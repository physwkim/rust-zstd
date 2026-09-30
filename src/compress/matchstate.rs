//! Persistent match-finder state: port of `ZSTD_MatchState_t`
//! (zstd_compress_internal.h) for the no-dictionary case.
//!
//! The finders address the input by index, as libzstd does through
//! `window.base`; no window starts below [`WINDOW_START_INDEX`], so index
//! `0` can mean "empty" in a table and `1` is `ZSTD_DUBT_UNSORTED_MARK`.
//! [`MatchState`] is the only owner of the position <-> index mapping
//! ([`MatchState::index`], [`MatchState::pos`], [`MatchState::view`]), kept
//! in its [`Window`].
//!
//! Indices continue from one input of a context to the next, as after
//! `ZSTD_resetCCtx_internal` with `ZSTDirp_continue`: a context's first
//! window starts at [`WINDOW_START_INDEX`], and each later one (the next
//! frame of a reused `Compressor`, the next job on a pooled context) starts
//! at the index where the previous input ended (`ZSTD_window_clear`). What
//! earlier inputs stored stays in the tables, all of it below the new
//! `window_low`, where every finder takes an entry for a miss as it takes
//! `0`. [`MatchState::reset`] alone decides to restart at
//! [`WINDOW_START_INDEX`] instead (`needsIndexReset`): on first use, when
//! the tables outgrow their allocation, or when the previous input ended
//! within 16 MiB of [`CURRENT_MAX`] (`ZSTD_indexTooCloseToMax`). Table
//! cells are zeroed only where [`Workspace`] cannot vouch for them (the
//! `tableValidEnd` of `ZSTD_cwksp`).
//!
//! Indices are stored as `u32`: before a block would end above
//! [`CURRENT_MAX`], [`MatchState::start_block`] moves the window's base
//! forward and reduces every stored index by the same amount
//! (`ZSTD_overflowCorrectIfNeeded`), so inputs of any size compress.
//!
//! A candidate index `c` is usable at index `cur` only if
//! `c >= window_low` and `cur - c <= (1 << window_log)`; see
//! [`MatchState::lowest_prefix_index`].
//!
//! The tables share one allocation ([`Workspace`], the table area of
//! `ZSTD_cwksp`), so the allocator sees one request per context rather
//! than three that straddle glibc's dynamic mmap threshold.

use std::ops::Range;

use super::common::Src;
use super::lazy::DUBT_UNSORTED_MARK;
use super::opt::OptState;
use super::params::{CParams, Strategy};

/// `ZSTD_WINDOW_START_INDEX`: the index of a fresh window's first byte.
pub const WINDOW_START_INDEX: usize = 2;

/// `ZSTD_INDEXOVERFLOW_MARGIN`: a reset restarts indices when the previous
/// input ended less than this below [`CURRENT_MAX`].
const INDEX_OVERFLOW_MARGIN: usize = 16 << 20;

/// `ZSTD_CURRENT_MAX` (64-bit): the highest index a block, or a long
/// distance matching chunk, may end at without its window being corrected
/// first. The `ZSTD_CHUNKSIZE_MAX` (596 MiB) indices above it exceed any
/// block or chunk.
pub const CURRENT_MAX: usize = 3500 << 20;

/// `ZSTD_window_t` without a dictionary (`lowLimit == dictLimit`, no
/// `dictBase`) over one contiguous input at a time: the position <-> index
/// mapping of a [`MatchState`] or an [`LdmState`](super::ldm::LdmState),
/// each of which owns one and alone moves it.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    /// `window.base` as a position of the input slice: the position of
    /// index 0 (wrapping, it may lie before the input).
    base: usize,
    /// `window.lowLimit` (== `dictLimit`): the lowest valid index.
    low: usize,
    /// `window.nextSrc` as a position of the input slice: the end of the
    /// input indexed so far, where the next input's indices continue.
    next_src: usize,
    /// `nbOverflowCorrections`.
    nb_overflow_corrections: u32,
    /// `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY`: correct whenever
    /// [`Window::can_overflow_correct`] allows it, not only near
    /// [`CURRENT_MAX`]. A test knob, see `CompressOptions`.
    correct_frequently: bool,
}

impl Window {
    /// `ZSTD_window_init`, then the input's first (non-contiguous)
    /// `ZSTD_window_update`: position `origin` is index
    /// [`WINDOW_START_INDEX`], the lowest valid one.
    pub fn new(origin: usize, correct_frequently: bool) -> Self {
        Self {
            base: origin.wrapping_sub(WINDOW_START_INDEX),
            low: WINDOW_START_INDEX,
            next_src: origin,
            nb_overflow_corrections: 0,
            correct_frequently,
        }
    }

    /// `ZSTD_window_clear`, then the next input's (non-contiguous)
    /// `ZSTD_window_update`: position `origin` of that input gets the index
    /// where the previous input ended, and becomes the lowest valid one, so
    /// every index stored so far lies below the window.
    fn continue_at(&mut self, origin: usize) {
        let end = self.index(self.next_src);
        self.base = origin.wrapping_sub(end);
        self.low = end;
        self.next_src = origin;
    }

    /// The contiguous `ZSTD_window_update`: the input now reaches position
    /// `end`.
    #[inline]
    fn extend_to(&mut self, end: usize) {
        self.next_src = end;
    }

    /// `ZSTD_indexTooCloseToMax`: whether the input ended less than
    /// `ZSTD_INDEXOVERFLOW_MARGIN` below [`CURRENT_MAX`].
    fn too_close_to_max(&self) -> bool {
        self.index(self.next_src) > CURRENT_MAX - INDEX_OVERFLOW_MARGIN
    }

    /// The index of position `pos`.
    #[inline(always)]
    pub fn index(&self, pos: usize) -> usize {
        pos.wrapping_sub(self.base)
    }

    /// The position of index `idx`.
    #[inline(always)]
    pub fn pos(&self, idx: usize) -> usize {
        idx.wrapping_add(self.base)
    }

    /// `lowLimit`: the lowest valid index.
    #[inline(always)]
    pub fn low(&self) -> usize {
        self.low
    }

    /// `nbOverflowCorrections`: the corrections since the window last
    /// started at [`WINDOW_START_INDEX`].
    pub fn nb_overflow_corrections(&self) -> u32 {
        self.nb_overflow_corrections
    }

    /// The `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY` test knob.
    pub fn correct_frequently(&self) -> bool {
        self.correct_frequently
    }

    /// Set the `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY` test knob.
    #[doc(hidden)]
    pub fn set_correct_frequently(&mut self, on: bool) {
        self.correct_frequently = on;
    }

    /// `ZSTD_window_canOverflowCorrect` without a dictionary
    /// (`loadedDictEnd == 0`): whether the index of position `src` is large
    /// enough for a correction that keeps the whole window. In `U32`, as
    /// libzstd computes it.
    fn can_overflow_correct(&self, cycle_log: u32, max_dist: u32, src: usize) -> bool {
        let cycle_size = 1u32 << cycle_log;
        let curr = self.index(src) as u32;
        let min_index_to_overflow_correct =
            cycle_size + max_dist.max(cycle_size) + WINDOW_START_INDEX as u32;
        // Adjust the min index to backoff the overflow correction frequency,
        // so we don't waste too much CPU in overflow correction. If this
        // computation overflows we don't really care, we just need to make
        // sure it is at least minIndexToOverflowCorrect.
        let adjustment = self.nb_overflow_corrections.wrapping_add(1);
        let adjusted_index = min_index_to_overflow_correct
            .wrapping_mul(adjustment)
            .max(min_index_to_overflow_correct);
        let index_large_enough = curr > adjusted_index;
        // Only overflow correct early if the dictionary is invalidated
        // already, so we don't hurt compression ratio.
        let dictionary_invalidated = curr > max_dist;
        index_large_enough && dictionary_invalidated
    }

    /// `ZSTD_window_needOverflowCorrection` (`loadedDictEnd == 0`): whether
    /// the window must be corrected before the positions `src..src_end` are
    /// indexed.
    #[inline]
    pub fn need_overflow_correction(
        &self,
        cycle_log: u32,
        max_dist: u32,
        src: usize,
        src_end: usize,
    ) -> bool {
        if self.correct_frequently && self.can_overflow_correct(cycle_log, max_dist, src) {
            return true;
        }
        self.index(src_end) > CURRENT_MAX
    }

    /// `ZSTD_window_correctOverflow`: move the base forward so that
    /// position `src` gets index `max(max_dist, cycle) + c`, `c` its low
    /// `cycle_log` bits (plus a cycle when below [`WINDOW_START_INDEX`]):
    /// chains and trees stay valid, and so do the `max_dist` indices below
    /// it. Returns the correction, which the owner subtracts from every
    /// index it stores.
    pub fn correct_overflow(&mut self, cycle_log: u32, max_dist: u32, src: usize) -> u32 {
        let cycle_size = 1u32 << cycle_log;
        let cycle_mask = cycle_size - 1;
        let curr = self.index(src) as u32;
        let current_cycle = curr & cycle_mask;
        // Ensure newCurrent - maxDist >= ZSTD_WINDOW_START_INDEX.
        let current_cycle_correction = if current_cycle < WINDOW_START_INDEX as u32 {
            cycle_size.max(WINDOW_START_INDEX as u32)
        } else {
            0
        };
        let new_current = current_cycle + current_cycle_correction + max_dist.max(cycle_size);
        debug_assert!(u32::try_from(self.index(src)).is_ok());
        // maxDist must be a power of two so that:
        //   (newCurrent & cycleMask) == (curr & cycleMask)
        // This is required to not corrupt the chains / binary tree.
        debug_assert!(max_dist.is_power_of_two());
        debug_assert_eq!(curr & cycle_mask, new_current & cycle_mask);
        debug_assert!(curr > new_current);
        let correction = curr - new_current;
        if !self.correct_frequently {
            // Loose bound, should be around 1<<29
            debug_assert!(correction > 1 << 28);
        }
        let reduced = correction as usize;
        self.base = self.base.wrapping_add(reduced);
        self.low = if self.low < reduced + WINDOW_START_INDEX {
            WINDOW_START_INDEX
        } else {
            self.low - reduced
        };
        // Ensure we can still reference the full window.
        debug_assert!(new_current - max_dist >= WINDOW_START_INDEX as u32);
        // Ensure that lowLimit didn't underflow.
        debug_assert!(self.low <= new_current as usize);
        self.nb_overflow_corrections = self.nb_overflow_corrections.wrapping_add(1);
        correction
    }

    /// `ZSTD_window_enforceMaxDist` (`loadedDictEnd == 0`): raise `lowLimit`
    /// to `max_dist` below the index of position `block_end`.
    #[inline]
    pub fn enforce_max_dist(&mut self, block_end: usize, max_dist: usize) {
        let block_end_idx = self.index(block_end);
        if block_end_idx > max_dist {
            self.low = self.low.max(block_end_idx - max_dist);
        }
    }

    /// `ZSTD_initStats_ultra`'s window move: `base -= len`, `dictLimit` and
    /// `lowLimit` up by `len`.
    fn skip(&mut self, len: usize) {
        self.base = self.base.wrapping_sub(len);
        self.low += len;
    }
}

/// `ZSTD_reduceTable_internal`: subtract `reducer` from every index of
/// `table`, squashing the ones that would fall below
/// [`WINDOW_START_INDEX`] to `0` (empty); with `preserve_mark`
/// (`ZSTD_reduceTable_btlazy2`) `ZSTD_DUBT_UNSORTED_MARK` stays as is.
fn reduce_table(table: &mut [u32], reducer: u32, preserve_mark: bool) {
    const MARK: u32 = DUBT_UNSORTED_MARK as u32;
    // Protect special index values < ZSTD_WINDOW_START_INDEX.
    let threshold = reducer + WINDOW_START_INDEX as u32;
    for cell in table {
        *cell = if preserve_mark && *cell == MARK {
            MARK
        } else if *cell < threshold {
            0
        } else {
            *cell - reducer
        };
    }
}

pub struct MatchState {
    pub cparams: CParams,
    /// `hashTable`, `chainTable` and `tagTable`, see [`Workspace::tables_mut`].
    /// Strategies borrow them as `ms.ws.tables_mut()`, which leaves the
    /// other fields accessible while the slices are live.
    pub ws: Workspace,
    /// `nextToUpdate`: index from which table insertion resumes.
    pub next_to_update: usize,
    /// `window`: the index space, see [`MatchState::window_low`].
    window: Window,
    /// `hashSalt`: salt of the row-based finder's hash (`ZSTD_hashPtrSalted`),
    /// advanced on every reset, see [`MatchState::reset`].
    pub hash_salt: u64,
    /// `hashSaltEntropy`: running sum of the row finder's search hashes over
    /// the context's life, mixed into the salt on every reset.
    pub hash_salt_entropy: u32,
    /// `opt`: the optimal parser's statistics and work tables, allocated
    /// for the opt strategies and kept (not shrunk) across resets.
    pub opt: Option<Box<OptState>>,
}

/// The table area of `ZSTD_cwksp`: one allocation holding
/// `hashTable` (`1 << hash_log` entries, every strategy), `chainTable`
/// (`1 << chain_log` entries; empty for `Fast`; `hashSmall` for `DFast`;
/// the hash-chain table for the lazy strategies; the binary tree, two
/// entries per node, for `BtLazy2` and the opt strategies), `hashTable3`
/// (`1 << hash_log3` entries, opt strategies with `min_match == 3` only)
/// and `tagTable` (`1 << hash_log` bytes, row-based lazy finder only, else
/// empty).
///
/// The first three are the index tables. `words[..valid]` is known to hold
/// only values below the owner's window end (`0`, `1` or indices of inputs
/// it has indexed; `ZSTD_cwksp`'s `tableValidEnd`), which after
/// `ZSTD_window_clear` are all misses: index tables laid over those words
/// need no zeroing.
#[derive(Default)]
pub struct Workspace {
    words: Vec<u32>,
    hash_len: usize,
    chain_len: usize,
    hash3_len: usize,
    tag_len: usize,
    valid: usize,
}

impl Workspace {
    /// `(hash, chain, hash3, tag)` lengths for `cparams.strategy`.
    fn lens(cparams: &CParams) -> (usize, usize, usize, usize) {
        let hash = 1usize << cparams.hash_log;
        let chain = 1usize << cparams.chain_log;
        match cparams.strategy {
            Strategy::Fast => (hash, 0, 0, 0),
            Strategy::DFast => (hash, chain, 0, 0),
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => (hash, chain, 0, hash),
            Strategy::BtLazy2 => (hash, chain, 0, 0),
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                let hash3 = match cparams.hash_log3() {
                    0 => 0,
                    log => 1usize << log,
                };
                (hash, chain, hash3, 0)
            }
        }
    }

    /// Words the tables for `cparams` take.
    fn words_for(cparams: &CParams) -> usize {
        let (hash_len, chain_len, hash3_len, tag_len) = Self::lens(cparams);
        hash_len + chain_len + hash3_len + tag_len.div_ceil(4)
    }

    /// Whether the tables for `cparams` fit the allocation (not
    /// `workspaceTooSmall`).
    fn fits(&self, cparams: &CParams) -> bool {
        Self::words_for(cparams) <= self.words.len()
    }

    /// Lay the tables for `cparams` out (`ZSTD_reset_matchState` with
    /// `ZSTDcrp_makeClean`). Tables that do not fit get a new zeroed
    /// allocation (`ZSTD_cwksp_create`; the owner resets its indices);
    /// otherwise `index_reset` forgets every word
    /// (`ZSTD_cwksp_mark_tables_dirty`), and the index tables' words past
    /// `valid` are zeroed (`ZSTD_cwksp_clean_tables`). The tag table is
    /// never zeroed here: `ZSTD_cwksp_reserve_aligned_init_once` only
    /// zeroes new memory.
    fn reset(&mut self, cparams: &CParams, index_reset: bool) {
        let (hash_len, chain_len, hash3_len, tag_len) = Self::lens(cparams);
        let index_end = hash_len + chain_len + hash3_len;
        let words = Self::words_for(cparams);
        if self.words.len() < words {
            debug_assert!(index_reset);
            // ZSTD_cwksp_free before ZSTD_cwksp_create: never both at once.
            drop(std::mem::take(&mut self.words));
            self.words = vec![0; words];
            self.valid = words;
        } else if index_reset {
            self.valid = 0;
        }
        if self.valid < index_end {
            self.words[self.valid..index_end].fill(0);
        }
        // Tags are no indices: past the index tables nothing is vouched for
        // any more, as a buffer reserved below `tableValidEnd` lowers it.
        self.valid = if tag_len > 0 {
            index_end
        } else {
            self.valid.max(index_end)
        };
        self.hash_len = hash_len;
        self.chain_len = chain_len;
        self.hash3_len = hash3_len;
        self.tag_len = tag_len;
    }

    /// `ZSTD_reduceIndex` between `ZSTD_cwksp_mark_tables_dirty` and
    /// `ZSTD_cwksp_mark_tables_clean`: [`reduce_table`] on `hashTable`,
    /// `chainTable` (keeping btlazy2's unsorted marks with `preserve_mark`)
    /// and `hashTable3`. Words past them keep indices from before the
    /// correction, so they are no longer vouched for.
    fn reduce(&mut self, correction: u32, preserve_mark: bool) {
        let (hash, chain, hash3) = self.opt_tables_mut();
        reduce_table(hash, correction, false);
        reduce_table(chain, correction, preserve_mark);
        reduce_table(hash3, correction, false);
        self.valid = self.hash_len + self.chain_len + self.hash3_len;
    }

    /// `(hashTable, chainTable, tagTable)`. Unchecked splits: the bounds
    /// checks of `split_at_mut` at a strategy's entry re-allocate the
    /// registers of its whole hot loop (fast L1 measured 7% slower).
    #[inline]
    pub fn tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u8]) {
        // SAFETY: `reset` is the only writer of the lengths and keeps
        // `words.len() >= hash_len + chain_len + hash3_len + tag_len.div_ceil(4)`.
        unsafe {
            let (hash, rest) = self.words.split_at_mut_unchecked(self.hash_len);
            let (chain, rest) = rest.split_at_mut_unchecked(self.chain_len);
            let tag = rest.get_unchecked_mut(self.hash3_len..);
            let tag: &mut [u8] = bytemuck::cast_slice_mut(tag);
            (hash, chain, tag.get_unchecked_mut(..self.tag_len))
        }
    }

    /// `(hashTable, chainTable, tagTable)`, see [`Workspace::tables_mut`].
    #[inline]
    pub fn tables(&self) -> (&[u32], &[u32], &[u8]) {
        // SAFETY: as in `tables_mut`.
        unsafe {
            let (hash, rest) = self.words.split_at_unchecked(self.hash_len);
            let (chain, rest) = rest.split_at_unchecked(self.chain_len);
            let tag = rest.get_unchecked(self.hash3_len..);
            let tag: &[u8] = bytemuck::cast_slice(tag);
            (hash, chain, tag.get_unchecked(..self.tag_len))
        }
    }

    /// `(hashTable, chainTable, hashTable3)` of the opt strategies.
    #[inline]
    pub fn opt_tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u32]) {
        // SAFETY: as in `tables_mut`.
        unsafe {
            let (hash, rest) = self.words.split_at_mut_unchecked(self.hash_len);
            let (chain, rest) = rest.split_at_mut_unchecked(self.chain_len);
            (hash, chain, rest.get_unchecked_mut(..self.hash3_len))
        }
    }

    /// `hashTable3`, see [`Workspace::opt_tables_mut`].
    #[inline]
    pub fn hash3(&self) -> &[u32] {
        // SAFETY: as in `tables_mut`.
        unsafe {
            self.words
                .get_unchecked(self.hash_len + self.chain_len..)
                .get_unchecked(..self.hash3_len)
        }
    }
}

/// The indices of one block about to be searched. Only
/// [`MatchState::start_block`] makes one, after the block-start
/// `nextToUpdate` clamp, so no finder can run without it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    start: usize,
    end: usize,
}

impl Block {
    #[inline]
    pub fn range(self) -> Range<usize> {
        self.start..self.end
    }
}

impl MatchState {
    /// Allocate zeroed tables for `cparams.strategy` and start a window at
    /// position `origin`, index [`WINDOW_START_INDEX`]: the first use of a
    /// context, see [`MatchState::reset`].
    pub fn new(cparams: CParams, origin: usize) -> Self {
        let mut ms = Self {
            cparams,
            ws: Workspace::default(),
            next_to_update: 0,
            window: Window::new(origin, false),
            hash_salt: 0,
            hash_salt_entropy: 0,
            opt: None,
        };
        ms.reset(cparams, origin);
        ms
    }

    /// `ZSTD_resetCCtx_internal` with `ZSTDcrp_makeClean` on a used context,
    /// for an input whose window starts at position `origin`: the one place
    /// that decides whether indices continue (`needsIndexReset`).
    ///
    /// Indices restart at [`WINDOW_START_INDEX`] (`ZSTDirp_reset`:
    /// `ZSTD_window_init`, every table word zeroed) when the tables for
    /// `cparams` outgrow the allocation (`workspaceTooSmall`; a new one is
    /// made) or the previous input ended too close to [`CURRENT_MAX`]
    /// (`ZSTD_indexTooCloseToMax`). Otherwise they continue
    /// (`ZSTDirp_continue`): `ZSTD_window_clear` puts position `origin` at
    /// the index where the previous input ended and makes it the window's
    /// low end, the tables keep their contents, all of it now below the
    /// window, and only words [`Workspace`] cannot vouch for are zeroed.
    /// Every finder takes an entry below the window for a miss, as it takes
    /// `0`, so either way the frames equal those of [`MatchState::new`].
    ///
    /// Every reset also advances the row finder's hash salt
    /// (`ZSTD_advanceHashSalt`; libzstd only when that finder is in use, and
    /// the salt is unused otherwise). The salt XORs the multiplied hash, so
    /// it only permutes rows and tags, and frames do not depend on it. It
    /// keeps the kept tags from matching the current input's: a stale entry
    /// is older than every current one in its row, so a match on it ends
    /// the search as an empty slot's would, but only after loading the
    /// entry. With a constant salt, input repeating the previous one met
    /// such a match in nearly every search (elf L11: 8% slower).
    ///
    /// The overflow correction knob ([`MatchState::set_correct_frequently`])
    /// is a property of the context and survives the reset.
    pub fn reset(&mut self, cparams: CParams, origin: usize) {
        let index_reset = !self.ws.fits(&cparams) || self.window.too_close_to_max();
        self.ws.reset(&cparams, index_reset);
        if index_reset {
            self.window = Window::new(origin, self.window.correct_frequently());
        } else {
            self.window.continue_at(origin);
        }
        self.cparams = cparams;
        // ZSTD_invalidateMatchState: `nextToUpdate = dictLimit`.
        self.next_to_update = self.window.low;
        self.hash_salt = super::lazy::advance_hash_salt(self.hash_salt, self.hash_salt_entropy);
        // ZSTD_invalidateMatchState: `opt.litLengthSum = 0` forces the next
        // opt block to initialize its statistics.
        if cparams.strategy.is_opt() {
            self.opt
                .get_or_insert_with(|| Box::new(OptState::new()))
                .invalidate();
        }
    }

    /// Test knob: `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY`, see
    /// [`Window::need_overflow_correction`].
    #[doc(hidden)]
    pub fn set_correct_frequently(&mut self, on: bool) {
        self.window.set_correct_frequently(on);
    }

    /// The index of position `pos` (`pos >= pos(0)`).
    #[inline]
    pub fn index(&self, pos: usize) -> usize {
        let idx = self.window.index(pos);
        debug_assert!(idx <= isize::MAX as usize, "position {pos} before index 0");
        idx
    }

    /// The position of index `idx`.
    #[inline]
    pub fn pos(&self, idx: usize) -> usize {
        self.window.pos(idx)
    }

    /// `window.lowLimit` / `window.dictLimit`: the lowest valid index.
    #[inline(always)]
    pub fn window_low(&self) -> usize {
        self.window.low
    }

    /// `window`.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// `data` from the window's lowest valid index on, addressed by index.
    #[inline]
    pub fn view<'a>(&self, data: &'a [u8]) -> Src<'a> {
        Src::new(data, self.window.pos(self.window.low), self.window.low)
    }

    /// `ZSTD_overflowCorrectIfNeeded` before the positions `range` (a block,
    /// or a loaded prefix) are indexed: when the window must be corrected
    /// ([`Window::need_overflow_correction`] with `ZSTD_cycleLog` and the
    /// window size), move it ([`Window::correct_overflow`]) and subtract the
    /// correction from every stored index (`ZSTD_reduceIndex`) and from
    /// `next_to_update`.
    pub fn correct_overflow_if_needed(&mut self, range: Range<usize>) {
        let cycle_log = self.cparams.chain_log - self.cparams.strategy.bt_scale();
        let max_dist = 1u32 << self.cparams.window_log;
        if self
            .window
            .need_overflow_correction(cycle_log, max_dist, range.start, range.end)
        {
            let correction = self
                .window
                .correct_overflow(cycle_log, max_dist, range.start);
            self.reduce_index(correction);
            self.next_to_update = self.next_to_update.saturating_sub(correction as usize);
        }
    }

    /// `ZSTD_reduceIndex` ([`Workspace::reduce`]), keeping btlazy2's
    /// unsorted marks. The row finder's tag table holds no index.
    fn reduce_index(&mut self, correction: u32) {
        let preserve_mark = self.cparams.strategy == Strategy::BtLazy2;
        self.ws.reduce(correction, preserve_mark);
    }

    /// `ZSTD_compress_frameChunk` and `ZSTD_buildSeqStore`'s set-up of a
    /// block: the overflow correction for `positions`
    /// ([`MatchState::correct_overflow_if_needed`]), then `data[positions]`
    /// as indices of this window, after the "limited update after a very
    /// long match" clamp: when the previous block left more than 384
    /// positions uninserted (its last match ran past the block end), insert
    /// at most the 192 positions before the block (fewer while the backlog
    /// is under 576) instead of the whole backlog.
    pub fn start_block<'a>(&mut self, data: &'a [u8], positions: Range<usize>) -> (Src<'a>, Block) {
        self.correct_overflow_if_needed(positions.clone());
        self.window.extend_to(positions.end);
        let src = self.view(data);
        let (start, end) = (self.index(positions.start), self.index(positions.end));
        assert!(
            u32::try_from(end).is_ok(),
            "block end index {end} exceeds the u32 index space"
        );
        if start > self.next_to_update + 384 {
            self.next_to_update = start - 192.min(start - self.next_to_update - 384);
        }
        (src, Block { start, end })
    }

    /// The literal run `range` of `block` (indices) that
    /// `ZSTD_ldm_blockCompress` hands to the block compressor, after
    /// `ZSTD_ldm_limitTableUpdate`: when more than 1024 positions before it
    /// are uninserted, insert at most the last 512 of them (fewer while the
    /// backlog is under 1536).
    pub fn ldm_sub_block(&mut self, block: Block, range: Range<usize>) -> Block {
        assert!(block.start <= range.start && range.start <= range.end && range.end <= block.end);
        let start = range.start;
        if start > self.next_to_update + 1024 {
            self.next_to_update = start - 512.min(start - self.next_to_update - 1024);
        }
        Block {
            start,
            end: range.end,
        }
    }

    /// Move the window past the `len` bytes that begin it, as
    /// `ZSTD_initStats_ultra` forgets its first pass: `base -= len`,
    /// `dictLimit` and `lowLimit` up by `len`, `nextToUpdate = dictLimit`.
    /// Every index already in the tables falls below the window, and every
    /// byte's index grows by `len`; returns `src` and `block` re-addressed.
    pub fn skip_window<'a>(
        &mut self,
        src: Src<'a>,
        block: Range<usize>,
        len: usize,
    ) -> (Src<'a>, Range<usize>) {
        debug_assert_eq!(block.start, self.window.low);
        self.window.skip(len);
        self.next_to_update = self.window.low;
        let block = block.start + len..block.end + len;
        assert!(
            u32::try_from(block.end).is_ok(),
            "window moved past the u32 index space"
        );
        (src.rebased(len), block)
    }

    /// `(hashTable, chainTable, tagTable)`; borrows the whole state, use
    /// `self.ws.tables_mut()` where the other fields must stay accessible.
    #[inline]
    pub fn tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u8]) {
        self.ws.tables_mut()
    }

    /// `(hashTable, chainTable, tagTable)`.
    #[inline]
    pub fn tables(&self) -> (&[u32], &[u32], &[u8]) {
        self.ws.tables()
    }

    /// `ZSTD_getLowestPrefixIndex(ms, cur, windowLog)` without a dictionary:
    /// the lowest index a match may reference from index `cur`.
    #[inline]
    pub fn lowest_prefix_index(&self, cur: usize) -> usize {
        let max_distance = 1usize << self.cparams.window_log;
        let window_low = self.window.low;
        debug_assert!(cur >= window_low);
        // C: `curr - lowestValid > maxDistance`
        if cur - window_low > max_distance {
            cur - max_distance
        } else {
            window_low
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_block_limits_update_after_long_match() {
        let data = vec![0u8; 500_000];
        let mut ms = MatchState::new(CParams::for_level(5, 1 << 20), 0);
        // The clamp for a block starting at index `idx`.
        let mut clamp = |next_to_update: Option<usize>, idx: usize| {
            if let Some(n) = next_to_update {
                ms.next_to_update = n;
            }
            let pos = idx - WINDOW_START_INDEX;
            let (_, block) = ms.start_block(&data, pos..pos);
            assert_eq!(block.range(), idx..idx);
            ms.next_to_update
        };
        // Backlog of exactly 384: untouched.
        assert_eq!(clamp(Some(1000), 1384), 1000);
        // Backlog 385..575: only the excess over 384 gets inserted.
        assert_eq!(clamp(Some(1000), 1385), 1384);
        assert_eq!(clamp(Some(1000), 1575), 1384);
        // Backlog >= 576: insert only the last 192 positions.
        assert_eq!(clamp(Some(1000), 1576), 1384);
        assert_eq!(clamp(Some(1000), 500_000), 500_000 - 192);
        // Idempotent.
        assert_eq!(clamp(None, 500_000), 500_000 - 192);
    }

    /// Position of index `idx` in a window whose origin is position 0.
    fn at(idx: usize) -> usize {
        idx - WINDOW_START_INDEX
    }

    /// `ZSTD_window_needOverflowCorrection`: past [`CURRENT_MAX`] at the
    /// block end; with the knob, past `cycle + max(maxDist, cycle) + 2`
    /// at the block start, times `nbOverflowCorrections + 1`.
    #[test]
    fn window_needs_correction_past_thresholds() {
        let (cycle_log, max_dist) = (12, 1u32 << 19);
        let stock = Window::new(0, false);
        // Whether a block from index `start` to index `end` needs one.
        let need = |w: &Window, start: usize, end: usize| {
            w.need_overflow_correction(cycle_log, max_dist, w.pos(start), w.pos(end))
        };
        assert!(!need(&stock, CURRENT_MAX - 10, CURRENT_MAX));
        assert!(need(&stock, CURRENT_MAX - 10, CURRENT_MAX + 1));
        let min = (1 << 12) + (1 << 19) + WINDOW_START_INDEX;
        let mut w = Window::new(0, true);
        assert!(!need(&w, min, min + 10));
        assert!(need(&w, min + 1, min + 10));
        for n in 1..4 {
            w.correct_overflow(cycle_log, max_dist, w.pos(min * n + 1));
            let threshold = min * (n + 1);
            assert!(!need(&w, threshold, threshold + 10), "after {n}");
            assert!(need(&w, threshold + 1, threshold + 10), "after {n}");
        }
        assert_eq!(w.nb_overflow_corrections(), 3);
    }

    /// `ZSTD_window_correctOverflow`: the block start keeps its low
    /// `cycle_log` bits (raised by a cycle when below
    /// [`WINDOW_START_INDEX`]) above `max(maxDist, cycle)`, and `lowLimit`
    /// moves down with it, but not below [`WINDOW_START_INDEX`].
    #[test]
    fn window_correction_keeps_cycle_and_window() {
        let (cycle_log, max_dist) = (12, 1u32 << 19);
        let mut w = Window::new(0, false);
        let idx = (3 << 30) + 5;
        let correction = w.correct_overflow(cycle_log, max_dist, at(idx));
        assert_eq!(w.index(at(idx)), (1 << 19) + 5);
        assert_eq!(correction as usize, idx - ((1 << 19) + 5));
        assert_eq!(w.low(), WINDOW_START_INDEX);
        // Current cycle 0: one cycle more. lowLimit exactly maxDist below
        // the block start comes down to the cycle's index.
        let mut w = Window::new(0, false);
        let idx = 3 << 30;
        w.enforce_max_dist(at(idx), 1 << 19);
        assert_eq!(w.low(), idx - (1 << 19));
        w.correct_overflow(cycle_log, max_dist, at(idx));
        assert_eq!(w.index(at(idx)), (1 << 19) + (1 << 12));
        assert_eq!(w.low(), 1 << 12);
        // Cycle log 0 (long distance matching): 2 above maxDist.
        let mut w = Window::new(0, false);
        w.correct_overflow(0, max_dist, at(idx));
        assert_eq!(w.index(at(idx)), (1 << 19) + 2);
    }

    /// `ZSTD_reduceTable` / `_btlazy2`: indices below the correction plus
    /// [`WINDOW_START_INDEX`] become 0 (empty); btlazy2 keeps its unsorted
    /// mark.
    #[test]
    fn reduce_table_squashes_below_window_start() {
        let cells = [0, 1, 2, 101, 102, 1000];
        let mut plain = cells;
        reduce_table(&mut plain, 100, false);
        assert_eq!(plain, [0, 0, 0, 0, 2, 900]);
        let mut marked = cells;
        reduce_table(&mut marked, 100, true);
        assert_eq!(marked, [0, DUBT_UNSORTED_MARK as u32, 0, 0, 2, 900]);
    }

    /// `strategy` with explicit table logs.
    fn table_params(strategy: Strategy, hash_log: u32, chain_log: u32) -> CParams {
        CParams {
            strategy,
            hash_log,
            chain_log,
            ..CParams::for_level(1, 1 << 20)
        }
    }

    /// `ZSTDirp_continue`: the next input's first byte gets the index where
    /// the previous input ended, which becomes the window's low end, and
    /// the tables keep what the previous input stored.
    #[test]
    fn reset_continues_indices_over_kept_tables() {
        let cp = table_params(Strategy::DFast, 10, 10);
        let data = vec![0u8; 5000];
        let mut ms = MatchState::new(cp, 0);
        ms.start_block(&data, 0..3000);
        ms.start_block(&data, 3000..5000);
        ms.ws.tables_mut().0.fill(77);
        ms.reset(cp, 100);
        assert_eq!(ms.window_low(), WINDOW_START_INDEX + 5000);
        assert_eq!(ms.index(100), ms.window_low());
        assert_eq!(ms.next_to_update, ms.window_low());
        assert!(ms.tables().0.iter().all(|&e| e == 77));
    }

    /// `ZSTD_indexTooCloseToMax`: indices restart, and the tables are
    /// zeroed, once the previous input ended above `CURRENT_MAX - 16 MiB`.
    #[test]
    fn reset_restarts_indices_close_to_max() {
        let cp = table_params(Strategy::Fast, 10, 0);
        let edge = CURRENT_MAX - INDEX_OVERFLOW_MARGIN;
        for (end, restarts) in [(edge, false), (edge + 1, true)] {
            let mut ms = MatchState::new(cp, 0);
            let end_pos = ms.pos(end);
            ms.window.extend_to(end_pos);
            ms.ws.tables_mut().0.fill(77);
            ms.reset(cp, 0);
            let low = if restarts { WINDOW_START_INDEX } else { end };
            assert_eq!(ms.window_low(), low, "end {end}");
            assert_eq!(ms.tables().0.iter().all(|&e| e == 0), restarts);
        }
    }

    /// `workspaceTooSmall`: tables that outgrow the allocation get a new,
    /// zeroed one and restart the indices.
    #[test]
    fn reset_restarts_indices_when_tables_outgrow() {
        let data = vec![0u8; 5000];
        let mut ms = MatchState::new(table_params(Strategy::Fast, 10, 0), 0);
        ms.start_block(&data, 0..5000);
        ms.ws.tables_mut().0.fill(77);
        ms.reset(table_params(Strategy::Fast, 11, 0), 0);
        assert_eq!(ms.window_low(), WINDOW_START_INDEX);
        assert_eq!(ms.tables().0.len(), 1 << 11);
        assert!(ms.tables().0.iter().all(|&e| e == 0));
    }

    /// `tableValidEnd`: index tables laid over words that held indices are
    /// not zeroed, whatever table held them; words past the reduced tables
    /// of a correction, words that held tags, and every word after an
    /// index reset are.
    #[test]
    fn workspace_zeroes_only_words_it_cannot_vouch_for() {
        let big = table_params(Strategy::DFast, 12, 12);
        let small = table_params(Strategy::Fast, 10, 0);
        let rows = table_params(Strategy::Lazy, 11, 11);
        let mut ws = Workspace::default();
        ws.reset(&big, true);
        assert_eq!(ws.words.len(), 8192);
        ws.words.fill(7);
        ws.reset(&small, false);
        ws.reset(&big, false);
        assert!(ws.words.iter().all(|&w| w == 7));
        // The correction reduces the small tables only.
        ws.reset(&small, false);
        ws.reduce(5, false);
        ws.reset(&big, false);
        assert!(ws.words[..1024].iter().all(|&w| w == 2));
        assert!(ws.words[1024..].iter().all(|&w| w == 0));
        // Index tables 4096 words, tags 512 words after them.
        ws.words.fill(7);
        ws.reset(&rows, false);
        ws.tables_mut().2.fill(9);
        ws.reset(&big, false);
        assert!(ws.words[..4096].iter().all(|&w| w == 7));
        assert!(ws.words[4096..].iter().all(|&w| w == 0));
        ws.words.fill(7);
        ws.reset(&small, true);
        assert!(ws.words[..1024].iter().all(|&w| w == 0));
        assert!(ws.words[1024..].iter().all(|&w| w == 7));
    }
}
