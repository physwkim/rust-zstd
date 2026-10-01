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
//! `0`. [`MatchState::reset_needing`] alone decides to restart at
//! [`WINDOW_START_INDEX`] instead (`needsIndexReset`): when the workspace
//! is resized (`Workspace::reserve`: on first use, when it is too small
//! for the reset, or when it has been three times too large after more
//! than 128 resets), or when the previous input ended within 16 MiB of
//! [`CURRENT_MAX`] (`ZSTD_indexTooCloseToMax`). Table cells are zeroed only
//! where [`Workspace`] cannot vouch for them (the `tableValidEnd` of
//! `ZSTD_cwksp`).
//!
//! Indices are stored as `u32`: before a block would end above
//! [`CURRENT_MAX`], [`MatchState::enter_block`] moves the window's base
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

use super::bt::ZSTD_OPT_SIZE;
use super::common::{Src, HASH_READ_SIZE};
use super::lazy::{default_search_method, SearchMethod, DUBT_UNSORTED_MARK};
use super::ldm::LdmParams;
use super::opt::OptState;
use super::params::{CParams, Strategy, ZSTD_HASHLOG3_MAX};
use super::seqstore::WILDCOPY_OVERLENGTH;
use crate::constants::{MAX_LL, MAX_ML, MAX_OFF, MEM_32BITS, ZSTD_BLOCKSIZE_MAX};

/// `ZSTD_WINDOW_START_INDEX`: the index of a fresh window's first byte.
pub const WINDOW_START_INDEX: usize = 2;

/// `ZSTD_INDEXOVERFLOW_MARGIN`: a reset restarts indices when the previous
/// input ended less than this below [`CURRENT_MAX`].
const INDEX_OVERFLOW_MARGIN: usize = 16 << 20;

/// `ZSTD_CURRENT_MAX`: the highest index a block, or a long distance
/// matching chunk, may end at without its window being corrected first;
/// 3500 MiB, or 2000 MiB where `size_t` is 32 bits. The
/// `ZSTD_CHUNKSIZE_MAX` (596 or 2096 MiB) indices above it exceed any block
/// or chunk.
pub const CURRENT_MAX: usize = if MEM_32BITS { 2000 << 20 } else { 3500 << 20 };

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
    /// for the opt strategies and kept until the workspace is resized.
    pub opt: Option<Box<OptState>>,
    /// The lazy match finder the tables are sized for (`useRowMatchFinder`
    /// resolved): [`MatchState::new`] and [`MatchState::reset`] take
    /// [`default_search_method`], [`MatchState::new_for`] and
    /// [`MatchState::reset_for`] a given one. Meaningful for the lazy
    /// strategies only.
    pub search_method: SearchMethod,
}

/// The table area of `ZSTD_cwksp`: one allocation holding `hashTable`
/// (`1 << hash_log` entries, every strategy), `chainTable` (`1 << chain_log`
/// entries; empty for `Fast` and the row-based lazy finder, as
/// `ZSTD_allocateChainTable` allocates none there; `hashSmall` for `DFast`;
/// the hash-chain table of the other lazy finders; the binary tree, two
/// entries per node, for `BtLazy2` and the opt strategies), `hashTable3`
/// (`1 << hash_log3` entries, opt strategies with `min_match == 3` only)
/// and `tagTable` (`1 << hash_log` bytes, row-based lazy finder only, else
/// empty), in that order.
///
/// Every table starts on a 4 KiB page boundary: the allocation is of
/// `Page`s and each table fills whole pages. A row of the row finder
/// (`1 << row_log` entries at a multiple of its size, at most 256 bytes of
/// `hashTable` and 64 of `tagTable`) therefore never crosses a page or a
/// cache line. `ZSTD_cwksp` only guarantees 64-byte alignment
/// (`ZSTD_CWKSP_ALIGNMENT_BYTES`); where its tables land within a page
/// depends on the sizes of the objects reserved before them.
///
/// The first three are the index tables; they and the padding between
/// them, which every reset zeroes, make up the index area. `words[..valid]`
/// is known to hold only values below the owner's window end (`0`, `1` or
/// indices of inputs it has indexed; `ZSTD_cwksp`'s `tableValidEnd`), which
/// after `ZSTD_window_clear` are all misses: index tables laid over those
/// words need no zeroing.
///
/// It also keeps `ZSTD_cwksp`'s size bookkeeping for the whole context
/// (see `Workspace::reserve`), whose resize frees the context's other
/// buffers too.
#[derive(Default)]
pub struct Workspace {
    pages: Vec<Page>,
    layout: Layout,
    valid: usize,
    /// `ZSTD_cwksp_sizeof`: the [`needed_space`] of the reset that last
    /// resized the workspace.
    size: usize,
    /// `ZSTD_cwksp_used` after the last reset: its [`needed_space`].
    used: usize,
    /// `workspaceOversizedDuration`.
    oversized_duration: u32,
}

/// Where the tables of [`Workspace`] lie, in words from the start of the
/// allocation.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Layout {
    hash_len: usize,
    chain_off: usize,
    chain_len: usize,
    hash3_off: usize,
    hash3_len: usize,
    tag_off: usize,
    tag_len: usize,
}

impl Layout {
    /// The tables for `cparams.strategy` and, for the lazy strategies, the
    /// finder `method`, each on its own pages.
    fn of(cparams: &CParams, method: SearchMethod) -> Self {
        let hash = 1usize << cparams.hash_log;
        let chain = 1usize << cparams.chain_log;
        let (hash_len, chain_len, hash3_len, tag_len) = match cparams.strategy {
            Strategy::Fast => (hash, 0, 0, 0),
            Strategy::DFast => (hash, chain, 0, 0),
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => match method {
                SearchMethod::RowHash => (hash, 0, 0, hash),
                SearchMethod::HashChain | SearchMethod::BinaryTree => (hash, chain, 0, 0),
            },
            Strategy::BtLazy2 => (hash, chain, 0, 0),
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
                let hash3 = match cparams.hash_log3() {
                    0 => 0,
                    log => 1usize << log,
                };
                (hash, chain, hash3, 0)
            }
        };
        let chain_off = whole_pages(hash_len);
        let hash3_off = chain_off + whole_pages(chain_len);
        Self {
            hash_len,
            chain_off,
            chain_len,
            hash3_off,
            hash3_len,
            tag_off: hash3_off + whole_pages(hash3_len),
            tag_len,
        }
    }

    /// The end of the index area.
    fn index_end(&self) -> usize {
        self.hash3_off + self.hash3_len
    }

    /// Words the tables take, a whole number of pages.
    fn words(&self) -> usize {
        self.tag_off + whole_pages(self.tag_len.div_ceil(4))
    }
}

/// The page size tables are aligned to.
const PAGE: usize = 4096;

/// `u32` entries per [`PAGE`] bytes.
const PAGE_WORDS: usize = PAGE / 4;

/// `words` rounded up to whole pages.
fn whole_pages(words: usize) -> usize {
    words.next_multiple_of(PAGE_WORDS)
}

/// One page of table entries: the allocation unit of [`Workspace`].
#[derive(Clone, Copy)]
#[repr(C, align(4096))]
struct Page([u32; PAGE_WORDS]);

// SAFETY: 1024 `u32`s are 4096 bytes, the alignment, so `Page` has no
// padding and every bit pattern is valid.
unsafe impl bytemuck::Zeroable for Page {}
unsafe impl bytemuck::Pod for Page {}

/// `n` zeroed `Page`s through `alloc_zeroed` (as `vec![0u32; n]` does), so
/// fresh tables stay untouched zero pages.
fn zeroed_pages(n: usize) -> Vec<Page> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<Page>(n).expect("workspace size");
    // SAFETY: `layout` is not zero-sized; zeroed memory is `n` valid `Page`s
    // (`Zeroable`), allocated with the layout `Vec<Page>` frees it with.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout).cast::<Page>();
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(p, n, n)
    }
}

impl Workspace {
    /// The allocation as table entries.
    #[inline]
    fn words(&self) -> &[u32] {
        bytemuck::cast_slice(&self.pages)
    }

    /// The allocation as table entries.
    #[inline]
    fn words_mut(&mut self) -> &mut [u32] {
        bytemuck::cast_slice_mut(&mut self.pages)
    }

    /// `ZSTD_resetCCtx_internal`'s workspace check for a reset that needs
    /// `needed` bytes ([`needed_space`]): `ZSTD_cwksp_bump_oversized_duration`,
    /// then `resizeWorkspace = workspaceTooSmall || workspaceWasteful`.
    /// Wasteful is three times `needed` free after the previous reset's use
    /// (`ZSTD_cwksp_check_too_large`) once more than
    /// [`WORKSPACE_TOO_LARGE_MAX_DURATION`] resets have passed since the
    /// last resize: libzstd bumps the duration for `additionalNeededSpace`
    /// 0, which is always free, so it counts every reset, oversized or not.
    /// A resize frees the tables (`ZSTD_cwksp_free`), which
    /// [`Workspace::reset`] then allocates at their size; the owner frees
    /// the context's other buffers. Returns whether the workspace was
    /// resized.
    ///
    /// `used` stands in for `ZSTD_cwksp_used`, which the workspace's
    /// alignment padding makes up to `2 * ZSTD_CWKSP_ALIGNMENT_BYTES` (128)
    /// smaller than the need: libzstd may find up to 128 bytes more free.
    fn reserve(&mut self, needed: usize) -> bool {
        let available = self.size - self.used;
        self.oversized_duration = self.oversized_duration.saturating_add(1);
        let too_small = self.size < needed;
        let wasteful = available >= needed.saturating_mul(WORKSPACE_TOO_LARGE_FACTOR)
            && self.oversized_duration > WORKSPACE_TOO_LARGE_MAX_DURATION;
        let resize = too_small || wasteful;
        if resize {
            self.pages = Vec::new();
            self.size = needed;
            self.oversized_duration = 0;
        }
        self.used = needed;
        resize
    }

    /// Lay the tables for `cparams` and `method` out
    /// (`ZSTD_reset_matchState` with `ZSTDcrp_makeClean`). Tables that do
    /// not fit get a new zeroed allocation: after [`Workspace::reserve`]
    /// freed them, or when tables of another shape outgrow them within a
    /// workspace libzstd keeps (its indices continue). Otherwise
    /// `index_reset` forgets every word (`ZSTD_cwksp_mark_tables_dirty`),
    /// and the index area's words past `valid` are zeroed
    /// (`ZSTD_cwksp_clean_tables`), as is the padding between the index
    /// tables. The tag table is never zeroed here:
    /// `ZSTD_cwksp_reserve_aligned_init_once` only zeroes new memory.
    fn reset(&mut self, cparams: &CParams, method: SearchMethod, index_reset: bool) {
        let layout = Layout::of(cparams, method);
        let index_end = layout.index_end();
        let pages = layout.words() / PAGE_WORDS;
        if self.pages.len() < pages {
            // ZSTD_cwksp_free before ZSTD_cwksp_create: never both at once.
            drop(std::mem::take(&mut self.pages));
            self.pages = zeroed_pages(pages);
            self.valid = pages * PAGE_WORDS;
        } else if index_reset {
            self.valid = 0;
        }
        let valid = self.valid;
        let words = self.words_mut();
        if valid < index_end {
            words[valid..index_end].fill(0);
        }
        // Padding holds no table, so no correction reduces it: zero it here
        // and it stays below every window end.
        words[layout.hash_len..layout.chain_off].fill(0);
        words[layout.chain_off + layout.chain_len..layout.hash3_off].fill(0);
        // Tags are no indices: past the index area nothing is vouched for
        // any more, as a buffer reserved below `tableValidEnd` lowers it.
        self.valid = if layout.tag_len > 0 {
            index_end
        } else {
            self.valid.max(index_end)
        };
        self.layout = layout;
    }

    /// `ZSTD_reduceIndex` between `ZSTD_cwksp_mark_tables_dirty` and
    /// `ZSTD_cwksp_mark_tables_clean`: [`reduce_table`] on `hashTable`,
    /// `chainTable` (keeping btlazy2's unsorted marks with `preserve_mark`)
    /// and `hashTable3`. Words past the index area keep indices from before
    /// the correction, so they are no longer vouched for.
    fn reduce(&mut self, correction: u32, preserve_mark: bool) {
        let (hash, chain, hash3) = self.opt_tables_mut();
        reduce_table(hash, correction, false);
        reduce_table(chain, correction, preserve_mark);
        reduce_table(hash3, correction, false);
        self.valid = self.layout.index_end();
    }

    /// `(hashTable, chainTable, tagTable)`. Unchecked splits: the bounds
    /// checks of `split_at_mut` at a strategy's entry re-allocate the
    /// registers of its whole hot loop (fast L1 measured 7% slower).
    #[inline]
    pub fn tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u8]) {
        let l = self.layout;
        // SAFETY: `reset` is the only writer of `layout` and sizes `pages`
        // to hold `l.words()` entries, with `l.hash_len <= l.chain_off`,
        // `l.chain_off + l.chain_len <= l.hash3_off <= l.tag_off`.
        unsafe {
            let words = self.words_mut();
            let (hash, rest) = words.split_at_mut_unchecked(l.chain_off);
            let (chain, rest) = rest.split_at_mut_unchecked(l.hash3_off - l.chain_off);
            let tag = rest.get_unchecked_mut(l.tag_off - l.hash3_off..);
            let tag: &mut [u8] = bytemuck::cast_slice_mut(tag);
            (
                hash.get_unchecked_mut(..l.hash_len),
                chain.get_unchecked_mut(..l.chain_len),
                tag.get_unchecked_mut(..l.tag_len),
            )
        }
    }

    /// `(hashTable, chainTable, tagTable)`, see [`Workspace::tables_mut`].
    #[inline]
    pub fn tables(&self) -> (&[u32], &[u32], &[u8]) {
        let l = self.layout;
        // SAFETY: as in `tables_mut`.
        unsafe {
            let words = self.words();
            let tag: &[u8] = bytemuck::cast_slice(words.get_unchecked(l.tag_off..));
            (
                words.get_unchecked(..l.hash_len),
                words.get_unchecked(l.chain_off..l.chain_off + l.chain_len),
                tag.get_unchecked(..l.tag_len),
            )
        }
    }

    /// `(hashTable, chainTable, hashTable3)` of the opt strategies.
    #[inline]
    pub fn opt_tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u32]) {
        let l = self.layout;
        // SAFETY: as in `tables_mut`.
        unsafe {
            let words = self.words_mut();
            let (hash, rest) = words.split_at_mut_unchecked(l.chain_off);
            let (chain, rest) = rest.split_at_mut_unchecked(l.hash3_off - l.chain_off);
            (
                hash.get_unchecked_mut(..l.hash_len),
                chain.get_unchecked_mut(..l.chain_len),
                rest.get_unchecked_mut(..l.hash3_len),
            )
        }
    }

    /// `hashTable3`, see [`Workspace::opt_tables_mut`].
    #[inline]
    pub fn hash3(&self) -> &[u32] {
        let l = self.layout;
        // SAFETY: as in `tables_mut`.
        unsafe {
            self.words()
                .get_unchecked(l.hash3_off..l.hash3_off + l.hash3_len)
        }
    }
}

/// `ZSTD_WORKSPACETOOLARGE_FACTOR`.
const WORKSPACE_TOO_LARGE_FACTOR: usize = 3;

/// `ZSTD_WORKSPACETOOLARGE_MAXDURATION`.
const WORKSPACE_TOO_LARGE_MAX_DURATION: u32 = 128;

/// `ZSTD_CONTENTSIZE_UNKNOWN`.
const CONTENTSIZE_UNKNOWN: u64 = u64::MAX;

/// `ZSTD_CWKSP_ALIGNMENT_BYTES`.
const CWKSP_ALIGNMENT: usize = 64;

/// `ZSTD_cwksp_aligned64_alloc_size`.
const fn aligned64(size: usize) -> usize {
    size.next_multiple_of(CWKSP_ALIGNMENT)
}

/// `TMP_WORKSPACE_SIZE`: `ENTROPY_WORKSPACE_SIZE`, `(8 << 10) + 512 +
/// 4 * (MaxSeq + 2)`, the larger of it and `ZSTD_SLIPBLOCK_WORKSPACESIZE`.
const TMP_WORKSPACE_SIZE: usize = 8920;

/// `sizeof(ZSTD_compressedBlockState_t)`: the Huffman table's 257
/// `size_t` entries, the FSE tables and the repeat offsets.
const COMPRESSED_BLOCK_STATE_SIZE: usize = if MEM_32BITS { 4596 } else { 5632 };

/// `optPotentialSpace` of `ZSTD_sizeof_matchState`: the optimal parser's
/// frequencies, its `ZSTD_match_t` (8 bytes) and its `ZSTD_optimal_t` (28
/// bytes) arrays.
const OPT_SPACE: usize = aligned64((MAX_ML + 1) * 4)
    + aligned64((MAX_LL + 1) * 4)
    + aligned64((MAX_OFF + 1) * 4)
    + aligned64(256 * 4)
    + aligned64(ZSTD_OPT_SIZE * 8)
    + aligned64(ZSTD_OPT_SIZE * 28);

/// `ZSTD_estimateCCtxSize_usingCCtxParams_internal`, the workspace bytes
/// `ZSTD_resetCCtx_internal` needs, for compression parameters `cparams`
/// with the lazy finder `method`, long distance matching parameters `ldm`
/// and an input of `pledged_src_size` bytes. The context is not static and
/// has no in or out buffer (`ZSTD_compress2` makes both `ZSTD_bm_stable`,
/// a ZSTDMT job is `ZSTDb_not_buffered`), no sequence producer and the
/// default `ZSTD_c_maxBlockSize`.
pub fn needed_space(
    cparams: &CParams,
    method: SearchMethod,
    ldm: Option<&LdmParams>,
    pledged_src_size: u64,
) -> usize {
    let window_size = pledged_src_size.clamp(1, 1 << cparams.window_log) as usize;
    let block_size = ZSTD_BLOCKSIZE_MAX.min(window_size);
    // ZSTD_maxNbSeq
    let max_nb_seq = block_size / if cparams.min_match == 3 { 3 } else { 4 };
    // Literals, `SeqDef` (8 bytes) sequences, their three code arrays.
    let token_space = WILDCOPY_OVERLENGTH + block_size + aligned64(max_nb_seq * 8) + 3 * max_nb_seq;
    // ZSTD_ldm_getTableSize (bucket offsets and 8-byte `ldmEntry_t`s), then
    // ZSTD_ldm_getMaxNbSeq 12-byte `rawSeq`s.
    let ldm_space = ldm.map_or(0, |p| {
        let buckets = 1usize << (p.hash_log - p.bucket_size_log.min(p.hash_log));
        buckets + (8 << p.hash_log) + aligned64(block_size / p.min_match_length as usize * 12)
    });
    TMP_WORKSPACE_SIZE
        + 2 * COMPRESSED_BLOCK_STATE_SIZE
        + ldm_space
        + match_state_size(cparams, method)
        + token_space
}

/// `ZSTD_sizeof_matchState` for a context (`forCCtx`, no dedicated
/// dictionary search). Unlike [`CParams::hash_log3`] it counts the 3-byte
/// hash table for every strategy with `min_match == 3`, as
/// `ZSTD_reset_matchState` reserves it.
fn match_state_size(cparams: &CParams, method: SearchMethod) -> usize {
    // ZSTD_rowMatchFinderUsed
    let row = cparams.row_match_finder_supported() && method == SearchMethod::RowHash;
    // ZSTD_allocateChainTable
    let chain = if cparams.strategy != Strategy::Fast && !row {
        1usize << cparams.chain_log
    } else {
        0
    };
    let hash = 1usize << cparams.hash_log;
    let hash3 = if cparams.min_match == 3 {
        1usize << cparams.window_log.min(ZSTD_HASHLOG3_MAX)
    } else {
        0
    };
    let opt = if cparams.strategy.is_opt() {
        OPT_SPACE
    } else {
        0
    };
    let tags = if row { aligned64(hash) } else { 0 };
    // ZSTD_cwksp_slack_space_required
    let slack = 2 * CWKSP_ALIGNMENT;
    4 * (chain + hash + hash3) + opt + slack + tags
}

/// The positions of a block inside the window, its overflow check done.
/// Only [`MatchState::enter_block`] makes one, and
/// [`MatchState::start_block`] takes it, so no block reaches a finder
/// without entering the window first.
#[derive(Debug)]
pub struct EnteredBlock(Range<usize>);

impl EnteredBlock {
    #[inline]
    pub fn positions(&self) -> Range<usize> {
        self.0.clone()
    }
}

/// The indexed suffix of a raw-content prefix inside the window, more
/// than [`HASH_READ_SIZE`] positions, its overflow check done and
/// `next_to_update` at its start. Only [`MatchState::enter_prefix`] makes
/// one, and the strategies' `load_prefix` take it, so no prefix reaches the
/// tables without entering the window first.
#[derive(Debug)]
pub struct EnteredPrefix(Range<usize>);

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
        Self::new_for(cparams, origin, default_search_method(&cparams))
    }

    /// [`MatchState::new`] with tables for the lazy finder `method`.
    pub fn new_for(cparams: CParams, origin: usize, method: SearchMethod) -> Self {
        let needed = needed_space(&cparams, method, None, CONTENTSIZE_UNKNOWN);
        Self::new_needing(cparams, origin, method, needed)
    }

    /// [`MatchState::new_for`] as the first reset of a context that needs
    /// `needed` bytes of workspace, see [`MatchState::reset_needing`].
    pub fn new_needing(
        cparams: CParams,
        origin: usize,
        method: SearchMethod,
        needed: usize,
    ) -> Self {
        let mut ms = Self {
            cparams,
            ws: Workspace::default(),
            next_to_update: 0,
            window: Window::new(origin, false),
            hash_salt: 0,
            hash_salt_entropy: 0,
            opt: None,
            search_method: method,
        };
        ms.reset_needing(cparams, origin, method, needed);
        ms
    }

    /// [`MatchState::reset_needing`] in a context that holds nothing but
    /// these tables and sizes them for any input: [`needed_space`] without
    /// long distance matching, for `ZSTD_CONTENTSIZE_UNKNOWN`.
    pub fn reset(&mut self, cparams: CParams, origin: usize) {
        self.reset_for(cparams, origin, default_search_method(&cparams))
    }

    /// [`MatchState::reset`] with tables for the lazy finder `method`.
    pub fn reset_for(&mut self, cparams: CParams, origin: usize, method: SearchMethod) {
        let needed = needed_space(&cparams, method, None, CONTENTSIZE_UNKNOWN);
        self.reset_needing(cparams, origin, method, needed);
    }

    /// `ZSTD_resetCCtx_internal` with `ZSTDcrp_makeClean` on a used context
    /// whose reset needs `needed` bytes of workspace ([`needed_space`]), for
    /// an input whose window starts at position `origin`, with tables for
    /// the lazy finder `method`: the one place that decides whether the
    /// workspace is resized (`Workspace::reserve`) and whether indices
    /// continue (`needsIndexReset`). Returns whether the workspace was
    /// resized, which frees the optimal parser's tables here and the
    /// context's other buffers at the caller.
    ///
    /// Indices restart at [`WINDOW_START_INDEX`] (`ZSTDirp_reset`:
    /// `ZSTD_window_init`, every table word zeroed) when the workspace is
    /// resized or the previous input ended too close to [`CURRENT_MAX`]
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
    pub fn reset_needing(
        &mut self,
        cparams: CParams,
        origin: usize,
        method: SearchMethod,
        needed: usize,
    ) -> bool {
        let resized = self.ws.reserve(needed);
        if resized {
            self.opt = None;
        }
        let index_reset = resized || self.window.too_close_to_max();
        self.ws.reset(&cparams, method, index_reset);
        self.search_method = method;
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
        resized
    }

    /// `ZSTD_cwksp_sizeof`: the bytes libzstd's workspace would hold for
    /// this state's context, see [`MatchState::reset_needing`].
    pub fn workspace_size(&self) -> usize {
        self.ws.size
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

    /// Positions `input`, the next of the input, enter the window: the
    /// contiguous `ZSTD_window_update`, then `ZSTD_overflowCorrectIfNeeded`
    /// over `indexed` if given.
    ///
    /// The window's invariant, kept here alone: every position a finder
    /// indexes into the tables, or is handed as a block, lies in
    /// `[low, next_src)` before any table write, and the overflow check
    /// runs for every block whatever its size. `next_src` thus covers every
    /// stored index, so the next reset's `ZSTD_window_clear` puts them all
    /// below the new window and `ZSTD_indexTooCloseToMax` sees where the
    /// input really ended. The only callers are [`MatchState::enter_block`]
    /// and [`MatchState::enter_prefix`], whose [`EnteredBlock`] and
    /// [`EnteredPrefix`] the finders and the strategies' `load_prefix`
    /// require; outside a test, this is the only caller of the private
    /// [`Window::extend_to`] and [`MatchState::correct_overflow_if_needed`].
    fn enter(&mut self, input: Range<usize>, indexed: Option<Range<usize>>) {
        assert!(
            self.window.next_src <= input.start && input.start <= input.end,
            "input {input:?} does not follow the window's end {}",
            self.window.next_src
        );
        self.window.extend_to(input.end);
        if let Some(indexed) = indexed {
            self.correct_overflow_if_needed(indexed);
        }
    }

    /// `ZSTD_compress_frameChunk`'s set-up of the block at `positions`, for
    /// every block whatever its size: the window covers it
    /// (`ZSTD_compressContinue_internal`'s `ZSTD_window_update`) and the
    /// overflow check runs for it (see `MatchState::enter`).
    pub fn enter_block(&mut self, positions: Range<usize>) -> EnteredBlock {
        self.enter(positions.clone(), Some(positions.clone()));
        EnteredBlock(positions)
    }

    /// `ZSTD_loadDictionaryContent` up to its table fill, for the
    /// raw-content prefix at `prefix`: the window covers all of it (see
    /// `MatchState::enter`), and `next_to_update` is the first of the
    /// positions to index, the last
    /// `1 << min(max(hashLog + 3, chainLog + 1), 31)` ("larger than we can
    /// reasonably index in our tables"). At most [`HASH_READ_SIZE`] of them
    /// are left unindexed (`None`, no overflow check and no table write);
    /// more get the overflow check and are returned.
    pub fn enter_prefix(&mut self, prefix: Range<usize>) -> Option<EnteredPrefix> {
        let cp = &self.cparams;
        let max_dict_size = 1usize << (cp.hash_log + 3).max(cp.chain_log + 1).min(31);
        let indexed = prefix.start.max(prefix.end.saturating_sub(max_dict_size))..prefix.end;
        let long = indexed.len() > HASH_READ_SIZE;
        self.enter(prefix, long.then(|| indexed.clone()));
        self.next_to_update = self.index(indexed.start);
        long.then_some(EnteredPrefix(indexed))
    }

    /// The indices of an entered prefix's suffix to index.
    pub fn prefix_indices(&self, prefix: EnteredPrefix) -> Range<usize> {
        self.index(prefix.0.start)..self.index(prefix.0.end)
    }

    /// `ZSTD_overflowCorrectIfNeeded` before the positions `range` (a block,
    /// or a loaded prefix) are indexed: when the window must be corrected
    /// ([`Window::need_overflow_correction`] with `ZSTD_cycleLog` and the
    /// window size), move it ([`Window::correct_overflow`]) and subtract the
    /// correction from every stored index (`ZSTD_reduceIndex`) and from
    /// `next_to_update`.
    fn correct_overflow_if_needed(&mut self, range: Range<usize>) {
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

    /// `ZSTD_buildSeqStore`'s set-up of a block that entered the window
    /// ([`MatchState::enter_block`]): `data[block]` as indices of this
    /// window, after the "limited update after a very long match" clamp:
    /// when the previous block left more than 384 positions uninserted (its
    /// last match ran past the block end), insert at most the 192 positions
    /// before the block (fewer while the backlog is under 576) instead of
    /// the whole backlog.
    pub fn start_block<'a>(&mut self, data: &'a [u8], block: EnteredBlock) -> (Src<'a>, Block) {
        let positions = block.0;
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
            let entered = ms.enter_block(pos..pos);
            let (_, block) = ms.start_block(&data, entered);
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
        let mut ms = MatchState::new(cp, 0);
        ms.enter_block(0..3000);
        ms.enter_block(3000..5000);
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
        let mut ms = MatchState::new(table_params(Strategy::Fast, 10, 0), 0);
        ms.enter_block(0..5000);
        ms.ws.tables_mut().0.fill(77);
        ms.reset(table_params(Strategy::Fast, 11, 0), 0);
        assert_eq!(ms.window_low(), WINDOW_START_INDEX);
        assert_eq!(ms.tables().0.len(), 1 << 11);
        assert!(ms.tables().0.iter().all(|&e| e == 0));
    }

    /// `workspaceWasteful`: a workspace three times a reset's need is kept,
    /// indices continuing, until the first reset past 128 since it was
    /// sized, which frees it and allocates the tables at their size.
    #[test]
    fn reset_shrinks_a_wasteful_workspace() {
        let big = CParams::for_level(19, 64 << 20);
        let small = CParams::for_level(19, 1024);
        let method = default_search_method(&big);
        let needed = needed_space(&big, method, None, 64 << 20);
        let mut ms = MatchState::new_needing(big, 0, method, needed);
        let big_pages = ms.ws.pages.len();
        let needed = needed_space(&small, method, None, 1024);
        assert!(3 * needed <= ms.workspace_size());
        for n in 1..=130 {
            ms.enter_block(0..1000);
            let resized = ms.reset_needing(small, 0, method, needed);
            assert_eq!(resized, n == 129, "reset {n}");
            let low = if n < 129 { 1000 * n } else { 1000 * (n - 129) };
            assert_eq!(ms.window_low(), WINDOW_START_INDEX + low, "reset {n}");
            let pages = Layout::of(&small, method).words() / PAGE_WORDS;
            assert_eq!(
                ms.ws.pages.len(),
                if n < 129 { big_pages } else { pages },
                "reset {n}"
            );
            assert!(ms.opt.is_some());
        }
        assert_eq!(ms.workspace_size(), needed);
    }

    /// `tableValidEnd`: index tables laid over words that held indices are
    /// not zeroed, whatever table held them; words past the reduced tables
    /// of a correction, words that held tags, padding between index tables
    /// and every word after an index reset are.
    #[test]
    fn workspace_zeroes_only_words_it_cannot_vouch_for() {
        const HC: SearchMethod = SearchMethod::HashChain;
        let big = table_params(Strategy::DFast, 12, 12);
        let small = table_params(Strategy::Fast, 10, 0);
        let rows = table_params(Strategy::Lazy, 11, 11);
        let padded = table_params(Strategy::DFast, 9, 9);
        let mut ws = Workspace::default();
        ws.reset(&big, HC, true);
        assert_eq!(ws.words().len(), 8192);
        ws.words_mut().fill(7);
        ws.reset(&small, HC, false);
        ws.reset(&big, HC, false);
        assert!(ws.words().iter().all(|&w| w == 7));
        // The correction reduces the small tables only.
        ws.reset(&small, HC, false);
        ws.reduce(5, false);
        ws.reset(&big, HC, false);
        assert!(ws.words()[..1024].iter().all(|&w| w == 2));
        assert!(ws.words()[1024..].iter().all(|&w| w == 0));
        // The row finder's index area is its 2048-word hash table (no
        // chain table), its tags follow on the next page.
        ws.words_mut().fill(7);
        ws.reset(&rows, SearchMethod::RowHash, false);
        ws.tables_mut().2.fill(9);
        ws.reset(&big, HC, false);
        assert!(ws.words()[..2048].iter().all(|&w| w == 7));
        assert!(ws.words()[2048..].iter().all(|&w| w == 0));
        // Two 512-word tables, each padded to a page.
        ws.words_mut().fill(7);
        ws.reset(&padded, HC, false);
        let w = ws.words();
        assert!(w[..512].iter().chain(&w[1024..1536]).all(|&w| w == 7));
        assert!(w[512..1024].iter().chain(&w[1536..2048]).all(|&w| w == 0));
        ws.words_mut().fill(7);
        ws.reset(&small, HC, true);
        assert!(ws.words()[..1024].iter().all(|&w| w == 0));
        assert!(ws.words()[1024..].iter().all(|&w| w == 7));
    }

    /// Every table of every strategy starts on a page boundary, fresh and
    /// after a reset that reuses a larger allocation, so every row-finder
    /// row (`1 << row_log` entries, `row_log` = `BOUNDED(4, search_log, 6)`)
    /// of `hashTable` and `tagTable` lies within one page.
    #[test]
    fn tables_are_page_aligned_and_rows_stay_in_a_page() {
        let check = |ms: &MatchState, level: i32| {
            let (hash, chain, tag) = ms.ws.tables();
            let h3 = ms.ws.hash3();
            for (name, p) in [
                ("hash", hash.as_ptr() as usize),
                ("chain", chain.as_ptr() as usize),
                ("tag", tag.as_ptr() as usize),
                ("hash3", h3.as_ptr() as usize),
            ] {
                assert_eq!(p % PAGE, 0, "level {level} {name} table at {p:#x}");
            }
            if !tag.is_empty() {
                let row = 1usize << ms.cparams.search_log.clamp(4, 6);
                let page = |p: usize| p / PAGE;
                for first in (0..tag.len()).step_by(row) {
                    let h = hash[first..first + row].as_ptr_range();
                    assert_eq!(page(h.start as usize), page(h.end as usize - 1));
                    let t = tag[first..first + row].as_ptr_range();
                    assert_eq!(page(t.start as usize), page(t.end as usize - 1));
                }
            }
        };
        let mut reused = MatchState::new(CParams::for_level(22, 1 << 20), 0);
        // One size per `clevels.h` table: small tables are smaller than a page.
        for size in [1 << 10, 100 << 10, 200 << 10, 1 << 20] {
            for level in 1..=22 {
                let cp = CParams::for_level(level, size);
                check(&MatchState::new(cp, 0), level);
                reused.reset(cp, 0);
                check(&reused, level);
            }
        }
    }

    /// `ZSTD_allocateChainTable`: the row finder gets a tag table and no
    /// chain table, the hash chain a chain table and no tag table, whichever
    /// finder the state held before.
    #[test]
    fn lazy_tables_follow_the_search_method() {
        let lens = |ms: &MatchState| {
            let (hash, chain, tag) = ms.tables();
            (hash.len(), chain.len(), tag.len())
        };
        // 1 KiB inputs get windowLog <= 14, where the hash chain is the default.
        for size in [1 << 10, 1 << 20] {
            for level in 3..=15 {
                let cp = CParams::for_level(level, size);
                if !cp.row_match_finder_supported() {
                    continue;
                }
                let (hash, chain) = (1 << cp.hash_log, 1 << cp.chain_log);
                let row = (hash, 0, hash);
                let hc = (hash, chain, 0);
                let default = MatchState::new(cp, 0);
                let expect = match default.search_method {
                    SearchMethod::RowHash => row,
                    _ => hc,
                };
                assert_eq!(lens(&default), expect, "level {level} size {size}");
                let mut ms = MatchState::new_for(cp, 0, SearchMethod::RowHash);
                assert_eq!(lens(&ms), row);
                ms.reset_for(cp, 0, SearchMethod::HashChain);
                assert_eq!(lens(&ms), hc);
                ms.reset_for(cp, 0, SearchMethod::RowHash);
                assert_eq!(lens(&ms), row);
            }
        }
    }
}
