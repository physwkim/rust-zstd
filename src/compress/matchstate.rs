//! Persistent match-finder state: port of `ZSTD_MatchState_t`
//! (zstd_compress_internal.h) for the no-dictionary case.
//!
//! The finders address the input by index, as libzstd does through
//! `window.base`: the window's first byte (position `origin` of the input
//! slice) is index [`WINDOW_START_INDEX`], so index `0` can mean "empty"
//! in a table and `1` is `ZSTD_DUBT_UNSORTED_MARK`. [`MatchState`] is the
//! only owner of the position <-> index mapping ([`MatchState::index`],
//! [`MatchState::pos`], [`MatchState::view`]). Indices are stored as `u32`,
//! so a job's window must stay below 4 GiB ([`MatchState::view`] asserts
//! it; libzstd would correct overflow instead).
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
use super::opt::OptState;
use super::params::{CParams, Strategy};

/// `ZSTD_WINDOW_START_INDEX`: the index of a window's first byte.
pub const WINDOW_START_INDEX: usize = 2;

pub struct MatchState {
    pub cparams: CParams,
    /// `hashTable`, `chainTable` and `tagTable`, see [`Workspace::tables_mut`].
    /// Strategies borrow them as `ms.ws.tables_mut()`, which leaves the
    /// other fields accessible while the slices are live.
    pub ws: Workspace,
    /// `nextToUpdate`: index from which table insertion resumes.
    pub next_to_update: usize,
    /// `window.dictLimit` / `window.lowLimit`: lowest valid index.
    pub window_low: usize,
    /// `window.base` as a position of the input slice: the position of
    /// index 0 (wrapping, it lies before the input).
    base: usize,
    /// Position of the window's first byte, the lowest readable one.
    origin: usize,
    /// `hashSalt`: salt of the row-based finder's hash (`ZSTD_hashPtrSalted`),
    /// so that a reused tag table does not produce phantom matches. Starts
    /// at the value a fresh `ZSTD_CCtx` has after its first
    /// `ZSTD_advanceHashSalt` (both inputs zero), see
    /// `lazy::initial_hash_salt`.
    pub hash_salt: u64,
    /// `hashSaltEntropy`: running sum of the row finder's search hashes,
    /// mixed into the next salt by `ZSTD_advanceHashSalt` on a context reset.
    pub hash_salt_entropy: u32,
    /// `opt`: the optimal parser's statistics and work tables, allocated
    /// for the opt strategies and kept (not shrunk) across resets.
    pub opt: Option<Box<OptState>>,
}

/// The table area of `ZSTD_cwksp`: one zeroed allocation holding
/// `hashTable` (`1 << hash_log` entries, every strategy), `chainTable`
/// (`1 << chain_log` entries; empty for `Fast`; `hashSmall` for `DFast`;
/// the hash-chain table for the lazy strategies; the binary tree, two
/// entries per node, for `BtLazy2` and the opt strategies), `hashTable3`
/// (`1 << hash_log3` entries, opt strategies with `min_match == 3` only)
/// and `tagTable` (`1 << hash_log` bytes, row-based lazy finder only, else
/// empty). Every table starts on a 64-byte boundary, as `ZSTD_cwksp`
/// places them (`ZSTD_CWKSP_ALIGNMENT_BYTES`): the allocation is of
/// 64-byte `Line`s and every table length is a multiple of 16 entries
/// (`hash_log`, `chain_log` and `hash_log3` are at least 6).
#[derive(Default)]
pub struct Workspace {
    lines: Vec<Line>,
    hash_len: usize,
    chain_len: usize,
    hash3_len: usize,
    tag_len: usize,
}

/// One 64-byte cache line of table entries: the allocation unit of
/// [`Workspace`], so that its tables are line-aligned.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u32; 16]);

// SAFETY: 16 `u32`s are 64 bytes, the alignment, so `Line` has no padding
// and every bit pattern is valid.
unsafe impl bytemuck::Zeroable for Line {}
unsafe impl bytemuck::Pod for Line {}

/// `n` zeroed `Line`s through `alloc_zeroed` (as `vec![0u32; n]` does), so
/// fresh tables stay untouched zero pages.
fn zeroed_lines(n: usize) -> Vec<Line> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<Line>(n).expect("workspace size");
    // SAFETY: `layout` is not zero-sized; zeroed memory is `n` valid `Line`s
    // (`Zeroable`), allocated with the layout `Vec<Line>` frees it with.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout).cast::<Line>();
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
        bytemuck::cast_slice(&self.lines)
    }

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

    /// Size for `cparams`: an allocation that is large enough is kept and
    /// exactly its used range zeroed (`ZSTD_cwksp_clean_tables`), else it
    /// is freed and a zeroed one allocated.
    fn reset(&mut self, cparams: &CParams) {
        let (hash_len, chain_len, hash3_len, tag_len) = Self::lens(cparams);
        let words = hash_len + chain_len + hash3_len + tag_len.div_ceil(4);
        debug_assert!((hash_len | chain_len | hash3_len) % 16 == 0);
        let lines = words.div_ceil(16);
        if self.lines.capacity() < lines {
            // ZSTD_cwksp_free before ZSTD_cwksp_create: never both at once.
            drop(std::mem::take(&mut self.lines));
            self.lines = zeroed_lines(lines);
        } else {
            self.lines.clear();
            // SAFETY: `lines <= capacity`, and zeroed memory is valid `Line`s
            // (`Zeroable`). `write_bytes` is glibc's memset, as in
            // `ZSTD_cwksp_clean_tables`; `resize` compiles to a store loop.
            unsafe {
                self.lines.as_mut_ptr().write_bytes(0, lines);
                self.lines.set_len(lines);
            }
        }
        self.hash_len = hash_len;
        self.chain_len = chain_len;
        self.hash3_len = hash3_len;
        self.tag_len = tag_len;
    }

    /// `(hashTable, chainTable, tagTable)`. Unchecked splits: the bounds
    /// checks of `split_at_mut` at a strategy's entry re-allocate the
    /// registers of its whole hot loop (fast L1 measured 7% slower).
    #[inline]
    pub fn tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u8]) {
        // SAFETY: `reset` is the only writer of the lengths and sizes `lines`
        // to hold `hash_len + chain_len + hash3_len + tag_len.div_ceil(4)`
        // entries.
        unsafe {
            let (hash, rest) = bytemuck::cast_slice_mut::<Line, u32>(&mut self.lines)
                .split_at_mut_unchecked(self.hash_len);
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
            let (hash, rest) = self.words().split_at_unchecked(self.hash_len);
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
            let (hash, rest) = bytemuck::cast_slice_mut::<Line, u32>(&mut self.lines)
                .split_at_mut_unchecked(self.hash_len);
            let (chain, rest) = rest.split_at_mut_unchecked(self.chain_len);
            (hash, chain, rest.get_unchecked_mut(..self.hash3_len))
        }
    }

    /// `hashTable3`, see [`Workspace::opt_tables_mut`].
    #[inline]
    pub fn hash3(&self) -> &[u32] {
        // SAFETY: as in `tables_mut`.
        unsafe {
            self.words()
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
    /// position `origin`, see [`MatchState::reset`].
    pub fn new(cparams: CParams, origin: usize) -> Self {
        let mut ms = Self {
            cparams,
            ws: Workspace::default(),
            next_to_update: 0,
            window_low: 0,
            base: 0,
            origin: 0,
            hash_salt: 0,
            hash_salt_entropy: 0,
            opt: None,
        };
        ms.reset(cparams, origin);
        ms
    }

    /// `ZSTD_reset_matchState` with `ZSTDcrp_makeClean` on a reused context:
    /// size the tables for `cparams`, keeping an allocation that is large
    /// enough and zeroing it (`ZSTD_cwksp_clean_tables`), and start a window
    /// whose first byte is position `origin` of the input, at index
    /// [`WINDOW_START_INDEX`] (`ZSTD_window_init`, then the non-contiguous
    /// `ZSTD_window_update` of the job's first input). The result equals
    /// [`MatchState::new`], so a reused context compresses identically.
    ///
    /// Deviation: libzstd keeps the row finder's stale tag table and
    /// advances the hash salt instead (`ZSTD_advanceHashSalt`,
    /// `ZSTD_cwksp_reserve_aligned_init_once`), which makes the frame depend
    /// on the context's history; the tag table is cleared here and the salt
    /// stays at its initial value.
    pub fn reset(&mut self, cparams: CParams, origin: usize) {
        self.ws.reset(&cparams);
        self.cparams = cparams;
        self.origin = origin;
        self.base = origin.wrapping_sub(WINDOW_START_INDEX);
        self.next_to_update = WINDOW_START_INDEX;
        self.window_low = WINDOW_START_INDEX;
        self.hash_salt = super::lazy::initial_hash_salt();
        self.hash_salt_entropy = 0;
        // ZSTD_invalidateMatchState: `opt.litLengthSum = 0` forces the next
        // opt block to initialize its statistics.
        if cparams.strategy.is_opt() {
            self.opt
                .get_or_insert_with(|| Box::new(OptState::new()))
                .invalidate();
        }
    }

    /// The index of position `pos` (`pos >= origin`).
    #[inline]
    pub fn index(&self, pos: usize) -> usize {
        debug_assert!(pos >= self.origin);
        pos.wrapping_sub(self.base)
    }

    /// The position of index `idx`.
    #[inline]
    pub fn pos(&self, idx: usize) -> usize {
        idx.wrapping_add(self.base)
    }

    /// `data[origin..]` addressed by index. Panics if an index of `data`
    /// does not fit the `u32` tables.
    #[inline]
    pub fn view<'a>(&self, data: &'a [u8]) -> Src<'a> {
        let src = Src::new(data, self.origin, self.index(self.origin));
        assert!(
            u32::try_from(src.end() - 1).is_ok(),
            "window of {} bytes exceeds the u32 index space",
            data.len() - self.origin
        );
        src
    }

    /// `ZSTD_buildSeqStore`'s set-up of a block: `data[positions]` as
    /// indices of this window, after the "limited update after a very long
    /// match" clamp: when the previous block left more than 384 positions
    /// uninserted (its last match ran past the block end), insert at most
    /// the 192 positions before the block (fewer while the backlog is under
    /// 576) instead of the whole backlog.
    pub fn start_block<'a>(&mut self, data: &'a [u8], positions: Range<usize>) -> (Src<'a>, Block) {
        let src = self.view(data);
        let (start, end) = (self.index(positions.start), self.index(positions.end));
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
        debug_assert_eq!(block.start, self.window_low);
        self.base = self.base.wrapping_sub(len);
        self.window_low += len;
        self.next_to_update = self.window_low;
        let src = src.rebased(len);
        assert!(
            u32::try_from(src.end() - 1).is_ok(),
            "window moved past the u32 index space"
        );
        (src, block.start + len..block.end + len)
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
        debug_assert!(cur >= self.window_low);
        // C: `curr - lowestValid > maxDistance`
        if cur - self.window_low > max_distance {
            cur - max_distance
        } else {
            self.window_low
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

    /// Every table of every strategy starts on a 64-byte boundary, fresh
    /// and after a reset that reuses a larger allocation.
    #[test]
    fn tables_are_line_aligned() {
        let aligned = |ms: &mut MatchState, level: i32| {
            let (hash, chain, tag) = ms.ws.tables();
            let (h, c, t) = (
                hash.as_ptr() as usize,
                chain.as_ptr() as usize,
                tag.as_ptr() as usize,
            );
            let h3 = ms.ws.hash3().as_ptr() as usize;
            for (name, p) in [("hash", h), ("chain", c), ("tag", t), ("hash3", h3)] {
                assert_eq!(p % 64, 0, "level {level} {name} table at {p:#x}");
            }
        };
        let mut reused = MatchState::new(CParams::for_level(22, 1 << 20), 0);
        for level in 1..=22 {
            let cp = CParams::for_level(level, 1 << 20);
            aligned(&mut MatchState::new(cp, 0), level);
            reused.reset(cp, 0);
            aligned(&mut reused, level);
        }
    }
}
