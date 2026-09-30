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
/// empty), in that order.
///
/// Every table starts on a 4 KiB page boundary: the allocation is of
/// `Page`s and each table fills whole pages. A row of the row finder
/// (`1 << row_log` entries at a multiple of its size, at most 256 bytes of
/// `hashTable` and 64 of `tagTable`) therefore never crosses a page or a
/// cache line. `ZSTD_cwksp` only guarantees 64-byte alignment
/// (`ZSTD_CWKSP_ALIGNMENT_BYTES`); where its tables land within a page
/// depends on the sizes of the objects reserved before them.
#[derive(Default)]
pub struct Workspace {
    pages: Vec<Page>,
    hash_len: usize,
    chain_off: usize,
    chain_len: usize,
    hash3_off: usize,
    hash3_len: usize,
    tag_off: usize,
    tag_len: usize,
}

/// The page size tables are aligned to.
const PAGE: usize = 4096;

/// `u32` entries per [`PAGE`] bytes.
const PAGE_WORDS: usize = PAGE / 4;

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

    /// `(hash, chain, hash3, tag)` lengths for `cparams.strategy`: entries,
    /// bytes for `tag`.
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
        let whole = |words: usize| words.next_multiple_of(PAGE_WORDS);
        self.hash_len = hash_len;
        self.chain_off = whole(hash_len);
        self.chain_len = chain_len;
        self.hash3_off = self.chain_off + whole(chain_len);
        self.hash3_len = hash3_len;
        self.tag_off = self.hash3_off + whole(hash3_len);
        self.tag_len = tag_len;
        let pages = (self.tag_off + whole(tag_len.div_ceil(4))) / PAGE_WORDS;
        if self.pages.capacity() < pages {
            // ZSTD_cwksp_free before ZSTD_cwksp_create: never both at once.
            drop(std::mem::take(&mut self.pages));
            self.pages = zeroed_pages(pages);
        } else {
            self.pages.clear();
            // SAFETY: `pages <= capacity`, and zeroed memory is valid `Page`s
            // (`Zeroable`). `write_bytes` is glibc's memset, as in
            // `ZSTD_cwksp_clean_tables`; `resize` compiles to a store loop.
            unsafe {
                self.pages.as_mut_ptr().write_bytes(0, pages);
                self.pages.set_len(pages);
            }
        }
    }

    /// `(hashTable, chainTable, tagTable)`. Unchecked splits: the bounds
    /// checks of `split_at_mut` at a strategy's entry re-allocate the
    /// registers of its whole hot loop (fast L1 measured 7% slower).
    #[inline]
    pub fn tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u8]) {
        // SAFETY: `reset` is the only writer of the offsets and lengths and
        // sizes `pages` to hold `tag_off + tag_len.div_ceil(4)` entries, with
        // `hash_len <= chain_off`, `chain_off + chain_len <= hash3_off <=
        // tag_off`.
        unsafe {
            let words = bytemuck::cast_slice_mut::<Page, u32>(&mut self.pages);
            let (hash, rest) = words.split_at_mut_unchecked(self.chain_off);
            let (chain, rest) = rest.split_at_mut_unchecked(self.hash3_off - self.chain_off);
            let tag = rest.get_unchecked_mut(self.tag_off - self.hash3_off..);
            let tag: &mut [u8] = bytemuck::cast_slice_mut(tag);
            (
                hash.get_unchecked_mut(..self.hash_len),
                chain.get_unchecked_mut(..self.chain_len),
                tag.get_unchecked_mut(..self.tag_len),
            )
        }
    }

    /// `(hashTable, chainTable, tagTable)`, see [`Workspace::tables_mut`].
    #[inline]
    pub fn tables(&self) -> (&[u32], &[u32], &[u8]) {
        // SAFETY: as in `tables_mut`.
        unsafe {
            let words = self.words();
            let tag: &[u8] = bytemuck::cast_slice(words.get_unchecked(self.tag_off..));
            (
                words.get_unchecked(..self.hash_len),
                words.get_unchecked(self.chain_off..self.chain_off + self.chain_len),
                tag.get_unchecked(..self.tag_len),
            )
        }
    }

    /// `(hashTable, chainTable, hashTable3)` of the opt strategies.
    #[inline]
    pub fn opt_tables_mut(&mut self) -> (&mut [u32], &mut [u32], &mut [u32]) {
        // SAFETY: as in `tables_mut`.
        unsafe {
            let words = bytemuck::cast_slice_mut::<Page, u32>(&mut self.pages);
            let (hash, rest) = words.split_at_mut_unchecked(self.chain_off);
            let (chain, rest) = rest.split_at_mut_unchecked(self.hash3_off - self.chain_off);
            (
                hash.get_unchecked_mut(..self.hash_len),
                chain.get_unchecked_mut(..self.chain_len),
                rest.get_unchecked_mut(..self.hash3_len),
            )
        }
    }

    /// `hashTable3`, see [`Workspace::opt_tables_mut`].
    #[inline]
    pub fn hash3(&self) -> &[u32] {
        // SAFETY: as in `tables_mut`.
        unsafe {
            self.words()
                .get_unchecked(self.hash3_off..self.hash3_off + self.hash3_len)
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
}
