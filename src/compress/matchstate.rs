//! Persistent match-finder state: port of `ZSTD_MatchState_t`
//! (zstd_compress_internal.h) for the no-dictionary case.
//!
//! Positions are absolute indices into the source slice handed to the block
//! functions and are stored as `u32`, so a job's source must be smaller than
//! 4 GiB. Table entry `0` means "empty"; `window_low >= 1` guarantees that no
//! real position is ever `0` (libzstd achieves the same by starting indices
//! at `ZSTD_WINDOW_START_INDEX` and skipping the first prefix byte with
//! `ip0 += (ip0 == prefixStart)`).
//!
//! A candidate index `c` is usable at position `cur` only if
//! `c >= window_low` and `cur - c <= (1 << window_log)`; see
//! [`MatchState::lowest_prefix_index`].
//!
//! The tables share one allocation ([`Workspace`], the table area of
//! `ZSTD_cwksp`), so the allocator sees one request per context rather
//! than three that straddle glibc's dynamic mmap threshold.

use super::opt::OptState;
use super::params::{CParams, Strategy};

pub struct MatchState {
    pub cparams: CParams,
    /// `hashTable`, `chainTable` and `tagTable`, see [`Workspace::tables_mut`].
    /// Strategies borrow them as `ms.ws.tables_mut()`, which leaves the
    /// other fields accessible while the slices are live.
    pub ws: Workspace,
    /// `nextToUpdate`: index from which table insertion resumes.
    pub next_to_update: usize,
    /// `window.dictLimit` / `window.lowLimit`: lowest valid index (`>= 1`).
    pub window_low: usize,
    /// `hashSalt`: salt of the row-based finder's hash (`ZSTD_hashPtrSalted`),
    /// so that a reused tag table does not produce phantom matches. Starts
    /// at the value a fresh `ZSTD_CCtx` has after its first
    /// `ZSTD_advanceHashSalt` (both inputs zero), see
    /// [`super::lazy::initial_hash_salt`].
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
/// the hash-chain table for the lazy strategies; the binary tree for the
/// opt strategies), `hashTable3` (`1 << hash_log3` entries, opt strategies
/// with `min_match == 3` only) and `tagTable` (`1 << hash_log` bytes,
/// row-based lazy finder only, else empty).
#[derive(Default)]
pub struct Workspace {
    words: Vec<u32>,
    hash_len: usize,
    chain_len: usize,
    hash3_len: usize,
    tag_len: usize,
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
        if self.words.capacity() < words {
            // ZSTD_cwksp_free before ZSTD_cwksp_create: never both at once.
            drop(std::mem::take(&mut self.words));
            self.words = vec![0; words];
        } else {
            self.words.clear();
            self.words.resize(words, 0);
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
        // SAFETY: `reset` is the only writer of the lengths and sizes `words`
        // to exactly `hash_len + chain_len + hash3_len + tag_len.div_ceil(4)`.
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

impl MatchState {
    /// Allocate zeroed tables for `cparams.strategy`; positions below
    /// `window_low` (which must be `>= 1`) are never referenced.
    pub fn new(cparams: CParams, window_low: usize) -> Self {
        let mut ms = Self {
            cparams,
            ws: Workspace::default(),
            next_to_update: 0,
            window_low: 0,
            hash_salt: 0,
            hash_salt_entropy: 0,
            opt: None,
        };
        ms.reset(cparams, window_low);
        ms
    }

    /// `ZSTD_reset_matchState` with `ZSTDcrp_makeClean` on a reused context:
    /// size the tables for `cparams`, keeping an allocation that is large
    /// enough and zeroing it (`ZSTD_cwksp_clean_tables`), and start over at
    /// `window_low`. The result equals [`MatchState::new`], so a reused
    /// context compresses identically.
    ///
    /// Deviation: libzstd keeps the row finder's stale tag table and
    /// advances the hash salt instead (`ZSTD_advanceHashSalt`,
    /// `ZSTD_cwksp_reserve_aligned_init_once`), which makes the frame depend
    /// on the context's history; the tag table is cleared here and the salt
    /// stays at its initial value.
    pub fn reset(&mut self, cparams: CParams, window_low: usize) {
        assert!(
            window_low >= 1,
            "window_low must be >= 1 (0 marks an empty table entry)"
        );
        self.ws.reset(&cparams);
        self.cparams = cparams;
        self.next_to_update = window_low;
        self.window_low = window_low;
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
    /// the lowest index a match may reference from position `cur`.
    #[inline]
    pub fn lowest_prefix_index(&self, cur: usize) -> usize {
        let max_distance = 1usize << self.cparams.window_log;
        // C: `curr - lowestValid > maxDistance`, rearranged so that it is
        // also total for `cur < window_low` (block 0 starts at position 0).
        if cur > self.window_low + max_distance {
            cur - max_distance
        } else {
            self.window_low
        }
    }
}
