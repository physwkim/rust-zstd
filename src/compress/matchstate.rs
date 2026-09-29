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

use super::params::{CParams, Strategy};

pub struct MatchState {
    pub cparams: CParams,
    /// `hashTable`: `1 << hash_log` entries. Every strategy.
    pub hash_table: Vec<u32>,
    /// `chainTable`: `1 << chain_log` entries. Empty for `Fast`. For `DFast`
    /// zstd_double_fast.c uses `hash_table` as `hashLong` and this table as
    /// `hashSmall`. For the lazy strategies it is the hash-chain table.
    pub chain_table: Vec<u32>,
    /// `tagTable`: `1 << hash_log` bytes, for the row-based lazy match finder.
    /// Empty unless the strategy is `Greedy`/`Lazy`/`Lazy2`.
    pub tag_table: Vec<u8>,
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
}

impl MatchState {
    /// Allocate zeroed tables for `cparams.strategy`; positions below
    /// `window_low` (which must be `>= 1`) are never referenced.
    pub fn new(cparams: CParams, window_low: usize) -> Self {
        assert!(
            window_low >= 1,
            "window_low must be >= 1 (0 marks an empty table entry)"
        );
        let hash_size = 1usize << cparams.hash_log;
        let chain_size = 1usize << cparams.chain_log;
        let (chain_table, tag_table) = match cparams.strategy {
            Strategy::Fast => (Vec::new(), Vec::new()),
            Strategy::DFast => (vec![0u32; chain_size], Vec::new()),
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => {
                (vec![0u32; chain_size], vec![0u8; hash_size])
            }
        };
        Self {
            cparams,
            hash_table: vec![0u32; hash_size],
            chain_table,
            tag_table,
            next_to_update: window_low,
            window_low,
            hash_salt: super::lazy::initial_hash_salt(),
            hash_salt_entropy: 0,
        }
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
