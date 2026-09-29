//! Greedy / lazy / lazy2 block compressors (`zstd_lazy.c`).
//!
//! Not ported yet: delegates to [`super::fast`]. The `MatchState` already
//! carries `hash_table`, `chain_table` and `tag_table` for the hash-chain
//! and row-based finders.

use super::matchstate::MatchState;
use super::seqstore::SeqStore;
use std::ops::Range;

/// See [`super::fast::compress_block`].
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    super::fast::compress_block(ms, src, block, rep, out)
}

/// See [`super::fast::load_prefix`] (`ZSTD_insertAndFindFirstIndex` /
/// `ZSTD_row_update` analogue).
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    super::fast::load_prefix(ms, src, range)
}
