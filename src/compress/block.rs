//! One block: `ZSTD_compressBlock_internal` plus the block-type decision of
//! `ZSTD_compress_frameChunk`, and the cross-block state
//! (`ZSTD_blockState_t`) with its commit rule.
//!
//! State-owner invariant: the decoder's repeat offsets and entropy tables
//! change only when a COMPRESSED block is emitted. [`CommittedBlockState`]
//! keeps that state in a private field; the entropy stage reads it through
//! [`CommittedBlockState::prev`] and produces a fresh [`BlockState`], and the
//! private `commit` in [`compress_block`] is the only path that installs it
//! (`ZSTD_blockState_confirmRepcodesAndEntropyTables`). RAW and RLE blocks
//! discard the candidate, including its repeat offsets.

use super::matchstate::MatchState;
use super::params::{CParams, Strategy};
use super::seqstore::SeqStore;
use super::{dfast, fast, lazy};
use crate::constants::*;
use crate::fse::{self, FseState};
use crate::huf::{self, HufState};
use std::ops::Range;

/// `ZSTD_blockHeaderSize`.
pub const ZSTD_BLOCKHEADERSIZE: usize = 3;
/// `MIN_CBLOCK_SIZE`: 1 (literals header) + 1 (RLE or RAW).
pub const MIN_CBLOCK_SIZE: usize = 2;
/// `rleMaxLength` in `ZSTD_compressBlock_internal`.
pub const RLE_MAX_LENGTH: usize = 25;

/// `ZSTD_compressedBlockState_t`: what the decoder holds after the last
/// COMPRESSED block. `Treeless` literals and `Repeat` sequence tables may
/// only reference the committed instance.
#[derive(Clone, Debug, Default)]
pub struct BlockState {
    pub rep: [u32; 3],
    pub huf: HufState,
    pub fse: FseState,
}

impl BlockState {
    /// `ZSTD_reset_compressedBlockState`: `repStartValue`, no entropy tables.
    pub fn initial() -> Self {
        BlockState {
            rep: [1, 4, 8],
            huf: HufState::None,
            fse: FseState::default(),
        }
    }

    /// `ZSTD_invalidateRepCodes`: a job after the first one does not know
    /// the repeat offsets the decoder holds, so no repcode may be emitted
    /// until a real offset has replaced the zero.
    pub fn invalidate_rep_codes(&mut self) {
        self.rep = [0; 3];
    }
}

/// `prevCBlock`: the committed cross-block state. Replaced only by the
/// private `commit`, exactly when a COMPRESSED block is written.
pub struct CommittedBlockState {
    prev: BlockState,
}

impl CommittedBlockState {
    pub fn new(prev: BlockState) -> Self {
        Self { prev }
    }

    /// The state the decoder holds right now.
    pub fn prev(&self) -> &BlockState {
        &self.prev
    }

    /// `ZSTD_blockState_confirmRepcodesAndEntropyTables`.
    fn commit(&mut self, next: BlockState) {
        self.prev = next;
    }
}

/// Reusable per-block buffers.
#[derive(Default)]
pub struct BlockScratch {
    pub store: SeqStore,
    pub cbuf: Vec<u8>,
}

impl BlockScratch {
    pub fn new(block_size: usize) -> Self {
        let mut scratch = Self::default();
        scratch.reserve(block_size);
        scratch
    }

    /// Grow to [`BlockScratch::new`]'s sizes for `block_size` if smaller;
    /// existing allocations are kept.
    pub fn reserve(&mut self, block_size: usize) {
        self.store.reserve(block_size);
        self.cbuf
            .reserve_exact(block_size.saturating_sub(self.cbuf.len()));
    }
}

/// Block type written by [`compress_block`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Raw,
    Rle,
    Compressed,
}

/// `ZSTD_isRLE`: every byte of `data` equals the first one.
pub fn is_rle(data: &[u8]) -> bool {
    match data.split_first() {
        Some((&first, rest)) => rest.iter().all(|&b| b == first),
        None => false,
    }
}

/// `ZSTD_noCompressBlock`.
pub fn write_raw_block(out: &mut Vec<u8>, data: &[u8], is_last: bool) {
    let header = (is_last as u32) | ((BLOCK_TYPE_RAW as u32) << 1) | ((data.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..ZSTD_BLOCKHEADERSIZE]);
    out.extend_from_slice(data);
}

/// `ZSTD_rleCompressBlock`.
pub fn write_rle_block(out: &mut Vec<u8>, byte: u8, repeat_count: usize, is_last: bool) {
    let header = (is_last as u32) | ((BLOCK_TYPE_RLE as u32) << 1) | ((repeat_count as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..ZSTD_BLOCKHEADERSIZE]);
    out.push(byte);
}

/// A COMPRESSED block: header then `compressed`.
pub fn write_compressed_block(out: &mut Vec<u8>, compressed: &[u8], is_last: bool) {
    let header =
        (is_last as u32) | ((BLOCK_TYPE_COMPRESSED as u32) << 1) | ((compressed.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..ZSTD_BLOCKHEADERSIZE]);
    out.extend_from_slice(compressed);
}

/// `ZSTD_loadDictionaryContent` for a raw-content prefix: index
/// `src[range]` into the strategy's tables before the first block of a job.
/// Only the last `1 << min(max(hashLog + 3, chainLog + 1), 31)` bytes are
/// indexed ("larger than we can reasonably index in our tables"); matches
/// may still reach the whole prefix, which `window_low` keeps valid.
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    let cp = &ms.cparams;
    let max_dict_size = 1usize << (cp.hash_log + 3).max(cp.chain_log + 1).min(31);
    let range = range.start.max(range.end.saturating_sub(max_dict_size))..range.end;
    match ms.cparams.strategy {
        Strategy::Fast => fast::load_prefix(ms, src, range),
        Strategy::DFast => dfast::load_prefix(ms, src, range),
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => lazy::load_prefix(ms, src, range),
    }
}

/// `ZSTD_buildSeqStore`, "limited update after a very long match": when the
/// previous block left more than 384 positions uninserted (its last match ran
/// past the block end), insert at most the 192 positions before `curr` (fewer
/// while the backlog is under 576) instead of the whole backlog.
#[inline]
fn limit_update_after_long_match(ms: &mut MatchState, curr: usize) {
    if curr > ms.next_to_update + 384 {
        ms.next_to_update = curr - 192.min(curr - ms.next_to_update - 384);
    }
}

/// `ZSTD_buildSeqStore` for a block worth compressing: reset `store`, apply
/// the nextToUpdate clamp, run the strategy's block compressor and store the
/// trailing literals (`ZSTD_storeLastLiterals`). `rep` holds the committed
/// repeat offsets on entry and the block's candidates on return.
///
/// Out of line so that every caller, tests/stage_bench.rs included, runs
/// the one instantiation the frame writer runs.
#[inline(never)]
pub fn build_seq_store(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    store: &mut SeqStore,
) {
    store.clear();
    limit_update_after_long_match(ms, block.start);
    let anchor = match ms.cparams.strategy {
        Strategy::Fast => fast::compress_block(ms, src, block.clone(), rep, store),
        Strategy::DFast => dfast::compress_block(ms, src, block.clone(), rep, store),
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => {
            lazy::compress_block(ms, src, block.clone(), rep, store)
        }
    };
    // ZSTD_storeLastLiterals
    store.lits.extend_from_slice(&src[anchor..block.end]);
    debug_assert_eq!(
        store.lits.len()
            + store
                .seqs
                .iter()
                .map(|s| s.match_len() as usize)
                .sum::<usize>(),
        block.len()
    );
}

/// `ZSTD_buildSeqStore` + `ZSTD_entropyCompressSeqStore`: fill
/// `scratch.store` for `src[block]` and write the block payload into
/// `scratch.cbuf`. Returns the candidate next state when the block is
/// compressible (`cbuf.len() < block_len - ZSTD_minGain`), else `None`.
fn build_and_entropy_compress(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    prev: &BlockState,
    scratch: &mut BlockScratch,
) -> Option<BlockState> {
    let block_len = block.len();
    let strategy = ms.cparams.strategy;
    // don't even attempt compression below a certain srcSize
    if block_len < MIN_CBLOCK_SIZE + ZSTD_BLOCKHEADERSIZE + 1 + 1 {
        return None;
    }
    let mut rep = prev.rep;
    build_seq_store(ms, src, block, &mut rep, &mut scratch.store);
    let store = &scratch.store;

    let cbuf = &mut scratch.cbuf;
    cbuf.clear();
    let huf =
        huf::compress_literals_with(cbuf, &store.lits, store.seqs.len(), &prev.huf, &ms.cparams);
    let fse = fse::encode_sequences_section_with(cbuf, &store.seqs, &prev.fse, &ms.cparams)?;

    let max_c_size = block_len - CParams::min_gain(block_len, strategy);
    if cbuf.len() >= max_c_size {
        return None;
    }
    debug_assert!(cbuf.len() < ZSTD_BLOCKSIZE_MAX);
    Some(BlockState { rep, huf, fse })
}

/// Compress `src[block]` (at most `ZSTD_BLOCKSIZE_MAX` bytes) and append one
/// complete block, header included, to `out`. `state` is committed only when
/// the block is written COMPRESSED. `is_first_block` disables RLE for the
/// first block of a frame, as libzstd does for decoders <= 1.4.3.
#[allow(clippy::too_many_arguments)]
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    is_first_block: bool,
    is_last: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    out: &mut Vec<u8>,
) -> BlockKind {
    debug_assert!(block.len() <= ZSTD_BLOCKSIZE_MAX);
    let data = &src[block.clone()];
    let next = build_and_entropy_compress(ms, src, block, state.prev(), scratch);
    let c_size = if next.is_some() {
        scratch.cbuf.len()
    } else {
        0
    };

    if !is_first_block && c_size < RLE_MAX_LENGTH && is_rle(data) {
        write_rle_block(out, data[0], data.len(), is_last);
        return BlockKind::Rle;
    }
    match next {
        None => {
            write_raw_block(out, data, is_last);
            BlockKind::Raw
        }
        Some(next) => {
            state.commit(next);
            write_compressed_block(out, &scratch.cbuf, is_last);
            BlockKind::Compressed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prefix longer than `1 << max(hashLog + 3, chainLog + 1)` is indexed
    /// only over that suffix; one byte shorter is indexed whole.
    #[test]
    fn load_prefix_indexes_only_the_table_sized_suffix() {
        let cp = CParams::for_level(1, 8 << 20);
        let cap = 1usize << (cp.hash_log + 3).max(cp.chain_log + 1);
        let mut x = 1u32;
        let src: Vec<u8> = (0..cap + 4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        let lowest = |len: usize| {
            let mut ms = MatchState::new(cp, 1);
            load_prefix(&mut ms, &src, src.len() - len..src.len());
            let (hash, _, _) = ms.tables();
            hash.iter().filter(|&&e| e != 0).min().copied().unwrap() as usize
        };
        assert!(lowest(cap + 1000) >= src.len() - cap);
        assert!(lowest(cap) >= src.len() - cap);
        assert!(lowest(cap - 1) > src.len() - cap);
        assert!(lowest(cap + 1000) < src.len() - cap + 64);
    }

    #[test]
    fn limit_update_after_long_match_boundaries() {
        let cp = CParams::for_level(5, 1 << 20);
        let mut ms = MatchState::new(cp, 1);
        // Backlog of exactly 384: untouched.
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1384);
        assert_eq!(ms.next_to_update, 1000);
        // Backlog 385..575: only the excess over 384 gets inserted.
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1385);
        assert_eq!(ms.next_to_update, 1384);
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1575);
        assert_eq!(ms.next_to_update, 1384);
        // Backlog >= 576: insert only the last 192 positions.
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1576);
        assert_eq!(ms.next_to_update, 1384);
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 500_000);
        assert_eq!(ms.next_to_update, 500_000 - 192);
        // Idempotent.
        limit_update_after_long_match(&mut ms, 500_000);
        assert_eq!(ms.next_to_update, 500_000 - 192);
    }
}
