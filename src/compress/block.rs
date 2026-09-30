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

use super::ldm::{self, LdmState, RawSeqStore};
use super::matchstate::MatchState;
use super::params::{CParams, Strategy};
use super::seqstore::SeqStore;
use super::{bt, dfast, fast, lazy, opt};
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
    /// The second sequence store of [`compress_blocks`]' pipelined loop;
    /// sized on first use.
    pub next: SeqStore,
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
        Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => bt::load_prefix(ms, src, range),
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

/// `ZSTD_selectBlockCompressor(strategy, useRowMatchFinder, ZSTD_noDict)`
/// run on `src[range]`: store its sequences into `out` and return the
/// anchor of the trailing literals.
pub fn run_block_compressor(
    ms: &mut MatchState,
    src: &[u8],
    range: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    match ms.cparams.strategy {
        Strategy::Fast => fast::compress_block(ms, src, range, rep, out),
        Strategy::DFast => dfast::compress_block(ms, src, range, rep, out),
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => {
            lazy::compress_block(ms, src, range, rep, out)
        }
        Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
            opt::compress_block(ms, src, range, rep, out)
        }
    }
}

/// Where a block's long distance matches come from: the branches of
/// `ZSTD_buildSeqStore`.
///
/// A block too small to compress ([`attempts_compression`]) is the last
/// one of its job, so the sequences libzstd skips over for it
/// (`ZSTD_ldm_skipSequences`) are never read again and are left alone.
pub enum BlockLdm<'a> {
    /// Long distance matching is off.
    Off,
    /// `ldmParams.enableLdm` on a single-threaded context: generate the
    /// block's sequences from the frame's persistent state, then
    /// `ZSTD_ldm_blockCompress`.
    Internal(&'a mut LdmState),
    /// `externSeqStore`: the job's sequences, generated ahead of it in job
    /// order (ZSTDMT). `ZSTD_ldm_blockCompress` while some remain, the
    /// plain block compressor once they are used up.
    External(&'a mut RawSeqStore),
}

/// `ZSTD_buildSeqStore` for a block worth compressing: reset `store`, apply
/// the nextToUpdate clamp, run the strategy's block compressor (through
/// `ZSTD_ldm_blockCompress` when `ldm` provides long matches) and store the
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
    ldm: &mut BlockLdm,
) {
    store.clear();
    limit_update_after_long_match(ms, block.start);
    let anchor = match ldm {
        BlockLdm::External(seqs) if !seqs.is_exhausted() => {
            ldm::block_compress(seqs, ms, src, block.clone(), rep, store)
        }
        BlockLdm::Internal(state) => {
            let seqs = state.generate_block_sequences(src, block.clone());
            ldm::block_compress(seqs, ms, src, block.clone(), rep, store)
        }
        BlockLdm::Off | BlockLdm::External(_) => {
            run_block_compressor(ms, src, block.clone(), rep, store)
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

/// `ZSTD_compressBlock_internal`: "don't even attempt compression below a
/// certain srcSize"; smaller blocks skip both stages and go RAW or RLE.
#[inline]
pub fn attempts_compression(block_len: usize) -> bool {
    block_len > MIN_CBLOCK_SIZE + ZSTD_BLOCKHEADERSIZE + 1
}

/// Proof, before entropy coding, that [`entropy_and_emit`] will write
/// `src[block]` COMPRESSED from `store`: the block is not RLE and the
/// section bounds already beat `block_len - ZSTD_minGain`.
#[cfg(feature = "parallel")]
fn proven_compressed(
    src: &[u8],
    block: Range<usize>,
    store: &SeqStore,
    strategy: Strategy,
) -> bool {
    let block_len = block.len();
    !is_rle(&src[block])
        && huf::literals_section_bound(store.lits.len()) + fse::sequences_section_bound(&store.seqs)
            < block_len - CParams::min_gain(block_len, strategy)
}

/// `ZSTD_entropyCompressSeqStore` and the block-type decision of
/// `ZSTD_compressBlock_internal`: from `built` (the block's sequence store
/// and the finder's repeat offsets after it, `None` when compression was
/// not attempted) write the payload into `cbuf`, then append one complete
/// block to `out`. `state` is committed only when the block is written
/// COMPRESSED. `is_first_block` disables RLE for the first block of a
/// frame, as libzstd does for decoders <= 1.4.3.
#[allow(clippy::too_many_arguments)]
fn entropy_and_emit(
    src: &[u8],
    block: Range<usize>,
    built: Option<(&SeqStore, [u32; 3])>,
    cparams: &CParams,
    is_first_block: bool,
    is_last: bool,
    state: &mut CommittedBlockState,
    cbuf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> BlockKind {
    debug_assert!(block.len() <= ZSTD_BLOCKSIZE_MAX);
    let block_len = block.len();
    let data = &src[block];
    let next = built.and_then(|(store, rep)| {
        let prev = state.prev();
        cbuf.clear();
        let huf =
            huf::compress_literals_with(cbuf, &store.lits, store.seqs.len(), &prev.huf, cparams);
        let fse = fse::encode_sequences_section_with(cbuf, &store.seqs, &prev.fse, cparams)?;
        let max_c_size = block_len - CParams::min_gain(block_len, cparams.strategy);
        if cbuf.len() >= max_c_size {
            return None;
        }
        debug_assert!(cbuf.len() < ZSTD_BLOCKSIZE_MAX);
        Some(BlockState { rep, huf, fse })
    });
    let c_size = if next.is_some() { cbuf.len() } else { 0 };

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
            write_compressed_block(out, cbuf, is_last);
            BlockKind::Compressed
        }
    }
}

/// Compress `src[block]` (at most `ZSTD_BLOCKSIZE_MAX` bytes) and append one
/// complete block, header included, to `out`: [`build_seq_store`] from the
/// committed repeat offsets, then [`entropy_and_emit`].
#[allow(clippy::too_many_arguments)]
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    is_first_block: bool,
    is_last: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    out: &mut Vec<u8>,
) -> BlockKind {
    let built = attempts_compression(block.len()).then(|| {
        let mut rep = state.prev().rep;
        build_seq_store(ms, src, block.clone(), &mut rep, &mut scratch.store, ldm);
        rep
    });
    let cparams = ms.cparams;
    entropy_and_emit(
        src,
        block,
        built.map(|rep| (&scratch.store, rep)),
        &cparams,
        is_first_block,
        is_last,
        state,
        &mut scratch.cbuf,
        out,
    )
}

/// `ZSTD_compress_frameChunk` over one job: `src[job]` in blocks of
/// `block_size`, appended to `out`, with long distance matches from `ldm`.
/// With `pipelined` (parallel feature only) block N+1's match finding runs
/// on rayon next to block N's entropy stage and emission whenever block N
/// is [proven](proven_compressed) to be written COMPRESSED, so that the
/// finder's repeat offsets after N are the ones the decoder will hold;
/// otherwise N's entropy stage runs first and N+1 starts from the committed
/// offsets. Output is identical either way.
#[allow(clippy::too_many_arguments)]
pub fn compress_blocks(
    ms: &mut MatchState,
    src: &[u8],
    job: Range<usize>,
    block_size: usize,
    first_job: bool,
    last_job: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    out: &mut Vec<u8>,
    pipelined: bool,
) {
    #[cfg(feature = "parallel")]
    if pipelined {
        compress_blocks_pipelined(
            ms, src, job, block_size, first_job, last_job, state, scratch, ldm, out,
        );
        return;
    }
    let _ = pipelined;
    let mut start = job.start;
    while start < job.end {
        let end = (start + block_size).min(job.end);
        compress_block(
            ms,
            src,
            start..end,
            first_job && start == job.start,
            last_job && end == job.end,
            state,
            scratch,
            ldm,
            out,
        );
        start = end;
    }
}

/// Counters of [`compress_blocks`]' pipelined loop, for benches.
#[cfg(feature = "parallel")]
pub static PIPELINE_OVERLAPPED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Blocks with a successor whose proof failed (entropy stage ran first).
#[cfg(feature = "parallel")]
pub static PIPELINE_SERIALIZED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn compress_blocks_pipelined(
    ms: &mut MatchState,
    src: &[u8],
    job: Range<usize>,
    block_size: usize,
    first_job: bool,
    last_job: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    out: &mut Vec<u8>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let cparams = ms.cparams;
    let blocks: Vec<Range<usize>> = (job.start..job.end)
        .step_by(block_size)
        .map(|start| start..(start + block_size).min(job.end))
        .collect();
    scratch.next.reserve(block_size);
    let BlockScratch { store, next, cbuf } = scratch;
    let (mut cur, mut nxt) = (store, next);
    // `built`: block i's store is in `cur`, with the finder's offsets after it.
    let mut built = blocks.first().and_then(|b| {
        attempts_compression(b.len()).then(|| {
            let mut rep = state.prev().rep;
            build_seq_store(ms, src, b.clone(), &mut rep, cur, ldm);
            rep
        })
    });
    for (i, block) in blocks.iter().enumerate() {
        let is_first_block = first_job && i == 0;
        let is_last = last_job && i + 1 == blocks.len();
        let following = blocks
            .get(i + 1)
            .filter(|b| attempts_compression(b.len()))
            .cloned();
        let Some(following) = following else {
            let built_cur = built.map(|rep| (&*cur, rep));
            entropy_and_emit(
                src,
                block.clone(),
                built_cur,
                &cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            );
            built = None;
            continue;
        };
        let proven = built.filter(|_| proven_compressed(src, block.clone(), cur, cparams.strategy));
        if let Some(rep) = proven {
            PIPELINE_OVERLAPPED.fetch_add(1, Relaxed);
            let mut rep_next = rep;
            let cur_store = &*cur;
            let (kind, ()) = rayon::join(
                || {
                    entropy_and_emit(
                        src,
                        block.clone(),
                        Some((cur_store, rep)),
                        &cparams,
                        is_first_block,
                        is_last,
                        state,
                        cbuf,
                        out,
                    )
                },
                || build_seq_store(ms, src, following.clone(), &mut rep_next, nxt, ldm),
            );
            // The proof is what made `rep_next` the decoder's offsets.
            assert_eq!(kind, BlockKind::Compressed, "section bound proof failed");
            built = Some(rep_next);
        } else {
            PIPELINE_SERIALIZED.fetch_add(1, Relaxed);
            let built_cur = built.map(|rep| (&*cur, rep));
            entropy_and_emit(
                src,
                block.clone(),
                built_cur,
                &cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            );
            let mut rep_next = state.prev().rep;
            build_seq_store(ms, src, following, &mut rep_next, nxt, ldm);
            built = Some(rep_next);
        }
        std::mem::swap(&mut cur, &mut nxt);
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
