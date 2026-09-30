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
use super::presplit::{PreSplitter, SPLIT_BLOCK_SIZE};
use super::seqstore::{Seq, SeqStore};
use super::split::{resolve_off_codes, BlockSplitter, Partition};
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
    /// The second sequence store of [`compress_blocks`]' pipelined loop;
    /// sized on first use.
    pub next: SeqStore,
    pub cbuf: Vec<u8>,
    /// The post-sequence splitter's partitions and estimator buffers.
    pub splitter: BlockSplitter,
    /// The pre-splitter's fingerprints.
    pub presplit: PreSplitter,
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

/// `ZSTD_compressBlock_internal`: "don't even attempt compression below a
/// certain srcSize"; smaller blocks skip both stages and go RAW or RLE.
#[inline]
pub fn attempts_compression(block_len: usize) -> bool {
    block_len > MIN_CBLOCK_SIZE + ZSTD_BLOCKHEADERSIZE + 1
}

/// The sections of one block about to be entropy coded: its literals and
/// sequences, and the repeat offsets the decoder holds after them, which
/// the commit installs if the block is written COMPRESSED.
#[derive(Clone, Copy)]
struct Sections<'a> {
    lits: &'a [u8],
    seqs: &'a [Seq],
    rep: [u32; 3],
}

impl<'a> Sections<'a> {
    fn whole(store: &'a SeqStore, rep: [u32; 3]) -> Self {
        Sections {
            lits: &store.lits,
            seqs: &store.seqs,
            rep,
        }
    }
}

/// Proof, before entropy coding, that [`entropy_and_emit`] will write
/// `src[block]` COMPRESSED from `lits` and `seqs`: the block is not RLE and
/// the section bounds already beat `block_len - ZSTD_minGain`.
#[cfg(feature = "parallel")]
fn proven_compressed(
    src: &[u8],
    block: Range<usize>,
    lits: &[u8],
    seqs: &[Seq],
    strategy: Strategy,
) -> bool {
    let block_len = block.len();
    !is_rle(&src[block])
        && huf::literals_section_bound(lits.len()) + fse::sequences_section_bound(seqs)
            < block_len - CParams::min_gain(block_len, strategy)
}

/// Proof, before entropy coding, that [`emit_block`] writes every block of
/// `src[block]` COMPRESSED from `store` cut into `parts`, returning the
/// repeat offsets the decoder then holds: the finder's `rep` for a block
/// kept whole; for a split one the history through every sequence from
/// `prev_rep`, since with no partition RAW or RLE [`resolve_off_codes`]
/// never sees `d_rep` and `c_rep` differ, rewrites nothing, and leaves
/// `d_rep` equal to `c_rep`.
#[cfg(feature = "parallel")]
fn proven_rep_after(
    src: &[u8],
    block: Range<usize>,
    store: &SeqStore,
    rep: [u32; 3],
    parts: Option<&[Partition]>,
    prev_rep: [u32; 3],
    strategy: Strategy,
) -> Option<[u32; 3]> {
    let parts = match parts {
        None | Some([]) => {
            return proven_compressed(src, block, &store.lits, &store.seqs, strategy)
                .then_some(rep);
        }
        Some(parts) => parts,
    };
    let mut start = block.start;
    for part in parts {
        let range = start..start + part.src_len;
        let lits = &store.lits[part.lits.clone()];
        if !proven_compressed(src, range, lits, &store.seqs[part.seqs.clone()], strategy) {
            return None;
        }
        start += part.src_len;
    }
    let mut rep = prev_rep;
    for seq in &store.seqs {
        super::seqstore::update_rep(&mut rep, seq.off_base, seq.lit_len == 0);
    }
    Some(rep)
}

/// `ZSTD_entropyCompressSeqStore` and the block-type decision of
/// `ZSTD_compressBlock_internal` / `ZSTD_compressSeqStore_singleBlock`:
/// from `built` (`None` when compression was not attempted) write the
/// payload into `cbuf`, then append one complete block to `out`. `state` is
/// committed, with `built.rep`, only when the block is written COMPRESSED.
/// `is_first_block` disables RLE for the first block of a frame, as
/// libzstd does for decoders <= 1.4.3.
#[allow(clippy::too_many_arguments)]
fn entropy_and_emit(
    src: &[u8],
    block: Range<usize>,
    built: Option<Sections>,
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
    let next = built.and_then(|Sections { lits, seqs, rep }| {
        let prev = state.prev();
        cbuf.clear();
        let huf = huf::compress_literals_with(cbuf, lits, seqs.len(), &prev.huf, cparams);
        let fse = fse::encode_sequences_section_with(cbuf, seqs, &prev.fse, cparams)?;
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

/// Append `src[block]` to `out` from `built` (the block's sequence store
/// and the finder's repeat offsets after it, `None` when compression was
/// not attempted). With the splitter off (`parts` is `None`) this is
/// `ZSTD_compressBlock_internal`'s entropy stage, one block. With it on,
/// `ZSTD_compressBlock_splitBlock` after `ZSTD_buildSeqStore`: `parts`
/// from [`BlockSplitter::derive`], one block when empty, else one per
/// partition (`ZSTD_compressBlock_splitBlock_internal`), each coded with
/// the repcodes [`resolve_off_codes`] made valid for the decoder and
/// committing those repcodes. Returns whether every block written is
/// COMPRESSED.
#[allow(clippy::too_many_arguments)]
fn emit_block(
    src: &[u8],
    block: Range<usize>,
    built: Option<(&mut SeqStore, [u32; 3])>,
    parts: Option<&[Partition]>,
    cparams: &CParams,
    is_first_block: bool,
    is_last: bool,
    state: &mut CommittedBlockState,
    cbuf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> bool {
    let (built, parts) = match (built, parts) {
        (built, None) => {
            let built = built.map(|(store, rep)| Sections::whole(store, rep));
            return entropy_and_emit(
                src,
                block,
                built,
                cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            ) == BlockKind::Compressed;
        }
        // ZSTDbss_noCompress: RAW, without the RLE check.
        (None, Some(_)) => {
            write_raw_block(out, &src[block], is_last);
            return false;
        }
        (Some((store, rep)), Some([])) => {
            return entropy_and_emit(
                src,
                block,
                Some(Sections::whole(store, rep)),
                cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            ) == BlockKind::Compressed;
        }
        (Some((store, _)), Some(parts)) => (store, parts),
    };
    let store = built;
    let mut d_rep = state.prev().rep;
    let mut c_rep = d_rep;
    let mut start = block.start;
    let mut all_compressed = true;
    for (i, part) in parts.iter().enumerate() {
        let d_rep_original = d_rep;
        resolve_off_codes(&mut d_rep, &mut c_rep, &mut store.seqs[part.seqs.clone()]);
        let kind = entropy_and_emit(
            src,
            start..start + part.src_len,
            Some(Sections {
                lits: &store.lits[part.lits.clone()],
                seqs: &store.seqs[part.seqs.clone()],
                rep: d_rep,
            }),
            cparams,
            is_first_block,
            is_last && i + 1 == parts.len(),
            state,
            cbuf,
            out,
        );
        if kind != BlockKind::Compressed {
            // The decoder's history is untouched by a RAW or RLE block.
            d_rep = d_rep_original;
            all_compressed = false;
        }
        start += part.src_len;
    }
    debug_assert_eq!(start, block.end);
    // `prevCBlock->rep = dRep`: the last COMPRESSED partition committed it.
    debug_assert_eq!(state.prev().rep, d_rep);
    all_compressed
}

/// Compress `src[block]` (at most `ZSTD_BLOCKSIZE_MAX` bytes) and append it,
/// as one block or, with `split`, as the blocks the post-sequence splitter
/// cuts it into, to `out`: [`build_seq_store`] from the committed repeat
/// offsets, then [`emit_block`].
#[allow(clippy::too_many_arguments)]
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    is_first_block: bool,
    is_last: bool,
    split: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    out: &mut Vec<u8>,
) {
    let cparams = ms.cparams;
    let BlockScratch {
        store,
        cbuf,
        splitter,
        ..
    } = scratch;
    let built = attempts_compression(block.len()).then(|| {
        let mut rep = state.prev().rep;
        build_seq_store(ms, src, block.clone(), &mut rep, store);
        rep
    });
    let parts = split.then(|| match built {
        Some(_) => splitter.derive(store, state.prev(), &cparams, block.len()),
        None => &[][..],
    });
    emit_block(
        src,
        block,
        built.map(|rep| (store, rep)),
        parts,
        &cparams,
        is_first_block,
        is_last,
        state,
        cbuf,
        out,
    );
}

/// How [`compress_blocks`] sizes blocks: `ZSTD_compress_frameChunk` with
/// `ZSTD_optimalBlockSize`.
#[derive(Clone, Copy, Debug)]
pub struct BlockSizing {
    /// `blockSizeMax`: `min(ZSTD_BLOCKSIZE_MAX, 1 << window_log)`.
    pub block_size_max: usize,
    /// The `ZSTD_splitBlock` level, `None` with the pre-splitter off (see
    /// [`split_level`](super::presplit::split_level)).
    pub split_level: Option<u8>,
    /// Source bytes per `ZSTD_compressContinue` call, from the job start;
    /// no block crosses the end of one. ZSTDMT feeds a job in chunks of
    /// `4 * ZSTD_BLOCKSIZE_MAX`; single-threaded `ZSTD_compress2` passes the
    /// whole input in one call (`usize::MAX`).
    pub chunk_size: usize,
    /// Frame header length. `producedCSize` counts it only after the call
    /// that wrote it, so job 0's `savings` owe it from its second chunk on.
    pub header_len: usize,
}

/// [`BlockSizing`] over one job, with the job's `savings` so far.
struct JobBlocks {
    sizing: BlockSizing,
    job: Range<usize>,
    first_job: bool,
    /// Source minus written bytes over the job's blocks so far.
    gained: i64,
}

impl JobBlocks {
    /// `min(remaining, blockSizeMax)` at `start`, with `remaining` the rest
    /// of its chunk: the block's size unless pre-split.
    fn unsplit_size(&self, start: usize) -> usize {
        let chunk = self.sizing.chunk_size;
        let offset = start - self.job.start;
        let chunk_end = (offset / chunk + 1)
            .saturating_mul(chunk)
            .min(self.job.len());
        (chunk_end - offset).min(self.sizing.block_size_max)
    }

    /// `savings` (`consumedSrcSize - producedCSize` plus the chunk's blocks
    /// so far) at `start`, had the blocks before it gained `gained`.
    fn savings(&self, start: usize, gained: i64) -> i64 {
        let owes_header = self.first_job && start - self.job.start >= self.sizing.chunk_size;
        let header = if owes_header {
            self.sizing.header_len
        } else {
            0
        };
        gained - header as i64
    }

    /// `ZSTD_optimalBlockSize`: the block starting at `start` with
    /// `savings`. Only a full 128 KiB block is split, and only once the
    /// job has saved 3 bytes, so the first block of a job never is.
    fn block(
        &self,
        src: &[u8],
        start: usize,
        savings: i64,
        presplit: &mut PreSplitter,
    ) -> Range<usize> {
        let unsplit = self.unsplit_size(start);
        let size = match self.sizing.split_level {
            Some(level) if unsplit == SPLIT_BLOCK_SIZE && savings >= 3 => {
                presplit.split_block(&src[start..start + unsplit], level)
            }
            _ => unsplit,
        };
        start..start + size
    }

    /// The block at `start`, from the blocks written so far.
    fn next(&self, src: &[u8], start: usize, presplit: &mut PreSplitter) -> Range<usize> {
        self.block(src, start, self.savings(start, self.gained), presplit)
    }

    /// The block at `start` while the blocks before it, not all written,
    /// are known to gain at least `least_gained`. [`JobBlocks::block`]
    /// depends on `savings` only through `savings >= 3` and only for a
    /// block that may be pre-split, so the size is fixed unless that holds
    /// and the least savings are under 3.
    #[cfg(feature = "parallel")]
    fn next_unwritten(
        &self,
        src: &[u8],
        start: usize,
        least_gained: i64,
        presplit: &mut PreSplitter,
    ) -> Option<Range<usize>> {
        let least_savings = self.savings(start, least_gained);
        let may_split =
            self.sizing.split_level.is_some() && self.unsplit_size(start) == SPLIT_BLOCK_SIZE;
        if may_split && least_savings < 3 {
            return None;
        }
        Some(self.block(src, start, least_savings, presplit))
    }

    /// Account `block`, written as `written` bytes.
    fn wrote(&mut self, block: &Range<usize>, written: usize) {
        self.gained += block.len() as i64 - written as i64;
    }
}

/// `ZSTD_compress_frameChunk` over one job: `src[job]` in blocks sized by
/// `sizing`, appended to `out`, each through the post-sequence splitter
/// when `split`. With `pipelined` (parallel feature only) block N+1's match
/// finding runs on rayon next to block N's entropy stage and emission
/// whenever every block N is written as is [proven](proven_rep_after) to
/// be COMPRESSED, so that the repeat offsets N+1 starts from are the ones
/// the decoder will hold, and N+1's size is fixed without N's compressed
/// size; otherwise N's entropy stage runs first and N+1 starts from the
/// committed offsets. Output is identical either way.
#[allow(clippy::too_many_arguments)]
pub fn compress_blocks(
    ms: &mut MatchState,
    src: &[u8],
    job: Range<usize>,
    sizing: BlockSizing,
    first_job: bool,
    last_job: bool,
    split: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    out: &mut Vec<u8>,
    pipelined: bool,
) {
    let mut blocks = JobBlocks {
        sizing,
        job: job.clone(),
        first_job,
        gained: 0,
    };
    #[cfg(feature = "parallel")]
    if pipelined {
        compress_blocks_pipelined(ms, src, &mut blocks, last_job, split, state, scratch, out);
        return;
    }
    let _ = pipelined;
    let mut start = job.start;
    while start < job.end {
        let block = blocks.next(src, start, &mut scratch.presplit);
        let written = out.len();
        compress_block(
            ms,
            src,
            block.clone(),
            first_job && start == job.start,
            last_job && block.end == job.end,
            split,
            state,
            scratch,
            out,
        );
        blocks.wrote(&block, out.len() - written);
        start = block.end;
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

/// The least `len - written` over the blocks [`emit_block`] writes for
/// `block_len` bytes cut into `parts` when all are COMPRESSED:
/// [`entropy_and_emit`] writes COMPRESSED only a payload under
/// `len - ZSTD_minGain`, so with its header a block takes at most
/// `len - min_gain + 2` bytes.
#[cfg(feature = "parallel")]
fn compressed_gain_bound(block_len: usize, parts: Option<&[Partition]>, strategy: Strategy) -> i64 {
    let gain =
        |len: usize| CParams::min_gain(len, strategy) as i64 + 1 - ZSTD_BLOCKHEADERSIZE as i64;
    match parts {
        None | Some([]) => gain(block_len),
        Some(parts) => parts.iter().map(|p| gain(p.src_len)).sum(),
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn compress_blocks_pipelined(
    ms: &mut MatchState,
    src: &[u8],
    blocks: &mut JobBlocks,
    last_job: bool,
    split: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    out: &mut Vec<u8>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let cparams = ms.cparams;
    let job = blocks.job.clone();
    if job.is_empty() {
        return;
    }
    scratch.next.reserve(blocks.sizing.block_size_max);
    let BlockScratch {
        store,
        next,
        cbuf,
        splitter,
        presplit,
    } = scratch;
    let (mut cur, mut nxt) = (store, next);
    let mut block = blocks.next(src, job.start, presplit);
    // `built`: `block`'s store is in `cur`, with the finder's offsets after it.
    let mut built = attempts_compression(block.len()).then(|| {
        let mut rep = state.prev().rep;
        build_seq_store(ms, src, block.clone(), &mut rep, cur);
        rep
    });
    loop {
        let is_first_block = blocks.first_job && block.start == job.start;
        let is_last = last_job && block.end == job.end;
        // ZSTD_deriveBlockSplits runs against the state committed by the
        // previous block, before the next block's finder may start.
        let parts = split.then(|| match built {
            Some(_) => splitter.derive(cur, state.prev(), &cparams, block.len()),
            None => &[][..],
        });
        let following_start = block.end;
        // A pre-split block is at least 8 KiB, so the unsplit size decides
        // whether the next block attempts compression.
        let following_builds =
            following_start < job.end && attempts_compression(blocks.unsplit_size(following_start));
        // Block N+1 may start before block N is written when N is proven
        // COMPRESSED (the offsets N+1 starts from) and N+1's size does not
        // depend on N's compressed size: it is not pre-split, or the least
        // savings a COMPRESSED N leaves already allow the split.
        let overlap = following_builds
            .then(|| {
                let rep = built?;
                let rep_next = proven_rep_after(
                    src,
                    block.clone(),
                    cur,
                    rep,
                    parts,
                    state.prev().rep,
                    cparams.strategy,
                )?;
                let least_gained =
                    blocks.gained + compressed_gain_bound(block.len(), parts, cparams.strategy);
                let following =
                    blocks.next_unwritten(src, following_start, least_gained, presplit)?;
                Some((rep, rep_next, least_gained, following))
            })
            .flatten();
        let written = out.len();
        if let Some((rep, rep_next, least_gained, following)) = overlap {
            PIPELINE_OVERLAPPED.fetch_add(1, Relaxed);
            let mut rep_following = rep_next;
            let cur_store = &mut *cur;
            let state = &mut *state;
            let (compressed, ()) = rayon::join(
                || {
                    emit_block(
                        src,
                        block.clone(),
                        Some((cur_store, rep)),
                        parts,
                        &cparams,
                        is_first_block,
                        is_last,
                        state,
                        cbuf,
                        out,
                    )
                },
                || build_seq_store(ms, src, following.clone(), &mut rep_following, nxt),
            );
            // The proof is what made block N+1 start from the decoder's
            // offsets, with the size the written block N gives it.
            assert!(compressed, "section bound proof failed");
            assert_eq!(state.prev().rep, rep_next, "repeat offset proof failed");
            blocks.wrote(&block, out.len() - written);
            assert!(blocks.gained >= least_gained, "savings bound proof failed");
            built = Some(rep_following);
            block = following;
        } else {
            if following_builds {
                PIPELINE_SERIALIZED.fetch_add(1, Relaxed);
            }
            emit_block(
                src,
                block.clone(),
                built.map(|rep| (&mut *cur, rep)),
                parts,
                &cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            );
            blocks.wrote(&block, out.len() - written);
            if block.end == job.end {
                return;
            }
            block = blocks.next(src, block.end, presplit);
            built = attempts_compression(block.len()).then(|| {
                let mut rep = state.prev().rep;
                build_seq_store(ms, src, block.clone(), &mut rep, nxt);
                rep
            });
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

    /// Builds a block and its sequence store one partition at a time,
    /// executing each sequence against the finder's repeat offsets.
    struct Builder {
        src: Vec<u8>,
        store: SeqStore,
        c_rep: [u32; 3],
        parts: Vec<Partition>,
        part_start: (usize, usize, usize),
    }

    impl Builder {
        fn seq(&mut self, lits: &[u8], off_base: u32, match_len: usize) {
            use super::super::seqstore::{offbase_is_repcode, offbase_to_offset, update_rep};
            let ll0 = lits.is_empty();
            let offset = if offbase_is_repcode(off_base) {
                match off_base - 1 + ll0 as u32 {
                    3 => self.c_rep[0] - 1,
                    i => self.c_rep[i as usize],
                }
            } else {
                offbase_to_offset(off_base)
            } as usize;
            self.src.extend_from_slice(lits);
            self.store.lits.extend_from_slice(lits);
            for _ in 0..match_len {
                self.src.push(self.src[self.src.len() - offset]);
            }
            self.store.seqs.push(Seq {
                lit_len: lits.len() as u32,
                off_base,
                ml_base: match_len as u32 - 3,
            });
            update_rep(&mut self.c_rep, off_base, ll0);
        }

        fn end_partition(&mut self) {
            let (seq, lit, src) = self.part_start;
            let end = (self.store.seqs.len(), self.store.lits.len(), self.src.len());
            self.parts.push(Partition {
                seqs: seq..end.0,
                lits: lit..end.1,
                src_len: end.2 - src,
            });
            self.part_start = end;
        }
    }

    /// A split block with an RLE and a RAW partition between COMPRESSED
    /// ones: the decoder's repeat offsets skip the RLE and RAW partitions'
    /// sequences, so the repcodes after each are rewritten to the finder's
    /// raw offsets, the frame decodes, and the committed offsets are the
    /// decoder's.
    #[test]
    fn split_block_resolves_repcodes_across_rle_and_raw_partitions() {
        use super::super::seqstore::{offset_to_offbase, repcode_to_offbase};
        let mut b = Builder {
            src: Vec::new(),
            store: SeqStore::new(),
            c_rep: BlockState::initial().rep,
            parts: Vec::new(),
            part_start: (0, 0, 0),
        };
        // COMPRESSED: offset 16 everywhere; both histories end [16, 16, 16].
        b.seq(
            b"the quick brown fox jumps over t",
            offset_to_offbase(16),
            12,
        );
        for _ in 0..399 {
            b.seq(b"abcd", offset_to_offbase(16), 12);
        }
        b.end_partition();
        // RLE: zeros, with a new offset 7 the decoder never sees.
        b.seq(&[0; 64], offset_to_offbase(7), 100);
        b.end_partition();
        // COMPRESSED: repcode 1 is 7 for the finder, 16 for the decoder.
        let c_first = b.store.seqs.len();
        for _ in 0..50 {
            b.seq(b"wxyz", repcode_to_offbase(1), 12);
        }
        b.end_partition();
        // RAW: incompressible, with a new offset 33 the decoder never sees.
        let mut x = 0x9e37_79b9u32;
        let noise: Vec<u8> = (0..60)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        b.seq(&noise, offset_to_offbase(33), 3);
        b.end_partition();
        // COMPRESSED: repcode 2 is 7 for the finder, 16 for the decoder.
        let e_first = b.store.seqs.len();
        b.seq(b"klmn", repcode_to_offbase(2), 12);
        for _ in 0..49 {
            b.seq(b"klmn", repcode_to_offbase(1), 12);
        }
        b.end_partition();

        let cparams = CParams::for_level(3, b.src.len());
        let mut state = CommittedBlockState::new(BlockState::initial());
        let (mut cbuf, mut blocks) = (Vec::new(), Vec::new());
        let all_compressed = emit_block(
            &b.src,
            0..b.src.len(),
            Some((&mut b.store, b.c_rep)),
            Some(&b.parts),
            &cparams,
            false,
            true,
            &mut state,
            &mut cbuf,
            &mut blocks,
        );
        assert!(!all_compressed);

        let mut kinds = Vec::new();
        let mut pos = 0;
        while pos < blocks.len() {
            let h = u32::from_le_bytes([blocks[pos], blocks[pos + 1], blocks[pos + 2], 0]);
            let ty = (h >> 1) & 3;
            kinds.push(ty);
            pos += 3 + if ty == 1 { 1 } else { (h >> 3) as usize };
        }
        // 0 RAW, 1 RLE, 2 COMPRESSED.
        assert_eq!(kinds, [2, 1, 2, 0, 2]);
        assert_eq!(b.store.seqs[c_first].off_base, offset_to_offbase(7));
        assert_eq!(b.store.seqs[c_first + 1].off_base, repcode_to_offbase(1));
        assert_eq!(b.store.seqs[e_first].off_base, offset_to_offbase(7));
        assert_eq!(state.prev().rep, [7, 7, 16]);

        let mut frame = Vec::new();
        super::super::write_frame_header(&mut frame, b.src.len() as u64, cparams.window_log);
        frame.extend_from_slice(&blocks);
        assert_eq!(crate::decompress(&frame).unwrap(), b.src);
        assert_eq!(zstd::stream::decode_all(&frame[..]).unwrap(), b.src);
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

    /// 1 MiB of period-21 text with noise over `[40 KiB, 128 KiB)`: the
    /// chunked pre-splitter levels cut the block at 0 at 40 KiB and keep
    /// every later one whole.
    fn presplit_input() -> Vec<u8> {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut src: Vec<u8> = (0..1 << 20)
            .map(|i| b"the quick brown foxes"[i % 21])
            .collect();
        for b in &mut src[40 << 10..SPLIT_BLOCK_SIZE] {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 56) as u8;
        }
        src
    }

    fn job_blocks(split_level: Option<u8>, chunk_size: usize, job: Range<usize>) -> JobBlocks {
        JobBlocks {
            sizing: BlockSizing {
                block_size_max: ZSTD_BLOCKSIZE_MAX,
                split_level,
                chunk_size,
                header_len: 10,
            },
            job,
            first_job: true,
            gained: 0,
        }
    }

    /// `ZSTD_optimalBlockSize`: a full block splits at `savings` 3, not 2;
    /// a block under 128 KiB (job end, a smaller `blockSizeMax`) or with
    /// the pre-splitter off never does.
    #[test]
    fn block_splits_full_blocks_once_savings_reach_three() {
        let src = presplit_input();
        let mut ps = PreSplitter::default();
        let on = job_blocks(Some(1), usize::MAX, 0..src.len());
        assert_eq!(on.block(&src, 0, 2, &mut ps), 0..SPLIT_BLOCK_SIZE);
        assert_eq!(on.block(&src, 0, 3, &mut ps), 0..40 << 10);
        assert_eq!(
            on.block(&src, 512 << 10, 3, &mut ps).len(),
            SPLIT_BLOCK_SIZE
        );
        let tail = src.len() - 1000;
        assert_eq!(on.block(&src, tail, 1 << 20, &mut ps), tail..src.len());
        let off = job_blocks(None, usize::MAX, 0..src.len());
        assert_eq!(off.block(&src, 0, 1 << 20, &mut ps), 0..SPLIT_BLOCK_SIZE);
        let mut small = job_blocks(Some(1), usize::MAX, 0..src.len());
        small.sizing.block_size_max = 64 << 10;
        assert_eq!(small.block(&src, 0, 1 << 20, &mut ps), 0..64 << 10);
    }

    /// ZSTDMT chunks: no block crosses `job.start + k * 512 KiB`, and job 0
    /// owes the frame header from its second chunk on; single-threaded,
    /// the job is one chunk and the header is never owed.
    #[test]
    fn chunks_bound_blocks_and_owe_the_header_from_the_second() {
        let chunk = 4 * ZSTD_BLOCKSIZE_MAX;
        let job = 1000..1000 + (700 << 10);
        let mt = job_blocks(Some(1), chunk, job.clone());
        let second = job.start + chunk;
        assert_eq!(mt.unsplit_size(job.start), SPLIT_BLOCK_SIZE);
        assert_eq!(mt.unsplit_size(second - (8 << 10)), 8 << 10);
        assert_eq!(mt.unsplit_size(second), SPLIT_BLOCK_SIZE);
        assert_eq!(mt.unsplit_size(job.end - 100), 100);
        assert_eq!(mt.savings(second - 1, 5), 5);
        assert_eq!(mt.savings(second, 5), -5);
        let later = JobBlocks {
            first_job: false,
            ..job_blocks(Some(1), chunk, job.clone())
        };
        assert_eq!(later.savings(second, 5), 5);
        let st = job_blocks(Some(1), usize::MAX, job.clone());
        assert_eq!(st.unsplit_size(second - (8 << 10)), SPLIT_BLOCK_SIZE);
        assert_eq!(st.savings(job.end - 1, 5), 5);
    }

    /// The pipelined loop fixes the next block before the current one is
    /// written only when the least savings decide it as the written block
    /// will: a block that may split needs least savings of 3 (the header
    /// debit included); any other block is fixed at any savings.
    #[cfg(feature = "parallel")]
    #[test]
    fn next_unwritten_waits_only_when_least_savings_cannot_decide() {
        let src = presplit_input();
        let mut ps = PreSplitter::default();
        let st = job_blocks(Some(1), usize::MAX, 0..src.len());
        assert_eq!(st.next_unwritten(&src, 0, 2, &mut ps), None);
        assert_eq!(st.next_unwritten(&src, 0, 3, &mut ps), Some(0..40 << 10));
        let tail = src.len() - 1000;
        assert_eq!(
            st.next_unwritten(&src, tail, -100, &mut ps),
            Some(tail..src.len())
        );
        let off = job_blocks(None, usize::MAX, 0..src.len());
        assert_eq!(
            off.next_unwritten(&src, 0, -100, &mut ps),
            Some(0..SPLIT_BLOCK_SIZE)
        );
        let chunk = 4 * ZSTD_BLOCKSIZE_MAX;
        let mt = job_blocks(Some(1), chunk, 0..src.len());
        assert_eq!(mt.next_unwritten(&src, chunk, 12, &mut ps), None);
        assert_eq!(
            mt.next_unwritten(&src, chunk, 13, &mut ps).map(|b| b.len()),
            Some(SPLIT_BLOCK_SIZE)
        );
        assert_eq!(
            mt.next_unwritten(&src, chunk - (8 << 10), -100, &mut ps),
            Some(chunk - (8 << 10)..chunk)
        );
    }
}
