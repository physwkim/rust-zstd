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

/// `ZSTD_compress_frameChunk` over one job: `src[job]` in blocks of
/// `block_size`, appended to `out`, each through the post-sequence splitter
/// when `split`. With `pipelined` (parallel feature only) and without
/// `split`, block N+1's match finding runs on rayon next to block N's
/// entropy stage and emission whenever N is
/// [proven](proven_compressed) to be written COMPRESSED, so that the repeat
/// offsets N+1 starts from are the ones the decoder will hold; otherwise
/// N's entropy stage runs first and N+1 starts from the committed offsets.
/// Output is identical either way.
#[allow(clippy::too_many_arguments)]
pub fn compress_blocks(
    ms: &mut MatchState,
    src: &[u8],
    job: Range<usize>,
    block_size: usize,
    first_job: bool,
    last_job: bool,
    split: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    out: &mut Vec<u8>,
    pipelined: bool,
) {
    #[cfg(feature = "parallel")]
    if pipelined && !split {
        compress_blocks_pipelined(
            ms, src, job, block_size, first_job, last_job, state, scratch, out,
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
            split,
            state,
            scratch,
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
    out: &mut Vec<u8>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let cparams = ms.cparams;
    let blocks: Vec<Range<usize>> = (job.start..job.end)
        .step_by(block_size)
        .map(|start| start..(start + block_size).min(job.end))
        .collect();
    scratch.next.reserve(block_size);
    let BlockScratch {
        store, next, cbuf, ..
    } = scratch;
    let (mut cur, mut nxt) = (store, next);
    // `built`: block i's store is in `cur`, with the finder's offsets after it.
    let mut built = blocks.first().and_then(|b| {
        attempts_compression(b.len()).then(|| {
            let mut rep = state.prev().rep;
            build_seq_store(ms, src, b.clone(), &mut rep, cur);
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
            emit_block(
                src,
                block.clone(),
                built.map(|rep| (&mut *cur, rep)),
                None,
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
        let proven = built.filter(|_| {
            proven_compressed(src, block.clone(), &cur.lits, &cur.seqs, cparams.strategy)
        });
        if let Some(rep) = proven {
            PIPELINE_OVERLAPPED.fetch_add(1, Relaxed);
            let mut rep_next = rep;
            let cur_store = &mut *cur;
            let (compressed, ()) = rayon::join(
                || {
                    emit_block(
                        src,
                        block.clone(),
                        Some((cur_store, rep)),
                        None,
                        &cparams,
                        is_first_block,
                        is_last,
                        state,
                        cbuf,
                        out,
                    )
                },
                || build_seq_store(ms, src, following.clone(), &mut rep_next, nxt),
            );
            // The proof is what made `rep_next` the decoder's offsets.
            assert!(compressed, "section bound proof failed");
            built = Some(rep_next);
        } else {
            PIPELINE_SERIALIZED.fetch_add(1, Relaxed);
            emit_block(
                src,
                block.clone(),
                built.map(|rep| (&mut *cur, rep)),
                None,
                &cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            );
            let mut rep_next = state.prev().rep;
            build_seq_store(ms, src, following, &mut rep_next, nxt);
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
}
