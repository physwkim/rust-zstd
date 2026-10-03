//! One block: `ZSTD_compressBlock_internal` plus the block-type decision of
//! `ZSTD_compress_frameChunk`, and the cross-block state
//! (`ZSTD_blockState_t`) with its commit rule.
//!
//! State-owner invariant: the decoder's repeat offsets and entropy tables
//! change only when a COMPRESSED block is emitted. [`CommittedBlockState`]
//! keeps that state in a private field; the entropy stage reads it through
//! [`CommittedBlockState::prev`] and produces a fresh [`BlockState`], and the
//! private `end_block`, which every written block goes through, is the only
//! path that installs it (`ZSTD_blockState_confirmRepcodesAndEntropyTables`).
//! RAW and RLE blocks discard the candidate, including its repeat offsets.

use super::common::Src;
use super::ldm::{self, LdmState, RawSeqStore, RawSeqView};
use super::matchstate::{Block, DictMatchState, EnteredBlock, EnteredPrefix, MatchState};
use super::params::{CParams, Strategy};
use super::presplit::{PreSplitter, SPLIT_BLOCK_SIZE};
use super::seqstore::{Seq, SeqStore};
use super::split::{resolve_off_codes, BlockSplitter, Partition};
use super::{bt, dfast, fast, lazy, opt};
use crate::constants::*;
use crate::fse::{self, FseState, FseTableState};
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
/// private `end_block`, exactly when a COMPRESSED block is written.
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

    /// The end of every written block, with `next` its candidate state if
    /// it was written COMPRESSED: commit it
    /// (`ZSTD_blockState_confirmRepcodesAndEntropyTables`), then, whatever
    /// the block's type, demote a dictionary's `Valid` offset table to
    /// `Check` (`ZSTD_compressBlock_internal`'s `out:`,
    /// `ZSTD_compressSeqStore_singleBlock` for each block a split writes):
    /// the dictionary checked that it codes every offset of a first block,
    /// `dictContentSize + 128 KiB`, and a later block reaches further.
    fn end_block(&mut self, next: Option<BlockState>) {
        if let Some(next) = next {
            self.prev = next;
        }
        self.prev.fse.of = match std::mem::take(&mut self.prev.fse.of) {
            FseTableState::Valid(table) => FseTableState::Check(table),
            of => of,
        };
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

/// `ZSTD_isRLE`: every byte of `data` equals the first one. As in C, 32
/// bytes are compared per step as words; a byte iterator's early exit
/// keeps LLVM from vectorizing and scans an RLE block a byte per cycle.
pub fn is_rle(data: &[u8]) -> bool {
    let Some(&first) = data.first() else {
        return false;
    };
    let splat = u64::from_ne_bytes([first; 8]);
    let (chunks, rest) = data.as_chunks::<32>();
    chunks.iter().all(|chunk| {
        let words = chunk.as_chunks::<8>().0;
        words
            .iter()
            .fold(0, |diff, &word| diff | (u64::from_ne_bytes(word) ^ splat))
            == 0
    }) && rest.iter().all(|&b| b == first)
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

/// `ZSTD_dictTableLoadMethod_e`: which positions of loaded content the
/// fast and dfast tables get. The other strategies insert every position
/// either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableLoad {
    /// `ZSTD_dtlm_fast`: every third position, as a context loads content.
    Fast,
    /// `ZSTD_dtlm_full` with `ZSTD_tfp_forCDict`: also the positions
    /// between them where their slot is empty, as a CDict loads its
    /// content once for many frames, and every fast and dfast entry tagged
    /// (`ZSTD_CDictIndicesAreTagged`, see [`MatchState::copy_dict`]).
    Full,
}

/// `ZSTD_loadDictionaryContent` for a job's raw-content prefix
/// ([`super::job_prefix`]): `data[range]` enters the window
/// ([`MatchState::enter_prefix`]) and its indexed suffix, unless
/// `HASH_READ_SIZE` bytes or less, goes into the strategy's tables before
/// the first block of the job; matches may still reach the whole prefix,
/// which `window_low` keeps valid. `range` is in positions of `data`,
/// starting at the window's origin.
pub fn load_prefix(ms: &mut MatchState, data: &[u8], range: Range<usize>) {
    if let Some(prefix) = ms.enter_prefix(range) {
        fill_tables(ms, data, prefix, TableLoad::Fast);
    }
}

/// [`load_prefix`] for dictionary content, which the window keeps valid
/// until the input passes the window size ([`MatchState::enter_dict`]),
/// filling the tables by `load`.
pub fn load_dict(ms: &mut MatchState, data: &[u8], range: Range<usize>, load: TableLoad) {
    if let Some(content) = ms.enter_dict(range) {
        fill_tables(ms, data, content, load);
    }
}

/// The strategy's table fill of `ZSTD_loadDictionaryContent` over entered
/// content.
fn fill_tables(ms: &mut MatchState, data: &[u8], content: EnteredPrefix, load: TableLoad) {
    let src = ms.view(data);
    match (ms.cparams.strategy, load) {
        (Strategy::Fast, TableLoad::Fast) => fast::load_prefix(ms, src, content),
        (Strategy::Fast, TableLoad::Full) => fast::load_dict_full(ms, src, content),
        (Strategy::DFast, TableLoad::Fast) => dfast::load_prefix(ms, src, content),
        (Strategy::DFast, TableLoad::Full) => dfast::load_dict_full(ms, src, content),
        (Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 | Strategy::BtLazy2, _) => {
            lazy::load_prefix(ms, src, content)
        }
        (Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2, _) => {
            bt::load_prefix(ms, src, content)
        }
    }
}

/// `ZSTD_selectBlockCompressor(strategy, useRowMatchFinder, dictMode)`
/// run on `block`: store its sequences into `out` and return the anchor of
/// the trailing literals. `dms` is the attached dictionary while it is
/// valid ([`MatchState::dict_match_state`]), which selects the
/// `ZSTD_dictMatchState` variants.
pub fn run_block_compressor(
    ms: &mut MatchState,
    src: Src,
    block: Block,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    dms: Option<DictMatchState>,
) -> usize {
    if let Some(dms) = dms {
        return run_dms_block_compressor(ms, src, block, rep, out, RawSeqView::default(), dms);
    }
    match ms.cparams.strategy {
        Strategy::Fast => fast::compress_block(ms, src, block, rep, out),
        Strategy::DFast => dfast::compress_block(ms, src, block, rep, out),
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 | Strategy::BtLazy2 => {
            lazy::compress_block(ms, src, block, rep, out)
        }
        Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
            opt::compress_block(ms, src, block, rep, out, RawSeqView::default())
        }
    }
}

/// The `ZSTD_dictMatchState` block compressors of
/// [`run_block_compressor`], the optimal parser's with the long distance
/// matches `ldm`. Out of line, so that the no-dictionary dispatch stays as
/// it is.
#[inline(never)]
pub fn run_dms_block_compressor(
    ms: &mut MatchState,
    src: Src,
    block: Block,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    ldm: RawSeqView,
    dms: DictMatchState,
) -> usize {
    match ms.cparams.strategy {
        Strategy::Fast => fast::compress_block_dms(ms, src, block, rep, out, dms),
        Strategy::DFast => dfast::compress_block_dms(ms, src, block, rep, out, dms),
        Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 | Strategy::BtLazy2 => {
            lazy::compress_block_dms(ms, src, block, rep, out, dms)
        }
        Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => {
            opt::compress_block_dms(ms, src, block, rep, out, ldm, dms)
        }
    }
}

/// Where a block's long distance matches come from: the branches of
/// `ZSTD_buildSeqStore`.
///
/// A block too small to compress ([`build_seq_store`] returns `None`) is
/// the last one of its job, so the sequences libzstd skips over for it
/// (`ZSTD_ldm_skipSequences`, or `ZSTD_ldm_skipRawSeqStoreBytes` from
/// btopt on) are never read again and are left alone.
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

/// `ZSTD_buildSeqStore` for a block that entered the window
/// ([`MatchState::enter_block`]): `None` for a block too small to attempt
/// compression, else reset `store`, set up the block's positions of `data`
/// ([`MatchState::start_block`], which applies the nextToUpdate clamp), run
/// the strategy's block compressor on it (through `ZSTD_ldm_blockCompress`
/// when `ldm` provides long matches) and store the trailing literals
/// (`ZSTD_storeLastLiterals`). `rep` is the committed repeat offsets; the
/// block's candidates are returned. `dms` is the frame's attached
/// dictionary, searched while it is valid.
///
/// Out of line so that every caller, tests/stage_bench.rs included, runs
/// the one instantiation the frame writer runs.
#[inline(never)]
pub fn build_seq_store(
    ms: &mut MatchState,
    data: &[u8],
    block: EnteredBlock,
    mut rep: [u32; 3],
    store: &mut SeqStore,
    ldm: &mut BlockLdm,
    dms: Option<DictMatchState>,
) -> Option<[u32; 3]> {
    let positions = block.positions();
    if !attempts_compression(positions.len()) {
        return None;
    }
    store.clear();
    let (block_len, block_end) = (positions.len(), positions.end);
    let (src, block) = ms.start_block(data, block);
    let dms = ms.dict_match_state(dms);
    let anchor = match ldm {
        BlockLdm::External(seqs) if !seqs.is_exhausted() => {
            ldm::block_compress(seqs, ms, src, block, &mut rep, store, dms)
        }
        BlockLdm::Internal(state) => {
            let seqs = state.generate_block_sequences(data, positions);
            ldm::block_compress(seqs, ms, src, block, &mut rep, store, dms)
        }
        BlockLdm::Off | BlockLdm::External(_) => {
            run_block_compressor(ms, src, block, &mut rep, store, dms)
        }
    };
    // ZSTD_storeLastLiterals; btultra2 may have moved the window
    // (`ZSTD_initStats_ultra`), so the anchor is read back through `ms`.
    store
        .lits
        .extend_from_slice(&data[ms.pos(anchor)..block_end]);
    debug_assert_eq!(
        store.lits.len()
            + store
                .seqs
                .iter()
                .map(|s| s.match_len() as usize)
                .sum::<usize>(),
        block_len
    );
    Some(rep)
}

/// `ZSTD_buildSeqStore`: "don't even attempt compression below a certain
/// srcSize"; smaller blocks skip both stages and go RAW.
#[inline]
fn attempts_compression(block_len: usize) -> bool {
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

/// `ZSTD_entropyCompressSeqStore`: code `sections` of a `block_len`-byte
/// block into `cbuf` from the committed tables `prev`, returning the state
/// the block commits if written COMPRESSED, `None` when the payload does not
/// beat `block_len - ZSTD_minGain`.
fn entropy_code(
    sections: Sections,
    block_len: usize,
    prev: &BlockState,
    cparams: &CParams,
    cbuf: &mut Vec<u8>,
) -> Option<BlockState> {
    let Sections { lits, seqs, rep } = sections;
    cbuf.clear();
    let huf = huf::compress_literals_with(cbuf, lits, seqs.len(), &prev.huf, cparams);
    let fse = fse::encode_sequences_section_with(cbuf, seqs, &prev.fse, cparams)?;
    let max_c_size = block_len - CParams::min_gain(block_len, cparams.strategy);
    if cbuf.len() >= max_c_size {
        return None;
    }
    debug_assert!(cbuf.len() < ZSTD_BLOCKSIZE_MAX);
    Some(BlockState { rep, huf, fse })
}

/// The block-type decision of `ZSTD_compressBlock_internal` /
/// `ZSTD_compressSeqStore_singleBlock` once compression was attempted:
/// [`entropy_code`] `sections` into `cbuf`, then append one complete block
/// to `out`. `state` is committed, with `sections.rep`, only when the block
/// is written COMPRESSED. `is_first_block` disables RLE for the first block
/// of a job, as libzstd does for decoders <= 1.4.3.
#[allow(clippy::too_many_arguments)]
fn entropy_and_emit(
    src: &[u8],
    block: Range<usize>,
    sections: Sections,
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
    let next = entropy_code(sections, block_len, state.prev(), cparams, cbuf);
    let c_size = if next.is_some() { cbuf.len() } else { 0 };

    if !is_first_block && c_size < RLE_MAX_LENGTH && is_rle(data) {
        write_rle_block(out, data[0], data.len(), is_last);
        state.end_block(None);
        return BlockKind::Rle;
    }
    match next {
        None => {
            write_raw_block(out, data, is_last);
            state.end_block(None);
            BlockKind::Raw
        }
        Some(next) => {
            state.end_block(Some(next));
            write_compressed_block(out, cbuf, is_last);
            BlockKind::Compressed
        }
    }
}

/// Append `src[block]` to `out` from `built` (the block's sequence store
/// and the finder's repeat offsets after it, `None` when compression was
/// not attempted: `ZSTDbss_noCompress`, written RAW without the RLE check
/// by both block functions). With the splitter off (`parts` is `None`)
/// this is `ZSTD_compressBlock_internal`'s entropy stage, one block. With
/// it on, `ZSTD_compressBlock_splitBlock` after `ZSTD_buildSeqStore`:
/// `parts` from [`BlockSplitter::derive`], one block when empty, else one
/// per partition (`ZSTD_compressBlock_splitBlock_internal`), each coded with
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
    let Some((store, rep)) = built else {
        write_raw_block(out, &src[block], is_last);
        state.end_block(None);
        return false;
    };
    let parts = match parts {
        None | Some([]) => {
            return entropy_and_emit(
                src,
                block,
                Sections::whole(store, rep),
                cparams,
                is_first_block,
                is_last,
                state,
                cbuf,
                out,
            ) == BlockKind::Compressed;
        }
        Some(parts) => parts,
    };
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
            Sections {
                lits: &store.lits[part.lits.clone()],
                seqs: &store.seqs[part.seqs.clone()],
                rep: d_rep,
            },
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
/// offsets, then `emit_block`.
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
    ldm: &mut BlockLdm,
    dms: Option<DictMatchState>,
    out: &mut Vec<u8>,
) {
    let cparams = ms.cparams;
    let BlockScratch {
        store,
        cbuf,
        splitter,
        ..
    } = scratch;
    let entered = ms.enter_block(block.clone());
    let built = build_seq_store(ms, src, entered, state.prev().rep, store, ldm, dms);
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

/// The least unsplit size of a block whose match finding [`compress_blocks`]'
/// pipelined loop runs next to the previous block's entropy stage. The
/// overlap hands that entropy stage, and the sequences it reads, to another
/// core while the finder waits for it, so it pays only when the finder has
/// this much to do: on one 8-core CCD, single-job rssrc and elf frames of
/// 128 KiB and a 4-8 KiB second block lost 3-13% at L1 and L3 to the
/// overlap, broke even at 16-28 KiB and gained 3-8% at 32 KiB.
pub(crate) const MIN_OVERLAP: usize = 32 << 10;

/// How far the input [`compress_blocks`] is handed reaches, and what
/// follows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputEnd {
    /// More input may follow position `end` (`ZSTD_e_continue`): only the
    /// blocks that no later input can change are compressed, those that
    /// start more than `block_size_max` before `end`.
    Open(usize),
    /// `end` ends a `ZSTD_compressContinue` chunk (`ZSTD_e_flush`): no
    /// block crosses it.
    Chunk(usize),
    /// `end` ends the job; its last block is the frame's last when the job
    /// is the last one.
    JobEnd(usize),
}

/// [`BlockSizing`] over one job, resumable: the job's blocks are cut and
/// compressed by one [`compress_blocks`] call over the whole job, or by
/// several over its input as it arrives, with the same result as long as
/// no call ends a [`InputEnd::Chunk`] the one call would not have.
pub struct JobBlocks {
    sizing: BlockSizing,
    /// Position of the job's first byte; once the input moved down past
    /// it ([`JobBlocks::rebase`]), it lies before the input (wrapping), so
    /// every offset from it is a `wrapping_sub`.
    job_start: usize,
    first_job: bool,
    last_job: bool,
    /// Offset from the job start where the next block starts.
    done: usize,
    /// Offset from the job start where its first `ZSTD_compressContinue`
    /// chunk ended: ZSTDMT's first `chunk_size`, or the first
    /// [`InputEnd::Chunk`] before it.
    first_chunk_end: usize,
    /// Source minus written bytes over the job's blocks so far.
    gained: i64,
}

impl JobBlocks {
    /// The job starting at position `job_start`, none of it compressed.
    pub fn new(sizing: BlockSizing, job_start: usize, first_job: bool, last_job: bool) -> Self {
        Self {
            sizing,
            job_start,
            first_job,
            last_job,
            done: 0,
            first_chunk_end: sizing.chunk_size,
            gained: 0,
        }
    }

    /// Where the next block starts.
    pub fn next_start(&self) -> usize {
        self.job_start.wrapping_add(self.done)
    }

    /// The input moved `shift` bytes down: every position is `shift` lower.
    pub fn rebase(&mut self, shift: usize) {
        self.job_start = self.job_start.wrapping_sub(shift);
    }

    /// `min(remaining, blockSizeMax)` at `start`, with `remaining` the rest
    /// of its chunk as far as `input` tells: the block's size unless
    /// pre-split.
    fn unsplit_size(&self, start: usize, input: InputEnd) -> usize {
        let chunk = self.sizing.chunk_size;
        let offset = start.wrapping_sub(self.job_start);
        let mut chunk_end = (offset / chunk + 1).saturating_mul(chunk);
        if let InputEnd::Chunk(end) | InputEnd::JobEnd(end) = input {
            chunk_end = chunk_end.min(end.wrapping_sub(self.job_start));
        }
        (chunk_end - offset).min(self.sizing.block_size_max)
    }

    /// Whether `input` makes the next block ready (`JobBlocks::ready`).
    pub fn has_ready(&self, input: InputEnd) -> bool {
        self.ready(self.next_start(), input)
    }

    /// Whether the block at `start` is to be compressed now: there is
    /// input left, and with more to come, enough that the block's size and
    /// last-block flag cannot depend on it.
    fn ready(&self, start: usize, input: InputEnd) -> bool {
        match input {
            InputEnd::Open(end) => end > start && end - start > self.sizing.block_size_max,
            InputEnd::Chunk(end) | InputEnd::JobEnd(end) => start < end,
        }
    }

    /// Whether [`compress_blocks`]' pipelined loop overlaps the block at
    /// `start`, the next one written, with the one before it: the block is
    /// ready and holds [`MIN_OVERLAP`] bytes unsplit.
    fn overlaps_at(&self, start: usize, input: InputEnd) -> bool {
        self.ready(start, input) && self.unsplit_size(start, input) >= MIN_OVERLAP
    }

    /// Whether `input` gives [`compress_blocks`]' pipelined loop blocks to
    /// overlap: whether it overlaps the second ready block, the first one
    /// taken unsplit, as no later one holds more. A job's first block is
    /// never pre-split, so on a job's whole input this is exact.
    pub fn overlaps(&self, input: InputEnd) -> bool {
        let start = self.next_start();
        self.ready(start, input) && self.overlaps_at(start + self.unsplit_size(start, input), input)
    }

    /// Whether `block` is the frame's last.
    fn is_last(&self, block: &Range<usize>, input: InputEnd) -> bool {
        self.last_job && input == InputEnd::JobEnd(block.end)
    }

    /// The input ended at `input` once every ready block is written: a
    /// chunk end is where the header debit of `savings` starts, if no
    /// earlier one is.
    fn input_ended(&mut self, input: InputEnd) {
        if let InputEnd::Chunk(end) = input {
            self.first_chunk_end = self.first_chunk_end.min(end.wrapping_sub(self.job_start));
        }
    }

    /// `savings` (`consumedSrcSize - producedCSize` plus the chunk's blocks
    /// so far) at `start`, had the blocks before it gained `gained`.
    /// `producedCSize` counts the frame header only after the call that
    /// wrote it, so job 0's `savings` owe it from its second chunk on.
    fn savings(&self, start: usize, gained: i64) -> i64 {
        let owes_header =
            self.first_job && start.wrapping_sub(self.job_start) >= self.first_chunk_end;
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
        input: InputEnd,
        savings: i64,
        presplit: &mut PreSplitter,
    ) -> Range<usize> {
        let unsplit = self.unsplit_size(start, input);
        let size = match self.sizing.split_level {
            Some(level) if unsplit == SPLIT_BLOCK_SIZE && savings >= 3 => {
                presplit.split_block(&src[start..start + unsplit], level)
            }
            _ => unsplit,
        };
        start..start + size
    }

    /// `isFirstBlock` at `block`: set by every job context's
    /// `ZSTD_compressBegin` and cleared by its first block; a later ZSTDMT
    /// job's header flush (`ZSTD_compressContinue` with no input) returns
    /// before any block, so it holds for the first block of every job.
    fn is_first(&self, block: &Range<usize>) -> bool {
        block.start == self.job_start
    }

    /// The next block, from the blocks written so far.
    fn next(&self, src: &[u8], input: InputEnd, presplit: &mut PreSplitter) -> Range<usize> {
        let start = self.next_start();
        self.block(
            src,
            start,
            input,
            self.savings(start, self.gained),
            presplit,
        )
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
        input: InputEnd,
        least_gained: i64,
        presplit: &mut PreSplitter,
    ) -> Option<Range<usize>> {
        let least_savings = self.savings(start, least_gained);
        let may_split = self.sizing.split_level.is_some()
            && self.unsplit_size(start, input) == SPLIT_BLOCK_SIZE;
        if may_split && least_savings < 3 {
            return None;
        }
        Some(self.block(src, start, input, least_savings, presplit))
    }

    /// Account the next block, `block`, written as `written` bytes.
    fn wrote(&mut self, block: &Range<usize>, written: usize) {
        debug_assert_eq!(block.start, self.next_start());
        self.done = block.end.wrapping_sub(self.job_start);
        self.gained += block.len() as i64 - written as i64;
    }
}

/// `ZSTD_compress_frameChunk` over the input of a job up to `input`: the
/// ready blocks of `src` from `blocks`' next one, sized by its
/// [`BlockSizing`], appended to `out`, each through the post-sequence
/// splitter when `split`, with long distance matches from `ldm` and the
/// attached dictionary `dms`. With
/// `pipelined` (parallel feature only) block N's entropy stage and
/// emission run on rayon next to block N+1's match finding whenever N+1
/// holds `MIN_OVERLAP` bytes unsplit and every
/// block N is written as is proven (`proven_rep_after`) to be COMPRESSED,
/// so that the repeat offsets N+1 starts from are the ones the decoder
/// will hold, and N+1's size is fixed without N's compressed size;
/// otherwise N's entropy stage runs first and N+1 starts from the
/// committed offsets. Output is identical either way, and no block is in
/// flight when it returns.
#[allow(clippy::too_many_arguments)]
pub fn compress_blocks(
    ms: &mut MatchState,
    src: &[u8],
    blocks: &mut JobBlocks,
    input: InputEnd,
    split: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    dms: Option<DictMatchState>,
    out: &mut Vec<u8>,
    pipelined: bool,
) {
    #[cfg(feature = "parallel")]
    if pipelined {
        compress_blocks_pipelined(ms, src, blocks, input, split, state, scratch, ldm, dms, out);
        blocks.input_ended(input);
        return;
    }
    let _ = pipelined;
    while blocks.ready(blocks.next_start(), input) {
        let block = blocks.next(src, input, &mut scratch.presplit);
        let written = out.len();
        compress_block(
            ms,
            src,
            block.clone(),
            blocks.is_first(&block),
            blocks.is_last(&block, input),
            split,
            state,
            scratch,
            ldm,
            dms,
            out,
        );
        blocks.wrote(&block, out.len() - written);
    }
    blocks.input_ended(input);
}

/// Counters of [`compress_blocks`]' pipelined loop, for benches.
#[cfg(feature = "parallel")]
pub static PIPELINE_OVERLAPPED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Blocks with a successor of `MIN_OVERLAP` bytes whose proof failed
/// (entropy stage ran first).
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
    input: InputEnd,
    split: bool,
    state: &mut CommittedBlockState,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    dms: Option<DictMatchState>,
    out: &mut Vec<u8>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let cparams = ms.cparams;
    if !blocks.ready(blocks.next_start(), input) {
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
    let mut block = blocks.next(src, input, presplit);
    // `built`: `block`'s store is in `cur`, with the finder's offsets after it.
    let entered = ms.enter_block(block.clone());
    let mut built = build_seq_store(ms, src, entered, state.prev().rep, cur, ldm, dms);
    loop {
        let is_first_block = blocks.is_first(&block);
        let is_last = blocks.is_last(&block, input);
        // ZSTD_deriveBlockSplits runs against the state committed by the
        // previous block, before the next block's finder may start.
        let parts = split.then(|| match built {
            Some(_) => splitter.derive(cur, state.prev(), &cparams, block.len()),
            None => &[][..],
        });
        let following_start = block.end;
        // Pre-split, a next block that overlaps is at least 8 KiB, so it
        // attempts compression.
        let following_overlaps = blocks.overlaps_at(following_start, input);
        // Block N+1 may start before block N is written when N is proven
        // COMPRESSED (the offsets N+1 starts from) and N+1's size does not
        // depend on N's compressed size: it is not pre-split, or the least
        // savings a COMPRESSED N leaves already allow the split.
        let overlap = following_overlaps
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
                    blocks.next_unwritten(src, following_start, input, least_gained, presplit)?;
                Some((rep, rep_next, least_gained, following))
            })
            .flatten();
        let written = out.len();
        if let Some((rep, rep_next, least_gained, following)) = overlap {
            PIPELINE_OVERLAPPED.fetch_add(1, Relaxed);
            let entered = ms.enter_block(following.clone());
            let cur_store = &mut *cur;
            let state = &mut *state;
            // The match state stays on this thread, whose caches hold its
            // tables; block N's entropy stage is the part a thief takes.
            let (built_following, compressed) = rayon::join(
                || build_seq_store(ms, src, entered, rep_next, nxt, ldm, dms),
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
            );
            // The proof is what made block N+1 start from the decoder's
            // offsets, with the size the written block N gives it.
            assert!(compressed, "section bound proof failed");
            assert_eq!(state.prev().rep, rep_next, "repeat offset proof failed");
            blocks.wrote(&block, out.len() - written);
            assert!(blocks.gained >= least_gained, "savings bound proof failed");
            built = built_following;
            block = following;
        } else {
            if following_overlaps {
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
            if !blocks.ready(block.end, input) {
                return;
            }
            block = blocks.next(src, input, presplit);
            let entered = ms.enter_block(block.clone());
            built = build_seq_store(ms, src, entered, state.prev().rep, nxt, ldm, dms);
        }
        std::mem::swap(&mut cur, &mut nxt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::common::HASH_READ_SIZE;
    use crate::compress::lazy::SearchMethod;

    /// `is_rle` against its definition at every length around the 32-byte
    /// step and its 8-byte words, with one byte changed at each position,
    /// including the first byte, the remainder and the last byte.
    #[test]
    fn is_rle_matches_its_definition() {
        for len in (0..=100).chain([ZSTD_BLOCKSIZE_MAX - 1, ZSTD_BLOCKSIZE_MAX]) {
            let mut data = vec![0xa7u8; len];
            assert_eq!(is_rle(&data), len > 0, "len {len}");
            let positions = (0..len.min(100)).chain(len.saturating_sub(40)..len);
            for at in positions {
                data[at] ^= 1;
                assert!(!is_rle(&data) || len == 1, "len {len} at {at}");
                data[at] ^= 1;
            }
        }
    }

    /// `ZSTD_loadDictionaryContent` leaves a prefix of `HASH_READ_SIZE`
    /// bytes or less unindexed, `nextToUpdate` at its start and the row
    /// finder's tags not even cleared; one byte more goes to the table fill
    /// (which for fast and dfast inserts nothing yet) and `nextToUpdate`
    /// moves to its end. Every strategy and lazy finder.
    #[test]
    fn prefix_of_hash_read_size_bytes_is_not_indexed() {
        let data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) ^ 0x5a).collect();
        let finders = [
            (1, SearchMethod::HashChain),
            (3, SearchMethod::HashChain),
            (5, SearchMethod::HashChain),
            (5, SearchMethod::RowHash),
            (13, SearchMethod::BinaryTree),
            (16, SearchMethod::BinaryTree),
        ];
        for (level, method) in finders {
            for len in [HASH_READ_SIZE, HASH_READ_SIZE + 1] {
                let name = format!("L{level} {method:?}, {len} bytes");
                let mut ms = MatchState::new_for(CParams::for_level(level, 1 << 20), 0, method);
                ms.tables_mut().2.fill(0xa5);
                load_prefix(&mut ms, &data, 0..len);
                let (hash, chain, tag) = ms.tables();
                let untouched =
                    hash.iter().chain(chain).all(|&e| e == 0) && tag.iter().all(|&t| t == 0xa5);
                if len <= HASH_READ_SIZE {
                    assert!(untouched, "{name}: tables written");
                    assert_eq!(ms.next_to_update, ms.index(0), "{name}");
                } else {
                    assert_eq!(ms.next_to_update, ms.index(len), "{name}");
                }
            }
        }
    }

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
            let origin = src.len() - len;
            let mut ms = MatchState::new(cp, origin);
            load_prefix(&mut ms, &src, origin..src.len());
            let (hash, _, _) = ms.tables();
            ms.pos(hash.iter().filter(|&&e| e != 0).min().copied().unwrap() as usize)
        };
        assert!(lowest(cap + 1000) >= src.len() - cap);
        assert!(lowest(cap) >= src.len() - cap);
        assert!(lowest(cap - 1) > src.len() - cap);
        assert!(lowest(cap + 1000) < src.len() - cap + 64);
    }

    /// `ZSTD_dtlm_full` keeps every slot `ZSTD_dtlm_fast` writes, as each
    /// third position overwrites its slot either way, and gives empty slots
    /// the positions between them: fast's table and dfast's large one gain
    /// entries, dfast's small table is unchanged. Its entries are tagged
    /// with the low byte of a hash 8 bits wider than the slot's.
    #[test]
    fn full_table_load_fills_only_empty_slots() {
        use crate::compress::common::{hash_ptr, SHORT_CACHE_TAG_BITS};
        let data = crate::compress::common::testutil::synthetic_text(100_000, 5);
        for level in [1, 3] {
            let cp = CParams::for_level(level, 1 << 20);
            let load = |how| {
                let mut ms = MatchState::new(cp, 0);
                load_dict(&mut ms, &data, 0..data.len(), how);
                let src = ms.view(&data);
                let (hash, chain, _) = ms.tables();
                (hash.to_vec(), chain.to_vec(), ms.index(0), src)
            };
            let (fast_hash, fast_chain, base, _) = load(TableLoad::Fast);
            let (tagged_hash, tagged_chain, _, src) = load(TableLoad::Full);
            let tbits = cp.hash_log + SHORT_CACHE_TAG_BITS;
            for &e in tagged_hash.iter().filter(|&&e| e != 0) {
                let idx = (e >> SHORT_CACHE_TAG_BITS) as usize;
                // SAFETY: a filled position has 8 bytes after it in `data`.
                let hash_and_tag = unsafe {
                    match (cp.strategy, cp.min_match) {
                        (Strategy::DFast, _) => hash_ptr::<8>(src, idx, tbits),
                        (_, 5) => hash_ptr::<5>(src, idx, tbits),
                        (_, 6) => hash_ptr::<6>(src, idx, tbits),
                        (_, 7) => hash_ptr::<7>(src, idx, tbits),
                        _ => hash_ptr::<4>(src, idx, tbits),
                    }
                };
                assert_eq!(e & 0xff, hash_and_tag as u32 & 0xff, "L{level}: tag");
            }
            let untag = |t: Vec<u32>| t.into_iter().map(|e| e >> SHORT_CACHE_TAG_BITS).collect();
            let (full_hash, full_chain): (Vec<u32>, Vec<u32>) =
                (untag(tagged_hash), untag(tagged_chain));
            for (&f, &g) in fast_hash.iter().zip(&full_hash) {
                assert!(f == 0 || f == g, "L{level}: slot {f} became {g}");
                assert!(
                    f != 0 || g == 0 || !(g as usize - base).is_multiple_of(3),
                    "L{level}"
                );
            }
            let count = |t: &[u32]| t.iter().filter(|&&e| e != 0).count();
            assert!(count(&full_hash) > count(&fast_hash), "L{level}");
            assert_eq!(fast_chain, full_chain, "L{level}");
        }
    }

    /// A dictionary's `Valid` offset table is `Check` after any written
    /// block, compressed with it as the candidate's or not; its other
    /// tables stay `Valid`.
    #[test]
    fn end_block_demotes_a_valid_offset_table() {
        use crate::fse::{FseCTable, FseRepeat};
        let valid = || FseTableState::Valid(FseCTable::build(&[16, 16], 1, 5));
        let dict = BlockState {
            fse: FseState {
                ll: valid(),
                of: valid(),
                ml: valid(),
            },
            ..BlockState::initial()
        };
        let repeats = |s: &CommittedBlockState| {
            let f = &s.prev().fse;
            (f.ll.repeat(), f.of.repeat(), f.ml.repeat())
        };
        let demoted = (FseRepeat::Valid, FseRepeat::Check, FseRepeat::Valid);
        for next in [None, Some(dict.clone())] {
            let mut state = CommittedBlockState::new(dict.clone());
            state.end_block(next);
            assert_eq!(repeats(&state), demoted);
            state.end_block(None);
            assert_eq!(repeats(&state), demoted);
        }
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
        super::super::write_frame_header(
            &mut frame,
            Some(b.src.len() as u64),
            cparams.window_log,
            false,
            0,
        );
        frame.extend_from_slice(&blocks);
        assert_eq!(crate::decompress(&frame).unwrap(), b.src);
        assert_eq!(zstd::stream::decode_all(&frame[..]).unwrap(), b.src);
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

    fn job_blocks(split_level: Option<u8>, chunk_size: usize, job_start: usize) -> JobBlocks {
        let sizing = BlockSizing {
            block_size_max: ZSTD_BLOCKSIZE_MAX,
            split_level,
            chunk_size,
            header_len: 10,
        };
        JobBlocks::new(sizing, job_start, true, true)
    }

    /// `ZSTD_optimalBlockSize`: a full block splits at `savings` 3, not 2;
    /// a block under 128 KiB (job end, a smaller `blockSizeMax`) or with
    /// the pre-splitter off never does.
    #[test]
    fn block_splits_full_blocks_once_savings_reach_three() {
        let src = presplit_input();
        let mut ps = PreSplitter::default();
        let on = job_blocks(Some(1), usize::MAX, 0);
        let end = InputEnd::JobEnd(src.len());
        assert_eq!(on.block(&src, 0, end, 2, &mut ps), 0..SPLIT_BLOCK_SIZE);
        assert_eq!(on.block(&src, 0, end, 3, &mut ps), 0..40 << 10);
        assert_eq!(
            on.block(&src, 512 << 10, end, 3, &mut ps).len(),
            SPLIT_BLOCK_SIZE
        );
        let tail = src.len() - 1000;
        assert_eq!(on.block(&src, tail, end, 1 << 20, &mut ps), tail..src.len());
        let off = job_blocks(None, usize::MAX, 0);
        assert_eq!(
            off.block(&src, 0, end, 1 << 20, &mut ps),
            0..SPLIT_BLOCK_SIZE
        );
        let mut small = job_blocks(Some(1), usize::MAX, 0);
        small.sizing.block_size_max = 64 << 10;
        assert_eq!(small.block(&src, 0, end, 1 << 20, &mut ps), 0..64 << 10);
    }

    /// ZSTDMT chunks: no block crosses `job.start + k * 512 KiB`, and job 0
    /// owes the frame header from its second chunk on; single-threaded,
    /// the job is one chunk and the header is never owed.
    #[test]
    fn chunks_bound_blocks_and_owe_the_header_from_the_second() {
        let chunk = 4 * ZSTD_BLOCKSIZE_MAX;
        let job = 1000..1000 + (700 << 10);
        let end = InputEnd::JobEnd(job.end);
        let mt = job_blocks(Some(1), chunk, job.start);
        let second = job.start + chunk;
        assert_eq!(mt.unsplit_size(job.start, end), SPLIT_BLOCK_SIZE);
        assert_eq!(mt.unsplit_size(second - (8 << 10), end), 8 << 10);
        assert_eq!(mt.unsplit_size(second, end), SPLIT_BLOCK_SIZE);
        assert_eq!(mt.unsplit_size(job.end - 100, end), 100);
        assert_eq!(mt.savings(second - 1, 5), 5);
        assert_eq!(mt.savings(second, 5), -5);
        let sizing = mt.sizing;
        let later = JobBlocks::new(sizing, job.start, false, true);
        assert_eq!(later.savings(second, 5), 5);
        let st = job_blocks(Some(1), usize::MAX, job.start);
        assert_eq!(st.unsplit_size(second - (8 << 10), end), SPLIT_BLOCK_SIZE);
        assert_eq!(st.savings(job.end - 1, 5), 5);
    }

    /// Streaming input: an open end compresses only blocks starting more
    /// than `blockSizeMax` before it; a chunk end bounds blocks like the
    /// job end without being the last, and job 0 owes the header after
    /// the first one.
    #[test]
    fn input_ends_bound_blocks_and_chunks_owe_the_header() {
        let mut st = job_blocks(Some(1), usize::MAX, 0);
        let bsm = ZSTD_BLOCKSIZE_MAX;
        assert!(!st.ready(0, InputEnd::Open(bsm)));
        assert!(st.ready(0, InputEnd::Open(bsm + 1)));
        assert_eq!(st.unsplit_size(0, InputEnd::Open(bsm + 1)), bsm);
        assert!(!st.ready(0, InputEnd::Chunk(0)));
        assert!(st.ready(0, InputEnd::Chunk(1)));
        assert_eq!(st.unsplit_size(0, InputEnd::Chunk(1000)), 1000);
        assert!(!st.is_last(&(0..1000), InputEnd::Chunk(1000)));
        assert!(st.is_last(&(0..1000), InputEnd::JobEnd(1000)));
        assert_eq!(st.savings(1000, 5), 5);
        st.input_ended(InputEnd::Open(5000));
        assert_eq!(st.savings(5000, 5), 5);
        st.input_ended(InputEnd::Chunk(1000));
        st.input_ended(InputEnd::Chunk(3000));
        assert_eq!(st.savings(999, 5), 5);
        assert_eq!(st.savings(1000, 5), -5);
        st.rebase(100);
        assert_eq!(st.savings(899, 5), 5);
        assert_eq!(st.savings(900, 5), -5);
    }

    /// The pipelined loop has blocks to overlap once the second ready
    /// block holds `MIN_OVERLAP` bytes unsplit: at a job or chunk end, from
    /// `blockSizeMax + MIN_OVERLAP` bytes on; open, once the second block
    /// is ready, past `2 * blockSizeMax`; never with `blockSizeMax` under
    /// `MIN_OVERLAP`.
    #[test]
    fn overlaps_needs_a_ready_second_block_of_min_overlap() {
        let bsm = ZSTD_BLOCKSIZE_MAX;
        let at = 1000 + bsm + MIN_OVERLAP;
        let mut st = job_blocks(Some(1), usize::MAX, 1000);
        for end in [InputEnd::JobEnd, InputEnd::Chunk] {
            assert!(!st.overlaps(end(1000 + bsm)));
            assert!(!st.overlaps(end(at - 1)));
            assert!(st.overlaps(end(at)));
        }
        assert!(!st.overlaps(InputEnd::Open(1000 + 2 * bsm)));
        assert!(st.overlaps(InputEnd::Open(1000 + 2 * bsm + 1)));
        st.sizing.block_size_max = MIN_OVERLAP - 1;
        assert!(!st.overlaps(InputEnd::JobEnd(1 << 20)));
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
        let end = InputEnd::JobEnd(src.len());
        let st = job_blocks(Some(1), usize::MAX, 0);
        assert_eq!(st.next_unwritten(&src, 0, end, 2, &mut ps), None);
        assert_eq!(
            st.next_unwritten(&src, 0, end, 3, &mut ps),
            Some(0..40 << 10)
        );
        let tail = src.len() - 1000;
        assert_eq!(
            st.next_unwritten(&src, tail, end, -100, &mut ps),
            Some(tail..src.len())
        );
        let off = job_blocks(None, usize::MAX, 0);
        assert_eq!(
            off.next_unwritten(&src, 0, end, -100, &mut ps),
            Some(0..SPLIT_BLOCK_SIZE)
        );
        let chunk = 4 * ZSTD_BLOCKSIZE_MAX;
        let mt = job_blocks(Some(1), chunk, 0);
        assert_eq!(mt.next_unwritten(&src, chunk, end, 12, &mut ps), None);
        assert_eq!(
            mt.next_unwritten(&src, chunk, end, 13, &mut ps)
                .map(|b| b.len()),
            Some(SPLIT_BLOCK_SIZE)
        );
        assert_eq!(
            mt.next_unwritten(&src, chunk - (8 << 10), end, -100, &mut ps),
            Some(chunk - (8 << 10)..chunk)
        );
    }
}
