//! Zstandard frame compressor.
//!
//! libzstd's block architecture (`ZSTD_compress_frameChunk`): match finding
//! runs per block on a persistent [`MatchState`], sequences never cross a
//! block boundary, and the entropy stage consumes a per-block [`SeqStore`]
//! against the committed cross-block [`BlockState`].
//!
//! Above that, ZSTDMT's job architecture: the input is cut into jobs, each
//! compressed independently with its own [`MatchState`] and block state after
//! indexing an overlap of the preceding bytes (`ZSTDMT_computeOverlapSize`),
//! and the job outputs are concatenated. The job loop is the same with and
//! without the `parallel` feature, so both builds emit identical frames.

pub mod block;
pub mod bt;
pub mod common;
pub mod dfast;
pub mod fast;
pub mod lazy;
pub mod ldm;
pub mod matchstate;
pub mod opt;
pub mod params;
pub mod seqstore;

use crate::constants::*;
use block::{
    write_raw_block, write_rle_block, BlockLdm, BlockScratch, BlockState, CommittedBlockState,
    ZSTD_BLOCKHEADERSIZE,
};
use ldm::{LdmParams, LdmState, RawSeqStore, LDM_DEFAULT_WINDOW_LOG};
use matchstate::MatchState;
pub use params::{CParams, ParamSwitch, Strategy};
use params::{ZSTD_CLEVEL_DEFAULT, ZSTD_WINDOWLOG_ABSOLUTEMIN};
pub use seqstore::{Seq, SeqStore};
use std::ops::Range;

/// `ZSTDMT_JOBSIZE_MIN`: lower bound of an explicit job size.
pub const JOBSIZE_MIN: usize = 512 << 10;
/// `ZSTDMT_JOBSIZE_MAX` (64-bit): upper bound of an explicit job size.
pub const JOBSIZE_MAX: usize = 1 << 30;
/// `ZSTDMT_JOBLOG_MAX` (64-bit): upper bound of the default job size log.
const JOBLOG_MAX: u32 = 30;

/// Options for [`Compressor`] and [`compress_with`].
///
/// Inputs must be smaller than 4 GiB: match positions are `u32` indices into
/// the input, and [`Compressor::compress`] asserts the limit.
#[derive(Clone, Debug)]
pub struct CompressOptions {
    /// Compression level, `ZSTD_c_compressionLevel`. `<= 0` emits raw/RLE
    /// blocks only; `1..=22` map to libzstd's parameter rows.
    pub level: i32,
    /// Job size in bytes (`ZSTD_c_jobSize`). The input is cut into jobs of
    /// this many bytes; each job is compressed independently and, with the
    /// `parallel` feature, on its own rayon task. `None` selects
    /// `ZSTDMT_computeTargetJobLog`: `1 << min(max(20, window_log + 2), 30)`,
    /// i.e. 1 MiB for windows up to 256 KiB and 16 MiB for a 4 MiB window.
    /// An explicit size is clamped to `[JOBSIZE_MIN, JOBSIZE_MAX]`
    /// (512 KiB to 1 GiB) and then used as is. The frame never depends on
    /// the thread count, and is the same for a given job size whether or
    /// not the `parallel` feature is enabled. Smaller jobs give more
    /// parallelism and slightly worse ratios, since a job only sees
    /// `min(overlap, job start)` bytes of history from the previous job, see
    /// `overlap_log`. The job size is raised to the overlap size when it is
    /// smaller (`ZSTDMT_initCStream_internal`).
    pub job_size: Option<usize>,
    /// `ZSTD_c_overlapLog`, `0..=9`: the history a job indexes from before
    /// its start, as a fraction of the window. `0` selects
    /// `ZSTDMT_overlapLog_default` (6 for `Fast`..`Lazy`, 7 for `Lazy2`),
    /// `1` means no overlap, and `n` in `2..=9` means `window >> (9 - n)`,
    /// so `9` is the full window. Values above 9 panic (libzstd rejects
    /// them with `parameter_outOfBound`). With long distance matching the
    /// fraction is of `min(window, job size of the LDM formula / 4)`
    /// instead, and `1` no longer means no overlap
    /// (`ZSTDMT_computeOverlapSize`).
    pub overlap_log: u8,
    /// `ZSTD_c_enableLongDistanceMatching`: find matches up to a window
    /// back with a rolling hash over the whole window (see [`ldm`]).
    /// `Auto` enables it for the `btopt` strategies and up with a window
    /// log of 27 or more (level 22 on inputs above 64 MiB), which no
    /// ported strategy reaches yet. `Enable` raises the window log to 27
    /// (`ZSTD_LDM_DEFAULT_WINDOW_LOG`) before the size adjustment, and
    /// with it the job size and overlap (see `job_size`, `overlap_log`).
    pub ldm: ParamSwitch,
    /// `ZSTD_c_ldmHashLog`: `0` derives it (window log minus hash rate
    /// log, within `6..=30`), else `6..=30`.
    pub ldm_hash_log: u32,
    /// `ZSTD_c_ldmMinMatch`: `0` derives it (64, 32 for `btultra` and up),
    /// else `4..=4096`.
    pub ldm_min_match: u32,
    /// `ZSTD_c_ldmBucketSizeLog`: `0` derives it (the strategy number
    /// within `4..=8`), else `1..=8`; never above the hash log.
    pub ldm_bucket_size_log: u32,
    /// `ZSTD_c_ldmHashRateLog`: `0` derives it (window log minus an
    /// explicit hash log, else `7 - strategy / 3`), else `1..=25`.
    ///
    /// Out-of-range LDM values panic, as `overlap_log` does.
    pub ldm_hash_rate_log: u32,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            level: ZSTD_CLEVEL_DEFAULT,
            job_size: None,
            overlap_log: 0,
            ldm: ParamSwitch::Auto,
            ldm_hash_log: 0,
            ldm_min_match: 0,
            ldm_bucket_size_log: 0,
            ldm_hash_rate_log: 0,
        }
    }
}

impl CompressOptions {
    /// Opt-in multi-job preset: 2 MiB jobs (`ZSTD_c_jobSize`) with
    /// `ZSTD_c_overlapLog` 8 (half-window overlap). Inputs above 2 MiB split
    /// into several jobs, so the `parallel` feature can compress them
    /// concurrently; the frame still does not depend on the thread count.
    ///
    /// Measured on 8 threads against the default options (`tests/mt_grid.rs`,
    /// median of four runs) on the 8 MiB ELF and Rust-source corpus files:
    /// the size cost is at most 0.17% at levels 1 to 7 and 0.55% at level
    /// 11 (ELF), and the speedup (ELF / source) is 1.14x / 1.03x at level 1,
    /// 1.85x / 2.48x at 3, 1.74x / 2.27x at 5, 2.01x / 2.14x at 7 and
    /// 2.29x / 1.92x at 11. A 1 MiB input is one job under either option,
    /// so its frame and speed are unchanged.
    pub fn parallel(level: i32) -> Self {
        Self {
            level,
            job_size: Some(2 << 20),
            overlap_log: 8,
            ..Self::default()
        }
    }

    /// `ZSTD_getCParamsFromCCtxParams` and `ZSTD_resolveEnableLdm` for an
    /// input of `src_size` bytes: the frame's compression parameters and,
    /// when long distance matching resolves to enabled, its parameters
    /// (`ZSTD_ldm_adjustParameters`).
    fn frame_params(&self, src_size: usize) -> (CParams, Option<LdmParams>) {
        let requested = LdmParams::requested(
            self.ldm_hash_log,
            self.ldm_min_match,
            self.ldm_bucket_size_log,
            self.ldm_hash_rate_log,
        );
        let mut cparams = CParams::for_level(self.level, src_size);
        if self.ldm == ParamSwitch::Enable {
            cparams.window_log = LDM_DEFAULT_WINDOW_LOG;
            cparams = cparams.adjust(src_size);
        }
        let enabled = match self.ldm {
            // wlog >= 27, strategy >= btopt
            ParamSwitch::Auto => cparams.strategy as u32 >= ZSTD_BTOPT && cparams.window_log >= 27,
            ParamSwitch::Enable => true,
            ParamSwitch::Disable => false,
        };
        (cparams, enabled.then(|| requested.adjusted(&cparams)))
    }
}

/// `ZSTD_btopt`.
const ZSTD_BTOPT: u32 = 7;

/// Compress `data` into a zstd frame at `level`.
///
/// Returns a valid zstd frame decompressible by any conformant decoder.
pub fn compress(data: &[u8], level: i32) -> Vec<u8> {
    compress_with(
        data,
        &CompressOptions {
            level,
            ..CompressOptions::default()
        },
    )
}

/// Convenience wrapper: [`compress`] at level 1.
pub fn compress_to_vec(data: &[u8]) -> Vec<u8> {
    compress(data, 1)
}

/// Compress `data` into a zstd frame with `opts`, through a one-off
/// [`Compressor`].
pub fn compress_with(data: &[u8], opts: &CompressOptions) -> Vec<u8> {
    Compressor::new(opts.clone()).compress_to_vec(data)
}

/// A reusable `ZSTD_CCtx`: the options plus, as ZSTDMT keeps one context
/// per job, a pool of per-job match states and block buffers that grows on
/// demand and is kept across calls. A call sizes the tables for its input
/// like `ZSTD_resetCCtx_internal` with `ZSTDcrp_makeClean`: allocations
/// that are large enough are zeroed and kept, smaller ones replaced. Every
/// frame is identical to [`compress_with`]'s.
pub struct Compressor {
    opts: CompressOptions,
    jobs: Vec<JobContext>,
    /// The long distance matching state, once a frame has used it.
    ldm: Option<LdmState>,
}

/// One job's reusable state: its match state once a job has run, its
/// block buffers and its long distance matches (ZSTDMT's `rawSeqStore`).
#[derive(Default)]
struct JobContext {
    ms: Option<MatchState>,
    scratch: BlockScratch,
    ldm_seqs: RawSeqStore,
}

impl Compressor {
    pub fn new(opts: CompressOptions) -> Self {
        Self {
            opts,
            jobs: Vec::new(),
            ldm: None,
        }
    }

    /// Append one frame holding `src` to `out`.
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) {
        assert!(
            src.len() < u32::MAX as usize,
            "inputs of 4 GiB or more are not supported (match indices are u32)"
        );
        let (cparams, ldm_params) = self.opts.frame_params(src.len());
        out.reserve(src.len() + 64);
        write_frame_header(out, src.len() as u64, cparams.window_log);

        if src.is_empty() {
            write_raw_block(out, &[], true);
            return;
        }

        // blockSizeMax = MIN(ZSTD_BLOCKSIZE_MAX, 1 << windowLog)
        let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
        let n_blocks = src.len().div_ceil(block_size);

        if self.opts.level <= 0 {
            for (i, chunk) in src.chunks(block_size).enumerate() {
                let is_last = i + 1 == n_blocks;
                if block::is_rle(chunk) {
                    write_rle_block(out, chunk[0], chunk.len(), is_last);
                } else {
                    write_raw_block(out, chunk, is_last);
                }
            }
            return;
        }

        let ldm_on = ldm_params.is_some();
        let overlap = overlap_size(&cparams, self.opts.overlap_log, ldm_on);
        let job_size = job_size_for(self.opts.job_size, &cparams, ldm_on, overlap);
        let jobs = job_ranges(src.len(), job_size);
        let n_jobs = jobs.len();
        if self.jobs.len() < n_jobs {
            self.jobs.resize_with(n_jobs, JobContext::default);
        }
        let pipelined = cfg!(feature = "parallel");
        if let Some(params) = ldm_params {
            let mut state = match self.ldm.take() {
                Some(mut state) => {
                    state.reset(params, 0);
                    state
                }
                None => LdmState::new(params, 0),
            };
            if src.len() <= JOBSIZE_MIN {
                // ZSTD_CCtx_init_compressStream2: no ZSTDMT up to
                // ZSTDMT_JOBSIZE_MIN; the single context generates each
                // block's sequences as it goes.
                debug_assert_eq!(n_jobs, 1);
                let ctx = &mut self.jobs[0];
                compress_job(
                    src,
                    cparams,
                    block_size,
                    overlap,
                    jobs[0].clone(),
                    true,
                    true,
                    pipelined,
                    &mut ctx.ms,
                    &mut ctx.scratch,
                    &mut BlockLdm::Internal(&mut state),
                    out,
                );
                self.ldm = Some(state);
                return;
            }
            // ZSTDMT_serialState_genSequences: every job's sequences from
            // the one state, in job order, before the jobs run.
            let max_seqs = job_size / params.min_match_length as usize;
            for (job, ctx) in jobs.iter().zip(&mut self.jobs) {
                state.generate_sequences(src, job.clone(), max_seqs, &mut ctx.ldm_seqs);
            }
            self.ldm = Some(state);
        }
        run_jobs(
            &jobs,
            &mut self.jobs[..n_jobs],
            pipelined,
            |k, job, ctx, out| {
                let JobContext {
                    ms,
                    scratch,
                    ldm_seqs,
                } = ctx;
                let mut ldm = if ldm_on {
                    BlockLdm::External(ldm_seqs)
                } else {
                    BlockLdm::Off
                };
                compress_job(
                    src,
                    cparams,
                    block_size,
                    overlap,
                    job,
                    k == 0,
                    k + 1 == n_jobs,
                    pipelined,
                    ms,
                    scratch,
                    &mut ldm,
                    out,
                )
            },
            out,
        );
    }

    /// One frame holding `src`.
    pub fn compress_to_vec(&mut self, src: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        self.compress(src, &mut out);
        out
    }
}

/// Upper bound on the blocks a job of `len` bytes produces: every block
/// RAW with its header.
fn job_bound(len: usize, block_size: usize) -> usize {
    len + ZSTD_BLOCKHEADERSIZE * len.div_ceil(block_size)
}

/// `ZSTDMT_compressionJob`: compress `data[job]` into a sequence of blocks
/// appended to `out`, on the match state in `ms_slot` reset for this job
/// (allocated on first use) with `scratch`'s buffers and the long distance
/// matches of `ldm`. Job 0 starts from `repStartValue` with
/// `window_low = 1`; a later job indexes `overlap` bytes before its start
/// (`ZSTD_loadDictionaryContent` on the raw-content prefix), starts with
/// invalidated repeat offsets and no entropy tables, so its first block
/// cannot reference state the decoder obtained from the previous job.
#[allow(clippy::too_many_arguments)]
fn compress_job(
    data: &[u8],
    cparams: CParams,
    block_size: usize,
    overlap: usize,
    job: Range<usize>,
    first_job: bool,
    last_job: bool,
    pipelined: bool,
    ms_slot: &mut Option<MatchState>,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    out: &mut Vec<u8>,
) {
    let window_low = if first_job {
        1
    } else {
        job.start.saturating_sub(overlap).max(1)
    };
    let mut ms = match ms_slot.take() {
        Some(mut ms) => {
            ms.reset(cparams, window_low);
            ms
        }
        None => MatchState::new(cparams, window_low),
    };
    let mut initial = BlockState::initial();
    if !first_job {
        block::load_prefix(&mut ms, data, window_low..job.start);
        initial.invalidate_rep_codes();
    }
    let mut state = CommittedBlockState::new(initial);
    scratch.reserve(block_size);
    out.reserve(job_bound(job.len(), block_size));
    block::compress_blocks(
        &mut ms, data, job, block_size, first_job, last_job, &mut state, scratch, ldm, out,
        pipelined,
    );
    *ms_slot = Some(ms);
}

/// Run `f` over every job, in job order, appending to `out`; `ctxs[k]` is
/// job `k`'s context. Job 0 always writes straight into `out`. With the
/// feature enabled and `parallel`, the remaining jobs run on rayon into
/// buffers of their own while job 0 runs, and are appended afterwards; the
/// serial loop hands every job `out`. The job function is the same either
/// way, so the frame is identical.
fn run_jobs<F>(
    jobs: &[Range<usize>],
    ctxs: &mut [JobContext],
    parallel: bool,
    f: F,
    out: &mut Vec<u8>,
) where
    F: Fn(usize, Range<usize>, &mut JobContext, &mut Vec<u8>) + Sync,
{
    debug_assert_eq!(jobs.len(), ctxs.len());
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        let (Some((first, rest)), Some((first_ctx, rest_ctxs))) =
            (jobs.split_first(), ctxs.split_first_mut())
        else {
            return;
        };
        let ((), rest_out) = rayon::join(
            || f(0, first.clone(), first_ctx, out),
            || {
                rest.par_iter()
                    .zip(rest_ctxs.par_iter_mut())
                    .enumerate()
                    .map(|(i, (job, ctx))| {
                        let mut o = Vec::new();
                        f(i + 1, job.clone(), ctx, &mut o);
                        o
                    })
                    .collect::<Vec<_>>()
            },
        );
        for o in &rest_out {
            out.extend_from_slice(o);
        }
        return;
    }
    let _ = parallel;
    for (k, (job, ctx)) in jobs.iter().zip(ctxs.iter_mut()).enumerate() {
        f(k, job.clone(), ctx, out);
    }
}

/// `ZSTDMT_initCStream_internal`'s `targetSectionSize`: an explicit size
/// clamped to `[ZSTDMT_JOBSIZE_MIN, ZSTDMT_JOBSIZE_MAX]`, else
/// `1 << ZSTDMT_computeTargetJobLog`, and at least `overlap` ("job size
/// must be >= overlap size"). `ldm` is whether long distance matching is
/// enabled.
pub fn job_size_for(
    requested: Option<usize>,
    cparams: &CParams,
    ldm: bool,
    overlap: usize,
) -> usize {
    let section = match requested {
        Some(n) => n.clamp(JOBSIZE_MIN, JOBSIZE_MAX),
        None => 1usize << target_job_log(cparams, ldm),
    };
    section.max(overlap)
}

/// `ZSTDMT_computeTargetJobLog`: from the window log, or with long distance
/// matching (whose window is typically oversized) from the cycle log.
fn target_job_log(cparams: &CParams, ldm: bool) -> u32 {
    let job_log = if ldm {
        // ZSTD_cycleLog(chainLog, strategy): btScale = strategy >= btlazy2
        let cycle_log = cparams.chain_log - (cparams.strategy as u32 >= ZSTD_BTLAZY2) as u32;
        21.max(cycle_log + 3)
    } else {
        20.max(cparams.window_log + 2)
    };
    job_log.min(JOBLOG_MAX)
}

/// `ZSTD_btlazy2`.
const ZSTD_BTLAZY2: u32 = 6;

/// Job boundaries: `[0, job_size)`, `[job_size, 2 * job_size)`, ... with the
/// last job truncated to `len`.
pub fn job_ranges(len: usize, job_size: usize) -> Vec<Range<usize>> {
    (0..len)
        .step_by(job_size)
        .map(|start| start..(start + job_size).min(len))
        .collect()
}

/// `ZSTDMT_computeOverlapSize`: `overlap_log` as `ZSTD_c_overlapLog` (see
/// [`CompressOptions::overlap_log`]); without long distance matching the
/// result is `0` or `1 << (window_log - (9 - overlap_log))`, with it
/// `1 << (min(window_log, target_job_log - 2) - (9 - overlap_log))`.
pub fn overlap_size(cparams: &CParams, overlap_log: u8, ldm: bool) -> usize {
    assert!(
        overlap_log <= 9,
        "overlap_log {overlap_log} out of range 0..=9"
    );
    // ZSTDMT_overlapLog
    let overlap_log = match overlap_log {
        // ZSTDMT_overlapLog_default
        0 => match cparams.strategy {
            Strategy::BtUltra2 => 9,
            Strategy::BtOpt | Strategy::BtUltra => 8,
            Strategy::Lazy2 => 7,
            Strategy::Fast | Strategy::DFast | Strategy::Greedy | Strategy::Lazy => 6,
        },
        n => n as u32,
    };
    let overlap_rlog = 9 - overlap_log;
    let ov_log = if ldm {
        // In Long Range Mode, the windowLog is typically oversized: ovLog
        // becomes a fraction of the jobSize, rather than windowSize.
        cparams.window_log.min(target_job_log(cparams, true) - 2) - overlap_rlog
    } else if overlap_rlog >= 8 {
        0
    } else {
        cparams.window_log - overlap_rlog
    };
    if ov_log == 0 {
        0
    } else {
        1usize << ov_log
    }
}

/// `ZSTD_writeFrameHeader` with no dictionary and no checksum:
/// Single_Segment iff the window covers the whole content, otherwise a
/// Window_Descriptor with mantissa 0 derived from `window_log`.
fn write_frame_header(out: &mut Vec<u8>, content_size: u64, window_log: u32) {
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
    let window_size = 1u64 << window_log;
    let single_segment = window_size >= content_size;
    let fcs_code = (content_size >= 256) as u8
        + (content_size >= 65536 + 256) as u8
        + (content_size >= 0xFFFF_FFFF) as u8;
    let descriptor = ((single_segment as u8) << 5) | (fcs_code << 6);
    out.push(descriptor);
    if !single_segment {
        out.push(((window_log - ZSTD_WINDOWLOG_ABSOLUTEMIN) << 3) as u8);
    }
    match fcs_code {
        0 => {
            if single_segment {
                out.push(content_size as u8);
            }
        }
        1 => out.extend_from_slice(&((content_size - 256) as u16).to_le_bytes()),
        2 => out.extend_from_slice(&(content_size as u32).to_le_bytes()),
        _ => out.extend_from_slice(&content_size.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_empty() {
        let compressed = compress(&[], 1);
        assert!(compressed.len() >= 5); // magic + header + empty block
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert_eq!(crate::decompress(&compressed).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn compress_small() {
        let data = b"hello world";
        let compressed = compress(data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert!(compressed.len() > 5);
        assert_eq!(crate::decompress(&compressed).unwrap(), data);
    }

    #[test]
    fn compress_repetitive() {
        let data = vec![42u8; 4096];
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert_eq!(crate::decompress(&compressed).unwrap(), data);
    }

    #[test]
    fn compress_real_data() {
        let data: Vec<u8> = (0..1024u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert_eq!(crate::decompress(&compressed).unwrap(), data);
    }

    /// Golden test: roundtrip through our compressor → our decompressor.
    #[test]
    fn roundtrip_self_contained() {
        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros", vec![0u8; 4096]),
            (
                "sequential",
                (0..4096u32).flat_map(|i| i.to_le_bytes()).collect(),
            ),
            (
                "f32_data",
                (0..256u32)
                    .flat_map(|i| (i as f32 * 1.5).to_le_bytes())
                    .collect(),
            ),
            ("repetitive", b"hello world! ".repeat(100)),
            ("small", b"abc".to_vec()),
        ];

        for (name, data) in &test_cases {
            for level in [0, 1, 3, 7, 11, 19] {
                let compressed = compress(data, level);
                let decompressed = crate::decompress(&compressed)
                    .unwrap_or_else(|e| panic!("{}: decompress failed: {}", name, e));
                assert_eq!(decompressed.len(), data.len(), "{}: length mismatch", name);
                assert_eq!(&decompressed, data, "{}: data mismatch", name);
            }
        }
    }

    /// A RAW block between two compressed blocks must not advance the
    /// committed Huffman table: block 3 may only go Treeless against the
    /// tree the decoder actually saw in block 1.
    #[test]
    fn raw_block_between_compressed_blocks_keeps_prev_state() {
        let mut data = Vec::new();
        let text: Vec<u8> = (0..ZSTD_BLOCKSIZE_MAX)
            .map(|i| b"the quick brown fox "[i % 20])
            .collect();
        data.extend_from_slice(&text);
        // pseudo-random, incompressible
        let mut x = 0x2545F4914F6CDD1Du64;
        for _ in 0..ZSTD_BLOCKSIZE_MAX {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            data.push((x >> 56) as u8);
        }
        data.extend_from_slice(&text);
        for level in [1, 3, 7] {
            let c = compress(&data, level);
            assert_eq!(crate::decompress(&c).unwrap(), data, "level {level}");
        }
    }

    /// Pseudo-random bytes (xorshift64), incompressible.
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 56) as u8
            })
            .collect()
    }

    fn text(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len + 64);
        let mut i = 0u32;
        while v.len() < len {
            v.extend_from_slice(format!("line {} of the corpus {}\n", i, i % 37).as_bytes());
            i += 1;
        }
        v.truncate(len);
        v
    }

    /// Level 1's parameters with `window_log` and `chain_log` replaced.
    fn logs(window_log: u32, chain_log: u32) -> CParams {
        let mut cp = CParams::for_level(1, 8 << 20);
        cp.window_log = window_log;
        cp.chain_log = chain_log;
        cp
    }

    #[test]
    fn job_sizing() {
        let wlog = |w| logs(w, 13);
        // explicit: clamped to [JOBSIZE_MIN, JOBSIZE_MAX], otherwise used as is
        assert_eq!(job_size_for(Some(0), &wlog(19), false, 0), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(1), &wlog(19), false, 0), JOBSIZE_MIN);
        assert_eq!(
            job_size_for(Some(JOBSIZE_MIN + 1), &wlog(19), false, 0),
            JOBSIZE_MIN + 1
        );
        assert_eq!(
            job_size_for(Some(usize::MAX), &wlog(19), false, 0),
            JOBSIZE_MAX
        );
        // default: 1 << min(max(20, window_log + 2), 30)
        assert_eq!(job_size_for(None, &wlog(10), false, 0), 1 << 20);
        assert_eq!(job_size_for(None, &wlog(18), false, 0), 1 << 20);
        assert_eq!(job_size_for(None, &wlog(19), false, 0), 1 << 21);
        assert_eq!(job_size_for(None, &wlog(22), false, 0), 1 << 24);
        assert_eq!(job_size_for(None, &wlog(31), false, 0), 1 << 30);
        assert_eq!(job_ranges(0, 1 << 17), Vec::<Range<usize>>::new());
        assert_eq!(
            job_ranges(1_000_001, 1_000_000),
            vec![0..1_000_000, 1_000_000..1_000_001]
        );
        assert_eq!(job_ranges(1, 1 << 17), vec![0..1]);
        assert_eq!(
            job_ranges((1 << 18) + 5, 1 << 17),
            vec![0..1 << 17, 1 << 17..1 << 18, 1 << 18..(1 << 18) + 5]
        );
        let fast = CParams::for_level(1, 8 << 20);
        assert_eq!(overlap_size(&fast, 0, false), 1 << (fast.window_log - 3));
        let lazy2 = CParams::for_level(11, 8 << 20);
        assert_eq!(lazy2.strategy, Strategy::Lazy2);
        assert_eq!(overlap_size(&lazy2, 0, false), 1 << (lazy2.window_log - 2));
        // ZSTD_c_overlapLog: 1 = none, n = window >> (9 - n), 9 = window.
        for cp in [fast, lazy2] {
            assert_eq!(overlap_size(&cp, 1, false), 0);
            assert_eq!(overlap_size(&cp, 2, false), 1 << (cp.window_log - 7));
            assert_eq!(overlap_size(&cp, 6, false), 1 << (cp.window_log - 3));
            assert_eq!(overlap_size(&cp, 9, false), 1 << cp.window_log);
        }
        // A job is at least the overlap; JOBSIZE_MIN applies to explicit sizes.
        assert_eq!(job_size_for(Some(1), &wlog(19), false, 1 << 22), 1 << 22);
        assert_eq!(
            job_size_for(Some(1), &wlog(19), false, 1 << 18),
            JOBSIZE_MIN
        );
        assert_eq!(job_size_for(None, &wlog(23), false, 1 << 23), 1 << 25);
        assert_eq!(
            job_size_for(Some(JOBSIZE_MIN), &wlog(23), false, 1 << 23),
            1 << 23
        );
    }

    /// With long distance matching the default job log is
    /// `min(max(21, cycleLog + 3), 30)` whatever the window, and the
    /// overlap is `1 << (min(window_log, job_log - 2) - (9 - overlap_log))`,
    /// nonzero even for overlap_log 1 (`ZSTDMT_computeTargetJobLog`,
    /// `ZSTDMT_computeOverlapSize`).
    #[test]
    fn job_sizing_with_ldm() {
        // cycleLog 13 + 3 < 21; the window does not matter.
        for w in [10, 23, 27, 31] {
            assert_eq!(job_size_for(None, &logs(w, 13), true, 0), 1 << 21);
        }
        assert_eq!(job_size_for(None, &logs(27, 18), true, 0), 1 << 21);
        assert_eq!(job_size_for(None, &logs(27, 19), true, 0), 1 << 22);
        assert_eq!(job_size_for(None, &logs(27, 24), true, 0), 1 << 27);
        assert_eq!(job_size_for(None, &logs(27, 28), true, 0), 1 << 30);
        // Explicit sizes and the overlap floor are unchanged.
        assert_eq!(job_size_for(Some(1), &logs(27, 24), true, 0), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(1), &logs(27, 13), true, 1 << 20), 1 << 20);
        // Job log 21: the overlap is a fraction of 1 << 19, not of the
        // window.
        let cp = logs(27, 13);
        assert_eq!(overlap_size(&cp, 9, true), 1 << 19);
        assert_eq!(overlap_size(&cp, 6, true), 1 << 16);
        assert_eq!(overlap_size(&cp, 0, true), 1 << 16);
        assert_eq!(overlap_size(&cp, 1, true), 1 << 11);
        // A window below job_log - 2 is the base.
        let small = logs(17, 13);
        assert_eq!(overlap_size(&small, 9, true), 1 << 17);
        assert_eq!(overlap_size(&small, 2, true), 1 << 10);
        // Job log 27 (chain log 24), Lazy2's default overlap log 7.
        let mut lazy2 = logs(27, 24);
        lazy2.strategy = Strategy::Lazy2;
        assert_eq!(overlap_size(&lazy2, 0, true), 1 << 23);
    }

    /// `len + gap` noise bytes, then the first `len` of them again: a
    /// repeat `len + gap` bytes back, beyond the window of levels 1-11
    /// without long distance matching once that exceeds 4 MiB.
    fn far_repeat(len: usize, gap: usize) -> Vec<u8> {
        let mut v = noise(len + gap, 11);
        v.extend_from_within(..len);
        v
    }

    fn ldm_opts(level: i32, ldm: ParamSwitch, job_size: Option<usize>) -> CompressOptions {
        CompressOptions {
            level,
            job_size,
            ldm,
            ..Default::default()
        }
    }

    /// Long distance matching frames decode through both decoders on the
    /// single-context path (up to `JOBSIZE_MIN`) and over one or several
    /// jobs (the last block 3 bytes, too small to compress), find the
    /// repeat that the frame without it stores as literals, and a reused
    /// `Compressor` writes what fresh ones do.
    #[test]
    fn ldm_frames_roundtrip_and_find_far_repeats() {
        let single = far_repeat(160 << 10, 160 << 10);
        let jobs = far_repeat(1 << 20, (4 << 20) + 3);
        let empty = Vec::new();
        for level in [1, 3, 7, 11] {
            for job_size in [None, Some(JOBSIZE_MIN)] {
                let opts = ldm_opts(level, ParamSwitch::Enable, job_size);
                let mut cx = Compressor::new(opts.clone());
                for data in [&jobs, &single, &empty, &jobs] {
                    let frame = cx.compress_to_vec(data);
                    let name = format!("L{level} job {job_size:?} {} bytes", data.len());
                    assert!(frame == compress_with(data, &opts), "{name}: reuse");
                    assert_eq!(&crate::decompress(&frame).unwrap(), data, "{name}");
                    let theirs = zstd::stream::decode_all(&frame[..]).unwrap();
                    assert_eq!(&theirs, data, "{name}");
                }
                let off = compress_with(&jobs, &ldm_opts(level, ParamSwitch::Disable, job_size));
                let on = compress_with(&jobs, &opts);
                assert!(
                    on.len() + (900 << 10) < off.len(),
                    "L{level} job {job_size:?}: {} with LDM, {} without",
                    on.len(),
                    off.len()
                );
            }
        }
    }

    /// With long distance matching over several jobs, the parallel and the
    /// serial job loop write the same bytes from the sequences generated in
    /// job order, and those are the frame's.
    #[test]
    fn ldm_parallel_and_serial_job_loops_agree() {
        let data = far_repeat(1 << 20, 4 << 20);
        let src = data.as_slice();
        for level in [1, 3, 7, 11] {
            let opts = ldm_opts(level, ParamSwitch::Enable, Some(JOBSIZE_MIN));
            let (cparams, ldm) = opts.frame_params(src.len());
            let ldm = ldm.expect("enabled");
            let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
            let overlap = overlap_size(&cparams, opts.overlap_log, true);
            let job_size = job_size_for(opts.job_size, &cparams, true, overlap);
            let jobs = job_ranges(src.len(), job_size);
            let n = jobs.len();
            assert!(n >= 6, "L{level}: {n} jobs");
            let generate = |ctxs: &mut [JobContext]| {
                let mut state = LdmState::new(ldm, 0);
                let max_seqs = job_size / ldm.min_match_length as usize;
                for (job, ctx) in jobs.iter().zip(ctxs) {
                    state.generate_sequences(src, job.clone(), max_seqs, &mut ctx.ldm_seqs);
                }
            };
            let f = |pipelined: bool| {
                move |k: usize, job: Range<usize>, ctx: &mut JobContext, out: &mut Vec<u8>| {
                    let JobContext {
                        ms,
                        scratch,
                        ldm_seqs,
                    } = ctx;
                    compress_job(
                        src,
                        cparams,
                        block_size,
                        overlap,
                        job,
                        k == 0,
                        k + 1 == n,
                        pipelined,
                        ms,
                        scratch,
                        &mut BlockLdm::External(ldm_seqs),
                        out,
                    )
                }
            };
            let mut ctxs: Vec<JobContext> = (0..n).map(|_| JobContext::default()).collect();
            generate(&mut ctxs);
            let mut par = Vec::new();
            run_jobs(&jobs, &mut ctxs, true, f(true), &mut par);
            generate(&mut ctxs);
            let mut seq = Vec::new();
            run_jobs(&jobs, &mut ctxs, false, f(false), &mut seq);
            assert!(par == seq, "L{level}: job outputs differ");
            assert!(compress_with(src, &opts).ends_with(&seq), "L{level}");
        }
    }

    /// `Enable` selects window log 27 before the size adjustment; `Auto`
    /// stays off for the ported strategies (all below `btopt`), so its
    /// frames are `Disable`'s.
    #[test]
    fn ldm_switch_resolution() {
        let data = far_repeat(300 << 10, 400 << 10);
        for level in 1..=22 {
            let (cp, ldm) = ldm_opts(level, ParamSwitch::Enable, None).frame_params(3 << 20);
            assert_eq!(cp.window_log, 22, "L{level}: 3 MiB adjusts 27 to 22");
            assert!(ldm.is_some_and(|p| p.window_log == 22));
            let big = ldm_opts(level, ParamSwitch::Enable, None).frame_params(1 << 30);
            assert_eq!(big.0.window_log, 27, "L{level}");
            for size in [3 << 20, 1 << 30] {
                let (auto_cp, auto_ldm) =
                    ldm_opts(level, ParamSwitch::Auto, None).frame_params(size);
                assert!(auto_ldm.is_none(), "L{level} {size}");
                assert_eq!(auto_cp, CParams::for_level(level, size), "L{level} {size}");
                let off = ldm_opts(level, ParamSwitch::Disable, None).frame_params(size);
                assert_eq!(off, (auto_cp, None), "L{level} {size}");
            }
            let auto = compress_with(&data, &ldm_opts(level, ParamSwitch::Auto, None));
            let off = compress_with(&data, &ldm_opts(level, ParamSwitch::Disable, None));
            assert!(auto == off, "L{level}");
        }
    }

    /// Out-of-range long distance matching parameters panic, as libzstd
    /// refuses them with `parameter_outOfBound`.
    #[test]
    #[should_panic(expected = "ldm_bucket_size_log 9 out of range")]
    fn ldm_parameter_out_of_range_panics() {
        compress_with(
            b"abc",
            &CompressOptions {
                ldm_bucket_size_log: 9,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn overlap_log_above_nine_panics() {
        overlap_size(&CParams::for_level(1, 1 << 20), 10, false);
    }

    /// The preset runs three jobs on 4.25 MiB, counted as the job contexts
    /// a fresh `Compressor` leaves holding a match state, and its frame
    /// decodes through both decoders.
    #[test]
    fn parallel_preset_roundtrips() {
        let mut data = text(2 << 20);
        data.extend_from_slice(&noise(256 << 10, 7));
        data.extend_from_slice(&text(2 << 20));
        for level in [1, 3, 7, 11] {
            let opts = CompressOptions::parallel(level);
            assert_eq!((opts.job_size, opts.overlap_log), (Some(2 << 20), 8));
            let mut cx = Compressor::new(opts.clone());
            let frame = cx.compress_to_vec(&data);
            let ran = cx.jobs.iter().filter(|j| j.ms.is_some()).count();
            assert_eq!(ran, 3, "L{level}: jobs run");
            assert!(frame == compress_with(&data, &opts), "L{level}");
            assert_eq!(crate::decompress(&frame).unwrap(), data, "L{level}");
            let theirs = zstd::stream::decode_all(&frame[..]).unwrap();
            assert_eq!(theirs, data, "L{level}");
        }
    }

    /// Every overlap boundary (none, smallest, default, full window) yields a
    /// frame both decoders accept, over several jobs; overlap_log 0 equals
    /// the strategy's explicit default.
    #[test]
    fn overlap_log_boundaries_roundtrip() {
        let mut data = text(1 << 20);
        data.extend_from_slice(&noise(256 << 10, 5));
        data.extend_from_slice(&text(1 << 20));
        for level in [1, 3, 7, 11] {
            let default_log = if CParams::for_level(level, data.len()).strategy == Strategy::Lazy2 {
                7
            } else {
                6
            };
            let frame = |overlap_log| {
                compress_with(
                    &data,
                    &CompressOptions {
                        level,
                        job_size: Some(JOBSIZE_MIN),
                        overlap_log,
                        ..Default::default()
                    },
                )
            };
            assert!(frame(0) == frame(default_log), "level {level}");
            for overlap_log in [1, 2, 9] {
                let f = frame(overlap_log);
                assert_eq!(
                    crate::decompress(&f).unwrap(),
                    data,
                    "L{level} ov{overlap_log}"
                );
                let theirs = zstd::stream::decode_all(&f[..]).unwrap();
                assert_eq!(theirs, data, "L{level} ov{overlap_log}");
            }
        }
    }

    /// The parallel and the serial job loop must produce the same bytes.
    #[test]
    fn parallel_and_serial_job_loops_agree() {
        let mut data = text(3 << 20);
        data.extend_from_slice(&noise(1 << 20, 7));
        data.extend_from_slice(&text(1 << 20));
        for level in [1, 3, 7, 11] {
            let opts = CompressOptions {
                level,
                job_size: Some(512 << 10),
                ..Default::default()
            };
            let cparams = CParams::for_level(level, data.len());
            let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
            let overlap = overlap_size(&cparams, opts.overlap_log, false);
            let job_size = job_size_for(opts.job_size, &cparams, false, overlap);
            let jobs = job_ranges(data.len(), job_size);
            assert!(jobs.len() >= 5, "level {level}: {} jobs", jobs.len());
            let n = jobs.len();
            let src = data.as_slice();
            let f = |pipelined: bool| {
                move |k: usize, job: Range<usize>, ctx: &mut JobContext, out: &mut Vec<u8>| {
                    compress_job(
                        src,
                        cparams,
                        block_size,
                        overlap,
                        job,
                        k == 0,
                        k + 1 == n,
                        pipelined,
                        &mut ctx.ms,
                        &mut ctx.scratch,
                        &mut BlockLdm::Off,
                        out,
                    )
                }
            };
            // The same contexts serve both runs, so the serial run also
            // covers the reset of used contexts.
            let mut ctxs: Vec<JobContext> = (0..n).map(|_| JobContext::default()).collect();
            let mut par = Vec::new();
            run_jobs(&jobs, &mut ctxs, true, f(true), &mut par);
            let mut seq = Vec::new();
            run_jobs(&jobs, &mut ctxs, false, f(false), &mut seq);
            assert!(par == seq, "level {level}: job outputs differ");
            let frame = compress_with(&data, &opts);
            assert!(
                frame.ends_with(&seq),
                "level {level}: frame != header + jobs"
            );
            assert_eq!(crate::decompress(&frame).unwrap(), data);
        }
    }

    /// One job through `compress_job`, pipelined or serial.
    #[cfg(feature = "parallel")]
    fn one_job(data: &[u8], level: i32, pipelined: bool) -> Vec<u8> {
        let cparams = CParams::for_level(level, data.len());
        let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
        let overlap = overlap_size(&cparams, 0, false);
        let mut ctx = JobContext::default();
        let mut out = Vec::new();
        let job = 0..data.len();
        compress_job(
            data,
            cparams,
            block_size,
            overlap,
            job,
            true,
            true,
            pipelined,
            &mut ctx.ms,
            &mut ctx.scratch,
            &mut BlockLdm::Off,
            &mut out,
        );
        out
    }

    /// The pipelined block loop writes the serial loop's bytes: on source
    /// text (the proof holds and blocks overlap), on random data (every
    /// block RAW, the proof fails) and with a RAW and an RLE block between
    /// compressed ones (the next block must start from the committed, not
    /// the finder's, repeat offsets). Levels 16, 18 and 19 run the opt
    /// parsers, whose statistics carry across blocks in the match state.
    #[cfg(feature = "parallel")]
    #[test]
    fn pipelined_block_loop_matches_serial() {
        use std::sync::atomic::Ordering::Relaxed;
        let sources = super::common::testutil::crate_sources();
        let random = noise(1 << 20, 9);
        let mut mixed = text(300 << 10);
        mixed.extend_from_slice(&noise(200 << 10, 4));
        mixed.extend_from_slice(&vec![0u8; 300 << 10]);
        mixed.extend_from_slice(&text(333 << 10));
        for level in [1, 3, 5, 11, 16, 18, 19] {
            for (name, data) in [
                ("sources", &sources),
                ("random", &random),
                ("mixed", &mixed),
            ] {
                let before = block::PIPELINE_OVERLAPPED.load(Relaxed);
                let serial = one_job(data, level, false);
                let piped = one_job(data, level, true);
                assert!(piped == serial, "{name} L{level}: pipelined != serial");
                let overlapped = block::PIPELINE_OVERLAPPED.load(Relaxed) - before;
                if name != "random" {
                    assert!(overlapped > 0, "{name} L{level}: never overlapped");
                }
                let frame = compress_with(
                    data,
                    &CompressOptions {
                        level,
                        ..Default::default()
                    },
                );
                assert!(frame.ends_with(&serial), "{name} L{level}: frame != job");
                assert_eq!(&crate::decompress(&frame).unwrap(), data, "{name} L{level}");
            }
        }
    }

    /// A `Compressor` fed different inputs back to back, so that its tables
    /// grow, shrink and grow again and its job pool is reused, produces the
    /// frames fresh `compress_with` calls produce.
    #[test]
    fn reused_compressor_matches_fresh_compress_with() {
        let a = text(1280 << 10);
        let mut b = noise(384 << 10, 3);
        b.extend_from_slice(&text(384 << 10));
        let c = text(100 << 10);
        let empty = Vec::new();
        for level in [1, 3, 7, 11] {
            for job_size in [None, Some(512 << 10)] {
                let opts = CompressOptions {
                    level,
                    job_size,
                    overlap_log: 0,
                    ..Default::default()
                };
                let mut cx = Compressor::new(opts.clone());
                for input in [&a, &b, &c, &a, &empty] {
                    let reused = cx.compress_to_vec(input);
                    assert!(
                        reused == compress_with(input, &opts),
                        "level {level}, job_size {job_size:?}, {} bytes",
                        input.len()
                    );
                }
                let mut out = b"prefix".to_vec();
                cx.compress(&b, &mut out);
                assert!(out.starts_with(b"prefix"));
                assert!(out[6..] == compress_with(&b, &opts)[..]);
            }
        }
    }

    /// An input smaller than one job compresses identically with the default
    /// and with any explicit job size that covers it (300 KiB clamps up to
    /// JOBSIZE_MIN and still covers the input).
    #[test]
    fn default_job_size_equals_explicit_for_single_job_input() {
        let data = text(300 << 10);
        for level in [1, 3] {
            assert_eq!(
                job_size_for(None, &CParams::for_level(level, data.len()), false, 0),
                1 << 21
            );
            let auto = compress_with(
                &data,
                &CompressOptions {
                    level,
                    job_size: None,
                    overlap_log: 0,
                    ..Default::default()
                },
            );
            for js in [300 << 10, 512 << 10, 1 << 20] {
                let explicit = compress_with(
                    &data,
                    &CompressOptions {
                        level,
                        job_size: Some(js),
                        overlap_log: 0,
                        ..Default::default()
                    },
                );
                assert!(auto == explicit, "level {level} job_size {js}");
            }
            assert_eq!(crate::decompress(&auto).unwrap(), data);
        }
    }

    /// Two jobs of the same text: job 1 must not reuse job 0's Huffman or
    /// FSE tables nor its repeat offsets. libzstd rejects the stream if it did.
    #[test]
    fn second_job_starts_from_fresh_state() {
        let unit = text(JOBSIZE_MIN);
        let mut data = unit.clone();
        data.extend_from_slice(&unit);
        for level in [1, 3, 7, 11] {
            let frame = compress_with(
                &data,
                &CompressOptions {
                    level,
                    job_size: Some(JOBSIZE_MIN),
                    overlap_log: 0,
                    ..Default::default()
                },
            );
            assert_eq!(crate::decompress(&frame).unwrap(), data, "level {level}");
            let theirs = zstd::stream::decode_all(&frame[..]).unwrap();
            assert_eq!(theirs, data, "level {level}");
        }
    }

    /// Job 1 may reference the overlap it indexed from job 0: a copy of the
    /// bytes right before the job boundary compresses to almost nothing.
    #[test]
    fn second_job_matches_into_overlap_of_first_job() {
        let job = 512 << 10;
        let copy = 32 << 10;
        let mut data = noise(job, 11);
        let tail = data[job - copy..].to_vec();
        data.extend_from_slice(&tail);
        let opts = CompressOptions {
            level: 1,
            job_size: Some(job),
            overlap_log: 0,
            ..Default::default()
        };
        let cparams = CParams::for_level(1, data.len());
        assert!(overlap_size(&cparams, 0, false) >= copy);
        let frame = compress_with(&data, &opts);
        assert!(
            frame.len() < job + copy / 4,
            "no cross-job matches: {} bytes for {} input",
            frame.len(),
            data.len()
        );
        assert_eq!(crate::decompress(&frame).unwrap(), data);
        assert_eq!(zstd::stream::decode_all(&frame[..]).unwrap(), data);
    }

    #[test]
    fn frame_header_window_descriptor_for_large_input() {
        // 1 MiB at level 1: windowLog 19 < 20 bits of content -> not single segment
        let data = vec![1u8; 1 << 20];
        let c = compress(&data, 1);
        let descriptor = c[4];
        assert_eq!(descriptor & (1 << 5), 0, "Single_Segment must be clear");
        assert_eq!(c[5] >> 3, 19 - 10, "Window_Descriptor exponent");
        assert_eq!(crate::decompress(&c).unwrap(), data);
    }
}
