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
pub mod matchstate;
pub mod opt;
pub mod params;
pub mod presplit;
pub mod seqstore;
pub mod split;

use crate::constants::*;
use block::{
    write_raw_block, write_rle_block, BlockScratch, BlockSizing, BlockState, CommittedBlockState,
    ZSTD_BLOCKHEADERSIZE,
};
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
    ///
    /// An explicit size, or an input of several jobs, also selects
    /// ZSTDMT's block sizing for inputs above `JOBSIZE_MIN` (below it
    /// libzstd does not start ZSTDMT): each job is fed in 512 KiB chunks
    /// that no block crosses and that bound the pre-split blocks of
    /// `block_splitter_level`. A single job of the default size is sized
    /// like single-threaded `ZSTD_compress2`, one chunk.
    pub job_size: Option<usize>,
    /// `ZSTD_c_overlapLog`, `0..=9`: the history a job indexes from before
    /// its start, as a fraction of the window. `0` selects
    /// `ZSTDMT_overlapLog_default` (6 for `Fast`..`Lazy`, 7 for `Lazy2` and
    /// `BtLazy2`, 8 for `BtOpt` and `BtUltra`, 9 for `BtUltra2`), `1` means
    /// no overlap, and `n` in `2..=9` means `window >> (9 - n)`,
    /// so `9` is the full window. Values above 9 panic (libzstd rejects
    /// them with `parameter_outOfBound`).
    pub overlap_log: u8,
    /// `ZSTD_c_splitAfterSequences`: after the match finder, cut a block
    /// into several where separate entropy tables are estimated to pay for
    /// the extra block headers. `Auto` (the default) enables it for
    /// `strategy >= ZSTD_btopt` with `window_log >= 17`, as libzstd does.
    pub split_after_sequences: ParamSwitch,
    /// `ZSTD_c_blockSplitterLevel`, `0..=6`: before match finding, end a
    /// full 128 KiB block early where its byte statistics change
    /// (`ZSTD_splitBlock`). `0` (the default) picks the heuristic by
    /// strategy from libzstd's `splitLevels`: `Fast` compares the block's
    /// borders, the others compare 8 KiB chunks sampled every 43 bytes
    /// (`DFast`), 11 (`Greedy`, `Lazy`), 5 (`Lazy2`, `btlazy2`) or 1
    /// (`btopt` and above). `1` disables it; `2` selects the borders
    /// and `3..=6` chunks sampled every 43, 11, 5 and 1 bytes. A block is
    /// split only once its job has saved 3 bytes, so the first block of
    /// every job is whole. Values above 6 panic (libzstd rejects them
    /// with `parameter_outOfBound`).
    pub block_splitter_level: u8,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            level: ZSTD_CLEVEL_DEFAULT,
            job_size: None,
            overlap_log: 0,
            split_after_sequences: ParamSwitch::Auto,
            block_splitter_level: 0,
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
    /// so its frame and speed are unchanged. Reproduce with four runs of
    /// the following and the per-cell median of `rs MB/s`:
    ///
    /// ```text
    /// ZSTD_BENCH_ITERS=11 ZSTD_GRID_JOBS=2048K ZSTD_GRID_OVERLAPS=8 \
    ///   flock /tmp/claude-1000/zstd-mtbench.lock taskset -c 4,6,7,8,10,12,14,15 \
    ///   cargo test --release --offline --test mt_grid -- --ignored --nocapture
    /// ```
    pub fn parallel(level: i32) -> Self {
        Self {
            level,
            job_size: Some(2 << 20),
            overlap_log: 8,
            ..Self::default()
        }
    }
}

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
}

/// One job's reusable state: its match state once a job has run, and its
/// block buffers.
#[derive(Default)]
struct JobContext {
    ms: Option<MatchState>,
    scratch: BlockScratch,
}

impl Compressor {
    pub fn new(opts: CompressOptions) -> Self {
        Self {
            opts,
            jobs: Vec::new(),
        }
    }

    /// Append one frame holding `src` to `out`.
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) {
        assert!(
            src.len() < u32::MAX as usize,
            "inputs of 4 GiB or more are not supported (match indices are u32)"
        );
        let cparams = CParams::for_level(self.opts.level, src.len());
        out.reserve(src.len() + 64);
        let header_start = out.len();
        write_frame_header(out, src.len() as u64, cparams.window_log);
        let header_len = out.len() - header_start;

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

        let overlap = overlap_size(&cparams, self.opts.overlap_log);
        let job_size = job_size_for(self.opts.job_size, cparams.window_log, overlap);
        let split = split::block_splitter_enabled(self.opts.split_after_sequences, &cparams);
        let jobs = job_ranges(src.len(), job_size);
        let n_jobs = jobs.len();
        let mt = multithreaded(&self.opts, src.len(), n_jobs);
        let sizing = block_sizing(&self.opts, &cparams, mt, header_len);
        if self.jobs.len() < n_jobs {
            self.jobs.resize_with(n_jobs, JobContext::default);
        }
        run_jobs(
            &jobs,
            &mut self.jobs[..n_jobs],
            cfg!(feature = "parallel"),
            |k, job, ctx, out| {
                compress_job(
                    src,
                    cparams,
                    sizing,
                    overlap,
                    job,
                    k == 0,
                    k + 1 == n_jobs,
                    split,
                    cfg!(feature = "parallel"),
                    ctx,
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

/// Whether libzstd would compress through ZSTDMT: `ZSTD_compressStream2`
/// runs it only for inputs above `ZSTDMT_JOBSIZE_MIN`, and a job size (set,
/// or several default jobs) is a ZSTDMT parameter.
fn multithreaded(opts: &CompressOptions, len: usize, n_jobs: usize) -> bool {
    len > JOBSIZE_MIN && (opts.job_size.is_some() || n_jobs > 1)
}

/// [`BlockSizing`] for `opts`: `blockSizeMax` from the window, the
/// pre-splitter level by strategy, and ZSTDMT's `4 * ZSTD_BLOCKSIZE_MAX`
/// chunks when `mt`.
fn block_sizing(
    opts: &CompressOptions,
    cparams: &CParams,
    mt: bool,
    header_len: usize,
) -> BlockSizing {
    BlockSizing {
        // blockSizeMax = MIN(ZSTD_BLOCKSIZE_MAX, 1 << windowLog)
        block_size_max: ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log),
        split_level: presplit::split_level(opts.block_splitter_level, cparams.strategy),
        chunk_size: if mt {
            4 * ZSTD_BLOCKSIZE_MAX
        } else {
            usize::MAX
        },
        header_len,
    }
}

/// Upper bound on the blocks a job of `len` bytes produces: every block
/// RAW with its header.
fn job_bound(len: usize, block_size: usize) -> usize {
    len + ZSTD_BLOCKHEADERSIZE * len.div_ceil(block_size)
}

/// `ZSTDMT_compressionJob`: compress `data[job]` into a sequence of blocks
/// appended to `out`, on `ctx`'s state reset for this job. Job 0 starts
/// from `repStartValue` with `window_low = 1`; a later job indexes
/// `overlap` bytes before its start (`ZSTD_loadDictionaryContent` on the
/// raw-content prefix), starts with invalidated repeat offsets and no
/// entropy tables, so its first block cannot reference state the decoder
/// obtained from the previous job. `sizing` cuts the job into blocks;
/// `split` runs every block through the post-sequence splitter.
#[allow(clippy::too_many_arguments)]
fn compress_job(
    data: &[u8],
    cparams: CParams,
    sizing: BlockSizing,
    overlap: usize,
    job: Range<usize>,
    first_job: bool,
    last_job: bool,
    split: bool,
    pipelined: bool,
    ctx: &mut JobContext,
    out: &mut Vec<u8>,
) {
    let window_low = if first_job {
        1
    } else {
        job.start.saturating_sub(overlap).max(1)
    };
    let mut ms = match ctx.ms.take() {
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
    ctx.scratch.reserve(sizing.block_size_max);
    out.reserve(job_bound(job.len(), sizing.block_size_max));
    block::compress_blocks(
        &mut ms,
        data,
        job,
        sizing,
        first_job,
        last_job,
        split,
        &mut state,
        &mut ctx.scratch,
        out,
        pipelined,
    );
    ctx.ms = Some(ms);
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
/// `1 << ZSTDMT_computeTargetJobLog` (no long-distance matching), and at
/// least `overlap` ("job size must be >= overlap size").
pub fn job_size_for(requested: Option<usize>, window_log: u32, overlap: usize) -> usize {
    let section = match requested {
        Some(n) => n.clamp(JOBSIZE_MIN, JOBSIZE_MAX),
        None => 1usize << 20.max(window_log + 2).min(JOBLOG_MAX),
    };
    section.max(overlap)
}

/// Job boundaries: `[0, job_size)`, `[job_size, 2 * job_size)`, ... with the
/// last job truncated to `len`.
pub fn job_ranges(len: usize, job_size: usize) -> Vec<Range<usize>> {
    (0..len)
        .step_by(job_size)
        .map(|start| start..(start + job_size).min(len))
        .collect()
}

/// `ZSTDMT_computeOverlapSize` without long-distance matching: `overlap_log`
/// as `ZSTD_c_overlapLog` (see [`CompressOptions::overlap_log`]); the result
/// is `0` or `1 << (window_log - (9 - overlap_log))`.
pub fn overlap_size(cparams: &CParams, overlap_log: u8) -> usize {
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
            Strategy::Lazy2 | Strategy::BtLazy2 => 7,
            Strategy::Fast | Strategy::DFast | Strategy::Greedy | Strategy::Lazy => 6,
        },
        n => n as u32,
    };
    let overlap_rlog = 9 - overlap_log;
    if overlap_rlog >= 8 {
        0
    } else {
        1usize << (cparams.window_log - overlap_rlog)
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

    #[test]
    fn job_sizing() {
        // explicit: clamped to [JOBSIZE_MIN, JOBSIZE_MAX], otherwise used as is
        assert_eq!(job_size_for(Some(0), 19, 0), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(1), 19, 0), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(JOBSIZE_MIN + 1), 19, 0), JOBSIZE_MIN + 1);
        assert_eq!(job_size_for(Some(usize::MAX), 19, 0), JOBSIZE_MAX);
        // default: 1 << min(max(20, window_log + 2), 30)
        assert_eq!(job_size_for(None, 10, 0), 1 << 20);
        assert_eq!(job_size_for(None, 18, 0), 1 << 20);
        assert_eq!(job_size_for(None, 19, 0), 1 << 21);
        assert_eq!(job_size_for(None, 22, 0), 1 << 24);
        assert_eq!(job_size_for(None, 31, 0), 1 << 30);
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
        assert_eq!(overlap_size(&fast, 0), 1 << (fast.window_log - 3));
        let lazy2 = CParams::for_level(11, 8 << 20);
        assert_eq!(lazy2.strategy, Strategy::Lazy2);
        assert_eq!(overlap_size(&lazy2, 0), 1 << (lazy2.window_log - 2));
        // ZSTD_c_overlapLog: 1 = none, n = window >> (9 - n), 9 = window.
        for cp in [fast, lazy2] {
            assert_eq!(overlap_size(&cp, 1), 0);
            assert_eq!(overlap_size(&cp, 2), 1 << (cp.window_log - 7));
            assert_eq!(overlap_size(&cp, 6), 1 << (cp.window_log - 3));
            assert_eq!(overlap_size(&cp, 9), 1 << cp.window_log);
        }
        // A job is at least the overlap; JOBSIZE_MIN applies to explicit sizes.
        assert_eq!(job_size_for(Some(1), 19, 1 << 22), 1 << 22);
        assert_eq!(job_size_for(Some(1), 19, 1 << 18), JOBSIZE_MIN);
        assert_eq!(job_size_for(None, 23, 1 << 23), 1 << 25);
        assert_eq!(job_size_for(Some(JOBSIZE_MIN), 23, 1 << 23), 1 << 23);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn overlap_log_above_nine_panics() {
        overlap_size(&CParams::for_level(1, 1 << 20), 10);
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
                overlap_log: 0,
                ..Default::default()
            };
            let cparams = CParams::for_level(level, data.len());
            let overlap = overlap_size(&cparams, opts.overlap_log);
            let job_size = job_size_for(opts.job_size, cparams.window_log, overlap);
            let jobs = job_ranges(data.len(), job_size);
            assert!(jobs.len() >= 5, "level {level}: {} jobs", jobs.len());
            let n = jobs.len();
            let mt = multithreaded(&opts, data.len(), n);
            let sizing = block_sizing(&opts, &cparams, mt, header_len(&data, &cparams));
            let src = data.as_slice();
            let f = |pipelined: bool| {
                move |k: usize, job: Range<usize>, ctx: &mut JobContext, out: &mut Vec<u8>| {
                    compress_job(
                        src,
                        cparams,
                        sizing,
                        overlap,
                        job,
                        k == 0,
                        k + 1 == n,
                        false,
                        pipelined,
                        ctx,
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

    /// The frame header length of `data` with `cparams`.
    fn header_len(data: &[u8], cparams: &CParams) -> usize {
        let mut header = Vec::new();
        write_frame_header(&mut header, data.len() as u64, cparams.window_log);
        header.len()
    }

    /// `data` as job 0 and the last through `compress_job`, pipelined or
    /// serial, with the post-sequence splitter on or off, the pre-splitter
    /// by strategy, and single-threaded or (`mt`) ZSTDMT block sizing.
    #[cfg(feature = "parallel")]
    fn one_job(data: &[u8], level: i32, split: bool, pipelined: bool, mt: bool) -> Vec<u8> {
        let opts = CompressOptions {
            level,
            ..Default::default()
        };
        let cparams = CParams::for_level(level, data.len());
        let sizing = block_sizing(&opts, &cparams, mt, header_len(data, &cparams));
        let overlap = overlap_size(&cparams, 0);
        let mut ctx = JobContext::default();
        let mut out = Vec::new();
        let job = 0..data.len();
        compress_job(
            data, cparams, sizing, overlap, job, true, true, split, pipelined, &mut ctx, &mut out,
        );
        out
    }

    /// The pipelined block loop writes the serial loop's bytes: on source
    /// text (the proof holds and blocks overlap), on random data (every
    /// block RAW, the proof fails) and with a RAW and an RLE block between
    /// compressed ones (the next block must start from the committed, not
    /// the finder's, repeat offsets). The pre-splitter is on, so block
    /// N+1's size is fixed from the least savings of a COMPRESSED block N;
    /// `mixed` also runs in ZSTDMT's 512 KiB chunks, where job 0 owes the
    /// frame header from its second chunk on. Levels 16, 18 and 19 run the
    /// opt parsers, whose statistics carry across blocks in the match state,
    /// and the post-splitter default options enable for them.
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
                // Default options: the post-splitter runs from btopt on.
                let cparams = CParams::for_level(level, data.len());
                let split = split::block_splitter_enabled(ParamSwitch::Auto, &cparams);
                let before = block::PIPELINE_OVERLAPPED.load(Relaxed);
                let serial = one_job(data, level, split, false, false);
                let piped = one_job(data, level, split, true, false);
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
                if name == "mixed" {
                    let unsplit = data.len().div_ceil(ZSTD_BLOCKSIZE_MAX);
                    assert!(
                        count_blocks(&frame) > unsplit,
                        "{name} L{level}: no pre-split"
                    );
                }
            }
            let cparams = CParams::for_level(level, mixed.len());
            let split = split::block_splitter_enabled(ParamSwitch::Auto, &cparams);
            let serial = one_job(&mixed, level, split, false, true);
            let piped = one_job(&mixed, level, split, true, true);
            assert!(
                piped == serial,
                "mixed L{level}: ZSTDMT pipelined != serial"
            );
            let frame = compress_with(
                &mixed,
                &CompressOptions {
                    level,
                    job_size: Some(2 << 20),
                    ..Default::default()
                },
            );
            assert!(
                frame.ends_with(&serial),
                "mixed L{level}: ZSTDMT frame != job"
            );
            assert_eq!(crate::decompress(&frame).unwrap(), mixed, "mixed L{level}");
            let unsplit = mixed.len().div_ceil(ZSTD_BLOCKSIZE_MAX);
            assert!(
                count_blocks(&frame) > unsplit,
                "mixed L{level}: no pre-split"
            );
        }
    }

    /// Blocks in a frame, from the block headers.
    #[cfg(feature = "parallel")]
    fn count_blocks(frame: &[u8]) -> usize {
        let fcs = |d: u8| match d >> 6 {
            0 => (d >> 5) as usize & 1,
            1 => 2,
            2 => 4,
            _ => 8,
        };
        let single = (frame[4] >> 5) & 1 == 1;
        let mut pos = 5 + (!single) as usize + fcs(frame[4]);
        let mut n = 0;
        loop {
            let h = u32::from_le_bytes([frame[pos], frame[pos + 1], frame[pos + 2], 0]);
            let ty = (h >> 1) & 3;
            pos += 3 + if ty == 1 { 1 } else { (h >> 3) as usize };
            n += 1;
            if h & 1 == 1 {
                assert_eq!(pos, frame.len());
                return n;
            }
        }
    }

    /// With the post-sequence splitter on, the pipelined block loop still
    /// writes the serial loop's bytes and still overlaps split blocks; the
    /// frame splits blocks and both decoders accept it.
    #[cfg(feature = "parallel")]
    #[test]
    fn pipelined_split_blocks_match_serial() {
        use std::sync::atomic::Ordering::Relaxed;
        let sources = super::common::testutil::crate_sources();
        let random = noise(1 << 20, 9);
        let mut mixed = text(300 << 10);
        mixed.extend_from_slice(&noise(200 << 10, 4));
        mixed.extend_from_slice(&vec![0u8; 300 << 10]);
        mixed.extend_from_slice(&sources[..333 << 10]);
        for level in [3, 5, 7, 11] {
            for (name, data) in [
                ("sources", &sources),
                ("random", &random),
                ("mixed", &mixed),
            ] {
                let before = block::PIPELINE_OVERLAPPED.load(Relaxed);
                let serial = one_job(data, level, true, false, false);
                let piped = one_job(data, level, true, true, false);
                assert!(piped == serial, "{name} L{level}: pipelined != serial");
                let overlapped = block::PIPELINE_OVERLAPPED.load(Relaxed) - before;
                if name != "random" {
                    assert!(overlapped > 0, "{name} L{level}: never overlapped");
                }
                let opts = |split_after_sequences| CompressOptions {
                    level,
                    split_after_sequences,
                    ..Default::default()
                };
                let frame = compress_with(data, &opts(ParamSwitch::Enable));
                assert!(frame.ends_with(&serial), "{name} L{level}: frame != job");
                assert_eq!(&crate::decompress(&frame).unwrap(), data, "{name} L{level}");
                let theirs = zstd::stream::decode_all(&frame[..]).unwrap();
                assert_eq!(&theirs, data, "{name} L{level}");
                if name == "sources" {
                    // The post-sequence splitter against 128 KiB blocks. It
                    // can cost a few bytes, as libzstd's does, so only the
                    // split is asserted; tests/split_parity pins where.
                    let unsplit = |split_after_sequences| CompressOptions {
                        block_splitter_level: 1,
                        ..opts(split_after_sequences)
                    };
                    let split = compress_with(data, &unsplit(ParamSwitch::Enable));
                    let whole = compress_with(data, &unsplit(ParamSwitch::Disable));
                    assert!(whole == compress_with(data, &unsplit(ParamSwitch::Auto)));
                    assert!(
                        count_blocks(&split) > count_blocks(&whole),
                        "{name} L{level}: no block split"
                    );
                }
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
                job_size_for(None, CParams::for_level(level, data.len()).window_log, 0),
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
        assert!(overlap_size(&cparams, 0) >= copy);
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
