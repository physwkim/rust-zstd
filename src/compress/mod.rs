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
pub mod presplit;
pub mod seqstore;
pub mod split;

use crate::constants::*;
use block::{
    write_raw_block, BlockLdm, BlockScratch, BlockSizing, BlockState, CommittedBlockState,
    ZSTD_BLOCKHEADERSIZE,
};
use ldm::{LdmParams, LdmState, RawSeqStore, LDM_DEFAULT_WINDOW_LOG};
use matchstate::MatchState;
pub use params::{CParams, ParamSwitch, Strategy};
use params::{ZSTD_CLEVEL_DEFAULT, ZSTD_WINDOWLOG_ABSOLUTEMIN};
pub use seqstore::{Seq, SeqStore};
use std::ops::Range;
#[cfg(feature = "parallel")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Condvar, Mutex,
};

/// `ZSTDMT_JOBSIZE_MIN`: lower bound of an explicit job size.
pub const JOBSIZE_MIN: usize = 512 << 10;
/// `ZSTDMT_JOBSIZE_MAX`: upper bound of an explicit job size, 1 GiB, or
/// 512 MiB where `size_t` is 32 bits.
pub const JOBSIZE_MAX: usize = if MEM_32BITS { 512 << 20 } else { 1 << 30 };
/// `ZSTDMT_JOBLOG_MAX`: upper bound of the job size log that sizes the
/// overlap under long distance matching, 30, or 29 where `size_t` is 32
/// bits.
const JOBLOG_MAX: u32 = if MEM_32BITS { 29 } else { 30 };

/// `ZSTD_OVERLAPLOG_MAX`: upper bound of `ZSTD_c_overlapLog`.
const OVERLAPLOG_MAX: u8 = 9;

/// Options for [`Compressor`] and [`compress_with`].
#[derive(Clone, Debug)]
pub struct CompressOptions {
    /// Compression level, `ZSTD_c_compressionLevel`, as libzstd reads it:
    /// `1..=22` select its parameter rows (higher clamps to 22), `0` is the
    /// default level 3, and a negative level is the fast strategy
    /// accelerated by `-level` (clamped at `ZSTD_minCLevel`, -131072).
    pub level: i32,
    /// Job size in bytes (`ZSTD_c_jobSize`). `None` (the default) compresses
    /// the input as one job, as single-threaded `ZSTD_compress2`
    /// (`ZSTD_c_nbWorkers` 0) does. An explicit size selects ZSTDMT: the
    /// input is cut into jobs of that many bytes, clamped to
    /// `[JOBSIZE_MIN, JOBSIZE_MAX]` (512 KiB to 1 GiB, or to 512 MiB where
    /// `usize` is 32 bits) and raised to the
    /// overlap size (`ZSTDMT_initCStream_internal`); each job is compressed
    /// independently and, with the `parallel` feature, on its own rayon
    /// task. The frame never depends on the thread count, and is the same
    /// for a given job size whether or not the `parallel` feature is
    /// enabled. Smaller jobs give more parallelism and slightly worse
    /// ratios, since a job only sees `min(overlap, job start)` bytes of
    /// history from the previous job, see `overlap_log`.
    ///
    /// For inputs above `JOBSIZE_MIN` (below it libzstd does not start
    /// ZSTDMT) an explicit size also selects ZSTDMT's block sizing: each
    /// job is fed in 512 KiB chunks that no block crosses and that bound
    /// the pre-split blocks of `block_splitter_level`.
    pub job_size: Option<usize>,
    /// `ZSTD_c_overlapLog`, `0..=9`: the history a job indexes from before
    /// its start, as a fraction of the window. `0` selects
    /// `ZSTDMT_overlapLog_default` (6 for `Fast`..`Lazy`, 7 for `Lazy2` and
    /// `BtLazy2`, 8 for `BtOpt` and `BtUltra`, 9 for `BtUltra2`), `1` means
    /// no overlap, and `n` in `2..=9` means `window >> (9 - n)`,
    /// so `9` is the full window. Values above 9 are 9, as
    /// `ZSTD_CCtx_setParameter` clamps them. With long distance matching the
    /// fraction is of `min(window, 1 << (job_log - 2))` instead, with
    /// `job_log = min(max(21, cycleLog + 3), 30)` (29 where `usize` is 32
    /// bits), and `1` no longer means
    /// no overlap (`ZSTDMT_computeOverlapSize`).
    pub overlap_log: u8,
    /// `ZSTD_c_enableLongDistanceMatching`: find matches up to a window
    /// back with a rolling hash over the whole window (see [`ldm`]).
    /// `Auto` enables it for the `btopt` strategies and up with a window
    /// log of 27 or more (level 22 on inputs above 64 MiB). `Enable`
    /// raises the window log to 27 (`ZSTD_LDM_DEFAULT_WINDOW_LOG`) before
    /// the size adjustment. Without an explicit `job_size` each block's
    /// matches are generated as the block is compressed; with one, each
    /// job's are generated in job order before the job, and the job
    /// overlap changes (see `overlap_log`).
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
    /// explicit hash log, else `7 - strategy / 3`), else `1..=25`
    /// (`1..=24` where `usize` is 32 bits).
    ///
    /// Out-of-range LDM values panic, where `ZSTD_CCtx_setParameter`
    /// returns `parameter_outOfBound`.
    pub ldm_hash_rate_log: u32,
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
    /// Test knob, libzstd's `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY`:
    /// correct the match state's and the long distance matcher's windows
    /// whenever a correction keeps the whole window, not only before an
    /// index would pass `ZSTD_CURRENT_MAX` (3500 MiB, or 2000 MiB where
    /// `usize` is 32 bits), so that small inputs exercise the
    /// correction. The frames then equal those of a libzstd built with
    /// that macro set to 1.
    #[doc(hidden)]
    pub overflow_correct_frequently: bool,
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
            split_after_sequences: ParamSwitch::Auto,
            block_splitter_level: 0,
            overflow_correct_frequently: false,
        }
    }
}

impl CompressOptions {
    /// Opt-in multi-job preset: 2 MiB jobs (`ZSTD_c_jobSize`) with
    /// `ZSTD_c_overlapLog` 8 (half-window overlap). Inputs above 2 MiB split
    /// into several jobs, so the `parallel` feature can compress them
    /// concurrently; the frame still does not depend on the thread count.
    ///
    /// Measured on 8 threads against the default options, one job
    /// (`tests/mt_grid.rs`, median of four runs), on the 8 MiB ELF and
    /// Rust-source corpus files: the size cost (ELF / source) is 0.02% /
    /// 0.75% at level 1, -0.02% / 0.28% at 3, 0.04% / 0.05% at 5, 0.20% /
    /// 0.07% at 7 and 0.58% / 0.08% at 11, and the speedup is 3.14x /
    /// 2.68x at level 1, 2.18x / 2.17x at 3, 1.82x / 2.51x at 5, 1.86x /
    /// 2.27x at 7 and 1.81x / 1.64x at 11. An input of at most 2 MiB is one
    /// job under either option. Reproduce with four runs of the following
    /// and the per-cell median of `rs MB/s`:
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
            ParamSwitch::Auto => cparams.strategy >= Strategy::BtOpt && cparams.window_log >= 27,
            ParamSwitch::Enable => true,
            ParamSwitch::Disable => false,
        };
        (cparams, enabled.then(|| requested.adjusted(&cparams)))
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
/// demand and is kept across calls. Each job's match state is reset for
/// its input like `ZSTD_resetCCtx_internal`: indices continue from its
/// previous input and its tables are kept, see [`MatchState::reset`].
/// Every frame is identical to [`compress_with`]'s.
pub struct Compressor {
    opts: CompressOptions,
    jobs: Vec<JobContext>,
    /// ZSTDMT's long distance matching state (`serialState.ldmState`),
    /// once a multithreaded frame has used it.
    serial_ldm: Option<LdmState>,
}

/// One job's reusable state: its match state once a job has run, its
/// block buffers, the long distance matching state a single-threaded frame
/// generates each block's matches from (the context's `ldmState`), and the
/// long distance matches ZSTDMT generated for the job (`rawSeqStore`).
#[derive(Default)]
struct JobContext {
    ms: Option<MatchState>,
    scratch: BlockScratch,
    ldm_state: Option<LdmState>,
    ldm_seqs: RawSeqStore,
}

impl Compressor {
    pub fn new(opts: CompressOptions) -> Self {
        Self {
            opts,
            jobs: Vec::new(),
            serial_ldm: None,
        }
    }

    /// Test hook for `CompressOptions::overflow_correct_frequently`: the
    /// window overflow corrections of the match states since each last
    /// restarted its indices, and of the long distance matchers since each
    /// was last reset; for a fresh `Compressor`, those of its one frame.
    #[doc(hidden)]
    pub fn overflow_corrections(&self) -> (u32, u32) {
        let ms = self.jobs.iter().filter_map(|ctx| ctx.ms.as_ref());
        let ldm = self.jobs.iter().filter_map(|ctx| ctx.ldm_state.as_ref());
        let ldm = ldm.chain(self.serial_ldm.as_ref());
        (
            ms.map(|ms| ms.window().nb_overflow_corrections()).sum(),
            ldm.map(|ldm| ldm.window().nb_overflow_corrections()).sum(),
        )
    }

    /// Append one frame holding `src` to `out`.
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) {
        let (cparams, ldm_params) = self.opts.frame_params(src.len());
        out.reserve(src.len() + 64);
        let header_start = out.len();
        write_frame_header(out, src.len() as u64, cparams.window_log);
        let header_len = out.len() - header_start;

        if src.is_empty() {
            write_raw_block(out, &[], true);
            return;
        }

        let ldm_on = ldm_params.is_some();
        let overlap = overlap_size(&cparams, self.opts.overlap_log, ldm_on);
        let job_size = job_size_for(self.opts.job_size, overlap);
        let split = split::block_splitter_enabled(self.opts.split_after_sequences, &cparams);
        let jobs = job_ranges(src.len(), job_size);
        let n_jobs = jobs.len();
        let mt = multithreaded(&self.opts, src.len());
        let sizing = block_sizing(&self.opts, &cparams, mt, header_len);
        if self.jobs.len() < n_jobs {
            self.jobs.resize_with(n_jobs, JobContext::default);
        }
        let pipelined = cfg!(feature = "parallel");
        // ZSTDMT_serialState: every job's long distance matches from the one
        // state, in job order, at most ZSTD_ldm_getMaxNbSeq of them. A
        // single-threaded frame generates each block's as it compresses the
        // block (ZSTD_buildSeqStore), from its context's state.
        let frequently = self.opts.overflow_correct_frequently;
        let mut serial_ldm = ldm_params
            .filter(|_| mt)
            .map(|params| reset_ldm_state(&mut self.serial_ldm, params, frequently));
        let max_seqs = ldm_params.map_or(0, |p| job_size / p.min_match_length as usize);
        run_jobs(
            &jobs,
            &mut self.jobs[..n_jobs],
            pipelined,
            // ZSTDMT_serialState_update
            |job, ctx| {
                if let Some(state) = &mut serial_ldm {
                    state.generate_sequences(src, job.clone(), max_seqs, &mut ctx.ldm_seqs);
                }
            },
            |k, job, ctx, out| {
                let JobContext {
                    ms,
                    scratch,
                    ldm_state,
                    ldm_seqs,
                } = ctx;
                let mut ldm = match ldm_params {
                    None => BlockLdm::Off,
                    Some(_) if mt => BlockLdm::External(ldm_seqs),
                    Some(params) => {
                        BlockLdm::Internal(reset_ldm_state(ldm_state, params, frequently))
                    }
                };
                compress_job(
                    src,
                    cparams,
                    frequently,
                    sizing,
                    overlap,
                    job,
                    k == 0,
                    k + 1 == n_jobs,
                    split,
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

/// The long distance matching state in `slot` reset for a frame with
/// `params`, allocated on first use, with the overflow correction knob
/// `frequently` (see [`CompressOptions`]).
fn reset_ldm_state(
    slot: &mut Option<LdmState>,
    params: LdmParams,
    frequently: bool,
) -> &mut LdmState {
    let state = match slot.take() {
        Some(mut state) => {
            state.reset(params, 0);
            slot.insert(state)
        }
        None => slot.insert(LdmState::new(params, 0)),
    };
    state.set_correct_frequently(frequently);
    state
}

/// Whether libzstd would compress through ZSTDMT: an explicit job size is
/// a ZSTDMT parameter, and `ZSTD_compressStream2` runs ZSTDMT only for
/// inputs above `ZSTDMT_JOBSIZE_MIN`.
fn multithreaded(opts: &CompressOptions, len: usize) -> bool {
    opts.job_size.is_some() && len > JOBSIZE_MIN
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
/// appended to `out`, on the match state in `ms_slot` reset for this job
/// (allocated on first use) with `scratch`'s buffers and the long distance
/// matches of `ldm`. Job 0 starts from `repStartValue` with its first byte
/// as the window start; a later job's window starts `overlap` bytes before
/// it, and the job indexes that prefix (`ZSTD_loadDictionaryContent` on the
/// raw-content prefix), starts with invalidated repeat offsets and no
/// entropy tables, so its first block cannot reference state the decoder
/// obtained from the previous job. `sizing` cuts the job into blocks;
/// `split` runs every block through the post-sequence splitter;
/// `frequently` is the overflow correction knob (see [`CompressOptions`]).
#[allow(clippy::too_many_arguments)]
fn compress_job(
    data: &[u8],
    cparams: CParams,
    frequently: bool,
    sizing: BlockSizing,
    overlap: usize,
    job: Range<usize>,
    first_job: bool,
    last_job: bool,
    split: bool,
    pipelined: bool,
    ms_slot: &mut Option<MatchState>,
    scratch: &mut BlockScratch,
    ldm: &mut BlockLdm,
    out: &mut Vec<u8>,
) {
    // ZSTDMT: a job's window starts at its prefix (ZSTD_dct_rawContent).
    let origin = if first_job {
        job.start
    } else {
        job.start.saturating_sub(overlap)
    };
    let mut ms = match ms_slot.take() {
        Some(mut ms) => {
            ms.reset(cparams, origin);
            ms
        }
        None => MatchState::new(cparams, origin),
    };
    ms.set_correct_frequently(frequently);
    let mut initial = BlockState::initial();
    if !first_job {
        block::load_prefix(&mut ms, data, origin..job.start);
        initial.invalidate_rep_codes();
    }
    let mut state = CommittedBlockState::new(initial);
    scratch.reserve(sizing.block_size_max);
    out.reserve(job_bound(job.len(), sizing.block_size_max));
    block::compress_blocks(
        &mut ms, data, job, sizing, first_job, last_job, split, &mut state, scratch, ldm, out,
        pipelined,
    );
    *ms_slot = Some(ms);
}

/// Run `f` over every job, in job order, appending to `out`; `ctxs[k]` is
/// job `k`'s context. `prepare` runs on each job's context, in job order and
/// one at a time, before the job starts: with the feature enabled and
/// `parallel`, on the calling task while the earlier jobs run on rayon, as
/// ZSTDMT serializes its long distance matching across jobs. Job 0 writes
/// straight into `out`, the others into buffers of their own that are
/// appended afterwards; the serial loop hands every job `out`. The job
/// function is the same either way, so the frame is identical.
///
/// Jobs are never rayon tasks: every worker gets one [`JobQueue`] runner
/// (`spawn_broadcast`), which claims jobs in order until none is left. A
/// thread waiting in a job's block `rayon::join` runs whatever rayon hands
/// it; were jobs tasks, it could take a queued job and finish its own a
/// whole job late. It can still take its own runner, or another frame's,
/// which is why a runner that starts inside a job claims nothing.
fn run_jobs<P, F>(
    jobs: &[Range<usize>],
    ctxs: &mut [JobContext],
    parallel: bool,
    mut prepare: P,
    f: F,
    out: &mut Vec<u8>,
) where
    P: FnMut(&Range<usize>, &mut JobContext) + Send,
    F: Fn(usize, Range<usize>, &mut JobContext, &mut Vec<u8>) + Sync,
{
    debug_assert_eq!(jobs.len(), ctxs.len());
    #[cfg(feature = "parallel")]
    if parallel {
        let mut rest_out = vec![Vec::new(); jobs.len().saturating_sub(1)];
        {
            let outs = std::iter::once(&mut *out).chain(&mut rest_out);
            let queue = JobQueue::new(jobs, ctxs.iter_mut().zip(outs));
            let (queue, f) = (&queue, &f);
            rayon::scope(|s| {
                if jobs.len() > 1 {
                    s.spawn_broadcast(move |_, _| {
                        if !IN_JOB.get() {
                            queue.run(f);
                        }
                    });
                }
                queue.prepare_all(&mut prepare);
                // The calling task runs jobs whether or not it is inside one:
                // the frame is its to finish.
                queue.run(f);
            });
        }
        for o in &rest_out {
            out.extend_from_slice(o);
        }
        return;
    }
    let _ = parallel;
    for (k, (job, ctx)) in jobs.iter().zip(ctxs.iter_mut()).enumerate() {
        prepare(job, ctx);
        f(k, job.clone(), ctx, out);
    }
}

#[cfg(feature = "parallel")]
thread_local! {
    /// Whether this thread is running one of [`JobQueue::run`]'s jobs.
    static IN_JOB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks this thread as inside a job until dropped.
#[cfg(feature = "parallel")]
struct InJob(bool);

#[cfg(feature = "parallel")]
impl InJob {
    fn enter() -> Self {
        InJob(IN_JOB.replace(true))
    }
}

#[cfg(feature = "parallel")]
impl Drop for InJob {
    fn drop(&mut self) {
        IN_JOB.set(self.0);
    }
}

/// A job's context and output.
#[cfg(feature = "parallel")]
type JobSlot<'a> = (&'a mut JobContext, &'a mut Vec<u8>);

/// One frame's jobs for [`run_jobs`]' runners, claimed in job order, each
/// once its `prepare` has run.
#[cfg(feature = "parallel")]
struct JobQueue<'a> {
    jobs: &'a [Range<usize>],
    /// Job `k`'s slot, until its runner takes it.
    slots: Vec<Mutex<Option<JobSlot<'a>>>>,
    /// The next job to claim.
    next: AtomicUsize,
    /// How many jobs, in job order, have been prepared; `None` once
    /// preparing unwound, so that no runner waits for the rest.
    prepared: Mutex<Option<usize>>,
    prepared_cv: Condvar,
}

#[cfg(feature = "parallel")]
impl<'a> JobQueue<'a> {
    fn new(jobs: &'a [Range<usize>], slots: impl Iterator<Item = JobSlot<'a>>) -> Self {
        Self {
            jobs,
            slots: slots.map(|s| Mutex::new(Some(s))).collect(),
            next: AtomicUsize::new(0),
            prepared: Mutex::new(Some(0)),
            prepared_cv: Condvar::new(),
        }
    }

    fn publish(&self, prepared: Option<usize>) {
        *self.prepared.lock().unwrap() = prepared;
        self.prepared_cv.notify_all();
    }

    /// `prepare` every job, in job order, releasing each to the runners.
    fn prepare_all<P>(&self, prepare: &mut P)
    where
        P: FnMut(&Range<usize>, &mut JobContext),
    {
        struct Unwinding<'q, 'a>(&'q JobQueue<'a>);
        impl Drop for Unwinding<'_, '_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.publish(None);
                }
            }
        }
        let _unwinding = Unwinding(self);
        for (k, (job, slot)) in self.jobs.iter().zip(&self.slots).enumerate() {
            if let Some((ctx, _)) = slot.lock().unwrap().as_mut() {
                prepare(job, ctx);
            }
            self.publish(Some(k + 1));
        }
    }

    /// Claim the next job, wait until it is prepared and run it, until no
    /// job is left.
    fn run<F>(&self, f: &F)
    where
        F: Fn(usize, Range<usize>, &mut JobContext, &mut Vec<u8>),
    {
        loop {
            let k = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(job) = self.jobs.get(k) else {
                return;
            };
            let mut prepared = self.prepared.lock().unwrap();
            loop {
                match *prepared {
                    None => return,
                    Some(n) if n > k => break,
                    Some(_) => prepared = self.prepared_cv.wait(prepared).unwrap(),
                }
            }
            drop(prepared);
            let (ctx, out) = self.slots[k]
                .lock()
                .unwrap()
                .take()
                .expect("job claimed twice");
            let _in_job = InJob::enter();
            f(k, job.clone(), ctx, out);
        }
    }
}

/// The job size of `requested`: `None` is single-threaded, one job of
/// unbounded size; an explicit size is `ZSTDMT_initCStream_internal`'s
/// `targetSectionSize`, clamped to `[ZSTDMT_JOBSIZE_MIN,
/// ZSTDMT_JOBSIZE_MAX]` and at least `overlap` ("job size must be >=
/// overlap size").
pub fn job_size_for(requested: Option<usize>, overlap: usize) -> usize {
    match requested {
        Some(n) => n.clamp(JOBSIZE_MIN, JOBSIZE_MAX).max(overlap),
        None => usize::MAX,
    }
}

/// Job boundaries: `[0, job_size)`, `[job_size, 2 * job_size)`, ... with the
/// last job truncated to `len`.
pub fn job_ranges(len: usize, job_size: usize) -> Vec<Range<usize>> {
    (0..len)
        .step_by(job_size)
        .map(|start| start..start + job_size.min(len - start))
        .collect()
}

/// `ZSTDMT_computeOverlapSize`: `overlap_log` as `ZSTD_c_overlapLog` (see
/// [`CompressOptions::overlap_log`]); without long distance matching the
/// result is `0` or `1 << (window_log - (9 - overlap_log))`, with it
/// `1 << (min(window_log, job_log - 2) - (9 - overlap_log))`.
pub fn overlap_size(cparams: &CParams, overlap_log: u8, ldm: bool) -> usize {
    // ZSTDMT_overlapLog, of the value ZSTD_cParam_clampBounds leaves
    let overlap_log = match overlap_log.min(OVERLAPLOG_MAX) {
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
    let ov_log = if ldm {
        // In Long Range Mode, the windowLog is typically oversized: ovLog
        // becomes a fraction of the jobSize, rather than windowSize.
        // ZSTDMT_computeTargetJobLog, from ZSTD_cycleLog(chainLog, strategy):
        let cycle_log = cparams.chain_log - cparams.strategy.bt_scale();
        let job_log = 21.max(cycle_log + 3).min(JOBLOG_MAX);
        cparams.window_log.min(job_log - 2) - overlap_rlog
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
    use crate::compress::matchstate::WINDOW_START_INDEX;

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

    /// An explicit job size clamps at `ZSTD_c_jobSize`'s upper bound,
    /// `ZSTDMT_JOBSIZE_MAX` for this target, and `ZSTDMT_JOBLOG_MAX` is its
    /// log.
    #[test]
    fn job_bounds_match_libzstd() {
        use crate::compress::common::testutil::c_bounds;
        use zstd::zstd_safe::zstd_sys::ZSTD_cParameter::ZSTD_c_jobSize;

        let max = c_bounds(ZSTD_c_jobSize).1 as usize;
        for (requested, clamped) in [
            (max - 1, max - 1),
            (max, max),
            (max + 1, max),
            (usize::MAX, max),
        ] {
            assert_eq!(job_size_for(Some(requested), 0), clamped, "{requested}");
        }
        assert_eq!(1 << JOBLOG_MAX, max);
    }

    #[test]
    fn job_sizing() {
        // explicit: clamped to [JOBSIZE_MIN, JOBSIZE_MAX], otherwise used as is
        assert_eq!(job_size_for(Some(0), 0), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(1), 0), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(JOBSIZE_MIN + 1), 0), JOBSIZE_MIN + 1);
        assert_eq!(job_size_for(Some(usize::MAX), 0), JOBSIZE_MAX);
        // default: one job, whatever the overlap
        assert_eq!(job_size_for(None, 0), usize::MAX);
        assert_eq!(job_size_for(None, 1 << 23), usize::MAX);
        assert_eq!(job_ranges(3 << 30, usize::MAX), vec![0..3 << 30]);
        assert_eq!(job_ranges(0, usize::MAX), Vec::<Range<usize>>::new());
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
        assert_eq!(job_size_for(Some(1), 1 << 22), 1 << 22);
        assert_eq!(job_size_for(Some(1), 1 << 18), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(JOBSIZE_MIN), 1 << 23), 1 << 23);
    }

    /// With long distance matching the overlap of explicit jobs is
    /// `1 << (min(window_log, job_log - 2) - (9 - overlap_log))`, nonzero
    /// even for overlap_log 1, with `job_log = min(max(21, cycleLog + 3),
    /// JOBLOG_MAX)` whatever the window (`ZSTDMT_computeOverlapSize`,
    /// `ZSTDMT_computeTargetJobLog`).
    #[test]
    fn overlap_with_ldm() {
        // Job log 21: the overlap is a fraction of 1 << 19, not of the
        // window.
        let cp = logs(27, 13);
        assert_eq!(overlap_size(&cp, 9, true), 1 << 19);
        assert_eq!(overlap_size(&cp, 6, true), 1 << 16);
        assert_eq!(overlap_size(&cp, 0, true), 1 << 16);
        assert_eq!(overlap_size(&cp, 1, true), 1 << 11);
        // cycleLog 18 + 3 is still 21; then 22, 27, and 31 capped to
        // JOBLOG_MAX.
        assert_eq!(overlap_size(&logs(27, 18), 9, true), 1 << 19);
        assert_eq!(overlap_size(&logs(27, 19), 9, true), 1 << 20);
        assert_eq!(overlap_size(&logs(27, 24), 9, true), 1 << 25);
        assert_eq!(
            overlap_size(&logs(ZSTD_WINDOWLOG_MAX, 28), 9, true),
            1 << (JOBLOG_MAX - 2)
        );
        // A window below job_log - 2 is the base.
        let small = logs(17, 13);
        assert_eq!(overlap_size(&small, 9, true), 1 << 17);
        assert_eq!(overlap_size(&small, 2, true), 1 << 10);
        // Job log 27 (chain log 24), Lazy2's default overlap log 7.
        let mut lazy2 = logs(27, 24);
        lazy2.strategy = Strategy::Lazy2;
        assert_eq!(overlap_size(&lazy2, 0, true), 1 << 23);
        // The binary-tree strategies halve the cycle: level 22's chain log
        // 27 gives job log 26 + 3, and btultra2's overlap log 9 the whole
        // job log - 2 window.
        let l22 = CParams::for_level(22, 1 << 30);
        assert_eq!(overlap_size(&l22, 0, true), 1 << 27);
        // An explicit job is at least that overlap.
        assert_eq!(
            job_size_for(Some(1), overlap_size(&logs(27, 24), 9, true)),
            1 << 25
        );
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

    /// One `Compressor` for `opts` writes, for each of `inputs` in turn,
    /// what a fresh one does, and the frames decode through both decoders.
    fn check_ldm_frames(opts: &CompressOptions, inputs: &[&Vec<u8>]) {
        let mut cx = Compressor::new(opts.clone());
        for &data in inputs {
            let frame = cx.compress_to_vec(data);
            let name = format!(
                "L{} job {:?} {} bytes",
                opts.level,
                opts.job_size,
                data.len()
            );
            assert!(frame == compress_with(data, opts), "{name}: reuse");
            assert_eq!(&crate::decompress(&frame).unwrap(), data, "{name}");
            let theirs = zstd::stream::decode_all(&frame[..]).unwrap();
            assert_eq!(&theirs, data, "{name}");
        }
    }

    /// Long distance matching frames decode through both decoders on the
    /// single-threaded path (the default, and explicit jobs up to
    /// `JOBSIZE_MIN`) and over several jobs (the last block 3 bytes, too
    /// small to compress), find the repeat that the frame without it stores
    /// as literals, and a reused `Compressor` writes what fresh ones do.
    #[test]
    fn ldm_frames_roundtrip_and_find_far_repeats() {
        let single = far_repeat(160 << 10, 160 << 10);
        let jobs = far_repeat(1 << 20, (4 << 20) + 3);
        let empty = Vec::new();
        for level in [1, 3, 7, 11] {
            for job_size in [None, Some(JOBSIZE_MIN)] {
                let opts = ldm_opts(level, ParamSwitch::Enable, job_size);
                check_ldm_frames(&opts, &[&jobs, &single, &empty, &jobs]);
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

    /// From btopt on the long distance matches are candidates of the
    /// optimal parser: the frames decode through both decoders on the
    /// single-threaded path and over explicit jobs, whose overlap with long
    /// distance matching is half the window at level 16 (two jobs here) and
    /// all of it at level 19 (one job, still fed from the job-order
    /// sequences).
    #[test]
    fn ldm_opt_frames_roundtrip() {
        let single = far_repeat(160 << 10, 160 << 10);
        let jobs = far_repeat(256 << 10, 1 << 20);
        for level in [16, 19] {
            for job_size in [None, Some(JOBSIZE_MIN)] {
                let opts = ldm_opts(level, ParamSwitch::Enable, job_size);
                check_ldm_frames(&opts, &[&jobs, &single, &jobs]);
            }
        }
    }

    /// With long distance matching over several jobs, the parallel job loop
    /// (generating each job's sequences while the earlier jobs run) and the
    /// serial one write the same bytes, and those are the frame's.
    #[test]
    fn ldm_parallel_and_serial_job_loops_agree() {
        let big = far_repeat(1 << 20, 4 << 20);
        let small = far_repeat(256 << 10, 1 << 20);
        for (level, data, min_jobs) in [
            (1, &big, 6),
            (3, &big, 6),
            (7, &big, 6),
            (11, &big, 6),
            (16, &small, 2),
            (18, &small, 2),
        ] {
            let src = data.as_slice();
            let opts = ldm_opts(level, ParamSwitch::Enable, Some(JOBSIZE_MIN));
            let (cparams, ldm) = opts.frame_params(src.len());
            let ldm = ldm.expect("enabled");
            let overlap = overlap_size(&cparams, opts.overlap_log, true);
            let job_size = job_size_for(opts.job_size, overlap);
            let mt = multithreaded(&opts, src.len());
            let sizing = block_sizing(&opts, &cparams, mt, header_len(src, &cparams));
            let split = split::block_splitter_enabled(opts.split_after_sequences, &cparams);
            let jobs = job_ranges(src.len(), job_size);
            let n = jobs.len();
            assert!(n >= min_jobs, "L{level}: {n} jobs");
            let max_seqs = job_size / ldm.min_match_length as usize;
            let generate = || {
                let mut state = LdmState::new(ldm, 0);
                move |job: &Range<usize>, ctx: &mut JobContext| {
                    state.generate_sequences(src, job.clone(), max_seqs, &mut ctx.ldm_seqs);
                }
            };
            let f = |pipelined: bool| {
                move |k: usize, job: Range<usize>, ctx: &mut JobContext, out: &mut Vec<u8>| {
                    let JobContext {
                        ms,
                        scratch,
                        ldm_seqs,
                        ..
                    } = ctx;
                    compress_job(
                        src,
                        cparams,
                        false,
                        sizing,
                        overlap,
                        job,
                        k == 0,
                        k + 1 == n,
                        split,
                        pipelined,
                        ms,
                        scratch,
                        &mut BlockLdm::External(ldm_seqs),
                        out,
                    )
                }
            };
            let mut ctxs: Vec<JobContext> = (0..n).map(|_| JobContext::default()).collect();
            let mut par = Vec::new();
            run_jobs(&jobs, &mut ctxs, true, generate(), f(true), &mut par);
            let mut seq = Vec::new();
            run_jobs(&jobs, &mut ctxs, false, generate(), f(false), &mut seq);
            assert!(par == seq, "L{level}: job outputs differ");
            assert!(compress_with(src, &opts).ends_with(&seq), "L{level}");
        }
    }

    /// `Enable` selects window log 27 before the size adjustment; `Auto`
    /// turns on for btopt and above at window log 27 (`ZSTD_resolveEnableLdm`),
    /// which only level 22 reaches, above 64 MiB, and otherwise leaves the
    /// parameters and frames `Disable`'s.
    #[test]
    fn ldm_switch_resolution() {
        let data = far_repeat(300 << 10, 400 << 10);
        for level in 1..=22 {
            let (cp, ldm) = ldm_opts(level, ParamSwitch::Enable, None).frame_params(3 << 20);
            assert_eq!(cp.window_log, 22, "L{level}: 3 MiB adjusts 27 to 22");
            assert!(ldm.is_some_and(|p| p.window_log == 22));
            let big = ldm_opts(level, ParamSwitch::Enable, None).frame_params(1 << 30);
            assert_eq!(big.0.window_log, 27, "L{level}");
            for size in [3 << 20, 64 << 20, (64 << 20) + 1, 1 << 30] {
                let auto = ldm_opts(level, ParamSwitch::Auto, None).frame_params(size);
                if level == 22 && size > 64 << 20 {
                    let on = ldm_opts(level, ParamSwitch::Enable, None).frame_params(size);
                    assert_eq!(auto, on, "L{level} {size}");
                    assert!(auto.1.is_some(), "L{level} {size}");
                } else {
                    assert!(auto.1.is_none(), "L{level} {size}");
                    assert_eq!(auto.0, CParams::for_level(level, size), "L{level} {size}");
                    let off = ldm_opts(level, ParamSwitch::Disable, None).frame_params(size);
                    assert_eq!(off, auto, "L{level} {size}");
                }
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

    /// `overlap_size` reads an overlap log as `ZSTD_CCtx_setParameter`
    /// stores it, clamped to `ZSTD_c_overlapLog`'s bounds.
    #[test]
    fn overlap_log_bounds_match_libzstd() {
        use crate::compress::common::testutil::{c_accepts, c_bounds};
        use zstd::zstd_safe::zstd_sys::ZSTD_cParameter::ZSTD_c_overlapLog;

        let (lo, hi) = c_bounds(ZSTD_c_overlapLog);
        assert_eq!(hi, OVERLAPLOG_MAX as i32);
        assert!(c_accepts(ZSTD_c_overlapLog, hi + 1));
        for cp in [
            CParams::for_level(1, 1 << 20),
            CParams::for_level(19, 1 << 20),
        ] {
            for ldm in [false, true] {
                for v in 0..=u8::MAX {
                    let clamped = (v as i32).clamp(lo, hi) as u8;
                    assert_eq!(
                        overlap_size(&cp, v, ldm),
                        overlap_size(&cp, clamped, ldm),
                        "{v} ldm {ldm}"
                    );
                }
            }
        }
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
            let overlap = overlap_size(&cparams, opts.overlap_log, false);
            let job_size = job_size_for(opts.job_size, overlap);
            let jobs = job_ranges(data.len(), job_size);
            assert!(jobs.len() >= 5, "level {level}: {} jobs", jobs.len());
            let n = jobs.len();
            let mt = multithreaded(&opts, data.len());
            let sizing = block_sizing(&opts, &cparams, mt, header_len(&data, &cparams));
            let src = data.as_slice();
            let f = |pipelined: bool| {
                move |k: usize, job: Range<usize>, ctx: &mut JobContext, out: &mut Vec<u8>| {
                    compress_job(
                        src,
                        cparams,
                        false,
                        sizing,
                        overlap,
                        job,
                        k == 0,
                        k + 1 == n,
                        false,
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
            run_jobs(&jobs, &mut ctxs, true, |_, _| {}, f(true), &mut par);
            let mut seq = Vec::new();
            run_jobs(&jobs, &mut ctxs, false, |_, _| {}, f(false), &mut seq);
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
        let overlap = overlap_size(&cparams, 0, false);
        let mut ctx = JobContext::default();
        let mut out = Vec::new();
        let job = 0..data.len();
        compress_job(
            data,
            cparams,
            false,
            sizing,
            overlap,
            job,
            true,
            true,
            split,
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

    /// A thread waiting in a job's block join runs no other job meanwhile:
    /// with more jobs than threads and each join's second half slower than
    /// its first, so that a thread waits whenever a half is stolen, no job
    /// runs within another's run on the same thread.
    #[cfg(feature = "parallel")]
    #[test]
    fn no_job_runs_inside_another() {
        use std::time::{Duration, Instant};
        let spin = |d: Duration| {
            let t = Instant::now();
            while t.elapsed() < d {
                std::hint::spin_loop();
            }
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let jobs: Vec<Range<usize>> = (0..24).map(|k| k..k + 1).collect();
        let mut ctxs: Vec<JobContext> = jobs.iter().map(|_| JobContext::default()).collect();
        let runs = Mutex::new(Vec::new());
        let t0 = Instant::now();
        pool.install(|| {
            let job = |k: usize, _: Range<usize>, _: &mut JobContext, _: &mut Vec<u8>| {
                let start = t0.elapsed();
                for _ in 0..4 {
                    rayon::join(
                        || spin(Duration::from_micros(50)),
                        || spin(Duration::from_micros(500)),
                    );
                }
                let run = (k, rayon::current_thread_index(), start, t0.elapsed());
                runs.lock().unwrap().push(run);
            };
            run_jobs(&jobs, &mut ctxs, true, |_, _| {}, job, &mut Vec::new());
        });
        let runs = runs.into_inner().unwrap();
        assert_eq!(runs.len(), jobs.len());
        for (k, thread, start, end) in &runs {
            for (inner, t, s, e) in &runs {
                assert!(
                    !(t == thread && s > start && e < end),
                    "job {inner} ran inside job {k} on thread {thread:?}"
                );
            }
        }
    }

    /// A `Compressor` fed different inputs back to back, so that its tables
    /// shrink and grow again within their allocations and its job pool is
    /// reused, continues every context's indices from its previous input
    /// and still produces the frames fresh `compress_with` calls produce,
    /// for every strategy.
    #[test]
    fn reused_compressor_matches_fresh_compress_with() {
        let a = text(1280 << 10);
        let mut b = noise(384 << 10, 3);
        b.extend_from_slice(&text(384 << 10));
        let c = text(100 << 10);
        let empty = Vec::new();
        // The bt levels on smaller inputs: this runs unoptimized.
        let (a_bt, b_bt, c_bt) = (&a[..600 << 10], &b[320 << 10..], &c[..40 << 10]);
        let cases = [-5, 1, 2, 3, 5, 7, 11]
            .map(|level| (level, [&a[..], &b, &c, &a, &empty]))
            .into_iter()
            .chain([13, 16, 19].map(|level| (level, [a_bt, b_bt, c_bt, a_bt, &empty])));
        for (level, inputs) in cases {
            for job_size in [None, Some(512 << 10)] {
                let opts = CompressOptions {
                    level,
                    job_size,
                    overlap_log: 0,
                    ..Default::default()
                };
                let mut cx = Compressor::new(opts.clone());
                for input in inputs {
                    let reused = cx.compress_to_vec(input);
                    assert!(
                        reused == compress_with(input, &opts),
                        "level {level}, job_size {job_size:?}, {} bytes",
                        input.len()
                    );
                }
                for ctx in &cx.jobs {
                    let low = ctx.ms.as_ref().unwrap().window_low();
                    assert!(low > WINDOW_START_INDEX, "level {level}: indices restarted");
                }
                let mut out = b"prefix".to_vec();
                cx.compress(inputs[1], &mut out);
                assert!(out.starts_with(b"prefix"));
                assert!(out[6..] == compress_with(inputs[1], &opts)[..]);
            }
        }
    }

    /// An input of at most `JOBSIZE_MIN` bytes compresses identically with
    /// the default and with any explicit job size, which covers it (300 KiB
    /// clamps up to JOBSIZE_MIN) and does not start ZSTDMT.
    #[test]
    fn default_job_size_equals_explicit_for_single_job_input() {
        let data = text(300 << 10);
        for level in [1, 3] {
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

    /// Text, then copies of 4 KiB from anywhere earlier, each followed by
    /// 64 noise bytes: matches at every distance up to the whole input.
    fn repeats(len: usize) -> Vec<u8> {
        let mut v = text(64 << 10);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        while v.len() < len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let start = (x >> 20) as usize % (v.len() - 4096);
            v.extend_from_within(start..start + 4096);
            v.extend_from_slice(&noise(64, x));
        }
        v.truncate(len);
        v
    }

    /// With `overflow_correct_frequently` the match state's window is
    /// corrected in single-job and multi-job frames (blocks of a 2 MiB job
    /// start past the correction threshold of a 1 MiB window), and the
    /// frames are those without it: a correction keeps every index the
    /// window reaches.
    #[test]
    fn frequent_overflow_correction_keeps_frames() {
        let data = repeats(3 << 20);
        let jobs = Some(2 << 20);
        for (level, job_size) in [
            (-5, None),
            (1, None),
            (2, None),
            (5, None),
            (-5, jobs),
            (1, jobs),
            (2, jobs),
        ] {
            let opts = CompressOptions {
                level,
                job_size,
                ..Default::default()
            };
            let mut cx = Compressor::new(CompressOptions {
                overflow_correct_frequently: true,
                ..opts.clone()
            });
            let frame = cx.compress_to_vec(&data);
            let name = format!("L{level} job {job_size:?}");
            assert!(cx.overflow_corrections().0 > 0, "{name}: no correction");
            assert!(frame == compress_with(&data, &opts), "{name}");
        }
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
