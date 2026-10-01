//! Zstandard frame compressor.
//!
//! libzstd's block architecture (`ZSTD_compress_frameChunk`): match finding
//! runs per block on a persistent [`MatchState`], sequences never cross a
//! block boundary, and the entropy stage consumes a per-block [`SeqStore`]
//! against the committed cross-block [`BlockState`].
//!
//! Above that, ZSTDMT's job architecture: the input is cut into jobs, each
//! compressed independently with a block state of its own, on a context
//! taken from a pool of at most one per worker thread (`ZSTDMT_CCtxPool`)
//! whose [`MatchState`] it resets, after indexing an overlap of the
//! preceding bytes (`ZSTDMT_computeOverlapSize`), and the job outputs are
//! concatenated. The job loop is the same with and without the `parallel`
//! feature, so both builds emit identical frames.

pub mod block;
pub mod bt;
pub mod common;
pub mod dfast;
pub mod dict;
mod error;
pub mod fast;
pub mod lazy;
pub mod ldm;
pub mod matchstate;
pub mod opt;
pub mod params;
pub mod presplit;
pub mod seqstore;
pub mod split;
mod stream;

use crate::constants::*;
use crate::xxhash::Xxh64;
use block::{
    write_raw_block, BlockLdm, BlockScratch, BlockSizing, BlockState, CommittedBlockState,
    InputEnd, JobBlocks, ZSTD_BLOCKHEADERSIZE,
};
use dict::FrameDict;
pub use dict::{CompressDict, DictContentType};
pub use error::CompressError;
use lazy::{default_search_method, SearchMethod};
use ldm::{LdmParams, LdmState, RawSeqStore, LDM_DEFAULT_WINDOW_LOG};
use matchstate::{needed_space, MatchState};
use params::{CParamMode, ZSTD_CLEVEL_DEFAULT, ZSTD_WINDOWLOG_ABSOLUTEMIN};
pub use params::{CParams, ParamSwitch, Strategy};
pub use seqstore::{Seq, SeqStore};
use std::ops::Range;
#[cfg(feature = "parallel")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Condvar,
};
use std::sync::{Arc, Mutex};
pub use stream::{Encoder, EndDirective};

/// `ZSTDMT_JOBSIZE_MIN`: lower bound of an explicit job size.
pub const JOBSIZE_MIN: usize = 512 << 10;
/// `ZSTDMT_JOBSIZE_MAX`: upper bound of an explicit job size, 1 GiB, or
/// 512 MiB where `size_t` is 32 bits.
pub const JOBSIZE_MAX: usize = if MEM_32BITS { 512 << 20 } else { 1 << 30 };
/// `ZSTDMT_JOBLOG_MAX`: upper bound of `ZSTDMT_computeTargetJobLog`, 30,
/// or 29 where `size_t` is 32 bits.
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
    /// `ZSTD_c_checksumFlag`: set the header's Content_Checksum_flag and
    /// end the frame with a Content_Checksum, the low 32 bits of the XXH64
    /// of the input. `false` (the default) writes none, as `ZSTD_compress2`
    /// does. With a `job_size` the one checksum still covers the whole
    /// input and follows the last job's blocks: ZSTDMT hashes each job's
    /// input in job order (`ZSTDMT_serialState_update`).
    pub checksum: bool,
    /// Job size in bytes (`ZSTD_c_jobSize`). `None` (the default) compresses
    /// the input as one job, as single-threaded `ZSTD_compress2`
    /// (`ZSTD_c_nbWorkers` 0) does. `Some` selects ZSTDMT: the input is cut
    /// into jobs of that many bytes, clamped to `[JOBSIZE_MIN, JOBSIZE_MAX]`
    /// (512 KiB to 1 GiB, or to 512 MiB where `usize` is 32 bits) and
    /// raised to the overlap size (`ZSTDMT_initCStream_internal`). `Some(0)`
    /// is libzstd's automatic size, `1 << ZSTDMT_computeTargetJobLog`:
    /// `1 << max(20, window_log + 2)`, with long distance matching
    /// `1 << max(21, cycleLog + 3)`, capped at 1 GiB (512 MiB where `usize`
    /// is 32 bits). Each job is compressed independently and, with the
    /// `parallel` feature, on its own rayon task. The frame never depends
    /// on the thread count, and is the same for a given job size whether or
    /// not the `parallel` feature is enabled. Smaller jobs give more
    /// parallelism and slightly worse ratios, since a job only sees
    /// `min(overlap, job start)` bytes of history from the previous job, see
    /// `overlap_log`.
    ///
    /// For inputs above `JOBSIZE_MIN` (below it libzstd does not start
    /// ZSTDMT) `Some` also selects ZSTDMT's block sizing: each job is fed in
    /// 512 KiB chunks that no block crosses and that bound the pre-split
    /// blocks of `block_splitter_level`.
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
    /// log, within `6..=30`), else `6..=30`. A hash rate log above the
    /// window log derives 6, where libzstd's unsigned subtraction wraps
    /// and derives 30 (an 8 GiB table). Where `usize` is 32 bits, a hash
    /// log above 27 (a table over `isize::MAX` bytes) panics before any
    /// output when long distance matching is enabled.
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
    /// Out-of-range LDM values panic before any output, where
    /// `ZSTD_CCtx_setParameter` returns `parameter_outOfBound`.
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
    /// every job is whole. Values above 6 panic before any output, where
    /// `ZSTD_CCtx_setParameter` returns `parameter_outOfBound`.
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
    /// `ZSTD_CCtx_refCDict`: compress every frame with this dictionary, as
    /// `ZSTD_compress2` does: its level supersedes `level`, the frame
    /// header carries its ID, and the frame is one job whatever
    /// `job_size` says (ZSTDMT with a dictionary is not supported yet).
    /// See [`CompressDict`] and [`dict`].
    pub dict: Option<Arc<CompressDict>>,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            level: ZSTD_CLEVEL_DEFAULT,
            checksum: false,
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
            dict: None,
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
    /// input of `src_size` bytes without a dictionary: the frame's
    /// compression parameters and, when long distance matching resolves to
    /// enabled, its parameters (`ZSTD_ldm_adjustParameters`).
    fn frame_params(&self, src_size: usize) -> (CParams, Option<LdmParams>) {
        let (cparams, ldm) = self.frame_cparams(self.level, src_size, 0, CParamMode::NoAttachDict);
        (cparams, ldm.map(|requested| requested.adjusted(&cparams)))
    }

    /// `ZSTD_getCParamsFromCCtxParams` and `ZSTD_resolveEnableLdm` at
    /// `level` for an input of `src_size` bytes and a dictionary of
    /// `dict_size` bytes used in `mode`: the frame's compression parameters
    /// and, when long distance matching resolves to enabled, the requested
    /// parameters, which `ZSTD_ldm_adjustParameters` completes for the
    /// parameters the frame is compressed with ([`LdmParams::adjusted`]).
    ///
    /// Also the one place options are checked against libzstd's bounds,
    /// panicking where `ZSTD_CCtx_setParameter` returns
    /// `parameter_outOfBound`: [`Compressor::compress`] calls it before
    /// writing anything, so an out-of-range option is rejected the same way
    /// whatever the input. So is an LDM hash log whose table cannot be
    /// allocated ([`LdmParams::adjusted`]). The other options have no
    /// rejected values: the level, job size and overlap log clamp as
    /// libzstd clamps them.
    fn frame_cparams(
        &self,
        level: i32,
        src_size: usize,
        dict_size: usize,
        mode: CParamMode,
    ) -> (CParams, Option<LdmParams>) {
        assert!(
            self.block_splitter_level <= presplit::BLOCK_SPLITTER_LEVEL_MAX,
            "block_splitter_level {} out of range 0..={}",
            self.block_splitter_level,
            presplit::BLOCK_SPLITTER_LEVEL_MAX
        );
        let requested = LdmParams::requested(
            self.ldm_hash_log,
            self.ldm_min_match,
            self.ldm_bucket_size_log,
            self.ldm_hash_rate_log,
        );
        let src = Some(src_size as u64);
        let mut cparams = CParams::for_level_with(level, src, dict_size, mode);
        if self.ldm == ParamSwitch::Enable {
            cparams.window_log = LDM_DEFAULT_WINDOW_LOG;
            cparams = cparams.adjust_with(src, dict_size, mode);
        }
        let enabled = match self.ldm {
            // wlog >= 27, strategy >= btopt
            ParamSwitch::Auto => cparams.strategy >= Strategy::BtOpt && cparams.window_log >= 27,
            ParamSwitch::Enable => true,
            ParamSwitch::Disable => false,
        };
        (cparams, enabled.then_some(requested))
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

/// Compress `data` into a zstd frame with the dictionary `dict`, at its
/// level (`ZSTD_compress2` with `ZSTD_CCtx_refCDict`): see
/// [`CompressOptions::dict`]. Decoding needs the same dictionary.
pub fn compress_with_dict(data: &[u8], dict: &CompressDict) -> Vec<u8> {
    let mut out = Vec::new();
    let opts = CompressOptions::default();
    let dict = FrameDict::of(dict, data.len(), &opts);
    Compressor::new(opts).compress_frame(data, Some(&dict), &mut out);
    out
}

/// Compress `data` into a zstd frame with `opts` and the raw-content
/// prefix `prefix` (`ZSTD_compress2` with `ZSTD_CCtx_refPrefix`), through a
/// one-off [`Compressor`]: see [`Compressor::compress_with_prefix`].
pub fn compress_with_prefix(data: &[u8], prefix: &[u8], opts: &CompressOptions) -> Vec<u8> {
    let mut out = Vec::new();
    Compressor::new(opts.clone()).compress_with_prefix(data, prefix, &mut out);
    out
}

/// A reusable `ZSTD_CCtx`: the options plus ZSTDMT's pool of contexts
/// (`ContextPool`), which every job, a single-threaded frame's one job
/// included, runs on, and which is kept across calls. A context is reset
/// for each job (and each empty frame) like `ZSTD_resetCCtx_internal`:
/// indices continue from its previous input and its tables are kept until
/// its workspace is resized, see [`MatchState::reset_needing`]. Every
/// frame is identical to [`compress_with`]'s.
pub struct Compressor {
    opts: CompressOptions,
    contexts: ContextPool,
    /// ZSTDMT's long distance matching state (`serialState.ldmState`),
    /// once a multithreaded frame has used it. Outside any context's
    /// workspace, its tables only grow, as `ZSTDMT_serialState_reset`'s.
    serial_ldm: Option<LdmState>,
    /// The streaming session ([`Compressor::compress_stream`]).
    stream: stream::Session,
}

/// A compression context (`ZSTD_CCtx`), which runs one job at a time: its
/// match state once a job has run, its block buffers, and the long distance
/// matching state a single-threaded frame generates each block's matches
/// from (`ldmState`). libzstd keeps all three in the context's workspace,
/// whose bookkeeping is in the match state's [`matchstate::Workspace`].
#[derive(Default)]
struct Context {
    ms: Option<MatchState>,
    scratch: BlockScratch,
    ldm_state: Option<LdmState>,
}

/// Where a job's long distance matches come from: nowhere, the frame's
/// serial state (ZSTDMT's `rawSeqStore`), or the context's own state with
/// these parameters, block by block (`ZSTD_buildSeqStore`).
enum JobLdm<'a> {
    Off,
    External(&'a mut RawSeqStore),
    Internal(LdmParams),
}

impl Context {
    /// `ZSTD_resetCCtx_internal` for an input of `pledged` bytes (the
    /// frame's, or a later ZSTDMT job's own) whose window starts at
    /// position `origin`, with tables for the lazy finder `method` and the
    /// overflow correction knob `frequently`.
    /// The match state's reset decides from what libzstd's workspace would
    /// need ([`needed_space`]) whether the workspace is resized; a resize
    /// frees the block buffers and the long distance matching tables too
    /// (`ZSTD_cwksp_free`). Returns the match state, the block buffers and
    /// the job's long distance matches.
    fn reset<'a>(
        &'a mut self,
        cparams: CParams,
        method: SearchMethod,
        origin: usize,
        ldm: JobLdm<'a>,
        pledged: usize,
        frequently: bool,
    ) -> (&'a mut MatchState, &'a mut BlockScratch, BlockLdm<'a>) {
        let ldm_params = match &ldm {
            JobLdm::Internal(params) => Some(params),
            JobLdm::Off | JobLdm::External(_) => None,
        };
        let needed = needed_space(&cparams, method, ldm_params, pledged as u64);
        let (ms, resized) = match &mut self.ms {
            Some(ms) => {
                let resized = ms.reset_needing(cparams, origin, method, needed);
                (ms, resized)
            }
            slot => {
                let ms = MatchState::new_needing(cparams, origin, method, needed);
                (slot.insert(ms), true)
            }
        };
        if resized {
            self.scratch = BlockScratch::default();
            self.ldm_state = None;
        }
        ms.set_correct_frequently(frequently);
        let ldm = match ldm {
            JobLdm::Off => BlockLdm::Off,
            JobLdm::External(seqs) => BlockLdm::External(seqs),
            JobLdm::Internal(params) => {
                BlockLdm::Internal(reset_ldm_state(&mut self.ldm_state, params, frequently))
            }
        };
        (ms, &mut self.scratch, ldm)
    }

    /// The match state, block buffers and single-context long distance
    /// matches (`ldm` set) of the job [`Context::reset`] last started,
    /// for a job compressed over several calls.
    fn resume(&mut self, ldm: bool) -> (&mut MatchState, &mut BlockScratch, BlockLdm<'_>) {
        let ms = self.ms.as_mut().expect("context never reset");
        let ldm = match &mut self.ldm_state {
            Some(state) if ldm => BlockLdm::Internal(state),
            _ => BlockLdm::Off,
        };
        (ms, &mut self.scratch, ldm)
    }

    /// The input of the job [`Context::resume`] continues moved `shift`
    /// bytes down ([`Window::rebase`]).
    ///
    /// [`Window::rebase`]: matchstate::Window::rebase
    fn rebase(&mut self, shift: usize, ldm: bool) {
        self.ms.as_mut().expect("context never reset").rebase(shift);
        if ldm {
            self.ldm_state
                .as_mut()
                .expect("ldm never reset")
                .rebase(shift);
        }
    }
}

/// `ZSTDMT_CCtxPool`: the contexts jobs run on, last in first out. A job
/// takes a context as it starts and gives it back when it finishes, so a
/// frame creates no more contexts than it runs jobs at once, at most one
/// per worker thread, and the pool keeps no more than that across frames.
#[derive(Default)]
struct ContextPool {
    free: Mutex<Vec<Context>>,
    /// `totalCCtx`: how many contexts the pool keeps.
    capacity: usize,
}

impl ContextPool {
    /// `ZSTDMT_expandCCtxPool`: keep a context for each of `workers`
    /// threads; the pool never shrinks.
    fn expand(&mut self, workers: usize) {
        self.capacity = self.capacity.max(workers);
    }

    /// Run `f` on the last context given back, or a new one if none is free
    /// (`ZSTDMT_getCCtx`), and give it back once `f` returns
    /// (`ZSTDMT_releaseCCtx`) unless the pool is full. A context `f` unwinds
    /// from is dropped, never reused.
    fn with_context<R>(&self, f: impl FnOnce(&mut Context) -> R) -> R {
        let mut ctx = self.take();
        let r = f(&mut ctx);
        self.give_back(ctx);
        r
    }

    /// `ZSTDMT_getCCtx`: the last context given back, or a new one.
    fn take(&self) -> Context {
        self.free.lock().unwrap().pop().unwrap_or_default()
    }

    /// `ZSTDMT_releaseCCtx`: keep `ctx` unless the pool is full.
    fn give_back(&self, ctx: Context) {
        let mut free = self.free.lock().unwrap();
        if free.len() < self.capacity {
            free.push(ctx);
        }
    }
}

impl Compressor {
    pub fn new(opts: CompressOptions) -> Self {
        Self {
            opts,
            contexts: ContextPool::default(),
            serial_ldm: None,
            stream: stream::Session::default(),
        }
    }

    /// Test hook for `CompressOptions::overflow_correct_frequently`: the
    /// window overflow corrections of the match states since each last
    /// restarted its indices, and of the long distance matchers since each
    /// was last reset; for a fresh `Compressor`, those of its one frame.
    #[doc(hidden)]
    pub fn overflow_corrections(&self) -> (u32, u32) {
        let contexts = self.contexts.free.lock().unwrap();
        let ms = contexts.iter().filter_map(|ctx| ctx.ms.as_ref());
        let ldm = contexts.iter().filter_map(|ctx| ctx.ldm_state.as_ref());
        let ldm = ldm.chain(self.serial_ldm.as_ref());
        (
            ms.map(|ms| ms.window().nb_overflow_corrections()).sum(),
            ldm.map(|ldm| ldm.window().nb_overflow_corrections()).sum(),
        )
    }

    /// Test hook for the workspace shrink: the workspace size
    /// (`ZSTD_cwksp_sizeof`) of each idle context that has been reset, in
    /// the order [`ContextPool::with_context`] hands them out.
    #[doc(hidden)]
    pub fn workspace_sizes(&self) -> Vec<usize> {
        let contexts = self.contexts.free.lock().unwrap();
        let ms = contexts.iter().rev().filter_map(|ctx| ctx.ms.as_ref());
        ms.map(MatchState::workspace_size).collect()
    }

    /// Append one frame holding `src` to `out`. A streaming frame in
    /// progress is abandoned first, as `ZSTD_compress2` resets the session.
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) {
        self.reset_stream();
        match self.opts.dict.clone() {
            Some(dict) => {
                let dict = FrameDict::of(&dict, src.len(), &self.opts);
                self.compress_frame(src, Some(&dict), out);
            }
            None => self.compress_frame(src, None, out),
        }
    }

    /// Append one frame holding `src` to `out`, compressed with the
    /// raw-content prefix `prefix` before it (`ZSTD_CCtx_refPrefix`, a
    /// dictionary of content only, for this frame alone): the parameters
    /// are sized for `src` and the prefix, the frame header carries no
    /// dictionary ID, and decoding needs the same prefix as a raw-content
    /// dictionary. A prefix under 8 bytes is ignored. Replaces
    /// [`CompressOptions::dict`] for this frame. A streaming frame in
    /// progress is abandoned first, as for [`Compressor::compress`].
    pub fn compress_with_prefix(&mut self, src: &[u8], prefix: &[u8], out: &mut Vec<u8>) {
        self.reset_stream();
        let dict = FrameDict::prefix(prefix, src.len(), &self.opts);
        self.compress_frame(src, Some(&dict), out);
    }

    /// Append one frame holding `src`, with `dict` if given, to `out`. A
    /// dictionary's content goes before `src` and the frame is one job (see
    /// [`dict`]).
    fn compress_frame(&mut self, src: &[u8], dict: Option<&FrameDict>, out: &mut Vec<u8>) {
        let (frame_cparams, cparams, ldm_params) = match dict {
            Some(dict) => dict.params(),
            None => {
                let (cparams, ldm) = self.opts.frame_params(src.len());
                (cparams, cparams, ldm)
            }
        };
        let joined;
        let data = match dict.map(FrameDict::content) {
            Some(content) if !content.is_empty() => {
                joined = [content, src].concat();
                &joined[..]
            }
            _ => src,
        };
        let src_start = data.len() - src.len();
        out.reserve(src.len() + 64);
        let header_start = out.len();
        write_frame_header(
            out,
            Some(src.len() as u64),
            cparams.window_log,
            self.opts.checksum,
            dict.map_or(0, FrameDict::id),
        );
        let header_len = out.len() - header_start;
        // XXH64_update over the input in job order, before each job is
        // compressed (ZSTDMT_serialState_update, ZSTD_compressContinue).
        let mut checksum = self.opts.checksum.then(Xxh64::new);

        let method = dict.map_or_else(|| default_search_method(&cparams), FrameDict::search_method);
        if src.is_empty() {
            // ZSTD_compress2 resets a context for the empty frame too.
            let ldm = ldm_params.map_or(JobLdm::Off, JobLdm::Internal);
            let frequently = self.opts.overflow_correct_frequently;
            self.contexts.expand(1);
            self.contexts.with_context(|ctx| {
                ctx.reset(cparams, method, 0, ldm, 0, frequently);
            });
            write_raw_block(out, &[], true);
            write_epilogue(out, checksum);
            return;
        }

        let ldm_on = ldm_params.is_some();
        let overlap = overlap_size(&cparams, self.opts.overlap_log, ldm_on);
        // A frame with a dictionary is one job.
        let requested = self.opts.job_size.filter(|_| dict.is_none());
        let job_size = job_size_for(requested, &cparams, ldm_on, overlap);
        let split = split::block_splitter_enabled(self.opts.split_after_sequences, &frame_cparams);
        let jobs: Vec<_> = job_ranges(src.len(), job_size)
            .into_iter()
            .map(|job| src_start + job.start..src_start + job.end)
            .collect();
        let n_jobs = jobs.len();
        let mt = dict.is_none() && multithreaded(&self.opts, src.len());
        let sizing = block_sizing(&self.opts, &cparams, mt, header_len);
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
            &mut self.contexts,
            pipelined,
            // ZSTDMT_serialState_update
            |job, seqs| {
                if let Some(checksum) = &mut checksum {
                    checksum.update(&data[job.clone()]);
                }
                if let Some(state) = &mut serial_ldm {
                    state.generate_sequences(data, job.clone(), max_seqs, seqs);
                }
            },
            |k, job, ctx, seqs, out| {
                let ldm = match ldm_params {
                    None => JobLdm::Off,
                    Some(_) if mt => JobLdm::External(seqs),
                    Some(params) => JobLdm::Internal(params),
                };
                compress_job(
                    data,
                    dict,
                    cparams,
                    method,
                    ldm,
                    frequently,
                    sizing,
                    overlap,
                    job,
                    k == 0,
                    k + 1 == n_jobs,
                    split,
                    pipelined,
                    ctx,
                    out,
                )
            },
            out,
        );
        write_epilogue(out, checksum);
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
/// appended to `out`, on `ctx` reset for this job ([`Context::reset`]) with
/// tables for `method` and the long distance matches of `ldm`. Job 0
/// starts from `repStartValue` with its first byte as the window start and
/// pledges the rest of `data`; a later job pledges itself, its window
/// starts `overlap` bytes before it, and the job indexes that prefix
/// (`ZSTD_loadDictionaryContent` on the raw-content prefix), starts with
/// invalidated repeat offsets and no entropy tables, so its first block
/// cannot reference state the decoder obtained from the previous job. The
/// one job of a frame with `dict` has the dictionary's content before it,
/// where its window starts, and starts from the dictionary instead
/// ([`FrameDict::preload`]). `sizing` cuts the job into blocks; `split`
/// runs every block through the post-sequence splitter; `frequently` is
/// the overflow correction knob (see [`CompressOptions`]).
#[allow(clippy::too_many_arguments)]
fn compress_job(
    data: &[u8],
    dict: Option<&FrameDict>,
    cparams: CParams,
    method: SearchMethod,
    ldm: JobLdm,
    frequently: bool,
    sizing: BlockSizing,
    overlap: usize,
    job: Range<usize>,
    first_job: bool,
    last_job: bool,
    split: bool,
    pipelined: bool,
    ctx: &mut Context,
    out: &mut Vec<u8>,
) {
    // ZSTDMT: a job's window starts at its prefix (ZSTD_dct_rawContent).
    // A dictionary's content is the one job's prefix instead.
    let prefix = match dict {
        Some(_) => 0..job.start,
        None => job_prefix(&job, first_job, overlap),
    };
    let pledged = if first_job {
        data.len() - job.start
    } else {
        job.len()
    };
    let (ms, scratch, mut ldm) = ctx.reset(cparams, method, prefix.start, ldm, pledged, frequently);
    let (mut blocks, mut state) =
        begin_job(ms, scratch, data, prefix, dict, sizing, first_job, last_job);
    out.reserve(job_bound(job.len(), sizing.block_size_max));
    block::compress_blocks(
        ms,
        data,
        &mut blocks,
        InputEnd::JobEnd(job.end),
        split,
        &mut state,
        scratch,
        &mut ldm,
        out,
        pipelined,
    );
}

/// The start of a job on a context just reset for it: a later ZSTDMT job
/// indexes its raw-content `prefix` of `data` and starts with invalidated
/// repeat offsets; job 0 (`prefix` empty, whatever `data`) starts from
/// `repStartValue`; the one job of a frame with `dict` starts from the
/// dictionary, whose content is `prefix` ([`FrameDict::preload`]). Returns
/// the job's block cursor, its first block at the prefix end, and its
/// committed block state. One-shot jobs and the streaming frame both start
/// here.
#[allow(clippy::too_many_arguments)]
fn begin_job(
    ms: &mut MatchState,
    scratch: &mut BlockScratch,
    data: &[u8],
    prefix: Range<usize>,
    dict: Option<&FrameDict>,
    sizing: BlockSizing,
    first_job: bool,
    last_job: bool,
) -> (JobBlocks, CommittedBlockState) {
    let initial = match dict {
        Some(dict) => dict.preload(ms, data),
        None if !first_job => {
            block::load_prefix(ms, data, prefix.clone());
            let mut initial = BlockState::initial();
            initial.invalidate_rep_codes();
            initial
        }
        None => BlockState::initial(),
    };
    scratch.reserve(sizing.block_size_max);
    let blocks = JobBlocks::new(sizing, prefix.end, first_job, last_job);
    (blocks, CommittedBlockState::new(initial))
}

/// Run `f` over every job, in job order, appending to `out`, each job on a
/// context of `contexts` ([`ContextPool::with_context`]) and with long
/// distance matches of its own (the job's `rawSeqStore`). `prepare`
/// generates job `k`'s matches, in job order and one job at a time, before
/// the job starts: with the feature enabled and `parallel`, on the calling
/// task while the earlier jobs run on rayon, as ZSTDMT serializes its long
/// distance matching across jobs. Job 0 writes straight into `out`, the
/// others into buffers of their own that are appended afterwards; the
/// serial loop hands every job `out`. The job function is the same either
/// way, so the frame is identical.
///
/// Jobs are never rayon tasks: every worker gets one [`JobQueue`] runner
/// (`spawn_broadcast`), which claims jobs in order until none is left. A
/// thread waiting in a job's block `rayon::join` runs whatever rayon hands
/// it; were jobs tasks, it could take a queued job and finish its own a
/// whole job late. It can still take its own runner, or another frame's,
/// which is why a runner that starts inside a job claims nothing. A thread
/// thus runs one of the frame's jobs at a time, and the frame holds at most
/// one context per worker thread.
fn run_jobs<P, F>(
    jobs: &[Range<usize>],
    contexts: &mut ContextPool,
    parallel: bool,
    mut prepare: P,
    f: F,
    out: &mut Vec<u8>,
) where
    P: FnMut(&Range<usize>, &mut RawSeqStore) + Send,
    F: Fn(usize, Range<usize>, &mut Context, &mut RawSeqStore, &mut Vec<u8>) + Sync,
{
    let mut seqs: Vec<RawSeqStore> = jobs.iter().map(|_| RawSeqStore::default()).collect();
    #[cfg(feature = "parallel")]
    if parallel {
        contexts.expand(rayon::current_num_threads());
        let mut rest_out = vec![Vec::new(); jobs.len().saturating_sub(1)];
        {
            let outs = std::iter::once(&mut *out).chain(&mut rest_out);
            let queue = JobQueue::new(jobs, contexts, seqs.iter_mut().zip(outs));
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
    contexts.expand(1);
    for (k, (job, seqs)) in jobs.iter().zip(&mut seqs).enumerate() {
        prepare(job, seqs);
        contexts.with_context(|ctx| f(k, job.clone(), ctx, seqs, out));
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

/// A job's long distance matches and output.
#[cfg(feature = "parallel")]
type JobSlot<'a> = (&'a mut RawSeqStore, &'a mut Vec<u8>);

/// One frame's jobs for [`run_jobs`]' runners, claimed in job order, each
/// once its `prepare` has run.
#[cfg(feature = "parallel")]
struct JobQueue<'a> {
    jobs: &'a [Range<usize>],
    contexts: &'a ContextPool,
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
    fn new(
        jobs: &'a [Range<usize>],
        contexts: &'a ContextPool,
        slots: impl Iterator<Item = JobSlot<'a>>,
    ) -> Self {
        Self {
            jobs,
            contexts,
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
        P: FnMut(&Range<usize>, &mut RawSeqStore),
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
            if let Some((seqs, _)) = slot.lock().unwrap().as_mut() {
                prepare(job, seqs);
            }
            self.publish(Some(k + 1));
        }
    }

    /// Claim the next job, wait until it is prepared and run it on a
    /// context of the pool, until no job is left.
    fn run<F>(&self, f: &F)
    where
        F: Fn(usize, Range<usize>, &mut Context, &mut RawSeqStore, &mut Vec<u8>),
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
            let (seqs, out) = self.slots[k]
                .lock()
                .unwrap()
                .take()
                .expect("job claimed twice");
            let _in_job = InJob::enter();
            self.contexts
                .with_context(|ctx| f(k, job.clone(), ctx, seqs, out));
        }
    }
}

/// The job size of `requested`: `None` is single-threaded, one job of
/// unbounded size; `Some` is `ZSTDMT_initCStream_internal`'s
/// `targetSectionSize`, `1 << target_job_log` for `Some(0)`, else
/// clamped to `[ZSTDMT_JOBSIZE_MIN, ZSTDMT_JOBSIZE_MAX]`, and at least
/// `overlap` ("job size must be >= overlap size").
pub fn job_size_for(
    requested: Option<usize>,
    cparams: &CParams,
    ldm: bool,
    overlap: usize,
) -> usize {
    let job_size = match requested {
        None => return usize::MAX,
        Some(0) => 1 << target_job_log(cparams, ldm),
        Some(n) => n.clamp(JOBSIZE_MIN, JOBSIZE_MAX),
    };
    job_size.max(overlap)
}

/// `ZSTDMT_computeTargetJobLog`: the log of the automatic job size. With
/// long distance matching the window is typically oversized, so the log
/// follows `ZSTD_cycleLog(chainLog, strategy)` instead.
fn target_job_log(cparams: &CParams, ldm: bool) -> u32 {
    let job_log = if ldm {
        21.max(cparams.chain_log - cparams.strategy.bt_scale() + 3)
    } else {
        20.max(cparams.window_log + 2)
    };
    job_log.min(JOBLOG_MAX)
}

/// Job boundaries: `[0, job_size)`, `[job_size, 2 * job_size)`, ... with the
/// last job truncated to `len`.
pub fn job_ranges(len: usize, job_size: usize) -> Vec<Range<usize>> {
    (0..len)
        .step_by(job_size)
        .map(|start| start..start + job_size.min(len - start))
        .collect()
}

/// The raw-content prefix of `job` (ZSTDMT's `job->prefix`), where its
/// window starts: the `overlap` bytes before it, none for the first job,
/// and none under 8 bytes, which `ZSTD_compress_insertDictionary` ignores
/// (`dictSize < 8`) before `ZSTD_loadDictionaryContent` would put them in
/// the window.
pub fn job_prefix(job: &Range<usize>, first_job: bool, overlap: usize) -> Range<usize> {
    let start = if first_job {
        job.start
    } else {
        job.start.saturating_sub(overlap)
    };
    if job.start - start < 8 {
        job.start..job.start
    } else {
        start..job.start
    }
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

/// `ZSTD_writeFrameHeader`: with a `content_size`, Single_Segment iff the
/// window covers it, otherwise (or without one, the `contentSizeFlag` 0 of
/// an unknown pledged size) a Window_Descriptor with mantissa 0 derived
/// from `window_log`; the Content_Checksum_flag of `checksum`, and
/// `dict_id` in the fewest of 1, 2 or 4 bytes, none for 0.
fn write_frame_header(
    out: &mut Vec<u8>,
    content_size: Option<u64>,
    window_log: u32,
    checksum: bool,
    dict_id: u32,
) {
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
    let window_size = 1u64 << window_log;
    let single_segment = content_size.is_some_and(|size| window_size >= size);
    let fcs_code = content_size.map_or(0, |size| {
        (size >= 256) as u8 + (size >= 65536 + 256) as u8 + (size >= 0xFFFF_FFFF) as u8
    });
    let content_size = content_size.unwrap_or(0);
    let dict_id_code = (dict_id > 0) as u8 + (dict_id >= 256) as u8 + (dict_id >= 65536) as u8;
    let descriptor =
        dict_id_code | ((checksum as u8) << 2) | ((single_segment as u8) << 5) | (fcs_code << 6);
    out.push(descriptor);
    if !single_segment {
        out.push(((window_log - ZSTD_WINDOWLOG_ABSOLUTEMIN) << 3) as u8);
    }
    match dict_id_code {
        0 => {}
        1 => out.push(dict_id as u8),
        2 => out.extend_from_slice(&(dict_id as u16).to_le_bytes()),
        _ => out.extend_from_slice(&dict_id.to_le_bytes()),
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

/// `ZSTD_writeEpilogue` after the last block: the Content_Checksum, when
/// the frame has one, from the XXH64 of its whole input.
fn write_epilogue(out: &mut Vec<u8>, checksum: Option<Xxh64>) {
    if let Some(checksum) = checksum {
        out.extend_from_slice(&(checksum.digest() as u32).to_le_bytes());
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

    /// A job's prefix: none for the first job or under 8 bytes, else the
    /// overlap, cut at the input start.
    #[test]
    fn job_prefix_boundaries() {
        let job = 100..200;
        assert_eq!(job_prefix(&job, true, 64), 100..100);
        assert_eq!(job_prefix(&job, false, 0), 100..100);
        assert_eq!(job_prefix(&job, false, 7), 100..100);
        assert_eq!(job_prefix(&job, false, 8), 92..100);
        assert_eq!(job_prefix(&job, false, 9), 91..100);
        assert_eq!(job_prefix(&job, false, 1000), 0..100);
        assert_eq!(job_prefix(&(7..20), false, 1000), 7..7);
        assert_eq!(job_prefix(&(8..20), false, 1000), 0..8);
    }

    /// `ZSTD_compress_insertDictionary` ignores a raw-content prefix under 8
    /// bytes: the job's window starts at the job, so it writes the blocks it
    /// writes without an overlap. The job repeats its 7-byte prefix, which a
    /// window starting there would reach.
    #[test]
    fn job_ignores_a_prefix_under_8_bytes() {
        let mut data = noise(7, 5);
        for _ in 0..300 {
            data.extend_from_within(..7);
        }
        data.extend_from_slice(&text(8 << 10));
        let job = 7..data.len();
        for level in [1, 3, 5, 7, 13, 16] {
            let opts = CompressOptions {
                level,
                ..Default::default()
            };
            let cparams = CParams::for_level(level, data.len());
            let run = |overlap: usize| {
                let mut ctx = Context::default();
                let mut out = Vec::new();
                compress_job(
                    &data,
                    None,
                    cparams,
                    default_search_method(&cparams),
                    JobLdm::Off,
                    false,
                    block_sizing(&opts, &cparams, true, 0),
                    overlap,
                    job.clone(),
                    false,
                    true,
                    false,
                    false,
                    &mut ctx,
                    &mut out,
                );
                out
            };
            assert!(run(7) == run(0), "L{level}");
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
        let cp = CParams::for_level(1, 1 << 20);
        for (requested, clamped) in [
            (max - 1, max - 1),
            (max, max),
            (max + 1, max),
            (usize::MAX, max),
        ] {
            let job_size = job_size_for(Some(requested), &cp, false, 0);
            assert_eq!(job_size, clamped, "{requested}");
        }
        assert_eq!(1 << JOBLOG_MAX, max);
    }

    #[test]
    fn job_sizing() {
        let cp = CParams::for_level(1, 8 << 20);
        let job_size = |requested, overlap| job_size_for(requested, &cp, false, overlap);
        // explicit: clamped to [JOBSIZE_MIN, JOBSIZE_MAX], otherwise used as is
        assert_eq!(job_size(Some(1), 0), JOBSIZE_MIN);
        assert_eq!(job_size(Some(JOBSIZE_MIN + 1), 0), JOBSIZE_MIN + 1);
        assert_eq!(job_size(Some(usize::MAX), 0), JOBSIZE_MAX);
        // default: one job, whatever the overlap
        assert_eq!(job_size(None, 0), usize::MAX);
        assert_eq!(job_size(None, 1 << 23), usize::MAX);
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
        assert_eq!(job_size(Some(1), 1 << 22), 1 << 22);
        assert_eq!(job_size(Some(1), 1 << 18), JOBSIZE_MIN);
        assert_eq!(job_size(Some(JOBSIZE_MIN), 1 << 23), 1 << 23);
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
        let cp = logs(27, 24);
        assert_eq!(
            job_size_for(Some(1), &cp, true, overlap_size(&cp, 9, true)),
            1 << 25
        );
    }

    /// `Some(0)` is `ZSTDMT_computeTargetJobLog`'s automatic size:
    /// `1 << max(20, window_log + 2)`, with long distance matching
    /// `1 << max(21, cycleLog + 3)`, either capped at `1 << JOBLOG_MAX`,
    /// and at least the overlap.
    #[test]
    fn automatic_job_size() {
        let auto = |cp: &CParams, ldm| job_size_for(Some(0), cp, ldm, 0);
        // Window log 18 + 2 is still 20; then 21, 25, and the largest capped.
        assert_eq!(auto(&logs(17, 24), false), 1 << 20);
        assert_eq!(auto(&logs(18, 24), false), 1 << 20);
        assert_eq!(auto(&logs(19, 24), false), 1 << 21);
        assert_eq!(auto(&logs(23, 24), false), 1 << 25);
        assert_eq!(auto(&logs(ZSTD_WINDOWLOG_MAX, 24), false), 1 << JOBLOG_MAX);
        // The chain log instead: 18 + 3 is still 21; then 22, 27, and 33
        // capped, whatever the window.
        assert_eq!(auto(&logs(27, 13), true), 1 << 21);
        assert_eq!(auto(&logs(27, 18), true), 1 << 21);
        assert_eq!(auto(&logs(27, 19), true), 1 << 22);
        assert_eq!(auto(&logs(17, 24), true), 1 << 27);
        assert_eq!(auto(&logs(27, 30), true), 1 << JOBLOG_MAX);
        // Level 22: window log 27 + 2, and chain log 27 of btultra2, whose
        // cycle is 26, + 3.
        let l22 = CParams::for_level(22, 1 << 30);
        assert_eq!(auto(&l22, false), 1 << 29);
        assert_eq!(auto(&l22, true), 1 << 29);
        // At least the overlap.
        let cp = logs(19, 24);
        assert_eq!(job_size_for(Some(0), &cp, false, 1 << 22), 1 << 22);
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
            let job_size = job_size_for(opts.job_size, &cparams, true, overlap);
            let mt = multithreaded(&opts, src.len());
            let sizing = block_sizing(&opts, &cparams, mt, header_len(src, &cparams));
            let split = split::block_splitter_enabled(opts.split_after_sequences, &cparams);
            let jobs = job_ranges(src.len(), job_size);
            let n = jobs.len();
            assert!(n >= min_jobs, "L{level}: {n} jobs");
            let max_seqs = job_size / ldm.min_match_length as usize;
            let generate = || {
                let mut state = LdmState::new(ldm, 0);
                move |job: &Range<usize>, seqs: &mut RawSeqStore| {
                    state.generate_sequences(src, job.clone(), max_seqs, seqs);
                }
            };
            let f = |pipelined: bool| {
                move |k: usize,
                      job: Range<usize>,
                      ctx: &mut Context,
                      seqs: &mut RawSeqStore,
                      out: &mut Vec<u8>| {
                    compress_job(
                        src,
                        None,
                        cparams,
                        default_search_method(&cparams),
                        JobLdm::External(seqs),
                        false,
                        sizing,
                        overlap,
                        job,
                        k == 0,
                        k + 1 == n,
                        split,
                        pipelined,
                        ctx,
                        out,
                    )
                }
            };
            let mut contexts = ContextPool::default();
            let mut par = Vec::new();
            run_jobs(&jobs, &mut contexts, true, generate(), f(true), &mut par);
            let mut seq = Vec::new();
            run_jobs(&jobs, &mut contexts, false, generate(), f(false), &mut seq);
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

    /// Options changed by `set` panic with a message starting with `name`
    /// before [`Compressor::compress`] writes to `out`, on empty, one-byte
    /// and multi-block input, single- and multithreaded.
    fn assert_panics_before_output(name: &str, set: impl Fn(&mut CompressOptions)) {
        let big = vec![b'x'; JOBSIZE_MIN + (300 << 10)];
        for job_size in [None, Some(JOBSIZE_MIN)] {
            for src in [&[][..], b"a", &big] {
                let mut opts = CompressOptions {
                    job_size,
                    ..Default::default()
                };
                set(&mut opts);
                let mut cx = Compressor::new(opts);
                let mut out = b"prefix".to_vec();
                let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cx.compress(src, &mut out)
                }))
                .expect_err(name);
                let msg = err.downcast_ref::<String>().map_or("", |m| m);
                let what = format!("{name} job_size {job_size:?} len {}", src.len());
                assert!(msg.starts_with(name), "{what}: {msg}");
                assert!(out == b"prefix", "{what}: out written");
            }
        }
    }

    /// Every option libzstd rejects panics before any output, and its bound
    /// is libzstd's.
    #[test]
    fn out_of_range_option_panics_before_output() {
        use crate::compress::common::testutil::{c_accepts, c_bounds};
        use zstd::zstd_safe::zstd_sys::ZSTD_cParameter::ZSTD_c_experimentalParam20;

        let (lo, hi) = c_bounds(ZSTD_c_experimentalParam20);
        assert_eq!((lo, hi), (0, presplit::BLOCK_SPLITTER_LEVEL_MAX as i32));
        assert!(!c_accepts(ZSTD_c_experimentalParam20, hi + 1));
        type Set = fn(&mut CompressOptions);
        let bad: [(&str, Set); 5] = [
            ("block_splitter_level 7", |o| o.block_splitter_level = 7),
            ("ldm_hash_log 31", |o| o.ldm_hash_log = 31),
            ("ldm_min_match 3", |o| o.ldm_min_match = 3),
            ("ldm_bucket_size_log 9", |o| o.ldm_bucket_size_log = 9),
            ("ldm_hash_rate_log 26", |o| o.ldm_hash_rate_log = 26),
        ];
        for (name, set) in bad {
            assert_panics_before_output(name, set);
        }
    }

    /// An LDM hash log whose table cannot be allocated panics before any
    /// output (R1-15): `28..=30` where `usize` is 32 bits, none on 64 bits.
    #[test]
    fn unallocatable_ldm_hash_log_panics_before_output() {
        let too_big = (ZSTD_HASHLOG_MIN..=ZSTD_HASHLOG_MAX).filter(|&l| l > ldm::HASHLOG_ALLOC_MAX);
        for log in too_big {
            assert_panics_before_output(&format!("ldm_hash_log {log}"), |o| {
                o.ldm = ParamSwitch::Enable;
                o.ldm_hash_log = log;
            });
        }
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

    /// The preset runs three jobs on 4.25 MiB, on at most three contexts,
    /// and its frame decodes through both decoders.
    #[test]
    fn parallel_preset_roundtrips() {
        let mut data = text(2 << 20);
        data.extend_from_slice(&noise(256 << 10, 7));
        data.extend_from_slice(&text(2 << 20));
        for level in [1, 3, 7, 11] {
            let opts = CompressOptions::parallel(level);
            assert_eq!((opts.job_size, opts.overlap_log), (Some(2 << 20), 8));
            let (cparams, _) = opts.frame_params(data.len());
            let overlap = overlap_size(&cparams, opts.overlap_log, false);
            let jobs = job_ranges(
                data.len(),
                job_size_for(opts.job_size, &cparams, false, overlap),
            );
            assert!(multithreaded(&opts, data.len()), "L{level}");
            assert_eq!(jobs.len(), 3, "L{level}: jobs");
            let mut cx = Compressor::new(opts.clone());
            let frame = cx.compress_to_vec(&data);
            let contexts = cx.contexts.free.get_mut().unwrap();
            assert!((1..=3).contains(&contexts.len()), "L{level}: contexts");
            assert!(contexts.iter().all(|ctx| ctx.ms.is_some()), "L{level}");
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
            let job_size = job_size_for(opts.job_size, &cparams, false, overlap);
            let jobs = job_ranges(data.len(), job_size);
            assert!(jobs.len() >= 5, "level {level}: {} jobs", jobs.len());
            let n = jobs.len();
            let mt = multithreaded(&opts, data.len());
            let sizing = block_sizing(&opts, &cparams, mt, header_len(&data, &cparams));
            let src = data.as_slice();
            let f = |pipelined: bool| {
                move |k: usize,
                      job: Range<usize>,
                      ctx: &mut Context,
                      _: &mut RawSeqStore,
                      out: &mut Vec<u8>| {
                    compress_job(
                        src,
                        None,
                        cparams,
                        default_search_method(&cparams),
                        JobLdm::Off,
                        false,
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
            // The same pool serves both runs, so the serial run also covers
            // the reset of used contexts.
            let mut contexts = ContextPool::default();
            let mut par = Vec::new();
            run_jobs(&jobs, &mut contexts, true, |_, _| {}, f(true), &mut par);
            let mut seq = Vec::new();
            run_jobs(&jobs, &mut contexts, false, |_, _| {}, f(false), &mut seq);
            assert!(par == seq, "level {level}: job outputs differ");
            let frame = compress_with(&data, &opts);
            assert!(
                frame.ends_with(&seq),
                "level {level}: frame != header + jobs"
            );
            assert_eq!(crate::decompress(&frame).unwrap(), data);
        }
    }

    /// The resize that frees a wasteful workspace frees the context's block
    /// buffers with it, as `ZSTD_cwksp_free` frees every buffer libzstd
    /// keeps there; until then they are kept.
    #[test]
    fn context_resize_frees_block_buffers() {
        let opts = CompressOptions {
            level: 19,
            ldm: ParamSwitch::Enable,
            ..Default::default()
        };
        let (big, big_ldm) = opts.frame_params(64 << 20);
        let (small, small_ldm) = opts.frame_params(1024);
        let mut ctx = Context::default();
        let ldm = JobLdm::Internal(big_ldm.unwrap());
        let method = default_search_method(&big);
        let (_, scratch, _) = ctx.reset(big, method, 0, ldm, 64 << 20, false);
        scratch.reserve(ZSTD_BLOCKSIZE_MAX);
        for n in 1..=129 {
            let ldm = JobLdm::Internal(small_ldm.unwrap());
            let method = default_search_method(&small);
            let (_, scratch, _) = ctx.reset(small, method, 0, ldm, 1024, false);
            let kept = scratch.cbuf.capacity() >= ZSTD_BLOCKSIZE_MAX;
            assert_eq!(kept, n < 129, "reset {n}");
        }
    }

    /// The frame header length of `data` with `cparams`.
    fn header_len(data: &[u8], cparams: &CParams) -> usize {
        let mut header = Vec::new();
        write_frame_header(
            &mut header,
            Some(data.len() as u64),
            cparams.window_log,
            false,
            0,
        );
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
        let mut ctx = Context::default();
        let mut out = Vec::new();
        let job = 0..data.len();
        compress_job(
            data,
            None,
            cparams,
            default_search_method(&cparams),
            JobLdm::Off,
            false,
            sizing,
            overlap,
            job,
            true,
            true,
            split,
            pipelined,
            &mut ctx,
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
    /// runs within another's run on the same thread, and the frame's jobs
    /// hold no more contexts than there are threads.
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
        let mut contexts = ContextPool::default();
        let runs = Mutex::new(Vec::new());
        // Contexts held, each by one running job, and the most at once.
        let (held, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let t0 = Instant::now();
        pool.install(|| {
            let job = |k: usize,
                       _: Range<usize>,
                       _: &mut Context,
                       _: &mut RawSeqStore,
                       _: &mut Vec<u8>| {
                most.fetch_max(held.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                let start = t0.elapsed();
                for _ in 0..4 {
                    rayon::join(
                        || spin(Duration::from_micros(50)),
                        || spin(Duration::from_micros(500)),
                    );
                }
                let run = (k, rayon::current_thread_index(), start, t0.elapsed());
                runs.lock().unwrap().push(run);
                held.fetch_sub(1, Ordering::SeqCst);
            };
            run_jobs(&jobs, &mut contexts, true, |_, _| {}, job, &mut Vec::new());
        });
        let runs = runs.into_inner().unwrap();
        assert_eq!(runs.len(), jobs.len());
        let most = most.into_inner();
        assert!((1..=4).contains(&most), "{most} contexts held at once");
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
    /// shrink and grow again within their allocations and its contexts are
    /// reused, continues a context's indices from its previous input and
    /// still produces the frames fresh `compress_with` calls produce, for
    /// every strategy. Single-job frames all run on one context; the
    /// multi-job ones run more jobs than they can hold contexts, so some
    /// context runs two.
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
                let contexts = cx.contexts.free.get_mut().unwrap();
                let continued = contexts
                    .iter()
                    .filter(|ctx| ctx.ms.as_ref().unwrap().window_low() > WINDOW_START_INDEX)
                    .count();
                assert!(continued > 0, "level {level}: indices restarted");
                if job_size.is_none() {
                    assert_eq!(contexts.len(), 1, "level {level}: contexts");
                }
                let mut out = b"prefix".to_vec();
                cx.compress(inputs[1], &mut out);
                assert!(out.starts_with(b"prefix"));
                assert!(out[6..] == compress_with(inputs[1], &opts)[..]);
            }
        }
    }

    /// With one worker thread, every job of a frame runs on the one context,
    /// which continues its indices from job to job, and the frame is
    /// `compress_with`'s.
    #[test]
    fn one_worker_runs_every_job_on_one_context() {
        let data = text(3 << 20);
        let opts = CompressOptions {
            level: 3,
            job_size: Some(JOBSIZE_MIN),
            ..Default::default()
        };
        let mut cx = Compressor::new(opts.clone());
        #[cfg(feature = "parallel")]
        let frame = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| cx.compress_to_vec(&data));
        #[cfg(not(feature = "parallel"))]
        let frame = cx.compress_to_vec(&data);
        assert!(frame == compress_with(&data, &opts));
        let contexts = cx.contexts.free.get_mut().unwrap();
        assert_eq!(contexts.len(), 1);
        // Past the five earlier jobs' inputs.
        let low = contexts[0].ms.as_ref().unwrap().window_low();
        assert!(low >= WINDOW_START_INDEX + 5 * JOBSIZE_MIN, "{low}");
    }

    /// Contexts in use at once beyond the pool's capacity are created, and
    /// the pool keeps `capacity` of them as they are given back.
    #[test]
    fn context_pool_keeps_at_most_its_capacity() {
        let mut pool = ContextPool::default();
        for capacity in [1, 2] {
            pool.expand(capacity);
            pool.with_context(|_| pool.with_context(|_| pool.with_context(|_| {})));
            assert_eq!(pool.free.get_mut().unwrap().len(), capacity);
        }
        pool.expand(1);
        pool.with_context(|_| {});
        assert_eq!(pool.free.get_mut().unwrap().len(), 2, "shrank");
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
