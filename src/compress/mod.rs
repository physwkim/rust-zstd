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
pub mod dfast;
pub mod fast;
pub mod lazy;
pub mod matchstate;
pub mod params;
pub mod seqstore;

use crate::constants::*;
use block::{write_raw_block, write_rle_block, BlockScratch, BlockState, CommittedBlockState};
use matchstate::MatchState;
pub use params::{CParams, Strategy};
use params::{ZSTD_CLEVEL_DEFAULT, ZSTD_WINDOWLOG_ABSOLUTEMIN};
pub use seqstore::{Seq, SeqStore};
use std::ops::Range;

/// `ZSTDMT_JOBSIZE_MIN`: lower bound of an explicit job size.
pub const JOBSIZE_MIN: usize = 512 << 10;
/// `ZSTDMT_JOBSIZE_MAX` (64-bit): upper bound of an explicit job size.
pub const JOBSIZE_MAX: usize = 1 << 30;
/// `ZSTDMT_JOBLOG_MAX` (64-bit): upper bound of the default job size log.
const JOBLOG_MAX: u32 = 30;

/// Options for [`compress_with`].
///
/// Inputs must be smaller than 4 GiB: match positions are `u32` indices into
/// the input, and [`compress_with`] asserts the limit.
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
    /// `min(window / 8, job start)` bytes of history from the previous job
    /// (`window / 4` for `Strategy::Lazy2`).
    pub job_size: Option<usize>,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            level: ZSTD_CLEVEL_DEFAULT,
            job_size: None,
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
            job_size: None,
        },
    )
}

/// Convenience wrapper: [`compress`] at level 1.
pub fn compress_to_vec(data: &[u8]) -> Vec<u8> {
    compress(data, 1)
}

/// Compress `data` into a zstd frame with `opts`.
pub fn compress_with(data: &[u8], opts: &CompressOptions) -> Vec<u8> {
    assert!(
        data.len() < u32::MAX as usize,
        "inputs of 4 GiB or more are not supported (match indices are u32)"
    );
    let cparams = CParams::for_level(opts.level, data.len());
    let mut out = Vec::with_capacity(data.len() + 64);
    write_frame_header(&mut out, data.len() as u64, cparams.window_log);

    if data.is_empty() {
        write_raw_block(&mut out, &[], true);
        return out;
    }

    // blockSizeMax = MIN(ZSTD_BLOCKSIZE_MAX, 1 << windowLog)
    let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
    let n_blocks = data.len().div_ceil(block_size);

    if opts.level <= 0 {
        for (i, chunk) in data.chunks(block_size).enumerate() {
            let is_last = i + 1 == n_blocks;
            if block::is_rle(chunk) {
                write_rle_block(&mut out, chunk[0], chunk.len(), is_last);
            } else {
                write_raw_block(&mut out, chunk, is_last);
            }
        }
        return out;
    }

    let job_size = job_size_for(opts.job_size, cparams.window_log);
    let jobs = job_ranges(data.len(), job_size);
    let overlap = overlap_size(&cparams);
    let n_jobs = jobs.len();
    let outputs = run_jobs(&jobs, cfg!(feature = "parallel"), |k, job| {
        compress_job(
            data,
            cparams,
            block_size,
            overlap,
            job,
            k == 0,
            k + 1 == n_jobs,
        )
    });
    for o in &outputs {
        out.extend_from_slice(o);
    }
    out
}

/// `ZSTDMT_compressionJob`: compress `data[job]` into a sequence of blocks.
/// Job 0 starts from `repStartValue` with `window_low = 1`; a later job
/// indexes `overlap` bytes before its start (`ZSTD_loadDictionaryContent` on
/// the raw-content prefix), starts with invalidated repeat offsets and no
/// entropy tables, so its first block cannot reference state the decoder
/// obtained from the previous job.
fn compress_job(
    data: &[u8],
    cparams: CParams,
    block_size: usize,
    overlap: usize,
    job: Range<usize>,
    first_job: bool,
    last_job: bool,
) -> Vec<u8> {
    let window_low = if first_job {
        1
    } else {
        job.start.saturating_sub(overlap).max(1)
    };
    let mut ms = MatchState::new(cparams, window_low);
    let mut initial = BlockState::initial();
    if !first_job {
        block::load_prefix(&mut ms, data, window_low..job.start);
        initial.invalidate_rep_codes();
    }
    let mut state = CommittedBlockState::new(initial);
    let mut scratch = BlockScratch::new(block_size);
    let mut out = Vec::with_capacity(job.len() + 3 * job.len().div_ceil(block_size));
    let mut start = job.start;
    while start < job.end {
        let end = (start + block_size).min(job.end);
        block::compress_block(
            &mut ms,
            data,
            start..end,
            first_job && start == job.start,
            last_job && end == job.end,
            &mut state,
            &mut scratch,
            &mut out,
        );
        start = end;
    }
    out
}

/// Run `f` over every job and return the outputs in job order. `parallel`
/// selects rayon when the feature is enabled; the serial loop is otherwise
/// the same, so the concatenated frame is identical either way.
fn run_jobs<F>(jobs: &[Range<usize>], parallel: bool, f: F) -> Vec<Vec<u8>>
where
    F: Fn(usize, Range<usize>) -> Vec<u8> + Sync,
{
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return jobs
            .par_iter()
            .enumerate()
            .map(|(k, job)| f(k, job.clone()))
            .collect();
    }
    let _ = parallel;
    jobs.iter()
        .enumerate()
        .map(|(k, job)| f(k, job.clone()))
        .collect()
}

/// `ZSTDMT_initCStream_internal`'s job size: an explicit size clamped to
/// `[ZSTDMT_JOBSIZE_MIN, ZSTDMT_JOBSIZE_MAX]`, else
/// `1 << ZSTDMT_computeTargetJobLog` (no long-distance matching).
fn job_size_for(requested: Option<usize>, window_log: u32) -> usize {
    match requested {
        Some(n) => n.clamp(JOBSIZE_MIN, JOBSIZE_MAX),
        None => 1usize << 20.max(window_log + 2).min(JOBLOG_MAX),
    }
}

/// Job boundaries: `[0, job_size)`, `[job_size, 2 * job_size)`, ... with the
/// last job truncated to `len`.
fn job_ranges(len: usize, job_size: usize) -> Vec<Range<usize>> {
    (0..len)
        .step_by(job_size)
        .map(|start| start..(start + job_size).min(len))
        .collect()
}

/// `ZSTDMT_computeOverlapSize` with `overlapLog = 0` (the default) and no
/// long-distance matching: `ZSTDMT_overlapLog_default` is 7 for `Lazy2` and
/// 6 for the other strategies, i.e. a quarter or an eighth of the window.
fn overlap_size(cparams: &CParams) -> usize {
    let overlap_log = match cparams.strategy {
        Strategy::Lazy2 => 7,
        Strategy::Fast | Strategy::DFast | Strategy::Greedy | Strategy::Lazy => 6,
    };
    let overlap_rlog = 9 - overlap_log;
    1usize << (cparams.window_log - overlap_rlog)
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
        assert_eq!(job_size_for(Some(0), 19), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(1), 19), JOBSIZE_MIN);
        assert_eq!(job_size_for(Some(JOBSIZE_MIN + 1), 19), JOBSIZE_MIN + 1);
        assert_eq!(job_size_for(Some(usize::MAX), 19), JOBSIZE_MAX);
        // default: 1 << min(max(20, window_log + 2), 30)
        assert_eq!(job_size_for(None, 10), 1 << 20);
        assert_eq!(job_size_for(None, 18), 1 << 20);
        assert_eq!(job_size_for(None, 19), 1 << 21);
        assert_eq!(job_size_for(None, 22), 1 << 24);
        assert_eq!(job_size_for(None, 31), 1 << 30);
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
        assert_eq!(overlap_size(&fast), 1 << (fast.window_log - 3));
        let lazy2 = CParams::for_level(11, 8 << 20);
        assert_eq!(lazy2.strategy, Strategy::Lazy2);
        assert_eq!(overlap_size(&lazy2), 1 << (lazy2.window_log - 2));
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
            };
            let cparams = CParams::for_level(level, data.len());
            let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
            let job_size = job_size_for(opts.job_size, cparams.window_log);
            let jobs = job_ranges(data.len(), job_size);
            assert!(jobs.len() >= 10, "level {level}: {} jobs", jobs.len());
            let overlap = overlap_size(&cparams);
            let n = jobs.len();
            let f = |k: usize, job: Range<usize>| {
                compress_job(
                    data.as_slice(),
                    cparams,
                    block_size,
                    overlap,
                    job,
                    k == 0,
                    k + 1 == n,
                )
            };
            let par = run_jobs(&jobs, true, f);
            let seq = run_jobs(&jobs, false, f);
            assert!(par == seq, "level {level}: job outputs differ");
            let frame = compress_with(&data, &opts);
            let blocks: Vec<u8> = seq.concat();
            assert!(
                frame.ends_with(&blocks),
                "level {level}: frame != header + jobs"
            );
            assert_eq!(crate::decompress(&frame).unwrap(), data);
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
                job_size_for(None, CParams::for_level(level, data.len()).window_log),
                1 << 21
            );
            let auto = compress_with(
                &data,
                &CompressOptions {
                    level,
                    job_size: None,
                },
            );
            for js in [300 << 10, 512 << 10, 1 << 20] {
                let explicit = compress_with(
                    &data,
                    &CompressOptions {
                        level,
                        job_size: Some(js),
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
        };
        let cparams = CParams::for_level(1, data.len());
        assert!(overlap_size(&cparams) >= copy);
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
