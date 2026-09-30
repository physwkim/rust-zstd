//! Job grid under multi-threading: rust-zstd with an 8-thread rayon pool
//! against libzstd 1.5.7 with `nbWorkers = 8`, both at the same
//! `jobSize` and `overlapLog`. Per cell: compressed size relative to the
//! same implementation's default options, and encode MB/s (median).
//! Ignored by default; run in release under the shared MT lock:
//!
//! ```text
//! flock /tmp/claude-1000/zstd-mtbench.lock taskset -c 4,6,7,8,10,12,14,15 \
//!   cargo test --release --offline --test mt_grid -- --ignored --nocapture
//! ```
//!
//! `ZSTD_CORPUS_DIR` overrides the corpus directory, `ZSTD_BENCH_ITERS`
//! the iterations per cell (default 5).
#![cfg(feature = "parallel")]

use rust_zstd::compress::{CompressOptions, Compressor};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use zstd::zstd_safe::CParameter;

const DEFAULT_CORPUS: &str = "/tmp/claude-1000/-home-stevek-work-rust-zstd/d30c8856-c9ae-4039-8110-94096bb23bce/scratchpad/corpus";
const FILES: [&str; 3] = ["elf_8M.bin", "rssrc_8M.txt", "words_1M.txt"];
const LEVELS: [i32; 5] = [1, 3, 5, 7, 11];
const JOB_SIZES: [Option<usize>; 4] = [Some(512 << 10), Some(1 << 20), Some(2 << 20), None];
const OVERLAP_LOGS: [u8; 4] = [0, 7, 8, 9];
const THREADS: usize = 8;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time<F: FnMut() -> Vec<u8>>(iters: usize, mut f: F) -> (Duration, Vec<u8>) {
    let mut frame = f();
    let mut t = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = Instant::now();
        frame = f();
        t.push(s.elapsed());
    }
    (median(t), frame)
}

#[test]
#[ignore]
fn mt_grid() {
    let dir = std::env::var_os("ZSTD_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CORPUS));
    let iters: usize = std::env::var("ZSTD_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap();
    eprintln!(
        "\n{:<13} {:>3} {:>5} {:>3} | {:>9} {:>7} {:>7} | {:>9} {:>7} {:>7} | {:>7} {:>7}",
        "dataset",
        "L",
        "job",
        "ov",
        "rs size",
        "rs rel",
        "rs MB/s",
        "C size",
        "C rel",
        "C MB/s",
        "rs/C sz",
        "rs/C sp"
    );
    for file in FILES {
        let path = dir.join(file);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}: {e} (set ZSTD_CORPUS_DIR)", path.display());
                continue;
            }
        };
        let mbs = |d: Duration| data.len() as f64 / (1 << 20) as f64 / d.as_secs_f64();
        for level in LEVELS {
            let mut rs_default = 0usize;
            let mut c_default = 0usize;
            // Default options first so every other cell has its reference.
            let cells = std::iter::once((None, 0u8)).chain(
                JOB_SIZES
                    .iter()
                    .flat_map(|&j| OVERLAP_LOGS.iter().map(move |&o| (j, o)))
                    .filter(|&(j, o)| !(j.is_none() && o == 0)),
            );
            for (job_size, overlap_log) in cells {
                let opts = CompressOptions {
                    level,
                    job_size,
                    overlap_log,
                    ..Default::default()
                };
                let mut cx = Compressor::new(opts);
                let (rs_t, rs_frame) = pool.install(|| time(iters, || cx.compress_to_vec(&data)));
                assert!(
                    zstd::decode_all(&rs_frame[..]).unwrap() == data,
                    "{file} L{level} {job_size:?} ov{overlap_log}: libzstd rejects our frame"
                );
                let mut c = zstd::bulk::Compressor::new(level).unwrap();
                c.set_parameter(CParameter::NbWorkers(THREADS as u32))
                    .unwrap();
                c.set_parameter(CParameter::JobSize(job_size.unwrap_or(0) as u32))
                    .unwrap();
                c.set_parameter(CParameter::OverlapSizeLog(overlap_log as u32))
                    .unwrap();
                let (c_t, c_frame) = time(iters, || c.compress(&data).unwrap());
                if job_size.is_none() && overlap_log == 0 {
                    rs_default = rs_frame.len();
                    c_default = c_frame.len();
                }
                let job = job_size.map_or("def".to_string(), |j| format!("{}K", j >> 10));
                eprintln!(
                    "{:<13} {:>3} {:>5} {:>3} | {:>9} {:>7.4} {:>7.0} | {:>9} {:>7.4} {:>7.0} | {:>7.4} {:>7.3}",
                    file, level, job, overlap_log,
                    rs_frame.len(), rs_frame.len() as f64 / rs_default as f64, mbs(rs_t),
                    c_frame.len(), c_frame.len() as f64 / c_default as f64, mbs(c_t),
                    rs_frame.len() as f64 / c_frame.len() as f64,
                    mbs(rs_t) / mbs(c_t)
                );
            }
        }
    }
}
