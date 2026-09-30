//! Single-job encode time of the block loop: the serial loop
//! (`--no-default-features`) against the pipelined loop (parallel feature,
//! run inside a rayon pool of `ZSTD_BENCH_THREADS` threads, default 2),
//! through `compress_with` and through a reused `Compressor`, plus the
//! pipeline's proof-failure rate. Ignored by default; run each build under
//! the shared MT lock on the same cores:
//!
//! ```text
//! flock /tmp/claude-1000/zstd-mtbench.lock taskset -c 4,6 \
//!   cargo test --release --offline --test pipeline_bench [--no-default-features] -- --ignored --nocapture
//! ```
//!
//! `ZSTD_CORPUS_DIR` overrides the corpus directory, `ZSTD_BENCH_ITERS`
//! the iterations per cell (default 11).

use rust_zstd::compress::{compress_with, CompressOptions, Compressor};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_CORPUS: &str = "/tmp/claude-1000/-home-stevek-work-rust-zstd/d30c8856-c9ae-4039-8110-94096bb23bce/scratchpad/corpus";
const FILES: [&str; 3] = ["elf_8M.bin", "rssrc_8M.txt", "words_1M.txt"];
const LEVELS: [i32; 4] = [1, 3, 5, 11];

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// (blocks overlapped, blocks whose proof failed) during `f`.
#[cfg(feature = "parallel")]
fn proof_counts<F: FnOnce()>(f: F) -> (usize, usize) {
    use rust_zstd::compress::block::{PIPELINE_OVERLAPPED, PIPELINE_SERIALIZED};
    use std::sync::atomic::Ordering::Relaxed;
    let (o, s) = (
        PIPELINE_OVERLAPPED.load(Relaxed),
        PIPELINE_SERIALIZED.load(Relaxed),
    );
    f();
    (
        PIPELINE_OVERLAPPED.load(Relaxed) - o,
        PIPELINE_SERIALIZED.load(Relaxed) - s,
    )
}

#[cfg(not(feature = "parallel"))]
fn proof_counts<F: FnOnce()>(f: F) -> (usize, usize) {
    f();
    (0, 0)
}

fn run() {
    let dir = std::env::var_os("ZSTD_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CORPUS));
    let iters: usize = std::env::var("ZSTD_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(11);
    let mode = if cfg!(feature = "parallel") {
        "pipelined"
    } else {
        "serial"
    };
    eprintln!(
        "\n{:<13} {:>3} {:>9} {:>9} {:>10} {:>10} {:>8} {:>8}",
        "dataset", "L", "mode", "size", "fresh ms", "reuse ms", "overlap", "failed"
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
        for level in LEVELS {
            let opts = CompressOptions {
                level,
                job_size: Some(1 << 30),
                ..CompressOptions::default()
            };
            let mut frame = Vec::new();
            let (overlapped, failed) = proof_counts(|| frame = compress_with(&data, &opts));
            let mut cx = Compressor::new(opts.clone());
            let _ = cx.compress_to_vec(&data);
            let (mut fresh, mut reuse) = (Vec::new(), Vec::new());
            for _ in 0..iters {
                let t = Instant::now();
                let f = compress_with(&data, &opts);
                fresh.push(t.elapsed());
                let t = Instant::now();
                let r = cx.compress_to_vec(&data);
                reuse.push(t.elapsed());
                assert!(f == frame && r == frame, "{file} L{level}: frames differ");
            }
            eprintln!(
                "{:<13} {:>3} {:>9} {:>9} {:>10.2} {:>10.2} {:>8} {:>8}",
                file,
                level,
                mode,
                frame.len(),
                ms(median(fresh)),
                ms(median(reuse)),
                overlapped,
                failed
            );
        }
    }
}

#[test]
#[ignore]
fn pipeline_split() {
    #[cfg(feature = "parallel")]
    {
        let threads = std::env::var("ZSTD_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        eprintln!("rayon pool: {threads} threads");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(run);
    }
    #[cfg(not(feature = "parallel"))]
    run();
}
