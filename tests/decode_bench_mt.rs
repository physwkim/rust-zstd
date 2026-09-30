//! Multi-threaded decoder throughput on libzstd-produced streams, for rayon
//! pools of 1, 2, 4 and 8 threads, next to single-threaded libzstd on the
//! same streams:
//!
//! ```text
//! taskset -c <8 cores> cargo test --release --offline --test decode_bench_mt -- --ignored --nocapture
//! ```
//!
//! Same environment variables as `decode_bench`; `ZSTD_BENCH_THREADS`
//! (comma-separated) overrides the pool sizes.

#![cfg(feature = "parallel")]

mod common;

use common::{datasets, zstd_bulk, MIB};
use std::time::Instant;

fn median_secs(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn bench<F: FnMut() -> usize>(mut f: F, repeats: usize) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        std::hint::black_box(f());
        samples.push(t.elapsed().as_secs_f64());
    }
    median_secs(samples)
}

fn env_list<T: std::str::FromStr + Clone>(name: &str, default: &[T]) -> Vec<T> {
    std::env::var(name)
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

#[test]
#[ignore]
fn decode_throughput_threads() {
    let levels: Vec<i32> = env_list("ZSTD_BENCH_LEVELS", &[1, 3, 11]);
    let threads: Vec<usize> = env_list("ZSTD_BENCH_THREADS", &[1, 2, 4, 8]);
    let only: Option<Vec<String>> = std::env::var("ZSTD_BENCH_DATASETS")
        .ok()
        .map(|s| s.split(',').map(|n| n.trim().to_string()).collect());
    let repeats: usize = std::env::var("ZSTD_BENCH_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21);
    let pools: Vec<rayon::ThreadPool> = threads
        .iter()
        .map(|&n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        })
        .collect();

    let mut header = format!("\n{:<12} {:>3} {:>8}", "dataset", "lvl", "libzstd");
    for n in &threads {
        header += &format!(" {:>7}", format!("t{}", n));
    }
    for n in &threads {
        header += &format!(" {:>6}", format!("x{}", n));
    }
    eprintln!("{}   (MiB/s; x = rust / libzstd)", header);
    for ds in datasets() {
        if ds.data.len() < 1024 {
            continue;
        }
        if let Some(only) = &only {
            if !only.iter().any(|n| n == ds.name) {
                continue;
            }
        }
        for &level in &levels {
            let compressed = zstd_bulk(&ds.data, level);
            let mib = ds.data.len() as f64 / MIB as f64;
            let cap = ds.data.len();
            let c_t = bench(
                || zstd::bulk::decompress(&compressed, cap).unwrap().len(),
                repeats,
            );
            let mut rust_t = Vec::new();
            for pool in &pools {
                pool.install(|| {
                    let out = rust_zstd::decompress(&compressed).expect("decode");
                    assert!(out == ds.data, "{}: decode mismatch", ds.name);
                    rust_t.push(bench(
                        || rust_zstd::decompress(&compressed).unwrap().len(),
                        repeats,
                    ));
                });
            }
            let mut line = format!("{:<12} {:>3} {:>8.0}", ds.name, level, mib / c_t);
            for t in &rust_t {
                line += &format!(" {:>7.0}", mib / t);
            }
            for t in &rust_t {
                line += &format!(" {:>6.2}", c_t / t);
            }
            eprintln!("{}", line);
        }
    }
}
