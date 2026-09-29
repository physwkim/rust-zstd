//! Decoder throughput on libzstd-produced streams (level 3 bulk frames).
//!
//! Run pinned to one core, in release mode:
//!
//! ```text
//! taskset -c 3 cargo test --release --offline --test decode_bench -- --ignored --nocapture
//! ```
//!
//! Reports the median MiB/s over repeated single-threaded decodes of the same
//! compressed stream, alongside libzstd decoding the identical stream.

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
        let n = f();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(n);
        samples.push(dt);
    }
    median_secs(samples)
}

#[test]
#[ignore]
fn decode_throughput() {
    let level: i32 = std::env::var("ZSTD_BENCH_LEVEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let repeats: usize = std::env::var("ZSTD_BENCH_REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21);

    eprintln!(
        "\n{:<12} {:>10} {:>10} {:>12} {:>12}",
        "dataset", "raw", "zst", "rust MiB/s", "libzstd MiB/s"
    );
    for ds in datasets() {
        if ds.data.len() < 1024 {
            continue;
        }
        let compressed = zstd_bulk(&ds.data, level);
        // Warm-up and correctness gate.
        let out = rust_zstd::decompress(&compressed).expect("decode");
        assert!(out == ds.data, "{}: decode mismatch", ds.name);
        drop(out);

        let mib = ds.data.len() as f64 / MIB as f64;
        let rust_t = bench(
            || rust_zstd::decompress(&compressed).map(|v| v.len()).unwrap(),
            repeats,
        );
        let cap = ds.data.len();
        let c_t = bench(
            || zstd::bulk::decompress(&compressed, cap).unwrap().len(),
            repeats,
        );
        eprintln!(
            "{:<12} {:>10} {:>10} {:>12.0} {:>12.0}",
            ds.name,
            ds.data.len(),
            compressed.len(),
            mib / rust_t,
            mib / c_t
        );
    }
}
