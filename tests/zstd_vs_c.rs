//! rust-zstd against libzstd 1.5.7 (the `zstd` crate, in process): sizes,
//! compression and decompression throughput, and cross-decoding of every
//! frame in both directions.

use std::time::Instant;
use zstd::bulk::{Compressor, Decompressor};

/// (compress seconds, decompress seconds, frame) for our codec.
fn bench_rust(data: &[u8], level: i32, iters: u32) -> (f64, f64, Vec<u8>) {
    let compressed = rust_zstd::compress(data, level);
    let _ = rust_zstd::decompress(&compressed);

    let start = Instant::now();
    let mut c = Vec::new();
    for _ in 0..iters {
        c = rust_zstd::compress(data, level);
    }
    let ct = start.elapsed().as_secs_f64() / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = rust_zstd::decompress(&c);
    }
    let dt = start.elapsed().as_secs_f64() / iters as f64;

    (ct, dt, c)
}

/// (compress seconds, decompress seconds, frame) for libzstd, one context
/// per direction reused across iterations like `ZSTD_compressCCtx`.
fn bench_c_zstd(data: &[u8], level: i32, iters: u32) -> (f64, f64, Vec<u8>) {
    let mut compressor = Compressor::new(level).unwrap();
    let mut decompressor = Decompressor::new().unwrap();

    // Warmup
    let c = compressor.compress(data).unwrap();
    let _ = decompressor.decompress(&c, data.len()).unwrap();

    let start = Instant::now();
    let mut c = Vec::new();
    for _ in 0..iters {
        c = compressor.compress(data).unwrap();
    }
    let ct = start.elapsed().as_secs_f64() / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = decompressor.decompress(&c, data.len()).unwrap();
    }
    let dt = start.elapsed().as_secs_f64() / iters as f64;

    (ct, dt, c)
}

#[test]
fn rust_vs_c_zstd() {
    let datasets: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_1M", vec![0u8; 1_048_576]),
        (
            "text_1M",
            b"The quick brown fox jumps over the lazy dog. Hello world! ".repeat(18000),
        ),
        (
            "f64_seq_1M",
            (0..131072u64)
                .flat_map(|i| (i as f64).to_le_bytes())
                .collect(),
        ),
        (
            "mixed_1M",
            (0..262144u32)
                .flat_map(|i| {
                    if i % 4 == 0 {
                        [0u8; 4]
                    } else {
                        i.to_le_bytes()
                    }
                })
                .collect(),
        ),
    ];

    let iters = 5;

    eprintln!("\n{:=<110}", "");
    eprintln!(
        "{:<12} {:>5} │ {:>8} {:>7} {:>7} │ {:>8} {:>7} {:>7} │ {:>6} {:>6}",
        "Dataset",
        "Level",
        "C_Size",
        "C_Comp",
        "C_Dec",
        "Rs_Size",
        "Rs_Comp",
        "Rs_Dec",
        "Ratio",
        "Speed"
    );
    eprintln!("{:=<110}", "");

    for (name, data) in &datasets {
        let mb = data.len() as f64 / (1024.0 * 1024.0);

        for level in [1, 3, 7, 11] {
            let (c_ct, c_dt, c_frame) = bench_c_zstd(data, level, iters);
            let (r_ct, r_dt, r_frame) = bench_rust(data, level, iters);

            let ours = rust_zstd::decompress(&r_frame)
                .unwrap_or_else(|e| panic!("{name} L{level}: our decoder on our frame: {e}"));
            assert!(
                ours == *data,
                "{name} L{level}: our decoder on our frame: wrong bytes"
            );
            let theirs = zstd::decode_all(&r_frame[..])
                .unwrap_or_else(|e| panic!("{name} L{level}: libzstd on our frame: {e}"));
            assert!(
                theirs == *data,
                "{name} L{level}: libzstd on our frame: wrong bytes"
            );
            let ours_on_c = rust_zstd::decompress(&c_frame)
                .unwrap_or_else(|e| panic!("{name} L{level}: our decoder on libzstd frame: {e}"));
            assert!(
                ours_on_c == *data,
                "{name} L{level}: our decoder on libzstd frame: wrong bytes"
            );

            let c_sz = c_frame.len();
            let r_sz = r_frame.len();
            let c_comp = mb / c_ct;
            let c_dec = mb / c_dt;
            let r_comp = mb / r_ct;
            let r_dec = mb / r_dt;

            // Compression ratio comparison (Rust size / C size)
            let size_ratio = r_sz as f64 / c_sz as f64;
            // Speed comparison (Rust speed / C speed for compress)
            let speed_ratio = r_comp / c_comp;

            eprintln!("{:<12} {:>5} │ {:>8} {:>6.0}M {:>6.0}M │ {:>8} {:>6.0}M {:>6.0}M │ {:>5.2}x {:>5.1}%",
                name, level,
                c_sz, c_comp, c_dec,
                r_sz, r_comp, r_dec,
                size_ratio, speed_ratio * 100.0);
        }
        eprintln!("{:-<110}", "");
    }
}
