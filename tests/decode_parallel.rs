//! The multi-threaded frame decoder (`parallel` feature) against libzstd
//! and against the serial decoder: byte-exact output on every dataset, and
//! the same `Err`-vs-`Ok` outcome (and the same output when `Ok`) on
//! truncated and corrupted multi-block frames.

#![cfg(feature = "parallel")]

mod common;

use common::{datasets, lcg_bytes, zstd_bulk, zstd_stream, LEVELS, MIB};
use rust_zstd::decode::{decompress_with_options, DecodeOptions};
use zstd::zstd_safe::zstd_sys as sys;

/// Every frame through the multi-threaded path, however few its blocks.
fn decode_mt(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_with_options(
        data,
        &DecodeOptions {
            min_parallel_blocks: 1,
            simd: true,
            window_log_max: 0,
        },
    )
}

/// Every frame through the fused serial path.
fn decode_serial(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_with_options(
        data,
        &DecodeOptions {
            min_parallel_blocks: usize::MAX,
            simd: true,
            window_log_max: 0,
        },
    )
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

/// libzstd frame with blocks of at most `max_block` bytes and Huffman
/// literals forced on, so that small inputs give many blocks, most of them
/// with treeless literals and repeat-mode FSE tables.
fn zstd_small_blocks(data: &[u8], level: i32, max_block: i32) -> Vec<u8> {
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let set = |p, v| {
            let r = sys::ZSTD_CCtx_setParameter(cctx, p, v);
            assert_eq!(sys::ZSTD_isError(r), 0, "set parameter {:?}", p);
        };
        set(sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level);
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam18, max_block);
        // ZSTD_c_literalCompressionMode = ZSTD_ps_enable.
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam5, 1);
        let mut out = vec![0u8; sys::ZSTD_compressBound(data.len())];
        let n = sys::ZSTD_compress2(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            data.as_ptr().cast(),
            data.len(),
        );
        assert_eq!(sys::ZSTD_isError(n), 0);
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

#[test]
fn mt_decodes_libzstd_streams_byte_exact() {
    let pool = pool(4);
    for ds in datasets() {
        for &level in &LEVELS {
            for (kind, compressed) in [
                ("bulk", zstd_bulk(&ds.data, level)),
                ("stream", zstd_stream(&ds.data, level)),
            ] {
                let decoded = pool
                    .install(|| decode_mt(&compressed))
                    .unwrap_or_else(|e| panic!("{} L{} {}: {}", ds.name, level, kind, e));
                assert!(
                    decoded == ds.data,
                    "{} L{} {}: output differs",
                    ds.name,
                    level,
                    kind
                );
            }
        }
    }
}

/// Multi-job frames from this crate's encoder: every job boundary starts
/// fresh entropy tables, and later jobs match into earlier jobs' output.
#[test]
fn mt_decodes_multi_job_frames_from_our_encoder() {
    let pool = pool(3);
    for ds in datasets() {
        let data = &ds.data[..ds.data.len().min(2 * MIB)];
        for &level in &LEVELS {
            let opts = rust_zstd::CompressOptions {
                level,
                job_size: Some(512 * 1024),
                ..rust_zstd::CompressOptions::default()
            };
            let compressed = rust_zstd::compress_with(data, &opts);
            let decoded = pool
                .install(|| decode_mt(&compressed))
                .unwrap_or_else(|e| panic!("{} L{}: {}", ds.name, level, e));
            assert!(decoded == data, "{} L{}: output differs", ds.name, level);
        }
    }
}

/// Frames of one to a few blocks, which `decompress` leaves to the serial
/// path, forced through the multi-threaded one; also a pool of one thread.
#[test]
fn mt_path_forced_on_small_frames() {
    let text = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
    let mut mixed = lcg_bytes(3000, 5);
    mixed.extend_from_slice(&[0u8; 5000]);
    mixed.extend_from_slice(&text);
    let cases = [
        ("one block", zstd_bulk(&text, 3), text.clone()),
        ("empty", zstd_bulk(&[], 3), Vec::new()),
        (
            "small blocks",
            zstd_small_blocks(&mixed, 3, 1024),
            mixed.clone(),
        ),
        (
            "small blocks L19",
            zstd_small_blocks(&text, 19, 1024),
            text.clone(),
        ),
    ];
    for threads in [1, 2, 8] {
        let pool = pool(threads);
        for (name, compressed, want) in &cases {
            let got = pool.install(|| decode_mt(compressed)).unwrap();
            assert!(
                &got == want,
                "{} ({} threads): output differs",
                name,
                threads
            );
        }
    }
    // Concatenated frames keep separate offset histories and tables.
    let mut two = cases[2].1.clone();
    two.extend_from_slice(&cases[3].1);
    let mut want = mixed;
    want.extend_from_slice(&text);
    assert!(pool(2).install(|| decode_mt(&two)).unwrap() == want);
}

fn same_outcome(name: &str, input: &[u8]) {
    let serial = std::panic::catch_unwind(|| decode_serial(input))
        .unwrap_or_else(|_| panic!("{}: serial path panicked", name));
    let mt = std::panic::catch_unwind(|| decode_mt(input))
        .unwrap_or_else(|_| panic!("{}: multi-threaded path panicked", name));
    match (&serial, &mt) {
        (Ok(a), Ok(b)) => assert!(a == b, "{}: outputs differ", name),
        (Err(_), Err(_)) => {}
        _ => panic!(
            "{}: serial {:?} vs multi-threaded {:?}",
            name,
            serial.as_ref().map(Vec::len),
            mt.as_ref().map(Vec::len)
        ),
    }
}

fn small_block_inputs() -> Vec<(String, Vec<u8>)> {
    let mut data = b"The quick brown fox jumps over the lazy dog. ".repeat(60);
    data.extend_from_slice(&lcg_bytes(1500, 11));
    data.extend_from_slice(&b"abcabcabd".repeat(200));
    let mut out = Vec::new();
    for &level in &[1, 3, 19] {
        let c = zstd_small_blocks(&data, level, 1024);
        assert_eq!(decode_mt(&c).unwrap(), data);
        out.push((format!("L{}", level), c));
    }
    out
}

/// Truncation at every offset of multi-block frames: `Err` (never a panic)
/// through the multi-threaded path exactly when the serial path errs.
#[test]
fn mt_truncation_matches_serial_outcome() {
    pool(4).install(|| {
        for (name, c) in small_block_inputs() {
            for cut in 0..c.len() {
                same_outcome(&format!("{} cut {}", name, cut), &c[..cut]);
                if cut > 0 {
                    assert!(decode_mt(&c[..cut]).is_err(), "{} cut {}", name, cut);
                }
            }
        }
    });
}

/// Byte corruptions of multi-block frames: same outcome as the serial path.
#[test]
fn mt_corruption_matches_serial_outcome() {
    pool(4).install(|| {
        for (name, c) in small_block_inputs() {
            for pos in 0..c.len() {
                for flip in [0x01u8, 0x80, 0xFF, 0x55] {
                    let mut bad = c.clone();
                    bad[pos] ^= flip;
                    same_outcome(&format!("{} byte {} ^ {:#x}", name, pos, flip), &bad);
                }
            }
        }
    });
}
