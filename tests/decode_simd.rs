//! The decoder's SIMD levels against each other and against libzstd: the
//! detected level (32-byte AVX2 copies where available) and the portable
//! 16-byte fallback must give byte-identical output, and the same
//! `Err`-vs-`Ok` outcome on corrupted input.

mod common;

use common::{datasets, lcg_bytes, zstd_bulk, zstd_stream, LEVELS};
use rust_zstd::decode::{decompress_with_options, DecodeOptions};

fn decode(data: &[u8], simd: bool) -> Result<Vec<u8>, String> {
    decode_with(data, simd, usize::MAX)
}

/// `min_parallel_blocks: 1` sends every frame through the MT path (when
/// the `parallel` feature is on).
fn decode_with(data: &[u8], simd: bool, min_parallel_blocks: usize) -> Result<Vec<u8>, String> {
    decompress_with_options(
        data,
        &DecodeOptions {
            min_parallel_blocks,
            simd,
        },
    )
}

#[test]
fn fallback_and_detected_levels_decode_libzstd_streams_byte_exact_serial_and_mt() {
    for ds in datasets() {
        for &level in &LEVELS {
            for (kind, compressed) in [
                ("bulk", zstd_bulk(&ds.data, level)),
                ("stream", zstd_stream(&ds.data, level)),
            ] {
                for (simd, min_blocks) in [
                    (false, usize::MAX),
                    (true, usize::MAX),
                    (false, 1),
                    (true, 1),
                ] {
                    let got = decode_with(&compressed, simd, min_blocks).unwrap_or_else(|e| {
                        panic!(
                            "{} L{} {} simd={} min_blocks={}: {}",
                            ds.name, level, kind, simd, min_blocks, e
                        )
                    });
                    assert!(
                        got == ds.data,
                        "{} L{} {} simd={} min_blocks={}: output differs",
                        ds.name,
                        level,
                        kind,
                        simd,
                        min_blocks
                    );
                }
            }
        }
    }
}

/// Periodic data with every period from 1 to 80 bytes, so that matches
/// with offsets below 16, from 16 to 31 (16-byte chunks under the 32-byte
/// level) and from 32 up (32-byte chunks) all occur, at lengths both
/// shorter and longer than one chunk.
#[test]
fn every_short_offset_decodes_at_both_levels() {
    let mut data = Vec::new();
    let noise = lcg_bytes(80 * 200, 21);
    for period in 1..=80usize {
        let unit = &noise[period * 100..period * 100 + period];
        for rep in 0..(700 / period + 3) {
            data.extend_from_slice(unit);
            // Break the period now and then so match lengths vary.
            if rep % 5 == 4 {
                data.push(noise[rep % noise.len()]);
            }
        }
    }
    for &level in &LEVELS {
        let compressed = zstd_bulk(&data, level);
        for simd in [false, true] {
            assert!(
                decode(&compressed, simd).unwrap() == data,
                "L{} simd={}",
                level,
                simd
            );
        }
    }
}

/// Runs of random periods from 1 to 80, each opened by fresh bytes, so
/// that most matches take a new offset rather than a repeat code and many
/// blocks' offset tables are dominated by offsets below 29, which selects
/// the shuffled match copy under AVX2 and the straight-line one of the
/// portable level (from the first sequence of the frame on).
#[test]
fn random_short_periods_decode_at_both_levels() {
    let noise = lcg_bytes(1 << 20, 33);
    let mut data = Vec::new();
    let mut pos = 0;
    while data.len() < 256 * 1024 {
        let period = 1 + usize::from(noise[pos]) % 80;
        let len = period + 1 + usize::from(noise[pos + 1]) % 96;
        let unit = &noise[pos + 2..pos + 2 + period];
        data.extend((0..len).map(|i| unit[i % period]));
        pos += 2 + period;
    }
    for &level in &LEVELS {
        let compressed = zstd_bulk(&data, level);
        for simd in [false, true] {
            assert!(
                decode(&compressed, simd).unwrap() == data,
                "L{} simd={}",
                level,
                simd
            );
        }
    }
}

/// Byte corruptions: the two levels agree on `Err` vs `Ok`, and on the
/// output when both succeed.
#[test]
fn corruption_outcome_is_level_independent() {
    let mut data = b"The quick brown fox jumps over the lazy dog. ".repeat(30);
    data.extend_from_slice(&lcg_bytes(400, 11));
    data.extend_from_slice(&b"0123456789abcdefghijklmnopqrstu".repeat(20));
    for &level in &[1, 3, 19] {
        let compressed = zstd_bulk(&data, level);
        for pos in 0..compressed.len() {
            for flip in [0x01u8, 0x80, 0xFF, 0x55] {
                let mut bad = compressed.clone();
                bad[pos] ^= flip;
                let fallback = std::panic::catch_unwind(|| decode(&bad, false))
                    .unwrap_or_else(|_| panic!("L{} byte {} ^ {:#x}: panic", level, pos, flip));
                let detected = std::panic::catch_unwind(|| decode(&bad, true))
                    .unwrap_or_else(|_| panic!("L{} byte {} ^ {:#x}: panic", level, pos, flip));
                match (&fallback, &detected) {
                    (Ok(a), Ok(b)) => assert!(a == b, "L{} byte {}: outputs differ", level, pos),
                    (Err(_), Err(_)) => {}
                    _ => panic!("L{} byte {} ^ {:#x}: outcomes differ", level, pos, flip),
                }
            }
        }
    }
}
