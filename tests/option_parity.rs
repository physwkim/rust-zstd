//! Option values that libzstd clamps or reads as a sentinel, against
//! libzstd 1.5.7 given the same value through `ZSTD_CCtx_setParameter`:
//! the frames must be byte-identical.

mod common;

use rust_zstd::{compress_with, decompress, CompressOptions};
use sys::ZSTD_cParameter::{
    ZSTD_c_compressionLevel, ZSTD_c_jobSize, ZSTD_c_nbWorkers, ZSTD_c_overlapLog,
};
use zstd::zstd_safe::zstd_sys as sys;

/// `[0xfe, 0xff]`, then runs of 16-letter noise shuffled with 64-byte to
/// 4 KiB copies from up to 1 MiB back, so that matches cross every job
/// boundary within any overlap.
fn input(len: usize) -> Vec<u8> {
    let noise = common::lcg_bytes(len, 1);
    let mut noise = noise.iter().map(|b| b'a' + (b & 15));
    let mut state = 7u64;
    let mut below = |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % n
    };
    let mut v = vec![0xfe, 0xff];
    while v.len() < len {
        if v.len() < 1 << 16 || below(2) == 0 {
            v.extend(noise.by_ref().take(1 + below(512)));
            continue;
        }
        let dist = 1 + below(v.len().min(1 << 20));
        let m = (64 + below(4032)).min(dist);
        let start = v.len() - dist;
        v.extend_from_within(start..start + m);
    }
    v.truncate(len);
    v
}

/// `ZSTD_c_overlapLog` above `ZSTD_OVERLAPLOG_MAX` is clamped to it.
#[test]
fn overlap_log_above_max_matches_libzstd() {
    let data = input(3 << 20);
    for level in [1, 3, 7] {
        let frame = |overlap_log| {
            compress_with(
                &data,
                &CompressOptions {
                    level,
                    job_size: Some(1 << 20),
                    overlap_log,
                    ..Default::default()
                },
            )
        };
        assert!(frame(9) != frame(0), "L{level}: overlap has no effect");
        for overlap_log in [10, u8::MAX] {
            let ours = frame(overlap_log);
            let c = common::c_compress2(
                &data,
                &[
                    (ZSTD_c_compressionLevel, level),
                    (ZSTD_c_nbWorkers, 2),
                    (ZSTD_c_jobSize, 1 << 20),
                    (ZSTD_c_overlapLog, overlap_log as i32),
                ],
            );
            let name = format!("L{level} overlap_log {overlap_log}");
            assert!(decompress(&ours).unwrap() == data, "{name}: roundtrip");
            assert!(ours == c, "{name}: frame differs from libzstd");
        }
    }
}
