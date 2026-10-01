//! A long distance matching hash rate log above the adjusted window log.
//! libzstd 1.5.7 derives the hash log as `windowLog - hashRateLog` in U32
//! (`ZSTD_ldm_adjustParameters`), which wraps and clamps to 30: it zeroes
//! a 2^30-entry (8 GiB) table to compress 1 KiB. This port derives
//! ZSTD_HASHLOG_MIN instead; at every rate up to the window log the
//! frames pass the encoder gate against libzstd's. One test, so that nothing else allocates while
//! it measures.

mod common;

use rust_zstd::constants::{ZSTD_HASHLOG_MIN, ZSTD_WINDOWLOG_MAX};
use rust_zstd::{compress_with, CompressOptions, ParamSwitch};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use sys::ZSTD_cParameter::{
    ZSTD_c_compressionLevel, ZSTD_c_enableLongDistanceMatching, ZSTD_c_ldmHashRateLog,
};
use zstd::zstd_safe::zstd_sys as sys;

/// The system allocator, recording the largest size it is asked for.
struct Largest;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System` unchanged.
unsafe impl GlobalAlloc for Largest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Largest = Largest;

/// Far above what compressing 1 KiB at level 3 needs, far below the
/// 8 GiB table of hash log 30.
const SMALL: usize = 1 << 20;

const LEVEL: i32 = 3;

/// The window log of 1 KiB: `ZSTD_c_enableLongDistanceMatching` raises it
/// to 27, then `ZSTD_adjustCParams_internal` shrinks it to the input's.
const WINDOW_LOG: u32 = 10;

/// `ZSTD_LDM_HASHRATELOG_MAX`: 25, or 24 where `usize` is 32 bits.
const RATE_MAX: u32 = ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN;

/// 1 KiB: 400 pseudo-random letters, the same 400 again, then 224 more,
/// so that long distance matching has a repeat to find.
fn input() -> Vec<u8> {
    let letters: Vec<u8> = common::lcg_bytes(624, 1)
        .iter()
        .map(|b| b'a' + (b & 15))
        .collect();
    let mut v = letters[..400].to_vec();
    v.extend_from_slice(&letters[..400]);
    v.extend_from_slice(&letters[400..]);
    assert_eq!(v.len(), 1 << 10);
    v
}

fn ours(data: &[u8], hash_rate_log: u32, job_size: Option<usize>) -> Vec<u8> {
    compress_with(
        data,
        &CompressOptions {
            level: LEVEL,
            job_size,
            ldm: ParamSwitch::Enable,
            ldm_hash_rate_log: hash_rate_log,
            ..Default::default()
        },
    )
}

fn theirs(data: &[u8], hash_rate_log: u32) -> Vec<u8> {
    common::c_compress2(
        data,
        &[
            (ZSTD_c_compressionLevel, LEVEL),
            (ZSTD_c_enableLongDistanceMatching, 1),
            (ZSTD_c_ldmHashRateLog, hash_rate_log as i32),
        ],
    )
}

#[test]
fn rate_above_window_log_keeps_the_ldm_table_small() {
    let data = input();
    for rate in [WINDOW_LOG + 1, RATE_MAX] {
        for job_size in [None, Some(1 << 20)] {
            LARGEST.store(0, Ordering::Relaxed);
            let frame = ours(&data, rate, job_size);
            let largest = LARGEST.load(Ordering::Relaxed);
            assert!(
                largest < SMALL,
                "rate {rate} job {job_size:?}: largest allocation {largest} bytes"
            );
            assert!(
                zstd::bulk::decompress(&frame, data.len()).unwrap() == data,
                "rate {rate} job {job_size:?}: libzstd decodes another input"
            );
        }
    }
    // 0 derives the rate (7 - strategy / 3); up to the window log the
    // difference does not wrap and the hash log is libzstd's.
    for rate in 0..=WINDOW_LOG {
        common::assert_gate(
            &format!("L{LEVEL} ldm_hash_rate_log {rate}"),
            &data,
            &ours(&data, rate, None),
            &theirs(&data, rate),
        );
    }
}
