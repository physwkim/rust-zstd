//! Long distance matching frames against libzstd 1.5.7 with
//! `ZSTD_c_enableLongDistanceMatching` 1 and both block splitters off, on
//! an input above `ZSTDMT_JOBSIZE_MIN` full of long repeats. The default
//! options are single-threaded `ZSTD_compress2`, which generates each
//! block's long distance matches as it compresses the block; an explicit
//! job size is ZSTDMT, which generates each job's in job order. The input
//! starts with two bytes that never recur, so that job 0 not matching from
//! `src[0]` leaves the frames alike; fast and dfast (levels 1-4) also start
//! their search a byte later than libzstd and are left out.

mod common;

use rust_zstd::{compress_with, CompressOptions, ParamSwitch};
use sys::ZSTD_cParameter::{
    ZSTD_c_compressionLevel, ZSTD_c_enableLongDistanceMatching, ZSTD_c_experimentalParam13,
    ZSTD_c_experimentalParam20, ZSTD_c_jobSize, ZSTD_c_nbWorkers,
};
use zstd::zstd_safe::zstd_sys as sys;

const LEVELS: [i32; 5] = [7, 11, 16, 19, 22];
const JOB_SIZE: usize = 1 << 20;

/// `[0xfe, 0xff]`, 512 KiB of 16-letter text, then 300-byte copies from
/// pseudo-random places of that text, each followed by 200 new letters,
/// up to 1.25 MiB: long matches that start and end anywhere in a block.
fn input() -> Vec<u8> {
    let letters = common::lcg_bytes(1 << 20, 1);
    let mut letters = letters.iter().map(|b| b'a' + (b & 15));
    let mut v = vec![0xfe, 0xff];
    v.extend(letters.by_ref().take(512 << 10));
    let mut state = 7u64;
    while v.len() < 5 << 18 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let from = 2 + (state >> 33) as usize % ((512 << 10) - 300);
        v.extend_from_within(from..from + 300);
        v.extend(letters.by_ref().take(200));
    }
    v
}

fn ours(data: &[u8], level: i32, job_size: Option<usize>) -> Vec<u8> {
    compress_with(
        data,
        &CompressOptions {
            level,
            job_size,
            ldm: ParamSwitch::Enable,
            split_after_sequences: ParamSwitch::Disable,
            block_splitter_level: 1,
            ..Default::default()
        },
    )
}

fn theirs(data: &[u8], level: i32, job_size: Option<usize>) -> Vec<u8> {
    let mut params = vec![
        (ZSTD_c_compressionLevel, level),
        (ZSTD_c_enableLongDistanceMatching, 1),
        // ZSTD_c_splitAfterSequences 2: off.
        (ZSTD_c_experimentalParam13, 2),
        // ZSTD_c_blockSplitterLevel 1: no pre-splitting.
        (ZSTD_c_experimentalParam20, 1),
    ];
    if let Some(job_size) = job_size {
        params.extend([(ZSTD_c_nbWorkers, 2), (ZSTD_c_jobSize, job_size as i32)]);
    }
    common::c_compress2(data, &params)
}

fn check(job_size: Option<usize>) {
    let data = input();
    let mut differ = Vec::new();
    for level in LEVELS {
        let frame = ours(&data, level, job_size);
        assert!(
            rust_zstd::decompress(&frame).unwrap() == data,
            "L{level} job {job_size:?}: our decoder"
        );
        let c = theirs(&data, level, job_size);
        if frame != c {
            differ.push(format!("L{level}: {} vs libzstd {}", frame.len(), c.len()));
        }
    }
    assert!(differ.is_empty(), "job {job_size:?}: {differ:?}");
}

#[test]
fn single_threaded_ldm_frames_match_libzstd() {
    check(None);
}

#[test]
fn multithreaded_ldm_frames_match_libzstd() {
    check(Some(JOB_SIZE));
}
