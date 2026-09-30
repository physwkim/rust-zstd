//! A reused `Compressor` after a frame whose last job is under 7 bytes.
//! Such a job's only block is too small to compress, and before, only a
//! compressed block moved the window's end: the window still ended where
//! the job's prefix began while the tables held the prefix's indices, so
//! the next input on that context (`ZSTD_window_clear`) took them for live
//! entries. Which job inherits a context depends on the rayon thread
//! count, and so did the frames; at level 16 the binary tree broke
//! (`match_index < curr` in bt.rs under debug assertions). A reused
//! `Compressor` must write the frame a fresh one writes, which is
//! libzstd's with `nbWorkers` at the same job size.

mod common;

use common::c_compress2;
use rust_zstd::compress::{CompressOptions, Compressor};
use zstd::zstd_safe::zstd_sys as sys;

/// `ZSTDMT_JOBSIZE_MIN`: an input one byte longer is two jobs.
const JOB: usize = 512 << 10;

/// Text of period 1..200 over a random alphabet with one byte in 30
/// replaced by noise.
fn text(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed;
    let mut below = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let period: Vec<u8> = {
        let p = 1 + below(200) as usize;
        let alphabet = 2 + below(254);
        (0..p).map(|_| below(alphabet) as u8).collect()
    };
    (0..len)
        .map(|i| match below(30) {
            0 => below(256) as u8,
            _ => period[i % period.len()],
        })
        .collect()
}

fn opts(level: i32) -> CompressOptions {
    CompressOptions {
        level,
        job_size: Some(JOB),
        ..CompressOptions::default()
    }
}

/// libzstd's frame with two workers at job size [`JOB`].
fn libzstd(data: &[u8], level: i32) -> Vec<u8> {
    use sys::ZSTD_cParameter::{ZSTD_c_compressionLevel, ZSTD_c_jobSize, ZSTD_c_nbWorkers};
    c_compress2(
        data,
        &[
            (ZSTD_c_compressionLevel, level),
            (ZSTD_c_nbWorkers, 2),
            (ZSTD_c_jobSize, JOB as i32),
        ],
    )
}

/// The frame of `second` from a `Compressor` that wrote `first` before.
fn reused(level: i32, first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut cx = Compressor::new(opts(level));
    cx.compress_to_vec(first);
    cx.compress_to_vec(second)
}

/// [`reused`] on one thread, where every job gets the context the job
/// before it gave back: the next frame's first job gets the one the last
/// job left.
fn reused_on_one_thread(level: i32, first: &[u8], second: &[u8]) -> Vec<u8> {
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| reused(level, first, second))
    }
    #[cfg(not(feature = "parallel"))]
    reused(level, first, second)
}

/// A last job of 1..=6 bytes, and 7 (its block compressed): the next
/// frame, three jobs with a 3-byte last one, is the fresh frame.
#[test]
fn frame_after_tiny_last_job_equals_fresh_frame() {
    let second = text(5, 2 * JOB + 3);
    let fresh = Compressor::new(opts(1)).compress_to_vec(&second);
    assert!(
        fresh == libzstd(&second, 1),
        "fresh frame differs from libzstd"
    );
    for tail in 1..=7 {
        let got = reused_on_one_thread(1, &text(1007, JOB + tail), &second);
        assert!(
            got == fresh,
            "last job of {tail} bytes: {} bytes, fresh {}",
            got.len(),
            fresh.len()
        );
    }
}

/// The same at 1, 2 and 8 rayon threads, a few times each: which job's
/// context the next frame's jobs inherit varies with the thread count
/// and the timing.
#[cfg(feature = "parallel")]
#[test]
fn frame_after_tiny_last_job_ignores_thread_count() {
    let second = text(5, 2 * JOB + 3);
    let fresh = Compressor::new(opts(1)).compress_to_vec(&second);
    for threads in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for tail in 1..=6 {
            let first = text(1007, JOB + tail);
            for trial in 0..3 {
                let got = pool.install(|| reused(1, &first, &second));
                assert!(
                    got == fresh,
                    "{threads} threads, last job of {tail} bytes, trial {trial}: {} bytes, fresh {}",
                    got.len(),
                    fresh.len()
                );
            }
        }
    }
}

/// Level 16 (btopt): the next frame's binary tree must not link the
/// previous prefix's nodes, whose indices lay above the old window end.
#[test]
fn binary_tree_after_tiny_last_job_equals_fresh_frame() {
    let second = text(4, 2 * JOB);
    let fresh = Compressor::new(opts(16)).compress_to_vec(&second);
    assert!(
        fresh == libzstd(&second, 16),
        "fresh frame differs from libzstd"
    );
    let got = reused_on_one_thread(16, &text(3, JOB + 6), &second);
    assert!(got == fresh, "{} bytes, fresh {}", got.len(), fresh.len());
}
