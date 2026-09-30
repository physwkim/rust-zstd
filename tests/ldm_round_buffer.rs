//! Multi-job long distance matching on an input several times ZSTDMT's
//! round buffer, against libzstd 1.5.7 with 2 workers at the same job
//! size and overlap. ZSTDMT keeps its input in a round buffer of
//! `max(window, jobSize * nbWorkers) + jobSize * (2 + (overlap > 0))`
//! bytes. On each wrap, `ZSTD_window_update` makes the serial long
//! distance window an extDict and raises its low limit over the bytes the
//! next job overwrites. Our jobs read one contiguous input, so the frames
//! match only if no overwritten byte is still inside the window. None is:
//! the buffer wraps only once less than a job fits, when it holds more
//! than a window plus a job and its overlap, so the next job and the
//! overlap copied before it end below the window.
//!
//! Ignored by default; release build:
//!
//! ```text
//! cargo test --release --test ldm_round_buffer -- --ignored
//! ```

mod common;

use rust_zstd::compress::{compress_with, CompressOptions, ParamSwitch};
use sys::ZSTD_cParameter::{
    ZSTD_c_compressionLevel, ZSTD_c_enableLongDistanceMatching, ZSTD_c_jobSize, ZSTD_c_nbWorkers,
    ZSTD_c_overlapLog,
};
use zstd::zstd_safe::zstd_sys as sys;

/// `ZSTD_LDM_DEFAULT_WINDOW_LOG`: enabling long distance matching raises
/// every level's window to 128 MiB.
const WINDOW: usize = 1 << 27;

/// Level, job size in MiB and `overlap_log`: level 1 at ZSTDMT's default
/// long distance job size, level 3 with a full overlap, and level 16
/// (`btopt`) at its default job size. Their round buffers hold 134, 152
/// and 176 MiB, and `input` is 512 MiB.
const CASES: [(i32, usize, u8); 3] = [(1, 2, 0), (3, 8, 9), (16, 16, 0)];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Runs of 16-letter noise shuffled with copies of 64 bytes to 64 KiB,
/// from anywhere back, from up to 32 MiB inside the window's lower edge
/// (where a low limit raised over overwritten bytes drops matches), and
/// from up to 1 MiB past it.
fn input(len: usize) -> Vec<u8> {
    let mut r = Lcg(1);
    let mut out = Vec::with_capacity(len + (64 << 10));
    while out.len() < len {
        let n = out.len();
        let k = r.below(4);
        if k == 0 || n < 1 << 16 {
            for _ in 0..1 + r.below(4096) {
                out.push(b'a' + (r.next() & 15) as u8);
            }
            continue;
        }
        let dist = match k {
            1 => 1 + r.below(n),
            2 => WINDOW - r.below(32 << 20),
            _ => WINDOW + r.below(1 << 20),
        }
        .min(n);
        let start = n - dist;
        let m = (64 + r.below((64 << 10) - 64)).min(dist);
        out.extend_from_within(start..start + m);
    }
    out.truncate(len);
    out
}

#[test]
#[ignore]
fn jobs_past_the_round_buffer_match_libzstd() {
    let data = input(512 << 20);
    let mut differ = Vec::new();
    for (level, job, overlap_log) in CASES {
        let name = format!("L{level} job {job} MiB overlap_log {overlap_log}");
        let frame = compress_with(
            &data,
            &CompressOptions {
                level,
                job_size: Some(job << 20),
                overlap_log,
                ldm: ParamSwitch::Enable,
                ..CompressOptions::default()
            },
        );
        let c = common::c_compress2(
            &data,
            &[
                (ZSTD_c_compressionLevel, level),
                (ZSTD_c_enableLongDistanceMatching, 1),
                (ZSTD_c_nbWorkers, 2),
                (ZSTD_c_jobSize, (job << 20) as i32),
                (ZSTD_c_overlapLog, overlap_log as i32),
            ],
        );
        eprintln!("{name}: {} bytes, libzstd {}", frame.len(), c.len());
        if frame != c {
            differ.push(name.clone());
        }
        assert!(rust_zstd::decompress(&frame).unwrap() == data, "{name}");
    }
    assert!(differ.is_empty(), "differ from libzstd: {differ:?}");
}
