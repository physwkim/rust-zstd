//! Encoder output identity gate: every frame of a fixed case grid is hashed
//! and compared with `tests/data/frame_identity.txt`. The grid covers
//! levels 1..=15 on prefixes at libzstd's table-row edges (`clevels.h`
//! picks its row by `(n <= 256 KiB) + (n <= 128 KiB) + (n <= 16 KiB)`), so
//! every strategy row, including the small-input btlazy2 rows, is reached,
//! plus a 4 MiB input as one job (default options) and in 512 KiB jobs
//! (several jobs with overlap), both also with a content checksum, those
//! frames gated against libzstd's (`ZSTD_c_checksumFlag` 1, and for the
//! jobs `ZSTD_c_nbWorkers` 2). Inputs are generated from fixed seeds, so the
//! gate needs no file outside the repository and both feature builds must
//! match the same file (serial and parallel agreement). Each frame is also
//! compressed on a reused `Compressor` and must equal the fresh frame, and
//! must decode through our decoder and libzstd.
//!
//! Ignored by default (release build recommended):
//!
//! ```text
//! cargo test --release --offline --test frame_identity -- --ignored
//! cargo test --release --offline --no-default-features --test frame_identity -- --ignored
//! ```
//!
//! On a mismatch the differing rows are printed with their strategy so an
//! intended change can be classified; after review, regenerate the file with
//! `ZSTD_BLESS=1` and commit it with the change that caused it.

mod common;

use rust_zstd::compress::{CParams, CompressOptions, Compressor};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

const EDGE_SIZES: [usize; 8] = [
    1000,
    16 << 10,
    (16 << 10) + 1,
    128 << 10,
    (128 << 10) + 1,
    256 << 10,
    (256 << 10) + 1,
    1 << 20,
];
const LARGE: usize = 4 << 20;
const LEVELS: std::ops::RangeInclusive<i32> = 1..=15;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Word text: a 2000-word vocabulary of random syllables drawn with a
/// skewed (roughly Zipf) distribution, with punctuation and line breaks.
fn text(len: usize, seed: u64) -> Vec<u8> {
    const SYL: [&str; 24] = [
        "ka", "to", "re", "mi", "su", "lo", "an", "de", "ri", "on", "ve", "ta", "sh", "el", "or",
        "in", "qu", "ba", "ne", "st", "ch", "um", "ex", "po",
    ];
    let mut r = Lcg(seed);
    let words: Vec<String> = (0..2000)
        .map(|_| {
            (0..1 + r.below(4))
                .map(|_| SYL[r.below(SYL.len() as u64) as usize])
                .collect()
        })
        .collect();
    let mut out = Vec::with_capacity(len + 16);
    while out.len() < len {
        // Product of two uniforms: small indices are much more frequent.
        let i = (r.below(2000) * r.below(2000) / 2000) as usize;
        out.extend_from_slice(words[i].as_bytes());
        out.push(match r.below(16) {
            0 => b'\n',
            1 => b',',
            2 => b'.',
            _ => b' ',
        });
    }
    out.truncate(len);
    out
}

/// Binary with LZ structure: copies of earlier data at short, medium and
/// long distances, repeat-offset runs, zero runs and noise literals.
fn binary(len: usize, seed: u64) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut out: Vec<u8> = Vec::with_capacity(len + 1024);
    let mut last_dist = 8usize;
    while out.len() < len {
        match r.below(8) {
            0..=2 => {
                for _ in 0..1 + r.below(24) {
                    out.push(r.next() as u8);
                }
            }
            3 => out.resize(out.len() + 1 + r.below(64) as usize, 0),
            _ if out.len() < 64 => out.push(r.next() as u8),
            k => {
                let max = match k {
                    4 => 256,
                    5 => 64 << 10,
                    _ => out.len(),
                };
                let dist = if k == 7 {
                    last_dist
                } else {
                    1 + r.below(max.min(out.len()) as u64) as usize
                };
                let dist = dist.min(out.len());
                last_dist = dist;
                let start = out.len() - dist;
                for i in 0..4 + r.below(60) as usize {
                    out.push(out[start + i]);
                }
            }
        }
    }
    out.truncate(len);
    out
}

/// FNV-1a 64.
fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/frame_identity.txt")
}

/// `input size level job` -> `strategy len hash`, one line per frame.
fn compute() -> BTreeMap<String, String> {
    let inputs = [("text", text(LARGE, 1)), ("binary", binary(LARGE, 2))];
    let mut rows = BTreeMap::new();
    for (name, data) in &inputs {
        for level in LEVELS {
            let mut cases: Vec<(usize, Option<usize>, bool)> =
                EDGE_SIZES.iter().map(|&n| (n, None, false)).collect();
            for checksum in [false, true] {
                cases.push((LARGE, None, checksum));
                cases.push((LARGE, Some(512 << 10), checksum));
            }
            for (n, job_size, checksum) in cases {
                let src = &data[..n];
                let opts = CompressOptions {
                    level,
                    checksum,
                    job_size,
                    ..CompressOptions::default()
                };
                let frame = rust_zstd::compress_with(src, &opts);
                let mut cx = Compressor::new(opts.clone());
                let _ = cx.compress_to_vec(&data[..1000]);
                assert!(
                    cx.compress_to_vec(src) == frame,
                    "{name} {n} L{level}: reused Compressor differs from compress_with"
                );
                assert!(
                    rust_zstd::decompress(&frame).unwrap() == src,
                    "{name} {n} L{level}: our decoder"
                );
                assert!(
                    zstd::bulk::decompress(&frame, n).unwrap() == src,
                    "{name} {n} L{level}: libzstd"
                );
                if checksum {
                    use zstd::zstd_safe::zstd_sys::ZSTD_cParameter::*;
                    let mut params =
                        vec![(ZSTD_c_compressionLevel, level), (ZSTD_c_checksumFlag, 1)];
                    if let Some(j) = job_size {
                        params.extend([(ZSTD_c_nbWorkers, 2), (ZSTD_c_jobSize, j as i32)]);
                    }
                    common::assert_gate(
                        &format!("{name} {n} L{level} checksum"),
                        src,
                        &frame,
                        &common::c_compress2(src, &params),
                    );
                }
                let strategy = CParams::for_level(level, n).strategy;
                let job = job_size.map_or("def".to_string(), |j| format!("{}K", j >> 10));
                let job = if checksum { job + "+c" } else { job };
                rows.insert(
                    format!("{name:<6} {n:>7} L{level:<2} {job:>4}"),
                    format!(
                        "{:<8} {:>8} {:016x}",
                        format!("{strategy:?}"),
                        frame.len(),
                        fnv64(&frame)
                    ),
                );
            }
        }
    }
    rows
}

/// `(input size level job, strategy len hash)` of one file line, with the
/// padding collapsed.
fn split_row(line: &str) -> (String, String) {
    let f: Vec<&str> = line.split_whitespace().collect();
    (f[..4].join(" "), f[4..].join(" "))
}

#[test]
#[ignore]
fn frame_identity() {
    let rows = compute();
    let mut text_out = String::new();
    for (k, v) in &rows {
        writeln!(text_out, "{k} {v}").unwrap();
    }
    if std::env::var_os("ZSTD_BLESS").is_some() {
        std::fs::write(golden_path(), text_out).unwrap();
        eprintln!("wrote {} rows to {}", rows.len(), golden_path().display());
        return;
    }
    let golden = std::fs::read_to_string(golden_path()).expect("golden file; bless first");
    let want: BTreeMap<String, String> = golden.lines().map(split_row).collect();
    let rows: BTreeMap<String, String> = text_out.lines().map(split_row).collect();
    let mut diffs = Vec::new();
    for (k, v) in &rows {
        match want.get(k) {
            Some(w) if w == v => {}
            Some(w) => diffs.push(format!("{k}: {w} -> {v}")),
            None => diffs.push(format!("{k}: missing -> {v}")),
        }
    }
    for k in want.keys().filter(|k| !rows.contains_key(*k)) {
        diffs.push(format!("{k}: no longer computed"));
    }
    assert!(
        diffs.is_empty(),
        "{} of {} frames differ from {}:\n{}",
        diffs.len(),
        rows.len(),
        golden_path().display(),
        diffs.join("\n")
    );
}
