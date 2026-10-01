//! Window overflow correction against libzstd 1.5.7 built with
//! `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY` set to 1: with
//! `CompressOptions::overflow_correct_frequently`, every case of
//! `tests/data/overflow_frequent.txt` writes that frame, corrects a window
//! at least once (the long distance matcher's when the case enables it), and
//! decodes through our decoder. The fixture comes from
//! `tests/data/overflow_frequent.c`, whose header has the build command.
//!
//! A single frame's window shrinks to its input, so each input exceeds
//! its level's window by a correction's threshold (`maxDist` plus a chain
//! cycle): 20 MiB up to level 19, as one job and in explicit jobs; 56, 104
//! and 200 MiB at levels 20, 21 and 22 (which enables long distance
//! matching there); 136 MiB with long distance matching enabled, which
//! raises the window log to 27.
//!
//! `input_over_4_gib_matches_libzstd` compresses 4.5 GiB, which stock
//! libzstd corrects once its indices pass 3500 MiB, at levels 1 and 3 and
//! level 3 with long distance matching.
//!
//! `window_low_follows_libzstd_low_limit` (not ignored) checks the match
//! state's window, `lowLimit` included, block by block against
//! `tests/data/window_low_frequent.txt`.
//!
//! Ignored by default; release build (the 4.5 GiB test needs about 12 GB
//! of memory):
//!
//! ```text
//! cargo test --release --test overflow_correction -- --ignored
//! ```

mod common;

use rust_zstd::compress::block::{build_seq_store, BlockLdm, BlockState};
use rust_zstd::compress::matchstate::MatchState;
use rust_zstd::compress::{CParams, CompressOptions, Compressor, ParamSwitch, SeqStore};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// `CASES` of `overflow_frequent.c`: level, long distance matching
/// enabled (else `Auto`), job size in MiB (0: one job), input MiB.
const CASES: [(i32, bool, usize, usize); 35] = [
    (-5, false, 0, 20),
    (-1, false, 0, 20),
    (1, false, 0, 20),
    (2, false, 0, 20),
    (3, false, 0, 20),
    (4, false, 0, 20),
    (5, false, 0, 20),
    (6, false, 0, 20),
    (7, false, 0, 20),
    (8, false, 0, 20),
    (9, false, 0, 20),
    (10, false, 0, 20),
    (11, false, 0, 20),
    (12, false, 0, 20),
    (13, false, 0, 20),
    (14, false, 0, 20),
    (15, false, 0, 20),
    (16, false, 0, 20),
    (17, false, 0, 20),
    (18, false, 0, 20),
    (19, false, 0, 20),
    (1, false, 4, 20),
    (3, false, 4, 20),
    (9, false, 16, 20),
    (16, false, 16, 20),
    (19, false, 18, 20),
    (20, false, 0, 56),
    (21, false, 0, 104),
    (22, false, 0, 200),
    (1, true, 0, 136),
    (3, true, 0, 136),
    (9, true, 0, 136),
    (16, true, 0, 136),
    (19, true, 0, 136),
    (3, true, 8, 136),
];

type Case = (i32, bool, usize, usize);

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

/// `gen` of `overflow_frequent.c`: noise literals, zero runs and copies
/// from 1..256, 1..64 Ki and 1..len bytes back or at the last distance.
/// A shorter input is a prefix of a longer one.
fn input(len: usize) -> Vec<u8> {
    let mut r = Lcg(1);
    let mut out = Vec::with_capacity(len + 512);
    let mut last = 1usize;
    while out.len() < len {
        let n = out.len();
        let k = r.below(8);
        if k < 3 || n < 64 {
            for _ in 0..1 + r.below(32) {
                out.push(r.next() as u8);
            }
        } else if k == 3 {
            out.resize(n + 1 + r.below(64) as usize, 0);
        } else {
            let dist = if k == 7 {
                last
            } else {
                let max = match k {
                    4 => 256,
                    5 => 65536,
                    _ => n,
                };
                1 + r.below(max.min(n) as u64) as usize
            };
            last = dist;
            let m = if k == 6 {
                16 + r.below(240)
            } else {
                4 + r.below(60)
            };
            let start = n - dist;
            for i in 0..m as usize {
                out.push(out[start + i]);
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

/// The fixture's first four columns: `L<level> auto|ldm def|<job>M <n>M`.
fn key(&(level, ldm, job, mib): &Case) -> String {
    let ldm = if ldm { "ldm" } else { "auto" };
    let job = if job == 0 {
        "def".to_string()
    } else {
        format!("{job}M")
    };
    format!("L{level} {ldm} {job} {mib}M")
}

/// Key -> `frame_len frame_fnv input_fnv`.
fn fixture() -> BTreeMap<String, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/overflow_frequent.txt");
    let text = std::fs::read_to_string(&path).expect("tests/data/overflow_frequent.txt");
    text.lines()
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            (f[..4].join(" "), f[4..].join(" "))
        })
        .collect()
}

fn check(cases: &[Case]) {
    let want = fixture();
    let data = input(cases.iter().map(|c| c.3 << 20).max().unwrap());
    let mut diffs = Vec::new();
    for case in cases {
        let &(level, ldm, job, mib) = case;
        let src = &data[..mib << 20];
        let mut cx = Compressor::new(CompressOptions {
            level,
            job_size: (job != 0).then_some(job << 20),
            ldm: if ldm {
                ParamSwitch::Enable
            } else {
                ParamSwitch::Auto
            },
            overflow_correct_frequently: true,
            ..CompressOptions::default()
        });
        let frame = cx.compress_to_vec(src);
        let (ms, lds) = cx.overflow_corrections();
        let name = key(case);
        let got = format!("{} {:016x} {:016x}", frame.len(), fnv64(&frame), fnv64(src));
        eprintln!("{name}: {got}, corrections {ms} + {lds} (long distance)");
        if want.get(&name) != Some(&got) {
            diffs.push(format!("{name}: {:?} -> {got}", want.get(&name)));
        }
        assert!(ms + lds > 0, "{name}: no correction");
        assert!(!ldm || lds > 0, "{name}: no long distance correction");
        assert!(rust_zstd::decompress(&frame).unwrap() == src, "{name}");
    }
    assert!(
        diffs.is_empty(),
        "differ from libzstd:\n{}",
        diffs.join("\n")
    );
}

/// The fixture has exactly the rows of `CASES`.
#[test]
fn fixture_lists_every_case() {
    let keys: Vec<String> = CASES.iter().map(key).collect();
    let fixture: Vec<String> = fixture().into_keys().collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(fixture, sorted);
}

#[test]
#[ignore]
fn frequent_correction_matches_libzstd_20_mib() {
    let cases: Vec<Case> = CASES.into_iter().filter(|c| c.3 == 20).collect();
    check(&cases);
}

#[test]
#[ignore]
fn frequent_correction_matches_libzstd_long_windows() {
    let cases: Vec<Case> = CASES.into_iter().filter(|c| c.3 > 20).collect();
    check(&cases);
}

/// Stock libzstd's frames (single-threaded `ZSTD_compress2`) of an input
/// past `ZSTD_CURRENT_MAX`, which our decoder reads back. A 4.5 GiB input
/// does not fit a 32-bit address space.
#[cfg(target_pointer_width = "64")]
#[test]
#[ignore]
fn input_over_4_gib_matches_libzstd() {
    use sys::ZSTD_cParameter::{ZSTD_c_compressionLevel, ZSTD_c_enableLongDistanceMatching};
    use zstd::zstd_safe::zstd_sys as sys;

    let data = input(4608 << 20);
    for (level, ldm) in [(1, false), (3, false), (3, true)] {
        let mut cx = Compressor::new(CompressOptions {
            level,
            ldm: if ldm {
                ParamSwitch::Enable
            } else {
                ParamSwitch::Auto
            },
            ..CompressOptions::default()
        });
        let frame = cx.compress_to_vec(&data);
        let (ms, lds) = cx.overflow_corrections();
        let name = format!("L{level} ldm {ldm}");
        eprintln!("{name}: {} bytes, corrections {ms} + {lds}", frame.len());
        assert_eq!((ms, lds), (1, ldm as u32), "{name}: corrections");
        let mut params = vec![(ZSTD_c_compressionLevel, level)];
        if ldm {
            params.push((ZSTD_c_enableLongDistanceMatching, 1));
        }
        assert!(common::c_compress2(&data, &params) == frame, "{name}");
        assert!(rust_zstd::decompress(&frame).unwrap() == data, "{name}");
    }
}

/// One reused context's window corrections after each of 40 frames equal
/// libzstd's (`tests/data/overflow_frequent_reuse.c`): a 128 KiB block and
/// a last one of 3 bytes, which is too small to compress but still gets
/// its overflow check, or of 7.
#[test]
fn frequent_correction_counts_on_reused_context() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/overflow_frequent_reuse.txt");
    let text = std::fs::read_to_string(&path).expect("tests/data/overflow_frequent_reuse.txt");
    let mut rows = 0;
    for line in text.lines() {
        let (case, want) = line.split_once(": ").unwrap();
        let (level, len) = case[1..].split_once(' ').unwrap();
        let src = input(len.parse().unwrap());
        let mut cx = Compressor::new(CompressOptions {
            level: level.parse().unwrap(),
            overflow_correct_frequently: true,
            ..CompressOptions::default()
        });
        let got: Vec<String> = want
            .split(' ')
            .map(|_| {
                cx.compress_to_vec(&src);
                cx.overflow_corrections().0.to_string()
            })
            .collect();
        assert_eq!(got.join(" "), want, "{case}");
        rows += 1;
    }
    assert_eq!(rows, 4);
}

/// The match state's window on every block equals libzstd's
/// (`tests/data/window_low_frequent.c`) as `ZSTD_compress_frameChunk`
/// leaves it: the index of the block's first byte, `nbOverflowCorrections`,
/// and `lowLimit`, which `ZSTD_window_enforceMaxDist` raises to the window
/// size below the block and `ZSTD_window_correctOverflow` moves down with
/// the indices. Inputs of 20 MiB, past each level's window, with frequent
/// corrections; the blocks libzstd compressed enter one match state in
/// turn, frames after the first on the reset (reused) state. Each frame's
/// first block is also compressed, where btultra2's `ZSTD_initStats_ultra`
/// moves the window past it.
#[test]
fn window_low_follows_libzstd_low_limit() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/window_low_frequent.txt");
    let text = std::fs::read_to_string(&path).expect("tests/data/window_low_frequent.txt");
    let fields = |line: &str| -> Vec<usize> {
        line.split(' ')
            .map(|f| f.parse().unwrap_or_else(|_| panic!("row {line:?}")))
            .collect()
    };
    let mut lines = text.lines().peekable();
    let mut cases = 0;
    while let Some(line) = lines.next() {
        let case = line.strip_prefix("case ").expect("case row");
        let (level, rest) = case.split_once(' ').unwrap();
        let level: i32 = level.parse().unwrap();
        let f = fields(rest);
        let (len, frames) = (f[0], f[1]);
        let cp = CParams::for_level(level, len);
        let applied = (
            cp.window_log as usize,
            cp.chain_log as usize,
            cp.strategy as usize,
        );
        assert_eq!(applied, (f[2], f[3], f[4]), "L{level}: parameters");
        let data = input(len);
        let mut store = SeqStore::new();
        let mut ms = MatchState::new(cp, 0);
        ms.set_correct_frequently(true);
        for frame in 0..frames {
            if frame > 0 {
                ms.reset(cp, 0);
            }
            let mut end = 0;
            while let Some(row) = lines.next_if(|l| !l.starts_with("frame ")) {
                let b = fields(row);
                let (start, size) = (b[0], b[1]);
                assert_eq!(start, end, "L{level} frame {frame}: blocks not contiguous");
                let entered = ms.enter_block(start..start + size);
                let got = (
                    ms.index(start),
                    ms.window_low(),
                    ms.window().nb_overflow_corrections() as usize,
                );
                let want = (b[2], b[3], b[4]);
                assert_eq!(got, want, "L{level} frame {frame}: block {row}");
                if start == 0 {
                    let rep = BlockState::initial().rep;
                    build_seq_store(&mut ms, &data, entered, rep, &mut store, &mut BlockLdm::Off);
                }
                end = start + size;
            }
            assert_eq!(end, len, "L{level} frame {frame}: blocks");
            lines.next().expect("frame row");
        }
        cases += 1;
    }
    assert_eq!(cases, 5);
}
