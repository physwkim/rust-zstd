//! Blocks of 1..=16 bytes against libzstd 1.5.7 default frames: as a
//! frame's first block, after a RAW and after a COMPRESSED full block, and
//! as the first block of a later ZSTDMT job, in several shapes at
//! L1/L3/L7/L13. libzstd writes a block below 7 bytes RAW without the RLE
//! check (`ZSTD_buildSeqStore` returns `ZSTDbss_noCompress`), and never
//! writes a job's first block RLE (`isFirstBlock`, which the header flush of
//! a later ZSTDMT job leaves set).
//!
//! Every frame must be libzstd's byte for byte, except where one of two
//! known differences applies, each left out by the condition that produces
//! it: a row whose libzstd strategy this crate runs as another one (L13's
//! btlazy2/btopt/btultra rows, see `Strategy`), and a frame in which
//! libzstd's matches reach the positions our job-0 search skips (see
//! [`c_uses_job_start`]). The two rules above decide the block without the
//! match finder, so they are checked on those cases too.

mod common;

use common::{c_compress2, frame_blocks, lcg_bytes};
use rust_zstd::compress::CParams;
use rust_zstd::{compress_with, CompressOptions};
use std::ffi::{c_int, c_void};
use zstd::zstd_safe::zstd_sys as sys;

const LEVELS: [i32; 4] = [1, 3, 7, 13];
const JOB: usize = 512 << 10;

/// `ZSTD_compressionParameters`.
#[repr(C)]
struct CCParams {
    window_log: u32,
    chain_log: u32,
    hash_log: u32,
    search_log: u32,
    min_match: u32,
    target_length: u32,
    strategy: c_int,
}

/// `ZSTD_Sequence`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CSequence {
    offset: u32,
    lit_length: u32,
    match_length: u32,
    rep: u32,
}

extern "C" {
    fn ZSTD_getCParams(level: c_int, src_size_hint: u64, dict_size: usize) -> CCParams;
    fn ZSTD_generateSequences(
        zc: *mut sys::ZSTD_CCtx,
        out: *mut CSequence,
        out_capacity: usize,
        src: *const c_void,
        src_size: usize,
    ) -> usize;
}

/// libzstd's strategy for `len` bytes at `level`, as a `ZSTD_strategy`.
fn c_strategy(level: i32, len: usize) -> i32 {
    // SAFETY: a pure table lookup.
    unsafe { ZSTD_getCParams(level, len as u64, 0).strategy }
}

/// Does libzstd's single-threaded `level` frame of `data` start a match
/// before position 3 or match from before position 2? Our search of a
/// job's first block begins at position 2 where libzstd's begins at 1, and
/// never indexes `src[0]` (nor `src[1]` in the fast finders), see the
/// `fast` and `lazy` module docs. A block libzstd does not attempt to
/// compress has no sequences; `ZSTD_generateSequences` reports an error.
fn c_uses_job_start(data: &[u8], level: i32) -> bool {
    let mut seqs = vec![CSequence::default(); data.len() + 2];
    // SAFETY: the context is used only here; `seqs` bounds the output.
    let n = unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let r =
            sys::ZSTD_CCtx_setParameter(cctx, sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level);
        assert_eq!(sys::ZSTD_isError(r), 0);
        let n = ZSTD_generateSequences(
            cctx,
            seqs.as_mut_ptr(),
            seqs.len(),
            data.as_ptr().cast(),
            data.len(),
        );
        sys::ZSTD_freeCCtx(cctx);
        n
    };
    // SAFETY: a pure function of `n`.
    if unsafe { sys::ZSTD_isError(n) } != 0 {
        return false;
    }
    let mut pos = 0usize;
    seqs[..n].iter().any(|s| {
        pos += s.lit_length as usize;
        let near_start = s.match_length != 0 && (pos < 3 || pos - (s.offset as usize) < 2);
        pos += s.match_length as usize;
        near_start
    })
}

fn shapes(n: usize) -> [(&'static str, Vec<u8>); 4] {
    [
        ("same", vec![b'a'; n]),
        (
            "two",
            lcg_bytes(n, 7).iter().map(|b| b'a' + (b >> 7)).collect(),
        ),
        ("alternating", (0..n).map(|i| b"ab"[i % 2]).collect()),
        ("random", lcg_bytes(n, 11)),
    ]
}

/// Space-separated words drawn from a small vocabulary after the bytes 0xff
/// 0xfe: compresses, so the block after it follows a COMPRESSED block, and
/// its first two bytes never recur, so libzstd cannot match from the
/// positions [`c_uses_job_start`] names.
fn text(len: usize) -> Vec<u8> {
    const WORDS: [&str; 16] = [
        "block", "frame", "the", "of", "literal", "sequence", "match", "offset", "huffman",
        "table", "raw", "window", "entropy", "a", "and", "repeat",
    ];
    let mut out = Vec::with_capacity(len + 16);
    out.extend_from_slice(&[0xff, 0xfe]);
    for b in lcg_bytes(len, 5) {
        if out.len() >= len {
            break;
        }
        out.extend_from_slice(WORDS[(b >> 4) as usize].as_bytes());
        out.push(b' ');
    }
    out.truncate(len);
    out
}

#[test]
fn small_blocks_match_libzstd() {
    use sys::ZSTD_cParameter::{ZSTD_c_compressionLevel, ZSTD_c_jobSize, ZSTD_c_nbWorkers};
    let raw_prefix = lcg_bytes(128 << 10, 3);
    // Above 256 KiB so that L13 is libzstd's btlazy2 row.
    let text_prefix = text(3 << 17);
    let job_prefix = text(JOB);
    let mut bad = Vec::new();
    for level in LEVELS {
        let (mut compared, mut by_rule, mut other_strategy) = (0, 0, 0);
        let mut job_start = Vec::new();
        for n in 1..=16 {
            for (shape, tail) in shapes(n) {
                let cases: [(&str, &[u8], Option<usize>); 4] = [
                    ("first", &[], None),
                    ("after raw", &raw_prefix, None),
                    ("after compressed", &text_prefix, None),
                    ("job start", &job_prefix, Some(JOB)),
                ];
                for (position, prefix, job_size) in cases {
                    let case = format!("L{level} {n} {shape} {position}");
                    let data = [prefix, &tail[..]].concat();
                    let opts = CompressOptions {
                        level,
                        job_size,
                        ..Default::default()
                    };
                    let ours = compress_with(&data, &opts);
                    let mut params = vec![(ZSTD_c_compressionLevel, level)];
                    if let Some(job) = job_size {
                        params.extend([(ZSTD_c_nbWorkers, 2), (ZSTD_c_jobSize, job as i32)]);
                    }
                    let theirs = c_compress2(&data, &params);
                    let (our_blocks, decoded) = frame_blocks(&ours, data.len());
                    assert!(decoded == data, "{case}: libzstd decodes our frame wrong");
                    assert!(
                        rust_zstd::decompress(&ours).unwrap() == data,
                        "{case}: our decoder"
                    );
                    let (c_blocks, _) = frame_blocks(&theirs, data.len());
                    let (our_last, c_last) =
                        (*our_blocks.last().unwrap(), *c_blocks.last().unwrap());
                    assert_eq!(c_last.size, n, "{case}: libzstd's last block");
                    // Without the match finder: RAW below 7 bytes, RLE for a
                    // run that is not a job's first block.
                    let job_first = position == "first" || position == "job start";
                    let run = tail.iter().all(|&b| b == tail[0]);
                    let by_rule_here = n < 7 || (run && !job_first);
                    by_rule += by_rule_here as usize;
                    if by_rule_here && our_last != c_last {
                        bad.push(format!("{case}: ours {our_last:?} libzstd {c_last:?}"));
                        continue;
                    }
                    if CParams::for_level(level, data.len()).strategy as i32
                        != c_strategy(level, data.len())
                    {
                        other_strategy += 1;
                        continue;
                    }
                    if ours != theirs && c_uses_job_start(&data, level) {
                        job_start.push(case);
                        continue;
                    }
                    let c_head = &theirs[..theirs.len() - c_last.c_size];
                    assert!(ours.starts_with(c_head), "{case}: frame before the block");
                    compared += 1;
                    if ours != theirs {
                        bad.push(format!("{case}: ours {our_last:?} libzstd {c_last:?}"));
                    }
                }
            }
        }
        eprintln!(
            "L{level}: {compared} frames compared, {by_rule} blocks by rule, \
             {other_strategy} other strategy, {} job start: {job_start:?}",
            job_start.len()
        );
        if level != 13 {
            assert_eq!(other_strategy, 0, "L{level}");
            assert!(compared >= 3 * 16 * 4, "L{level}: compared only {compared}");
        }
    }
    assert!(bad.is_empty(), "{} differ:\n{}", bad.len(), bad.join("\n"));
}
