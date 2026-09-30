//! Frame-level checks of ZSTD_decompressFrame / ZSTD_decompressMultiFrame:
//! libzstd 1.5.7 frames, whole or altered, get the verdict of its one-shot
//! (`zstd::bulk`) and streaming (`zstd::stream`) decoders, and their bytes
//! when accepted, serial and MT at both SIMD levels.

use rust_zstd::decode::{decompress_with_options, DecodeOptions};

/// Output capacity given to libzstd's one-shot decoder, above every
/// frame's content here.
const CAPACITY: usize = 8 << 20;

/// Check that libzstd's one-shot and streaming decoders both accept `f`
/// when `accept` and both reject it otherwise, and that ours gives the same
/// outcome and bytes.
fn check(name: &str, f: &[u8], accept: bool) {
    let theirs = zstd::bulk::decompress(f, CAPACITY);
    assert_eq!(
        theirs.is_ok(),
        accept,
        "{name}: libzstd one-shot {:?}",
        theirs.as_ref().map(Vec::len)
    );
    let stream = zstd::stream::decode_all(f);
    assert_eq!(
        stream.is_ok(),
        accept,
        "{name}: libzstd streaming {:?}",
        stream.as_ref().map(Vec::len)
    );
    for simd in [false, true] {
        for min_parallel_blocks in [usize::MAX, 1] {
            let options = DecodeOptions {
                min_parallel_blocks,
                simd,
            };
            let ours = decompress_with_options(f, &options);
            match (&theirs, &ours) {
                (Ok(a), Ok(b)) => assert!(
                    a == b,
                    "{name} simd={simd} min_parallel_blocks={min_parallel_blocks}: output differs"
                ),
                (Err(_), Err(_)) => {}
                _ => panic!(
                    "{name} simd={simd} min_parallel_blocks={min_parallel_blocks}: libzstd {:?}, ours {:?}",
                    theirs.as_ref().map(Vec::len),
                    ours
                ),
            }
        }
    }
}

fn lcg_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

fn text(len: usize) -> Vec<u8> {
    let words = [
        "frame ",
        "block ",
        "checksum ",
        "literal ",
        "sequence ",
        "offset ",
        "window ",
    ];
    let mut v = Vec::with_capacity(len + 16);
    let mut x = 7u64;
    while v.len() < len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.extend_from_slice(words[(x >> 60) as usize % words.len()].as_bytes());
    }
    v.truncate(len);
    v
}

/// libzstd's frame of `data` at `level`, with a Content_Checksum.
fn with_checksum(data: &[u8], level: i32) -> Vec<u8> {
    let mut cctx = zstd::bulk::Compressor::new(level).unwrap();
    cctx.include_checksum(true).unwrap();
    cctx.compress(data).unwrap()
}

/// The content ranges of `f`'s raw blocks (a single frame).
fn raw_block_contents(f: &[u8]) -> Vec<std::ops::Range<usize>> {
    let fhd = f[4];
    let single_segment = fhd >> 5 & 1;
    let fcs = [usize::from(single_segment), 2, 4, 8][usize::from(fhd >> 6)];
    let dict_id = [0, 1, 2, 4][usize::from(fhd & 3)];
    let mut at = 5 + usize::from(1 - single_segment) + dict_id + fcs;
    let mut raw = Vec::new();
    loop {
        let h = u32::from_le_bytes([f[at], f[at + 1], f[at + 2], 0]);
        let size = (h >> 3) as usize;
        let (ty, len) = ((h >> 1) & 3, if (h >> 1) & 3 == 1 { 1 } else { size });
        if ty == 0 {
            raw.push(at + 3..at + 3 + size);
        }
        at += 3 + len;
        if h & 1 == 1 {
            return raw;
        }
    }
}

/// A frame whose Content_Checksum_flag is set decodes only when its
/// checksum is the low half of the content's XXH64 (ZSTD_decompressFrame's
/// checksum_wrong), whether the content or the checksum is altered, and in
/// any frame of a multi-frame input.
#[test]
fn content_checksum_is_verified() {
    let inputs = [
        ("empty", vec![]),
        ("1 byte", vec![42]),
        ("text 1000", text(1000)),
        ("text 1 MiB", text(1 << 20)),
        ("random 300000", lcg_bytes(300_000, 5)),
    ];
    let mut raw_altered = 0;
    for (name, data) in &inputs {
        for level in [1, 19] {
            let f = with_checksum(data, level);
            check(&format!("{name} L{level}"), &f, true);
            for i in 1..=4 {
                let mut g = f.clone();
                let at = g.len() - i;
                g[at] ^= 0x01;
                check(
                    &format!("{name} L{level}: checksum byte -{i} flipped"),
                    &g,
                    false,
                );
            }
            let mut g = f.clone();
            g.pop();
            check(&format!("{name} L{level}: checksum truncated"), &g, false);
            // Altered raw block content decodes, to other bytes.
            for r in raw_block_contents(&f).into_iter().filter(|r| !r.is_empty()) {
                let mut g = f.clone();
                g[(r.start + r.end) / 2] ^= 0x80;
                check(
                    &format!("{name} L{level}: raw block at {} altered", r.start),
                    &g,
                    false,
                );
                raw_altered += 1;
            }
        }
    }
    // The random input is stored in raw blocks at both levels.
    assert!(raw_altered >= 6, "{raw_altered} raw blocks altered");

    // The second frame's checksum covers its own content only.
    let (a, b) = (text(5000), lcg_bytes(200_000, 9));
    let mut f = with_checksum(&a, 3);
    f.extend_from_slice(&with_checksum(&b, 3));
    check("two frames", &f, true);
    let mut g = f.clone();
    let at = g.len() - 1;
    g[at] ^= 0x01;
    check("two frames: second checksum flipped", &g, false);
}
