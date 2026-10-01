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

/// The Block_Type and Block_Content range of each block of `f` (a single
/// frame).
fn blocks(f: &[u8]) -> Vec<(u32, std::ops::Range<usize>)> {
    let fhd = f[4];
    let single_segment = fhd >> 5 & 1;
    let fcs = [usize::from(single_segment), 2, 4, 8][usize::from(fhd >> 6)];
    let dict_id = [0, 1, 2, 4][usize::from(fhd & 3)];
    let mut at = 5 + usize::from(1 - single_segment) + dict_id + fcs;
    let mut blocks = Vec::new();
    loop {
        let h = u32::from_le_bytes([f[at], f[at + 1], f[at + 2], 0]);
        let ty = (h >> 1) & 3;
        let len = if ty == 1 { 1 } else { (h >> 3) as usize };
        blocks.push((ty, at + 3..at + 3 + len));
        at += 3 + len;
        if h & 1 == 1 {
            return blocks;
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
            let raw = blocks(&f).into_iter().filter(|(ty, _)| *ty == 0);
            for (_, r) in raw.filter(|(_, r)| !r.is_empty()) {
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

const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Single-segment frame of one raw block holding `content`.
fn raw_frame(content: &[u8]) -> Vec<u8> {
    assert!(content.len() < 256);
    let mut f = MAGIC.to_vec();
    f.extend_from_slice(&[0x20, content.len() as u8]);
    f.extend_from_slice(&(1 | (content.len() as u32) << 3).to_le_bytes()[..3]);
    f.extend_from_slice(content);
    f
}

/// Skippable frame whose Frame_Size field says `size`, with `payload`.
fn skippable(size: u32, payload: &[u8]) -> Vec<u8> {
    let mut f = 0x184D_2A53u32.to_le_bytes().to_vec();
    f.extend_from_slice(&size.to_le_bytes());
    f.extend_from_slice(payload);
    f
}

/// ZSTD_decompressMultiFrame takes frames while at least 5 bytes
/// (ZSTD_startingInputLength) remain and fails on any byte left over:
/// after a frame, whether it decoded to bytes or to none, garbage, a
/// truncated frame or a truncated skippable frame is an error, as is an
/// input of 1 to 4 bytes.
#[test]
fn every_input_byte_belongs_to_a_frame() {
    let hello = raw_frame(b"hello");
    let tails: Vec<(&str, Vec<u8>, bool)> = vec![
        ("nothing", vec![], true),
        ("an empty frame", raw_frame(b""), true),
        ("an empty skippable frame", skippable(0, &[]), true),
        ("1 byte", vec![1], false),
        ("4 bytes", vec![1, 2, 3, 4], false),
        ("5 bytes", vec![1, 2, 3, 4, 5], false),
        ("10 bytes", b"garbage!!!".to_vec(), false),
        ("a zstd magic number", MAGIC.to_vec(), false),
        (
            "a magic number and descriptor",
            [&MAGIC[..], &[0x20]].concat(),
            false,
        ),
        ("a truncated frame", hello[..10].to_vec(), false),
        (
            "5 bytes of a skippable frame",
            skippable(16, &[])[..5].to_vec(),
            false,
        ),
        (
            "7 bytes of a skippable frame",
            skippable(16, &[])[..7].to_vec(),
            false,
        ),
        (
            "a skippable frame short of its size",
            skippable(4, &[1, 2, 3]),
            false,
        ),
        ("a skippable frame and 1 byte", skippable(1, &[9, 7]), false),
    ];
    let leads: Vec<(&str, Vec<u8>)> = vec![
        ("nothing", vec![]),
        ("a frame", hello.clone()),
        ("an empty frame", raw_frame(b"")),
        ("a skippable frame", skippable(3, &[1, 2, 3])),
        (
            "a multi-block frame",
            zstd::bulk::compress(&text(1 << 20), 3).unwrap(),
        ),
    ];
    for (lead_name, lead) in &leads {
        for (tail_name, tail, accept) in &tails {
            if lead.is_empty() && tail.is_empty() {
                continue;
            }
            let f = [&lead[..], &tail[..]].concat();
            check(&format!("{lead_name} then {tail_name}"), &f, *accept);
        }
    }
    for n in 1..hello.len() {
        check(&format!("first {n} bytes of a frame"), &hello[..n], false);
    }

    // An empty input decodes to nothing in one shot. (The zstd crate's
    // streaming reader reports an incomplete frame instead.)
    assert_eq!(
        zstd::bulk::decompress(&[], CAPACITY).unwrap(),
        Vec::<u8>::new()
    );
    assert_eq!(rust_zstd::decompress(&[]).unwrap(), Vec::<u8>::new());
}

/// Frame of one raw block holding `content`, whose header carries the
/// Dictionary_ID `dict_id` in the `dict_id_flag` form (1, 2 or 4 bytes),
/// single-segment or with a window descriptor.
fn dict_id_frame(dict_id_flag: u8, dict_id: u32, single_segment: bool, content: &[u8]) -> Vec<u8> {
    assert!(content.len() < 256);
    let mut f = MAGIC.to_vec();
    f.push(if single_segment { 0x20 } else { 0 } | dict_id_flag);
    if !single_segment {
        f.push(0);
    }
    let len = [0, 1, 2, 4][usize::from(dict_id_flag)];
    f.extend_from_slice(&dict_id.to_le_bytes()[..len]);
    if single_segment {
        f.push(content.len() as u8);
    }
    f.extend_from_slice(&(1 | (content.len() as u32) << 3).to_le_bytes()[..3]);
    f.extend_from_slice(content);
    f
}

/// With no dictionary loaded, a frame naming a dictionary (any non-zero
/// Dictionary_ID, in every field size) is ZSTD_decodeFrameHeader's
/// dictionary_wrong; a Dictionary_ID field holding 0 names none.
#[test]
fn dictionary_id_needs_a_dictionary() {
    let cases: [(u8, u32, bool); 12] = [
        (1, 0, true),
        (1, 1, false),
        (1, 0xff, false),
        (2, 0, true),
        (2, 1, false),
        (2, 0x100, false),
        (2, 0xffff, false),
        (3, 0, true),
        (3, 1, false),
        (3, 0x100, false),
        (3, 0x0100_0000, false),
        (3, 0xffff_ffff, false),
    ];
    let hello = raw_frame(b"hello");
    for (flag, id, accept) in cases {
        for single_segment in [true, false] {
            let f = dict_id_frame(flag, id, single_segment, b"dict");
            let name = format!("flag {flag} id {id:#x} single_segment {single_segment}");
            check(&name, &f, accept);
            check(
                &format!("a frame then {name}"),
                &[&hello[..], &f[..]].concat(),
                accept,
            );
        }
    }
}

/// The offset of each compressed block's Symbol_Compression_Modes byte in
/// `f` (a single frame), and the length of the Number_of_Sequences field
/// before it; blocks without sequences have none.
fn modes_bytes(f: &[u8]) -> Vec<(usize, usize)> {
    let mut modes = Vec::new();
    for (_, r) in blocks(f).into_iter().filter(|(ty, _)| *ty == 2) {
        let b = &f[r.clone()];
        let word = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        // Literals_Section_Header length and the literals' length after it.
        let (header, literals) = match (b[0] & 3, b[0] >> 2 & 3) {
            (0, 0 | 2) => (1, usize::from(b[0] >> 3)),
            (0, 1) => (2, (word >> 4 & 0xfff) as usize),
            (0, _) => (3, (word >> 4 & 0xf_ffff) as usize),
            (1, 0 | 2) => (1, 1),
            (1, 1) => (2, 1),
            (1, _) => (3, 1),
            (_, 0 | 1) => (3, (word >> 14 & 0x3ff) as usize),
            (_, 2) => (4, (word >> 18) as usize),
            (_, _) => (5, (word >> 22) as usize | usize::from(b[4]) << 10),
        };
        let at = header + literals;
        let (count, len) = match b[at] {
            0..=127 => (u32::from(b[at]), 1),
            128..=254 => (u32::from(b[at] & 0x7f) << 8 | u32::from(b[at + 1]), 2),
            255 => (1, 3),
        };
        if count != 0 {
            modes.push((r.start + at + len, len));
        }
    }
    modes
}

/// Copies of 4 bytes from alternately 24 and 40 bytes back: each is one
/// sequence without literals at levels 4 and up, over 0x7F00 in a block.
fn alternating_copies(len: usize) -> Vec<u8> {
    let mut data = lcg_bytes(40, 3);
    for k in 0.. {
        if data.len() >= len {
            break;
        }
        let back = if k % 2 == 0 { 24 } else { 40 };
        for _ in 0..4 {
            data.push(data[data.len() - back]);
        }
    }
    data.truncate(len);
    data
}

/// Reserved bits must be zero: Frame_Header_Descriptor bit 3
/// (ZSTD_getFrameHeader_advanced's frameParameter_unsupported; unused bit 4
/// is ignored) and Symbol_Compression_Modes bits 1-0, after a
/// Number_of_Sequences field of each length (ZSTD_decodeSeqHeaders'
/// corruption_detected).
#[test]
fn reserved_bits_must_be_zero() {
    let hello = raw_frame(b"hello");
    for (bits, accept) in [(0x08, false), (0x10, true), (0x18, false)] {
        for single_segment in [true, false] {
            let mut f = dict_id_frame(0, 0, single_segment, b"reserved");
            f[4] |= bits;
            let name = format!("descriptor bits {bits:#04x} single_segment {single_segment}");
            check(&name, &f, accept);
            check(
                &format!("a frame then {name}"),
                &[&hello[..], &f[..]].concat(),
                accept,
            );
        }
    }

    let frames = [
        (
            "text 1000 L1",
            zstd::bulk::compress(&text(1000), 1).unwrap(),
        ),
        (
            "text 1000 L19",
            zstd::bulk::compress(&text(1000), 19).unwrap(),
        ),
        (
            "text 300000 L3",
            zstd::bulk::compress(&text(300_000), 3).unwrap(),
        ),
        (
            "alternating copies L5",
            zstd::bulk::compress(&alternating_copies(1 << 17), 5).unwrap(),
        ),
    ];
    let mut count_lengths = [0; 4];
    for (name, f) in &frames {
        check(name, f, true);
        for (at, len) in modes_bytes(f) {
            count_lengths[len] += 1;
            for bits in 1..=3 {
                let mut g = f.clone();
                g[at] |= bits;
                check(&format!("{name}: modes byte at {at} | {bits}"), &g, false);
            }
        }
    }
    assert!(
        count_lengths[1..].iter().all(|&n| n > 0),
        "Number_of_Sequences lengths {count_lengths:?}"
    );
}

/// A skippable Frame_Size of 0xFFFFFFF8 or more overflows 32 bits with the
/// 8-byte header, which ZSTD_readSkippableFrameSize refuses as
/// frameParameter_unsupported on every build, before it compares the length
/// with the input: a header alone gets that error, not srcSize_wrong.
#[test]
fn skippable_frame_length_fits_32_bits() {
    let hello = raw_frame(b"hello");
    for (size, unsupported) in [
        (0xFFFF_FFF7, false),
        (0xFFFF_FFF8, true),
        (0xFFFF_FFFF, true),
    ] {
        for (lead_name, lead) in [("", &[][..]), ("a frame then ", &hello[..])] {
            let f = [lead, &skippable(size, &[1, 2, 3])].concat();
            let name = format!("{lead_name}a skippable frame of size {size:#x}");
            check(&name, &f, false);
            let theirs = zstd::bulk::decompress(&f, CAPACITY).unwrap_err();
            let want = if unsupported {
                "Unsupported frame parameter"
            } else {
                "Src size is incorrect"
            };
            assert_eq!(theirs.to_string(), want, "{name}");
            let ours = rust_zstd::decompress(&f).unwrap_err();
            assert_eq!(
                ours.contains("overflows 32 bits"),
                unsupported,
                "{name}: {ours}"
            );
        }
    }
}
