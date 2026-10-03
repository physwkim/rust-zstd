//! Frame-level checks of ZSTD_decompressFrame / ZSTD_decompressMultiFrame:
//! libzstd 1.5.7 frames, whole or altered, get the verdict of its one-shot
//! (`zstd::bulk`) and streaming (`zstd::stream`) decoders, and their bytes
//! when accepted, serial and MT at both SIMD levels, except where RFC 8878
//! decides otherwise, as a test notes.

mod common;

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
                min_parallel_bytes: 0,
                simd,
                window_log_max: 0,
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
    common::assert_stream_parity(name, f);
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
        // Past level 1's 512 KiB window and the round buffer's margin.
        ("text 2 MiB", text(2 << 20)),
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
    common::assert_stream_parity("empty input", &[]);
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

/// libzstd's frame of `content` at `level`, streamed with a flush (a block
/// end) after each of `chunks` bytes.
fn flushed_frame(content: &[u8], chunks: &[usize], level: i32) -> Vec<u8> {
    use std::io::Write;
    let mut enc = zstd::stream::Encoder::new(Vec::new(), level).unwrap();
    let mut at = 0;
    for &n in chunks {
        enc.write_all(&content[at..at + n]).unwrap();
        enc.flush().unwrap();
        at += n;
    }
    enc.write_all(&content[at..]).unwrap();
    enc.finish().unwrap()
}

/// Repeat_Mode reuses the table the previous block's sequences decoded
/// with, the predefined one included (ZSTD_buildSeqTable's set_repeat keeps
/// LLTptr, which set_basic points at LL_defaultDTable): libzstd frames
/// decode to the same content with each Predefined_Mode that follows one of
/// the same table switched to Repeat_Mode, after a block with no tables or
/// its own ones.
#[test]
fn repeat_mode_after_predefined_mode() {
    let content = text(60_000);
    let frames: [(i32, &[usize]); 4] = [
        (1, &[100, 100, 100]),
        (1, &[30_000, 20, 20, 20]),
        (1, &[500, 500, 500, 500]),
        (19, &[100, 100, 100]),
    ];
    for (level, chunks) in frames {
        let f = flushed_frame(&content, chunks, level);
        let mut g = f.clone();
        let mut switched = 0;
        for w in modes_bytes(&f).windows(2) {
            let (prev, at) = (f[w[0].0], w[1].0);
            // LL, OF, ML modes: bits 7-6, 5-4, 3-2.
            for shift in [6, 4, 2] {
                if prev >> shift & 3 == 0 && f[at] >> shift & 3 == 0 {
                    g[at] |= 3 << shift;
                    switched += 1;
                }
            }
        }
        let name = format!("L{level} flushed after {chunks:?}, {switched} modes switched");
        assert!(switched >= 2, "{name}");
        check(&name, &g, true);
        assert!(
            zstd::bulk::decompress(&g, CAPACITY).unwrap() == content,
            "{name}"
        );
    }
}

/// A frame's first block has no earlier block's tables: Treeless literals
/// or a Repeat_Mode table there are rejected, after a frame that built
/// them in the same input, or in the call before on a reused
/// `Decompressor` (`check`). Each such frame is a later block of a libzstd
/// frame made the only block of a frame of its own: one with Treeless
/// literals, or one of raw literals with its modes set to Repeat_Mode.
#[test]
fn first_block_reuses_no_table() {
    // Every block after the first has Treeless literals of 16 letters.
    let letters: Vec<u8> = lcg_bytes(60_000, 5).iter().map(|b| b'a' + b % 16).collect();
    let mut found = [0; 2];
    for (content, repeat) in [(letters, false), (text(60_000), true)] {
        let f = flushed_frame(&content, &[1000, 1000, 1000, 1000], 1);
        let list = blocks(&f);
        let header = &f[..list[0].1.start - 3];
        let modes = modes_bytes(&f);
        for (i, (ty, r)) in list.into_iter().enumerate().skip(1) {
            let mut block = f[r.clone()].to_vec();
            let treeless = block[0] & 3 == 3;
            let what = match modes.iter().find(|m| r.contains(&m.0)) {
                _ if ty != 2 => continue,
                Some(&(at, _)) if repeat && !treeless => {
                    block[at - r.start] |= 0xFC;
                    "Repeat_Mode tables"
                }
                _ if !repeat && treeless => "Treeless literals",
                _ => continue,
            };
            found[usize::from(repeat)] += 1;
            let mut tail = header.to_vec();
            let block_header = (block.len() as u32) << 3 | 2 << 1 | 1;
            tail.extend_from_slice(&block_header.to_le_bytes()[..3]);
            tail.extend_from_slice(&block);
            let name = format!("block {i} alone, {what}");
            check(&format!("{name}: its frame"), &f, true);
            check(&name, &tail, false);
            check(
                &format!("{name} after its frame"),
                &[&f[..], &tail].concat(),
                false,
            );
        }
    }
    assert!(found[0] >= 1 && found[1] >= 1, "{found:?}");
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

/// A skippable frame's Frame_Size is any 32-bit value (RFC 8878 lines
/// 1325-1329), the largest too; an input that stops short of it is
/// truncated, which libzstd rejects as well.
#[test]
fn skippable_frame_short_of_its_size() {
    let hello = raw_frame(b"hello");
    for size in [0xFFFF_FFF7, 0xFFFF_FFF8, 0xFFFF_FFFF] {
        for (lead_name, lead) in [("", &[][..]), ("a frame then ", &hello[..])] {
            let f = [lead, &skippable(size, &[1, 2, 3])].concat();
            let name = format!("{lead_name}3 bytes of a skippable frame of size {size:#x}");
            check(&name, &f, false);
            let ours = rust_zstd::decompress(&f).unwrap_err();
            assert!(ours.contains("past end of input"), "{name}: {ours}");
        }
    }
}

/// A skippable frame of each of the largest Frame_Sizes, up to 2^32 + 7
/// bytes with its header, is skipped and decoding resumes after it (RFC
/// 8878 lines 1306-1308, 1325-1329). libzstd 1.5.7's one-shot decoder
/// refuses a Frame_Size of 0xFFFFFFF8 or more, as its 32-bit frame length
/// wraps.
#[cfg(target_pointer_width = "64")]
#[test]
fn skippable_frame_of_any_32_bit_size_is_skipped() {
    let hello = raw_frame(b"hello");
    for size in [0xFFFF_FFF7u32, 0xFFFF_FFF8, 0xFFFF_FFFF] {
        // Zeroed memory is mapped lazily: only the pages written below,
        // the frames around the skippable one and its header, take memory.
        let mut f = vec![0u8; 2 * hello.len() + 8 + size as usize];
        let n = f.len();
        f[..hello.len()].copy_from_slice(&hello);
        f[hello.len()..][..8].copy_from_slice(&skippable(size, &[]));
        f[n - hello.len()..].copy_from_slice(&hello);
        let name = format!("a skippable frame of size {size:#x} between two frames");
        let theirs = zstd::bulk::decompress(&f, CAPACITY);
        assert_eq!(
            theirs.is_ok(),
            size < 0xFFFF_FFF8,
            "{name}: libzstd {theirs:?}"
        );
        for simd in [false, true] {
            for min_parallel_blocks in [usize::MAX, 1] {
                let options = DecodeOptions {
                    min_parallel_blocks,
                    min_parallel_bytes: 0,
                    simd,
                    window_log_max: 0,
                };
                let ours = decompress_with_options(&f, &options);
                assert_eq!(
                    ours.as_deref(),
                    Ok(&b"hellohello"[..]),
                    "{name} simd={simd} min_parallel_blocks={min_parallel_blocks}"
                );
                if let Ok(theirs) = &theirs {
                    assert_eq!(theirs, b"hellohello", "{name}: libzstd");
                }
            }
        }
        // Not a byte or 7 at a time: 4 GiB.
        common::assert_stream_parity_at(&name, &f, &[4096, usize::MAX]);
    }
}
