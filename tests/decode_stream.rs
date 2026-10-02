//! Streaming decompression, `Decompressor::decompress_stream` and
//! `DecompressReader` over it, against the one-shot decoder on libzstd's
//! frames: every level, input and output in pieces of any size, frames
//! past their window, blocks ended by `ZSTD_compressStream2` flushes, frame
//! sequences with skippable frames.

mod common;

use common::{c_compress2, datasets, decompress_streaming, read_all, zstd_bulk, zstd_stream, MIB};
use rust_zstd::decode::DecodeOptions;
use rust_zstd::Decompressor;
use std::io;
use sys::ZSTD_cParameter::{
    ZSTD_c_checksumFlag, ZSTD_c_compressionLevel, ZSTD_c_contentSizeFlag, ZSTD_c_windowLog,
};
use zstd::zstd_safe::zstd_sys as sys;

/// The whole input in one piece.
const WHOLE: usize = usize::MAX;

/// Just past one block of content.
const SHORT: usize = (128 << 10) + 3;

fn dataset(name: &str) -> Vec<u8> {
    datasets()
        .into_iter()
        .find(|d| d.name == name)
        .unwrap_or_else(|| panic!("no dataset {name}"))
        .data
}

/// `frame` decodes to `want` in one shot, and streaming with input chunks
/// of each of `chunks` and as much output room (all it takes for `WHOLE`).
fn check(name: &str, frame: &[u8], want: &[u8], chunks: &[usize]) {
    let one_shot = rust_zstd::decompress(frame).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(one_shot == want, "{name}: one-shot content differs");
    for &chunk in chunks {
        let room = if chunk == WHOLE {
            want.len().max(1)
        } else {
            chunk
        };
        let got = decompress_streaming(frame, chunk, room, &DecodeOptions::default())
            .unwrap_or_else(|e| panic!("{name} chunk {chunk}: {e}"));
        assert!(
            got == one_shot,
            "{name} chunk {chunk}: {} bytes differ from one-shot's {}",
            got.len(),
            one_shot.len()
        );
    }
}

/// libzstd's frames of `name`, one-shot (with Frame_Content_Size) and
/// streamed (without), at levels 1 to 22: of its first MiB at chunks of 7,
/// 64 KiB and whole, of its first `SHORT` bytes at chunks of 1, and of all
/// of it at a few levels, and, of 512 KiB to 2 MiB, of all of it twice
/// over at level 1, past the 512 KiB window and the round buffer's margin.
fn levels(name: &str) {
    let data = dataset(name);
    let mib = &data[..data.len().min(MIB)];
    let short = &data[..data.len().min(SHORT)];
    for level in 1..=22 {
        for (kind, compress) in [
            ("bulk", zstd_bulk as fn(&[u8], i32) -> Vec<u8>),
            ("stream", zstd_stream),
        ] {
            let at = format!("{name} level {level} {kind}");
            check(&at, &compress(mib, level), mib, &[7, 1 << 16, WHOLE]);
            check(&format!("{at} short"), &compress(short, level), short, &[1]);
            if data.len() > MIB && [1, 19, 22].contains(&level) {
                check(
                    &format!("{at} all"),
                    &compress(&data, level),
                    &data,
                    &[1 << 16, WHOLE],
                );
            }
            if level == 1 && (MIB / 2..2 * MIB).contains(&data.len()) {
                let twice = data.repeat(2);
                check(
                    &format!("{at} twice"),
                    &compress(&twice, level),
                    &twice,
                    &[7, 1 << 16, WHOLE],
                );
            }
        }
    }
}

#[test]
fn levels_rust_src() {
    levels("rust_src_8m");
}

#[test]
fn levels_elf() {
    levels("elf_8m");
}

#[test]
fn levels_words() {
    levels("words_1m");
}

#[test]
fn levels_text() {
    levels("text_1m");
}

#[test]
fn levels_random() {
    levels("random_1m");
}

#[test]
fn levels_zeros() {
    levels("zeros_1m");
}

#[test]
fn levels_one_byte() {
    levels("one_byte");
}

#[test]
fn levels_empty() {
    levels("empty");
}

/// Frames with windows of 1 KiB to 256 KiB over a MiB of content, so that
/// the round buffer starts over many times and matches reach across its
/// segments, with and without Frame_Content_Size.
#[test]
fn frames_past_small_windows() {
    let data = dataset("rust_src_8m");
    let mib = &data[..MIB];
    for window_log in [10, 11, 17, 18] {
        for level in [1, 3, 9, 16, 19] {
            for content_size in [true, false] {
                let frame = c_compress2(
                    mib,
                    &[
                        (ZSTD_c_compressionLevel, level),
                        (ZSTD_c_windowLog, window_log),
                        (ZSTD_c_contentSizeFlag, content_size as i32),
                        (ZSTD_c_checksumFlag, 1),
                    ],
                );
                let at =
                    format!("window log {window_log} level {level} content size {content_size}");
                check(&at, &frame, mib, &[7, 4096, WHOLE]);
            }
        }
    }
}

/// libzstd's frame of `data` through `ZSTD_compressStream2` at `level`,
/// flushed after each piece of `pieces` (cycling) and with a checksum:
/// blocks end wherever a piece does.
fn c_stream_flushed(data: &[u8], level: i32, pieces: &[usize]) -> Vec<u8> {
    use sys::ZSTD_EndDirective::{ZSTD_e_end, ZSTD_e_flush};
    let mut frame = Vec::new();
    let mut out = vec![0u8; 1 << 17];
    // SAFETY: the context is used only here, with in-bounds buffers.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        for (param, value) in [(ZSTD_c_compressionLevel, level), (ZSTD_c_checksumFlag, 1)] {
            assert_eq!(
                sys::ZSTD_isError(sys::ZSTD_CCtx_setParameter(cctx, param, value)),
                0
            );
        }
        let mut pos = 0;
        let mut piece = pieces.iter().cycle();
        loop {
            let len = (data.len() - pos).min(*piece.next().unwrap());
            let end = pos + len == data.len();
            let mut input = sys::ZSTD_inBuffer {
                src: data[pos..].as_ptr().cast(),
                size: len,
                pos: 0,
            };
            loop {
                let mut output = sys::ZSTD_outBuffer {
                    dst: out.as_mut_ptr().cast(),
                    size: out.len(),
                    pos: 0,
                };
                let directive = if end { ZSTD_e_end } else { ZSTD_e_flush };
                let left = sys::ZSTD_compressStream2(cctx, &mut output, &mut input, directive);
                assert_eq!(sys::ZSTD_isError(left), 0);
                frame.extend_from_slice(&out[..output.pos]);
                if left == 0 && input.pos == input.size {
                    break;
                }
            }
            pos += len;
            if end {
                break;
            }
        }
        sys::ZSTD_freeCCtx(cctx);
    }
    frame
}

/// Frames of `ZSTD_compressStream2` flushed after pieces of 1 byte to
/// 200 KB: blocks of any size end mid-frame, mid-match for the encoder. At
/// level 1, of 2 MiB, past the 512 KiB window and the round buffer's
/// margin.
#[test]
fn compress_stream2_flushed_frames() {
    let data = dataset("rust_src_8m");
    for level in [1, 3, 12, 19] {
        let content = &data[..if level == 1 { 2 * MIB } else { MIB }];
        for pieces in [
            &[1, 2, 3, 100, 5000][..],
            &[70_000, 1, 200_000, 131_072, 131_073][..],
        ] {
            let frame = c_stream_flushed(content, level, pieces);
            check(
                &format!("level {level} pieces {pieces:?}"),
                &frame,
                content,
                &[1, 7, 1 << 16, WHOLE],
            );
        }
    }
}

/// Skippable frame of `size` bytes of User_Data, `fill` each.
fn skippable(size: u32, fill: u8) -> Vec<u8> {
    let mut f = 0x184D2A5Au32.to_le_bytes().to_vec();
    f.extend_from_slice(&size.to_le_bytes());
    f.resize(8 + size as usize, fill);
    f
}

/// Frames, skippable ones too, one after another, with the contents they
/// decode to (none for a skippable frame).
fn frame_sequence() -> Vec<(Vec<u8>, Vec<u8>)> {
    let data = dataset("words_1m");
    let a = data[..300_000].to_vec();
    let b = data[300_000..310_000].to_vec();
    let c = data[310_000..].to_vec();
    vec![
        (zstd_bulk(&a, 3), a),
        (skippable(0, 0), Vec::new()),
        (zstd_stream(&b, 19), b),
        (skippable(100, 0xAB), Vec::new()),
        (zstd_bulk(&[], 1), Vec::new()),
        (
            c_compress2(
                &c,
                &[(ZSTD_c_compressionLevel, 7), (ZSTD_c_checksumFlag, 1)],
            ),
            c,
        ),
        (skippable(5, 0), Vec::new()),
    ]
}

/// A sequence of frames streams at every chunk size, `decompress_stream`
/// returning 0 once at the end of each frame, when its content is all
/// written out.
#[test]
fn frame_sequence_ends_each_frame() {
    let frames = frame_sequence();
    let input: Vec<u8> = frames.iter().flat_map(|(f, _)| f.clone()).collect();
    let mut ends = Vec::new();
    let mut want = Vec::new();
    for (_, content) in &frames {
        want.extend_from_slice(content);
        ends.push(want.len());
    }
    for chunk in [1, 7, 4096, WHOLE] {
        for room in [1, 7, 1 << 16] {
            let mut d = Decompressor::new();
            let mut out = vec![0u8; room];
            let mut got = Vec::new();
            let mut got_ends = Vec::new();
            let mut pos = 0usize;
            loop {
                let src = &input[pos..input.len().min(pos.saturating_add(chunk))];
                let (mut read, mut written) = (0, 0);
                let hint = d
                    .decompress_stream(src, &mut read, &mut out, &mut written)
                    .unwrap();
                got.extend_from_slice(&out[..written]);
                pos += read;
                if hint == 0 {
                    got_ends.push(got.len());
                } else if read == 0 && written == 0 {
                    assert!(src.is_empty(), "chunk {chunk} room {room}: stalled");
                    break;
                }
            }
            d.finish().unwrap();
            assert!(got == want, "chunk {chunk} room {room}: content differs");
            assert_eq!(got_ends, ends, "chunk {chunk} room {room}: frame ends");
        }
    }
}

/// Fed exactly the input each hint asks for, with room for a block, the
/// decoder takes all of it in every call: the hint is the rest of the
/// current unit (frame header, block, checksum, skippable content).
#[test]
fn hints_ask_for_the_rest_of_the_unit() {
    let frames = frame_sequence();
    let input: Vec<u8> = frames.iter().flat_map(|(f, _)| f.clone()).collect();
    let want: Vec<u8> = frames.iter().flat_map(|(_, c)| c.clone()).collect();
    let mut d = Decompressor::new();
    let mut out = vec![0u8; 1 << 17];
    let mut got = Vec::new();
    let mut pos = 0;
    let mut hint = d.decompress_stream(&[], &mut 0, &mut out, &mut 0).unwrap();
    assert_eq!(hint, 5, "a frame header's first bytes");
    let mut calls = 0;
    while pos < input.len() {
        let src = &input[pos..pos + hint.max(1)];
        let (mut read, mut written) = (0, 0);
        hint = d
            .decompress_stream(src, &mut read, &mut out, &mut written)
            .unwrap();
        assert_eq!(read, src.len(), "at {pos}: takes all it asked for");
        got.extend_from_slice(&out[..written]);
        pos += read;
        calls += 1;
    }
    d.finish().unwrap();
    assert!(got == want);
    assert!(calls > frames.len() * 3, "{calls} calls");
}

/// A frame's checksum is computed across calls: a wrong one fails at every
/// chunk size, and with it every later call.
#[test]
fn checksum_mismatch_stops_the_decoder() {
    let data = dataset("text_1m");
    let mut frame = c_compress2(
        &data[..200_000],
        &[(ZSTD_c_compressionLevel, 3), (ZSTD_c_checksumFlag, 1)],
    );
    *frame.last_mut().unwrap() ^= 1;
    let one_shot = rust_zstd::decompress(&frame).unwrap_err();
    assert!(one_shot.contains("checksum"), "{one_shot}");
    for chunk in [1, 7, 4096, WHOLE] {
        let opts = DecodeOptions::default();
        let got = decompress_streaming(&frame, chunk, 4096, &opts).unwrap_err();
        assert_eq!(got, one_shot, "chunk {chunk}");
    }
    let mut d = Decompressor::new();
    let mut out = vec![0u8; 1 << 20];
    let (mut read, mut written) = (0, 0);
    let first = d
        .decompress_stream(&frame, &mut read, &mut out, &mut written)
        .unwrap_err();
    assert_eq!(first, one_shot);
    let again = d
        .decompress_stream(&frame[read..], &mut 0, &mut out, &mut 0)
        .unwrap_err();
    assert_eq!(again, first, "the error sticks");
    assert_eq!(d.finish().unwrap_err(), first);
    // After `reset` it decodes a frame again.
    d.reset();
    let good = zstd_bulk(&data[..1000], 1);
    let (mut read, mut written) = (0, 0);
    let hint = d
        .decompress_stream(&good, &mut read, &mut out, &mut written)
        .unwrap();
    assert_eq!((hint, read), (0, good.len()));
    assert!(out[..written] == data[..1000]);
    d.finish().unwrap();
}

/// Stream all of `input` through `d` with 4 KiB of output room, then
/// `finish`.
fn stream_all(d: &mut Decompressor, input: &[u8]) -> Result<Vec<u8>, String> {
    let mut content = Vec::new();
    let mut out = vec![0u8; 4096];
    let mut pos = 0;
    loop {
        let mut written = 0;
        d.decompress_stream(input, &mut pos, &mut out, &mut written)?;
        content.extend_from_slice(&out[..written]);
        if pos == input.len() && written == 0 {
            break;
        }
    }
    d.finish()?;
    Ok(content)
}

/// `Decompressor::decompress` gives the outcome of the function
/// `decompress` wherever streaming stood, mid-frame or failed, and
/// streaming then starts on a new frame, after content or an error.
#[test]
fn decompress_resets_the_stream() {
    let data = dataset("text_1m");
    let a = zstd_bulk(&data[..300_000], 3);
    let b = zstd_stream(&data[..1000], 19);
    let cut = &a[..a.len() - 10];
    let mut d = Decompressor::new();

    // Mid-frame, with content to write out, then with part of a block in
    // hand.
    let mut out = vec![0u8; 100];
    let (mut read, mut written) = (0, 0);
    d.decompress_stream(&a, &mut read, &mut out, &mut written)
        .unwrap();
    assert!(read < a.len() && written == out.len());
    assert_eq!(d.decompress(&b), Ok(data[..1000].to_vec()));
    let (mut read, mut written) = (0, 0);
    d.decompress_stream(&a[..20], &mut read, &mut out, &mut written)
        .unwrap();
    assert_eq!((read, written), (20, 0));
    assert_eq!(d.decompress(&b), Ok(data[..1000].to_vec()));
    assert_eq!(stream_all(&mut d, &a), Ok(data[..300_000].to_vec()));

    // Stopped by an error.
    let bad = [0u8; 8];
    assert!(d.decompress_stream(&bad, &mut 0, &mut out, &mut 0).is_err());
    assert_eq!(d.decompress(&b), Ok(data[..1000].to_vec()));
    assert_eq!(stream_all(&mut d, &b), Ok(data[..1000].to_vec()));

    // A one-shot error inside a frame.
    let one_shot = rust_zstd::decompress(cut).unwrap_err();
    assert_eq!(d.decompress(cut), Err(one_shot));
    assert_eq!(stream_all(&mut d, &a), Ok(data[..300_000].to_vec()));
    assert_eq!(d.decompress(&a), Ok(data[..300_000].to_vec()));
}

/// `finish` while content is left to write out, or mid-frame, fails; after
/// a frame's last byte is written out it succeeds.
#[test]
fn finish_needs_the_frame_written_out() {
    let data = dataset("words_1m");
    let frame = zstd_bulk(&data[..10_000], 3);
    let mut d = Decompressor::new();
    let mut out = vec![0u8; 100];
    let (mut read, mut written) = (0, 0);
    let hint = d
        .decompress_stream(&frame, &mut read, &mut out, &mut written)
        .unwrap();
    assert_eq!((hint, written), (1, 100), "room for 100 bytes");
    let left = d.finish().unwrap_err();
    assert!(left.contains("left to write out"), "{left}");
    let mut got = out[..written].to_vec();
    loop {
        let mut written = 0;
        let hint = d
            .decompress_stream(&[], &mut 0, &mut out, &mut written)
            .unwrap();
        got.extend_from_slice(&out[..written]);
        if hint == 0 {
            break;
        }
    }
    assert!(got == data[..10_000]);
    d.finish().unwrap();

    let mut d = Decompressor::new();
    let mut out = vec![0u8; 1 << 17];
    let half = frame.len() / 2;
    d.decompress_stream(&frame[..half], &mut 0, &mut out, &mut 0)
        .unwrap();
    assert_eq!(
        d.finish().unwrap_err(),
        rust_zstd::decompress(&frame[..half]).unwrap_err()
    );
}

/// Frame of a 1 KiB window (no Frame_Content_Size) holding `fill` RLE
/// blocks of 1 KiB, raw blocks of `raws` bytes, then a compressed block of
/// `literals` raw literals (at most 31) and one sequence: the literals,
/// then 4 bytes from `offset` back.
fn edge_frame(fill: usize, raws: &[usize], literals: usize, offset: u32) -> Vec<u8> {
    let mut f = vec![0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00];
    for _ in 0..fill {
        let h = (1u32 << 1) | (1024 << 3);
        f.extend_from_slice(&h.to_le_bytes()[..3]);
        f.push(0);
    }
    let mut block = |ty: u32, last: bool, body: &[u8]| {
        let h = u32::from(last) | (ty << 1) | ((body.len() as u32) << 3);
        f.extend_from_slice(&h.to_le_bytes()[..3]);
        f.extend_from_slice(body);
    };
    for (i, &n) in raws.iter().enumerate() {
        let content: Vec<u8> = (0..n).map(|j| (i * 7 + j % 251 + 1) as u8).collect();
        block(0, false, &content);
    }
    // Literals_Length code and its extra bits (RFC 8878 lines 1663-1676).
    let (ll_code, ll_bits, ll_extra) = match literals {
        0..=15 => (literals, 0, 0),
        16..=23 => (16 + (literals - 16) / 2, 1, (literals - 16) % 2),
        24..=31 => (20 + (literals - 24) / 4, 2, (literals - 24) % 4),
        _ => unreachable!("a 1-byte Raw_Literals_Block header"),
    };
    let mut body = vec![(literals as u8) << 3];
    body.extend((0..literals).map(|j| 0xC0 | j as u8));
    // One sequence in RLE-mode tables (no state bits): Offset_Value =
    // offset + 3 in `of_code` extra bits, read first, so highest; ML code 1
    // (4 bytes, no bits); the literals length bits lowest.
    let value = offset + 3;
    let of_code = 31 - value.leading_zeros();
    body.extend_from_slice(&[1, 0x54, ll_code as u8, of_code as u8, 1]);
    let stream = ll_extra as u64
        | u64::from(value - (1 << of_code)) << ll_bits
        | 1 << (ll_bits + of_code as usize);
    body.extend_from_slice(&stream.to_le_bytes()[..(ll_bits + of_code as usize) / 8 + 1]);
    block(2, true, &body);
    f
}

/// A match up to Window_Size back, right after the round buffer starts a
/// new segment, reads the previous segment past every byte the literals
/// before it wrote, overshoot included: after 8 KiB + `k` bytes for each
/// `k` a block can end the segment at (the window and the margin of seven
/// blocks), with up to 31 literals.
#[test]
fn match_at_the_window_edge_after_a_new_segment() {
    let window = 1024u32;
    for k in 1..=48 {
        for literals in [0, 5, 15, 16, 17, 24, 31] {
            for offset in window - 48..=window {
                let f = edge_frame(7, &[1024, k], literals, offset);
                let at = format!("k {k} literals {literals} offset {offset}");
                let want = rust_zstd::decompress(&f).unwrap_or_else(|e| panic!("{at}: {e}"));
                assert_eq!(want.len(), 8 * 1024 + k + literals + 4, "{at}");
                for chunk in [7, WHOLE] {
                    let got = decompress_streaming(&f, chunk, 4096, &DecodeOptions::default())
                        .unwrap_or_else(|e| panic!("{at} chunk {chunk}: {e}"));
                    assert!(got == want, "{at} chunk {chunk}: content differs");
                }
            }
        }
    }
}

/// The reader gives the content of a sequence of frames, whatever the
/// pieces it reads and the buffers it fills.
#[test]
fn reader_decodes_frame_sequence() {
    let frames = frame_sequence();
    let input: Vec<u8> = frames.iter().flat_map(|(f, _)| f.clone()).collect();
    let want: Vec<u8> = frames.iter().flat_map(|(_, c)| c.clone()).collect();
    for piece in [1, 7, 4096, WHOLE] {
        for room in [7, 1 << 16] {
            let got = read_all(&input, piece, room, None).unwrap();
            assert!(got == want, "piece {piece} room {room}: content differs");
        }
    }
    assert!(read_all(&[], 1, 1, None).unwrap().is_empty());
}

/// The reader fails with `InvalidData` and `decompress`'s error where
/// `decompress` fails: on a frame cut short anywhere, and on one with a
/// bad checksum.
#[test]
fn reader_fails_where_decompress_does() {
    let data = dataset("words_1m");
    let frame = c_compress2(
        &data[..50_000],
        &[(ZSTD_c_compressionLevel, 3), (ZSTD_c_checksumFlag, 1)],
    );
    let mut bad = frame.clone();
    *bad.last_mut().unwrap() ^= 1;
    let mut cases: Vec<&[u8]> = (1..frame.len()).step_by(97).map(|n| &frame[..n]).collect();
    cases.extend([&frame[..frame.len() - 1], &bad[..]]);
    for input in cases {
        let want = rust_zstd::decompress(input).unwrap_err();
        for piece in [7, WHOLE] {
            let e = read_all(input, piece, 4096, None).unwrap_err();
            assert_eq!(
                e.kind(),
                io::ErrorKind::InvalidData,
                "{} bytes",
                input.len()
            );
            assert_eq!(e.to_string(), want, "{} bytes", input.len());
        }
    }
}
