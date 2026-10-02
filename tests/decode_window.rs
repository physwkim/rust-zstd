//! Match offsets reach back at most Window_Size bytes (RFC 8878 lines
//! 590-593 and 1204-1206; an offset of exactly Window_Size is accepted) and
//! never before the frame's start, serial and MT at both SIMD levels.
//! libzstd 1.5.7's one-shot decoder bounds offsets by the frame start
//! alone; each frame states its outcome, checked, and where both accept
//! the bytes must match.

mod common;

use rust_zstd::decode::{decompress_with_options, DecodeOptions};

const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Frame of raw blocks of `raws` bytes each, then a compressed block
/// copying 4 bytes from `offset` back, after header `header`.
fn frame(header: &[u8], raws: &[usize], offset: u32) -> Vec<u8> {
    let mut f = header.to_vec();
    let mut block = |ty: u32, last: bool, body: &[u8]| {
        let h = u32::from(last) | (ty << 1) | ((body.len() as u32) << 3);
        f.extend_from_slice(&h.to_le_bytes()[..3]);
        f.extend_from_slice(body);
    };
    for (i, &n) in raws.iter().enumerate() {
        block(
            0,
            false,
            &(0..n).map(|j| (i * 7 + j % 251) as u8).collect::<Vec<_>>(),
        );
    }
    // No literals; one sequence in RLE-mode tables: LL code 0, offset code
    // `code` carrying Offset_Value = offset + 3 in `code` extra bits, ML
    // code 1 (4 bytes).
    let value = offset + 3;
    let code = 31 - value.leading_zeros();
    let mut body = vec![0x00, 1, 0x54, 0, code as u8, 1];
    let stream = (1u64 << code) | u64::from(value - (1 << code));
    body.extend_from_slice(&stream.to_le_bytes()[..code as usize / 8 + 1]);
    block(2, true, &body);
    f
}

/// Header without Frame_Content_Size, of Window_Descriptor `wd`.
fn windowed(wd: u8) -> Vec<u8> {
    let mut h = MAGIC.to_vec();
    h.extend_from_slice(&[0x00, wd]);
    h
}

/// Single-segment header: Window_Size is the content size.
fn single_segment(content_size: u16) -> Vec<u8> {
    let mut h = MAGIC.to_vec();
    h.push(0x20 | (1 << 6));
    h.extend_from_slice(&(content_size - 256).to_le_bytes());
    h
}

/// Decode `f` serial and MT at both SIMD levels and check the outcome is
/// `accept`; check libzstd's one-shot outcome is `libzstd`, and where both
/// accept, that the bytes match.
fn check_vs(name: &str, f: &[u8], accept: bool, libzstd: bool) {
    let theirs = zstd::bulk::decompress(f, 1 << 20);
    assert_eq!(
        theirs.is_ok(),
        libzstd,
        "{name}: libzstd {:?}",
        theirs.as_ref().map(Vec::len)
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
            let at = format!("{name} simd={simd} min_parallel_blocks={min_parallel_blocks}");
            assert_eq!(
                ours.is_ok(),
                accept,
                "{at}: ours {:?}",
                ours.as_ref().map(Vec::len)
            );
            if let (Ok(a), Ok(b)) = (&theirs, &ours) {
                assert!(a == b, "{at}: output differs");
            }
        }
    }
    common::assert_stream_parity(name, f);
}

/// Windows of 1 KiB, 1 KiB + 1/8 and 2 KiB, each after more decoded bytes
/// than the window: offsets up to Window_Size are accepted, larger ones
/// rejected though the frame holds them.
#[test]
fn offset_is_at_most_window_size() {
    for (wd, window, raws) in [
        (0, 1024, &[1000, 1000][..]),
        (1, 1152, &[1000, 1000][..]),
        (1 << 3, 2048, &[1000, 1000, 1000][..]),
    ] {
        let decoded: u32 = raws.iter().sum::<usize>() as u32;
        for offset in [window - 1, window, window + 1, decoded, decoded + 1] {
            let f = frame(&windowed(wd), raws, offset);
            check_vs(
                &format!("window {window}: offset {offset} after {decoded} bytes"),
                &f,
                offset <= window,
                offset <= decoded,
            );
        }
    }
}

/// Before Window_Size bytes are decoded, the frame's start bounds the
/// offset; a single-segment frame's window is its content size, which the
/// frame's start bounds first.
#[test]
fn offset_reaches_back_to_the_frame_start() {
    for offset in [600, 601] {
        let f = frame(&windowed(1 << 3), &[600], offset);
        check_vs(
            &format!("2 KiB window: offset {offset} after 600 bytes"),
            &f,
            offset <= 600,
            offset <= 600,
        );
        let f = frame(&single_segment(604), &[600], offset);
        check_vs(
            &format!("single segment 604: offset {offset} after 600 bytes"),
            &f,
            offset <= 600,
            offset <= 600,
        );
    }
}

/// A frame's matches do not reach into the frame before it, whatever its
/// window.
#[test]
fn offset_does_not_reach_the_previous_frame() {
    let first = frame(&windowed(1 << 3), &[1000], 1000);
    for offset in [500, 501] {
        let mut f = first.clone();
        f.extend_from_slice(&frame(&windowed(1 << 3), &[500], offset));
        check_vs(
            &format!("second frame: offset {offset} after 500 bytes"),
            &f,
            offset <= 500,
            offset <= 500,
        );
    }
}
