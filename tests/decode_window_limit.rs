//! The streaming window limit (`DecodeOptions::window_log_max`,
//! ZSTD_d_windowLogMax) against libzstd 1.5.7's streaming decoder fed the
//! same pieces: a frame whose Window_Size is above the limit is refused at
//! its header, unless one call gets all of it with room for its
//! Frame_Content_Size, exactly where libzstd refuses it, at the default
//! limit and at limits set; one-shot decoding takes it, as libzstd's does.

mod common;

use common::{
    c_compress2, c_streaming, is_window_limit, outcome, refuses_window, stream_room, stream_with,
    STREAM_CHUNKS, WINDOW_LOG_MAX,
};
use rust_zstd::decode::{decompress, decompress_with_options, DecodeOptions};
use rust_zstd::{DecompressReader, Decompressor};
use std::io::Read;
use zstd::zstd_safe::zstd_sys as sys;

const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

const WINDOW_TOO_LARGE: sys::ZSTD_ErrorCode =
    sys::ZSTD_ErrorCode::ZSTD_error_frameParameter_windowTooLarge;

/// A block: Block_Type, Block_Size, content.
type Block<'a> = (u32, u32, &'a [u8]);

/// The frame of header `header` and `blocks`, then `checksum` if given.
fn frame(header: &[u8], blocks: &[Block], checksum: Option<[u8; 4]>) -> Vec<u8> {
    let mut f = header.to_vec();
    for (i, &(ty, size, content)) in blocks.iter().enumerate() {
        let h = u32::from(i + 1 == blocks.len()) | ty << 1 | size << 3;
        f.extend_from_slice(&h.to_le_bytes()[..3]);
        f.extend_from_slice(content);
    }
    f.extend(checksum.into_iter().flatten());
    f
}

/// A frame header of Window_Descriptor `wd`, with the 4-byte
/// Frame_Content_Size `fcs` if given, and the Content_Checksum_flag if
/// `checksum`.
fn windowed(wd: u8, fcs: Option<u32>, checksum: bool) -> Vec<u8> {
    let mut h = MAGIC.to_vec();
    h.push(if fcs.is_some() { 2 << 6 } else { 0 } | u8::from(checksum) << 2);
    h.push(wd);
    h.extend(fcs.into_iter().flat_map(u32::to_le_bytes));
    h
}

/// A single-segment frame header of the 8-byte Frame_Content_Size `fcs`:
/// its Window_Size is `fcs`.
fn single_segment(fcs: u64) -> Vec<u8> {
    let mut h = MAGIC.to_vec();
    h.push(3 << 6 | 1 << 5);
    h.extend(fcs.to_le_bytes());
    h
}

/// The Window_Descriptor of window log `log` and mantissa `mantissa`.
fn wd(log: u8, mantissa: u8) -> u8 {
    (log - 10) << 3 | mantissa
}

/// "hi" in one raw block.
const HI: &[Block] = &[(0, 2, b"hi")];

/// The Content_Checksum of `content`, from libzstd's frame of it.
fn checksum_of(content: &[u8]) -> [u8; 4] {
    let f = c_compress2(content, &[(sys::ZSTD_cParameter::ZSTD_c_checksumFlag, 1)]);
    f[f.len() - 4..].try_into().unwrap()
}

/// Our streaming outcome on `input` with `window_log_max`, fed at most
/// `chunk` bytes of input and `room` bytes of output a call, the same at
/// both SIMD levels, and libzstd's streaming decoder's with
/// ZSTD_d_windowLogMax `window_log_max` (unset for 0) fed alike, after
/// checking that both refuse the window (`refuses_window`) or neither
/// does.
fn streams(
    what: &str,
    input: &[u8],
    chunk: usize,
    room: usize,
    window_log_max: u32,
) -> (
    Result<Vec<u8>, String>,
    Result<Vec<u8>, sys::ZSTD_ErrorCode>,
) {
    let lib = c_streaming(input, chunk, room, window_log_max as i32, None);
    let mut ours = Vec::new();
    for simd in [false, true] {
        let mut d = Decompressor::with_options(&DecodeOptions {
            min_parallel_blocks: usize::MAX,
            simd,
            window_log_max,
        });
        ours.push(stream_with(&mut d, input, chunk, room));
    }
    let got = ours.pop().unwrap();
    assert!(ours[0] == got, "{what}: the SIMD levels disagree");
    let at = format!("{what}, window_log_max {window_log_max}, chunk {chunk}, room {room}");
    assert!(
        refuses_window(&got) == (lib == Err(WINDOW_TOO_LARGE)),
        "{at}: ours gives {} where libzstd gives {:?}",
        outcome(&got),
        lib.map(|c| c.len())
    );
    (got, lib)
}

/// `streams`, checking also that both give the same content or both fail.
fn stream_both(
    what: &str,
    input: &[u8],
    chunk: usize,
    room: usize,
    window_log_max: u32,
) -> Result<Vec<u8>, String> {
    let (got, lib) = streams(what, input, chunk, room, window_log_max);
    assert!(
        got.as_ref().ok() == lib.as_ref().ok(),
        "{what}, window_log_max {window_log_max}, chunk {chunk}, room {room}: ours gives {} \
         where libzstd gives {:?}",
        outcome(&got),
        lib.map(|c| c.len())
    );
    got
}

/// One-shot decoding of `input` takes it whatever the window limit, as
/// libzstd's ZSTD_decompressDCtx does with ZSTD_d_windowLogMax at its
/// least: `decompress`, `decompress_with_options` and a `Decompressor`'s
/// own `decompress`, with the limit at 10.
fn assert_one_shot_takes(what: &str, input: &[u8], content: &[u8]) {
    // SAFETY: the context is used only here, with a buffer of its size.
    let lib = unsafe {
        let dctx = sys::ZSTD_createDCtx();
        let r = sys::ZSTD_DCtx_setParameter(dctx, sys::ZSTD_dParameter::ZSTD_d_windowLogMax, 10);
        assert_eq!(sys::ZSTD_isError(r), 0);
        let mut out = vec![0u8; content.len() + 1];
        let n = sys::ZSTD_decompressDCtx(
            dctx,
            out.as_mut_ptr().cast(),
            out.len(),
            input.as_ptr().cast(),
            input.len(),
        );
        sys::ZSTD_freeDCtx(dctx);
        assert_eq!(sys::ZSTD_isError(n), 0, "{what}: libzstd one-shot");
        out.truncate(n);
        out
    };
    assert_eq!(lib, content, "{what}: libzstd one-shot");
    let opts = DecodeOptions {
        min_parallel_blocks: usize::MAX,
        simd: true,
        window_log_max: 10,
    };
    let mut d = Decompressor::with_options(&opts);
    for (got, how) in [
        (decompress(input), "decompress"),
        (
            decompress_with_options(input, &opts),
            "decompress_with_options",
        ),
        (d.decompress(input), "Decompressor::decompress"),
    ] {
        assert!(
            got.as_deref() == Ok(content),
            "{what}: {how} gives {}",
            outcome(&got)
        );
    }
}

/// By default the limit is window log 27: Window_Size `1 << 27` is taken,
/// the next one, at window log 27 and mantissa 1, and window log 28 are
/// refused, at every input piece size; a limit of 28 takes window log 28
/// and a limit of 27 refuses mantissa 1; one-shot decoding takes them all.
#[test]
fn default_limit_is_window_log_27() {
    for (log, mantissa, window_log_max, taken) in [
        (27, 0, 0, true),
        (27, 1, 0, false),
        (28, 0, 0, false),
        (31, 7, 0, false),
        (27, 0, 27, true),
        (27, 1, 27, false),
        (28, 0, 28, true),
        (28, 1, 28, false),
    ] {
        let f = frame(&windowed(wd(log, mantissa), None, false), HI, None);
        let what = format!("window log {log} mantissa {mantissa} limit {window_log_max}");
        for chunk in STREAM_CHUNKS {
            let got = stream_both(&what, &f, chunk, stream_room(chunk), window_log_max);
            if taken {
                assert_eq!(got.as_deref(), Ok(&b"hi"[..]), "{what} chunk {chunk}");
            } else {
                assert!(
                    is_window_limit(&got),
                    "{what} chunk {chunk}: {}",
                    outcome(&got)
                );
            }
        }
        assert_one_shot_takes(&what, &f, b"hi");
    }
}

/// `DecompressReader` refuses a frame above the limit with `InvalidData`.
#[test]
fn reader_refuses_above_the_limit() {
    for (log, taken) in [(27, true), (28, false)] {
        let f = frame(&windowed(wd(log, 0), None, false), HI, None);
        let mut content = Vec::new();
        let got = DecompressReader::new(&f[..])
            .read_to_end(&mut content)
            .map(|_| content)
            .map_err(|e| {
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
                e.to_string()
            });
        if taken {
            assert_eq!(got.as_deref(), Ok(&b"hi"[..]), "window log {log}");
        } else {
            assert!(is_window_limit(&got), "window log {log}: {}", outcome(&got));
        }
    }
}

/// Every Window_Descriptor, without and with a Frame_Content_Size, at the
/// default limit and at limits set from the least to the largest: ours and
/// libzstd's streaming decoders refuse the same windows at every input
/// piece size and output room. A Window_Size up to the limit, `1 << n` for
/// window_log_max `n`, is taken, and so is a larger one in a frame that
/// one call has whole with room for its content; window logs above
/// `ZSTD_WINDOWLOG_MAX` are refused by the header.
#[test]
fn every_window_descriptor_at_every_limit() {
    for descriptor in 0..=u8::MAX {
        let log = 10 + u32::from(descriptor >> 3);
        let window = (1u64 << log) / 8 * (8 + u64::from(descriptor & 7));
        for fcs in [None, Some(2)] {
            let f = frame(&windowed(descriptor, fcs, false), HI, None);
            for window_log_max in [0, 10, 20, 27, 28, WINDOW_LOG_MAX] {
                let limit = match window_log_max {
                    0 => (1 << 27) + 1,
                    n => 1 << n,
                };
                for chunk in STREAM_CHUNKS {
                    for room in [1, stream_room(chunk)] {
                        let what = format!("descriptor {descriptor:#x}, content size {fcs:?}");
                        let got = stream_both(&what, &f, chunk, room, window_log_max);
                        if log <= WINDOW_LOG_MAX {
                            assert_eq!(
                                is_window_limit(&got),
                                window > limit && !(fcs.is_some() && chunk >= f.len() && room >= 2),
                                "{what}, window_log_max {window_log_max}, chunk {chunk}, \
                                 room {room}: {}",
                                outcome(&got)
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A single-segment frame's Window_Size is its Frame_Content_Size, and the
/// default limit is libzstd's `ZSTD_MAXWINDOWSIZE_DEFAULT`, `(1 << 27) + 1`:
/// content of `(1 << 27) + 1` bytes is taken, one byte more refused, while
/// window_log_max 27 refuses the former too, as ZSTD_d_windowLogMax 27
/// does. Only the header and a first RLE block are given, the frame's
/// verdict being taken at its header.
#[test]
fn single_segment_limit_is_one_past_window_log_27() {
    for (fcs, window_log_max, taken) in [
        ((1 << 27) + 1, 0, true),
        ((1 << 27) + 2, 0, false),
        (1 << 27, 27, true),
        ((1 << 27) + 1, 27, false),
        ((1 << 31) + 1, WINDOW_LOG_MAX, false),
    ] {
        let rle: &[Block] = &[(1, 1 << 17, b"x"), (1, 1 << 17, b"x")];
        let mut f = frame(&single_segment(fcs), rle, None);
        // The first block is not the last.
        f.truncate(f.len() - 4);
        let what = format!("single segment {fcs}");
        for chunk in STREAM_CHUNKS {
            let got = stream_both(&what, &f, chunk, stream_room(chunk), window_log_max);
            assert_eq!(
                is_window_limit(&got),
                !taken,
                "{what}, window_log_max {window_log_max}, chunk {chunk}: {}",
                outcome(&got)
            );
        }
    }
}

/// libzstd decodes a frame in one pass, without the limit, when one call
/// has all of it, by ZSTD_findFrameCompressedSize, and room for its
/// Frame_Content_Size: so does ours, at window log 28 under the default
/// limit. A frame cut short, a header that came in pieces, too little
/// room, or a walk stopped by a reserved block type keep the limit; a
/// Block_Size above Block_Maximum_Size does not stop the walk.
#[test]
fn whole_frames_skip_the_limit() {
    let big = wd(28, 0);
    let hi = frame(&windowed(big, Some(2), false), HI, None);
    let checked = frame(&windowed(big, Some(2), true), HI, Some(checksum_of(b"hi")));
    let small = frame(&windowed(wd(10, 0), Some(2), false), HI, None);
    let mut skippable = vec![0x50, 0x2a, 0x4d, 0x18, 3, 0, 0, 0, 1, 2, 3];
    skippable.extend_from_slice(&hi);
    let two = [small.clone(), hi.clone()].concat();
    let rle = frame(&windowed(big, Some(1000), false), &[(1, 1000, b"r")], None);
    let reserved = frame(&windowed(big, Some(2), false), &[(3, 2, b"hi")], None);
    let raw = vec![7u8; (1 << 17) + 1];
    let oversized = frame(
        &windowed(big, Some(raw.len() as u32), false),
        &[(0, raw.len() as u32, &raw)],
        None,
    );
    let header_len = windowed(big, Some(2), false).len();
    for (name, input, chunk, room, taken) in [
        ("whole", &hi, usize::MAX, 1 << 16, true),
        (
            "whole, room for the content alone",
            &hi,
            usize::MAX,
            2,
            true,
        ),
        ("whole, one byte short of room", &hi, usize::MAX, 1, false),
        (
            "in pieces larger than the frame",
            &hi,
            hi.len(),
            1 << 16,
            true,
        ),
        (
            "in pieces one byte short",
            &hi,
            hi.len() - 1,
            1 << 16,
            false,
        ),
        ("header in pieces", &hi, header_len - 1, 1 << 16, false),
        ("with its checksum", &checked, usize::MAX, 1 << 16, true),
        (
            "after a skippable frame",
            &skippable,
            usize::MAX,
            1 << 16,
            true,
        ),
        ("after a small frame", &two, usize::MAX, 1 << 16, true),
        ("an RLE block", &rle, usize::MAX, 1000, true),
        (
            "a reserved block type",
            &reserved,
            usize::MAX,
            1 << 16,
            false,
        ),
    ] {
        let got = stream_both(name, input, chunk, room, 0);
        assert_eq!(is_window_limit(&got), !taken, "{name}: {}", outcome(&got));
    }
    // Past the limit, ours decodes the block as one-shot decoding does, and
    // refuses it by RFC 8878; libzstd's one-shot decoder takes a raw block
    // past Block_Maximum_Size.
    let (got, lib) = streams(
        "a block above Block_Maximum_Size",
        &oversized,
        usize::MAX,
        1 << 18,
        0,
    );
    assert_eq!(
        got,
        Err("Block size 131073 exceeds Block_Maximum_Size 131072".to_string())
    );
    assert_eq!(lib, Ok(raw));
    let cut = &checked[..checked.len() - 1];
    let got = stream_both("checksum cut short", cut, usize::MAX, 1 << 16, 0);
    assert!(
        is_window_limit(&got),
        "checksum cut short: {}",
        outcome(&got)
    );
    for (name, input, content) in [
        ("whole", &hi, &b"hi"[..]),
        ("with its checksum", &checked, b"hi"),
        ("an RLE block", &rle, &[b'r'; 1000]),
    ] {
        assert_one_shot_takes(name, input, content);
    }
}

/// A window limit libzstd's ZSTD_DCtx_setParameter refuses
/// (`parameter_outOfBound`) panics in `Decompressor::with_options`; 0 is
/// the default in both.
#[test]
fn window_log_max_bounds_match_libzstd() {
    for window_log_max in 0..=40u32 {
        // SAFETY: the context is used only here.
        let lib = unsafe {
            let dctx = sys::ZSTD_createDCtx();
            let r = sys::ZSTD_DCtx_setParameter(
                dctx,
                sys::ZSTD_dParameter::ZSTD_d_windowLogMax,
                window_log_max as i32,
            );
            sys::ZSTD_freeDCtx(dctx);
            sys::ZSTD_isError(r) == 0
        };
        let ours = std::panic::catch_unwind(|| {
            Decompressor::with_options(&DecodeOptions {
                min_parallel_blocks: usize::MAX,
                simd: true,
                window_log_max,
            })
        })
        .is_ok();
        assert_eq!(ours, lib, "window_log_max {window_log_max}");
    }
}
