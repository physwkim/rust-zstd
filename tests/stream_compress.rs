//! Streaming compression (`Compressor::compress_stream`).
//!
//! Self-consistency: without `Flush`, a stream of any chunking with the
//! size pledged is our one-shot frame byte for byte, and without a pledged
//! size the frame does not depend on the chunking. With flushes: round trip
//! through our decoder and libzstd's (one-shot and streaming), and the size
//! gate against libzstd's `ZSTD_compressStream2` flushing at the same
//! points. The full corpus grid is ignored by default (release build):
//!
//! ```text
//! cargo nextest run --release --test stream_compress --run-ignored all
//! ```

mod common;

use common::{assert_gate, assert_round_trip, datasets, MIB};
use rust_zstd::compress::{CompressError, CompressOptions, Compressor, EndDirective};
use zstd::zstd_safe::zstd_sys as sys;

const CHUNKS: [usize; 4] = [1, 7, 1013, 64 << 10];
const GRID_LEVELS: [i32; 5] = [1, 3, 9, 12, 19];

fn opts(level: i32) -> CompressOptions {
    CompressOptions {
        level,
        ..CompressOptions::default()
    }
}

/// `data` through `compress_stream` in `chunk`-byte calls of `Continue`,
/// with `Flush` after every input position in `flushes`, then `End`, into
/// an output buffer of `dst_size` bytes drained after every call.
fn stream(
    cctx: &mut Compressor,
    data: &[u8],
    chunk: usize,
    flushes: &[usize],
    dst_size: usize,
) -> Vec<u8> {
    let mut frame = Vec::new();
    let mut dst = vec![0u8; dst_size];
    let mut call = |src: &[u8], op: EndDirective| {
        let mut src_pos = 0;
        loop {
            let mut dst_pos = 0;
            let left = cctx
                .compress_stream(src, &mut src_pos, &mut dst, &mut dst_pos, op)
                .expect("compress_stream");
            frame.extend_from_slice(&dst[..dst_pos]);
            let done = match op {
                EndDirective::Continue => src_pos == src.len() && dst_pos < dst.len(),
                _ => left == 0,
            };
            if done {
                assert_eq!(src_pos, src.len());
                return;
            }
        }
    };
    let mut start = 0;
    let mut cuts: Vec<usize> = (chunk..data.len()).step_by(chunk).collect();
    cuts.extend_from_slice(flushes);
    cuts.sort_unstable();
    cuts.dedup();
    for cut in cuts {
        call(&data[start..cut], EndDirective::Continue);
        if flushes.contains(&cut) {
            call(&[], EndDirective::Flush);
        }
        start = cut;
    }
    call(&data[start..], EndDirective::Continue);
    call(&[], EndDirective::End);
    frame
}

/// `stream` with the size pledged first.
fn stream_pledged(cctx: &mut Compressor, data: &[u8], chunk: usize, flushes: &[usize]) -> Vec<u8> {
    cctx.set_pledged_src_size(Some(data.len() as u64)).unwrap();
    stream(cctx, data, chunk, flushes, 1 << 17)
}

/// The no-flush gate on `data` at `level`: pledged streams equal the
/// one-shot frame, unpledged streams equal each other and round-trip.
fn check_no_flush(name: &str, data: &[u8], level: i32, chunks: &[usize]) {
    let one_shot = rust_zstd::compress(data, level);
    let mut cctx = Compressor::new(opts(level));
    let mut unpledged: Option<Vec<u8>> = None;
    for &chunk in chunks {
        let what = format!("{name} L{level} chunk {chunk}");
        let frame = stream_pledged(&mut cctx, data, chunk, &[]);
        assert!(
            frame == one_shot,
            "{what}: pledged stream != one-shot frame"
        );
        let frame = stream(&mut cctx, data, chunk, &[], 1 << 17);
        match &unpledged {
            None => {
                assert_round_trip(&what, data, &frame);
                unpledged = Some(frame);
            }
            Some(first) => assert!(&frame == first, "{what}: unpledged stream differs"),
        }
    }
}

/// libzstd's `ZSTD_compressStream2` frame of `data` at `level` with the
/// size pledged or not, cut and flushed as [`stream`] does.
fn c_stream(data: &[u8], level: i32, pledged: bool, chunk: usize, flushes: &[usize]) -> Vec<u8> {
    // SAFETY: the context is used only here; every buffer outlives the
    // calls that reference it.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let r =
            sys::ZSTD_CCtx_setParameter(cctx, sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level);
        assert_eq!(sys::ZSTD_isError(r), 0);
        if pledged {
            let r = sys::ZSTD_CCtx_setPledgedSrcSize(cctx, data.len() as u64);
            assert_eq!(sys::ZSTD_isError(r), 0);
        }
        let mut frame = Vec::new();
        let mut dst = vec![0u8; sys::ZSTD_CStreamOutSize()];
        let mut call = |src: &[u8], op: sys::ZSTD_EndDirective| {
            let mut input = sys::ZSTD_inBuffer {
                src: src.as_ptr().cast(),
                size: src.len(),
                pos: 0,
            };
            loop {
                let mut output = sys::ZSTD_outBuffer {
                    dst: dst.as_mut_ptr().cast(),
                    size: dst.len(),
                    pos: 0,
                };
                let left = sys::ZSTD_compressStream2(cctx, &mut output, &mut input, op);
                assert_eq!(sys::ZSTD_isError(left), 0, "ZSTD_compressStream2");
                frame.extend_from_slice(&dst[..output.pos]);
                let done = match op {
                    sys::ZSTD_EndDirective::ZSTD_e_continue => input.pos == input.size,
                    _ => left == 0,
                };
                if done {
                    return;
                }
            }
        };
        let mut start = 0;
        let mut cuts: Vec<usize> = (chunk..data.len()).step_by(chunk).collect();
        cuts.extend_from_slice(flushes);
        cuts.sort_unstable();
        cuts.dedup();
        for cut in cuts {
            call(&data[start..cut], sys::ZSTD_EndDirective::ZSTD_e_continue);
            if flushes.contains(&cut) {
                call(&[], sys::ZSTD_EndDirective::ZSTD_e_flush);
            }
            start = cut;
        }
        call(&data[start..], sys::ZSTD_EndDirective::ZSTD_e_continue);
        call(&[], sys::ZSTD_EndDirective::ZSTD_e_end);
        sys::ZSTD_freeCCtx(cctx);
        frame
    }
}

/// The flush gate: round trip through both decoders, libzstd's streaming
/// decoder included, and the size gate against libzstd flushing at the
/// same points.
fn check_flushes(name: &str, data: &[u8], level: i32, pledged: bool, flushes: &[usize]) {
    let chunk = 64 << 10;
    let mut cctx = Compressor::new(opts(level));
    let ours = if pledged {
        stream_pledged(&mut cctx, data, chunk, flushes)
    } else {
        stream(&mut cctx, data, chunk, flushes, 1 << 17)
    };
    let what = format!(
        "{name} L{level} pledged {pledged} flushes {}",
        flushes.len()
    );
    let decoded = zstd::stream::decode_all(&ours[..]).expect(&what);
    assert!(decoded == data, "{what}: libzstd streaming decode differs");
    let lib = c_stream(data, level, pledged, chunk, flushes);
    assert_gate(&what, data, &ours, &lib);
}

/// Flush points: every `step` bytes, plus a few odd ones.
fn flush_points(len: usize, step: usize) -> Vec<usize> {
    let mut points: Vec<usize> = (step..len).step_by(step).collect();
    points.extend([1, 17, len / 3].into_iter().filter(|&p| p > 0 && p < len));
    points.sort_unstable();
    points.dedup();
    points
}

/// Dataset `i` of [`datasets`].
fn set(i: usize) -> Vec<u8> {
    datasets().swap_remove(i).data
}

fn prefix(data: &[u8], len: usize) -> &[u8] {
    &data[..len.min(data.len())]
}

#[test]
fn no_flush_stream_is_the_one_shot_frame() {
    for set in datasets() {
        let data = prefix(&set.data, 200 << 10);
        for level in [1, 3, 9] {
            check_no_flush(set.name, data, level, &[7, 1013, 64 << 10]);
        }
    }
    let words = &set(2);
    check_no_flush("words 4k", prefix(words, 4096), 19, &CHUNKS);
}

/// The buffer moves its last window down several times: at level 1 the
/// window is 512 KiB, so a 4 MiB input slides past 1.25 MiB.
#[test]
fn sliding_buffer_keeps_the_one_shot_frame() {
    let src = &set(0)[..4 * MIB];
    check_no_flush("rust_src 4m", src, 1, &[1013, 64 << 10, 1 << 20]);
    let elf = &set(1)[..3 * MIB];
    check_no_flush("elf 3m", elf, 2, &[64 << 10, 300 << 10]);
}

#[test]
fn flushed_streams_pass_the_gate() {
    for set in datasets() {
        let data = prefix(&set.data, 300 << 10);
        for level in [1, 3, 9] {
            for pledged in [false, true] {
                check_flushes(
                    set.name,
                    data,
                    level,
                    pledged,
                    &flush_points(data.len(), 50_000),
                );
            }
        }
    }
}

/// A small output buffer: every call's output is drained over many calls,
/// and the frame is the same.
#[test]
fn small_output_buffer_gives_the_same_frame() {
    let data = &set(0);
    let data = prefix(data, 600 << 10);
    let flushes = flush_points(data.len(), 100_000);
    for level in [1, 12] {
        let mut cctx = Compressor::new(opts(level));
        let big = stream(&mut cctx, data, 4096, &flushes, 1 << 17);
        for dst_size in [1, 13, 4096] {
            let small = stream(&mut cctx, data, 4096, &flushes, dst_size);
            assert!(small == big, "L{level} dst {dst_size}: frame differs");
        }
    }
}

/// The empty frame: one `End` call, or `Continue` then `End`, pledged or
/// not, round-trips; pledged it is the one-shot frame. A flush with no
/// input buffered writes nothing.
#[test]
fn empty_frames() {
    let mut cctx = Compressor::new(opts(3));
    let one_shot = rust_zstd::compress(&[], 3);
    assert_eq!(stream_pledged(&mut cctx, &[], 1, &[]), one_shot);
    let frame = stream(&mut cctx, &[], 1, &[], 64);
    assert_round_trip("empty unpledged", &[], &frame);
    let mut dst = [0u8; 64];
    let (mut src_pos, mut dst_pos) = (0, 0);
    let left = cctx
        .compress_stream(
            &[],
            &mut src_pos,
            &mut dst,
            &mut dst_pos,
            EndDirective::Flush,
        )
        .unwrap();
    assert_eq!((left, dst_pos), (0, 0));
}

/// `End` as the first call is the one-shot frame, the size pledged by the
/// call; `End` after a flush adds libzstd's empty last block.
#[test]
fn first_call_end_and_end_after_flush() {
    let data = &set(0);
    let data = prefix(data, 100 << 10);
    let mut cctx = Compressor::new(opts(5));
    let mut dst = vec![0u8; 1 << 18];
    let (mut src_pos, mut dst_pos) = (0, 0);
    let left = cctx
        .compress_stream(
            data,
            &mut src_pos,
            &mut dst,
            &mut dst_pos,
            EndDirective::End,
        )
        .unwrap();
    assert_eq!(left, 0);
    assert_eq!(&dst[..dst_pos], &rust_zstd::compress(data, 5)[..]);
    let frame = stream(&mut cctx, data, 1 << 20, &[data.len()], 1 << 18);
    assert_eq!(
        &frame[frame.len() - 3..],
        &[1, 0, 0],
        "empty last RAW block"
    );
    assert_round_trip("end after flush", data, &frame);
}

/// The pledged size is checked like libzstd's `srcSize_wrong`: more input
/// than pledged fails the call that brings it, consuming none of it; less
/// fails `End`. Either way the stream fails until reset, and the pledge
/// is only accepted before a frame starts.
#[test]
fn wrong_pledged_size_is_an_error() {
    let data = &set(0);
    let data = prefix(data, 10_000);
    let mut dst = vec![0u8; 1 << 16];
    let mut cctx = Compressor::new(opts(3));
    cctx.set_pledged_src_size(Some(5000)).unwrap();
    let (mut src_pos, mut dst_pos) = (0, 0);
    let r = cctx.compress_stream(
        &data[..4000],
        &mut src_pos,
        &mut dst,
        &mut dst_pos,
        EndDirective::Continue,
    );
    assert!(r.is_ok());
    assert_eq!(
        cctx.set_pledged_src_size(None),
        Err(CompressError::StageWrong)
    );
    let mut more = 0;
    let r = cctx.compress_stream(
        &data[4000..],
        &mut more,
        &mut dst,
        &mut dst_pos,
        EndDirective::Continue,
    );
    assert_eq!(
        r,
        Err(CompressError::SrcSizeWrong {
            pledged: 5000,
            consumed: 10_000
        })
    );
    assert_eq!(more, 0);
    let r = cctx.compress_stream(&[], &mut more, &mut dst, &mut dst_pos, EndDirective::End);
    assert_eq!(r, Err(CompressError::StageWrong));
    cctx.reset_stream();

    for first_end in [false, true] {
        cctx.set_pledged_src_size(Some(5000)).unwrap();
        let (mut src_pos, mut dst_pos) = (0, 0);
        let op = if first_end {
            EndDirective::End
        } else {
            EndDirective::Continue
        };
        let src = &data[..4000];
        let r = cctx.compress_stream(src, &mut src_pos, &mut dst, &mut dst_pos, op);
        let r = if first_end {
            r
        } else {
            assert!(r.is_ok());
            cctx.compress_stream(&[], &mut 0, &mut dst, &mut dst_pos, EndDirective::End)
        };
        let consumed = 4000;
        assert_eq!(
            r,
            Err(CompressError::SrcSizeWrong {
                pledged: 5000,
                consumed
            })
        );
        cctx.reset_stream();
    }
    // After a reset the context compresses normally.
    let frame = stream_pledged(&mut cctx, data, 999, &[]);
    assert_eq!(frame, rust_zstd::compress(data, 3));
}

/// A `job_size` stream that would be multithreaded is not implemented.
#[test]
fn multithreaded_streaming_is_unsupported() {
    let mut cctx = Compressor::new(CompressOptions {
        job_size: Some(0),
        ..opts(3)
    });
    let mut dst = vec![0u8; 1 << 16];
    let (mut src_pos, mut dst_pos) = (0, 0);
    let r = cctx.compress_stream(
        b"abc",
        &mut src_pos,
        &mut dst,
        &mut dst_pos,
        EndDirective::Continue,
    );
    assert!(matches!(r, Err(CompressError::Unsupported(_))));
    cctx.reset_stream();
    // Pledged at most JOBSIZE_MIN, libzstd runs it single-threaded.
    let data = &set(0);
    let data = prefix(data, 300 << 10);
    let frame = stream_pledged(&mut cctx, data, 4096, &[]);
    let one_shot = rust_zstd::compress_with(
        data,
        &CompressOptions {
            job_size: Some(0),
            ..opts(3)
        },
    );
    assert_eq!(frame, one_shot);
}

/// The corpus grid of the self-consistency gate: chunk sizes 1, 7, 1013
/// and 64 KiB at levels 1, 3, 9, 12 and 19 on every dataset.
#[test]
#[ignore]
fn corpus_grid() {
    for set in datasets() {
        for level in GRID_LEVELS {
            check_no_flush(set.name, &set.data, level, &CHUNKS);
        }
    }
}

/// The corpus flush gate at the grid levels.
#[test]
#[ignore]
fn corpus_flush_grid() {
    for set in datasets() {
        for level in GRID_LEVELS {
            let flushes = flush_points(set.data.len(), 1 << 20);
            for pledged in [false, true] {
                check_flushes(set.name, &set.data, level, pledged, &flushes);
            }
        }
    }
}

/// The buffer slides at every strategy: windows of 2 to 8 MiB over 12 and
/// 20 MiB inputs (the corpus grid slides only at levels 1 and 3).
#[test]
#[ignore]
fn sliding_grid() {
    let mut data = set(0);
    data.extend_from_slice(&set(1));
    data.extend_from_within(..4 * MIB);
    for level in [5, 7, 12, 13, 16] {
        check_no_flush("rust+elf 12m", &data[..12 * MIB], level, &[1013, 64 << 10]);
    }
    check_no_flush("rust+elf+rust 20m", &data, 19, &[64 << 10]);
}
