//! Shared dataset construction for the decoder verification and benchmark
//! tests. Datasets are cached under `target/decoder-datasets/` so that every
//! run (before and after an optimization) works on byte-identical inputs and
//! therefore byte-identical libzstd streams.

#![allow(dead_code)]

use rust_zstd::decode::{decompress_with_options, DecodeDict, DecodeOptions};
use rust_zstd::{DecompressReader, Decompressor};
use std::cell::RefCell;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use zstd::zstd_safe::zstd_sys as sys;

pub const MIB: usize = 1024 * 1024;

pub struct Dataset {
    pub name: &'static str,
    pub data: Vec<u8>,
}

fn cache_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/decoder-datasets");
    std::fs::create_dir_all(&dir).expect("create dataset cache dir");
    dir
}

fn cached(name: &str, build: impl FnOnce() -> Vec<u8>) -> Vec<u8> {
    let path = cache_dir().join(name);
    if let Ok(bytes) = std::fs::read(&path) {
        return bytes;
    }
    let bytes = build();
    std::fs::write(&path, &bytes).expect("write dataset cache");
    bytes
}

/// Deterministic 64-bit LCG (Knuth MMIX constants); returns the high byte.
pub fn lcg_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((state >> 56) as u8);
    }
    out
}

/// Concatenation of `.rs` files (sorted by path) found under the cargo
/// registry source cache, truncated to 8 MiB. Falls back to repeating the
/// crate's own sources if the registry is unavailable.
fn rust_source_8m() -> Vec<u8> {
    let target = 8 * MIB;
    let mut out = Vec::with_capacity(target);
    let own = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut roots: Vec<PathBuf> = vec![own];
    if let Some(home) = std::env::var_os("HOME") {
        let registry = Path::new(&home).join(".cargo/registry/src");
        if registry.is_dir() {
            roots.push(registry);
        }
    }
    let mut files = Vec::new();
    for root in roots {
        collect_rs(&root, &mut files);
    }
    files.sort();
    for f in files {
        if out.len() >= target {
            break;
        }
        if let Ok(bytes) = std::fs::read(&f) {
            out.extend_from_slice(&bytes);
        }
    }
    if out.is_empty() {
        panic!("no .rs sources found");
    }
    while out.len() < target {
        let n = out.len();
        out.extend_from_within(..n.min(target - n));
    }
    out.truncate(target);
    out
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// The running test binary itself (a real ELF), capped at 8 MiB. Cached on
/// first use so that later builds of the test binary do not change the input.
fn elf_8m() -> Vec<u8> {
    let exe = std::env::current_exe().expect("current_exe");
    let mut bytes = std::fs::read(exe).expect("read current_exe");
    bytes.truncate(8 * MIB);
    bytes
}

/// 1 MiB of dictionary words separated by spaces, chosen with an LCG.
fn words_1m() -> Vec<u8> {
    let dict = std::fs::read_to_string("/usr/share/dict/words").unwrap_or_default();
    let words: Vec<&str> = dict.lines().filter(|w| !w.is_empty()).collect();
    let mut out = Vec::with_capacity(MIB + 64);
    let mut state = 0x9E3779B97F4A7C15u64;
    while out.len() < MIB {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if words.is_empty() {
            // Synthetic fallback: short pseudo-words.
            let len = 2 + ((state >> 58) as usize % 8);
            for i in 0..len {
                out.push(b'a' + ((state >> (i * 5)) as u8 % 26));
            }
        } else {
            let w = words[(state >> 33) as usize % words.len()];
            out.extend_from_slice(w.as_bytes());
        }
        out.push(b' ');
    }
    out.truncate(MIB);
    out
}

/// All datasets named in the task, in a fixed order.
pub fn datasets() -> Vec<Dataset> {
    vec![
        Dataset {
            name: "rust_src_8m",
            data: cached("rust_src_8m.bin", rust_source_8m),
        },
        Dataset {
            name: "elf_8m",
            data: cached("elf_8m.bin", elf_8m),
        },
        Dataset {
            name: "words_1m",
            data: cached("words_1m.bin", words_1m),
        },
        Dataset {
            name: "text_1m",
            data: b"The quick brown fox jumps over the lazy dog. Hello world! ".repeat(18000),
        },
        Dataset {
            name: "random_1m",
            data: lcg_bytes(MIB, 42),
        },
        Dataset {
            name: "zeros_1m",
            data: vec![0u8; MIB],
        },
        Dataset {
            name: "one_byte",
            data: vec![0x5A],
        },
        Dataset {
            name: "empty",
            data: Vec::new(),
        },
    ]
}

pub const LEVELS: [i32; 5] = [1, 3, 7, 11, 19];

/// libzstd one-shot compression (frame carries Frame_Content_Size).
pub fn zstd_bulk(data: &[u8], level: i32) -> Vec<u8> {
    zstd::bulk::compress(data, level).expect("libzstd bulk compress")
}

/// libzstd streaming compression (no Frame_Content_Size, window descriptor).
pub fn zstd_stream(data: &[u8], level: i32) -> Vec<u8> {
    zstd::stream::encode_all(data, level).expect("libzstd stream compress")
}

extern "C" {
    fn ZSTD_compressContinue(
        cctx: *mut sys::ZSTD_CCtx,
        dst: *mut u8,
        dst_capacity: usize,
        src: *const u8,
        src_size: usize,
    ) -> usize;
    fn ZSTD_compressEnd(
        cctx: *mut sys::ZSTD_CCtx,
        dst: *mut u8,
        dst_capacity: usize,
        src: *const u8,
        src_size: usize,
    ) -> usize;
    fn ZSTD_decompressBegin(dctx: *mut sys::ZSTD_DCtx) -> usize;
    fn ZSTD_nextSrcSizeToDecompress(dctx: *mut sys::ZSTD_DCtx) -> usize;
    fn ZSTD_nextInputType(dctx: *mut sys::ZSTD_DCtx) -> i32;
    fn ZSTD_decompressContinue(
        dctx: *mut sys::ZSTD_DCtx,
        dst: *mut u8,
        dst_capacity: usize,
        src: *const u8,
        src_size: usize,
    ) -> usize;
}

/// libzstd's stream frame of `src` without its input buffering, the frame
/// our streams are gated against: on a fresh context `setup` prepares, the
/// size pledged if `pledged`, an empty `ZSTD_compressStream2` call starts
/// the frame as a stream does, then each piece between `flushes` is
/// compressed whole by `ZSTD_compressContinue`, the last by
/// `ZSTD_compressEnd`. `ZSTD_compressStream2` itself compresses its input
/// buffer in 128 KiB units and wraps it, so its blocks (a pre-split block
/// never crosses a unit) and its fast finders' extDict mode depend on that
/// buffer, and ours by design do not.
pub fn c_stream_unbuffered(
    src: &[u8],
    pledged: bool,
    flushes: &[usize],
    setup: impl FnOnce(*mut sys::ZSTD_CCtx),
) -> Vec<u8> {
    // SAFETY: the context is used only here; every buffer outlives the
    // calls that reference it, with the capacity it is given.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        setup(cctx);
        if pledged {
            let r = sys::ZSTD_CCtx_setPledgedSrcSize(cctx, src.len() as u64);
            assert_eq!(sys::ZSTD_isError(r), 0);
        }
        let mut header = [0u8; 1];
        let mut output = sys::ZSTD_outBuffer {
            dst: header.as_mut_ptr().cast(),
            size: header.len(),
            pos: 0,
        };
        let mut input = sys::ZSTD_inBuffer {
            src: src.as_ptr().cast(),
            size: 0,
            pos: 0,
        };
        let r = sys::ZSTD_compressStream2(
            cctx,
            &mut output,
            &mut input,
            sys::ZSTD_EndDirective::ZSTD_e_continue,
        );
        assert_eq!(sys::ZSTD_isError(r), 0, "ZSTD_compressStream2");
        assert_eq!(output.pos, 0, "the start writes nothing");
        let mut frame = Vec::new();
        let mut start = 0;
        for (i, end) in flushes.iter().copied().chain([src.len()]).enumerate() {
            let piece = &src[start..end];
            // Room for the frame header, the empty last block and the
            // checksum besides the piece's blocks.
            let room = sys::ZSTD_compressBound(piece.len()) + 32;
            let at = frame.len();
            frame.resize(at + room, 0);
            let compress = if i == flushes.len() {
                ZSTD_compressEnd
            } else {
                ZSTD_compressContinue
            };
            let n = compress(
                cctx,
                frame.as_mut_ptr().add(at),
                room,
                piece.as_ptr(),
                piece.len(),
            );
            assert_eq!(
                sys::ZSTD_isError(n),
                0,
                "{}",
                std::ffi::CStr::from_ptr(sys::ZSTD_getErrorName(n)).to_string_lossy()
            );
            frame.truncate(at + n);
            start = end;
        }
        sys::ZSTD_freeCCtx(cctx);
        frame
    }
}

/// One block: type (0 RAW, 1 RLE, 2 COMPRESSED), decompressed size and
/// size in the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub ty: u32,
    pub size: usize,
    pub c_size: usize,
}

/// Decode a single-frame `frame` with libzstd's buffer-less decoder,
/// returning its blocks and content.
pub fn frame_blocks(frame: &[u8], content_size: usize) -> (Vec<Block>, Vec<u8>) {
    // ZSTD_nextInputType: ZSTDnit_blockHeader, ZSTDnit_block,
    // ZSTDnit_lastBlock.
    const BLOCK_HEADER: i32 = 1;
    const BLOCK: i32 = 2;
    const LAST_BLOCK: i32 = 3;
    let mut out = vec![0u8; content_size];
    let mut list = Vec::new();
    let (mut ip, mut op) = (0usize, 0usize);
    let mut header = None;
    // SAFETY: every call gets the context it created and in-bounds buffers.
    unsafe {
        let dctx = sys::ZSTD_createDCtx();
        assert_eq!(sys::ZSTD_isError(ZSTD_decompressBegin(dctx)), 0);
        loop {
            let n = ZSTD_nextSrcSizeToDecompress(dctx);
            if n == 0 {
                break;
            }
            let kind = ZSTD_nextInputType(dctx);
            if kind == BLOCK_HEADER {
                let h = u32::from_le_bytes([frame[ip], frame[ip + 1], frame[ip + 2], 0]);
                let ty = (h >> 1) & 3;
                header = Some((ty, 3 + if ty == 1 { 1 } else { (h >> 3) as usize }));
            }
            let r = ZSTD_decompressContinue(
                dctx,
                out.as_mut_ptr().add(op),
                out.len() - op,
                frame.as_ptr().add(ip),
                n,
            );
            assert_eq!(sys::ZSTD_isError(r), 0, "libzstd rejects the block at {ip}");
            if kind == BLOCK || kind == LAST_BLOCK {
                let (ty, c_size) = header.take().expect("block without header");
                list.push(Block {
                    ty,
                    size: r,
                    c_size,
                });
            }
            ip += n;
            op += r;
        }
        sys::ZSTD_freeDCtx(dctx);
    }
    assert_eq!(ip, frame.len());
    out.truncate(op);
    (list, out)
}

/// libzstd's frame for `data` through `ZSTD_compress2` with `params` set.
pub fn c_compress2(data: &[u8], params: &[(sys::ZSTD_cParameter, i32)]) -> Vec<u8> {
    // SAFETY: the context is used only here; buffers are sized by
    // ZSTD_compressBound.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        for &(param, value) in params {
            let r = sys::ZSTD_CCtx_setParameter(cctx, param, value);
            assert_eq!(sys::ZSTD_isError(r), 0, "set parameter {param:?}");
        }
        let mut out = vec![0u8; sys::ZSTD_compressBound(data.len())];
        let n = sys::ZSTD_compress2(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            data.as_ptr().cast(),
            data.len(),
        );
        assert_eq!(sys::ZSTD_isError(n), 0);
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

/// The encoder gate's size tolerance over libzstd's frame for the same
/// input and parameters: `lib / SIZE_SLACK_DIVISOR + SIZE_SLACK_BYTES`.
pub const SIZE_SLACK_DIVISOR: usize = 1000;
/// See [`SIZE_SLACK_DIVISOR`].
pub const SIZE_SLACK_BYTES: usize = 16;

/// The largest size the gate accepts against libzstd's `lib` bytes.
pub fn size_limit(lib: usize) -> usize {
    lib + lib / SIZE_SLACK_DIVISOR + SIZE_SLACK_BYTES
}

/// The gate's size check: `ours` bytes are at most [`size_limit`] of
/// libzstd's `lib` bytes; the error names the case, both sizes and the
/// delta.
pub fn check_size(what: &str, ours: usize, lib: usize) -> Result<(), String> {
    if ours <= size_limit(lib) {
        return Ok(());
    }
    Err(format!(
        "{what}: ours {ours} bytes, libzstd {lib}, delta +{} over the {} allowed",
        ours - lib,
        size_limit(lib) - lib
    ))
}

/// The gate's round trip: `frame` decodes to `src` through our decoder and
/// through libzstd's (window log up to `ZSTD_WINDOWLOG_MAX`).
pub fn assert_round_trip(what: &str, src: &[u8], frame: &[u8]) {
    assert!(
        rust_zstd::decompress(frame).expect(what) == src,
        "{what}: our decoder decodes another input"
    );
    let mut dctx = zstd::bulk::Decompressor::new().unwrap();
    let window_log_max = if cfg!(target_pointer_width = "64") {
        31
    } else {
        30
    };
    dctx.set_parameter(zstd::stream::raw::DParameter::WindowLogMax(window_log_max))
        .unwrap();
    assert!(
        dctx.decompress(frame, src.len()).expect(what) == src,
        "{what}: libzstd decodes another input"
    );
}

/// The encoder gate on our frame `ours` of `src` against libzstd's frame
/// `lib` for the same input and parameters: [`assert_round_trip`] (which
/// panics), then [`check_size`].
pub fn gate(what: &str, src: &[u8], ours: &[u8], lib: &[u8]) -> Result<(), String> {
    assert_round_trip(what, src, ours);
    check_size(what, ours.len(), lib.len())
}

/// [`gate`], panicking on a size failure.
pub fn assert_gate(what: &str, src: &[u8], ours: &[u8], lib: &[u8]) {
    if let Err(e) = gate(what, src, ours, lib) {
        panic!("{e}");
    }
}

/// Input chunk sizes the streaming decoder is checked at: 1, 7 and 4096
/// bytes, and the whole input in one piece.
pub const STREAM_CHUNKS: [usize; 4] = [1, 7, 4096, usize::MAX];

/// Decode `input` with `Decompressor::decompress_stream`, handing it at most
/// `chunk` bytes of input and `room` bytes to write in each call, until it
/// has read all of the input and written out all it decoded, then
/// `finish`.
pub fn decompress_streaming(
    input: &[u8],
    chunk: usize,
    room: usize,
    opts: &DecodeOptions,
) -> Result<Vec<u8>, String> {
    stream_with(&mut Decompressor::with_options(opts), input, chunk, room)
}

/// `decompress_streaming` with `d`, which must stand at the start of a
/// frame.
pub fn stream_with(
    d: &mut Decompressor,
    input: &[u8],
    chunk: usize,
    room: usize,
) -> Result<Vec<u8>, String> {
    let mut content = Vec::new();
    let mut out = vec![0u8; room];
    let mut pos = 0usize;
    loop {
        let src = &input[pos..input.len().min(pos.saturating_add(chunk))];
        let (mut read, mut written) = (0, 0);
        let hint = d.decompress_stream(src, &mut read, &mut out, &mut written)?;
        content.extend_from_slice(&out[..written]);
        pos += read;
        if read == 0 && written == 0 && hint != 0 {
            assert!(
                src.is_empty(),
                "no progress with {} bytes to read",
                src.len()
            );
            break;
        }
    }
    d.finish()?;
    Ok(content)
}

/// The output room `assert_stream_parity_at` gives with input chunks of
/// `chunk` bytes.
pub fn stream_room(chunk: usize) -> usize {
    chunk.min(1 << 16)
}

/// Output room for four blocks of 128 KiB, at which a call can take
/// several whole blocks to decode in parallel; below the 1 MiB that
/// decode_header_alloc holds decoding to, whose streaming checks
/// allocate it.
pub const PARALLEL_ROOM: usize = 1 << 19;

/// Feed `input` to `serial` and `parallel` alike, at most `chunk` bytes of
/// input and `room` bytes of output a call, as `stream_with` does, checking
/// that every call returns, reads and writes the same, and then that
/// `finish` gives the same.
pub fn assert_lockstep(
    what: &str,
    serial: &mut Decompressor,
    parallel: &mut Decompressor,
    input: &[u8],
    chunk: usize,
    room: usize,
) {
    let (mut a, mut b) = (vec![0u8; room], vec![0u8; room]);
    let mut pos = 0usize;
    for call in 0.. {
        let src = &input[pos..input.len().min(pos.saturating_add(chunk))];
        let (mut read, mut written, mut b_read, mut b_written) = (0, 0, 0, 0);
        let hint = serial.decompress_stream(src, &mut read, &mut a, &mut written);
        let b_hint = parallel.decompress_stream(src, &mut b_read, &mut b, &mut b_written);
        assert!(
            (&hint, read, &a[..written]) == (&b_hint, b_read, &b[..b_written]),
            "{what}, chunk {chunk}, room {room}, call {call}: serial gives {hint:?}, reading \
             {read} and writing {written}; parallel {b_hint:?}, reading {b_read} and writing \
             {b_written}"
        );
        pos += read;
        if hint.is_err() || read == 0 && written == 0 && hint != Ok(0) {
            break;
        }
    }
    assert_eq!(serial.finish(), parallel.finish(), "{what}: finish");
}

/// `assert_lockstep` of `input` at `chunk`, with the room `stream_room`
/// gives and with `PARALLEL_ROOM`, between serial decompressors and ones
/// that decode every frame's whole blocks in parallel, at SIMD level
/// `simd`, each with `dict` if given.
///
/// Only at chunks of 3 bytes or more, the least a whole block takes, and
/// at the detected SIMD level: the parallel decoder executes a block with
/// the serial one's code at either level.
#[cfg(feature = "parallel")]
pub fn assert_parallel_streams(
    what: &str,
    input: &[u8],
    chunk: usize,
    simd: bool,
    dict: Option<&DecodeDict>,
) {
    if chunk < 3 || !simd {
        return;
    }
    for room in [stream_room(chunk), PARALLEL_ROOM] {
        let [mut serial, mut parallel] = [usize::MAX, 1].map(|min_parallel_blocks| {
            let mut d = Decompressor::with_options(&DecodeOptions {
                min_parallel_blocks,
                min_parallel_bytes: 0,
                simd,
                window_log_max: 0,
            });
            d.set_dict(dict);
            d
        });
        assert_lockstep(what, &mut serial, &mut parallel, input, chunk, room);
    }
}

thread_local! {
    /// A serial decompressor per SIMD level, and the buffer its
    /// `decompress_into` calls fill, for every `assert_stream_parity_at`
    /// call on the thread, so each input decodes with the tables and
    /// buffers the inputs before it, rejected ones included, left.
    static REUSED: RefCell<[(Decompressor, Vec<u8>); 2]> = RefCell::new([false, true].map(|simd| {
        let d = Decompressor::with_options(&DecodeOptions {
            min_parallel_blocks: usize::MAX,
            min_parallel_bytes: 0,
            simd,
            window_log_max: 0,
        });
        (d, Vec::new())
    }));
}

/// A `decompress_into` result and the `dst` it left, as the result
/// `decompress` gives, checking that a failed call leaves `dst` empty.
pub fn into_result(got: Result<(), String>, dst: &[u8]) -> Result<Vec<u8>, String> {
    match got {
        Ok(()) => Ok(dst.to_vec()),
        Err(e) => {
            assert!(
                dst.is_empty(),
                "failed decompress_into left {} bytes",
                dst.len()
            );
            Err(e)
        }
    }
}

/// A decode outcome, for messages.
pub fn outcome(r: &Result<Vec<u8>, String>) -> String {
    match r {
        Ok(content) => format!("{} bytes", content.len()),
        Err(e) => format!("error {e:?}"),
    }
}

/// `ZSTD_WINDOWLOG_MAX`, the largest `DecodeOptions::window_log_max`.
pub const WINDOW_LOG_MAX: u32 = if usize::BITS == 32 { 30 } else { 31 };

/// Whether `r` is our streaming decoder's refusal of a frame whose
/// Window_Size is above its limit (`DecodeOptions::window_log_max`).
pub fn is_window_limit<T>(r: &Result<T, String>) -> bool {
    matches!(r, Err(e) if e.contains("the streaming limit is"))
}

/// libzstd's streaming decoder, ZSTD_decompressStream from a new DCtx with
/// `ZSTD_d_windowLogMax` `window_log_max` unless 0 and the dictionary of
/// bytes `dict` referenced if given (ZSTD_DCtx_refDDict), fed as
/// `stream_with` feeds ours: at most `chunk` bytes of input and `room`
/// bytes of output a call, until a call moves nothing. The content or the
/// error code; input that ends inside a frame is
/// `ZSTD_error_srcSize_wrong`.
pub fn c_streaming(
    input: &[u8],
    chunk: usize,
    room: usize,
    window_log_max: i32,
    dict: Option<&[u8]>,
) -> Result<Vec<u8>, sys::ZSTD_ErrorCode> {
    // SAFETY: the context and the DDict are used only here, the DDict
    // outliving the context, with buffers whose sizes they are given.
    unsafe {
        let dctx = sys::ZSTD_createDCtx();
        assert!(!dctx.is_null(), "create DCtx");
        if window_log_max != 0 {
            let r = sys::ZSTD_DCtx_setParameter(
                dctx,
                sys::ZSTD_dParameter::ZSTD_d_windowLogMax,
                window_log_max,
            );
            assert_eq!(sys::ZSTD_isError(r), 0, "windowLogMax {window_log_max}");
        }
        let ddict = match dict {
            Some(dict) => {
                let ddict = sys::ZSTD_createDDict(dict.as_ptr().cast(), dict.len());
                assert!(!ddict.is_null(), "create DDict");
                assert_eq!(sys::ZSTD_isError(sys::ZSTD_DCtx_refDDict(dctx, ddict)), 0);
                ddict
            }
            None => std::ptr::null_mut(),
        };
        let mut content = Vec::new();
        let mut out = vec![0u8; room];
        let mut pos = 0usize;
        // No input is no frame, which ends between frames.
        let mut between = true;
        let result = loop {
            let end = input.len().min(pos.saturating_add(chunk));
            let mut src = sys::ZSTD_inBuffer {
                src: input[pos..end].as_ptr().cast(),
                size: end - pos,
                pos: 0,
            };
            let mut dst = sys::ZSTD_outBuffer {
                dst: out.as_mut_ptr().cast(),
                size: out.len(),
                pos: 0,
            };
            let hint = sys::ZSTD_decompressStream(dctx, &mut dst, &mut src);
            if sys::ZSTD_isError(hint) != 0 {
                break Err(sys::ZSTD_getErrorCode(hint));
            }
            content.extend_from_slice(&out[..dst.pos]);
            pos += src.pos;
            if src.pos == 0 && dst.pos == 0 {
                break if between {
                    Ok(content)
                } else {
                    Err(sys::ZSTD_ErrorCode::ZSTD_error_srcSize_wrong)
                };
            }
            // 0: a frame has been decoded and written out.
            between = hint == 0;
        };
        sys::ZSTD_freeDCtx(dctx);
        if !ddict.is_null() {
            sys::ZSTD_freeDDict(ddict);
        }
        result
    }
}

/// Whether `r` refuses a frame's window: above the streaming limit, or
/// above `ZSTD_WINDOWLOG_MAX` in its header, as one-shot decoding does too.
/// libzstd gives both `ZSTD_error_frameParameter_windowTooLarge`.
pub fn refuses_window(r: &Result<Vec<u8>, String>) -> bool {
    is_window_limit(r) || matches!(r, Err(e) if e.starts_with("Window log "))
}

/// Check `got`, what our serial streaming decoder at SIMD level `simd`
/// with the default window limit, holding the dictionary `dict` (parsed,
/// and its bytes) if given, gave on `input` fed `chunk` bytes and
/// `stream_room(chunk)` bytes of output a call, against `want`, the
/// one-shot outcome. libzstd's streaming decoder fed alike
/// (`c_streaming`) refuses a window exactly where ours does
/// (`refuses_window`). `got` equals `want` unless it refuses a frame above
/// the window limit; then so does ours with the limit at its largest,
/// `ZSTD_WINDOWLOG_MAX`, exactly where libzstd's does with that limit, and
/// otherwise it gives `want`.
pub fn check_streamed(
    what: &str,
    input: &[u8],
    chunk: usize,
    simd: bool,
    dict: Option<(&DecodeDict, &[u8])>,
    got: &Result<Vec<u8>, String>,
    want: &Result<Vec<u8>, String>,
) {
    let room = stream_room(chunk);
    let raw = dict.map(|(_, raw)| raw);
    let window_too_large = Err(sys::ZSTD_ErrorCode::ZSTD_error_frameParameter_windowTooLarge);
    let lib = c_streaming(input, chunk, room, 0, raw);
    assert!(
        refuses_window(got) == (lib == window_too_large),
        "{what}: streaming gives {} where libzstd's streaming decoder gives {:?}",
        outcome(got),
        lib.map(|c| c.len())
    );
    if !is_window_limit(got) {
        assert!(
            got == want,
            "{what}: streaming gives {} where one-shot gives {}",
            outcome(got),
            outcome(want)
        );
        return;
    }
    let mut d = Decompressor::with_options(&DecodeOptions {
        min_parallel_blocks: usize::MAX,
        min_parallel_bytes: 0,
        simd,
        window_log_max: WINDOW_LOG_MAX,
    });
    d.set_dict(dict.map(|(d, _)| d));
    let lifted = stream_with(&mut d, input, chunk, room);
    let lib = c_streaming(input, chunk, room, WINDOW_LOG_MAX as i32, raw);
    assert!(
        refuses_window(&lifted) == (lib == window_too_large),
        "{what}: streaming with window_log_max {WINDOW_LOG_MAX} gives {} where libzstd's \
         streaming decoder gives {:?}",
        outcome(&lifted),
        lib.map(|c| c.len())
    );
    assert!(
        is_window_limit(&lifted) || lifted == *want,
        "{what}: streaming with window_log_max {WINDOW_LOG_MAX} gives {} where one-shot gives {}",
        outcome(&lifted),
        outcome(want)
    );
}

/// `decompress_streaming` of `input` at each of `chunks`, with as much
/// output room up to 64 KiB, has the outcome of `decompress_with_options`,
/// serial, at both SIMD levels: the same content or the same error, or
/// the window limit's refusal where libzstd's streaming decoder refuses
/// too (`check_streamed`); with the `parallel` feature, decoding whole
/// blocks in parallel changes no call (`assert_parallel_streams`). So does
/// `Decompressor::decompress` on a decompressor every call reuses, and its
/// `decompress_into` into a buffer every call reuses, without the
/// exception.
pub fn assert_stream_parity_at(name: &str, input: &[u8], chunks: &[usize]) {
    for simd in [false, true] {
        let opts = DecodeOptions {
            min_parallel_blocks: usize::MAX,
            min_parallel_bytes: 0,
            simd,
            window_log_max: 0,
        };
        let want = decompress_with_options(input, &opts);
        let (reused, into) = REUSED.with_borrow_mut(|r| {
            let (d, dst) = &mut r[usize::from(simd)];
            let reused = d.decompress(input);
            let into = d.decompress_into(input, dst);
            (reused, into_result(into, dst))
        });
        assert!(
            reused == want,
            "{name} simd={simd}: a reused decompressor gives {} where one-shot gives {}",
            outcome(&reused),
            outcome(&want)
        );
        assert!(
            into == want,
            "{name} simd={simd}: decompress_into gives {} where one-shot gives {}",
            outcome(&into),
            outcome(&want)
        );
        for &chunk in chunks {
            let got = decompress_streaming(input, chunk, stream_room(chunk), &opts);
            let what = format!("{name} simd={simd} chunk {chunk}");
            check_streamed(&what, input, chunk, simd, None, &got, &want);
            #[cfg(feature = "parallel")]
            assert_parallel_streams(&what, input, chunk, simd, None);
        }
    }
}

/// `assert_stream_parity_at` at `STREAM_CHUNKS`.
pub fn assert_stream_parity(name: &str, input: &[u8]) {
    assert_stream_parity_at(name, input, &STREAM_CHUNKS);
}

/// A reader of `data` that gives at most `piece` bytes a read, and fails
/// with `Interrupted` before each other one.
pub struct Pieces<'a> {
    data: &'a [u8],
    piece: usize,
    interrupt: bool,
}

impl Read for Pieces<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.interrupt = !self.interrupt;
        if self.interrupt {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let n = self.data.len().min(self.piece).min(buf.len());
        buf[..n].copy_from_slice(&self.data[..n]);
        self.data = &self.data[n..];
        Ok(n)
    }
}

/// `DecompressReader` over a reader of `input` in pieces of `piece` bytes,
/// with `dict` if given, read into a buffer of `room` bytes until it ends,
/// retrying `Interrupted`; the content, or the error.
pub fn read_all(
    input: &[u8],
    piece: usize,
    room: usize,
    dict: Option<&DecodeDict>,
) -> io::Result<Vec<u8>> {
    let pieces = Pieces {
        data: input,
        piece,
        interrupt: false,
    };
    let mut r = match dict {
        Some(dict) => DecompressReader::with_dict(pieces, dict),
        None => DecompressReader::new(pieces),
    };
    let mut content = Vec::new();
    let mut buf = vec![0u8; room];
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => content.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    // It stays at the end.
    assert_eq!(r.read(&mut buf).unwrap(), 0);
    Ok(content)
}
