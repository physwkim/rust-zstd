//! Shared dataset construction for the decoder verification and benchmark
//! tests. Datasets are cached under `target/decoder-datasets/` so that every
//! run (before and after an optimization) works on byte-identical inputs and
//! therefore byte-identical libzstd streams.

#![allow(dead_code)]

use rust_zstd::decode::{decompress_with_options, DecodeOptions};
use rust_zstd::Decompressor;
use std::cell::RefCell;
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
    let mut d = Decompressor::with_options(opts);
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

thread_local! {
    /// A serial decompressor per SIMD level, and the buffer its
    /// `decompress_into` calls fill, for every `assert_stream_parity_at`
    /// call on the thread, so each input decodes with the tables and
    /// buffers the inputs before it, rejected ones included, left.
    static REUSED: RefCell<[(Decompressor, Vec<u8>); 2]> = RefCell::new([false, true].map(|simd| {
        let d = Decompressor::with_options(&DecodeOptions {
            min_parallel_blocks: usize::MAX,
            simd,
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

/// `decompress_streaming` of `input` at each of `chunks`, with as much
/// output room up to 64 KiB, has the outcome of `decompress_with_options`,
/// serial, at both SIMD levels: the same content or the same error. So
/// does `Decompressor::decompress` on a decompressor every call reuses,
/// and its `decompress_into` into a buffer every call reuses.
pub fn assert_stream_parity_at(name: &str, input: &[u8], chunks: &[usize]) {
    fn outcome(r: &Result<Vec<u8>, String>) -> String {
        match r {
            Ok(content) => format!("{} bytes", content.len()),
            Err(e) => format!("error {e:?}"),
        }
    }
    for simd in [false, true] {
        let opts = DecodeOptions {
            min_parallel_blocks: usize::MAX,
            simd,
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
            assert!(
                got == want,
                "{name} simd={simd} chunk {chunk}: streaming gives {} where one-shot gives {}",
                outcome(&got),
                outcome(&want)
            );
        }
    }
}

/// `assert_stream_parity_at` at `STREAM_CHUNKS`.
pub fn assert_stream_parity(name: &str, input: &[u8]) {
    assert_stream_parity_at(name, input, &STREAM_CHUNKS);
}
