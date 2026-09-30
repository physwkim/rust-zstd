//! Shared dataset construction for the decoder verification and benchmark
//! tests. Datasets are cached under `target/decoder-datasets/` so that every
//! run (before and after an optimization) works on byte-identical inputs and
//! therefore byte-identical libzstd streams.

#![allow(dead_code)]

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
