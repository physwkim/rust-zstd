//! The post-sequence block splitter (`ZSTD_c_splitAfterSequences`) against
//! libzstd 1.5.7 with the splitter enabled and the pre-splitter off
//! (`ZSTD_c_blockSplitterLevel` 1), single job: every frame decodes on both
//! decoders, and where the unsplit frames already agree block for block
//! (so the match finders produced the same sequences) the split frames have
//! the same blocks: type and decompressed size of each, in order.
//!
//! `split_blocks_match_libzstd` runs on the shared datasets (the 8 MiB ones
//! cut to 2 MiB); the ignored `split_blocks_match_libzstd_on_corpus` on the
//! files of the corpus directory (`ZSTD_CORPUS_DIR` overrides it):
//!
//! ```text
//! cargo nextest run --release --test split_parity --run-ignored all
//! ```

mod common;

use rust_zstd::{compress_with, CompressOptions, ParamSwitch};
use std::path::PathBuf;
use zstd::zstd_safe::zstd_sys as sys;

const DEFAULT_CORPUS: &str = "/tmp/claude-1000/-home-stevek-work-rust-zstd/d30c8856-c9ae-4039-8110-94096bb23bce/scratchpad/corpus";
const FILES: [&str; 3] = ["elf_8M.bin", "rssrc_8M.txt", "words_1M.txt"];
const LEVELS: [i32; 4] = [3, 5, 7, 11];

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
struct Block {
    ty: u32,
    size: usize,
    c_size: usize,
}

/// Decode a single-frame `frame` with libzstd's buffer-less decoder,
/// returning its blocks and content.
fn blocks(frame: &[u8], content_size: usize) -> (Vec<Block>, Vec<u8>) {
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

/// libzstd's single-threaded frame with `ZSTD_c_splitAfterSequences` set to
/// `split` (1 enable, 2 disable) and the pre-splitter off.
fn c_frame(data: &[u8], level: i32, split: i32) -> Vec<u8> {
    // SAFETY: the context is used only here; buffers are sized by
    // ZSTD_compressBound.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let set = |p, v| {
            let r = sys::ZSTD_CCtx_setParameter(cctx, p, v);
            assert_eq!(sys::ZSTD_isError(r), 0, "set parameter {p:?}");
        };
        set(sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level);
        // ZSTD_c_splitAfterSequences.
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam13, split);
        // ZSTD_c_blockSplitterLevel 1: no pre-splitting.
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam20, 1);
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

/// Our frame as one job, with the pre-splitter off like [`c_frame`].
fn ours(data: &[u8], level: i32, split_after_sequences: ParamSwitch) -> Vec<u8> {
    compress_with(
        data,
        &CompressOptions {
            level,
            job_size: Some(1 << 30),
            split_after_sequences,
            block_splitter_level: 1,
            ..Default::default()
        },
    )
}

/// Compare every input at every level in [`LEVELS`]; returns the number
/// of cases compared in which our frame split blocks.
fn compare(inputs: Vec<(String, Vec<u8>)>) -> usize {
    let key = |b: &Block| (b.ty, b.size);
    let (mut compared, mut split_cases) = (0, 0);
    let mut skipped = Vec::new();
    for (name, data) in inputs {
        for level in LEVELS {
            let split = ours(&data, level, ParamSwitch::Enable);
            let whole = ours(&data, level, ParamSwitch::Disable);
            let (split_blocks, decoded) = blocks(&split, data.len());
            assert!(
                decoded == data,
                "{name} L{level}: libzstd decodes our frame wrong"
            );
            assert!(
                rust_zstd::decompress(&split).unwrap() == data,
                "{name} L{level}: our decoder"
            );
            let (whole_blocks, _) = blocks(&whole, data.len());
            let (c_whole_blocks, _) = blocks(&c_frame(&data, level, 2), data.len());
            if whole_blocks != c_whole_blocks {
                // The match finders differ on this input; the splitter
                // would be compared on different sequences.
                skipped.push(format!("{name} L{level}"));
                continue;
            }
            let (c_split_blocks, _) = blocks(&c_frame(&data, level, 1), data.len());
            let ours_keys: Vec<_> = split_blocks.iter().map(key).collect();
            let theirs_keys: Vec<_> = c_split_blocks.iter().map(key).collect();
            assert_eq!(ours_keys, theirs_keys, "{name} L{level}: block boundaries");
            compared += 1;
            split_cases += (split_blocks.len() > whole_blocks.len()) as usize;
        }
    }
    eprintln!("compared {compared}, with splits {split_cases}, skipped {skipped:?}");
    split_cases
}

#[test]
fn split_blocks_match_libzstd() {
    let inputs = common::datasets()
        .into_iter()
        .filter(|ds| ["rust_src_8m", "elf_8m", "words_1m", "text_1m"].contains(&ds.name))
        .map(|ds| {
            let len = ds.data.len().min(2 << 20);
            (ds.name.to_string(), ds.data[..len].to_vec())
        })
        .collect();
    let split_cases = compare(inputs);
    assert!(split_cases >= 4, "too few inputs split: {split_cases}");
}

#[test]
#[ignore]
fn split_blocks_match_libzstd_on_corpus() {
    let dir = std::env::var_os("ZSTD_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CORPUS));
    let inputs: Vec<_> = FILES
        .iter()
        .filter_map(|f| Some((f.to_string(), std::fs::read(dir.join(f)).ok()?)))
        .collect();
    if inputs.is_empty() {
        eprintln!("no corpus files in {}", dir.display());
        return;
    }
    compare(inputs);
}
