//! The pre-splitter (`ZSTD_c_blockSplitterLevel`) against libzstd 1.5.7.
//!
//! `ZSTD_splitBlock` itself is compared on 128 KiB windows at every level.
//! Frames pass the encoder gate against libzstd's in two configurations:
//! - default options against libzstd's defaults (single-threaded
//!   `ZSTD_compress2`);
//! - `CompressOptions::parallel(level)` against ZSTDMT at 2 MiB jobs and
//!   overlap log 8.
//!
//! The frames with the pre-splitter off must pass it too, and enough of
//! ours must be pre-split.
//!
//! `blocks_match_libzstd` runs on the shared datasets (the 8 MiB ones cut to
//! 3 MiB); the ignored `blocks_match_libzstd_on_corpus` runs on the files of
//! the corpus directory (`ZSTD_CORPUS_DIR` overrides it):
//!
//! ```text
//! cargo nextest run --release --test presplit_parity --run-ignored all
//! ```

mod common;

use common::{assert_gate, frame_blocks};
use rust_zstd::compress::presplit::{PreSplitter, SPLIT_BLOCK_SIZE};
use rust_zstd::{compress_with, CompressOptions};
use std::path::PathBuf;
use zstd::zstd_safe::zstd_sys as sys;

const DEFAULT_CORPUS: &str = "/tmp/claude-1000/-home-stevek-work-rust-zstd/d30c8856-c9ae-4039-8110-94096bb23bce/scratchpad/corpus";
const FILES: [&str; 3] = ["elf_8M.bin", "rssrc_8M.txt", "words_1M.txt"];
const LEVELS: [i32; 6] = [1, 3, 5, 7, 11, 12];

extern "C" {
    /// `zstd_preSplit.h`; `workspace` holds `ZSTD_SLIPBLOCK_WORKSPACESIZE`
    /// bytes.
    fn ZSTD_splitBlock(
        block: *const u8,
        block_size: usize,
        level: i32,
        workspace: *mut u8,
        workspace_size: usize,
    ) -> usize;
}

/// `ZSTD_SLIPBLOCK_WORKSPACESIZE`.
const WORKSPACE_SIZE: usize = 8208;

#[test]
fn split_block_matches_libzstd() {
    let mut inputs: Vec<Vec<u8>> = common::datasets()
        .into_iter()
        .filter(|ds| ds.data.len() >= SPLIT_BLOCK_SIZE)
        .map(|ds| ds.data[..ds.data.len().min(2 << 20)].to_vec())
        .collect();
    // Runs of noise, text and 16-letter noise, so that windows straddle
    // changes of statistics.
    let mut mix = Vec::new();
    let mut x = 88172645463325252u64;
    for k in 0..48 {
        for i in 0..k * 997 % 20000 + 3000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            mix.push(match k % 3 {
                0 => (x >> 24) as u8,
                1 => b"lorem ipsum dolor "[i % 18],
                _ => b'a' + (x >> 60) as u8,
            });
        }
    }
    inputs.push(mix);
    let mut workspace = vec![0u64; WORKSPACE_SIZE / 8];
    let mut ours = PreSplitter::default();
    let (mut compared, mut cut) = (0, 0);
    for data in &inputs {
        for start in (0..=data.len() - SPLIT_BLOCK_SIZE).step_by(12289) {
            let block = &data[start..start + SPLIT_BLOCK_SIZE];
            for level in 0..=4u8 {
                // SAFETY: `block` and `workspace` are valid for the sizes
                // passed.
                let theirs = unsafe {
                    ZSTD_splitBlock(
                        block.as_ptr(),
                        block.len(),
                        level as i32,
                        workspace.as_mut_ptr().cast(),
                        WORKSPACE_SIZE,
                    )
                };
                let size = ours.split_block(block, level);
                assert_eq!(size, theirs, "window at {start}, level {level}");
                compared += 1;
                cut += (size < SPLIT_BLOCK_SIZE) as usize;
            }
        }
    }
    eprintln!("compared {compared}, cut {cut}");
    assert!(
        cut > compared / 10 && cut < compared,
        "cut {cut} of {compared}"
    );
}

/// libzstd's frame with `level`, `extra` parameters and the pre-splitter
/// at `block_splitter_level`.
fn c_frame(
    data: &[u8],
    level: i32,
    extra: &[(sys::ZSTD_cParameter, i32)],
    block_splitter_level: u8,
) -> Vec<u8> {
    let mut params = vec![
        (sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level),
        // ZSTD_c_blockSplitterLevel.
        (
            sys::ZSTD_cParameter::ZSTD_c_experimentalParam20,
            block_splitter_level as i32,
        ),
    ];
    params.extend_from_slice(extra);
    common::c_compress2(data, &params)
}

/// Compare every input at every level of [`LEVELS`] in both
/// configurations; returns the number of compared cases whose blocks were
/// pre-split.
fn compare(inputs: Vec<(String, Vec<u8>)>) -> usize {
    use sys::ZSTD_cParameter::{ZSTD_c_jobSize, ZSTD_c_nbWorkers, ZSTD_c_overlapLog};
    let (mut compared, mut presplit) = (0, 0);
    for (name, data) in &inputs {
        for level in LEVELS {
            let parallel_mt: &[_] = &[
                (ZSTD_c_nbWorkers, 2),
                (ZSTD_c_jobSize, 2 << 20),
                (ZSTD_c_overlapLog, 8),
            ];
            let configs = [
                (
                    "default",
                    CompressOptions {
                        level,
                        ..Default::default()
                    },
                    &[][..],
                ),
                ("parallel", CompressOptions::parallel(level), parallel_mt),
            ];
            for (config, opts, c_params) in configs {
                let case = format!("{name} L{level} {config}");
                let off = CompressOptions {
                    block_splitter_level: 1,
                    ..opts.clone()
                };
                let ours_off = compress_with(data, &off);
                let theirs_off = c_frame(data, level, c_params, 1);
                assert_gate(
                    &format!("{case} pre-splitter off"),
                    data,
                    &ours_off,
                    &theirs_off,
                );
                let ours = compress_with(data, &opts);
                assert_gate(&case, data, &ours, &c_frame(data, level, c_params, 0));
                compared += 1;
                let blocks = |frame| frame_blocks(frame, data.len()).0.len();
                presplit += (blocks(&ours) > blocks(&ours_off)) as usize;
            }
        }
    }
    eprintln!("compared {compared}, pre-split {presplit}");
    presplit
}

#[test]
fn blocks_match_libzstd() {
    let inputs = common::datasets()
        .into_iter()
        .filter(|ds| ds.data.len() > 1)
        .map(|ds| {
            let len = ds.data.len().min(3 << 20);
            (ds.name.to_string(), ds.data[..len].to_vec())
        })
        .collect();
    let presplit = compare(inputs);
    assert!(presplit >= 20, "too few inputs pre-split: {presplit}");
}

#[test]
#[ignore]
fn blocks_match_libzstd_on_corpus() {
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
