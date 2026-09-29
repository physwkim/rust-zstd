//! Compressor stage split on the QA corpus: end-to-end `compress_with`
//! (fresh), the same frame through one reused [`Compressor`] (reuse), and
//! libzstd 1.5.7 through a reused `ZSTD_CCtx` (C), against the block stage
//! (`MatchState` setup, prefix load and `build_seq_store`), the literal
//! stage (`compress_literals_with`) and the sequence stage
//! (`encode_sequences_section_with`), each timed in place inside a replica
//! of the driver's job and block loop whose output is checked byte for byte
//! against the real frame. The remainder is driver overhead of the fresh
//! path. Ignored by default; run pinned and in release:
//!
//! ```text
//! taskset -c 3 cargo test --release --offline --test stage_bench -- --ignored --nocapture
//! ```
//!
//! `ZSTD_CORPUS_DIR` overrides the corpus directory, `ZSTD_BENCH_ITERS` the
//! number of iterations per cell (default 5, odd values keep the median exact).

use rust_zstd::compress::block::{
    self, BlockScratch, BlockState, MIN_CBLOCK_SIZE, RLE_MAX_LENGTH, ZSTD_BLOCKHEADERSIZE,
};
use rust_zstd::compress::matchstate::MatchState;
use rust_zstd::compress::{
    compress_with, job_ranges, job_size_for, overlap_size, CParams, CompressOptions, Compressor,
};
use rust_zstd::constants::ZSTD_BLOCKSIZE_MAX;
use rust_zstd::{fse, huf};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_CORPUS: &str = "/tmp/claude-1000/-home-stevek-work-rust-zstd/d30c8856-c9ae-4039-8110-94096bb23bce/scratchpad/corpus";
const FILES: [&str; 3] = ["elf_8M.bin", "rssrc_8M.txt", "words_1M.txt"];
const LEVELS: [i32; 4] = [1, 3, 5, 11];
const DEFAULT_ITERS: usize = 5;

#[derive(Clone, Copy, Default)]
struct Stages {
    block: Duration,
    lits: Duration,
    seqs: Duration,
    pass: Duration,
}

#[derive(Default)]
struct Layout {
    jobs: usize,
    blocks: usize,
}

/// `compress_with`'s job and block loop with a timer around each stage.
/// Returns the block area of the frame (everything after the frame header).
fn stage_pass(data: &[u8], cparams: CParams, st: &mut Stages, layout: &mut Layout) -> Vec<u8> {
    let t_pass = Instant::now();
    let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
    let overlap = overlap_size(&cparams, 0);
    let jobs = job_ranges(data.len(), job_size_for(None, cparams.window_log, overlap));
    layout.jobs = jobs.len();
    layout.blocks = 0;
    let mut out = Vec::new();
    for (k, job) in jobs.iter().enumerate() {
        let first_job = k == 0;
        let last_job = k + 1 == jobs.len();
        let window_low = if first_job {
            1
        } else {
            job.start.saturating_sub(overlap).max(1)
        };
        let t = Instant::now();
        let mut ms = MatchState::new(cparams, window_low);
        let mut prev = BlockState::initial();
        if !first_job {
            block::load_prefix(&mut ms, data, window_low..job.start);
            prev.invalidate_rep_codes();
        }
        st.block += t.elapsed();
        let mut scratch = BlockScratch::new(block_size);
        let mut start = job.start;
        while start < job.end {
            let end = (start + block_size).min(job.end);
            let block_len = end - start;
            let is_first = first_job && start == job.start;
            let is_last = last_job && end == job.end;
            layout.blocks += 1;
            let mut next = None;
            // block.rs: `block_len < MIN_CBLOCK_SIZE + ZSTD_BLOCKHEADERSIZE + 1 + 1` -> RAW
            if block_len > MIN_CBLOCK_SIZE + ZSTD_BLOCKHEADERSIZE + 1 {
                let mut rep = prev.rep;
                let t = Instant::now();
                block::build_seq_store(&mut ms, data, start..end, &mut rep, &mut scratch.store);
                st.block += t.elapsed();
                let store = &scratch.store;
                let cbuf = &mut scratch.cbuf;
                cbuf.clear();
                let t = Instant::now();
                let huf = huf::compress_literals_with(
                    cbuf,
                    &store.lits,
                    store.seqs.len(),
                    &prev.huf,
                    &cparams,
                );
                st.lits += t.elapsed();
                let t = Instant::now();
                let fse =
                    fse::encode_sequences_section_with(cbuf, &store.seqs, &prev.fse, &cparams);
                st.seqs += t.elapsed();
                if let Some(fse) = fse {
                    if cbuf.len() < block_len - CParams::min_gain(block_len, cparams.strategy) {
                        next = Some(BlockState { rep, huf, fse });
                    }
                }
            }
            let bdata = &data[start..end];
            let c_size = if next.is_some() {
                scratch.cbuf.len()
            } else {
                0
            };
            if !is_first && c_size < RLE_MAX_LENGTH && block::is_rle(bdata) {
                block::write_rle_block(&mut out, bdata[0], bdata.len(), is_last);
            } else if let Some(next) = next {
                prev = next;
                block::write_compressed_block(&mut out, &scratch.cbuf, is_last);
            } else {
                block::write_raw_block(&mut out, bdata, is_last);
            }
            start = end;
        }
    }
    st.pass += t_pass.elapsed();
    out
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
#[ignore]
fn stage_split() {
    let dir = std::env::var_os("ZSTD_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CORPUS));
    let iters: usize = std::env::var("ZSTD_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_ITERS);
    eprintln!(
        "\n{:<13} {:>3} {:>4} {:>6} {:>9} {:>9} | {:>8} {:>8} {:>6} {:>7} {:>6} {:>7} | {:>8} | {:>8} {:>5} | {:>8} {:>5} | {:>8} {:>5} | {:>8} {:>5}",
        "dataset", "L", "jobs", "blocks", "size", "C size",
        "fresh ms", "reuse ms", "MB/s", "C ms", "MB/s", "reuse/C", "pass ms",
        "block ms", "%", "lits ms", "%", "seqs ms", "%", "rem ms", "%"
    );
    for file in FILES {
        let path = dir.join(file);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}: {e} (set ZSTD_CORPUS_DIR)", path.display());
                continue;
            }
        };
        for level in LEVELS {
            let opts = CompressOptions {
                level,
                ..CompressOptions::default()
            };
            let cparams = CParams::for_level(level, data.len());
            let mut cx = Compressor::new(opts.clone());
            let mut czstd = zstd::bulk::Compressor::new(level).unwrap();
            let mut frame = Vec::new();
            let mut c_frame = Vec::new();
            let mut e2e = Vec::with_capacity(iters);
            let mut reuse = Vec::with_capacity(iters);
            let mut c_e2e = Vec::with_capacity(iters);
            let mut runs = Vec::with_capacity(iters);
            let mut layout = Layout::default();
            // Interleaved so that clock drift hits every side alike.
            for _ in 0..iters {
                let t = Instant::now();
                frame = compress_with(&data, &opts);
                e2e.push(t.elapsed());
                let t = Instant::now();
                let reused = cx.compress_to_vec(&data);
                reuse.push(t.elapsed());
                assert!(
                    reused == frame,
                    "{file} L{level}: reused Compressor diverged from compress_with"
                );
                let t = Instant::now();
                c_frame = czstd.compress(&data).unwrap();
                c_e2e.push(t.elapsed());
                let mut st = Stages::default();
                let blocks = stage_pass(&data, cparams, &mut st, &mut layout);
                assert!(
                    frame.ends_with(&blocks) && frame.len() - blocks.len() <= 14,
                    "{file} L{level}: stage replica diverged from compress_with"
                );
                runs.push(st);
            }
            let e2e = median(e2e);
            let reuse = median(reuse);
            let c_e2e = median(c_e2e);
            let block = median(runs.iter().map(|s| s.block).collect());
            let lits = median(runs.iter().map(|s| s.lits).collect());
            let seqs = median(runs.iter().map(|s| s.seqs).collect());
            let pass = median(runs.iter().map(|s| s.pass).collect());
            let rem = e2e.as_secs_f64() - (block + lits + seqs).as_secs_f64();
            let pct = |d: f64| 100.0 * d / e2e.as_secs_f64();
            let mbs = |d: Duration| data.len() as f64 / (1 << 20) as f64 / d.as_secs_f64();
            eprintln!(
                "{:<13} {:>3} {:>4} {:>6} {:>9} {:>9} | {:>8.2} {:>8.2} {:>6.1} {:>7.2} {:>6.1} {:>7.3} | {:>8.2} | {:>8.2} {:>5.1} | {:>8.2} {:>5.1} | {:>8.2} {:>5.1} | {:>8.2} {:>5.1}",
                file, level, layout.jobs, layout.blocks, frame.len(), c_frame.len(),
                ms(e2e), ms(reuse), mbs(reuse), ms(c_e2e), mbs(c_e2e),
                reuse.as_secs_f64() / c_e2e.as_secs_f64(), ms(pass),
                ms(block), pct(block.as_secs_f64()),
                ms(lits), pct(lits.as_secs_f64()),
                ms(seqs), pct(seqs.as_secs_f64()),
                rem * 1e3, pct(rem)
            );
        }
    }
}
