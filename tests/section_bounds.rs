//! `huf::literals_section_bound` and `fse::sequences_section_bound` must
//! bound the bytes the two section encoders append for every block the
//! compressor produces, under the real previous-block state and under a
//! fresh one, and whenever the two bounds prove a block compressed
//! (`sum < block_len - ZSTD_minGain`) the block must come out compressed.
//! The blocks come from a replica of the driver's job and block loop whose
//! output is checked against `compress_with` with the pre-splitter off
//! (`block_splitter_level` 1), since the replica cuts 128 KiB blocks.
//!
//! The corpus sweep is ignored by default; run it in release:
//!
//! ```text
//! taskset -c 2 cargo test --release --test section_bounds -- --ignored --nocapture
//! ```
//!
//! `ZSTD_CORPUS_DIR` overrides the corpus directory.

use rust_zstd::compress::block::{self, BlockScratch, BlockState, RLE_MAX_LENGTH};
use rust_zstd::compress::matchstate::MatchState;
use rust_zstd::compress::seqstore::Seq;
use rust_zstd::compress::{
    compress_with, job_prefix, job_ranges, job_size_for, overlap_size, CParams, CompressOptions,
};
use rust_zstd::constants::ZSTD_BLOCKSIZE_MAX;
use rust_zstd::fse::{self, sequences_section_bound, FseState};
use rust_zstd::huf::{self, literals_section_bound};
use std::path::{Path, PathBuf};

const DEFAULT_CORPUS: &str = "/tmp/claude-1000/-home-stevek-work-rust-zstd/d30c8856-c9ae-4039-8110-94096bb23bce/scratchpad/corpus";

#[derive(Default, Clone, Copy)]
struct Tally {
    blocks: usize,
    compressed: usize,
    proven: usize,
}

/// Run both stages on one block's store against `prev`, check the bounds
/// and return `(section bytes, BlockState)` when the block compresses.
fn stages(
    store_lits: &[u8],
    seqs: &[Seq],
    prev: &BlockState,
    rep: [u32; 3],
    cparams: &CParams,
    block_len: usize,
    what: &str,
) -> Option<(Vec<u8>, BlockState)> {
    let lit_bound = literals_section_bound(store_lits.len());
    let seq_bound = sequences_section_bound(seqs);
    let mut cbuf = Vec::new();
    let huf = huf::compress_literals_with(&mut cbuf, store_lits, seqs.len(), &prev.huf, cparams);
    let lit_size = cbuf.len();
    assert!(
        lit_size <= lit_bound,
        "{what}: literals {lit_size} > bound {lit_bound}"
    );
    let fse =
        fse::encode_sequences_section_with(&mut cbuf, seqs, &mut Vec::new(), &prev.fse, cparams);
    let seq_size = cbuf.len() - lit_size;
    assert!(
        seq_size <= seq_bound,
        "{what}: sequences {seq_size} > bound {seq_bound} ({} seqs)",
        seqs.len()
    );
    let limit = block_len - CParams::min_gain(block_len, cparams.strategy);
    let proven = lit_bound + seq_bound < limit;
    let compressed = fse.is_some() && cbuf.len() < limit;
    assert!(
        !proven || compressed,
        "{what}: bounds {lit_bound}+{seq_bound} < {limit} but the block is not compressed"
    );
    let fse = fse?;
    compressed.then_some((cbuf, BlockState { rep, huf, fse }))
}

/// `compress_with`'s job and block loop around [`stages`]. Returns the
/// block area of the frame.
fn sweep(data: &[u8], level: i32, name: &str, tally: &mut Tally) -> Vec<u8> {
    let cparams = CParams::for_level(level, data.len());
    let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
    let overlap = overlap_size(&cparams, 0, false);
    let jobs = job_ranges(data.len(), job_size_for(None, &cparams, false, overlap));
    let fresh = BlockState::initial();
    let mut out = Vec::new();
    for (k, job) in jobs.iter().enumerate() {
        let first_job = k == 0;
        let last_job = k + 1 == jobs.len();
        let prefix = job_prefix(job, first_job, overlap);
        let mut ms = MatchState::new(cparams, prefix.start);
        let mut prev = BlockState::initial();
        if !first_job {
            block::load_prefix(&mut ms, data, prefix);
            prev.invalidate_rep_codes();
        }
        let mut scratch = BlockScratch::new(block_size);
        let mut start = job.start;
        while start < job.end {
            let end = (start + block_size).min(job.end);
            let block_len = end - start;
            let is_first = first_job && start == job.start;
            let is_last = last_job && end == job.end;
            let what = format!("{name} L{level} block @{start}");
            let mut next = None;
            let entered = ms.enter_block(start..end);
            let built = block::build_seq_store(
                &mut ms,
                data,
                entered,
                prev.rep,
                &mut scratch.store,
                &mut block::BlockLdm::Off,
                None,
            );
            // None below 7 bytes (RAW)
            if let Some(rep) = built {
                tally.blocks += 1;
                let store = &scratch.store;
                let proven = literals_section_bound(store.lits.len())
                    + sequences_section_bound(&store.seqs)
                    < block_len - CParams::min_gain(block_len, cparams.strategy);
                tally.proven += proven as usize;
                // any previous state: the fresh one
                stages(
                    &store.lits,
                    &store.seqs,
                    &fresh,
                    rep,
                    &cparams,
                    block_len,
                    &format!("{what} (fresh state)"),
                );
                next = stages(
                    &store.lits,
                    &store.seqs,
                    &prev,
                    rep,
                    &cparams,
                    block_len,
                    &what,
                );
            }
            let bdata = &data[start..end];
            let c_size = next.as_ref().map_or(0, |(c, _)| c.len());
            if !is_first && c_size < RLE_MAX_LENGTH && block::is_rle(bdata) {
                block::write_rle_block(&mut out, bdata[0], bdata.len(), is_last);
            } else if let Some((cbuf, state)) = next {
                tally.compressed += 1;
                prev = state;
                block::write_compressed_block(&mut out, &cbuf, is_last);
            } else {
                block::write_raw_block(&mut out, bdata, is_last);
            }
            start = end;
        }
    }
    out
}

fn check(data: &[u8], level: i32, name: &str) -> Tally {
    let mut tally = Tally::default();
    let blocks = sweep(data, level, name, &mut tally);
    let frame = compress_with(
        data,
        &CompressOptions {
            level,
            block_splitter_level: 1,
            ..CompressOptions::default()
        },
    );
    assert!(
        frame.ends_with(&blocks) && frame.len() - blocks.len() <= 14,
        "{name} L{level}: replica diverged from compress_with"
    );
    tally
}

/// xorshift64*.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let uniform: Vec<u8> = (0..200_000).map(|_| r.next() as u8).collect();
    let skewed: Vec<u8> = (0..200_000).map(|_| b"aaabbcd"[r.below(7)]).collect();
    // Huffman-compressible literals with few matches: compressed blocks
    // the bounds cannot prove
    let literal_heavy: Vec<u8> = (0..300_000)
        .map(|_| {
            let x = r.next();
            (x as u8 & 0x3F) & ((x >> 8) as u8 & 0x3F) | 0x40
        })
        .collect();
    let vocab = [
        "the ",
        "of ",
        "zstd ",
        "entropy ",
        "block ",
        "literal ",
        "sequence ",
        "\n",
        "fse ",
        "huffman ",
        "table ",
        "offset ",
    ];
    let mut words = Vec::new();
    while words.len() < 300_000 {
        words.extend_from_slice(vocab[r.below(vocab.len())].as_bytes());
    }
    // copies of earlier spans at every offset scale, with mutations
    let mut copies: Vec<u8> = (0..64).map(|_| r.next() as u8).collect();
    while copies.len() < 400_000 {
        let scale = 1usize << (1 + r.below(20));
        let off = 1 + r.below(copies.len().min(scale));
        let max_len = if r.below(8) == 0 { 2000 } else { 40 };
        let len = 3 + r.below(max_len);
        for _ in 0..len {
            let b = copies[copies.len() - off];
            copies.push(b);
        }
        for _ in 0..r.below(6) {
            copies.push(r.next() as u8);
        }
    }
    // zero runs (RLE blocks and long matches) mixed with noise
    let mut runs = Vec::new();
    while runs.len() < 400_000 {
        let n = r.below(70_000);
        runs.extend(std::iter::repeat_n(0u8, n));
        let m = r.below(3000);
        runs.extend((0..m).map(|_| r.next() as u8));
    }
    vec![
        ("uniform", uniform),
        ("skewed", skewed),
        ("literal_heavy", literal_heavy),
        ("words", words),
        ("copies", copies),
        ("runs", runs),
    ]
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name),
    )
    .unwrap()
}

#[test]
fn bounds_hold_on_fixtures_and_random_data() {
    let mut inputs = vec![
        ("minfail.bin", fixture("minfail.bin")),
        ("minfail2.bin", fixture("minfail2.bin")),
    ];
    inputs.extend(random_inputs());
    for (name, data) in &inputs {
        for level in 1..=12 {
            let t = check(data, level, name);
            if level == 1 || level == 3 {
                eprintln!(
                    "{name:<13} L{level}: {} blocks, {} compressed, {} proven",
                    t.blocks, t.compressed, t.proven
                );
            }
        }
    }
}

/// One boundary per branch of the 1.3.4-workaround guard: a single
/// sequence can never use a new table, two sequences with at most 2 extra
/// bits may, three extra bits rule the 1-byte bitstream out.
#[test]
fn sequences_bound_guards_the_decoder_workaround() {
    let rep0 = |lit_len| Seq {
        lit_len,
        off_base: 1,
        ml_base: 0,
    };
    let small = sequences_section_bound(&[rep0(0)]);
    assert!(small <= ZSTD_BLOCKSIZE_MAX, "one sequence: {small}");
    // off_base 2 -> OF code 1, one extra bit each
    let two_bits = [
        Seq {
            off_base: 2,
            ..rep0(0)
        },
        Seq {
            off_base: 3,
            ..rep0(1)
        },
    ];
    assert!(sequences_section_bound(&two_bits) > ZSTD_BLOCKSIZE_MAX);
    assert!(sequences_section_bound(&[rep0(0), rep0(1)]) > ZSTD_BLOCKSIZE_MAX);
    // off_base 4 -> OF code 2, two extra bits; plus one more
    let three_bits = [
        Seq {
            off_base: 4,
            ..rep0(0)
        },
        Seq {
            off_base: 2,
            ..rep0(1)
        },
    ];
    assert!(sequences_section_bound(&three_bits) <= ZSTD_BLOCKSIZE_MAX);
    assert_eq!(sequences_section_bound(&[]), 1);
    // the empty section appends only the count byte
    let mut out = Vec::new();
    let st = fse::encode_sequences_section_with(
        &mut out,
        &[],
        &mut Vec::new(),
        &FseState::default(),
        &CParams::for_level(1, 1 << 20),
    );
    assert!(st.is_some() && out.len() == 1);
}

#[test]
#[ignore]
fn bounds_hold_on_corpus() {
    let dir = std::env::var_os("ZSTD_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CORPUS));
    eprintln!(
        "\n{:<14} {:>3} {:>6} {:>10} {:>7} {:>9} {:>11}",
        "file", "L", "blocks", "compressed", "proven", "proven/all", "proven/cmp"
    );
    for file in ["elf_8M.bin", "rssrc_8M.txt", "words_1M.txt"] {
        let data = std::fs::read(dir.join(file))
            .unwrap_or_else(|e| panic!("{file}: {e} (set ZSTD_CORPUS_DIR)"));
        for level in 1..=12 {
            let t = check(&data, level, file);
            eprintln!(
                "{:<14} {:>3} {:>6} {:>10} {:>7} {:>9.3} {:>11.3}",
                file,
                level,
                t.blocks,
                t.compressed,
                t.proven,
                t.proven as f64 / t.blocks as f64,
                t.proven as f64 / t.compressed.max(1) as f64
            );
        }
    }
}
