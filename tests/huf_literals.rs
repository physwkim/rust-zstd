//! Literals-section conformance: `huf::compress_literals_with` must produce
//! the bytes libzstd 1.5.7's `ZSTD_compressLiterals` produces for the same
//! literals, previous Huffman state and strategy (`tests/data/huf`), and
//! every section must decode through our decoder and libzstd.

use rust_zstd::compress::{CParams, Strategy};
use rust_zstd::huf::{compress_literals_with, HufState};
use std::path::Path;

fn cparams(strategy: Strategy, target_length: u32) -> CParams {
    CParams {
        window_log: 19,
        chain_log: 12,
        hash_log: 12,
        search_log: 1,
        min_match: 4,
        target_length,
        strategy,
    }
}

fn strategy(n: u32) -> Strategy {
    match n {
        1 => Strategy::Fast,
        2 => Strategy::DFast,
        3 => Strategy::Greedy,
        4 => Strategy::Lazy,
        5 => Strategy::Lazy2,
        _ => panic!("strategy {n}"),
    }
}

/// One frame whose blocks are `sections` (each a literals section followed
/// by an empty sequences section), so a Treeless block decodes with the
/// table its predecessor installed. The window is 1 MiB rather than
/// single-segment: libzstd caps a compressed block at
/// `min(windowSize, 128 KiB)`, and a raw-literals section is larger than
/// the content it carries.
fn frame_around(sections: &[Vec<u8>], content_size: usize) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&0xFD2F_B528u32.to_le_bytes());
    f.push(0x80); // FCS 4 bytes, window descriptor present
    f.push((20 - 10) << 3); // 1 MiB window
    f.extend_from_slice(&(content_size as u32).to_le_bytes());
    for (i, section) in sections.iter().enumerate() {
        let block_size = section.len() + 1;
        let last = (i + 1 == sections.len()) as u32;
        let bh = last | (2 << 1) | ((block_size as u32) << 3);
        f.extend_from_slice(&bh.to_le_bytes()[..3]);
        f.extend_from_slice(section);
        f.push(0); // nbSeq = 0
    }
    f
}

fn decodes_to(sections: &[Vec<u8>], literals: &[u8], what: &str) {
    let frame = frame_around(sections, literals.len());
    let ours = rust_zstd::decompress(&frame).unwrap_or_else(|e| panic!("{what}: ours {e:?}"));
    assert!(ours == literals, "{what}: our decoder mismatch");
    let theirs =
        zstd::stream::decode_all(&frame[..]).unwrap_or_else(|e| panic!("{what}: libzstd {e}"));
    assert!(theirs == literals, "{what}: libzstd mismatch");
}

#[test]
fn literals_sections_match_libzstd_and_decode() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/huf");
    let chains = std::fs::read_to_string(dir.join("chains.txt")).unwrap();
    let mut n_steps = 0;
    for line in chains
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut it = line.split_whitespace();
        let chain = it.next().unwrap();
        let cp = cparams(strategy(it.next().unwrap().parse().unwrap()), 0);
        let mut prev = HufState::None;
        let mut sections = Vec::new();
        let mut all_literals = Vec::new();
        for step in it {
            let (input, nb_seq) = step.split_once(':').unwrap();
            let nb_seq: usize = nb_seq.parse().unwrap();
            let literals = std::fs::read(dir.join(format!("{input}.in"))).unwrap();
            let expected = std::fs::read(dir.join(format!("{chain}__{input}.expected"))).unwrap();
            let mut out = Vec::new();
            prev = compress_literals_with(&mut out, &literals, nb_seq, &prev, &cp);
            assert!(
                out == expected,
                "{chain}/{input}: section differs from libzstd"
            );
            sections.push(out);
            all_literals.extend_from_slice(&literals);
            n_steps += 1;
        }
        decodes_to(&sections, &all_literals, chain);
    }
    assert_eq!(n_steps, 15);
}

#[test]
fn negative_level_disables_literal_compression() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/huf");
    let literals = std::fs::read(dir.join("skew3000_0.in")).unwrap();
    let mut out = Vec::new();
    let next = compress_literals_with(
        &mut out,
        &literals,
        150,
        &HufState::None,
        &cparams(Strategy::Fast, 0),
    );
    assert_eq!(out[0] & 3, 2, "compressible input compresses at level 1");
    assert!(matches!(next, HufState::Check(_)));

    // ZSTD_literalsCompressionIsDisabled: ZSTD_fast with targetLength > 0
    let mut out = Vec::new();
    let next = compress_literals_with(
        &mut out,
        &literals,
        150,
        &HufState::None,
        &cparams(Strategy::Fast, 1),
    );
    assert_eq!(out[0] & 3, 0, "raw block");
    assert_eq!(
        out.len(),
        2 + literals.len(),
        "2-byte raw header + literals"
    );
    assert_eq!(next, HufState::None);
    decodes_to(&[out], &literals, "target_length 1");
}
