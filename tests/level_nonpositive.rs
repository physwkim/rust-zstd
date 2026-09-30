//! Levels `<= 0` as libzstd reads them (`ZSTD_getCParams_internal`): `0`
//! is `ZSTD_CLEVEL_DEFAULT`, a negative level is row 0's fast strategy with
//! `targetLength = -level`. Whole frames must equal libzstd's.

mod common;

use rust_zstd::{compress_with, decompress, CompressOptions};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// `\x00\x01` followed by words of a 16-word vocabulary.
fn text(len: usize) -> Vec<u8> {
    const WORDS: [&str; 16] = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "cat", "sat", "on",
        "a", "mat", "with", "hat",
    ];
    let mut rng = Lcg(7);
    let mut out = vec![0u8, 1];
    while out.len() < len {
        out.extend_from_slice(WORDS[(rng.next() % 16) as usize].as_bytes());
        out.push(b' ');
    }
    out.truncate(len);
    out
}

fn random(len: usize) -> Vec<u8> {
    let mut rng = Lcg(11);
    (0..len).map(|_| (rng.next() >> 16) as u8).collect()
}

fn check(name: &str, data: &[u8], level: i32) -> Vec<u8> {
    let ours = compress_with(
        data,
        &CompressOptions {
            level,
            ..Default::default()
        },
    );
    assert!(
        decompress(&ours).unwrap() == data,
        "{name} L{level}: roundtrip"
    );
    assert!(
        ours == common::zstd_bulk(data, level),
        "{name} L{level}: frame differs from libzstd"
    );
    ours
}

#[test]
fn nonpositive_levels_match_libzstd() {
    let text = text(300 << 10);
    for level in [0, -1] {
        let frame = check("text", &text, level);
        assert!(
            frame.len() < text.len() / 2,
            "text L{level}: not compressed"
        );
    }
    for len in [1, 1000, 200 << 10] {
        let data = random(len);
        for level in [0, -1, -2, -5, -7, -100, -131072] {
            check(&format!("random {len}"), &data, level);
        }
    }
}
