//! Job 0 indexes its first byte at `ZSTD_WINDOW_START_INDEX`, as libzstd
//! does, so its search starts at the second byte and can match the first.
//! Before, `src[0]` sat at the empty-entry index and the search began one
//! byte later, which changed whole parses: 9-symbol random text (the block
//! splitter's input) came out with 977 sequences at L1 where libzstd has
//! 1471, and a 1 MiB f64 array (QA's `f64_1M`) 560 bytes larger than
//! libzstd's at L22.

mod common;

use common::{assert_gate, c_compress2, lcg_bytes};
use rust_zstd::{compress_with, CompressOptions};
use zstd::zstd_safe::zstd_sys as sys;

/// `0xff` then random letters of `"abcdefgh "`.
fn nine(len: usize) -> Vec<u8> {
    let mut v = vec![0xffu8];
    v.extend(
        lcg_bytes(len, 9)
            .iter()
            .map(|b| b"abcdefgh "[(b % 9) as usize]),
    );
    v.truncate(len);
    v
}

/// QA's `f64_1M`: `0.0, 1.0, .., 131071.0` little-endian.
fn f64_1m() -> Vec<u8> {
    (0..131072u64)
        .flat_map(|i| (i as f64).to_le_bytes())
        .collect()
}

/// Our frame and libzstd's at `level`, both splitters at their defaults or
/// both off.
fn check(name: &str, data: &[u8], level: i32, splitters: bool) {
    use sys::ZSTD_cParameter::{
        ZSTD_c_compressionLevel, ZSTD_c_experimentalParam13, ZSTD_c_experimentalParam20,
    };
    let mut opts = CompressOptions {
        level,
        ..Default::default()
    };
    let mut params = vec![(ZSTD_c_compressionLevel, level)];
    if !splitters {
        opts.block_splitter_level = 1;
        opts.split_after_sequences = rust_zstd::ParamSwitch::Disable;
        // ZSTD_c_blockSplitterLevel 1, ZSTD_c_splitAfterSequences disable.
        params.push((ZSTD_c_experimentalParam20, 1));
        params.push((ZSTD_c_experimentalParam13, 2));
    }
    let ours = compress_with(data, &opts);
    let theirs = c_compress2(data, &params);
    assert_gate(
        &format!("{name} L{level} splitters {splitters}"),
        data,
        &ours,
        &theirs,
    );
}

#[test]
fn nine_symbol_text_matches_libzstd() {
    let data = nine(128 << 10);
    for level in [1, 3, -1] {
        check("nine", &data, level, true);
    }
}

#[test]
fn f64_array_opt_levels_match_libzstd() {
    let data = f64_1m();
    for level in 16..=22 {
        for splitters in [false, true] {
            check("f64_1M", &data, level, splitters);
        }
    }
}
