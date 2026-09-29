//! Regression fixtures for the FSE normalization defect: sequence-code
//! distributions in these inputs made the pre-port normalization sum to
//! something other than `1 << table_log`, which the decoders reject or
//! `FseCTable::build` panics on. Every level must decode byte-exact through
//! our decoder and through libzstd.

use std::path::Path;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name),
    )
    .unwrap()
}

fn check_all_levels(name: &str) {
    let data = fixture(name);
    for level in 1..=12 {
        let compressed = rust_zstd::compress(&data, level);
        let ours = rust_zstd::decompress(&compressed)
            .unwrap_or_else(|e| panic!("{name} level {level}: our decoder failed: {e}"));
        assert!(
            ours == data,
            "{name} level {level}: our decoder produced wrong bytes"
        );
        let theirs = zstd::stream::decode_all(&compressed[..])
            .unwrap_or_else(|e| panic!("{name} level {level}: libzstd failed: {e}"));
        assert!(
            theirs == data,
            "{name} level {level}: libzstd produced wrong bytes"
        );
    }
}

#[test]
fn minfail_all_levels() {
    check_all_levels("minfail.bin");
}

#[test]
fn minfail2_all_levels() {
    check_all_levels("minfail2.bin");
}
