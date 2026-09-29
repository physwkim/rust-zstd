//! Real-data roundtrips: the crate's own sources (text) and the test binary
//! itself (an ELF), byte-exact through our decoder and through libzstd.

use std::fs;
use std::path::Path;

const LEVELS: [i32; 4] = [1, 3, 7, 11];

fn crate_sources() -> Vec<u8> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut paths = Vec::new();
    let mut stack = vec![src];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                paths.push(path);
            }
        }
    }
    paths.sort();
    let mut data = Vec::new();
    for p in paths {
        data.extend_from_slice(&fs::read(p).unwrap());
    }
    assert!(
        data.len() > 100_000,
        "expected the crate sources to be > 100 KB"
    );
    data
}

fn current_exe_bytes() -> Vec<u8> {
    let exe = std::env::current_exe().unwrap();
    let data = fs::read(exe).unwrap();
    assert_eq!(&data[..4], b"\x7fELF");
    data
}

fn check(name: &str, data: &[u8]) {
    for level in LEVELS {
        let compressed = rust_zstd::compress(data, level);
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
        eprintln!(
            "{name} level {level}: {} -> {} ({:.3}x)",
            data.len(),
            compressed.len(),
            data.len() as f64 / compressed.len() as f64
        );
    }
}

#[test]
fn crate_sources_roundtrip() {
    check("src/*.rs", &crate_sources());
}

#[test]
fn crate_sources_tiled_to_8mib_roundtrip() {
    let unit = crate_sources();
    let mut data = Vec::with_capacity(8 << 20);
    while data.len() < 8 << 20 {
        data.extend_from_slice(&unit);
    }
    check("src/*.rs x N (8 MiB)", &data);
}

#[test]
fn current_exe_roundtrip() {
    check("current_exe", &current_exe_bytes());
}
