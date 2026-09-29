//! Byte-exact verification of `rust_zstd::decompress` against streams produced
//! by libzstd 1.5.7 (via the `zstd` crate), plus malformed-input robustness.

mod common;

use common::{datasets, lcg_bytes, zstd_bulk, zstd_stream, LEVELS};

#[test]
fn decodes_libzstd_streams_byte_exact() {
    for ds in datasets() {
        for &level in &LEVELS {
            for (kind, compressed) in [
                ("bulk", zstd_bulk(&ds.data, level)),
                ("stream", zstd_stream(&ds.data, level)),
            ] {
                let decoded = rust_zstd::decompress(&compressed).unwrap_or_else(|e| {
                    panic!(
                        "{} level {} ({}): decode error: {}",
                        ds.name, level, kind, e
                    )
                });
                assert!(
                    decoded == ds.data,
                    "{} level {} ({}): output differs (got {} bytes, want {})",
                    ds.name,
                    level,
                    kind,
                    decoded.len(),
                    ds.data.len()
                );
            }
        }
    }
}

#[test]
fn decodes_concatenated_frames_and_skippable_frames() {
    let a = b"first frame payload ".repeat(50);
    let b = lcg_bytes(3000, 7);
    let mut stream = zstd_bulk(&a, 3);
    // Skippable frame: magic 0x184D2A5? + LE32 size + payload.
    stream.extend_from_slice(&0x184D2A50u32.to_le_bytes());
    stream.extend_from_slice(&4u32.to_le_bytes());
    stream.extend_from_slice(&[1, 2, 3, 4]);
    stream.extend_from_slice(&zstd_stream(&b, 1));
    let decoded = rust_zstd::decompress(&stream).unwrap();
    let mut want = a.clone();
    want.extend_from_slice(&b);
    assert!(decoded == want);
}

/// Truncating a valid stream at every byte offset must yield `Err` (or, for
/// the empty prefix, `Ok(empty)`), never a panic.
#[test]
fn truncated_streams_return_err_without_panic() {
    let inputs: Vec<(&str, Vec<u8>)> = vec![
        (
            "text",
            b"The quick brown fox jumps over the lazy dog. ".repeat(40),
        ),
        ("random", lcg_bytes(600, 3)),
        ("mixed", {
            let mut v = lcg_bytes(300, 9);
            v.extend_from_slice(&b"abcabcabc".repeat(60));
            v
        }),
    ];
    for (name, data) in inputs {
        for &level in &[1, 3, 19] {
            for compressed in [zstd_bulk(&data, level), zstd_stream(&data, level)] {
                assert_eq!(rust_zstd::decompress(&compressed).unwrap(), data);
                for cut in 0..compressed.len() {
                    let prefix = &compressed[..cut];
                    let result = std::panic::catch_unwind(|| rust_zstd::decompress(prefix));
                    let result = result.unwrap_or_else(|_| {
                        panic!(
                            "{} level {}: panic at truncation offset {}",
                            name, level, cut
                        )
                    });
                    if cut == 0 {
                        assert_eq!(result.unwrap(), Vec::<u8>::new());
                    } else {
                        assert!(
                            result.is_err(),
                            "{} level {}: truncation at {} of {} did not error",
                            name,
                            level,
                            cut,
                            compressed.len()
                        );
                    }
                }
            }
        }
    }
}

/// Corrupting a valid stream (every byte position, several bit patterns) must
/// never panic. The output may be an error or garbage; both are acceptable.
#[test]
fn corrupted_streams_never_panic() {
    let data = {
        let mut v = b"The quick brown fox jumps over the lazy dog. ".repeat(30);
        v.extend_from_slice(&lcg_bytes(400, 11));
        v
    };
    for &level in &[1, 3, 19] {
        for compressed in [zstd_bulk(&data, level), zstd_stream(&data, level)] {
            for pos in 0..compressed.len() {
                for flip in [0x01u8, 0x80, 0xFF, 0x55] {
                    let mut bad = compressed.clone();
                    bad[pos] ^= flip;
                    let result = std::panic::catch_unwind(|| rust_zstd::decompress(&bad));
                    assert!(
                        result.is_ok(),
                        "level {}: panic with byte {} xor {:#x}",
                        level,
                        pos,
                        flip
                    );
                }
            }
        }
    }
}
