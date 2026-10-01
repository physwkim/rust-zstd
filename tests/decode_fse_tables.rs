//! FSE table descriptions RFC 8878 rejects and libzstd 1.5.7 decodes: the
//! verdict is the RFC's, the same serial and MT at both SIMD levels and
//! streaming, while libzstd's decoding of the same frame is asserted to
//! show the divergence.

mod common;

use rust_zstd::decode::{decompress_with_options, DecodeOptions};

/// A single-segment frame of `content_size` bytes: a raw block of
/// "abcdefgh", then compressed block `body`, the last.
fn frame(content_size: u8, body: &[u8]) -> Vec<u8> {
    let mut f = vec![0x28, 0xB5, 0x2F, 0xFD, 0x20, content_size];
    f.extend_from_slice(&[8 << 3, 0, 0]);
    f.extend_from_slice(b"abcdefgh");
    let header = (body.len() as u32) << 3 | 2 << 1 | 1;
    f.extend_from_slice(&header.to_le_bytes()[..3]);
    f.extend_from_slice(body);
    f
}

/// `f` decodes to `want` on every path when `want` is given, and fails
/// with an error containing `error` otherwise.
fn check(name: &str, f: &[u8], want: Result<&[u8], &str>) {
    for simd in [false, true] {
        for min_parallel_blocks in [usize::MAX, 1] {
            let opts = DecodeOptions {
                min_parallel_blocks,
                simd,
            };
            let got = decompress_with_options(f, &opts);
            let path = format!("{name} simd={simd} min_parallel_blocks={min_parallel_blocks}");
            match (want, &got) {
                (Ok(w), Ok(g)) => assert!(g == w, "{path}: wrong output"),
                (Err(e), Err(g)) => assert!(g.contains(e), "{path}: {g}"),
                _ => panic!("{path}: {got:?}"),
            }
        }
    }
    common::assert_stream_parity(name, f);
}

/// R3-1: an LL table in FSE_Compressed_Mode whose only symbol of nonzero
/// probability is code 0 (32 of 32 at accuracy log 5) is rejected (RFC
/// 8878 lines 1372-1373, 925-927). libzstd decodes it as the RLE_Mode
/// frame for code 0.
#[test]
fn one_symbol_fse_table_is_rejected() {
    // Literals "xyz" (raw); one sequence of literal length 0, offset 8
    // (Offset_Value 11: OF code 3, extra bits 3) and match length 3 (ML
    // code 0); OF and ML in RLE_Mode.
    let lits = [0x18, b'x', b'y', b'z', 0x01];
    // LL table: accuracy log 5 (4 bits 0), then count 32 + 1 = 33 as the
    // 6-bit value 63; the bitstream holds 5 bits of LL state then 3 offset
    // extra bits.
    let fse = [&lits[..], &[0x94, 0xF0, 0x03, 3, 0, 0x03, 0x01]].concat();
    // The same sequence with LL in RLE_Mode: no LL state bits.
    let rle = [&lits[..], &[0x54, 0, 3, 0, 0x0B]].concat();
    let want = b"abcdefghabcxyz";
    check("RLE LL", &frame(14, &rle), Ok(want));
    let probe = frame(14, &fse);
    assert_eq!(
        zstd::bulk::decompress(&probe, 64).as_deref().ok(),
        Some(&want[..]),
        "libzstd no longer decodes the one-symbol table"
    );
    check("one-symbol LL", &probe, Err("fewer than two symbols"));
}
