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
                window_log_max: 0,
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

extern "C" {
    // lib/common/fse.h and huf.h; linked from zstd-sys's static libzstd.
    fn FSE_writeNCount(
        buffer: *mut u8,
        buffer_size: usize,
        normalized_counter: *const i16,
        max_symbol_value: u32,
        table_log: u32,
    ) -> usize;
    fn HUF_readStats(
        huff_weight: *mut u8,
        hw_size: usize,
        rank_stats: *mut u32,
        nb_symbols: *mut u32,
        table_log: *mut u32,
        src: *const u8,
        src_size: usize,
    ) -> usize;
}

fn is_error(r: usize) -> bool {
    // SAFETY: a pure function of its argument.
    unsafe { zstd::zstd_safe::zstd_sys::ZSTD_isError(r) != 0 }
}

/// The FSE table description, by libzstd's FSE_writeNCount, of accuracy
/// log `log` in which weight 1 has every cell but one and symbol `listed`
/// the last (probability "less than 1"), which no state decodes to.
fn ncount_listing(listed: usize, log: u32) -> Vec<u8> {
    let mut norm = vec![0i16; listed + 1];
    norm[1] = (1 << log) - 1;
    norm[listed] = -1;
    let mut out = [0u8; 64];
    // SAFETY: the buffers have the sizes passed.
    let n = unsafe {
        FSE_writeNCount(
            out.as_mut_ptr(),
            out.len(),
            norm.as_ptr(),
            listed as u32,
            log,
        )
    };
    assert!(!is_error(n), "FSE_writeNCount");
    out[..n].to_vec()
}

/// A Huffman tree description of FSE-compressed weights: header byte,
/// `ncount`, then weight bitstream `stream`.
fn description(ncount: &[u8], stream: &[u8]) -> Vec<u8> {
    let len = (ncount.len() + stream.len()) as u8;
    [&[len][..], ncount, stream].concat()
}

/// The description with `ncount` and the shortest weight bitstream
/// HUF_readStats accepts, if any.
fn accepted_description(ncount: &[u8]) -> Option<Vec<u8>> {
    (1..=2usize).find_map(|len| {
        (1u32 << (8 * len - 8)..1 << (8 * len)).find_map(|bits| {
            let desc = description(ncount, &bits.to_le_bytes()[..len]);
            let mut weights = [0u8; 256];
            let mut rank_stats = [0u32; 13];
            let (mut nb_symbols, mut table_log) = (0u32, 0u32);
            // SAFETY: the buffers have the sizes HUF_readStats is given.
            let r = unsafe {
                HUF_readStats(
                    weights.as_mut_ptr(),
                    weights.len(),
                    rank_stats.as_mut_ptr(),
                    &mut nb_symbols,
                    &mut table_log,
                    desc.as_ptr(),
                    desc.len(),
                )
            };
            (!is_error(r) && r == desc.len()).then_some(desc)
        })
    })
}

/// A frame (1 KiB window, no content size) of one block holding two
/// literals coded (one stream) with tree description `desc` into
/// `stream`, and no sequences.
fn huffman_frame(desc: &[u8], stream: &[u8]) -> Vec<u8> {
    const REGEN: u32 = 2;
    let comp = (desc.len() + stream.len()) as u32;
    let header = 2 | REGEN << 4 | comp << 14;
    let mut body = header.to_le_bytes()[..3].to_vec();
    body.extend_from_slice(desc);
    body.extend_from_slice(stream);
    body.push(0);
    let mut f = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x00];
    let bh = (body.len() as u32) << 3 | 2 << 1 | 1;
    f.extend_from_slice(&bh.to_le_bytes()[..3]);
    f.extend_from_slice(&body);
    f
}

/// R3-2: the weights' FSE table may list symbols 0 to 11 only, the
/// weights a tree can have, at every accuracy log (RFC 8878 lines
/// 1432-1436, 1541-1543). libzstd's bound depends on the log: of the R3-A
/// probe shapes it accepts a listed 12, 20 or 91 at log 5 and rejects 12 at
/// log 6, as its workspace allows.
#[test]
fn weight_table_lists_symbols_up_to_11() {
    for (listed, log, libzstd_accepts) in [
        (11, 5, true),
        (11, 6, true),
        (12, 5, true),
        (20, 5, true),
        (91, 5, true),
        (12, 6, false),
    ] {
        let name = format!("weights listing {listed} at log {log}");
        let ncount = ncount_listing(listed, log);
        let Some(desc) = accepted_description(&ncount) else {
            assert!(!libzstd_accepts, "{name}: HUF_readStats rejects it");
            // Our verdict comes from the table description alone.
            let f = huffman_frame(&description(&ncount, &[0x81]), &[0x81]);
            assert!(zstd::bulk::decompress(&f, 64).is_err(), "{name}");
            check(&name, &f, Err(SYMBOL_PAST_11));
            continue;
        };
        assert!(libzstd_accepts, "{name}: HUF_readStats accepts it");
        let (f, out) = (1..1u32 << 16)
            .find_map(|s| {
                let stream = &s.to_le_bytes()[..if s < 256 { 1 } else { 2 }];
                let f = huffman_frame(&desc, stream);
                zstd::bulk::decompress(&f, 64).ok().map(|out| (f, out))
            })
            .unwrap_or_else(|| panic!("{name}: no literal stream libzstd decodes"));
        if listed <= 11 {
            check(&name, &f, Ok(&out));
        } else {
            check(&name, &f, Err(SYMBOL_PAST_11));
        }
    }
}

/// What a weight table listing a symbol past 11 fails with: its counts
/// stop at symbol 11 with cells left over.
const SYMBOL_PAST_11: &str = "unassigned";
