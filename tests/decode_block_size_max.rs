//! Block_Maximum_Size = min(Window_Size, 128 KiB) (RFC 8878 lines
//! 557-564): hand-built frames at the boundary of each limit it puts on a
//! block get the RFC's accept / reject outcome, serial and MT at both SIMD
//! levels. Each frame also states libzstd 1.5.7's one-shot
//! (ZSTD_decompressDCtx) outcome, checked, and where both accept the bytes
//! must match.

mod common;

use rust_zstd::decode::{decompress_with_options, DecodeOptions};

const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

enum Block {
    Raw(Vec<u8>),
    Rle(u8, usize),
    Compressed(Vec<u8>),
}

/// Frame header without Frame_Content_Size, of Window_Descriptor `wd`.
fn windowed(wd: u8) -> Vec<u8> {
    let mut h = MAGIC.to_vec();
    h.extend_from_slice(&[0x00, wd]);
    h
}

/// Single-segment frame header: the window is the content size.
fn single_segment(content_size: u32) -> Vec<u8> {
    let mut h = MAGIC.to_vec();
    h.push(0x20 | (2 << 6));
    h.extend_from_slice(&content_size.to_le_bytes());
    h
}

fn frame(header: &[u8], blocks: &[Block]) -> Vec<u8> {
    let mut f = header.to_vec();
    for (i, b) in blocks.iter().enumerate() {
        let (ty, size, body) = match b {
            Block::Raw(v) => (0, v.len(), &v[..]),
            Block::Rle(byte, n) => (1, *n, std::slice::from_ref(byte)),
            Block::Compressed(v) => (2, v.len(), &v[..]),
        };
        let h = u32::from(i + 1 == blocks.len()) | (ty << 1) | ((size as u32) << 3);
        f.extend_from_slice(&h.to_le_bytes()[..3]);
        f.extend_from_slice(body);
    }
    f
}

/// Header of a Raw (type 0) or RLE (type 1) literals section of `n` bytes.
fn plain_literals_header(ty: u32, n: usize) -> Vec<u8> {
    let n = n as u32;
    match n {
        0..32 => vec![(ty | n << 3) as u8],
        32..4096 => (ty | 1 << 2 | n << 4).to_le_bytes()[..2].to_vec(),
        _ => (ty | 3 << 2 | n << 4).to_le_bytes()[..3].to_vec(),
    }
}

/// Compressed block of `n` raw literals and no sequences.
fn raw_literals_block(n: usize) -> Vec<u8> {
    let mut v = plain_literals_header(0, n);
    v.extend((0..n).map(|i| b'a' + (i % 23) as u8));
    v.push(0);
    v
}

/// Compressed block of exactly `size` bytes of raw literals.
fn raw_literals_block_of_size(size: usize) -> Vec<u8> {
    let n = (0..size)
        .rev()
        .find(|&n| raw_literals_block(n).len() <= size);
    let v = raw_literals_block(n.unwrap());
    assert_eq!(v.len(), size);
    v
}

/// RLE literals section of `n` copies of `byte`.
fn rle_literals(n: usize, byte: u8) -> Vec<u8> {
    let mut v = plain_literals_header(1, n);
    v.push(byte);
    v
}

/// Huffman literals section of `n` zero bytes in four streams (Compressed,
/// or Treeless reusing the previous block's description): the description
/// gives symbols 0 and 1 one-bit codes, and every stream is zero bits.
fn huf_zero_literals(n: usize, treeless: bool) -> Vec<u8> {
    let seg = n.div_ceil(4);
    let streams: Vec<Vec<u8>> = [seg, seg, seg, n - 3 * seg]
        .iter()
        .map(|&k| {
            let mut s = vec![0u8; k / 8 + 1];
            s[k / 8] = 1 << (k % 8);
            s
        })
        .collect();
    // Direct weights: one weight (1) for symbol 0; symbol 1's is implied.
    let mut body = if treeless { vec![] } else { vec![128, 0x10] };
    for s in &streams[..3] {
        body.extend_from_slice(&(s.len() as u16).to_le_bytes());
    }
    for s in &streams {
        body.extend_from_slice(s);
    }
    let (ty, n, c) = (if treeless { 3u64 } else { 2 }, n as u64, body.len() as u64);
    let mut v = if n < 1 << 10 && c < 1 << 10 {
        (ty | 1 << 2 | n << 4 | c << 14).to_le_bytes()[..3].to_vec()
    } else if n < 1 << 14 && c < 1 << 14 {
        (ty | 2 << 2 | n << 4 | c << 18).to_le_bytes()[..4].to_vec()
    } else {
        (ty | 3 << 2 | n << 4 | c << 22).to_le_bytes()[..5].to_vec()
    };
    v.extend_from_slice(&body);
    v
}

/// Compressed block of `literals` and no sequences.
fn literals_only(literals: Vec<u8>) -> Block {
    let mut v = literals;
    v.push(0);
    Block::Compressed(v)
}

/// Compressed block of `lits` RLE literals and one sequence per entry of
/// `mls` (RLE-mode tables, so all of one match length code), each taking
/// one literal and matching `ml` bytes at repeat offset 1. It decodes to
/// `lits` plus the match lengths, the last `lits - mls.len()` literals
/// after the last sequence.
fn sequences_block(lits: usize, mls: &[u32]) -> Block {
    const ML_BASE: [u32; 21] = [
        35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387,
        32771, 65539,
    ];
    const ML_BITS: [u32; 21] = [
        1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    ];
    let code = |ml: u32| match ML_BASE.iter().rposition(|&b| b <= ml) {
        Some(i) => (32 + i as u8, ml - ML_BASE[i], ML_BITS[i]),
        None => (ml as u8 - 3, 0, 0),
    };
    let (ml_code, _, bits) = code(mls[0]);
    assert!(lits >= mls.len() && mls.iter().all(|&ml| code(ml).0 == ml_code));
    let mut v = rle_literals(lits, b'q');
    // LL, OF and ML in RLE mode: LL code 1, OF code 0 (repeat offset 1
    // after a literal), ML `ml_code`.
    v.extend_from_slice(&[mls.len() as u8, 0x54, 1, 0, ml_code]);
    // Only the match lengths have extra bits, read from below the end
    // marker down: the first sequence's first.
    let mut stream = 1u128;
    for &ml in mls {
        let extra = code(ml).1;
        assert!(extra < 1 << bits);
        stream = stream << bits | u128::from(extra);
    }
    let len = mls.len() * bits as usize / 8 + 1;
    v.extend_from_slice(&stream.to_le_bytes()[..len]);
    Block::Compressed(v)
}

/// Output capacity given to libzstd. ZSTD_decompressFrame bounds raw and
/// RLE blocks by the capacity left, and places a block's literals by it
/// (ZSTD_allocateLiteralsBuffer); a capacity well above every frame here
/// keeps its outcomes those of an ample buffer.
const CAPACITY: usize = 8 << 20;

/// Decode `f` serial and MT at both SIMD levels and check the outcome is
/// `accept`; check libzstd's one-shot outcome is `libzstd`, and where both
/// accept, that the bytes match.
fn check_vs(name: &str, f: &[u8], accept: bool, libzstd: bool) {
    let theirs = zstd::bulk::decompress(f, CAPACITY);
    assert_eq!(
        theirs.is_ok(),
        libzstd,
        "{name}: libzstd {:?}",
        theirs.as_ref().map(Vec::len)
    );
    for simd in [false, true] {
        for min_parallel_blocks in [usize::MAX, 1] {
            let options = DecodeOptions {
                min_parallel_blocks,
                simd,
            };
            let ours = decompress_with_options(f, &options);
            let at = format!("{name} simd={simd} min_parallel_blocks={min_parallel_blocks}");
            assert_eq!(
                ours.is_ok(),
                accept,
                "{at}: ours {:?}",
                ours.as_ref().map(Vec::len)
            );
            if let (Ok(a), Ok(b)) = (&theirs, &ours) {
                assert!(a == b, "{at}: output differs");
            }
        }
    }
    common::assert_stream_parity(name, f);
}

/// `check_vs` for a frame libzstd gives the same outcome.
fn check(name: &str, f: &[u8], accept: bool) {
    check_vs(name, f, accept, accept);
}

/// (Window_Descriptor, Block_Maximum_Size): windows of 1 KiB and 96 KiB,
/// below 128 KiB, and 1 MiB, above it.
const WINDOWS: [(u8, usize); 3] = [(0, 1024), (6 << 3 | 4, 96 << 10), (10 << 3, 128 << 10)];

/// `case` between a raw block and an RLE block, so that it is neither the
/// first nor the last block of the frame.
fn around(wd: u8, case: Block) -> Vec<u8> {
    frame(
        &windowed(wd),
        &[
            Block::Raw(b"0123456789".to_vec()),
            case,
            Block::Rle(b'z', 5),
        ],
    )
}

#[test]
fn compressed_block_size_is_at_most_block_maximum_size() {
    for (wd, max) in WINDOWS {
        for (size, accept) in [(max - 1, true), (max, true), (max + 1, false)] {
            let f = around(wd, Block::Compressed(raw_literals_block_of_size(size)));
            check(
                &format!("max {max}: compressed block of {size}"),
                &f,
                accept,
            );
        }
    }
}

#[test]
fn literals_size_is_at_most_block_maximum_size() {
    for (wd, max) in WINDOWS {
        for (n, accept) in [(max, true), (max + 1, false)] {
            let f = around(wd, literals_only(rle_literals(n, 7)));
            check(&format!("max {max}: {n} RLE literals"), &f, accept);
            let f = around(wd, literals_only(huf_zero_literals(n, false)));
            check(&format!("max {max}: {n} Huffman literals"), &f, accept);
            let f = frame(
                &windowed(wd),
                &[
                    literals_only(huf_zero_literals(100, false)),
                    literals_only(huf_zero_literals(n, true)),
                ],
            );
            check(&format!("max {max}: {n} treeless literals"), &f, accept);
        }
    }
}

/// Block_Maximum_Size bounds what a compressed block decodes to as well
/// (RFC 8878 lines 566-568): a block decoding to that many bytes is
/// accepted and to one more rejected, whether the last sequence or the
/// literals after it cross the bound. libzstd's one-shot decoder bounds it
/// by where it keeps the literals, Block_Maximum_Size + WILDCOPY_OVERLENGTH
/// (32) past the block's start, so it takes up to 32 bytes more.
#[test]
fn decoded_size_is_at_most_block_maximum_size() {
    for (wd, max) in WINDOWS {
        for (n, accept, libzstd) in [
            (max, true, true),
            (max + 1, false, true),
            (max + 32, false, true),
            (max + 33, false, false),
        ] {
            let ml = (n - 2) as u32;
            let f = around(wd, sequences_block(2, &[ml / 2, ml - ml / 2]));
            check_vs(
                &format!("max {max}: sequences decoding to {n}"),
                &f,
                accept,
                libzstd,
            );
            let lits = n / 2;
            let f = around(wd, sequences_block(lits, &[(n - lits) as u32]));
            check_vs(
                &format!("max {max}: sequence then literals decoding to {n}"),
                &f,
                accept,
                libzstd,
            );
        }
    }
}

/// A raw or RLE block's Block_Size is bounded by Block_Maximum_Size too
/// (RFC 8878 lines 545-555), in windowed frames and in single-segment
/// frames, whose window is the content size. libzstd's one-shot decoder
/// bounds them by its output capacity alone (ZSTD_copyRawBlock,
/// ZSTD_setRleBlock), up to the 21-bit Block_Size maximum.
#[test]
fn raw_and_rle_block_size_is_at_most_block_maximum_size() {
    const BLOCK_SIZE_FIELD_MAX: usize = (1 << 21) - 1;
    for (wd, max) in WINDOWS {
        for (n, accept) in [
            (max, true),
            (max + 1, false),
            ((128 << 10) + 1, false),
            (BLOCK_SIZE_FIELD_MAX, false),
        ] {
            let f = around(wd, Block::Raw(vec![3; n]));
            check_vs(&format!("max {max}: raw block of {n}"), &f, accept, true);
            let f = around(wd, Block::Rle(4, n));
            check_vs(&format!("max {max}: RLE block of {n}"), &f, accept, true);
        }
    }
    for (n, accept) in [
        (128 << 10, true),
        ((128 << 10) + 1, false),
        (BLOCK_SIZE_FIELD_MAX, false),
    ] {
        let f = frame(&single_segment(n as u32), &[Block::Raw(vec![5; n])]);
        check_vs(
            &format!("single segment: raw block of {n}"),
            &f,
            accept,
            true,
        );
        let f = frame(&single_segment(n as u32), &[Block::Rle(6, n)]);
        check_vs(
            &format!("single segment: RLE block of {n}"),
            &f,
            accept,
            true,
        );
    }
    let n = 128 << 10;
    let f = frame(
        &single_segment(2 * n as u32),
        &[Block::Raw(vec![7; n]), Block::Rle(8, n)],
    );
    check("single segment: two blocks of 128 KiB", &f, true);
}

/// A single-segment frame's window is its content size, below 1 KiB too:
/// a raw block of `r` bytes, then a compressed block of `n` raw literals
/// whose size, `n + 3`, is the window exactly for `r = 3`.
#[test]
fn single_segment_block_maximum_size_is_the_content_size() {
    for n in [100, 3000] {
        for (r, accept) in [(3, true), (4, true), (2, false)] {
            let blocks = [
                Block::Raw(vec![1; r]),
                Block::Compressed(raw_literals_block(n)),
            ];
            let f = frame(&single_segment((r + n) as u32), &blocks);
            check(&format!("{n} literals after {r} raw bytes"), &f, accept);
        }
    }
}
