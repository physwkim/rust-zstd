//! Block_Maximum_Size = min(Window_Size, 128 KiB) (fParams.blockSizeMax):
//! hand-built frames at the boundary of each limit libzstd 1.5.7's one-shot
//! decoder (ZSTD_decompressDCtx) puts on a block decode to the same
//! accept / reject outcome, and to the same bytes when accepted, serial and
//! MT at both SIMD levels.

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

/// Compressed block decoding to `1 + ml` bytes: one RLE literal and one
/// sequence (RLE-mode tables) matching `ml` bytes at repeat offset 1.
fn one_sequence_block(ml: u32) -> Block {
    const ML_BASE: [u32; 21] = [
        35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387,
        32771, 65539,
    ];
    const ML_BITS: [u32; 21] = [
        1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    ];
    let (code, extra, bits) = match ML_BASE.iter().rposition(|&b| b <= ml) {
        Some(i) => (32 + i as u8, ml - ML_BASE[i], ML_BITS[i]),
        None => (ml as u8 - 3, 0, 0),
    };
    assert!(extra < 1 << bits || bits == 0 && extra == 0);
    let mut v = rle_literals(1, b'q');
    // One sequence; LL, OF and ML in RLE mode: LL code 1, OF code 0
    // (repeat offset 1 after a literal), ML `code`.
    v.extend_from_slice(&[1, 0x54, 1, 0, code]);
    // Only the match length has extra bits; then the end marker.
    let stream = u64::from(extra) | 1 << bits;
    v.extend_from_slice(&stream.to_le_bytes()[..bits as usize / 8 + 1]);
    Block::Compressed(v)
}

/// Decode `f` with libzstd's one-shot decoder, check its outcome is
/// `accept`, and check ours gives the same outcome and bytes.
fn check(name: &str, f: &[u8], accept: bool) {
    let theirs = zstd::bulk::decompress(f, 1 << 20);
    assert_eq!(
        theirs.is_ok(),
        accept,
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
            match (&theirs, &ours) {
                (Ok(a), Ok(b)) => assert!(
                    a == b,
                    "{name} simd={simd} min_parallel_blocks={min_parallel_blocks}: output differs"
                ),
                (Err(_), Err(_)) => {}
                _ => panic!(
                    "{name} simd={simd} min_parallel_blocks={min_parallel_blocks}: libzstd {:?}, ours {:?}",
                    theirs.as_ref().map(Vec::len),
                    ours.as_ref().map(Vec::len)
                ),
            }
        }
    }
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

/// Neither bound applies to the decoded size of a compressed block, nor to
/// raw or RLE blocks, in ZSTD_decompressFrame (only ZSTD_decompressStream
/// checks them): below 128 KiB, both decoders take them past the window.
#[test]
fn decoded_and_uncompressed_blocks_past_the_window_decode_like_libzstd() {
    for (wd, max) in WINDOWS.into_iter().filter(|&(_, max)| max < 128 << 10) {
        for n in [max, max + 1] {
            let f = around(wd, one_sequence_block(n as u32 - 1));
            check(&format!("max {max}: block decoding to {n}"), &f, true);
            let f = around(wd, Block::Raw(vec![3; n]));
            check(&format!("max {max}: raw block of {n}"), &f, true);
            let f = around(wd, Block::Rle(4, n));
            check(&format!("max {max}: RLE block of {n}"), &f, true);
        }
    }
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
