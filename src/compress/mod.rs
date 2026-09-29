//! Zstandard frame compressor.
//!
//! libzstd's block architecture (`ZSTD_compress_frameChunk`): match finding
//! runs per block on a persistent [`MatchState`], sequences never cross a
//! block boundary, and the entropy stage consumes a per-block [`SeqStore`]
//! against the committed cross-block [`BlockState`].

pub mod block;
pub mod dfast;
pub mod fast;
pub mod lazy;
pub mod matchstate;
pub mod params;
pub mod seqstore;

use crate::constants::*;
use block::{write_raw_block, write_rle_block, BlockScratch, BlockState, CommittedBlockState};
use matchstate::MatchState;
pub use params::{CParams, Strategy};
use params::{ZSTD_CLEVEL_DEFAULT, ZSTD_WINDOWLOG_ABSOLUTEMIN};
pub use seqstore::{Seq, SeqStore};

/// Options for [`compress_with`].
#[derive(Clone, Debug)]
pub struct CompressOptions {
    /// Compression level, `ZSTD_c_compressionLevel`. `<= 0` emits raw/RLE
    /// blocks only; `1..=22` map to libzstd's parameter rows.
    pub level: i32,
    /// Job size for multi-threaded compression. Accepted but not used yet:
    /// every frame is compressed as a single job.
    pub job_size: Option<usize>,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            level: ZSTD_CLEVEL_DEFAULT,
            job_size: None,
        }
    }
}

/// Compress `data` into a zstd frame at `level`.
///
/// Returns a valid zstd frame decompressible by any conformant decoder.
pub fn compress(data: &[u8], level: i32) -> Vec<u8> {
    compress_with(
        data,
        &CompressOptions {
            level,
            job_size: None,
        },
    )
}

/// Convenience wrapper: [`compress`] at level 1.
pub fn compress_to_vec(data: &[u8]) -> Vec<u8> {
    compress(data, 1)
}

/// Compress `data` into a zstd frame with `opts`.
pub fn compress_with(data: &[u8], opts: &CompressOptions) -> Vec<u8> {
    assert!(
        data.len() < u32::MAX as usize,
        "inputs of 4 GiB or more are not supported (match indices are u32)"
    );
    let cparams = CParams::for_level(opts.level, data.len());
    let mut out = Vec::with_capacity(data.len() + 64);
    write_frame_header(&mut out, data.len() as u64, cparams.window_log);

    if data.is_empty() {
        write_raw_block(&mut out, &[], true);
        return out;
    }

    // blockSizeMax = MIN(ZSTD_BLOCKSIZE_MAX, 1 << windowLog)
    let block_size = ZSTD_BLOCKSIZE_MAX.min(1usize << cparams.window_log);
    let n_blocks = data.len().div_ceil(block_size);

    if opts.level <= 0 {
        for (i, chunk) in data.chunks(block_size).enumerate() {
            let is_last = i + 1 == n_blocks;
            if block::is_rle(chunk) {
                write_rle_block(&mut out, chunk[0], chunk.len(), is_last);
            } else {
                write_raw_block(&mut out, chunk, is_last);
            }
        }
        return out;
    }

    // Single job: rep = repStartValue, window_low = 1.
    let mut ms = MatchState::new(cparams, 1);
    let mut state = CommittedBlockState::new(BlockState::initial());
    let mut scratch = BlockScratch::new(block_size);
    let mut start = 0usize;
    while start < data.len() {
        let end = (start + block_size).min(data.len());
        block::compress_block(
            &mut ms,
            data,
            start..end,
            start == 0,
            end == data.len(),
            &mut state,
            &mut scratch,
            &mut out,
        );
        start = end;
    }
    out
}

/// `ZSTD_writeFrameHeader` with no dictionary and no checksum:
/// Single_Segment iff the window covers the whole content, otherwise a
/// Window_Descriptor with mantissa 0 derived from `window_log`.
fn write_frame_header(out: &mut Vec<u8>, content_size: u64, window_log: u32) {
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
    let window_size = 1u64 << window_log;
    let single_segment = window_size >= content_size;
    let fcs_code = (content_size >= 256) as u8
        + (content_size >= 65536 + 256) as u8
        + (content_size >= 0xFFFF_FFFF) as u8;
    let descriptor = ((single_segment as u8) << 5) | (fcs_code << 6);
    out.push(descriptor);
    if !single_segment {
        out.push(((window_log - ZSTD_WINDOWLOG_ABSOLUTEMIN) << 3) as u8);
    }
    match fcs_code {
        0 => {
            if single_segment {
                out.push(content_size as u8);
            }
        }
        1 => out.extend_from_slice(&((content_size - 256) as u16).to_le_bytes()),
        2 => out.extend_from_slice(&(content_size as u32).to_le_bytes()),
        _ => out.extend_from_slice(&content_size.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_empty() {
        let compressed = compress(&[], 1);
        assert!(compressed.len() >= 5); // magic + header + empty block
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert_eq!(crate::decompress(&compressed).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn compress_small() {
        let data = b"hello world";
        let compressed = compress(data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert!(compressed.len() > 5);
        assert_eq!(crate::decompress(&compressed).unwrap(), data);
    }

    #[test]
    fn compress_repetitive() {
        let data = vec![42u8; 4096];
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert_eq!(crate::decompress(&compressed).unwrap(), data);
    }

    #[test]
    fn compress_real_data() {
        let data: Vec<u8> = (0..1024u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert_eq!(crate::decompress(&compressed).unwrap(), data);
    }

    /// Golden test: roundtrip through our compressor → our decompressor.
    #[test]
    fn roundtrip_self_contained() {
        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros", vec![0u8; 4096]),
            (
                "sequential",
                (0..4096u32).flat_map(|i| i.to_le_bytes()).collect(),
            ),
            (
                "f32_data",
                (0..256u32)
                    .flat_map(|i| (i as f32 * 1.5).to_le_bytes())
                    .collect(),
            ),
            ("repetitive", b"hello world! ".repeat(100)),
            ("small", b"abc".to_vec()),
        ];

        for (name, data) in &test_cases {
            for level in [0, 1, 3, 7, 11, 19] {
                let compressed = compress(data, level);
                let decompressed = crate::decompress(&compressed)
                    .unwrap_or_else(|e| panic!("{}: decompress failed: {}", name, e));
                assert_eq!(decompressed.len(), data.len(), "{}: length mismatch", name);
                assert_eq!(&decompressed, data, "{}: data mismatch", name);
            }
        }
    }

    /// A RAW block between two compressed blocks must not advance the
    /// committed Huffman table: block 3 may only go Treeless against the
    /// tree the decoder actually saw in block 1.
    #[test]
    fn raw_block_between_compressed_blocks_keeps_prev_state() {
        let mut data = Vec::new();
        let text: Vec<u8> = (0..ZSTD_BLOCKSIZE_MAX)
            .map(|i| b"the quick brown fox "[i % 20])
            .collect();
        data.extend_from_slice(&text);
        // pseudo-random, incompressible
        let mut x = 0x2545F4914F6CDD1Du64;
        for _ in 0..ZSTD_BLOCKSIZE_MAX {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            data.push((x >> 56) as u8);
        }
        data.extend_from_slice(&text);
        for level in [1, 3, 7] {
            let c = compress(&data, level);
            assert_eq!(crate::decompress(&c).unwrap(), data, "level {level}");
        }
    }

    #[test]
    fn frame_header_window_descriptor_for_large_input() {
        // 1 MiB at level 1: windowLog 19 < 20 bits of content -> not single segment
        let data = vec![1u8; 1 << 20];
        let c = compress(&data, 1);
        let descriptor = c[4];
        assert_eq!(descriptor & (1 << 5), 0, "Single_Segment must be clear");
        assert_eq!(c[5] >> 3, 19 - 10, "Window_Descriptor exponent");
        assert_eq!(crate::decompress(&c).unwrap(), data);
    }
}
