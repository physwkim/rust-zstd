#![allow(clippy::manual_is_multiple_of, clippy::identity_op)]
//! Self-contained Zstandard decompressor.
//!
//! Ported from ruzstd 0.8.2 by Moritz Borcherding, used under the MIT license.
//!
//! ```text
//! MIT License
//!
//! Copyright (c) ruzstd contributors
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
//! ```
//!
//! Public API: `decompress(data: &[u8]) -> Result<Vec<u8>, String>`, and
//! `Decompressor` for input and output in pieces, `DecompressReader` over it.
//!
//! Supports raw blocks, RLE blocks, and compressed blocks with Huffman
//! literals and FSE sequences, with or without a dictionary
//! (`decompress_with_dict`).

#![allow(
    clippy::needless_range_loop,
    clippy::len_without_is_empty,
    clippy::upper_case_acronyms,
    clippy::manual_range_contains,
    dead_code
)]

use crate::constants::ZSTD_WINDOWLOG_MAX;
use crate::xxhash::Xxh64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Avx2;
use fearless_simd::{Fallback, Level};
use std::ptr;
use std::sync::LazyLock;

mod dict;
pub use dict::DecodeDict;
use dict::DictEntropy;

mod stream;
pub use stream::{DecompressReader, Decompressor};

// ============================================================
// Constants
// ============================================================

const ZSTD_MAGIC: u32 = 0xFD2F_B528;
const MIN_WINDOW_SIZE: u64 = 1024;
/// Block_Maximum_Size's 128 KiB cap (RFC 8878 lines 557-564): no block
/// of any frame has more content or decodes to more, so it is also the
/// sequence executors' constant output bound.
const MAX_BLOCK_SIZE: usize = 128 * 1024;
/// Largest Huffman table log, and so weight, the decoder takes (libzstd
/// HUF_TABLELOG_MAX is 12): RFC 8878 §4.2.1 (rfc8878.txt:1537-1540) limits
/// the maximum code length to 11 bits.
const HUF_TABLELOG_MAX: u32 = 11;
const ACC_LOG_OFFSET: u8 = 5;

const MAX_LITERAL_LENGTH_CODE: u8 = 35;
const MAX_MATCH_LENGTH_CODE: u8 = 52;
const MAX_OFFSET_CODE: u8 = 31;

const LL_MAX_LOG: u8 = 9;
const ML_MAX_LOG: u8 = 9;
const OF_MAX_LOG: u8 = 8;

const LL_DEFAULT_ACC_LOG: u8 = 6;
const ML_DEFAULT_ACC_LOG: u8 = 6;
const OF_DEFAULT_ACC_LOG: u8 = 5;

const LITERALS_LENGTH_DEFAULT_DISTRIBUTION: [i32; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];

const MATCH_LENGTH_DEFAULT_DISTRIBUTION: [i32; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];

const OFFSET_DEFAULT_DISTRIBUTION: [i32; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

// ============================================================
// Errors
// ============================================================

/// Why decoding failed: a message, made out of line. Every function of the
/// decoder returns it, and the public API turns it into a `String`. A
/// `Result` carrying it comes back in registers when its value takes a
/// word, and making it writes nothing into the `Result`'s slot. A `String`
/// error is three words, written there by `format!`: that call keeps the
/// slot, `Ok` value included, in memory, where the value's fields are
/// stored one by one and copied in wider loads, which the CPU cannot
/// forward from those stores.
#[derive(Debug, PartialEq)]
struct DecodeError(Box<str>);

impl From<String> for DecodeError {
    #[cold]
    #[inline(never)]
    fn from(msg: String) -> Self {
        DecodeError(msg.into_boxed_str())
    }
}

impl From<&str> for DecodeError {
    #[cold]
    #[inline(never)]
    fn from(msg: &str) -> Self {
        DecodeError(msg.into())
    }
}

impl From<DecodeError> for String {
    fn from(e: DecodeError) -> String {
        e.0.into_string()
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ============================================================
// Public API
// ============================================================

/// Decode Huffman weights from FSE-compressed data (for encoder verification).
pub fn decode_huf_weights_from_fse(source: &[u8], header: u8) -> Result<Vec<u8>, String> {
    let mut ht = HuffmanTable::new();
    let mut full = vec![header];
    full.extend_from_slice(source);
    let (_, nb_weights) = ht.read_weights(&full)?;
    Ok(ht.weights[..nb_weights].to_vec())
}

/// Decode a Huffman tree description and reconstruct canonical codes.
/// Returns codes array matching compress.rs format: [(code, nbits); 256].
pub fn parse_fse_header(source: &[u8], max_log: u8) -> Result<(u8, Vec<i32>, usize), String> {
    let mut table = FSETable::new(255);
    let bytes = table.read_probabilities(source, max_log)?;
    Ok((
        table.accuracy_log,
        table.symbol_probabilities.clone(),
        bytes,
    ))
}

/// Decompress a zstd-compressed byte slice, returning the uncompressed data.
///
/// Supports any number of concatenated zstd frames. Skippable frames are
/// skipped. Every input byte must belong to a frame: an empty input decodes
/// to nothing, and bytes after the last frame are an error. A frame that
/// names a dictionary is an error; see `decompress_with_dict`.
///
/// With the `parallel` feature, frames of two or more compressed blocks,
/// of 32 KiB or more in all and 163 B or more each on average, are decoded
/// on the current rayon pool when it has more than one thread; the output
/// is the same either way.
///
/// Which frames decode follows RFC 8878, not libzstd: every block, raw,
/// RLE or compressed, holds and decodes to at most Block_Maximum_Size
/// bytes (lines 545-569).
///
/// Content of at most 1 KiB comes back in a `Vec` of exactly its length;
/// larger content may come back with the room the decoder reserved past
/// it.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_with_options(data, &DecodeOptions::default())
}

/// `decompress` with dictionary `dict` (ZSTD_decompress_usingDict): every
/// frame starts from its tables and repeat offsets, and its content is the
/// history before every frame (ZSTD_refDictContent; RFC 8878 lines
/// 1835-1837). A match may reach it past Window_Size while the frame has
/// decoded at most Window_Size bytes (lines 1838-1844). A frame that names
/// another nonzero Dictionary_ID is an error (dictionary_wrong); one with
/// no Dictionary_ID is decoded with `dict` too.
///
/// RFC 8878 lines 442-443 make a frame naming an unregistered ID in the
/// reserved ranges an error. That is not enforced: registration cannot be
/// known offline, and the caller supplying the dictionary is the "private
/// arrangement" of lines 439-440, so a matching ID is accepted in any
/// range, as libzstd does.
pub fn decompress_with_dict(data: &[u8], dict: &DecodeDict) -> Result<Vec<u8>, String> {
    decompress_with_dict_options(data, Some(dict), &DecodeOptions::default())
}

/// Decoder paths to force, for testing each of them on any input.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    /// The whole blocks of a frame in the input decode on the current rayon
    /// pool, whatever its size, when at least this many of them are
    /// compressed, of `min_parallel_bytes` or more in all and
    /// `min_parallel_bytes / 200` or more each on average; `usize::MAX`
    /// never do.
    pub min_parallel_blocks: usize,
    /// See `min_parallel_blocks`.
    pub min_parallel_bytes: usize,
    /// Use the SIMD level detected at run time; false forces the portable
    /// code.
    pub simd: bool,
    /// `ZSTD_d_windowLogMax`: the window limit of
    /// `Decompressor::decompress_stream`, as
    /// `Decompressor::set_window_log_max` sets it; `0`, the default, is
    /// that of a new ZSTD_DCtx.
    pub window_log_max: u32,
}

impl DecodeOptions {
    /// The SIMD level `simd` picks.
    fn simd_level(&self) -> Level {
        if self.simd {
            Level::new()
        } else {
            Level::fallback()
        }
    }
}

impl Default for DecodeOptions {
    /// What `decompress` uses.
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        let (min_parallel_blocks, min_parallel_bytes) = (
            if rayon::current_num_threads() > 1 {
                parallel::MIN_BLOCKS
            } else {
                usize::MAX
            },
            parallel::MIN_BYTES,
        );
        #[cfg(not(feature = "parallel"))]
        let (min_parallel_blocks, min_parallel_bytes) = (usize::MAX, usize::MAX);
        DecodeOptions {
            min_parallel_blocks,
            min_parallel_bytes,
            simd: true,
            window_log_max: 0,
        }
    }
}

/// `decompress` with the paths chosen by `opts`.
#[doc(hidden)]
pub fn decompress_with_options(data: &[u8], opts: &DecodeOptions) -> Result<Vec<u8>, String> {
    decompress_with_dict_options(data, None, opts)
}

/// `decompress_with_dict`, or `decompress` without `dict`, with the paths
/// chosen by `opts`.
#[doc(hidden)]
pub fn decompress_with_dict_options(
    data: &[u8],
    dict: Option<&DecodeDict>,
    opts: &DecodeOptions,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    decompress_frames(&mut FrameDecoder::new(opts), data, dict, &mut output)?;
    Ok(take_output(&mut output))
}

/// The most content a Vec-returning decode (`decompress` and the like, and
/// `Decompressor::decompress` and the like) copies out of the buffer it
/// decoded into, into a `Vec` of exactly its length. That buffer has
/// `MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH` bytes of room past the content
/// (`Dst`), so handed over as it is, a few bytes of content would come
/// back in a 131 KB allocation, which glibc serves from the top of the
/// heap, merging it back on free. The copy costs less than that
/// allocation up to about 1 KiB of content (about 280 fewer instructions a
/// call on 120-960 B frames) and more from about 1.2 KiB on (about 350
/// more), where glibc no longer serves it from its thread cache (chunks
/// of up to 1032 bytes); larger content is handed over with the room.
const COPY_OUT_MAX: usize = 1024;

/// The `Vec` a Vec-returning decode returns for the content decoded into
/// `buf`: a copy of exactly its length if that is at most `COPY_OUT_MAX`,
/// leaving `buf` empty with its room for the next decode, or else `buf`
/// itself.
fn take_output(buf: &mut Vec<u8>) -> Vec<u8> {
    if buf.len() > COPY_OUT_MAX {
        return std::mem::take(buf);
    }
    let out = buf.as_slice().to_vec();
    buf.clear();
    out
}

/// The one-shot driver (ZSTD_decompressMultiFrame): decode the frames of
/// `data`, whole, each from `dict` if given, onto the end of `output`,
/// with `dec`, which stands between frames: a fresh one for `decompress`,
/// a `Decompressor`'s own for its `decompress`.
fn decompress_frames(
    dec: &mut FrameDecoder,
    data: &[u8],
    dict: Option<&DecodeDict>,
    output: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    let mut out = VecOut {
        output,
        prefix: Prefix {
            start: 0,
            window: 0,
        },
        dict: dict.map_or(&[], DecodeDict::content),
    };
    let mut pos = 0usize;
    loop {
        let rest = &data[pos..];
        let (skipped, _) = dec.skip(rest.len());
        pos += skipped;
        let rest = &data[pos..];
        let len = dec.unit_len(rest);
        if len > rest.len() {
            return dec.end_of_input(rest);
        }
        if dec.process(&rest[..len], &mut out, dict)? != Event::FrameStarted {
            pos += len;
            continue;
        }
        pos += len;
        let (frame, _) = dec.frame_start();
        reserve_frame(out.output, frame, &data[pos..])?;
        #[cfg(feature = "parallel")]
        {
            let mut read = 0;
            let decoded = dec.decode_blocks_parallel(
                &data[pos..],
                dict,
                &mut out,
                usize::MAX,
                &mut read,
                |_| true,
            );
            pos += read;
            decoded?;
        }
    }
}

// ============================================================
// FSE Table and Decoder
// ============================================================

/// One FSE decoding-table cell, laid out like libzstd's `ZSTD_seqSymbol`.
#[derive(Copy, Clone, Debug, Default)]
struct FSEEntry {
    /// Baseline of the next state; the next `num_bits` stream bits are added.
    next_state: u16,
    num_bits: u8,
    /// Extra bits the sequence code reads from the stream (0 for weight tables).
    extra_bits: u8,
    /// Sequence code base value, or the raw symbol for weight tables.
    base_value: u32,
}

/// Table log of the largest FSE table (libzstd MaxFSELog), that of the
/// literal and match length tables; offsets use 8 and Huffman weights 6.
const FSE_MAX_TABLE_LOG: u8 = LL_MAX_LOG;
const FSE_MAX_TABLE_SIZE: usize = 1 << FSE_MAX_TABLE_LOG;

#[derive(Debug, Clone)]
struct FSETable {
    max_symbol: u8,
    /// Cells for the largest table, like libzstd's fixed DTable arrays,
    /// sized by the first build: a build writes the cells of its own table
    /// and clears nothing.
    cells: Vec<FSEEntry>,
    /// Cells of the built table, `1 << accuracy_log`, or 0 while none is.
    size: usize,
    accuracy_log: u8,
    symbol_probabilities: Vec<i32>,
    /// Per-symbol next-state counter while building (libzstd symbolNext),
    /// 256 entries once sized.
    symbol_next: Vec<u16>,
    /// Symbols laid out in order before spreading, with room for the last
    /// 8-byte write (libzstd spread), sized like `symbol_next`.
    spread: Vec<u8>,
}

impl FSETable {
    fn new(max_symbol: u8) -> FSETable {
        FSETable {
            max_symbol,
            cells: Vec::new(),
            size: 0,
            accuracy_log: 0,
            symbol_probabilities: Vec::new(),
            symbol_next: Vec::new(),
            spread: Vec::new(),
        }
    }

    /// The decoding table: `1 << accuracy_log` cells once built, else none.
    fn decode(&self) -> &[FSEEntry] {
        &self.cells[..self.size]
    }

    fn reset(&mut self) {
        self.symbol_probabilities.clear();
        self.size = 0;
        self.accuracy_log = 0;
    }

    /// One-cell table for an RLE-coded sequence section
    /// (ZSTD_buildSeqTable_rle): accuracy log 0, no state bits.
    fn build_rle(&mut self, symbol: u8, base: &[u32], bits: &[u8]) {
        self.reset();
        self.cells.resize(FSE_MAX_TABLE_SIZE, FSEEntry::default());
        self.cells[0] = FSEEntry {
            next_state: 0,
            num_bits: 0,
            extra_bits: bits[symbol as usize],
            base_value: base[symbol as usize],
        };
        self.size = 1;
    }

    /// Parse an FSE table description and build the decoding table. With
    /// `codes`, each cell carries the sequence code's base value and extra
    /// bits; without, it carries the symbol.
    fn build_decoder(
        &mut self,
        source: &[u8],
        max_log: u8,
        codes: Option<(&[u32], &[u8])>,
    ) -> Result<usize, DecodeError> {
        self.reset();
        let bytes_read = self.read_probabilities(source, max_log)?;
        // RFC 8878 lines 1372-1373: two or more symbols of nonzero
        // probability; one alone is RLE_Mode's (lines 925-927). libzstd's
        // FSE_readNCount does not check this.
        let nonzero = self.symbol_probabilities.iter().filter(|&&c| c != 0);
        if nonzero.count() < 2 {
            return Err("FSE table has fewer than two symbols of nonzero probability".into());
        }
        self.build_decoding_table(codes);
        Ok(bytes_read)
    }

    /// Build the decoding table from counts that did not come from a table
    /// description, so are checked here as `read_ncount_body` checks those.
    fn build_from_probabilities(
        &mut self,
        acc_log: u8,
        probs: &[i32],
        codes: Option<(&[u32], &[u8])>,
    ) -> Result<(), DecodeError> {
        self.reset();
        let cells: i64 = probs.iter().map(|&p| i64::from(p.abs())).sum();
        if !(ACC_LOG_OFFSET..=FSE_MAX_TABLE_LOG).contains(&acc_log)
            || probs.len() > usize::from(self.max_symbol) + 1
            || probs.iter().any(|&p| p < -1)
            || cells != 1 << acc_log
        {
            return Err(format!(
                "Invalid FSE distribution: {} counts over {} cells at accuracy log {}",
                probs.len(),
                cells,
                acc_log
            )
            .into());
        }
        self.symbol_probabilities.extend_from_slice(probs);
        self.accuracy_log = acc_log;
        self.build_decoding_table(codes);
        Ok(())
    }

    /// Port of ZSTD_buildFSETable_body (with `codes`) and
    /// FSE_buildDTable_internal (without): lay low-probability symbols at
    /// the top, spread the rest, then derive each cell's bit count and next
    /// state from `symbol_next` in one pass. The counts tile the table:
    /// `read_ncount_body` or `build_from_probabilities` checked them.
    fn build_decoding_table(&mut self, codes: Option<(&[u32], &[u8])>) {
        let table_log = u32::from(self.accuracy_log);
        assert!(table_log <= u32::from(FSE_MAX_TABLE_LOG));
        let table_size = 1usize << table_log;
        let mask = table_size - 1;
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        // No-ops after the first build.
        self.cells.resize(FSE_MAX_TABLE_SIZE, FSEEntry::default());
        self.symbol_next.resize(256, 0);
        self.spread.resize(FSE_MAX_TABLE_SIZE + 8, 0);
        let dt = self.cells.first_chunk_mut::<FSE_MAX_TABLE_SIZE>().unwrap();
        let symbol_next = self.symbol_next.first_chunk_mut::<256>().unwrap();
        let counts = &self.symbol_probabilities[..];

        // Low-probability symbols occupy the highest cells.
        let mut high_threshold = table_size;
        for ((s, &n), next) in counts.iter().enumerate().zip(&mut *symbol_next) {
            if n == -1 {
                high_threshold -= 1;
                dt[high_threshold].base_value = s as u32;
                *next = 1;
            } else {
                *next = n as u16;
            }
        }

        if high_threshold == table_size {
            // No low-probability symbols: lay the symbols down in order with
            // 8-byte writes, then scatter them across the table, so neither
            // loop has a data-dependent trip count.
            let spread = self
                .spread
                .first_chunk_mut::<{ FSE_MAX_TABLE_SIZE + 8 }>()
                .unwrap();
            let mut pos = 0;
            let mut sv = 0u64;
            for &n in counts {
                let n = n as usize;
                spread[pos..pos + 8].copy_from_slice(&sv.to_le_bytes());
                let mut i = 8;
                while i < n {
                    spread[pos + i..pos + i + 8].copy_from_slice(&sv.to_le_bytes());
                    i += 8;
                }
                pos += n;
                sv = sv.wrapping_add(0x0101_0101_0101_0101);
            }
            let mut position = 0;
            for s in (0..table_size).step_by(2) {
                dt[position].base_value = u32::from(spread[s]);
                dt[(position + step) & mask].base_value = u32::from(spread[s + 1]);
                position = (position + 2 * step) & mask;
            }
        } else {
            let mut position = 0;
            for (s, &n) in counts.iter().enumerate() {
                for _ in 0..n.max(0) {
                    dt[position].base_value = s as u32;
                    position = (position + step) & mask;
                    while position >= high_threshold {
                        position = (position + step) & mask;
                    }
                }
            }
        }

        let cells = &mut dt[..table_size];
        match codes {
            Some((base, bits)) => {
                for cell in cells {
                    let symbol = usize::from(cell.base_value as u8);
                    let (num_bits, next_state) = fse_cell_state(symbol_next, symbol, table_log);
                    *cell = FSEEntry {
                        next_state,
                        num_bits,
                        extra_bits: bits[symbol],
                        base_value: base[symbol],
                    };
                }
            }
            None => {
                for cell in cells {
                    let symbol = usize::from(cell.base_value as u8);
                    (cell.num_bits, cell.next_state) =
                        fse_cell_state(symbol_next, symbol, table_log);
                }
            }
        }
        self.size = table_size;
    }

    /// Read the normalized counts header (FSE_readNCount): four bits of
    /// accuracy log, then one count per symbol, with repeat flags after
    /// each zero count. Returns the header's length in bytes.
    fn read_probabilities(&mut self, source: &[u8], max_log: u8) -> Result<usize, DecodeError> {
        if source.len() < 8 {
            // The body reads 4 bytes at a time up to the header's end.
            let mut buffer = [0u8; 8];
            buffer[..source.len()].copy_from_slice(source);
            let n = self.read_ncount_body(&buffer, max_log)?;
            if n > source.len() {
                return Err("FSE table header extends past its input".into());
            }
            return Ok(n);
        }
        self.read_ncount_body(source, max_log)
    }

    /// FSE_readNCount_body; requires `src.len() >= 8`.
    fn read_ncount_body(&mut self, src: &[u8], max_log: u8) -> Result<usize, DecodeError> {
        debug_assert!(src.len() >= 8);
        let iend = src.len();
        let read32 = |at: usize| u32::from_le_bytes(src[at..at + 4].try_into().unwrap());
        let max_sv1 = self.max_symbol as usize + 1;
        let counts = &mut self.symbol_probabilities;
        counts.clear();
        counts.resize(max_sv1, 0);

        let mut ip = 0usize;
        let mut bit_stream = read32(ip);
        let mut nb_bits = (bit_stream & 0xF) + u32::from(ACC_LOG_OFFSET);
        if nb_bits > u32::from(max_log) {
            return Err(format!("Accuracy log {} exceeds max {}", nb_bits, max_log).into());
        }
        self.accuracy_log = nb_bits as u8;
        bit_stream >>= 4;
        let mut bit_count = 4u32;
        let mut remaining = (1i32 << nb_bits) + 1;
        let mut threshold = 1i32 << nb_bits;
        nb_bits += 1;
        let mut charnum = 0usize;
        let mut previous0 = false;

        // Advance `ip` by the whole bytes consumed, clamping at the last
        // 4-byte window, and reload the 32-bit window.
        let advance = |ip: &mut usize, bit_count: &mut u32| {
            if *ip + 7 <= iend || *ip + (*bit_count >> 3) as usize + 4 <= iend {
                *ip += (*bit_count >> 3) as usize;
                *bit_count &= 7;
            } else {
                *bit_count = bit_count.wrapping_sub(8 * (iend - 4 - *ip) as u32) & 31;
                *ip = iend - 4;
            }
        };

        loop {
            if previous0 {
                // Each 0b11 repeat code adds three zero-count symbols.
                let mut repeats = ((!bit_stream | 0x8000_0000).trailing_zeros() >> 1) as usize;
                while repeats >= 12 {
                    charnum += 3 * 12;
                    if ip + 7 <= iend {
                        ip += 3;
                    } else {
                        // `iend - 7 - ip` is negative here (signed in the C).
                        let back = 8 * (iend as i64 - 7 - ip as i64);
                        bit_count = (i64::from(bit_count) - back) as u32 & 31;
                        ip = iend - 4;
                    }
                    bit_stream = read32(ip) >> bit_count;
                    repeats = ((!bit_stream | 0x8000_0000).trailing_zeros() >> 1) as usize;
                }
                charnum += 3 * repeats;
                bit_stream >>= 2 * repeats;
                bit_count += 2 * repeats as u32;
                charnum += (bit_stream & 3) as usize;
                bit_count += 2;
                if charnum >= max_sv1 {
                    break;
                }
                advance(&mut ip, &mut bit_count);
                bit_stream = read32(ip) >> bit_count;
            }

            let max = (2 * threshold - 1) - remaining;
            let mut count;
            if ((bit_stream & (threshold as u32 - 1)) as i32) < max {
                count = (bit_stream & (threshold as u32 - 1)) as i32;
                bit_count += nb_bits - 1;
            } else {
                count = (bit_stream & (2 * threshold as u32 - 1)) as i32;
                if count >= threshold {
                    count -= max;
                }
                bit_count += nb_bits;
            }
            count -= 1;
            if count >= 0 {
                remaining -= count;
            } else {
                remaining += count;
            }
            counts[charnum] = count;
            charnum += 1;
            previous0 = count == 0;

            if remaining < threshold {
                if remaining <= 1 {
                    break;
                }
                nb_bits = highest_bit_set(remaining as u32);
                threshold = 1 << (nb_bits - 1);
            }
            if charnum >= max_sv1 {
                break;
            }
            advance(&mut ip, &mut bit_count);
            bit_stream = read32(ip) >> bit_count;
        }
        if remaining != 1 {
            return Err(format!(
                "FSE counts leave {} of {} cells unassigned",
                remaining - 1,
                1u32 << self.accuracy_log
            )
            .into());
        }
        if charnum > max_sv1 {
            return Err(format!("Too many symbols: {}", charnum).into());
        }
        if bit_count > 32 {
            return Err("FSE table header extends past its input".into());
        }
        counts.truncate(charnum);
        Ok(ip + ((bit_count + 7) >> 3) as usize)
    }
}

pub(crate) fn highest_bit_set(x: u32) -> u32 {
    assert!(x > 0);
    u32::BITS - x.leading_zeros()
}

/// Bit count and next-state baseline of the next cell of `symbol`, taking
/// its next state from `symbol_next` (the last pass of the FSE table builds).
#[inline(always)]
fn fse_cell_state(symbol_next: &mut [u16; 256], symbol: usize, table_log: u32) -> (u8, u16) {
    let next_state = u32::from(symbol_next[symbol]);
    symbol_next[symbol] += 1;
    let nb_bits = table_log - (u32::BITS - 1 - next_state.leading_zeros());
    (
        nb_bits as u8,
        ((next_state << nb_bits) - (1 << table_log)) as u16,
    )
}

/// Decode the weights of a Huffman tree description from an FSE bitstream
/// with two interleaved states (FSE_decompress_usingDTable_generic): four
/// symbols per reload while the stream lasts, then one at a time until it
/// overflows. Returns the number of weights.
fn fse_decompress_weights(
    table: &FSETable,
    src: &[u8],
    out: &mut [u8; 255],
) -> Result<usize, DecodeError> {
    let dt = table.decode();
    let table_log = u32::from(table.accuracy_log);
    let mut br = BitDStream::new(src)?;
    // FSE_initDState reloads after each initial state.
    let mut state1 = br.read_bits(table_log);
    br.reload();
    let mut state2 = br.read_bits(table_log);
    br.reload();
    if br.reload() == HufStreamStatus::Overflow {
        return Err("Huffman weights stream is too short".into());
    }
    // FSE_decodeSymbol: the state's symbol, then the next state. Every
    // cell's `next_state` plus its `num_bits` bits stays below the table
    // size, so a state is always a valid index.
    let decode = |state: &mut usize, br: &mut BitDStream<'_>| {
        let cell = dt[*state];
        *state = usize::from(cell.next_state) + br.read_bits(u32::from(cell.num_bits));
        cell.base_value as u8
    };
    let too_many = || Err("Too many Huffman weights".into());
    let omax = out.len();
    let mut op = 0;
    while br.reload() == HufStreamStatus::Unfinished && op < omax - 3 {
        out[op] = decode(&mut state1, &mut br);
        out[op + 1] = decode(&mut state2, &mut br);
        out[op + 2] = decode(&mut state1, &mut br);
        out[op + 3] = decode(&mut state2, &mut br);
        op += 4;
    }
    loop {
        if op > omax - 2 {
            return too_many();
        }
        out[op] = decode(&mut state1, &mut br);
        op += 1;
        if br.reload() == HufStreamStatus::Overflow {
            out[op] = decode(&mut state2, &mut br);
            return Ok(op + 1);
        }
        if op > omax - 2 {
            return too_many();
        }
        out[op] = decode(&mut state2, &mut br);
        op += 1;
        if br.reload() == HufStreamStatus::Overflow {
            out[op] = decode(&mut state1, &mut br);
            return Ok(op + 1);
        }
    }
}

// ============================================================
// Huffman Table and Decoder
// ============================================================

/// Single-symbol table cell (libzstd HUF_DEltX1).
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
struct HuffmanEntry {
    symbol: u8,
    num_bits: u8,
}

// SAFETY: two `u8` fields under `repr(C)`: no padding, and every bit
// pattern is a valid cell.
unsafe impl bytemuck::Zeroable for HuffmanEntry {}
unsafe impl bytemuck::Pod for HuffmanEntry {}

/// Double-symbol table cell (libzstd HUF_DEltX2): `sequence` holds one or
/// two symbols little-endian, `length` how many.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
struct HufEntryX2 {
    sequence: u16,
    nb_bits: u8,
    length: u8,
}

// SAFETY: a `u16` and two `u8` fields under `repr(C)`: no padding, and every
// bit pattern is a valid cell.
unsafe impl bytemuck::Zeroable for HufEntryX2 {}
unsafe impl bytemuck::Pod for HufEntryX2 {}

impl HufEntryX2 {
    /// `length`, which is 1 or 2 in every built cell, computed so that the
    /// compiler knows it is: a write position advanced by it then stays in
    /// a window checked once for several cells.
    #[inline(always)]
    fn advance(self) -> usize {
        1 + usize::from(self.length >> 1 & 1)
    }
}

/// Table log of every decoding table (libzstd HUF_DECODER_FAST_TABLELOG):
/// the longest code length, so that codes of any length are scaled up to
/// it and the 4-stream fast loops index with a constant shift.
const HUF_FAST_TABLE_LOG: u32 = HUF_TABLELOG_MAX;

/// The cells of a built decoding table: a lookup of `HUF_FAST_TABLE_LOG`
/// bits is in range by type.
type HufCells<T> = [T; 1 << HUF_FAST_TABLE_LOG];

/// `cells` as a built table; a build sizes it once, and every later build
/// writes every cell.
fn huf_cells<T>(cells: &[T]) -> Result<&HufCells<T>, DecodeError> {
    cells
        .try_into()
        .map_err(|_| "Huffman table is uninitialized".into())
}

/// Relative cost of the single- and double-symbol decoders, indexed by the
/// compression ratio quantile (libzstd algoTime: table build time, then
/// decode time per 256 bytes).
const HUF_ALGO_TIME: [[(u32, u32); 2]; 16] = [
    [(0, 0), (1, 1)],
    [(0, 0), (1, 1)],
    [(150, 216), (381, 119)],
    [(170, 205), (514, 112)],
    [(177, 199), (539, 110)],
    [(197, 194), (644, 107)],
    [(221, 192), (735, 107)],
    [(256, 189), (881, 106)],
    [(359, 188), (1167, 109)],
    [(582, 187), (1570, 114)],
    [(688, 187), (1712, 122)],
    [(825, 186), (1965, 136)],
    [(976, 185), (2131, 150)],
    [(1180, 186), (2070, 175)],
    [(1377, 185), (1731, 202)],
    [(1412, 185), (1695, 202)],
];

/// Pick the double-symbol decoder when its estimated time is lower
/// (HUF_selectDecoder).
fn huf_select_x2(dst_size: usize, src_size: usize) -> bool {
    let q = if src_size >= dst_size {
        15
    } else {
        src_size * 16 / dst_size
    };
    let d256 = (dst_size >> 8) as u32;
    let (t0, d0) = HUF_ALGO_TIME[q][0];
    let (t1, d1) = HUF_ALGO_TIME[q][1];
    let time0 = t0 + d0 * d256;
    let mut time1 = t1 + d1 * d256;
    time1 += time1 >> 5;
    time1 < time0
}

/// One 8-bit counter per Huffman weight 0..=15, kept in two registers:
/// `lo` holds weights 0..8 and `hi` weights 8..16, a byte each.
#[derive(Default)]
struct WeightLanes {
    lo: u64,
    hi: u64,
}

impl WeightLanes {
    /// Add `n << (8 * (w % 8))` to the half that holds weight `w`.
    #[inline(always)]
    fn add(&mut self, w: u8, n: u64) {
        let v = n << (8 * (w & 7));
        if w & 8 == 0 {
            self.lo = self.lo.wrapping_add(v);
        } else {
            self.hi = self.hi.wrapping_add(v);
        }
    }

    /// Add one to the counter of weight `w`.
    #[inline(always)]
    fn bump(&mut self, w: u8) {
        const ONE: [[u64; 2]; 16] = {
            let mut t = [[0; 2]; 16];
            let mut w = 0;
            while w < 16 {
                t[w][w / 8] = 1 << (8 * (w % 8));
                w += 1;
            }
            t
        };
        let [lo, hi] = ONE[usize::from(w & 15)];
        self.lo = self.lo.wrapping_add(lo);
        self.hi = self.hi.wrapping_add(hi);
    }

    /// The counter of weight `w`.
    #[inline(always)]
    fn get(&self, w: u8) -> u8 {
        let half = if w & 8 == 0 { self.lo } else { self.hi };
        (half >> (8 * (w & 7))) as u8
    }
}

struct HuffmanTable {
    /// Single-symbol table, `1 << HUF_FAST_TABLE_LOG` cells once the first
    /// single-symbol build sized it; the built table when `!is_x2`.
    decode: Vec<HuffmanEntry>,
    /// Double-symbol table, sized like `decode` by the first double-symbol
    /// build; the built table when `is_x2`.
    decode_x2: Vec<HufEntryX2>,
    is_x2: bool,
    /// Weight per symbol (libzstd huffWeight): after a build, the first
    /// `nb_symbols`, the last of them implied by the others.
    weights: [u8; 256],
    nb_symbols: usize,
    /// Table log of the Huffman code; 0 while no table is built.
    max_num_bits: u8,
    /// Number of symbols of each weight (libzstd rankStats).
    rank_stats: [u32; HUF_TABLELOG_MAX as usize + 1],
    /// Symbols ordered by weight (libzstd symbols, sortedSymbol).
    sorted: [u8; 256],
    /// The weights' FSE table, over weights 0..=HUF_TABLELOG_MAX at every
    /// accuracy log: a description listing a higher symbol is corrupt (RFC
    /// 8878 lines 1432-1436, 1541-1543), which `read_ncount_body`'s symbol
    /// bound enforces. libzstd's bound instead depends on the log
    /// (HUF_READ_STATS_WORKSPACE_SIZE_U32).
    fse_table: FSETable,
}

impl HuffmanTable {
    fn new() -> HuffmanTable {
        HuffmanTable {
            decode: Vec::new(),
            decode_x2: Vec::new(),
            is_x2: false,
            weights: [0; 256],
            nb_symbols: 0,
            max_num_bits: 0,
            rank_stats: [0; HUF_TABLELOG_MAX as usize + 1],
            sorted: [0; 256],
            fse_table: FSETable::new(HUF_TABLELOG_MAX as u8),
        }
    }

    /// Make this the decoding table `src` is, reusing this one's
    /// allocations; the weights' FSE table is build scratch and not copied.
    fn copy_from(&mut self, src: &HuffmanTable) {
        if src.is_x2 {
            self.decode_x2.clone_from(&src.decode_x2);
        } else {
            self.decode.clone_from(&src.decode);
        }
        self.is_x2 = src.is_x2;
        self.weights = src.weights;
        self.nb_symbols = src.nb_symbols;
        self.max_num_bits = src.max_num_bits;
        self.rank_stats = src.rank_stats;
        self.sorted = src.sorted;
    }

    /// Forget the table. Its cells stay allocated for the next build, which
    /// writes every one of them.
    fn reset(&mut self) {
        self.is_x2 = false;
        self.nb_symbols = 0;
        self.max_num_bits = 0;
        self.fse_table.reset();
    }

    /// Read the tree description at the start of `source` (the whole
    /// compressed literals section) and build the decoding table. Four-stream
    /// sections pick the table kind by libzstd's cost model; single-stream
    /// sections always use the single-symbol table
    /// (ZSTD_decodeLiteralsBlock).
    fn build_decoder(
        &mut self,
        source: &[u8],
        dst_size: usize,
        four_streams: bool,
    ) -> Result<u32, DecodeError> {
        self.max_num_bits = 0;
        let (bytes_used, nb_weights) = self.read_weights(source)?;
        self.weight_stats(nb_weights)?;
        self.fill(four_streams && huf_select_x2(dst_size, source.len()));
        Ok(bytes_used as u32)
    }

    /// Fill the double-symbol table when `x2`, else the single-symbol one.
    fn fill(&mut self, x2: bool) {
        self.is_x2 = x2;
        if x2 {
            self.fill_x2();
        } else {
            self.fill_x1();
        }
    }

    /// Code length of `symbol` (RFC 8878 §4.2.1, rfc8878.txt:1550:
    /// Number_of_Bits = Max_Number_of_Bits + 1 - Weight); `symbol` is one
    /// the built table decodes, so its weight is not 0.
    #[inline(always)]
    fn code_len(&self, symbol: u8) -> u32 {
        u32::from(self.max_num_bits) + 1 - u32::from(self.weights[usize::from(symbol)])
    }

    /// Read the weights of a tree description (HUF_readStats_body before the
    /// statistics): a header byte of 128 or more is followed by
    /// `header - 127` 4-bit weights, a smaller one by that many bytes of
    /// FSE-compressed weights. Returns the length of the description and
    /// the number of weights read.
    fn read_weights(&mut self, source: &[u8]) -> Result<(usize, usize), DecodeError> {
        let Some(&header) = source.first() else {
            return Err("Huffman source is empty".into());
        };
        let header = usize::from(header);
        if header >= 128 {
            let nb_weights = header - 127;
            let size = nb_weights.div_ceil(2);
            let Some(packed) = source.get(1..1 + size) else {
                return Err(
                    format!("Not enough bytes for {} raw Huffman weights", nb_weights).into(),
                );
            };
            for (pair, &b) in self.weights.as_chunks_mut::<2>().0.iter_mut().zip(packed) {
                pair[0] = b >> 4;
                pair[1] = b & 15;
            }
            Ok((1 + size, nb_weights))
        } else {
            let Some(src) = source.get(1..1 + header) else {
                return Err(format!(
                    "Not enough bytes for weights: have {}, need {}",
                    source.len() - 1,
                    header
                )
                .into());
            };
            let ncount = self.fse_table.build_decoder(src, 6, None)?;
            let out = self.weights.first_chunk_mut::<255>().unwrap();
            let nb_weights = fse_decompress_weights(&self.fse_table, &src[ncount..], out)?;
            Ok((1 + header, nb_weights))
        }
    }

    /// The statistics and checks of HUF_readStats_body over the
    /// `nb_weights` weights read: symbols per weight, the table log, the
    /// implied last weight and a full binary tree.
    fn weight_stats(&mut self, nb_weights: usize) -> Result<(), DecodeError> {
        let weights = &self.weights[..nb_weights];
        // Weights are at most 15 (4 raw bits) and fewer than 256: count
        // them in `WeightLanes` rather than in memory, where runs of one
        // weight would wait on store forwarding.
        let mut counts = WeightLanes::default();
        for &w in weights {
            counts.bump(w);
        }
        if (HUF_TABLELOG_MAX as u8 + 1..16).any(|w| counts.get(w) != 0) {
            let w = weights.iter().find(|&&w| u32::from(w) > HUF_TABLELOG_MAX);
            return Err(format!("Weight {} exceeds max {}", w.unwrap(), HUF_TABLELOG_MAX).into());
        }
        let mut rank_stats = [0u32; HUF_TABLELOG_MAX as usize + 1];
        let mut weight_total = 0u32;
        for (w, n) in rank_stats.iter_mut().enumerate() {
            *n = u32::from(counts.get(w as u8));
            weight_total += *n * ((1 << w) >> 1);
        }
        if weight_total == 0 {
            return Err("Missing weights".into());
        }
        let table_log = highest_bit_set(weight_total);
        if table_log > HUF_TABLELOG_MAX {
            return Err(format!("Max bits {} too high", table_log).into());
        }
        // The last weight completes the total to a power of 2.
        let rest = (1 << table_log) - weight_total;
        if !rest.is_power_of_two() {
            return Err(format!("Leftover {} is not a power of 2", rest).into());
        }
        let last_weight = highest_bit_set(rest);
        self.weights[nb_weights] = last_weight as u8;
        rank_stats[last_weight as usize] += 1;
        // A full binary tree has an even number of leaves at its deepest
        // level, and at least two.
        if rank_stats[1] < 2 || rank_stats[1] & 1 != 0 {
            return Err(format!("Huffman tree has {} symbols of weight 1", rank_stats[1]).into());
        }
        self.rank_stats = rank_stats;
        self.nb_symbols = nb_weights + 1;
        self.max_num_bits = table_log as u8;
        Ok(())
    }

    /// Order the symbols by weight, then by value, into `sorted` (libzstd
    /// symbols[], sortedSymbol[]) and return where each weight's symbols
    /// start, followed by the end of the last weight.
    fn sort_symbols(&mut self) -> [usize; HUF_TABLELOG_MAX as usize + 2] {
        let mut rank_start = [0usize; HUF_TABLELOG_MAX as usize + 2];
        // The next slot of each weight, as in `weight_stats`. A slot of 256
        // only follows the last symbol, and its carry reaches lanes of
        // higher weights, which have no symbols.
        let mut next = WeightLanes::default();
        let mut start = 0usize;
        for (w, &n) in self.rank_stats.iter().enumerate() {
            rank_start[w] = start;
            next.add(w as u8, start as u64);
            start += n as usize;
        }
        rank_start[HUF_TABLELOG_MAX as usize + 1] = start;
        for (s, &w) in self.weights[..self.nb_symbols].iter().enumerate() {
            self.sorted[usize::from(next.get(w))] = s as u8;
            next.bump(w);
        }
        rank_start
    }

    /// Fill the single-symbol table at `HUF_FAST_TABLE_LOG` bits
    /// (HUF_readDTableX1_wksp with HUF_rescaleStats): each symbol of `n`
    /// bits owns `1 << (HUF_FAST_TABLE_LOG - n)` consecutive cells, ordered
    /// by code length. Scaling the weights up leaves every code length unchanged,
    /// so only the cell counts differ from a `max_bits` table. The weights
    /// tile the table (`weight_stats`): every cell is written.
    fn fill_x1(&mut self) {
        let max_bits = u32::from(self.max_num_bits);
        let dt_log = HUF_FAST_TABLE_LOG;
        let rescale = dt_log - max_bits;
        let rank_start = self.sort_symbols();
        let rank_stats = &self.rank_stats;

        // Fill the table one weight at a time, so that the run length is a
        // constant of each loop, writing each symbol's cells four to a word
        // (HUF_DEltX1_set4) over the table's bytes.
        self.decode.resize(1 << dt_log, HuffmanEntry::default());
        let cells: &mut [u8] = bytemuck::cast_slice_mut(&mut self.decode[..]);
        let mut u = 0usize;
        for w in 1..=max_bits as usize {
            let count = rank_stats[w] as usize;
            let length = 1usize << (w - 1 + rescale as usize);
            let num_bits = (max_bits + 1 - w as u32) as u8;
            let syms = &self.sorted[rank_start[w]..rank_start[w + 1]];
            let d4 = |s: u8| u64::from(u16::from_le_bytes([s, num_bits])) * 0x0001_0001_0001_0001;
            let run = &mut cells[2 * u..2 * (u + count * length)];
            match length {
                1 => {
                    for (c, &s) in run.as_chunks_mut::<2>().0.iter_mut().zip(syms) {
                        *c = (d4(s) as u16).to_le_bytes();
                    }
                }
                2 => {
                    for (c, &s) in run.as_chunks_mut::<4>().0.iter_mut().zip(syms) {
                        *c = (d4(s) as u32).to_le_bytes();
                    }
                }
                4 => {
                    for (c, &s) in run.as_chunks_mut::<8>().0.iter_mut().zip(syms) {
                        *c = d4(s).to_le_bytes();
                    }
                }
                8 => {
                    for (c, &s) in run.as_chunks_mut::<16>().0.iter_mut().zip(syms) {
                        *c = bytemuck::cast([d4(s).to_le_bytes(); 2]);
                    }
                }
                _ => {
                    for (r, &s) in run.chunks_exact_mut(2 * length).zip(syms) {
                        let d16: [u8; 32] = bytemuck::cast([d4(s).to_le_bytes(); 4]);
                        for c in r.as_chunks_mut::<32>().0 {
                            *c = d16;
                        }
                    }
                }
            }
            u += count * length;
        }
    }

    /// Fill the double-symbol table (HUF_readDTableX2_wksp after
    /// HUF_readStats): sort symbols by weight, compute where each weight's
    /// run starts for every number of already-consumed bits, then tile the
    /// table so that a cell holds two symbols whenever both fit in
    /// `HUF_FAST_TABLE_LOG` bits. Every cell is written, as in `fill_x1`.
    fn fill_x2(&mut self) {
        let table_log = u32::from(self.max_num_bits);
        let target_log = HUF_FAST_TABLE_LOG;
        let nb_bits_baseline = table_log + 1;

        // Highest weight in use; weight 1 is always present.
        let mut max_w = table_log as usize;
        while self.rank_stats[max_w] == 0 {
            max_w -= 1;
        }

        // Weight-0 symbols come first, and are never read.
        let rank_start = self.sort_symbols();

        // rank_val[consumed][w]: first cell of weight w once `consumed` bits
        // of the lookup have been used by a first symbol.
        let rescale = target_log - table_log; // shift of (w - 1 + rescale)
        let mut rank_val = [[0u32; HUF_TABLELOG_MAX as usize + 1]; HUF_TABLELOG_MAX as usize];
        let mut next_val = 0u32;
        for w in 1..=max_w {
            rank_val[0][w] = next_val;
            next_val += self.rank_stats[w] << (w as u32 - 1 + rescale);
        }
        let min_bits = nb_bits_baseline - max_w as u32;
        for consumed in min_bits..=(target_log - min_bits) {
            for w in 1..=max_w {
                rank_val[consumed as usize][w] = rank_val[0][w] >> consumed;
            }
        }

        self.decode_x2
            .resize(1 << target_log, HufEntryX2::default());
        let dt = &mut self.decode_x2[..];
        let sorted = &self.sorted[..];
        let scale_log = nb_bits_baseline as i32 - target_log as i32;

        for w in 1..=max_w {
            let begin = rank_start[w];
            let end = rank_start[w + 1];
            let nb_bits = nb_bits_baseline - w as u32;

            if target_log - nb_bits >= min_bits {
                // Room for a second symbol.
                let length = 1usize << (target_log - nb_bits);
                let min_weight = (nb_bits as i32 + scale_log).max(1) as usize;
                let mut start = rank_val[0][w] as usize;
                for &base_seq in &sorted[begin..end] {
                    huf_fill_x2_level2(
                        &mut dt[start..start + length],
                        target_log,
                        nb_bits,
                        &rank_val[nb_bits as usize],
                        min_weight,
                        max_w + 1,
                        sorted,
                        &rank_start,
                        nb_bits_baseline,
                        base_seq,
                    );
                    start += length;
                }
            } else {
                huf_fill_x2_for_weight(
                    &mut dt[rank_val[0][w] as usize..],
                    &sorted[begin..end],
                    nb_bits,
                    target_log,
                    0,
                    1,
                );
            }
        }
    }
}

#[inline(always)]
fn huf_build_x2(symbol: u8, nb_bits: u32, base_seq: u8, level: u8) -> HufEntryX2 {
    let sequence = if level == 1 {
        u16::from(symbol)
    } else {
        u16::from(base_seq) | (u16::from(symbol) << 8)
    };
    HufEntryX2 {
        sequence,
        nb_bits: nb_bits as u8,
        length: level,
    }
}

/// Write every symbol of one weight into consecutive runs of
/// `1 << (target_log - nb_bits)` cells (HUF_fillDTableX2ForWeight), with
/// one loop per run length.
fn huf_fill_x2_for_weight(
    dt: &mut [HufEntryX2],
    symbols: &[u8],
    nb_bits: u32,
    target_log: u32,
    base_seq: u8,
    level: u8,
) {
    let length = 1usize << (target_log - nb_bits);
    let cell =
        |symbol: u8| -> [u8; 4] { bytemuck::cast(huf_build_x2(symbol, nb_bits, base_seq, level)) };
    // HUF_buildDEltX2U64: the cell twice in a word.
    let cell2 = |symbol: u8| -> [u8; 8] { bytemuck::cast([cell(symbol); 2]) };
    let cells: &mut [u8] = bytemuck::cast_slice_mut(&mut dt[..symbols.len() * length]);
    match length {
        1 => {
            for (c, &symbol) in cells.as_chunks_mut::<4>().0.iter_mut().zip(symbols) {
                *c = cell(symbol);
            }
        }
        2 => {
            for (c, &symbol) in cells.as_chunks_mut::<8>().0.iter_mut().zip(symbols) {
                *c = cell2(symbol);
            }
        }
        4 => {
            for (c, &symbol) in cells.as_chunks_mut::<16>().0.iter_mut().zip(symbols) {
                *c = bytemuck::cast([cell2(symbol); 2]);
            }
        }
        _ => {
            for (run, &symbol) in cells.chunks_exact_mut(4 * length).zip(symbols) {
                let c8: [u8; 32] = bytemuck::cast([cell2(symbol); 4]);
                for c in run.as_chunks_mut::<32>().0 {
                    *c = c8;
                }
            }
        }
    }
}

/// Fill the cells of first symbol `base_seq` (HUF_fillDTableX2Level2):
/// second symbols too long to fit get a single-symbol cell, the rest are
/// laid out by weight.
#[allow(clippy::too_many_arguments)]
fn huf_fill_x2_level2(
    dt: &mut [HufEntryX2],
    target_log: u32,
    consumed_bits: u32,
    rank_val: &[u32],
    min_weight: usize,
    max_weight1: usize,
    sorted: &[u8],
    rank_start: &[usize],
    nb_bits_baseline: u32,
    base_seq: u8,
) {
    if min_weight > 1 {
        // Whole runs of 2, 4 or 8 cells as in libzstd; the cells written
        // past `skip` are overwritten by the second symbols below.
        let cell: [u8; 4] = bytemuck::cast(huf_build_x2(base_seq, consumed_bits, 0, 1));
        let cell2: [u8; 8] = bytemuck::cast([cell; 2]);
        let skip = rank_val[min_weight] as usize;
        let cells: &mut [u8] = bytemuck::cast_slice_mut(&mut dt[..]);
        match cells.len() / 4 {
            2 => cells.copy_from_slice(&cell2),
            4 => cells.copy_from_slice(&bytemuck::cast::<_, [u8; 16]>([cell2; 2])),
            _ => {
                let c8: [u8; 32] = bytemuck::cast([cell2; 4]);
                for c in cells[..4 * skip.next_multiple_of(8)]
                    .as_chunks_mut::<32>()
                    .0
                {
                    *c = c8;
                }
            }
        }
    }
    for w in min_weight..max_weight1 {
        let begin = rank_start[w];
        let end = rank_start[w + 1];
        let nb_bits = nb_bits_baseline - w as u32;
        huf_fill_x2_for_weight(
            &mut dt[rank_val[w] as usize..],
            &sorted[begin..end],
            nb_bits + consumed_bits,
            target_log,
            base_seq,
            2,
        );
    }
}

// ------------------------------------------------------------
// Backward bit stream and Huffman symbol decoding
//
// Port of libzstd's BIT_DStream_t (common/bitstream.h), shared by the
// literals and sequences decoders, and of the X1 single-symbol decoders
// in decompress/huf_decompress.c
// (HUF_decodeSymbolX1, HUF_decodeStreamX1,
// HUF_decompress1X1_usingDTable_internal_body,
// HUF_decompress4X1_usingDTable_internal_body).
// ------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum HufStreamStatus {
    /// At least 57 bits are loaded; keep decoding without checks.
    Unfinished,
    /// The container holds the final bytes; fewer than 64 bits are unread.
    EndOfBuffer,
    /// Every input bit has been consumed.
    Completed,
    /// More bits were consumed than the stream holds (or the fast reload
    /// reached the last 8 bytes).
    Overflow,
}

/// Backward bit reader over one Huffman or sequences stream.
///
/// `container` holds the 8 bytes starting at `ptr`; bits are consumed from
/// its high end (the stream is read from its last byte backwards). Below
/// 64 bits consumed, a lookup reaching past the container's low end reads
/// zeros there. From 64 on, lookups shift by `bits_consumed & 63`, as
/// BIT_lookBitsFast masks its shift, and so re-read the container from its
/// top. Every Huffman symbol consumes at least one bit, so a lookup at or
/// past 64 leaves the stream consumed past 64 bits, which `is_finished`
/// rejects.
struct BitDStream<'s> {
    src: &'s [u8],
    ptr: usize,
    container: u64,
    bits_consumed: u32,
}

impl<'s> BitDStream<'s> {
    fn new(src: &'s [u8]) -> Result<Self, DecodeError> {
        let Some(&last) = src.last() else {
            return Err("Huffman stream is empty".into());
        };
        if last == 0 {
            return Err("Huffman stream has no end mark".into());
        }
        // Zero padding above the end mark, plus the mark itself.
        let padding = last.leading_zeros() + 1;
        if src.len() >= 8 {
            let ptr = src.len() - 8;
            Ok(BitDStream {
                src,
                ptr,
                container: read_le64(src, ptr),
                bits_consumed: padding,
            })
        } else {
            Ok(BitDStream {
                src,
                ptr: 0,
                container: read_le_short(src),
                bits_consumed: padding + (8 - src.len() as u32) * 8,
            })
        }
    }

    /// Next `n` (1..=56) unread bits, without consuming them.
    #[inline(always)]
    fn look_bits(&self, n: u32) -> usize {
        ((self.container << (self.bits_consumed & 63)) >> (64 - n)) as usize
    }

    /// Next `n` (0..=56) unread bits, without consuming them (BIT_lookBits).
    #[inline(always)]
    fn look_bits_any(&self, n: u32) -> usize {
        (((self.container << (self.bits_consumed & 63)) >> 1) >> ((63 - n) & 63)) as usize
    }

    #[inline(always)]
    fn skip_bits(&mut self, n: u32) {
        self.bits_consumed += n;
    }

    /// Read `n` (0..=56) bits (BIT_readBits).
    #[inline(always)]
    fn read_bits(&mut self, n: u32) -> usize {
        let value = self.look_bits_any(n);
        self.skip_bits(n);
        value
    }

    /// Read `n` (1..=56) bits (BIT_readBitsFast).
    #[inline(always)]
    fn read_bits_fast(&mut self, n: u32) -> usize {
        let value = self.look_bits(n);
        self.skip_bits(n);
        value
    }

    /// Requires `ptr >= 8` and `bits_consumed <= 64`.
    #[inline(always)]
    fn reload_internal(&mut self) -> HufStreamStatus {
        debug_assert!(self.ptr >= 8 && self.bits_consumed <= 64);
        self.ptr -= (self.bits_consumed >> 3) as usize;
        self.bits_consumed &= 7;
        // SAFETY: `ptr + 8 <= src.len()` is a struct invariant: `new` sets
        // `ptr = src.len() - 8` when the stream has 8 bytes or more, `ptr`
        // only ever decreases, and a shorter stream keeps `ptr == 0`, which
        // no caller of this function accepts.
        self.container = unsafe { read_le64_unchecked(self.src, self.ptr) };
        HufStreamStatus::Unfinished
    }

    /// Reload for the interleaved main loop. Requires `bits_consumed <= 64`
    /// and stops (Overflow) once fewer than 8 unread bytes remain.
    #[inline(always)]
    fn reload_fast(&mut self) -> HufStreamStatus {
        if self.ptr < 8 {
            return HufStreamStatus::Overflow;
        }
        self.reload_internal()
    }

    #[inline(always)]
    fn reload(&mut self) -> HufStreamStatus {
        if self.bits_consumed > 64 {
            return HufStreamStatus::Overflow;
        }
        if self.ptr >= 8 {
            return self.reload_internal();
        }
        if self.ptr == 0 {
            return if self.bits_consumed < 64 {
                HufStreamStatus::EndOfBuffer
            } else {
                HufStreamStatus::Completed
            };
        }
        // 0 < ptr < 8: only `ptr` bytes are left before the stream start.
        let mut nb_bytes = (self.bits_consumed >> 3) as usize;
        let mut result = HufStreamStatus::Unfinished;
        if self.ptr < nb_bytes {
            nb_bytes = self.ptr;
            result = HufStreamStatus::EndOfBuffer;
        }
        self.ptr -= nb_bytes;
        self.bits_consumed -= (nb_bytes * 8) as u32;
        self.container = read_le64(self.src, self.ptr);
        result
    }

    /// True when exactly every bit of the stream was consumed.
    fn is_finished(&self) -> bool {
        self.ptr == 0 && self.bits_consumed == 64
    }
}

/// `src`, of 1 to 8 bytes, as a little-endian number, read in registers as
/// BIT_initDStream reads it. Copied into a zeroed `[u8; 8]` instead, its
/// bytes reach the 8-byte load through narrower stores, which the CPU
/// cannot forward.
#[inline(always)]
fn read_le_short(src: &[u8]) -> u64 {
    let n = src.len();
    if n >= 4 {
        // Overlapping reads: the bytes both cover land in the same place.
        let lo = u32::from_le_bytes(src[..4].try_into().unwrap());
        let hi = u32::from_le_bytes(src[n - 4..].try_into().unwrap());
        u64::from(lo) | u64::from(hi) << (8 * (n - 4))
    } else {
        let mid = n / 2;
        u64::from(src[0])
            | u64::from(src[mid]) << (8 * mid)
            | u64::from(src[n - 1]) << (8 * (n - 1))
    }
}

#[inline(always)]
fn read_le64(src: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(src[at..at + 8].try_into().unwrap())
}

/// `MEM_readLE64(src + at)` without a bounds check.
///
/// # Safety
/// `at + 8 <= src.len()`.
#[inline(always)]
unsafe fn read_le64_unchecked(src: &[u8], at: usize) -> u64 {
    debug_assert!(at + 8 <= src.len());
    u64::from_le_bytes(*(src.as_ptr().add(at) as *const [u8; 8]))
}

/// `dt[i]` without a bounds check.
///
/// # Safety
/// `i < dt.len()`.
#[inline(always)]
unsafe fn table_entry<T: Copy>(dt: &[T], i: usize) -> T {
    debug_assert!(i < dt.len());
    *dt.get_unchecked(i)
}

#[inline(always)]
fn huf_decode_symbol_x1(br: &mut BitDStream<'_>, dt: &HufCells<HuffmanEntry>) -> u8 {
    let entry = dt[br.look_bits(HUF_FAST_TABLE_LOG)];
    br.skip_bits(u32::from(entry.num_bits));
    entry.symbol
}

/// Decode `out.len()` symbols from one stream (HUF_decodeStreamX1).
#[inline(always)]
fn huf_decode_stream_x1(out: &mut [u8], br: &mut BitDStream<'_>, dt: &HufCells<HuffmanEntry>) {
    let end = out.len();
    let mut p = 0;
    if end > 3 {
        // Up to 4 symbols per reload: a reload that reports Unfinished
        // guarantees at least 57 bits, and a symbol takes at most 11.
        while br.reload() == HufStreamStatus::Unfinished && p < end - 3 {
            let a = huf_decode_symbol_x1(br, dt);
            let b = huf_decode_symbol_x1(br, dt);
            let c = huf_decode_symbol_x1(br, dt);
            let d = huf_decode_symbol_x1(br, dt);
            out[p..p + 4].copy_from_slice(&[a, b, c, d]);
            p += 4;
        }
    } else {
        br.reload();
    }
    // Either at most 3 symbols remain with >= 57 bits loaded, or the
    // container already holds the last bytes of the stream.
    while p < end {
        out[p] = huf_decode_symbol_x1(br, dt);
        p += 1;
    }
}

/// Single-stream literals (HUF_decompress1X1_usingDTable_internal_body).
#[inline(never)]
fn huf_decompress_1x1(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), DecodeError> {
    let dt = huf_cells(&table.decode)?;
    let mut br = BitDStream::new(src)?;
    huf_decode_stream_x1(out, &mut br, dt);
    if !br.is_finished() {
        return Err("Huffman stream not fully consumed".into());
    }
    Ok(())
}

/// Stream and segment layout of a 4-stream section: the jump table checks
/// shared by HUF_decompress4X1/4X2_usingDTable_internal_body.
struct HufStreams {
    /// First byte of each stream within the section.
    istart: [usize; 4],
    /// One past the last byte of each stream within the section.
    iend: [usize; 4],
    /// Segment size; stream `s` writes `[s * segment, segment_end(s))`.
    segment: usize,
}

impl HufStreams {
    fn split(src: &[u8], dst_size: usize) -> Result<HufStreams, DecodeError> {
        if src.len() < 10 {
            return Err(format!("Huffman 4-stream input too short: {} bytes", src.len()).into());
        }
        let len1 = usize::from(u16::from_le_bytes([src[0], src[1]]));
        let len2 = usize::from(u16::from_le_bytes([src[2], src[3]]));
        let len3 = usize::from(u16::from_le_bytes([src[4], src[5]]));
        if 6 + len1 + len2 + len3 > src.len() {
            return Err("Huffman jump table exceeds input".into());
        }
        let istart = [6, 6 + len1, 6 + len1 + len2, 6 + len1 + len2 + len3];
        let iend = [istart[1], istart[2], istart[3], src.len()];
        // RFC 8878 §3.1.1.3.1.6 (rfc8878.txt:796-799): the first three
        // streams decode (Regenerated_Size+3)/4 bytes each and the last one
        // the rest, so a size is valid exactly when that rest is not
        // negative: 4 splits 1,1,1,1, 5 would leave -1. libzstd's
        // MIN_LITERALS_FOR_4_STREAMS (6) also rejects 0, 3 and 4 (R2-6).
        let segment = dst_size.div_ceil(4);
        if 3 * segment > dst_size {
            return Err("Huffman 4-stream segments exceed output".into());
        }
        Ok(HufStreams {
            istart,
            iend,
            segment,
        })
    }

    fn stream<'s>(&self, src: &'s [u8], s: usize) -> &'s [u8] {
        &src[self.istart[s]..self.iend[s]]
    }

    fn segment_end(&self, s: usize, dst_size: usize) -> usize {
        ((s + 1) * self.segment).min(dst_size)
    }
}

/// Per-stream state of the 4-stream fast loops (HUF_DecompressFastArgs):
/// `ip[s]` indexes the 8 section bytes held in `bits[s]`, `op[s]` the next
/// output byte. Each container keeps its unread bits at the top and a
/// sentinel 1 bit right below them, so `trailing_zeros` is the number of
/// bits already consumed from the loaded bytes.
#[derive(Clone, Copy)]
struct HufFastArgs {
    ip: [usize; 4],
    op: [usize; 4],
    bits: [u64; 4],
}

/// Set up the fast loops (HUF_DecompressFastArgs_init). `None` sends the
/// section to the plain loops: each stream must hold 8 bytes to fill a
/// container, and the fourth segment must not be empty.
fn huf_fast_args_init(
    streams: &HufStreams,
    src: &[u8],
    dst_size: usize,
) -> Result<Option<HufFastArgs>, DecodeError> {
    let segment = streams.segment;
    if 3 * segment >= dst_size {
        return Ok(None);
    }
    let mut ip = [0usize; 4];
    let mut bits = [0u64; 4];
    for s in 0..4 {
        if streams.iend[s] - streams.istart[s] < 8 {
            return Ok(None);
        }
        let last = src[streams.iend[s] - 1];
        if last == 0 {
            return Err("Huffman stream has no end mark".into());
        }
        // HUF_initFastDStream: the padding above the end mark and the mark
        // itself count as consumed.
        ip[s] = streams.iend[s] - 8;
        bits[s] = (read_le64(src, ip[s]) | 1) << (last.leading_zeros() + 1);
    }
    Ok(Some(HufFastArgs {
        ip,
        op: [0, segment, 2 * segment, 3 * segment],
        bits,
    }))
}

/// Continue stream `s` with the plain decoder (HUF_initRemainingDStream).
/// The container's consumed-bit count, plus 8 per byte the container sits
/// below the stream start, is the position within the stream's own bytes.
fn huf_remaining_dstream<'s>(
    args: &HufFastArgs,
    streams: &HufStreams,
    s: usize,
    src: &'s [u8],
) -> Result<BitDStream<'s>, DecodeError> {
    let start = streams.istart[s];
    let stream = streams.stream(src, s);
    let ip = args.ip[s];
    // A fully consumed stream leaves the container at most 8 bytes below
    // its start; anything lower is corruption.
    if ip + 8 < start {
        return Err("Huffman stream overran its start".into());
    }
    let (ptr, below) = if ip >= start {
        (ip - start, 0)
    } else {
        (0, start - ip)
    };
    let bits_consumed = args.bits[s].trailing_zeros() + below as u32 * 8;
    if bits_consumed > 64 {
        return Err("Huffman stream overran its start".into());
    }
    Ok(BitDStream {
        src: stream,
        ptr,
        container: read_le64(stream, ptr),
        bits_consumed,
    })
}

/// Five symbols per stream per iteration, reloading by `trailing_zeros`
/// (HUF_decompress4X1_usingDTable_internal_fast_c_loop). Each iteration
/// writes 5 bytes per stream and consumes at most 55 bits, under 7 bytes,
/// per stream, and every stream's input lies at or above stream 0's; so
/// `iters` iterations stay inside `src` and `out` without further checks.
/// A stream that crosses the previous one (corruption) ends the loop.
///
/// # Safety
/// `args` was produced by `huf_fast_args_init` on `src` and an output of
/// `out.len()` bytes; `dt.len() == 1 << HUF_FAST_TABLE_LOG`.
unsafe fn huf_4x1_fast_loop(
    args: &mut HufFastArgs,
    out: &mut [u8],
    src: &[u8],
    dt: &[HuffmanEntry],
) {
    let HufFastArgs {
        mut ip,
        mut op,
        mut bits,
    } = *args;
    let oend = out.len();
    let o = out.as_mut_ptr();
    loop {
        let oiters = (oend - op[3]) / 5;
        let iiters = ip[0] / 7;
        let olimit = op[3] + oiters.min(iiters) * 5;
        if op[3] == olimit {
            break;
        }
        if ip[1] < ip[0] || ip[2] < ip[1] || ip[3] < ip[2] {
            break;
        }
        // Table cells hold at most HUF_FAST_TABLE_LOG bits, so the
        // sentinel stays inside the container between reloads.
        macro_rules! decode {
            ($s:expr, $k:expr) => {{
                let entry = table_entry(dt, (bits[$s] >> (64 - HUF_FAST_TABLE_LOG)) as usize);
                bits[$s] <<= u32::from(entry.num_bits);
                *o.add(op[$s] + $k) = entry.symbol;
            }};
        }
        macro_rules! reload {
            ($s:expr) => {{
                let ctz = bits[$s].trailing_zeros();
                op[$s] += 5;
                ip[$s] -= (ctz >> 3) as usize;
                bits[$s] = (read_le64_unchecked(src, ip[$s]) | 1) << (ctz & 7);
            }};
        }
        loop {
            decode!(0, 0);
            decode!(1, 0);
            decode!(2, 0);
            decode!(3, 0);
            decode!(0, 1);
            decode!(1, 1);
            decode!(2, 1);
            decode!(3, 1);
            decode!(0, 2);
            decode!(1, 2);
            decode!(2, 2);
            decode!(3, 2);
            decode!(0, 3);
            decode!(1, 3);
            decode!(2, 3);
            decode!(3, 3);
            decode!(0, 4);
            decode!(1, 4);
            decode!(2, 4);
            decode!(3, 4);
            reload!(0);
            reload!(1);
            reload!(2);
            reload!(3);
            if op[3] >= olimit {
                break;
            }
        }
    }
    *args = HufFastArgs { ip, op, bits };
}

/// Five cells per stream per iteration, up to 10 bytes each
/// (HUF_decompress4X2_usingDTable_internal_fast_c_loop). Streams advance at
/// their own pace, so `iters` is the minimum over the four output bounds;
/// the fourth stream's cells are decoded around the reloads to relieve
/// register pressure. Every cell write is 2 bytes wide and lands below
/// the stream's bound because `op[s] + 10 <= oend[s]` at each iteration.
///
/// # Safety
/// `args` was produced by `huf_fast_args_init` on `src` and an output of
/// `out.len()` bytes; `dt.len() == 1 << HUF_FAST_TABLE_LOG`.
unsafe fn huf_4x2_fast_loop(args: &mut HufFastArgs, out: &mut [u8], src: &[u8], dt: &[HufEntryX2]) {
    let HufFastArgs {
        mut ip,
        mut op,
        mut bits,
    } = *args;
    let oend = [op[1], op[2], op[3], out.len()];
    let o = out.as_mut_ptr();
    loop {
        let mut iters = ip[0] / 7;
        for s in 0..4 {
            iters = iters.min((oend[s] - op[s]) / 10);
        }
        let olimit = op[3] + iters * 5;
        if op[3] == olimit {
            break;
        }
        if ip[1] < ip[0] || ip[2] < ip[1] || ip[3] < ip[2] {
            break;
        }
        macro_rules! decode {
            ($s:expr) => {{
                let entry = table_entry(dt, (bits[$s] >> (64 - HUF_FAST_TABLE_LOG)) as usize);
                ptr::copy_nonoverlapping(entry.sequence.to_le_bytes().as_ptr(), o.add(op[$s]), 2);
                bits[$s] <<= u32::from(entry.nb_bits);
                op[$s] += usize::from(entry.length);
            }};
        }
        macro_rules! reload {
            ($s:expr) => {{
                decode!(3);
                let ctz = bits[$s].trailing_zeros();
                ip[$s] -= (ctz >> 3) as usize;
                bits[$s] = (read_le64_unchecked(src, ip[$s]) | 1) << (ctz & 7);
            }};
        }
        loop {
            decode!(0);
            decode!(1);
            decode!(2);
            decode!(0);
            decode!(1);
            decode!(2);
            decode!(0);
            decode!(1);
            decode!(2);
            decode!(0);
            decode!(1);
            decode!(2);
            decode!(0);
            decode!(1);
            decode!(2);
            decode!(3);
            reload!(0);
            reload!(1);
            reload!(2);
            reload!(3);
            if op[3] >= olimit {
                break;
            }
        }
    }
    *args = HufFastArgs { ip, op, bits };
}

/// Four interleaved literal streams
/// (HUF_decompress4X1_usingDTable_internal_body and _fast).
///
/// The output is split into four segments of `(len + 3) / 4` bytes (the
/// last one holds the remainder); stream `i` produces segment `i`. The fast
/// loop takes sections with 8 bytes or more per stream; the plain loop
/// advances all four streams in lockstep, 4 symbols each per reload. Both
/// finish each stream with `huf_decode_stream_x1`.
#[inline(never)]
fn huf_decompress_4x1(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), DecodeError> {
    let dt = huf_cells(&table.decode)?;
    let dst_size = out.len();
    let streams = HufStreams::split(src, dst_size)?;

    if let Some(mut args) = huf_fast_args_init(&streams, src, dst_size)? {
        // SAFETY: `args` comes from `huf_fast_args_init` on this `src` and
        // `out`, and `dt` has exactly `1 << HUF_FAST_TABLE_LOG` cells.
        unsafe { huf_4x1_fast_loop(&mut args, out, src, dt) };
        for s in 0..4 {
            let end = streams.segment_end(s, dst_size);
            if args.op[s] > end {
                return Err("Huffman stream overran its segment".into());
            }
            let mut br = huf_remaining_dstream(&args, &streams, s, src)?;
            huf_decode_stream_x1(&mut out[args.op[s]..end], &mut br, dt);
            if !br.is_finished() {
                return Err("Huffman stream not fully consumed".into());
            }
        }
        return Ok(());
    }

    let segment = streams.segment;
    let (o1, rest) = out.split_at_mut(segment);
    let (o2, rest) = rest.split_at_mut(segment);
    let (o3, o4) = rest.split_at_mut(segment);

    let mut b1 = BitDStream::new(streams.stream(src, 0))?;
    let mut b2 = BitDStream::new(streams.stream(src, 1))?;
    let mut b3 = BitDStream::new(streams.stream(src, 2))?;
    let mut b4 = BitDStream::new(streams.stream(src, 3))?;

    // Common write position within each segment; `o4` is the shortest
    // segment, so bounding `p` by it bounds all four.
    let mut p = 0;
    if o4.len() >= 8 {
        let mut end_signal = true;
        while end_signal && p + 4 <= o4.len() {
            let mut w1 = [0u8; 4];
            let mut w2 = [0u8; 4];
            let mut w3 = [0u8; 4];
            let mut w4 = [0u8; 4];
            for i in 0..4 {
                w1[i] = huf_decode_symbol_x1(&mut b1, dt);
                w2[i] = huf_decode_symbol_x1(&mut b2, dt);
                w3[i] = huf_decode_symbol_x1(&mut b3, dt);
                w4[i] = huf_decode_symbol_x1(&mut b4, dt);
            }
            o1[p..p + 4].copy_from_slice(&w1);
            o2[p..p + 4].copy_from_slice(&w2);
            o3[p..p + 4].copy_from_slice(&w3);
            o4[p..p + 4].copy_from_slice(&w4);
            p += 4;
            end_signal &= b1.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b2.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b3.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b4.reload_fast() == HufStreamStatus::Unfinished;
        }
    }

    huf_decode_stream_x1(&mut o1[p..], &mut b1, dt);
    huf_decode_stream_x1(&mut o2[p..], &mut b2, dt);
    huf_decode_stream_x1(&mut o3[p..], &mut b3, dt);
    huf_decode_stream_x1(&mut o4[p..], &mut b4, dt);

    if !(b1.is_finished() && b2.is_finished() && b3.is_finished() && b4.is_finished()) {
        return Err("Huffman stream not fully consumed".into());
    }
    Ok(())
}

/// Decode `N / 2` cells of one or two symbols each into `w`, the room
/// they can fill; returns the number of bytes they filled.
#[inline(always)]
fn huf_decode_cells_x2<const N: usize>(
    w: &mut [u8; N],
    br: &mut BitDStream<'_>,
    dt: &HufCells<HufEntryX2>,
) -> usize {
    let mut o = 0;
    for _ in 0..N / 2 {
        // HUF_decodeSymbolX2: always writes two bytes.
        let entry = dt[br.look_bits(HUF_FAST_TABLE_LOG)];
        w[o..o + 2].copy_from_slice(&entry.sequence.to_le_bytes());
        br.skip_bits(u32::from(entry.nb_bits));
        o += entry.advance();
    }
    o
}

/// Decode the final symbol of a stream: only the first symbol of the cell
/// is wanted, and it consumes its own code length, as X1 would. RFC 8878
/// §4.2.2 (rfc8878.txt:1779-1782) requires the bitstream to be consumed
/// exactly, so neither a second symbol's bits nor a lookup past the
/// stream's start may pass the end check. libzstd's
/// HUF_decodeLastSymbolX2 skips the whole cell, clamped to the container,
/// and skips nothing once the container is spent (R1-6, R2-3).
#[inline(always)]
fn huf_decode_last_symbol_x2(
    out: &mut [u8],
    op: usize,
    br: &mut BitDStream<'_>,
    dt: &HufCells<HufEntryX2>,
    table: &HuffmanTable,
) {
    let symbol = dt[br.look_bits(HUF_FAST_TABLE_LOG)].sequence as u8;
    out[op] = symbol;
    br.skip_bits(table.code_len(symbol));
}

/// Decode symbols into all of `out` (HUF_decodeStreamX2).
#[inline(always)]
fn huf_decode_stream_x2(
    out: &mut [u8],
    br: &mut BitDStream<'_>,
    dt: &HufCells<HufEntryX2>,
    table: &HuffmanTable,
) {
    let end = out.len();
    let mut op = 0;
    if end >= 8 {
        // Up to 10 symbols per reload: 5 cells of at most 11 bits each.
        while br.reload() == HufStreamStatus::Unfinished {
            let Some(w) = out[op..].first_chunk_mut() else {
                break;
            };
            op += huf_decode_cells_x2::<10>(w, br, dt);
        }
    } else {
        br.reload();
    }

    if end - op >= 2 {
        while br.reload() == HufStreamStatus::Unfinished {
            let Some(w) = out[op..].first_chunk_mut() else {
                break;
            };
            op += huf_decode_cells_x2::<2>(w, br, dt);
        }
        // The container holds the last bytes of the stream: no reloads.
        while let Some(w) = out[op..].first_chunk_mut() {
            op += huf_decode_cells_x2::<2>(w, br, dt);
        }
    }

    if op < end {
        huf_decode_last_symbol_x2(out, op, br, dt, table);
    }
}

/// Single-stream literals with the double-symbol table
/// (HUF_decompress1X2_usingDTable_internal_body).
#[inline(never)]
fn huf_decompress_1x2(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), DecodeError> {
    let dt = huf_cells(&table.decode_x2)?;
    let mut br = BitDStream::new(src)?;
    huf_decode_stream_x2(out, &mut br, dt, table);
    if !br.is_finished() {
        return Err("Huffman stream not fully consumed".into());
    }
    Ok(())
}

/// Four interleaved literal streams with the double-symbol table
/// (HUF_decompress4X2_usingDTable_internal_body and _fast). Streams write
/// at their own pace, so each is checked against its segment end after the
/// shared loop; the plain loop's trip count is bounded by the last stream.
/// The fast loop takes the same sections as in `huf_decompress_4x1`.
#[inline(never)]
fn huf_decompress_4x2(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), DecodeError> {
    let dt = huf_cells(&table.decode_x2)?;
    let oend = out.len();
    let streams = HufStreams::split(src, oend)?;

    if let Some(mut args) = huf_fast_args_init(&streams, src, oend)? {
        // SAFETY: `args` comes from `huf_fast_args_init` on this `src` and
        // `out`, and `dt` has exactly `1 << HUF_FAST_TABLE_LOG` cells.
        unsafe { huf_4x2_fast_loop(&mut args, out, src, dt) };
        for s in 0..4 {
            let end = streams.segment_end(s, oend);
            if args.op[s] > end {
                return Err("Huffman stream overran its segment".into());
            }
            let mut br = huf_remaining_dstream(&args, &streams, s, src)?;
            huf_decode_stream_x2(&mut out[args.op[s]..end], &mut br, dt, table);
            if !br.is_finished() {
                return Err("Huffman stream not fully consumed".into());
            }
        }
        return Ok(());
    }

    let segment = streams.segment;
    let op_start2 = segment;
    let op_start3 = 2 * segment;
    let op_start4 = 3 * segment;

    let mut b1 = BitDStream::new(streams.stream(src, 0))?;
    let mut b2 = BitDStream::new(streams.stream(src, 1))?;
    let mut b3 = BitDStream::new(streams.stream(src, 2))?;
    let mut b4 = BitDStream::new(streams.stream(src, 3))?;

    let mut op1 = 0;
    let mut op2 = op_start2;
    let mut op3 = op_start3;
    let mut op4 = op_start4;

    // 4 cells per stream per iteration, at most 8 bytes; every stream has
    // those 8 bytes inside `out` because none can outrun the last one by
    // more than a factor of two.
    if oend - op4 >= 8 {
        let mut end_signal = true;
        while end_signal && op4 + 8 <= oend {
            op1 += huf_decode_cells_x2::<8>(out[op1..].first_chunk_mut().unwrap(), &mut b1, dt);
            op2 += huf_decode_cells_x2::<8>(out[op2..].first_chunk_mut().unwrap(), &mut b2, dt);
            end_signal &= b1.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b2.reload_fast() == HufStreamStatus::Unfinished;
            op3 += huf_decode_cells_x2::<8>(out[op3..].first_chunk_mut().unwrap(), &mut b3, dt);
            op4 += huf_decode_cells_x2::<8>(out[op4..].first_chunk_mut().unwrap(), &mut b4, dt);
            end_signal &= b3.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b4.reload_fast() == HufStreamStatus::Unfinished;
        }
    }

    if op1 > op_start2 || op2 > op_start3 || op3 > op_start4 {
        return Err("Huffman stream overran its segment".into());
    }

    huf_decode_stream_x2(&mut out[op1..op_start2], &mut b1, dt, table);
    huf_decode_stream_x2(&mut out[op2..op_start3], &mut b2, dt, table);
    huf_decode_stream_x2(&mut out[op3..op_start4], &mut b3, dt, table);
    huf_decode_stream_x2(&mut out[op4..], &mut b4, dt, table);

    if !(b1.is_finished() && b2.is_finished() && b3.is_finished() && b4.is_finished()) {
        return Err("Huffman stream not fully consumed".into());
    }
    Ok(())
}

// ============================================================
// Block types and headers
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockType {
    Raw,
    RLE,
    Compressed,
    Reserved,
}

#[derive(Clone, Copy, PartialEq)]
struct BlockHeader {
    last_block: bool,
    block_type: BlockType,
    decompressed_size: u32,
    content_size: u32,
}

// ============================================================
// Literals Section
// ============================================================

#[derive(Clone, Copy)]
enum LiteralsSectionType {
    Raw,
    RLE,
    Compressed,
    Treeless,
}

#[derive(Clone, Copy)]
struct LiteralsSection {
    regenerated_size: u32,
    /// The section's bytes after its header: a Raw section's literals, an
    /// RLE section's byte, or the tree description and streams.
    content_size: u32,
    ls_type: LiteralsSectionType,
    /// Huffman-coded literals in four streams rather than one.
    four_streams: bool,
}

impl LiteralsSection {
    /// Parse the Literals_Section_Header that starts `raw`: the section and
    /// the header's length.
    #[inline(always)]
    fn parse(raw: &[u8]) -> Result<(LiteralsSection, usize), DecodeError> {
        let short = |need: usize| {
            format!(
                "Not enough bytes for literals header: have {}, need {}",
                raw.len(),
                need
            )
        };
        let Some(&first) = raw.first() else {
            return Err(short(1).into());
        };
        let ls_type = match first & 3 {
            0 => LiteralsSectionType::Raw,
            1 => LiteralsSectionType::RLE,
            2 => LiteralsSectionType::Compressed,
            _ => LiteralsSectionType::Treeless,
        };
        let size_format = (first >> 2) & 3;
        let len = match (ls_type, size_format) {
            (LiteralsSectionType::Raw | LiteralsSectionType::RLE, 0 | 2) => 1,
            (LiteralsSectionType::Raw | LiteralsSectionType::RLE, 1) => 2,
            (LiteralsSectionType::Raw | LiteralsSectionType::RLE, _) => 3,
            (_, 0 | 1) => 3,
            (_, 2) => 4,
            (_, _) => 5,
        };
        let h = raw.get(..len).ok_or_else(|| short(len))?;
        let b = |i: usize| u32::from(h[i]);
        let (regenerated_size, content_size, four_streams) = match ls_type {
            LiteralsSectionType::Raw | LiteralsSectionType::RLE => {
                let size = match len {
                    1 => b(0) >> 3,
                    2 => (b(0) >> 4) + (b(1) << 4),
                    _ => (b(0) >> 4) + (b(1) << 4) + (b(2) << 12),
                };
                let content = match ls_type {
                    LiteralsSectionType::RLE => 1,
                    _ => size,
                };
                (size, content, false)
            }
            LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
                let (size, compressed) = match len {
                    3 => (
                        (b(0) >> 4) + ((b(1) & 0x3f) << 4),
                        (b(1) >> 6) + (b(2) << 2),
                    ),
                    4 => (
                        (b(0) >> 4) + (b(1) << 4) + ((b(2) & 0x3) << 12),
                        (b(2) >> 2) + (b(3) << 6),
                    ),
                    _ => (
                        (b(0) >> 4) + (b(1) << 4) + ((b(2) & 0x3F) << 12),
                        (b(2) >> 6) + (b(3) << 2) + (b(4) << 10),
                    ),
                };
                (size, compressed, size_format != 0)
            }
        };
        let section = LiteralsSection {
            regenerated_size,
            content_size,
            ls_type,
            four_streams,
        };
        Ok((section, len))
    }
}

// ============================================================
// Sequences Section
// ============================================================

#[derive(Copy, Clone)]
struct CompressionModes(u8);

#[derive(Clone, Copy)]
enum ModeType {
    Predefined,
    RLE,
    FSECompressed,
    Repeat,
}

impl CompressionModes {
    /// The Symbol_Compression_Modes byte, whose Reserved bits 1-0 must be
    /// zero (ZSTD_decodeSeqHeaders' corruption_detected).
    fn new(byte: u8) -> Result<Self, DecodeError> {
        if byte & 3 != 0 {
            return Err(format!("Symbol compression modes {byte:#04x}: reserved bits set").into());
        }
        Ok(Self(byte))
    }

    fn decode_mode(m: u8) -> ModeType {
        match m {
            0 => ModeType::Predefined,
            1 => ModeType::RLE,
            2 => ModeType::FSECompressed,
            3 => ModeType::Repeat,
            _ => panic!("Invalid mode value"),
        }
    }
    fn ll_mode(self) -> ModeType {
        Self::decode_mode(self.0 >> 6)
    }
    fn of_mode(self) -> ModeType {
        Self::decode_mode((self.0 >> 4) & 0x3)
    }
    fn ml_mode(self) -> ModeType {
        Self::decode_mode((self.0 >> 2) & 0x3)
    }
}

#[derive(Clone, Copy)]
struct SequencesHeader {
    num_sequences: u32,
    /// The Symbol_Compression_Modes; all Predefined, unread, without
    /// sequences, where the header has no such byte.
    modes: CompressionModes,
}

impl SequencesHeader {
    /// Parse the Sequences_Section_Header that starts `source`: the header
    /// and its length.
    #[inline(always)]
    fn parse(source: &[u8]) -> Result<(SequencesHeader, usize), DecodeError> {
        let short = |need: usize| {
            format!(
                "Not enough bytes for sequences header: have {}, need {}",
                source.len(),
                need
            )
        };
        let (num_sequences, len) = match *source {
            [] => return Err("Sequences header source is empty".into()),
            [0, ..] => (0, 1),
            [n @ 1..=127, ..] => (u32::from(n), 1),
            [n @ 128..=254, low, ..] => (((u32::from(n) - 128) << 8) + u32::from(low), 2),
            [128..=254] => return Err(short(2).into()),
            [255, low, high, ..] => (u32::from(low) + (u32::from(high) << 8) + 0x7F00, 3),
            [255, ..] => return Err(short(4).into()),
        };
        let modes = match num_sequences {
            // No Symbol_Compression_Modes byte follows a zero count.
            0 => CompressionModes(0),
            _ => CompressionModes::new(*source.get(len).ok_or_else(|| short(len + 1))?)?,
        };
        let header = SequencesHeader {
            num_sequences,
            modes,
        };
        Ok((header, len + usize::from(num_sequences != 0)))
    }
}

// ============================================================
// Scratch space
// ============================================================

struct HuffmanScratch {
    table: HuffmanTable,
}

struct FSEScratch {
    offsets: FSETable,
    literal_lengths: FSETable,
    match_lengths: FSETable,
    /// Where the table blocks decode each code with is, in `SEQ_TABLES`
    /// order (libzstd's LLTptr, OFTptr, MLTptr).
    source: [SeqTableSource; 3],
}

/// Where the table blocks decode a sequence code with is.
#[derive(Clone, Copy)]
enum SeqTableSource {
    /// The scratch's own table: the last FSE_Compressed or RLE
    /// description's, or none.
    Own,
    /// `PREDEFINED_TABLES`: the last description was Predefined_Mode.
    Predefined,
    /// The table of the dictionary the frame started from: no block of
    /// the frame has described one yet (`DecoderScratch::load_dict`).
    Dict,
}

/// The LL, OF and ML tables of the predefined distributions, in
/// `SEQ_TABLES` order, built on first use and shared by every decoder, as
/// libzstd's static LL_defaultDTable, OF_defaultDTable and ML_defaultDTable.
static PREDEFINED_TABLES: LazyLock<[FSETable; 3]> = LazyLock::new(|| {
    std::array::from_fn(|t| {
        let kind = &SEQ_TABLES[t];
        let mut table = FSETable::new(kind.max_code);
        table
            .build_from_probabilities(
                kind.default_log,
                kind.default_distribution,
                Some((kind.base, kind.bits)),
            )
            .expect("predefined distributions tile their tables");
        table
    })
});

struct DecoderScratch {
    huf: HuffmanScratch,
    fse: FSEScratch,
    offset_hist: [u32; 3],
    /// Literals of the current block plus `WILDCOPY_OVERLENGTH` zero bytes.
    literals_buffer: Vec<u8>,
    /// The Huffman table in use is that of the dictionary the frame
    /// started from, not `huf`: true from `load_dict` until a block of the
    /// frame builds one. The sequence tables' counterpart is
    /// `SeqTableSource::Dict`. The dictionary's tables are not kept here:
    /// every call that decodes the frame's blocks is given them, as libzstd
    /// points its DCtx at the DDict's rather than copying them.
    huf_from_dict: bool,
}

impl DecoderScratch {
    fn new() -> DecoderScratch {
        DecoderScratch {
            huf: HuffmanScratch {
                table: HuffmanTable::new(),
            },
            fse: FSEScratch::new(),
            offset_hist: [1, 4, 8],
            literals_buffer: Vec::new(),
            huf_from_dict: false,
        }
    }

    fn reset(&mut self) {
        self.offset_hist = [1, 4, 8];
        self.literals_buffer.clear();
        self.huf_from_dict = false;
        self.fse.reset();
        self.huf.table.reset();
    }
}

// ============================================================
// Frame header
// ============================================================

struct FrameDescriptor(u8);

impl FrameDescriptor {
    fn frame_content_size_flag(&self) -> u8 {
        self.0 >> 6
    }

    fn single_segment_flag(&self) -> bool {
        ((self.0 >> 5) & 0x1) == 1
    }

    fn content_checksum_flag(&self) -> bool {
        ((self.0 >> 2) & 0x1) == 1
    }

    fn dict_id_flag(&self) -> u8 {
        self.0 & 0x3
    }

    fn frame_content_size_bytes(&self) -> Result<u8, DecodeError> {
        match self.frame_content_size_flag() {
            0 => {
                if self.single_segment_flag() {
                    Ok(1)
                } else {
                    Ok(0)
                }
            }
            1 => Ok(2),
            2 => Ok(4),
            3 => Ok(8),
            other => Err(format!("Invalid frame content size flag: {}", other).into()),
        }
    }

    fn dictionary_id_bytes(&self) -> Result<u8, DecodeError> {
        match self.dict_id_flag() {
            0 => Ok(0),
            1 => Ok(1),
            2 => Ok(2),
            3 => Ok(4),
            other => Err(format!("Invalid dict id flag: {}", other).into()),
        }
    }
}

struct FrameHeader {
    descriptor: FrameDescriptor,
    window_descriptor: u8,
    /// Frame_Content_Size, or None when the header omits it.
    frame_content_size: Option<u64>,
    /// Dictionary_ID, 0 when the header omits it.
    dict_id: u32,
}

impl FrameHeader {
    fn window_size(&self) -> Result<u64, DecodeError> {
        if self.descriptor.single_segment_flag() {
            Ok(self.frame_content_size.unwrap_or(0))
        } else {
            let exp = self.window_descriptor >> 3;
            let mantissa = self.window_descriptor & 0x7;

            let window_log = 10 + u32::from(exp);
            // frameParameter_windowTooLarge of ZSTD_getFrameHeader
            if window_log > ZSTD_WINDOWLOG_MAX {
                return Err(format!("Window log {} too large", window_log).into());
            }
            let window_base = 1u64 << window_log;
            let window_add = (window_base / 8) * u64::from(mantissa);

            let window_size = window_base + window_add;

            if window_size < MIN_WINDOW_SIZE {
                Err(format!("Window size {} too small", window_size).into())
            } else {
                Ok(window_size)
            }
        }
    }

    fn frame_content_size(&self) -> Option<u64> {
        self.frame_content_size
    }
}

// ============================================================
// Error wrapper for skip frames
// ============================================================

struct FrameDecoderError {
    msg: DecodeError,
    /// A skippable frame's Frame_Size: the length of its User_Data.
    skip_size: Option<u32>,
}

impl FrameDecoderError {
    fn new(msg: DecodeError) -> Self {
        Self {
            msg,
            skip_size: None,
        }
    }

    fn skip(size: u32) -> Self {
        Self {
            msg: format!("Skippable frame with Frame_Size {}", size).into(),
            skip_size: Some(size),
        }
    }

    fn skip_frame_size(&self) -> Option<u32> {
        self.skip_size
    }
}

impl std::fmt::Display for FrameDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

// ============================================================
// Frame header parsing
// ============================================================

/// Magic number plus Frame_Size of a skippable frame.
const SKIPPABLE_FRAME_HEADER_LEN: usize = 8;

/// ZSTD_startingInputLength: magic number plus Frame_Header_Descriptor, the
/// fewest bytes ZSTD_decompressMultiFrame takes for another frame.
const FRAME_HEADER_PREFIX_LEN: usize = 5;

#[inline(always)]
fn parse_frame_header(src: &[u8]) -> Result<(FrameHeader, usize), FrameDecoderError> {
    let magic_num = src
        .get(..4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| FrameDecoderError::new("Error reading magic number: truncated".into()))?;
    let mut pos = 4;

    // Skippable frames
    if (0x184D2A50..=0x184D2A5F).contains(&magic_num) {
        let skip_size = src
            .get(4..8)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| {
                FrameDecoderError::new("Error reading skip frame size: truncated".into())
            })?;
        return Err(FrameDecoderError::skip(skip_size));
    }

    if magic_num != ZSTD_MAGIC {
        return Err(FrameDecoderError::new(
            format!("Bad magic number: 0x{:X}", magic_num).into(),
        ));
    }

    let desc = FrameDescriptor(*src.get(pos).ok_or_else(|| {
        FrameDecoderError::new("Error reading frame descriptor: truncated".into())
    })?);
    pos += 1;
    // ZSTD_getFrameHeader_advanced: bit 3 is reserved and must be zero
    // (frameParameter_unsupported); bit 4, unused, is ignored.
    if desc.0 & 0x08 != 0 {
        return Err(FrameDecoderError::new(
            format!("Frame header descriptor {:#04x}: reserved bit set", desc.0).into(),
        ));
    }

    let mut frame_header = FrameHeader {
        descriptor: FrameDescriptor(desc.0),
        frame_content_size: None,
        window_descriptor: 0,
        dict_id: 0,
    };

    if !desc.single_segment_flag() {
        frame_header.window_descriptor = *src.get(pos).ok_or_else(|| {
            FrameDecoderError::new("Error reading window descriptor: truncated".into())
        })?;
        pos += 1;
    }

    let dict_id_len = desc.dictionary_id_bytes().map_err(FrameDecoderError::new)? as usize;
    let dict_id = src
        .get(pos..pos + dict_id_len)
        .ok_or_else(|| FrameDecoderError::new("Error reading dictionary id: truncated".into()))?;
    pos += dict_id_len;
    frame_header.dict_id = dict_id
        .iter()
        .rev()
        .fold(0u32, |id, &b| id << 8 | u32::from(b));

    let fcs_len = desc
        .frame_content_size_bytes()
        .map_err(FrameDecoderError::new)? as usize;
    if fcs_len != 0 {
        let fcs_bytes = src.get(pos..pos + fcs_len).ok_or_else(|| {
            FrameDecoderError::new("Error reading frame content size: truncated".into())
        })?;
        pos += fcs_len;
        let mut fcs = 0u64;
        for (i, &b) in fcs_bytes.iter().enumerate() {
            fcs += u64::from(b) << (8 * i);
        }
        if fcs_len == 2 {
            fcs += 256;
        }
        frame_header.frame_content_size = Some(fcs);
    }

    Ok((frame_header, pos))
}

// ============================================================
// Block header parsing
// ============================================================

/// Parse the block header at the start of `src`, in a frame whose
/// Block_Maximum_Size is `block_size_max`.
#[inline(always)]
fn parse_block_header(src: &[u8], block_size_max: usize) -> Result<BlockHeader, DecodeError> {
    let buf: [u8; 3] = src
        .get(..3)
        .ok_or_else(|| DecodeError::from("Error reading block header: truncated"))?
        .try_into()
        .unwrap();

    let last_block = buf[0] & 0x1 == 1;
    let block_type_raw = (buf[0] >> 1) & 0x3;
    let block_type = match block_type_raw {
        0 => BlockType::Raw,
        1 => BlockType::RLE,
        2 => BlockType::Compressed,
        3 => BlockType::Reserved,
        _ => unreachable!(),
    };

    if block_type == BlockType::Reserved {
        return Err("Found reserved block type".into());
    }

    let block_size = u32::from(buf[0] >> 3) | (u32::from(buf[1]) << 5) | (u32::from(buf[2]) << 13);
    // RFC 8878 lines 545-569: Block_Size, a raw or compressed block's
    // content size and an RLE block's decoded size alike, is at most
    // Block_Maximum_Size.
    if block_size as usize > block_size_max {
        return Err(format!(
            "Block size {} exceeds Block_Maximum_Size {}",
            block_size, block_size_max
        )
        .into());
    }

    let decompressed_size = match block_type {
        BlockType::Raw | BlockType::RLE => block_size,
        BlockType::Compressed | BlockType::Reserved => 0,
    };
    let content_size = match block_type {
        BlockType::Raw | BlockType::Compressed => block_size,
        BlockType::RLE => 1,
        BlockType::Reserved => 0,
    };

    Ok(BlockHeader {
        last_block,
        block_type,
        decompressed_size,
        content_size,
    })
}

// ============================================================
// Literals section decoder
// ============================================================

fn decode_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    repeat: Option<&HuffmanTable>,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, DecodeError> {
    match section.ls_type {
        LiteralsSectionType::Raw => {
            target.extend(&source[0..section.regenerated_size as usize]);
            Ok(section.regenerated_size)
        }
        LiteralsSectionType::RLE => {
            target.resize(target.len() + section.regenerated_size as usize, source[0]);
            Ok(1)
        }
        LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
            decompress_literals(section, scratch, repeat, source, target)
        }
    }
}

/// Compressed literals build `scratch`'s table; Treeless ones use `repeat`
/// when given, else `scratch`'s.
fn decompress_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    repeat: Option<&HuffmanTable>,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, DecodeError> {
    let four_streams = section.four_streams;
    let regenerated_size = section.regenerated_size as usize;

    let source = &source[0..section.content_size as usize];
    let mut bytes_read = 0usize;

    let table = match section.ls_type {
        LiteralsSectionType::Compressed => {
            bytes_read += scratch
                .table
                .build_decoder(source, regenerated_size, four_streams)?
                as usize;
            &scratch.table
        }
        _ => repeat.unwrap_or(&scratch.table),
    };
    if table.max_num_bits == 0 {
        return Err("Uninitialized Huffman table for treeless literals".into());
    }

    let source = &source[bytes_read..];
    let start = target.len();
    target.resize(start + regenerated_size, 0);
    huf_decompress(&mut target[start..], source, four_streams, table)?;
    bytes_read += source.len();

    Ok(bytes_read as u32)
}

/// Decode the Huffman streams of a literals section with the built table.
fn huf_decompress(
    out: &mut [u8],
    source: &[u8],
    four_streams: bool,
    t: &HuffmanTable,
) -> Result<(), DecodeError> {
    match (four_streams, t.is_x2) {
        (true, true) => huf_decompress_4x2(out, source, t),
        (true, false) => huf_decompress_4x1(out, source, t),
        (false, true) => huf_decompress_1x2(out, source, t),
        (false, false) => huf_decompress_1x1(out, source, t),
    }
}

// ============================================================
// Sequence section: FSE tables, then fused decode + execute
//
// Port of ZSTD_decodeSequence / ZSTD_execSequence /
// ZSTD_decompressSequences_body (decompress/zstd_decompress_block.c).
// ============================================================

// Base value and extra-bit count per sequence code
// (libzstd LL_base / LL_bits / ML_base / ML_bits / OF_base / OF_bits).
const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];

const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
    2051, 4099, 8195, 16387, 32771, 65539,
];

const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

const OF_BASE: [u32; 32] = [
    0, 1, 1, 5, 13, 29, 61, 125, 253, 509, 1021, 2045, 4093, 8189, 16381, 32765, 65533, 131069,
    262141, 524285, 1048573, 2097149, 4194301, 8388605, 16777213, 33554429, 67108861, 134217725,
    268435453, 536870909, 1073741821, 2147483645,
];

const OF_BITS: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// Bytes of slack kept after the literals and after the block's output
/// limit so that copies may overshoot by a whole vector
/// (libzstd WILDCOPY_OVERLENGTH).
const WILDCOPY_OVERLENGTH: usize = 32;

/// Copies with offsets at or above this never overlap a 16-byte chunk
/// (libzstd WILDCOPY_VECLEN).
const WILDCOPY_VECLEN: usize = 16;

/// Number of repeat offsets (libzstd ZSTD_REP_NUM).
const ZSTD_REP_NUM: usize = 3;

/// Parameters of one of the three sequence code tables.
struct SeqTableKind {
    max_log: u8,
    max_code: u8,
    default_log: u8,
    default_distribution: &'static [i32],
    base: &'static [u32],
    bits: &'static [u8],
    name: &'static str,
}

/// LL, OF, ML: the order of their descriptions in a sequences section.
const SEQ_TABLES: [SeqTableKind; 3] = [
    SeqTableKind {
        max_log: LL_MAX_LOG,
        max_code: MAX_LITERAL_LENGTH_CODE,
        default_log: LL_DEFAULT_ACC_LOG,
        default_distribution: &LITERALS_LENGTH_DEFAULT_DISTRIBUTION,
        base: &LL_BASE,
        bits: &LL_BITS,
        name: "LL",
    },
    SeqTableKind {
        max_log: OF_MAX_LOG,
        max_code: MAX_OFFSET_CODE,
        default_log: OF_DEFAULT_ACC_LOG,
        default_distribution: &OFFSET_DEFAULT_DISTRIBUTION,
        base: &OF_BASE,
        bits: &OF_BITS,
        name: "OF",
    },
    SeqTableKind {
        max_log: ML_MAX_LOG,
        max_code: MAX_MATCH_LENGTH_CODE,
        default_log: ML_DEFAULT_ACC_LOG,
        default_distribution: &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        base: &ML_BASE,
        bits: &ML_BITS,
        name: "ML",
    },
];

impl CompressionModes {
    /// Modes of the LL, OF and ML tables, in `SEQ_TABLES` order.
    fn all(self) -> [ModeType; 3] {
        [self.ll_mode(), self.of_mode(), self.ml_mode()]
    }
}

impl FSEScratch {
    fn new() -> FSEScratch {
        FSEScratch {
            offsets: FSETable::new(MAX_OFFSET_CODE),
            literal_lengths: FSETable::new(MAX_LITERAL_LENGTH_CODE),
            match_lengths: FSETable::new(MAX_MATCH_LENGTH_CODE),
            source: [SeqTableSource::Own; 3],
        }
    }

    /// No tables, for a new frame.
    fn reset(&mut self) {
        self.literal_lengths.reset();
        self.match_lengths.reset();
        self.offsets.reset();
        self.source = [SeqTableSource::Own; 3];
    }

    /// The table blocks decode `SEQ_TABLES[t]` codes with, `dict` being
    /// the tables of the dictionary the frame started from; none yet while
    /// its `decode()` is empty.
    #[inline]
    fn table<'s>(&'s self, t: usize, dict: Option<&'s FSEScratch>) -> &'s FSETable {
        match (self.source[t], dict) {
            (SeqTableSource::Own, _) => self.own(t),
            (SeqTableSource::Predefined, _) => &PREDEFINED_TABLES[t],
            (SeqTableSource::Dict, Some(d)) => d.own(t),
            (SeqTableSource::Dict, None) => {
                unreachable!("only a frame started from a dictionary uses its tables")
            }
        }
    }

    /// The LL, OF, ML tables (`SEQ_TABLES` order) blocks decode with.
    #[inline]
    fn tables<'s>(&'s self, dict: Option<&'s FSEScratch>) -> [&'s FSETable; 3] {
        std::array::from_fn(|t| self.table(t, dict))
    }

    /// The scratch's own table for `SEQ_TABLES[t]`, which descriptions
    /// build.
    fn own(&self, t: usize) -> &FSETable {
        match t {
            0 => &self.literal_lengths,
            1 => &self.offsets,
            _ => &self.match_lengths,
        }
    }

    /// Whether Repeat_Mode may use the `SEQ_TABLES[t]` table: one is in
    /// use, a dictionary's, the predefined one or one a block described.
    fn repeatable(&self, t: usize) -> bool {
        !matches!(self.source[t], SeqTableSource::Own) || !self.own(t).decode().is_empty()
    }

    /// `own`, mutable.
    fn own_mut(&mut self, t: usize) -> &mut FSETable {
        match t {
            0 => &mut self.literal_lengths,
            1 => &mut self.offsets,
            _ => &mut self.match_lengths,
        }
    }
}

/// Build (or reuse) the three FSE tables for this block's sequences and
/// return the number of header bytes consumed (ZSTD_decodeSeqHeaders).
fn build_sequence_tables(
    section: &SequencesHeader,
    source: &[u8],
    scratch: &mut FSEScratch,
) -> Result<usize, DecodeError> {
    let mut bytes_read = 0;
    for (t, mode) in section.modes.all().into_iter().enumerate() {
        bytes_read += build_sequence_table(mode, &source[bytes_read..], scratch, t)?;
    }
    Ok(bytes_read)
}

/// Share of short offsets, out of 256, above which a block's sequences run
/// with `Avx2ShortOffsets` or `FallbackShortOffsets`. Below it the offset
/// tests of the match copy are predictable and cheaper than the straight-line
/// copies that replace them.
const SHORT_OFFSET_SHARE_MIN: usize = 32;

/// Cells of an offsets table with code 2..=4 (new offsets 1..=28), scaled
/// to a 256-cell table, the largest. ZSTD_getOffsetInfo counts its long
/// offsets by scanning the cells; the normalized counts the table was
/// built from give the same number, or the one cell of an RLE table.
fn short_offset_share(table: &FSETable) -> usize {
    let short = if table.accuracy_log == 0 {
        usize::from(
            table
                .decode()
                .first()
                .is_some_and(|e| (2..=4).contains(&e.extra_bits)),
        )
    } else {
        // A count of -1 is one cell.
        table
            .symbol_probabilities
            .iter()
            .take(5)
            .skip(2)
            .map(|&p| p.unsigned_abs() as usize)
            .sum()
    };
    short << (usize::from(OF_MAX_LOG) - usize::from(table.accuracy_log))
}

/// Make the table `mode` gives the one `scratch` decodes `SEQ_TABLES[t]`
/// codes with, and return the length of its description in `source`
/// (ZSTD_buildSeqTable). Predefined_Mode builds nothing: it selects
/// `PREDEFINED_TABLES`.
fn build_sequence_table(
    mode: ModeType,
    source: &[u8],
    scratch: &mut FSEScratch,
    t: usize,
) -> Result<usize, DecodeError> {
    let kind = &SEQ_TABLES[t];
    let codes = Some((kind.base, kind.bits));
    match mode {
        ModeType::FSECompressed => {
            scratch.source[t] = SeqTableSource::Own;
            scratch
                .own_mut(t)
                .build_decoder(source, kind.max_log, codes)
        }
        ModeType::RLE => {
            let Some(&code) = source.first() else {
                return Err(format!("Missing byte for RLE {} table", kind.name).into());
            };
            if code > kind.max_code {
                return Err(format!("RLE {} code {} exceeds max", kind.name, code).into());
            }
            scratch.source[t] = SeqTableSource::Own;
            scratch.own_mut(t).build_rle(code, kind.base, kind.bits);
            Ok(1)
        }
        ModeType::Predefined => {
            scratch.source[t] = SeqTableSource::Predefined;
            Ok(0)
        }
        ModeType::Repeat => {
            if !scratch.repeatable(t) {
                return Err(format!("Repeat mode without a previous {} table", kind.name).into());
            }
            Ok(0)
        }
    }
}

/// Why a sequence could not be executed; turned into a message only after
/// the hot loop has exited.
#[derive(Clone, Copy)]
enum SeqError {
    NotEnoughLiterals,
    BlockTooLarge,
    OffsetTooFar,
    OffsetBeforeDict,
    OffsetPastWindow,
}

#[cold]
#[inline(never)]
fn seq_error_message(e: SeqError) -> DecodeError {
    match e {
        SeqError::NotEnoughLiterals => "Sequence needs more literals than the block has".into(),
        SeqError::BlockTooLarge => "Block content exceeds block size limit".into(),
        SeqError::OffsetTooFar => "Match offset reaches before the frame start".into(),
        SeqError::OffsetBeforeDict => "Match offset reaches before the dictionary start".into(),
        SeqError::OffsetPastWindow => "Match offset exceeds Window_Size".into(),
    }
}

/// The output a frame's matches may copy from: back to the frame's first
/// byte, at output position `start`, and at most `window` bytes, its
/// Window_Size (RFC 8878 lines 590-593). An offset of exactly Window_Size
/// is accepted (line 592).
#[derive(Clone, Copy)]
struct Prefix {
    start: usize,
    window: usize,
}

impl Prefix {
    /// The destination of the frame's next block in `output`, the frame
    /// being `output[self.start..]`: `room` bytes are reserved past its
    /// end. All of the frame's history is in it (`ExtHistory::NONE`).
    fn dst(self, output: &mut Vec<u8>, room: usize) -> Dst {
        output.reserve(room);
        Dst {
            // SAFETY: `start` is at most `output.len()`.
            base: unsafe { output.as_mut_ptr().add(self.start) },
            op: output.len() - self.start,
            window: self.window,
        }
    }
}

/// Where a block decodes to: the current segment of the frame's output
/// starts at `base` (libzstd's `prefixStart`) and holds the `op` bytes
/// before the block. A match reaches back at most `window` bytes,
/// Window_Size (`Prefix`), through the segment and then its
/// `ExtHistory`.
///
/// Users of a `Dst` rely on: `base..base + op` is initialized, and
/// `base + op..base + op + MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH` is valid
/// for writes in the allocation of `base`.
#[derive(Clone, Copy)]
struct Dst {
    base: *mut u8,
    op: usize,
    window: usize,
}

/// Output of the frame that precedes a `Dst`'s segment and lies apart from
/// it: the `len` bytes that end at `end` (libzstd's `virtualStart` to
/// `dictEnd`). They are initialized; writes to the segment may land in
/// them (the streaming round buffer reuses them), so they are only read
/// through raw pointers. A match that starts in them, at `avail` bytes
/// into the segment, starts in another allocation than the segment's, or
/// more than `WILDCOPY_OVERLENGTH` bytes past `base + avail`.
#[derive(Clone, Copy)]
struct ExtHistory {
    end: *const u8,
    len: usize,
    /// The bytes are a dictionary's content, which a match may reach past
    /// Window_Size while the frame has decoded at most Window_Size bytes
    /// (RFC 8878 lines 1838-1844), the segment then starting at the
    /// frame's first byte. Any other history is bounded by Window_Size.
    dict: bool,
}

impl ExtHistory {
    /// No history outside the segment.
    const NONE: ExtHistory = ExtHistory {
        end: ptr::null(),
        len: 0,
        dict: false,
    };

    /// The history of a frame's first segment: the content of the
    /// dictionary it started from, empty without one.
    fn dict(content: &[u8]) -> ExtHistory {
        ExtHistory {
            end: content.as_ptr_range().end,
            len: content.len(),
            dict: true,
        }
    }
}

/// A block's sequences, executed into the frame by `execute_with_copies`
/// with the copies it picks.
trait BlockSequences {
    /// Execute the sequences straight into `dst`, from `dst.op` on, and
    /// return the block's end (from `dst.base`). All copies use
    /// fixed-size chunks and may overshoot, within the room `Dst`
    /// guarantees.
    ///
    /// # Safety
    /// `dst` meets the `Dst` contract.
    unsafe fn execute<W: WildCopy>(
        self,
        w: W,
        offset_hist: &mut [u32; 3],
        dst: Dst,
    ) -> Result<usize, DecodeError>;
}

/// `execute_with` for a block whose offsets table is `offsets`: with the
/// `ShortOffsets` copies when it gives many short offsets.
///
/// # Safety
/// `dst` meets the `Dst` contract.
unsafe fn execute_with_copies<S: BlockSequences>(
    simd: Level,
    offsets: &FSETable,
    seqs: S,
    offset_hist: &mut [u32; 3],
    block_size_max: usize,
    dst: Dst,
) -> Result<usize, DecodeError> {
    let short = short_offset_share(offsets) >= SHORT_OFFSET_SHARE_MIN;
    execute_with(simd, short, seqs, offset_hist, block_size_max, dst)
}

/// Execute `seqs` with the copies for its block, for the fused decoder and
/// the MT decoder's stage 3 alike: 32-byte ones on the AVX2 level, 16-byte
/// ones otherwise, and the `ShortOffsets` variants when `short`. Each copy
/// type runs in a function of its own. The block may decode to
/// `block_size_max` bytes (RFC 8878 lines 566-568). Returns the block's end
/// in `dst`.
///
/// # Safety
/// `dst` meets the `Dst` contract.
#[inline(always)]
unsafe fn execute_with<S: BlockSequences>(
    simd: Level,
    short: bool,
    seqs: S,
    offset_hist: &mut [u32; 3],
    block_size_max: usize,
    dst: Dst,
) -> Result<usize, DecodeError> {
    let Dst { base, op, window } = dst;
    let end = match simd {
        // SAFETY: fearless_simd makes an `Avx2` only after detecting AVX2
        // and FMA on this CPU (`Level::new`).
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(w) => {
            if short {
                execute_avx2(Avx2ShortOffsets(w), seqs, offset_hist, base, op, window)
            } else {
                execute_avx2(w, seqs, offset_hist, base, op, window)
            }
        }
        _ if short => {
            let w = FallbackShortOffsets(Fallback::new());
            execute_portable(w, seqs, offset_hist, base, op, window)
        }
        _ => execute_portable(Fallback::new(), seqs, offset_hist, base, op, window),
    }?;
    let decoded = end - op;
    if decoded > block_size_max {
        return Err(format!(
            "Block decodes to {} bytes, past Block_Maximum_Size {}",
            decoded, block_size_max
        )
        .into());
    }
    Ok(end)
}

/// `seqs.execute` into the `Dst` of `base`, `op` and `window`. The fields
/// come apart so that they arrive in registers: passed as one `Dst`, by
/// reference, the AVX2 short-offset loop ran 6% more instructions
/// (words_1M L1).
///
/// # Safety
/// The `Dst` meets its contract.
#[inline(never)]
unsafe fn execute_portable<W: WildCopy, S: BlockSequences>(
    w: W,
    seqs: S,
    offset_hist: &mut [u32; 3],
    base: *mut u8,
    op: usize,
    window: usize,
) -> Result<usize, DecodeError> {
    seqs.execute(w, offset_hist, Dst { base, op, window })
}

/// `execute_portable` compiled with AVX2.
///
/// # Safety
/// The CPU supports AVX2, and the `Dst` meets its contract.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn execute_avx2<W: WildCopy, S: BlockSequences>(
    w: W,
    seqs: S,
    offset_hist: &mut [u32; 3],
    base: *mut u8,
    op: usize,
    window: usize,
) -> Result<usize, DecodeError> {
    seqs.execute(w, offset_hist, Dst { base, op, window })
}

/// A compressed block's sequences section after its tables, with the
/// block's literals.
#[derive(Clone, Copy)]
struct SeqInput<'a> {
    num_sequences: u32,
    bit_stream: &'a [u8],
    /// LL, OF, ML tables (`SEQ_TABLES` order).
    fse: [&'a FSETable; 3],
    literals: &'a [u8],
}

/// Decoding each sequence and executing it at once. `literals` holds the
/// block's decoded literals followed by exactly `WILDCOPY_OVERLENGTH` bytes
/// of slack.
impl BlockSequences for SeqInput<'_> {
    #[inline(always)]
    unsafe fn execute<W: WildCopy>(
        self,
        w: W,
        offset_hist: &mut [u32; 3],
        dst: Dst,
    ) -> Result<usize, DecodeError> {
        run_sequences::<W, false>(w, self, ExtHistory::NONE, offset_hist, dst)
    }
}

/// `SeqInput` whose matches reach on into `ext` before the segment,
/// executed by its own instantiation of the loop: the AVX2 loop that can
/// copy from `ext` and go on runs 2-3% more instructions (words_1M L1,
/// rssrc_8M L3, elf_8M L9), so a block without `ext`, as every one-shot
/// one is, takes the loop in which a match before the segment exits. Both
/// give the same verdicts.
struct ExtSeqInput<'a> {
    seqs: SeqInput<'a>,
    ext: ExtHistory,
}

impl BlockSequences for ExtSeqInput<'_> {
    #[inline(always)]
    unsafe fn execute<W: WildCopy>(
        self,
        w: W,
        offset_hist: &mut [u32; 3],
        dst: Dst,
    ) -> Result<usize, DecodeError> {
        run_sequences::<W, true>(w, self.seqs, self.ext, offset_hist, dst)
    }
}

/// Execute the block's sequences into `dst` from `dst.op` on. Returns the
/// block's end, at most `dst.op + MAX_BLOCK_SIZE`, with every byte of
/// `dst.op..end` written; bytes past `end` may have been written too, and
/// nothing before `dst.op` is. Matches reach into `ext` only when `EXT`;
/// otherwise they stop at the segment (`exec_sequence`).
///
/// # Safety
/// `dst` meets the `Dst` contract, and `ext` the `ExtHistory` one when
/// `EXT`.
#[inline(always)]
unsafe fn run_sequences<W: WildCopy, const EXT: bool>(
    w: W,
    seqs: SeqInput<'_>,
    ext: ExtHistory,
    offset_hist: &mut [u32; 3],
    dst: Dst,
) -> Result<usize, DecodeError> {
    let Dst {
        base: out,
        op,
        window,
    } = dst;
    let SeqInput {
        num_sequences,
        bit_stream,
        fse,
        literals,
    } = seqs;
    let [ll_t, of_t, ml_t] = fse;
    let ll_dt = ll_t.decode();
    let of_dt = of_t.decode();
    let ml_dt = ml_t.decode();
    let ll_log = u32::from(ll_t.accuracy_log);
    let of_log = u32::from(of_t.accuracy_log);
    let ml_log = u32::from(ml_t.accuracy_log);
    // The state lookups below are unchecked: an initial state is
    // `accuracy_log` bits, and every cell of a table built by
    // `build_decoding_table` or `build_rle` satisfies
    // `next_state + (1 << num_bits) <= table size`, so a state is always a
    // valid index of a table with exactly `1 << accuracy_log` cells.
    if ll_dt.len() != 1 << ll_log || of_dt.len() != 1 << of_log || ml_dt.len() != 1 << ml_log {
        return Err("FSE table is uninitialized".into());
    }
    let literals_len = literals.len() - WILDCOPY_OVERLENGTH;
    let oend = op + MAX_BLOCK_SIZE;

    let mut br = BitDStream::new(bit_stream)?;
    // ZSTD_initFseState: LL, OF, ML order, each followed by a reload.
    let ll_state = br.read_bits(ll_log);
    br.reload();
    let of_state = br.read_bits(of_log);
    br.reload();
    let ml_state = br.read_bits(ml_log);
    br.reload();

    let mut st = SeqState {
        ll: ll_state,
        ml: ml_state,
        of: of_state,
        hist: [
            offset_hist[0] as usize,
            offset_hist[1] as usize,
            offset_hist[2] as usize,
        ],
    };
    let lit_start = literals.as_ptr();
    // SAFETY: `op <= oend` are within the writable range of the function
    // contract and `literals_len < literals.len()`, so every pointer below
    // stays inside its buffer.
    let mut cur = unsafe {
        SeqCursor {
            op: out.add(op),
            lit: lit_start,
        }
    };
    let lim = unsafe {
        SeqLimits {
            oend_w: out.add(oend),
            lit_limit: lit_start.add(literals_len),
            prefix: out,
            ext,
            window,
        }
    };

    for _ in 1..num_sequences {
        let (ll, ml, offset) = decode_sequence(&mut br, &mut st, ll_dt, ml_dt, of_dt, false);
        exec_sequence::<W, EXT>(w, &mut cur, &lim, ll, ml, offset).map_err(seq_error_message)?;
    }
    let (ll, ml, offset) = decode_sequence(&mut br, &mut st, ll_dt, ml_dt, of_dt, true);
    exec_sequence::<W, EXT>(w, &mut cur, &lim, ll, ml, offset).map_err(seq_error_message)?;
    let hist = st.hist;
    // Both cursors only ever advance within their slices (see
    // `exec_sequence`), so these differences are in-bounds indexes.
    let mut op = cur.op as usize - out as usize;
    let lit_pos = cur.lit as usize - lit_start as usize;

    if !br.is_finished() {
        return Err("Sequence bitstream not fully consumed".into());
    }

    // Last literals segment.
    let rest = literals_len - lit_pos;
    if op + rest > oend {
        return Err(seq_error_message(SeqError::BlockTooLarge));
    }
    // SAFETY: `op + rest <= oend` is writable per the function contract,
    // and `lit_pos + rest == literals_len <= literals.len()`.
    ptr::copy_nonoverlapping(literals.as_ptr().add(lit_pos), out.add(op), rest);
    op += rest;

    *offset_hist = [hist[0] as u32, hist[1] as u32, hist[2] as u32];
    Ok(op)
}

/// FSE states and repeat offsets of the sequences decoder (seqState_t).
struct SeqState {
    ll: usize,
    ml: usize,
    of: usize,
    hist: [usize; 3],
}

/// Where the next sequence writes its output and reads its literals.
///
/// Invariant: `op <= SeqLimits::oend_w` and `lit <= SeqLimits::lit_limit`
/// of the limits it is executed against, so both point into their buffers.
struct SeqCursor {
    op: *mut u8,
    lit: *const u8,
}

/// Bounds of one block's sequence execution: the output limit less
/// `WILDCOPY_OVERLENGTH`, the literals end less `WILDCOPY_OVERLENGTH`, and
/// how far back a match may copy from: the current segment's first byte,
/// then the history before it, and Window_Size (`Dst`).
#[derive(Clone, Copy)]
struct SeqLimits {
    oend_w: *mut u8,
    lit_limit: *const u8,
    prefix: *mut u8,
    ext: ExtHistory,
    window: usize,
}

/// Decode one sequence (ZSTD_decodeSequence): literal length, match
/// length, offset. `is_last` skips the state update, which would read
/// past the end of the stream.
#[inline(always)]
fn decode_sequence(
    br: &mut BitDStream<'_>,
    st: &mut SeqState,
    ll_dt: &[FSEEntry],
    ml_dt: &[FSEEntry],
    of_dt: &[FSEEntry],
    is_last: bool,
) -> (usize, usize, usize) {
    // SAFETY: each state is below its table's length (see `run_sequences`).
    let (ll_e, ml_e, of_e) = unsafe {
        (
            table_entry(ll_dt, st.ll),
            table_entry(ml_dt, st.ml),
            table_entry(of_dt, st.of),
        )
    };

    let mut ll = ll_e.base_value as usize;
    let mut ml = ml_e.base_value as usize;
    let ll_bits = u32::from(ll_e.extra_bits);
    let ml_bits = u32::from(ml_e.extra_bits);
    let of_bits = u32::from(of_e.extra_bits);
    let total_bits = ll_bits + ml_bits + of_bits;
    let hist = &mut st.hist;

    // Offset and repeat-offset history.
    let offset = if of_bits > 1 {
        let o = of_e.base_value as usize + br.read_bits_fast(of_bits);
        hist[2] = hist[1];
        hist[1] = hist[0];
        hist[0] = o;
        o
    } else {
        let ll0 = usize::from(ll == 0);
        if of_bits == 0 {
            let o = hist[ll0];
            hist[1] = hist[1 - ll0];
            hist[0] = o;
            o
        } else {
            // Offset code 1: base value 1 plus one extra bit selects a
            // repeat offset 1..=3.
            let o = 1 + ll0 + br.read_bits_fast(1);
            let mut temp = if o == 3 {
                hist[0].wrapping_sub(1)
            } else {
                hist[o]
            };
            if temp == 0 {
                // Corrupt input: force an offset that execution rejects.
                temp = usize::MAX;
            }
            if o != 1 {
                hist[2] = hist[1];
            }
            hist[1] = hist[0];
            hist[0] = temp;
            temp
        }
    };

    if ml_bits > 0 {
        ml += br.read_bits_fast(ml_bits);
    }
    // A reload guarantees 57 bits; the three state updates below need
    // up to 26 more, so reload now if this sequence's extra bits could
    // leave fewer than that.
    if total_bits >= 57 - 26 {
        br.reload();
    }
    if ll_bits > 0 {
        ll += br.read_bits_fast(ll_bits);
    }

    if !is_last {
        st.ll = usize::from(ll_e.next_state) + br.read_bits(u32::from(ll_e.num_bits));
        st.ml = usize::from(ml_e.next_state) + br.read_bits(u32::from(ml_e.num_bits));
        st.of = usize::from(of_e.next_state) + br.read_bits(u32::from(of_e.num_bits));
        br.reload();
    }
    (ll, ml, offset)
}

/// Copy `ll` literals then `ml` match bytes from `offset` back
/// (ZSTD_execSequenceSplitLitBuffer) and advance `cur`. Both buffers carry
/// `WILDCOPY_OVERLENGTH` bytes of slack past their limits. A match that
/// starts before the segment is copied from `lim.ext` when `EXT`, and is
/// `OffsetTooFar` otherwise, as it is with an empty `lim.ext`.
#[inline(always)]
fn exec_sequence<W: WildCopy, const EXT: bool>(
    w: W,
    cur: &mut SeqCursor,
    lim: &SeqLimits,
    ll: usize,
    ml: usize,
    offset: usize,
) -> Result<(), SeqError> {
    let op = cur.op;
    let lit = cur.lit;
    // The window has a branch of its own, ahead of the rest: folded into
    // the segment-start check with `min` it cost AVX2 words_1M decode 2-6%,
    // and after that check 2.5-3.7% of cycles, against 0.6-2.4% here.
    if offset > lim.window {
        if !EXT {
            return Err(SeqError::OffsetPastWindow);
        }
        return exec_sequence_past_window(w, cur, lim, ll, ml, offset);
    }
    // Addresses are compared as integers: `ll` and `ml` are below 2^32
    // and pointers are below 2^63, so these sums cannot wrap.
    let o_lit_end = op as usize + ll;
    let o_match_end = o_lit_end + ml;
    if lit as usize + ll > lim.lit_limit as usize {
        return Err(SeqError::NotEnoughLiterals);
    }
    if o_match_end > lim.oend_w as usize {
        return Err(SeqError::BlockTooLarge);
    }
    // Rejects offset 0 as well (it wraps to usize::MAX).
    if offset.wrapping_sub(1) >= o_lit_end - lim.prefix as usize {
        if !EXT {
            return Err(SeqError::OffsetTooFar);
        }
        let avail = o_lit_end - lim.prefix as usize;
        // A match that starts `back` bytes before the end of `ext` and ends
        // `WILDCOPY_OVERLENGTH` bytes or more before it (`offset` is then
        // not 0) copies on here: out of line, its copy cost streamed
        // rssrc_8M L3 MT decode 7%, with every wrap of the round buffer.
        let back = offset.wrapping_sub(avail);
        if back <= lim.ext.len && back >= ml + WILDCOPY_OVERLENGTH {
            // SAFETY: as below for the literals. The match reads `ml + 31`
            // bytes from `back` bytes before the end of `ext`, within it,
            // and lies apart from the segment or ahead of `dst` by more
            // than `W::WIDTH` bytes (the `ExtHistory` contract).
            unsafe {
                copy_literals(w, op, lit, ll);
                w.wildcopy(op.add(ll), lim.ext.end.sub(back), ml);
                cur.lit = lit.add(ll);
                cur.op = op.add(ll + ml);
            }
            return Ok(());
        }
        // SAFETY: as below, the checks above hold; the match starts before
        // the segment, which `exec_sequence_ext` takes from there.
        unsafe {
            exec_sequence_ext(w, op, lit, (ll, ml, offset), avail, lim.ext)?;
            cur.lit = lit.add(ll);
            cur.op = op.add(ll + ml);
        }
        return Ok(());
    }

    // SAFETY: the checks above give, with `ml >= 1`,
    //   lit + ll <= lit_limit, which has 32 readable bytes after it,
    //   o_match_end <= oend_w, which has 32 writable bytes after it,
    //   1 <= offset <= min(o_lit_end - prefix, window).
    // Literals come from another buffer and are read from `lit` and
    // written from `op` with at most 31 bytes of overshoot. The match
    // meets `copy_match`'s contract with `avail = o_lit_end - prefix`:
    // `prefix..o_lit_end` is initialized (by the caller up to the block,
    // then by this block's earlier sequences and these literals), and
    // `o_lit_end + ml + 31` is writable. The advanced cursors keep the
    // `SeqCursor` invariant.
    unsafe {
        copy_literals(w, op, lit, ll);
        cur.lit = lit.add(ll);

        w.copy_match(op.add(ll), offset, ml, o_lit_end - lim.prefix as usize);
        cur.op = op.add(ll + ml);
    }
    Ok(())
}

/// `exec_sequence` for a match past Window_Size, which only a dictionary
/// takes: one in `lim.ext`, while the segment holds at most Window_Size
/// bytes before the match (`ExtHistory::dict`). The match is then bounded
/// by the dictionary's first byte alone.
#[cold]
#[inline(never)]
fn exec_sequence_past_window<W: WildCopy>(
    w: W,
    cur: &mut SeqCursor,
    lim: &SeqLimits,
    ll: usize,
    ml: usize,
    offset: usize,
) -> Result<(), SeqError> {
    // As in `exec_sequence`, the sum cannot wrap.
    let avail = cur.op as usize + ll - lim.prefix as usize;
    if !lim.ext.dict || avail > lim.window {
        return Err(SeqError::OffsetPastWindow);
    }
    let lim = SeqLimits {
        window: usize::MAX,
        ..*lim
    };
    exec_sequence::<W, true>(w, cur, &lim, ll, ml, offset)
}

/// Copy `ll` literals from `lit` to `op`, overshooting by up to 31 bytes.
///
/// # Safety
/// `ll + 31` bytes readable at `lit` and writable at `op`, in different
/// buffers.
#[inline(always)]
unsafe fn copy_literals<W: WildCopy>(w: W, op: *mut u8, lit: *const u8, ll: usize) {
    // Nearly always at most 16 bytes.
    copy16(op, lit);
    if ll > 16 {
        w.wildcopy(op.add(16), lit.add(16), ll - 16);
    }
}

/// `exec_sequence` for the sequence `(ll, ml, offset)` at `op` whose match
/// starts before the current segment, `avail` bytes of which precede the
/// match's destination (ZSTD_execSequence's extDict branch): it starts in
/// `ext`, or before it, which is `OffsetBeforeDict` when `ext` is a
/// dictionary and `OffsetTooFar` otherwise, as offset 0 is. A match that
/// runs past the end of `ext` continues from the segment's first byte.
///
/// # Safety
/// `exec_sequence`'s checks before its segment-start check hold, and so
/// do the `Dst` contract for the segment and the `ExtHistory` one for
/// `ext`.
#[cold]
#[inline(never)]
unsafe fn exec_sequence_ext<W: WildCopy>(
    w: W,
    op: *mut u8,
    lit: *const u8,
    (ll, ml, offset): (usize, usize, usize),
    avail: usize,
    ext: ExtHistory,
) -> Result<(), SeqError> {
    // `offset > avail` here unless it is 0.
    if offset == 0 {
        return Err(SeqError::OffsetTooFar);
    }
    if offset - avail > ext.len {
        return Err(if ext.dict {
            SeqError::OffsetBeforeDict
        } else {
            SeqError::OffsetTooFar
        });
    }
    copy_literals(w, op, lit, ll);
    let dst = op.add(ll);
    // The match starts `back` bytes before the end of `ext`, and its first
    // `head` bytes are there. `ext` may share a buffer with the segment,
    // so the copy allows overlap.
    let back = offset - avail;
    let head = back.min(ml);
    ptr::copy(ext.end.sub(back), dst, head);
    if ml > head {
        // The rest starts at the segment's first byte, `offset` bytes
        // before `dst + head`, with all `offset` bytes between written.
        w.copy_match(dst.add(head), offset, ml - head, offset);
    }
    Ok(())
}

/// ZSTD_copy16.
///
/// # Safety
/// 16 bytes readable at `src` and writable at `dst`, not overlapping.
#[inline(always)]
unsafe fn copy16(dst: *mut u8, src: *const u8) {
    ptr::copy_nonoverlapping(src, dst, 16);
}

/// ZSTD_wildcopy(no_overlap): 16-byte chunks that may overshoot `len` by up
/// to 31 bytes on both sides.
///
/// # Safety
/// `len + 31` bytes readable at `src` and writable at `dst`, and either the
/// two ranges are disjoint or `dst - src >= 16` or `src - dst >= 16`.
#[inline(always)]
unsafe fn wildcopy(mut dst: *mut u8, mut src: *const u8, len: usize) {
    copy16(dst, src);
    if len <= 16 {
        return;
    }
    let end = dst.add(len);
    dst = dst.add(16);
    src = src.add(16);
    loop {
        copy16(dst, src);
        copy16(dst.add(16), src.add(16));
        dst = dst.add(32);
        src = src.add(32);
        if dst >= end {
            break;
        }
    }
}

/// The wide copy of sequence execution at one SIMD level.
trait WildCopy: Copy {
    /// Bytes per chunk of `wildcopy`: its least safe `dst - src` for
    /// overlapping ranges.
    const WIDTH: usize;

    /// ZSTD_wildcopy(no_overlap) in `WIDTH`-byte chunks, overshooting `len`
    /// by up to 31 bytes.
    ///
    /// # Safety
    /// `len + 31` bytes readable at `src` and writable at `dst`, and either
    /// the two ranges are disjoint or `dst - src >= WIDTH` or `src - dst >=
    /// WIDTH`: the chunks go forward, and each is read before it is written.
    unsafe fn wildcopy(self, dst: *mut u8, src: *const u8, len: usize);

    /// Copy the `ml`-byte match that starts `offset` bytes before `dst`,
    /// which repeats with period `offset` where `offset < ml`, overshooting
    /// by up to 31 bytes (ZSTD_execSequence).
    ///
    /// # Safety
    /// `1 <= offset <= avail` and `ml >= 1`; the `avail` bytes before `dst`
    /// are initialized, and `ml + 31` bytes from `dst` are writable, all in
    /// one allocation.
    unsafe fn copy_match(self, dst: *mut u8, offset: usize, ml: usize, avail: usize);
}

impl WildCopy for Fallback {
    const WIDTH: usize = 16;

    #[inline(always)]
    unsafe fn wildcopy(self, dst: *mut u8, src: *const u8, len: usize) {
        wildcopy(dst, src, len)
    }

    #[inline(always)]
    unsafe fn copy_match(self, dst: *mut u8, offset: usize, ml: usize, avail: usize) {
        let _ = avail;
        let src = dst.sub(offset) as *const u8;
        if offset >= 16 {
            // Two chunks before the first length test: `wildcopy`'s
            // `len <= 16` exit and loop exit mispredict on source code's
            // matches. Past a match of at most 16 bytes the second chunk
            // is overshoot and copies `src` again: from `src + 16`, below
            // offset 32 it loads the first chunk's store, a wait on every
            // sequence that left mixed_1M (offset 16, 7 bytes) 19% slower.
            copy16(dst, src);
            let s2 = if ml > 16 { src.add(16) } else { src };
            copy16(dst.add(16), s2);
            if ml > 32 {
                wildcopy(dst.add(32), src.add(32), ml - 32);
            }
        } else {
            let (dst, src) = overlap_copy8(dst, src, offset);
            if ml > 8 {
                wildcopy_overlap8(dst, src, ml - 8);
            }
        }
    }
}

/// Portable copies for blocks with many short offsets
/// (`execute_with_copies`), whose `offset < 8` and `offset < 16`
/// tests in `copy_match` are unpredictable: the first 16 bytes of a match
/// take the same straight-line copy at every offset, continued in 8-byte
/// chunks.
#[derive(Clone, Copy)]
struct FallbackShortOffsets(Fallback);

impl WildCopy for FallbackShortOffsets {
    const WIDTH: usize = 16;

    #[inline(always)]
    unsafe fn wildcopy(self, dst: *mut u8, src: *const u8, len: usize) {
        // SAFETY: the caller's.
        unsafe { self.0.wildcopy(dst, src, len) }
    }

    #[inline(always)]
    unsafe fn copy_match(self, dst: *mut u8, offset: usize, ml: usize, avail: usize) {
        let _ = avail;
        let src = dst.sub(offset) as *const u8;
        // ZSTD_overlapCopy8, with row 8 for every `offset >= 8`: four single
        // bytes, then 4 from `src + DEC32`; afterwards `dst - src` is a
        // multiple of the offset of at least 8.
        const DEC32: [usize; 9] = [0, 1, 2, 1, 4, 4, 4, 4, 4];
        const DEC64: [usize; 9] = [8, 8, 8, 7, 8, 9, 10, 11, 4];
        let o = offset.min(8);
        *dst = *src;
        *dst.add(1) = *src.add(1);
        *dst.add(2) = *src.add(2);
        *dst.add(3) = *src.add(3);
        let s2 = src.add(DEC32[o]);
        ptr::copy_nonoverlapping(s2, dst.add(4), 4);
        let (dst, src) = (dst.add(8), s2.add(8).sub(DEC64[o]));
        ptr::copy_nonoverlapping(src, dst, 8);
        if ml > 16 {
            wildcopy_overlap8(dst.add(8), src.add(8), ml - 16);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl WildCopy for Avx2 {
    const WIDTH: usize = 32;

    #[inline(always)]
    unsafe fn wildcopy(self, dst: *mut u8, src: *const u8, len: usize) {
        // SAFETY: `self` proves AVX2; the ranges are the caller's.
        unsafe { wildcopy32(dst, src, len) }
    }

    #[inline(always)]
    unsafe fn copy_match(self, dst: *mut u8, offset: usize, ml: usize, avail: usize) {
        let _ = avail;
        let src = dst.sub(offset) as *const u8;
        if offset >= 32 {
            // The first 32 bytes end at or before `dst`.
            ptr::copy_nonoverlapping(src, dst, 32);
            if ml > 32 {
                // SAFETY: `self` proves AVX2; `offset >= 32` and the rest
                // is the caller's.
                unsafe { wildcopy_periodic_outlined(dst.add(32), offset, ml - 32) }
            }
        } else if offset >= WILDCOPY_VECLEN {
            wildcopy(dst, src, ml);
        } else {
            // Copy 8 bytes and spread the offset to at least 8, then
            // continue with 8-byte chunks.
            let (dst, src) = overlap_copy8(dst, src, offset);
            if ml > 8 {
                wildcopy_overlap8(dst, src, ml - 8);
            }
        }
    }
}

/// AVX2 copies for blocks with many short offsets
/// (`execute_with_copies`), whose `offset >= 32` and `offset >= 16`
/// tests in `copy_match` are unpredictable: the first 32 bytes of a match
/// take one shuffled store at every offset.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct Avx2ShortOffsets(Avx2);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl WildCopy for Avx2ShortOffsets {
    const WIDTH: usize = 32;

    #[inline(always)]
    unsafe fn wildcopy(self, dst: *mut u8, src: *const u8, len: usize) {
        // SAFETY: the caller's.
        unsafe { self.0.wildcopy(dst, src, len) }
    }

    #[inline(always)]
    unsafe fn copy_match(self, dst: *mut u8, offset: usize, ml: usize, avail: usize) {
        if avail < 32 {
            // Within 32 bytes of the frame start there are too few bytes
            // before `dst` for the loads of `copy_match_short`.
            // SAFETY: the caller's.
            unsafe { self.0.copy_match(dst, offset, ml, avail) }
        } else {
            // SAFETY: `self.0` proves AVX2; `avail >= 32` and the rest is
            // the caller's.
            unsafe { copy_match_short(dst, offset, ml) }
        }
    }
}

/// vpshufb masks of `copy_match_short`, one row per `min(offset, 32)`;
/// row 0 is unused. Output byte `i` of a match at `m = min(offset, 32)`
/// is byte `32 - m + i % m` of the 32 bytes loaded from `max(offset, 32)`
/// back: mask 0 picks it from their low half, broadcast to both lanes,
/// mask 1 from their high half, and the other mask zeroes it (0x80). Row
/// 32 (`offset >= 32`) copies the 32 bytes unchanged.
#[repr(align(64))]
struct MatchMasks([[[u8; 32]; 2]; 33]);

const MATCH_MASKS: MatchMasks = {
    let mut t = [[[0x80u8; 32]; 2]; 33];
    let mut m = 1;
    while m <= 32 {
        let mut i = 0;
        while i < 32 {
            let idx = 32 - m + i % m;
            if idx < 16 {
                t[m][0][i] = idx as u8;
            } else {
                t[m][1][i] = (idx - 16) as u8;
            }
            i += 1;
        }
        m += 1;
    }
    MatchMasks(t)
};

/// For an offset below 16, its least multiple of at least 16: a distance
/// at which 16-byte chunks continue the period.
const PERIOD_SPREAD: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut o = 1;
    while o < 16 {
        t[o] = (o * 16_usize.div_ceil(o)) as u8;
        o += 1;
    }
    t
};

/// `WildCopy::copy_match` with no branch on the offset for the first 32
/// bytes: the 32 bytes from `max(offset, 32)` back are shuffled into the
/// match, which is a plain copy for `offset >= 32` and repeats the period
/// below it (ZSTD_overlapCopy8). A match past 32 bytes continues in
/// `wildcopy_periodic`.
///
/// # Safety
/// The CPU supports AVX2, `avail >= 32`, and `WildCopy::copy_match`'s
/// contract.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn copy_match_short(dst: *mut u8, offset: usize, ml: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, _mm256_broadcastsi128_si256, _mm256_load_si256, _mm256_or_si256,
        _mm256_shuffle_epi8, _mm256_storeu_si256, _mm_loadu_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, _mm256_broadcastsi128_si256, _mm256_load_si256, _mm256_or_si256,
        _mm256_shuffle_epi8, _mm256_storeu_si256, _mm_loadu_si128,
    };
    let masks = &MATCH_MASKS.0[offset.min(32)];
    // `offset <= avail` and `32 <= avail`, so the 32 bytes read are
    // initialized and end at or before `dst`.
    let src = dst.sub(offset.max(32));
    let lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(src.cast()));
    let hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(src.add(16).cast()));
    let head = _mm256_or_si256(
        _mm256_shuffle_epi8(lo, _mm256_load_si256(masks[0].as_ptr().cast::<__m256i>())),
        _mm256_shuffle_epi8(hi, _mm256_load_si256(masks[1].as_ptr().cast::<__m256i>())),
    );
    _mm256_storeu_si256(dst.cast(), head);
    if ml > 32 {
        // The 32 bytes written repeat the period, so any multiple of
        // `offset` continues it.
        let dist = if offset >= 16 {
            offset
        } else {
            usize::from(PERIOD_SPREAD[offset])
        };
        wildcopy_periodic(dst.add(32), dist, ml - 32);
    }
}

/// Continue a periodic match at `dst` from `dist` bytes back: in 32-byte
/// chunks where they are the faster ones (`wide_chunks`), else in 16-byte
/// chunks as ZSTD_wildcopy does.
///
/// # Safety
/// The CPU supports AVX2, `dist >= 16`, the `dist` bytes before `dst` are
/// initialized, and `len + 31` bytes from `dst` are writable, all in one
/// allocation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wildcopy_periodic(dst: *mut u8, dist: usize, len: usize) {
    let src = dst.sub(dist) as *const u8;
    if wide_chunks(dist) {
        wildcopy32(dst, src, len);
    } else {
        wildcopy(dst, src, len);
    }
}

/// `wildcopy_periodic` out of line, for `Avx2::copy_match`: inlined into
/// the sequence loop, its two loops changed the loop's register
/// allocation to one more store per sequence, 6% slower on f64_1M, whose
/// matches never get here.
///
/// # Safety
/// `wildcopy_periodic`'s.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn wildcopy_periodic_outlined(dst: *mut u8, dist: usize, len: usize) {
    wildcopy_periodic(dst, dist, len);
}

/// Whether 32-byte chunks copy a match at distance `dist` faster than
/// 16-byte ones. A chunk's load that partly overlaps the store of an
/// earlier chunk waits for that store to commit (no store-to-load
/// forwarding), so a copy has `dist` rounded down to a multiple of its
/// chunk size in flight per wait: at 58 bytes (text_1M) 32 in 32-byte
/// chunks against 48, 0.66x. 16-byte chunks keep at least as many bytes
/// in flight up to 128, and at the odd multiples of 16 they forward
/// whole from one store each, where 32-byte chunks ran 0.65x (144) to
/// 0.96x (368) on Zen 5. Past these, 32-byte chunks were up to 1.2x.
#[inline(always)]
fn wide_chunks(dist: usize) -> bool {
    dist > 128 && dist % 32 != 16
}

/// `wildcopy` in 32-byte chunks (one AVX2 load and store each).
///
/// # Safety
/// The CPU supports AVX2; `len + 31` bytes readable at `src` and writable
/// at `dst`, and either the two ranges are disjoint or `dst - src >= 32` or
/// `src - dst >= 32`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wildcopy32(mut dst: *mut u8, mut src: *const u8, len: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};
    let end = dst.add(len);
    loop {
        _mm256_storeu_si256(
            dst.cast::<__m256i>(),
            _mm256_loadu_si256(src.cast::<__m256i>()),
        );
        dst = dst.add(32);
        src = src.add(32);
        if dst >= end {
            break;
        }
    }
}

/// ZSTD_wildcopy(overlap_src_before_dst) with `8 <= dst - src < 16`:
/// 8-byte chunks, overshooting by up to 7 bytes.
///
/// # Safety
/// `len + 7` bytes readable at `src` and writable at `dst`, `dst - src >= 8`.
#[inline(always)]
unsafe fn wildcopy_overlap8(mut dst: *mut u8, mut src: *const u8, len: usize) {
    let end = dst.add(len);
    loop {
        ptr::copy_nonoverlapping(src, dst, 8);
        dst = dst.add(8);
        src = src.add(8);
        if dst >= end {
            break;
        }
    }
}

/// ZSTD_overlapCopy8: copy 8 bytes from `src` to `dst` and return both
/// advanced so that afterwards `dst - src >= 8`.
///
/// # Safety
/// `dst - src == offset` with `1 <= offset < 16`; 12 bytes readable at `src`
/// and 8 writable at `dst`, all within one allocation.
#[inline(always)]
unsafe fn overlap_copy8(dst: *mut u8, src: *const u8, offset: usize) -> (*mut u8, *const u8) {
    if offset < 8 {
        const DEC32: [usize; 8] = [0, 1, 2, 1, 4, 4, 4, 4];
        const DEC64: [usize; 8] = [8, 8, 8, 7, 8, 9, 10, 11];
        *dst = *src;
        *dst.add(1) = *src.add(1);
        *dst.add(2) = *src.add(2);
        *dst.add(3) = *src.add(3);
        // `dst + 4` is at least 4 bytes past `src + DEC32[offset]`.
        let s2 = src.add(DEC32[offset]);
        ptr::copy_nonoverlapping(s2, dst.add(4), 4);
        (dst.add(8), s2.add(8).sub(DEC64[offset]))
    } else {
        ptr::copy_nonoverlapping(src, dst, 8);
        (dst.add(8), src.add(8))
    }
}

// ============================================================
// Frame decoder
// ============================================================

/// Where a `FrameDecoder` stands in its input (ZSTD_decompressContinue's
/// ZSTDds_* stages). Each stage but `Skip` takes one unit: a frame header,
/// a block header, a block's content, or a Content_Checksum, whole
/// (`FrameDecoder::unit_len`).
enum Stage {
    /// Between frames: the next unit is a frame header, or a skippable
    /// frame's magic number and Frame_Size.
    FrameHeader,
    /// `left` bytes of a skippable frame's User_Data still to skip, in
    /// pieces of any size.
    Skip { left: u64 },
    /// The blocks of a frame: the next unit is a block header
    /// (ZSTDds_decodeBlockHeader), or with `header`, the content of the
    /// block it heads (ZSTDds_decompressBlock).
    Block {
        frame: Frame,
        header: Option<BlockHeader>,
    },
    /// The frame's Content_Checksum, which must equal `computed`.
    Checksum { computed: u32 },
}

// Every frame builds a `Stage::Block`, moves it and leaves it: with drop
// glue, each of those cost a frame of one block a call or more.
const _: () = assert!(!std::mem::needs_drop::<Stage>());

/// What the unit `FrameDecoder::process` took was.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Event {
    /// A frame's header: its blocks follow.
    FrameStarted,
    /// A skippable frame's header or a block, not the frame's last unit.
    Continue,
    /// A frame's last unit: the decoder is between frames again.
    FrameEnded,
}

/// The frame being decoded: the bounds its header sets and what its blocks
/// have decoded to (ZSTD_DCtx's `fParams`, `decodedSize`, `xxhState`).
struct Frame {
    /// Window_Size: how far back a match reaches, `usize::MAX` for a window
    /// past the address space.
    window: usize,
    /// Block_Maximum_Size (RFC 8878 lines 557-564), the bound
    /// `parse_block_header` puts on every Block_Size, `split_block` on a
    /// compressed block's literals and `execute_with_copies` on what its
    /// sequences decode to.
    block_size_max: usize,
    content_size: Option<u64>,
    decoded: u64,
    checksum: Option<Xxh64>,
}

impl Frame {
    fn new(header: &FrameHeader) -> Result<Frame, DecodeError> {
        let window_size = header.window_size()?;
        Ok(Frame {
            window: usize::try_from(window_size).unwrap_or(usize::MAX),
            block_size_max: window_size.min(MAX_BLOCK_SIZE as u64) as usize,
            content_size: header.frame_content_size(),
            decoded: 0,
            checksum: header.descriptor.content_checksum_flag().then(Xxh64::new),
        })
    }

    /// Account for the bytes a block decoded to, while they are in cache: a
    /// frame that decodes past its Frame_Content_Size fails here, which
    /// keeps a decoder's buffer within that size.
    fn block_decoded(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        self.decoded += bytes.len() as u64;
        if let Some(h) = &mut self.checksum {
            h.update(bytes);
        }
        match self.content_size {
            Some(fcs) if self.decoded > fcs => Err(self.content_size_mismatch(fcs)),
            _ => Ok(()),
        }
    }

    /// Frame_Content_Size says the frame fits in one block. An encoder may
    /// still have split it into several, but its content is too small to
    /// decode in parallel, and reserving room for all of it costs at most
    /// one block more than its blocks can decode to.
    fn fits_one_block(&self) -> bool {
        self.content_size
            .is_some_and(|fcs| fcs <= self.block_size_max as u64)
    }

    fn content_size_mismatch(&self, fcs: u64) -> DecodeError {
        format!(
            "Frame content size mismatch: header says {}, decoded {}",
            fcs, self.decoded
        )
        .into()
    }
}

/// Where a `FrameDecoder` decodes frames to: for `decompress`, the output
/// `Vec` (`VecOut`); for `Decompressor`, its window's round buffer
/// (`RingOut`).
trait FrameOut {
    /// Make ready for a frame whose matches reach at most `window` bytes
    /// back, and which decodes to `content_size` bytes if that is known.
    /// Neither claim allocates: only what the blocks decode to does.
    fn start(&mut self, window: usize, content_size: Option<u64>);

    /// The destination of the frame's next block and the history before
    /// its segment, which meet the `Dst` and `ExtHistory` contracts, or an
    /// error if there is no room for it.
    fn block_dst(&mut self) -> Result<(Dst, ExtHistory), DecodeError>;

    /// Take the block written to the last `block_dst` up to `end` (from its
    /// `base`), and return the block's bytes.
    ///
    /// # Safety
    /// Every byte of that destination from its `op` to `end` is written.
    unsafe fn commit(&mut self, end: usize) -> &[u8];
}

/// The resumable core of both decoders (ZSTD_decompressContinue): it takes
/// the input one unit at a time and decodes each block into a `FrameOut`.
/// `decompress` hands it units of the whole input, `Decompressor` the
/// units it gathers from the pieces it gets, so both take every verdict
/// here.
struct FrameDecoder {
    stage: Stage,
    scratch: Option<DecoderScratch>,
    simd: Level,
    /// When `decode_blocks_parallel` takes blocks to the rayon pool;
    /// `None` if never.
    #[cfg(feature = "parallel")]
    parallel: Option<parallel::Gate>,
    /// The blocks of the frame being decoded on pool tasks, kept from call
    /// to call: none until the frame first takes blocks to the pool, and
    /// none again once the decoder leaves the frame (`leave_frame`). Here
    /// rather than in `Frame`, so that `Stage` has no drop glue: a frame
    /// that never takes blocks to the pool builds, moves and drops none.
    #[cfg(feature = "parallel")]
    pipeline: Option<Box<parallel::Pipeline>>,
}

impl FrameDecoder {
    fn new(opts: &DecodeOptions) -> FrameDecoder {
        FrameDecoder {
            stage: Stage::FrameHeader,
            scratch: None,
            simd: opts.simd_level(),
            #[cfg(feature = "parallel")]
            parallel: parallel::Gate::new(opts),
            #[cfg(feature = "parallel")]
            pipeline: None,
        }
    }

    /// Leave the current stage for `Stage::FrameHeader`, returning it: the
    /// only way out of `Stage::Block`, so that a frame's pipeline ends with
    /// the frame.
    fn leave_frame(&mut self) -> Stage {
        // The test inline, the drop out of line: as `self.pipeline = None`,
        // a frame with no pipeline paid a call into the drop glue.
        #[cfg(feature = "parallel")]
        if let Some(pipeline) = self.pipeline.take() {
            std::hint::cold_path();
            drop(pipeline);
        }
        std::mem::replace(&mut self.stage, Stage::FrameHeader)
    }

    /// Whether a chain of the frame's pipeline is running.
    #[cfg(feature = "parallel")]
    fn chain_active(&self) -> bool {
        self.pipeline.as_ref().is_some_and(|p| p.active())
    }

    /// Inside a skippable frame's User_Data.
    fn skipping(&self) -> bool {
        matches!(self.stage, Stage::Skip { .. })
    }

    /// Right after `Event::FrameStarted`: the frame, and the scratch with
    /// the tables and repeat offsets it starts from, which select those of
    /// the dictionary given to `process` if the frame started from one.
    fn frame_start(&mut self) -> (&mut Frame, &DecoderScratch) {
        match (&mut self.stage, &self.scratch) {
            (Stage::Block { frame, .. }, Some(scratch)) => (frame, scratch),
            _ => unreachable!("a started frame has blocks and a scratch"),
        }
    }

    /// How many bytes the current unit takes, `head` being its first bytes
    /// so far, which may be fewer, or more: a frame header's length follows
    /// from its first `FRAME_HEADER_PREFIX_LEN` bytes, so with fewer the
    /// length is what it takes to learn it. The bytes of a skippable frame
    /// left to skip, in `Stage::Skip`.
    fn unit_len(&self, head: &[u8]) -> usize {
        match &self.stage {
            Stage::FrameHeader => frame_header_len(head),
            Stage::Skip { left } => usize::try_from(*left).unwrap_or(usize::MAX),
            Stage::Block { header: None, .. } => BLOCK_HEADER_LEN,
            Stage::Block {
                header: Some(block),
                ..
            } => block.content_size as usize,
            Stage::Checksum { .. } => CHECKSUM_LEN,
        }
    }

    /// Skip up to `avail` bytes of a skippable frame: returns how many it
    /// skipped and whether that ended the frame.
    fn skip(&mut self, avail: usize) -> (usize, bool) {
        let Stage::Skip { left } = &mut self.stage else {
            return (0, false);
        };
        let n = usize::try_from(*left).map_or(avail, |left| left.min(avail));
        *left -= n as u64;
        if *left != 0 {
            return (n, false);
        }
        self.stage = Stage::FrameHeader;
        (n, true)
    }

    /// Take the current unit, `unit_len(unit)` bytes, and decode it into
    /// `out`, a frame header starting the frame from `dict` if given. Every
    /// call of a frame passes the `dict` it started from. Not for
    /// `Stage::Skip`.
    fn process(
        &mut self,
        unit: &[u8],
        out: &mut impl FrameOut,
        dict: Option<&DecodeDict>,
    ) -> Result<Event, DecodeError> {
        match &mut self.stage {
            Stage::FrameHeader => self.frame_header(unit, out, dict),
            Stage::Skip { .. } => unreachable!("skippable frame content is skipped, not a unit"),
            Stage::Block { frame, header } => {
                let Some(block) = header.take() else {
                    *header = Some(parse_block_header(unit, frame.block_size_max)?);
                    return Ok(Event::Continue);
                };
                let scratch = self.scratch.get_or_insert_with(DecoderScratch::new);
                // The serial decoder's tables are those the frame's chain
                // left in use.
                #[cfg(feature = "parallel")]
                if let Some(pipeline) = &mut self.pipeline {
                    pipeline.hand_back(scratch)?;
                }
                let dict = dict.and_then(DecodeDict::entropy);
                decode_block(&block, unit, frame, scratch, dict, out, self.simd)?;
                if !block.last_block {
                    return Ok(Event::Continue);
                }
                self.blocks_ended()
            }
            Stage::Checksum { computed } => {
                let stored = u32::from_le_bytes(unit.try_into().unwrap());
                if stored != *computed {
                    return Err(format!(
                        "Content checksum mismatch: frame says {:#010x}, content hashes to {:#010x}",
                        stored, computed
                    ).into());
                }
                self.stage = Stage::FrameHeader;
                Ok(Event::FrameEnded)
            }
        }
    }

    fn frame_header(
        &mut self,
        unit: &[u8],
        out: &mut impl FrameOut,
        dict: Option<&DecodeDict>,
    ) -> Result<Event, DecodeError> {
        let header = match parse_frame_header(unit) {
            Ok((header, _)) => header,
            Err(e) => match e.skip_frame_size() {
                Some(size) => {
                    self.stage = Stage::Skip {
                        left: u64::from(size),
                    };
                    return Ok(Event::Continue);
                }
                None => return Err(frame_header_error(e)),
            },
        };
        let frame = Frame::new(&header)?;
        // ZSTD_decodeFrameHeader's dictionary_wrong check.
        match (header.dict_id, dict) {
            (0, _) => {}
            (id, Some(d)) if id == d.id() => {}
            (id, Some(d)) => {
                return Err(format!(
                    "Frame needs dictionary {id}, dictionary {} is loaded",
                    d.id()
                )
                .into())
            }
            (id, None) => return Err(format!("Frame needs dictionary {id}, none is loaded").into()),
        }
        let scratch = self.scratch.get_or_insert_with(DecoderScratch::new);
        scratch.reset();
        if let Some(e) = dict.and_then(DecodeDict::entropy) {
            scratch.load_dict(e);
        }
        out.start(frame.window, frame.content_size);
        #[cfg(feature = "parallel")]
        debug_assert!(
            self.pipeline.is_none(),
            "the last frame's pipeline ended with it"
        );
        self.stage = Stage::Block {
            frame,
            header: None,
        };
        Ok(Event::FrameStarted)
    }

    /// After the frame's last block: check its size, then expect its
    /// checksum, if it has one.
    fn blocks_ended(&mut self) -> Result<Event, DecodeError> {
        let Stage::Block { frame, .. } = self.leave_frame() else {
            unreachable!("blocks end in Stage::Block");
        };
        if let Some(fcs) = frame.content_size {
            if frame.decoded != fcs {
                return Err(frame.content_size_mismatch(fcs));
            }
        }
        match frame.checksum {
            Some(h) => {
                self.stage = Stage::Checksum {
                    computed: h.digest() as u32,
                };
                Ok(Event::Continue)
            }
            None => Ok(Event::FrameEnded),
        }
    }

    /// The verdict on input that ends `partial` bytes into the current
    /// unit, fewer than `unit_len(partial)`: the bytes the one-shot input
    /// ends with, or those `Decompressor` holds when it finishes. It may
    /// end only between frames.
    fn end_of_input(&self, partial: &[u8]) -> Result<(), DecodeError> {
        Err(match &self.stage {
            Stage::FrameHeader if partial.is_empty() => return Ok(()),
            // ZSTD_decompressMultiFrame: a frame starts wherever at least
            // FRAME_HEADER_PREFIX_LEN bytes remain, and no byte may be left
            // over.
            Stage::FrameHeader if partial.len() < FRAME_HEADER_PREFIX_LEN => format!(
                "Input not entirely consumed: {} bytes left, too few for a frame",
                partial.len()
            )
            .into(),
            Stage::FrameHeader => match parse_frame_header(partial) {
                Err(e) => frame_header_error(e),
                Ok(_) => unreachable!("a whole frame header is a unit"),
            },
            Stage::Skip { .. } => "Skippable frame extends past end of input".into(),
            Stage::Block {
                frame,
                header: None,
            } => match parse_block_header(partial, frame.block_size_max) {
                Err(e) => e,
                Ok(_) => unreachable!("a whole block header is a unit"),
            },
            Stage::Block {
                header: Some(_), ..
            } => BLOCK_CONTENT_TRUNCATED.into(),
            Stage::Checksum { .. } => "Error reading checksum: truncated".into(),
        })
    }
}

/// Block_Header's length.
const BLOCK_HEADER_LEN: usize = 3;

const BLOCK_CONTENT_TRUNCATED: &str = "Block content extends past end of input";

/// Content_Checksum's length.
const CHECKSUM_LEN: usize = 4;

fn frame_header_error(e: FrameDecoderError) -> DecodeError {
    format!("Frame header error: {}", e).into()
}

/// The length of the frame header that starts `head` (ZSTD_frameHeaderSize),
/// as far as `parse_frame_header` reads, or `FRAME_HEADER_PREFIX_LEN` while
/// `head` is shorter than that or does not start a frame.
fn frame_header_len(head: &[u8]) -> usize {
    let Some(&[m0, m1, m2, m3, descriptor]) = head.get(..FRAME_HEADER_PREFIX_LEN) else {
        return FRAME_HEADER_PREFIX_LEN;
    };
    let magic = u32::from_le_bytes([m0, m1, m2, m3]);
    if (0x184D2A50..=0x184D2A5F).contains(&magic) {
        return SKIPPABLE_FRAME_HEADER_LEN;
    }
    if magic != ZSTD_MAGIC {
        return FRAME_HEADER_PREFIX_LEN;
    }
    let d = FrameDescriptor(descriptor);
    FRAME_HEADER_PREFIX_LEN
        + usize::from(!d.single_segment_flag())
        + d.dictionary_id_bytes().unwrap_or(0) as usize
        + d.frame_content_size_bytes().unwrap_or(0) as usize
}

/// The block at the start of `src`, in a frame whose Block_Maximum_Size is
/// `block_size_max`: its header and content, if `src` holds it whole and
/// `parse_block_header` takes its header.
fn locate_block(src: &[u8], block_size_max: usize) -> Option<(BlockHeader, &[u8])> {
    let block = parse_block_header(src.get(..BLOCK_HEADER_LEN)?, block_size_max).ok()?;
    let content = src.get(BLOCK_HEADER_LEN..BLOCK_HEADER_LEN + block.content_size as usize)?;
    Some((block, content))
}

/// Decode `block` with `content` into `out`, a block of `frame`, which
/// started from the tables of `dict` if given.
///
/// Inline in `process` and, with the `parallel` feature, in
/// `decode_serially`: out of line, the call cost a frame of one block
/// about 70 instructions.
#[inline(always)]
fn decode_block(
    block: &BlockHeader,
    content: &[u8],
    frame: &mut Frame,
    scratch: &mut DecoderScratch,
    dict: Option<&DictEntropy>,
    out: &mut impl FrameOut,
    simd: Level,
) -> Result<(), DecodeError> {
    let (dst, ext) = out.block_dst()?;
    // SAFETY: `block_dst` meets the `Dst` and `ExtHistory` contracts. A raw
    // or RLE block decodes to at most `block_size_max <= MAX_BLOCK_SIZE`
    // bytes (`parse_block_header`), within the room `Dst` has, from input
    // apart from it; on success each arm wrote the block up to `end`.
    let bytes = unsafe {
        let at = dst.base.add(dst.op);
        let end = match block.block_type {
            BlockType::Raw => {
                ptr::copy_nonoverlapping(content.as_ptr(), at, content.len());
                dst.op + content.len()
            }
            BlockType::RLE => {
                let len = block.decompressed_size as usize;
                ptr::write_bytes(at, content[0], len);
                dst.op + len
            }
            BlockType::Compressed => {
                decompress_block(content, frame.block_size_max, scratch, dict, dst, ext, simd)?
            }
            BlockType::Reserved => return Err("Reserved block type encountered".into()),
        };
        out.commit(end)
    };
    frame.block_decoded(bytes)
}

/// The one-shot driver's `FrameOut`: frames decode straight into one
/// `Vec`, each block after the last, so a frame's history is all in it.
struct VecOut<'d> {
    output: &'d mut Vec<u8>,
    /// The current frame's.
    prefix: Prefix,
    /// The dictionary content every frame's history starts with, empty
    /// without a dictionary.
    dict: &'d [u8],
}

impl FrameOut for VecOut<'_> {
    fn start(&mut self, window: usize, _content_size: Option<u64>) {
        self.prefix = Prefix {
            start: self.output.len(),
            window,
        };
    }

    fn block_dst(&mut self) -> Result<(Dst, ExtHistory), DecodeError> {
        let dst = self
            .prefix
            .dst(self.output, MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH);
        Ok((dst, ExtHistory::dict(self.dict)))
    }

    unsafe fn commit(&mut self, end: usize) -> &[u8] {
        let start = self.output.len();
        // The bytes up to `end` are written, and `end` is within the room
        // `Prefix::dst` reserved.
        self.output.set_len(self.prefix.start + end);
        &self.output[start..]
    }
}

/// Reserve room in `output` for `frame`, whose blocks start `blocks`, plus
/// what a block's destination reserves past its start, so that no block
/// has to grow the buffer (and move everything decoded). A content size
/// past what the blocks can decode to fails the size check, so it gets no
/// room beyond that, unless the frame fits in one block.
fn reserve_frame(output: &mut Vec<u8>, frame: &Frame, blocks: &[u8]) -> Result<(), DecodeError> {
    let Some(fcs) = frame.content_size else {
        return Ok(());
    };
    let content = if frame.fits_one_block() {
        fcs
    } else {
        fcs.min(blocks_decoded_bound(blocks, frame.block_size_max))
    };
    let want = usize::try_from(content)
        .ok()
        .and_then(|n| n.checked_add(MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH))
        .ok_or_else(|| DecodeError::from(format!("Frame content size {} too large", fcs)))?;
    output
        .try_reserve(want)
        .map_err(|e| format!("Cannot reserve {} bytes of output: {}", want, e).into())
}

/// The most the blocks at the start of `data` decode to: the sum of each
/// raw block's content, RLE block's size and compressed block's
/// `block_size_max`, up to the last block or up to the first one the input
/// does not hold or `parse_block_header` rejects, where decoding fails.
fn blocks_decoded_bound(data: &[u8], block_size_max: usize) -> u64 {
    let mut rest = data;
    let mut bound = 0u64;
    while let Some((block, content)) = locate_block(rest, block_size_max) {
        rest = &rest[BLOCK_HEADER_LEN + content.len()..];
        bound = bound.saturating_add(block_bound(&block, block_size_max) as u64);
        if block.last_block {
            break;
        }
    }
    bound
}

/// The most `block` decodes to, in a frame whose Block_Maximum_Size is
/// `block_size_max`: a raw or RLE block's size, or `block_size_max`.
fn block_bound(block: &BlockHeader, block_size_max: usize) -> usize {
    match block.block_type {
        BlockType::Compressed => block_size_max,
        _ => block.decompressed_size as usize,
    }
}

// ============================================================
// Block decoder
// ============================================================

/// A compressed block's sections, located from their headers
/// (ZSTD_decodeLiteralsBlock's and ZSTD_decodeSeqHeaders' size parsing).
#[derive(Clone, Copy)]
struct BlockParts<'a> {
    literals: LiteralsSection,
    /// The literals section after its header: the raw bytes, the RLE byte,
    /// or the Huffman tree description followed by the streams.
    literals_src: &'a [u8],
    sequences: SequencesHeader,
    /// Everything after the sequences header: the FSE table descriptions,
    /// then the sequence bitstream.
    sequences_src: &'a [u8],
}

/// Locate the sections of compressed block `raw`, which
/// `parse_block_header` has held to `block_size_max`, in a frame whose
/// Block_Maximum_Size that is. The literals decode into the block, so they
/// are held to it too, before anything is sized from their header.
#[inline(always)]
fn split_block(raw: &[u8], block_size_max: usize) -> Result<BlockParts<'_>, DecodeError> {
    let (section, bytes_in_literals_header) = LiteralsSection::parse(raw)?;
    if section.regenerated_size as usize > block_size_max {
        return Err(format!(
            "Literals size {} exceeds Block_Maximum_Size {}",
            section.regenerated_size, block_size_max
        )
        .into());
    }
    let raw = &raw[bytes_in_literals_header..];

    let upper_limit_for_literals = section.content_size as usize;

    if raw.len() < upper_limit_for_literals {
        return Err(format!(
            "Malformed section header: expected {} bytes, have {}",
            upper_limit_for_literals,
            raw.len()
        )
        .into());
    }

    let literals_src = &raw[..upper_limit_for_literals];
    let raw = &raw[upper_limit_for_literals..];

    let (sequences, bytes_in_sequence_header) = SequencesHeader::parse(raw)?;
    Ok(BlockParts {
        literals: section,
        literals_src,
        sequences,
        sequences_src: &raw[bytes_in_sequence_header..],
    })
}

/// Decode the block's literals into `target` (cleared first) and append
/// `WILDCOPY_OVERLENGTH` bytes of slack, in one allocation when `target`
/// has to grow (`split_block` bounded the literals' size). Treeless
/// literals use `repeat` when given, else `huf`'s table.
fn decode_block_literals(
    parts: &BlockParts<'_>,
    huf: &mut HuffmanScratch,
    repeat: Option<&HuffmanTable>,
    target: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    target.clear();
    target.reserve(parts.literals.regenerated_size as usize + WILDCOPY_OVERLENGTH);
    let used = decode_literals(&parts.literals, huf, repeat, parts.literals_src, target)?;
    assert!(
        parts.literals.regenerated_size == target.len() as u32,
        "Wrong number of literals: {}, Should have been: {}",
        target.len(),
        parts.literals.regenerated_size
    );
    assert!(used as usize == parts.literals_src.len());
    target.resize(target.len() + WILDCOPY_OVERLENGTH, 0);
    Ok(())
}

/// Decode compressed block `raw` into `dst`, after `ext`, in a frame whose
/// Block_Maximum_Size is `block_size_max` and which started from the
/// tables of `dict` if given; returns the block's end.
///
/// # Safety
/// `dst` and `ext` meet their contracts.
unsafe fn decompress_block(
    raw: &[u8],
    block_size_max: usize,
    workspace: &mut DecoderScratch,
    dict: Option<&DictEntropy>,
    dst: Dst,
    ext: ExtHistory,
    simd: Level,
) -> Result<usize, DecodeError> {
    let parts = split_block(raw, block_size_max)?;
    let repeat = dict
        .filter(|_| workspace.huf_from_dict)
        .map(|d| &d.huf.table);
    decode_block_literals(
        &parts,
        &mut workspace.huf,
        repeat,
        &mut workspace.literals_buffer,
    )?;
    workspace.huf_from_dict &= !matches!(parts.literals.ls_type, LiteralsSectionType::Compressed);
    let literals_len = workspace.literals_buffer.len() - WILDCOPY_OVERLENGTH;
    let seq_section = parts.sequences;
    let raw = parts.sequences_src;

    if seq_section.num_sequences != 0 {
        let table_bytes = build_sequence_tables(&seq_section, raw, &mut workspace.fse)?;
        let fse = workspace.fse.tables(dict.map(|d| &d.fse));
        let seqs = SeqInput {
            num_sequences: seq_section.num_sequences,
            bit_stream: &raw[table_bytes..],
            fse,
            literals: &workspace.literals_buffer,
        };
        let hist = &mut workspace.offset_hist;
        if ext.len == 0 {
            execute_with_copies(simd, fse[1], seqs, hist, block_size_max, dst)
        } else {
            let seqs = ExtSeqInput { seqs, ext };
            execute_with_copies(simd, fse[1], seqs, hist, block_size_max, dst)
        }
    } else {
        if !raw.is_empty() {
            return Err(format!("Extra bits remaining: {} bits", raw.len() as isize * 8).into());
        }
        // `split_block` held the literals to `block_size_max`, within the
        // room `dst` has.
        let literals = &workspace.literals_buffer[..literals_len];
        ptr::copy_nonoverlapping(literals.as_ptr(), dst.base.add(dst.op), literals.len());
        Ok(dst.op + literals.len())
    }
}

// ============================================================
// Multi-threaded frame decoder (feature `parallel`)
//
// Stage 1 (sequential) locates every block and resolves which earlier
// block defined each Huffman / FSE table a block reuses. Stage 2 (rayon
// tasks, at most a ring's worth of blocks ahead) builds the tables, decodes
// the literals and decodes the sequences with their offsets still in
// OFFBASE form. Stage 3 (one thread, in block order, as soon as each block
// is decoded) resolves the repeat offsets and executes the sequences into
// the output.
//
// Blocks a call has room for all of decode in one rayon scope, from the
// call's input. Others go to the frame's `Pipeline`: detached tasks that
// own copies of what they read, so that those planned past the room keep
// decoding after the call returns, for the next call to take.
// ============================================================

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use std::any::Any;
    use std::collections::VecDeque;
    use std::panic::{self, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};

    /// Batches with fewer compressed blocks decode on the calling thread,
    /// a single block having no other to decode alongside: two blocks of
    /// 32-57 KiB took 11-30% less time on the pool on eight cores sharing
    /// an L3, and 2-20% less on eight over two L3s; three of 34-86 KiB,
    /// 27-46% and 20-37% less (timed as for `MIN_BYTES`).
    pub(super) const MIN_BLOCKS: usize = 2;

    /// Batches whose compressed blocks hold fewer bytes decode on the
    /// calling thread. Timed on 16 different frames of two or more 128 KiB
    /// blocks decoded in turn (one frame decoded over and over lets the
    /// branch predictor learn it): on eight cores sharing an L3, the pool
    /// took less time from 4 KiB up for three or more blocks of 300 B or
    /// more, and from 16 KiB for two. On eight cores over two L3s, blocks
    /// of 1 KiB or more took up to 40% more time below 16 KiB (two of them
    /// 64% more) and less from 20 KiB, though two blocks still took up to
    /// 3% more at 15-29 KiB and less from 31 KiB; blocks of 300 B took up
    /// to 7% more at 19-26 KiB and less from 28 KiB.
    pub(super) const MIN_BYTES: usize = 32 * 1024;

    /// Batches whose compressed blocks hold fewer bytes each, on average,
    /// decode on the calling thread: 163 B, the per-block floor of a gate
    /// of `MIN_BYTES`. Timed as for `MIN_BYTES` on frames of 200-240
    /// blocks of 32-41 KiB: blocks of 147-159 B took 2-3% more time on the
    /// pool on eight cores sharing an L3 and 8% more on eight over two
    /// L3s; blocks of 167-178 B, 5-7% less on one L3 and 2-4% more on two.
    /// Blocks of 186-411 B in frames of 32-50 KiB took 8-19% less on one
    /// L3 and from 5% more to 2% less on two.
    pub(super) const MIN_BLOCK_BYTES: usize = MIN_BYTES / BLOCK_SHARE;

    /// A gate's per-block floor is its `min_bytes` over this, so that the
    /// floor scales with the gate.
    const BLOCK_SHARE: usize = 200;

    /// When a batch of blocks decodes on the rayon pool
    /// (`DecodeOptions::min_parallel_blocks` and `min_parallel_bytes`).
    #[derive(Clone, Copy)]
    pub(super) struct Gate {
        min_blocks: usize,
        min_bytes: usize,
        min_block_bytes: usize,
    }

    impl Gate {
        /// The gate of `opts`; `None` if it takes no batch. Its per-block
        /// floor scales with `min_parallel_bytes`, so that `0` still takes
        /// every batch of `min_parallel_blocks`.
        pub(super) fn new(opts: &DecodeOptions) -> Option<Gate> {
            (opts.min_parallel_blocks != usize::MAX).then_some(Gate {
                min_blocks: opts.min_parallel_blocks,
                min_bytes: opts.min_parallel_bytes,
                min_block_bytes: opts.min_parallel_bytes / BLOCK_SHARE,
            })
        }

        /// Whether the blocks `located` gives carry enough compressed work
        /// for the pool: `min_blocks` compressed blocks of `min_bytes` or
        /// more in all and `min_block_bytes` or more each on average. Only
        /// compressed blocks have a stage 2 to hand it, and that work grows
        /// with their bytes, against the fixed cost of starting tasks and
        /// waiting for them and a cost per block handed: stage 3 executes
        /// every block on one thread either way, so blocks of few bytes
        /// leave the pool that cost with little to share (`MIN_BLOCK_BYTES`).
        fn pools<'a>(self, blocks: impl Iterator<Item = (BlockHeader, &'a [u8], usize)>) -> bool {
            let (mut compressed, mut bytes) = (0, 0);
            for (block, content, _) in blocks {
                if matches!(block.block_type, BlockType::Compressed) {
                    compressed += 1;
                    bytes += content.len();
                }
            }
            compressed >= self.min_blocks
                && bytes >= self.min_bytes
                && bytes >= compressed.saturating_mul(self.min_block_bytes)
        }
    }

    /// The tables and repeat offsets a run of blocks starts from: `init`,
    /// the serial decoder's scratch before its first block, which may
    /// select those of `dict`, the dictionary the frame started from. `id`
    /// stands for them where a plan names the block that defined a table
    /// (`Defs`): a dictionary's, or one a block before the run described.
    #[derive(Clone, Copy)]
    struct FrameStart<'a> {
        id: u64,
        init: &'a DecoderScratch,
        dict: Option<&'a DictEntropy>,
    }

    impl FrameStart<'_> {
        /// The tables in use before the run's first block: `id` for each
        /// table `init` has one of.
        fn defs(self) -> Defs {
            let mut defs = [None; 4];
            defs[0] = (self.init.huf_table(self.dict).max_num_bits != 0).then_some(self.id);
            for t in 0..3 {
                defs[1 + t] = self.init.fse.repeatable(t).then_some(self.id);
            }
            defs
        }
    }

    /// For the Huffman table, then the LL, OF and ML tables (`SEQ_TABLES`
    /// order), the id of the block that defined the one in use, or of the
    /// `FrameStart`; `None` while none is.
    type Defs = [Option<u64>; 4];

    /// One block of a frame, as located by the pre-pass.
    enum Plan<'a> {
        Raw(&'a [u8]),
        Rle(u8, usize),
        Compressed(CompressedPlan<'a>),
    }

    struct CompressedPlan<'a> {
        parts: BlockParts<'a>,
        /// For Huffman-coded literals, the block whose tree description
        /// they use: this block for Compressed literals, the last earlier
        /// block with Compressed literals for Treeless ones, or the start.
        huf_def: Option<u64>,
        /// For LL, OF and ML (`SEQ_TABLES` order), the block whose mode
        /// defined the table this block uses, or the start. Unused without
        /// sequences.
        fse_def: [u64; 3],
    }

    /// The blocks at the start of `data` that it holds whole and whose
    /// headers parse, each with where it ends in `data`, up to the frame's
    /// last; past the first, only while the blocks before may decode to
    /// `room` bytes at most (`block_bound`).
    fn located(
        data: &[u8],
        block_size_max: usize,
        room: usize,
    ) -> impl Iterator<Item = (BlockHeader, &[u8], usize)> {
        let (mut pos, mut bound, mut ended) = (0, 0usize, false);
        std::iter::from_fn(move || {
            if ended || bound > room {
                return None;
            }
            let (block, content) = locate_block(&data[pos..], block_size_max)?;
            pos += BLOCK_HEADER_LEN + content.len();
            bound = bound.saturating_add(block_bound(&block, block_size_max));
            ended = block.last_block;
            Some((block, content, pos))
        })
    }

    /// The blocks `plan_blocks` takes: their plans, where each ends in the
    /// input, whether the last is the frame's last, and the id of the
    /// first, the others' following.
    struct Batch<'a> {
        plans: Vec<Plan<'a>>,
        ends: Vec<usize>,
        last: bool,
        base: u64,
    }

    impl<'a> Batch<'a> {
        /// The sections of compressed block `d` of the batch.
        fn def(&self, d: u64) -> Option<&BlockParts<'a>> {
            match self
                .plans
                .get(usize::try_from(d.checked_sub(self.base)?).ok()?)?
            {
                Plan::Compressed(c) => Some(&c.parts),
                _ => None,
            }
        }
    }

    /// Stage 1: the block loop of ZSTD_decompressFrame over the blocks
    /// `located` gives, resolving Treeless / Repeat references the way the
    /// serial decoder's scratch tables carry them from block to block, from
    /// the tables in `start`, naming each block by the next of `ids`. It
    /// stops before a block it cannot plan, which the serial decoder then
    /// decodes, giving its verdict on it.
    fn plan_blocks<'a>(
        data: &'a [u8],
        block_size_max: usize,
        start: FrameStart<'_>,
        room: usize,
        ids: &mut u64,
    ) -> Batch<'a> {
        let mut batch = Batch {
            plans: Vec::new(),
            ends: Vec::new(),
            last: false,
            base: *ids,
        };
        let mut defs = start.defs();
        for (block, content, end) in located(data, block_size_max, room) {
            let i = *ids;
            let plan = match block.block_type {
                BlockType::Raw => Plan::Raw(content),
                BlockType::RLE => Plan::Rle(content[0], block.decompressed_size as usize),
                BlockType::Compressed => {
                    let Some(plan) = plan_compressed(content, block_size_max, i, &mut defs) else {
                        break;
                    };
                    Plan::Compressed(plan)
                }
                BlockType::Reserved => break,
            };
            *ids += 1;
            batch.plans.push(plan);
            batch.ends.push(end);
            batch.last = block.last_block;
        }
        batch
    }

    /// Plan compressed block `id`, of `content`, `defs` being the tables
    /// in use before it, which it updates; `None`, leaving them, if the
    /// block fails before decoding.
    fn plan_compressed<'a>(
        content: &'a [u8],
        block_size_max: usize,
        id: u64,
        defs: &mut Defs,
    ) -> Option<CompressedPlan<'a>> {
        let parts = split_block(content, block_size_max).ok()?;
        let mut after = *defs;
        let huf = match parts.literals.ls_type {
            LiteralsSectionType::Compressed => {
                after[0] = Some(id);
                after[0]
            }
            LiteralsSectionType::Treeless => Some(after[0]?),
            LiteralsSectionType::Raw | LiteralsSectionType::RLE => None,
        };
        let mut fse = [0; 3];
        if parts.sequences.num_sequences != 0 {
            for (t, mode) in parts.sequences.modes.all().into_iter().enumerate() {
                if !matches!(mode, ModeType::Repeat) {
                    after[1 + t] = Some(id);
                }
                fse[t] = after[1 + t]?;
            }
        }
        *defs = after;
        Some(CompressedPlan {
            parts,
            huf_def: huf,
            fse_def: fse,
        })
    }

    /// One sequence as stage 2 publishes it: its offset is relative to the
    /// repeat offsets its block starts with (`RelReps`), which only the
    /// executing thread knows, and which it resolves (`RelBase`).
    ///
    /// Stage 2 resolves the repeat codes, not the executing thread: there
    /// the branches on them had no history of the sequence's decode to
    /// predict them by, and the executing thread alone mispredicted 2.3x
    /// as often as the serial loop's decode and execution together.
    #[derive(Clone, Copy)]
    struct RelSeq {
        /// The literal length, below 2^17, and the offset's `from` in the
        /// top two bits.
        ll_from: u32,
        ml: u32,
        off: u32,
    }

    /// `RelSeq::ll_from`'s literal length.
    const LL_MASK: u32 = (1 << 30) - 1;

    impl RelSeq {
        /// Lengths below 2^17, `o` a `RelReps` entry.
        #[inline(always)]
        fn new(ll: u32, ml: u32, o: u64) -> RelSeq {
            RelSeq {
                ll_from: ll | ((o >> 32) as u32) << 30,
                ml,
                off: o as u32,
            }
        }
    }

    /// `RelReps`' `from` of an offset the block's sequences give.
    const REL_NONE: u64 = 3 << 32;

    /// What a `RelReps` entry adds to a starting repeat offset: those only
    /// ever decrease, by one per sequence at most, so from here they never
    /// borrow into `from`.
    const REL_BIAS: u32 = 1 << 31;

    /// Stage 2's repeat offsets (`decode_sequence`'s), each `from << 32 |
    /// off`: for `from` below 3 the block's starting repeat offset `from`
    /// plus `off - REL_BIAS`, for `REL_NONE` the offset `off`.
    #[derive(Clone, Copy)]
    struct RelReps([u64; 3]);

    impl RelReps {
        /// The block's starting repeat offsets.
        fn start() -> RelReps {
            RelReps([0, 1, 2].map(|k| k << 32 | u64::from(REL_BIAS)))
        }

        /// Offset code 2 or more: the new offset `offset`, below 2^32.
        #[inline(always)]
        fn new_offset(&mut self, offset: u64) -> u64 {
            let h = &mut self.0;
            let o = REL_NONE | offset;
            *h = [o, h[0], h[1]];
            o
        }

        /// Offset code 0: the first repeat offset, or with `ll0` (no
        /// literals) the second.
        #[inline(always)]
        fn code0(&mut self, ll0: usize) -> u64 {
            let h = &mut self.0;
            let o = h[ll0];
            h[1] = h[1 - ll0];
            h[0] = o;
            o
        }

        /// Offset code 1: repeat offset `o` in 1..=3, 3 being the first
        /// minus one. An offset of 0 there (corrupt input) wraps into
        /// `from` on the next decrement, after the sequence that gave it
        /// failed to execute.
        #[inline(always)]
        fn code1(&mut self, o: usize) -> u64 {
            let h = &mut self.0;
            let temp = if o == 3 { h[0].wrapping_sub(1) } else { h[o] };
            if o != 1 {
                h[2] = h[1];
            }
            h[1] = h[0];
            h[0] = temp;
            temp
        }
    }

    /// The repeat offsets a block starts with, to resolve its sequences'
    /// `RelReps` entries against.
    struct RelBase([u32; 4]);

    impl RelBase {
        fn new(hist: [u32; 3]) -> RelBase {
            let [a, b, c] = hist.map(|h| h.wrapping_sub(REL_BIAS));
            RelBase([a, b, c, 0])
        }

        /// The offset `from` and `off` give.
        #[inline(always)]
        fn resolve(&self, from: u32, off: u32) -> u32 {
            self.0[from as usize & 3].wrapping_add(off)
        }

        /// A sequence's offset, 0 (corrupt input: code 3 from a first
        /// offset of 1) forced to one execution rejects.
        #[inline(always)]
        fn offset(&self, s: &RelSeq) -> usize {
            match self.resolve(s.ll_from >> 30, s.off) {
                0 => usize::MAX,
                o => o as usize,
            }
        }

        /// The repeat offsets `end` stands for.
        fn hist(&self, end: RelReps) -> [u32; 3] {
            end.0.map(|e| self.resolve((e >> 32) as u32, e as u32))
        }
    }

    /// Where the stage 2 of one compressed block after another decodes to,
    /// each block named by a ticket: the ticket of a block it is handed is
    /// planned (`hand`), then taken by whoever decodes it first, a pool task
    /// or the executing thread (`take`), then done once its decode is in
    /// `slot` (`decoded`). A task whose block someone else took, or whose
    /// cell has since been handed another block, finds a ticket it cannot
    /// take and returns. The decode does the block's literals, then its
    /// sequences; for the block the executing thread executes next, the
    /// sequences first, then the literals unless that thread has taken
    /// those meanwhile (`literals`). While the decode runs, the executing
    /// thread reads only what it has published (`Published`), the literals
    /// once they are done, and the slot once it is done.
    struct Cell {
        /// `2 * ticket` while the block of `ticket` is planned, one more
        /// once it is taken.
        claim: AtomicUsize,
        /// `2 * ticket + 1` once the block of `ticket` is decoded.
        done: AtomicUsize,
        /// `2 * ticket` while the literals of the block of `ticket` are to
        /// decode, one more once someone has taken them.
        lit_claim: AtomicUsize,
        /// `2 * ticket + 1` once the literals of the block of `ticket` are
        /// decoded.
        lit_done: AtomicUsize,
        /// Whether the decode of the block last handed does its sequences
        /// first.
        sequences_first: AtomicBool,
        /// What the decode of the block of the ticket last handed has
        /// published.
        published: Published,
        slot: Mutex<Slot>,
        lits: Mutex<LitSlot>,
    }

    impl Cell {
        fn new() -> Cell {
            let DecoderScratch { huf, fse, .. } = DecoderScratch::new();
            Cell {
                // Taken, of no block: nothing is to decode yet.
                claim: AtomicUsize::new(1),
                done: AtomicUsize::new(0),
                lit_claim: AtomicUsize::new(1),
                lit_done: AtomicUsize::new(0),
                sequences_first: AtomicBool::new(false),
                published: Published::new(),
                slot: Mutex::new(Slot::new(fse)),
                lits: Mutex::new(LitSlot::new(huf)),
            }
        }

        /// Plan the block of `ticket`, once the cell's block before it has
        /// been executed or given up and no task decodes into the cell;
        /// `next` if the executing thread executes it next, so its decode
        /// does the sequences first, the literals being that thread's to
        /// decode meanwhile.
        fn hand(&self, ticket: usize, next: bool) {
            // Before the decode by the claim's release, and before the
            // executing thread's loads: the planning thread is that thread,
            // or one whatever moved the decoder there synchronized with.
            self.published.progress.store(0, Ordering::Relaxed);
            self.sequences_first.store(next, Ordering::Relaxed);
            self.lit_claim
                .store(ticket.wrapping_mul(2), Ordering::Relaxed);
            self.claim.store(ticket.wrapping_mul(2), Ordering::Release);
        }

        /// Decode the literals of the block of `ticket` into the cell with
        /// `decode`, unless someone has taken them, and return them, locked.
        /// The executing thread takes them when it comes to the block before
        /// the block's decode has, which, for the block `hand` expects that
        /// thread to execute next, decodes the sequences first. A panic of
        /// `decode` leaves them failed.
        fn literals(
            &self,
            ticket: usize,
            decode: impl FnOnce(&mut LitSlot) -> Result<(), DecodeError>,
        ) -> Option<MutexGuard<'_, LitSlot>> {
            let planned = ticket.wrapping_mul(2);
            self.lit_claim
                .compare_exchange(planned, planned | 1, Ordering::AcqRel, Ordering::Relaxed)
                .ok()?;
            let _done = MarkDone(&self.lit_done, planned | 1);
            let mut lits = lock(&self.lits);
            match panic::catch_unwind(AssertUnwindSafe(|| decode(&mut lits))) {
                Ok(result) => lits.result = result,
                Err(payload) => {
                    lits.result = Err("Literals decode panicked".into());
                    drop(lits);
                    panic::resume_unwind(payload);
                }
            }
            Some(lits)
        }

        /// Take the decode of the block of `ticket`; false if someone has,
        /// or the cell holds another block.
        fn take(&self, ticket: usize) -> bool {
            let planned = ticket.wrapping_mul(2);
            self.claim
                .compare_exchange(planned, planned | 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        }

        fn decoded(&self, ticket: usize) -> bool {
            self.done.load(Ordering::Acquire) == ticket.wrapping_mul(2) | 1
        }

        /// Whether the executing thread can start on the block of
        /// `ticket`: its decode is done, or has published and its literals
        /// are done.
        fn startable(&self, ticket: usize) -> bool {
            self.decoded(ticket)
                || (self.published.progress.load(Ordering::Acquire) != 0
                    && self.lit_done.load(Ordering::Acquire) == ticket.wrapping_mul(2) | 1)
        }

        /// The literals of the block of `ticket`, which someone has taken,
        /// once they are done.
        fn decoded_literals(&self, ticket: usize) -> MutexGuard<'_, LitSlot> {
            let mut spins = 0;
            while self.lit_done.load(Ordering::Acquire) != ticket.wrapping_mul(2) | 1 {
                pause(&mut spins);
            }
            lock(&self.lits)
        }

        /// The slot of the block of `ticket` once its decode is done: its
        /// stage 2, or the panic of its decode resumed.
        fn finished(&self, ticket: usize) -> MutexGuard<'_, Slot> {
            let mut spins = 0;
            while !self.decoded(ticket) {
                pause(&mut spins);
            }
            decoded_slot(self)
        }

        /// Decode the block of `ticket`, taken, into the slot with `decode`,
        /// which publishes into the cell's `Published` as it goes, and the
        /// literals unless taken, then mark it done, on a panic too, which
        /// the slot keeps for the executing thread to resume.
        fn decode(
            &self,
            ticket: usize,
            decode: impl FnOnce(&mut Slot, Decoding<'_>) -> Result<(), DecodeError>,
        ) {
            let _done = MarkDone(&self.done, ticket.wrapping_mul(2) | 1);
            let mut slot = lock(&self.slot);
            let slot = &mut *slot;
            let at = Decoding { cell: self, ticket };
            let decoded = panic::catch_unwind(AssertUnwindSafe(|| decode(&mut *slot, at)));
            slot.result = match decoded {
                Ok(result) => result,
                Err(payload) => {
                    slot.panic = Some(payload);
                    Err("Block decode panicked".into())
                }
            };
        }
    }

    /// The decode of the block of `ticket` in `cell`, taken: where it
    /// publishes the sequences, and the claim on the literals.
    #[derive(Clone, Copy)]
    struct Decoding<'c> {
        cell: &'c Cell,
        ticket: usize,
    }

    impl Decoding<'_> {
        fn published(&self) -> &Published {
            &self.cell.published
        }

        /// Decode the block's literals with `literals`, unless the
        /// executing thread has taken them, and its sequences with
        /// `sequences`, in the order `hand` set.
        fn stage2(
            &self,
            sequences: impl FnOnce() -> Result<(), DecodeError>,
            literals: impl FnOnce(&mut LitSlot) -> Result<(), DecodeError>,
        ) -> Result<(), DecodeError> {
            if self.cell.sequences_first.load(Ordering::Relaxed) {
                sequences()?;
                drop(self.cell.literals(self.ticket, literals));
            } else {
                drop(self.cell.literals(self.ticket, literals));
                sequences()?;
            }
            Ok(())
        }
    }

    /// Wait a little for another thread: spin, then hand the CPU to a
    /// worker the kernel may have queued on it.
    fn pause(spins: &mut u32) {
        if *spins < 64 {
            *spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }

    /// What the decode of a cell's block makes readable before it ends, so
    /// that the executing thread executes the block's sequences while the
    /// rest decode: the first `progress` of them, at `seqs`, and with the
    /// first whether the block's offsets are short
    /// (`SHORT_OFFSET_SHARE_MIN`), and with the last the repeat offsets
    /// after them (`end`). The decode writes none of those again, nor
    /// moves them, and they stay until the cell is handed another block,
    /// which waits for the executing thread to be done with this one.
    struct Published {
        progress: AtomicUsize,
        seqs: AtomicPtr<RelSeq>,
        short: AtomicBool,
        /// `end`'s offsets, and its `from`s two bits each.
        end: [AtomicU32; 3],
        end_from: AtomicU32,
    }

    impl Published {
        fn new() -> Published {
            Published {
                progress: AtomicUsize::new(0),
                seqs: AtomicPtr::new(ptr::null_mut()),
                short: AtomicBool::new(false),
                end: [const { AtomicU32::new(0) }; 3],
                end_from: AtomicU32::new(0),
            }
        }

        /// Where the block's sequences go, and whether its offsets are
        /// short, to publish with the first.
        fn start(&self, seqs: *mut RelSeq, short: bool) {
            self.seqs.store(seqs, Ordering::Relaxed);
            self.short.store(short, Ordering::Relaxed);
        }

        /// Publish the first `n` sequences, decoded, `n` above 0.
        fn decoded(&self, n: usize) {
            self.progress.store(n, Ordering::Release);
        }

        /// Publish the last of the block's `n` sequences, and `end`, the
        /// repeat offsets after them.
        fn decoded_all(&self, n: usize, end: RelReps) {
            let mut from = 0;
            for (k, (e, r)) in self.end.iter().zip(end.0).enumerate() {
                e.store(r as u32, Ordering::Relaxed);
                from |= ((r >> 32) as u32) << (2 * k);
            }
            self.end_from.store(from, Ordering::Relaxed);
            self.decoded(n);
        }

        /// The repeat offsets after the block's sequences, once all are
        /// published.
        fn end(&self) -> RelReps {
            let from = self.end_from.load(Ordering::Relaxed);
            RelReps([0, 1, 2].map(|k| {
                u64::from(from >> (2 * k) & 3) << 32
                    | u64::from(self.end[k].load(Ordering::Relaxed))
            }))
        }
    }

    /// A cell's slot or literals. A decode that panics leaves its payload
    /// in the slot, or its literals failed, rather than poisoning the lock.
    fn lock<T>(slot: &Mutex<T>) -> MutexGuard<'_, T> {
        slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The slot of a block that the executing thread is to execute, its
    /// decode done: its stage 2, or the panic of its decode resumed.
    fn decoded_slot(cell: &Cell) -> MutexGuard<'_, Slot> {
        let mut slot = lock(&cell.slot);
        if let Some(payload) = slot.panic.take() {
            drop(slot);
            panic::resume_unwind(payload);
        }
        slot
    }

    /// Publishes a finished decode on drop.
    struct MarkDone<'a>(&'a AtomicUsize, usize);

    impl Drop for MarkDone<'_> {
        fn drop(&mut self) {
            self.0.store(self.1, Ordering::Release);
        }
    }

    /// Per-cell state of the decode of a block's sequences: tables tagged
    /// with the id of the block that defined them (so a run of blocks
    /// reusing one table builds it once; ids are never reused within the
    /// frame), and the decoded sequences of the current block.
    struct Slot {
        fse: FSEScratch,
        fse_from: [Option<u64>; 3],
        /// Room for the block's sequences, which stage 2 writes past the
        /// length and publishes (`Published`).
        seqs: Vec<RelSeq>,
        result: Result<(), DecodeError>,
        /// The detached decode of the block handed to the cell, until it
        /// is taken.
        job: Option<Arc<Job>>,
        /// What the last decode panicked with.
        panic: Option<Box<dyn Any + Send>>,
    }

    impl Slot {
        fn new(fse: FSEScratch) -> Slot {
            Slot {
                fse,
                fse_from: [None; 3],
                seqs: Vec::new(),
                result: Ok(()),
                job: None,
                panic: None,
            }
        }
    }

    /// Per-cell state of the decode of a block's literals, by whoever took
    /// them (`Cell::literals`): the Huffman table, tagged as `Slot`'s, the
    /// literals followed by `WILDCOPY_OVERLENGTH` bytes of slack, and how
    /// their decode ended.
    struct LitSlot {
        huf: HuffmanScratch,
        huf_from: Option<u64>,
        literals: Vec<u8>,
        result: Result<(), DecodeError>,
    }

    impl LitSlot {
        fn new(huf: HuffmanScratch) -> LitSlot {
            LitSlot {
                huf,
                huf_from: None,
                literals: Vec::new(),
                result: Ok(()),
            }
        }
    }

    /// The tables and repeat offsets a chain starts from: a copy of those
    /// of the serial decoder's scratch when it started, with the dictionary
    /// the frame started from.
    struct StartTables {
        id: u64,
        init: DecoderScratch,
        dict: Option<DecodeDict>,
        block_size_max: usize,
    }

    impl StartTables {
        fn view(&self) -> FrameStart<'_> {
            FrameStart {
                id: self.id,
                init: &self.init,
                dict: self.dict.as_ref().and_then(DecodeDict::entropy),
            }
        }
    }

    impl DecoderScratch {
        /// A copy of the tables and repeat offsets.
        fn copy_tables(&self) -> DecoderScratch {
            let mut huf = HuffmanTable::new();
            huf.copy_from(&self.huf.table);
            DecoderScratch {
                huf: HuffmanScratch { table: huf },
                fse: FSEScratch {
                    offsets: self.fse.offsets.clone(),
                    literal_lengths: self.fse.literal_lengths.clone(),
                    match_lengths: self.fse.match_lengths.clone(),
                    source: self.fse.source,
                },
                offset_hist: self.offset_hist,
                literals_buffer: Vec::new(),
                huf_from_dict: self.huf_from_dict,
            }
        }
    }

    /// The stage 2 of a compressed block of a chain, for a pool task: it
    /// owns all it reads.
    struct Job {
        id: u64,
        content: Arc<[u8]>,
        huf_def: Option<u64>,
        fse_def: [u64; 3],
        /// For the Huffman, then LL, OF, ML tables, the earlier block of
        /// the chain that described the one the block uses, and its
        /// content; `None` for one the block describes or the chain
        /// started from, or does not use.
        defs: [Option<(u64, Arc<[u8]>)>; 4],
        start: Arc<StartTables>,
    }

    /// The sections of the blocks a block takes table descriptions from.
    type Described<'a> = [Option<(u64, BlockParts<'a>)>; 4];

    /// The section of block `d` among `defs`.
    fn def_in<'p, 'a>(defs: &'p Described<'a>) -> impl Fn(u64) -> Option<&'p BlockParts<'a>> {
        move |d| {
            defs.iter()
                .flatten()
                .find(|(id, _)| *id == d)
                .map(|(_, p)| p)
        }
    }

    impl Job {
        /// The block's plan, and the sections of the blocks whose table
        /// descriptions it uses.
        fn split(&self) -> Result<(CompressedPlan<'_>, Described<'_>), DecodeError> {
            let block_size_max = self.start.block_size_max;
            let plan = CompressedPlan {
                parts: split_block(&self.content, block_size_max)?,
                huf_def: self.huf_def,
                fse_def: self.fse_def,
            };
            let defs = self.defs.each_ref().map(|d| {
                let (id, content) = d.as_ref()?;
                Some((*id, split_block(content, block_size_max).ok()?))
            });
            Ok((plan, defs))
        }

        /// Stage 2 into `slot`, as `at` publishes it.
        fn decode(&self, slot: &mut Slot, at: Decoding<'_>) -> Result<(), DecodeError> {
            let (plan, defs) = self.split()?;
            decode_block(slot, at, self.id, &plan, def_in(&defs), self.start.view())
        }

        /// The block's literals into `lits`.
        fn literals(&self, lits: &mut LitSlot) -> Result<(), DecodeError> {
            let (plan, defs) = self.split()?;
            decode_literals(lits, self.id, &plan, def_in(&defs), self.start.view())
        }
    }

    /// A pool task: the decode of the block of `ticket`, handed to `cell`,
    /// unless someone has taken it.
    fn run_job(cell: &Cell, ticket: usize) {
        if cell.take(ticket) {
            run_taken(cell, ticket);
        }
    }

    /// The decode of the block of `ticket`, taken, from the job its cell
    /// holds.
    fn run_taken(cell: &Cell, ticket: usize) {
        cell.decode(ticket, |slot, at| match slot.job.take() {
            Some(job) => job.decode(slot, at),
            None => Err("Block decode without a job".into()),
        });
    }

    /// The tables in use after a block of a chain: their `Defs`, and the
    /// content of each block of the chain among them.
    #[derive(Clone)]
    struct ChainDefs {
        ids: Defs,
        blocks: [Option<Arc<[u8]>>; 4],
    }

    /// A block a chain planned and has not executed.
    struct Planned {
        /// Its Block_Header.
        block: BlockHeader,
        /// A compressed block's stage 2 and the cell it decodes into.
        job: Option<(Arc<Job>, Arc<Cell>)>,
        /// The tables in use before it.
        before: ChainDefs,
        /// The last call of the pipeline whose input held it.
        seen: u64,
    }

    impl Planned {
        /// Its bytes in the input, header included.
        fn len(&self) -> usize {
            BLOCK_HEADER_LEN + self.block.content_size as usize
        }

        /// Whether `block`, with `content`, is the block: the same header,
        /// which a block's header bytes are a function of, and a compressed
        /// block's content. A raw or RLE block's content is read from the
        /// input it is executed from.
        fn holds(&self, block: &BlockHeader, content: &[u8]) -> bool {
            *block == self.block
                && self
                    .job
                    .as_ref()
                    .is_none_or(|(job, _)| content == &job.content[..])
        }
    }

    /// A call's input to a chain: the block whose header the serial
    /// decoder took, with its content, if the frame is at one, then whole
    /// blocks from a header on. Its positions run through that block's
    /// header and content, then `data`.
    #[derive(Clone, Copy)]
    struct Input<'a> {
        held: Option<(BlockHeader, &'a [u8])>,
        data: &'a [u8],
    }

    impl<'a> Input<'a> {
        /// Where `data` starts.
        fn data_start(&self) -> usize {
            self.held
                .map_or(0, |(_, content)| BLOCK_HEADER_LEN + content.len())
        }

        /// The block at `at`, a block's start: the held one, or one `data`
        /// holds whole (`locate_block`).
        fn block(&self, at: usize, block_size_max: usize) -> Option<(BlockHeader, &'a [u8])> {
            match self.held {
                Some(held) if at == 0 => Some(held),
                _ => locate_block(self.data.get(at - self.data_start()..)?, block_size_max),
            }
        }
    }

    /// The frame's blocks planned from one start, the serial decoder's
    /// scratch then: those executed, through their repeat offsets and the
    /// tables they leave in use, and those planned after them.
    struct Chain {
        start: Arc<StartTables>,
        /// The repeat offsets after the blocks executed.
        hist: [u32; 3],
        /// The tables in use after the blocks planned.
        defs: ChainDefs,
        /// The blocks planned and not executed, the frame's next block
        /// first.
        queue: VecDeque<Planned>,
        /// Their bytes in the input.
        queued_len: usize,
    }

    /// The frame's stage 2 on detached pool tasks: a chain of blocks, kept
    /// from call to call, and the cells their tasks decode into.
    ///
    /// Invariant: a task reads only what its `Job` owns, copies of its
    /// block, of the earlier blocks whose table descriptions it uses and of
    /// the tables and repeat offsets the chain started from; nothing of
    /// the frame, the decoder, its dictionary or a caller's input. It
    /// writes only its cell's slot, the cell's literals unless the
    /// executing thread has taken them, and what it publishes there
    /// (`Published`), so it may outlive the call that planned it, and the
    /// frame. Only `run`, on the thread that executes the frame's blocks,
    /// applies a task's result: in plan order, to a block whose bytes the
    /// call's input holds where it executes it, and only while the frame
    /// has decoded no block outside the chain since it started; the serial
    /// decoder's `hand_back` ends the chain before it decodes one. It
    /// decodes a block's literals itself if it comes to the block before
    /// the task has taken them, executes its sequences as the task
    /// publishes them, and keeps the block, and its repeat offsets, once
    /// the task and the literals have ended without an error. A block
    /// leaves the chain only by being executed or through `retire`, on
    /// every other way out: input that differs (`rewind`), a block decoded
    /// outside the chain (`hand_back`), the frame's end or reset or drop
    /// (`Drop`). `retire` cancels a decode no task has started; one running
    /// finishes into its cell, which is handed a block planned after only
    /// once no task holds it (`idle_cell`).
    #[derive(Default)]
    pub(super) struct Pipeline {
        /// The next block or start id: none is used twice in the frame.
        next_id: u64,
        /// The calls of `run` so far.
        calls: u64,
        /// The chain running, if any.
        chain: Option<Box<Chain>>,
        /// Cells no block of the chain has, the pool's tasks perhaps still
        /// holding some.
        free: Vec<Arc<Cell>>,
        /// The cells allocated.
        cells: usize,
    }

    impl Drop for Pipeline {
        fn drop(&mut self) {
            self.outdate();
        }
    }

    /// Give up a planned block: take its decode if no task has, so that
    /// the task returns at once, and free its cell.
    fn retire(p: Planned, free: &mut Vec<Arc<Cell>>) {
        if let Some((job, cell)) = p.job {
            if cell.take(job.id as usize) {
                lock(&cell.slot).job = None;
            }
            free.push(cell);
        }
    }

    /// A free cell that no task holds, most recently freed first, or a new
    /// one while fewer than `cap` are allocated. `Arc::get_mut` sees every
    /// write of the tasks that held it.
    fn idle_cell(free: &mut Vec<Arc<Cell>>, cells: &mut usize, cap: usize) -> Option<Arc<Cell>> {
        if let Some(k) = free.iter_mut().rposition(|c| Arc::get_mut(c).is_some()) {
            return Some(free.swap_remove(k));
        }
        (*cells < cap).then(|| {
            *cells += 1;
            Arc::new(Cell::new())
        })
    }

    /// For the Huffman, then LL, OF, ML tables compressed block `plan`
    /// (`id`) uses, the earlier block of the chain that described it, from
    /// the tables in use before it, `before`.
    fn job_defs(
        plan: &CompressedPlan<'_>,
        id: u64,
        start: u64,
        before: &ChainDefs,
    ) -> [Option<(u64, Arc<[u8]>)>; 4] {
        let mut used = [plan.huf_def, None, None, None];
        if plan.parts.sequences.num_sequences != 0 {
            used[1..].copy_from_slice(&plan.fse_def.map(Some));
        }
        std::array::from_fn(|k| {
            let d = used[k].filter(|&d| d != id && d != start)?;
            Some((d, before.blocks[k].clone()?))
        })
    }

    impl Pipeline {
        fn new_id(&mut self) -> u64 {
            self.next_id += 1;
            self.next_id - 1
        }

        /// Whether a chain is running: until it ends, every block of the
        /// frame decoded in parallel is one of its.
        pub(super) fn active(&self) -> bool {
            self.chain.is_some()
        }

        /// `len` cells for a batch on the pool, idle ones first; given back
        /// to `free` after it.
        fn ring(&mut self, len: usize) -> Vec<Arc<Cell>> {
            (0..len)
                .map(|_| {
                    idle_cell(&mut self.free, &mut self.cells, usize::MAX)
                        .expect("cells are allocated without bound")
                })
                .collect()
        }

        /// Start a chain from `scratch`, the serial decoder's, in a frame
        /// that started from `dict` if given.
        fn start(
            &mut self,
            scratch: &DecoderScratch,
            dict: Option<&DecodeDict>,
            block_size_max: usize,
        ) {
            let start = Arc::new(StartTables {
                id: self.new_id(),
                init: scratch.copy_tables(),
                dict: dict.cloned(),
                block_size_max,
            });
            self.chain = Some(Box::new(Chain {
                hist: scratch.offset_hist,
                defs: ChainDefs {
                    ids: start.view().defs(),
                    blocks: Default::default(),
                },
                queue: VecDeque::new(),
                queued_len: 0,
                start,
            }));
        }

        /// End the chain: give up every block planned.
        fn outdate(&mut self) {
            if let Some(chain) = self.chain.take() {
                for p in chain.queue {
                    retire(p, &mut self.free);
                }
            }
        }

        /// Before the serial decoder decodes a block of the frame: end the
        /// chain, leaving `scratch` as the serial decoder would have left
        /// it after the blocks the chain executed.
        #[inline]
        pub(super) fn hand_back(
            &mut self,
            scratch: &mut DecoderScratch,
        ) -> Result<(), DecodeError> {
            match self.chain {
                Some(_) => self.hand_back_chain(scratch),
                None => Ok(()),
            }
        }

        #[cold]
        #[inline(never)]
        fn hand_back_chain(&mut self, scratch: &mut DecoderScratch) -> Result<(), DecodeError> {
            let Some(chain) = &self.chain else {
                return Ok(());
            };
            let defs = chain.queue.front().map_or(&chain.defs, |p| &p.before);
            let block_size_max = chain.start.block_size_max;
            let parts = defs
                .blocks
                .each_ref()
                .map(|b| split_block(b.as_deref()?, block_size_max).ok());
            let def = |d| {
                let k = defs.ids.iter().position(|&id| id == Some(d))?;
                parts[k].as_ref()
            };
            // The chain started from `scratch`'s tables, which no block has
            // changed since.
            let synced = sync_scratch(scratch, chain.hist, defs.ids, chain.start.id, def);
            self.outdate();
            synced
        }

        /// The input differs from the next block's: give up the queue, to
        /// plan again from the tables in use before that block.
        fn rewind(&mut self) {
            let Some(chain) = &mut self.chain else {
                return;
            };
            if let Some(front) = chain.queue.front() {
                chain.defs = front.before.clone();
            }
            chain.queued_len = 0;
            for p in chain.queue.drain(..) {
                retire(p, &mut self.free);
            }
        }

        /// Plan the blocks of `input` after the queue, the first at `pos`,
        /// until `depth` are queued, handing each compressed one to a pool
        /// task; it stops past the frame's last block, at a block `input`
        /// does not hold whole, at one with no idle cell for, and at one it
        /// cannot plan, setting `blocked`, which the serial decoder then
        /// decodes, giving its verdict on it.
        fn fill(&mut self, input: Input<'_>, pos: usize, depth: usize, blocked: &mut bool) {
            let call = self.calls;
            let Pipeline {
                next_id,
                chain: Some(chain),
                free,
                cells,
                ..
            } = self
            else {
                return;
            };
            let block_size_max = chain.start.block_size_max;
            while !*blocked
                && chain.queue.len() < depth
                && !chain.queue.back().is_some_and(|p| p.block.last_block)
            {
                let Some((block, content)) = input.block(pos + chain.queued_len, block_size_max)
                else {
                    return;
                };
                let before = chain.defs.clone();
                let job = match block.block_type {
                    BlockType::Raw | BlockType::RLE => None,
                    BlockType::Compressed => {
                        let mut ids = before.ids;
                        let Some(plan) =
                            plan_compressed(content, block_size_max, *next_id, &mut ids)
                        else {
                            *blocked = true;
                            return;
                        };
                        let Some(cell) = idle_cell(free, cells, 2 * depth) else {
                            return;
                        };
                        let id = *next_id;
                        *next_id += 1;
                        let content: Arc<[u8]> = content.into();
                        let job = Arc::new(Job {
                            id,
                            huf_def: plan.huf_def,
                            fse_def: plan.fse_def,
                            defs: job_defs(&plan, id, chain.start.id, &before),
                            content: content.clone(),
                            start: chain.start.clone(),
                        });
                        for (k, d) in ids.iter().enumerate() {
                            if *d == Some(id) {
                                chain.defs.blocks[k] = Some(content.clone());
                            }
                        }
                        chain.defs.ids = ids;
                        lock(&cell.slot).job = Some(job.clone());
                        cell.hand(id as usize, chain.queue.is_empty());
                        let task = cell.clone();
                        rayon::spawn_fifo(move || run_job(&task, id as usize));
                        Some((job, cell))
                    }
                    BlockType::Reserved => {
                        *blocked = true;
                        return;
                    }
                };
                let planned = Planned {
                    block,
                    job,
                    before,
                    seen: call,
                };
                chain.queued_len += planned.len();
                chain.queue.push_back(planned);
            }
        }
    }

    impl Pipeline {
        /// Execute the chain's blocks that `input` starts with into `out`,
        /// blocks of `frame`, calling `next` before each block but the
        /// first and stopping if it returns false, while planning the
        /// blocks of `input` after them a ring's worth ahead, which keep
        /// decoding after it returns. Returns whether the last block it
        /// executed was the frame's last, `None` if it executed none, and
        /// sets `read` to where the blocks executed end in `input.data`, on
        /// failure too. It stops before a block `input` does not hold
        /// whole, and before one whose decode fails or that it cannot plan,
        /// which the serial decoder then decodes, giving its verdict on it.
        fn run<O: FrameOut>(
            &mut self,
            input: Input<'_>,
            frame: &mut Frame,
            out: &mut O,
            simd: Level,
            read: &mut usize,
            next: &mut impl FnMut(&mut O) -> bool,
        ) -> Result<Option<bool>, DecodeError> {
            self.calls += 1;
            let depth = 2 * rayon::current_num_threads();
            let block_size_max = frame.block_size_max;
            let (mut pos, mut last, mut blocked) = (0, None, false);
            loop {
                self.fill(input, pos, depth, &mut blocked);
                let Some(chain) = &mut self.chain else {
                    break;
                };
                let Some(front) = chain.queue.front_mut() else {
                    break;
                };
                let Some((block, content)) = input.block(pos, block_size_max) else {
                    break;
                };
                // Planned in an earlier call, on its input.
                if front.seen != self.calls {
                    if !front.holds(&block, content) {
                        self.rewind();
                        continue;
                    }
                    front.seen = self.calls;
                }
                if last.is_some() && !next(out) {
                    break;
                }
                let Pipeline {
                    chain: Some(chain),
                    free,
                    ..
                } = self
                else {
                    break;
                };
                let p = &chain.queue[0];
                let plan = match (p.block.block_type, &p.job) {
                    (BlockType::Raw, _) => Plan::Raw(content),
                    (BlockType::RLE, _) => {
                        Plan::Rle(content[0], p.block.decompressed_size as usize)
                    }
                    (_, Some((job, _))) => {
                        let Ok(parts) = split_block(content, block_size_max) else {
                            break;
                        };
                        Plan::Compressed(CompressedPlan {
                            parts,
                            huf_def: job.huf_def,
                            fse_def: job.fse_def,
                        })
                    }
                    (_, None) => break,
                };
                let stage2 = match &p.job {
                    Some((job, cell)) => {
                        let ticket = job.id as usize;
                        if cell.take(ticket) {
                            run_taken(cell, ticket);
                        }
                        // While a task decodes the sequences.
                        let literals = cell.literals(ticket, |lits| job.literals(lits));
                        while !cell.startable(ticket) {
                            // If no task has started the block after either,
                            // the pool is behind: decode it here meanwhile.
                            match chain.queue.get(1).and_then(|n| n.job.as_ref()) {
                                Some((job, after)) if after.take(job.id as usize) => {
                                    run_taken(after, job.id as usize)
                                }
                                _ => std::thread::yield_now(),
                            }
                        }
                        Some(Stage2 {
                            cell,
                            ticket,
                            literals,
                        })
                    }
                    None => None,
                };
                let hist = &mut chain.hist;
                let Ok(bytes) = execute_block(&plan, stage2, hist, block_size_max, out, simd)
                else {
                    break;
                };
                let p = chain.queue.pop_front().unwrap();
                let (len, ended) = (p.len(), p.block.last_block);
                chain.queued_len -= len;
                if let Some((_, cell)) = p.job {
                    free.push(cell);
                }
                pos += len;
                *read = pos - input.data_start();
                last = Some(ended);
                frame.block_decoded(bytes)?;
                if ended {
                    break;
                }
            }
            Ok(last)
        }
    }

    /// Stage 2 for compressed block `id`, of a run starting with the
    /// tables in `start`: its literals unless the executing thread has
    /// taken them, and its sequences into `slot`, published as `at` gives
    /// as they decode, in the order `at` gives; `def` gives the sections of
    /// the earlier blocks of the run whose table descriptions it uses.
    fn decode_block<'p, 'a: 'p>(
        slot: &mut Slot,
        at: Decoding<'_>,
        id: u64,
        plan: &CompressedPlan<'_>,
        def: impl Fn(u64) -> Option<&'p BlockParts<'a>>,
        start: FrameStart<'_>,
    ) -> Result<(), DecodeError> {
        at.stage2(
            || decode_block_sequences(slot, at.published(), id, plan, &def, start),
            |lits| decode_literals(lits, id, plan, &def, start),
        )
    }

    /// The literals of compressed block `id` into `lits`, as for
    /// `decode_block`.
    fn decode_literals<'p, 'a: 'p>(
        lits: &mut LitSlot,
        id: u64,
        plan: &CompressedPlan<'_>,
        def: impl Fn(u64) -> Option<&'p BlockParts<'a>>,
        start: FrameStart<'_>,
    ) -> Result<(), DecodeError> {
        if let Some(d) = plan.huf_def {
            if d == id {
                // `decode_block_literals` builds it from this block.
                lits.huf_from = None;
            } else if d == start.id {
                if lits.huf_from != Some(d) {
                    lits.huf.table.copy_from(start.init.huf_table(start.dict));
                    lits.huf_from = Some(d);
                }
            } else if lits.huf_from != Some(d) {
                lits.huf_from = None;
                build_huf_from(described(&def, d)?, &mut lits.huf.table)?;
                lits.huf_from = Some(d);
            }
        }
        decode_block_literals(&plan.parts, &mut lits.huf, None, &mut lits.literals)?;
        if plan.huf_def == Some(id) {
            lits.huf_from = Some(id);
        }
        Ok(())
    }

    /// The sequences of compressed block `id` into `slot`, published into
    /// `publish`, as for `decode_block`.
    fn decode_block_sequences<'p, 'a: 'p>(
        slot: &mut Slot,
        publish: &Published,
        id: u64,
        plan: &CompressedPlan<'_>,
        def: impl Fn(u64) -> Option<&'p BlockParts<'a>>,
        start: FrameStart<'_>,
    ) -> Result<(), DecodeError> {
        slot.seqs.clear();
        let seq = plan.parts.sequences;
        let src = plan.parts.sequences_src;
        if seq.num_sequences == 0 {
            if !src.is_empty() {
                return Err(
                    format!("Extra bits remaining: {} bits", src.len() as isize * 8).into(),
                );
            }
            return Ok(());
        }
        let mut used = 0;
        for (t, mode) in seq.modes.all().into_iter().enumerate() {
            let d = plan.fse_def[t];
            if d == id {
                slot.fse_from[t] = None;
                used += build_sequence_table(mode, &src[used..], &mut slot.fse, t)?;
                slot.fse_from[t] = Some(id);
            } else if d != start.id && slot.fse_from[t] != Some(d) {
                slot.fse_from[t] = None;
                build_table_from(described(&def, d)?, t, &mut slot.fse)?;
                slot.fse_from[t] = Some(d);
            }
        }
        let tables = seq_tables(plan, &slot.fse, start);
        decode_sequences(
            seq.num_sequences,
            &src[used..],
            tables,
            &mut slot.seqs,
            publish,
        )
    }

    /// The sections of block `d`, which described a table a later block
    /// uses, as `def` gives them.
    fn described<'p, 'a>(
        def: &impl Fn(u64) -> Option<&'p BlockParts<'a>>,
        d: u64,
    ) -> Result<&'p BlockParts<'a>, DecodeError> {
        def(d).ok_or_else(|| format!("No description of block {d}'s tables").into())
    }

    /// The LL, OF, ML tables compressed block `plan` decodes with: those
    /// `start` has in use where no block of the batch described one, those
    /// of `fse`, a slot's, built for it otherwise.
    fn seq_tables<'s>(
        plan: &CompressedPlan<'_>,
        fse: &'s FSEScratch,
        start: FrameStart<'s>,
    ) -> [&'s FSETable; 3] {
        std::array::from_fn(|t| {
            if plan.fse_def[t] == start.id {
                start.init.fse.table(t, start.dict.map(|d| &d.fse))
            } else {
                fse.table(t, None)
            }
        })
    }

    /// Build `table` from the tree description of the literals of the
    /// compressed block of sections `def`, with the arguments of its own
    /// build in `decompress_literals`, so the same table kind (X1 / X2).
    fn build_huf_from(def: &BlockParts<'_>, table: &mut HuffmanTable) -> Result<(), DecodeError> {
        let lit = &def.literals;
        table
            .build_decoder(
                def.literals_src,
                lit.regenerated_size as usize,
                lit.four_streams,
            )
            .map(|_| ())
    }

    /// Build table `t` from its description in the earlier block of
    /// sections `def`, skipping the descriptions that precede it there.
    fn build_table_from(
        def: &BlockParts<'_>,
        t: usize,
        fse: &mut FSEScratch,
    ) -> Result<(), DecodeError> {
        let modes = def.sequences.modes.all();
        let src = def.sequences_src;
        let mut used = 0;
        for (u, kind) in SEQ_TABLES.iter().enumerate().take(t) {
            used += match modes[u] {
                ModeType::FSECompressed => {
                    FSETable::new(kind.max_code).read_probabilities(&src[used..], kind.max_log)?
                }
                ModeType::RLE if used < src.len() => 1,
                ModeType::RLE => {
                    return Err(format!("Missing byte for RLE {} table", kind.name).into())
                }
                ModeType::Predefined | ModeType::Repeat => 0,
            };
        }
        build_sequence_table(modes[t], &src[used..], fse, t).map(|_| ())
    }

    /// The block's three sequence tables, checked for the unchecked state
    /// lookups of `decode_rel_sequence`, and the bitstream positioned after
    /// the initial states (ZSTD_initFseState). Same checks and reads as the
    /// start of `run_sequences`.
    fn seq_stream_begin<'a>(
        bit_stream: &'a [u8],
        tables: [&'a FSETable; 3],
    ) -> Result<SeqStream<'a>, DecodeError> {
        let logs = tables.map(|t| u32::from(t.accuracy_log));
        // Every state is `accuracy_log` bits or `next_state + bits` of a
        // cell, which `build_decoding_table` / `build_rle` keep below
        // `1 << accuracy_log`, the table length checked here.
        if tables
            .iter()
            .zip(logs)
            .any(|(t, log)| t.decode().len() != 1 << log)
        {
            return Err("FSE table is uninitialized".into());
        }
        let mut br = BitDStream::new(bit_stream)?;
        let mut states = [0; 3];
        for (s, log) in states.iter_mut().zip(logs) {
            *s = br.read_bits(log);
            br.reload();
        }
        Ok((br, states, tables.map(|t| t.decode())))
    }

    /// A sequence bitstream after its initial LL, OF, ML states, with the
    /// LL, OF, ML decoding tables.
    type SeqStream<'a> = (BitDStream<'a>, [usize; 3], [&'a [FSEEntry]; 3]);

    /// Sequences stage 2 decodes between two publications: each costs a
    /// store, and the executing thread, which executes them faster than
    /// they decode, waits for at most as many.
    const PUBLISH_EVERY: usize = 256;

    /// Stage 2's sequence loop: `run_sequences` without execution. It
    /// publishes the sequences into `publish` `PUBLISH_EVERY` at a time,
    /// the last only once the bitstream is checked.
    fn decode_sequences(
        num_sequences: u32,
        bit_stream: &[u8],
        tables: [&FSETable; 3],
        seqs: &mut Vec<RelSeq>,
        publish: &Published,
    ) -> Result<(), DecodeError> {
        let (mut br, [ll, of, ml], [ll_dt, of_dt, ml_dt]) = seq_stream_begin(bit_stream, tables)?;
        let mut st = [ll, ml, of];
        let mut reps = RelReps::start();
        // Zero-fill first: when the executing thread last read these lines
        // from another CCD, the fill's bulk stores take ownership of them
        // at memory bandwidth, where the loop's 12-byte stores would stall
        // on one cross-CCD invalidation per line (4x slower on Zen 5).
        let n = num_sequences as usize;
        if n == 0 {
            return Err("Missing sequences".into());
        }
        seqs.clear();
        seqs.reserve(n);
        // The sequences go past the length, through `base` alone: the
        // executing thread reads those published while the rest decode.
        let base = seqs.as_mut_ptr();
        // SAFETY: `n` elements are reserved.
        unsafe { ptr::write_bytes(base, 0, n) };
        publish.start(
            base,
            short_offset_share(tables[1]) >= SHORT_OFFSET_SHARE_MIN,
        );
        let mut at = 0;
        while at < n - 1 {
            let end = (at + PUBLISH_EVERY).min(n - 1);
            for i in at..end {
                let s =
                    decode_rel_sequence(&mut br, &mut st, &mut reps, ll_dt, ml_dt, of_dt, false);
                // SAFETY: `n` elements are reserved.
                unsafe { base.add(i).write(s) };
            }
            at = end;
            publish.decoded(at);
        }
        let last = decode_rel_sequence(&mut br, &mut st, &mut reps, ll_dt, ml_dt, of_dt, true);
        // SAFETY: as in the loop.
        unsafe { base.add(n - 1).write(last) };
        if !br.is_finished() {
            return Err("Sequence bitstream not fully consumed".into());
        }
        publish.decoded_all(n, reps);
        Ok(())
    }

    /// `decode_sequence` with the repeat offsets relative to those the
    /// block starts with, `reps`; `st` is the LL, ML, OF states. Kept
    /// separate from `decode_sequence`: sharing one body changed the fused
    /// loop's register allocation and cost it 2-3%.
    #[inline(always)]
    fn decode_rel_sequence(
        br: &mut BitDStream<'_>,
        st: &mut [usize; 3],
        reps: &mut RelReps,
        ll_dt: &[FSEEntry],
        ml_dt: &[FSEEntry],
        of_dt: &[FSEEntry],
        is_last: bool,
    ) -> RelSeq {
        // SAFETY: each state is below its table's length (see
        // `seq_stream_begin`).
        let (ll_e, ml_e, of_e) = unsafe {
            (
                table_entry(ll_dt, st[0]),
                table_entry(ml_dt, st[1]),
                table_entry(of_dt, st[2]),
            )
        };
        let mut ll = ll_e.base_value as usize;
        let mut ml = ml_e.base_value as usize;
        let ll_bits = u32::from(ll_e.extra_bits);
        let ml_bits = u32::from(ml_e.extra_bits);
        let of_bits = u32::from(of_e.extra_bits);
        let total_bits = ll_bits + ml_bits + of_bits;

        // As in `decode_sequence`. The offset of a code above 1 is below
        // 2^32 (code 31: base 2^31 - 3 plus 31 extra bits).
        let off = if of_bits > 1 {
            reps.new_offset((of_e.base_value as usize + br.read_bits_fast(of_bits)) as u64)
        } else {
            let ll0 = usize::from(ll == 0);
            if of_bits == 0 {
                reps.code0(ll0)
            } else {
                reps.code1(1 + ll0 + br.read_bits_fast(1))
            }
        };
        if ml_bits > 0 {
            ml += br.read_bits_fast(ml_bits);
        }
        // Same reload rule as `decode_sequence`.
        if total_bits >= 57 - 26 {
            br.reload();
        }
        if ll_bits > 0 {
            ll += br.read_bits_fast(ll_bits);
        }
        if !is_last {
            st[0] = usize::from(ll_e.next_state) + br.read_bits(u32::from(ll_e.num_bits));
            st[1] = usize::from(ml_e.next_state) + br.read_bits(u32::from(ml_e.num_bits));
            st[2] = usize::from(of_e.next_state) + br.read_bits(u32::from(of_e.num_bits));
            br.reload();
        }
        // Lengths are below 2^17.
        RelSeq::new(ll as u32, ml as u32, off)
    }

    /// The stage 2 of a compressed block for `execute_block`: the cell and
    /// ticket it decodes under, and its literals if the executing thread
    /// decoded them (`Cell::literals`), else whoever took them does.
    struct Stage2<'c> {
        cell: &'c Cell,
        ticket: usize,
        literals: Option<MutexGuard<'c, LitSlot>>,
    }

    /// Stage 3 for one block, whose stage 2, if compressed, is `stage2`:
    /// write it to `out` and return its bytes. The block's sequences
    /// execute as stage 2 publishes them, once its literals are done; the
    /// block and its repeat offsets count once stage 2 and the literals
    /// have ended without an error.
    fn execute_block<'o>(
        plan: &Plan<'_>,
        stage2: Option<Stage2<'_>>,
        hist: &mut [u32; 3],
        block_size_max: usize,
        out: &'o mut impl FrameOut,
        simd: Level,
    ) -> Result<&'o [u8], DecodeError> {
        let (dst, ext) = out.block_dst()?;
        // SAFETY: `block_dst` meets the `Dst` and `ExtHistory` contracts. A
        // raw or RLE block decodes to at most `block_size_max <=
        // MAX_BLOCK_SIZE` bytes (`parse_block_header`), and literals
        // without sequences too (`split_block`), within the room `Dst` has,
        // from input apart from it; on success each arm wrote the block up
        // to `end`.
        unsafe {
            let at = dst.base.add(dst.op);
            let end = match plan {
                Plan::Raw(content) => {
                    ptr::copy_nonoverlapping(content.as_ptr(), at, content.len());
                    dst.op + content.len()
                }
                Plan::Rle(byte, len) => {
                    ptr::write_bytes(at, *byte, *len);
                    dst.op + len
                }
                Plan::Compressed(cp) => {
                    let Stage2 {
                        cell,
                        ticket,
                        literals,
                    } = stage2.expect("a compressed block is executed from its stage 2");
                    let lits = literals.unwrap_or_else(|| cell.decoded_literals(ticket));
                    let num = cp.parts.sequences.num_sequences as usize;
                    let mut h = *hist;
                    let end = match stage2_result(&lits.result) {
                        Err(e) => Err(e),
                        Ok(()) if num == 0 => {
                            let len = lits.literals.len() - WILDCOPY_OVERLENGTH;
                            ptr::copy_nonoverlapping(lits.literals.as_ptr(), at, len);
                            Ok(dst.op + len)
                        }
                        Ok(()) => {
                            let runs = PublishedRuns { cell, ticket, num };
                            let literals = &lits.literals[..];
                            execute_published(
                                runs,
                                literals,
                                ext,
                                &mut h,
                                block_size_max,
                                dst,
                                simd,
                            )
                        }
                    };
                    // Stage 2's error first: it may have published only
                    // part of the sequences.
                    stage2_result(&cell.finished(ticket).result)?;
                    let end = end?;
                    *hist = h;
                    end
                }
            };
            Ok(out.commit(end))
        }
    }

    /// Whether a stage 2 decode, of a block's sequences or literals, with
    /// `result` succeeded. Its error stays in the cell: a chain that stops
    /// at the block may execute it again.
    fn stage2_result(result: &Result<(), DecodeError>) -> Result<(), DecodeError> {
        match result {
            Ok(()) => Ok(()),
            Err(_) => Err("Block's stage 2 failed".into()),
        }
    }

    /// Execute the sequences `runs` publishes, with `literals`, followed by
    /// `WILDCOPY_OVERLENGTH` bytes of slack, into `dst`, as `execute_with`
    /// does, waiting for each run of them; returns the block's end.
    ///
    /// # Safety
    /// `dst` meets the `Dst` contract and `ext` the `ExtHistory` one.
    unsafe fn execute_published(
        runs: PublishedRuns<'_>,
        literals: &[u8],
        ext: ExtHistory,
        hist: &mut [u32; 3],
        block_size_max: usize,
        dst: Dst,
        simd: Level,
    ) -> Result<usize, DecodeError> {
        // Where they are, and whether the offsets are short, is published
        // with the first run.
        runs.ready(0)?;
        let p = &runs.cell.published;
        let short = p.short.load(Ordering::Relaxed);
        let seqs = DecodedSeqs {
            seqs: p.seqs.load(Ordering::Relaxed),
            runs,
            literals,
        };
        if ext.len == 0 {
            execute_with(simd, short, seqs, hist, block_size_max, dst)
        } else {
            let seqs = ExtDecodedSeqs { seqs, ext };
            execute_with(simd, short, seqs, hist, block_size_max, dst)
        }
    }

    /// The runs of the `num` sequences the decode of the block of `ticket`
    /// publishes in `cell`.
    #[derive(Clone, Copy)]
    struct PublishedRuns<'a> {
        cell: &'a Cell,
        ticket: usize,
        num: usize,
    }

    impl PublishedRuns<'_> {
        /// How many sequences are published, once more than `have` are,
        /// or an error once the decode has ended without publishing more.
        #[inline(always)]
        fn ready(self, have: usize) -> Result<usize, DecodeError> {
            let n = self.cell.published.progress.load(Ordering::Acquire);
            if n > have {
                return Ok(n);
            }
            self.wait(have)
        }

        #[cold]
        #[inline(never)]
        fn wait(self, have: usize) -> Result<usize, DecodeError> {
            let mut spins = 0;
            loop {
                // Done before the count: a decode that is done has
                // published all it does.
                let done = self.cell.decoded(self.ticket);
                let n = self.cell.published.progress.load(Ordering::Acquire);
                if n > have {
                    return Ok(n);
                }
                if done {
                    return Err("Block's sequences not decoded".into());
                }
                pause(&mut spins);
            }
        }
    }

    /// A block's sequences as stage 2 publishes them, at `seqs`, with its
    /// literals followed by `WILDCOPY_OVERLENGTH` bytes of slack.
    struct DecodedSeqs<'a> {
        seqs: *const RelSeq,
        runs: PublishedRuns<'a>,
        literals: &'a [u8],
    }

    impl BlockSequences for DecodedSeqs<'_> {
        #[inline(always)]
        unsafe fn execute<W: WildCopy>(
            self,
            w: W,
            offset_hist: &mut [u32; 3],
            dst: Dst,
        ) -> Result<usize, DecodeError> {
            execute_sequences::<W, false>(w, self, ExtHistory::NONE, offset_hist, dst)
        }
    }

    /// `DecodedSeqs` whose matches reach on into `ext` before the segment,
    /// in their own instantiation of the loop, as `ExtSeqInput`.
    struct ExtDecodedSeqs<'a> {
        seqs: DecodedSeqs<'a>,
        ext: ExtHistory,
    }

    impl BlockSequences for ExtDecodedSeqs<'_> {
        #[inline(always)]
        unsafe fn execute<W: WildCopy>(
            self,
            w: W,
            offset_hist: &mut [u32; 3],
            dst: Dst,
        ) -> Result<usize, DecodeError> {
            execute_sequences::<W, true>(w, self.seqs, self.ext, offset_hist, dst)
        }
    }

    /// Execute decoded sequences into `dst` from `dst.op` on, each run as
    /// it is published; returns the block's end. Same contract as
    /// `run_sequences`.
    ///
    /// # Safety
    /// `dst` meets the `Dst` contract, and `ext` the `ExtHistory` one when
    /// `EXT`.
    #[inline(always)]
    unsafe fn execute_sequences<W: WildCopy, const EXT: bool>(
        w: W,
        DecodedSeqs {
            seqs,
            runs,
            literals,
        }: DecodedSeqs<'_>,
        ext: ExtHistory,
        offset_hist: &mut [u32; 3],
        dst: Dst,
    ) -> Result<usize, DecodeError> {
        let out = dst.base;
        let base = RelBase::new(*offset_hist);
        let lit = literals.as_ptr();
        // In bounds by the contract, and `literals` ends with
        // `WILDCOPY_OVERLENGTH` bytes of slack.
        let mut cur = SeqCursor {
            op: out.add(dst.op),
            lit,
        };
        let lim = SeqLimits {
            oend_w: out.add(dst.op + MAX_BLOCK_SIZE),
            lit_limit: lit.add(literals.len() - WILDCOPY_OVERLENGTH),
            prefix: out,
            ext,
            window: dst.window,
        };
        let mut at = 0;
        while at < runs.num {
            let end = runs.ready(at)?;
            // Published, so written for good.
            for s in std::slice::from_raw_parts(seqs.add(at), end - at) {
                let ll = (s.ll_from & LL_MASK) as usize;
                exec_sequence::<W, EXT>(w, &mut cur, &lim, ll, s.ml as usize, base.offset(s))
                    .map_err(seq_error_message)?;
            }
            at = end;
        }
        // Last literals; both cursors only advanced within their buffers.
        let rest = lim.lit_limit as usize - cur.lit as usize;
        if cur.op as usize + rest > lim.oend_w as usize {
            return Err(seq_error_message(SeqError::BlockTooLarge));
        }
        ptr::copy_nonoverlapping(cur.lit, cur.op, rest);
        // All published, with the last.
        *offset_hist = base.hist(runs.cell.published.end());
        Ok(cur.op as usize + rest - out as usize)
    }

    impl FrameDecoder {
        /// Decode the blocks of the frame being decoded at the start of
        /// `data`, from a block header on, into `out`, the frame having
        /// started from `dict` if given, unless the frame fits in one
        /// block: on the current rayon pool while the frame's `Pipeline`
        /// runs a chain or if `Gate::pools` takes the blocks `located`
        /// gives for `room` bytes of output, else those one after another
        /// as `process` would. Before each block but the first it calls
        /// `next`, where the serial driver writes out the blocks before,
        /// and stops if it returns false. Returns the `Event` of the last
        /// block it decoded, or `None` if it decoded none, and sets `read`
        /// to how much of `data` those blocks take, on failure too.
        ///
        /// If `room` takes all the whole blocks of `data`, or the rest of
        /// the frame's content, none is left to decode ahead, and the pool
        /// decodes them in one scope, from `data` as borrowed. Otherwise
        /// the pipeline's chain takes them, and those it planned past the
        /// last one executed keep decoding after the call, for the next.
        /// It stops before a block that fails, or that it cannot plan, and
        /// leaves the serial decoder's state as the serial decoder would
        /// after the blocks before (on `Pipeline::hand_back`, for a chain),
        /// which then decodes that one and gives its verdict. The frame's
        /// size checks, after each block and after the last, are the serial
        /// ones, and fail here as there.
        ///
        /// Inline down to the stage and frame it never takes, which cost
        /// the serial driver a branch on every unit.
        #[inline]
        pub(super) fn decode_blocks_parallel<O: FrameOut>(
            &mut self,
            data: &[u8],
            dict: Option<&DecodeDict>,
            out: &mut O,
            room: usize,
            read: &mut usize,
            next: impl FnMut(&mut O) -> bool,
        ) -> Result<Option<Event>, DecodeError> {
            match &self.stage {
                Stage::Block {
                    frame,
                    header: None,
                } if self.parallel.is_some() && !frame.fits_one_block() => {
                    self.decode_block_batch(data, dict, out, room, read, next)
                }
                _ => Ok(None),
            }
        }

        /// `decode_blocks_parallel` at a block header of a frame it may
        /// take.
        #[inline(never)]
        fn decode_block_batch<O: FrameOut>(
            &mut self,
            data: &[u8],
            dict: Option<&DecodeDict>,
            out: &mut O,
            room: usize,
            read: &mut usize,
            next: impl FnMut(&mut O) -> bool,
        ) -> Result<Option<Event>, DecodeError> {
            let (Stage::Block { frame, .. }, Some(gate)) = (&self.stage, self.parallel) else {
                return Ok(None);
            };
            let block_size_max = frame.block_size_max;
            // Room for the rest of the frame's content is room for all its
            // blocks, however much more they could decode to.
            let room = match frame.content_size {
                Some(fcs) if fcs - frame.decoded <= room as u64 => usize::MAX,
                _ => room,
            };
            if !self.chain_active() {
                if !gate.pools(located(data, block_size_max, room)) {
                    return self.decode_serially(data, dict, out, room, read, next);
                }
                if room_takes_all(data, block_size_max, room) {
                    return self.decode_scoped(data, dict, out, room, read, next);
                }
            }
            self.decode_detached(Input { held: None, data }, dict, out, read, next)
        }

        /// `decode_blocks_parallel` at the content of a block whose header
        /// the serial decoder took, `content`: if the frame's chain runs,
        /// it takes that block, then the blocks `data` starts with, as
        /// `decode_blocks_parallel` takes blocks, and sets `read` to how
        /// much of `data` those after it take. A block that straddles two
        /// calls' input so stays in the chain, which goes on after it.
        #[inline]
        pub(super) fn decode_held_block_parallel<O: FrameOut>(
            &mut self,
            content: &[u8],
            data: &[u8],
            dict: Option<&DecodeDict>,
            out: &mut O,
            read: &mut usize,
            next: impl FnMut(&mut O) -> bool,
        ) -> Result<Option<Event>, DecodeError> {
            match &self.stage {
                Stage::Block {
                    header: Some(block),
                    ..
                } if self.chain_active() => {
                    let held = Some((*block, content));
                    self.decode_detached(Input { held, data }, dict, out, read, next)
                }
                _ => Ok(None),
            }
        }

        /// `decode_block_batch` on blocks `room` takes all of: stages 2 and
        /// 3 in one rayon scope.
        fn decode_scoped<O: FrameOut>(
            &mut self,
            data: &[u8],
            dict: Option<&DecodeDict>,
            out: &mut O,
            room: usize,
            read: &mut usize,
            mut next: impl FnMut(&mut O) -> bool,
        ) -> Result<Option<Event>, DecodeError> {
            let (Stage::Block { frame, .. }, Some(scratch)) = (&mut self.stage, &mut self.scratch)
            else {
                return Ok(None);
            };
            let block_size_max = frame.block_size_max;
            let pipeline = self.pipeline.get_or_insert_default();
            let start = FrameStart {
                id: pipeline.new_id(),
                init: scratch,
                dict: dict.and_then(DecodeDict::entropy),
            };
            let batch = plan_blocks(data, block_size_max, start, room, &mut pipeline.next_id);
            let ring = pipeline.ring((2 * rayon::current_num_threads()).min(batch.plans.len()));
            let (done, hist, accounted) =
                run_batch(&batch, frame, start, out, self.simd, &mut next, &ring);
            self.pipeline.get_or_insert_default().free.extend(ring);
            let Some(&end) = done.checked_sub(1).and_then(|i| batch.ends.get(i)) else {
                return Ok(None);
            };
            *read = end;
            accounted?;
            if done == batch.plans.len() && batch.last {
                return self.blocks_ended().map(Some);
            }
            let mut defs = [None; 4];
            for plan in &batch.plans[..done] {
                if let Plan::Compressed(cp) = plan {
                    defs[0] = cp.huf_def.or(defs[0]);
                    if cp.parts.sequences.num_sequences != 0 {
                        defs[1..].copy_from_slice(&cp.fse_def.map(Some));
                    }
                }
            }
            let start = start.id;
            sync_scratch(scratch, hist, defs, start, |d| batch.def(d))?;
            Ok(Some(Event::Continue))
        }

        /// `decode_block_batch`, or `decode_held_block_parallel`, on the
        /// frame's pipeline, which starts a chain from the scratch if none
        /// runs.
        #[inline(never)]
        fn decode_detached<O: FrameOut>(
            &mut self,
            input: Input<'_>,
            dict: Option<&DecodeDict>,
            out: &mut O,
            read: &mut usize,
            mut next: impl FnMut(&mut O) -> bool,
        ) -> Result<Option<Event>, DecodeError> {
            let simd = self.simd;
            let (Stage::Block { frame, header }, Some(scratch)) = (&mut self.stage, &self.scratch)
            else {
                return Ok(None);
            };
            let pipeline = self.pipeline.get_or_insert_default();
            if !pipeline.active() {
                pipeline.start(scratch, dict, frame.block_size_max);
            }
            let ran = pipeline.run(input, frame, out, simd, read, &mut next);
            // The held block, if any, is the first it executed.
            if !matches!(ran, Ok(None)) {
                *header = None;
            }
            match ran? {
                None => Ok(None),
                Some(false) => Ok(Some(Event::Continue)),
                Some(true) => self.blocks_ended().map(Some),
            }
        }

        /// `decode_block_batch` on blocks the pool does not take, with no
        /// chain running: the blocks `located` gives, decoded one after
        /// another as `process` decodes them, here rather than by the
        /// serial driver so that none is located twice.
        fn decode_serially<O: FrameOut>(
            &mut self,
            data: &[u8],
            dict: Option<&DecodeDict>,
            out: &mut O,
            room: usize,
            read: &mut usize,
            mut next: impl FnMut(&mut O) -> bool,
        ) -> Result<Option<Event>, DecodeError> {
            let (Stage::Block { frame, .. }, Some(scratch)) = (&mut self.stage, &mut self.scratch)
            else {
                return Ok(None);
            };
            let dict = dict.and_then(DecodeDict::entropy);
            let mut last = None;
            for (i, (block, content, end)) in located(data, frame.block_size_max, room).enumerate()
            {
                if i > 0 && !next(out) {
                    break;
                }
                *read = end;
                super::decode_block(&block, content, frame, scratch, dict, out, self.simd)?;
                last = Some(block.last_block);
            }
            match last {
                None => Ok(None),
                Some(false) => Ok(Some(Event::Continue)),
                Some(true) => self.blocks_ended().map(Some),
            }
        }
    }

    /// Whether `room` bytes of output take all the whole blocks of `data`
    /// (`located`).
    fn room_takes_all(data: &[u8], block_size_max: usize, room: usize) -> bool {
        located(data, block_size_max, room).count()
            == located(data, block_size_max, usize::MAX).count()
    }

    /// Stages 2 and 3 of `batch`, blocks of `frame`, from the tables and
    /// repeat offsets in `start`, into `out`, calling `next` before each
    /// block but the first, with tasks decoding into `ring`. Returns how
    /// many blocks it decoded, from the first, up to one whose stage 2 or 3
    /// fails or before which `next` returns false; the repeat offsets after
    /// them, and the frame's size check on them, which ends them where it
    /// fails.
    fn run_batch<O: FrameOut>(
        batch: &Batch<'_>,
        frame: &mut Frame,
        start: FrameStart<'_>,
        out: &mut O,
        simd: Level,
        next: &mut impl FnMut(&mut O) -> bool,
        ring: &[Arc<Cell>],
    ) -> (usize, [u32; 3], Result<(), DecodeError>) {
        // Block `i` is decoded into `ring[i % ring.len()]` by a rayon task
        // spawned once block `i - ring.len()` has been executed from it, or
        // by the executing thread if no task has started it by the time
        // that thread needs block `i`, or waits for block `i - 1`. It
        // decodes no block further ahead, so a block it needs is never left
        // waiting behind the decode of a later one.
        //
        // The executing thread is the caller's, not a worker's, so output
        // it faults in is freed on the CPU that faulted it: a fresh buffer
        // faulted on a worker and freed by the caller took half again as
        // much kernel time per fault.
        let block_size_max = frame.block_size_max;
        let (plans, base) = (&batch.plans[..], batch.base);
        let def = |d| batch.def(d);
        let ticket = |i: usize| (base + i as u64) as usize;
        let decode = |i: usize, cell: &Cell| {
            if let Plan::Compressed(cp) = &plans[i] {
                let id = base + i as u64;
                cell.decode(ticket(i), |slot, at| {
                    decode_block(slot, at, id, cp, def, start)
                });
            }
        };
        let mut hist = start.init.offset_hist;
        let (mut done, mut stopped) = (0, false);
        let accounted = rayon::in_place_scope_fifo(|s| {
            let spawn_decode = |i: usize| {
                let Some(Plan::Compressed(_)) = plans.get(i) else {
                    return;
                };
                let cell = &*ring[i % ring.len()];
                cell.hand(ticket(i), i == 0);
                s.spawn_fifo(move |_| {
                    if cell.take(ticket(i)) {
                        decode(i, cell);
                    }
                });
            };
            for i in 0..ring.len() {
                spawn_decode(i);
            }
            for (i, plan) in plans.iter().enumerate() {
                if i != 0 {
                    if !next(out) {
                        stopped = true;
                        break;
                    }
                    // Block `i - 1` has been executed from its cell.
                    spawn_decode(i - 1 + ring.len());
                }
                let stage2 = match plan {
                    Plan::Compressed(cp) => {
                        let cell = &*ring[i % ring.len()];
                        if cell.take(ticket(i)) {
                            decode(i, cell);
                        }
                        // While a task decodes the sequences.
                        let id = base + i as u64;
                        let literals = cell
                            .literals(ticket(i), |lits| decode_literals(lits, id, cp, def, start));
                        while !cell.startable(ticket(i)) {
                            // If no task has started block `i + 1` either,
                            // the decoders are behind: decode it here while
                            // block `i` finishes. Its cell is free, as block
                            // `i + 1 - ring.len()` has been executed.
                            let after = &*ring[(i + 1) % ring.len()];
                            match plans.get(i + 1) {
                                Some(Plan::Compressed(_)) if after.take(ticket(i + 1)) => {
                                    decode(i + 1, after)
                                }
                                // Hand the CPU to a worker the kernel may
                                // have queued on it.
                                _ => std::thread::yield_now(),
                            }
                        }
                        Some(Stage2 {
                            cell,
                            ticket: ticket(i),
                            literals,
                        })
                    }
                    _ => None,
                };
                let Ok(bytes) = execute_block(plan, stage2, &mut hist, block_size_max, out, simd)
                else {
                    break;
                };
                done = i + 1;
                frame.block_decoded(bytes)?;
            }
            if stopped {
                // Take the decodes no task has started, so that their tasks
                // return; the scope waits for the others.
                for j in done..plans.len().min(done + ring.len()) {
                    ring[j % ring.len()].take(ticket(j));
                }
            }
            Ok(())
        });
        (done, hist, accounted)
    }

    /// Leave `scratch` as the serial decoder leaves it after a run of
    /// blocks from the tables `scratch` had, named `start`, all decoded:
    /// with repeat offsets `hist`, and the Huffman and sequence tables
    /// `defs` names rebuilt from the descriptions `def` gives, those of
    /// blocks of the run, for the blocks after to repeat.
    fn sync_scratch<'p, 'a: 'p>(
        scratch: &mut DecoderScratch,
        hist: [u32; 3],
        defs: Defs,
        start: u64,
        def: impl Fn(u64) -> Option<&'p BlockParts<'a>>,
    ) -> Result<(), DecodeError> {
        scratch.offset_hist = hist;
        if let Some(d) = defs[0].filter(|&d| d != start) {
            build_huf_from(described(&def, d)?, &mut scratch.huf.table)?;
            scratch.huf_from_dict = false;
        }
        for (t, d) in defs[1..].iter().enumerate() {
            if let Some(d) = d.filter(|&d| d != start) {
                build_table_from(described(&def, d)?, t, &mut scratch.fse)?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::{mpsc, Weak};

        const BLOCK_SIZE_MAX: usize = 1 << 17;

        /// Blocks of `kinds`, `r` raw, `z` RLE and `c` compressed, each of
        /// Block_Size `n` but RLE's 1, the last one last.
        fn blocks(kinds: &str, n: usize) -> Vec<u8> {
            let mut out = Vec::new();
            for (i, kind) in kinds.bytes().enumerate() {
                let (ty, size) = match kind {
                    b'r' => (0, n),
                    b'z' => (1, 1),
                    _ => (2, n),
                };
                let last = (i + 1 == kinds.len()) as u32;
                let header = last | ty << 1 | (n as u32) << 3;
                out.extend_from_slice(&header.to_le_bytes()[..3]);
                out.resize(out.len() + size, 7);
            }
            out
        }

        fn pools(kinds: &str, n: usize, min_blocks: usize, min_bytes: usize) -> bool {
            let data = blocks(kinds, n);
            let span = located(&data, BLOCK_SIZE_MAX, usize::MAX);
            assert_eq!(span.count(), kinds.len(), "{kinds}");
            let span = located(&data, BLOCK_SIZE_MAX, usize::MAX);
            let opts = DecodeOptions {
                min_parallel_blocks: min_blocks,
                min_parallel_bytes: min_bytes,
                ..DecodeOptions::default()
            };
            Gate::new(&opts).unwrap().pools(span)
        }

        /// `Gate::pools` on either side of its per-block floor: under a gate
        /// of `MIN_BYTES`, 240 compressed blocks of `MIN_BLOCK_BYTES` pool
        /// and of one byte fewer do not, though over `MIN_BYTES` in all; RLE
        /// blocks among them leave the average alone; a gate of no bytes
        /// takes blocks of one byte.
        #[test]
        fn gate_pools_at_its_block_floor() {
            let c = "c".repeat(240);
            const { assert!(240 * (MIN_BLOCK_BYTES - 1) > MIN_BYTES) };
            assert!(pools(&c, MIN_BLOCK_BYTES, 2, MIN_BYTES));
            assert!(!pools(&c, MIN_BLOCK_BYTES - 1, 2, MIN_BYTES));
            assert!(pools(&"cz".repeat(240), MIN_BLOCK_BYTES, 2, MIN_BYTES));
            assert!(pools(&c, 1, 2, 0));
        }

        /// `Gate::pools` on either side of its two thresholds: blocks of no
        /// compressed one, one fewer than `min_blocks`, and that many of
        /// one byte fewer than `min_bytes` or exactly that many.
        #[test]
        fn gate_pools_at_its_thresholds() {
            assert!(!pools("rzrzr", 100, 1, 0));
            assert!(pools("rzczr", 100, 1, 0));
            assert!(!pools("crrzc", 4000, 3, 0));
            // Three compressed blocks of 100 bytes.
            for (min_bytes, want) in [(0, true), (299, true), (300, true), (301, false)] {
                assert_eq!(pools("crczc", 100, 3, min_bytes), want, "{min_bytes}");
            }
        }

        /// A cell's tickets: one handed is taken once, by whoever comes
        /// first, and is done once decoded; an earlier ticket, a task's
        /// whose block someone else took or whose cell was handed another,
        /// is neither takeable nor done. So for its literals: taken once,
        /// an earlier ticket's not at all. A ticket handed has published
        /// nothing and has no literals, whatever the cell's decode before
        /// it did, and is startable once its decode is done, or has
        /// published with its literals done. A decode that panics is done,
        /// and its panic resumes on the thread that executes the block.
        #[test]
        fn cell_tickets_are_taken_once() {
            let cell = Cell::new();
            assert!(!cell.take(0) && !cell.decoded(0), "new");
            assert!(cell.literals(0, |_| Ok(())).is_none(), "new literals");
            cell.hand(7, false);
            assert!(!cell.decoded(7));
            assert!(cell.take(7), "planned");
            assert!(!cell.take(7), "taken");
            cell.decode(7, |_, at| {
                at.published().decoded(1);
                Ok(())
            });
            assert!(cell.decoded(7) && cell.startable(7));
            cell.hand(8, true);
            assert!(!cell.take(7) && !cell.decoded(8), "handed the next");
            assert!(!cell.startable(8), "nothing published");
            assert!(cell.literals(7, |_| Ok(())).is_none(), "earlier literals");
            cell.published.decoded(1);
            assert!(!cell.startable(8), "published, literals to decode");
            assert!(cell.literals(8, |_| Ok(())).is_some(), "literals planned");
            let twice = cell.literals(8, |_| panic!("literals decoded twice"));
            assert!(twice.is_none(), "literals taken");
            assert!(cell.startable(8), "published, literals done");
            cell.hand(9, false);
            cell.published.decoded(1);
            assert!(!cell.startable(9), "the literals of the ticket before");
            assert!(cell.take(9));
            cell.decode(9, |_, _| panic!("stage 2"));
            assert!(cell.decoded(9));
            let resumed = panic::catch_unwind(AssertUnwindSafe(|| drop(decoded_slot(&cell))));
            let payload = resumed.expect_err("the decode's panic");
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"stage 2"));
            // Resumed once; the slot is not poisoned.
            assert!(decoded_slot(&cell).result.is_err());
        }

        /// The libzstd frame of `data` in blocks of 1 KiB, with a 1 KiB
        /// window and Huffman literals forced on; returns it with where its
        /// blocks start.
        fn small_blocks(data: &[u8]) -> (Vec<u8>, usize) {
            use zstd::zstd_safe::zstd_sys as sys;
            unsafe {
                let cctx = sys::ZSTD_createCCtx();
                for (p, v) in [
                    (sys::ZSTD_cParameter::ZSTD_c_compressionLevel, 3),
                    (sys::ZSTD_cParameter::ZSTD_c_windowLog, 10),
                    (sys::ZSTD_cParameter::ZSTD_c_experimentalParam18, 1024),
                    (sys::ZSTD_cParameter::ZSTD_c_experimentalParam5, 1),
                ] {
                    assert_eq!(
                        sys::ZSTD_isError(sys::ZSTD_CCtx_setParameter(cctx, p, v)),
                        0
                    );
                }
                let mut out = vec![0u8; sys::ZSTD_compressBound(data.len())];
                let n = sys::ZSTD_compress2(
                    cctx,
                    out.as_mut_ptr().cast(),
                    out.len(),
                    data.as_ptr().cast(),
                    data.len(),
                );
                assert_eq!(sys::ZSTD_isError(n), 0);
                sys::ZSTD_freeCCtx(cctx);
                out.truncate(n);
                let header = frame_header_len(&out);
                (out, header)
            }
        }

        /// 40 KiB of words, compressed blocks of 1 KiB each.
        fn words() -> Vec<u8> {
            let words: [&[u8]; 6] = [b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"lazy "];
            let (mut data, mut x) = (Vec::new(), 1u64);
            while data.len() < 40 << 10 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                data.extend_from_slice(words[(x >> 33) as usize % words.len()]);
            }
            data.truncate(40 << 10);
            data
        }

        /// What `retire` leaves of the blocks a chain planned: each block's
        /// job, the cell it decodes into, and its id.
        struct PlannedJobs {
            jobs: Vec<(Weak<Job>, Arc<Cell>, u64)>,
            start: Weak<StartTables>,
        }

        impl PlannedJobs {
            fn of(dec: &FrameDecoder) -> PlannedJobs {
                assert!(matches!(dec.stage, Stage::Block { .. }), "inside the frame");
                let chain = dec
                    .pipeline
                    .as_ref()
                    .and_then(|p| p.chain.as_ref())
                    .expect("a chain");
                let jobs = chain
                    .queue
                    .iter()
                    .filter_map(|p| p.job.as_ref())
                    .map(|(job, cell)| (Arc::downgrade(job), cell.clone(), job.id))
                    .collect();
                PlannedJobs {
                    jobs,
                    start: Arc::downgrade(&chain.start),
                }
            }

            /// Every job is gone and its ticket taken from its cell, which
            /// decoded none of them, and so is what the chain started from.
            fn assert_retired(&self, what: &str) {
                for (job, cell, id) in &self.jobs {
                    let ticket = *id as usize;
                    assert!(job.upgrade().is_none(), "{what}: job {id} dropped");
                    assert!(!cell.take(ticket), "{what}: job {id} taken");
                    assert!(!cell.decoded(ticket), "{what}: job {id} not decoded");
                }
                assert!(self.start.upgrade().is_none(), "{what}: start dropped");
            }

            /// Wait until no task holds a cell: every task the chain spawned
            /// has returned.
            fn wait_for_tasks(&self, what: &str) {
                let t0 = std::time::Instant::now();
                for (_, cell, id) in &self.jobs {
                    while Arc::strong_count(cell) != 1 {
                        assert!(t0.elapsed().as_secs() < 10, "{what}: task {id} returned");
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        }

        /// A pool of two threads, one of them held until the returned
        /// sender sends or drops, so that tasks spawned from the other,
        /// which runs `f`, stay queued while it runs.
        fn on_held_pool<R: Send>(f: impl FnOnce(mpsc::Sender<()>) -> R + Send) -> R {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            pool.install(|| {
                let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
                rayon::spawn(move || {
                    held.0.send(()).unwrap();
                    let _ = release.1.recv();
                });
                held.1.recv().unwrap();
                f(release.0)
            })
        }

        /// One call on the frame of `words` through the pipeline that
        /// executes its first block only, the pool's second thread held:
        /// the decoder, its output, and where the call stopped reading.
        fn one_block_decoded() -> (FrameDecoder, Vec<u8>, usize) {
            let (frame, header) = small_blocks(&words());
            let mut dec = FrameDecoder::new(&DecodeOptions {
                min_parallel_blocks: 1,
                min_parallel_bytes: 0,
                simd: true,
                window_log_max: 0,
            });
            let mut output = Vec::new();
            let mut out = VecOut {
                output: &mut output,
                prefix: Prefix {
                    start: 0,
                    window: 0,
                },
                dict: &[],
            };
            let started = dec.process(&frame[..header], &mut out, None);
            assert!(matches!(started, Ok(Event::FrameStarted)));
            let mut read = 0;
            let decoded = dec.decode_blocks_parallel(
                &frame[header..],
                None,
                &mut out,
                2048,
                &mut read,
                |_| false,
            );
            assert!(matches!(decoded, Ok(Some(Event::Continue))));
            (dec, output, header + read)
        }

        /// A frame's reset or drop, or a block the serial decoder decodes,
        /// with the blocks the chain planned queued on a held pool: each
        /// one's ticket is taken from its cell and its job dropped there
        /// and then, with what the chain started from, and once the pool
        /// runs their tasks, none decodes. After the serial block the frame
        /// decodes as `decompress` decodes it.
        #[test]
        fn pipeline_retires_planned_blocks_at_once() {
            for what in ["reset", "drop", "serial block"] {
                on_held_pool(|release| {
                    let (mut dec, mut output, read) = one_block_decoded();
                    let planned = PlannedJobs::of(&dec);
                    assert_eq!(
                        planned.jobs.len(),
                        2 * rayon::current_num_threads(),
                        "{what}"
                    );
                    for (_, cell, id) in &planned.jobs {
                        assert!(!cell.decoded(*id as usize), "{what}: {id} queued");
                    }
                    match what {
                        "reset" => {
                            dec.leave_frame();
                        }
                        "drop" => drop(dec),
                        _ => {
                            let (frame, _) = small_blocks(&words());
                            let mut out = VecOut {
                                output: &mut output,
                                // The frame's, as `process` started it.
                                prefix: Prefix {
                                    start: 0,
                                    window: 1 << 10,
                                },
                                dict: &[],
                            };
                            let mut pos = read;
                            loop {
                                let len = dec.unit_len(&frame[pos..]);
                                let event = dec.process(&frame[pos..pos + len], &mut out, None);
                                pos += len;
                                if event.unwrap() == Event::FrameEnded {
                                    break;
                                }
                                if pos == read + 3 {
                                    // The block's header only.
                                    continue;
                                }
                                planned.assert_retired(what);
                            }
                            assert!(output == words(), "{what}: content");
                        }
                    }
                    planned.assert_retired(what);
                    release.send(()).unwrap();
                    planned.wait_for_tasks(what);
                    for (_, cell, id) in &planned.jobs {
                        assert!(!cell.decoded(*id as usize), "{what}: {id} never decoded");
                    }
                });
            }
        }

        /// A block that straddles two calls' input, whose header the
        /// serial decoder took: the chain the call before ran takes it,
        /// with no new start, and goes on with the blocks after it, and
        /// the frame decodes as `decompress` decodes it.
        #[test]
        fn chain_takes_a_straddling_block() {
            let (frame, header) = small_blocks(&words());
            let mut dec = FrameDecoder::new(&DecodeOptions {
                min_parallel_blocks: 1,
                min_parallel_bytes: 0,
                simd: true,
                window_log_max: 0,
            });
            let mut output = Vec::new();
            let mut out = VecOut {
                output: &mut output,
                prefix: Prefix {
                    start: 0,
                    window: 0,
                },
                dict: &[],
            };
            let started = dec.process(&frame[..header], &mut out, None);
            assert!(matches!(started, Ok(Event::FrameStarted)));
            // Where the 4th, 5th and 9th blocks start.
            let mut ends = vec![header];
            while ends.len() < 10 {
                let at = *ends.last().unwrap();
                let (_, content) = locate_block(&frame[at..], 1 << 10).unwrap();
                ends.push(at + BLOCK_HEADER_LEN + content.len());
            }
            let (pos, after, stop) = (ends[3], ends[4], ends[8]);
            // A room the chain takes three blocks for, and part of the 4th.
            let mut read = 0;
            let cut = &frame[header..pos + BLOCK_HEADER_LEN + 1];
            let decoded =
                dec.decode_blocks_parallel(cut, None, &mut out, 1024, &mut read, |_| true);
            assert!(matches!(decoded, Ok(Some(Event::Continue))));
            assert_eq!(header + read, pos);
            let start = PlannedJobs::of(&dec).start;
            let block = dec.process(&frame[pos..pos + BLOCK_HEADER_LEN], &mut out, None);
            assert!(matches!(block, Ok(Event::Continue)));
            let content = &frame[pos + BLOCK_HEADER_LEN..after];
            let mut read = 0;
            let decoded = dec.decode_held_block_parallel(
                content,
                &frame[after..stop + 1],
                None,
                &mut out,
                &mut read,
                |_| true,
            );
            assert!(matches!(decoded, Ok(Some(Event::Continue))));
            assert_eq!(after + read, stop);
            assert!(Weak::ptr_eq(&start, &PlannedJobs::of(&dec).start));
            let Stage::Block { header: None, .. } = &dec.stage else {
                panic!("at the 9th block's header");
            };
            let mut pos = stop;
            while pos < frame.len() {
                let len = dec.unit_len(&frame[pos..]);
                dec.process(&frame[pos..pos + len], &mut out, None).unwrap();
                pos += len;
            }
            assert!(output == words());
        }

        /// A frame dropped while a task decodes one of its blocks: the
        /// task keeps what it reads, its job and the tables the chain
        /// started from, decodes from them after the frame is gone, and
        /// then drops them.
        #[test]
        fn running_task_outlives_the_frame_on_its_own_copies() {
            on_held_pool(|_release| {
                let (dec, _, _) = one_block_decoded();
                let planned = PlannedJobs::of(&dec);
                let (job, cell, id) = &planned.jobs[0];
                // The task's take.
                assert!(cell.take(*id as usize));
                drop(dec);
                assert!(job.upgrade().is_some(), "the running task's job");
                assert!(planned.start.upgrade().is_some(), "its start tables");
                for (other, _, _) in &planned.jobs[1..] {
                    assert!(other.upgrade().is_none(), "jobs no task started");
                }
                run_taken(cell, *id as usize);
                assert!(cell.decoded(*id as usize));
                assert!(decoded_slot(cell).result.is_ok());
                assert!(job.upgrade().is_none() && planned.start.upgrade().is_none());
            });
        }

        /// A faked sequence: lengths and libzstd OFFBASE (1..=3 a repeat
        /// code, larger values the offset plus `ZSTD_REP_NUM`).
        #[derive(Clone, Copy)]
        struct RawSeq {
            ll: u32,
            ml: u32,
            off_base: u32,
        }

        /// `s` as stage 2 publishes it, after the sequences `reps` follows.
        fn rel(reps: &mut RelReps, s: RawSeq) -> RelSeq {
            let ll0 = usize::from(s.ll == 0);
            let o = match s.off_base as usize {
                1 => reps.code0(ll0),
                b @ 2..=3 => reps.code1(b - 1 + ll0),
                b => reps.new_offset((b - ZSTD_REP_NUM) as u64),
            };
            RelSeq::new(s.ll, s.ml, o)
        }

        /// `decode_sequence`'s repeat-offset update on OFFBASE, from the
        /// repeat offsets `hist`, 0 forced to `usize::MAX`.
        fn absolute(hist: &mut [usize; 3], s: RawSeq) -> usize {
            let off_base = s.off_base as usize;
            if off_base > ZSTD_REP_NUM {
                *hist = [off_base - ZSTD_REP_NUM, hist[0], hist[1]];
                return hist[0];
            }
            let idx = off_base - 1 + usize::from(s.ll == 0);
            if idx == 0 {
                return hist[0];
            }
            let mut o = if idx == 3 {
                hist[0].wrapping_sub(1)
            } else {
                hist[idx]
            };
            if o == 0 {
                o = usize::MAX;
            }
            *hist = [o, hist[0], if idx == 1 { hist[2] } else { hist[1] }];
            o
        }

        /// Stage 2's offsets, relative to the block's starting repeat
        /// offsets, resolve on the executing thread to `decode_sequence`'s:
        /// every repeat code with and without literals, from starting
        /// offsets, from new offsets, through runs of code 3 below a
        /// starting offset, to a code 3 from 1 (corrupt), and the repeat
        /// offsets after the block.
        #[test]
        fn relative_offsets_resolve_as_decode_sequence() {
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = |m: u64| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % m
            };
            let seq = |ll: u32, off_base: u32| RawSeq {
                ll,
                ml: 3,
                off_base,
            };
            let mut blocks: Vec<([u32; 3], Vec<RawSeq>)> = vec![
                // Each code from the starting offsets, then from new ones.
                (
                    [1, 4, 8],
                    [1, 2, 3]
                        .iter()
                        .flat_map(|&b| [seq(5, b), seq(0, b)])
                        .collect(),
                ),
                (
                    [7, 300, 9],
                    [8, 2, 1, 3, 9, 3, 2, 1]
                        .iter()
                        .map(|&b| seq(b & 1, b + 1))
                        .collect(),
                ),
                // Code 3 without literals down a starting offset to 1, then
                // once more (0: corrupt).
                ([5, 6, 7], vec![seq(0, 3); 5]),
                // Only repeat codes: the end stays relative to the start.
                ([10, 20, 30], vec![seq(1, 2), seq(0, 1), seq(1, 3)]),
                // The largest offset a code gives, and a new 1.
                ([2, 3, 4], vec![seq(1, u32::MAX), seq(1, 4), seq(0, 1)]),
            ];
            for _ in 0..200 {
                let start = [0, 0, 0].map(|_| 1 + next(1 << 20) as u32);
                let seqs = (0..64)
                    .map(|_| {
                        let off_base = match next(4) {
                            0 => 4 + next(1 << 16) as u32,
                            _ => 1 + next(3) as u32,
                        };
                        seq(next(2) as u32 * (1 + next(40) as u32), off_base)
                    })
                    .collect();
                blocks.push((start, seqs));
            }
            for (start, seqs) in blocks {
                let base = RelBase::new(start);
                let mut reps = RelReps::start();
                let mut hist = start.map(|h| h as usize);
                for (i, &s) in seqs.iter().enumerate() {
                    let r = rel(&mut reps, s);
                    assert_eq!(r.ll_from & LL_MASK, s.ll, "{start:?} {i}");
                    let want = absolute(&mut hist, s);
                    assert_eq!(base.offset(&r), want, "{start:?} {i}");
                    if want == usize::MAX {
                        break;
                    }
                    let end = hist.map(|h| h as u32);
                    assert_eq!(base.hist(reps), end, "{start:?} after {i}");
                }
            }
        }

        /// `n` sequences of four literals and a match of them, at offset 4:
        /// the sequences, their literals, and what they decode to.
        fn echoes(n: usize) -> (Vec<RawSeq>, Vec<u8>, Vec<u8>) {
            let offset = 4 + ZSTD_REP_NUM as u32;
            let seqs = vec![
                RawSeq {
                    ll: 4,
                    ml: 4,
                    off_base: offset,
                };
                n
            ];
            let literals: Vec<u8> = (0..4 * n).map(|i| (i * 7 + i / 251) as u8).collect();
            let decoded = literals.chunks(4).flat_map(|c| [c, c].concat()).collect();
            (seqs, literals, decoded)
        }

        /// A compressed block of `n` sequences, whose stage 2 the tests
        /// below fake.
        fn block_of(n: usize) -> Vec<u8> {
            // No literals, `n`, predefined tables, a byte of bitstream.
            let mut block = vec![0];
            if n < 128 {
                block.push(n as u8);
            } else {
                block.extend_from_slice(&[(n >> 8) as u8 + 128, n as u8]);
            }
            if n != 0 {
                block.extend_from_slice(&[0, 1]);
            }
            block
        }

        /// The stage 2 of a block, faked: its sequences, its literals, and
        /// whether its offsets are short.
        #[derive(Clone, Copy)]
        struct Fake<'a> {
            seqs: &'a [RawSeq],
            literals: &'a [u8],
            short: bool,
        }

        /// The sequences of `fake` into `slot`, published up to each of
        /// `ends` in turn, calling `step` after each, then ending with
        /// `result`.
        fn fake_sequences(
            slot: &mut Slot,
            publish: &Published,
            fake: Fake<'_>,
            ends: &[usize],
            step: &dyn Fn(),
            result: Result<(), DecodeError>,
        ) -> Result<(), DecodeError> {
            let mut reps = RelReps::start();
            let seqs: Vec<RelSeq> = fake.seqs.iter().map(|&s| rel(&mut reps, s)).collect();
            slot.seqs.clear();
            slot.seqs.reserve(seqs.len());
            let base = slot.seqs.as_mut_ptr();
            publish.start(base, fake.short);
            let mut at = 0;
            for &end in ends {
                for (i, s) in seqs.iter().enumerate().take(end).skip(at) {
                    // SAFETY: reserved.
                    unsafe { base.add(i).write(*s) };
                }
                if end == seqs.len() {
                    publish.decoded_all(end, reps);
                } else {
                    publish.decoded(end);
                }
                at = end;
                step();
            }
            result
        }

        /// The literals `literals` into `lits`, or none and an error unless
        /// `ok`.
        fn fake_literals(lits: &mut LitSlot, literals: &[u8], ok: bool) -> Result<(), DecodeError> {
            lits.literals.clear();
            if !ok {
                return Err("literals".into());
            }
            lits.literals.extend_from_slice(literals);
            lits.literals
                .resize(literals.len() + WILDCOPY_OVERLENGTH, 0);
            Ok(())
        }

        /// A task's stage 2 of `fake` at `at`: `fake_sequences`, and the
        /// literals unless taken, which call `step` and fail unless
        /// `literals_ok`, setting `took`, in the order `at` gives.
        fn fake_stage2(
            slot: &mut Slot,
            at: Decoding<'_>,
            fake: Fake<'_>,
            ends: &[usize],
            step: &dyn Fn(),
            (result, literals_ok): (Result<(), DecodeError>, bool),
            took: &AtomicBool,
        ) -> Result<(), DecodeError> {
            at.stage2(
                || fake_sequences(slot, at.published(), fake, ends, step, result),
                |lits| {
                    step();
                    took.store(true, Ordering::Relaxed);
                    fake_literals(lits, fake.literals, literals_ok)
                },
            )
        }

        /// The executing thread's claim on the literals of ticket 1 in
        /// `cell`: `fake_literals` after `wait`, setting `took`, unless
        /// someone has taken them.
        fn claim<'c>(
            cell: &'c Cell,
            literals: &[u8],
            ok: bool,
            wait: &dyn Fn(),
            took: &AtomicBool,
        ) -> Option<MutexGuard<'c, LitSlot>> {
            cell.literals(1, |lits| {
                wait();
                took.store(true, Ordering::Relaxed);
                fake_literals(lits, literals, ok)
            })
        }

        /// Stage 3 of the block of `n` sequences in `cell`, ticket 1, with
        /// `literals` if the executing thread decoded them, into `output`,
        /// with a dictionary of `dict`: the result, and the repeat offsets
        /// after it, from 1, 4 and 8.
        fn stage3(
            cell: &Cell,
            literals: Option<MutexGuard<'_, LitSlot>>,
            n: usize,
            dict: &[u8],
            output: &mut Vec<u8>,
            simd: Level,
        ) -> (Result<usize, DecodeError>, [u32; 3]) {
            let block = block_of(n);
            let plan = Plan::Compressed(CompressedPlan {
                parts: split_block(&block, BLOCK_SIZE_MAX).unwrap(),
                huf_def: None,
                fse_def: [0; 3],
            });
            let mut out = VecOut {
                output,
                prefix: Prefix {
                    start: 0,
                    window: 1 << 20,
                },
                dict,
            };
            let mut hist = [1, 4, 8];
            let stage2 = Stage2 {
                cell,
                ticket: 1,
                literals,
            };
            let executed = execute_block(
                &plan,
                Some(stage2),
                &mut hist,
                BLOCK_SIZE_MAX,
                &mut out,
                simd,
            );
            (executed.map(<[u8]>::len), hist)
        }

        /// The repeat offsets after `n` sequences of `echoes`.
        fn echoes_hist(n: usize) -> [u32; 3] {
            (0..n.min(3)).fold([1, 4, 8], |h, _| [4, h[0], h[1]])
        }

        /// Runs that end at each of `ends` up to `n`, the last at `n`.
        fn runs_of(n: usize, every: usize) -> Vec<usize> {
            let mut ends: Vec<usize> = (every..n).step_by(every).collect();
            ends.push(n);
            ends
        }

        /// When the executing thread claims a block's literals: before its
        /// task starts, between two of its publishes, or once the task has
        /// taken them.
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Claim {
            Before,
            During,
            After,
        }

        /// Stage 3 executes a block's sequences as stage 2 publishes them,
        /// at once or run by run, around `PUBLISH_EVERY` and past two of
        /// its runs, with stage 2 done before stage 3 starts or still
        /// publishing on another thread, with and without history before
        /// the frame and short-offset copies, and with the literals decoded
        /// once, before or after the sequences, by the task or by the
        /// executing thread, before the task starts or while it publishes:
        /// each way the block decodes as a whole, with the same repeat
        /// offsets.
        #[test]
        fn published_runs_execute_as_the_whole_block() {
            let mut levels = vec![Level::fallback()];
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Level::Avx2(w) = Level::new() {
                levels.push(Level::Avx2(w));
            }
            let every = PUBLISH_EVERY;
            let handoffs = [
                (false, Claim::After),
                (false, Claim::Before),
                (false, Claim::During),
                (true, Claim::Before),
                (true, Claim::During),
                (true, Claim::After),
            ];
            for n in [1, 2, 3, every - 1, every, every + 1, 2 * every + 1, 1000] {
                let (seqs, literals, decoded) = echoes(n);
                let runs = [vec![n], runs_of(n, every), runs_of(n, 1), runs_of(n, 7)];
                for (ends, threaded) in runs.iter().flat_map(|e| [(e, false), (e, true)]) {
                    for (k, &(first, when)) in handoffs.iter().enumerate() {
                        // Every handoff with each level and history, over
                        // the runs.
                        let simd = levels[k % levels.len()];
                        let (dict, short) = [(&[][..], false), (&[9u8; 16][..], true)][k / 3];
                        let what = format!(
                            "{n} in {} runs, threaded {threaded}, sequences first {first}, \
                             {when:?}",
                            ends.len()
                        );
                        let fake = Fake {
                            seqs: &seqs,
                            literals: &literals,
                            short,
                        };
                        let (task_took, mine_took) =
                            (AtomicBool::new(false), AtomicBool::new(false));
                        let cell = Cell::new();
                        cell.hand(1, first);
                        assert!(!cell.startable(1), "{what}: handed");
                        assert!(cell.take(1));
                        let mine =
                            |wait: &dyn Fn()| claim(&cell, &literals, true, wait, &mine_took);
                        let step = || {
                            if threaded {
                                std::thread::sleep(std::time::Duration::from_micros(20));
                            } else if when == Claim::During {
                                drop(mine(&|| ()));
                            }
                        };
                        let decode = || {
                            cell.decode(1, |slot, at| {
                                fake_stage2(slot, at, fake, ends, &step, (Ok(()), true), &task_took)
                            });
                        };
                        let mut output = Vec::new();
                        let (executed, hist) = std::thread::scope(|s| {
                            let mut lits = None;
                            if when == Claim::Before {
                                lits = mine(&|| ());
                                assert!(lits.is_some(), "{what}: claimed first");
                            }
                            if threaded {
                                s.spawn(decode);
                                let waited = |taken: &dyn Fn() -> bool| {
                                    while !taken() && !cell.decoded(1) {
                                        std::thread::yield_now();
                                    }
                                };
                                match when {
                                    Claim::Before => {}
                                    Claim::During => {
                                        waited(&|| {
                                            cell.published.progress.load(Ordering::Acquire) != 0
                                        });
                                        let slow = || {
                                            std::thread::sleep(std::time::Duration::from_micros(
                                                100,
                                            ))
                                        };
                                        lits = mine(&slow);
                                    }
                                    Claim::After => {
                                        waited(&|| cell.lit_claim.load(Ordering::Acquire) & 1 == 1);
                                        lits = mine(&|| ());
                                    }
                                }
                                // Stage 3 waits for the task's literals
                                // itself.
                                while when != Claim::After && !cell.startable(1) {
                                    std::thread::yield_now();
                                }
                            } else {
                                decode();
                                if when == Claim::After {
                                    lits = mine(&|| ());
                                }
                            }
                            stage3(&cell, lits, n, dict, &mut output, simd)
                        });
                        assert_eq!(executed, Ok(8 * n), "{what}");
                        assert!(output == decoded, "{what}: output");
                        assert_eq!(hist, echoes_hist(n), "{what}: offsets");
                        let (task, mine) = (task_took.into_inner(), mine_took.into_inner());
                        assert!(task != mine, "{what}: literals decoded once");
                        // The executing thread decodes them when it comes
                        // first; between publishes, it does only before a
                        // task that does the sequences first takes them.
                        match (when, first, threaded) {
                            (Claim::Before, _, _) | (Claim::During, true, false) => {
                                assert!(mine, "{what}: the executing thread's")
                            }
                            (Claim::After, _, _) | (Claim::During, false, false) => {
                                assert!(task, "{what}: the task's")
                            }
                            (Claim::During, _, true) => {}
                        }
                    }
                }
            }
        }

        /// Stage 3 of a block, from `stage2_then_3`.
        struct Executed {
            result: Result<usize, DecodeError>,
            output: Vec<u8>,
            hist: [u32; 3],
            /// The result of executing the block again, as a chain that
            /// stops at it does.
            again: Result<usize, DecodeError>,
        }

        /// Stage 2 of `fake` in a new cell, its sequences published up to
        /// each of `ends` and ending with `result`, its literals failing
        /// unless `literals_ok`, decoded by the executing thread before the
        /// task starts if `mine`, the sequences first if `first`; then
        /// stage 3 of it after `[5]`, from repeat offsets 1, 4 and 8, as
        /// the executing thread claims the literals before it.
        fn stage2_then_3(
            fake: Fake<'_>,
            ends: &[usize],
            result: Result<(), DecodeError>,
            (literals_ok, mine, first): (bool, bool, bool),
        ) -> Executed {
            let simd = Level::new();
            let took = AtomicBool::new(false);
            let cell = Cell::new();
            cell.hand(1, first);
            assert!(cell.take(1));
            let mut lits = None;
            if mine {
                lits = claim(&cell, fake.literals, literals_ok, &|| (), &took);
                assert!(lits.is_some());
            }
            cell.decode(1, |slot, at| {
                fake_stage2(slot, at, fake, ends, &|| (), (result, literals_ok), &took)
            });
            if lits.is_none() {
                lits = claim(&cell, fake.literals, literals_ok, &|| (), &took);
            }
            assert!(cell.startable(1));
            let n = fake.seqs.len();
            let mut output = vec![5];
            let (result, hist) = stage3(&cell, lits, n, &[], &mut output, simd);
            assert!(cell.literals(1, |_| panic!("decoded again")).is_none());
            let (again, _) = stage3(&cell, None, n, &[], &mut Vec::new(), simd);
            Executed {
                result,
                output,
                hist,
                again,
            }
        }

        /// A stage 2 whose sequences fail, before they publish, after part
        /// of them, or after all of them, or whose literals fail, the
        /// task's or the executing thread's, before or after the sequences,
        /// or that succeeds with a last sequence stage 3 rejects, or that
        /// panics, after part of the sequences or in the literals, the
        /// task's or the executing thread's: stage 3 fails, or resumes the
        /// panic, and neither commits the block nor changes the repeat
        /// offsets, whatever of it executed, and fails again for a chain
        /// that executes it again. A block without sequences commits its
        /// literals once stage 2 is done, and only if all of it succeeded.
        #[test]
        fn failed_stage2_keeps_its_block_out() {
            let simd = Level::new();
            let n = 2 * PUBLISH_EVERY + 1;
            let (seqs, literals, _) = echoes(n);
            let fake = Fake {
                seqs: &seqs,
                literals: &literals,
                short: false,
            };
            let all = runs_of(n, PUBLISH_EVERY);
            let orders = [(false, false), (false, true), (true, false), (true, true)];
            let kept = |what: &str, e: Executed| {
                assert!(e.result.is_err(), "{what}");
                assert_eq!((e.output, e.hist), (vec![5], [1, 4, 8]), "{what}");
                assert!(e.again.is_err(), "{what} again");
            };
            for (mine, first) in orders {
                for ends in [vec![], vec![PUBLISH_EVERY], all.clone()] {
                    let what = format!("sequences failing at {ends:?}, mine {mine}, first {first}");
                    kept(
                        &what,
                        stage2_then_3(fake, &ends, Err("stage 2".into()), (true, mine, first)),
                    );
                }
                let what = format!("literals failing, mine {mine}, first {first}");
                kept(
                    &what,
                    stage2_then_3(fake, &all, Ok(()), (false, mine, first)),
                );
                let mut far = seqs.clone();
                far[n - 1].off_base = (1 << 16) + ZSTD_REP_NUM as u32;
                let rejected = Fake { seqs: &far, ..fake };
                let e = stage2_then_3(rejected, &all, Ok(()), (true, mine, first));
                let what = format!("rejected, mine {mine}, first {first}");
                assert_eq!(
                    e.result,
                    Err("Match offset reaches before the frame start".into()),
                    "{what}"
                );
                assert_eq!((e.output, e.hist), (vec![5], [1, 4, 8]), "{what}");
            }
            let took = AtomicBool::new(false);
            for (first, in_literals) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let what = format!("panic, first {first}, in the literals {in_literals}");
                let cell = Cell::new();
                cell.hand(1, first);
                assert!(cell.take(1));
                cell.decode(1, |slot, at| {
                    at.stage2(
                        || {
                            let ends = if in_literals {
                                &all[..]
                            } else {
                                &[PUBLISH_EVERY]
                            };
                            fake_sequences(slot, at.published(), fake, ends, &|| (), Ok(()))?;
                            if !in_literals {
                                panic!("stage 2");
                            }
                            Ok(())
                        },
                        |lits| {
                            fake_literals(lits, fake.literals, true)?;
                            if in_literals {
                                panic!("stage 2");
                            }
                            Ok(())
                        },
                    )
                });
                let lits = claim(&cell, fake.literals, true, &|| (), &took);
                let mut output = vec![5];
                let mut hist = [1, 4, 8];
                let resumed = panic::catch_unwind(AssertUnwindSafe(|| {
                    (_, hist) = stage3(&cell, lits, n, &[], &mut output, simd);
                }));
                let payload = resumed.expect_err(&what);
                assert_eq!(payload.downcast_ref::<&str>(), Some(&"stage 2"), "{what}");
                assert_eq!((output, hist), (vec![5], [1, 4, 8]), "{what}");
                let (again, _) = stage3(&cell, None, n, &[], &mut Vec::new(), simd);
                assert!(again.is_err(), "{what} again");
            }
            // The executing thread's literals decode panics, unwinding it,
            // and leaves them failed.
            let cell = Cell::new();
            cell.hand(1, true);
            assert!(cell.take(1));
            let resumed = panic::catch_unwind(AssertUnwindSafe(|| {
                drop(cell.literals(1, |_| panic!("literals")));
            }));
            let payload = resumed.expect_err("the literals' panic");
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"literals"));
            let task = AtomicBool::new(false);
            cell.decode(1, |slot, at| {
                fake_stage2(slot, at, fake, &all, &|| (), (Ok(()), true), &task)
            });
            assert!(!task.into_inner(), "the literals taken");
            let mut output = vec![5];
            let (executed, hist) = stage3(&cell, None, n, &[], &mut output, simd);
            assert!(executed.is_err(), "panicked literals");
            assert_eq!((output, hist), (vec![5], [1, 4, 8]), "panicked literals");
            let none = Fake {
                seqs: &[],
                literals: &literals[..100],
                short: false,
            };
            for (sequences_ok, literals_ok, mine) in [
                (true, true, false),
                (true, true, true),
                (false, true, false),
                (true, false, false),
                (true, false, true),
            ] {
                let ok = sequences_ok && literals_ok;
                let result = if sequences_ok {
                    Ok(())
                } else {
                    Err("stage 2".into())
                };
                let e = stage2_then_3(none, &[], result, (literals_ok, mine, mine));
                let what = format!("no sequences, {sequences_ok} {literals_ok} {mine}");
                assert_eq!((e.result.is_ok(), e.again.is_ok()), (ok, ok), "{what}");
                let committed = [&[5][..], if ok { none.literals } else { &[] }].concat();
                assert_eq!((e.output, e.hist), (committed, [1, 4, 8]), "{what}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `copy_match` against a byte-at-a-time copy at every offset up to
    /// 200 (past 128 `wide_chunks` turns on the offset mod 32) and every
    /// length up to 150, with the bytes before the match both fewer and
    /// more than the 32 that `copy_match_short` loads.
    fn check_copy_match<W: WildCopy>(w: W, name: &str) {
        let init: Vec<u8> = (0..200u8).map(|i| i.wrapping_mul(37) ^ 0x5a).collect();
        for avail in [1, 2, 7, 8, 15, 16, 31, 32, 33, 47, 64, 100, 200] {
            for offset in 1..=avail {
                for ml in 1..=150 {
                    let mut buf = vec![0xEEu8; avail + ml + 31];
                    buf[..avail].copy_from_slice(&init[..avail]);
                    let mut want = buf.clone();
                    for i in avail..avail + ml {
                        want[i] = want[i - offset];
                    }
                    // SAFETY: `1 <= offset <= avail`, `avail` initialized
                    // bytes before `dst` and `ml + 31` bytes after it.
                    unsafe { w.copy_match(buf.as_mut_ptr().add(avail), offset, ml, avail) };
                    assert!(
                        buf[..avail + ml] == want[..avail + ml],
                        "{name}: avail {avail} offset {offset} ml {ml}"
                    );
                }
            }
        }
    }

    extern "C" {
        // lib/common/huf.h; linked from zstd-sys's static libzstd.
        fn HUF_readStats(
            huff_weight: *mut u8,
            hw_size: usize,
            rank_stats: *mut u32,
            nb_symbols: *mut u32,
            table_log: *mut u32,
            src: *const u8,
            src_size: usize,
        ) -> usize;
        fn HUF_buildCTable_wksp(
            tree: *mut usize,
            count: *const u32,
            max_symbol_value: u32,
            max_nb_bits: u32,
            workspace: *mut u64,
            wksp_size: usize,
        ) -> usize;
        fn HUF_writeCTable_wksp(
            dst: *mut u8,
            max_dst_size: usize,
            ctable: *const usize,
            max_symbol_value: u32,
            huff_log: u32,
            workspace: *mut u64,
            wksp_size: usize,
        ) -> usize;
        fn HUF_compress1X_usingCTable(
            dst: *mut u8,
            dst_size: usize,
            src: *const u8,
            src_size: usize,
            ctable: *const usize,
            flags: i32,
        ) -> usize;
        fn HUF_compress4X_usingCTable(
            dst: *mut u8,
            dst_size: usize,
            src: *const u8,
            src_size: usize,
            ctable: *const usize,
            flags: i32,
        ) -> usize;
    }

    /// (description length, weights with the implied last one, rank
    /// statistics of weights 0 to 12, table log) of a tree description.
    type HufStats = (usize, Vec<u8>, Vec<u32>, u32);

    fn huf_stats_c(src: &[u8]) -> Option<HufStats> {
        let mut weights = [0u8; 256];
        let mut rank_stats = [0u32; 13];
        let (mut nb_symbols, mut table_log) = (0u32, 0u32);
        // SAFETY: the buffers have the sizes HUF_readStats is given (rankStats
        // takes libzstd's HUF_TABLELOG_MAX + 1 = 13 entries).
        let (r, error) = unsafe {
            let r = HUF_readStats(
                weights.as_mut_ptr(),
                weights.len(),
                rank_stats.as_mut_ptr(),
                &mut nb_symbols,
                &mut table_log,
                src.as_ptr(),
                src.len(),
            );
            (r, zstd::zstd_safe::zstd_sys::ZSTD_isError(r) != 0)
        };
        if error {
            return None;
        }
        let n = nb_symbols as usize;
        Some((r, weights[..n].to_vec(), rank_stats.to_vec(), table_log))
    }

    fn huf_stats_ours(src: &[u8]) -> Option<HufStats> {
        let mut t = HuffmanTable::new();
        let (used, nb_weights) = t.read_weights(src).ok()?;
        t.weight_stats(nb_weights).ok()?;
        let n = t.nb_symbols;
        // libzstd's rankStats has a slot for weight 12 too.
        let mut rank_stats = t.rank_stats.to_vec();
        rank_stats.resize(13, 0);
        Some((
            used,
            t.weights[..n].to_vec(),
            rank_stats,
            u32::from(t.max_num_bits),
        ))
    }

    /// libzstd's tree description of a code for `counts` of at most
    /// `max_bits` bits; `None` when HUF_writeCTable cannot write it (more
    /// than 128 symbols whose weights FSE does not compress).
    fn huf_description_c(counts: &[u32], max_bits: u32) -> Option<Vec<u8>> {
        let max_sv = counts.len() as u32 - 1;
        let mut ctable = [0usize; 258];
        let mut wksp = [0u64; 2048];
        let mut out = [0u8; 256];
        // SAFETY: `ctable` holds HUF_CTABLE_SIZE_ST(255) entries and `wksp`
        // exceeds HUF_WORKSPACE_SIZE.
        unsafe {
            let bits = HUF_buildCTable_wksp(
                ctable.as_mut_ptr(),
                counts.as_ptr(),
                max_sv,
                max_bits,
                wksp.as_mut_ptr(),
                wksp.len() * 8,
            );
            assert_eq!(zstd::zstd_safe::zstd_sys::ZSTD_isError(bits), 0);
            let n = HUF_writeCTable_wksp(
                out.as_mut_ptr(),
                out.len(),
                ctable.as_ptr(),
                max_sv,
                bits as u32,
                wksp.as_mut_ptr(),
                wksp.len() * 8,
            );
            (zstd::zstd_safe::zstd_sys::ZSTD_isError(n) == 0).then(|| out[..n].to_vec())
        }
    }

    /// libzstd's code of at most `max_bits` bits for `symbols`, and a
    /// literals section of them: the tree description followed by one or
    /// four streams (HUF_compress1X/4X_usingCTable). `None` when libzstd
    /// does not compress them.
    fn huf_section_c(symbols: &[u8], max_bits: u32, four: bool) -> Option<Vec<u8>> {
        let mut counts = [0u32; 256];
        for &s in symbols {
            counts[usize::from(s)] += 1;
        }
        let max_sv = counts.iter().rposition(|&c| c > 0).unwrap() as u32;
        let mut ctable = [0usize; 258];
        let mut wksp = [0u64; 2048];
        let mut out = vec![0u8; 256 + 2 * symbols.len() + 64];
        // SAFETY: `ctable` holds HUF_CTABLE_SIZE_ST(255) entries, `wksp`
        // exceeds HUF_WORKSPACE_SIZE and the lengths are the buffers' own.
        unsafe {
            let bits = HUF_buildCTable_wksp(
                ctable.as_mut_ptr(),
                counts.as_ptr(),
                max_sv,
                max_bits,
                wksp.as_mut_ptr(),
                wksp.len() * 8,
            );
            assert_eq!(zstd::zstd_safe::zstd_sys::ZSTD_isError(bits), 0);
            let n = HUF_writeCTable_wksp(
                out.as_mut_ptr(),
                256,
                ctable.as_ptr(),
                max_sv,
                bits as u32,
                wksp.as_mut_ptr(),
                wksp.len() * 8,
            );
            if zstd::zstd_safe::zstd_sys::ZSTD_isError(n) != 0 {
                return None;
            }
            let compress = if four {
                HUF_compress4X_usingCTable
            } else {
                HUF_compress1X_usingCTable
            };
            let m = compress(
                out.as_mut_ptr().add(n),
                out.len() - n,
                symbols.as_ptr(),
                symbols.len(),
                ctable.as_ptr(),
                0,
            );
            if m == 0 || zstd::zstd_safe::zstd_sys::ZSTD_isError(m) != 0 {
                return None;
            }
            out.truncate(n + m);
        }
        Some(out)
    }

    /// Decode `section` into `dst_size` literals with the single-symbol
    /// (`x2` false) or the double-symbol table, whatever HUF_selectDecoder
    /// would pick.
    fn huf_decode_forced(
        section: &[u8],
        dst_size: usize,
        four: bool,
        x2: bool,
    ) -> Result<Vec<u8>, DecodeError> {
        let mut t = HuffmanTable::new();
        let (used, nb_weights) = t.read_weights(section)?;
        t.weight_stats(nb_weights)?;
        t.fill(x2);
        let mut out = vec![0u8; dst_size];
        huf_decompress(&mut out, &section[used..], four, &t)?;
        Ok(out)
    }

    /// `read_le_short` reads every length from 1 to 8 bytes as the bytes
    /// zero-padded to 8 read.
    #[test]
    fn read_le_short_matches_padded_read() {
        let bytes = [0x11, 0x82, 0x23, 0xF4, 0x45, 0x96, 0x67, 0xA8];
        for n in 1..=8 {
            let mut padded = [0u8; 8];
            padded[..n].copy_from_slice(&bytes[..n]);
            assert_eq!(
                read_le_short(&bytes[..n]),
                u64::from_le_bytes(padded),
                "{n} bytes"
            );
        }
    }

    /// RFC 8878 §4.2.2: a Huffman stream is consumed exactly, so the
    /// verdict and bytes depend on the section alone, not on whether X1 or
    /// X2 decodes it (R1-6, R2-3). Over libzstd sections of one and four
    /// streams, short (plain loops) and long (fast loops): one literal
    /// fewer leaves a symbol undecoded and one more runs a stream dry, and
    /// both decoders reject both; random byte and size mutations get the
    /// same verdict and output from both.
    #[test]
    fn huf_x1_x2_same_verdict() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let both = |section: &[u8], dst_size: usize, four: bool| {
            let x1 = huf_decode_forced(section, dst_size, four, false);
            let x2 = huf_decode_forced(section, dst_size, four, true);
            assert_eq!(
                x1.is_ok(),
                x2.is_ok(),
                "four {four} size {dst_size}: X1 {:?} X2 {:?}",
                x1.as_ref().err(),
                x2.as_ref().err()
            );
            if let (Ok(a), Ok(b)) = (&x1, &x2) {
                assert!(a == b, "four {four} size {dst_size}: bytes differ");
            }
            x1.ok()
        };
        let (mut sections, mut mutants_ok, mut fast) = (0, 0, 0);
        for case in 0..1500 {
            let n = match case % 5 {
                0 => 1 + rand() as usize % 40,
                1 => 40 + rand() as usize % 200,
                _ => 200 + rand() as usize % 6000,
            };
            let alphabet = 2 + rand() % 60;
            let skew = rand() % 4;
            let symbols: Vec<u8> = (0..n)
                .map(|_| {
                    let mut v = rand() % alphabet;
                    for _ in 0..skew {
                        v = v.min(rand() % alphabet);
                    }
                    v as u8
                })
                .collect();
            // A code needs two symbols; libzstd sends one as RLE.
            if symbols.iter().all(|&s| s == symbols[0]) {
                continue;
            }
            let four = case % 2 == 1;
            let max_bits = 6 + rand() % 6;
            let Some(section) = huf_section_c(&symbols, max_bits, four) else {
                continue;
            };
            sections += 1;
            let segment = n.div_ceil(4);
            fast += usize::from(four && n > 3 * segment && section.len() > 300);
            assert_eq!(both(&section, n, four).as_deref(), Some(&symbols[..]));
            // A 4-stream size change keeps the segments only while
            // ceil(size / 4) stays the same.
            if !four || (n - 1).div_ceil(4) == segment {
                assert_eq!(both(&section, n - 1, four), None, "extra symbol");
            }
            if !four || (n + 1).div_ceil(4) == segment {
                assert_eq!(both(&section, n + 1, four), None, "exhausted");
            }
            for _ in 0..20 {
                let mut bad = section.clone();
                let pos = rand() as usize % bad.len();
                if rand() % 2 == 0 {
                    bad[pos] ^= 1 << (rand() % 8);
                } else {
                    bad[pos] = rand() as u8;
                }
                let size = (n as i64 + i64::from(rand() % 7) - 3).max(1) as usize;
                mutants_ok += usize::from(both(&bad, size, four).is_some());
            }
        }
        assert!(
            sections > 1200 && fast > 300 && mutants_ok > 1000,
            "{sections} sections, {fast} fast, {mutants_ok} mutants accepted"
        );
    }

    /// 4-stream sections of every Regenerated_Size up to 9 under a code of
    /// two 1-bit symbols: the RFC's split ((size + 3) / 4 bytes for the
    /// first three streams, the rest for the last) is valid exactly when
    /// the rest is not negative, so sizes 1, 2 and 5 fail and 0, 3 and 4
    /// decode, under X1 and X2 alike (R2-6).
    #[test]
    fn huf_four_streams_follow_rfc_split() {
        // One raw weight: symbol 0 of weight 1, the implied symbol 1 too.
        let desc = [128u8, 0x10];
        for size in 0..=9usize {
            let segment = size.div_ceil(4);
            let last = size as isize - 3 * segment as isize;
            let want: Vec<u8> = (0..size).map(|i| (i % 3 == 1) as u8).collect();
            let mut streams: Vec<Vec<u8>> = Vec::new();
            for s in 0..4 {
                let len = if s < 3 { segment } else { last.max(0) as usize };
                let begin = (s * segment).min(size);
                // The end mark, then each symbol's 1-bit code, read from
                // the top bit down.
                let mut byte = 1u8;
                for &b in &want[begin..(begin + len).min(size)] {
                    byte = byte << 1 | b;
                }
                streams.push(vec![byte]);
            }
            let mut section = desc.to_vec();
            for st in &streams[..3] {
                section.extend_from_slice(&(st.len() as u16).to_le_bytes());
            }
            for st in &streams {
                section.extend_from_slice(st);
            }
            for x2 in [false, true] {
                let got = huf_decode_forced(&section, size, true, x2);
                if last < 0 {
                    assert!(got.is_err(), "size {size} x2 {x2}");
                } else {
                    assert_eq!(got.as_deref(), Ok(&want[..]), "size {size} x2 {x2}");
                }
            }
        }
    }

    /// `read_weights` + `weight_stats` against HUF_readStats on libzstd's
    /// descriptions of random codes (raw and FSE-compressed, 2 to 256
    /// symbols, 6- to 12-bit), every truncation of them, single-byte
    /// corruptions and random bytes: the same outcome, and on success the
    /// same length, weights, statistics and table log, except for two
    /// kinds libzstd accepts and we reject: descriptions of 12-bit codes
    /// (RFC 8878 §4.2.1 caps codes at 11 bits; R2-7), and weight FSE tables
    /// listing a symbol past 11, the highest weight (R3-2). Hundreds of the
    /// corrupted and random inputs are valid descriptions, over 150
    /// describe 12-bit codes, and over 10 list such a symbol.
    #[test]
    fn huf_stats_match_libzstd() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        // The FSE-compressed weights of `src` list a symbol past 11.
        let wide = |src: &[u8]| {
            let header = usize::from(src[0]);
            header < 128
                && src.get(1..1 + header).is_some_and(|desc| {
                    parse_fse_header(desc, 6).is_ok_and(|(_, counts, _)| counts.len() > 12)
                })
        };
        let (mut log12, mut wide_tables) = (0, 0);
        let mut check = |src: &[u8]| {
            let ours = huf_stats_ours(src);
            let c = huf_stats_c(src);
            if matches!(c, Some((.., 12))) {
                assert_eq!(ours, None, "input {src:02x?}");
                log12 += 1;
            } else if c.is_some() && wide(src) {
                assert_eq!(ours, None, "input {src:02x?}");
                wide_tables += 1;
            } else {
                assert_eq!(ours, c, "input {src:02x?}");
            }
            usize::from(ours.is_some())
        };
        let (mut raw, mut written, mut bad_ok, mut random_ok) = (0, 0, 0, 0);
        for case in 0..3000 {
            let nb = 2 + rand() as usize % 255;
            let skew = rand() % 20;
            let counts: Vec<u32> = (0..nb)
                .map(|_| 1 + (rand() >> (31 - skew % 31)) % 5000)
                .collect();
            // HUF_minTableLog: at least enough bits for `nb` leaves.
            let max_bits = (6 + case % 7).max(highest_bit_set(nb as u32) + 1);
            let Some(desc) = huf_description_c(&counts, max_bits) else {
                continue;
            };
            written += 1;
            raw += usize::from(desc[0] >= 128);
            for len in 0..=desc.len() {
                check(&desc[..len]);
            }
            let mut bad = desc.clone();
            let pos = rand() as usize % desc.len();
            bad[pos] ^= 1 << (rand() % 8);
            bad_ok += check(&bad);
            bad[pos] = rand() as u8;
            bad_ok += check(&bad);
        }
        assert!(raw > 100 && written - raw > 1000, "{raw} raw of {written}");
        for _ in 0..200_000 {
            let len = 1 + rand() as usize % 40;
            let mut src: Vec<u8> = (0..len).map(|_| rand() as u8).collect();
            src[0] %= 1 + len as u8;
            random_ok += check(&src);
        }
        assert!(
            bad_ok > 500 && random_ok > 1000 && log12 > 150 && wide_tables > 10,
            "{bad_ok} {random_ok} accepted, {log12} of 12 bits, {wide_tables} wide"
        );
    }

    /// Raw 4-bit weights (an odd number of them, and those of an 11-bit
    /// code) through the statistics and both fills, each build over the
    /// previous one: the tables have 11 bits, each symbol
    /// of weight `w` owns `1 << (w - 1 + rescale)` single-symbol cells of
    /// `max_bits + 1 - w` bits, in weight order, and each double-symbol
    /// cell holds the single-symbol lookup of its index plus, exactly when
    /// both fit in the table log, the lookup that follows it.
    #[test]
    fn raw_weights_fill_both_tables() {
        // Weights 3 3 2 2 1 0 1 sum to 14 halves; the implied eighth is 2.
        let short: (&[u8], &[u8], u32) = (
            &[127 + 7, 0x33, 0x22, 0x10, 0x10],
            &[3, 3, 2, 2, 1, 0, 1, 2],
            4,
        );
        // Weights 11 down to 1 sum to 2047 halves; the implied 12th is 1.
        let long: (&[u8], &[u8], u32) = (
            &[127 + 11, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10],
            &[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1],
            11,
        );
        let mut t = HuffmanTable::new();
        // Weights 12 down to 1 describe a 12-bit code (R2-7).
        let (_, nb_weights) = t
            .read_weights(&[127 + 12, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21])
            .unwrap();
        assert!(t.weight_stats(nb_weights).is_err());
        for (src, weights, max_bits) in [short, long, short] {
            let n = weights.len();
            let (used, nb_weights) = t.read_weights(src).unwrap();
            assert_eq!((used, nb_weights), (src.len(), n - 1));
            t.weight_stats(nb_weights).unwrap();
            assert_eq!((t.nb_symbols, u32::from(t.max_num_bits)), (n, max_bits));
            assert_eq!(&t.weights[..n], weights);
            t.fill_x1();
            t.fill_x2();
            let log = HUF_FAST_TABLE_LOG;
            let x1 = &t.decode[..];
            assert_eq!((x1.len(), t.decode_x2.len()), (1 << log, 1 << log));
            let mut cells = vec![0usize; n];
            let mut last_key = (0, 0);
            for e in x1 {
                let w = u32::from(weights[usize::from(e.symbol)]);
                assert_eq!(u32::from(e.num_bits), max_bits + 1 - w);
                assert!((w, e.symbol) >= last_key);
                last_key = (w, e.symbol);
                cells[usize::from(e.symbol)] += 1;
            }
            for (s, &w) in weights.iter().enumerate() {
                let w = u32::from(w);
                let want = if w == 0 {
                    0
                } else {
                    1 << (w - 1 + log - max_bits)
                };
                assert_eq!(cells[s], want, "symbol {s}");
            }
            for (i, x) in t.decode_x2.iter().enumerate() {
                let first = x1[i];
                let next = x1[(i << first.num_bits) & ((1 << log) - 1)];
                let both = u32::from(first.num_bits + next.num_bits) <= log;
                assert_eq!(x.length, if both { 2 } else { 1 }, "cell {i}");
                assert_eq!(x.sequence as u8, first.symbol, "cell {i}");
                if both {
                    assert_eq!((x.sequence >> 8) as u8, next.symbol, "cell {i}");
                    assert_eq!(x.nb_bits, first.num_bits + next.num_bits, "cell {i}");
                } else {
                    assert_eq!(x.nb_bits, first.num_bits, "cell {i}");
                }
            }
        }
    }

    /// Both tables of libzstd's descriptions of random codes (raw and
    /// FSE-compressed, 2 to 256 symbols, some of weight 0, 6- to 11-bit),
    /// each build over the previous one, cell for cell against their
    /// definition: single-symbol cells in order of weight, then symbol, a
    /// symbol of weight `w` owning `1 << (w - 1 + rescale)` of them; a
    /// double-symbol cell is the single-symbol lookup of its index plus,
    /// exactly when both fit in the table log, the lookup that follows it.
    #[test]
    fn huf_tables_match_definition() {
        let mut seed = 0x6a09_e667_f3bc_c908u64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let log = HUF_FAST_TABLE_LOG;
        let mut t = HuffmanTable::new();
        let (mut raw, mut zero, mut full) = (0, 0, 0);
        for case in 0..2000 {
            let nb = 2 + rand() as usize % 255;
            let skew = rand() % 20;
            let sparse = rand() % 4;
            let mut counts: Vec<u32> = (0..nb)
                .map(|_| {
                    let c = 1 + (rand() >> (31 - skew % 31)) % 5000;
                    if rand() % 4 < sparse {
                        0
                    } else {
                        c
                    }
                })
                .collect();
            // Two symbols at least, the last one among them.
            counts[0] = counts[0].max(1);
            counts[nb - 1] = counts[nb - 1].max(1);
            let max_bits = (6 + case % 6).max(highest_bit_set(nb as u32) + 1);
            let Some(desc) = huf_description_c(&counts, max_bits) else {
                continue;
            };
            let (_, nb_weights) = t.read_weights(&desc).unwrap();
            t.weight_stats(nb_weights).unwrap();
            let max = u32::from(t.max_num_bits);
            let weights = &t.weights[..t.nb_symbols];
            raw += usize::from(desc[0] >= 128);
            zero += usize::from(weights.contains(&0));
            full += usize::from(weights.len() == 256);

            let mut order: Vec<usize> = (0..weights.len()).filter(|&s| weights[s] != 0).collect();
            order.sort_by_key(|&s| weights[s]);
            let mut x1 = Vec::new();
            for s in order {
                let w = u32::from(weights[s]);
                let cell = (s as u8, (max + 1 - w) as u8);
                x1.extend(std::iter::repeat_n(cell, 1 << (w - 1 + log - max)));
            }
            let x2: Vec<(u16, u8, u8)> = (0..1 << log)
                .map(|i| {
                    let (a, a_bits) = x1[i];
                    let (b, b_bits) = x1[(i << a_bits) & ((1 << log) - 1)];
                    if u32::from(a_bits + b_bits) <= log {
                        (u16::from(a) | u16::from(b) << 8, a_bits + b_bits, 2)
                    } else {
                        (u16::from(a), a_bits, 1)
                    }
                })
                .collect();

            t.fill_x1();
            let got: Vec<(u8, u8)> = t.decode.iter().map(|e| (e.symbol, e.num_bits)).collect();
            assert_eq!(got, x1, "case {case}: {desc:02x?}");
            t.fill_x2();
            let got: Vec<(u16, u8, u8)> = t
                .decode_x2
                .iter()
                .map(|e| (e.sequence, e.nb_bits, e.length))
                .collect();
            assert_eq!(got, x2, "case {case}: {desc:02x?}");
        }
        assert!(
            raw > 100 && zero > 500 && full > 3,
            "{raw} raw, {zero} with weight 0, {full} of 256"
        );
    }

    /// A table rebuilt over a larger one, an RLE one or one with -1 counts
    /// equals the same table built fresh: builds clear no cells, so none of
    /// the previous table's may show through.
    #[test]
    fn rebuilt_table_equals_fresh_build() {
        let ll = &SEQ_TABLES[0];
        let codes = Some((ll.base, ll.bits));
        let cells = |t: &FSETable| {
            t.decode()
                .iter()
                .map(|e| (e.next_state, e.num_bits, e.extra_bits, e.base_value))
                .collect::<Vec<_>>()
        };
        let mut wide = vec![-1i32; 36];
        wide[0] = 512 - 35;
        let mut narrow = vec![0i32; 36];
        narrow[1] = 20;
        narrow[7] = 12;
        let dists: [(u8, &[i32]); 4] = [
            (9, &wide),
            (5, &narrow),
            (ll.default_log, ll.default_distribution),
            (5, &narrow),
        ];
        let mut reused = FSETable::new(MAX_LITERAL_LENGTH_CODE);
        for (i, (log, probs)) in dists.into_iter().enumerate() {
            if i == 3 {
                reused.build_rle(3, ll.base, ll.bits);
            }
            reused.build_from_probabilities(log, probs, codes).unwrap();
            let mut fresh = FSETable::new(MAX_LITERAL_LENGTH_CODE);
            fresh.build_from_probabilities(log, probs, codes).unwrap();
            assert_eq!(reused.decode().len(), 1 << log);
            assert!(cells(&reused) == cells(&fresh), "build {i}");
        }
    }

    /// Predefined_Mode selects the shared `PREDEFINED_TABLES` and builds
    /// nothing; Repeat_Mode keeps the table selected, the dictionary's,
    /// predefined or own, and fails while there is none, as in a new frame
    /// without a dictionary.
    #[test]
    fn predefined_mode_selects_shared_tables() {
        let mut fse = FSEScratch::new();
        let mut dict = FSEScratch::new();
        for t in 0..3 {
            build_sequence_table(ModeType::RLE, &[0], &mut dict, t).unwrap();
            let predefined: *const FSETable = &PREDEFINED_TABLES[t];
            let own: *const FSETable = fse.own(t);
            // As `DecoderScratch::load_dict` starts a frame.
            fse.source[t] = SeqTableSource::Dict;
            assert_eq!(
                build_sequence_table(ModeType::Repeat, &[], &mut fse, t),
                Ok(0)
            );
            assert!(
                ptr::eq(fse.table(t, Some(&dict)), dict.own(t)),
                "dictionary's"
            );
            build_sequence_table(ModeType::Predefined, &[], &mut fse, t).unwrap();
            assert!(fse.own(t).decode().is_empty(), "built nothing");
            // Description length and the table selected after a block.
            let mut block = |mode, src: &[u8]| {
                let len = build_sequence_table(mode, src, &mut fse, t)?;
                Ok::<_, String>((len, fse.table(t, Some(&dict)) as *const FSETable))
            };
            assert_eq!(block(ModeType::Repeat, &[]), Ok((0, predefined)));
            assert_eq!(block(ModeType::RLE, &[1]), Ok((1, own)));
            assert_eq!(block(ModeType::Repeat, &[]), Ok((0, own)));
            assert_eq!(block(ModeType::Predefined, &[]), Ok((0, predefined)));
            assert_eq!(block(ModeType::Repeat, &[]), Ok((0, predefined)));
            fse.reset();
            assert!(build_sequence_table(ModeType::Repeat, &[], &mut fse, t).is_err());
        }
    }

    /// `build_from_probabilities` refuses what `read_ncount_body` refuses
    /// in a table description: counts that do not tile the table, counts
    /// below -1, too many symbols and accuracy logs out of range.
    #[test]
    fn build_from_probabilities_checks_counts() {
        let mut t = FSETable::new(MAX_OFFSET_CODE);
        // Each failing case breaks one rule and keeps the others.
        let mut probs = vec![1i32; 32];
        assert!(t.build_from_probabilities(5, &probs, None).is_ok());
        probs[0] = 2;
        assert!(t.build_from_probabilities(5, &probs, None).is_err());
        assert!(t.decode().is_empty());
        probs[0] = -2;
        probs[1] = 0;
        assert!(t.build_from_probabilities(5, &probs, None).is_err());
        let mut many = vec![-1i32; 32];
        assert!(t.build_from_probabilities(5, &many, None).is_ok());
        many.insert(0, 0);
        assert!(t.build_from_probabilities(5, &many, None).is_err());
        assert!(t.build_from_probabilities(4, &[-1; 16], None).is_err());
        let mut big = vec![0i32; 32];
        big[0] = 1024;
        assert!(t.build_from_probabilities(10, &big, None).is_err());
    }

    /// `short_offset_share` from the counts equals a scan of the cells, for
    /// RLE tables, the predefined table and built ones with -1 counts.
    #[test]
    fn short_offset_share_counts_cells() {
        let of = &SEQ_TABLES[1];
        let codes = Some((of.base, of.bits));
        let scan = |t: &FSETable| {
            let short = t
                .decode()
                .iter()
                .filter(|e| (2..=4).contains(&e.extra_bits))
                .count();
            short * 256 / t.decode().len()
        };
        let mut t = FSETable::new(MAX_OFFSET_CODE);
        for code in 0..=MAX_OFFSET_CODE {
            t.build_rle(code, of.base, of.bits);
            assert_eq!(short_offset_share(&t), scan(&t), "RLE {code}");
        }
        let predefined = &PREDEFINED_TABLES[1];
        assert_eq!(
            short_offset_share(predefined),
            scan(predefined),
            "predefined"
        );
        let mut probs = vec![-1i32; 32];
        probs[2] = 100;
        probs[3] = 60;
        probs[5] = 67;
        t.build_from_probabilities(8, &probs, codes).unwrap();
        assert_eq!(short_offset_share(&t), scan(&t), "built");
        assert_eq!(short_offset_share(&t), 100 + 60 + 1);
    }

    /// Fails with the name of the copy type it is executed with.
    struct CopyProbe;

    impl BlockSequences for CopyProbe {
        unsafe fn execute<W: WildCopy>(
            self,
            _: W,
            _: &mut [u32; 3],
            _: Dst,
        ) -> Result<usize, DecodeError> {
            let name = std::any::type_name::<W>();
            Err(name.rsplit("::").next().unwrap_or(name).into())
        }
    }

    /// `execute_with_copies` runs the `ShortOffsets` copies from a share of
    /// `SHORT_OFFSET_SHARE_MIN` on and the plain ones below it, on the
    /// portable level and on AVX2.
    #[test]
    fn execute_with_copies_follows_offsets_table() {
        assert_eq!(SHORT_OFFSET_SHARE_MIN, 32);
        let of = &SEQ_TABLES[1];
        let codes = Some((of.base, of.bits));
        let mut levels = vec![(Level::fallback(), "Fallback", "FallbackShortOffsets")];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Level::Avx2(w) = Level::new() {
            levels.push((Level::Avx2(w), "Avx2", "Avx2ShortOffsets"));
        }
        let pick = |level, t: &FSETable| {
            let prefix = Prefix {
                start: 0,
                window: usize::MAX,
            };
            let mut out = Vec::new();
            let dst = prefix.dst(&mut out, MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH);
            // SAFETY: `Prefix::dst` meets the `Dst` contract.
            let e = unsafe { execute_with_copies(level, t, CopyProbe, &mut [1, 4, 8], 0, dst) };
            String::from(e.unwrap_err())
        };
        let mut t = FSETable::new(MAX_OFFSET_CODE);
        for (level, plain, short) in levels {
            for (share, want) in [(31, plain), (32, short), (0, plain), (256, short)] {
                let mut probs = vec![0i32; 11];
                probs[3] = share;
                probs[10] = 256 - share;
                t.build_from_probabilities(8, &probs, codes).unwrap();
                assert_eq!(short_offset_share(&t), share as usize);
                assert_eq!(pick(level, &t), want, "share {share}");
            }
            t.build_rle(3, of.base, of.bits);
            assert_eq!(pick(level, &t), short, "RLE 3");
            t.build_rle(10, of.base, of.bits);
            assert_eq!(pick(level, &t), plain, "RLE 10");
        }
    }

    /// Compressed block of one sequence in RLE-mode tables: no literals,
    /// then `ml` bytes (3..=34) copied from `offset` back.
    fn one_match_block(offset: u32, ml: u8) -> Vec<u8> {
        let value = offset + 3;
        let code = 31 - value.leading_zeros();
        let mut body = vec![0x00, 1, 0x54, 0, code as u8, ml - 3];
        let stream = (1u64 << code) | u64::from(value - (1 << code));
        body.extend_from_slice(&stream.to_le_bytes()[..code as usize / 8 + 1]);
        body
    }

    /// A match reaches back through the segment into `ExtHistory` up to
    /// its first byte and no further, wholly inside it (ending
    /// `WILDCOPY_OVERLENGTH` bytes or more before its end, or nearer) or
    /// running on into the segment, still bounded by Window_Size; without history it stops
    /// at the segment, as in one-shot decoding. Into a dictionary it may
    /// reach past Window_Size, while the segment before it holds at most
    /// Window_Size bytes. Each rejection names the bound it crossed.
    #[test]
    fn match_reaches_into_ext_history() {
        let ext: Vec<u8> = (0..64).collect();
        let seg: Vec<u8> = (100..110).collect();
        let mut levels = vec![Level::fallback()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Level::Avx2(w) = Level::new() {
            levels.push(Level::Avx2(w));
        }
        const OK: Option<&str> = None;
        const FRAME: Option<&str> = Some("Match offset reaches before the frame start");
        const DICT: Option<&str> = Some("Match offset reaches before the dictionary start");
        const WINDOW: Option<&str> = Some("Match offset exceeds Window_Size");
        // (offset, match length, ext bytes, window, error from history,
        // from a dictionary); the match follows the 10 segment bytes.
        let cases = [
            (10, 4, 64, 1 << 20, OK, OK),
            (11, 4, 64, 1 << 20, OK, OK),
            (11, 30, 64, 1 << 20, OK, OK),
            (20, 4, 64, 1 << 20, OK, OK),
            (20, 34, 64, 1 << 20, OK, OK),
            (45, 4, 64, 1 << 20, OK, OK),
            (46, 4, 64, 1 << 20, OK, OK),
            (71, 30, 64, 1 << 20, OK, OK),
            (72, 30, 64, 1 << 20, OK, OK),
            (74, 4, 64, 1 << 20, OK, OK),
            (74, 34, 64, 1 << 20, OK, OK),
            (75, 4, 64, 1 << 20, FRAME, DICT),
            (11, 4, 0, 1 << 20, FRAME, FRAME),
            (11, 4, 1, 1 << 20, OK, OK),
            (12, 4, 1, 1 << 20, FRAME, DICT),
            (60, 4, 64, 60, OK, OK),
            (60, 4, 64, 59, WINDOW, OK),
            (11, 4, 64, 10, WINDOW, OK),
            (74, 4, 64, 10, WINDOW, OK),
            (74, 34, 64, 10, WINDOW, OK),
            (75, 4, 64, 10, WINDOW, DICT),
            (10, 4, 64, 9, WINDOW, WINDOW),
            (60, 4, 64, 9, WINDOW, WINDOW),
        ];
        for simd in levels {
            for (offset, ml, ext_len, window, error, error_dict) in cases {
                for dict in [false, true] {
                    let name = format!(
                        "offset {offset} ml {ml} ext {ext_len} window {window} dict {dict}"
                    );
                    let ext = &ext[ext.len() - ext_len..];
                    let mut out = seg.clone();
                    out.reserve(MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH);
                    let dst = Dst {
                        base: out.as_mut_ptr(),
                        op: out.len(),
                        window,
                    };
                    let history = ExtHistory {
                        end: ext.as_ptr_range().end,
                        len: ext.len(),
                        dict,
                    };
                    let block = one_match_block(offset as u32, ml);
                    let mut scratch = DecoderScratch::new();
                    // SAFETY: `out` holds the segment with the room `Dst`
                    // needs reserved, and `ext` is the history.
                    let got = unsafe {
                        decompress_block(
                            &block,
                            MAX_BLOCK_SIZE,
                            &mut scratch,
                            None,
                            dst,
                            history,
                            simd,
                        )
                        .map(|end| {
                            out.set_len(end);
                            out
                        })
                    };
                    if let Some(e) = if dict { error_dict } else { error } {
                        assert_eq!(got.err().map(String::from).as_deref(), Some(e), "{name}");
                        continue;
                    }
                    let mut want = [ext, &seg[..]].concat();
                    for _ in 0..ml {
                        want.push(want[want.len() - offset]);
                    }
                    assert_eq!(got.as_deref(), Ok(&want[ext.len()..]), "{name}");
                }
            }
        }
    }

    #[test]
    fn copy_match_is_bytewise_copy() {
        check_copy_match(Fallback::new(), "Fallback");
        check_copy_match(
            FallbackShortOffsets(Fallback::new()),
            "FallbackShortOffsets",
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Level::Avx2(w) = Level::new() {
            check_copy_match(w, "Avx2");
            check_copy_match(Avx2ShortOffsets(w), "Avx2ShortOffsets");
        }
    }

    #[test]
    fn test_empty_input() {
        let result = decompress(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_bad_magic() {
        let result = decompress(&[0, 0, 0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_raw() {
        // A minimal zstd frame: magic + frame header + single raw block
        // This test builds a valid frame with a raw block containing "hello"
        let data = b"hello";
        let mut frame = Vec::new();
        // Magic number
        frame.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        // Frame descriptor: single_segment=1, no checksum, no dict, fcs_flag=0
        // So FCS field = 1 byte
        frame.push(0x20); // single_segment_flag set
                          // FCS = 5 (length of "hello")
        frame.push(5);
        // Block header: last_block=1, type=raw(0), size=5
        // Encoding: bit0=last(1), bit1-2=type(0), bit3-20=size(5)
        let bh = 1u32 | (0u32 << 1) | (5u32 << 3);
        frame.push((bh & 0xFF) as u8);
        frame.push(((bh >> 8) & 0xFF) as u8);
        frame.push(((bh >> 16) & 0xFF) as u8);
        // Block content
        frame.extend_from_slice(data);

        let result = decompress(&frame).unwrap();
        assert_eq!(result, data);
    }

    /// A frame declaring `window_descriptor` (no content size) and holding
    /// one raw block of "hi".
    fn windowed_frame(window_descriptor: u8) -> Vec<u8> {
        let mut frame = ZSTD_MAGIC.to_le_bytes().to_vec();
        frame.push(0x00); // no single segment, no content size
        frame.push(window_descriptor);
        let bh = 1u32 | (2u32 << 3); // last, raw, 2 bytes
        frame.extend_from_slice(&bh.to_le_bytes()[..3]);
        frame.extend_from_slice(b"hi");
        frame
    }

    /// One-shot libzstd refuses a window only above `ZSTD_WINDOWLOG_MAX`
    /// (its default limit, window log 27, binds streaming alone): window
    /// logs 27, 28, 31 and 32, at the smallest and largest mantissa, decode
    /// exactly where `zstd::bulk` does.
    #[test]
    fn test_window_limit_is_one_shot_zstd() {
        for window_log in [27, 28, 31, 32] {
            for mantissa in [0, 7] {
                let frame = windowed_frame((window_log - 10) << 3 | mantissa);
                let want = zstd::bulk::decompress(&frame, 16).ok();
                assert_eq!(
                    want.is_some(),
                    u32::from(window_log) <= ZSTD_WINDOWLOG_MAX,
                    "window log {window_log}"
                );
                assert_eq!(
                    decompress(&frame).ok(),
                    want,
                    "window log {window_log} mantissa {mantissa}"
                );
            }
        }
    }

    /// The frame header refuses a window log exactly where libzstd's
    /// `ZSTD_getFrameHeader` does: above `ZSTD_WINDOWLOG_MAX`, the upper
    /// bound of `ZSTD_d_windowLogMax`, for this target.
    #[test]
    fn window_log_bounds_match_libzstd() {
        use zstd::zstd_safe::{self, zstd_sys as sys};

        // SAFETY: reads no memory of ours.
        let bounds =
            unsafe { sys::ZSTD_dParam_getBounds(sys::ZSTD_dParameter::ZSTD_d_windowLogMax) };
        assert_eq!(bounds.error, 0);
        assert_eq!(bounds.upperBound, ZSTD_WINDOWLOG_MAX as i32);
        for descriptor in 0..=u8::MAX {
            let frame = windowed_frame(descriptor);
            let Ok((header, _)) = parse_frame_header(&frame) else {
                panic!("descriptor {descriptor:#x}: header refused");
            };
            assert_eq!(
                header.window_size().is_ok(),
                zstd_safe::get_frame_content_size(&frame).is_ok(),
                "descriptor {descriptor:#x}"
            );
        }
    }

    #[test]
    fn test_roundtrip_rle() {
        // Frame with an RLE block: 10 copies of byte 0x42
        let mut frame = Vec::new();
        frame.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        frame.push(0x20); // single_segment_flag set
        frame.push(10); // FCS = 10
                        // Block header: last_block=1, type=RLE(1), size=10
        let bh = 1u32 | (1u32 << 1) | (10u32 << 3);
        frame.push((bh & 0xFF) as u8);
        frame.push(((bh >> 8) & 0xFF) as u8);
        frame.push(((bh >> 16) & 0xFF) as u8);
        // Single RLE byte
        frame.push(0x42);

        let result = decompress(&frame).unwrap();
        assert_eq!(result, vec![0x42; 10]);
    }

    #[test]
    fn test_roundtrip_with_compressor() {
        // Use the crate's own compressor to produce a valid zstd frame,
        // then decompress with our decoder.
        let data = b"Hello, world! This is a test of the zstd compression and decompression round-trip. \
                      The quick brown fox jumps over the lazy dog. \
                      AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA \
                      BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB \
                      Hello, world! This is a test of the zstd compression and decompression round-trip.";
        let compressed = crate::compress::compress_to_vec(data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_roundtrip_larger() {
        // Test with larger data that triggers compressed blocks.
        let data = Vec::with_capacity(16384);
        let compressed = crate::compress::compress_to_vec(&data);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Every Vec-returning decode, with and without a dictionary, returns
    /// content of at most `COPY_OUT_MAX` bytes in a `Vec` of exactly its
    /// length and larger content with the room `reserve_frame` left past
    /// it (`take_output`); a `Decompressor` does so call after call, below
    /// the bound again after content above it.
    #[test]
    fn vec_output_capacity_bound() {
        let raw_dict: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
        let dict = DecodeDict::new(&raw_dict).unwrap();
        let mut with_dict = zstd::bulk::Compressor::with_dictionary(3, &raw_dict).unwrap();
        let mut d = crate::Decompressor::new();
        let room = MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH;
        for n in [
            COPY_OUT_MAX - 1,
            COPY_OUT_MAX,
            COPY_OUT_MAX + 1,
            COPY_OUT_MAX - 1,
        ] {
            let content: Vec<u8> = (0..n)
                .map(|i| b"zstd frame "[i % 11] ^ (i / 97) as u8)
                .collect();
            let plain = zstd::bulk::compress(&content, 3).unwrap();
            let framed = with_dict.compress(&content).unwrap();
            for (api, out) in [
                ("decompress", decompress(&plain)),
                ("decompress_with_dict", decompress_with_dict(&framed, &dict)),
                ("Decompressor::decompress", d.decompress(&plain)),
                (
                    "Decompressor::decompress_with_dict",
                    d.decompress_with_dict(&framed, &dict),
                ),
            ] {
                let out = out.unwrap();
                assert!(out == content, "{api} {n}: content differs");
                if n <= COPY_OUT_MAX {
                    assert_eq!(out.capacity(), n, "{api} {n}");
                } else {
                    assert!(out.capacity() >= n + room, "{api} {n}: {}", out.capacity());
                }
            }
        }
    }
}
