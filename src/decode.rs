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
//! Public API: `decompress(data: &[u8]) -> Result<Vec<u8>, String>`
//!
//! Supports raw blocks, RLE blocks, and compressed blocks with Huffman
//! literals and FSE sequences. No dictionary support.

#![allow(
    clippy::needless_range_loop,
    clippy::len_without_is_empty,
    clippy::upper_case_acronyms,
    clippy::manual_range_contains,
    dead_code
)]

use crate::constants::ZSTD_WINDOWLOG_MAX;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Avx2;
use fearless_simd::{Fallback, Level};
use std::ptr;

// ============================================================
// Constants
// ============================================================

const ZSTD_MAGIC: u32 = 0xFD2F_B528;
const MIN_WINDOW_SIZE: u64 = 1024;
const MAX_BLOCK_SIZE: u32 = 128 * 1024;
/// `ZSTD_MAXWINDOWSIZE_DEFAULT`: libzstd's default decoder limit,
/// `(1 << ZSTD_WINDOWLOG_LIMIT_DEFAULT) + 1`, which admits the window log
/// 27 frames of level 22 and of long distance matching on large inputs.
const MAXIMUM_ALLOWED_WINDOW_SIZE: u64 = (1 << 27) + 1;
/// Largest Huffman table log, and so weight, the decoder takes (libzstd
/// HUF_TABLELOG_MAX): the format caps the log at 11, libzstd's decoder at 12.
const HUF_TABLELOG_MAX: u32 = 12;
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
/// Supports one or more concatenated zstd frames. Skippable frames are skipped.
/// Dictionary frames are not supported.
///
/// With the `parallel` feature, frames of four or more blocks are decoded on
/// the current rayon pool when it has more than one thread; the output is
/// the same either way.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_with_options(data, &DecodeOptions::default())
}

/// Decoder paths to force, for testing each of them on any input.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    /// Frames of at least this many blocks are decoded on the current rayon
    /// pool, whatever its size; `usize::MAX` never does.
    pub min_parallel_blocks: usize,
    /// Use the SIMD level detected at run time; false forces the portable
    /// code.
    pub simd: bool,
}

impl Default for DecodeOptions {
    /// What `decompress` uses.
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        let min_parallel_blocks = if rayon::current_num_threads() > 1 {
            parallel::MIN_BLOCKS
        } else {
            usize::MAX
        };
        #[cfg(not(feature = "parallel"))]
        let min_parallel_blocks = usize::MAX;
        DecodeOptions {
            min_parallel_blocks,
            simd: true,
        }
    }
}

/// `decompress` with the paths chosen by `opts`.
#[doc(hidden)]
pub fn decompress_with_options(data: &[u8], opts: &DecodeOptions) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut scratch: Option<DecoderScratch> = None;
    let mut pos = 0usize;

    while pos < data.len() {
        let (frame_header, header_len) = match parse_frame_header(&data[pos..]) {
            Ok(parsed) => parsed,
            Err(e) => {
                if let Some(skip_len) = e.skip_frame_length() {
                    let end = pos
                        .checked_add(SKIPPABLE_FRAME_HEADER_LEN)
                        .and_then(|p| p.checked_add(skip_len as usize))
                        .filter(|&end| end <= data.len())
                        .ok_or_else(|| "Skippable frame extends past end of input".to_string())?;
                    pos = end;
                    continue;
                }
                // If we already have output and hit an error, it might just be trailing data
                if !output.is_empty() {
                    break;
                }
                return Err(format!("Frame header error: {}", e));
            }
        };
        pos += header_len;

        let scratch = scratch.get_or_insert_with(DecoderScratch::new);
        scratch.reset();
        let simd = if opts.simd {
            Level::new()
        } else {
            Level::fallback()
        };
        decode_frame(
            &frame_header,
            data,
            &mut pos,
            scratch,
            &mut output,
            opts.min_parallel_blocks,
            simd,
        )?;
    }

    Ok(output)
}

// ============================================================
// BitReader (forward)
// ============================================================

struct BitReader<'s> {
    idx: usize,
    source: &'s [u8],
}

impl<'s> BitReader<'s> {
    fn new(source: &'s [u8]) -> BitReader<'s> {
        BitReader { idx: 0, source }
    }

    fn bits_left(&self) -> usize {
        self.source.len() * 8 - self.idx
    }

    fn bits_read(&self) -> usize {
        self.idx
    }

    fn return_bits(&mut self, n: usize) {
        if n > self.idx {
            panic!("Cannot return more bits than have been read");
        }
        self.idx -= n;
    }

    fn get_bits(&mut self, n: usize) -> Result<u64, String> {
        if n > 64 {
            return Err(format!("Cannot read {} bits, maximum is 64", n));
        }
        if self.bits_left() < n {
            return Err(format!(
                "Cannot read {} bits, only {} remaining",
                n,
                self.bits_left()
            ));
        }

        let old_idx = self.idx;
        let bits_left_in_current_byte = 8 - (self.idx % 8);
        let bits_not_needed_in_current_byte = 8 - bits_left_in_current_byte;

        let mut value = u64::from(self.source[self.idx / 8] >> bits_not_needed_in_current_byte);

        if bits_left_in_current_byte >= n {
            value &= (1 << n) - 1;
            self.idx += n;
        } else {
            self.idx += bits_left_in_current_byte;
            let full_bytes_needed = (n - bits_left_in_current_byte) / 8;
            let bits_in_last_byte_needed = n - bits_left_in_current_byte - full_bytes_needed * 8;

            let mut bit_shift = bits_left_in_current_byte;

            for _ in 0..full_bytes_needed {
                value |= u64::from(self.source[self.idx / 8]) << bit_shift;
                self.idx += 8;
                bit_shift += 8;
            }

            if bits_in_last_byte_needed > 0 {
                let val_last_byte =
                    u64::from(self.source[self.idx / 8]) & ((1 << bits_in_last_byte_needed) - 1);
                value |= val_last_byte << bit_shift;
                self.idx += bits_in_last_byte_needed;
            }
        }

        debug_assert!(self.idx == old_idx + n);
        Ok(value)
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
    /// True while the table holds a predefined sequence distribution, so the
    /// next block in Predefined mode can reuse it without rebuilding.
    predefined: bool,
}

impl FSETable {
    fn new(max_symbol: u8) -> FSETable {
        FSETable {
            max_symbol,
            cells: Vec::new(),
            size: 0,
            accuracy_log: 0,
            symbol_probabilities: Vec::with_capacity(256),
            symbol_next: Vec::new(),
            spread: Vec::new(),
            predefined: false,
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
        self.predefined = false;
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
    ) -> Result<usize, String> {
        self.reset();
        let bytes_read = self.read_probabilities(source, max_log)?;
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
    ) -> Result<(), String> {
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
            ));
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
    fn read_probabilities(&mut self, source: &[u8], max_log: u8) -> Result<usize, String> {
        if source.len() < 8 {
            // The body reads 4 bytes at a time up to the header's end.
            let mut buffer = [0u8; 8];
            buffer[..source.len()].copy_from_slice(source);
            let n = self.read_ncount_body(&buffer, max_log)?;
            if n > source.len() {
                return Err("FSE table header extends past its input".to_string());
            }
            return Ok(n);
        }
        self.read_ncount_body(source, max_log)
    }

    /// FSE_readNCount_body; requires `src.len() >= 8`.
    fn read_ncount_body(&mut self, src: &[u8], max_log: u8) -> Result<usize, String> {
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
            return Err(format!("Accuracy log {} exceeds max {}", nb_bits, max_log));
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
            ));
        }
        if charnum > max_sv1 {
            return Err(format!("Too many symbols: {}", charnum));
        }
        if bit_count > 32 {
            return Err("FSE table header extends past its input".to_string());
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

/// Size in u32 words of FSE_decompress_wksp's workspace for a table of
/// `table_log` over symbols `0..=max_symbol` (FSE_DECOMPRESS_WKSP_SIZE_U32).
const fn fse_decompress_wksp_u32(table_log: usize, max_symbol: usize) -> usize {
    let dtable = 1 + (1 << table_log);
    let build = (2 * (max_symbol + 1) + (1 << table_log) + 8).div_ceil(4);
    dtable + 1 + build + 256 / 2 + 1
}

/// Decode the weights of a Huffman tree description from an FSE bitstream
/// with two interleaved states (FSE_decompress_usingDTable_generic): four
/// symbols per reload while the stream lasts, then one at a time until it
/// overflows. Returns the number of weights.
fn fse_decompress_weights(
    table: &FSETable,
    src: &[u8],
    out: &mut [u8; 255],
) -> Result<usize, String> {
    let dt = table.decode();
    let table_log = u32::from(table.accuracy_log);
    let mut br = BitDStream::new(src)?;
    // FSE_initDState reloads after each initial state.
    let mut state1 = br.read_bits(table_log);
    br.reload();
    let mut state2 = br.read_bits(table_log);
    br.reload();
    if br.reload() == HufStreamStatus::Overflow {
        return Err("Huffman weights stream is too short".to_string());
    }
    // FSE_decodeSymbol: the state's symbol, then the next state. Every
    // cell's `next_state` plus its `num_bits` bits stays below the table
    // size, so a state is always a valid index.
    let decode = |state: &mut usize, br: &mut BitDStream<'_>| {
        let cell = dt[*state];
        *state = usize::from(cell.next_state) + br.read_bits(u32::from(cell.num_bits));
        cell.base_value as u8
    };
    let too_many = || Err("Too many Huffman weights".to_string());
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

/// Table log of the decoding tables of codes up to this many bits (libzstd
/// HUF_DECODER_FAST_TABLELOG): shorter codes are scaled up to it so that
/// the 4-stream fast loops index with a constant shift. A 12-bit code keeps
/// its own log and takes the plain loops.
const HUF_FAST_TABLE_LOG: u32 = 11;

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

struct HuffmanTable {
    /// Single-symbol table; when `!is_x2`, the built table, of
    /// `1 << dt_log()` cells.
    decode: Vec<HuffmanEntry>,
    /// Double-symbol table; when `is_x2`, the built table, of
    /// `1 << dt_log()` cells.
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
            fse_table: FSETable::new(255),
        }
    }

    /// Forget the table. Its cells stay allocated for the next build, which
    /// writes every one of them.
    fn reset(&mut self) {
        self.is_x2 = false;
        self.nb_symbols = 0;
        self.max_num_bits = 0;
        self.fse_table.reset();
    }

    /// Log of the built table (libzstd DTableDesc.tableLog): codes of up to
    /// HUF_FAST_TABLE_LOG bits are scaled up to it (HUF_rescaleStats, and
    /// HUF_readDTableX2_wksp's maxTableLog), a 12-bit code keeps its own.
    fn dt_log(&self) -> u32 {
        u32::from(self.max_num_bits).max(HUF_FAST_TABLE_LOG)
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
    ) -> Result<u32, String> {
        self.max_num_bits = 0;
        let (bytes_used, nb_weights) = self.read_weights(source)?;
        self.weight_stats(nb_weights)?;
        self.is_x2 = four_streams && huf_select_x2(dst_size, source.len());
        if self.is_x2 {
            self.fill_x2();
        } else {
            self.fill_x1();
        }
        Ok(bytes_used as u32)
    }

    /// Read the weights of a tree description (HUF_readStats_body before the
    /// statistics): a header byte of 128 or more is followed by
    /// `header - 127` 4-bit weights, a smaller one by that many bytes of
    /// FSE-compressed weights. Returns the length of the description and
    /// the number of weights read.
    fn read_weights(&mut self, source: &[u8]) -> Result<(usize, usize), String> {
        let Some(&header) = source.first() else {
            return Err("Huffman source is empty".to_string());
        };
        let header = usize::from(header);
        if header >= 128 {
            let nb_weights = header - 127;
            let size = nb_weights.div_ceil(2);
            let Some(packed) = source.get(1..1 + size) else {
                return Err(format!(
                    "Not enough bytes for {} raw Huffman weights",
                    nb_weights
                ));
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
                ));
            };
            let ncount = self.fse_table.build_decoder(src, 6, None)?;
            // FSE_decompress_wksp's table must fit HUF_readStats's workspace
            // (HUF_READ_STATS_WORKSPACE_SIZE_U32), sized for 6-bit tables
            // over the weights below HUF_TABLELOG_MAX.
            let max_symbol = self.fse_table.symbol_probabilities.len() - 1;
            let log = usize::from(self.fse_table.accuracy_log);
            let wksp = fse_decompress_wksp_u32(6, HUF_TABLELOG_MAX as usize - 1);
            if fse_decompress_wksp_u32(log, max_symbol) > wksp {
                return Err(format!(
                    "Huffman weights table of log {} over {} symbols is too large",
                    log,
                    max_symbol + 1
                ));
            }
            let out = self.weights.first_chunk_mut::<255>().unwrap();
            let nb_weights = fse_decompress_weights(&self.fse_table, &src[ncount..], out)?;
            Ok((1 + header, nb_weights))
        }
    }

    /// The statistics and checks of HUF_readStats_body over the
    /// `nb_weights` weights read: symbols per weight, the table log, the
    /// implied last weight and a full binary tree.
    fn weight_stats(&mut self, nb_weights: usize) -> Result<(), String> {
        let mut rank_stats = [0u32; HUF_TABLELOG_MAX as usize + 1];
        let mut weight_total = 0u32;
        for &w in &self.weights[..nb_weights] {
            if u32::from(w) > HUF_TABLELOG_MAX {
                return Err(format!("Weight {} exceeds max {}", w, HUF_TABLELOG_MAX));
            }
            rank_stats[usize::from(w)] += 1;
            weight_total += (1 << w) >> 1;
        }
        if weight_total == 0 {
            return Err("Missing weights".to_string());
        }
        let table_log = highest_bit_set(weight_total);
        if table_log > HUF_TABLELOG_MAX {
            return Err(format!("Max bits {} too high", table_log));
        }
        // The last weight completes the total to a power of 2.
        let rest = (1 << table_log) - weight_total;
        if !rest.is_power_of_two() {
            return Err(format!("Leftover {} is not a power of 2", rest));
        }
        let last_weight = highest_bit_set(rest);
        self.weights[nb_weights] = last_weight as u8;
        rank_stats[last_weight as usize] += 1;
        // A full binary tree has an even number of leaves at its deepest
        // level, and at least two.
        if rank_stats[1] < 2 || rank_stats[1] & 1 != 0 {
            return Err(format!(
                "Huffman tree has {} symbols of weight 1",
                rank_stats[1]
            ));
        }
        self.rank_stats = rank_stats;
        self.nb_symbols = nb_weights + 1;
        self.max_num_bits = table_log as u8;
        Ok(())
    }

    /// Fill the single-symbol table at `dt_log()` bits
    /// (HUF_readDTableX1_wksp with HUF_rescaleStats): each symbol of `n`
    /// bits owns `1 << (dt_log() - n)` consecutive cells, ordered by code
    /// length. Scaling the weights up leaves every code length unchanged,
    /// so only the cell counts differ from a `max_bits` table. The weights
    /// tile the table (`weight_stats`): every cell is written.
    fn fill_x1(&mut self) {
        let max_bits = u32::from(self.max_num_bits);
        let dt_log = self.dt_log();
        let rescale = dt_log - max_bits;
        let rank_stats = &self.rank_stats;

        // Symbols ordered by weight, then by value (libzstd symbols[]).
        let mut rank_start = [0usize; HUF_TABLELOG_MAX as usize + 1];
        let mut next = 0usize;
        for w in 0..=max_bits as usize {
            rank_start[w] = next;
            next += rank_stats[w] as usize;
        }
        for (s, &w) in self.weights[..self.nb_symbols].iter().enumerate() {
            let r = &mut rank_start[usize::from(w)];
            self.sorted[*r] = s as u8;
            *r += 1;
        }

        // Fill the table one weight at a time, so that the run length is a
        // constant of each loop, writing each symbol's cells four to a word
        // (HUF_DEltX1_set4) over the table's bytes.
        self.decode.resize(1 << dt_log, HuffmanEntry::default());
        let cells: &mut [u8] = bytemuck::cast_slice_mut(&mut self.decode[..]);
        let mut symbol = rank_stats[0] as usize;
        let mut u = 0usize;
        for w in 1..=max_bits as usize {
            let count = rank_stats[w] as usize;
            let length = 1usize << (w - 1 + rescale as usize);
            let num_bits = (max_bits + 1 - w as u32) as u8;
            let syms = &self.sorted[symbol..symbol + count];
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
            symbol += count;
        }
    }

    /// Fill the double-symbol table (HUF_readDTableX2_wksp after
    /// HUF_readStats): sort symbols by weight, compute where each weight's
    /// run starts for every number of already-consumed bits, then tile the
    /// table so that a cell holds two symbols whenever both fit in
    /// `dt_log()` bits. Every cell is written, as in `fill_x1`.
    fn fill_x2(&mut self) {
        let table_log = u32::from(self.max_num_bits);
        let target_log = self.dt_log();
        let nb_bits_baseline = table_log + 1;

        // Highest weight in use; weight 1 is always present.
        let mut max_w = table_log as usize;
        while self.rank_stats[max_w] == 0 {
            max_w -= 1;
        }

        // rank_start[w]: first index of weight w in the sorted list.
        let mut rank_start = [0usize; HUF_TABLELOG_MAX as usize + 2];
        let mut next = 0usize;
        for w in 1..=max_w {
            rank_start[w] = next;
            next += self.rank_stats[w] as usize;
        }
        rank_start[max_w + 1] = next;

        // Weight-0 symbols go after all others, and are never read.
        let mut fill = rank_start;
        fill[0] = next;
        for (s, &w) in self.weights[..self.nb_symbols].iter().enumerate() {
            let r = &mut fill[usize::from(w)];
            self.sorted[*r] = s as u8;
            *r += 1;
        }

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

/// Fewest literals for which the 4-stream layout is legal
/// (libzstd MIN_LITERALS_FOR_4_STREAMS).
const MIN_LITERALS_FOR_4_STREAMS: usize = 6;

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
/// its high end (the stream is read from its last byte backwards). Once
/// `bits_consumed` exceeds the data actually loaded, reads return zeros and
/// the final `is_finished` check rejects the stream.
struct BitDStream<'s> {
    src: &'s [u8],
    ptr: usize,
    container: u64,
    bits_consumed: u32,
}

impl<'s> BitDStream<'s> {
    fn new(src: &'s [u8]) -> Result<Self, String> {
        let Some(&last) = src.last() else {
            return Err("Huffman stream is empty".to_string());
        };
        if last == 0 {
            return Err("Huffman stream has no end mark".to_string());
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
            let mut buf = [0u8; 8];
            buf[..src.len()].copy_from_slice(src);
            Ok(BitDStream {
                src,
                ptr: 0,
                container: u64::from_le_bytes(buf),
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
fn huf_decode_symbol_x1(br: &mut BitDStream<'_>, dt: &[HuffmanEntry], dt_log: u32) -> u8 {
    let entry = dt[br.look_bits(dt_log)];
    br.skip_bits(u32::from(entry.num_bits));
    entry.symbol
}

/// Decode `out.len()` symbols from one stream (HUF_decodeStreamX1).
#[inline(always)]
fn huf_decode_stream_x1(out: &mut [u8], br: &mut BitDStream<'_>, dt: &[HuffmanEntry], dt_log: u32) {
    let end = out.len();
    let mut p = 0;
    if end > 3 {
        // Up to 4 symbols per reload: a reload that reports Unfinished
        // guarantees at least 57 bits, and a symbol takes at most 12.
        while br.reload() == HufStreamStatus::Unfinished && p < end - 3 {
            let a = huf_decode_symbol_x1(br, dt, dt_log);
            let b = huf_decode_symbol_x1(br, dt, dt_log);
            let c = huf_decode_symbol_x1(br, dt, dt_log);
            let d = huf_decode_symbol_x1(br, dt, dt_log);
            out[p..p + 4].copy_from_slice(&[a, b, c, d]);
            p += 4;
        }
    } else {
        br.reload();
    }
    // Either at most 3 symbols remain with >= 57 bits loaded, or the
    // container already holds the last bytes of the stream.
    while p < end {
        out[p] = huf_decode_symbol_x1(br, dt, dt_log);
        p += 1;
    }
}

/// Single-stream literals (HUF_decompress1X1_usingDTable_internal_body)
/// with a table of `DT_LOG` bits.
#[inline(never)]
fn huf_decompress_1x1<const DT_LOG: u32>(
    out: &mut [u8],
    src: &[u8],
    table: &HuffmanTable,
) -> Result<(), String> {
    let dt = &table.decode[..];
    let mut br = BitDStream::new(src)?;
    huf_decode_stream_x1(out, &mut br, dt, DT_LOG);
    if !br.is_finished() {
        return Err("Huffman stream not fully consumed".to_string());
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
    fn split(src: &[u8], dst_size: usize) -> Result<HufStreams, String> {
        if src.len() < 10 {
            return Err(format!(
                "Huffman 4-stream input too short: {} bytes",
                src.len()
            ));
        }
        if dst_size < MIN_LITERALS_FOR_4_STREAMS {
            return Err(format!(
                "Huffman 4-stream output too small: {} bytes",
                dst_size
            ));
        }
        let len1 = usize::from(u16::from_le_bytes([src[0], src[1]]));
        let len2 = usize::from(u16::from_le_bytes([src[2], src[3]]));
        let len3 = usize::from(u16::from_le_bytes([src[4], src[5]]));
        if 6 + len1 + len2 + len3 > src.len() {
            return Err("Huffman jump table exceeds input".to_string());
        }
        let istart = [6, 6 + len1, 6 + len1 + len2, 6 + len1 + len2 + len3];
        let iend = [istart[1], istart[2], istart[3], src.len()];
        let segment = dst_size.div_ceil(4);
        if 3 * segment > dst_size {
            return Err("Huffman 4-stream segments exceed output".to_string());
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
) -> Result<Option<HufFastArgs>, String> {
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
            return Err("Huffman stream has no end mark".to_string());
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
) -> Result<BitDStream<'s>, String> {
    let start = streams.istart[s];
    let stream = streams.stream(src, s);
    let ip = args.ip[s];
    // A fully consumed stream leaves the container at most 8 bytes below
    // its start; anything lower is corruption.
    if ip + 8 < start {
        return Err("Huffman stream overran its start".to_string());
    }
    let (ptr, below) = if ip >= start {
        (ip - start, 0)
    } else {
        (0, start - ip)
    };
    let bits_consumed = args.bits[s].trailing_zeros() + below as u32 * 8;
    if bits_consumed > 64 {
        return Err("Huffman stream overran its start".to_string());
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
/// loop takes sections with 8 bytes or more per stream and a table of
/// `HUF_FAST_TABLE_LOG` bits (HUF_DecompressFastArgs_init); the plain loop
/// advances all four streams in lockstep, 4 symbols each per reload. Both
/// finish each stream with `huf_decode_stream_x1`.
#[inline(never)]
fn huf_decompress_4x1<const DT_LOG: u32>(
    out: &mut [u8],
    src: &[u8],
    table: &HuffmanTable,
) -> Result<(), String> {
    let dt = &table.decode[..];
    if dt.len() != 1 << DT_LOG {
        return Err("Huffman table is uninitialized".to_string());
    }
    let dt_log = DT_LOG;
    let dst_size = out.len();
    let streams = HufStreams::split(src, dst_size)?;

    let fast = if DT_LOG == HUF_FAST_TABLE_LOG {
        huf_fast_args_init(&streams, src, dst_size)?
    } else {
        None
    };
    if let Some(mut args) = fast {
        // SAFETY: `args` comes from `huf_fast_args_init` on this `src` and
        // `out`, and `dt` has exactly `1 << HUF_FAST_TABLE_LOG` cells.
        unsafe { huf_4x1_fast_loop(&mut args, out, src, dt) };
        for s in 0..4 {
            let end = streams.segment_end(s, dst_size);
            if args.op[s] > end {
                return Err("Huffman stream overran its segment".to_string());
            }
            let mut br = huf_remaining_dstream(&args, &streams, s, src)?;
            huf_decode_stream_x1(&mut out[args.op[s]..end], &mut br, dt, dt_log);
            if !br.is_finished() {
                return Err("Huffman stream not fully consumed".to_string());
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
                w1[i] = huf_decode_symbol_x1(&mut b1, dt, dt_log);
                w2[i] = huf_decode_symbol_x1(&mut b2, dt, dt_log);
                w3[i] = huf_decode_symbol_x1(&mut b3, dt, dt_log);
                w4[i] = huf_decode_symbol_x1(&mut b4, dt, dt_log);
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

    huf_decode_stream_x1(&mut o1[p..], &mut b1, dt, dt_log);
    huf_decode_stream_x1(&mut o2[p..], &mut b2, dt, dt_log);
    huf_decode_stream_x1(&mut o3[p..], &mut b3, dt, dt_log);
    huf_decode_stream_x1(&mut o4[p..], &mut b4, dt, dt_log);

    if !(b1.is_finished() && b2.is_finished() && b3.is_finished() && b4.is_finished()) {
        return Err("Huffman stream not fully consumed".to_string());
    }
    Ok(())
}

/// Decode one cell's one or two symbols (HUF_decodeSymbolX2). Always
/// writes two bytes; the caller keeps `op + 2 <= out.len()`.
#[inline(always)]
fn huf_decode_symbol_x2(
    out: &mut [u8],
    op: usize,
    br: &mut BitDStream<'_>,
    dt: &[HufEntryX2],
    dt_log: u32,
) -> usize {
    let entry = dt[br.look_bits(dt_log)];
    out[op..op + 2].copy_from_slice(&entry.sequence.to_le_bytes());
    br.skip_bits(u32::from(entry.nb_bits));
    op + usize::from(entry.length)
}

/// Decode the final symbol of a stream (HUF_decodeLastSymbolX2): only the
/// first symbol of a two-symbol cell is wanted, and the bit count for it
/// alone is unknown, so consumption is clamped to the container.
#[inline(always)]
fn huf_decode_last_symbol_x2(
    out: &mut [u8],
    op: usize,
    br: &mut BitDStream<'_>,
    dt: &[HufEntryX2],
    dt_log: u32,
) -> usize {
    let entry = dt[br.look_bits(dt_log)];
    out[op] = entry.sequence as u8;
    if entry.length == 1 {
        br.skip_bits(u32::from(entry.nb_bits));
    } else if br.bits_consumed < 64 {
        br.skip_bits(u32::from(entry.nb_bits));
        if br.bits_consumed > 64 {
            br.bits_consumed = 64;
        }
    }
    op + 1
}

/// Decode symbols into `out[op..end]` (HUF_decodeStreamX2); returns the
/// position reached.
#[inline(always)]
fn huf_decode_stream_x2(
    out: &mut [u8],
    mut op: usize,
    end: usize,
    br: &mut BitDStream<'_>,
    dt: &[HufEntryX2],
    dt_log: u32,
) -> usize {
    if end - op >= 8 {
        if dt_log <= HUF_FAST_TABLE_LOG {
            // Up to 10 symbols per reload: 5 cells of at most 11 bits each.
            while br.reload() == HufStreamStatus::Unfinished && op + 9 < end {
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
            }
        } else {
            // Up to 8 symbols per reload: 4 cells of at most 12 bits each.
            while br.reload() == HufStreamStatus::Unfinished && op + 7 < end {
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
                op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
            }
        }
    } else {
        br.reload();
    }

    if end - op >= 2 {
        while br.reload() == HufStreamStatus::Unfinished && op + 2 <= end {
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
        }
        // The container holds the last bytes of the stream: no reloads.
        while op + 2 <= end {
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
        }
    }

    if op < end {
        op = huf_decode_last_symbol_x2(out, op, br, dt, dt_log);
    }
    op
}

/// Single-stream literals with the double-symbol table of `DT_LOG` bits
/// (HUF_decompress1X2_usingDTable_internal_body).
#[inline(never)]
fn huf_decompress_1x2<const DT_LOG: u32>(
    out: &mut [u8],
    src: &[u8],
    table: &HuffmanTable,
) -> Result<(), String> {
    let dt = &table.decode_x2[..];
    let mut br = BitDStream::new(src)?;
    huf_decode_stream_x2(out, 0, out.len(), &mut br, dt, DT_LOG);
    if !br.is_finished() {
        return Err("Huffman stream not fully consumed".to_string());
    }
    Ok(())
}

/// Four interleaved literal streams with the double-symbol table
/// (HUF_decompress4X2_usingDTable_internal_body and _fast). Streams write
/// at their own pace, so each is checked against its segment end after the
/// shared loop; the plain loop's trip count is bounded by the last stream.
/// The fast loop takes the same sections as in `huf_decompress_4x1`.
#[inline(never)]
fn huf_decompress_4x2<const DT_LOG: u32>(
    out: &mut [u8],
    src: &[u8],
    table: &HuffmanTable,
) -> Result<(), String> {
    let dt = &table.decode_x2[..];
    if dt.len() != 1 << DT_LOG {
        return Err("Huffman table is uninitialized".to_string());
    }
    let dt_log = DT_LOG;
    let oend = out.len();
    let streams = HufStreams::split(src, oend)?;

    let fast = if DT_LOG == HUF_FAST_TABLE_LOG {
        huf_fast_args_init(&streams, src, oend)?
    } else {
        None
    };
    if let Some(mut args) = fast {
        // SAFETY: `args` comes from `huf_fast_args_init` on this `src` and
        // `out`, and `dt` has exactly `1 << HUF_FAST_TABLE_LOG` cells.
        unsafe { huf_4x2_fast_loop(&mut args, out, src, dt) };
        for s in 0..4 {
            let end = streams.segment_end(s, oend);
            if args.op[s] > end {
                return Err("Huffman stream overran its segment".to_string());
            }
            let mut br = huf_remaining_dstream(&args, &streams, s, src)?;
            huf_decode_stream_x2(out, args.op[s], end, &mut br, dt, dt_log);
            if !br.is_finished() {
                return Err("Huffman stream not fully consumed".to_string());
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

    // 4 cells per stream per iteration, at most 8 bytes each; every stream
    // stays inside `out` because none can outrun the last one by more than
    // a factor of two.
    if oend - op4 >= 8 {
        let mut end_signal = true;
        while end_signal && op4 + 8 <= oend {
            for _ in 0..4 {
                op1 = huf_decode_symbol_x2(out, op1, &mut b1, dt, dt_log);
            }
            for _ in 0..4 {
                op2 = huf_decode_symbol_x2(out, op2, &mut b2, dt, dt_log);
            }
            end_signal &= b1.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b2.reload_fast() == HufStreamStatus::Unfinished;
            for _ in 0..4 {
                op3 = huf_decode_symbol_x2(out, op3, &mut b3, dt, dt_log);
            }
            for _ in 0..4 {
                op4 = huf_decode_symbol_x2(out, op4, &mut b4, dt, dt_log);
            }
            end_signal &= b3.reload_fast() == HufStreamStatus::Unfinished;
            end_signal &= b4.reload_fast() == HufStreamStatus::Unfinished;
        }
    }

    if op1 > op_start2 || op2 > op_start3 || op3 > op_start4 {
        return Err("Huffman stream overran its segment".to_string());
    }

    huf_decode_stream_x2(out, op1, op_start2, &mut b1, dt, dt_log);
    huf_decode_stream_x2(out, op2, op_start3, &mut b2, dt, dt_log);
    huf_decode_stream_x2(out, op3, op_start4, &mut b3, dt, dt_log);
    huf_decode_stream_x2(out, op4, oend, &mut b4, dt, dt_log);

    if !(b1.is_finished() && b2.is_finished() && b3.is_finished() && b4.is_finished()) {
        return Err("Huffman stream not fully consumed".to_string());
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
    compressed_size: Option<u32>,
    num_streams: Option<u8>,
    ls_type: LiteralsSectionType,
}

impl LiteralsSection {
    fn new() -> LiteralsSection {
        LiteralsSection {
            regenerated_size: 0,
            compressed_size: None,
            num_streams: None,
            ls_type: LiteralsSectionType::Raw,
        }
    }

    fn section_type(raw: u8) -> Result<LiteralsSectionType, String> {
        let t = raw & 0x3;
        match t {
            0 => Ok(LiteralsSectionType::Raw),
            1 => Ok(LiteralsSectionType::RLE),
            2 => Ok(LiteralsSectionType::Compressed),
            3 => Ok(LiteralsSectionType::Treeless),
            other => Err(format!("Illegal literal section type: {}", other)),
        }
    }

    fn header_bytes_needed(&self, first_byte: u8) -> Result<u8, String> {
        let ls_type = Self::section_type(first_byte)?;
        let size_format = (first_byte >> 2) & 0x3;
        match ls_type {
            LiteralsSectionType::RLE | LiteralsSectionType::Raw => match size_format {
                0 | 2 => Ok(1),
                1 => Ok(2),
                3 => Ok(3),
                _ => unreachable!(),
            },
            LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => match size_format {
                0 | 1 => Ok(3),
                2 => Ok(4),
                3 => Ok(5),
                _ => unreachable!(),
            },
        }
    }

    fn parse_from_header(&mut self, raw: &[u8]) -> Result<u8, String> {
        let mut br = BitReader::new(raw);
        let block_type = br.get_bits(2)? as u8;
        self.ls_type = Self::section_type(block_type)?;
        let size_format = br.get_bits(2)? as u8;

        let byte_needed = self.header_bytes_needed(raw[0])?;
        if raw.len() < byte_needed as usize {
            return Err(format!(
                "Not enough bytes for literals header: have {}, need {}",
                raw.len(),
                byte_needed
            ));
        }

        match self.ls_type {
            LiteralsSectionType::RLE | LiteralsSectionType::Raw => {
                self.compressed_size = None;
                match size_format {
                    0 | 2 => {
                        self.regenerated_size = u32::from(raw[0]) >> 3;
                        Ok(1)
                    }
                    1 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4) + (u32::from(raw[1]) << 4);
                        Ok(2)
                    }
                    3 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4)
                            + (u32::from(raw[1]) << 4)
                            + (u32::from(raw[2]) << 12);
                        Ok(3)
                    }
                    _ => unreachable!(),
                }
            }
            LiteralsSectionType::Compressed | LiteralsSectionType::Treeless => {
                match size_format {
                    0 => {
                        self.num_streams = Some(1);
                    }
                    1..=3 => {
                        self.num_streams = Some(4);
                    }
                    _ => unreachable!(),
                };

                match size_format {
                    0 | 1 => {
                        self.regenerated_size =
                            (u32::from(raw[0]) >> 4) + ((u32::from(raw[1]) & 0x3f) << 4);
                        self.compressed_size =
                            Some(u32::from(raw[1] >> 6) + (u32::from(raw[2]) << 2));
                        Ok(3)
                    }
                    2 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4)
                            + (u32::from(raw[1]) << 4)
                            + ((u32::from(raw[2]) & 0x3) << 12);
                        self.compressed_size =
                            Some((u32::from(raw[2]) >> 2) + (u32::from(raw[3]) << 6));
                        Ok(4)
                    }
                    3 => {
                        self.regenerated_size = (u32::from(raw[0]) >> 4)
                            + (u32::from(raw[1]) << 4)
                            + ((u32::from(raw[2]) & 0x3F) << 12);
                        self.compressed_size = Some(
                            (u32::from(raw[2]) >> 6)
                                + (u32::from(raw[3]) << 2)
                                + (u32::from(raw[4]) << 10),
                        );
                        Ok(5)
                    }
                    _ => unreachable!(),
                }
            }
        }
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
    modes: Option<CompressionModes>,
}

impl SequencesHeader {
    fn new() -> SequencesHeader {
        SequencesHeader {
            num_sequences: 0,
            modes: None,
        }
    }

    fn parse_from_header(&mut self, source: &[u8]) -> Result<u8, String> {
        let mut bytes_read = 0;
        if source.is_empty() {
            return Err("Sequences header source is empty".to_string());
        }

        match source[0] {
            0 => {
                self.num_sequences = 0;
                bytes_read += 1;
            }
            1..=127 => {
                if source.len() < 2 {
                    return Err(format!(
                        "Not enough bytes for sequences header: have {}, need 2",
                        source.len()
                    ));
                }
                self.num_sequences = u32::from(source[0]);
                self.modes = Some(CompressionModes(source[1]));
                bytes_read += 2;
            }
            128..=254 => {
                if source.len() < 2 {
                    return Err(format!(
                        "Not enough bytes for sequences header: have {}, need 2",
                        source.len()
                    ));
                }
                self.num_sequences = ((u32::from(source[0]) - 128) << 8) + u32::from(source[1]);
                bytes_read += 2;
                if self.num_sequences != 0 {
                    if source.len() < 3 {
                        return Err(format!(
                            "Not enough bytes for sequences header: have {}, need 3",
                            source.len()
                        ));
                    }
                    self.modes = Some(CompressionModes(source[2]));
                    bytes_read += 1;
                }
            }
            255 => {
                if source.len() < 4 {
                    return Err(format!(
                        "Not enough bytes for sequences header: have {}, need 4",
                        source.len()
                    ));
                }
                self.num_sequences = u32::from(source[1]) + (u32::from(source[2]) << 8) + 0x7F00;
                self.modes = Some(CompressionModes(source[3]));
                bytes_read += 4;
            }
        }

        Ok(bytes_read)
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
}

struct DecoderScratch {
    huf: HuffmanScratch,
    fse: FSEScratch,
    offset_hist: [u32; 3],
    /// Literals of the current block plus `WILDCOPY_OVERLENGTH` zero bytes.
    literals_buffer: Vec<u8>,
}

impl DecoderScratch {
    fn new() -> DecoderScratch {
        DecoderScratch {
            huf: HuffmanScratch {
                table: HuffmanTable::new(),
            },
            fse: FSEScratch {
                offsets: FSETable::new(MAX_OFFSET_CODE),
                literal_lengths: FSETable::new(MAX_LITERAL_LENGTH_CODE),
                match_lengths: FSETable::new(MAX_MATCH_LENGTH_CODE),
            },
            offset_hist: [1, 4, 8],
            literals_buffer: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.offset_hist = [1, 4, 8];
        self.literals_buffer.clear();
        self.fse.literal_lengths.reset();
        self.fse.match_lengths.reset();
        self.fse.offsets.reset();
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

    fn frame_content_size_bytes(&self) -> Result<u8, String> {
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
            other => Err(format!("Invalid frame content size flag: {}", other)),
        }
    }

    fn dictionary_id_bytes(&self) -> Result<u8, String> {
        match self.dict_id_flag() {
            0 => Ok(0),
            1 => Ok(1),
            2 => Ok(2),
            3 => Ok(4),
            other => Err(format!("Invalid dict id flag: {}", other)),
        }
    }
}

struct FrameHeader {
    descriptor: FrameDescriptor,
    window_descriptor: u8,
    /// Frame_Content_Size, or None when the header omits it.
    frame_content_size: Option<u64>,
}

impl FrameHeader {
    fn window_size(&self) -> Result<u64, String> {
        if self.descriptor.single_segment_flag() {
            Ok(self.frame_content_size.unwrap_or(0))
        } else {
            let exp = self.window_descriptor >> 3;
            let mantissa = self.window_descriptor & 0x7;

            let window_log = 10 + u32::from(exp);
            // frameParameter_windowTooLarge of ZSTD_getFrameHeader
            if window_log > ZSTD_WINDOWLOG_MAX {
                return Err(format!("Window log {} too large", window_log));
            }
            let window_base = 1u64 << window_log;
            let window_add = (window_base / 8) * u64::from(mantissa);

            let window_size = window_base + window_add;

            if window_size < MIN_WINDOW_SIZE {
                Err(format!("Window size {} too small", window_size))
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
    msg: String,
    skip_length: Option<u32>,
}

impl FrameDecoderError {
    fn new(msg: String) -> Self {
        Self {
            msg,
            skip_length: None,
        }
    }

    fn skip(length: u32) -> Self {
        Self {
            msg: format!("Skippable frame with length {}", length),
            skip_length: Some(length),
        }
    }

    fn skip_frame_length(&self) -> Option<u32> {
        self.skip_length
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
        return Err(FrameDecoderError::new(format!(
            "Bad magic number: 0x{:X}",
            magic_num
        )));
    }

    let desc = FrameDescriptor(*src.get(pos).ok_or_else(|| {
        FrameDecoderError::new("Error reading frame descriptor: truncated".into())
    })?);
    pos += 1;

    let mut frame_header = FrameHeader {
        descriptor: FrameDescriptor(desc.0),
        frame_content_size: None,
        window_descriptor: 0,
    };

    if !desc.single_segment_flag() {
        frame_header.window_descriptor = *src.get(pos).ok_or_else(|| {
            FrameDecoderError::new("Error reading window descriptor: truncated".into())
        })?;
        pos += 1;
    }

    // We don't support dictionaries, but we still need to skip these bytes
    let dict_id_len = desc.dictionary_id_bytes().map_err(FrameDecoderError::new)? as usize;
    if src.len() < pos + dict_id_len {
        return Err(FrameDecoderError::new(
            "Error reading dictionary id: truncated".into(),
        ));
    }
    pos += dict_id_len;

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

fn parse_block_header(src: &[u8]) -> Result<(BlockHeader, usize), String> {
    let buf: [u8; 3] = src
        .get(..3)
        .ok_or_else(|| "Error reading block header: truncated".to_string())?
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
        return Err("Found reserved block type".to_string());
    }

    // Raw and RLE blocks are bounded by the output alone (ZSTD_copyRawBlock,
    // ZSTD_setRleBlock); `split_block` bounds a compressed one.
    let block_size = u32::from(buf[0] >> 3) | (u32::from(buf[1]) << 5) | (u32::from(buf[2]) << 13);

    let decompressed_size = match block_type {
        BlockType::Raw | BlockType::RLE => block_size,
        BlockType::Compressed | BlockType::Reserved => 0,
    };
    let content_size = match block_type {
        BlockType::Raw | BlockType::Compressed => block_size,
        BlockType::RLE => 1,
        BlockType::Reserved => 0,
    };

    Ok((
        BlockHeader {
            last_block,
            block_type,
            decompressed_size,
            content_size,
        },
        3,
    ))
}

// ============================================================
// Literals section decoder
// ============================================================

fn decode_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, String> {
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
            decompress_literals(section, scratch, source, target)
        }
    }
}

fn decompress_literals(
    section: &LiteralsSection,
    scratch: &mut HuffmanScratch,
    source: &[u8],
    target: &mut Vec<u8>,
) -> Result<u32, String> {
    let compressed_size = section
        .compressed_size
        .ok_or_else(|| "Missing compressed size".to_string())? as usize;
    let num_streams = section
        .num_streams
        .ok_or_else(|| "Missing num_streams".to_string())?;
    let regenerated_size = section.regenerated_size as usize;

    let source = &source[0..compressed_size];
    let mut bytes_read = 0usize;

    match section.ls_type {
        LiteralsSectionType::Compressed => {
            bytes_read += scratch
                .table
                .build_decoder(source, regenerated_size, num_streams == 4)?
                as usize;
        }
        LiteralsSectionType::Treeless if scratch.table.max_num_bits == 0 => {
            return Err("Uninitialized Huffman table for treeless literals".to_string());
        }
        _ => {}
    }

    let source = &source[bytes_read..];
    let start = target.len();
    target.resize(start + regenerated_size, 0);
    let out = &mut target[start..];

    // Each decoder is instantiated per table log, so that a 12-bit table
    // leaves the fast loops out and codes up to 11 bits keep constant shifts.
    const FAST: u32 = HUF_FAST_TABLE_LOG;
    const MAX: u32 = HUF_TABLELOG_MAX;
    let t = &scratch.table;
    match (num_streams == 4, t.is_x2, t.dt_log() == FAST) {
        (true, true, true) => huf_decompress_4x2::<FAST>(out, source, t)?,
        (true, true, false) => huf_decompress_4x2::<MAX>(out, source, t)?,
        (true, false, true) => huf_decompress_4x1::<FAST>(out, source, t)?,
        (true, false, false) => huf_decompress_4x1::<MAX>(out, source, t)?,
        (false, true, true) => huf_decompress_1x2::<FAST>(out, source, t)?,
        (false, true, false) => huf_decompress_1x2::<MAX>(out, source, t)?,
        (false, false, true) => huf_decompress_1x1::<FAST>(out, source, t)?,
        (false, false, false) => huf_decompress_1x1::<MAX>(out, source, t)?,
    }
    bytes_read += source.len();

    Ok(bytes_read as u32)
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

/// The most bytes a compressed block decodes to in a frame whose
/// Block_Maximum_Size is `block_size_max`. With room left in dst,
/// ZSTD_decompressFrame's blocks keep their literals from
/// `dst + block_size_max + WILDCOPY_OVERLENGTH` on
/// (ZSTD_allocateLiteralsBuffer), and that is where the sequences' output
/// must end (`oend` of ZSTD_decompressSequences_body). This decoder's
/// output grows as needed, so it always has that room.
const fn decoded_block_max(block_size_max: usize) -> usize {
    block_size_max + WILDCOPY_OVERLENGTH
}

/// The sequence executors' output bound, the most any block decodes to;
/// `execute_with_copies` then holds a block to its own frame's
/// `decoded_block_max`, which keeps the executors' loops on a constant.
const DECODED_BLOCK_MAX: usize = decoded_block_max(MAX_BLOCK_SIZE as usize);

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
    /// The table for `SEQ_TABLES[t]`.
    fn table_mut(&mut self, t: usize) -> &mut FSETable {
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
) -> Result<usize, String> {
    let modes = section
        .modes
        .ok_or_else(|| "Missing compression mode".to_string())?;

    let mut bytes_read = 0;
    for (t, mode) in modes.all().into_iter().enumerate() {
        bytes_read += build_sequence_table(
            mode,
            &source[bytes_read..],
            scratch.table_mut(t),
            &SEQ_TABLES[t],
        )?;
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

fn build_sequence_table(
    mode: ModeType,
    source: &[u8],
    table: &mut FSETable,
    kind: &SeqTableKind,
) -> Result<usize, String> {
    let codes = Some((kind.base, kind.bits));
    match mode {
        ModeType::FSECompressed => table.build_decoder(source, kind.max_log, codes),
        ModeType::RLE => {
            let Some(&code) = source.first() else {
                return Err(format!("Missing byte for RLE {} table", kind.name));
            };
            if code > kind.max_code {
                return Err(format!("RLE {} code {} exceeds max", kind.name, code));
            }
            table.build_rle(code, kind.base, kind.bits);
            Ok(1)
        }
        ModeType::Predefined => {
            if !table.predefined {
                table.build_from_probabilities(
                    kind.default_log,
                    kind.default_distribution,
                    codes,
                )?;
                table.predefined = true;
            }
            Ok(0)
        }
        ModeType::Repeat => {
            if table.decode().is_empty() {
                return Err(format!(
                    "Repeat mode without a previous {} table",
                    kind.name
                ));
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
}

#[cold]
#[inline(never)]
fn seq_error_message(e: SeqError) -> String {
    match e {
        SeqError::NotEnoughLiterals => "Sequence needs more literals than the block has".into(),
        SeqError::BlockTooLarge => "Block content exceeds block size limit".into(),
        SeqError::OffsetTooFar => "Match offset reaches before the frame start".into(),
    }
}

/// A block's sequences, executed into the frame by `execute_with_copies`
/// with the copies it picks.
trait BlockSequences {
    /// Execute the sequences straight into `out`, whose bytes from
    /// `prefix_start` on are the frame so far: matches reach back no
    /// further. `out` is grown by the block limit plus slack up front so
    /// that all copies use fixed-size chunks and may overshoot; it is
    /// truncated to the real length on return.
    fn execute<W: WildCopy>(
        self,
        w: W,
        offset_hist: &mut [u32; 3],
        prefix_start: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), String>;
}

/// Execute `seqs` with the copies for its block, for the fused decoder and
/// the MT decoder's stage 3 alike: 32-byte ones on the AVX2 level, 16-byte
/// ones otherwise, and the `ShortOffsets` variants when the block's offsets
/// table gives many short offsets. Each copy type runs in a function of its
/// own. The block may decode to `decoded_block_max(block_size_max)` bytes.
fn execute_with_copies<S: BlockSequences>(
    simd: Level,
    offsets: &FSETable,
    seqs: S,
    offset_hist: &mut [u32; 3],
    block_size_max: usize,
    prefix_start: usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let base = out.len();
    let short = short_offset_share(offsets) >= SHORT_OFFSET_SHARE_MIN;
    match simd {
        // SAFETY: fearless_simd makes an `Avx2` only after detecting AVX2
        // and FMA on this CPU (`Level::new`).
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(w) => unsafe {
            if short {
                execute_avx2(Avx2ShortOffsets(w), seqs, offset_hist, prefix_start, out)
            } else {
                execute_avx2(w, seqs, offset_hist, prefix_start, out)
            }
        },
        _ if short => execute_portable(
            FallbackShortOffsets(Fallback::new()),
            seqs,
            offset_hist,
            prefix_start,
            out,
        ),
        _ => execute_portable(Fallback::new(), seqs, offset_hist, prefix_start, out),
    }?;
    let decoded = out.len() - base;
    if decoded > decoded_block_max(block_size_max) {
        return Err(format!(
            "Block decodes to {} bytes, past Block_Maximum_Size {} + {}",
            decoded, block_size_max, WILDCOPY_OVERLENGTH
        ));
    }
    Ok(())
}

#[inline(never)]
fn execute_portable<W: WildCopy, S: BlockSequences>(
    w: W,
    seqs: S,
    offset_hist: &mut [u32; 3],
    prefix_start: usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    seqs.execute(w, offset_hist, prefix_start, out)
}

/// `execute_portable` compiled with AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline(never)]
fn execute_avx2<W: WildCopy, S: BlockSequences>(
    w: W,
    seqs: S,
    offset_hist: &mut [u32; 3],
    prefix_start: usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    seqs.execute(w, offset_hist, prefix_start, out)
}

/// A compressed block's sequences section after its tables, with the
/// block's literals.
#[derive(Clone, Copy)]
struct SeqInput<'a> {
    num_sequences: u32,
    bit_stream: &'a [u8],
    fse: &'a FSEScratch,
    literals: &'a [u8],
}

/// Decoding each sequence and executing it at once. `literals` holds the
/// block's decoded literals followed by exactly `WILDCOPY_OVERLENGTH` bytes
/// of slack.
impl BlockSequences for SeqInput<'_> {
    #[inline(always)]
    fn execute<W: WildCopy>(
        self,
        w: W,
        offset_hist: &mut [u32; 3],
        prefix_start: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), String> {
        let base = out.len();
        // Spare capacity only: the block's bytes are written by the copies
        // in `exec_sequence`, so zero-filling them first is wasted work.
        out.reserve(DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH);
        // SAFETY: `prefix_start <= base <= capacity`, and the capacity holds
        // `DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH` bytes past `base`, which
        // is the extent `run_sequences` may write (see its contract).
        let end = unsafe {
            run_sequences(
                w,
                self,
                offset_hist,
                out.as_mut_ptr().add(prefix_start),
                base - prefix_start,
            )?
        };
        // SAFETY: on success `run_sequences` initialized every byte of
        // `prefix_start + (base - prefix_start)..prefix_start + end`, and
        // `end <= base - prefix_start + DECODED_BLOCK_MAX` keeps the length
        // within the reserved capacity.
        unsafe { out.set_len(prefix_start + end) };
        Ok(())
    }
}

/// Execute the block's sequences into the buffer at `out`, which starts at
/// the frame's first byte; `op` is where this block starts. Returns the
/// block's end, at most `op + DECODED_BLOCK_MAX`, with every byte of
/// `op..end` written; bytes past `end` may have been written too, and
/// nothing before `op` is.
///
/// # Safety
/// `out..out + op` is initialized and
/// `out..out + op + DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH` is valid for
/// writes.
#[inline(always)]
unsafe fn run_sequences<W: WildCopy>(
    w: W,
    seqs: SeqInput<'_>,
    offset_hist: &mut [u32; 3],
    out: *mut u8,
    op: usize,
) -> Result<usize, String> {
    let SeqInput {
        num_sequences,
        bit_stream,
        fse,
        literals,
    } = seqs;
    let ll_dt = fse.literal_lengths.decode();
    let of_dt = fse.offsets.decode();
    let ml_dt = fse.match_lengths.decode();
    let ll_log = u32::from(fse.literal_lengths.accuracy_log);
    let of_log = u32::from(fse.offsets.accuracy_log);
    let ml_log = u32::from(fse.match_lengths.accuracy_log);
    // The state lookups below are unchecked: an initial state is
    // `accuracy_log` bits, and every cell of a table built by
    // `build_decoding_table` or `build_rle` satisfies
    // `next_state + (1 << num_bits) <= table size`, so a state is always a
    // valid index of a table with exactly `1 << accuracy_log` cells.
    if ll_dt.len() != 1 << ll_log || of_dt.len() != 1 << of_log || ml_dt.len() != 1 << ml_log {
        return Err("FSE table is uninitialized".to_string());
    }
    let literals_len = literals.len() - WILDCOPY_OVERLENGTH;
    let oend = op + DECODED_BLOCK_MAX;

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
        }
    };

    for _ in 1..num_sequences {
        let (ll, ml, offset) = decode_sequence(&mut br, &mut st, ll_dt, ml_dt, of_dt, false);
        exec_sequence(w, &mut cur, &lim, ll, ml, offset).map_err(seq_error_message)?;
    }
    let (ll, ml, offset) = decode_sequence(&mut br, &mut st, ll_dt, ml_dt, of_dt, true);
    exec_sequence(w, &mut cur, &lim, ll, ml, offset).map_err(seq_error_message)?;
    let hist = st.hist;
    // Both cursors only ever advance within their slices (see
    // `exec_sequence`), so these differences are in-bounds indexes.
    let mut op = cur.op as usize - out as usize;
    let lit_pos = cur.lit as usize - lit_start as usize;

    if !br.is_finished() {
        return Err("Sequence bitstream not fully consumed".to_string());
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
/// the earliest byte a match may copy from.
struct SeqLimits {
    oend_w: *mut u8,
    lit_limit: *const u8,
    prefix: *mut u8,
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
/// `WILDCOPY_OVERLENGTH` bytes of slack past their limits.
#[inline(always)]
fn exec_sequence<W: WildCopy>(
    w: W,
    cur: &mut SeqCursor,
    lim: &SeqLimits,
    ll: usize,
    ml: usize,
    offset: usize,
) -> Result<(), SeqError> {
    let op = cur.op;
    let lit = cur.lit;
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
        return Err(SeqError::OffsetTooFar);
    }

    // SAFETY: the three checks above give, with `ml >= 1`,
    //   lit + ll <= lit_limit, which has 32 readable bytes after it,
    //   o_match_end <= oend_w, which has 32 writable bytes after it,
    //   1 <= offset <= o_lit_end - prefix.
    // Literals come from another buffer and are read from `lit` and
    // written from `op` with at most 31 bytes of overshoot. The match
    // meets `copy_match`'s contract with `avail = o_lit_end - prefix`:
    // `prefix..o_lit_end` is initialized (by the caller up to the block,
    // then by this block's earlier sequences and these literals), and
    // `o_lit_end + ml + 31` is writable. The advanced cursors keep the
    // `SeqCursor` invariant.
    unsafe {
        // Literals: nearly always at most 16 bytes.
        copy16(op, lit);
        if ll > 16 {
            w.wildcopy(op.add(16), lit.add(16), ll - 16);
        }
        cur.lit = lit.add(ll);

        w.copy_match(op.add(ll), offset, ml, o_lit_end - lim.prefix as usize);
        cur.op = op.add(ll + ml);
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
/// two ranges are disjoint or `dst - src >= 16`.
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
    /// the two ranges are disjoint or `dst - src >= WIDTH`.
    unsafe fn wildcopy(self, dst: *mut u8, src: *const u8, len: usize);

    /// Copy the `ml`-byte match that starts `offset` bytes before `dst`,
    /// which repeats with period `offset` where `offset < ml`, overshooting
    /// by up to 31 bytes (ZSTD_execSequence).
    ///
    /// # Safety
    /// `1 <= offset <= avail` and `ml >= 1`; the `avail` bytes before `dst`
    /// are initialized, and `ml + 31` bytes from `dst` are writable, all in
    /// one allocation.
    #[inline(always)]
    unsafe fn copy_match(self, dst: *mut u8, offset: usize, ml: usize, avail: usize) {
        let _ = avail;
        let src = dst.sub(offset) as *const u8;
        // Sequential chunks stay correct for overlapping periodic matches
        // while `dst - src` is at least the chunk size.
        if offset >= Self::WIDTH {
            self.wildcopy(dst, src, ml);
        } else if offset >= WILDCOPY_VECLEN {
            // Only for `WIDTH > 16`.
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
            // matches.
            copy16(dst, src);
            copy16(dst.add(16), src.add(16));
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
/// below it (ZSTD_overlapCopy8). A match past 32 bytes continues in chunks
/// of 32, or 16 when the distance is shorter.
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
        let src = dst.add(32).sub(dist) as *const u8;
        if dist >= 32 {
            wildcopy32(dst.add(32), src, ml - 32);
        } else {
            wildcopy(dst.add(32), src, ml - 32);
        }
    }
}

/// `wildcopy` in 32-byte chunks (one AVX2 load and store each).
///
/// # Safety
/// The CPU supports AVX2; `len + 31` bytes readable at `src` and writable
/// at `dst`, and either the two ranges are disjoint or `dst - src >= 32`.
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
// Block decoder
// ============================================================

/// Decode every block of one frame from `data[*pos..]` straight into
/// `output`, then skip the checksum. Matches may only reach back to the
/// frame's own start (ZSTD_decompressFrame). Frames of at least
/// `min_parallel_blocks` blocks are decoded by `parallel` when enabled.
#[inline(never)]
fn decode_frame(
    header: &FrameHeader,
    data: &[u8],
    pos: &mut usize,
    scratch: &mut DecoderScratch,
    output: &mut Vec<u8>,
    min_parallel_blocks: usize,
    simd: Level,
) -> Result<(), String> {
    let window_size = header.window_size()?;
    if window_size > MAXIMUM_ALLOWED_WINDOW_SIZE {
        return Err(format!(
            "Window size {} exceeds maximum allowed {}",
            window_size, MAXIMUM_ALLOWED_WINDOW_SIZE
        ));
    }
    // Block_Maximum_Size (fParams.blockSizeMax), the bound `split_block`
    // puts on every compressed block and its literals, and
    // `execute_with_copies`, through `decoded_block_max`, on what the block
    // decodes to.
    let block_size_max = window_size.min(u64::from(MAX_BLOCK_SIZE)) as usize;
    let frame_base = output.len();

    if let Some(fcs) = header.frame_content_size() {
        // Room for the whole frame plus what a compressed block may write
        // past its start, so that no block has to grow the buffer (and
        // move everything decoded).
        let want = usize::try_from(fcs)
            .ok()
            .and_then(|n| n.checked_add(DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH))
            .ok_or_else(|| format!("Frame content size {} too large", fcs))?;
        output
            .try_reserve(want)
            .map_err(|e| format!("Cannot reserve {} bytes of output: {}", want, e))?;
    }

    #[cfg(feature = "parallel")]
    let decoded = parallel::decode_frame_blocks(
        data,
        pos,
        block_size_max,
        frame_base,
        output,
        min_parallel_blocks,
        simd,
    )?;
    #[cfg(not(feature = "parallel"))]
    let decoded = {
        let _ = min_parallel_blocks;
        false
    };
    if !decoded {
        decode_blocks(data, pos, block_size_max, scratch, frame_base, output, simd)?;
    }

    // Skip the checksum if present; this decoder does not verify it.
    if header.descriptor.content_checksum_flag() {
        if data.len() - *pos < 4 {
            return Err("Error reading checksum: truncated".to_string());
        }
        *pos += 4;
    }

    if let Some(fcs) = header.frame_content_size() {
        let decoded = (output.len() - frame_base) as u64;
        if decoded != fcs {
            return Err(format!(
                "Frame content size mismatch: header says {}, decoded {}",
                fcs, decoded
            ));
        }
    }
    Ok(())
}

/// The serial block loop of `decode_frame`: decode every block of the
/// frame at `data[*pos..]` into `output`.
fn decode_blocks(
    data: &[u8],
    pos: &mut usize,
    block_size_max: usize,
    scratch: &mut DecoderScratch,
    frame_base: usize,
    output: &mut Vec<u8>,
    simd: Level,
) -> Result<(), String> {
    loop {
        let (block, header_len) = parse_block_header(&data[*pos..])?;
        *pos += header_len;
        let content = data
            .get(*pos..*pos + block.content_size as usize)
            .ok_or_else(|| "Block content extends past end of input".to_string())?;
        *pos += content.len();

        match block.block_type {
            BlockType::Raw => output.extend_from_slice(content),
            BlockType::RLE => {
                output.resize(output.len() + block.decompressed_size as usize, content[0])
            }
            BlockType::Compressed => {
                decompress_block(content, block_size_max, scratch, frame_base, output, simd)?
            }
            BlockType::Reserved => return Err("Reserved block type encountered".to_string()),
        }

        if block.last_block {
            break;
        }
    }
    Ok(())
}

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

/// Locate the sections of compressed block `raw` in a frame whose
/// Block_Maximum_Size is `block_size_max`, which bounds the block and its
/// literals (ZSTD_decompressBlock_internal, ZSTD_decodeLiteralsBlock).
fn split_block(raw: &[u8], block_size_max: usize) -> Result<BlockParts<'_>, String> {
    if raw.len() > block_size_max {
        return Err(format!(
            "Compressed block size {} exceeds Block_Maximum_Size {}",
            raw.len(),
            block_size_max
        ));
    }
    let mut section = LiteralsSection::new();
    let bytes_in_literals_header = section.parse_from_header(raw)?;
    if section.regenerated_size as usize > block_size_max {
        return Err(format!(
            "Literals size {} exceeds Block_Maximum_Size {}",
            section.regenerated_size, block_size_max
        ));
    }
    let raw = &raw[bytes_in_literals_header as usize..];

    let upper_limit_for_literals = match section.compressed_size {
        Some(x) => x as usize,
        None => match section.ls_type {
            LiteralsSectionType::RLE => 1,
            LiteralsSectionType::Raw => section.regenerated_size as usize,
            _ => return Err("Bug: unexpected literals section type".to_string()),
        },
    };

    if raw.len() < upper_limit_for_literals {
        return Err(format!(
            "Malformed section header: expected {} bytes, have {}",
            upper_limit_for_literals,
            raw.len()
        ));
    }

    let literals_src = &raw[..upper_limit_for_literals];
    let raw = &raw[upper_limit_for_literals..];

    let mut sequences = SequencesHeader::new();
    let bytes_in_sequence_header = sequences.parse_from_header(raw)?;
    Ok(BlockParts {
        literals: section,
        literals_src,
        sequences,
        sequences_src: &raw[bytes_in_sequence_header as usize..],
    })
}

/// Decode the block's literals into `target` (cleared first) and append
/// `WILDCOPY_OVERLENGTH` bytes of slack.
fn decode_block_literals(
    parts: &BlockParts<'_>,
    huf: &mut HuffmanScratch,
    target: &mut Vec<u8>,
) -> Result<(), String> {
    target.clear();
    let used = decode_literals(&parts.literals, huf, parts.literals_src, target)?;
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

fn decompress_block(
    raw: &[u8],
    block_size_max: usize,
    workspace: &mut DecoderScratch,
    frame_base: usize,
    output: &mut Vec<u8>,
    simd: Level,
) -> Result<(), String> {
    let parts = split_block(raw, block_size_max)?;
    decode_block_literals(&parts, &mut workspace.huf, &mut workspace.literals_buffer)?;
    let literals_len = workspace.literals_buffer.len() - WILDCOPY_OVERLENGTH;
    let seq_section = parts.sequences;
    let raw = parts.sequences_src;

    if seq_section.num_sequences != 0 {
        let table_bytes = build_sequence_tables(&seq_section, raw, &mut workspace.fse)?;
        let seqs = SeqInput {
            num_sequences: seq_section.num_sequences,
            bit_stream: &raw[table_bytes..],
            fse: &workspace.fse,
            literals: &workspace.literals_buffer,
        };
        execute_with_copies(
            simd,
            &workspace.fse.offsets,
            seqs,
            &mut workspace.offset_hist,
            block_size_max,
            frame_base,
            output,
        )?;
    } else {
        if !raw.is_empty() {
            return Err(format!(
                "Extra bits remaining: {} bits",
                raw.len() as isize * 8
            ));
        }
        output.extend_from_slice(&workspace.literals_buffer[..literals_len]);
    }

    Ok(())
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
// ============================================================

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Frames with fewer blocks take the fused serial path.
    pub(super) const MIN_BLOCKS: usize = 4;

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
        /// block with Compressed literals for Treeless ones.
        huf_def: Option<usize>,
        /// For LL, OF and ML (`SEQ_TABLES` order), the block whose mode
        /// defined the table this block uses. Unused without sequences.
        fse_def: [usize; 3],
    }

    fn compressed<'p, 'a>(plan: &'p Plan<'a>) -> &'p CompressedPlan<'a> {
        match plan {
            Plan::Compressed(c) => c,
            // `plan_frame` only records compressed blocks as definitions.
            _ => unreachable!("table definition in a non-compressed block"),
        }
    }

    /// Stage 1: the block loop of ZSTD_decompressFrame, locating blocks
    /// and resolving Treeless / Repeat references the way the serial
    /// decoder's scratch tables carry them from block to block.
    fn plan_frame<'a>(
        data: &'a [u8],
        pos: &mut usize,
        block_size_max: usize,
    ) -> Result<Vec<Plan<'a>>, String> {
        let mut plans = Vec::new();
        let mut huf_def = None;
        let mut fse_def: [Option<usize>; 3] = [None; 3];
        loop {
            let (block, header_len) = parse_block_header(&data[*pos..])?;
            *pos += header_len;
            let content = data
                .get(*pos..*pos + block.content_size as usize)
                .ok_or_else(|| "Block content extends past end of input".to_string())?;
            *pos += content.len();
            let i = plans.len();
            plans.push(match block.block_type {
                BlockType::Raw => Plan::Raw(content),
                BlockType::RLE => Plan::Rle(content[0], block.decompressed_size as usize),
                BlockType::Compressed => {
                    let parts = split_block(content, block_size_max)?;
                    let huf = match parts.literals.ls_type {
                        LiteralsSectionType::Compressed => {
                            huf_def = Some(i);
                            huf_def
                        }
                        LiteralsSectionType::Treeless => Some(huf_def.ok_or_else(|| {
                            "Uninitialized Huffman table for treeless literals".to_string()
                        })?),
                        LiteralsSectionType::Raw | LiteralsSectionType::RLE => None,
                    };
                    let mut defs = [0; 3];
                    if parts.sequences.num_sequences != 0 {
                        let modes = parts
                            .sequences
                            .modes
                            .ok_or_else(|| "Missing compression mode".to_string())?;
                        for (t, mode) in modes.all().into_iter().enumerate() {
                            if !matches!(mode, ModeType::Repeat) {
                                fse_def[t] = Some(i);
                            }
                            defs[t] = fse_def[t].ok_or_else(|| {
                                format!(
                                    "Repeat mode without a previous {} table",
                                    SEQ_TABLES[t].name
                                )
                            })?;
                        }
                    }
                    Plan::Compressed(CompressedPlan {
                        parts,
                        huf_def: huf,
                        fse_def: defs,
                    })
                }
                BlockType::Reserved => return Err("Reserved block type encountered".to_string()),
            });
            if block.last_block {
                return Ok(plans);
            }
        }
    }

    /// One sequence before repeat-offset resolution.
    #[derive(Clone, Copy)]
    struct RawSeq {
        ll: u32,
        ml: u32,
        /// libzstd OFFBASE: 1..=3 name a repeat offset, larger values are
        /// the offset plus `ZSTD_REP_NUM`.
        off_base: u32,
    }

    /// A ring position: its slot, and the index plus one of the last block
    /// claimed for decoding into it and of the last block decoded into it.
    struct RingSlot {
        claimed: AtomicUsize,
        done: AtomicUsize,
        slot: Mutex<Slot>,
    }

    impl RingSlot {
        /// Take block `i`'s decode; false if someone already has. A task
        /// that runs after block `i` has been decoded and its position
        /// reused finds a later block's claim and fails too.
        fn claim(&self, i: usize) -> bool {
            self.claimed.fetch_max(i + 1, Ordering::AcqRel) < i + 1
        }
    }

    /// Publishes a finished decode on drop.
    struct MarkDone<'a>(&'a AtomicUsize, usize);

    impl Drop for MarkDone<'_> {
        fn drop(&mut self) {
            self.0.store(self.1, Ordering::Release);
        }
    }

    /// Per-ring-position worker state: tables tagged with the block that
    /// defined them (so a run of blocks reusing one table builds it once),
    /// and the decoded literals and sequences of the current block.
    struct Slot {
        huf: HuffmanScratch,
        huf_from: Option<usize>,
        fse: FSEScratch,
        fse_from: [Option<usize>; 3],
        literals: Vec<u8>,
        seqs: Vec<RawSeq>,
        result: Result<(), String>,
    }

    impl Slot {
        fn new() -> Slot {
            let scratch = DecoderScratch::new();
            Slot {
                huf: scratch.huf,
                huf_from: None,
                fse: scratch.fse,
                fse_from: [None; 3],
                literals: Vec::new(),
                seqs: Vec::new(),
                result: Ok(()),
            }
        }
    }

    /// Stage 2 for compressed block `i`.
    fn decode_block(
        slot: &mut Slot,
        i: usize,
        plan: &CompressedPlan<'_>,
        plans: &[Plan<'_>],
    ) -> Result<(), String> {
        if let Some(d) = plan.huf_def {
            if d == i {
                // `decode_block_literals` builds it from this block.
                slot.huf_from = None;
            } else if slot.huf_from != Some(d) {
                slot.huf_from = None;
                // The same arguments as the defining block's own build in
                // `decompress_literals`, so the same table kind (X1 / X2).
                let def = compressed(&plans[d]);
                let lit = &def.parts.literals;
                slot.huf.table.build_decoder(
                    def.parts.literals_src,
                    lit.regenerated_size as usize,
                    lit.num_streams == Some(4),
                )?;
                slot.huf_from = Some(d);
            }
        }
        decode_block_literals(&plan.parts, &mut slot.huf, &mut slot.literals)?;
        if plan.huf_def == Some(i) {
            slot.huf_from = Some(i);
        }

        slot.seqs.clear();
        let seq = plan.parts.sequences;
        let src = plan.parts.sequences_src;
        if seq.num_sequences == 0 {
            if !src.is_empty() {
                return Err(format!(
                    "Extra bits remaining: {} bits",
                    src.len() as isize * 8
                ));
            }
            return Ok(());
        }
        let modes = seq
            .modes
            .ok_or_else(|| "Missing compression mode".to_string())?;
        let mut used = 0;
        for (t, mode) in modes.all().into_iter().enumerate() {
            let d = plan.fse_def[t];
            if d == i {
                slot.fse_from[t] = None;
                used += build_sequence_table(
                    mode,
                    &src[used..],
                    slot.fse.table_mut(t),
                    &SEQ_TABLES[t],
                )?;
                slot.fse_from[t] = Some(i);
            } else if slot.fse_from[t] != Some(d) {
                slot.fse_from[t] = None;
                build_table_from(compressed(&plans[d]), t, slot.fse.table_mut(t))?;
                slot.fse_from[t] = Some(d);
            }
        }
        decode_sequences(seq.num_sequences, &src[used..], &slot.fse, &mut slot.seqs)
    }

    /// Build table `t` from its description in the earlier block `def`,
    /// skipping the descriptions that precede it there.
    fn build_table_from(
        def: &CompressedPlan<'_>,
        t: usize,
        table: &mut FSETable,
    ) -> Result<(), String> {
        let modes = def
            .parts
            .sequences
            .modes
            .ok_or_else(|| "Missing compression mode".to_string())?
            .all();
        let src = def.parts.sequences_src;
        let mut used = 0;
        for (u, kind) in SEQ_TABLES.iter().enumerate().take(t) {
            used += match modes[u] {
                ModeType::FSECompressed => {
                    FSETable::new(kind.max_code).read_probabilities(&src[used..], kind.max_log)?
                }
                ModeType::RLE if used < src.len() => 1,
                ModeType::RLE => return Err(format!("Missing byte for RLE {} table", kind.name)),
                ModeType::Predefined | ModeType::Repeat => 0,
            };
        }
        build_sequence_table(modes[t], &src[used..], table, &SEQ_TABLES[t]).map(|_| ())
    }

    /// The block's three sequence tables, checked for the unchecked state
    /// lookups of `decode_raw_sequence`, and the bitstream positioned after
    /// the initial states (ZSTD_initFseState). Same checks and reads as the
    /// start of `run_sequences`.
    fn seq_stream_begin<'a>(
        bit_stream: &'a [u8],
        fse: &'a FSEScratch,
    ) -> Result<SeqStream<'a>, String> {
        let tables = [&fse.literal_lengths, &fse.offsets, &fse.match_lengths];
        let logs = tables.map(|t| u32::from(t.accuracy_log));
        // Every state is `accuracy_log` bits or `next_state + bits` of a
        // cell, which `build_decoding_table` / `build_rle` keep below
        // `1 << accuracy_log`, the table length checked here.
        if tables
            .iter()
            .zip(logs)
            .any(|(t, log)| t.decode().len() != 1 << log)
        {
            return Err("FSE table is uninitialized".to_string());
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

    /// Stage 2's sequence loop: `run_sequences` without execution.
    fn decode_sequences(
        num_sequences: u32,
        bit_stream: &[u8],
        fse: &FSEScratch,
        seqs: &mut Vec<RawSeq>,
    ) -> Result<(), String> {
        let (mut br, [ll, of, ml], [ll_dt, of_dt, ml_dt]) = seq_stream_begin(bit_stream, fse)?;
        let mut st = [ll, ml, of];
        // Zero-fill first: when the executing thread last read these lines
        // from another CCD, the fill's bulk stores take ownership of them
        // at memory bandwidth, where the loop's 12-byte stores would stall
        // on one cross-CCD invalidation per line (4x slower on Zen 5).
        let n = num_sequences as usize;
        seqs.clear();
        seqs.reserve(n);
        // SAFETY: `n` elements are reserved, and zero bytes are a valid
        // `RawSeq` (three u32s).
        unsafe {
            ptr::write_bytes(seqs.as_mut_ptr(), 0, n);
            seqs.set_len(n);
        }
        let (last, rest) = seqs
            .split_last_mut()
            .ok_or_else(|| "Missing sequences".to_string())?;
        for s in rest {
            *s = decode_raw_sequence(&mut br, &mut st, ll_dt, ml_dt, of_dt, false);
        }
        *last = decode_raw_sequence(&mut br, &mut st, ll_dt, ml_dt, of_dt, true);
        if !br.is_finished() {
            return Err("Sequence bitstream not fully consumed".to_string());
        }
        Ok(())
    }

    /// `decode_sequence` with the offset left as OFFBASE; `st` is the LL,
    /// ML, OF states. Kept separate from `decode_sequence`: sharing one
    /// body changed the fused loop's register allocation and cost it 2-3%.
    #[inline(always)]
    fn decode_raw_sequence(
        br: &mut BitDStream<'_>,
        st: &mut [usize; 3],
        ll_dt: &[FSEEntry],
        ml_dt: &[FSEEntry],
        of_dt: &[FSEEntry],
        is_last: bool,
    ) -> RawSeq {
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

        // Offset codes 0 and 1 are repeat codes (base value 0, or base
        // value 1 plus one extra bit) and give 1..=3; larger codes carry
        // the offset, stored plus ZSTD_REP_NUM.
        let off_base = if of_bits > 1 {
            of_e.base_value as usize + br.read_bits_fast(of_bits) + ZSTD_REP_NUM
        } else if of_bits == 1 {
            of_e.base_value as usize + br.read_bits_fast(1) + 1
        } else {
            of_e.base_value as usize + 1
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
        // Lengths are below 2^17 and OFFBASE at most 2^32 - 1 (offset code
        // 31: base 2^31 - 3 plus 31 extra bits plus 3), so all fit in u32.
        RawSeq {
            ll: ll as u32,
            ml: ml as u32,
            off_base: off_base as u32,
        }
    }

    /// The repeat-offset update of `decode_sequence`, applied to OFFBASE.
    /// `ll` is the sequence's literal length.
    #[inline(always)]
    fn resolve_offset(hist: &mut [usize; 3], off_base: usize, ll: usize) -> usize {
        if off_base > ZSTD_REP_NUM {
            let o = off_base - ZSTD_REP_NUM;
            hist[2] = hist[1];
            hist[1] = hist[0];
            hist[0] = o;
            return o;
        }
        // Without literals the repeat codes shift by one: code 1 names the
        // second offset, and code 3 means the first offset minus one.
        let idx = off_base - 1 + usize::from(ll == 0);
        if idx == 0 {
            return hist[0];
        }
        let mut temp = if idx == 3 {
            hist[0].wrapping_sub(1)
        } else {
            hist[idx]
        };
        if temp == 0 {
            // Corrupt input: force an offset that execution rejects.
            temp = usize::MAX;
        }
        if idx != 1 {
            hist[2] = hist[1];
        }
        hist[1] = hist[0];
        hist[0] = temp;
        temp
    }

    /// Stage 3 for one block.
    fn execute_block(
        plan: &Plan<'_>,
        slot: &mut Slot,
        hist: &mut [u32; 3],
        block_size_max: usize,
        frame_base: usize,
        output: &mut Vec<u8>,
        simd: Level,
    ) -> Result<(), String> {
        match plan {
            Plan::Raw(content) => output.extend_from_slice(content),
            Plan::Rle(byte, len) => output.resize(output.len() + len, *byte),
            Plan::Compressed(cp) => {
                std::mem::replace(&mut slot.result, Ok(()))?;
                let literals_len = slot.literals.len() - WILDCOPY_OVERLENGTH;
                if cp.parts.sequences.num_sequences == 0 {
                    output.extend_from_slice(&slot.literals[..literals_len]);
                    return Ok(());
                }
                let seqs = DecodedSeqs {
                    seqs: &slot.seqs,
                    literals: &slot.literals,
                };
                execute_with_copies(
                    simd,
                    &slot.fse.offsets,
                    seqs,
                    hist,
                    block_size_max,
                    frame_base,
                    output,
                )?;
            }
        }
        Ok(())
    }

    /// A block's sequences as stage 2 decoded them, with its literals
    /// followed by `WILDCOPY_OVERLENGTH` bytes of slack.
    struct DecodedSeqs<'a> {
        seqs: &'a [RawSeq],
        literals: &'a [u8],
    }

    impl BlockSequences for DecodedSeqs<'_> {
        #[inline(always)]
        fn execute<W: WildCopy>(
            self,
            w: W,
            offset_hist: &mut [u32; 3],
            prefix_start: usize,
            out: &mut Vec<u8>,
        ) -> Result<(), String> {
            let base = out.len();
            out.reserve(DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH);
            // SAFETY: `prefix_start <= base`, and the capacity holds
            // `DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH` bytes past `base`,
            // the extent `execute_sequences` may write.
            let end = unsafe {
                execute_sequences(
                    w,
                    self.seqs,
                    self.literals,
                    offset_hist,
                    out.as_mut_ptr().add(prefix_start),
                    base - prefix_start,
                )?
            };
            // SAFETY: on success every byte up to `prefix_start + end` is
            // initialized, within the reserved capacity.
            unsafe { out.set_len(prefix_start + end) };
            Ok(())
        }
    }

    /// Execute decoded sequences from `op` in the buffer at `out` (the
    /// frame start); returns the block's end. Same contract as
    /// `run_sequences`.
    ///
    /// # Safety
    /// `out..out + op` is initialized and
    /// `out..out + op + DECODED_BLOCK_MAX + WILDCOPY_OVERLENGTH` is writable.
    #[inline(always)]
    unsafe fn execute_sequences<W: WildCopy>(
        w: W,
        seqs: &[RawSeq],
        literals: &[u8],
        offset_hist: &mut [u32; 3],
        out: *mut u8,
        op: usize,
    ) -> Result<usize, String> {
        let mut hist = offset_hist.map(|o| o as usize);
        let lit = literals.as_ptr();
        // In bounds by the contract, and `literals` ends with
        // `WILDCOPY_OVERLENGTH` bytes of slack.
        let mut cur = SeqCursor {
            op: out.add(op),
            lit,
        };
        let lim = SeqLimits {
            oend_w: out.add(op + DECODED_BLOCK_MAX),
            lit_limit: lit.add(literals.len() - WILDCOPY_OVERLENGTH),
            prefix: out,
        };
        for s in seqs {
            let ll = s.ll as usize;
            let offset = resolve_offset(&mut hist, s.off_base as usize, ll);
            exec_sequence(w, &mut cur, &lim, ll, s.ml as usize, offset)
                .map_err(seq_error_message)?;
        }
        // Last literals; both cursors only advanced within their buffers.
        let rest = lim.lit_limit as usize - cur.lit as usize;
        if cur.op as usize + rest > lim.oend_w as usize {
            return Err(seq_error_message(SeqError::BlockTooLarge));
        }
        ptr::copy_nonoverlapping(cur.lit, cur.op, rest);
        *offset_hist = hist.map(|o| o as u32);
        Ok(cur.op as usize + rest - out as usize)
    }

    /// Decode the blocks of the frame at `data[*pos..]` into `output` on the
    /// current rayon pool. Returns `Ok(false)` without consuming input when
    /// the frame has fewer than `min_blocks` blocks.
    pub(super) fn decode_frame_blocks(
        data: &[u8],
        pos: &mut usize,
        block_size_max: usize,
        frame_base: usize,
        output: &mut Vec<u8>,
        min_blocks: usize,
        simd: Level,
    ) -> Result<bool, String> {
        if min_blocks == usize::MAX {
            return Ok(false);
        }
        let mut end = *pos;
        let plans = plan_frame(data, &mut end, block_size_max)?;
        if plans.len() < min_blocks {
            return Ok(false);
        }
        let plans = &plans[..];

        // Block `i` is decoded into `ring[i % ring.len()]` by a rayon task
        // spawned once block `i - ring.len()` has been executed from it, or
        // by the executing thread if no task has started it by the time
        // that thread needs block `i`, or waits for block `i - 1`. It
        // decodes no block further ahead, so a block it needs is never left
        // waiting behind the decode of a later one.
        let ring: Vec<RingSlot> = (0..(2 * rayon::current_num_threads()).min(plans.len()))
            .map(|_| RingSlot {
                claimed: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                slot: Mutex::new(Slot::new()),
            })
            .collect();
        let ring = &ring[..];
        let mut hist = [1u32, 4, 8];
        rayon::scope_fifo(|s| {
            let spawn_decode = |i: usize| {
                let Some(Plan::Compressed(cp)) = plans.get(i) else {
                    return;
                };
                let cell = &ring[i % ring.len()];
                s.spawn_fifo(move |_| {
                    if !cell.claim(i) {
                        return;
                    }
                    // Marks the block done even if decoding panics, so that
                    // the executing thread finds the poisoned lock instead
                    // of waiting forever.
                    let _done = MarkDone(&cell.done, i + 1);
                    let mut slot = cell.slot.lock().unwrap();
                    slot.result = decode_block(&mut slot, i, cp, plans);
                });
            };
            for i in 0..ring.len() {
                spawn_decode(i);
            }
            for (i, plan) in plans.iter().enumerate() {
                let cell = &ring[i % ring.len()];
                let mut slot = match plan {
                    Plan::Compressed(cp) if cell.claim(i) => {
                        let mut slot = cell.slot.lock().unwrap();
                        slot.result = decode_block(&mut slot, i, cp, plans);
                        slot
                    }
                    Plan::Compressed(_) => {
                        while cell.done.load(Ordering::Acquire) != i + 1 {
                            // If no task has started block `i + 1` either,
                            // the decoders are behind: decode it here while
                            // block `i` finishes. Its position is free, as
                            // block `i + 1 - ring.len()` has been executed.
                            let next = &ring[(i + 1) % ring.len()];
                            match plans.get(i + 1) {
                                Some(Plan::Compressed(np)) if next.claim(i + 1) => {
                                    let _done = MarkDone(&next.done, i + 2);
                                    let mut slot = next.slot.lock().unwrap();
                                    slot.result = decode_block(&mut slot, i + 1, np, plans);
                                }
                                // Hand the CPU to a worker the kernel may
                                // have queued on it.
                                _ => std::thread::yield_now(),
                            }
                        }
                        cell.slot.lock().unwrap()
                    }
                    _ => cell.slot.lock().unwrap(),
                };
                execute_block(
                    plan,
                    &mut slot,
                    &mut hist,
                    block_size_max,
                    frame_base,
                    output,
                    simd,
                )?;
                drop(slot);
                spawn_decode(i + ring.len());
            }
            Ok::<(), String>(())
        })?;
        *pos = end;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `copy_match` against a byte-at-a-time copy at every offset and
    /// every length up to 100, with the bytes before the match both fewer
    /// and more than the 32 that `copy_match_short` loads.
    fn check_copy_match<W: WildCopy>(w: W, name: &str) {
        let init: Vec<u8> = (0..100u8).map(|i| i.wrapping_mul(37) ^ 0x5a).collect();
        for avail in [1, 2, 7, 8, 15, 16, 31, 32, 33, 47, 64, 100] {
            for offset in 1..=avail {
                for ml in 1..=100 {
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
    }

    /// (description length, weights with the implied last one, rank
    /// statistics of weights 0 to 12, table log) of a tree description.
    type HufStats = (usize, Vec<u8>, Vec<u32>, u32);

    fn huf_stats_c(src: &[u8]) -> Option<HufStats> {
        let mut weights = [0u8; 256];
        let mut rank_stats = [0u32; 13];
        let (mut nb_symbols, mut table_log) = (0u32, 0u32);
        // SAFETY: the buffers have the sizes HUF_readStats is given (rankStats
        // takes HUF_TABLELOG_MAX + 1 = 13 entries).
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
        Some((
            used,
            t.weights[..n].to_vec(),
            t.rank_stats.to_vec(),
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

    /// `read_weights` + `weight_stats` against HUF_readStats on libzstd's
    /// descriptions of random codes (raw and FSE-compressed, 2 to 256
    /// symbols, 6- to 12-bit), every truncation of them, single-byte
    /// corruptions and random bytes: the same outcome, and on success the
    /// same length, weights, statistics and table log. Hundreds of the
    /// corrupted and random inputs are valid descriptions, and over 150 of
    /// the accepted inputs describe 12-bit codes, which the format excludes
    /// and libzstd decodes.
    #[test]
    fn huf_stats_match_libzstd() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut log12_ok = 0;
        let mut check = |src: &[u8]| {
            let ours = huf_stats_ours(src);
            assert_eq!(ours, huf_stats_c(src), "input {src:02x?}");
            log12_ok += usize::from(matches!(ours, Some((.., 12))));
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
            bad_ok > 500 && random_ok > 1000 && log12_ok > 150,
            "{bad_ok} {random_ok} accepted, {log12_ok} of 12 bits"
        );
    }

    /// Raw 4-bit weights (an odd number of them, and those of a 12-bit
    /// code) through the statistics and both fills, each build over the
    /// previous one: the tables have `max(max_bits, 11)` bits, each symbol
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
        // Weights 12 down to 1 sum to 4095 halves; the implied 13th is 1.
        let long: (&[u8], &[u8], u32) = (
            &[127 + 12, 0xcb, 0xa9, 0x87, 0x65, 0x43, 0x21],
            &[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1],
            12,
        );
        let mut t = HuffmanTable::new();
        for (src, weights, max_bits) in [short, long, short] {
            let n = weights.len();
            let (used, nb_weights) = t.read_weights(src).unwrap();
            assert_eq!((used, nb_weights), (src.len(), n - 1));
            t.weight_stats(nb_weights).unwrap();
            assert_eq!((t.nb_symbols, u32::from(t.max_num_bits)), (n, max_bits));
            assert_eq!(&t.weights[..n], weights);
            t.fill_x1();
            t.fill_x2();
            let log = t.dt_log();
            assert_eq!(log, max_bits.max(HUF_FAST_TABLE_LOG));
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
        t.build_from_probabilities(of.default_log, of.default_distribution, codes)
            .unwrap();
        assert_eq!(short_offset_share(&t), scan(&t), "predefined");
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
        fn execute<W: WildCopy>(
            self,
            _: W,
            _: &mut [u32; 3],
            _: usize,
            _: &mut Vec<u8>,
        ) -> Result<(), String> {
            let name = std::any::type_name::<W>();
            Err(name.rsplit("::").next().unwrap_or(name).to_string())
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
            execute_with_copies(level, t, CopyProbe, &mut [1, 4, 8], 0, 0, &mut Vec::new())
                .unwrap_err()
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

    /// libzstd's default limit: a 128 MiB window (exponent 17) decodes,
    /// the next larger one (mantissa 1, 144 MiB) is refused.
    #[test]
    fn test_window_limit_is_zstd_default() {
        assert_eq!(decompress(&windowed_frame(17 << 3)).unwrap(), b"hi");
        let err = decompress(&windowed_frame((17 << 3) | 1)).unwrap_err();
        assert!(err.contains("exceeds maximum allowed"), "{err}");
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
}
