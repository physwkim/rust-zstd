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

// ============================================================
// Constants
// ============================================================

const ZSTD_MAGIC: u32 = 0xFD2F_B528;
const MIN_WINDOW_SIZE: u64 = 1024;
const MAX_WINDOW_SIZE: u64 = (1 << 41) + 7 * (1 << 38);
const MAX_BLOCK_SIZE: u32 = 128 * 1024;
const MAXIMUM_ALLOWED_WINDOW_SIZE: u64 = 1024 * 1024 * 100;
const MAX_MAX_NUM_BITS: u8 = 11;
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
    let _ = ht.read_weights(&full)?;
    Ok(ht.weights.clone())
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
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
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

        let window_size = frame_header.window_size()?;
        if window_size > MAXIMUM_ALLOWED_WINDOW_SIZE {
            return Err(format!(
                "Window size {} exceeds maximum allowed {}",
                window_size, MAXIMUM_ALLOWED_WINDOW_SIZE
            ));
        }

        let scratch = scratch.get_or_insert_with(DecoderScratch::new);
        scratch.reset();
        decode_frame(&frame_header, data, &mut pos, scratch, &mut output)?;
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
// BitReaderReversed
// ============================================================

struct BitReaderReversed<'s> {
    index: usize,
    bits_consumed: u8,
    extra_bits: usize,
    source: &'s [u8],
    bit_container: u64,
}

impl<'s> BitReaderReversed<'s> {
    fn bits_remaining(&self) -> isize {
        self.index as isize * 8 + (64 - self.bits_consumed as isize) - self.extra_bits as isize
    }

    fn new(source: &'s [u8]) -> BitReaderReversed<'s> {
        BitReaderReversed {
            index: source.len(),
            bits_consumed: 64,
            source,
            bit_container: 0,
            extra_bits: 0,
        }
    }

    #[cold]
    fn refill(&mut self) {
        let bytes_consumed = self.bits_consumed as usize / 8;
        if bytes_consumed == 0 {
            return;
        }

        if self.index >= bytes_consumed {
            self.index -= bytes_consumed;
            self.bits_consumed &= 7;
            let remaining = self.source.len() - self.index;
            if remaining >= 8 {
                self.bit_container =
                    u64::from_le_bytes(self.source[self.index..][..8].try_into().unwrap());
            } else {
                let mut value = [0u8; 8];
                value[..remaining].copy_from_slice(&self.source[self.index..]);
                self.bit_container = u64::from_le_bytes(value);
            }
        } else if self.index > 0 {
            if self.source.len() >= 8 {
                self.bit_container = u64::from_le_bytes(self.source[..8].try_into().unwrap());
            } else {
                let mut value = [0; 8];
                value[..self.source.len()].copy_from_slice(self.source);
                self.bit_container = u64::from_le_bytes(value);
            }

            self.bits_consumed -= 8 * self.index as u8;
            self.index = 0;

            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else if self.bits_consumed < 64 {
            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else {
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            self.bit_container = 0;
        }

        debug_assert!(self.bits_consumed < 8);
    }

    #[inline(always)]
    fn get_bits(&mut self, n: u8) -> u64 {
        if self.bits_consumed + n > 64 {
            self.refill();
        }
        let value = self.peek_bits(n);
        self.consume(n);
        value
    }

    #[inline(always)]
    fn peek_bits(&mut self, n: u8) -> u64 {
        if n == 0 {
            return 0;
        }
        let mask = (1u64 << n) - 1u64;
        let shift_by = 64 - self.bits_consumed - n;
        (self.bit_container >> shift_by) & mask
    }

    #[inline(always)]
    fn peek_bits_triple(&mut self, sum: u8, n1: u8, n2: u8, n3: u8) -> (u64, u64, u64) {
        if sum == 0 {
            return (0, 0, 0);
        }
        let all_three = self.bit_container >> (64 - self.bits_consumed - sum);

        let mask1 = (1u64 << n1) - 1u64;
        let val1 = (all_three >> (n3 + n2)) & mask1;

        let mask2 = (1u64 << n2) - 1u64;
        let val2 = (all_three >> n3) & mask2;

        let mask3 = (1u64 << n3) - 1u64;
        let val3 = all_three & mask3;

        (val1, val2, val3)
    }

    #[inline(always)]
    fn consume(&mut self, n: u8) {
        self.bits_consumed += n;
        debug_assert!(self.bits_consumed <= 64);
    }

    #[inline(always)]
    fn get_bits_triple(&mut self, n1: u8, n2: u8, n3: u8) -> (u64, u64, u64) {
        let sum = n1 + n2 + n3;
        if sum <= 56 {
            self.refill();
            let triple = self.peek_bits_triple(sum, n1, n2, n3);
            self.consume(sum);
            return triple;
        }
        (self.get_bits(n1), self.get_bits(n2), self.get_bits(n3))
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

#[derive(Debug, Clone)]
struct FSETable {
    max_symbol: u8,
    decode: Vec<FSEEntry>,
    accuracy_log: u8,
    symbol_probabilities: Vec<i32>,
    /// Per-symbol next-state counter while building (libzstd symbolNext).
    symbol_counter: Vec<u32>,
    /// Symbols laid out in order before spreading (libzstd spread).
    spread: Vec<u8>,
    /// True while `decode` holds a predefined sequence distribution, so the
    /// next block in Predefined mode can reuse it without rebuilding.
    predefined: bool,
}

impl FSETable {
    fn new(max_symbol: u8) -> FSETable {
        FSETable {
            max_symbol,
            symbol_probabilities: Vec::with_capacity(256),
            symbol_counter: Vec::with_capacity(256),
            spread: Vec::new(),
            decode: Vec::new(),
            accuracy_log: 0,
            predefined: false,
        }
    }

    fn reset(&mut self) {
        self.symbol_counter.clear();
        self.symbol_probabilities.clear();
        self.decode.clear();
        self.accuracy_log = 0;
        self.predefined = false;
    }

    /// One-cell table for an RLE-coded sequence section
    /// (ZSTD_buildSeqTable_rle): accuracy log 0, no state bits.
    fn build_rle(&mut self, symbol: u8, base: &[u32], bits: &[u8]) {
        self.reset();
        self.decode.push(FSEEntry {
            next_state: 0,
            num_bits: 0,
            extra_bits: bits[symbol as usize],
            base_value: base[symbol as usize],
        });
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
        self.accuracy_log = 0;
        self.predefined = false;
        let bytes_read = self.read_probabilities(source, max_log)?;
        self.build_decoding_table(codes)?;
        Ok(bytes_read)
    }

    fn build_from_probabilities(
        &mut self,
        acc_log: u8,
        probs: &[i32],
        codes: Option<(&[u32], &[u8])>,
    ) -> Result<(), String> {
        if acc_log == 0 {
            return Err("Accuracy log is zero".to_string());
        }
        self.symbol_probabilities.clear();
        self.symbol_probabilities.extend_from_slice(probs);
        self.accuracy_log = acc_log;
        self.predefined = false;
        self.build_decoding_table(codes)
    }

    /// Port of ZSTD_buildFSETable_body: lay low-probability symbols at the
    /// top, spread the rest, then derive each cell's bit count and next
    /// state from a per-symbol counter in one pass.
    fn build_decoding_table(&mut self, codes: Option<(&[u32], &[u8])>) -> Result<(), String> {
        let num_symbols = self.symbol_probabilities.len();
        if num_symbols > self.max_symbol as usize + 1 {
            return Err(format!(
                "Too many symbols: {}, max: {}",
                num_symbols,
                self.max_symbol + 1
            ));
        }
        let table_log = u32::from(self.accuracy_log);
        let table_size = 1usize << table_log;
        let total: i64 = self
            .symbol_probabilities
            .iter()
            .map(|&p| if p < 0 { 1 } else { i64::from(p) })
            .sum();
        if total != table_size as i64 {
            return Err(format!(
                "FSE probabilities sum to {}, expected {}",
                total, table_size
            ));
        }

        self.decode.clear();
        self.decode.resize(table_size, FSEEntry::default());
        self.symbol_counter.clear();
        self.symbol_counter.resize(num_symbols, 0);

        // Low-probability symbols occupy the highest cells.
        let mut high_threshold = table_size;
        for (symbol, &prob) in self.symbol_probabilities.iter().enumerate() {
            if prob == -1 {
                high_threshold -= 1;
                self.decode[high_threshold].base_value = symbol as u32;
                self.symbol_counter[symbol] = 1;
            } else {
                self.symbol_counter[symbol] = prob as u32;
            }
        }

        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;
        if high_threshold == table_size {
            // No low-probability symbols: lay the symbols down in order with
            // 8-byte writes, then scatter them across the table, so neither
            // loop has a data-dependent trip count.
            self.spread.clear();
            self.spread.resize(table_size + 8, 0);
            let mut pos = 0;
            for (symbol, &prob) in self.symbol_probabilities.iter().enumerate() {
                let n = prob as usize;
                let sv = [symbol as u8; 8];
                self.spread[pos..pos + 8].copy_from_slice(&sv);
                let mut i = 8;
                while i < n {
                    self.spread[pos + i..pos + i + 8].copy_from_slice(&sv);
                    i += 8;
                }
                pos += n;
            }
            let mut position = 0;
            for s in (0..table_size).step_by(2) {
                self.decode[position].base_value = u32::from(self.spread[s]);
                self.decode[(position + step) & mask].base_value = u32::from(self.spread[s + 1]);
                position = (position + 2 * step) & mask;
            }
        } else {
            let mut position = 0;
            for (symbol, &prob) in self.symbol_probabilities.iter().enumerate() {
                for _ in 0..prob.max(0) {
                    self.decode[position].base_value = symbol as u32;
                    position = (position + step) & mask;
                    while position >= high_threshold {
                        position = (position + step) & mask;
                    }
                }
            }
        }

        for cell in &mut self.decode {
            let symbol = cell.base_value as usize;
            let next_state = self.symbol_counter[symbol];
            self.symbol_counter[symbol] += 1;
            let nb_bits = table_log - (u32::BITS - 1 - next_state.leading_zeros());
            cell.num_bits = nb_bits as u8;
            cell.next_state = ((next_state << nb_bits) - table_size as u32) as u16;
            if let Some((base, bits)) = codes {
                cell.extra_bits = bits[symbol];
                cell.base_value = base[symbol];
            }
        }
        Ok(())
    }

    fn read_probabilities(&mut self, source: &[u8], max_log: u8) -> Result<usize, String> {
        self.symbol_probabilities.clear();

        let mut br = BitReader::new(source);
        self.accuracy_log = ACC_LOG_OFFSET + (br.get_bits(4)? as u8);
        if self.accuracy_log > max_log {
            return Err(format!(
                "Accuracy log {} exceeds max {}",
                self.accuracy_log, max_log
            ));
        }
        if self.accuracy_log == 0 {
            return Err("Accuracy log is zero".to_string());
        }

        let probability_sum = 1u32 << self.accuracy_log;
        let mut probability_counter = 0u32;

        while probability_counter < probability_sum {
            let max_remaining_value = probability_sum - probability_counter + 1;
            let bits_to_read = highest_bit_set(max_remaining_value);

            let unchecked_value = br.get_bits(bits_to_read as usize)? as u32;

            let low_threshold = ((1 << bits_to_read) - 1) - max_remaining_value;
            let mask = (1 << (bits_to_read - 1)) - 1;
            let small_value = unchecked_value & mask;

            let value = if small_value < low_threshold {
                br.return_bits(1);
                small_value
            } else if unchecked_value > mask {
                unchecked_value - low_threshold
            } else {
                unchecked_value
            };

            let prob = (value as i32) - 1;
            self.symbol_probabilities.push(prob);

            if prob != 0 {
                if prob > 0 {
                    probability_counter += prob as u32;
                } else {
                    // probability -1 counts as 1
                    probability_counter += 1;
                }
            } else {
                loop {
                    let skip_amount = br.get_bits(2)? as usize;
                    self.symbol_probabilities
                        .resize(self.symbol_probabilities.len() + skip_amount, 0);
                    if skip_amount != 3 {
                        break;
                    }
                }
            }
        }

        if probability_counter != probability_sum {
            return Err(format!(
                "Probability counter {} does not match expected sum {}",
                probability_counter, probability_sum
            ));
        }
        if self.symbol_probabilities.len() > self.max_symbol as usize + 1 {
            return Err(format!(
                "Too many symbols: {}",
                self.symbol_probabilities.len()
            ));
        }

        let bytes_read = if br.bits_read() % 8 == 0 {
            br.bits_read() / 8
        } else {
            (br.bits_read() / 8) + 1
        };

        Ok(bytes_read)
    }
}

fn fse_next_position(mut p: usize, table_size: usize) -> usize {
    p += (table_size >> 1) + (table_size >> 3) + 3;
    p &= table_size - 1;
    p
}

pub(crate) fn highest_bit_set(x: u32) -> u32 {
    assert!(x > 0);
    u32::BITS - x.leading_zeros()
}

struct FSEDecoder<'table> {
    state: FSEEntry,
    table: &'table FSETable,
}

impl<'t> FSEDecoder<'t> {
    fn new(table: &'t FSETable) -> FSEDecoder<'t> {
        FSEDecoder {
            state: table.decode.first().copied().unwrap_or_default(),
            table,
        }
    }

    fn decode_symbol(&self) -> u8 {
        self.state.base_value as u8
    }

    fn init_state(&mut self, bits: &mut BitReaderReversed<'_>) -> Result<(), String> {
        if self.table.accuracy_log == 0 {
            return Err("FSE table is uninitialized".to_string());
        }
        let new_state = bits.get_bits(self.table.accuracy_log);
        self.state = self.table.decode[new_state as usize];
        Ok(())
    }

    fn update_state(&mut self, bits: &mut BitReaderReversed<'_>) {
        let num_bits = self.state.num_bits;
        let add = bits.get_bits(num_bits);
        let new_state = usize::from(self.state.next_state) + add as usize;
        self.state = self.table.decode[new_state];
    }
}

// ============================================================
// Huffman Table and Decoder
// ============================================================

/// Single-symbol table cell (libzstd HUF_DEltX1).
#[derive(Copy, Clone, Debug, Default)]
struct HuffmanEntry {
    symbol: u8,
    num_bits: u8,
}

/// Double-symbol table cell (libzstd HUF_DEltX2): `sequence` holds one or
/// two symbols little-endian, `length` how many.
#[derive(Copy, Clone, Debug, Default)]
struct HufEntryX2 {
    sequence: u16,
    nb_bits: u8,
    length: u8,
}

/// Table log of every double-symbol table (libzstd HUF_DECODER_FAST_TABLELOG).
const HUF_X2_TABLE_LOG: u32 = 11;

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
    /// Single-symbol table, `1 << max_num_bits` cells, when `!is_x2`.
    decode: Vec<HuffmanEntry>,
    /// Double-symbol table, `1 << HUF_X2_TABLE_LOG` cells, when `is_x2`.
    decode_x2: Vec<HufEntryX2>,
    is_x2: bool,
    /// Weight per symbol, including the implied last one after a build.
    weights: Vec<u8>,
    /// Table log of the Huffman code; 0 while no table is built.
    max_num_bits: u8,
    /// Number of symbols of each weight (libzstd rankStats).
    rank_stats: [u32; MAX_MAX_NUM_BITS as usize + 2],
    bits: Vec<u8>,
    bit_ranks: Vec<u32>,
    rank_indexes: Vec<usize>,
    /// Symbols ordered by weight (libzstd sortedSymbol).
    sorted: Vec<u8>,
    fse_table: FSETable,
}

impl HuffmanTable {
    fn new() -> HuffmanTable {
        HuffmanTable {
            decode: Vec::new(),
            decode_x2: Vec::new(),
            is_x2: false,
            weights: Vec::with_capacity(256),
            max_num_bits: 0,
            rank_stats: [0; MAX_MAX_NUM_BITS as usize + 2],
            bits: Vec::with_capacity(256),
            bit_ranks: Vec::with_capacity(11),
            rank_indexes: Vec::with_capacity(11),
            sorted: Vec::with_capacity(256),
            fse_table: FSETable::new(255),
        }
    }

    fn reset(&mut self) {
        self.decode.clear();
        self.decode_x2.clear();
        self.is_x2 = false;
        self.weights.clear();
        self.max_num_bits = 0;
        self.bits.clear();
        self.bit_ranks.clear();
        self.rank_indexes.clear();
        self.sorted.clear();
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
    ) -> Result<u32, String> {
        self.max_num_bits = 0;
        let bytes_used = self.read_weights(source)?;
        self.weight_stats()?;
        self.is_x2 = four_streams && huf_select_x2(dst_size, source.len());
        if self.is_x2 {
            self.fill_x2();
        } else {
            self.fill_x1()?;
        }
        Ok(bytes_used)
    }

    fn read_weights(&mut self, source: &[u8]) -> Result<u32, String> {
        if source.is_empty() {
            return Err("Huffman source is empty".to_string());
        }
        let header = source[0];
        let mut bits_read = 8;

        match header {
            0..=127 => {
                let fse_stream = &source[1..];
                if (header as usize) > fse_stream.len() {
                    return Err(format!(
                        "Not enough bytes for weights: have {}, need {}",
                        fse_stream.len(),
                        header
                    ));
                }
                let bytes_used_by_fse_header = self.fse_table.build_decoder(fse_stream, 6, None)?;

                if bytes_used_by_fse_header > header as usize {
                    return Err(format!(
                        "FSE table used {} bytes but only {} available",
                        bytes_used_by_fse_header, header
                    ));
                }

                let mut dec1 = FSEDecoder::new(&self.fse_table);
                let mut dec2 = FSEDecoder::new(&self.fse_table);

                let compressed_start = bytes_used_by_fse_header;
                let compressed_length = header as usize - bytes_used_by_fse_header;

                let compressed_weights = &fse_stream[compressed_start..];
                if compressed_weights.len() < compressed_length {
                    return Err(format!(
                        "Not enough bytes to decompress weights: have {}, need {}",
                        compressed_weights.len(),
                        compressed_length
                    ));
                }
                let compressed_weights = &compressed_weights[..compressed_length];
                let mut br = BitReaderReversed::new(compressed_weights);

                bits_read += (bytes_used_by_fse_header + compressed_length) * 8;

                let mut skipped_bits = 0;
                loop {
                    let val = br.get_bits(1);
                    skipped_bits += 1;
                    if val == 1 || skipped_bits > 8 {
                        break;
                    }
                }
                if skipped_bits > 8 {
                    return Err(format!("Extra padding: {} bits skipped", skipped_bits));
                }

                dec1.init_state(&mut br)?;
                dec2.init_state(&mut br)?;

                self.weights.clear();

                loop {
                    let w = dec1.decode_symbol();
                    self.weights.push(w);
                    dec1.update_state(&mut br);

                    if br.bits_remaining() <= -1 {
                        self.weights.push(dec2.decode_symbol());
                        break;
                    }

                    let w = dec2.decode_symbol();
                    self.weights.push(w);
                    dec2.update_state(&mut br);

                    if br.bits_remaining() <= -1 {
                        self.weights.push(dec1.decode_symbol());
                        break;
                    }
                    if self.weights.len() > 255 {
                        return Err(format!("Too many weights: {}", self.weights.len()));
                    }
                }
                if self.weights.len() > 255 {
                    return Err(format!("Too many weights: {}", self.weights.len()));
                }
            }
            _ => {
                let weights_raw = &source[1..];
                let num_weights = header - 127;
                self.weights.resize(num_weights as usize, 0);

                let bytes_needed = if num_weights % 2 == 0 {
                    num_weights as usize / 2
                } else {
                    (num_weights as usize / 2) + 1
                };

                if weights_raw.len() < bytes_needed {
                    return Err(format!(
                        "Not enough bytes in source: have {}, need {}",
                        weights_raw.len(),
                        bytes_needed
                    ));
                }

                for idx in 0..num_weights {
                    if idx % 2 == 0 {
                        self.weights[idx as usize] = weights_raw[idx as usize / 2] >> 4;
                    } else {
                        self.weights[idx as usize] = weights_raw[idx as usize / 2] & 0xF;
                    }
                    bits_read += 4;
                }
            }
        }

        let bytes_read = if bits_read % 8 == 0 {
            bits_read / 8
        } else {
            (bits_read / 8) + 1
        };
        Ok(bytes_read as u32)
    }

    /// Validate the weights, derive the table log and the implied last
    /// weight, and count symbols per weight (the checks of HUF_readStats).
    fn weight_stats(&mut self) -> Result<(), String> {
        let mut weight_sum: u32 = 0;
        for w in &self.weights {
            if *w > MAX_MAX_NUM_BITS {
                return Err(format!("Weight {} exceeds max {}", w, MAX_MAX_NUM_BITS));
            }
            weight_sum += if *w > 0 { 1_u32 << (*w - 1) } else { 0 };
        }

        if weight_sum == 0 {
            return Err("Missing weights".to_string());
        }

        let max_bits = highest_bit_set(weight_sum) as u8;
        if max_bits > MAX_MAX_NUM_BITS {
            return Err(format!("Max bits {} too high", max_bits));
        }
        let left_over = (1u32 << max_bits) - weight_sum;

        if !left_over.is_power_of_two() {
            return Err(format!("Leftover {} is not a power of 2", left_over));
        }

        let last_weight = highest_bit_set(left_over) as u8;
        self.weights.push(last_weight);

        self.rank_stats = [0; MAX_MAX_NUM_BITS as usize + 2];
        for &w in &self.weights {
            self.rank_stats[usize::from(w)] += 1;
        }
        // A full binary tree has an even number of leaves at its deepest
        // level, and at least two.
        if self.rank_stats[1] < 2 || self.rank_stats[1] & 1 != 0 {
            return Err(format!(
                "Huffman tree has {} symbols of weight 1",
                self.rank_stats[1]
            ));
        }

        self.max_num_bits = max_bits;
        Ok(())
    }

    /// Fill the single-symbol table: each symbol of `n` bits owns
    /// `1 << (max_bits - n)` consecutive cells, ordered by code length.
    fn fill_x1(&mut self) -> Result<(), String> {
        let max_bits = self.max_num_bits;
        self.bits.clear();
        self.bits.resize(self.weights.len(), 0);
        for symbol in 0..self.weights.len() {
            let bits = if self.weights[symbol] > 0 {
                max_bits + 1 - self.weights[symbol]
            } else {
                0
            };
            self.bits[symbol] = bits;
        }

        self.bit_ranks.clear();
        self.bit_ranks.resize((max_bits + 1) as usize, 0);
        for num_bits in &self.bits {
            self.bit_ranks[(*num_bits) as usize] += 1;
        }

        self.decode.clear();
        self.decode.resize(1 << max_bits, HuffmanEntry::default());

        self.rank_indexes.clear();
        self.rank_indexes.resize((max_bits + 1) as usize, 0);

        self.rank_indexes[max_bits as usize] = 0;
        for bits in (1..self.rank_indexes.len() as u8).rev() {
            self.rank_indexes[bits as usize - 1] = self.rank_indexes[bits as usize]
                + self.bit_ranks[bits as usize] as usize * (1 << (max_bits - bits));
        }

        if self.rank_indexes[0] != self.decode.len() {
            return Err(format!(
                "Huffman code lengths cover {} of {} cells",
                self.rank_indexes[0],
                self.decode.len()
            ));
        }

        for symbol in 0..self.bits.len() {
            let bits_for_symbol = self.bits[symbol];
            if bits_for_symbol != 0 {
                let base_idx = self.rank_indexes[bits_for_symbol as usize];
                let len = 1 << (max_bits - bits_for_symbol);
                self.rank_indexes[bits_for_symbol as usize] += len;
                for idx in 0..len {
                    self.decode[base_idx + idx].symbol = symbol as u8;
                    self.decode[base_idx + idx].num_bits = bits_for_symbol;
                }
            }
        }

        Ok(())
    }

    /// Fill the double-symbol table (HUF_readDTableX2_wksp after
    /// HUF_readStats): sort symbols by weight, compute where each weight's
    /// run starts for every number of already-consumed bits, then tile the
    /// table so that a cell holds two symbols whenever both fit in
    /// `HUF_X2_TABLE_LOG` bits.
    fn fill_x2(&mut self) {
        let table_log = u32::from(self.max_num_bits);
        let target_log = HUF_X2_TABLE_LOG;
        let nb_bits_baseline = table_log + 1;
        let nb_symbols = self.weights.len();

        // Highest weight in use; weight 1 is always present.
        let mut max_w = table_log as usize;
        while self.rank_stats[max_w] == 0 {
            max_w -= 1;
        }

        // rank_start[w]: first index of weight w in the sorted list.
        let mut rank_start = [0usize; MAX_MAX_NUM_BITS as usize + 3];
        let mut next = 0usize;
        for w in 1..=max_w {
            rank_start[w] = next;
            next += self.rank_stats[w] as usize;
        }
        rank_start[max_w + 1] = next;

        self.sorted.clear();
        self.sorted.resize(nb_symbols, 0);
        {
            let mut fill = rank_start;
            let mut zero_at = next;
            for s in 0..nb_symbols {
                let w = usize::from(self.weights[s]);
                if w == 0 {
                    self.sorted[zero_at] = s as u8;
                    zero_at += 1;
                } else {
                    self.sorted[fill[w]] = s as u8;
                    fill[w] += 1;
                }
            }
        }

        // rank_val[consumed][w]: first cell of weight w once `consumed` bits
        // of the lookup have been used by a first symbol.
        let rescale = target_log - table_log; // shift of (w - 1 + rescale)
        let mut rank_val = [[0u32; MAX_MAX_NUM_BITS as usize + 2]; HUF_X2_TABLE_LOG as usize + 1];
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

        self.decode_x2.clear();
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
/// `1 << (target_log - nb_bits)` cells (HUF_fillDTableX2ForWeight).
fn huf_fill_x2_for_weight(
    dt: &mut [HufEntryX2],
    symbols: &[u8],
    nb_bits: u32,
    target_log: u32,
    base_seq: u8,
    level: u8,
) {
    let length = 1usize << (target_log - nb_bits);
    let mut off = 0;
    for &symbol in symbols {
        dt[off..off + length].fill(huf_build_x2(symbol, nb_bits, base_seq, level));
        off += length;
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
        let skip = rank_val[min_weight] as usize;
        dt[..skip].fill(huf_build_x2(base_seq, consumed_bits, 0, 1));
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

    #[inline(always)]
    fn reload_internal(&mut self) -> HufStreamStatus {
        self.ptr -= (self.bits_consumed >> 3) as usize;
        self.bits_consumed &= 7;
        self.container = read_le64(self.src, self.ptr);
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
        // guarantees at least 57 bits, and a symbol takes at most 11.
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

/// Single-stream literals (HUF_decompress1X1_usingDTable_internal_body).
#[inline(never)]
fn huf_decompress_1x1(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), String> {
    let dt_log = u32::from(table.max_num_bits);
    let dt = &table.decode[..];
    let mut br = BitDStream::new(src)?;
    huf_decode_stream_x1(out, &mut br, dt, dt_log);
    if !br.is_finished() {
        return Err("Huffman stream not fully consumed".to_string());
    }
    Ok(())
}

/// Four interleaved literal streams
/// (HUF_decompress4X1_usingDTable_internal_body).
///
/// The output is split into four segments of `(len + 3) / 4` bytes (the
/// last one holds the remainder); stream `i` produces segment `i`. The main
/// loop advances all four streams in lockstep, 4 symbols each per reload,
/// so the four dependency chains overlap in the CPU.
#[inline(never)]
fn huf_decompress_4x1(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), String> {
    if src.len() < 10 {
        return Err(format!(
            "Huffman 4-stream input too short: {} bytes",
            src.len()
        ));
    }
    let dst_size = out.len();
    if dst_size < MIN_LITERALS_FOR_4_STREAMS {
        return Err(format!(
            "Huffman 4-stream output too small: {} bytes",
            dst_size
        ));
    }
    let len1 = usize::from(u16::from_le_bytes([src[0], src[1]]));
    let len2 = usize::from(u16::from_le_bytes([src[2], src[3]]));
    let len3 = usize::from(u16::from_le_bytes([src[4], src[5]]));
    let body = &src[6..];
    if len1 + len2 + len3 > body.len() {
        return Err("Huffman jump table exceeds input".to_string());
    }
    let (s1, rest) = body.split_at(len1);
    let (s2, rest) = rest.split_at(len2);
    let (s3, s4) = rest.split_at(len3);

    let segment = dst_size.div_ceil(4);
    if 3 * segment > dst_size {
        return Err("Huffman 4-stream segments exceed output".to_string());
    }
    let (o1, rest) = out.split_at_mut(segment);
    let (o2, rest) = rest.split_at_mut(segment);
    let (o3, o4) = rest.split_at_mut(segment);

    let mut b1 = BitDStream::new(s1)?;
    let mut b2 = BitDStream::new(s2)?;
    let mut b3 = BitDStream::new(s3)?;
    let mut b4 = BitDStream::new(s4)?;

    let dt_log = u32::from(table.max_num_bits);
    let dt = &table.decode[..];

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
        // Up to 10 symbols per reload: 5 cells of at most 11 bits each.
        while br.reload() == HufStreamStatus::Unfinished && op + 9 < end {
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
            op = huf_decode_symbol_x2(out, op, br, dt, dt_log);
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

/// Single-stream literals with the double-symbol table
/// (HUF_decompress1X2_usingDTable_internal_body).
#[inline(never)]
fn huf_decompress_1x2(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), String> {
    let dt = &table.decode_x2[..];
    let mut br = BitDStream::new(src)?;
    huf_decode_stream_x2(out, 0, out.len(), &mut br, dt, HUF_X2_TABLE_LOG);
    if !br.is_finished() {
        return Err("Huffman stream not fully consumed".to_string());
    }
    Ok(())
}

/// Four interleaved literal streams with the double-symbol table
/// (HUF_decompress4X2_usingDTable_internal_body). Streams write at their
/// own pace, so the first three are checked against their segment ends
/// after the shared loop, whose trip count is bounded by the last stream.
#[inline(never)]
fn huf_decompress_4x2(out: &mut [u8], src: &[u8], table: &HuffmanTable) -> Result<(), String> {
    if src.len() < 10 {
        return Err(format!(
            "Huffman 4-stream input too short: {} bytes",
            src.len()
        ));
    }
    let oend = out.len();
    if oend < MIN_LITERALS_FOR_4_STREAMS {
        return Err(format!("Huffman 4-stream output too small: {} bytes", oend));
    }
    let len1 = usize::from(u16::from_le_bytes([src[0], src[1]]));
    let len2 = usize::from(u16::from_le_bytes([src[2], src[3]]));
    let len3 = usize::from(u16::from_le_bytes([src[4], src[5]]));
    let body = &src[6..];
    if len1 + len2 + len3 > body.len() {
        return Err("Huffman jump table exceeds input".to_string());
    }
    let (s1, rest) = body.split_at(len1);
    let (s2, rest) = rest.split_at(len2);
    let (s3, s4) = rest.split_at(len3);

    let segment = oend.div_ceil(4);
    let op_start2 = segment;
    let op_start3 = 2 * segment;
    let op_start4 = 3 * segment;
    if op_start4 > oend {
        return Err("Huffman 4-stream segments exceed output".to_string());
    }

    let mut b1 = BitDStream::new(s1)?;
    let mut b2 = BitDStream::new(s2)?;
    let mut b3 = BitDStream::new(s3)?;
    let mut b4 = BitDStream::new(s4)?;

    let dt = &table.decode_x2[..];
    let dt_log = HUF_X2_TABLE_LOG;
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

enum LiteralsSectionType {
    Raw,
    RLE,
    Compressed,
    Treeless,
}

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

            let window_log = 10 + u64::from(exp);
            let window_base = 1u64 << window_log;
            let window_add = (window_base / 8) * u64::from(mantissa);

            let window_size = window_base + window_add;

            if window_size < MIN_WINDOW_SIZE {
                Err(format!("Window size {} too small", window_size))
            } else if window_size >= MAX_WINDOW_SIZE {
                Err(format!("Window size {} too big", window_size))
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

    let block_size = u32::from(buf[0] >> 3) | (u32::from(buf[1]) << 5) | (u32::from(buf[2]) << 13);

    if block_size > MAX_BLOCK_SIZE {
        return Err(format!(
            "Block size {} exceeds max {}",
            block_size, MAX_BLOCK_SIZE
        ));
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
    if regenerated_size > MAX_BLOCK_SIZE as usize {
        return Err(format!(
            "Literals size {} exceeds block size limit",
            regenerated_size
        ));
    }

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

    match (num_streams == 4, scratch.table.is_x2) {
        (true, true) => huf_decompress_4x2(out, source, &scratch.table)?,
        (true, false) => huf_decompress_4x1(out, source, &scratch.table)?,
        (false, true) => huf_decompress_1x2(out, source, &scratch.table)?,
        (false, false) => huf_decompress_1x1(out, source, &scratch.table)?,
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

/// Copies with offsets at or above this never overlap a 16-byte chunk
/// (libzstd WILDCOPY_VECLEN).
const WILDCOPY_VECLEN: usize = 16;

/// Matches longer than this are copied with `copy_within` (memmove)
/// instead of fixed 16-byte chunks.
const LONG_COPY_THRESHOLD: usize = 32;

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
    bytes_read += build_sequence_table(
        modes.ll_mode(),
        &source[bytes_read..],
        &mut scratch.literal_lengths,
        LL_MAX_LOG,
        MAX_LITERAL_LENGTH_CODE,
        LL_DEFAULT_ACC_LOG,
        &LITERALS_LENGTH_DEFAULT_DISTRIBUTION,
        &LL_BASE,
        &LL_BITS,
        "LL",
    )?;
    bytes_read += build_sequence_table(
        modes.of_mode(),
        &source[bytes_read..],
        &mut scratch.offsets,
        OF_MAX_LOG,
        MAX_OFFSET_CODE,
        OF_DEFAULT_ACC_LOG,
        &OFFSET_DEFAULT_DISTRIBUTION,
        &OF_BASE,
        &OF_BITS,
        "OF",
    )?;
    bytes_read += build_sequence_table(
        modes.ml_mode(),
        &source[bytes_read..],
        &mut scratch.match_lengths,
        ML_MAX_LOG,
        MAX_MATCH_LENGTH_CODE,
        ML_DEFAULT_ACC_LOG,
        &MATCH_LENGTH_DEFAULT_DISTRIBUTION,
        &ML_BASE,
        &ML_BITS,
        "ML",
    )?;
    Ok(bytes_read)
}

#[allow(clippy::too_many_arguments)]
fn build_sequence_table(
    mode: ModeType,
    source: &[u8],
    table: &mut FSETable,
    max_log: u8,
    max_code: u8,
    default_log: u8,
    default_distribution: &[i32],
    base: &[u32],
    bits: &[u8],
    name: &str,
) -> Result<usize, String> {
    match mode {
        ModeType::FSECompressed => table.build_decoder(source, max_log, Some((base, bits))),
        ModeType::RLE => {
            let Some(&code) = source.first() else {
                return Err(format!("Missing byte for RLE {} table", name));
            };
            if code > max_code {
                return Err(format!("RLE {} code {} exceeds max", name, code));
            }
            table.build_rle(code, base, bits);
            Ok(1)
        }
        ModeType::Predefined => {
            if !table.predefined {
                table.build_from_probabilities(
                    default_log,
                    default_distribution,
                    Some((base, bits)),
                )?;
                table.predefined = true;
            }
            Ok(0)
        }
        ModeType::Repeat => {
            if table.decode.is_empty() {
                return Err(format!("Repeat mode without a previous {} table", name));
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

/// Decode every sequence of the block and execute it straight into `out`;
/// matches may reach back no further than `prefix_start`.
///
/// `literals` holds the block's decoded literals followed by exactly
/// `WILDCOPY_OVERLENGTH` bytes of slack. `out` is grown by the block limit
/// plus slack up front so that all copies use fixed-size chunks and may
/// overshoot; it is truncated to the real length on return.
#[inline(never)]
fn decode_and_execute_sequences(
    num_sequences: u32,
    bit_stream: &[u8],
    fse: &FSEScratch,
    literals: &[u8],
    offset_hist: &mut [u32; 3],
    prefix_start: usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let base = out.len();
    out.resize(base + MAX_BLOCK_SIZE as usize + WILDCOPY_OVERLENGTH, 0);
    let result = run_sequences(
        num_sequences,
        bit_stream,
        fse,
        literals,
        offset_hist,
        &mut out[prefix_start..],
        base - prefix_start,
    );
    match result {
        Ok(end) => {
            out.truncate(prefix_start + end);
            Ok(())
        }
        Err(e) => {
            out.truncate(base);
            Err(e)
        }
    }
}

/// `buf` starts at the frame's first byte and ends `WILDCOPY_OVERLENGTH`
/// bytes past the block's output limit; `op` is where this block starts.
fn run_sequences(
    num_sequences: u32,
    bit_stream: &[u8],
    fse: &FSEScratch,
    literals: &[u8],
    offset_hist: &mut [u32; 3],
    buf: &mut [u8],
    mut op: usize,
) -> Result<usize, String> {
    let ll_dt = &fse.literal_lengths.decode[..];
    let of_dt = &fse.offsets.decode[..];
    let ml_dt = &fse.match_lengths.decode[..];
    if ll_dt.is_empty() || of_dt.is_empty() || ml_dt.is_empty() {
        return Err("FSE table is uninitialized".to_string());
    }
    let literals_len = literals.len() - WILDCOPY_OVERLENGTH;
    let oend = buf.len() - WILDCOPY_OVERLENGTH;

    let mut br = BitDStream::new(bit_stream)?;
    // ZSTD_initFseState: LL, OF, ML order, each followed by a reload.
    let mut ll_state = br.read_bits(u32::from(fse.literal_lengths.accuracy_log));
    br.reload();
    let mut of_state = br.read_bits(u32::from(fse.offsets.accuracy_log));
    br.reload();
    let mut ml_state = br.read_bits(u32::from(fse.match_lengths.accuracy_log));
    br.reload();

    let mut hist = [
        offset_hist[0] as usize,
        offset_hist[1] as usize,
        offset_hist[2] as usize,
    ];
    let mut lit_pos = 0usize;

    for remaining in (1..=num_sequences).rev() {
        let ll_e = ll_dt[ll_state];
        let ml_e = ml_dt[ml_state];
        let of_e = of_dt[of_state];

        let mut ll = ll_e.base_value as usize;
        let mut ml = ml_e.base_value as usize;
        let ll_bits = u32::from(ll_e.extra_bits);
        let ml_bits = u32::from(ml_e.extra_bits);
        let of_bits = u32::from(of_e.extra_bits);
        let total_bits = ll_bits + ml_bits + of_bits;

        // Offset and repeat-offset history (ZSTD_decodeSequence).
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
                let o = of_e.base_value as usize + ll0 + br.read_bits_fast(1);
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

        if remaining > 1 {
            ll_state = usize::from(ll_e.next_state) + br.read_bits(u32::from(ll_e.num_bits));
            ml_state = usize::from(ml_e.next_state) + br.read_bits(u32::from(ml_e.num_bits));
            of_state = usize::from(of_e.next_state) + br.read_bits(u32::from(of_e.num_bits));
            br.reload();
        }

        op = exec_sequence(buf, op, literals, &mut lit_pos, ll, ml, offset)
            .map_err(seq_error_message)?;
    }

    if !br.is_finished() {
        return Err("Sequence bitstream not fully consumed".to_string());
    }

    // Last literals segment.
    let rest = literals_len - lit_pos;
    if op + rest > oend {
        return Err(seq_error_message(SeqError::BlockTooLarge));
    }
    buf[op..op + rest].copy_from_slice(&literals[lit_pos..literals_len]);
    op += rest;

    *offset_hist = [hist[0] as u32, hist[1] as u32, hist[2] as u32];
    Ok(op)
}

/// Copy `ll` literals then `ml` match bytes from `offset` back
/// (ZSTD_execSequence). Returns the new output position. Both slices carry
/// `WILDCOPY_OVERLENGTH` bytes of slack past their logical end.
#[inline(always)]
fn exec_sequence(
    buf: &mut [u8],
    op: usize,
    literals: &[u8],
    lit_pos: &mut usize,
    ll: usize,
    ml: usize,
    offset: usize,
) -> Result<usize, SeqError> {
    let lit_start = *lit_pos;
    let o_lit_end = op + ll;
    let o_match_end = o_lit_end + ml;
    if lit_start + ll + WILDCOPY_OVERLENGTH > literals.len() {
        return Err(SeqError::NotEnoughLiterals);
    }
    if o_match_end + WILDCOPY_OVERLENGTH > buf.len() {
        return Err(SeqError::BlockTooLarge);
    }

    // Literals: nearly always at most 16 bytes.
    copy16_from(buf, op, literals, lit_start);
    if ll > 16 {
        wildcopy_from(buf, op + 16, literals, lit_start + 16, ll - 16);
    }
    *lit_pos = lit_start + ll;

    if offset > o_lit_end {
        return Err(SeqError::OffsetTooFar);
    }
    let mut src = o_lit_end - offset;
    let mut dst = o_lit_end;
    if offset >= WILDCOPY_VECLEN {
        if ml <= LONG_COPY_THRESHOLD {
            wildcopy_within(buf, dst, src, ml);
        } else if offset >= ml {
            buf.copy_within(src..src + ml, dst);
        } else {
            // Periodic pattern: copy the whole prefix decoded so far, whose
            // length doubles each round, so long matches take O(log n) memcpys.
            let mut done = 0;
            while done < ml {
                let chunk = (offset + done).min(ml - done);
                buf.copy_within(src..src + chunk, dst + done);
                done += chunk;
            }
        }
    } else {
        // Copy 8 bytes and spread the offset to at least 8, then continue
        // with 8-byte chunks.
        overlap_copy8(buf, &mut dst, &mut src, offset);
        if ml > 8 {
            wildcopy_overlap8(buf, dst, src, ml - 8);
        }
    }
    Ok(o_match_end)
}

#[inline(always)]
fn copy16_from(buf: &mut [u8], dst: usize, src: &[u8], sp: usize) {
    let chunk: [u8; 16] = src[sp..sp + 16].try_into().unwrap();
    buf[dst..dst + 16].copy_from_slice(&chunk);
}

#[inline(always)]
fn copy16_within(buf: &mut [u8], dst: usize, src: usize) {
    let chunk: [u8; 16] = buf[src..src + 16].try_into().unwrap();
    buf[dst..dst + 16].copy_from_slice(&chunk);
}

#[inline(always)]
fn copy8_within(buf: &mut [u8], dst: usize, src: usize) {
    let chunk: [u8; 8] = buf[src..src + 8].try_into().unwrap();
    buf[dst..dst + 8].copy_from_slice(&chunk);
}

/// ZSTD_wildcopy(no_overlap) from another buffer: 16-byte chunks that may
/// overshoot `len` by up to 31 bytes on both sides.
#[inline(always)]
fn wildcopy_from(buf: &mut [u8], mut dst: usize, src: &[u8], mut sp: usize, len: usize) {
    copy16_from(buf, dst, src, sp);
    if len <= 16 {
        return;
    }
    let end = dst + len;
    dst += 16;
    sp += 16;
    loop {
        copy16_from(buf, dst, src, sp);
        copy16_from(buf, dst + 16, src, sp + 16);
        dst += 32;
        sp += 32;
        if dst >= end {
            break;
        }
    }
}

/// ZSTD_wildcopy(no_overlap) within `buf`; `dst - src >= 16`.
#[inline(always)]
fn wildcopy_within(buf: &mut [u8], mut dst: usize, mut src: usize, len: usize) {
    copy16_within(buf, dst, src);
    if len <= 16 {
        return;
    }
    let end = dst + len;
    dst += 16;
    src += 16;
    loop {
        copy16_within(buf, dst, src);
        copy16_within(buf, dst + 16, src + 16);
        dst += 32;
        src += 32;
        if dst >= end {
            break;
        }
    }
}

/// ZSTD_wildcopy(overlap_src_before_dst) with `8 <= dst - src < 16`:
/// 8-byte chunks, overshooting by up to 7 bytes.
#[inline(always)]
fn wildcopy_overlap8(buf: &mut [u8], mut dst: usize, mut src: usize, len: usize) {
    let end = dst + len;
    loop {
        copy8_within(buf, dst, src);
        dst += 8;
        src += 8;
        if dst >= end {
            break;
        }
    }
}

/// ZSTD_overlapCopy8: copy 8 bytes from `src` to `dst` (`src <= dst`) and
/// advance both so that afterwards `dst - src >= 8`.
#[inline(always)]
fn overlap_copy8(buf: &mut [u8], dst: &mut usize, src: &mut usize, offset: usize) {
    if offset < 8 {
        const DEC32: [usize; 8] = [0, 1, 2, 1, 4, 4, 4, 4];
        const DEC64: [usize; 8] = [8, 8, 8, 7, 8, 9, 10, 11];
        let (d, s) = (*dst, *src);
        buf[d] = buf[s];
        buf[d + 1] = buf[s + 1];
        buf[d + 2] = buf[s + 2];
        buf[d + 3] = buf[s + 3];
        let s2 = s + DEC32[offset];
        let chunk: [u8; 4] = buf[s2..s2 + 4].try_into().unwrap();
        buf[d + 4..d + 8].copy_from_slice(&chunk);
        *src = s2 + 8 - DEC64[offset];
    } else {
        copy8_within(buf, *dst, *src);
        *src += 8;
    }
    *dst += 8;
}

// ============================================================
// Block decoder
// ============================================================

/// Decode every block of one frame from `data[*pos..]` straight into
/// `output`, then skip the checksum. Matches may only reach back to the
/// frame's own start (ZSTD_decompressFrame).
#[inline(never)]
fn decode_frame(
    header: &FrameHeader,
    data: &[u8],
    pos: &mut usize,
    scratch: &mut DecoderScratch,
    output: &mut Vec<u8>,
) -> Result<(), String> {
    let frame_base = output.len();

    if let Some(fcs) = header.frame_content_size() {
        // Room for the whole frame plus one block of copy slack, so that
        // no block has to grow the buffer (and move everything decoded).
        let want = usize::try_from(fcs)
            .ok()
            .and_then(|n| n.checked_add(MAX_BLOCK_SIZE as usize + WILDCOPY_OVERLENGTH))
            .ok_or_else(|| format!("Frame content size {} too large", fcs))?;
        output
            .try_reserve(want)
            .map_err(|e| format!("Cannot reserve {} bytes of output: {}", want, e))?;
    }

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
            BlockType::Compressed => decompress_block(content, scratch, frame_base, output)?,
            BlockType::Reserved => return Err("Reserved block type encountered".to_string()),
        }

        if block.last_block {
            break;
        }
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

fn decompress_block(
    raw: &[u8],
    workspace: &mut DecoderScratch,
    frame_base: usize,
    output: &mut Vec<u8>,
) -> Result<(), String> {
    let content_size = raw.len() as u32;
    let mut section = LiteralsSection::new();
    let bytes_in_literals_header = section.parse_from_header(raw)?;
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

    let raw_literals = &raw[..upper_limit_for_literals];

    workspace.literals_buffer.clear();
    let bytes_used_in_literals_section = decode_literals(
        &section,
        &mut workspace.huf,
        raw_literals,
        &mut workspace.literals_buffer,
    )?;
    assert!(
        section.regenerated_size == workspace.literals_buffer.len() as u32,
        "Wrong number of literals: {}, Should have been: {}",
        workspace.literals_buffer.len(),
        section.regenerated_size
    );
    assert!(bytes_used_in_literals_section == upper_limit_for_literals as u32);
    let literals_len = workspace.literals_buffer.len();
    workspace
        .literals_buffer
        .resize(literals_len + WILDCOPY_OVERLENGTH, 0);

    let raw = &raw[upper_limit_for_literals..];

    let mut seq_section = SequencesHeader::new();
    let bytes_in_sequence_header = seq_section.parse_from_header(raw)?;
    let raw = &raw[bytes_in_sequence_header as usize..];

    assert!(
        u32::from(bytes_in_literals_header)
            + bytes_used_in_literals_section
            + u32::from(bytes_in_sequence_header)
            + raw.len() as u32
            == content_size
    );

    if seq_section.num_sequences != 0 {
        let table_bytes = build_sequence_tables(&seq_section, raw, &mut workspace.fse)?;
        decode_and_execute_sequences(
            seq_section.num_sequences,
            &raw[table_bytes..],
            &workspace.fse,
            &workspace.literals_buffer,
            &mut workspace.offset_hist,
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

#[cfg(test)]
mod tests {
    use super::*;

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
