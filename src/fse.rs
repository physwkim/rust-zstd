//! FSE (Finite State Entropy) encoder.
//! Ported from zstd C source: lib/common/fse.h, lib/compress/fse_compress.c.

use super::bitstream::BitCStream;
use super::huf;
use crate::compress::entropy::{FseHeld, FseNext, FseTables, Held, Next, Repeat, TableRef};
use crate::compress::seqstore::Seq;
use crate::compress::{CParams, Strategy};
use crate::constants::*;

/// `MaxSeq`: the largest sequence code of any kind.
const MAX_SEQ: usize = MAX_ML;
/// `DefaultMaxOff`: the largest offset code the default table covers.
const DEFAULT_MAX_OFF: usize = 28;
/// `LONGNBSEQ`.
const LONGNBSEQ: usize = 0x7F00;
/// `FSE_NCOUNTBOUND`.
const FSE_NCOUNTBOUND: usize = 512;

/// Per-symbol compression transform (matches C FSE_symbolCompressionTransform).
#[derive(Clone, Copy, Debug, Default)]
pub struct SymbolTT {
    pub delta_find_state: i32,
    pub delta_nb_bits: u32,
}

/// The largest table log an [`FseCTable`] holds: `LLFSELog` and
/// `MLFSELog` (offsets use 8, Huffman weight descriptions at most 6).
const FSE_CTABLE_MAX_LOG: u32 = 9;

/// Compiled FSE compression table (`FSE_CTable`), with room for any table
/// built here so that a table is rebuilt in place, never allocated.
#[derive(Clone, Debug)]
pub struct FseCTable {
    pub table_log: u32,
    /// `stateTable`: the first `1 << table_log` entries are the table's.
    pub state_table: [u16; 1 << FSE_CTABLE_MAX_LOG],
    /// `symbolTT` for every sequence code; past `max_symbol` the transform
    /// of a zero-probability symbol.
    pub symbol_tt: [SymbolTT; MAX_SEQ + 1],
    pub max_symbol: usize,
}

impl Default for FseCTable {
    fn default() -> Self {
        Self {
            table_log: 0,
            state_table: [0; 1 << FSE_CTABLE_MAX_LOG],
            symbol_tt: [SymbolTT::default(); MAX_SEQ + 1],
            max_symbol: 0,
        }
    }
}

impl FseCTable {
    /// Build an FSE compression table from normalized counts in place.
    /// Ported from FSE_buildCTable_wksp() in fse_compress.c.
    pub fn build(&mut self, norm: &[i16], max_symbol: usize, table_log: u32) {
        assert!(table_log <= FSE_CTABLE_MAX_LOG && max_symbol <= MAX_SEQ);
        let table_size = 1u32 << table_log;
        let table_mask = table_size - 1;

        // 1. Build cumulative counts and place low-probability symbols.
        // `cumul` and `tableSymbol` live in the C workspace.
        let mut cumul = [0u16; MAX_SEQ + 2];
        let mut high_threshold = table_size - 1;
        let mut table_symbol = [0u8; 1 << FSE_CTABLE_MAX_LOG];

        for s in 0..=max_symbol {
            if norm[s] == -1 {
                cumul[s + 1] = cumul[s] + 1;
                table_symbol[high_threshold as usize] = s as u8;
                high_threshold = high_threshold.wrapping_sub(1);
            } else {
                cumul[s + 1] = cumul[s] + norm[s] as u16;
            }
        }
        cumul[max_symbol + 1] = (table_size + 1) as u16;

        // 2. Spread symbols into the table using the exact C formula.
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mut pos = 0u32;
        for s in 0..=max_symbol {
            let count = if norm[s] <= 0 { 0 } else { norm[s] as u32 };
            for _ in 0..count {
                table_symbol[pos as usize] = s as u8;
                pos = (pos + step) & table_mask;
                while pos > high_threshold {
                    pos = (pos + step) & table_mask;
                }
            }
        }
        debug_assert_eq!(pos, 0);

        // 3. Build state transition table sorted by symbol order.
        for u in 0..table_size {
            let s = table_symbol[u as usize] as usize;
            let idx = cumul[s] as usize;
            self.state_table[idx] = (table_size + u) as u16;
            cumul[s] += 1;
        }

        // 4. Build per-symbol compression transforms.
        // Use the decoder's baseline/numbits calculation for compatibility.
        // For each state in the table, compute its decoder-compatible numbits,
        // then derive the CTable's delta_nb_bits and delta_find_state from that.
        let mut total = 0u32;
        for (s, tt) in self.symbol_tt.iter_mut().enumerate() {
            let prob = match norm[..=max_symbol].get(s) {
                None => 0,
                Some(-1) => 1,
                Some(&n) => n.max(0) as u32,
            };
            *tt = if prob == 0 {
                SymbolTT {
                    delta_find_state: 0,
                    delta_nb_bits: ((table_log + 1) << 16) - table_size,
                }
            } else if prob == 1 {
                SymbolTT {
                    delta_find_state: total as i32 - 1,
                    delta_nb_bits: (table_log << 16) - table_size,
                }
            } else {
                // Use the same formula as C zstd FSE_buildCTable
                let max_bits_out = table_log - highest_bit(prob - 1);
                let min_state_plus = prob << max_bits_out;
                SymbolTT {
                    delta_find_state: total as i32 - prob as i32,
                    delta_nb_bits: (max_bits_out << 16).wrapping_sub(min_state_plus),
                }
            };
            total += prob;
        }
        self.table_log = table_log;
        self.max_symbol = max_symbol;
    }

    /// `FSE_buildCTable_rle` in place: single symbol, 0 bits per encode.
    /// `init_state` returns 0 and every transition stays at state 0;
    /// table_log = 0, matching the decoder's RLE behavior.
    pub fn build_rle(&mut self, symbol: u8) {
        let s = symbol as usize;
        // Handcraft a table where everything resolves to 0 bits, state=0:
        // nb_bits = (state + delta_nb_bits) >> 16 = 0 for state 0, and
        // new_state = state_table[(0 >> 0) + 0] = 0
        self.state_table[0] = 0;
        self.symbol_tt.fill(SymbolTT::default());
        self.table_log = 0;
        self.max_symbol = s;
    }

    /// `FSE_initCState2`: the state that encodes `symbol` first.
    #[inline(always)]
    pub fn init_state(&self, symbol: u8) -> u32 {
        let stt = self.symbol_tt[symbol as usize];
        let nb_bits = ((stt.delta_nb_bits as u64 + (1 << 15)) >> 16) as u32;
        let base_val = (nb_bits << 16).wrapping_sub(stt.delta_nb_bits);
        self.next_state(base_val >> nb_bits, stt.delta_find_state)
    }

    /// `stateTable[(state >> nbBitsOut) + deltaFindState]` for a state
    /// this table produced (`init_state` or an earlier `next_state`) and
    /// a symbol's own transform.
    #[inline(always)]
    fn next_state(&self, shifted: u32, delta_find_state: i32) -> u32 {
        let idx = shifted.wrapping_add_signed(delta_find_state) as usize;
        debug_assert!(idx < 1 << self.table_log);
        // SAFETY: `build` gives symbol `s` with normalized count `p`
        // (1 for a low-probability symbol) `deltaFindState = start_s - p`
        // and a `deltaNbBits` that makes `state >> nbBitsOut` fall in
        // `p..2p` for every state `tableSize..2*tableSize`, so `idx` lies
        // in `start_s..start_s + p`, the states `build` assigned to `s`,
        // all below `tableSize <= state_table.len()`. Every other code,
        // zero-count or past `max_symbol`, gets `deltaFindState = 0` and
        // `nbBitsOut = tableLog + 1`, so `idx == 0`; `build_rle` gives
        // every code `idx == 0`, its one state. `init_state` derives
        // `shifted` from the same transform, so its `idx` is in the same
        // range. A code above `MaxSeq` panics on the `symbol_tt` index
        // before reaching here.
        unsafe { *self.state_table.get_unchecked(idx) as u32 }
    }
}

/// `FSE_CState_t`: one encoder state over its table.
struct FseCState<'a> {
    value: u32,
    table: &'a FseCTable,
}

impl<'a> FseCState<'a> {
    /// `FSE_initCState2`.
    #[inline(always)]
    fn new(table: &'a FseCTable, symbol: u8) -> Self {
        Self {
            value: table.init_state(symbol),
            table,
        }
    }

    /// `FSE_encodeSymbol`: `BIT_addBits` keeps the low `nbBitsOut` bits of
    /// the state.
    #[inline(always)]
    fn encode(&mut self, bw: &mut BitCStream, symbol: u8) {
        let stt = self.table.symbol_tt[symbol as usize];
        let nb_bits = self.value.wrapping_add(stt.delta_nb_bits) >> 16;
        bw.add_bits(self.value as u64, nb_bits);
        self.value = self
            .table
            .next_state(self.value >> nb_bits, stt.delta_find_state);
    }

    /// `FSE_flushCState`.
    #[inline(always)]
    fn flush(&self, bw: &mut BitCStream) {
        bw.add_bits(self.value as u64, self.table.table_log);
        bw.flush_bits();
    }
}

fn highest_bit(v: u32) -> u32 {
    if v == 0 {
        return 0;
    }
    31 - v.leading_zeros()
}

/// `LL_bits` and `ML_bits` indexed by any byte, so a code from
/// `ll_code` / `ml_code` (never above `MAX_LL` / `MAX_ML`) needs no bounds
/// check.
static LL_BITS_BY_CODE: [u8; 256] = pad_bits(&LL_BITS);
static ML_BITS_BY_CODE: [u8; 256] = pad_bits(&ML_BITS);

const fn pad_bits<const N: usize>(bits: &[u8; N]) -> [u8; 256] {
    let mut out = [0u8; 256];
    let mut i = 0;
    while i < N {
        out[i] = bits[i];
        i += 1;
    }
    out
}

/// `ZSTD_encodeSequences`: FSE-encode `sequences` with the three tables
/// into a backward bitstream appended to `out`, and return its size.
/// `nb_seq >= 1` and the code slices are `nb_seq` long. `extra_bits` is
/// the total of the raw literal-length, match-length and offset bits of
/// `sequences`; with the per-symbol FSE bits bounded by the table logs, it
/// sizes the region the stream is written into. Runs the BMI2 build of
/// the body where the CPU has it (`DYNAMIC_BMI2`).
#[allow(clippy::too_many_arguments)]
pub fn encode_sequences(
    out: &mut Vec<u8>,
    ll_table: &FseCTable,
    of_table: &FseCTable,
    ml_table: &FseCTable,
    ll_codes: &[u8],
    of_codes: &[u8],
    ml_codes: &[u8],
    sequences: &[Seq],
    extra_bits: usize,
) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if super::bitstream::cpu_supports_bmi2() {
        // SAFETY: BMI2 was detected just above.
        return unsafe {
            encode_sequences_bmi2(
                out, ll_table, of_table, ml_table, ll_codes, of_codes, ml_codes, sequences,
                extra_bits,
            )
        };
    }
    encode_sequences_body(
        out, ll_table, of_table, ml_table, ll_codes, of_codes, ml_codes, sequences, extra_bits,
    )
}

/// `ZSTD_encodeSequences_bmi2`: [`encode_sequences_body`] compiled with
/// BMI2 enabled.
///
/// # Safety
///
/// The CPU must support BMI2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "bmi2")]
#[allow(clippy::too_many_arguments)]
unsafe fn encode_sequences_bmi2(
    out: &mut Vec<u8>,
    ll_table: &FseCTable,
    of_table: &FseCTable,
    ml_table: &FseCTable,
    ll_codes: &[u8],
    of_codes: &[u8],
    ml_codes: &[u8],
    sequences: &[Seq],
    extra_bits: usize,
) -> usize {
    encode_sequences_body(
        out, ll_table, of_table, ml_table, ll_codes, of_codes, ml_codes, sequences, extra_bits,
    )
}

/// `ZSTD_encodeSequences_body` (the `MEM_64bits` variant; `longOffsets`
/// only exists for 32-bit accumulators).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn encode_sequences_body(
    out: &mut Vec<u8>,
    ll_table: &FseCTable,
    of_table: &FseCTable,
    ml_table: &FseCTable,
    ll_codes: &[u8],
    of_codes: &[u8],
    ml_codes: &[u8],
    sequences: &[Seq],
    extra_bits: usize,
) -> usize {
    let nb_seq = sequences.len();
    debug_assert!(nb_seq >= 1);
    assert!(ll_codes.len() == nb_seq && of_codes.len() == nb_seq && ml_codes.len() == nb_seq);
    let start = out.len();
    let max_bits = extra_bits + nb_seq * (LL_FSE_LOG + OFF_FSE_LOG + ML_FSE_LOG) as usize + 1;
    out.resize(start + BitCStream::capacity_for(max_bits), 0);
    let mut bw = BitCStream::new(&mut out[start..]);

    // first symbols
    let last = nb_seq - 1;
    let mut state_ml = FseCState::new(ml_table, ml_codes[last]);
    let mut state_of = FseCState::new(of_table, of_codes[last]);
    let mut state_ll = FseCState::new(ll_table, ll_codes[last]);
    // `BIT_addBits` keeps the low `nbBits` of the raw value: every base is a
    // multiple of `1 << nbBits`, so that equals `value - base`.
    bw.add_bits(
        sequences[last].lit_len as u64,
        LL_BITS_BY_CODE[ll_codes[last] as usize] as u32,
    );
    bw.add_bits(
        sequences[last].ml_base as u64,
        ML_BITS_BY_CODE[ml_codes[last] as usize] as u32,
    );
    bw.add_bits(sequences[last].off_base as u64, of_codes[last] as u32);
    bw.flush_bits();

    let sequences = &sequences[..last];
    let ll_codes = &ll_codes[..last];
    let of_codes = &of_codes[..last];
    let ml_codes = &ml_codes[..last];
    for n in (0..last).rev() {
        let ll_code = ll_codes[n];
        let of_code = of_codes[n];
        let ml_code = ml_codes[n];
        let ll_bits = LL_BITS_BY_CODE[ll_code as usize] as u32;
        let of_bits = of_code as u32;
        let ml_bits = ML_BITS_BY_CODE[ml_code as usize] as u32;
        state_of.encode(&mut bw, of_code);
        state_ml.encode(&mut bw, ml_code);
        state_ll.encode(&mut bw, ll_code);
        if of_bits + ml_bits + ll_bits >= 64 - 7 - (LL_FSE_LOG + ML_FSE_LOG + OFF_FSE_LOG) {
            bw.flush_bits();
        }
        bw.add_bits(sequences[n].lit_len as u64, ll_bits);
        bw.add_bits(sequences[n].ml_base as u64, ml_bits);
        if of_bits + ml_bits + ll_bits > 56 {
            bw.flush_bits();
        }
        bw.add_bits(sequences[n].off_base as u64, of_bits);
        bw.flush_bits();
    }

    state_ml.flush(&mut bw);
    state_of.flush(&mut bw);
    state_ll.flush(&mut bw);

    let size = bw.close();
    debug_assert!(size != 0, "bitstream exceeded its bound");
    out.truncate(start + size);
    size
}

/// `FSE_compress_usingCTable` (the 64-bit `FSE_compress_usingCTable_generic`):
/// two interleaved states over `src`, appended to `out` as a backward
/// bitstream. Returns the byte size, or `0` for `src.len() <= 2`. The
/// region is `FSE_BLOCKBOUND(srcSize)`, which the stream never exceeds
/// with a table log of at most 6 bits per symbol.
pub fn compress_using_ctable(out: &mut Vec<u8>, src: &[u8], ct: &FseCTable) -> usize {
    let mut src_size = src.len();
    if src_size <= 2 {
        return 0;
    }
    let start = out.len();
    out.resize(start + src_size + (src_size >> 7) + 4 + 8, 0);
    let mut bw = BitCStream::new(&mut out[start..]);
    let mut ip = src_size;
    let next = |ip: &mut usize| {
        *ip -= 1;
        src[*ip]
    };

    let (mut state1, mut state2);
    if src_size & 1 != 0 {
        state1 = FseCState::new(ct, next(&mut ip));
        state2 = FseCState::new(ct, next(&mut ip));
        state1.encode(&mut bw, next(&mut ip));
        bw.flush_bits();
    } else {
        state2 = FseCState::new(ct, next(&mut ip));
        state1 = FseCState::new(ct, next(&mut ip));
    }

    // join to mod 4
    src_size -= 2;
    if src_size & 2 != 0 {
        state2.encode(&mut bw, next(&mut ip));
        state1.encode(&mut bw, next(&mut ip));
        bw.flush_bits();
    }

    // 4 encoding per loop
    while ip > 0 {
        state2.encode(&mut bw, next(&mut ip));
        state1.encode(&mut bw, next(&mut ip));
        state2.encode(&mut bw, next(&mut ip));
        state1.encode(&mut bw, next(&mut ip));
        bw.flush_bits();
    }

    state2.flush(&mut bw);
    state1.flush(&mut bw);
    let size = bw.close();
    out.truncate(start + size);
    size
}

/// `SymbolEncodingType_e`: how one sequence table is transmitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolEncodingType {
    Basic = 0,
    Rle = 1,
    Compressed = 2,
    Repeat = 3,
}

/// `FSE_MIN_TABLELOG`.
pub const FSE_MIN_TABLELOG: u32 = 5;
/// `FSE_MAX_TABLELOG`.
pub const FSE_MAX_TABLELOG: u32 = 12;
/// `FSE_MAX_SYMBOL_VALUE`.
pub const FSE_MAX_SYMBOL_VALUE: usize = 255;
/// `FSE_DEFAULT_TABLELOG`.
pub const FSE_DEFAULT_TABLELOG: u32 = 11;

/// `ERROR(GENERIC)` / `ERROR(tableLog_tooLarge)` from `FSE_normalizeCount`:
/// the counts cannot be represented at the requested table log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormalizeError;

/// `FSE_minTableLog`: the minimum table log that can represent
/// `src_size` symbols drawn from `0..=max_symbol`.
fn min_table_log(src_size: usize, max_symbol: usize) -> u32 {
    let min_bits_src = highest_bit(src_size as u32) + 1;
    let min_bits_symbols = highest_bit(max_symbol as u32) + 2;
    min_bits_src.min(min_bits_symbols)
}

/// `FSE_optimalTableLog_internal`. `src_size` must be `> 1`.
pub fn optimal_table_log_internal(
    max_table_log: u32,
    src_size: usize,
    max_symbol: usize,
    minus: u32,
) -> u32 {
    debug_assert!(src_size > 1);
    // C: unsigned arithmetic, so `highbit(1) - 2` wraps and never lowers
    // the table log.
    let max_bits_src = highest_bit((src_size - 1) as u32).wrapping_sub(minus);
    let min_bits = min_table_log(src_size, max_symbol);
    let mut table_log = max_table_log;
    if table_log == 0 {
        table_log = FSE_DEFAULT_TABLELOG;
    }
    if max_bits_src < table_log {
        table_log = max_bits_src;
    }
    if min_bits > table_log {
        table_log = min_bits;
    }
    table_log.clamp(FSE_MIN_TABLELOG, FSE_MAX_TABLELOG)
}

/// `FSE_optimalTableLog` (`minus == 2`).
pub fn optimal_table_log(max_table_log: u32, src_size: usize, max_symbol: usize) -> u32 {
    optimal_table_log_internal(max_table_log, src_size, max_symbol, 2)
}

/// `FSE_normalizeCount`: scale `counts[..=max_symbol]` (summing to `total`)
/// to a distribution summing to `1 << table_log`, written to `norm`.
/// Symbols at or below `total >> table_log` get `-1` when
/// `use_low_prob_count` (`ZSTD_useLowProbCount`) and `1` otherwise.
/// Returns the table log used, or `Ok(0)` without touching `norm` when one
/// symbol carries every count (the caller must emit RLE). `table_log == 0`
/// selects `FSE_DEFAULT_TABLELOG`.
pub fn normalize_count(
    norm: &mut [i16],
    table_log: u32,
    counts: &[u32],
    total: usize,
    max_symbol: usize,
    use_low_prob_count: bool,
) -> Result<u32, NormalizeError> {
    static RTB_TABLE: [u64; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];

    let table_log = if table_log == 0 {
        FSE_DEFAULT_TABLELOG
    } else {
        table_log
    };
    if !(FSE_MIN_TABLELOG..=FSE_MAX_TABLELOG).contains(&table_log)
        || table_log < min_table_log(total, max_symbol)
    {
        return Err(NormalizeError);
    }

    let low_prob_count: i16 = if use_low_prob_count { -1 } else { 1 };
    let scale = 62 - table_log;
    let step = (1u64 << 62) / total as u64;
    let v_step = 1u64 << (scale - 20);
    let mut still_to_distribute = 1i32 << table_log;
    let mut largest = 0usize;
    let mut largest_p = 0i16;
    let low_threshold = (total >> table_log) as u32;

    for s in 0..=max_symbol {
        let count = counts[s];
        if count as usize == total {
            return Ok(0);
        }
        if count == 0 {
            norm[s] = 0;
            continue;
        }
        if count <= low_threshold {
            norm[s] = low_prob_count;
            still_to_distribute -= 1;
        } else {
            let scaled = count as u64 * step;
            let mut proba = (scaled >> scale) as i16;
            if proba < 8 {
                let rest_to_beat = v_step * RTB_TABLE[proba as usize];
                proba += (scaled - ((proba as u64) << scale) > rest_to_beat) as i16;
            }
            if proba > largest_p {
                largest_p = proba;
                largest = s;
            }
            norm[s] = proba;
            still_to_distribute -= proba as i32;
        }
    }
    if -still_to_distribute >= (norm[largest] >> 1) as i32 {
        // corner case, need another normalization method
        normalize_m2(norm, table_log, counts, total, max_symbol, low_prob_count)?;
    } else {
        norm[largest] += still_to_distribute as i16;
    }
    Ok(table_log)
}

/// `FSE_normalizeM2`: secondary normalization, used when the primary
/// method over-allocates.
fn normalize_m2(
    norm: &mut [i16],
    table_log: u32,
    counts: &[u32],
    mut total: usize,
    max_symbol: usize,
    low_prob_count: i16,
) -> Result<(), NormalizeError> {
    const NOT_YET_ASSIGNED: i16 = -2;
    let mut distributed = 0u32;
    let low_threshold = (total >> table_log) as u32;
    let mut low_one = ((total * 3) >> (table_log + 1)) as u32;

    for s in 0..=max_symbol {
        let count = counts[s];
        if count == 0 {
            norm[s] = 0;
            continue;
        }
        if count <= low_threshold {
            norm[s] = low_prob_count;
            distributed += 1;
            total -= count as usize;
            continue;
        }
        if count <= low_one {
            norm[s] = 1;
            distributed += 1;
            total -= count as usize;
            continue;
        }
        norm[s] = NOT_YET_ASSIGNED;
    }
    let mut to_distribute = (1u32 << table_log) - distributed;

    if to_distribute == 0 {
        return Ok(());
    }

    if (total / to_distribute as usize) as u32 > low_one {
        // risk of rounding to zero
        low_one = ((total * 3) / (to_distribute as usize * 2)) as u32;
        for s in 0..=max_symbol {
            if norm[s] == NOT_YET_ASSIGNED && counts[s] <= low_one {
                norm[s] = 1;
                distributed += 1;
                total -= counts[s] as usize;
            }
        }
        to_distribute = (1u32 << table_log) - distributed;
    }

    if distributed as usize == max_symbol + 1 {
        // all values are pretty poor; give all remaining points to max
        let mut max_v = 0usize;
        let mut max_c = 0u32;
        for s in 0..=max_symbol {
            if counts[s] > max_c {
                max_v = s;
                max_c = counts[s];
            }
        }
        norm[max_v] += to_distribute as i16;
        return Ok(());
    }

    if total == 0 {
        // all of the symbols were low enough for the lowOne or lowThreshold
        let mut s = 0usize;
        while to_distribute > 0 {
            if norm[s] > 0 {
                to_distribute -= 1;
                norm[s] += 1;
            }
            s = (s + 1) % (max_symbol + 1);
        }
        return Ok(());
    }

    let v_step_log = 62 - table_log;
    let mid = (1u64 << (v_step_log - 1)) - 1;
    // scale on remaining
    let r_step = ((1u64 << v_step_log) * to_distribute as u64 + mid) / total as u64;
    let mut tmp_total = mid;
    for s in 0..=max_symbol {
        if norm[s] == NOT_YET_ASSIGNED {
            let end = tmp_total + counts[s] as u64 * r_step;
            let s_start = (tmp_total >> v_step_log) as u32;
            let s_end = (end >> v_step_log) as u32;
            let weight = s_end - s_start;
            if weight < 1 {
                return Err(NormalizeError);
            }
            norm[s] = weight as i16;
            tmp_total = end;
        }
    }
    Ok(())
}

/// `FSE_writeNCount`: append the normalized-count header for
/// `norm[..=max_symbol]` to `out` and return its byte size. Fails as the C
/// `GENERIC` cases do when `norm` does not sum to `1 << table_log`.
pub fn write_ncount(
    out: &mut Vec<u8>,
    norm: &[i16],
    max_symbol: usize,
    table_log: u32,
) -> Result<usize, NormalizeError> {
    if !(FSE_MIN_TABLELOG..=FSE_MAX_TABLELOG).contains(&table_log) {
        return Err(NormalizeError);
    }
    let start = out.len();
    let table_size = 1i32 << table_log;
    let mut remaining = table_size + 1; // +1 for extra accuracy
    let mut threshold = table_size;
    let mut nb_bits = table_log as i32 + 1;
    let mut bit_stream: u32 = table_log - FSE_MIN_TABLELOG;
    let mut bit_count: i32 = 4;
    let mut symbol = 0usize;
    let alphabet_size = max_symbol + 1;
    let mut previous_is_0 = false;

    while symbol < alphabet_size && remaining > 1 {
        if previous_is_0 {
            let mut start_sym = symbol;
            while symbol < alphabet_size && norm[symbol] == 0 {
                symbol += 1;
            }
            if symbol == alphabet_size {
                break; // incorrect distribution
            }
            while symbol >= start_sym + 24 {
                start_sym += 24;
                bit_stream += 0xFFFF << bit_count;
                out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
                bit_stream >>= 16;
            }
            while symbol >= start_sym + 3 {
                start_sym += 3;
                bit_stream += 3 << bit_count;
                bit_count += 2;
            }
            bit_stream += ((symbol - start_sym) as u32) << bit_count;
            bit_count += 2;
            if bit_count > 16 {
                out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
                bit_stream >>= 16;
                bit_count -= 16;
            }
        }
        {
            let mut count = norm[symbol] as i32;
            symbol += 1;
            let max = (2 * threshold - 1) - remaining;
            remaining -= count.abs();
            count += 1; // +1 for extra accuracy
            if count >= threshold {
                count += max; // [0..max[ [max..threshold[ (...) [threshold+max 2*threshold[
            }
            bit_stream += (count as u32) << bit_count;
            bit_count += nb_bits;
            bit_count -= (count < max) as i32;
            previous_is_0 = count == 1;
            if remaining < 1 {
                out.truncate(start);
                return Err(NormalizeError);
            }
            while remaining < threshold {
                nb_bits -= 1;
                threshold >>= 1;
            }
        }
        if bit_count > 16 {
            out.extend_from_slice(&(bit_stream as u16).to_le_bytes());
            bit_stream >>= 16;
            bit_count -= 16;
        }
    }

    if remaining != 1 {
        out.truncate(start);
        return Err(NormalizeError); // incorrect normalized distribution
    }

    // flush remaining bitStream
    let tail = [bit_stream as u8, (bit_stream >> 8) as u8];
    out.extend_from_slice(&tail[..((bit_count + 7) / 8) as usize]);
    Ok(out.len() - start)
}

/// `kInverseProbabilityLog256`: `floor(-log2(x / 256) * 256)` for
/// `x in 1..256`, `0` at `x == 0`.
static K_INVERSE_PROBABILITY_LOG256: [u32; 256] = [
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162, 1130, 1100, 1073, 1047,
    1024, 1001, 980, 960, 941, 923, 906, 889, 874, 859, 844, 830, 817, 804, 791, 779, 768, 756,
    745, 734, 724, 714, 704, 694, 685, 676, 667, 658, 650, 642, 633, 626, 618, 610, 603, 595, 588,
    581, 574, 567, 561, 554, 548, 542, 535, 529, 523, 517, 512, 506, 500, 495, 489, 484, 478, 473,
    468, 463, 458, 453, 448, 443, 438, 434, 429, 424, 420, 415, 411, 407, 402, 398, 394, 390, 386,
    382, 377, 373, 370, 366, 362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322, 318, 315,
    311, 308, 305, 302, 298, 295, 292, 289, 286, 282, 279, 276, 273, 270, 267, 264, 261, 258, 256,
    253, 250, 247, 244, 241, 239, 236, 233, 230, 228, 225, 222, 220, 217, 215, 212, 209, 207, 204,
    202, 199, 197, 194, 192, 190, 187, 185, 182, 180, 178, 175, 173, 171, 168, 166, 164, 162, 159,
    157, 155, 153, 151, 149, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 123, 121, 119,
    117, 115, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85, 83, 82, 80,
    78, 76, 74, 73, 71, 69, 67, 66, 64, 62, 61, 59, 57, 55, 54, 52, 50, 49, 47, 46, 44, 42, 41, 39,
    37, 36, 34, 33, 31, 30, 28, 26, 25, 23, 22, 20, 19, 17, 16, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1,
];

/// `ZSTD_useLowProbCount`.
fn use_low_prob_count(nb_seq: usize) -> bool {
    nb_seq >= 2048
}

/// `ZSTD_NCountCost`: byte size of the normalized-count header for
/// `counts`, or `None` where the C returns an error. The header is written
/// past the end of `wksp`, which is left as it was.
fn ncount_cost(
    wksp: &mut Vec<u8>,
    counts: &[u32],
    max: usize,
    nb_seq: usize,
    fse_log: u32,
) -> Option<usize> {
    let table_log = optimal_table_log(fse_log, nb_seq, max);
    let mut norm = [0i16; MAX_SEQ + 1];
    let log = normalize_count(
        &mut norm,
        table_log,
        counts,
        nb_seq,
        max,
        use_low_prob_count(nb_seq),
    )
    .ok()?;
    if log == 0 {
        return None;
    }
    let start = wksp.len();
    let size = write_ncount(wksp, &norm, max, table_log);
    wksp.truncate(start);
    size.ok()
}

/// `ZSTD_entropyCost`: bits to encode `counts` at the entropy bound.
fn entropy_cost(counts: &[u32], max: usize, total: usize) -> u64 {
    debug_assert!(total > 0);
    let mut cost = 0u64;
    for &count in &counts[..=max] {
        let mut norm = (256 * count as u64 / total as u64) as usize;
        if count != 0 && norm == 0 {
            norm = 1;
        }
        debug_assert!((count as usize) < total);
        cost += count as u64 * K_INVERSE_PROBABILITY_LOG256[norm] as u64;
    }
    cost >> 8
}

/// `ZSTD_crossEntropyCost`: bits to encode `counts` with the table
/// described by `norm` (which must cover every counted symbol).
fn cross_entropy_cost(norm: &[i16], accuracy_log: u32, counts: &[u32], max: usize) -> u64 {
    let shift = 8 - accuracy_log;
    debug_assert!(accuracy_log <= 8);
    let mut cost = 0u64;
    for s in 0..=max {
        let norm_acc = if norm[s] != -1 { norm[s] as u32 } else { 1 };
        let norm256 = norm_acc << shift;
        debug_assert!(norm256 > 0 && norm256 < 256);
        cost += counts[s] as u64 * K_INVERSE_PROBABILITY_LOG256[norm256 as usize] as u64;
    }
    cost >> 8
}

/// `FSE_bitCost`: cost of `symbol` in `1 / (1 << accuracy_log)` bits,
/// linearly interpolated between its two possible bit counts.
fn fse_bit_cost(tt: &SymbolTT, table_log: u32, accuracy_log: u32) -> u32 {
    let min_nb_bits = tt.delta_nb_bits >> 16;
    let threshold = (min_nb_bits + 1) << 16;
    debug_assert!(table_log < 16);
    debug_assert!(accuracy_log < 31 - table_log);
    let table_size = 1u32 << table_log;
    let delta_from_threshold = threshold.wrapping_sub(tt.delta_nb_bits.wrapping_add(table_size));
    let normalized_delta_from_threshold = (delta_from_threshold << accuracy_log) >> table_log;
    let bit_multiplier = 1u32 << accuracy_log;
    debug_assert!(tt.delta_nb_bits.wrapping_add(table_size) <= threshold);
    debug_assert!(normalized_delta_from_threshold <= bit_multiplier);
    (min_nb_bits + 1) * bit_multiplier - normalized_delta_from_threshold
}

impl FseCTable {
    /// `ZSTD_fseBitCost`: bits to encode `counts` with this table, or
    /// `None` when the table cannot represent every counted symbol.
    pub fn bit_cost(&self, counts: &[u32], max: usize) -> Option<u64> {
        const K_ACCURACY_LOG: u32 = 8;
        if self.max_symbol < max {
            return None;
        }
        let bad_cost = (self.table_log + 1) << K_ACCURACY_LOG;
        let mut cost = 0u64;
        for s in 0..=max {
            let bit_cost = fse_bit_cost(&self.symbol_tt[s], self.table_log, K_ACCURACY_LOG);
            if counts[s] == 0 {
                continue;
            }
            if bit_cost >= bad_cost {
                return None;
            }
            cost += counts[s] as u64 * bit_cost as u64;
        }
        Some(cost >> K_ACCURACY_LOG)
    }
}

/// `ZSTD_selectEncodingType`. Costs the C reports as errors are modelled
/// as `None`, which is never selected; when nothing is selectable
/// (unreachable in libzstd, which asserts) the result is `Compressed` and
/// [`build_ctable`] reports the failure. `wksp` is [`ncount_cost`]'s.
#[allow(clippy::too_many_arguments)]
fn select_encoding_type(
    wksp: &mut Vec<u8>,
    repeat_mode: &mut Repeat,
    counts: &[u32],
    max: usize,
    most_frequent: usize,
    nb_seq: usize,
    fse_log: u32,
    prev_ctable: Option<&FseCTable>,
    default_norm: &[i16],
    default_norm_log: u32,
    is_default_allowed: bool,
    strategy: Strategy,
) -> SymbolEncodingType {
    if most_frequent == nb_seq {
        *repeat_mode = Repeat::None;
        if is_default_allowed && nb_seq <= 2 {
            // Prefer set_basic over set_rle when there are 2 or fewer
            // symbols, since RLE uses 1 byte, but set_basic uses 5-6 bits
            // per symbol. If basic encoding isn't possible, always choose RLE.
            return SymbolEncodingType::Basic;
        }
        return SymbolEncodingType::Rle;
    }
    if strategy < Strategy::Lazy {
        if is_default_allowed {
            let static_fse_nb_seq_max = 1000;
            let mult = 10 - strategy as usize;
            let base_log = 3;
            // 28-36 for offset, 56-72 for lengths
            let dynamic_fse_nb_seq_min = ((1usize << default_norm_log) * mult) >> base_log;
            debug_assert!((5..=6).contains(&default_norm_log));
            debug_assert!((7..=9).contains(&mult));
            if *repeat_mode == Repeat::Valid && nb_seq < static_fse_nb_seq_max {
                return SymbolEncodingType::Repeat;
            }
            if nb_seq < dynamic_fse_nb_seq_min || most_frequent < (nb_seq >> (default_norm_log - 1))
            {
                // The format allows default tables to be repeated, but it
                // isn't useful: don't confuse them with dictionaries.
                *repeat_mode = Repeat::None;
                return SymbolEncodingType::Basic;
            }
        }
    } else {
        let basic_cost = is_default_allowed
            .then(|| cross_entropy_cost(default_norm, default_norm_log, counts, max));
        let repeat_cost = match (*repeat_mode, prev_ctable) {
            (Repeat::None, _) | (_, None) => None,
            (_, Some(table)) => table.bit_cost(counts, max),
        };
        let compressed_cost = ncount_cost(wksp, counts, max, nb_seq, fse_log)
            .map(|ncount| ((ncount as u64) << 3) + entropy_cost(counts, max, nb_seq));
        let repeat_or_max = repeat_cost.unwrap_or(u64::MAX);
        let compressed_or_max = compressed_cost.unwrap_or(u64::MAX);
        if let Some(basic) = basic_cost {
            if basic <= repeat_or_max && basic <= compressed_or_max {
                *repeat_mode = Repeat::None;
                return SymbolEncodingType::Basic;
            }
        }
        if let Some(repeat) = repeat_cost {
            if repeat <= compressed_or_max {
                return SymbolEncodingType::Repeat;
            }
        }
    }
    *repeat_mode = Repeat::Check;
    SymbolEncodingType::Compressed
}

/// `ZSTD_buildCTable`: write the table description for `ty` to `out` and
/// return the table to encode with, the held one for `Repeat`, else built
/// in the spare slot, plus the number of bytes written. `None` where the
/// C fails (normalization or NCount write errors, or `Repeat` without a
/// held table).
#[allow(clippy::too_many_arguments)]
fn build_ctable<'t>(
    out: &mut Vec<u8>,
    fse_log: u32,
    ty: SymbolEncodingType,
    counts: &mut [u32],
    max: usize,
    codes: &[u8],
    nb_seq: usize,
    default_norm: &[i16],
    default_norm_log: u32,
    default_max: usize,
    table: TableRef<'t, FseCTable>,
) -> Option<(&'t FseCTable, usize)> {
    let TableRef { held, spare } = table;
    match ty {
        SymbolEncodingType::Rle => {
            out.push(codes[0]);
            spare.build_rle(max as u8);
            Some((spare, 1))
        }
        SymbolEncodingType::Repeat => Some((held.table()?, 0)),
        SymbolEncodingType::Basic => {
            spare.build(default_norm, default_max, default_norm_log);
            Some((spare, 0))
        }
        SymbolEncodingType::Compressed => {
            let mut nb_seq_1 = nb_seq;
            let table_log = optimal_table_log(fse_log, nb_seq, max);
            let last = codes[nb_seq - 1] as usize;
            if counts[last] > 1 {
                counts[last] -= 1;
                nb_seq_1 -= 1;
            }
            debug_assert!(nb_seq_1 > 1);
            let mut norm = [0i16; MAX_SEQ + 1];
            let log = normalize_count(
                &mut norm,
                table_log,
                counts,
                nb_seq_1,
                max,
                use_low_prob_count(nb_seq_1),
            )
            .ok()?;
            if log == 0 {
                return None;
            }
            let ncount_size = write_ncount(out, &norm[..=max], max, table_log).ok()?;
            spare.build(&norm, max, table_log);
            Some((spare, ncount_size))
        }
    }
}

/// Select, describe and build one sequence table
/// (one `ZSTD_selectEncodingType` + `ZSTD_buildCTable` step of
/// `ZSTD_buildSequencesStatistics`). `counts`, `max` and `most_frequent`
/// are the `HIST_countFast_wksp` result for `codes`. Returns the table to
/// encode with, what the block does to the decoder's table, the encoding
/// type and the description size.
#[allow(clippy::too_many_arguments)]
fn build_seq_table<'t>(
    out: &mut Vec<u8>,
    codes: &[u8],
    counts: &mut [u32; 256],
    max: usize,
    most_frequent: usize,
    fse_log: u32,
    table: TableRef<'t, FseCTable>,
    default_norm: &[i16],
    default_norm_log: u32,
    default_max: usize,
    strategy: Strategy,
) -> Option<(&'t FseCTable, Next, SymbolEncodingType, usize)> {
    let nb_seq = codes.len();
    // We can only use the basic table if max <= DefaultMaxOff, otherwise
    // the offsets are too large (a no-op for LL/ML, whose default tables
    // span every code).
    let is_default_allowed = max <= default_max;
    let mut repeat_mode = table.held.repeat();
    let ty = select_encoding_type(
        out,
        &mut repeat_mode,
        &counts[..],
        max,
        most_frequent,
        nb_seq,
        fse_log,
        table.held.table(),
        default_norm,
        default_norm_log,
        is_default_allowed,
        strategy,
    );
    // We don't copy tables: Basic and Rle leave nothing to repeat.
    debug_assert!(
        matches!(
            ty,
            SymbolEncodingType::Compressed | SymbolEncodingType::Repeat
        ) || repeat_mode == Repeat::None
    );
    // the mode select_encoding_type left (`FSE_repeat`), applied when the
    // block is committed
    let next = match ty {
        SymbolEncodingType::Repeat => Next::Keep,
        SymbolEncodingType::Compressed => Next::New,
        SymbolEncodingType::Basic | SymbolEncodingType::Rle => Next::None,
    };
    debug_assert!(next != Next::New || repeat_mode == Repeat::Check);
    let (table, size) = build_ctable(
        out,
        fse_log,
        ty,
        &mut counts[..],
        max,
        codes,
        nb_seq,
        default_norm,
        default_norm_log,
        default_max,
        table,
    )?;
    Some((table, next, ty, size))
}

/// `FSE_NCountWriteBound`: maximum size of an `FSE_writeNCount` table
/// description for symbols `0..=max_symbol` at `table_log`.
fn ncount_write_bound(max_symbol: usize, table_log: u32) -> usize {
    if max_symbol == 0 {
        return FSE_NCOUNTBOUND;
    }
    ((max_symbol + 1) * table_log as usize + 4 + 2) / 8 + 1 + 2
}

/// Upper bound on the bytes [`encode_sequences_section_with`] appends for
/// `seqs`, whatever the tables the decoder holds and the [`CParams`], over
/// every encoding type `ZSTD_selectEncodingType` can pick per stream:
/// the sequence-count header, the modes byte, per stream the larger of the
/// RLE byte and `FSE_NCountWriteBound(max code, *FSELog)` (Basic and
/// Repeat write nothing), and the bitstream: per sequence its exact extra
/// bits plus `LLFSELog + OffFSELog + MLFSELog` state bits (no table,
/// default, repeated or new, exceeds those logs, and `FSE_encodeSymbol`
/// emits at most `tableLog` bits; the last sequence's symbols seed the
/// states and the final flushes spend the same budget), then the end mark
/// and the padding to a byte.
///
/// The failure returns of [`encode_sequences_section_with`]:
/// - `build_ctable` for `Repeat` without a previous table: unreachable,
///   `select_encoding_type` returns `Repeat` only from a `Valid` repeat
///   mode or a repeat cost, both of which need the table;
/// - `normalize_count` failing or returning 0, and `write_ncount` failing:
///   unreachable, `Compressed` is only picked when no code covers every
///   sequence (else `Rle`/`Basic`), so at least two codes remain after the
///   last-symbol decrement, and `optimal_table_log` keeps the log within
///   `FSE_MIN_TABLELOG..=*FSELog` and at least `FSE_minTableLog`;
/// - the 1.3.4 workaround (`lastCountSize + bitstreamSize < 4`): a new
///   table needs two sequences, its description takes at least 2 bytes
///   and its flush at least `FSE_MIN_TABLELOG` bits, so the bitstream must
///   fit in 1 byte with at most 2 extra bits. For such `seqs` the bound is
///   raised above `ZSTD_BLOCKSIZE_MAX`, so it can never prove a block
///   compressed; it still bounds the section.
///
/// When `literals_section_bound + sequences_section_bound < block_len -
/// ZSTD_minGain` the block is therefore emitted compressed (RLE blocks and
/// blocks below `MIN_CBLOCK_SIZE` are decided before either stage).
pub fn sequences_section_bound(seqs: &[Seq]) -> usize {
    let nb_seq = seqs.len();
    let header = 1 + (nb_seq >= 128) as usize + (nb_seq >= LONGNBSEQ) as usize;
    if nb_seq == 0 {
        return header;
    }
    let (mut ll_max, mut of_max, mut ml_max) = (0u8, 0u8, 0u8);
    let mut extra_bits = 0usize;
    for seq in seqs {
        let (ll, of, ml) = (
            ll_code(seq.lit_len),
            off_code(seq.off_base),
            ml_code(seq.ml_base),
        );
        ll_max = ll_max.max(ll);
        of_max = of_max.max(of);
        ml_max = ml_max.max(ml);
        extra_bits += LL_BITS[ll as usize] as usize + ML_BITS[ml as usize] as usize + of as usize;
    }
    let descriptions = ncount_write_bound(ll_max as usize, LL_FSE_LOG)
        + ncount_write_bound(of_max as usize, OFF_FSE_LOG)
        + ncount_write_bound(ml_max as usize, ML_FSE_LOG);
    let state_bits = nb_seq * (LL_FSE_LOG + OFF_FSE_LOG + ML_FSE_LOG) as usize;
    let bitstream = (extra_bits + state_bits + 1).div_ceil(8);
    let bound = header + 1 + descriptions + bitstream;
    if nb_seq >= 2 && extra_bits <= 2 {
        // the 1.3.4 workaround may fire: never prove compression
        return bound.max(ZSTD_BLOCKSIZE_MAX + 1);
    }
    bound
}

/// `ZSTD_seqToCodes`: the literal-length, offset and match-length codes of
/// `sequences`, written to `codes` (`3 * sequences.len()` bytes) and
/// returned as three slices.
#[inline(always)]
fn seq_to_codes<'a>(
    sequences: &[Seq],
    codes: &'a mut [u8],
) -> (&'a mut [u8], &'a mut [u8], &'a mut [u8]) {
    let nb_seq = sequences.len();
    let (ll_codes, rest) = codes.split_at_mut(nb_seq);
    let (of_codes, ml_codes) = rest.split_at_mut(nb_seq);
    for (((seq, ll), of), ml) in sequences
        .iter()
        .zip(&mut *ll_codes)
        .zip(&mut *of_codes)
        .zip(&mut *ml_codes)
    {
        *ll = ll_code(seq.lit_len);
        *of = off_code(seq.off_base);
        *ml = ml_code(seq.ml_base);
    }
    (ll_codes, of_codes, &mut ml_codes[..nb_seq])
}

/// Write the sequences section (Sequences_Section_Header onward, as in
/// `ZSTD_entropyCompressSeqStore_internal`) against the tables the decoder
/// holds, building new ones in the spare slots of `tables`, and return
/// what the section does to them (`nextEntropy->fse`). With `nb_seq == 0`
/// the tables carry over unchanged (`nextEntropy->fse =
/// prevEntropy->fse`). `codes` is the buffer the sequence codes are
/// written to (`ZSTD_seqToCodes`' `llCode`, `ofCode` and `mlCode`), kept
/// across blocks.
///
/// `None` means the block must be emitted uncompressed: libzstd returns 0
/// for the 1.3.4 decoder workaround (the last table description plus the
/// bitstream under 4 bytes) and fails the compression when a table cannot
/// be built; both end here as a raw block. `out` may then hold a partial
/// section. [`sequences_section_bound`] lists when each is reachable.
pub fn encode_sequences_section_with(
    out: &mut Vec<u8>,
    sequences: &[Seq],
    codes: &mut Vec<u8>,
    tables: FseTables<'_>,
    cparams: &CParams,
) -> Option<FseNext> {
    let strategy = cparams.strategy;
    let nb_seq = sequences.len();

    // Sequences Header
    if nb_seq < 128 {
        out.push(nb_seq as u8);
    } else if nb_seq < LONGNBSEQ {
        out.push(((nb_seq >> 8) as u8) + 0x80);
        out.push(nb_seq as u8);
    } else {
        out.push(0xFF);
        out.extend_from_slice(&((nb_seq - LONGNBSEQ) as u16).to_le_bytes());
    }
    if nb_seq == 0 {
        // Copy the old tables over as if we repeated them
        return Some(FseNext::KEEP);
    }
    let seq_head = out.len();
    out.push(0);

    if codes.len() < 3 * nb_seq {
        codes.resize(3 * nb_seq, 0);
    }
    let (ll_codes, of_codes, ml_codes) = seq_to_codes(sequences, &mut codes[..3 * nb_seq]);

    // The `HIST_countFast_wksp` of each `ZSTD_buildSequencesStatistics`
    // step, taken up front so the histograms also total the raw bits the
    // bitstream carries.
    let mut ll_counts = [0u32; 256];
    let mut of_counts = [0u32; 256];
    let mut ml_counts = [0u32; 256];
    let (ll_most, ll_max) = huf::hist_count(&mut ll_counts, ll_codes, MAX_LL);
    let (of_most, of_max) = huf::hist_count(&mut of_counts, of_codes, MAX_OFF);
    let (ml_most, ml_max) = huf::hist_count(&mut ml_counts, ml_codes, MAX_ML);
    let raw_bits = |counts: &[u32], bits: &dyn Fn(usize) -> usize| {
        counts
            .iter()
            .enumerate()
            .map(|(c, &n)| n as usize * bits(c))
            .sum::<usize>()
    };
    let extra_bits = raw_bits(&ll_counts[..=ll_max], &|c| LL_BITS[c] as usize)
        + raw_bits(&ml_counts[..=ml_max], &|c| ML_BITS[c] as usize)
        + raw_bits(&of_counts[..=of_max], &|c| c);

    // ZSTD_buildSequencesStatistics: LL, then OF, then ML
    let mut last_count_size = 0;
    let (ll_table, ll_next, ll_type, size) = build_seq_table(
        out,
        ll_codes,
        &mut ll_counts,
        ll_max,
        ll_most as usize,
        LL_FSE_LOG,
        tables.ll,
        &LL_DEFAULT_NORM,
        LL_DEFAULT_NORM_LOG,
        MAX_LL,
        strategy,
    )?;
    if ll_type == SymbolEncodingType::Compressed {
        last_count_size = size;
    }
    let (of_table, of_next, of_type, size) = build_seq_table(
        out,
        of_codes,
        &mut of_counts,
        of_max,
        of_most as usize,
        OFF_FSE_LOG,
        tables.of,
        &OF_DEFAULT_NORM,
        OF_DEFAULT_NORM_LOG,
        DEFAULT_MAX_OFF,
        strategy,
    )?;
    if of_type == SymbolEncodingType::Compressed {
        last_count_size = size;
    }
    let (ml_table, ml_next, ml_type, size) = build_seq_table(
        out,
        ml_codes,
        &mut ml_counts,
        ml_max,
        ml_most as usize,
        ML_FSE_LOG,
        tables.ml,
        &ML_DEFAULT_NORM,
        ML_DEFAULT_NORM_LOG,
        MAX_ML,
        strategy,
    )?;
    if ml_type == SymbolEncodingType::Compressed {
        last_count_size = size;
    }
    out[seq_head] = ((ll_type as u8) << 6) | ((of_type as u8) << 4) | ((ml_type as u8) << 2);

    let bitstream_size = encode_sequences(
        out, ll_table, of_table, ml_table, ll_codes, of_codes, ml_codes, sequences, extra_bits,
    );
    // zstd versions <= 1.3.4 mistakenly report corruption when
    // FSE_readNCount() receives a buffer < 4 bytes: emit an uncompressed
    // block instead.
    if last_count_size != 0 && last_count_size + bitstream_size < 4 {
        debug_assert_eq!(last_count_size + bitstream_size, 3);
        return None;
    }

    Some(FseNext {
        ll: ll_next,
        of: of_next,
        ml: ml_next,
    })
}

/// Reusable buffers of [`estimate_sequences_section`], and the table a
/// candidate is built in.
#[derive(Default)]
pub struct EstimateScratch {
    codes: Vec<u8>,
    descriptions: Vec<u8>,
    table: FseCTable,
}

/// `ZSTD_buildBlockEntropyStats_sequences` followed by
/// `ZSTD_estimateBlockSize_sequences` with `writeEntropy`: the size the
/// post-sequence block splitter estimates for the sequences section of
/// `sequences` coded against `prev`, header and table descriptions
/// included. `None` where `ZSTD_buildSequencesStatistics` fails.
///
/// Per table the estimate is the bits `ZSTD_selectEncodingType` /
/// `ZSTD_buildCTable` would spend on the codes (the default distribution's
/// cross entropy for `Basic`, nothing for `Rle`, `ZSTD_fseBitCost` of the
/// table for `Compressed` and `Repeat`, or a flat `10` bytes per sequence
/// when that table cannot code a symbol) plus the codes' extra bits,
/// rounded down to bytes table by table.
pub fn estimate_sequences_section(
    sequences: &[Seq],
    prev: FseHeld<'_>,
    cparams: &CParams,
    scratch: &mut EstimateScratch,
) -> Option<usize> {
    let nb_seq = sequences.len();
    // seqHead + the smallest sequence count, whatever `nb_seq` is
    let header = 1 + 1 + (nb_seq >= 128) as usize + (nb_seq >= LONGNBSEQ) as usize;
    if nb_seq == 0 {
        // ZSTD_buildDummySequencesStatistics: every table Basic, no codes
        return Some(header);
    }
    scratch.codes.resize(3 * nb_seq, 0);
    let (ll_codes, of_codes, ml_codes) = seq_to_codes(sequences, &mut scratch.codes);
    scratch.descriptions.clear();
    let mut bytes = 0;
    /// One code stream and the tables `build_seq_table` chooses between.
    struct Stream<'a> {
        codes: &'a [u8],
        fse_log: u32,
        prev: Held<'a, FseCTable>,
        default_norm: &'a [i16],
        default_norm_log: u32,
        default_max: usize,
        /// The largest code the stream can hold.
        max_code: usize,
        /// Extra bits per code; `None` for offsets, whose code is the count.
        extra: Option<&'a [u8]>,
    }
    let streams = [
        Stream {
            codes: ll_codes,
            fse_log: LL_FSE_LOG,
            prev: prev.ll,
            default_norm: &LL_DEFAULT_NORM,
            default_norm_log: LL_DEFAULT_NORM_LOG,
            default_max: MAX_LL,
            max_code: MAX_LL,
            extra: Some(&LL_BITS),
        },
        Stream {
            codes: of_codes,
            fse_log: OFF_FSE_LOG,
            prev: prev.of,
            default_norm: &OF_DEFAULT_NORM,
            default_norm_log: OF_DEFAULT_NORM_LOG,
            default_max: DEFAULT_MAX_OFF,
            max_code: MAX_OFF,
            extra: None,
        },
        Stream {
            codes: ml_codes,
            fse_log: ML_FSE_LOG,
            prev: prev.ml,
            default_norm: &ML_DEFAULT_NORM,
            default_norm_log: ML_DEFAULT_NORM_LOG,
            default_max: MAX_ML,
            max_code: MAX_ML,
            extra: Some(&ML_BITS),
        },
    ];
    for Stream {
        codes,
        fse_log,
        prev,
        default_norm,
        default_norm_log,
        default_max,
        max_code,
        extra,
    } in streams
    {
        let mut counts = [0u32; 256];
        let (most, max) = huf::hist_count(&mut counts, codes, max_code);
        // `build_seq_table` lowers the last code's count for a new table.
        let mut table_counts = counts;
        let (table, _, ty, _) = build_seq_table(
            &mut scratch.descriptions,
            codes,
            &mut table_counts,
            max,
            most as usize,
            fse_log,
            TableRef {
                held: prev,
                spare: &mut scratch.table,
            },
            default_norm,
            default_norm_log,
            default_max,
            cparams.strategy,
        )?;
        // ZSTD_estimateBlockSize_symbolType
        let symbol_bits = match ty {
            SymbolEncodingType::Basic => Some(cross_entropy_cost(
                default_norm,
                default_norm_log,
                &counts,
                max,
            )),
            SymbolEncodingType::Rle => Some(0),
            SymbolEncodingType::Compressed | SymbolEncodingType::Repeat => {
                table.bit_cost(&counts, max)
            }
        };
        let Some(symbol_bits) = symbol_bits else {
            bytes += nb_seq * 10;
            continue;
        };
        // For offsets the code is also the number of extra bits.
        let extra_bits: u64 = counts[..=max]
            .iter()
            .enumerate()
            .map(|(c, &n)| n as u64 * extra.map_or(c as u64, |bits| bits[c] as u64))
            .sum();
        bytes += ((symbol_bits + extra_bits) >> 3) as usize;
    }
    // fseTablesSize: every description ZSTD_buildCTable wrote
    Some(bytes + scratch.descriptions.len() + header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::entropy::FseSlots;

    fn built(norm: &[i16], max_symbol: usize, table_log: u32) -> FseCTable {
        let mut table = FseCTable::default();
        table.build(norm, max_symbol, table_log);
        table
    }

    /// The table's 64 states are `64..128`, each once.
    fn assert_states_of_log_6(table: &FseCTable) {
        assert_eq!(table.table_log, 6);
        let mut states = table.state_table[..64].to_vec();
        states.sort_unstable();
        assert!(states.iter().copied().eq(64..128));
    }

    #[test]
    fn build_ll_default_table() {
        assert_states_of_log_6(&built(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG));
    }

    #[test]
    fn build_ml_default_table() {
        assert_states_of_log_6(&built(&ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG));
    }

    /// A table rebuilt in place over a larger one equals a fresh build:
    /// nothing of the old table survives in the states or transforms the
    /// new one uses, and every code past its `max_symbol` gets the
    /// zero-probability transform.
    #[test]
    fn rebuild_in_place_matches_fresh_build() {
        let mut table = built(&ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);
        table.build(&[16, 16], 1, 5);
        let fresh = built(&[16, 16], 1, 5);
        assert_eq!(table.state_table[..32], fresh.state_table[..32]);
        assert_eq!(table.max_symbol, 1);
        for (s, (a, b)) in table.symbol_tt.iter().zip(&fresh.symbol_tt).enumerate() {
            assert_eq!(
                (a.delta_find_state, a.delta_nb_bits),
                (b.delta_find_state, b.delta_nb_bits)
            );
            if s > 1 {
                assert_eq!((a.delta_find_state, a.delta_nb_bits), (0, (6 << 16) - 32));
            }
        }
        table.build_rle(7);
        assert_eq!((table.table_log, table.max_symbol), (0, 7));
        assert!(table
            .symbol_tt
            .iter()
            .all(|tt| (tt.delta_find_state, tt.delta_nb_bits) == (0, 0)));
    }

    fn norm_sum(norm: &[i16]) -> u32 {
        norm.iter().map(|&n| n.unsigned_abs() as u32).sum()
    }

    /// The `total` of `counts`. `FSE_normalizeCount` casts it to `U32`, so a
    /// total past `u32::MAX` is no histogram on any target; panic rather
    /// than wrap in a 32-bit `usize`.
    fn count_total(counts: &[u32]) -> usize {
        let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        u32::try_from(total).unwrap_or_else(|_| panic!("count total {total} does not fit u32"))
            as usize
    }

    /// `normalize_count` on `counts` at every table log from
    /// `FSE_minTableLog` up to `max_log`: a success must sum to
    /// `1 << table_log`, and the `FSE_optimalTableLog` choice must succeed.
    fn check_normalize(counts: &[u32], max_log: u32) {
        let max_symbol = counts.len() - 1;
        let total = count_total(counts);
        let optimal = optimal_table_log(max_log, total, max_symbol);
        for use_low_prob in [false, true] {
            for table_log in FSE_MIN_TABLELOG..=max_log {
                let mut norm = vec![0i16; max_symbol + 1];
                let r = normalize_count(
                    &mut norm,
                    table_log,
                    counts,
                    total,
                    max_symbol,
                    use_low_prob,
                );
                match r {
                    Ok(0) => panic!("RLE result for a multi-symbol input {counts:?}"),
                    Ok(log) => {
                        assert_eq!(log, table_log);
                        assert_eq!(
                            norm_sum(&norm),
                            1u32 << table_log,
                            "table_log {table_log} low_prob {use_low_prob} counts {counts:?} norm {norm:?}"
                        );
                        for s in 0..=max_symbol {
                            assert_eq!(counts[s] == 0, norm[s] == 0, "symbol {s} of {counts:?}");
                        }
                    }
                    Err(NormalizeError) => assert_ne!(
                        table_log, optimal,
                        "optimal table log {optimal} failed for {counts:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn normalize_dominant_symbol_plus_rare() {
        // One symbol carries almost everything; the rest are rare enough to
        // hit the lowThreshold / lowOne paths and drive the M2 fallback.
        for &(dominant, rare_symbols, rare_count) in &[
            (100_000u32, 40usize, 1u32),
            (100_000, 40, 3),
            (50_000, 200, 1),
            (20_000, 250, 2),
            (4_000, 60, 1),
            (1_000, 30, 1),
            (600, 52, 1),
            (60, 30, 1),
            (3_000, 100, 7),
        ] {
            let mut counts = vec![rare_count; rare_symbols + 1];
            counts[0] = dominant;
            check_normalize(&counts, 9);
            counts.reverse();
            check_normalize(&counts, 9);
            // rare symbols with a geometric tail: rare_count doubling every 8
            // symbols, shifted down as a whole (the low end saturating at 1)
            // so that no tail symbol exceeds half the dominant
            let levels = (rare_symbols as u32 - 1) / 8;
            let top = (u64::from(rare_count) << levels).min(u64::from(dominant / 2)) as u32;
            let mut geometric: Vec<u32> = (0..rare_symbols as u32)
                .map(|i| (top >> (levels - i / 8)).max(1))
                .collect();
            geometric.insert(0, dominant);
            check_normalize(&geometric, 9);
        }
    }

    #[test]
    fn normalize_rle_input_returns_zero() {
        let counts = [0u32, 17, 0];
        let mut norm = [7i16; 3];
        assert_eq!(normalize_count(&mut norm, 6, &counts, 17, 2, false), Ok(0));
    }

    #[test]
    fn normalize_rejects_too_small_table_log() {
        let counts = [3u32; 70];
        let mut norm = [0i16; 70];
        // FSE_minTableLog(210, 69) = min(8 + 1, 6 + 2) = 8
        assert_eq!(
            normalize_count(&mut norm, 7, &counts, 210, 69, false),
            Err(NormalizeError)
        );
        assert_eq!(
            normalize_count(&mut norm, 8, &counts, 210, 69, false),
            Ok(8)
        );
        assert_eq!(norm_sum(&norm), 256);
    }

    #[test]
    fn optimal_table_log_matches_c() {
        // FSE_optimalTableLog(9, 586, 18): maxBitsSrc = highbit(585) - 2 = 7,
        // minBits = min(highbit(586) + 1, highbit(18) + 2) = 6 -> 7
        assert_eq!(optimal_table_log(9, 586, 18), 7);
        // srcSize 2: highbit(1) - 2 wraps, tableLog stays at the maximum
        assert_eq!(optimal_table_log(6, 2, 1), 6);
        // clamp to FSE_MIN_TABLELOG
        assert_eq!(optimal_table_log(9, 9, 3), 5);
        // HUF weights: minus = 1
        assert_eq!(optimal_table_log_internal(6, 100, 12, 1), 5);
        assert_eq!(optimal_table_log_internal(6, 255, 12, 1), 6);
        // capped by maxTableLog
        assert_eq!(optimal_table_log(8, 100_000, 31), 8);
    }

    #[test]
    fn normalize_fixture_count_vectors() {
        let text = include_str!("../tests/data/minfail_counts.txt");
        let mut rows = 0;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let fields: Vec<&str> = line.split(' ').collect();
            let recorded_log: u32 = fields[2].parse().unwrap();
            let counts: Vec<u32> = fields[5].split(',').map(|c| c.parse().unwrap()).collect();
            let total = count_total(&counts);
            assert_eq!(total, fields[3].parse::<usize>().unwrap());
            // The log the pre-port code chose must now either be exact or be
            // refused; the offset (8) and LL/ML (9) maxima must both work.
            let max_symbol = counts.len() - 1;
            let mut norm = vec![0i16; max_symbol + 1];
            if normalize_count(&mut norm, recorded_log, &counts, total, max_symbol, false).is_ok() {
                assert_eq!(norm_sum(&norm), 1u32 << recorded_log, "{line}");
            }
            check_normalize(&counts, 8);
            check_normalize(&counts, 9);
            rows += 1;
        }
        assert_eq!(rows, 46);
    }

    /// Sequences whose LL codes are skewed (so the cost path prefers a
    /// custom table) with rep-code offsets and short matches.
    fn skewed_seqs(nb_seq: usize) -> Vec<Seq> {
        (0..nb_seq)
            .map(|i| Seq {
                lit_len: [0, 0, 0, 1, 2, 5, 12, 40][i * 7 % 8],
                off_base: 1 + (i % 3) as u32,
                ml_base: (i % 5) as u32,
            })
            .collect()
    }

    /// A `CParams` whose only field the sequences encoder reads is `strategy`.
    fn cparams(strategy: Strategy) -> CParams {
        CParams {
            window_log: 19,
            chain_log: 12,
            hash_log: 12,
            search_log: 1,
            min_match: 4,
            target_length: 0,
            strategy,
        }
    }

    /// Encoding types from the Sequences_Section_Header written by
    /// `encode_sequences_section_with` for `seqs` against `tables`, which
    /// then commit the section (`nb_seq >= 128` and `< LONGNBSEQ`
    /// assumed, so the count takes 2 bytes).
    fn section_types(seqs: &[Seq], tables: &mut FseSlots, strategy: Strategy) -> (u8, u8, u8) {
        let mut out = Vec::new();
        let next = encode_sequences_section_with(
            &mut out,
            seqs,
            &mut Vec::new(),
            tables.split(),
            &cparams(strategy),
        )
        .unwrap();
        tables.commit(next);
        assert!((128..LONGNBSEQ).contains(&seqs.len()));
        let head = out[2];
        (head >> 6, (head >> 4) & 3, (head >> 2) & 3)
    }

    fn repeats(tables: &FseSlots) -> (Repeat, Repeat, Repeat) {
        let held = tables.held();
        (held.ll.repeat(), held.of.repeat(), held.ml.repeat())
    }

    #[test]
    fn cost_path_repeats_previous_custom_table() {
        // 200 sequences: the NCount header outweighs the difference between
        // the interpolated FSE cost of the previous table and the entropy
        // bound, so Repeat wins (at 3000 the fresh table wins again, as in
        // libzstd).
        let seqs = skewed_seqs(200);
        let mut tables = FseSlots::default();
        let types = section_types(&seqs, &mut tables, Strategy::Lazy2);
        assert_eq!(types, (2, 2, 2), "first block: custom tables");
        let check = (Repeat::Check, Repeat::Check, Repeat::Check);
        assert_eq!(repeats(&tables), check);
        let first_ll = tables.held().ll.table().unwrap().state_table;
        let mut wider_tables = tables.clone();
        let types = section_types(&seqs, &mut tables, Strategy::Lazy2);
        assert_eq!(types, (3, 3, 3), "second block: repeat");
        assert_eq!(repeats(&tables), check);
        assert_eq!(tables.held().ll.table().unwrap().state_table, first_ll);
        // A symbol the previous table cannot encode rules Repeat out.
        let mut wider = seqs.clone();
        wider[10].lit_len = 70_000;
        let (ll, _, _) = section_types(&wider, &mut wider_tables, Strategy::Lazy2);
        assert_eq!(ll, 2);
    }

    #[test]
    fn heuristic_path_never_repeats_check_tables() {
        let seqs = skewed_seqs(3000);
        let mut tables = FseSlots::default();
        let (ll, _, _) = section_types(&seqs, &mut tables, Strategy::Fast);
        assert_eq!(ll, 2);
        assert_eq!(tables.held().ll.repeat(), Repeat::Check);
        let (ll, _, _) = section_types(&seqs, &mut tables.clone(), Strategy::Fast);
        assert_eq!(ll, 2);
        let (ll, _, _) = section_types(&seqs, &mut tables.clone(), Strategy::Greedy);
        assert_eq!(ll, 2);
    }

    #[test]
    fn heuristic_path_uses_basic_below_dynamic_fse_nb_seq_min() {
        // dynamicFse_nbSeq_min for LL with Fast: (64 * 9) >> 3 = 72
        let seqs = skewed_seqs(71);
        let mut out = Vec::new();
        encode_sequences_section_with(
            &mut out,
            &seqs,
            &mut Vec::new(),
            FseSlots::default().split(),
            &cparams(Strategy::Fast),
        )
        .unwrap();
        assert_eq!(out[1] >> 6, 0);
        let seqs = skewed_seqs(72);
        let mut out = Vec::new();
        encode_sequences_section_with(
            &mut out,
            &seqs,
            &mut Vec::new(),
            FseSlots::default().split(),
            &cparams(Strategy::Fast),
        )
        .unwrap();
        assert_eq!(out[1] >> 6, 2);
    }

    #[test]
    fn single_symbol_is_rle_unless_two_or_fewer() {
        let seq = Seq {
            lit_len: 3,
            off_base: 4,
            ml_base: 0,
        };
        for (nb_seq, expected) in [(1, 0u8), (2, 0), (3, 1)] {
            let seqs = vec![seq; nb_seq];
            let mut out = Vec::new();
            let next = encode_sequences_section_with(
                &mut out,
                &seqs,
                &mut Vec::new(),
                FseSlots::default().split(),
                &cparams(Strategy::Lazy),
            )
            .unwrap();
            assert_eq!(out[1] >> 6, expected, "nb_seq {nb_seq}");
            assert_eq!(next.ll, Next::None);
        }
    }

    #[test]
    fn write_ncount_matches_c_layout() {
        // FSE_writeNCount of the three default distributions, bytes taken
        // from libzstd 1.5.7.
        fn hex(bytes: &[u8]) -> String {
            bytes.iter().map(|b| format!("{b:02x}")).collect()
        }
        let mut out = vec![0xAA];
        let n = write_ncount(&mut out, &LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG).unwrap();
        assert_eq!(n, out.len() - 1);
        assert_eq!(hex(&out[1..]), "5110638c31c618630c21c4186366668646920400");
        let mut out = Vec::new();
        write_ncount(&mut out, &ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG).unwrap();
        assert_eq!(
            hex(&out),
            "2114c418638c2184104208218410420821444444444444444424090000"
        );
        let mut out = Vec::new();
        write_ncount(
            &mut out,
            &OF_DEFAULT_NORM,
            OF_DEFAULT_NORM.len() - 1,
            OF_DEFAULT_NORM_LOG,
        )
        .unwrap();
        assert_eq!(hex(&out), "2084104266464444444424490200");
        // A distribution that does not sum to the table size is refused and
        // leaves `out` untouched.
        let mut out = vec![1, 2];
        assert_eq!(write_ncount(&mut out, &[3, 3], 1, 5), Err(NormalizeError));
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn init_state_in_range() {
        let table = built(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        // Encoder states are stored in the [table_size, 2 * table_size) range.
        let state = table.init_state(0);
        let table_size = 1u32 << table.table_log;
        assert!((table_size..(table_size * 2)).contains(&state));
    }
}
