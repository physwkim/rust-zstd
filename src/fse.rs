//! FSE (Finite State Entropy) encoder.
//! Ported from zstd C source: lib/common/fse.h, lib/compress/fse_compress.c.

use super::bitstream::BackwardBitWriter;
use crate::compress::seqstore::Seq;
use crate::constants::*;

/// Per-symbol compression transform (matches C FSE_symbolCompressionTransform).
#[derive(Clone, Copy, Debug, Default)]
pub struct SymbolTT {
    pub delta_find_state: i32,
    pub delta_nb_bits: u32,
}

/// Compiled FSE compression table.
#[derive(Clone, Debug)]
pub struct FseCTable {
    pub table_log: u32,
    pub state_table: Vec<u16>,
    pub symbol_tt: Vec<SymbolTT>,
    pub max_symbol: usize,
}

impl FseCTable {
    /// Build an FSE compression table from normalized counts.
    /// Ported from FSE_buildCTable_wksp() in fse_compress.c.
    pub fn build(norm: &[i16], max_symbol: usize, table_log: u32) -> Self {
        let table_size = 1u32 << table_log;
        let table_mask = table_size - 1;

        // 1. Build cumulative counts and place low-probability symbols.
        let mut cumul = vec![0u16; max_symbol + 2];
        let mut high_threshold = table_size - 1;
        let mut table_symbol = vec![0u8; table_size as usize];

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
        let mut state_table = vec![0u16; table_size as usize];
        for u in 0..table_size {
            let s = table_symbol[u as usize] as usize;
            let idx = cumul[s] as usize;
            state_table[idx] = (table_size + u) as u16;
            cumul[s] += 1;
        }

        // 4. Build per-symbol compression transforms.
        // Use the decoder's baseline/numbits calculation for compatibility.
        // For each state in the table, compute its decoder-compatible numbits,
        // then derive the CTable's delta_nb_bits and delta_find_state from that.
        let mut symbol_tt = vec![SymbolTT::default(); max_symbol + 1];
        let _sym_count_tt = vec![0u32; max_symbol + 1];
        let mut total = 0u32;
        for s in 0..=max_symbol {
            let prob = if norm[s] == -1 {
                1
            } else {
                norm[s].max(0) as u32
            };
            if prob == 0 {
                symbol_tt[s].delta_nb_bits = ((table_log + 1) << 16) - table_size;
            } else if prob == 1 {
                symbol_tt[s].delta_nb_bits = (table_log << 16) - table_size;
                symbol_tt[s].delta_find_state = total as i32 - 1;
            } else {
                // Use the same formula as C zstd FSE_buildCTable
                let max_bits_out = table_log - highest_bit(prob - 1);
                let min_state_plus = prob << max_bits_out;
                symbol_tt[s].delta_nb_bits = (max_bits_out << 16).wrapping_sub(min_state_plus);
                symbol_tt[s].delta_find_state = total as i32 - prob as i32;
            }
            total += prob;
        }

        Self {
            table_log,
            state_table,
            symbol_tt,
            max_symbol,
        }
    }

    /// Build an RLE compression table: single symbol, 0 bits per encode.
    /// init_state returns 0, encode_symbol always returns (0, 0, 0).
    /// table_log = 0, matching the decoder's RLE behavior.
    pub fn build_rle(symbol: u8) -> Self {
        let s = symbol as usize;
        let max_symbol = s;
        // Handcraft a table where everything resolves to 0 bits, state=0.
        let state_table = vec![0u16; 1]; // state_table[0] = 0
        let mut symbol_tt = vec![SymbolTT::default(); max_symbol + 1];
        // We need encode_symbol to return (0, 0, 0):
        // nb_bits = (state + delta_nb_bits) >> 16
        //   For state=0: nb_bits = delta_nb_bits >> 16 = 0 (if delta_nb_bits < 65536)
        // bits_out = state & ((1 << 0) - 1) = state & 0 = 0
        // new_state = state_table[(state >> 0) + delta_find_state]
        //   state >> 0 = 0, delta_find_state = 0 → state_table[0] = 0
        symbol_tt[s] = SymbolTT {
            delta_find_state: 0,
            delta_nb_bits: 0,
        };
        Self {
            table_log: 0,
            state_table,
            symbol_tt,
            max_symbol,
        }
    }

    /// Initialize FSE state for the first symbol (FSE_initCState2).
    pub fn init_state(&self, symbol: usize) -> u32 {
        let stt = &self.symbol_tt[symbol];
        let nb_bits = ((stt.delta_nb_bits as u64 + (1 << 15)) >> 16) as u32;
        let base_val = (nb_bits << 16).wrapping_sub(stt.delta_nb_bits);
        self.state_table[((base_val >> nb_bits) as i32 + stt.delta_find_state) as usize] as u32
    }

    /// Encode a symbol: output bits from current state, then transition.
    /// Returns (bits_to_output, nb_bits, new_state).
    pub fn encode_symbol(&self, state: u32, symbol: usize) -> (u32, u32, u32) {
        let stt = &self.symbol_tt[symbol];
        let nb_bits = (state.wrapping_add(stt.delta_nb_bits)) >> 16;
        let bits_out = state & ((1 << nb_bits) - 1);
        let new_state =
            self.state_table[((state >> nb_bits) as i32 + stt.delta_find_state) as usize] as u32;
        (bits_out, nb_bits, new_state)
    }
}

fn highest_bit(v: u32) -> u32 {
    if v == 0 {
        return 0;
    }
    31 - v.leading_zeros()
}

#[allow(clippy::too_many_arguments)]
/// Encode sequences using predefined FSE tables.
/// Exact port of ZSTD_encodeSequences_body from zstd_compress_sequences.c.
pub fn encode_sequences(
    ll_table: &FseCTable,
    off_table: &FseCTable,
    ml_table: &FseCTable,
    ll_codes: &[u8],
    off_codes: &[u8],
    ml_codes: &[u8],
    ll_values: &[u32],  // literal length values (for extra bits)
    ml_values: &[u32],  // match length - MINMATCH values (for extra bits)
    off_values: &[u32], // offset values (for extra bits)
) -> Vec<u8> {
    let nb_seq = ll_codes.len();
    if nb_seq == 0 {
        return vec![];
    }

    let mut bw = BackwardBitWriter::new();

    // Initialize states from the last sequence (first in encoding order)
    let last = nb_seq - 1;
    let mut state_ll = ll_table.init_state(ll_codes[last] as usize);
    let mut state_off = off_table.init_state(off_codes[last] as usize);
    let mut state_ml = ml_table.init_state(ml_codes[last] as usize);

    // Encode extra bits for the last sequence
    let ll_bits_n = LL_BITS[ll_codes[last] as usize] as u32;
    bw.add_bits(ll_values[last] as u64, ll_bits_n);
    if ll_bits_n > 0 {
        bw.flush_bits();
    }

    let ml_bits_n = ML_BITS[ml_codes[last] as usize] as u32;
    bw.add_bits(ml_values[last] as u64, ml_bits_n);
    if ml_bits_n > 0 {
        bw.flush_bits();
    }

    let of_bits_n = off_codes[last] as u32;
    bw.add_bits(off_values[last] as u64, of_bits_n);
    bw.flush_bits();

    // Encode remaining sequences in reverse order
    if nb_seq >= 2 {
        for n in (0..last).rev() {
            let llc = ll_codes[n] as usize;
            let ofc = off_codes[n] as usize;
            let mlc = ml_codes[n] as usize;

            // FSE encode: OFF, ML, LL (order matters!)
            let (bits, nb, new_state) = off_table.encode_symbol(state_off, ofc);
            bw.add_bits(bits as u64, nb);
            state_off = new_state;

            let (bits, nb, new_state) = ml_table.encode_symbol(state_ml, mlc);
            bw.add_bits(bits as u64, nb);
            state_ml = new_state;

            let (bits, nb, new_state) = ll_table.encode_symbol(state_ll, llc);
            bw.add_bits(bits as u64, nb);
            state_ll = new_state;

            bw.flush_bits();

            // Extra bits: LL, ML, OFF
            let ll_eb = LL_BITS[llc] as u32;
            bw.add_bits(ll_values[n] as u64, ll_eb);

            let ml_eb = ML_BITS[mlc] as u32;
            bw.add_bits(ml_values[n] as u64, ml_eb);

            let of_eb = ofc as u32;
            bw.add_bits(off_values[n] as u64, of_eb);
            bw.flush_bits();
        }
    }

    // Flush final states
    bw.add_bits(state_ml as u64, ml_table.table_log);
    bw.flush_bits();
    bw.add_bits(state_off as u64, off_table.table_log);
    bw.flush_bits();
    bw.add_bits(state_ll as u64, ll_table.table_log);
    bw.flush_bits();

    bw.finish()
}

/// Encode sequences with cross-block Repeat mode support.
/// If the current block's symbol distribution matches the previous block, use Repeat mode
/// (no table header needed). Otherwise choose best of Predefined/RLE/Custom FSE.
/// `FSE_repeat` state of one sequence table held by the decoder. `None`
/// (`FSE_repeat_none`): no table. `Check` (`FSE_repeat_check`): a custom
/// table the next block may reference with `Repeat` if it covers its
/// symbols. `Valid` (`FSE_repeat_valid`): dictionaries only, never produced
/// here.
#[derive(Clone, Debug, Default)]
pub enum FseTableState {
    #[default]
    None,
    Check(FseCTable),
    Valid(FseCTable),
}

impl FseTableState {
    pub fn table(&self) -> Option<&FseCTable> {
        match self {
            FseTableState::None => None,
            FseTableState::Check(t) | FseTableState::Valid(t) => Some(t),
        }
    }

    /// State after writing a table in `mode`: a custom table becomes `Check`
    /// (`ZSTD_selectEncodingType` sets `FSE_repeat_check` for
    /// `set_compressed`); Predefined and RLE leave `None`.
    fn after(mode: &SeqTableMode, table: &FseCTable) -> Self {
        match mode {
            SeqTableMode::Fse { .. } => FseTableState::Check(table.clone()),
            SeqTableMode::Predefined | SeqTableMode::Rle(_) => FseTableState::None,
        }
    }
}

/// `ZSTD_fseCTables_t`: literal-length, offset and match-length tables.
#[derive(Clone, Debug, Default)]
pub struct FseState {
    pub ll: FseTableState,
    pub of: FseTableState,
    pub ml: FseTableState,
}

/// Write the sequences section (Sequences_Section_Header onward, as in
/// `ZSTD_entropyCompressSeqStore_internal`) and return the FSE state the
/// decoder holds afterwards (`nextEntropy->fse`).
///
/// With `nb_seq == 0` the tables carry over unchanged
/// (`nextEntropy->fse = prevEntropy->fse`). Mode selection is the
/// pre-existing Predefined / RLE / custom-FSE heuristic: `Repeat` is not
/// emitted yet, so `prev` is only carried, never referenced.
pub fn encode_sequences_section(out: &mut Vec<u8>, sequences: &[Seq], prev: &FseState) -> FseState {
    let nb_seq = sequences.len();

    // Number of sequences header
    if nb_seq < 128 {
        out.push(nb_seq as u8);
    } else if nb_seq < 0x7F00 {
        out.push(((nb_seq >> 8) as u8) + 128);
        out.push(nb_seq as u8);
    } else {
        out.push(255);
        out.extend_from_slice(&((nb_seq - 0x7F00) as u16).to_le_bytes());
    }

    if nb_seq == 0 {
        return prev.clone();
    }

    // Convert sequences to codes + extra bit values
    let mut ll_codes_v = Vec::with_capacity(nb_seq);
    let mut ml_codes_v = Vec::with_capacity(nb_seq);
    let mut off_codes_v = Vec::with_capacity(nb_seq);
    let mut ll_values = Vec::with_capacity(nb_seq);
    let mut ml_values = Vec::with_capacity(nb_seq);
    let mut off_values = Vec::with_capacity(nb_seq);

    for seq in sequences {
        let llc = ll_code(seq.lit_len);
        let mlc = ml_code(seq.ml_base);
        let ofc = off_code(seq.off_base);

        ll_codes_v.push(llc);
        ml_codes_v.push(mlc);
        off_codes_v.push(ofc);
        ll_values.push(seq.lit_len - LL_BASE[llc as usize]);
        ml_values.push(seq.match_len() - ML_BASE[mlc as usize]);
        off_values.push(if ofc > 0 {
            seq.off_base - (1u32 << ofc)
        } else {
            0
        });
    }

    // Choose best mode for each table: Predefined vs RLE vs Custom FSE
    let ll_mode = choose_seq_mode(
        &ll_codes_v,
        MAX_LL,
        LL_DEFAULT_NORM_LOG,
        &LL_DEFAULT_NORM,
        LL_FSE_LOG,
    );
    let of_mode = choose_seq_mode(
        &off_codes_v,
        OF_DEFAULT_NORM.len() - 1,
        OF_DEFAULT_NORM_LOG,
        &OF_DEFAULT_NORM,
        OFF_FSE_LOG,
    );
    let ml_mode = choose_seq_mode(
        &ml_codes_v,
        MAX_ML,
        ML_DEFAULT_NORM_LOG,
        &ML_DEFAULT_NORM,
        ML_FSE_LOG,
    );

    // Write compression modes byte
    let mode_byte = (ll_mode.tag() << 6) | (of_mode.tag() << 4) | (ml_mode.tag() << 2);
    out.push(mode_byte);

    // Write table descriptions for non-predefined modes, then build tables
    let ll_table =
        write_seq_table_and_build(out, &ll_mode, &LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
    let of_table = write_seq_table_and_build(
        out,
        &of_mode,
        &OF_DEFAULT_NORM,
        OF_DEFAULT_NORM.len() - 1,
        OF_DEFAULT_NORM_LOG,
    );
    let ml_table =
        write_seq_table_and_build(out, &ml_mode, &ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);

    let next = FseState {
        ll: FseTableState::after(&ll_mode, &ll_table),
        of: FseTableState::after(&of_mode, &of_table),
        ml: FseTableState::after(&ml_mode, &ml_table),
    };

    // Encode with FSE sequence encoder
    let bitstream = encode_sequences(
        &ll_table,
        &of_table,
        &ml_table,
        &ll_codes_v,
        &off_codes_v,
        &ml_codes_v,
        &ll_values,
        &ml_values,
        &off_values,
    );
    out.extend_from_slice(&bitstream);
    next
}
// =========================================================================
// Custom FSE table mode selection for sequences
// =========================================================================

/// Chosen compression mode for a sequence table.
pub enum SeqTableMode {
    Predefined,
    Rle(u8),
    Fse {
        norm: Vec<i16>,
        max_symbol: usize,
        table_log: u32,
        header_bytes: Vec<u8>,
    },
}

impl SeqTableMode {
    pub fn tag(&self) -> u8 {
        match self {
            SeqTableMode::Predefined => SEQ_MODE_PREDEFINED,
            SeqTableMode::Rle(_) => SEQ_MODE_RLE,
            SeqTableMode::Fse { .. } => SEQ_MODE_FSE,
        }
    }
}

/// `FSE_MIN_TABLELOG`.
pub const FSE_MIN_TABLELOG: u32 = 5;
/// `FSE_MAX_TABLELOG`.
pub const FSE_MAX_TABLELOG: u32 = 12;
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

/// Encode an FSE probability header (the variable-bit format from the spec).
/// Returns the serialized header bytes.
pub fn encode_fse_header(norm: &[i16], max_symbol: usize, table_log: u32) -> Vec<u8> {
    let table_size = 1u32 << table_log;
    let mut bb: u64 = (table_log - 5) as u64; // accuracy_log = 5 + low4bits
    let mut bp = 4u32;
    let mut out = Vec::with_capacity(32);
    let mut counter = 0u32;

    let mut s = 0usize;
    while s <= max_symbol && counter < table_size {
        let prob = norm[s] as i32;
        let value = (prob + 1) as u32;

        let max_remaining = table_size - counter + 1;
        let bits_to_read = 32 - max_remaining.leading_zeros();
        let low_threshold = ((1u32 << bits_to_read) - 1) - max_remaining;
        let mask = (1u32 << (bits_to_read - 1)) - 1;

        if value < low_threshold {
            bb |= (value as u64) << bp;
            bp += bits_to_read - 1;
        } else if value <= mask {
            bb |= (value as u64) << bp;
            bp += bits_to_read;
        } else {
            let encoded = value + low_threshold;
            bb |= (encoded as u64) << bp;
            bp += bits_to_read;
        }

        while bp >= 8 {
            out.push(bb as u8);
            bb >>= 8;
            bp -= 8;
        }

        if prob > 0 {
            counter += prob as u32;
        } else if prob == -1 {
            counter += 1;
        }

        // Handle zero-probability repeat flags
        if prob == 0 {
            // Count consecutive zeros after this one
            let mut repeat = 0u32;
            while s + 1 + repeat as usize <= max_symbol
                && norm[s + 1 + repeat as usize] == 0
                && repeat < 3
            {
                repeat += 1;
            }
            bb |= (repeat as u64) << bp;
            bp += 2;
            while bp >= 8 {
                out.push(bb as u8);
                bb >>= 8;
                bp -= 8;
            }
            s += repeat as usize; // skip the zeros we just flagged

            // If repeat == 3, keep emitting 2-bit repeat flags
            while repeat == 3 {
                repeat = 0;
                while s + 1 + repeat as usize <= max_symbol
                    && norm[s + 1 + repeat as usize] == 0
                    && repeat < 3
                {
                    repeat += 1;
                }
                bb |= (repeat as u64) << bp;
                bp += 2;
                while bp >= 8 {
                    out.push(bb as u8);
                    bb >>= 8;
                    bp -= 8;
                }
                s += repeat as usize;
            }
        }

        s += 1;
    }

    if bp > 0 {
        out.push(bb as u8);
    }

    out
}

/// Estimate the compressed size (in bits) of encoding `codes` with a given normalized distribution.
/// Cross-entropy cost of encoding `counts` using distribution `norm` at `table_log`.
/// Returns approximate total bits needed to encode all symbols.
pub fn cross_entropy_cost(
    norm: &[i16],
    table_log: u32,
    counts: &[u32; 256],
    max_sym: usize,
) -> u64 {
    let mut cost = 0u64;
    for s in 0..=max_sym {
        if counts[s] == 0 {
            continue;
        }
        if s >= norm.len() || norm[s] == 0 {
            return u64::MAX;
        }
        let prob = if norm[s] == -1 { 1u64 } else { norm[s] as u64 };
        // bits per symbol ≈ table_log - floor(log2(prob))
        let log2_prob = 63 - prob.leading_zeros() as u64;
        cost += counts[s] as u64 * (table_log as u64 - log2_prob);
    }
    cost + table_log as u64 // add state init cost
}

/// Choose best mode considering Repeat from previous block.
pub fn choose_seq_mode(
    codes: &[u8],
    max_symbol_default: usize,
    default_log: u32,
    default_norm: &[i16],
    max_log: u32,
) -> SeqTableMode {
    if codes.is_empty() {
        return SeqTableMode::Predefined;
    }

    // Count symbol frequencies
    let mut counts = [0u32; 256];
    let mut max_sym = 0usize;
    for &c in codes {
        counts[c as usize] += 1;
        if c as usize > max_sym {
            max_sym = c as usize;
        }
    }

    let n_used = counts[..=max_sym].iter().filter(|&&c| c > 0).count();

    // RLE: only one distinct symbol
    if n_used == 1 {
        let sym = codes[0];
        return SeqTableMode::Rle(sym);
    }

    // Check if predefined table can represent all our symbols
    let predefined_ok = max_sym <= max_symbol_default
        && codes.iter().all(|&c| {
            let s = c as usize;
            s < default_norm.len() && default_norm[s] != 0
        });

    // Custom FSE table: `ZSTD_buildCTable` picks `FSE_optimalTableLog` and
    // normalizes with `ZSTD_useLowProbCount(nbSeq)`. A `GENERIC` failure
    // falls back to Predefined, which always covers our codes: LL/ML
    // defaults have no zero entry and offset codes stay <= 27 while
    // window_log <= 27.
    let nb_seq = codes.len();
    let table_log = optimal_table_log(max_log, nb_seq, max_sym);
    let mut custom_norm = vec![0i16; max_sym + 1];
    if normalize_count(
        &mut custom_norm,
        table_log,
        &counts,
        nb_seq,
        max_sym,
        nb_seq >= 2048,
    )
    .is_err()
    {
        debug_assert!(predefined_ok);
        return SeqTableMode::Predefined;
    }
    debug_assert_eq!(
        custom_norm
            .iter()
            .map(|&n| n.unsigned_abs() as u32)
            .sum::<u32>(),
        1u32 << table_log
    );

    let header_bytes = encode_fse_header(&custom_norm, max_sym, table_log);

    // Bit-cost comparison (port of C zstd's ZSTD_selectEncodingType approach)
    let _nb_seq = codes.len();

    // Cross-entropy cost for predefined table: sum of log2(tableSize/prob) per symbol
    let predefined_cost = if predefined_ok {
        cross_entropy_cost(default_norm, default_log, &counts, max_sym)
    } else {
        u64::MAX
    };

    // Custom FSE cost: header bytes + cross-entropy with custom table
    let _custom_table_size = 1u64 << table_log;
    let mut custom_stream_cost = 0u64;
    for s in 0..=max_sym {
        if counts[s] > 0 {
            let prob = if custom_norm[s] == -1 {
                1u64
            } else {
                custom_norm[s] as u64
            };
            if prob == 0 {
                custom_stream_cost = u64::MAX;
                break;
            }
            // Cost in 256ths of a bit: count * log2(tableSize/prob) * 256
            // log2(tableSize/prob) = table_log - log2(prob)
            let log2_prob = 63 - prob.leading_zeros() as u64;
            custom_stream_cost += counts[s] as u64 * (table_log as u64 - log2_prob);
        }
    }
    let custom_header_cost = header_bytes.len() as u64 * 8;
    let custom_total_cost = custom_header_cost + custom_stream_cost + table_log as u64;

    if predefined_ok && predefined_cost <= custom_total_cost {
        SeqTableMode::Predefined
    } else {
        SeqTableMode::Fse {
            norm: custom_norm,
            max_symbol: max_sym,
            table_log,
            header_bytes,
        }
    }
}

/// Write the table description to `out` and return the built FSE compression table.
pub fn write_seq_table_and_build(
    out: &mut Vec<u8>,
    mode: &SeqTableMode,
    default_norm: &[i16],
    default_max_symbol: usize,
    default_log: u32,
) -> FseCTable {
    match mode {
        SeqTableMode::Predefined => FseCTable::build(default_norm, default_max_symbol, default_log),
        SeqTableMode::Rle(sym) => {
            out.push(*sym);
            FseCTable::build_rle(*sym)
        }
        SeqTableMode::Fse {
            norm,
            max_symbol,
            table_log,
            header_bytes,
        } => {
            out.extend_from_slice(header_bytes);
            FseCTable::build(norm, *max_symbol, *table_log)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ll_default_table() {
        let table = FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        assert_eq!(table.table_log, 6);
        assert_eq!(table.state_table.len(), 64);
    }

    #[test]
    fn build_ml_default_table() {
        let table = FseCTable::build(&ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);
        assert_eq!(table.table_log, 6);
        assert_eq!(table.state_table.len(), 64);
    }

    fn norm_sum(norm: &[i16]) -> u32 {
        norm.iter().map(|&n| n.unsigned_abs() as u32).sum()
    }

    /// `normalize_count` on `counts` at every table log from
    /// `FSE_minTableLog` up to `max_log`: a success must sum to
    /// `1 << table_log`, and the `FSE_optimalTableLog` choice must succeed.
    fn check_normalize(counts: &[u32], max_log: u32) {
        let max_symbol = counts.len() - 1;
        let total: usize = counts.iter().map(|&c| c as usize).sum();
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
            // rare symbols with a geometric tail
            let mut geometric: Vec<u32> = (0..rare_symbols as u32)
                .map(|i| (rare_count << (i / 8)).max(1))
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
            let total: usize = counts.iter().map(|&c| c as usize).sum();
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

    #[test]
    fn init_state_in_range() {
        let table = FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        // Encoder states are stored in the [table_size, 2 * table_size) range.
        let state = table.init_state(0);
        let table_size = 1u32 << table.table_log;
        assert!((table_size..(table_size * 2)).contains(&state));
    }
}
