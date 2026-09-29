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

/// Normalize symbol counts to probability distribution for FSE table.
/// Port of C zstd's FSE_normalizeCount() with 62-bit precision scaling.
pub fn normalize_counts(counts: &[u32], max_symbol: usize, table_log: u32) -> Vec<i16> {
    let table_size = 1u32 << table_log;
    let total: u64 = counts[..=max_symbol].iter().map(|&c| c as u64).sum();
    if total == 0 {
        return vec![0i16; max_symbol + 1];
    }

    let mut norm = vec![0i16; max_symbol + 1];

    // Use C zstd's high-precision scaling: step = (1<<62) / total
    let scale: u32 = 62 - table_log;
    let step: u64 = (1u64 << 62) / total;
    let v_step: u64 = 1u64 << (scale - 20);
    let low_threshold: u64 = total >> table_log;

    // C zstd's rtbTable for precise rounding of small probabilities
    static RTB_TABLE: [u32; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];

    // Use lowProbCount = -1 for large blocks (>= 2048 sequences), 1 otherwise
    let use_low_prob_count = total >= 2048;
    let low_prob_count: i16 = if use_low_prob_count { -1 } else { 1 };

    let mut still_to_distribute = table_size as i32;
    let mut largest_sym = 0usize;
    let mut largest_prob = 0i16;

    for s in 0..=max_symbol {
        if counts[s] as u64 == total {
            // Single-symbol dominance
            norm[s] = table_size as i16;
            return norm;
        }
        if counts[s] == 0 {
            continue;
        }

        if (counts[s] as u64) <= low_threshold {
            norm[s] = low_prob_count;
            still_to_distribute -= 1;
        } else {
            let mut proba = ((counts[s] as u64 * step) >> scale) as i16;
            if proba < 8 {
                // Use rtbTable for precise rounding
                let rest_to_beat = v_step as u128 * RTB_TABLE[proba as usize] as u128;
                let actual = (counts[s] as u128 * step as u128) - ((proba as u128) << scale);
                if actual > rest_to_beat {
                    proba += 1;
                }
            }
            if proba > (table_size >> 1) as i16 {
                proba = (table_size >> 1) as i16; // cap at half table
            }
            norm[s] = std::cmp::max(1, proba);
            still_to_distribute -= norm[s] as i32;
        }

        if norm[s] > largest_prob {
            largest_prob = norm[s];
            largest_sym = s;
        }
    }

    // Adjust largest symbol to distribute remaining
    if -still_to_distribute >= (norm[largest_sym] >> 1) as i32 {
        // Pathological case: use proportional redistribution
        normalize_counts_m2(&mut norm, counts, max_symbol, table_log, total);
    } else {
        norm[largest_sym] += still_to_distribute as i16;
    }

    norm
}

/// Fallback normalization for pathological distributions (port of FSE_normalizeM2).
pub fn normalize_counts_m2(
    norm: &mut [i16],
    counts: &[u32],
    max_symbol: usize,
    table_log: u32,
    total: u64,
) {
    let table_size = 1u32 << table_log;

    // Reset and recalculate
    let mut to_distribute = table_size as i32;

    // First pass: identify symbols that will get probability >= 1
    let low_one = (total * 3) / ((to_distribute as u64) * 2);
    for s in 0..=max_symbol {
        if counts[s] == 0 {
            norm[s] = 0;
        } else if (counts[s] as u64) <= low_one {
            norm[s] = -1;
            to_distribute -= 1;
        } else {
            norm[s] = 0; // will be set in second pass
        }
    }

    // Second pass: proportional scaling for remaining symbols
    let remaining_total: u64 = counts[..=max_symbol]
        .iter()
        .enumerate()
        .filter(|&(s, _)| norm[s] == 0 && counts[s] > 0)
        .map(|(_, &c)| c as u64)
        .sum();

    if remaining_total == 0 || to_distribute <= 0 {
        return;
    }

    let v_step_log = 62u32.saturating_sub(table_log);
    let r_step = ((1u128 << v_step_log) * to_distribute as u128 + remaining_total as u128 / 2)
        / remaining_total as u128;

    let mut tmp_total = 0u128;
    for s in 0..=max_symbol {
        if norm[s] == 0 && counts[s] > 0 {
            let end = tmp_total + counts[s] as u128 * r_step;
            let s_start = (tmp_total >> v_step_log) as i16;
            let s_end = (end >> v_step_log) as i16;
            let proba = s_end - s_start;
            norm[s] = std::cmp::max(1, proba);
            tmp_total = end;
        }
    }
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

    // Try custom FSE table
    // Choose table_log: use max_log for best compression, but cap by number of symbols
    let table_log = {
        let min_log = 5u32;
        let symbol_log = if n_used <= 2 {
            min_log
        } else {
            std::cmp::min(max_log, (32 - (n_used as u32).leading_zeros()).max(min_log))
        };
        std::cmp::min(max_log, std::cmp::max(min_log, symbol_log))
    };

    let custom_norm = normalize_counts(&counts, max_sym, table_log);

    // Verify all symbols are covered
    let all_covered = codes.iter().all(|&c| {
        let s = c as usize;
        s <= max_sym && custom_norm[s] != 0
    });

    // normalize_counts is known to yield distributions whose sum is not
    // 1 << table_log on real data; FseCTable::build cannot represent those.
    // Predefined always covers our codes: LL/ML defaults have no zero entry
    // and offset codes stay <= 27 while window_log <= 27.
    let norm_sum: i64 = custom_norm
        .iter()
        .map(|&n| if n == -1 { 1 } else { n as i64 })
        .sum();
    if !all_covered || norm_sum != (1i64 << table_log) {
        debug_assert!(predefined_ok);
        return SeqTableMode::Predefined;
    }

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

    #[test]
    fn init_state_in_range() {
        let table = FseCTable::build(&LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
        // Encoder states are stored in the [table_size, 2 * table_size) range.
        let state = table.init_state(0);
        let table_size = 1u32 << table.table_log;
        assert!((table_size..(table_size * 2)).contains(&state));
    }
}
