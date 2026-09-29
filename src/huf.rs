//! Huffman literals encoder.
//!
//! Ported from zstd C source: lib/compress/huf_compress.c and
//! lib/compress/zstd_compress_literals.c. Moved out of compress.rs unchanged
//! except for the block-state plumbing at the bottom of this file.

use crate::constants::*;

/// A built Huffman table: per-symbol `(code, nb_bits)` and the longest code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HufTable {
    pub codes: [(u32, u8); 256],
    pub max_bits: u8,
}

/// `ZSTD_hufCTables_t`: the Huffman table the decoder currently holds and its
/// `HUF_repeat` mode. `None` (`HUF_repeat_none`): no table. `Check`
/// (`HUF_repeat_check`): the table may be reused if it covers every symbol.
/// `Valid` (`HUF_repeat_valid`): known to cover the symbols (dictionaries
/// only; never produced here).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum HufState {
    #[default]
    None,
    Check(HufTable),
    Valid(HufTable),
}

impl HufState {
    pub fn table(&self) -> Option<&HufTable> {
        match self {
            HufState::None => None,
            HufState::Check(t) | HufState::Valid(t) => Some(t),
        }
    }
}

/// Encode literals using a previous Huffman tree (Treeless_Literals_Block, type=3).
pub fn encode_literals_treeless(literals: &[u8], prev_codes: &[(u32, u8); 256]) -> Option<Vec<u8>> {
    let use_4 = literals.len() >= 1024;
    let streams = if use_4 {
        encode_huf_4streams(literals, prev_codes)
    } else {
        encode_huf_1stream(literals, prev_codes)
    };
    let regen = literals.len();
    let comp = streams.len();
    if comp >= regen {
        return None;
    }
    let lh_size = 3 + (regen >= 1024) as usize + (regen >= 16384) as usize;
    let mut out = Vec::with_capacity(lh_size + comp);
    let htype = LIT_TYPE_TREELESS as u32;
    match lh_size {
        3 => {
            let sf = if use_4 { 1u32 } else { 0 };
            out.extend_from_slice(
                &(htype | (sf << 2) | ((regen as u32) << 4) | ((comp as u32) << 14)).to_le_bytes()
                    [..3],
            );
        }
        4 => out.extend_from_slice(
            &(htype | (2u32 << 2) | ((regen as u32) << 4) | ((comp as u32) << 18)).to_le_bytes()
                [..4],
        ),
        _ => {
            let v = htype | (3u32 << 2) | ((regen as u32) << 4) | ((comp as u32) << 22);
            out.extend_from_slice(&v.to_le_bytes()[..4]);
            out.push((comp >> 10) as u8);
        }
    }
    out.extend_from_slice(&streams);
    Some(out)
}

/// Encode `literals` as a Compressed_Literals_Block with a new tree.
/// Returns the section bytes and the table the decoder will hold afterwards.
pub fn encode_literals_huffman(literals: &[u8]) -> Option<(Vec<u8>, HufTable)> {
    // Count frequencies
    let mut counts = [0u32; 256];
    let mut max_sym = 0u8;
    for &b in literals {
        counts[b as usize] += 1;
        if b > max_sym {
            max_sym = b;
        }
    }
    let n_used = counts.iter().filter(|&&c| c > 0).count();
    if n_used < 2 {
        return None;
    }

    // Build length-limited Huffman (max 11 bits)
    let (codes, max_bits) = build_huffman_codes(&counts, max_sym as usize)?;

    // Encode tree description (weights packed as 4-bit pairs)
    let tree_desc = encode_huffman_tree(&codes, max_bits, max_sym as usize);
    if tree_desc.is_empty() {
        return None;
    }

    // Encode streams: single stream for < 1KB, 4 streams for >= 1KB
    let use_4 = literals.len() >= 1024;
    let streams = if use_4 {
        encode_huf_4streams(literals, &codes)
    } else {
        encode_huf_1stream(literals, &codes)
    };

    let regen = literals.len();
    let comp = tree_desc.len() + streams.len();
    let lh_size = 3 + (regen >= 1024) as usize + (regen >= 16384) as usize;

    let mut out = Vec::with_capacity(lh_size + comp);
    let htype = LIT_TYPE_COMPRESSED as u32;

    match lh_size {
        3 => {
            // bit[1:0]=type(2), bit[2]=streams_flag, bit[3]=0, bit[13:4]=regen, bit[23:14]=comp
            let sf = if use_4 { 1u32 } else { 0u32 };
            let lhc = htype | (sf << 2) | ((regen as u32) << 4) | ((comp as u32) << 14);
            out.extend_from_slice(&lhc.to_le_bytes()[..3]);
        }
        4 => {
            let lhc = htype | (2u32 << 2) | ((regen as u32) << 4) | ((comp as u32) << 18);
            out.extend_from_slice(&lhc.to_le_bytes()[..4]);
        }
        _ => {
            let lhc = htype | (3u32 << 2) | ((regen as u32) << 4) | ((comp as u32) << 22);
            out.extend_from_slice(&lhc.to_le_bytes()[..4]);
            out.push((comp >> 10) as u8);
        }
    }

    out.extend_from_slice(&tree_desc);
    out.extend_from_slice(&streams);
    Some((out, HufTable { codes, max_bits }))
}

/// Build Huffman codes — faithful port of C zstd's HUF_buildCTable_wksp.
/// 1. Sort symbols descending by count
/// 2. Build binary Huffman tree (two-queue merge)
/// 3. Enforce max depth via HUF_setMaxHeight
/// 4. Generate canonical codes using C zstd's min >>= 1 algorithm
pub fn build_huffman_codes(counts: &[u32; 256], max_sym: usize) -> Option<([(u32, u8); 256], u8)> {
    const MAX_BITS: u8 = 11;

    // Collect and sort symbols descending by count
    let mut syms: Vec<(u32, u8)> = (0..=max_sym)
        .filter(|&s| counts[s] > 0)
        .map(|s| (counts[s], s as u8))
        .collect();
    syms.sort_by_key(|a| std::cmp::Reverse(a.0));
    let n = syms.len();
    if n < 2 {
        return None;
    }

    // --- Step 1: Build Huffman tree (two-queue merge) ---
    let mut node_count = vec![0u64; 2 * n];
    let mut node_parent = vec![0u32; 2 * n];
    let mut node_nbits = vec![0u8; 2 * n];
    for i in 0..n {
        node_count[i] = syms[i].0 as u64;
    }
    for i in n..2 * n {
        node_count[i] = u64::MAX / 2;
    }

    let mut low_s = n as i32 - 1;
    let mut low_n = n;
    let mut next_node = n;

    // Helper: pick smallest from symbol queue or node queue
    let pick_smallest =
        |node_count: &[u64], low_s: &mut i32, low_n: &mut usize, next_node: usize| -> usize {
            if *low_s >= 0
                && (*low_n >= next_node || node_count[*low_s as usize] < node_count[*low_n])
            {
                let r = *low_s as usize;
                *low_s -= 1;
                r
            } else if *low_n < next_node {
                let r = *low_n;
                *low_n += 1;
                r
            } else {
                usize::MAX // shouldn't happen
            }
        };

    while next_node < 2 * n - 1 {
        let n1 = pick_smallest(&node_count, &mut low_s, &mut low_n, next_node);
        let n2 = pick_smallest(&node_count, &mut low_s, &mut low_n, next_node);
        if n1 == usize::MAX || n2 == usize::MAX {
            break;
        }
        node_count[next_node] = node_count[n1] + node_count[n2];
        node_parent[n1] = next_node as u32;
        node_parent[n2] = next_node as u32;
        next_node += 1;
    }
    let root = next_node - 1;

    // Assign bit lengths top-down
    node_nbits[root] = 0;
    for i in (n..=root).rev() {
        if i < root {
            node_nbits[i] = node_nbits[node_parent[i] as usize] + 1;
        }
    }
    for i in 0..n {
        node_nbits[i] = node_nbits[node_parent[i] as usize] + 1;
    }

    // --- Step 2: HUF_setMaxHeight ---
    // If tree is too deep, fall back to raw instead of trying to fix it.
    // Uses rankLast[] to track the least-frequent symbol at each depth,
    // ensuring both Kraft inequality and valid weight-sum.
    let largest_bits = *node_nbits[..n].iter().max().unwrap_or(&0);
    if largest_bits > MAX_BITS {
        let target = MAX_BITS;

        // Phase 1: Clamp all > target to target, compute totalCost
        let base_cost = 1i32 << (largest_bits - target);
        let mut total_cost = 0i32;
        // Scan backward (least frequent first, they have the longest codes)
        let mut last_non_null = n - 1;
        while node_nbits[last_non_null] > target {
            total_cost += base_cost - (1i32 << (largest_bits - node_nbits[last_non_null]));
            node_nbits[last_non_null] = target;
            if last_non_null == 0 {
                break;
            }
            last_non_null -= 1;
        }
        total_cost >>= largest_bits - target;

        // Build rankLast[]: position of last (least frequent) symbol at each rank
        // rankLast[k] = index of least-frequent symbol using (target - k) bits
        const NO_SYMBOL: u32 = 0xF0F0F0F0;
        let mut rank_last = [NO_SYMBOL; 16];
        {
            let mut current_bits = target;
            for pos in (0..=last_non_null).rev() {
                if node_nbits[pos] >= current_bits {
                    continue;
                }
                current_bits = node_nbits[pos];
                rank_last[(target - current_bits) as usize] = pos as u32;
            }
        }

        // Phase 2: Repay cost by lengthening symbols (increasing their nbBits)
        while total_cost > 0 {
            // Find the best rank to decrease: target the next power-of-2 chunk
            let mut n_bits_to_decrease = 32 - (total_cost as u32).leading_zeros();
            // but don't exceed available ranks
            if n_bits_to_decrease > largest_bits as u32 - target as u32 + 1 {
                n_bits_to_decrease = largest_bits as u32 - target as u32 + 1;
            }

            // Try to find best rank: prefer promoting cheap symbols
            while n_bits_to_decrease > 1 {
                let high_pos = rank_last[n_bits_to_decrease as usize];
                let low_pos = rank_last[n_bits_to_decrease as usize - 1];
                if high_pos == NO_SYMBOL {
                    n_bits_to_decrease -= 1;
                    continue;
                }
                if low_pos == NO_SYMBOL {
                    break;
                }
                let high_total = syms[high_pos as usize].0;
                let low_total = 2 * syms[low_pos as usize].0;
                if high_total <= low_total {
                    break;
                }
                n_bits_to_decrease -= 1;
            }

            // Find a non-empty rank if current is empty
            while n_bits_to_decrease as usize <= 14
                && rank_last[n_bits_to_decrease as usize] == NO_SYMBOL
            {
                n_bits_to_decrease += 1;
            }
            if n_bits_to_decrease as usize > 14
                || rank_last[n_bits_to_decrease as usize] == NO_SYMBOL
            {
                break; // can't repay
            }

            // Promote the symbol: increase its nbBits by 1 (C allows overshoot here)
            total_cost -= 1i32 << (n_bits_to_decrease - 1);
            let pos = rank_last[n_bits_to_decrease as usize] as usize;
            node_nbits[pos] += 1;

            // Update rankLast for the new rank
            if rank_last[n_bits_to_decrease as usize - 1] == NO_SYMBOL {
                rank_last[n_bits_to_decrease as usize - 1] = rank_last[n_bits_to_decrease as usize];
            }

            // Update rankLast for the old rank
            if rank_last[n_bits_to_decrease as usize] == 0 {
                rank_last[n_bits_to_decrease as usize] = NO_SYMBOL;
            } else {
                let prev = rank_last[n_bits_to_decrease as usize] - 1;
                rank_last[n_bits_to_decrease as usize] = prev;
                if node_nbits[prev as usize] != target - n_bits_to_decrease as u8 {
                    rank_last[n_bits_to_decrease as usize] = NO_SYMBOL;
                }
            }
        }

        // Phase 3: Overshoot correction (totalCost < 0)
        // Port of C zstd: demote rank-0 symbols to rank-1 (decrease nbBits by 1)
        while total_cost < 0 {
            if rank_last[1] == NO_SYMBOL {
                // No rank-1 symbols. Find last rank-0 symbol and demote it.
                let mut p = last_non_null;
                while p > 0 && node_nbits[p] == target {
                    p -= 1;
                }
                // p+1 is a rank-0 symbol (using target bits)
                if p + 1 < n && node_nbits[p + 1] == target {
                    node_nbits[p + 1] -= 1; // demote: target → target-1
                    rank_last[1] = (p + 1) as u32;
                    total_cost += 1;
                } else {
                    break; // can't correct
                }
            } else {
                // Demote the symbol just after rankLast[1] boundary
                let next = rank_last[1] as usize + 1;
                if next < n && node_nbits[next] == target {
                    node_nbits[next] -= 1;
                    rank_last[1] += 1;
                    total_cost += 1;
                } else {
                    // rankLast[1]+1 is not a rank-0 symbol, need to find one
                    rank_last[1] = NO_SYMBOL;
                    // Will retry with the NO_SYMBOL path above
                }
            }
        }

        // If still not zero, fall back
        if total_cost != 0 {
            for i in 0..n {
                node_nbits[i] = 0;
            } // will fail Kraft
        }
    }

    // --- Step 3: Extract code lengths and validate ---
    let mut lengths = [0u8; 256];
    for i in 0..n {
        lengths[syms[i].1 as usize] = node_nbits[i];
    }

    let max_bits = *lengths.iter().max().unwrap_or(&0);
    if max_bits == 0 {
        return None;
    }

    // Verify Kraft inequality: sum of 2^(max-len) must equal 2^max
    let kraft: u64 = (0..=max_sym)
        .filter(|&s| lengths[s] > 0)
        .map(|s| 1u64 << (max_bits - lengths[s]))
        .sum();
    if kraft != (1u64 << max_bits) {
        return None;
    }

    // Weight-sum is automatically valid when Kraft is valid:
    // encode_huffman_tree pops the last non-zero weight, and the remaining
    // weight_sum = kraft_sum - 2^(last_weight-1) = 2^max - 2^k, which
    // leaves a valid power-of-2 leftover for the implicit last weight.

    // Count symbols per rank
    let mut nb_per_rank = [0u32; 16];
    for &l in &lengths {
        if l > 0 {
            nb_per_rank[l as usize] += 1;
        }
    }

    // zstd-style canonical code generation — exact mirror of decoder's rank_indexes.
    // Decoder: rank_indexes[max_bits] = 0
    //          rank_indexes[bits-1] = rank_indexes[bits] + bit_ranks[bits] * (1 << (max_bits - bits))
    // Code for a symbol at rank `bits` = rank_indexes[bits] / (1 << (max_bits - bits))
    let mut rank_indexes = [0u32; 16];
    rank_indexes[max_bits as usize] = 0;
    for bits in (1..=max_bits as usize).rev() {
        rank_indexes[bits - 1] =
            rank_indexes[bits] + nb_per_rank[bits] * (1u32 << (max_bits as usize - bits));
    }

    // Assign codes: within each bit length, symbols get consecutive codes
    let mut next_code = [0u32; 16];
    for bits in 1..=max_bits as usize {
        next_code[bits] = rank_indexes[bits] >> (max_bits as usize - bits);
    }

    let mut codes = [(0u32, 0u8); 256];
    for s in 0..=max_sym {
        if lengths[s] > 0 {
            codes[s] = (next_code[lengths[s] as usize], lengths[s]);
            next_code[lengths[s] as usize] += 1;
        }
    }

    Some((codes, max_bits))
}

pub fn encode_huffman_tree(codes: &[(u32, u8); 256], max_bits: u8, max_sym: usize) -> Vec<u8> {
    if max_bits == 0 {
        return vec![];
    }
    let mut weights: Vec<u8> = (0..=max_sym)
        .map(|s| {
            if codes[s].1 > 0 {
                max_bits + 1 - codes[s].1
            } else {
                0
            }
        })
        .collect();
    while weights.last() == Some(&0) && weights.len() > 1 {
        weights.pop();
    }
    if !weights.is_empty() {
        weights.pop();
    } // last weight is implicit
    if weights.is_empty() || weights.len() > 255 {
        return vec![];
    }

    // Check all weights fit in 4 bits
    if weights.iter().any(|&w| w > 12) {
        return vec![];
    }

    let num = weights.len();

    if num <= 128 {
        // Direct mode: header = num + 127, packed 4-bit pairs
        let mut desc = Vec::with_capacity(1 + num.div_ceil(2));
        desc.push((num as u8) + 127);
        for pair in weights.chunks(2) {
            let w0 = pair[0];
            let w1 = if pair.len() > 1 { pair[1] } else { 0 };
            desc.push((w0 << 4) | (w1 & 0x0F));
        }
        desc
    } else {
        // >128 weights: FSE-compressed 2-stream interleaved encoding
        let fse_result = encode_weights_fse(&weights);
        match fse_result {
            Some(compressed) if compressed.len() < 127 => {
                let header_byte = compressed.len() as u8;
                // Verify roundtrip — only use if weights match exactly
                let verify = crate::decode::decode_huf_weights_from_fse(&compressed, header_byte);
                if let Ok(ref dw) = verify {
                    if *dw == weights {
                        let mut desc = Vec::with_capacity(1 + compressed.len());
                        desc.push(header_byte);
                        desc.extend_from_slice(&compressed);
                        return desc;
                    }
                }
                // FSE roundtrip failed — fall through to direct mode fallback
                if num <= 128 {
                    let mut desc = Vec::with_capacity(1 + num.div_ceil(2));
                    desc.push((num as u8) + 127);
                    for pair in weights.chunks(2) {
                        let w0 = pair[0];
                        let w1 = if pair.len() > 1 { pair[1] } else { 0 };
                        desc.push((w0 << 4) | (w1 & 0x0F));
                    }
                    desc
                } else {
                    vec![]
                }
            }
            _ => {
                // FSE too large or failed — try direct if num <= 128
                if num <= 128 {
                    let mut desc = Vec::with_capacity(1 + num.div_ceil(2));
                    desc.push((num as u8) + 127);
                    for pair in weights.chunks(2) {
                        let w0 = pair[0];
                        let w1 = if pair.len() > 1 { pair[1] } else { 0 };
                        desc.push((w0 << 4) | (w1 & 0x0F));
                    }
                    desc
                } else {
                    vec![] // can't encode 129+ weights without FSE
                }
            }
        }
    }
}

/// Calculate baseline and num_bits for FSE decode table entry.
/// EXACT copy of decode.rs fse_calc_baseline_and_numbits.
/// FSE-compress weights using 2-stream interleaved encoding.
/// Uses decoder's FSE table directly for encoding to guarantee compatibility.
pub fn encode_weights_fse(weights: &[u8]) -> Option<Vec<u8>> {
    let mut counts = [0u32; 13];
    let mut max_w = 0u8;
    for &w in weights {
        counts[w as usize] += 1;
        if w > max_w {
            max_w = w;
        }
    }
    if max_w == 0 {
        return None;
    }

    let table_log = 6u32;
    let table_size = 1u32 << table_log;
    let total = weights.len() as u32;

    // Normalize
    let mut norm = [0i16; 13];
    let mut dist = 0u32;
    for s in 0..=max_w as usize {
        if counts[s] == 0 {
            continue;
        }
        norm[s] = std::cmp::max(
            1,
            (counts[s] as u64 * table_size as u64 / total as u64) as i16,
        );
        dist += norm[s] as u32;
    }
    while dist > table_size {
        for s in 0..=max_w as usize {
            if norm[s] > 1 {
                norm[s] -= 1;
                dist -= 1;
                break;
            }
        }
    }
    while dist < table_size {
        let best = (0..=max_w as usize).max_by_key(|&s| counts[s]).unwrap_or(0);
        norm[best] += 1;
        dist += 1;
    }

    let fse = super::fse::FseCTable::build(&norm, max_w as usize, table_log);

    // --- FSE table header (must match decode.rs read_probabilities exactly) ---
    // Decoder: accuracy_log = 5 + get_bits(4)
    //          loop: max_remaining = prob_sum - counter + 1
    //                bits_to_read = highest_bit_set(max_remaining)
    //                read bits_to_read, apply low_threshold logic
    //                prob = value - 1
    let mut hdr = Vec::with_capacity(16);
    let mut bb: u64 = (table_log - 5) as u64;
    let mut bp = 4u32;
    let prob_sum = table_size;
    let mut counter = 0u32;

    let mut s = 0usize;
    while s <= max_w as usize && counter < prob_sum {
        let prob = norm[s] as i32;
        let value = (prob + 1) as u32;

        let max_remaining = prob_sum - counter + 1;
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
            bb |= ((value + low_threshold) as u64) << bp;
            bp += bits_to_read;
        }
        while bp >= 8 {
            hdr.push(bb as u8);
            bb >>= 8;
            bp -= 8;
        }

        if prob > 0 {
            counter += prob as u32;
        } else if prob == -1 {
            counter += 1;
        }

        if prob == 0 {
            let mut repeat = 0u32;
            while s + 1 + repeat as usize <= max_w as usize
                && norm[s + 1 + repeat as usize] == 0
                && repeat < 3
            {
                repeat += 1;
            }
            bb |= (repeat as u64) << bp;
            bp += 2;
            while bp >= 8 {
                hdr.push(bb as u8);
                bb >>= 8;
                bp -= 8;
            }
            s += repeat as usize;
            while repeat == 3 {
                repeat = 0;
                while s + 1 + repeat as usize <= max_w as usize
                    && norm[s + 1 + repeat as usize] == 0
                    && repeat < 3
                {
                    repeat += 1;
                }
                bb |= (repeat as u64) << bp;
                bp += 2;
                while bp >= 8 {
                    hdr.push(bb as u8);
                    bb >>= 8;
                    bp -= 8;
                }
                s += repeat as usize;
            }
        }
        s += 1;
    }
    if bp > 0 {
        hdr.push(bb as u8);
    }

    // --- Build decoder-compatible FSE table and encode using it ---
    // This guarantees encode/decode compatibility by using the same table structure.
    let ts = table_size as usize;

    // Build decode table (same algorithm as decode.rs)
    let mut dec_symbol = vec![0u8; ts];
    let mut dec_baseline = vec![0u32; ts];
    let mut dec_numbits = vec![0u8; ts];

    // Place -1 symbols at the end
    let mut neg_idx = ts;
    for s in 0..=max_w as usize {
        if norm[s] == -1 {
            neg_idx -= 1;
            dec_symbol[neg_idx] = s as u8;
            dec_baseline[neg_idx] = 0;
            dec_numbits[neg_idx] = table_log as u8;
        }
    }

    // Spread remaining symbols
    let mut pos = 0usize;
    for s in 0..=max_w as usize {
        if norm[s] <= 0 {
            continue;
        }
        for _ in 0..norm[s] {
            dec_symbol[pos] = s as u8;
            pos += (ts >> 1) + (ts >> 3) + 3;
            pos &= ts - 1;
            while pos >= neg_idx {
                pos += (ts >> 1) + (ts >> 3) + 3;
                pos &= ts - 1;
            }
        }
    }

    // CTable-based 2-stream interleaved encoding.
    // FSE backward encoding: init with LAST symbol, encode second-to-last down to first.
    // After all encodes, the final state carries the FIRST symbol (decoder's first output).
    let stream1: Vec<u8> = weights.iter().step_by(2).copied().collect();
    let stream2: Vec<u8> = weights.iter().skip(1).step_by(2).copied().collect();
    let len1 = stream1.len();
    let len2 = stream2.len();

    // Init with LAST symbol of each stream (= first encoded, last decoded)
    let mut st1 = fse.init_state(*stream1.last().unwrap() as usize);
    let mut st2 = if len2 > 0 {
        fse.init_state(*stream2.last().unwrap() as usize)
    } else {
        0
    };

    let mut bw = super::bitstream::BackwardBitWriter::new();

    // Encode from second-to-last down to first (index 0).
    // After all encodes, state carries stream[0]'s symbol.
    // Decoder reads: init(state) → peek(stream[0]) → update → peek(stream[1]) → ...
    let _max_idx = std::cmp::max(len1, len2);
    // Encode each stream from second-to-last down to first.
    // init handles the last element, so encode [0..last-1].
    // Interleave order: decoder reads st1 update first, so write st2 first (backward).
    let max_encode = std::cmp::max(len1.saturating_sub(1), len2.saturating_sub(1));
    for i in (0..max_encode).rev() {
        if i < len2.saturating_sub(1) {
            let (bits, nb, ns) = fse.encode_symbol(st2, stream2[i] as usize);
            bw.add_bits(bits as u64, nb);
            bw.flush_bits();
            st2 = ns;
        }
        if i < len1.saturating_sub(1) {
            let (bits, nb, ns) = fse.encode_symbol(st1, stream1[i] as usize);
            bw.add_bits(bits as u64, nb);
            bw.flush_bits();
            st1 = ns;
        }
    }

    // Write init states — convert CTable state to decoder index
    bw.add_bits((st2 - table_size) as u64, table_log);
    bw.flush_bits();
    bw.add_bits((st1 - table_size) as u64, table_log);
    bw.flush_bits();

    let bitstream = bw.finish();
    let mut out = hdr;
    out.extend_from_slice(&bitstream);
    Some(out)
}

/// Encode one Huffman stream (symbols in reverse, padded with sentinel bit).
pub fn encode_huf_1stream(data: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
    let mut bw = super::bitstream::BackwardBitWriter::new();
    // Encode symbols in reverse (backward bitstream convention)
    for &sym in data.iter().rev() {
        let (code, nb) = codes[sym as usize];
        if nb == 0 {
            continue;
        }
        bw.add_bits(code as u64, nb as u32);
        bw.flush_bits();
    }
    bw.finish() // adds sentinel, no reverse needed
}

pub fn encode_huf_4streams(data: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
    let q = data.len().div_ceil(4);
    let ends = [
        q,
        std::cmp::min(q * 2, data.len()),
        std::cmp::min(q * 3, data.len()),
        data.len(),
    ];
    let starts = [0, q, ends[1], ends[2]];

    let c: Vec<Vec<u8>> = (0..4)
        .map(|i| encode_huf_1stream(&data[starts[i]..ends[i]], codes))
        .collect();

    let mut out = Vec::with_capacity(6 + c.iter().map(|v| v.len()).sum::<usize>());
    // Jump table: sizes of first 3 streams (u16 LE each)
    for i in 0..3 {
        out.extend_from_slice(&(c[i].len() as u16).to_le_bytes());
    }
    for stream in &c {
        out.extend_from_slice(stream);
    }
    out
}

// =========================================================================
// Literals section encoding (Raw mode)
// =========================================================================

pub fn encode_literals_rle(out: &mut Vec<u8>, byte: u8, size: usize) {
    if size <= 31 {
        out.push(LIT_TYPE_RLE | ((size as u8) << 3));
    } else if size <= 4095 {
        let h = (LIT_TYPE_RLE as u16) | (1 << 2) | ((size as u16) << 4);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        let h = (LIT_TYPE_RLE as u32) | (3 << 2) | ((size as u32) << 4);
        out.extend_from_slice(&h.to_le_bytes()[..3]);
    }
    out.push(byte);
}

pub fn encode_literals_raw(out: &mut Vec<u8>, literals: &[u8]) {
    let size = literals.len();

    if size <= 31 {
        // 1-byte header: type=0 (raw), size in 5 bits
        out.push(LIT_TYPE_RAW | ((size as u8) << 3));
    } else if size <= 4095 {
        // 2-byte header
        let h = (LIT_TYPE_RAW as u16) | (1 << 2) | ((size as u16) << 4);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        // 3-byte header
        let h = (LIT_TYPE_RAW as u32) | (3 << 2) | ((size as u32) << 4);
        out.extend_from_slice(&h.to_le_bytes()[..3]);
    }

    out.extend_from_slice(literals);
}

// =========================================================================
// Literals section driver (ZSTD_compressLiterals analogue)
// =========================================================================

/// Write the literals section for one block and return the Huffman state the
/// decoder holds afterwards (`nextHuf`).
///
/// Mode selection is the pre-existing heuristic: RLE when every byte is
/// equal; for at least 64 literals a new tree or, when `prev` covers every
/// symbol, a Treeless reuse of `prev`, whichever is smaller and saves bytes;
/// raw otherwise. Like `ZSTD_compressLiterals`, a raw, RLE or Treeless
/// outcome leaves the state at `prev`; only a newly written tree yields
/// `HufState::Check(new)`.
pub fn compress_literals(out: &mut Vec<u8>, literals: &[u8], prev: &HufState) -> HufState {
    if !literals.is_empty() && literals.iter().all(|&b| b == literals[0]) {
        encode_literals_rle(out, literals[0], literals.len());
        return prev.clone();
    }
    if literals.len() >= 64 {
        let new_result = encode_literals_huffman(literals);
        // Treeless: reuse the previous tree only if it has a code for every symbol.
        let treeless_result = prev.table().and_then(|t| {
            if literals.iter().all(|&b| t.codes[b as usize].1 > 0) {
                encode_literals_treeless(literals, &t.codes)
            } else {
                None
            }
        });
        match (new_result, treeless_result) {
            (Some((ne, _)), Some(te)) if te.len() <= ne.len() && te.len() < literals.len() => {
                out.extend_from_slice(&te);
                return prev.clone();
            }
            (Some((ne, table)), _) if ne.len() < literals.len() => {
                out.extend_from_slice(&ne);
                return HufState::Check(table);
            }
            (None, Some(te)) if te.len() < literals.len() => {
                out.extend_from_slice(&te);
                return prev.clone();
            }
            _ => {}
        }
    }
    encode_literals_raw(out, literals);
    prev.clone()
}
