//! Zstandard frame compressor.
//!
//! Produces valid zstd frames decompressible by any standard decoder.
//! Uses greedy hash-based matching (equivalent to zstd level 1).

use super::constants::*;

/// Compress data into a zstd frame.
///
/// `level` controls the compression strategy:
/// - 0: no compression (raw blocks, fastest)
/// - 1-2: greedy matching (fast, hash_log=14)
/// - 3-5: lazy matching (better ratio, hash_log=15)
/// - 6-8: lazy matching + deeper search (hash_log=16)
/// - 9-11: lazy matching + deepest search (hash_log=17)
///
/// Returns a valid zstd frame decompressible by any conformant decoder.
pub fn compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    write_frame_header(&mut out, data.len() as u64);

    if data.is_empty() {
        write_raw_block(&mut out, &[], true);
        return out;
    }

    if level <= 0 {
        // Level 0: raw/RLE blocks
        let blocks: Vec<&[u8]> = data.chunks(ZSTD_BLOCKSIZE_MAX).collect();
        let n_blocks = blocks.len();
        for (i, block) in blocks.iter().enumerate() {
            if is_rle_block(block) {
                write_rle_block(&mut out, block[0], block.len(), i == n_blocks - 1);
            } else {
                write_raw_block(&mut out, block, i == n_blocks - 1);
            }
        }
        return out;
    }

    // Level 1+: run match finder on entire input (cross-block window), then
    // split sequences into blocks for encoding.
    let params = MatchParams::from_level(level);
    let all_sequences = find_matches(data, &params);
    let all_encoded = resolve_repeat_offsets(&all_sequences);

    // Split sequences into blocks of up to ZSTD_BLOCKSIZE_MAX output bytes.
    // Each sequence produces ll + ml output bytes. Track cumulative output
    // and cut a new block when we'd exceed the limit.
    let mut block_ranges: Vec<(usize, usize, usize)> = Vec::new(); // (seq_start, seq_end, data_end)
    let mut seq_start = 0usize;
    let mut data_pos = 0usize;
    let mut block_output = 0usize;

    for (i, (raw, _enc)) in all_sequences.iter().zip(all_encoded.iter()).enumerate() {
        let seq_output = raw.ll as usize + raw.ml as usize;
        if block_output + seq_output > ZSTD_BLOCKSIZE_MAX && i > seq_start {
            // End current block before this sequence
            block_ranges.push((seq_start, i, data_pos));
            seq_start = i;
            block_output = 0;
        }
        block_output += seq_output;
        data_pos += seq_output;
    }
    // Final block: includes remaining sequences + trailing literals
    block_ranges.push((seq_start, all_sequences.len(), data.len()));

    let n_blocks = block_ranges.len();

    for (bi, &(s_start, s_end, d_end)) in block_ranges.iter().enumerate() {
        let is_last = bi == n_blocks - 1;
        let block_seqs = &all_encoded[s_start..s_end];
        let raw_seqs = &all_sequences[s_start..s_end];

        // Determine the byte range for this block
        let d_start = if s_start == 0 {
            0
        } else {
            // Sum up all ll + ml of sequences before this block
            let mut p = 0usize;
            for s in &all_sequences[..s_start] {
                p += s.ll as usize + s.ml as usize;
            }
            p
        };
        let block_data = &data[d_start..d_end];

        if block_seqs.is_empty() {
            // No sequences in this block - use RLE if possible, else raw
            if is_rle_block(block_data) {
                write_rle_block(&mut out, block_data[0], block_data.len(), is_last);
            } else {
                write_raw_block(&mut out, block_data, is_last);
            }
            continue;
        }

        // Collect literals for this block
        let mut literals = Vec::with_capacity(block_data.len());
        let mut pos = 0usize;
        for seq in raw_seqs {
            literals.extend_from_slice(&block_data[pos..pos + seq.ll as usize]);
            pos += seq.ll as usize + seq.ml as usize;
        }
        // Trailing literals
        literals.extend_from_slice(&block_data[pos..]);

        // Encode block
        let mut block = Vec::with_capacity(block_data.len());

        // Literals section: try Huffman, fall back to raw
        let mut used_huf = false;
        if literals.len() >= 64 {
            if let Some(huf) = encode_literals_huffman(&literals) {
                if huf.len() < literals.len() {
                    block.extend_from_slice(&huf);
                    used_huf = true;
                }
            }
        }
        if !used_huf {
            encode_literals_raw(&mut block, &literals);
        }

        // Sequences section
        encode_sequences_section(&mut block, block_seqs);

        // Choose smallest block type: RLE < compressed < raw
        if is_rle_block(block_data) {
            write_rle_block(&mut out, block_data[0], block_data.len(), is_last);
        } else if block.len() < block_data.len() {
            write_compressed_block(&mut out, &block, is_last);
        } else {
            write_raw_block(&mut out, block_data, is_last);
        }
    }

    out
}

/// Convenience wrapper.
pub fn compress_to_vec(data: &[u8]) -> Vec<u8> {
    compress(data, 1)
}

// =========================================================================
// Frame header
// =========================================================================

fn write_frame_header(out: &mut Vec<u8>, content_size: u64) {
    // Magic number (LE)
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

    // Frame_Header_Descriptor:
    // bit 7-6: Frame_Content_Size_flag (determines FCS field size)
    // bit 5:   Single_Segment_flag (1 = no Window_Descriptor)
    // bit 4:   unused
    // bit 3:   reserved
    // bit 2:   Content_Checksum_flag (0 = no checksum)
    // bit 1-0: Dictionary_ID_flag (0 = no dict)

    let (fcs_flag, fcs_bytes) = if content_size <= 255 {
        (0u8, 1) // 1 byte FCS (but flag 0 means 0 bytes normally...)
    } else if content_size <= 65535 + 256 {
        (1u8, 2) // 2 bytes
    } else if content_size <= u32::MAX as u64 {
        (2u8, 4)
    } else {
        (3u8, 8)
    };

    // For single-segment, FCS_flag=0 means 1 byte (when single_segment=1)
    let single_segment = 1u8; // always single-segment for simplicity
    let descriptor = (fcs_flag << 6) | (single_segment << 5);
    out.push(descriptor);

    // No Window_Descriptor (single_segment = 1)

    // Frame_Content_Size
    match fcs_bytes {
        1 => out.push(content_size as u8),
        2 => out.extend_from_slice(&((content_size - 256) as u16).to_le_bytes()),
        4 => out.extend_from_slice(&(content_size as u32).to_le_bytes()),
        8 => out.extend_from_slice(&content_size.to_le_bytes()),
        _ => {}
    }
}

// =========================================================================
// Block writing
// =========================================================================

fn write_raw_block(out: &mut Vec<u8>, data: &[u8], is_last: bool) {
    let header = (is_last as u32) | ((BLOCK_TYPE_RAW as u32) << 1) | ((data.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.extend_from_slice(data);
}

fn write_rle_block(out: &mut Vec<u8>, byte: u8, repeat_count: usize, is_last: bool) {
    let header =
        (is_last as u32) | ((BLOCK_TYPE_RLE as u32) << 1) | ((repeat_count as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.push(byte);
}

/// Check if all bytes in a slice are identical (RLE candidate).
fn is_rle_block(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    let first = data[0];
    data.iter().all(|&b| b == first)
}

fn write_compressed_block(out: &mut Vec<u8>, compressed: &[u8], is_last: bool) {
    let header =
        (is_last as u32) | ((BLOCK_TYPE_COMPRESSED as u32) << 1) | ((compressed.len() as u32) << 3);
    out.extend_from_slice(&header.to_le_bytes()[..3]);
    out.extend_from_slice(compressed);
}

// =========================================================================
// Block compression (greedy matching + raw literals + predefined FSE)
// =========================================================================

/// A sequence: (literal_length, offset_value, match_length).
/// `off` is the raw back-reference distance.
/// After repeat offset resolution, it becomes an "offset value" for encoding.
struct Sequence {
    ll: u32,
    off: u32, // raw back-reference distance
    ml: u32,  // actual match length (>= ZSTD_MINMATCH)
}

/// Offset value after repeat-offset resolution.
/// In zstd, offset_value 1/2/3 = repeat offsets, >3 = new offset + 3.
struct EncodedSequence {
    ll: u32,
    of_value: u32, // offset value for encoding (1..3 = repcode, >3 = new)
    ml: u32,
}

/// Match finder parameters, derived from compression level.
struct MatchParams {
    hash_log: u32,
    lazy_depth: u32,   // 0=greedy, 1=lazy, 2=lazy2
    search_depth: u32, // hash chain search depth
}

impl MatchParams {
    /// Parameters aligned with C zstd compression parameters.
    /// C zstd level 1: hashLog=17, strategy=fast (no chains, greedy)
    /// C zstd level 3: hashLog=18, chainLog=19, searchLog=4, strategy=dfast
    /// C zstd level 7: hashLog=19, chainLog=22, searchLog=6, strategy=lazy2
    fn from_level(level: i32) -> Self {
        match level {
            0..=2 => Self {
                hash_log: 17,
                lazy_depth: 0,
                search_depth: 4,
            },
            3..=5 => Self {
                hash_log: 18,
                lazy_depth: 1,
                search_depth: 16,
            },
            6..=8 => Self {
                hash_log: 19,
                lazy_depth: 1,
                search_depth: 64,
            },
            _ => Self {
                hash_log: 20,
                lazy_depth: 1,
                search_depth: 256,
            },
        }
    }
}

fn compress_block(data: &[u8], params: &MatchParams) -> Option<Vec<u8>> {
    let sequences = find_matches(data, params);

    if sequences.is_empty() {
        return None;
    }

    // === 1. Resolve repeat offsets per zstd spec ===
    let encoded_seqs = resolve_repeat_offsets(&sequences);

    // === 2. Collect literals ===
    let mut literals = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    for seq in &sequences {
        literals.extend_from_slice(&data[pos..pos + seq.ll as usize]);
        pos += seq.ll as usize + seq.ml as usize;
    }
    literals.extend_from_slice(&data[pos..]);

    // === 3. Encode block ===
    let mut block = Vec::with_capacity(data.len());

    // Literals section: try Huffman, fall back to raw
    let mut used_huf = false;
    if literals.len() >= 64 {
        if let Some(huf) = encode_literals_huffman(&literals) {
            if huf.len() < literals.len() {
                block.extend_from_slice(&huf);
                used_huf = true;
            }
        }
    }
    if !used_huf {
        encode_literals_raw(&mut block, &literals);
    }

    // Sequences section
    encode_sequences_section(&mut block, &encoded_seqs);

    Some(block)
}

/// Resolve repeat offsets per zstd spec (RFC 8878 §3.1.2.5).
///
/// Encoder chooses offset_value for each sequence:
/// - If raw_offset matches a repeat offset → use repcode (1/2/3)
/// - Otherwise → use raw_offset + 3
///
/// After each sequence, the repeat offset table is updated:
/// - New offset → shift: rep = [new, old_rep0, old_rep1]
/// - Repeat 1 → no change
/// - Repeat 2 → rotate: rep = [rep1, rep0, rep2]
/// - Repeat 3 → rotate: rep = [rep2, rep0, rep1]
fn resolve_repeat_offsets(sequences: &[Sequence]) -> Vec<EncodedSequence> {
    let mut rep = [1u32, 4, 8]; // initial repeat offsets
    let mut out = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let raw_off = seq.off;
        let of_value;

        if seq.ll > 0 {
            // Normal case (ll > 0)
            if raw_off == rep[0] {
                of_value = 1;
                // rep unchanged
            } else if raw_off == rep[1] {
                of_value = 2;
                // rotate: [rep1, rep0, rep2]
                rep = [rep[1], rep[0], rep[2]];
            } else if raw_off == rep[2] {
                of_value = 3;
                // rotate: [rep2, rep0, rep1]
                rep = [rep[2], rep[0], rep[1]];
            } else {
                of_value = raw_off + 3;
                // shift: [new, old0, old1]
                rep = [raw_off, rep[0], rep[1]];
            }
        } else {
            // ll == 0: offsets are shifted by 1
            // of_value 1 → rep[1], of_value 2 → rep[2], of_value 3 → rep[0]-1
            if raw_off == rep[1] {
                of_value = 1;
                rep = [rep[1], rep[0], rep[2]];
            } else if raw_off == rep[2] {
                of_value = 2;
                rep = [rep[2], rep[0], rep[1]];
            } else if raw_off == rep[0].wrapping_sub(1) && rep[0] > 1 {
                of_value = 3;
                rep = [rep[0] - 1, rep[0], rep[1]];
            } else {
                of_value = raw_off + 3;
                rep = [raw_off, rep[0], rep[1]];
            }
        }

        out.push(EncodedSequence {
            ll: seq.ll,
            of_value,
            ml: seq.ml,
        });
    }

    out
}

/// Hash chain match finder with configurable lazy depth.
///
/// - `lazy_depth=0`: greedy — take first match (level 1-2)
/// - `lazy_depth=1`: lazy — check next position for better match (level 3+)
fn find_matches(data: &[u8], params: &MatchParams) -> Vec<Sequence> {
    if data.len() < ZSTD_MINMATCH + 1 {
        return vec![];
    }

    let hash_size = 1usize << params.hash_log;
    let hash_mask = (hash_size - 1) as u32;
    let mut hash_table = vec![0u32; hash_size];
    let mut chain = vec![0u32; data.len()];
    let mut sequences = Vec::new();
    let mut anchor = 0usize;
    let mut ip = 0usize;

    while ip + ZSTD_MINMATCH < data.len() {
        // Try to find a match at current position
        let best = find_best_at(
            data,
            ip,
            &hash_table,
            &chain,
            hash_mask,
            params.search_depth,
        );

        if let Some((offset, match_len)) = best {
            let mut final_off = offset;
            let mut final_len = match_len;
            let mut final_ip = ip;

            // Lazy matching: check if next position gives a better match
            if params.lazy_depth >= 1 && ip + 1 + ZSTD_MINMATCH < data.len() {
                insert_hash(&mut hash_table, &mut chain, data, ip, hash_mask);
                if let Some((off2, len2)) = find_best_at(
                    data,
                    ip + 1,
                    &hash_table,
                    &chain,
                    hash_mask,
                    params.search_depth,
                ) {
                    if len2 > final_len + 1 {
                        final_off = off2;
                        final_len = len2;
                        final_ip = ip + 1;
                    }
                }
            }

            let ll = (final_ip - anchor) as u32;
            sequences.push(Sequence {
                ll,
                off: final_off as u32,
                ml: final_len as u32,
            });

            // Insert all positions within the match for future matching
            for p in ip..std::cmp::min(
                final_ip + final_len,
                data.len().saturating_sub(ZSTD_MINMATCH),
            ) {
                insert_hash(&mut hash_table, &mut chain, data, p, hash_mask);
            }

            ip = final_ip + final_len;
            anchor = ip;
        } else {
            insert_hash(&mut hash_table, &mut chain, data, ip, hash_mask);
            ip += 1;
        }
    }

    sequences
}

/// Insert position into hash chain.
#[inline]
fn insert_hash(hash_table: &mut [u32], chain: &mut [u32], data: &[u8], pos: usize, mask: u32) {
    if pos + 4 > data.len() {
        return;
    }
    let h = hash4(&data[pos..], mask);
    chain[pos] = hash_table[h];
    hash_table[h] = pos as u32;
}

/// Find the best match at `pos` by walking the hash chain.
fn find_best_at(
    data: &[u8],
    pos: usize,
    hash_table: &[u32],
    chain: &[u32],
    mask: u32,
    max_depth: u32,
) -> Option<(usize, usize)> {
    if pos + ZSTD_MINMATCH > data.len() {
        return None;
    }
    let h = hash4(&data[pos..], mask);
    let mut candidate = hash_table[h] as usize;
    let mut best_len = ZSTD_MINMATCH - 1;
    let mut best_off = 0;

    for _ in 0..max_depth {
        if candidate >= pos || pos - candidate > (1 << 24) {
            break;
        }
        if candidate + ZSTD_MINMATCH > data.len() {
            break;
        }

        // Quick 4-byte check
        if data[candidate..candidate + 4] == data[pos..pos + 4] {
            // Cap match length at spec maximum (ML code 52: 65539 + 65535 = 131074)
            let max_ml = std::cmp::min(ZSTD_BLOCKSIZE_MAX, data.len() - pos);
            let cand_max = std::cmp::min(max_ml, data.len() - candidate);
            let ml = common_prefix_len(&data[candidate..candidate + cand_max], &data[pos..pos + cand_max]);
            if ml > best_len {
                best_len = ml;
                best_off = pos - candidate;
            }
        }

        let next = chain[candidate] as usize;
        if next >= candidate {
            break;
        }
        candidate = next;
    }

    if best_len >= ZSTD_MINMATCH {
        Some((best_off, best_len))
    } else {
        None
    }
}

/// Fast common prefix length using 8-byte chunks.
#[inline]
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let max = std::cmp::min(a.len(), b.len());
    let mut i = 0;
    while i + 8 <= max {
        let va = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let vb = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if va != vb {
            return i + ((va ^ vb).trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < max && a[i] == b[i] {
        i += 1;
    }
    i
}

/// 4-byte multiplicative hash, result masked to table size.
#[inline]
fn hash4(data: &[u8], mask: u32) -> usize {
    let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    (v.wrapping_mul(0x9E3779B1) as usize) & (mask as usize)
}

// =========================================================================
// Huffman literal compression
// =========================================================================

/// Build Huffman codes, encode tree + streams. Returns None if Huffman doesn't help.
fn encode_literals_huffman(literals: &[u8]) -> Option<Vec<u8>> {
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
    Some(out)
}

/// Build Huffman codes using the C zstd method:
/// 1. Sort symbols by count
/// 2. Build binary tree (classic Huffman)
/// 3. Clamp to MAX_BITS using HUF_setMaxHeight
/// 4. Generate canonical codes
fn build_huffman_codes(counts: &[u32; 256], max_sym: usize) -> Option<([(u32, u8); 256], u8)> {
    const MAX_BITS: u8 = 11;

    let mut syms: Vec<(u32, u8)> = (0..=max_sym)
        .filter(|&s| counts[s] > 0)
        .map(|s| (counts[s], s as u8))
        .collect();
    syms.sort_by(|a, b| b.0.cmp(&a.0)); // descending by count (C convention)
    let n = syms.len();
    if n < 2 {
        return None;
    }

    // --- Step 1: Build Huffman tree ---
    // Nodes: [0..n) = leaf nodes (symbols), [n..2n-1) = internal nodes
    let mut node_count = vec![0u64; 2 * n];
    let mut node_parent = vec![0u32; 2 * n];
    let mut node_nbits = vec![0u8; 2 * n];

    for i in 0..n {
        node_count[i] = syms[i].0 as u64;
    }

    let mut low_sym = n as i32 - 1; // pointer into leaf nodes (right to left)
    let mut low_node = n; // pointer into internal nodes (left to right)
    let mut node_nb = n; // next internal node to create

    // Create first internal node
    if n >= 2 {
        node_count[node_nb] = node_count[low_sym as usize] + node_count[(low_sym - 1) as usize];
        node_parent[low_sym as usize] = node_nb as u32;
        node_parent[(low_sym - 1) as usize] = node_nb as u32;
        node_nb += 1;
        low_sym -= 2;
    }

    let _node_root = node_nb + low_sym as usize; // not exactly right for general case
                                                 // Fill remaining internal nodes with large counts
    for i in node_nb..2 * n {
        node_count[i] = u64::MAX / 2;
    }

    while node_nb < 2 * n - 1 {
        let n1 = if low_sym >= 0 && node_count[low_sym as usize] < node_count[low_node] {
            let r = low_sym as usize;
            low_sym -= 1;
            r
        } else if low_node < node_nb {
            let r = low_node;
            low_node += 1;
            r
        } else {
            break;
        };

        let n2 = if low_sym >= 0 && node_count[low_sym as usize] < node_count[low_node] {
            let r = low_sym as usize;
            low_sym -= 1;
            r
        } else if low_node < node_nb {
            let r = low_node;
            low_node += 1;
            r
        } else {
            break;
        };

        node_count[node_nb] = node_count[n1] + node_count[n2];
        node_parent[n1] = node_nb as u32;
        node_parent[n2] = node_nb as u32;
        node_nb += 1;
    }

    let root = node_nb - 1;

    // --- Step 2: Assign bit lengths from tree ---
    node_nbits[root] = 0;
    for i in (n..root).rev() {
        node_nbits[i] = node_nbits[node_parent[i] as usize] + 1;
    }
    for i in 0..n {
        node_nbits[i] = node_nbits[node_parent[i] as usize] + 1;
    }

    // --- Step 3: Clamp to MAX_BITS (HUF_setMaxHeight) ---
    let largest_bits = *node_nbits[..n].iter().max().unwrap_or(&0);
    if largest_bits > MAX_BITS {
        let base_cost = 1i32 << (largest_bits - MAX_BITS);
        let mut total_cost = 0i32;

        // Clamp all > MAX_BITS to MAX_BITS, accumulate cost
        for i in 0..n {
            if node_nbits[i] > MAX_BITS {
                total_cost += base_cost - (1 << (largest_bits - node_nbits[i]));
                node_nbits[i] = MAX_BITS;
            }
        }

        // Normalize cost
        total_cost >>= largest_bits - MAX_BITS;

        // Repay cost by shortening symbols (making some codes shorter)
        while total_cost > 0 {
            // Find a symbol with nbits < MAX_BITS that can absorb cost
            let mut found = false;
            for nb in (1..MAX_BITS).rev() {
                for i in 0..n {
                    if node_nbits[i] == nb {
                        node_nbits[i] += 1; // lengthen by 1 → saves 2^(MAX_BITS-nb-1)
                        total_cost -= 1 << (MAX_BITS - nb - 1);
                        if total_cost <= 0 {
                            found = true;
                            break;
                        }
                    }
                }
                if found || total_cost <= 0 {
                    break;
                }
            }
            if !found {
                break;
            }
        }
    }

    // --- Step 4: Build canonical codes from lengths ---
    let mut lengths = [0u8; 256];
    for i in 0..n {
        lengths[syms[i].1 as usize] = node_nbits[i];
    }

    let max_bits = *lengths.iter().max().unwrap_or(&0);
    if max_bits == 0 {
        return None;
    }

    // Verify Kraft
    let kraft: u64 = (0..256)
        .filter(|&s| lengths[s] > 0)
        .map(|s| 1u64 << (max_bits - lengths[s]))
        .sum();
    if !kraft.is_power_of_two() {
        return None;
    }

    let mut bl_count = [0u32; 16];
    for &l in &lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }

    let mut next_code = [0u32; 16];
    for bits in 1..=max_bits as usize {
        next_code[bits] = (next_code[bits - 1] + bl_count[bits - 1]) << 1;
    }

    let mut codes = [(0u32, 0u8); 256];
    for s in 0..256 {
        if lengths[s] > 0 {
            codes[s] = (next_code[lengths[s] as usize], lengths[s]);
            next_code[lengths[s] as usize] += 1;
        }
    }

    Some((codes, max_bits))
}

fn encode_huffman_tree(codes: &[(u32, u8); 256], max_bits: u8, max_sym: usize) -> Vec<u8> {
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
        #[cfg(test)]
        if let Some(ref c) = fse_result {
            eprintln!(
                "FSE weight: {} weights -> {} bytes (limit {})",
                num,
                c.len(),
                num.div_ceil(2)
            );
        } else {
            eprintln!(
                "FSE weight: encode_weights_fse returned None for {} weights",
                num
            );
        }
        match fse_result {
            Some(compressed) if compressed.len() < 127 && compressed.len() < num.div_ceil(2) => {
                // Verify: try decoding the FSE weights to check roundtrip
                let header_byte = compressed.len() as u8;
                let verify =
                    crate::decode::decode_huf_weights_from_fse(&compressed, header_byte);
                match verify {
                    Ok(decoded_weights) if decoded_weights == weights => {
                        let mut desc = Vec::with_capacity(1 + compressed.len());
                        desc.push(header_byte);
                        desc.extend_from_slice(&compressed);
                        desc
                    }
                    Ok(_d) => {
                        #[cfg(test)]
                        eprintln!(
                            "FSE weight mismatch: {} encoded, {} decoded, first diff at {}",
                            weights.len(),
                            _d.len(),
                            weights
                                .iter()
                                .zip(_d.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(9999)
                        );
                        vec![]
                    }
                    Err(_e) => {
                        #[cfg(test)]
                        eprintln!("FSE weight decode error: {}", _e);
                        vec![]
                    }
                }
            }
            _ => vec![],
        }
    }
}

/// FSE-compress weights using 2-stream interleaved encoding.
///
/// Decoder reads: sentinel → init_state(dec1) → init_state(dec2) →
/// loop { dec1.decode + update, dec2.decode + update } until exhausted.
///
/// Encoder writes backward: data pairs → state2 → state1 → sentinel.
/// After bw.finish() (reverse + sentinel), decoder reads correctly.
fn encode_weights_fse(weights: &[u8]) -> Option<Vec<u8>> {
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

    for s in 0..=max_w as usize {
        if counter >= prob_sum {
            break;
        }
        let prob = norm[s] as i32;
        let value = (prob + 1) as u32; // prob=-1 → value=0, prob=0 → value=1, etc.

        let max_remaining = prob_sum - counter + 1;
        let bits_to_read = 32 - max_remaining.leading_zeros(); // = highest_bit_set(max_remaining)

        let low_threshold = ((1u32 << bits_to_read) - 1) - max_remaining;
        let mask = (1u32 << (bits_to_read - 1)) - 1;

        if value < low_threshold {
            // Case 1: decoder reads (btr-1) bits, gets value directly
            bb |= (value as u64) << bp;
            bp += bits_to_read - 1;
        } else if value <= mask {
            // Case 3: decoder reads btr bits, unchecked = value, value <= mask, value >= low_threshold
            bb |= (value as u64) << bp;
            bp += bits_to_read;
        } else {
            // Case 2: decoder reads btr bits, unchecked > mask, value = unchecked - low_threshold
            let encoded = value + low_threshold;
            bb |= (encoded as u64) << bp;
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
        // prob == 0 doesn't contribute to counter
    }
    if bp > 0 {
        hdr.push(bb as u8);
    }

    // --- 2-stream interleaved backward bitstream ---
    let _n = weights.len();
    let mut bw = super::bitstream::BackwardBitWriter::new();

    // Decoder reads:
    //   1. sentinel (1-bit + padding zeros)
    //   2. dec1.init_state (table_log bits)
    //   3. dec2.init_state (table_log bits)
    //   4. loop: dec1.decode_symbol, dec1.update_state(num_bits),
    //            dec2.decode_symbol, dec2.update_state(num_bits), ...
    //
    // decode_symbol reads nothing (just returns state.symbol).
    // update_state reads state.num_bits bits → new_state = base_line + bits_read.
    //
    // Encoder must write in reverse:
    //   First: update_state bits for early symbols
    //   ...
    //   Last: init states → sentinel
    //
    // The FSE CTable.encode_symbol(state, sym) returns:
    //   (bits_out, num_bits, new_state)
    // where bits_out are the low bits of the OLD state.
    // This corresponds to what update_state reads.

    // Split: stream1 = w[0],w[2],w[4],... stream2 = w[1],w[3],w[5],...
    let stream1: Vec<u8> = weights.iter().step_by(2).copied().collect();
    let stream2: Vec<u8> = weights.iter().skip(1).step_by(2).copied().collect();

    // Init states: decoder outputs stream1[0] first, so init with that symbol.
    // In backward encoding, init is the LAST thing written → FIRST thing read.
    // encode_symbol processes stream1[len1-2]..stream1[1] (skipping [0] and last).
    // The last symbol (stream1.last()) gets its state from the final encode_symbol.
    // The first symbol (stream1[0]) is the init state — decoded first.
    // BUT: the loop processes down to index 0, so actually we init with stream1.last()
    // and the loop's final encode_symbol at i=0 produces the state that carries stream1[0].
    // Wait... let me re-think.
    //
    // FSE backward encoding protocol:
    // 1. init_state(last_symbol) → set initial state for that symbol
    // 2. for each symbol from second-to-last down to first:
    //    encode_symbol(state, symbol) → output bits, update state
    // 3. flush final state
    //
    // Decoder:
    // 1. read init state → state
    // 2. output state.symbol (= LAST symbol from encoding, = FIRST output from decoder)
    // 3. read update bits → new state, output new state.symbol
    //    (= second-to-last symbol from encoding, = second output)
    //
    // So init_state(last_symbol) is correct! The decoder's first output IS the last
    // symbol encoded. But we process stream1 elements as: last → init, then
    // second-to-last → encode, ..., first → encode.
    // Decoder outputs: last (from init), second-to-last, ..., first.
    // That means decoder output = REVERSED stream1.
    //
    // But we want decoder to output stream1 in FORWARD order!
    // So we need to REVERSE the encoding order:
    // init_state(stream1[0]), encode stream1[1], stream1[2], ..., stream1[last]
    //
    // That way decoder outputs: stream1[0] (init), stream1[1], ..., stream1[last]
    let mut st1 = fse.init_state(stream1[0] as usize);
    let mut st2 = fse.init_state(stream2[0] as usize);

    // Encode in reverse order, alternating stream2 then stream1
    // (because decoder reads stream1 first after init)
    let len1 = stream1.len();
    let len2 = stream2.len();

    // Backward encoding: init with [0], encode [last] first → [1] last.
    // Decoder reads: init([0]) → update with [1] bits → update with [2] → ... → [last].
    let max_idx = std::cmp::max(len1, len2);
    for i in (1..max_idx).rev() {
        if i < len2 {
            let (bits, nb, ns) = fse.encode_symbol(st2, stream2[i] as usize);
            bw.add_bits(bits as u64, nb);
            bw.flush_bits();
            st2 = ns;
        }
        if i < len1 {
            let (bits, nb, ns) = fse.encode_symbol(st1, stream1[i] as usize);
            bw.add_bits(bits as u64, nb);
            bw.flush_bits();
            st1 = ns;
        }
    }

    // Write init states as decode table indices (state - table_size).
    // dec1 is read first → written last (backward bitstream convention).
    let table_size = 1u32 << table_log;
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
fn encode_huf_1stream(data: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
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

fn encode_huf_4streams(data: &[u8], codes: &[(u32, u8); 256]) -> Vec<u8> {
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

fn encode_literals_raw(out: &mut Vec<u8>, literals: &[u8]) {
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
// Sequences section encoding using exact C-compatible FSE tables
// =========================================================================

fn encode_sequences_section(out: &mut Vec<u8>, sequences: &[EncodedSequence]) {
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
        return;
    }

    // Convert sequences to codes + extra bit values
    let mut ll_codes_v = Vec::with_capacity(nb_seq);
    let mut ml_codes_v = Vec::with_capacity(nb_seq);
    let mut off_codes_v = Vec::with_capacity(nb_seq);
    let mut ll_values = Vec::with_capacity(nb_seq);
    let mut ml_values = Vec::with_capacity(nb_seq);
    let mut off_values = Vec::with_capacity(nb_seq);

    for seq in sequences {
        let llc = ll_code(seq.ll);
        let ml_base = seq.ml - ZSTD_MINMATCH as u32;
        let mlc = ml_code(ml_base);
        let ofc = off_code(seq.of_value);

        ll_codes_v.push(llc);
        ml_codes_v.push(mlc);
        off_codes_v.push(ofc);
        ll_values.push(seq.ll - LL_BASE[llc as usize]);
        ml_values.push(seq.ml - ML_BASE[mlc as usize]);
        off_values.push(if ofc > 0 {
            seq.of_value - (1u32 << ofc)
        } else {
            0
        });
    }

    // Choose best mode for each table: Predefined vs RLE vs Custom FSE
    let ll_mode = choose_seq_mode(&ll_codes_v, MAX_LL, LL_DEFAULT_NORM_LOG, &LL_DEFAULT_NORM, LL_FSE_LOG);
    let of_mode = choose_seq_mode(&off_codes_v, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG, &OF_DEFAULT_NORM, OFF_FSE_LOG);
    let ml_mode = choose_seq_mode(&ml_codes_v, MAX_ML, ML_DEFAULT_NORM_LOG, &ML_DEFAULT_NORM, ML_FSE_LOG);

    // Write compression modes byte
    let mode_byte = (ll_mode.tag() << 6) | (of_mode.tag() << 4) | (ml_mode.tag() << 2);
    out.push(mode_byte);

    // Write table descriptions for non-predefined modes, then build tables
    let ll_table = write_seq_table_and_build(out, &ll_mode, &LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
    let of_table = write_seq_table_and_build(out, &of_mode, &OF_DEFAULT_NORM, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG);
    let ml_table = write_seq_table_and_build(out, &ml_mode, &ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);

    // Encode with FSE sequence encoder
    let bitstream = super::fse::encode_sequences(
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
}

// =========================================================================
// Custom FSE table mode selection for sequences
// =========================================================================

/// Chosen compression mode for a sequence table.
enum SeqTableMode {
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
    fn tag(&self) -> u8 {
        match self {
            SeqTableMode::Predefined => SEQ_MODE_PREDEFINED,
            SeqTableMode::Rle(_) => SEQ_MODE_RLE,
            SeqTableMode::Fse { .. } => SEQ_MODE_FSE,
        }
    }
}

/// Normalize counts to fit `table_log` total, producing FSE-compatible normalized frequencies.
fn normalize_counts(counts: &[u32], max_symbol: usize, table_log: u32) -> Vec<i16> {
    let table_size = 1u32 << table_log;
    let total: u64 = counts[..=max_symbol].iter().map(|&c| c as u64).sum();
    if total == 0 {
        return vec![0i16; max_symbol + 1];
    }

    let mut norm = vec![0i16; max_symbol + 1];
    let mut distributed = 0i32;
    let mut largest_sym = 0usize;
    let mut largest_count = 0u32;

    for s in 0..=max_symbol {
        if counts[s] == 0 {
            continue;
        }
        if counts[s] > largest_count {
            largest_count = counts[s];
            largest_sym = s;
        }
        // Proportional allocation, minimum 1 (or -1 for very rare)
        let prob = (counts[s] as u64 * table_size as u64 / total) as i16;
        if prob == 0 {
            // Very rare symbol: assign probability -1 (special "less than 1")
            norm[s] = -1;
            distributed += 1;
        } else {
            norm[s] = std::cmp::max(1, prob);
            distributed += norm[s] as i32;
        }
    }

    // Adjust the most frequent symbol to make the total exactly table_size
    let target = table_size as i32;
    norm[largest_sym] += (target - distributed) as i16;
    // Ensure it stays positive
    if norm[largest_sym] < 1 {
        // Fallback: redistribute from others
        let deficit = 1 - norm[largest_sym] as i32;
        norm[largest_sym] = 1;
        let mut remaining = deficit;
        for s in (0..=max_symbol).rev() {
            if s == largest_sym || norm[s] <= 1 {
                continue;
            }
            let take = std::cmp::min(remaining, norm[s] as i32 - 1);
            norm[s] -= take as i16;
            remaining -= take;
            if remaining <= 0 {
                break;
            }
        }
    }

    norm
}

/// Encode an FSE probability header (the variable-bit format from the spec).
/// Returns the serialized header bytes.
fn encode_fse_header(norm: &[i16], max_symbol: usize, table_log: u32) -> Vec<u8> {
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

/// Choose the best compression mode for a sequence symbol type.
fn choose_seq_mode(
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
        let symbol_log = if n_used <= 2 { min_log } else {
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

    if !all_covered {
        if predefined_ok {
            return SeqTableMode::Predefined;
        }
        // Last resort: can't encode
        return SeqTableMode::Predefined;
    }

    let header_bytes = encode_fse_header(&custom_norm, max_sym, table_log);

    // Estimate costs: predefined vs custom using cross-entropy against each distribution
    let total = codes.len() as f64;
    let default_table_size = (1u64 << default_log) as f64;

    // Predefined cost: sum of -log2(predefined_prob(s)) for each symbol occurrence
    let predefined_bits = if predefined_ok {
        let mut bits = 0.0f64;
        for s in 0..=max_sym {
            if counts[s] > 0 {
                let norm_val = default_norm[s];
                let prob = if norm_val == -1 { 1.0 } else { norm_val as f64 };
                // Each symbol costs approximately log2(table_size / prob) bits
                bits += counts[s] as f64 * (default_table_size / prob).log2();
            }
        }
        // Add FSE state overhead (table_log bits for initial states × 3 tables shared)
        (bits.ceil() as u64).saturating_add(default_log as u64)
    } else {
        u64::MAX
    };

    // Custom cost: header + Shannon entropy (lower bound on custom FSE cost)
    let custom_header_bits = header_bytes.len() as u64 * 8;
    let mut custom_stream_bits = 0.0f64;
    for s in 0..=max_sym {
        if counts[s] > 0 {
            let p = counts[s] as f64 / total;
            custom_stream_bits += counts[s] as f64 * (-p.log2());
        }
    }
    // Add table_log bits for initial state + rounding overhead
    let custom_total_bits = custom_header_bits + custom_stream_bits.ceil() as u64 + table_log as u64 + 8;

    if predefined_ok && predefined_bits <= custom_total_bits {
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
fn write_seq_table_and_build(
    out: &mut Vec<u8>,
    mode: &SeqTableMode,
    default_norm: &[i16],
    default_max_symbol: usize,
    default_log: u32,
) -> super::fse::FseCTable {
    match mode {
        SeqTableMode::Predefined => {
            super::fse::FseCTable::build(default_norm, default_max_symbol, default_log)
        }
        SeqTableMode::Rle(sym) => {
            out.push(*sym);
            // RLE table: single symbol, 0 bits per symbol, accuracy_log=0
            super::fse::FseCTable::build_rle(*sym)
        }
        SeqTableMode::Fse {
            norm,
            max_symbol,
            table_log,
            header_bytes,
        } => {
            out.extend_from_slice(header_bytes);
            super::fse::FseCTable::build(norm, *max_symbol, *table_log)
        }
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
    }

    #[test]
    fn compress_small() {
        let data = b"hello world";
        let compressed = compress(data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
        assert!(compressed.len() > 5);
    }

    #[test]
    fn compress_repetitive() {
        let data = vec![42u8; 4096];
        let compressed = compress(&data, 1);
        // Valid zstd frame (raw blocks are larger than input due to framing)
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
    }

    #[test]
    fn compress_real_data() {
        let data: Vec<u8> = (0..1024u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let compressed = compress(&data, 1);
        assert_eq!(&compressed[..4], &ZSTD_MAGIC.to_le_bytes());
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
            let compressed = compress(data, 1);
            let decompressed = crate::decompress(&compressed)
                .unwrap_or_else(|e| panic!("{}: decompress failed: {}", name, e));

            assert_eq!(decompressed.len(), data.len(), "{}: length mismatch", name);
            assert_eq!(&decompressed, data, "{}: data mismatch", name);
        }
    }
}
