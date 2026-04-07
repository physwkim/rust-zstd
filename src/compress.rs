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

    // Cross-block state for Treeless Huffman and Repeat FSE
    let mut prev_huf_codes: Option<([(u32, u8); 256], u8)> = None;
    let mut prev_ll_mode: Option<SeqTableMode> = None;
    let mut prev_of_mode: Option<SeqTableMode> = None;
    let mut prev_ml_mode: Option<SeqTableMode> = None;

    for (bi, &(s_start, s_end, d_end)) in block_ranges.iter().enumerate() {
        let is_last = bi == n_blocks - 1;
        let block_seqs = &all_encoded[s_start..s_end];
        let raw_seqs = &all_sequences[s_start..s_end];

        // Determine the byte range for this block
        let d_start = if s_start == 0 {
            0
        } else {
            let mut p = 0usize;
            for s in &all_sequences[..s_start] {
                p += s.ll as usize + s.ml as usize;
            }
            p
        };
        let block_data = &data[d_start..d_end];

        if block_seqs.is_empty() {
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
        literals.extend_from_slice(&block_data[pos..]);

        // Encode block
        let mut block = Vec::with_capacity(block_data.len());

        // === Literals section: try Huffman with Treeless, fall back to new tree ===
        let mut used_huf = false;
        if literals.len() >= 64 {
            // Build a new Huffman tree for this block's literals
            let new_result = encode_literals_huffman(&literals);

            // Try Treeless: reuse previous tree (saves tree description bytes)
            let treeless_result = if let Some((prev, _pm)) = &prev_huf_codes {
                let mut counts = [0u32; 256];
                let mut max_sym = 0u8;
                for &b in literals.iter() {
                    counts[b as usize] += 1;
                    if b > max_sym { max_sym = b; }
                }
                let all_covered = (0..=max_sym as usize).all(|s| counts[s] == 0 || prev[s].1 > 0);
                if all_covered {
                    encode_literals_treeless(&literals, prev)
                } else {
                    None
                }
            } else {
                None
            };

            // Pick the smaller of: new tree vs treeless vs raw
            match (&new_result, &treeless_result) {
                (Some(new_enc), Some(treeless_enc)) => {
                    if treeless_enc.len() <= new_enc.len() && treeless_enc.len() < literals.len() {
                        block.extend_from_slice(treeless_enc);
                        // ALWAYS update prev with new tree so next block has fresh comparison
                        update_prev_huf(&literals, &mut prev_huf_codes);
                        used_huf = true;
                    } else if new_enc.len() < literals.len() {
                        block.extend_from_slice(new_enc);
                        update_prev_huf(&literals, &mut prev_huf_codes);
                        used_huf = true;
                    }
                }
                (Some(new_enc), None) => {
                    if new_enc.len() < literals.len() {
                        block.extend_from_slice(new_enc);
                        update_prev_huf(&literals, &mut prev_huf_codes);
                        used_huf = true;
                    }
                }
                (None, Some(treeless_enc)) => {
                    if treeless_enc.len() < literals.len() {
                        block.extend_from_slice(treeless_enc);
                        used_huf = true;
                    }
                }
                (None, None) => {}
            }
        }
        if !used_huf {
            encode_literals_raw(&mut block, &literals);
        }

        encode_sequences_section_with_reuse(
            &mut block, block_seqs,
            &mut prev_ll_mode, &mut prev_of_mode, &mut prev_ml_mode,
        );

        // Choose smallest block type
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
    hash_bytes: usize,    // 4, 5, 6, or 7 — number of bytes used by hash function
    lazy_depth: u32,      // 0=greedy, 1=lazy, 2=lazy2
    search_depth: u32,    // hash chain search depth
}

impl MatchParams {
    /// Parameters aligned with C zstd compression parameters.
    /// C zstd level 1: hashLog=14, minMatch=7, strategy=fast (7-byte hash)
    /// C zstd level 3: hashLog=17, minMatch=6, strategy=dfast (6-byte hash)
    /// C zstd level 7+: hashLog=19+, minMatch=5, strategy=lazy/lazy2
    ///
    /// Key: longer hash → fewer but higher-quality matches → less sequence
    /// overhead per byte. C level 1's 7-byte hash intentionally skips short matches.
    fn from_level(level: i32) -> Self {
        match level {
            0..=2 => Self {
                hash_log: 14,      // C zstd level 1 uses hashLog=14
                hash_bytes: 7,     // 7-byte hash like C zstd level 1 (minMatch=7)
                lazy_depth: 0,
                search_depth: 4,
            },
            3..=5 => Self {
                hash_log: 18,
                hash_bytes: 5,
                lazy_depth: 1,
                search_depth: 16,
            },
            6..=8 => Self {
                hash_log: 19,
                hash_bytes: 5,
                lazy_depth: 1,
                search_depth: 64,
            },
            _ => Self {
                hash_log: 20,
                hash_bytes: 5,
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

/// Match finder with rep-code integration and lazy evaluation.
/// All levels use hash-chain matching with rep-code chaining after matches.
fn find_matches(data: &[u8], params: &MatchParams) -> Vec<Sequence> {
    find_matches_lazy(data, params)
}

/// Greedy fast match finder modeled on ZSTD_compressBlock_fast_noDict_generic.
/// Single hash table lookup (no chains), rep-code priority, step increment.
fn find_matches_fast(data: &[u8], params: &MatchParams) -> Vec<Sequence> {
    const MIN_INPUT: usize = 8;
    if data.len() < MIN_INPUT {
        return vec![];
    }

    let hlog = params.hash_log;
    let hash_size = 1usize << hlog;
    let mls = params.hash_bytes; // minimum match length for hash
    let mut ht = vec![0u32; hash_size]; // hash table: hash → position
    let mut sequences = Vec::new();

    let ilimit = data.len() - MIN_INPUT;
    let mut anchor = 0usize;
    let mut ip0 = 0usize;

    let mut rep1 = 0u32; // most recent offset
    let mut rep2 = 0u32; // second most recent offset

    const K_SEARCH_STRENGTH: u32 = 8;
    const K_STEP_INCR: usize = 1 << (K_SEARCH_STRENGTH - 1); // 128

    'outer: loop {
        let mut step: usize = 2; // initial step between search pairs
        let mut next_step = ip0 + K_STEP_INCR;

        let mut ip1 = ip0 + 1;

        if ip1 > ilimit { break; }

        // Pre-hash ip0
        let mut h0 = hash_n(&data[ip0..], (hash_size - 1) as u32, mls);

        loop {
            let ip2 = ip0 + step;
            let ip3 = ip1 + step;
            if ip3 > ilimit { break 'outer; }

            // Get match candidate from hash table for ip0
            let match_idx0 = ht[h0] as usize;

            // Hash ip1, write ip0 to hash table
            let h1 = hash_n(&data[ip1..], (hash_size - 1) as u32, mls);
            ht[h0] = ip0 as u32;

            // === Rep-code check at ip2 (before hash match) ===
            if rep1 > 0 && ip2 >= rep1 as usize {
                let rep_cand = ip2 - rep1 as usize;
                if rep_cand + 4 <= data.len() && ip2 + 4 <= data.len()
                    && read32(data, ip2) == read32(data, rep_cand)
                {
                    // Rep match at ip2! Write hash for ip1 first
                    ht[h1] = ip1 as u32;

                    let mut mlen = 4 + count_match(data, ip2 + 4, rep_cand + 4);
                    // Backward extension
                    let mut start = ip2;
                    let mut mstart = rep_cand;
                    while start > anchor && mstart > 0 && mlen < MAX_MATCH_LEN && data[start - 1] == data[mstart - 1] {
                        start -= 1;
                        mstart -= 1;
                        mlen += 1;
                    }

                    let ll = (start - anchor) as u32;
                    sequences.push(Sequence { ll, off: rep1, ml: mlen as u32 });
                    ip0 = start + mlen;
                    anchor = ip0;

                    // Rep-code chaining: check rep2 at new position
                    rep_chain(data, &mut ip0, &mut anchor, &mut sequences,
                              &mut rep1, &mut rep2, &mut ht, hlog, mls, ilimit);
                    continue 'outer;
                }
            }

            // === Hash match check at ip0 ===
            if match_idx0 < ip0 && ip0 - match_idx0 <= (1 << 24)
                && match_idx0 + 4 <= data.len()
                && read32(data, ip0) == read32(data, match_idx0)
            {
                ht[h1] = ip1 as u32;
                let match0 = match_idx0;

                let mut mlen = 4 + count_match(data, ip0 + 4, match0 + 4);
                let mut start = ip0;
                let mut mstart = match0;
                while start > anchor && mstart > 0 && data[start - 1] == data[mstart - 1] {
                    start -= 1;
                    mstart -= 1;
                    mlen += 1;
                }

                let offset = (start - mstart) as u32;
                rep2 = rep1;
                rep1 = offset;

                let ll = (start - anchor) as u32;
                sequences.push(Sequence { ll, off: offset, ml: mlen as u32 });
                ip0 = start + mlen;
                anchor = ip0;

                // Fill hash table for end-of-match positions
                if ip0 > 2 && ip0 + MIN_INPUT <= data.len() {
                    ht[hash_n(&data[ip0 - 2..], (hash_size - 1) as u32, mls)] = (ip0 - 2) as u32;
                }

                rep_chain(data, &mut ip0, &mut anchor, &mut sequences,
                          &mut rep1, &mut rep2, &mut ht, hlog, mls, ilimit);
                continue 'outer;
            }

            // === No match at ip0 — check ip1 ===
            let match_idx1 = ht[h1] as usize;
            h0 = hash_n(&data[ip2..], (hash_size - 1) as u32, mls);
            ht[h1] = ip1 as u32;

            if match_idx1 < ip1 && ip1 - match_idx1 <= (1 << 24)
                && match_idx1 + 4 <= data.len()
                && read32(data, ip1) == read32(data, match_idx1)
            {
                let match0 = match_idx1;
                let mut mlen = 4 + count_match(data, ip1 + 4, match0 + 4);
                let mut start = ip1;
                let mut mstart = match0;
                while start > anchor && mstart > 0 && data[start - 1] == data[mstart - 1] {
                    start -= 1;
                    mstart -= 1;
                    mlen += 1;
                }

                let offset = (start - mstart) as u32;
                rep2 = rep1;
                rep1 = offset;

                let ll = (start - anchor) as u32;
                sequences.push(Sequence { ll, off: offset, ml: mlen as u32 });
                ip0 = start + mlen;
                anchor = ip0;

                if ip0 > 2 && ip0 + MIN_INPUT <= data.len() {
                    ht[hash_n(&data[ip0 - 2..], (hash_size - 1) as u32, mls)] = (ip0 - 2) as u32;
                }

                rep_chain(data, &mut ip0, &mut anchor, &mut sequences,
                          &mut rep1, &mut rep2, &mut ht, hlog, mls, ilimit);
                continue 'outer;
            }

            // No match at ip0 or ip1 — advance with step
            ip0 = ip2;
            ip1 = ip3;

            // Step increment: search less densely in non-matching regions
            if ip0 >= next_step {
                step += 1;
                next_step += K_STEP_INCR;
            }
        }
    }

    sequences
}

/// Rep-code chaining: after a match, check if the next position matches rep2.
#[inline]
fn rep_chain(
    data: &[u8], ip: &mut usize, anchor: &mut usize,
    sequences: &mut Vec<Sequence>,
    rep1: &mut u32, rep2: &mut u32,
    ht: &mut [u32], hlog: u32, mls: usize, _ilimit: usize,
) {
    let hash_size = 1usize << hlog;
    while *rep2 > 0 && *ip + 4 <= data.len() && *ip >= *rep2 as usize {
        let cand = *ip - *rep2 as usize;
        if cand + 4 > data.len() || read32(data, *ip) != read32(data, cand) {
            break;
        }
        let mlen = 4 + count_match(data, *ip + 4, cand + 4);
        // Swap rep codes
        let tmp = *rep2;
        *rep2 = *rep1;
        *rep1 = tmp;

        // Update hash table
        if *ip + 8 <= data.len() {
            ht[hash_n(&data[*ip..], (hash_size - 1) as u32, mls)] = *ip as u32;
        }

        sequences.push(Sequence { ll: 0, off: *rep1, ml: mlen as u32 });
        *ip += mlen;
        *anchor = *ip;
    }
}

#[inline]
fn read32(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]])
}

/// Max match length for ML code 52: baseline 65539 + (1<<16)-1 = 131074
const MAX_MATCH_LEN: usize = 131074;

#[inline]
fn count_match(data: &[u8], mut a: usize, mut b: usize) -> usize {
    let start = a;
    let max_extend = MAX_MATCH_LEN - 4;
    let limit = std::cmp::min(data.len(), start + max_extend);
    while a + 8 <= limit && b + 8 <= data.len() {
        let va = u64::from_le_bytes(data[a..a+8].try_into().unwrap());
        let vb = u64::from_le_bytes(data[b..b+8].try_into().unwrap());
        if va != vb {
            return a - start + (va ^ vb).trailing_zeros() as usize / 8;
        }
        a += 8;
        b += 8;
    }
    while a < limit && b < data.len() && data[a] == data[b] {
        a += 1;
        b += 1;
    }
    a - start
}

/// Dual-hash match finder with rep-code priority and optional lazy evaluation.
/// Uses two hash tables: short (4-byte) for nearby matches and long (7-byte)
/// for long-range matches. Takes the best match from either table.
/// - Rep-code matches checked first at each position (free offset cost)
/// - After each match, rep-code chaining checks rep2 immediately
/// - Lazy evaluation (level 3+) checks ip+1 for better matches
fn find_matches_lazy(data: &[u8], params: &MatchParams) -> Vec<Sequence> {
    if data.len() < 8 {
        return vec![];
    }

    let hash_size = 1usize << params.hash_log;
    let hash_mask = (hash_size - 1) as u32;
    let long_hash_size = 1usize << std::cmp::min(params.hash_log, 17);
    let long_hash_mask = (long_hash_size - 1) as u32;
    let mut ht_short = vec![0u32; hash_size];     // 4-byte hash → position
    let mut ht_long = vec![0u32; long_hash_size]; // 7-byte hash → position
    let mut chain = vec![0u32; data.len()];
    let mut sequences = Vec::new();
    let mut anchor = 0usize;
    let mut ip = 0usize;
    let mut rep1 = 0u32;
    let mut rep2 = 0u32;
    let lazy = params.lazy_depth >= 1;

    while ip + 8 <= data.len() {
        // === 1. Rep-code check (near-free offset, ~12 bits overhead) ===
        // Rep saves offset bits but still has LL+ML FSE overhead (~12 bits).
        // A rep match of N bytes saves N*5 bits (literals), costs ~12 bits → profitable if N >= 3.
        // But at N=4-5 with ll=2, the net savings are thin. Require N >= ZSTD_MINMATCH (always true).
        let rep_match = if rep1 > 0 && ip >= rep1 as usize && ip + 4 <= data.len() {
            let cand = ip - rep1 as usize;
            if cand + 4 <= data.len() && read32(data, ip) == read32(data, cand) {
                let ml = 4 + count_match(data, ip + 4, cand + 4);
                // Rep overhead: ~12 bits (LL FSE + ML FSE, no offset bits)
                // Savings: ml * 5 bits. Net positive if ml*5 > 20 → ml > 4
                // Rep overhead: ~12 bits for LL+ML FSE.
                // Savings: ml * literal_cost (~5 bits/byte).
                // Net benefit must exceed per-sequence fixed cost to be worthwhile.
                // For high-entropy data, a 6-byte rep match at ll=2 is marginal.
                // Require ml >= 7 to skip borderline short matches that
                // produce too many sequences (key insight from C zstd analysis).
                if ml * 5 > 40 { Some((rep1 as usize, ml)) } else { None }
            } else { None }
        } else { None };

        // === 2. Hash-chain match using configured hash_bytes ===
        let short_match = find_best_at_n(data, ip, &ht_short, &chain, hash_mask, params.search_depth, std::cmp::min(params.hash_bytes, 4));

        // === 3. Long hash match (7-byte, single lookup, for long-range) ===
        let long_match = if ip + 7 <= data.len() {
            let lh = hash7(&data[ip..], long_hash_mask);
            let lidx = ht_long[lh] as usize;
            ht_long[lh] = ip as u32;
            if lidx < ip && ip - lidx <= (1 << 24) && lidx + 4 <= data.len()
                && read32(data, ip) == read32(data, lidx)
            {
                let ml = 4 + count_match(data, ip + 4, lidx + 4);
                Some((ip - lidx, ml))
            } else { None }
        } else { None };

        // === 4. Pick best overall ===
        // Filter hash matches for profitability (rep matches are always free)
        let short_match = short_match.filter(|&(off, ml)| is_match_profitable(ml, off));
        let long_match = long_match.filter(|&(off, ml)| is_match_profitable(ml, off));

        let best_hash = match (short_match, long_match) {
            (Some((so, sl)), Some((lo, ll))) => {
                if ll >= sl + 2 { Some((lo, ll)) } else { Some((so, sl)) }
            }
            (Some(s), None) => Some(s),
            (None, Some(l)) => Some(l),
            (None, None) => None,
        };

        let chosen = match (rep_match, best_hash) {
            (Some((roff, rml)), Some((hoff, hml))) => {
                let off_bits = 32u32.saturating_sub((hoff as u32).leading_zeros());
                if rml + (off_bits as usize / 4) >= hml { Some((roff, rml)) }
                else { Some((hoff, hml)) }
            }
            (Some(r), None) => Some(r),
            (None, Some(h)) => Some(h),
            (None, None) => None,
        };

        if let Some((offset, match_len)) = chosen {
            let mut final_off = offset;
            let mut final_len = match_len;
            let mut final_ip = ip;

            // Lazy: check ip+1 for better match
            if lazy && ip + 1 + 8 <= data.len() {
                insert_hash_n(&mut ht_short, &mut chain, data, ip, hash_mask, 4);
                let mut next_best = None;
                // Rep at ip+1
                if rep1 > 0 && ip + 1 >= rep1 as usize && ip + 5 <= data.len() {
                    let c = ip + 1 - rep1 as usize;
                    if c + 4 <= data.len() && read32(data, ip + 1) == read32(data, c) {
                        let rl = 4 + count_match(data, ip + 5, c + 4);
                        if rl > final_len { next_best = Some((rep1 as usize, rl)); }
                    }
                }
                // Hash chain at ip+1
                if let Some((off2, len2)) = find_best_at_n(
                    data, ip + 1, &ht_short, &chain, hash_mask, params.search_depth, 4,
                ) {
                    if len2 > final_len + 1 && (next_best.is_none() || len2 > next_best.unwrap().1) {
                        next_best = Some((off2, len2));
                    }
                }
                if let Some((off2, len2)) = next_best {
                    if len2 > final_len + 1 {
                        final_off = off2;
                        final_len = len2;
                        final_ip = ip + 1;
                    }
                }
            }

            let ll = (final_ip - anchor) as u32;
            sequences.push(Sequence { ll, off: final_off as u32, ml: final_len as u32 });

            if final_off as u32 != rep1 {
                rep2 = rep1;
                rep1 = final_off as u32;
            }

            let end = std::cmp::min(final_ip + final_len, data.len().saturating_sub(4));
            for p in ip..end {
                insert_hash_n(&mut ht_short, &mut chain, data, p, hash_mask, 4);
                if p + 7 <= data.len() {
                    ht_long[hash7(&data[p..], long_hash_mask)] = p as u32;
                }
            }

            ip = final_ip + final_len;
            anchor = ip;

            // === Rep-code chaining: check rep2 ===
            while rep2 > 0 && ip + 4 <= data.len() && ip >= rep2 as usize {
                let cand = ip - rep2 as usize;
                if cand + 4 > data.len() || read32(data, ip) != read32(data, cand) { break; }
                let mlen = 4 + count_match(data, ip + 4, cand + 4);
                if mlen < ZSTD_MINMATCH { break; }

                let tmp = rep2; rep2 = rep1; rep1 = tmp;
                sequences.push(Sequence { ll: 0, off: rep1, ml: mlen as u32 });

                let end2 = std::cmp::min(ip + mlen, data.len().saturating_sub(4));
                for p in ip..end2 {
                    insert_hash_n(&mut ht_short, &mut chain, data, p, hash_mask, 4);
                    if p + 7 <= data.len() {
                        ht_long[hash7(&data[p..], long_hash_mask)] = p as u32;
                    }
                }
                ip += mlen;
                anchor = ip;
            }
        } else {
            insert_hash_n(&mut ht_short, &mut chain, data, ip, hash_mask, 4);
            if ip + 7 <= data.len() {
                ht_long[hash7(&data[ip..], long_hash_mask)] = ip as u32;
            }
            ip += 1;
        }
    }

    sequences
}

/// Check if a match at `offset` of `length` bytes saves more than it costs.
///
/// Key insight from C zstd analysis: each sequence has ~17+ bits of FSE overhead
/// (LL state + OF state + ML state + offset extra bits). A match only "saves"
/// bytes that would otherwise be literals. Literals compress well with Huffman
/// (~5-6 bits/byte for typical data). So a match of N bytes saves ~N*5 bits
/// but costs ~17+offset_code bits.
///
/// For match_len=6, offset=8: saves 30 bits, costs ~20 bits → marginal
/// For match_len=4, offset=1000: saves 20 bits, costs ~27 bits → unprofitable!
#[inline]
fn is_match_profitable(match_len: usize, offset: usize) -> bool {
    let off_code = if offset > 1 { 32 - (offset as u32).leading_zeros() } else { 1 };
    // Sequence overhead: ~6 (LL) + 5 (OF_FSE) + off_code (OF extra) + 6 (ML) = 17 + off_code bits
    // Match saves: match_len * literal_bits_per_byte
    // Use literal cost of 5 bits/byte (conservative Huffman estimate)
    let overhead_bits = 17 + off_code;
    let savings_bits = match_len as u32 * 5;
    savings_bits > overhead_bits + 8 // require >1 byte net benefit
}

/// After a match ends, immediately check for a rep-code match at the current
/// position using rep[1] (the second repeat offset). This chains consecutive
/// matches without re-entering the main loop, matching C zstd behavior.
fn chain_rep_matches(
    data: &[u8],
    ip: &mut usize,
    anchor: &mut usize,
    sequences: &mut Vec<Sequence>,
    rep: &mut [u32; 3],
    hash_table: &mut [u32],
    chain: &mut [u32],
    hash_mask: u32,
    hash_bytes: usize,
) {
    // Try rep[1] (second most recent offset) for continuation match.
    // This is very cheap: ll=0, offset=rep[1] (encoded as rep-code 2).
    loop {
        if rep[1] == 0 { break; }
        let rep_off = rep[1] as usize;
        if *ip < rep_off || *ip + 4 > data.len() { break; }
        let cand = *ip - rep_off;
        if cand + 4 > data.len() { break; }
        if data[*ip..*ip + 4] != data[cand..cand + 4] { break; }

        let max_ml = std::cmp::min(ZSTD_BLOCKSIZE_MAX, std::cmp::min(data.len() - *ip, data.len() - cand));
        let ml = common_prefix_len(&data[cand..cand + max_ml], &data[*ip..*ip + max_ml]);
        if ml < ZSTD_MINMATCH { break; }

        sequences.push(Sequence { ll: 0, off: rep_off as u32, ml: ml as u32 });
        *rep = [rep[1], rep[0], rep[2]];

        let end = std::cmp::min(*ip + ml, data.len().saturating_sub(hash_bytes));
        for p in *ip..end { insert_hash_n(hash_table, chain, data, p, hash_mask, hash_bytes); }
        *ip += ml;
        *anchor = *ip;
    }
}

/// Insert position into hash chain using N-byte hash.
#[inline]
fn insert_hash_n(hash_table: &mut [u32], chain: &mut [u32], data: &[u8], pos: usize, mask: u32, hash_bytes: usize) {
    if pos + hash_bytes > data.len() {
        return;
    }
    let h = hash_n(&data[pos..], mask, hash_bytes);
    chain[pos] = hash_table[h];
    hash_table[h] = pos as u32;
}

/// Insert position into hash chain (4-byte hash, used by lazy lookups).
#[inline]
fn insert_hash(hash_table: &mut [u32], chain: &mut [u32], data: &[u8], pos: usize, mask: u32) {
    insert_hash_n(hash_table, chain, data, pos, mask, 4);
}

/// N-byte hash function. Uses longer hashes for fewer collisions.
#[inline]
fn hash_n(data: &[u8], mask: u32, n: usize) -> usize {
    match n {
        5 => hash5(data, mask),
        6 => hash6(data, mask),
        7 => hash7(data, mask),
        _ => hash4(data, mask),
    }
}

/// 6-byte hash
#[inline]
fn hash6(data: &[u8], mask: u32) -> usize {
    let v = u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], 0, 0]);
    ((v.wrapping_mul(227718039650203u64)) >> 24) as usize & mask as usize
}

/// 7-byte hash matching C zstd's ZSTD_hash7Ptr (prime=58295818150454627).
/// Critical for f64 data where first 4-6 bytes are often identical (0x00).
#[inline]
fn hash7(data: &[u8], mask: u32) -> usize {
    let v = u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], data[6], 0]);
    ((v.wrapping_mul(58295818150454627u64)) >> 24) as usize & mask as usize
}

/// Find the best match at `pos` by walking the hash chain.
fn find_best_at_n(
    data: &[u8],
    pos: usize,
    hash_table: &[u32],
    chain: &[u32],
    mask: u32,
    max_depth: u32,
    hash_bytes: usize,
) -> Option<(usize, usize)> {
    if pos + hash_bytes > data.len() {
        return None;
    }
    let h = hash_n(&data[pos..], mask, hash_bytes);
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

/// 5-byte multiplicative hash for better collision avoidance.
/// Matches C zstd's hash5 using prime 889523592379.
#[inline]
fn hash5(data: &[u8], mask: u32) -> usize {
    let v = u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], 0, 0, 0]);
    ((v.wrapping_mul(889523592379u64)) >> 24) as usize & mask as usize
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

/// Update prev_huf_codes from current literals
fn update_prev_huf(literals: &[u8], prev: &mut Option<([(u32, u8); 256], u8)>) {
    let mut counts = [0u32; 256];
    let mut max_sym = 0u8;
    for &b in literals { counts[b as usize] += 1; if b > max_sym { max_sym = b; } }
    if let Some((codes, mb)) = build_huffman_codes(&counts, max_sym as usize) {
        *prev = Some((codes, mb));
    }
}

/// Encode literals using a previous Huffman tree (Treeless_Literals_Block, type=3).
fn encode_literals_treeless(literals: &[u8], prev_codes: &[(u32, u8); 256]) -> Option<Vec<u8>> {
    let use_4 = literals.len() >= 1024;
    let streams = if use_4 {
        encode_huf_4streams(literals, prev_codes)
    } else {
        encode_huf_1stream(literals, prev_codes)
    };

    let regen = literals.len();
    let comp = streams.len(); // no tree description for treeless!
    let lh_size = 3 + (regen >= 1024) as usize + (regen >= 16384) as usize;

    let mut out = Vec::with_capacity(lh_size + comp);
    let htype = LIT_TYPE_TREELESS as u32;

    match lh_size {
        3 => {
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
    out.extend_from_slice(&streams);
    Some(out)
}

/// Encode literals with Huffman, optionally reusing previous tree (Treeless mode).
/// Returns (encoded_bytes, codes, max_bits) on success.
fn encode_literals_huffman_with_reuse(
    literals: &[u8],
    prev_codes: &Option<([(u32, u8); 256], u8)>,
) -> Option<(Vec<u8>, [(u32, u8); 256], u8)> {
    // Count frequencies
    let mut counts = [0u32; 256];
    let mut max_sym = 0u8;
    for &b in literals {
        counts[b as usize] += 1;
        if b > max_sym { max_sym = b; }
    }
    let n_used = counts.iter().filter(|&&c| c > 0).count();
    if n_used < 2 { return None; }

    // Build new Huffman codes
    let (new_codes, new_max_bits) = build_huffman_codes(&counts, max_sym as usize)?;
    let new_tree_desc = encode_huffman_tree(&new_codes, new_max_bits, max_sym as usize);
    if new_tree_desc.is_empty() { return None; }

    // Try both: new tree (Compressed) and reused tree (Treeless)
    let use_4 = literals.len() >= 1024;

    // Option A: new tree (type=2)
    let streams_new = if use_4 {
        encode_huf_4streams(literals, &new_codes)
    } else {
        encode_huf_1stream(literals, &new_codes)
    };
    let comp_new = new_tree_desc.len() + streams_new.len();

    // Option B: treeless reuse (type=3) if previous tree exists and covers all symbols
    let mut best_type = LIT_TYPE_COMPRESSED;
    let mut best_codes = new_codes;
    let mut best_max_bits = new_max_bits;
    let mut best_streams = streams_new;
    let best_comp = comp_new;
    let mut best_tree_desc = new_tree_desc;

    if let Some((prev, prev_mb)) = prev_codes {
        // Check: does prev tree cover all symbols in current literals?
        let all_covered = (0..=max_sym as usize).all(|s| counts[s] == 0 || prev[s].1 > 0);
        if all_covered {
            let streams_reuse = if use_4 {
                encode_huf_4streams(literals, prev)
            } else {
                encode_huf_1stream(literals, prev)
            };
            let comp_reuse = streams_reuse.len(); // no tree desc!
            if comp_reuse < best_comp {
                best_type = LIT_TYPE_TREELESS;
                best_codes = *prev;
                best_max_bits = *prev_mb;
                best_streams = streams_reuse;
                let _ = comp_reuse;
                best_tree_desc = vec![]; // no tree header for treeless
            }
        }
    }

    let regen = literals.len();
    let comp = best_tree_desc.len() + best_streams.len();
    let lh_size = 3 + (regen >= 1024) as usize + (regen >= 16384) as usize;

    let mut out = Vec::with_capacity(lh_size + comp);
    let htype = best_type as u32;

    match lh_size {
        3 => {
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

    out.extend_from_slice(&best_tree_desc);
    out.extend_from_slice(&best_streams);
    Some((out, best_codes, best_max_bits))
}

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

/// Encode sequences with cross-block Repeat mode support.
/// If the current block's symbol distribution matches the previous block, use Repeat mode
/// (no table header needed). Otherwise choose best of Predefined/RLE/Custom FSE.
fn encode_sequences_section_with_reuse(
    out: &mut Vec<u8>,
    sequences: &[EncodedSequence],
    prev_ll: &mut Option<SeqTableMode>,
    prev_of: &mut Option<SeqTableMode>,
    prev_ml: &mut Option<SeqTableMode>,
) {
    // Encode without reuse first
    let mut no_reuse = Vec::new();
    encode_sequences_section(&mut no_reuse, sequences);

    // Try encoding with Repeat mode if previous tables exist
    if prev_ll.is_some() || prev_of.is_some() || prev_ml.is_some() {
        let mut with_reuse = Vec::new();
        encode_sequences_section_repeat(&mut with_reuse, sequences, prev_ll, prev_of, prev_ml);

        if with_reuse.len() < no_reuse.len() {
            out.extend_from_slice(&with_reuse);
            // Keep prev modes for next block (they were reused successfully)
            return;
        }
    }

    // No reuse or reuse was worse: use non-reuse version and update prev modes
    out.extend_from_slice(&no_reuse);

    // Update prev modes from the non-reuse encoding's chosen modes
    update_prev_modes(sequences, prev_ll, prev_of, prev_ml);
}

/// Encode sequences using Repeat mode for all three tables.
fn encode_sequences_section_repeat(
    out: &mut Vec<u8>,
    sequences: &[EncodedSequence],
    prev_ll: &Option<SeqTableMode>,
    prev_of: &Option<SeqTableMode>,
    prev_ml: &Option<SeqTableMode>,
) {
    let nb_seq = sequences.len();
    if nb_seq < 128 { out.push(nb_seq as u8); }
    else if nb_seq < 0x7F00 { out.push(((nb_seq >> 8) as u8) + 128); out.push(nb_seq as u8); }
    else { out.push(255); out.extend_from_slice(&((nb_seq - 0x7F00) as u16).to_le_bytes()); }
    if nb_seq == 0 { return; }

    let mut ll_codes_v = Vec::with_capacity(nb_seq);
    let mut ml_codes_v = Vec::with_capacity(nb_seq);
    let mut off_codes_v = Vec::with_capacity(nb_seq);
    let mut ll_values = Vec::with_capacity(nb_seq);
    let mut ml_values = Vec::with_capacity(nb_seq);
    let mut off_values = Vec::with_capacity(nb_seq);

    for seq in sequences {
        let llc = ll_code(seq.ll);
        let mlc = ml_code(seq.ml - ZSTD_MINMATCH as u32);
        let ofc = off_code(seq.of_value);
        ll_codes_v.push(llc);
        ml_codes_v.push(mlc);
        off_codes_v.push(ofc);
        ll_values.push(seq.ll - LL_BASE[llc as usize]);
        ml_values.push(seq.ml - ML_BASE[mlc as usize]);
        off_values.push(if ofc > 0 { seq.of_value - (1u32 << ofc) } else { 0 });
    }

    // Use Repeat for tables that have a previous mode, otherwise use best fresh mode
    let ll_mode = if can_repeat(&ll_codes_v, prev_ll) { SeqTableMode::Repeat }
        else { choose_seq_mode(&ll_codes_v, MAX_LL, LL_DEFAULT_NORM_LOG, &LL_DEFAULT_NORM, LL_FSE_LOG) };
    let of_mode = if can_repeat(&off_codes_v, prev_of) { SeqTableMode::Repeat }
        else { choose_seq_mode(&off_codes_v, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG, &OF_DEFAULT_NORM, OFF_FSE_LOG) };
    let ml_mode = if can_repeat(&ml_codes_v, prev_ml) { SeqTableMode::Repeat }
        else { choose_seq_mode(&ml_codes_v, MAX_ML, ML_DEFAULT_NORM_LOG, &ML_DEFAULT_NORM, ML_FSE_LOG) };

    let mode_byte = (ll_mode.tag() << 6) | (of_mode.tag() << 4) | (ml_mode.tag() << 2);
    out.push(mode_byte);

    let ll_table = write_seq_table_and_build(out, &ll_mode, prev_ll, &LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
    let of_table = write_seq_table_and_build(out, &of_mode, prev_of, &OF_DEFAULT_NORM, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG);
    let ml_table = write_seq_table_and_build(out, &ml_mode, prev_ml, &ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);

    let bitstream = super::fse::encode_sequences(
        &ll_table, &of_table, &ml_table,
        &ll_codes_v, &off_codes_v, &ml_codes_v,
        &ll_values, &ml_values, &off_values,
    );
    out.extend_from_slice(&bitstream);
}

fn can_repeat(codes: &[u8], prev: &Option<SeqTableMode>) -> bool {
    let Some(prev_mode) = prev else { return false; };
    match prev_mode {
        SeqTableMode::Rle(sym) => codes.iter().all(|&c| c == *sym),
        SeqTableMode::Fse { norm, max_symbol, .. } => {
            codes.iter().all(|&c| (c as usize) <= *max_symbol && norm[c as usize] != 0)
        }
        SeqTableMode::Predefined => true, // predefined always valid for standard codes
        SeqTableMode::Repeat => false,
    }
}

fn update_prev_modes(
    sequences: &[EncodedSequence],
    prev_ll: &mut Option<SeqTableMode>,
    prev_of: &mut Option<SeqTableMode>,
    prev_ml: &mut Option<SeqTableMode>,
) {
    if sequences.is_empty() { return; }
    let mut ll_codes = Vec::with_capacity(sequences.len());
    let mut of_codes = Vec::with_capacity(sequences.len());
    let mut ml_codes = Vec::with_capacity(sequences.len());
    for seq in sequences {
        ll_codes.push(ll_code(seq.ll));
        ml_codes.push(ml_code(seq.ml - ZSTD_MINMATCH as u32));
        of_codes.push(off_code(seq.of_value));
    }
    *prev_ll = Some(choose_seq_mode(&ll_codes, MAX_LL, LL_DEFAULT_NORM_LOG, &LL_DEFAULT_NORM, LL_FSE_LOG));
    *prev_of = Some(choose_seq_mode(&of_codes, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG, &OF_DEFAULT_NORM, OFF_FSE_LOG));
    *prev_ml = Some(choose_seq_mode(&ml_codes, MAX_ML, ML_DEFAULT_NORM_LOG, &ML_DEFAULT_NORM, ML_FSE_LOG));
}

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
    let no_prev = None;
    let ll_table = write_seq_table_and_build(out, &ll_mode, &no_prev, &LL_DEFAULT_NORM, MAX_LL, LL_DEFAULT_NORM_LOG);
    let of_table = write_seq_table_and_build(out, &of_mode, &no_prev, &OF_DEFAULT_NORM, OF_DEFAULT_NORM.len() - 1, OF_DEFAULT_NORM_LOG);
    let ml_table = write_seq_table_and_build(out, &ml_mode, &no_prev, &ML_DEFAULT_NORM, MAX_ML, ML_DEFAULT_NORM_LOG);

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
    Repeat, // Reuse previous block's table (no header written)
}

impl SeqTableMode {
    fn tag(&self) -> u8 {
        match self {
            SeqTableMode::Predefined => SEQ_MODE_PREDEFINED,
            SeqTableMode::Rle(_) => SEQ_MODE_RLE,
            SeqTableMode::Fse { .. } => SEQ_MODE_FSE,
            SeqTableMode::Repeat => SEQ_MODE_REPEAT,
        }
    }
}

/// Normalize symbol counts to probability distribution for FSE table.
/// Port of C zstd's FSE_normalizeCount() with 62-bit precision scaling.
fn normalize_counts(counts: &[u32], max_symbol: usize, table_log: u32) -> Vec<i16> {
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
fn normalize_counts_m2(norm: &mut [i16], counts: &[u32], max_symbol: usize, table_log: u32, total: u64) {
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
    let remaining_total: u64 = counts[..=max_symbol].iter().enumerate()
        .filter(|&(s, _)| norm[s] == 0 && counts[s] > 0)
        .map(|(_, &c)| c as u64)
        .sum();

    if remaining_total == 0 || to_distribute <= 0 {
        return;
    }

    let v_step_log = 62u32.saturating_sub(table_log);
    let r_step = ((1u128 << v_step_log) * to_distribute as u128 + remaining_total as u128 / 2) / remaining_total as u128;

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

/// Cross-entropy cost of encoding `counts` using distribution `norm` at `table_log`.
/// Returns approximate total bits needed to encode all symbols.
fn cross_entropy_cost(norm: &[i16], table_log: u32, counts: &[u32; 256], max_sym: usize) -> u64 {
    let mut cost = 0u64;
    for s in 0..=max_sym {
        if counts[s] == 0 { continue; }
        if s >= norm.len() || norm[s] == 0 { return u64::MAX; }
        let prob = if norm[s] == -1 { 1u64 } else { norm[s] as u64 };
        // bits per symbol ≈ table_log - floor(log2(prob))
        let log2_prob = 63 - (prob as u64).leading_zeros() as u64;
        cost += counts[s] as u64 * (table_log as u64 - log2_prob);
    }
    cost + table_log as u64 // add state init cost
}

/// Choose best mode considering Repeat from previous block.
fn choose_seq_mode_with_repeat(
    codes: &[u8],
    max_symbol_default: usize,
    default_log: u32,
    default_norm: &[i16],
    max_log: u32,
    prev_mode: &Option<SeqTableMode>,
) -> SeqTableMode {
    // First get the best non-repeat mode
    let best = choose_seq_mode(codes, max_symbol_default, default_log, default_norm, max_log);

    // If we have a previous mode, check if Repeat is valid and cheaper
    if let Some(prev) = prev_mode {
        if codes.is_empty() {
            return best;
        }
        // Repeat is valid if the previous table can represent all current symbols
        let repeat_valid = match prev {
            SeqTableMode::Predefined => {
                codes.iter().all(|&c| {
                    let s = c as usize;
                    s <= max_symbol_default && s < default_norm.len() && default_norm[s] != 0
                })
            }
            SeqTableMode::Rle(sym) => codes.iter().all(|&c| c == *sym),
            SeqTableMode::Fse { norm, max_symbol, .. } => {
                codes.iter().all(|&c| {
                    let s = c as usize;
                    s <= *max_symbol && norm[s] != 0
                })
            }
            SeqTableMode::Repeat => false, // can't repeat a repeat
        };

        if repeat_valid {
            // Repeat saves header bytes but might use a worse distribution.
            // Only use Repeat if the previous table is FSE or RLE (custom).
            // Predefined→Repeat is pointless (Predefined also has no header).
            let worth_repeating = match prev {
                SeqTableMode::Fse { .. } | SeqTableMode::Rle(_) => true,
                _ => false,
            };
            if worth_repeating {
                let best_header_cost = match &best {
                    SeqTableMode::Predefined => 0,
                    SeqTableMode::Rle(_) => 1,
                    SeqTableMode::Fse { header_bytes, .. } => header_bytes.len(),
                    SeqTableMode::Repeat => 0,
                };
                // Only use Repeat if the current best mode requires a header
                if best_header_cost > 0 {
                    return SeqTableMode::Repeat;
                }
            }
        }
    }

    best
}

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
            let prob = if custom_norm[s] == -1 { 1u64 } else { custom_norm[s] as u64 };
            if prob == 0 { custom_stream_cost = u64::MAX; break; }
            // Cost in 256ths of a bit: count * log2(tableSize/prob) * 256
            // log2(tableSize/prob) = table_log - log2(prob)
            let log2_prob = 63 - (prob as u64).leading_zeros() as u64;
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
fn write_seq_table_and_build(
    out: &mut Vec<u8>,
    mode: &SeqTableMode,
    prev_mode: &Option<SeqTableMode>,
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
        SeqTableMode::Repeat => {
            // No header written. Rebuild table from previous block's mode.
            match prev_mode {
                Some(SeqTableMode::Predefined) => {
                    super::fse::FseCTable::build(default_norm, default_max_symbol, default_log)
                }
                Some(SeqTableMode::Rle(sym)) => super::fse::FseCTable::build_rle(*sym),
                Some(SeqTableMode::Fse { norm, max_symbol, table_log, .. }) => {
                    super::fse::FseCTable::build(norm, *max_symbol, *table_log)
                }
                _ => super::fse::FseCTable::build(default_norm, default_max_symbol, default_log),
            }
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
