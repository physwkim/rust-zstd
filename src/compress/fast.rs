//! Fast (single hash table) block compressor: port of
//! `ZSTD_compressBlock_fast_noDict_generic`, `ZSTD_compressBlock_fast` and
//! `ZSTD_fillHashTable` (zstd_fast.c, libzstd 1.5.7), no-dictionary case.
//!
//! Positions are the indices [`MatchState`] assigns, `window.base`-relative
//! as in libzstd, so table contents and every search decision are
//! index-exact.
//!
//! The search loop reads `src` and the hash table without bounds checks
//! (the checks cost 10-16% of the throughput). Every read is covered by
//! one of the invariants stated in `compress_block_generic`.

use super::common::{
    byte, candidate_valid, hash_ptr, index_overlap_check, prefetch, read32, simd_level, tget, tset,
    write_tagged, MatchCount, Src, HASH_READ_SIZE, K_SEARCH_STRENGTH, SHORT_CACHE_TAG_BITS,
};
use super::matchstate::{Block, EnteredPrefix, MatchState};
use super::seqstore::{offset_to_offbase, SeqStore, REPCODE1_TO_OFFBASE};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Avx2;
use fearless_simd::{Fallback, Level};
use std::ops::Range;

/// `kStepIncr` of the fast strategy.
const K_STEP_INCR: usize = 1 << (K_SEARCH_STRENGTH - 1);

/// `ZSTD_match4Found_cmov` / `ZSTD_match4Found_branch`: does the 4-byte
/// candidate at `match_idx` equal `src[cur..]`? A candidate is valid iff
/// `idx_low_limit <= match_idx < cur` (see [`candidate_valid`]).
///
/// # Safety
/// `cur + 4 <= src.end()`.
#[inline(always)]
unsafe fn match4_found<const CMOV: bool>(
    src: Src,
    cur: usize,
    match_idx: usize,
    idx_low_limit: usize,
) -> bool {
    let valid = candidate_valid(match_idx, idx_low_limit, cur);
    if CMOV {
        // The C version loads from a dummy array when the index is out of
        // range so that the range test compiles to a conditional move.
        // Loading `cur` and flipping a bit guarantees the same mismatch.
        let pos = if valid { match_idx } else { cur };
        let mval = read32(src, pos) ^ (!valid as u32);
        read32(src, cur) == mval
    } else {
        let mval = if valid {
            read32(src, match_idx)
        } else {
            read32(src, cur) ^ 1 // guaranteed to not match
        };
        read32(src, cur) == mval
    }
}

/// How the pipelined search loop exited.
enum Found {
    /// Repcode hit at `ip2` (`goto _match`): `match0` and `m_length` known.
    Rep { match0: usize, m_length: usize },
    /// Hash-table hit at `ip0` (`goto _offset`).
    Offset,
    /// Hash-table hit at `ip0` after the advance, in `ZSTD_extDict` mode
    /// (`EXT`): the hashed next position `ip1` (`hash1`) goes in the table
    /// after the match if the match covers it.
    OffsetExt { hash1: usize, ip1: usize },
    /// `while (ip3 < ilimit)` failed (`_cleanup`).
    Cleanup,
}

/// `ZSTD_compressBlock_fast_noDict_generic(ms, seqStore, rep, src, srcSize,
/// mls, useCmov)`, monomorphized over `MLS` and `CMOV`; with `EXT`,
/// `ZSTD_compressBlock_fast_extDict_generic`, which libzstd runs while
/// dictionary content is in reach ([`MatchState::ext_dict_in_reach`]),
/// the content `[prefix_start, dict_limit)` being its `dictBase` segment.
/// Over the one contiguous window the two differ at that segment's end
/// and in one table write. extDict disables a repcode of more than `ip0 -
/// prefix_start` at the block start (noDict: more than `ip0 -
/// window_low`), takes no repcode at `ip2` starting in
/// `[dict_limit - 3, dict_limit]` nor an immediate one straddling
/// `dict_limit` ([`index_overlap_check`]), and stops a catch-up at the
/// start of the match's segment. The table entry of the next position
/// after a match found past the first position goes in before the match
/// when `step <= 4` (noDict), after it when the match covers the position
/// (extDict).
///
/// Bounds invariants covering every unchecked read below:
///
/// * (I1) ip-derived positions: inside the search loop
///   `ip0 < ip1 < ip2 < ip3 < ilimit = iend - 8`, so 8-byte reads at
///   `ip0..=ip2` end before `iend <= src.end()`; after a match the reads at
///   `current0 + 2`, `ip0 - 2` and `ip0` are guarded by `ip0 <= ilimit`
///   (`current0 + 4 <= ip0`). `ip0 - 1 >= prefix_start >= window_low >= src.lo()`.
/// * (I2) candidates: a table entry is used only after [`match4_found`]
///   established `prefix_start <= match_idx < ip0`, so `match_idx + 4 <=
///   ip0 + 4 <= iend` and `ip0 - match_idx >= 1`.
/// * (I3) repcodes: on entry `rep_offset1/2 <= ip0 - lowest_match_index(ip0)`;
///   afterwards `rep_offset1 = ip0 - match_idx` with (I2), and
///   `rep_offset2` is a former `rep_offset1`. A repcode is only applied at
///   positions `p >= ` the `ip0` it was derived at, hence
///   `1 <= p - rep < p` and `p - rep >= window_low >= src.lo()` (`rep == 0` means
///   disabled and reads `p` itself).
/// * (I4) `hash_ptr` returns `< 1 << hlog == hash_table.len()`.
///
/// Not forced inline: `#[inline(always)]` re-allocates the scalar search
/// loop with one more stack reload than when the inliner takes it. The
/// inliner still puts the AVX2 monomorphs into [`compress_block_avx2`]; one
/// left out of line stays correct and calls the AVX2 count out of line.
fn compress_block_generic<const MLS: u32, const CMOV: bool, const EXT: bool, C: MatchCount>(
    mc: C,
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let hlog = ms.cparams.hash_log;
    let target_length = ms.cparams.target_length as usize;
    let step_size = target_length + (target_length == 0) as usize + 1; // min 2
    let istart = block.start;
    let iend = block.end;
    assert!(istart <= iend && iend <= src.end());
    assert!((1..=32).contains(&hlog));
    // C bounds the block by `ZSTD_getLowestPrefixIndex(endIndex)`; the
    // bound of its last position holds for every position.
    let prefix_start = ms.lowest_match_index(iend - 1);
    // extDict: the input's first index (`prefixStartIndex`; C's
    // `dictStartIndex` is `prefix_start` here). Unused without `EXT`.
    let dict_limit = ms.window().dict_limit();
    // C: ilimit = iend - HASH_READ_SIZE, possibly below istart; every
    // comparison against it then sends the loop to _cleanup.
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

    let mut anchor = istart;
    // C: ip0 = istart; ip0 += (ip0 == prefixStart)
    let mut ip0 = istart;
    ip0 += (ip0 == prefix_start) as usize;

    let mut rep_offset1 = rep[0];
    let mut rep_offset2 = rep[1];
    let (mut offset_saved1, mut offset_saved2) = (0u32, 0u32);
    {
        // extDict: the block-wide bound (libzstd disables `offset >= curr -
        // dictStartIndex`, a repcode at `dictStartIndex` too).
        let max_rep = if EXT {
            (ip0 - prefix_start) as u32
        } else {
            (ip0 - ms.lowest_match_index(ip0)) as u32
        };
        if rep_offset2 > max_rep {
            offset_saved2 = rep_offset2;
            rep_offset2 = 0;
        }
        if rep_offset1 > max_rep {
            offset_saved1 = rep_offset1;
            rep_offset1 = 0;
        }
    }

    let (hash_table, _, _) = ms.ws.tables_mut();
    assert_eq!(hash_table.len(), 1usize << hlog); // (I4)

    // _start: requires ip0
    'start: loop {
        let mut step = step_size;
        let mut next_step = ip0 + K_STEP_INCR;

        // calculate positions, ip0 - anchor == 0, so we skip step calc.
        // C keeps four pointers ip0 < ip1 == ip0 + 1 < ip2 < ip3 == ip2 + 1;
        // the two odd ones are derived where they are used so that the loop
        // carries two registers less (the C names are kept in the comments).
        let mut ip2 = ip0 + step;

        if ip2 + 1 >= ilimit {
            break 'start; // _cleanup
        }

        // SAFETY: (I1) for ip0, ip0 + 1; (I4) for the table.
        let (mut hash0, mut hash1, mut match_idx) = unsafe {
            let hash0 = hash_ptr::<MLS>(src, ip0, hlog);
            (
                hash0,
                hash_ptr::<MLS>(src, ip0 + 1, hlog),
                tget(hash_table, hash0),
            )
        };
        let mut current0;

        // C tests `(MEM_read32(ip2) == rval) & (rep_offset1 > 0)`; with
        // rep_offset1 == 0 rval is read at ip2 itself, so flipping one bit
        // of it makes the same test fail without a second condition.
        let rep_mask = (rep_offset1 == 0) as u32;
        // SAFETY: (I1) for every ip-derived read, (I2) for the candidate
        // reads inside match4_found, (I3) for `ip2 - rep_offset1` and the
        // bytes before it, (I4) for every table access.
        let found = unsafe {
            loop {
                // Here ip1 == ip0 + 1 and ip3 == ip2 + 1.

                // load repcode match for ip[2]
                let rval = if EXT {
                    // intentional underflow: no repcode starting in
                    // [dict_limit - 3, dict_limit]
                    let rep_index = (ip2 as u32).wrapping_sub(rep_offset1);
                    if (dict_limit as u32).wrapping_sub(rep_index) >= 4 && rep_offset1 > 0 {
                        read32(src, rep_index as usize)
                    } else {
                        read32(src, ip2) ^ 1 // guaranteed to not match
                    }
                } else {
                    read32(src, ip2 - rep_offset1 as usize) ^ rep_mask
                };

                // write back hash table entry
                current0 = ip0;
                tset(hash_table, hash0, current0);

                // check repcode at ip[2]
                if read32(src, ip2) == rval {
                    let ip1 = ip0 + 1;
                    ip0 = ip2;
                    let mut match0 = ip0 - rep_offset1 as usize;
                    let m_length = (byte(src, ip0 - 1) == byte(src, match0 - 1)) as usize;
                    ip0 -= m_length;
                    match0 -= m_length;
                    // Write next hash table entry: it's already calculated.
                    // This write is known to be safe because ip1 is before
                    // the repcode (ip2).
                    tset(hash_table, hash1, ip1);
                    break Found::Rep {
                        match0,
                        m_length: m_length + 4,
                    };
                }

                if match4_found::<CMOV>(src, ip0, match_idx, prefix_start) {
                    // Write next hash table entry (it's already calculated).
                    // This write is known to be safe because the ip1 == ip0
                    // + 1, so searching will resume after ip1.
                    tset(hash_table, hash1, ip0 + 1);
                    break Found::Offset;
                }

                // lookup ip[1]
                match_idx = tget(hash_table, hash1);

                // hash ip[2]
                hash0 = hash1;
                hash1 = hash_ptr::<MLS>(src, ip2, hlog);

                // advance to next positions: ip0 = ip1, ip1 = ip2, ip2 = ip3.
                // From here on ip1 == ip2 (this variable) and C's ip2 is
                // ip2 + 1.
                ip0 += 1;

                // write back hash table entry
                current0 = ip0;
                tset(hash_table, hash0, current0);

                if match4_found::<CMOV>(src, ip0, match_idx, prefix_start) {
                    if EXT {
                        break Found::OffsetExt { hash1, ip1: ip2 };
                    }
                    // Write next hash table entry, since it's already calculated
                    if step <= 4 {
                        // Avoid writing an index if it's >= position where
                        // search will resume. The minimum possible match has
                        // length 4, so search can resume at ip0 + 4.
                        tset(hash_table, hash1, ip2); // ip1
                    }
                    break Found::Offset;
                }

                // lookup ip[1]
                match_idx = tget(hash_table, hash1);

                // hash ip[2]
                hash0 = hash1;
                hash1 = hash_ptr::<MLS>(src, ip2 + 1, hlog);

                // advance to next positions: ip0 = ip1, ip1 = ip2, ip2 = ip0
                // + step, ip3 = ip1 + step == ip2 + 1.
                ip0 = ip2;
                ip2 += step;

                // calculate step
                if ip2 >= next_step {
                    step += 1;
                    prefetch(src, ip0 + 1 + 64); // ip1
                    prefetch(src, ip0 + 1 + 128);
                    next_step += K_STEP_INCR;
                }

                if ip2 + 1 >= ilimit {
                    break Found::Cleanup;
                }
            }
        };

        let mut ext_next = None;
        let (match0, offcode, mut m_length) = match found {
            Found::Cleanup => break 'start,
            Found::Rep { match0, m_length } => (match0, REPCODE1_TO_OFFBASE, m_length),
            found @ (Found::Offset | Found::OffsetExt { .. }) => {
                if let Found::OffsetExt { hash1, ip1 } = found {
                    ext_next = Some((hash1, ip1));
                }
                // _offset: requires ip0, idx. Compute the offset code.
                let mut match0 = match_idx;
                rep_offset2 = rep_offset1;
                rep_offset1 = (ip0 - match0) as u32;
                let offcode = offset_to_offbase(rep_offset1);
                let mut m_length = 4;
                // extDict: the start of the match's segment.
                let low_match = if EXT && match0 >= dict_limit {
                    dict_limit
                } else {
                    prefix_start
                };
                // Count the backwards match length.
                // SAFETY: ip0 > anchor >= istart and match0 > low_match >=
                // prefix_start keep both indices >= src.lo() and below ip0 <
                // iend.
                unsafe {
                    while ((ip0 > anchor) & (match0 > low_match))
                        && byte(src, ip0 - 1) == byte(src, match0 - 1)
                    {
                        ip0 -= 1;
                        match0 -= 1;
                        m_length += 1;
                    }
                }
                (match0, offcode, m_length)
            }
        };

        // _match: requires ip0, match0, offcode. Count the forward length.
        // SAFETY: match0 < ip0 (I2/I3) and ip0 + m_length <= ip2 + 4 < iend.
        m_length += unsafe { mc.count(src, ip0 + m_length, match0 + m_length, iend) };

        out.store_seq(src, anchor, ip0 - anchor, iend, offcode, m_length);

        ip0 += m_length;
        anchor = ip0;

        // extDict: write next hash table entry
        if let Some((hash1, ip1)) = ext_next {
            if ip1 < ip0 {
                // SAFETY: (I4).
                unsafe { tset(hash_table, hash1, ip1) };
            }
        }

        // Fill table and check for immediate repcode.
        if ip0 <= ilimit {
            // SAFETY: (I1) with ip0 <= ilimit; (I3) for ip0 - rep_offset2;
            // (I4) for the table.
            unsafe {
                // Fill Table: here because current+2 could be > iend-8
                tset(
                    hash_table,
                    hash_ptr::<MLS>(src, current0 + 2, hlog),
                    current0 + 2,
                );
                tset(hash_table, hash_ptr::<MLS>(src, ip0 - 2, hlog), ip0 - 2);

                // rep_offset2 == 0 means rep_offset2 is invalidated
                if rep_offset2 > 0 {
                    while ip0 <= ilimit
                        && (!EXT
                            || index_overlap_check(
                                dict_limit,
                                (ip0 as u32).wrapping_sub(rep_offset2),
                            ))
                        && read32(src, ip0) == read32(src, ip0 - rep_offset2 as usize)
                    {
                        // store sequence
                        let r_length =
                            mc.count(src, ip0 + 4, ip0 + 4 - rep_offset2 as usize, iend) + 4;
                        std::mem::swap(&mut rep_offset1, &mut rep_offset2);
                        tset(hash_table, hash_ptr::<MLS>(src, ip0, hlog), ip0);
                        ip0 += r_length;
                        out.store_seq(src, anchor, 0, iend, REPCODE1_TO_OFFBASE, r_length);
                        anchor = ip0;
                    }
                }
            }
        }
    }

    // _cleanup. When the repcodes are outside of the prefix, they were set to
    // zero before the loop; if still zero they are restored. If rep_offset1
    // started invalid (offsetSaved1 != 0) and became valid (rep_offset1 !=
    // 0), then rep[0] = rep_offset1 and rep[1] = offsetSaved1.
    offset_saved2 = if offset_saved1 != 0 && rep_offset1 != 0 {
        offset_saved1
    } else {
        offset_saved2
    };

    // save reps for next block
    rep[0] = if rep_offset1 != 0 {
        rep_offset1
    } else {
        offset_saved1
    };
    rep[1] = if rep_offset2 != 0 {
        rep_offset2
    } else {
        offset_saved2
    };

    // Return the anchor of the last literals
    anchor
}

/// `ZSTD_compressBlock_fast`: find matches in `src[block]` and store them
/// into `out`. Returns the anchor: the start of the trailing literals
/// `src[anchor..block.end]`, which the caller appends to `out.lits`
/// (`ZSTD_storeLastLiterals`).
///
/// `rep` is the repeat-offset history on entry and is updated on exit. A
/// repcode that would reach below the window at the block start is disabled
/// (`0`) for this block and restored on exit when no match replaced it.
pub fn compress_block(
    ms: &mut MatchState,
    src: Src,
    block: Block,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let block = block.range();
    // ZSTD_selectBlockCompressor: ZSTD_compressBlock_fast_extDict while
    // dictionary content is in reach.
    let ext = ms.ext_dict_in_reach(block.end);
    match simd_level() {
        // SAFETY: fearless_simd constructs the witness only after detecting
        // AVX2 on this CPU, and BMI2 is detected here.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(w) if std::arch::is_x86_feature_detected!("bmi2") => unsafe {
            if ext {
                compress_block_avx2::<true>(w, ms, src, block, rep, out)
            } else {
                compress_block_avx2::<false>(w, ms, src, block, rep, out)
            }
        },
        _ if ext => compress_block_scalar::<true>(ms, src, block, rep, out),
        _ => compress_block_scalar::<false>(ms, src, block, rep, out),
    }
}

/// [`compress_block`] with the 8-byte [`count`](super::common::count).
#[inline(never)]
fn compress_block_scalar<const EXT: bool>(
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    compress_block_level::<EXT, _>(Fallback::new(), ms, src, block, rep, out)
}

/// [`compress_block`] compiled with AVX2, counting 32 bytes per step, and
/// BMI2, whose `shrx` takes the hash shift count in any register: with the
/// `shr r, cl` form the loop spends `rcx` on it and reloads it from the
/// stack.
///
/// # Safety
///
/// The CPU must support AVX2 (the witness proves it) and BMI2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn compress_block_avx2<const EXT: bool>(
    mc: Avx2,
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    compress_block_level::<EXT, _>(mc, ms, src, block, rep, out)
}

#[inline(always)]
fn compress_block_level<const EXT: bool, C: MatchCount>(
    mc: C,
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    // ZSTD_compressBlock_fast_extDict uses no cmov.
    if EXT {
        return match ms.cparams.min_match {
            5 => compress_block_generic::<5, false, true, C>(mc, ms, src, block, rep, out),
            6 => compress_block_generic::<6, false, true, C>(mc, ms, src, block, rep, out),
            7 => compress_block_generic::<7, false, true, C>(mc, ms, src, block, rep, out),
            _ => compress_block_generic::<4, false, true, C>(mc, ms, src, block, rep, out),
        };
    }
    // use cmov when "candidate in range" branch is likely unpredictable
    let use_cmov = ms.cparams.window_log < 19;
    match (use_cmov, ms.cparams.min_match) {
        (true, 5) => compress_block_generic::<5, true, false, C>(mc, ms, src, block, rep, out),
        (true, 6) => compress_block_generic::<6, true, false, C>(mc, ms, src, block, rep, out),
        (true, 7) => compress_block_generic::<7, true, false, C>(mc, ms, src, block, rep, out),
        (true, _) => compress_block_generic::<4, true, false, C>(mc, ms, src, block, rep, out),
        (false, 5) => compress_block_generic::<5, false, false, C>(mc, ms, src, block, rep, out),
        (false, 6) => compress_block_generic::<6, false, false, C>(mc, ms, src, block, rep, out),
        (false, 7) => compress_block_generic::<7, false, false, C>(mc, ms, src, block, rep, out),
        (false, _) => compress_block_generic::<4, false, false, C>(mc, ms, src, block, rep, out),
    }
}

/// `ZSTD_fillHashTableForCCtx(ms, end, ZSTD_dtlm_fast)` without
/// `FOR_CDICT`; with it `ZSTD_fillHashTableForCDict(ms, end,
/// ZSTD_dtlm_full)`, which also inserts the two positions after each third
/// one where their slot is empty, and tags every entry ([`write_tagged`]):
/// the slot is the high bits of a hash [`SHORT_CACHE_TAG_BITS`] wider,
/// the same slot as the untagged hash's.
fn fill_hash_table<const MLS: u32, const FOR_CDICT: bool>(
    ms: &mut MatchState,
    src: Src,
    start: usize,
    end: usize,
) {
    const FAST_HASH_FILL_STEP: usize = 3;
    let hbits = ms.cparams.hash_log;
    assert!((1..=32).contains(&hbits));
    assert!(!FOR_CDICT || hbits + SHORT_CACHE_TAG_BITS <= 32);
    assert!(end <= src.end());
    let (hash_table, _, _) = ms.ws.tables_mut();
    assert_eq!(hash_table.len(), 1usize << hbits);
    let mut ip = start;
    // C: for (; ip + fastHashFillStep < iend + 2; ip += fastHashFillStep)
    // with iend = end - HASH_READ_SIZE. Always insert every
    // fastHashFillStep position into the hash table.
    while ip + FAST_HASH_FILL_STEP + HASH_READ_SIZE < end + 2 {
        if FOR_CDICT {
            let tbits = hbits + SHORT_CACHE_TAG_BITS;
            // SAFETY: ip + 10 <= end <= src.end(); a hash of `tbits` bits
            // shifted down by the tag is < 1 << hbits == len.
            unsafe {
                write_tagged(hash_table, hash_ptr::<MLS>(src, ip, tbits), ip);
                // Only load extra positions for ZSTD_dtlm_full, where their
                // entry is still empty.
                for p in 1..FAST_HASH_FILL_STEP {
                    // ip + p + 8 <= ip + 10 <= end.
                    let hash_and_tag = hash_ptr::<MLS>(src, ip + p, tbits);
                    if tget(hash_table, hash_and_tag >> SHORT_CACHE_TAG_BITS) == 0 {
                        write_tagged(hash_table, hash_and_tag, ip + p);
                    }
                }
            }
        } else {
            // SAFETY: ip + 10 <= end <= src.end(); hash < 1 << hbits == len.
            unsafe { tset(hash_table, hash_ptr::<MLS>(src, ip, hbits), ip) };
        }
        ip += FAST_HASH_FILL_STEP;
    }
}

/// [`fill_hash_table`] for `ms.cparams.min_match`.
fn fill_hash_table_from<const FOR_CDICT: bool>(
    ms: &mut MatchState,
    src: Src,
    start: usize,
    end: usize,
) {
    match ms.cparams.min_match {
        5 => fill_hash_table::<5, FOR_CDICT>(ms, src, start, end),
        6 => fill_hash_table::<6, FOR_CDICT>(ms, src, start, end),
        7 => fill_hash_table::<7, FOR_CDICT>(ms, src, start, end),
        _ => fill_hash_table::<4, FOR_CDICT>(ms, src, start, end),
    }
}

/// `ZSTD_fillHashTable(ms, end, ZSTD_dtlm_fast, ZSTD_tfp_forCCtx)` for an
/// entered prefix: insert every third position from `ms.next_to_update`
/// (its start) into the hash table, then set `next_to_update` to its end.
pub fn load_prefix(ms: &mut MatchState, src: Src, prefix: EnteredPrefix) {
    let end = ms.prefix_indices(prefix).end;
    assert!(end <= src.end());
    fill_hash_table_from::<false>(ms, src, ms.next_to_update, end);
    ms.next_to_update = end;
}

/// `ZSTD_fillHashTable(ms, end, ZSTD_dtlm_full, ZSTD_tfp_forCDict)` for a
/// dictionary's entered content: [`load_prefix`] that also inserts the two
/// positions after each third one where their entry is empty, every entry
/// tagged (see [`fill_hash_table`]).
pub fn load_dict_full(ms: &mut MatchState, src: Src, content: EnteredPrefix) {
    let end = ms.prefix_indices(content).end;
    assert!(end <= src.end());
    fill_hash_table_from::<true>(ms, src, ms.next_to_update, end);
    ms.next_to_update = end;
}

/// `ZSTD_fillHashTable(ms, end, ZSTD_dtlm_fast, ZSTD_tfp_forCCtx)` as
/// `ZSTD_ldm_fillFastTables` calls it before each block compressor run:
/// insert every third position from `ms.next_to_update` up to `end`,
/// leaving `next_to_update` where it is.
pub fn fill_hash_table_to(ms: &mut MatchState, src: Src, end: usize) {
    fill_hash_table_from::<false>(ms, src, ms.next_to_update, end);
}

#[cfg(test)]
mod tests {
    use super::super::common::testutil::*;
    use super::*;
    use crate::compress::params::{CParams, Strategy};

    fn finder() -> Finder {
        Finder {
            compress_block,
            load_prefix,
        }
    }

    #[test]
    fn roundtrip_crate_sources_128k_blocks() {
        let data = crate_sources();
        for level in [1, 2] {
            let stats = roundtrip_blocks(
                &finder(),
                &data,
                CParams::for_level(level, data.len()),
                1 << 17,
                [1, 4, 8],
            );
            assert!(
                stats.cross_block_matches > 0,
                "level {level}: no match reached an earlier block"
            );
        }
    }

    #[test]
    fn roundtrip_current_exe_128k_blocks() {
        let data = current_exe_bytes();
        let stats = roundtrip_blocks(
            &finder(),
            &data,
            CParams::for_level(1, data.len()),
            1 << 17,
            [1, 4, 8],
        );
        assert!(stats.cross_block_matches > 0);
    }

    #[test]
    fn job_start_with_overlap_prefix_and_zero_reps() {
        let data = synthetic_text(600_000, 7);
        let cp = CParams::for_level(1, data.len());
        let origin = 200_000;
        let job_start = origin + (1 << 16);
        let stats = roundtrip_job(&finder(), &data, cp, 1 << 17, origin, job_start, [0, 0, 0]);
        assert!(
            stats.prefix_matches > 0,
            "no match referenced the loaded prefix"
        );
    }

    #[test]
    fn tiny_inputs_and_block_boundary() {
        let text = synthetic_text((1 << 17) + 1, 3);
        for len in [0usize, 1, 7, 8, 9, 100, (1 << 17) + 1] {
            let data = &text[..len];
            for level in [1, 2] {
                roundtrip_blocks(
                    &finder(),
                    data,
                    CParams::for_level(level, len),
                    1 << 17,
                    [1, 4, 8],
                );
            }
        }
    }

    #[test]
    fn every_mls_step_and_cmov_variant() {
        let data = synthetic_text(400_000, 11);
        for min_match in 4..=7u32 {
            for window_log in [18u32, 19] {
                for target_length in [0u32, 1, 3] {
                    let cp = CParams {
                        window_log,
                        chain_log: 13,
                        hash_log: 15,
                        search_log: 1,
                        min_match,
                        target_length,
                        strategy: Strategy::Fast,
                    };
                    let stats = roundtrip_blocks(&finder(), &data, cp, 1 << 17, [1, 4, 8]);
                    assert!(
                        stats.seqs > 100,
                        "mls {min_match} wlog {window_log} tl {target_length}"
                    );
                }
            }
        }
    }

    #[test]
    fn window_limited_matches_stay_inside_the_window() {
        // window_log 10 with 300 KiB of input: matches must never reach
        // further back than 1024 bytes.
        let data = synthetic_text(300_000, 5);
        let cp = CParams {
            window_log: 10,
            chain_log: 10,
            hash_log: 11,
            search_log: 1,
            min_match: 5,
            target_length: 0,
            strategy: Strategy::Fast,
        };
        roundtrip_blocks(&finder(), &data, cp, 1 << 10, [1, 4, 8]);
        roundtrip_blocks(&finder(), &data, cp, 1000, [1, 4, 8]);
    }

    #[test]
    fn rep_disabled_at_block_start_is_restored() {
        let src = vec![7u8; 4000];
        let cp = CParams::for_level(1, src.len());
        let mut ms = MatchState::new(cp, 0);
        let mut store = SeqStore::new();
        // rep[0] = 100 cannot be used from position 1: it is disabled and
        // then restored when a new offset replaces rep1 (saved1 -> rep[1]).
        let mut rep = [100u32, 4, 8];
        let anchor = run_block(
            compress_block,
            &mut ms,
            &src,
            0..src.len(),
            &mut rep,
            &mut store,
        );
        store.lits.extend_from_slice(&src[anchor..]);
        assert_eq!(store.reconstruct(&[], [100, 4, 8]), src);
        assert_eq!(rep[1], 100);
        assert_eq!(rep[2], 8);
    }

    /// A table entry pointing past the current position (tables carried
    /// over from a longer input) must be a miss, never a read past `src`.
    #[test]
    fn stale_table_entries_beyond_the_input_are_ignored() {
        let long = synthetic_text(300_000, 9);
        let cp = CParams::for_level(1, long.len());
        let mut stale = MatchState::new(cp, 0);
        let mut store = SeqStore::new();
        let mut rep = [1u32, 4, 8];
        run_block(
            compress_block,
            &mut stale,
            &long,
            0..long.len(),
            &mut rep,
            &mut store,
        );
        let mut ms = MatchState::new(cp, 0);
        ms.ws = stale.ws;
        let short = &long[..20_000];
        store.clear();
        let mut rep = [1u32, 4, 8];
        let anchor = run_block(
            compress_block,
            &mut ms,
            short,
            0..short.len(),
            &mut rep,
            &mut store,
        );
        store.lits.extend_from_slice(&short[anchor..]);
        assert_eq!(store.reconstruct(&[], [1, 4, 8]), short);
    }
}
