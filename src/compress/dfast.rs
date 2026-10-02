//! Double-fast block compressor: port of
//! `ZSTD_compressBlock_doubleFast_noDict_generic`,
//! `ZSTD_compressBlock_doubleFast_extDict_generic`,
//! `ZSTD_compressBlock_doubleFast` and `ZSTD_fillDoubleHashTable`
//! (zstd_double_fast.c, libzstd 1.5.7).
//!
//! The hash table is `hashLong` (`hBitsL = hash_log`, 8-byte hash) and the
//! chain table is `hashSmall` (`hBitsS = chain_log`, `mls`-byte hash).
//! Position conventions and the unchecked-read policy are those of
//! [`super::fast`].

use super::common::{
    byte, candidate_valid, count_2segments, count_dms, dms_rep_source, hash_ptr,
    index_overlap_check, prefetch, read32, read64, simd_level, tags_match, tget, tset,
    write_tagged, MatchCount, Src, HASH_READ_SIZE, K_SEARCH_STRENGTH, SHORT_CACHE_TAG_BITS,
};
use super::matchstate::{Block, DictMatchState, EnteredPrefix, MatchState, WINDOW_START_INDEX};
use super::seqstore::{offset_to_offbase, SeqStore, REPCODE1_TO_OFFBASE};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Avx2;
use fearless_simd::{Fallback, Level};
use std::ops::Range;

/// `kStepIncr` of the double-fast strategy: how many positions to search
/// before increasing the step size.
const K_STEP_INCR: usize = 1 << K_SEARCH_STRENGTH;

/// How the inner search loop exited.
enum Found {
    /// Repcode at `ip + 1`, sequence already stored (`goto _match_stored`).
    Stored { m_length: usize },
    /// Long or short hash match at `ip` (`_match_found`).
    Match { offset: u32, m_length: usize },
    /// `while (ip1 <= ilimit)` failed (`_cleanup`).
    Cleanup,
}

/// `ZSTD_compressBlock_doubleFast_noDict_generic(ms, seqStore, rep, src,
/// srcSize, mls)`, monomorphized over `MLS`.
///
/// Bounds invariants covering every unchecked read below:
///
/// * (I1) ip-derived positions: inside the search loop
///   `ip < ip1 <= ilimit = iend - 8`, so 8-byte reads at `ip`, `ip1` and
///   the 4-byte read at `ip + 1` end before `iend <= src.end()`; after a
///   match the reads at `curr + 2`, `ip - 2`, `ip - 1` and `ip` are guarded
///   by `ip <= ilimit` (`curr + 4 <= ip`). `ip - 1 >= prefix_lowest >= src.lo()`.
/// * (I2) candidates: a table entry is used only when
///   `prefix_lowest_index <= idx < ip` (`idxl1`: `< ip1`), see
///   [`candidate_valid`].
/// * (I3) repcodes: on entry `offset_1/2 <= ip - lowest_match_index(ip)`; afterwards
///   `offset_1 = ip - idx` with (I2) and `offset_2` is a former `offset_1`.
///   A repcode is only applied at positions `p >=` the `ip` it was derived
///   at, hence `1 <= p - offset < p` (`0` means disabled and reads `p`).
/// * (I4) `hash_ptr` returns `< 1 << hbits == table.len()` for both tables.
fn compress_block_generic<const MLS: u32, C: MatchCount>(
    mc: C,
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let hbits_l = ms.cparams.hash_log;
    let hbits_s = ms.cparams.chain_log;
    let istart = block.start;
    let iend = block.end;
    assert!(istart <= iend && iend <= src.end());
    assert!((1..=32).contains(&hbits_l) && (1..=32).contains(&hbits_s));
    // presumes that, if there is a dictionary, it must be using Attach mode
    // C bounds the block by `ZSTD_getLowestPrefixIndex(endIndex)`; the
    // bound of its last position holds for every position.
    let prefix_lowest_index = ms.lowest_match_index(iend - 1);
    let prefix_lowest = prefix_lowest_index;
    // C: ilimit = iend - HASH_READ_SIZE, possibly below istart (see fast.rs).
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

    let mut anchor = istart;
    // init: ip += ((ip - prefixLowest) == 0)
    let mut ip = istart;
    ip += (ip == prefix_lowest) as usize;

    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let (mut offset_saved1, mut offset_saved2) = (0u32, 0u32);
    {
        let window_low = ms.lowest_match_index(ip);
        let max_rep = (ip - window_low) as u32;
        if offset_2 > max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }

    let (hash_long, hash_small, _) = ms.ws.tables_mut();
    assert_eq!(hash_long.len(), 1usize << hbits_l); // (I4)
    assert_eq!(hash_small.len(), 1usize << hbits_s); // (I4)

    // Outer Loop: one iteration per match found and stored
    'outer: loop {
        let mut step = 1usize; // the current step size
                               // the position at which to increment the step size if no match is found
        let mut next_step = ip + K_STEP_INCR;
        let mut ip1 = ip + step; // the next position

        if ip1 > ilimit {
            break 'outer; // _cleanup
        }

        // SAFETY: (I1) for ip; (I4) for the table.
        let (mut hl0, mut idxl0) = unsafe {
            let hl0 = hash_ptr::<8>(src, ip, hbits_l); // the long hash at ip
            (hl0, tget(hash_long, hl0)) // the long match index for ip
        };
        let mut hl1 = 0usize; // the long hash at ip1; set before every use
        let mut curr;

        // Inner Loop: one iteration per search / position.
        // SAFETY: (I1) for every ip-derived read, (I2) for the candidate
        // reads, (I3) for `ip + 1 - offset_1`, (I4) for every table access.
        let found = unsafe {
            loop {
                let hs0 = hash_ptr::<MLS>(src, ip, hbits_s);
                let idxs0 = tget(hash_small, hs0);
                curr = ip;

                // update hash tables
                tset(hash_long, hl0, curr);
                tset(hash_small, hs0, curr);

                // check noDict repcode
                if (offset_1 > 0) & (read32(src, ip + 1 - offset_1 as usize) == read32(src, ip + 1))
                {
                    let m_length =
                        mc.count(src, ip + 1 + 4, ip + 1 + 4 - offset_1 as usize, iend) + 4;
                    ip += 1;
                    out.store_seq(
                        src,
                        anchor,
                        ip - anchor,
                        iend,
                        REPCODE1_TO_OFFBASE,
                        m_length,
                    );
                    break Found::Stored { m_length };
                }

                hl1 = hash_ptr::<8>(src, ip1, hbits_l);

                // idxl0 >= prefixLowestIndex is a (somewhat) unpredictable
                // branch; the C code selects a dummy address so that it
                // becomes a conditional move. Reading `ip` and flipping a
                // bit guarantees the same mismatch (see fast::match4_found).
                {
                    let valid = candidate_valid(idxl0, prefix_lowest_index, ip);
                    let pos = if valid { idxl0 } else { ip };
                    let mval = read64(src, pos) ^ (!valid as u64);
                    // check prefix long match
                    if read64(src, ip) == mval {
                        let mut matchl0 = idxl0;
                        let mut m_length = mc.count(src, ip + 8, matchl0 + 8, iend) + 8;
                        let offset = (ip - matchl0) as u32;
                        // catch up
                        while ((ip > anchor) & (matchl0 > prefix_lowest))
                            && byte(src, ip - 1) == byte(src, matchl0 - 1)
                        {
                            ip -= 1;
                            matchl0 -= 1;
                            m_length += 1;
                        }
                        break Found::Match { offset, m_length };
                    }
                }

                let idxl1 = tget(hash_long, hl1); // the long match index for ip1

                // Same optimization as matchl0 above: check prefix short match
                let valid = candidate_valid(idxs0, prefix_lowest_index, ip);
                let pos = if valid { idxs0 } else { ip };
                let mval = read32(src, pos) ^ (!valid as u32);
                if read32(src, ip) == mval {
                    // _search_next_long: short match found, check for a longer one
                    let mut matchs0 = idxs0;
                    let mut m_length = mc.count(src, ip + 4, matchs0 + 4, iend) + 4;
                    let mut offset = (ip - matchs0) as u32;

                    // check long match at +1 position
                    if candidate_valid(idxl1, prefix_lowest_index + 1, ip1)
                        && read64(src, idxl1) == read64(src, ip1)
                    {
                        let l1len = mc.count(src, ip1 + 8, idxl1 + 8, iend) + 8;
                        if l1len > m_length {
                            // use the long match instead
                            ip = ip1;
                            m_length = l1len;
                            offset = (ip - idxl1) as u32;
                            matchs0 = idxl1;
                        }
                    }

                    // complete backward
                    while ((ip > anchor) & (matchs0 > prefix_lowest))
                        && byte(src, ip - 1) == byte(src, matchs0 - 1)
                    {
                        ip -= 1;
                        matchs0 -= 1;
                        m_length += 1;
                    }
                    break Found::Match { offset, m_length };
                }

                if ip1 >= next_step {
                    prefetch(src, ip1 + 64);
                    prefetch(src, ip1 + 128);
                    step += 1;
                    next_step += K_STEP_INCR;
                }
                ip = ip1;
                ip1 += step;

                hl0 = hl1;
                idxl0 = idxl1;

                if ip1 > ilimit {
                    break Found::Cleanup;
                }
            }
        };

        let m_length = match found {
            Found::Cleanup => break 'outer,
            Found::Stored { m_length } => m_length,
            Found::Match { offset, m_length } => {
                // _match_found: requires ip, offset, mLength
                offset_2 = offset_1;
                offset_1 = offset;

                if step < 4 {
                    // It is unsafe to write this value back to the hashtable
                    // when ip1 is greater than or equal to the new ip we
                    // will have after we're done processing this match. The
                    // minmatch even if we take a short match is 4 bytes, so
                    // as long as step, the distance between ip and ip1
                    // (initially) is less than 4, we know ip1 < new ip.
                    // SAFETY: (I4); hl1 was computed on every Match path.
                    unsafe { tset(hash_long, hl1, ip1) };
                }

                out.store_seq(
                    src,
                    anchor,
                    ip - anchor,
                    iend,
                    offset_to_offbase(offset),
                    m_length,
                );
                m_length
            }
        };

        // _match_stored: match found
        ip += m_length;
        anchor = ip;

        if ip <= ilimit {
            // SAFETY: (I1) with ip <= ilimit; (I3) for ip - offset_2; (I4).
            unsafe {
                // Complementary insertion, done after iLimit test, as
                // candidates could be > iend-8
                let index_to_insert = curr + 2;
                tset(
                    hash_long,
                    hash_ptr::<8>(src, index_to_insert, hbits_l),
                    index_to_insert,
                );
                tset(hash_long, hash_ptr::<8>(src, ip - 2, hbits_l), ip - 2);
                tset(
                    hash_small,
                    hash_ptr::<MLS>(src, index_to_insert, hbits_s),
                    index_to_insert,
                );
                tset(hash_small, hash_ptr::<MLS>(src, ip - 1, hbits_s), ip - 1);

                // check immediate repcode
                while ip <= ilimit
                    && ((offset_2 > 0) & (read32(src, ip) == read32(src, ip - offset_2 as usize)))
                {
                    // store sequence
                    let r_length = mc.count(src, ip + 4, ip + 4 - offset_2 as usize, iend) + 4;
                    std::mem::swap(&mut offset_1, &mut offset_2);
                    tset(hash_small, hash_ptr::<MLS>(src, ip, hbits_s), ip);
                    tset(hash_long, hash_ptr::<8>(src, ip, hbits_l), ip);
                    out.store_seq(src, anchor, 0, iend, REPCODE1_TO_OFFBASE, r_length);
                    ip += r_length;
                    anchor = ip;
                }
            }
        }
    }

    // _cleanup: if offset_1 started invalid (offsetSaved1 != 0) and became
    // valid (offset_1 != 0), rotate saved offsets (see fast.rs).
    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 {
        offset_saved1
    } else {
        offset_saved2
    };

    // save reps for next block
    rep[0] = if offset_1 != 0 {
        offset_1
    } else {
        offset_saved1
    };
    rep[1] = if offset_2 != 0 {
        offset_2
    } else {
        offset_saved2
    };

    // Return the anchor of the last literals
    anchor
}

/// `ZSTD_compressBlock_doubleFast_extDict_generic(ms, seqStore, rep, src,
/// srcSize, mls)`, monomorphized over `MLS`: the loop libzstd runs while
/// dictionary content is in reach ([`MatchState::ext_dict_in_reach`]),
/// with the content `[dict_start, prefix_start)` as its `dictBase`
/// segment. Over the one contiguous window `ZSTD_count_2segments` is a
/// plain count; what differs from the no-dictionary loop is the search
/// order (repcode at `ip + 1`, long, short then long at `ip + 1`), the
/// step, no repcode reset at the block start, repcodes whose first four
/// bytes would straddle `prefix_start` ([`index_overlap_check`]), and a
/// catch-up bounded by the segment of the match.
///
/// Bounds invariants covering every unchecked read below:
///
/// * (E1) ip-derived positions: inside the loop `ip < ilimit = iend - 8`,
///   so 8-byte reads at `ip` and `ip + 1` end by `iend <= src.end()`;
///   after a match the reads at `curr + 2`, `ip - 2`, `ip - 1` and `ip`
///   are guarded by `ip <= ilimit` (`curr + 4 <= ip`).
/// * (E2) candidates: a table entry is used only when `dict_start <= idx <
///   cur` ([`candidate_valid`]) for the position `cur` it is compared at
///   (libzstd: `dictStartIndex < idx`; [`MatchState::lowest_match_index`]
///   is the one inclusive window bound).
/// * (E3) repcodes: nonzero on entry (a dictionary's are, and the initial
///   ones), afterwards the distance to an (E2) candidate; a repcode is
///   read at `p - offset` only after `offset <= p - dict_start`, so
///   `dict_start <= p - offset < p`.
/// * (E4) `hash_ptr` returns `< 1 << hbits == table.len()` for both tables.
fn compress_block_ext_generic<const MLS: u32, C: MatchCount>(
    mc: C,
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let hbits_l = ms.cparams.hash_log;
    let hbits_s = ms.cparams.chain_log;
    let istart = block.start;
    let iend = block.end;
    assert!(istart <= iend && iend <= src.end());
    assert!((1..=32).contains(&hbits_l) && (1..=32).contains(&hbits_s));
    // C: lowLimit = ZSTD_getLowestMatchIndex(ms, endIndex, windowLog).
    let dict_start = ms.lowest_match_index(iend - 1);
    let prefix_start = ms.window().dict_limit();
    assert!(dict_start < prefix_start && prefix_start <= istart);
    // C: ilimit = iend - 8, possibly below istart.
    let ilimit = iend.saturating_sub(8);

    let mut ip = istart;
    let mut anchor = istart;
    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    assert!(offset_1 > 0 && offset_2 > 0); // (E3)

    let (hash_long, hash_small, _) = ms.ws.tables_mut();
    assert_eq!(hash_long.len(), 1usize << hbits_l); // (E4)
    assert_eq!(hash_small.len(), 1usize << hbits_s); // (E4)

    // The segment a match at `idx` lies in starts here: catch-up stops at
    // it.
    let low_of = |idx: usize| {
        if idx < prefix_start {
            dict_start
        } else {
            prefix_start
        }
    };

    // Search Loop. SAFETY, for every unchecked access below: (E1) for the
    // ip-derived reads, (E2) for the candidate reads, (E3) for the repcode
    // reads, (E4) for the tables.
    unsafe {
        // < instead of <=, because (ip+1)
        while ip < ilimit {
            let h_small = hash_ptr::<MLS>(src, ip, hbits_s);
            let match_index = tget(hash_small, h_small);
            let h_long = hash_ptr::<8>(src, ip, hbits_l);
            let match_long_index = tget(hash_long, h_long);
            let curr = ip;
            // offset_1 expected <= curr + 1
            let rep_index = (curr as u32 + 1).wrapping_sub(offset_1);
            // update hash table
            tset(hash_small, h_small, curr);
            tset(hash_long, h_long, curr);

            let m_length;
            // note: we are searching at curr+1
            if index_overlap_check(prefix_start, rep_index)
                && offset_1 as usize <= curr + 1 - dict_start
                && read32(src, rep_index as usize) == read32(src, ip + 1)
            {
                let rep_index = rep_index as usize;
                m_length = mc.count(src, ip + 1 + 4, rep_index + 4, iend) + 4;
                ip += 1;
                out.store_seq(
                    src,
                    anchor,
                    ip - anchor,
                    iend,
                    REPCODE1_TO_OFFBASE,
                    m_length,
                );
            } else {
                let (offset, length) = if candidate_valid(match_long_index, dict_start, curr)
                    && read64(src, match_long_index) == read64(src, ip)
                {
                    let mut match_long = match_long_index;
                    let low = low_of(match_long);
                    let mut length = mc.count(src, ip + 8, match_long + 8, iend) + 8;
                    let offset = (curr - match_long_index) as u32;
                    // catch up
                    while ((ip > anchor) & (match_long > low))
                        && byte(src, ip - 1) == byte(src, match_long - 1)
                    {
                        ip -= 1;
                        match_long -= 1;
                        length += 1;
                    }
                    (offset, length)
                } else if candidate_valid(match_index, dict_start, curr)
                    && read32(src, match_index) == read32(src, ip)
                {
                    let h3 = hash_ptr::<8>(src, ip + 1, hbits_l);
                    let match_index3 = tget(hash_long, h3);
                    tset(hash_long, h3, curr + 1);
                    if candidate_valid(match_index3, dict_start, curr + 1)
                        && read64(src, match_index3) == read64(src, ip + 1)
                    {
                        let mut match3 = match_index3;
                        let low = low_of(match3);
                        let mut length = mc.count(src, ip + 9, match3 + 8, iend) + 8;
                        ip += 1;
                        let offset = (curr + 1 - match_index3) as u32;
                        // catch up
                        while ((ip > anchor) & (match3 > low))
                            && byte(src, ip - 1) == byte(src, match3 - 1)
                        {
                            ip -= 1;
                            match3 -= 1;
                            length += 1;
                        }
                        (offset, length)
                    } else {
                        let mut matchs = match_index;
                        let low = low_of(matchs);
                        let mut length = mc.count(src, ip + 4, matchs + 4, iend) + 4;
                        let offset = (curr - match_index) as u32;
                        // catch up
                        while ((ip > anchor) & (matchs > low))
                            && byte(src, ip - 1) == byte(src, matchs - 1)
                        {
                            ip -= 1;
                            matchs -= 1;
                            length += 1;
                        }
                        (offset, length)
                    }
                } else {
                    ip += ((ip - anchor) >> K_SEARCH_STRENGTH) + 1;
                    continue;
                };
                offset_2 = offset_1;
                offset_1 = offset;
                m_length = length;
                out.store_seq(
                    src,
                    anchor,
                    ip - anchor,
                    iend,
                    offset_to_offbase(offset),
                    m_length,
                );
            }

            // move to next sequence start
            ip += m_length;
            anchor = ip;

            if ip <= ilimit {
                // Complementary insertion, done after iLimit test, as
                // candidates could be > iend-8
                let index_to_insert = curr + 2;
                tset(
                    hash_long,
                    hash_ptr::<8>(src, index_to_insert, hbits_l),
                    index_to_insert,
                );
                tset(hash_long, hash_ptr::<8>(src, ip - 2, hbits_l), ip - 2);
                tset(
                    hash_small,
                    hash_ptr::<MLS>(src, index_to_insert, hbits_s),
                    index_to_insert,
                );
                tset(hash_small, hash_ptr::<MLS>(src, ip - 1, hbits_s), ip - 1);

                // check immediate repcode
                while ip <= ilimit {
                    let current2 = ip;
                    let rep_index2 = (current2 as u32).wrapping_sub(offset_2);
                    if index_overlap_check(prefix_start, rep_index2)
                        && offset_2 as usize <= current2 - dict_start
                        && read32(src, rep_index2 as usize) == read32(src, ip)
                    {
                        let rep_length2 = mc.count(src, ip + 4, rep_index2 as usize + 4, iend) + 4;
                        // swap offset_2 <=> offset_1
                        std::mem::swap(&mut offset_1, &mut offset_2);
                        out.store_seq(src, anchor, 0, iend, REPCODE1_TO_OFFBASE, rep_length2);
                        tset(hash_small, hash_ptr::<MLS>(src, ip, hbits_s), current2);
                        tset(hash_long, hash_ptr::<8>(src, ip, hbits_l), current2);
                        ip += rep_length2;
                        anchor = ip;
                        continue;
                    }
                    break;
                }
            }
        }
    }

    // save reps for next block
    rep[0] = offset_1;
    rep[1] = offset_2;

    // Return the anchor of the last literals
    anchor
}

/// `ZSTD_compressBlock_doubleFast`. Contract as [`super::fast::compress_block`].
pub fn compress_block(
    ms: &mut MatchState,
    src: Src,
    block: Block,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let block = block.range();
    // ZSTD_selectBlockCompressor: ZSTD_compressBlock_doubleFast_extDict
    // while dictionary content is in reach. Each loop has its own out of
    // line instance, so the dictionary one leaves the other's code as is.
    let ext = ms.ext_dict_in_reach(block.end);
    match simd_level() {
        // SAFETY: fearless_simd constructs the witness only after detecting
        // AVX2 on this CPU.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(w) => unsafe {
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

/// [`compress_block`] compiled with AVX2, counting 32 bytes per step.
///
/// # Safety
///
/// The CPU must support AVX2 (the witness proves it).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
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
    if EXT {
        return match ms.cparams.min_match {
            5 => compress_block_ext_generic::<5, C>(mc, ms, src, block, rep, out),
            6 => compress_block_ext_generic::<6, C>(mc, ms, src, block, rep, out),
            7 => compress_block_ext_generic::<7, C>(mc, ms, src, block, rep, out),
            _ => compress_block_ext_generic::<4, C>(mc, ms, src, block, rep, out),
        };
    }
    match ms.cparams.min_match {
        5 => compress_block_generic::<5, C>(mc, ms, src, block, rep, out),
        6 => compress_block_generic::<6, C>(mc, ms, src, block, rep, out),
        7 => compress_block_generic::<7, C>(mc, ms, src, block, rep, out),
        _ => compress_block_generic::<4, C>(mc, ms, src, block, rep, out),
    }
}

/// `ZSTD_compressBlock_doubleFast_dictMatchState`: [`compress_block`] with
/// the dictionary `dms` attached, searched by the `ZSTD_dictMatchState`
/// rules.
pub fn compress_block_dms(
    ms: &mut MatchState,
    src: Src,
    block: Block,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    dms: DictMatchState,
) -> usize {
    let block = block.range();
    match ms.cparams.min_match {
        5 => compress_block_dms_generic::<5>(ms, src, block, rep, out, dms),
        6 => compress_block_dms_generic::<6>(ms, src, block, rep, out, dms),
        7 => compress_block_dms_generic::<7>(ms, src, block, rep, out, dms),
        _ => compress_block_dms_generic::<4>(ms, src, block, rep, out, dms),
    }
}

/// `ZSTD_compressBlock_doubleFast_dictMatchState_generic(ms, seqStore,
/// rep, src, srcSize, mls)`, monomorphized over `MLS`. Each position checks
/// the repcode at the next one, then the frame's long candidate, else the
/// dictionary's (its tables are tagged, [`tags_match`]), then a short
/// candidate, the frame's or, where the frame's is no candidate, the
/// dictionary's, which a long one at the next position supersedes. The
/// input starts the window (`prefix_lowest`), the dictionary's content
/// lies below it ([`DictMatchState::src_below`]), and no repcode is
/// disabled.
///
/// Bounds: inside the search loop `ip < ilimit = iend - 8`, and after a
/// match `curr + 2 < ip <= ilimit` for the complementary insertion; a
/// frame candidate is read only when `prefix_lowest <= idx` and it is
/// below the position it is compared with, a dictionary one only above
/// the content's start, the dictionary's tables holding no index but `0`
/// and loaded positions at least 8 bytes before its end; a repcode only
/// where [`dms_rep_source`] admits it.
#[inline(never)]
fn compress_block_dms_generic<const MLS: u32>(
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    dms: DictMatchState,
) -> usize {
    let mc = Fallback::new();
    let h_bits_l = ms.cparams.hash_log;
    let h_bits_s = ms.cparams.chain_log;
    let istart = block.start;
    let iend = block.end;
    // ZSTD_getLowestPrefixIndex with a dictionary: the input's start.
    let prefix_lowest = ms.window().dict_limit();
    assert!(src.lo() <= prefix_lowest && prefix_lowest <= istart && istart <= iend);
    assert!(iend <= src.end());
    assert!((1..=32).contains(&h_bits_l) && (1..=32).contains(&h_bits_s));
    let dict = dms.src_below(prefix_lowest);
    let dict_start = dict.lo();
    let dict_h_bits_l = dms.ms.cparams.hash_log + SHORT_CACHE_TAG_BITS;
    let dict_h_bits_s = dms.ms.cparams.chain_log + SHORT_CACHE_TAG_BITS;
    assert!(dict_h_bits_l <= 32 && dict_h_bits_s <= 32);
    let (dict_hash_long, dict_hash_small, _) = dms.ms.tables();
    assert_eq!(dict_hash_long.len(), 1usize << dms.ms.cparams.hash_log);
    assert_eq!(dict_hash_small.len(), 1usize << dms.ms.cparams.chain_log);
    // A dictionary table entry as an index of this window: `0` stays below
    // the content.
    let dict_index =
        |packed: usize| (packed >> SHORT_CACHE_TAG_BITS) + (dict_start - WINDOW_START_INDEX);
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

    let mut ip = istart;
    let mut anchor = istart;
    let mut offset_1 = rep[0] as usize;
    let mut offset_2 = rep[1] as usize;
    let dict_and_prefix_length = ip - dict_start;
    ip += (dict_and_prefix_length == 0) as usize;

    let (hash_long, hash_small, _) = ms.ws.tables_mut();
    assert_eq!(hash_long.len(), 1usize << h_bits_l);
    assert_eq!(hash_small.len(), 1usize << h_bits_s);

    // SAFETY: the bounds above, for every read and table access.
    unsafe {
        // Main Search Loop: < instead of <=, because repcode check at (ip+1)
        'outer: while ip < ilimit {
            let h2 = hash_ptr::<8>(src, ip, h_bits_l);
            let h = hash_ptr::<MLS>(src, ip, h_bits_s);
            let dict_hash_and_tag_l = hash_ptr::<8>(src, ip, dict_h_bits_l);
            let dict_hash_and_tag_s = hash_ptr::<MLS>(src, ip, dict_h_bits_s);
            let dict_match_index_and_tag_l =
                tget(dict_hash_long, dict_hash_and_tag_l >> SHORT_CACHE_TAG_BITS);
            let dict_match_index_and_tag_s =
                tget(dict_hash_small, dict_hash_and_tag_s >> SHORT_CACHE_TAG_BITS);
            let dict_tags_match_l = tags_match(dict_match_index_and_tag_l, dict_hash_and_tag_l);
            let dict_tags_match_s = tags_match(dict_match_index_and_tag_s, dict_hash_and_tag_s);
            let curr = ip;
            let match_index_l = tget(hash_long, h2);
            let mut match_index_s = tget(hash_small, h);
            let rep_index = (curr + 1).wrapping_sub(offset_1);
            // update hash tables
            tset(hash_long, h2, curr);
            tset(hash_small, h, curr);

            let found = 'search: {
                // check repcode
                if let Some(rep_src) = dms_rep_source(src, dict, rep_index, curr + 1) {
                    if read32(rep_src, rep_index) == read32(src, ip + 1) {
                        let m_length =
                            count_dms(mc, src, ip + 1 + 4, iend, dict, rep_index + 4) + 4;
                        ip += 1;
                        out.store_seq(
                            src,
                            anchor,
                            ip - anchor,
                            iend,
                            REPCODE1_TO_OFFBASE,
                            m_length,
                        );
                        break 'search Found::Stored { m_length };
                    }
                }

                if candidate_valid(match_index_l, prefix_lowest, curr)
                    && read64(src, match_index_l) == read64(src, ip)
                {
                    // check prefix long match
                    let mut match_long = match_index_l;
                    let mut m_length = mc.count(src, ip + 8, match_long + 8, iend) + 8;
                    let offset = ip - match_long;
                    // catch up
                    while ip > anchor
                        && match_long > prefix_lowest
                        && byte(src, ip - 1) == byte(src, match_long - 1)
                    {
                        ip -= 1;
                        match_long -= 1;
                        m_length += 1;
                    }
                    break 'search Found::Match {
                        offset: offset as u32,
                        m_length,
                    };
                } else if dict_tags_match_l {
                    // check dictMatchState long match
                    let mut dict_match_l = dict_index(dict_match_index_and_tag_l);
                    if dict_match_l > dict_start && read64(dict, dict_match_l) == read64(src, ip) {
                        let mut m_length =
                            count_2segments(mc, src, ip + 8, iend, dict, dict_match_l + 8) + 8;
                        let offset = curr - dict_match_l;
                        // catch up
                        while ip > anchor
                            && dict_match_l > dict_start
                            && byte(src, ip - 1) == byte(dict, dict_match_l - 1)
                        {
                            ip -= 1;
                            dict_match_l -= 1;
                            m_length += 1;
                        }
                        break 'search Found::Match {
                            offset: offset as u32,
                            m_length,
                        };
                    }
                }

                let short_found = if match_index_s > prefix_lowest {
                    // short match candidate
                    match_index_s < curr && read32(src, match_index_s) == read32(src, ip)
                } else if dict_tags_match_s {
                    // check dictMatchState short match
                    match_index_s = dict_index(dict_match_index_and_tag_s);
                    match_index_s > dict_start && read32(dict, match_index_s) == read32(src, ip)
                } else {
                    false
                };
                if !short_found {
                    ip += ((ip - anchor) >> K_SEARCH_STRENGTH) + 1;
                    continue 'outer;
                }

                // _search_next_long
                {
                    let hl3 = hash_ptr::<8>(src, ip + 1, h_bits_l);
                    let dict_hash_and_tag_l3 = hash_ptr::<8>(src, ip + 1, dict_h_bits_l);
                    let match_index_l3 = tget(hash_long, hl3);
                    let dict_match_index_and_tag_l3 =
                        tget(dict_hash_long, dict_hash_and_tag_l3 >> SHORT_CACHE_TAG_BITS);
                    let dict_tags_match_l3 =
                        tags_match(dict_match_index_and_tag_l3, dict_hash_and_tag_l3);
                    tset(hash_long, hl3, curr + 1);

                    if candidate_valid(match_index_l3, prefix_lowest, curr + 1)
                        && read64(src, match_index_l3) == read64(src, ip + 1)
                    {
                        // check prefix long +1 match
                        let mut match_l3 = match_index_l3;
                        let mut m_length = mc.count(src, ip + 9, match_l3 + 8, iend) + 8;
                        ip += 1;
                        let offset = ip - match_l3;
                        // catch up
                        while ip > anchor
                            && match_l3 > prefix_lowest
                            && byte(src, ip - 1) == byte(src, match_l3 - 1)
                        {
                            ip -= 1;
                            match_l3 -= 1;
                            m_length += 1;
                        }
                        break 'search Found::Match {
                            offset: offset as u32,
                            m_length,
                        };
                    } else if dict_tags_match_l3 {
                        // check dict long +1 match
                        let mut dict_match_l3 = dict_index(dict_match_index_and_tag_l3);
                        if dict_match_l3 > dict_start
                            && read64(dict, dict_match_l3) == read64(src, ip + 1)
                        {
                            let mut m_length =
                                count_2segments(mc, src, ip + 1 + 8, iend, dict, dict_match_l3 + 8)
                                    + 8;
                            ip += 1;
                            let offset = curr + 1 - dict_match_l3;
                            // catch up
                            while ip > anchor
                                && dict_match_l3 > dict_start
                                && byte(src, ip - 1) == byte(dict, dict_match_l3 - 1)
                            {
                                ip -= 1;
                                dict_match_l3 -= 1;
                                m_length += 1;
                            }
                            break 'search Found::Match {
                                offset: offset as u32,
                                m_length,
                            };
                        }
                    }
                }

                // if no long +1 match, explore the short match we found
                let mut m = match_index_s;
                if match_index_s < prefix_lowest {
                    let mut m_length = count_2segments(mc, src, ip + 4, iend, dict, m + 4) + 4;
                    let offset = curr - m;
                    // catch up
                    while ip > anchor && m > dict_start && byte(src, ip - 1) == byte(dict, m - 1) {
                        ip -= 1;
                        m -= 1;
                        m_length += 1;
                    }
                    Found::Match {
                        offset: offset as u32,
                        m_length,
                    }
                } else {
                    let mut m_length = mc.count(src, ip + 4, m + 4, iend) + 4;
                    let offset = ip - m;
                    // catch up
                    while ip > anchor && m > prefix_lowest && byte(src, ip - 1) == byte(src, m - 1)
                    {
                        ip -= 1;
                        m -= 1;
                        m_length += 1;
                    }
                    Found::Match {
                        offset: offset as u32,
                        m_length,
                    }
                }
            };

            let m_length = match found {
                Found::Stored { m_length } => m_length,
                Found::Match { offset, m_length } => {
                    // _match_found
                    offset_2 = offset_1;
                    offset_1 = offset as usize;
                    out.store_seq(
                        src,
                        anchor,
                        ip - anchor,
                        iend,
                        offset_to_offbase(offset),
                        m_length,
                    );
                    m_length
                }
                Found::Cleanup => unreachable!(),
            };

            // _match_stored
            ip += m_length;
            anchor = ip;

            if ip <= ilimit {
                // Complementary insertion: done after iLimit test, as
                // candidates could be > iend-8.
                let index_to_insert = curr + 2;
                tset(
                    hash_long,
                    hash_ptr::<8>(src, index_to_insert, h_bits_l),
                    index_to_insert,
                );
                tset(hash_long, hash_ptr::<8>(src, ip - 2, h_bits_l), ip - 2);
                tset(
                    hash_small,
                    hash_ptr::<MLS>(src, index_to_insert, h_bits_s),
                    index_to_insert,
                );
                tset(hash_small, hash_ptr::<MLS>(src, ip - 1, h_bits_s), ip - 1);

                // check immediate repcode
                while ip <= ilimit {
                    let current2 = ip;
                    let rep_index2 = current2.wrapping_sub(offset_2);
                    let Some(rep_src) = dms_rep_source(src, dict, rep_index2, current2) else {
                        break;
                    };
                    if read32(rep_src, rep_index2) != read32(src, ip) {
                        break;
                    }
                    let rep_length2 = count_dms(mc, src, ip + 4, iend, dict, rep_index2 + 4) + 4;
                    // swap offset_2 <=> offset_1
                    std::mem::swap(&mut offset_1, &mut offset_2);
                    out.store_seq(src, anchor, 0, iend, REPCODE1_TO_OFFBASE, rep_length2);
                    tset(hash_small, hash_ptr::<MLS>(src, ip, h_bits_s), current2);
                    tset(hash_long, hash_ptr::<8>(src, ip, h_bits_l), current2);
                    ip += rep_length2;
                    anchor = ip;
                }
            }
        }
    }

    // save reps for next block
    rep[0] = offset_1 as u32;
    rep[1] = offset_2 as u32;
    anchor
}

/// `ZSTD_fillDoubleHashTableForCCtx(ms, end, ZSTD_dtlm_fast)` without
/// `FOR_CDICT`; with it `ZSTD_fillDoubleHashTableForCDict(ms, end,
/// ZSTD_dtlm_full)`, which also gives the large table the two positions
/// after each third one where their entry is empty, and tags every entry
/// of both tables ([`write_tagged`]): the slot is the high bits of a hash
/// [`SHORT_CACHE_TAG_BITS`] wider, the same slot as the untagged hash's.
fn fill_double_hash_table<const MLS: u32, const FOR_CDICT: bool>(
    ms: &mut MatchState,
    src: Src,
    start: usize,
    end: usize,
) {
    const FAST_HASH_FILL_STEP: usize = 3;
    let hbits_l = ms.cparams.hash_log;
    let hbits_s = ms.cparams.chain_log;
    assert!((1..=32).contains(&hbits_l) && (1..=32).contains(&hbits_s));
    assert!(!FOR_CDICT || hbits_l.max(hbits_s) + SHORT_CACHE_TAG_BITS <= 32);
    assert!(end <= src.end());
    let (hash_long, hash_small, _) = ms.ws.tables_mut();
    assert_eq!(hash_long.len(), 1usize << hbits_l);
    assert_eq!(hash_small.len(), 1usize << hbits_s);
    let mut ip = start;
    // C: for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep)
    // with iend = end - HASH_READ_SIZE. Both tables get every
    // fastHashFillStep position.
    while ip + FAST_HASH_FILL_STEP - 1 + HASH_READ_SIZE <= end {
        if FOR_CDICT {
            let (tbits_l, tbits_s) = (
                hbits_l + SHORT_CACHE_TAG_BITS,
                hbits_s + SHORT_CACHE_TAG_BITS,
            );
            // SAFETY: ip + 10 <= end <= src.end(); a hash of `tbits` bits
            // shifted down by the tag is < its table's size.
            unsafe {
                write_tagged(hash_small, hash_ptr::<MLS>(src, ip, tbits_s), ip);
                write_tagged(hash_long, hash_ptr::<8>(src, ip, tbits_l), ip);
                for i in 1..FAST_HASH_FILL_STEP {
                    // ip + i + 8 <= ip + 10 <= end.
                    let hash_and_tag = hash_ptr::<8>(src, ip + i, tbits_l);
                    if tget(hash_long, hash_and_tag >> SHORT_CACHE_TAG_BITS) == 0 {
                        write_tagged(hash_long, hash_and_tag, ip + i);
                    }
                }
            }
        } else {
            // SAFETY: ip + 10 <= end <= src.end(); hashes < their table sizes.
            unsafe {
                tset(hash_small, hash_ptr::<MLS>(src, ip, hbits_s), ip);
                tset(hash_long, hash_ptr::<8>(src, ip, hbits_l), ip);
            }
        }
        ip += FAST_HASH_FILL_STEP;
    }
}

/// [`fill_double_hash_table`] for `ms.cparams.min_match`.
fn fill_double_hash_table_from<const FOR_CDICT: bool>(
    ms: &mut MatchState,
    src: Src,
    start: usize,
    end: usize,
) {
    match ms.cparams.min_match {
        5 => fill_double_hash_table::<5, FOR_CDICT>(ms, src, start, end),
        6 => fill_double_hash_table::<6, FOR_CDICT>(ms, src, start, end),
        7 => fill_double_hash_table::<7, FOR_CDICT>(ms, src, start, end),
        _ => fill_double_hash_table::<4, FOR_CDICT>(ms, src, start, end),
    }
}

/// `ZSTD_fillDoubleHashTable(ms, end, ZSTD_dtlm_fast, ZSTD_tfp_forCCtx)`
/// for an entered prefix: insert every third position from
/// `ms.next_to_update` (its start) into both tables, then set
/// `next_to_update` to its end.
pub fn load_prefix(ms: &mut MatchState, src: Src, prefix: EnteredPrefix) {
    let end = ms.prefix_indices(prefix).end;
    assert!(end <= src.end());
    fill_double_hash_table_from::<false>(ms, src, ms.next_to_update, end);
    ms.next_to_update = end;
}

/// `ZSTD_fillDoubleHashTable(ms, end, ZSTD_dtlm_full, ZSTD_tfp_forCDict)`
/// for a dictionary's entered content: [`load_prefix`] that also inserts
/// the two positions after each third one into the large table where their
/// entry is empty, every entry tagged (see `fill_double_hash_table`).
pub fn load_dict_full(ms: &mut MatchState, src: Src, content: EnteredPrefix) {
    let end = ms.prefix_indices(content).end;
    assert!(end <= src.end());
    fill_double_hash_table_from::<true>(ms, src, ms.next_to_update, end);
    ms.next_to_update = end;
}

/// `ZSTD_fillDoubleHashTable(ms, end, ZSTD_dtlm_fast, ZSTD_tfp_forCCtx)` as
/// `ZSTD_ldm_fillFastTables` calls it before each block compressor run:
/// insert every third position from `ms.next_to_update` up to `end` into
/// both tables, leaving `next_to_update` where it is.
pub fn fill_double_hash_table_to(ms: &mut MatchState, src: Src, end: usize) {
    fill_double_hash_table_from::<false>(ms, src, ms.next_to_update, end);
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
        for level in [3, 4] {
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
            CParams::for_level(3, data.len()),
            1 << 17,
            [1, 4, 8],
        );
        assert!(stats.cross_block_matches > 0);
    }

    #[test]
    fn job_start_with_overlap_prefix_and_zero_reps() {
        let data = synthetic_text(600_000, 7);
        let cp = CParams::for_level(3, data.len());
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
            for level in [3, 4] {
                // Level 4 is greedy at some sizes: keep its table sizes but
                // give the finder the dfast tables it runs on.
                let cp = CParams {
                    strategy: Strategy::DFast,
                    ..CParams::for_level(level, len)
                };
                roundtrip_blocks(&finder(), data, cp, 1 << 17, [1, 4, 8]);
            }
        }
    }

    #[test]
    fn every_mls_variant() {
        let data = synthetic_text(400_000, 11);
        for min_match in 4..=7u32 {
            let cp = CParams {
                window_log: 19,
                chain_log: 14,
                hash_log: 16,
                search_log: 1,
                min_match,
                target_length: 0,
                strategy: Strategy::DFast,
            };
            let stats = roundtrip_blocks(&finder(), &data, cp, 1 << 17, [1, 4, 8]);
            assert!(stats.seqs > 100, "mls {min_match}");
        }
    }

    #[test]
    fn window_limited_matches_stay_inside_the_window() {
        let data = synthetic_text(300_000, 5);
        let cp = CParams {
            window_log: 10,
            chain_log: 10,
            hash_log: 11,
            search_log: 1,
            min_match: 5,
            target_length: 0,
            strategy: Strategy::DFast,
        };
        roundtrip_blocks(&finder(), &data, cp, 1 << 10, [1, 4, 8]);
        roundtrip_blocks(&finder(), &data, cp, 1000, [1, 4, 8]);
    }

    #[test]
    fn rep_disabled_at_block_start_is_restored() {
        let src = vec![7u8; 4000];
        let cp = CParams::for_level(3, src.len());
        let mut ms = MatchState::new(cp, 0);
        let mut store = SeqStore::new();
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
        let cp = CParams::for_level(3, long.len());
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
