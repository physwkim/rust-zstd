//! Double-fast block compressor: port of
//! `ZSTD_compressBlock_doubleFast_noDict_generic`,
//! `ZSTD_compressBlock_doubleFast` and `ZSTD_fillDoubleHashTable`
//! (zstd_double_fast.c, libzstd 1.5.7), no-dictionary case.
//!
//! The hash table is `hashLong` (`hBitsL = hash_log`, 8-byte hash) and the
//! chain table is `hashSmall` (`hBitsS = chain_log`, `mls`-byte hash).
//! Position conventions and the unchecked-read policy are those of
//! [`super::fast`].

use super::common::{
    byte, candidate_valid, hash_ptr, prefetch_unbounded, read32, read64, simd_level, tget, tset,
    MatchCount, HASH_READ_SIZE, K_SEARCH_STRENGTH,
};
use super::matchstate::MatchState;
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
///   the 4-byte read at `ip + 1` end before `iend <= src.len()`; after a
///   match the reads at `curr + 2`, `ip - 2`, `ip - 1` and `ip` are guarded
///   by `ip <= ilimit` (`curr + 4 <= ip`). `ip - 1 >= prefix_lowest >= 1`.
/// * (I2) candidates: a table entry is used only when
///   `prefix_lowest_index <= idx < ip` (`idxl1`: `< ip1`), see
///   [`candidate_valid`].
/// * (I3) repcodes: on entry `offset_1/2 <= ip - window_low(ip)`; afterwards
///   `offset_1 = ip - idx` with (I2) and `offset_2` is a former `offset_1`.
///   A repcode is only applied at positions `p >=` the `ip` it was derived
///   at, hence `1 <= p - offset < p` (`0` means disabled and reads `p`).
/// * (I4) `hash_ptr` returns `< 1 << hbits == table.len()` for both tables.
fn compress_block_generic<const MLS: u32, C: MatchCount>(
    mc: C,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let hbits_l = ms.cparams.hash_log;
    let hbits_s = ms.cparams.chain_log;
    let istart = block.start;
    let iend = block.end;
    assert!(istart <= iend && iend <= src.len());
    assert!((1..=32).contains(&hbits_l) && (1..=32).contains(&hbits_s));
    // presumes that, if there is a dictionary, it must be using Attach mode
    let prefix_lowest_index = ms.lowest_prefix_index(iend);
    let prefix_lowest = prefix_lowest_index;
    // C: ilimit = iend - HASH_READ_SIZE, possibly below istart (see fast.rs).
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);

    let mut anchor = istart;
    // init: ip += ((ip - prefixLowest) == 0), after the block-0 clamp of fast.rs.
    let mut ip = istart.max(prefix_lowest);
    ip += (ip == prefix_lowest) as usize;

    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let (mut offset_saved1, mut offset_saved2) = (0u32, 0u32);
    {
        let window_low = ms.lowest_prefix_index(ip);
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
                    prefetch_unbounded(src, ip1 + 64);
                    prefetch_unbounded(src, ip1 + 128);
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

/// `ZSTD_compressBlock_doubleFast`. Contract as [`super::fast::compress_block`].
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    match simd_level() {
        // SAFETY: fearless_simd constructs the witness only after detecting
        // AVX2 on this CPU.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(w) => unsafe { compress_block_avx2(w, ms, src, block, rep, out) },
        _ => compress_block_scalar(ms, src, block, rep, out),
    }
}

/// [`compress_block`] with the 8-byte [`count`](super::common::count).
#[inline(never)]
fn compress_block_scalar(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    compress_block_level(Fallback::new(), ms, src, block, rep, out)
}

/// [`compress_block`] compiled with AVX2, counting 32 bytes per step.
///
/// # Safety
///
/// The CPU must support AVX2 (the witness proves it).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn compress_block_avx2(
    mc: Avx2,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    compress_block_level(mc, ms, src, block, rep, out)
}

#[inline(always)]
fn compress_block_level<C: MatchCount>(
    mc: C,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    match ms.cparams.min_match {
        5 => compress_block_generic::<5, C>(mc, ms, src, block, rep, out),
        6 => compress_block_generic::<6, C>(mc, ms, src, block, rep, out),
        7 => compress_block_generic::<7, C>(mc, ms, src, block, rep, out),
        _ => compress_block_generic::<4, C>(mc, ms, src, block, rep, out),
    }
}

/// `ZSTD_fillDoubleHashTableForCCtx(ms, end, ZSTD_dtlm_fast)`.
fn fill_double_hash_table<const MLS: u32>(
    ms: &mut MatchState,
    src: &[u8],
    start: usize,
    end: usize,
) {
    const FAST_HASH_FILL_STEP: usize = 3;
    let hbits_l = ms.cparams.hash_log;
    let hbits_s = ms.cparams.chain_log;
    assert!((1..=32).contains(&hbits_l) && (1..=32).contains(&hbits_s));
    assert!(end <= src.len());
    let (hash_long, hash_small, _) = ms.ws.tables_mut();
    assert_eq!(hash_long.len(), 1usize << hbits_l);
    assert_eq!(hash_small.len(), 1usize << hbits_s);
    let mut ip = start;
    // C: for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep)
    // with iend = end - HASH_READ_SIZE. Only i == 0 is loaded for
    // ZSTD_dtlm_fast: both tables get every fastHashFillStep position.
    while ip + FAST_HASH_FILL_STEP - 1 + HASH_READ_SIZE <= end {
        // SAFETY: ip + 10 <= end <= src.len(); hashes < their table sizes.
        unsafe {
            tset(hash_small, hash_ptr::<MLS>(src, ip, hbits_s), ip);
            tset(hash_long, hash_ptr::<8>(src, ip, hbits_l), ip);
        }
        ip += FAST_HASH_FILL_STEP;
    }
}

/// `ZSTD_fillDoubleHashTable(ms, end, ZSTD_dtlm_fast, ZSTD_tfp_forCCtx)`:
/// insert every third position of `src[range]` from `ms.next_to_update`
/// into both tables, then set `next_to_update = range.end`.
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    let end = range.end;
    assert!(end <= src.len());
    let start = ms.next_to_update.max(range.start);
    debug_assert!(start >= 1, "position 0 is the empty-entry sentinel");
    match ms.cparams.min_match {
        5 => fill_double_hash_table::<5>(ms, src, start, end),
        6 => fill_double_hash_table::<6>(ms, src, start, end),
        7 => fill_double_hash_table::<7>(ms, src, start, end),
        _ => fill_double_hash_table::<4>(ms, src, start, end),
    }
    ms.next_to_update = end;
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
                1,
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
            1,
            [1, 4, 8],
        );
        assert!(stats.cross_block_matches > 0);
    }

    #[test]
    fn job_start_with_overlap_prefix_and_zero_reps() {
        let data = synthetic_text(600_000, 7);
        let cp = CParams::for_level(3, data.len());
        let window_low = 200_001;
        let job_start = window_low + (1 << 16);
        let stats = roundtrip_job(
            &finder(),
            &data,
            cp,
            1 << 17,
            window_low,
            job_start,
            [0, 0, 0],
        );
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
                roundtrip_blocks(
                    &finder(),
                    data,
                    CParams::for_level(level, len),
                    1 << 17,
                    1,
                    [1, 4, 8],
                );
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
            let stats = roundtrip_blocks(&finder(), &data, cp, 1 << 17, 1, [1, 4, 8]);
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
        roundtrip_blocks(&finder(), &data, cp, 1 << 10, 1, [1, 4, 8]);
        roundtrip_blocks(&finder(), &data, cp, 1000, 1, [1, 4, 8]);
    }

    #[test]
    fn rep_disabled_at_block_start_is_restored() {
        let src = vec![7u8; 4000];
        let cp = CParams::for_level(3, src.len());
        let mut ms = MatchState::new(cp, 1);
        let mut store = SeqStore::new();
        let mut rep = [100u32, 4, 8];
        let anchor = compress_block(&mut ms, &src, 0..src.len(), &mut rep, &mut store);
        store.lits.extend_from_slice(&src[anchor..]);
        assert_eq!(store.reconstruct(&[], [100, 4, 8]), src);
        assert_eq!(rep[1], 100);
        assert_eq!(rep[2], 8);
    }

    /// A table entry pointing past the current position (a MatchState
    /// reused on a shorter input) must be a miss, never a read past `src`.
    #[test]
    fn stale_table_entries_beyond_the_input_are_ignored() {
        let long = synthetic_text(300_000, 9);
        let cp = CParams::for_level(3, long.len());
        let mut ms = MatchState::new(cp, 1);
        let mut store = SeqStore::new();
        let mut rep = [1u32, 4, 8];
        compress_block(&mut ms, &long, 0..long.len(), &mut rep, &mut store);
        let short = &long[..20_000];
        store.clear();
        let mut rep = [1u32, 4, 8];
        let anchor = compress_block(&mut ms, short, 0..short.len(), &mut rep, &mut store);
        store.lits.extend_from_slice(&short[anchor..]);
        assert_eq!(store.reconstruct(&[], [1, 4, 8]), short);
    }
}
