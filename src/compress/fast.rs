//! Fast (single hash table, greedy) block compressor.
//!
//! Block-scoped form of the previous whole-input `find_matches_fast`,
//! restructured after `ZSTD_compressBlock_fast_noDict_generic`
//! (zstd_fast.c, libzstd 1.5.7): persistent `ms.hash_table`, absolute
//! positions, matches confined to `block`, `off_base` emitted directly.
//! Temporary implementation; to be replaced by a faithful port.

use super::matchstate::MatchState;
use super::seqstore::{offset_to_offbase, SeqStore, REPCODE1_TO_OFFBASE};
use std::ops::Range;

/// `HASH_READ_SIZE`: hashing reads up to 8 bytes.
const HASH_READ_SIZE: usize = 8;
/// `kSearchStrength`.
const K_SEARCH_STRENGTH: usize = 8;
const K_STEP_INCR: usize = 1 << (K_SEARCH_STRENGTH - 1);

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;
const PRIME7: u64 = 58295818150454627;
const PRIME8: u64 = 0xCF1BBCDCB7A56463;

#[inline]
fn read32(src: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(src[pos..pos + 4].try_into().unwrap())
}

#[inline]
fn read64(src: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(src[pos..pos + 8].try_into().unwrap())
}

/// `ZSTD_hashPtr(p, hlog, mls)`: hash of the `mls` bytes at `pos`
/// (`mls` in `4..=8`; other values hash 4 bytes like libzstd's default arm).
/// Reads 8 bytes for `mls >= 5`, so `pos + 8 <= src.len()` is required.
#[inline]
pub fn hash_ptr(src: &[u8], pos: usize, hlog: u32, mls: u32) -> usize {
    match mls {
        5 => (((read64(src, pos) << 24).wrapping_mul(PRIME5)) >> (64 - hlog)) as usize,
        6 => (((read64(src, pos) << 16).wrapping_mul(PRIME6)) >> (64 - hlog)) as usize,
        7 => (((read64(src, pos) << 8).wrapping_mul(PRIME7)) >> (64 - hlog)) as usize,
        8 => ((read64(src, pos).wrapping_mul(PRIME8)) >> (64 - hlog)) as usize,
        _ => (read32(src, pos).wrapping_mul(PRIME4) >> (32 - hlog)) as usize,
    }
}

/// `ZSTD_count(pIn, pMatch, pInLimit)`: length of the common prefix of
/// `src[a..limit]` and `src[b..]`, with `b < a`.
#[inline]
pub fn count(src: &[u8], a: usize, b: usize, limit: usize) -> usize {
    debug_assert!(b < a && a <= limit);
    let start = a;
    let (mut a, mut b) = (a, b);
    while a + 8 <= limit {
        let diff = read64(src, a) ^ read64(src, b);
        if diff != 0 {
            return a - start + (diff.trailing_zeros() / 8) as usize;
        }
        a += 8;
        b += 8;
    }
    while a < limit && src[a] == src[b] {
        a += 1;
        b += 1;
    }
    a - start
}

/// `ZSTD_match4Found_branch`.
#[inline]
fn match4_found(src: &[u8], cur: usize, match_idx: usize, idx_low_limit: usize) -> bool {
    match_idx >= idx_low_limit && read32(src, cur) == read32(src, match_idx)
}

enum Hit {
    None,
    Rep,
    Offset,
}

/// Find matches in `src[block]` and store them into `out`. Returns the anchor:
/// the start of the trailing literals `src[anchor..block.end]`, which the
/// caller appends to `out.lits` (`ZSTD_storeLastLiterals`).
///
/// `rep` is the repeat-offset history on entry and is updated on exit. A
/// repcode larger than `block.start - window_low` is disabled (`0`) for this
/// block and restored on exit when no match replaced it, as in libzstd.
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let hlog = ms.cparams.hash_log;
    let mls = ms.cparams.min_match;
    let target_length = ms.cparams.target_length as usize;
    let step_size = target_length + (target_length == 0) as usize + 1; // min 2
    let istart = block.start;
    let iend = block.end;
    debug_assert!(iend <= src.len());
    if iend < istart + HASH_READ_SIZE {
        return istart;
    }
    let ilimit = iend - HASH_READ_SIZE;
    let prefix_start = ms.lowest_prefix_index(iend);

    let mut anchor = istart;
    // In C the first input byte sits at index ZSTD_WINDOW_START_INDEX ==
    // lowestValid; here block 0 starts at position 0 < window_low, so clamp
    // first, then skip the prefix start exactly like `ip0 += (ip0 == prefixStart)`.
    let mut ip0 = istart.max(prefix_start);
    ip0 += (ip0 == prefix_start) as usize;

    let mut rep1 = rep[0];
    let mut rep2 = rep[1];
    let (mut saved1, mut saved2) = (0u32, 0u32);
    {
        let max_rep = (ip0 - prefix_start) as u32;
        if rep2 > max_rep {
            saved2 = rep2;
            rep2 = 0;
        }
        if rep1 > max_rep {
            saved1 = rep1;
            rep1 = 0;
        }
    }

    let ht = &mut ms.hash_table[..];

    // _start
    loop {
        let mut step = step_size;
        let mut next_step = ip0 + K_STEP_INCR;
        let mut ip1 = ip0 + 1;
        let mut ip2 = ip0 + step;
        let mut ip3 = ip2 + 1;
        if ip3 >= ilimit {
            break;
        }
        let mut hash0 = hash_ptr(src, ip0, hlog, mls);
        let mut hash1 = hash_ptr(src, ip1, hlog, mls);
        let mut match_idx = ht[hash0] as usize;

        let mut current0;
        let mut match0 = 0usize;
        let mut off_base = 0u32;
        let mut m_len = 0usize;

        let hit = loop {
            // load repcode match for ip[2]
            let rep_hit = rep1 > 0
                && ip2 >= rep1 as usize
                && read32(src, ip2) == read32(src, ip2 - rep1 as usize);

            // write back hash table entry
            current0 = ip0;
            ht[hash0] = ip0 as u32;

            // check repcode at ip[2]
            if rep_hit {
                ip0 = ip2;
                match0 = ip0 - rep1 as usize;
                let back = (src[ip0 - 1] == src[match0 - 1]) as usize;
                ip0 -= back;
                match0 -= back;
                off_base = REPCODE1_TO_OFFBASE;
                m_len = back + 4;
                // ip1 is before the repcode (ip2), so this write is safe.
                ht[hash1] = ip1 as u32;
                break Hit::Rep;
            }

            if match4_found(src, ip0, match_idx, prefix_start) {
                // ip1 == ip0 + 1, searching will resume after ip1.
                ht[hash1] = ip1 as u32;
                break Hit::Offset;
            }

            // lookup ip[1], hash ip[2], advance
            match_idx = ht[hash1] as usize;
            hash0 = hash1;
            hash1 = hash_ptr(src, ip2, hlog, mls);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip3;

            current0 = ip0;
            ht[hash0] = ip0 as u32;

            if match4_found(src, ip0, match_idx, prefix_start) {
                // Avoid writing an index >= the position where search resumes
                // (ip0 + 4 at least).
                if step <= 4 {
                    ht[hash1] = ip1 as u32;
                }
                break Hit::Offset;
            }

            match_idx = ht[hash1] as usize;
            hash0 = hash1;
            hash1 = hash_ptr(src, ip2, hlog, mls);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip0 + step;
            ip3 = ip1 + step;

            if ip2 >= next_step {
                step += 1;
                next_step += K_STEP_INCR;
            }
            if ip3 >= ilimit {
                break Hit::None;
            }
        };

        match hit {
            Hit::None => break,
            Hit::Offset => {
                match0 = match_idx;
                rep2 = rep1;
                rep1 = (ip0 - match0) as u32;
                off_base = offset_to_offbase(rep1);
                m_len = 4;
                // Count the backwards match length.
                while ip0 > anchor && match0 > prefix_start && src[ip0 - 1] == src[match0 - 1] {
                    ip0 -= 1;
                    match0 -= 1;
                    m_len += 1;
                }
            }
            Hit::Rep => {}
        }

        // _match: count the forward length.
        m_len += count(src, ip0 + m_len, match0 + m_len, iend);
        out.store_seq(src, anchor, ip0 - anchor, off_base, m_len);
        ip0 += m_len;
        anchor = ip0;

        // Fill table and check for immediate repcode.
        if ip0 <= ilimit {
            ht[hash_ptr(src, current0 + 2, hlog, mls)] = (current0 + 2) as u32;
            ht[hash_ptr(src, ip0 - 2, hlog, mls)] = (ip0 - 2) as u32;

            if rep2 > 0 {
                while ip0 <= ilimit
                    && ip0 >= rep2 as usize
                    && read32(src, ip0) == read32(src, ip0 - rep2 as usize)
                {
                    let r_length = count(src, ip0 + 4, ip0 + 4 - rep2 as usize, iend) + 4;
                    std::mem::swap(&mut rep1, &mut rep2);
                    ht[hash_ptr(src, ip0, hlog, mls)] = ip0 as u32;
                    ip0 += r_length;
                    out.store_seq(src, anchor, 0, REPCODE1_TO_OFFBASE, r_length);
                    anchor = ip0;
                }
            }
        }
    }

    // _cleanup: restore repcodes disabled at block start if still unused.
    if saved1 != 0 && rep1 != 0 {
        saved2 = saved1;
    }
    rep[0] = if rep1 != 0 { rep1 } else { saved1 };
    rep[1] = if rep2 != 0 { rep2 } else { saved2 };
    anchor
}

/// `ZSTD_fillHashTable(ms, end, ZSTD_dtlm_fast, ZSTD_tfp_forCCtx)`: insert
/// every third position of `src[range]` (from `ms.next_to_update`) into the
/// hash table, then set `next_to_update = range.end`.
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    const FAST_HASH_FILL_STEP: usize = 3;
    let hlog = ms.cparams.hash_log;
    let mls = ms.cparams.min_match;
    let end = range.end;
    debug_assert!(end <= src.len());
    let mut ip = ms.next_to_update.max(range.start).max(ms.window_low);
    // C: for (; ip + 3 < (end - HASH_READ_SIZE) + 2; ip += 3)
    while ip + FAST_HASH_FILL_STEP + HASH_READ_SIZE < end + 2 {
        ms.hash_table[hash_ptr(src, ip, hlog, mls)] = ip as u32;
        ip += FAST_HASH_FILL_STEP;
    }
    ms.next_to_update = end;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::params::CParams;

    fn run(src: &[u8], level: i32, block_size: usize) {
        let cp = CParams::for_level(level, src.len());
        let mut ms = MatchState::new(cp, 1);
        let mut rep = [1u32, 4, 8];
        let mut store = SeqStore::new();
        let mut start = 0;
        while start < src.len() {
            let end = (start + block_size).min(src.len());
            store.clear();
            let rep_in = rep;
            let anchor = compress_block(&mut ms, src, start..end, &mut rep, &mut store);
            store.lits.extend_from_slice(&src[anchor..end]);
            let got = store.reconstruct(&src[..start], rep_in);
            assert_eq!(
                got,
                &src[start..end],
                "block {start}..{end} at level {level}"
            );
            for s in &store.seqs {
                assert!(s.lit_len as usize + s.match_len() as usize <= end - start);
            }
            start = end;
        }
    }

    #[test]
    fn roundtrip_blocks_via_reconstruct() {
        let mut text = Vec::new();
        for i in 0..20000u32 {
            text.extend_from_slice(
                format!("line {} of the test corpus {}\n", i, i % 37).as_bytes(),
            );
        }
        for level in [1, 2] {
            run(&text, level, 1 << 17);
            run(&text, level, 1000);
        }
        run(&vec![0u8; 300_000], 1, 1 << 17);
        let f64s: Vec<u8> = (0..40000u64)
            .flat_map(|i| (i as f64 * 0.25).to_le_bytes())
            .collect();
        run(&f64s, 1, 1 << 17);
        run(b"short", 1, 1 << 17);
        run(b"", 1, 1 << 17);
    }

    #[test]
    fn rep_disabled_at_block_start_is_restored() {
        let src = vec![7u8; 4000];
        let cp = CParams::for_level(1, src.len());
        let mut ms = MatchState::new(cp, 1);
        let mut store = SeqStore::new();
        // rep[0] = 100 cannot be used from position 1: it is disabled and
        // then restored when a new offset replaces rep1 (saved1 -> rep[1]).
        let mut rep = [100u32, 4, 8];
        let anchor = compress_block(&mut ms, &src, 0..src.len(), &mut rep, &mut store);
        store.lits.extend_from_slice(&src[anchor..]);
        assert_eq!(store.reconstruct(&[], [100, 4, 8]), src);
        assert_eq!(rep[1], 100);
        assert_eq!(rep[2], 8);
    }
}
