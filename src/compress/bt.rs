//! Binary-tree match finder: port of the tree half of `zstd_opt.c`
//! (libzstd 1.5.7), no-dictionary mode.
//!
//! The chain table holds the tree: node `idx` owns the two entries at
//! `2 * (idx & bt_mask)` (smaller child) and `2 * (idx & bt_mask) + 1`
//! (larger child), `bt_mask = (1 << (chain_log - 1)) - 1`, so the tree is a
//! rolling buffer over the last `bt_mask` positions. Nodes are ordered by
//! the suffix starting at their position; every insertion makes the new
//! position the root of its hash bucket and splits the old tree around it.
//!
//! Positions follow the [`MatchState`] convention: absolute indices into
//! `src`, `window_low >= 1`, table entry `0` means empty.
//!
//! [`bt_get_all_matches`] is the optimal parser's finder
//! (`ZSTD_btGetAllMatches`): it inserts the position and collects every
//! repcode, 3-byte-hash and tree match that is longer than the previous one.

use super::common::{byte, read32, tget, MatchCount, HASH_READ_SIZE};
use super::matchstate::MatchState;
use super::seqstore::{offset_to_offbase, repcode_to_offbase, ZSTD_REP_NUM};
use fearless_simd::Fallback;
use std::ops::Range;

/// `ZSTD_OPT_NUM`: positions of one optimal-parser series.
pub const ZSTD_OPT_NUM: usize = 1 << 12;
/// `ZSTD_OPT_SIZE`: entries of the match and price tables.
pub const ZSTD_OPT_SIZE: usize = ZSTD_OPT_NUM + 3;

/// `ZSTD_match_t`: `off` is an `offBase` (repcode or `offset + 3`), `len`
/// the full match length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Match {
    pub off: u32,
    pub len: u32,
}

/// `prime3bytes`.
const PRIME3: u32 = 506832829;

/// `ZSTD_hash3Ptr(p, h)`: hash of the 3 bytes at `src[pos]`.
///
/// # Safety
/// `pos + 4 <= src.len()` (`MEM_readLE32`).
#[inline(always)]
unsafe fn hash3_ptr(src: &[u8], pos: usize, h: u32) -> usize {
    debug_assert!((1..=32).contains(&h));
    ((read32(src, pos) << 8).wrapping_mul(PRIME3) >> (32 - h)) as usize
}

/// `ZSTD_readMINMATCH(p, length)`: 4 bytes, or the low 3 bytes shifted up
/// for `length == 3`; only for comparisons.
///
/// # Safety
/// `pos + 4 <= src.len()`.
#[inline(always)]
unsafe fn read_min_match(src: &[u8], pos: usize, length: u32) -> u32 {
    if length == 3 {
        read32(src, pos) << 8
    } else {
        read32(src, pos)
    }
}

/// Checks behind every unchecked table access of this module: the tables
/// have the sizes the tree needs and `end` lies inside `src`.
pub(crate) fn assert_tree_bounds(ms: &MatchState, src: &[u8], end: usize) {
    let cp = &ms.cparams;
    let (hash_table, chain_table, _) = ms.tables();
    assert!(
        end <= src.len(),
        "tree end {end} past src.len() {}",
        src.len()
    );
    assert_eq!(hash_table.len(), 1usize << cp.hash_log, "hash_table size");
    assert!(cp.chain_log >= 1, "chain_log {} too small", cp.chain_log);
    assert_eq!(
        chain_table.len(),
        1usize << cp.chain_log,
        "chain_table size"
    );
}

/// `ZSTD_insertBt1(ms, ip, iend, target, mls, extDict = 0)`: add position
/// `ip` to the tree and return how many positions the caller may advance
/// (more than one over a repetitive pattern, where the positions after `ip`
/// would only re-insert the same long match).
///
/// `MLS` is the hash width of `ZSTD_hashPtr` (`3` hashes like `4`).
///
/// # Safety
/// `ip <= target`, `ip + HASH_READ_SIZE <= iend <= src.len()`, and the
/// tables pass [`assert_tree_bounds`].
#[inline(always)]
pub(crate) unsafe fn insert_bt1<M: MatchCount, const MLS: u32>(
    m: M,
    ms: &mut MatchState,
    src: &[u8],
    ip: usize,
    iend: usize,
    target: usize,
) -> usize {
    let cp = ms.cparams;
    let bt_mask = (1usize << (cp.chain_log - 1)) - 1;
    let curr = ip;
    let bt_low = curr.saturating_sub(bt_mask);
    // windowLow is based on target because we only need positions that will
    // be in the window at the end of the tree update.
    let window_low = ms.lowest_prefix_index(target);
    let (hash_table, bt, _) = ms.ws.tables_mut();
    let h = super::common::hash_ptr::<MLS>(src, ip, cp.hash_log);
    let mut match_index = tget(hash_table, h);
    let mut common_length_smaller = 0usize;
    let mut common_length_larger = 0usize;
    let mut match_end_idx = curr + 8 + 1;
    let mut best_length = 8usize;
    let mut nb_compares = 1u32 << cp.search_log;

    debug_assert!(curr <= target);
    *hash_table.get_unchecked_mut(h) = curr as u32; // Update Hash Table

    // `smaller_ptr` / `larger_ptr` point into `bt` or at `dummy32`, as in C;
    // every access to `bt` below goes through `bt_ptr`.
    let bt_ptr = bt.as_mut_ptr();
    let mut dummy32 = 0u32;
    let mut smaller_ptr: *mut u32 = bt_ptr.add(2 * (curr & bt_mask));
    let mut larger_ptr: *mut u32 = smaller_ptr.add(1);

    debug_assert!(window_low > 0);
    while nb_compares > 0 && match_index >= window_low {
        let next_ptr = bt_ptr.add(2 * (match_index & bt_mask));
        // guaranteed minimum nb of common bytes
        let mut match_length = common_length_smaller.min(common_length_larger);
        debug_assert!(match_index < curr);
        // `match_index + match_length < ip + match_length <= iend`: every
        // common length was counted against `iend`.
        match_length += m.count(src, ip + match_length, match_index + match_length, iend);

        if match_length > best_length {
            best_length = match_length;
            if match_length > match_end_idx - match_index {
                match_end_idx = match_index + match_length;
            }
        }

        if ip + match_length == iend {
            // equal : no way to know if inf or sup; drop, to guarantee
            // consistency (miss a bit of compression, but other solutions
            // can corrupt the tree)
            break;
        }

        // `ip + match_length < iend`: both bytes are inside `src`.
        if byte(src, match_index + match_length) < byte(src, ip + match_length) {
            // match is smaller than current
            *smaller_ptr = match_index as u32;
            common_length_smaller = match_length;
            if match_index <= bt_low {
                // beyond tree size, stop searching
                smaller_ptr = &mut dummy32;
                break;
            }
            // new "candidate" => larger than match, which was smaller than target
            smaller_ptr = next_ptr.add(1);
            // new matchIndex, larger than previous and closer to current
            match_index = *next_ptr.add(1) as usize;
        } else {
            // match is larger than current
            *larger_ptr = match_index as u32;
            common_length_larger = match_length;
            if match_index <= bt_low {
                // beyond tree size, stop searching
                larger_ptr = &mut dummy32;
                break;
            }
            larger_ptr = next_ptr;
            match_index = *next_ptr as usize;
        }
        nb_compares -= 1;
    }

    *smaller_ptr = 0;
    *larger_ptr = 0;
    let positions = if best_length > 384 {
        // speed optimization
        192.min(best_length - 384)
    } else {
        0
    };
    debug_assert!(match_end_idx > curr + 8);
    positions.max(match_end_idx - (curr + 8))
}

/// `ZSTD_updateTree_internal(ms, ip, iend, mls, ZSTD_noDict)`: insert
/// `[next_to_update, ip)` into the tree and set `next_to_update = ip`.
///
/// # Safety
/// `ip + HASH_READ_SIZE <= iend <= src.len()` and the tables pass
/// [`assert_tree_bounds`].
#[inline(always)]
pub(crate) unsafe fn update_tree_internal<M: MatchCount, const MLS: u32>(
    m: M,
    ms: &mut MatchState,
    src: &[u8],
    ip: usize,
    iend: usize,
) {
    let target = ip;
    let mut idx = ms.next_to_update;
    while idx < target {
        let forward = insert_bt1::<M, MLS>(m, ms, src, idx, iend, target);
        debug_assert!(forward > 0);
        idx += forward;
    }
    ms.next_to_update = target;
}

/// `ZSTD_insertAndFindFirstIndexHash3`: insert `[next_to_update3, ip)` into
/// the 3-byte hash table, set `next_to_update3 = ip` and return the entry of
/// `ip`'s hash.
///
/// # Safety
/// `ip + 4 <= src.len()`; `hash_table3` holds `1 << hash_log3` entries,
/// `hash_log3 >= 1`.
#[inline(always)]
unsafe fn insert_and_find_first_index_hash3(
    hash_table3: &mut [u32],
    hash_log3: u32,
    next_to_update3: &mut usize,
    src: &[u8],
    ip: usize,
) -> usize {
    let mut idx = *next_to_update3;
    let target = ip;
    let hash3 = hash3_ptr(src, ip, hash_log3);
    while idx < target {
        *hash_table3.get_unchecked_mut(hash3_ptr(src, idx, hash_log3)) = idx as u32;
        idx += 1;
    }
    *next_to_update3 = target;
    tget(hash_table3, hash3)
}

/// `ZSTD_insertBtAndGetAllMatches(..., dictMode = ZSTD_noDict, mls)`:
/// insert `ip` into the tree and write to `matches` every candidate longer
/// than all previous ones, starting above `length_to_beat - 1`: repcodes
/// first (`ll0` shifts them as the decoder does after a zero literal
/// length), then the 3-byte hash (`MLS == 3`), then the tree walk. Returns
/// the number of matches, in increasing length.
///
/// # Safety
/// `ip + HASH_READ_SIZE <= i_limit <= src.len()`, `ip >= ms.window_low`,
/// `ms.next_to_update >= ip`, and the tables pass [`assert_opt_bounds`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn insert_bt_and_get_all_matches<M: MatchCount, const MLS: u32>(
    m: M,
    matches: &mut [Match; ZSTD_OPT_SIZE],
    ms: &mut MatchState,
    next_to_update3: &mut usize,
    src: &[u8],
    ip: usize,
    i_limit: usize,
    rep: &[u32; 3],
    ll0: u32,
    length_to_beat: u32,
) -> u32 {
    let cp = ms.cparams;
    let sufficient_len = (cp.target_length as usize).min(ZSTD_OPT_NUM - 1);
    let curr = ip;
    let min_match: u32 = if MLS == 3 { 3 } else { 4 };
    let bt_mask = (1usize << (cp.chain_log - 1)) - 1;
    let dict_limit = ms.window_low;
    let bt_low = curr.saturating_sub(bt_mask);
    let window_low = ms.lowest_prefix_index(curr);
    // `matchLow = windowLow ? windowLow : 1`; `window_low >= 1` here.
    let match_low = window_low;
    let hash_log3 = cp.hash_log3();
    let (hash_table, bt, hash_table3) = ms.ws.opt_tables_mut();
    let h = super::common::hash_ptr::<MLS>(src, ip, cp.hash_log);
    let mut match_index = tget(hash_table, h);
    let mut common_length_smaller = 0usize;
    let mut common_length_larger = 0usize;
    // farthest referenced position of any match => detects repetitive patterns
    let mut match_end_idx = curr + 8 + 1;
    let mut mnum = 0usize;
    let mut nb_compares = 1u32 << cp.search_log;
    let mut best_length = (length_to_beat - 1) as usize;

    // check repCode
    debug_assert!(ll0 <= 1);
    {
        let last_r = ZSTD_REP_NUM + ll0;
        for rep_code in ll0..last_r {
            let rep_offset = if rep_code == ZSTD_REP_NUM {
                rep[0].wrapping_sub(1)
            } else {
                rep[rep_code as usize]
            };
            let mut rep_len = 0usize;
            debug_assert!(curr >= dict_limit);
            // intentional overflow, discards 0 and -1: `curr > repIndex >=
            // dictLimit`
            if (rep_offset.wrapping_sub(1) as usize) < curr - dict_limit {
                let rep_index = curr - rep_offset as usize;
                // We must validate the repcode offset because when we're using
                // a dictionary the valid offset range shrinks when the
                // dictionary goes out of bounds.
                if rep_index >= window_low
                    && read_min_match(src, ip, min_match)
                        == read_min_match(src, rep_index, min_match)
                {
                    rep_len = m.count(
                        src,
                        ip + min_match as usize,
                        rep_index + min_match as usize,
                        i_limit,
                    ) + min_match as usize;
                }
            }
            // repIndex < dictLimit || repIndex >= curr: no extDict or
            // dictMatchState here.
            // save longer solution
            if rep_len > best_length {
                best_length = rep_len;
                // expect value between 1 and 3
                *matches.get_unchecked_mut(mnum) = Match {
                    off: repcode_to_offbase(rep_code - ll0 + 1),
                    len: rep_len as u32,
                };
                mnum += 1;
                if rep_len > sufficient_len || ip + rep_len == i_limit {
                    // best possible
                    return mnum as u32;
                }
            }
        }
    }

    // HC3 match finder
    if MLS == 3 && best_length < MLS as usize {
        let match_index3 =
            insert_and_find_first_index_hash3(hash_table3, hash_log3, next_to_update3, src, ip);
        // heuristic : longer distance likely too expensive
        if match_index3 >= match_low && curr.wrapping_sub(match_index3) < (1 << 18) {
            // `match_index3 < curr`: insertions stop below the current
            // position and the parser only moves forward.
            debug_assert!(match_index3 < curr);
            let mlen = m.count(src, ip, match_index3, i_limit);
            // save best solution
            if mlen >= MLS as usize {
                best_length = mlen;
                debug_assert!(mnum == 0); // no prior solution
                *matches.get_unchecked_mut(0) = Match {
                    off: offset_to_offbase((curr - match_index3) as u32),
                    len: mlen as u32,
                };
                mnum = 1;
                if mlen > sufficient_len || ip + mlen == i_limit {
                    // best possible length
                    ms.next_to_update = curr + 1; // skip insertion
                    return 1;
                }
            }
        }
    }

    *hash_table.get_unchecked_mut(h) = curr as u32; // Update Hash Table

    let bt_ptr = bt.as_mut_ptr();
    let mut dummy32 = 0u32;
    let mut smaller_ptr: *mut u32 = bt_ptr.add(2 * (curr & bt_mask));
    let mut larger_ptr: *mut u32 = smaller_ptr.add(1);

    while nb_compares > 0 && match_index >= match_low {
        let next_ptr = bt_ptr.add(2 * (match_index & bt_mask));
        // guaranteed minimum nb of common bytes
        let mut match_length = common_length_smaller.min(common_length_larger);
        debug_assert!(curr > match_index);
        match_length += m.count(src, ip + match_length, match_index + match_length, i_limit);

        if match_length > best_length {
            debug_assert!(match_end_idx > match_index);
            if match_length > match_end_idx - match_index {
                match_end_idx = match_index + match_length;
            }
            best_length = match_length;
            *matches.get_unchecked_mut(mnum) = Match {
                off: offset_to_offbase((curr - match_index) as u32),
                len: match_length as u32,
            };
            mnum += 1;
            if match_length > ZSTD_OPT_NUM || ip + match_length == i_limit {
                // equal : no way to know if inf or sup; drop, to preserve bt
                // consistency (miss a little bit of compression)
                break;
            }
        }

        // Every length reaching `i_limit` is a new best (earlier ones
        // returned or broke), so both bytes are inside `src`.
        debug_assert!(ip + match_length < i_limit);
        if byte(src, match_index + match_length) < byte(src, ip + match_length) {
            // match smaller than current
            *smaller_ptr = match_index as u32;
            common_length_smaller = match_length;
            if match_index <= bt_low {
                // beyond tree size, stop the search
                smaller_ptr = &mut dummy32;
                break;
            }
            smaller_ptr = next_ptr.add(1);
            match_index = *next_ptr.add(1) as usize;
        } else {
            *larger_ptr = match_index as u32;
            common_length_larger = match_length;
            if match_index <= bt_low {
                // beyond tree size, stop the search
                larger_ptr = &mut dummy32;
                break;
            }
            larger_ptr = next_ptr;
            match_index = *next_ptr as usize;
        }
        nb_compares -= 1;
    }

    *smaller_ptr = 0;
    *larger_ptr = 0;

    debug_assert!(match_end_idx > curr + 8);
    ms.next_to_update = match_end_idx - 8; // skip repetitive patterns
    mnum as u32
}

/// `ZSTD_btGetAllMatches_internal(..., ZSTD_noDict, mls)`: nothing inside an
/// area a previous long match let the tree skip, else bring the tree up to
/// `ip` and collect `ip`'s matches, see [`insert_bt_and_get_all_matches`].
///
/// # Safety
/// `ip + HASH_READ_SIZE <= i_high_limit <= src.len()`,
/// `ip >= ms.window_low`, and the tables pass [`assert_opt_bounds`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn bt_get_all_matches<M: MatchCount, const MLS: u32>(
    m: M,
    matches: &mut [Match; ZSTD_OPT_SIZE],
    ms: &mut MatchState,
    next_to_update3: &mut usize,
    src: &[u8],
    ip: usize,
    i_high_limit: usize,
    rep: &[u32; 3],
    ll0: u32,
    length_to_beat: u32,
) -> u32 {
    if ip < ms.next_to_update {
        return 0; // skipped area
    }
    update_tree_internal::<M, MLS>(m, ms, src, ip, i_high_limit);
    insert_bt_and_get_all_matches::<M, MLS>(
        m,
        matches,
        ms,
        next_to_update3,
        src,
        ip,
        i_high_limit,
        rep,
        ll0,
        length_to_beat,
    )
}

/// [`assert_tree_bounds`] plus the 3-byte hash table of `hash_log3`.
pub(crate) fn assert_opt_bounds(ms: &MatchState, src: &[u8], end: usize) {
    assert_tree_bounds(ms, src, end);
    let log3 = ms.cparams.hash_log3();
    let want = if log3 == 0 { 0 } else { 1usize << log3 };
    assert_eq!(ms.ws.hash3().len(), want, "hash_table3 size");
}

/// Empty the `hashTable` bucket, and the `hashTable3` bucket when there is
/// one, of every position in `range`, hashed as [`bt_get_all_matches`]
/// hashes them (`mls = BOUNDED(3, minMatch, 6)`). Undoes the insertions of
/// a parse over `range` into tables that were empty before it.
pub(crate) fn clear_hash_buckets(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    assert!(
        range.is_empty() || range.end - 1 + HASH_READ_SIZE <= src.len(),
        "range {range:?} too close to src end {}",
        src.len()
    );
    assert_opt_bounds(ms, src, 0);
    let cp = ms.cparams;
    let hash_log3 = cp.hash_log3();
    let (hash_table, _, hash_table3) = ms.ws.opt_tables_mut();
    fn clear<const MLS: u32>(
        hash_table: &mut [u32],
        hash_log: u32,
        hash_table3: &mut [u32],
        hash_log3: u32,
        src: &[u8],
        range: Range<usize>,
    ) {
        for p in range {
            // SAFETY: `p + HASH_READ_SIZE <= src.len()` (asserted above);
            // the hashes are below the asserted table sizes.
            unsafe {
                *hash_table.get_unchecked_mut(super::common::hash_ptr::<MLS>(src, p, hash_log)) = 0;
                if hash_log3 != 0 {
                    *hash_table3.get_unchecked_mut(hash3_ptr(src, p, hash_log3)) = 0;
                }
            }
        }
    }
    let (h, h3) = (cp.hash_log, hash_log3);
    match cp.min_match.clamp(3, 6) {
        5 => clear::<5>(hash_table, h, hash_table3, h3, src, range),
        6 => clear::<6>(hash_table, h, hash_table3, h3, src, range),
        _ => clear::<4>(hash_table, h, hash_table3, h3, src, range),
    }
}

/// `ZSTD_loadDictionaryContent`, binary-tree arm, for a raw-content prefix
/// `src[range]` (already cut to the table-sized suffix): `nextToUpdate` at
/// the prefix start, then, unless the prefix is at most `HASH_READ_SIZE`
/// bytes, `ZSTD_updateTree(ms, end - HASH_READ_SIZE, end)` and
/// `nextToUpdate = end`.
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    assert!(
        range.start >= ms.window_low,
        "prefix start {} below window_low {}",
        range.start,
        ms.window_low
    );
    ms.next_to_update = range.start;
    if range.len() <= HASH_READ_SIZE {
        return;
    }
    update_tree(ms, src, range.end - HASH_READ_SIZE, range.end);
    ms.next_to_update = range.end;
}

/// `ZSTD_updateTree(ms, ip, iend)`: [`update_tree_internal`] with the hash
/// width `minMatch` (as C passes it, not bounded to `3..=6`), e.g.
/// `ZSTD_loadDictionaryContent`'s `ZSTD_updateTree(ms, iend - 8, iend)`
/// that sorts a prefix into the tree before a job's first block.
///
/// Panics unless `ip + HASH_READ_SIZE <= iend <= src.len()` and the match
/// state has a chain table of `1 << chain_log` entries.
pub fn update_tree(ms: &mut MatchState, src: &[u8], ip: usize, iend: usize) {
    assert!(
        ip + super::common::HASH_READ_SIZE <= iend,
        "update_tree target {ip} closer than HASH_READ_SIZE to {iend}"
    );
    assert_tree_bounds(ms, src, iend);
    let m = Fallback::new();
    // SAFETY: the bounds were just asserted.
    unsafe {
        match ms.cparams.min_match {
            5 => update_tree_internal::<_, 5>(m, ms, src, ip, iend),
            6 => update_tree_internal::<_, 6>(m, ms, src, ip, iend),
            7 => update_tree_internal::<_, 7>(m, ms, src, ip, iend),
            _ => update_tree_internal::<_, 4>(m, ms, src, ip, iend),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::params::{CParams, Strategy};

    fn xorshift_text(len: usize, alphabet: u8) -> Vec<u8> {
        let mut x = 0x9E37_79B9u32;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                b'a' + (x % alphabet as u32) as u8
            })
            .collect()
    }

    /// Parameters with a chain table of `1 << chain_log` entries (the lazy
    /// strategies allocate one).
    fn tree_params(chain_log: u32, search_log: u32, min_match: u32) -> CParams {
        CParams {
            window_log: 20,
            chain_log,
            hash_log: 12,
            search_log,
            min_match,
            target_length: 0,
            strategy: Strategy::Lazy2,
        }
    }

    /// `(smaller, larger)` children of node `idx`.
    fn children(ms: &MatchState, idx: usize) -> (usize, usize) {
        let bt_mask = (1usize << (ms.cparams.chain_log - 1)) - 1;
        let (_, bt, _) = ms.tables();
        (
            bt[2 * (idx & bt_mask)] as usize,
            bt[2 * (idx & bt_mask) + 1] as usize,
        )
    }

    /// Every linked child orders against its parent as the tree requires:
    /// the smaller child's suffix sorts below the parent's, the larger
    /// child's above it (suffixes compared up to `end`).
    #[test]
    fn children_are_ordered_by_suffix() {
        for (alphabet, min_match) in [(2u8, 4u32), (4, 5), (26, 6), (3, 3)] {
            let src = xorshift_text(20_000, alphabet);
            let cp = tree_params(16, 6, min_match);
            let mut ms = MatchState::new(cp, 1);
            let end = src.len();
            update_tree(&mut ms, &src, end - 8, end);
            assert_eq!(ms.next_to_update, end - 8);
            let mut linked = 0;
            for idx in 1..end - 8 {
                let (smaller, larger) = children(&ms, idx);
                if smaller != 0 {
                    assert!(smaller < idx, "smaller child {smaller} of {idx} not older");
                    assert!(src[smaller..end] < src[idx..end], "smaller child of {idx}");
                    linked += 1;
                }
                if larger != 0 {
                    assert!(larger < idx, "larger child {larger} of {idx} not older");
                    assert!(src[larger..end] > src[idx..end], "larger child of {idx}");
                    linked += 1;
                }
            }
            assert!(linked > end / 2, "alphabet {alphabet}: only {linked} links");
        }
    }

    /// Over a run the first insertion finds a match reaching the end of the
    /// run and skips the positions it covers: `forward = matchEndIdx -
    /// (curr + 8)`, and the update still ends at `target`.
    #[test]
    fn repetitive_run_is_skipped() {
        let mut src = xorshift_text(4096, 26);
        src[1000..3000].fill(b'z');
        src[3000] = b'a';
        let cp = tree_params(16, 6, 4);
        let mut ms = MatchState::new(cp, 1);
        let end = src.len();
        assert_tree_bounds(&ms, &src, end);
        // SAFETY: bounds asserted; 1000 + 8 <= end.
        unsafe {
            ms.next_to_update = 1;
            update_tree_internal::<_, 4>(Fallback::new(), &mut ms, &src, 1001, end);
            assert_eq!(ms.next_to_update, 1001);
            // Position 1001 matches 1000 for 1999 bytes (up to 2999), so
            // the match ends at 1000 + 1999 and the next 1990 positions are
            // skipped.
            let forward = insert_bt1::<_, 4>(Fallback::new(), &mut ms, &src, 1001, end, 1001);
            assert_eq!(forward, 1000 + 1999 - (1001 + 8));
        }
    }

    /// A far match longer than 384 bytes that ends before `curr + 9` still
    /// skips `min(192, best_length - 384)` positions (speed optimization).
    #[test]
    fn long_far_match_forward_is_capped() {
        let mut src = xorshift_text(8192, 26);
        let (a, b) = (100usize, 5000usize);
        let copy: Vec<u8> = src[a..a + 1000].to_vec();
        src[b..b + 1000].copy_from_slice(&copy);
        let cp = tree_params(16, 6, 4);
        let mut ms = MatchState::new(cp, 1);
        let end = src.len();
        assert_tree_bounds(&ms, &src, end);
        // SAFETY: bounds asserted.
        unsafe {
            ms.next_to_update = 1;
            update_tree_internal::<_, 4>(Fallback::new(), &mut ms, &src, b, end);
            let forward = insert_bt1::<_, 4>(Fallback::new(), &mut ms, &src, b, end, b);
            // The match at `a` ends at `a + 1000` (or a little later by
            // chance), far below `b + 9`: only the 384-rule applies.
            assert_eq!(forward, 192);
        }
    }

    #[test]
    #[should_panic(expected = "closer than HASH_READ_SIZE")]
    fn update_tree_rejects_target_near_end() {
        let src = xorshift_text(100, 4);
        let mut ms = MatchState::new(tree_params(10, 4, 4), 1);
        update_tree(&mut ms, &src, 95, 100);
    }
}
