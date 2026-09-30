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

use super::common::{byte, tget, MatchCount};
use super::matchstate::MatchState;
use fearless_simd::Fallback;

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
