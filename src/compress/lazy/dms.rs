//! The lazy block compressors with a dictionary attached
//! (`ZSTD_dictMatchState`): `ZSTD_compressBlock_lazy_generic` with
//! `dictMode == ZSTD_dictMatchState`, whose searches go on into the
//! dictionary's own tables once the frame's are exhausted
//! (`ZSTD_HcFindBestMatch`, `ZSTD_RowFindBestMatch`,
//! `ZSTD_DUBT_findBestMatch` with `ZSTD_DUBT_findBetterDictMatch`).
//!
//! The frame's half of each search is the no-dictionary one, written out
//! again with the attempts it leaves counted: the no-dictionary loops
//! stay as they are, instruction for instruction. The input starts the
//! window (`prefixLowest`), the dictionary's content lies below it by the
//! frame's indices ([`DictMatchState::src_below`]), and no repcode is ever
//! disabled.

use super::{
    assert_block_bounds, depth_of, hash_salted, highbit32, mls_of, row_log_of, BtParams, BtSearch,
    HcSearch, RowSearch, RowTables, Search, SearchMethod, TagMask, BT_DUMMY, DUBT_UNSORTED_MARK,
    K_LAZY_SKIPPING_STEP, K_SEARCH_STRENGTH, ROW_HASH_TAG_BITS, ROW_HASH_TAG_MASK,
};
use crate::compress::common::{
    byte, candidate_valid, count_2segments, count_dms, dms_rep_source, prefetch, read32, tget,
    tset, MatchCount, Src,
};
use crate::compress::matchstate::{Block, DictMatchState, MatchState, WINDOW_START_INDEX};
use crate::compress::seqstore::{
    offbase_is_offset, offbase_to_offset, offset_to_offbase, SeqStore, REPCODE1_TO_OFFBASE,
};
use fearless_simd::Fallback;
use std::ops::Range;

/// The attached dictionary as the searches read it (`ms->dictMatchState`).
#[derive(Clone, Copy)]
struct Dms<'a> {
    /// The content by the frame's indices, ending at the input's start.
    dict: Src<'a>,
    /// `dms->hashTable`, `dms->chainTable`, `dms->tagTable`.
    hash: &'a [u32],
    chain: &'a [u32],
    tag: &'a [u8],
    /// `dms->cParams.hashLog`, `chainLog`; the row hash salt (libzstd's
    /// is `0` for a dictionary; any salt permutes the rows and tags alike).
    hash_log: u32,
    chain_log: u32,
    hash_salt: u64,
    /// `dmsIndexDelta`: a dictionary index plus this is the frame's.
    delta: usize,
    /// `dms->window.nextSrc - dms->window.base`: the content's end by the
    /// dictionary's indices (`dmsSize`, `dictHighLimit`).
    end: usize,
}

impl<'a> Dms<'a> {
    fn new(dms: DictMatchState<'a>, prefix_start: usize) -> Self {
        let (hash, chain, tag) = dms.ms.tables();
        let end = dms.src().end();
        Dms {
            dict: dms.src_below(prefix_start),
            hash,
            chain,
            tag,
            hash_log: dms.ms.cparams.hash_log,
            chain_log: dms.ms.cparams.chain_log,
            hash_salt: dms.ms.hash_salt,
            delta: prefix_start - end,
            end,
        }
    }

    /// The highest dictionary index whose 4 bytes are content: a table
    /// entry above it is a miss, which a dictionary's tables never hold.
    #[inline(always)]
    fn read_limit(self) -> usize {
        self.end.saturating_sub(3)
    }
}

/// `ZSTD_HcFindBestMatch` (`ZSTD_dictMatchState`).
struct HcDmsSearch<'a, const MLS: u32> {
    d: Dms<'a>,
}

impl<const MLS: u32> Search for HcDmsSearch<'_, MLS> {
    const ILIMIT_MARGIN: usize = 8;

    #[inline(always)]
    fn refill(&mut self, _ms: &mut MatchState, _src: Src, _ilimit: usize) {}

    #[inline(always)]
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: Src,
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        lazy_skipping: bool,
    ) -> usize {
        let chain_size = 1usize << ms.cparams.chain_log;
        let chain_mask = chain_size - 1;
        let curr = ip;
        let low_limit = ms.lowest_match_index(curr);
        let min_chain = curr.saturating_sub(chain_size);
        let mut nb_attempts = 1u32 << ms.cparams.search_log;
        let mut ml = 4 - 1;

        // SAFETY: as in `HcSearch::search_max`.
        let mut match_index =
            unsafe { HcSearch::<MLS>::insert_and_find_first_index(ms, src, ip, lazy_skipping) }
                as usize;
        let (_, chain_table, _) = ms.ws.tables();
        while candidate_valid(match_index, low_limit, curr) && nb_attempts > 0 {
            // SAFETY: as in `HcSearch::search_max`.
            let better = unsafe { read32(src, match_index + ml - 3) == read32(src, ip + ml - 3) };
            if better {
                // SAFETY: `match_index < ip <= iend <= src.end()`.
                let current_ml = unsafe { Search::count(self, src, ip, match_index, iend) };
                if current_ml > ml {
                    ml = current_ml;
                    *off_base = offset_to_offbase((curr - match_index) as u32);
                    if ip + current_ml == iend {
                        break;
                    }
                }
            }
            if match_index <= min_chain {
                break;
            }
            // SAFETY: `match_index & chain_mask < chain_table.len()`.
            match_index = unsafe { tget(chain_table, match_index & chain_mask) };
            nb_attempts -= 1;
        }

        let d = self.d;
        let dms_chain_size = 1usize << d.chain_log;
        let dms_chain_mask = dms_chain_size - 1;
        let dms_min_chain = d.end.saturating_sub(dms_chain_size);
        // SAFETY: `ip + 8 <= iend`; the hash is `< 1 << d.hash_log`, the
        // dictionary's hash table's length (asserted per block).
        let mut match_index =
            unsafe { tget(d.hash, hash_salted::<MLS>(src, ip, d.hash_log, 0) as usize) };
        // C tests `matchIndex >= dmsLowestIndex` only.
        while candidate_valid(match_index, WINDOW_START_INDEX, d.read_limit()) && nb_attempts > 0 {
            let m = match_index + d.delta;
            let mut current_ml = 0;
            // SAFETY: `m + 4 <= d.dict.end() < ip + 4 <= iend`.
            unsafe {
                if read32(d.dict, m) == read32(src, ip) {
                    current_ml =
                        count_2segments(Fallback::new(), src, ip + 4, iend, d.dict, m + 4) + 4;
                }
            }
            // save best solution
            if current_ml > ml {
                ml = current_ml;
                *off_base = offset_to_offbase((curr - m) as u32);
                if ip + current_ml == iend {
                    break; // best possible, avoids read overflow on next attempt
                }
            }
            if match_index <= dms_min_chain {
                break;
            }
            // SAFETY: `match_index & dms_chain_mask < d.chain.len()`.
            match_index = unsafe { tget(d.chain, match_index & dms_chain_mask) };
            nb_attempts -= 1;
        }
        ml
    }
}

/// `ZSTD_RowFindBestMatch` (`ZSTD_dictMatchState`).
struct RowDmsSearch<'a, M: TagMask, const MLS: u32, const ROW_LOG: u32> {
    row: RowSearch<M, MLS, ROW_LOG>,
    d: Dms<'a>,
}

impl<M: TagMask + MatchCount, const MLS: u32, const ROW_LOG: u32> Search
    for RowDmsSearch<'_, M, MLS, ROW_LOG>
{
    const ILIMIT_MARGIN: usize = RowSearch::<M, MLS, ROW_LOG>::ILIMIT_MARGIN;

    #[inline(always)]
    unsafe fn count(&self, src: Src, a: usize, b: usize, limit: usize) -> usize {
        self.row.mask.count(src, a, b, limit)
    }

    #[inline(always)]
    fn refill(&mut self, ms: &mut MatchState, src: Src, ilimit: usize) {
        self.row.refill(ms, src, ilimit)
    }

    #[inline(always)]
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: Src,
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        lazy_skipping: bool,
    ) -> usize {
        type R<M, const MLS: u32, const ROW_LOG: u32> = RowSearch<M, MLS, ROW_LOG>;
        let row_entries = R::<M, MLS, ROW_LOG>::ROW_ENTRIES;
        let row_mask = R::<M, MLS, ROW_LOG>::ROW_MASK;
        let curr = ip;
        let low_limit = ms.lowest_match_index(curr);
        // nb of searches is capped at nb entries per row
        let capped_search_log = ms.cparams.search_log.min(ROW_LOG);
        let group_width = M::group_width(row_entries as u32);
        let mut nb_attempts = 1u32 << capped_search_log;
        let mut ml = 4 - 1;

        // The dictionary's row (prefetched first in C).
        let d = self.d;
        // SAFETY: `ip + 8 <= iend`; the dictionary's row log is the
        // frame's and its tables are `1 << d.hash_log` long (asserted per
        // block), so the row lies inside them, see `RowSearch::hash`.
        let (dms_tag_row, dms_row, dms_tag) = unsafe {
            let dms_hash = R::<M, MLS, ROW_LOG>::hash(d.hash_log, d.hash_salt, src, ip);
            let rel = ((dms_hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize;
            (
                d.tag.get_unchecked(rel..rel + row_entries),
                d.hash.get_unchecked(rel..rel + row_entries),
                (dms_hash & ROW_HASH_TAG_MASK) as u8,
            )
        };

        let (hash_log, hash_salt) = (ms.cparams.hash_log, ms.hash_salt);
        let (hash, _, tag) = ms.ws.tables_mut();
        let mut t = RowTables {
            hash,
            tag,
            hash_log,
            hash_salt,
        };
        // Update the hashTable and tagTable up to (but not including) ip
        // SAFETY: as in `RowSearch::search_max`.
        let hash = unsafe {
            if !lazy_skipping {
                self.row
                    .update_internal(&mut t, &mut ms.next_to_update, src, ip, true);
                self.row.next_cached_hash(&t, src, curr)
            } else {
                ms.next_to_update = curr;
                R::<M, MLS, ROW_LOG>::hash(hash_log, hash_salt, src, ip)
            }
        };
        ms.hash_salt_entropy = ms.hash_salt_entropy.wrapping_add(hash); // collect salt entropy

        let rel_row = ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize;
        let tag = (hash & ROW_HASH_TAG_MASK) as u8;
        // SAFETY: `rel_row + ROW_ENTRIES <= tag_table.len()`, see `hash`.
        let tag_row = unsafe { t.tag.get_unchecked(rel_row..rel_row + row_entries) };
        let head_grouped = ((tag_row[0] as u32) & row_mask) * group_width;
        let mut num_matches = 0usize;
        let mut matches = self
            .row
            .mask
            .match_mask::<ROW_LOG>(tag_row, tag, head_grouped);

        // Cycle through the matches and prefetch
        while matches > 0 && nb_attempts > 0 {
            let match_pos = ((head_grouped + matches.trailing_zeros()) / group_width) & row_mask;
            matches &= matches - 1;
            // SAFETY: `rel_row + match_pos < hash_table.len()`.
            let match_index = unsafe { tget(t.hash, rel_row + match_pos as usize) };
            if match_pos == 0 {
                continue;
            }
            if !candidate_valid(match_index, low_limit, curr) {
                break;
            }
            prefetch(src, match_index);
            // SAFETY: `num_matches < nb_attempts <= ROW_HASH_MAX_ENTRIES`.
            unsafe { *self.row.match_buffer.get_unchecked_mut(num_matches) = match_index as u32 };
            num_matches += 1;
            nb_attempts -= 1;
        }

        // Speed opt: insert current byte into hashtable too.
        // SAFETY: `rel_row + pos < rel_row + ROW_ENTRIES <= table len`.
        unsafe {
            let pos = R::<M, MLS, ROW_LOG>::next_index(R::<M, MLS, ROW_LOG>::head(t.tag, rel_row));
            *t.tag.get_unchecked_mut(rel_row + pos) = tag;
            tset(t.hash, rel_row + pos, ms.next_to_update);
            ms.next_to_update += 1;
        }

        // Return the longest match
        for &match_index in &self.row.match_buffer[..num_matches] {
            let match_index = match_index as usize;
            // SAFETY: as in `RowSearch::search_max`.
            let better = unsafe { read32(src, match_index + ml - 3) == read32(src, ip + ml - 3) };
            if better {
                // SAFETY: `match_index < ip <= iend <= src.end()`.
                let current_ml = unsafe { Search::count(self, src, ip, match_index, iend) };
                if current_ml > ml {
                    ml = current_ml;
                    *off_base = offset_to_offbase((curr - match_index) as u32);
                    if ip + current_ml == iend {
                        break; // best possible, avoids read overflow on next attempt
                    }
                }
            }
        }

        // The dictionary's row.
        let head_grouped = ((dms_tag_row[0] as u32) & row_mask) * group_width;
        let mut num_matches = 0usize;
        let mut matches = self
            .row
            .mask
            .match_mask::<ROW_LOG>(dms_tag_row, dms_tag, head_grouped);
        while matches > 0 && nb_attempts > 0 {
            let match_pos = ((head_grouped + matches.trailing_zeros()) / group_width) & row_mask;
            matches &= matches - 1;
            let match_index = dms_row[match_pos as usize] as usize;
            if match_pos == 0 {
                continue;
            }
            // C tests `matchIndex < dmsLowestIndex` only.
            if !candidate_valid(match_index, WINDOW_START_INDEX, d.read_limit()) {
                break;
            }
            // SAFETY: as above.
            unsafe { *self.row.match_buffer.get_unchecked_mut(num_matches) = match_index as u32 };
            num_matches += 1;
            nb_attempts -= 1;
        }
        for &match_index in &self.row.match_buffer[..num_matches] {
            let m = match_index as usize + d.delta;
            let mut current_ml = 0;
            // SAFETY: `m + 4 <= d.dict.end() < ip + 4 <= iend`.
            unsafe {
                if read32(d.dict, m) == read32(src, ip) {
                    current_ml =
                        count_2segments(Fallback::new(), src, ip + 4, iend, d.dict, m + 4) + 4;
                }
            }
            if current_ml > ml {
                ml = current_ml;
                *off_base = offset_to_offbase((curr - m) as u32);
                if ip + current_ml == iend {
                    break;
                }
            }
        }
        ml
    }
}

/// `ZSTD_BtFindBestMatch` (`ZSTD_dictMatchState`).
struct BtDmsSearch<'a, M, const MLS: u32> {
    bt: BtSearch<M, MLS>,
    d: Dms<'a>,
}

impl<M: MatchCount, const MLS: u32> BtDmsSearch<'_, M, MLS> {
    /// `ZSTD_DUBT_findBestMatch` (`ZSTD_dictMatchState`): the frame's tree
    /// as without a dictionary, then, unless a match reached `iend` or the
    /// compares ran out, [`BtDmsSearch::find_better_dict_match`].
    ///
    /// # Safety
    /// As `BtSearch::find_best_match`.
    #[inline(always)]
    unsafe fn find_best_match(
        &self,
        ms: &mut MatchState,
        src: Src,
        ip: usize,
        iend: usize,
        off_base: &mut u32,
    ) -> usize {
        let cp = ms.cparams;
        let bt_search = self.bt;
        let p = bt_search.p;
        let bt_mask = p.bt_mask;
        let curr = ip;
        let window_low = p.lowest_match_index(curr);
        let bt_low = curr.saturating_sub(bt_mask);
        let unsort_low = (bt_low + 1).max(window_low);
        let mut nb_compares = 1u32 << cp.search_log;
        let mut nb_candidates = nb_compares;
        let mut previous_candidate = 0usize;

        let (hash_table, bt, _) = ms.ws.tables_mut();
        let h = hash_salted::<MLS>(src, ip, cp.hash_log, 0) as usize;
        let mut match_index = tget(hash_table, h);

        // reach end of unsorted candidates list
        let mut node = 2 * (match_index & bt_mask);
        while candidate_valid(match_index, unsort_low, curr)
            && tget(bt, node + 1) == DUBT_UNSORTED_MARK
            && nb_candidates > 1
        {
            tset(bt, node + 1, previous_candidate);
            previous_candidate = match_index;
            match_index = tget(bt, node);
            node = 2 * (match_index & bt_mask);
            nb_candidates -= 1;
        }

        // nullify last candidate if it's still unsorted
        if candidate_valid(match_index, unsort_low, curr)
            && tget(bt, node + 1) == DUBT_UNSORTED_MARK
        {
            tset(bt, node, 0);
            tset(bt, node + 1, 0);
        }

        // batch sort stacked candidates
        match_index = previous_candidate;
        while match_index != 0 {
            let next_candidate = tget(bt, 2 * (match_index & bt_mask) + 1);
            bt_search.insert_dubt1(bt, src, match_index, iend, nb_candidates, unsort_low);
            match_index = next_candidate;
            nb_candidates += 1;
        }

        // find longest match
        let mut common_smaller = 0usize;
        let mut common_larger = 0usize;
        let mut smaller_ptr = 2 * (curr & bt_mask);
        let mut larger_ptr = smaller_ptr + 1;
        let mut match_end_idx = curr + 8 + 1;
        let mut best_length = 0usize;

        match_index = tget(hash_table, h);
        tset(hash_table, h, curr);

        while nb_compares > 0 && candidate_valid(match_index, window_low, curr) {
            let next = 2 * (match_index & bt_mask);
            let mut match_length = common_smaller.min(common_larger);
            match_length +=
                bt_search
                    .count
                    .count(src, ip + match_length, match_index + match_length, iend);
            if match_length > best_length {
                if match_length > match_end_idx - match_index {
                    match_end_idx = match_index + match_length;
                }
                if (4 * (match_length - best_length)) as i32
                    > highbit32((curr - match_index + 1) as u32) as i32
                        - highbit32(*off_base) as i32
                {
                    best_length = match_length;
                    *off_base = offset_to_offbase((curr - match_index) as u32);
                }
                if ip + match_length == iend {
                    // equal: no way to know if inf or sup; skip checking in
                    // the dictionary too
                    nb_compares = 0;
                    break;
                }
            }
            if byte(src, match_index + match_length) < byte(src, ip + match_length) {
                // match is smaller than current
                tset(bt, smaller_ptr, match_index);
                common_smaller = match_length;
                if match_index <= bt_low {
                    smaller_ptr = BT_DUMMY;
                    break;
                }
                smaller_ptr = next + 1;
                match_index = tget(bt, next + 1);
            } else {
                // match is larger than current
                tset(bt, larger_ptr, match_index);
                common_larger = match_length;
                if match_index <= bt_low {
                    larger_ptr = BT_DUMMY;
                    break;
                }
                larger_ptr = next;
                match_index = tget(bt, next);
            }
            nb_compares -= 1;
        }
        if smaller_ptr != BT_DUMMY {
            tset(bt, smaller_ptr, 0);
        }
        if larger_ptr != BT_DUMMY {
            tset(bt, larger_ptr, 0);
        }

        if nb_compares > 0 {
            best_length =
                self.find_better_dict_match(src, ip, iend, off_base, best_length, nb_compares);
        }

        // skip repetitive patterns
        ms.next_to_update = match_end_idx - 8;
        best_length
    }

    /// `ZSTD_DUBT_findBetterDictMatch`: descend the dictionary's sorted
    /// tree with the compares left, for a match better than `best_length`
    /// by the tree's cost rule (which here counts the current offset base
    /// plus one).
    ///
    /// # Safety
    /// `ip + 8 <= iend <= src.end()`; the dictionary's tables have the
    /// sizes asserted per block.
    #[inline(always)]
    unsafe fn find_better_dict_match(
        &self,
        src: Src,
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        mut best_length: usize,
        mut nb_compares: u32,
    ) -> usize {
        let d = self.d;
        let curr = ip;
        let mut dict_match_index =
            tget(d.hash, hash_salted::<MLS>(src, ip, d.hash_log, 0) as usize);
        let dict_high_limit = d.end;
        let dict_low_limit = WINDOW_START_INDEX;
        let bt_mask = (1usize << (d.chain_log - 1)) - 1;
        let bt_low = if bt_mask >= dict_high_limit - dict_low_limit {
            dict_low_limit
        } else {
            dict_high_limit - bt_mask
        };
        let mut common_smaller = 0usize;
        let mut common_larger = 0usize;

        // C tests `dictMatchIndex > dictLowLimit` only; a dictionary's tree
        // holds no index past its content.
        while nb_compares > 0 && candidate_valid(dict_match_index, dict_low_limit + 1, d.end) {
            let next = 2 * (dict_match_index & bt_mask);
            let mut match_length = common_smaller.min(common_larger);
            let m = dict_match_index + d.delta;
            match_length += count_2segments(
                Fallback::new(),
                src,
                ip + match_length,
                iend,
                d.dict,
                m + match_length,
            );

            if match_length > best_length {
                if (4 * (match_length - best_length)) as i32
                    > highbit32((curr - m + 1) as u32) as i32 - highbit32(*off_base + 1) as i32
                {
                    best_length = match_length;
                    *off_base = offset_to_offbase((curr - m) as u32);
                }
                if ip + match_length == iend {
                    // reached end of input: ip[matchLength] is not valid
                    break;
                }
            }
            // `ip + match_length < iend`: the frame's tree ended the search
            // at a match reaching `iend`, so `best_length < iend - ip` and a
            // match reaching it broke out above. The byte after the common
            // part is in the content, or past its end in the input.
            let next_byte = if m + match_length >= d.dict.end() {
                byte(src, m + match_length)
            } else {
                byte(d.dict, m + match_length)
            };
            if next_byte < byte(src, ip + match_length) {
                if dict_match_index <= bt_low {
                    break; // beyond tree size, stop the search
                }
                common_smaller = match_length;
                dict_match_index = tget(d.chain, next + 1);
            } else {
                // match is larger than current
                if dict_match_index <= bt_low {
                    break;
                }
                common_larger = match_length;
                dict_match_index = tget(d.chain, next);
            }
            nb_compares -= 1;
        }
        best_length
    }
}

impl<M: MatchCount, const MLS: u32> Search for BtDmsSearch<'_, M, MLS> {
    const ILIMIT_MARGIN: usize = 8;

    #[inline(always)]
    unsafe fn count(&self, src: Src, a: usize, b: usize, limit: usize) -> usize {
        self.bt.count.count(src, a, b, limit)
    }

    #[inline(always)]
    fn refill(&mut self, _ms: &mut MatchState, _src: Src, _ilimit: usize) {}

    #[inline(always)]
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: Src,
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        _lazy_skipping: bool,
    ) -> usize {
        if ip < ms.next_to_update {
            return 0; // skipped area
        }
        // SAFETY: `ip < ilimit` with `ilimit + 8 <= iend <= src.end()`;
        // table sizes asserted per block.
        unsafe {
            BtSearch::<M, MLS>::update_dubt(ms, src, ip);
            self.find_best_match(ms, src, ip, iend, off_base)
        }
    }
}

/// `ZSTD_compressBlock_lazy_generic(..., ZSTD_dictMatchState)`: the block
/// loop of `super::lazy_generic` with the dictionary's rules. A repcode is
/// checked wherever it is in reach, in the content or the input, never
/// across the input's start ([`dms_rep_source`]), and counted across it
/// ([`count_dms`]); a catch-up stops at the start of the match's segment.
/// Returns the anchor of the trailing literals.
#[inline(always)]
fn lazy_dms_generic<S: Search>(
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    mut search: S,
    dict: Src,
) -> usize {
    let mc = Fallback::new();
    let depth = depth_of(ms.cparams.strategy);
    let istart = block.start;
    let iend = block.end;
    let mut anchor = istart;
    let ilimit = iend.saturating_sub(S::ILIMIT_MARGIN);
    let prefix_lowest = ms.window().dict_limit();
    assert!(src.lo() <= prefix_lowest && prefix_lowest <= istart && dict.end() == prefix_lowest);

    let mut offset_1 = rep[0] as usize;
    let mut offset_2 = rep[1] as usize;

    // C: `ip += (dictAndPrefixLength == 0)`
    let mut ip = istart;
    ip += (ip == dict.lo()) as usize;

    let mut lazy_skipping = false;
    search.refill(ms, src, ilimit);

    // SAFETY, for every unchecked read below: `ip <= ilimit` implies `ip +
    // ILIMIT_MARGIN <= iend <= src.end()` with `ILIMIT_MARGIN >= 8`, and a
    // repcode is read only where `dms_rep_source` admits it.
    unsafe {
        while ip < ilimit {
            let mut match_length = 0usize;
            let mut off_base = REPCODE1_TO_OFFBASE;
            let mut start = ip + 1;

            // check repCode
            let mut rep_at_depth0 = false;
            let rep_index = (ip + 1).wrapping_sub(offset_1);
            if let Some(rep_src) = dms_rep_source(src, dict, rep_index, ip + 1) {
                if read32(rep_src, rep_index) == read32(src, ip + 1) {
                    match_length = count_dms(mc, src, ip + 1 + 4, iend, dict, rep_index + 4) + 4;
                    if depth == 0 {
                        rep_at_depth0 = true; // goto _storeSequence
                    }
                }
            }

            if !rep_at_depth0 {
                // first search (depth 0)
                {
                    let mut offbase_found = 999_999_999u32;
                    let ml2 =
                        search.search_max(ms, src, ip, iend, &mut offbase_found, lazy_skipping);
                    if ml2 > match_length {
                        match_length = ml2;
                        start = ip;
                        off_base = offbase_found;
                    }
                }

                if match_length < 4 {
                    // jump faster over incompressible sections
                    let step = ((ip - anchor) >> K_SEARCH_STRENGTH) + 1;
                    ip += step;
                    lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                    continue;
                }

                // let's try to find a better solution
                if depth >= 1 {
                    while ip < ilimit {
                        ip += 1;
                        let rep_index = ip.wrapping_sub(offset_1);
                        if let Some(rep_src) = dms_rep_source(src, dict, rep_index, ip) {
                            if read32(rep_src, rep_index) == read32(src, ip) {
                                let ml_rep =
                                    count_dms(mc, src, ip + 4, iend, dict, rep_index + 4) + 4;
                                let gain2 = (ml_rep * 3) as i32;
                                let gain1 =
                                    (match_length * 3) as i32 - highbit32(off_base) as i32 + 1;
                                if ml_rep >= 4 && gain2 > gain1 {
                                    match_length = ml_rep;
                                    off_base = REPCODE1_TO_OFFBASE;
                                    start = ip;
                                }
                            }
                        }
                        {
                            let mut ofb_candidate = 999_999_999u32;
                            let ml2 = search.search_max(
                                ms,
                                src,
                                ip,
                                iend,
                                &mut ofb_candidate,
                                lazy_skipping,
                            );
                            let gain2 = (ml2 * 4) as i32 - highbit32(ofb_candidate) as i32;
                            let gain1 = (match_length * 4) as i32 - highbit32(off_base) as i32 + 4;
                            if ml2 >= 4 && gain2 > gain1 {
                                match_length = ml2;
                                off_base = ofb_candidate;
                                start = ip;
                                continue; // search a better one
                            }
                        }

                        // let's find an even better one
                        if depth == 2 && ip < ilimit {
                            ip += 1;
                            let rep_index = ip.wrapping_sub(offset_1);
                            if let Some(rep_src) = dms_rep_source(src, dict, rep_index, ip) {
                                if read32(rep_src, rep_index) == read32(src, ip) {
                                    let ml_rep =
                                        count_dms(mc, src, ip + 4, iend, dict, rep_index + 4) + 4;
                                    let gain2 = (ml_rep * 4) as i32;
                                    let gain1 =
                                        (match_length * 4) as i32 - highbit32(off_base) as i32 + 1;
                                    if ml_rep >= 4 && gain2 > gain1 {
                                        match_length = ml_rep;
                                        off_base = REPCODE1_TO_OFFBASE;
                                        start = ip;
                                    }
                                }
                            }
                            {
                                let mut ofb_candidate = 999_999_999u32;
                                let ml2 = search.search_max(
                                    ms,
                                    src,
                                    ip,
                                    iend,
                                    &mut ofb_candidate,
                                    lazy_skipping,
                                );
                                let gain2 = (ml2 * 4) as i32 - highbit32(ofb_candidate) as i32;
                                let gain1 =
                                    (match_length * 4) as i32 - highbit32(off_base) as i32 + 7;
                                if ml2 >= 4 && gain2 > gain1 {
                                    match_length = ml2;
                                    off_base = ofb_candidate;
                                    start = ip;
                                    continue;
                                }
                            }
                        }
                        break; // nothing found : store previous solution
                    }
                }

                // catch up
                if offbase_is_offset(off_base) {
                    let offset = offbase_to_offset(off_base) as usize;
                    let mut m = start - offset;
                    let (m_src, m_start) = if m < prefix_lowest {
                        (dict, dict.lo())
                    } else {
                        (src, prefix_lowest)
                    };
                    while start > anchor
                        && m > m_start
                        && byte(src, start - 1) == byte(m_src, m - 1)
                    {
                        start -= 1;
                        m -= 1;
                        match_length += 1;
                    }
                    offset_2 = offset_1;
                    offset_1 = offset;
                }
            }

            // store sequence
            out.store_seq(src, anchor, start - anchor, iend, off_base, match_length);
            ip = start + match_length;
            anchor = ip;

            if lazy_skipping {
                // We've found a match, disable lazy skipping mode, and
                // refill the hash cache.
                search.refill(ms, src, ilimit);
                lazy_skipping = false;
            }

            // check immediate repcode
            while ip <= ilimit {
                let rep_index = ip.wrapping_sub(offset_2);
                let Some(rep_src) = dms_rep_source(src, dict, rep_index, ip) else {
                    break;
                };
                if read32(rep_src, rep_index) != read32(src, ip) {
                    break;
                }
                let match_length = count_dms(mc, src, ip + 4, iend, dict, rep_index + 4) + 4;
                std::mem::swap(&mut offset_1, &mut offset_2); // swap offset_2 <=> offset_1
                out.store_seq(src, anchor, 0, iend, REPCODE1_TO_OFFBASE, match_length);
                ip += match_length;
                anchor = ip;
            }
        }
    }

    // save reps for next block
    rep[0] = offset_1 as u32;
    rep[1] = offset_2 as u32;
    anchor
}

/// `ZSTD_compressBlock_greedy/lazy/lazy2[_row]/btlazy2_dictMatchState`:
/// [`super::compress_block`] with the dictionary `dms` attached, whose
/// tables are for the same finder (`useRowMatchFinder` from the
/// dictionary) with the same row log. The scalar kernels serve every SIMD
/// level.
pub fn compress_block_dms(
    ms: &mut MatchState,
    src: Src,
    block: Block,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    dms: DictMatchState,
) -> usize {
    let block = block.range();
    assert_block_bounds(ms, src, block.end);
    assert_eq!(
        ms.search_method, dms.ms.search_method,
        "tables of another finder"
    );
    let d = Dms::new(dms, ms.window().dict_limit());
    let dcp = &dms.ms.cparams;
    assert_eq!(d.hash.len(), 1usize << dcp.hash_log);
    match ms.search_method {
        SearchMethod::HashChain | SearchMethod::BinaryTree => {
            assert_eq!(d.chain.len(), 1usize << dcp.chain_log);
        }
        SearchMethod::RowHash => {
            assert_eq!(d.tag.len(), 1usize << dcp.hash_log);
            assert_eq!(row_log_of(dcp), row_log_of(&ms.cparams));
            assert!(dcp.hash_log >= row_log_of(dcp));
        }
    }
    hc_row_bt_dms(ms, src, block, rep, out, d)
}

/// The block loop of [`compress_block_dms`] for `ms.search_method`,
/// specialised on `mls` (and the row log) as without a dictionary.
#[inline(never)]
fn hc_row_bt_dms(
    ms: &mut MatchState,
    src: Src,
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    d: Dms,
) -> usize {
    let dict = d.dict;
    macro_rules! go {
        ($search:expr) => {
            lazy_dms_generic(ms, src, block, rep, out, $search, dict)
        };
    }
    macro_rules! row {
        ($mls:literal, $row_log:literal) => {
            go!(RowDmsSearch::<Fallback, $mls, $row_log> {
                row: RowSearch::new(Fallback::new()),
                d,
            })
        };
    }
    match ms.search_method {
        SearchMethod::HashChain => match mls_of(&ms.cparams) {
            4 => go!(HcDmsSearch::<4> { d }),
            5 => go!(HcDmsSearch::<5> { d }),
            _ => go!(HcDmsSearch::<6> { d }),
        },
        SearchMethod::BinaryTree => {
            let p = BtParams::of(ms);
            let count = Fallback::new();
            match mls_of(&ms.cparams) {
                4 => go!(BtDmsSearch::<_, 4> {
                    bt: BtSearch { count, p },
                    d
                }),
                5 => go!(BtDmsSearch::<_, 5> {
                    bt: BtSearch { count, p },
                    d
                }),
                _ => go!(BtDmsSearch::<_, 6> {
                    bt: BtSearch { count, p },
                    d
                }),
            }
        }
        SearchMethod::RowHash => match (mls_of(&ms.cparams), row_log_of(&ms.cparams)) {
            (4, 4) => row!(4, 4),
            (4, 5) => row!(4, 5),
            (4, _) => row!(4, 6),
            (5, 4) => row!(5, 4),
            (5, 5) => row!(5, 5),
            (5, _) => row!(5, 6),
            (_, 4) => row!(6, 4),
            (_, 5) => row!(6, 5),
            _ => row!(6, 6),
        },
    }
}
