//! Greedy / lazy / lazy2 block compressors: port of `zstd_lazy.c`
//! (`ZSTD_compressBlock_lazy_generic`, no-dictionary mode) with both match
//! finders: the hash chain (`ZSTD_HcFindBestMatch`) and the row-based finder
//! (`ZSTD_RowFindBestMatch`, tag table + SIMD tag compare). The method is
//! chosen like `ZSTD_resolveRowMatchFinderMode(ZSTD_ps_auto)`: rows whenever
//! the strategy supports them and `window_log > 14`.
//!
//! Positions follow the [`MatchState`](super::matchstate) convention: absolute
//! indices into `src`, `window_low >= 1`, table entry `0` means empty. libzstd
//! indexes the same data from `ZSTD_WINDOW_START_INDEX = 1`, so a Rust
//! position `p` corresponds to the C index `p + 1`; the only observable
//! difference is that the first byte of a stream can never be the start of a
//! match here (its position would be `0`), and that the block loop therefore
//! begins one byte later than C on block 0 (`ip = max(istart, window_low)`
//! followed by C's own `ip += (dictAndPrefixLength == 0)` skip).

use super::common::{
    byte, candidate_valid, count, read32, read64, tget, tset, MatchCount, HASH_READ_SIZE,
};
use super::matchstate::MatchState;
use super::params::{CParams, Strategy};
use super::seqstore::{
    offbase_is_offset, offbase_to_offset, offset_to_offbase, SeqStore, REPCODE1_TO_OFFBASE,
};
#[cfg(target_arch = "aarch64")]
use fearless_simd::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::{Avx2, Sse4_2};
use fearless_simd::{Fallback, Level};
use std::ops::Range;
use std::sync::OnceLock;

/// `ZSTD_ROW_HASH_TAG_BITS`: low hash bits kept in the tag table.
const ROW_HASH_TAG_BITS: u32 = 8;
const ROW_HASH_TAG_MASK: u32 = (1 << ROW_HASH_TAG_BITS) - 1;
/// `ZSTD_ROW_HASH_CACHE_SIZE`: positions hashed ahead of the update pointer.
const ROW_HASH_CACHE_SIZE: usize = 8;
const ROW_HASH_CACHE_MASK: usize = ROW_HASH_CACHE_SIZE - 1;
/// `ZSTD_ROW_HASH_MAX_ENTRIES`: entries of the widest row (`rowLog == 6`).
const ROW_HASH_MAX_ENTRIES: usize = 64;
/// `kSearchStrength`: `step = ((ip - anchor) >> kSearchStrength) + 1` when no
/// match is found.
const K_SEARCH_STRENGTH: usize = 8;
/// `kLazySkippingStep`: skipping more than this many bytes at once enters the
/// lazy-skipping mode (only searched positions are inserted).
const K_LAZY_SKIPPING_STEP: usize = 8;

// ---------------------------------------------------------------------------
// Small helpers. Unchecked reads follow `compress/common.rs`: every call site
// states the bound that makes it sound. Inside a block those bounds rest on
// `block.end <= src.len()` and the table sizes, both asserted once per block
// in [`compress_block_with`] / [`load_prefix_with`], plus the loop limits
// (`ip < ilimit`, `ilimit + ILIMIT_MARGIN <= iend`, `ILIMIT_MARGIN >= 8`).
// ---------------------------------------------------------------------------

/// `ZSTD_highbit32`: index of the highest set bit (`v != 0`).
#[inline(always)]
fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;

/// `ZSTD_hashPtrSalted(src + pos, hbits, mls, salt)` for `mls` 4..=6 and
/// `hbits <= 32` (`ZSTD_hashPtr` is the same with `salt == 0`). `mls == 4`
/// only uses the low 32 bits of the salt, like C. The result is
/// `< 1 << hbits`.
///
/// # Safety
/// `pos + HASH_READ_SIZE <= src.len()` (`MLS >= 5` reads 8 bytes).
#[inline(always)]
unsafe fn hash_salted<const MLS: u32>(src: &[u8], pos: usize, hbits: u32, salt: u64) -> u32 {
    debug_assert!((1..=32).contains(&hbits));
    debug_assert!(pos + HASH_READ_SIZE <= src.len());
    match MLS {
        4 => (read32(src, pos).wrapping_mul(PRIME4) ^ (salt as u32)) >> (32 - hbits),
        5 => (((read64(src, pos) << 24).wrapping_mul(PRIME5) ^ salt) >> (64 - hbits)) as u32,
        6 => (((read64(src, pos) << 16).wrapping_mul(PRIME6) ^ salt) >> (64 - hbits)) as u32,
        _ => unreachable!("mls is clamped to 4..=6"),
    }
}

/// `ZSTD_bitmix` (zstd_compress.c).
const fn bitmix(mut val: u64, len: u64) -> u64 {
    val ^= val.rotate_right(49) ^ val.rotate_right(24);
    val = val.wrapping_mul(0x9FB21C651E98DF25);
    val ^= (val >> 35).wrapping_add(len);
    val = val.wrapping_mul(0x9FB21C651E98DF25);
    val ^ (val >> 28)
}

/// The `hashSalt` of a freshly created `ZSTD_CCtx`: `ZSTD_advanceHashSalt`
/// applied to `hashSalt == 0` and `hashSaltEntropy == 0`.
pub(super) const fn initial_hash_salt() -> u64 {
    bitmix(0, 8) ^ bitmix(0, 4)
}

/// `PREFETCH_L1(&slice[idx])`. `idx` may point past the end: C prefetches
/// `base + matchIndex` and table rows the same way, and the hint never
/// faults.
#[inline(always)]
fn prefetch_l1<T>(slice: &[T], idx: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "sse")]
        #[inline]
        fn prefetch(p: *const i8) {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(p)
        }
        // SAFETY: SSE is part of the x86_64 baseline, so the target feature
        // the callee asks for is always present.
        unsafe { prefetch(slice.as_ptr().wrapping_add(idx) as *const i8) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (slice, idx);
    }
}

/// `BOUNDED(4, minMatch, 6)`.
#[inline]
fn mls_of(cp: &CParams) -> u32 {
    cp.min_match.clamp(4, 6)
}

/// `BOUNDED(4, searchLog, 6)`: log2 of the entries per tag-table row.
#[inline]
fn row_log_of(cp: &CParams) -> u32 {
    cp.search_log.clamp(4, 6)
}

/// Lazy depth of a strategy: greedy 0, lazy 1, lazy2 and btlazy2 2.
#[inline]
fn depth_of(strategy: Strategy) -> u32 {
    match strategy {
        Strategy::Greedy => 0,
        Strategy::Lazy => 1,
        Strategy::Lazy2 | Strategy::BtLazy2 => 2,
        Strategy::Fast
        | Strategy::DFast
        | Strategy::BtOpt
        | Strategy::BtUltra
        | Strategy::BtUltra2 => unreachable!("not a lazy strategy"),
    }
}

// ---------------------------------------------------------------------------
// Match finders (`ZSTD_searchMax` back ends).
// ---------------------------------------------------------------------------

/// One `searchMethod_e` of `ZSTD_compressBlock_lazy_generic`.
trait Search {
    /// `iend - ilimit`: 8 for the hash chain, `8 + ZSTD_ROW_HASH_CACHE_SIZE`
    /// for the row finder.
    const ILIMIT_MARGIN: usize;

    /// `ZSTD_count` for this finder's SIMD level.
    ///
    /// # Safety
    /// As [`count`].
    #[inline(always)]
    unsafe fn count(&self, src: &[u8], a: usize, b: usize, limit: usize) -> usize {
        count(src, a, b, limit)
    }

    /// Work done at block start and whenever lazy skipping ends
    /// (`ZSTD_row_fillHashCache` for the row finder; nothing for chains).
    fn refill(&mut self, ms: &mut MatchState, src: &[u8], ilimit: usize);

    /// `ZSTD_searchMax`: longest match at `ip` (limit `iend`), at least 4 to
    /// count; `off_base` receives its `OFFSET_TO_OFFBASE` when one is found.
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        lazy_skipping: bool,
    ) -> usize;
}

/// `search_hashChain`: `ZSTD_HcFindBestMatch` over the hash table heads and
/// the chain table links.
struct HcSearch<const MLS: u32>;

impl<const MLS: u32> HcSearch<MLS> {
    /// `ZSTD_insertAndFindFirstIndex_internal`: insert `[next_to_update, ip)`
    /// (only one position while lazy skipping) and return the chain head of
    /// `ip`'s hash.
    ///
    /// # Safety
    /// `ip + HASH_READ_SIZE <= src.len()`; `hash_table` and `chain_table`
    /// hold `1 << hash_log` and `1 << chain_log` entries.
    #[inline(always)]
    unsafe fn insert_and_find_first_index(
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        lazy_skipping: bool,
    ) -> u32 {
        let hash_log = ms.cparams.hash_log;
        let chain_mask = (1usize << ms.cparams.chain_log) - 1;
        let (hash_table, chain_table, _) = ms.ws.tables_mut();
        let target = ip;
        let mut idx = ms.next_to_update;
        // Every hashed position is `<= ip`; `h < 1 << hash_log` and
        // `idx & chain_mask < 1 << chain_log`.
        while idx < target {
            let h = hash_salted::<MLS>(src, idx, hash_log, 0) as usize;
            tset(chain_table, idx & chain_mask, tget(hash_table, h));
            tset(hash_table, h, idx);
            idx += 1;
            if lazy_skipping {
                break;
            }
        }
        ms.next_to_update = target;
        tget(
            hash_table,
            hash_salted::<MLS>(src, ip, hash_log, 0) as usize,
        ) as u32
    }
}

impl<const MLS: u32> Search for HcSearch<MLS> {
    const ILIMIT_MARGIN: usize = 8;

    #[inline(always)]
    fn refill(&mut self, _ms: &mut MatchState, _src: &[u8], _ilimit: usize) {}

    /// `ZSTD_HcFindBestMatch` (`ZSTD_noDict`).
    #[inline(always)]
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        lazy_skipping: bool,
    ) -> usize {
        let chain_size = 1usize << ms.cparams.chain_log;
        let chain_mask = chain_size - 1;
        let curr = ip;
        let low_limit = ms.lowest_prefix_index(curr);
        let min_chain = curr.saturating_sub(chain_size);
        let mut nb_attempts = 1u32 << ms.cparams.search_log;
        let mut ml = 4 - 1;

        // SAFETY: `ip < ilimit` with `ilimit + ILIMIT_MARGIN <= iend <=
        // src.len()`, `ILIMIT_MARGIN == HASH_READ_SIZE`; table sizes
        // asserted per block ([`assert_block_bounds`]).
        let mut match_index =
            unsafe { Self::insert_and_find_first_index(ms, src, ip, lazy_skipping) } as usize;
        let (_, chain_table, _) = ms.ws.tables();
        // C only tests `matchIndex >= lowLimit`; the upper bound is folded
        // into the same compare so that a stale table entry is a miss, not
        // an out-of-bounds read.
        while candidate_valid(match_index, low_limit, curr) && nb_attempts > 0 {
            // SAFETY: `match_index < ip` and `ip + ml < iend <= src.len()`
            // (`ml` starts at 3 with `ip + 8 <= iend`, and a match reaching
            // `iend` ends the loop), so the reads at `+ ml - 3` and the
            // count stay inside `src`.
            let better = unsafe { read32(src, match_index + ml - 3) == read32(src, ip + ml - 3) };
            if better {
                // SAFETY: `match_index < ip <= iend <= src.len()`.
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
            // SAFETY: `match_index & chain_mask < 1 << chain_log ==
            // chain_table.len()`.
            match_index = unsafe { tget(chain_table, match_index & chain_mask) };
            nb_attempts -= 1;
        }
        ml
    }
}

// ---------------------------------------------------------------------------
// Binary-tree match finder (`ZSTD_BtFindBestMatch`, btlazy2).
// ---------------------------------------------------------------------------

/// `ZSTD_DUBT_UNSORTED_MARK`: the second slot of a node that was inserted
/// by [`BtSearch::update_dubt`] but not yet sorted into the tree. No real
/// candidate is ever `1`: candidates are `> window_low >= 1`.
const DUBT_UNSORTED_MARK: usize = 1;

/// `search_binaryTree`: a "delayed update binary tree" in the chain table,
/// two entries per position (`bt[2 * (idx & btMask)]` the smaller child,
/// `+ 1` the larger one; `btLog = chainLog - 1`). Positions are appended
/// unsorted as a hash chain and sorted lazily when a search reaches them.
#[derive(Clone, Copy)]
struct BtSearch<M, const MLS: u32> {
    /// The [`MatchCount`] level of the block loop.
    count: M,
    /// The block's tree constants.
    p: BtParams,
}

/// C's `dummy32`: the slot a descent writes once it ran past `btLow`.
const BT_DUMMY: usize = usize::MAX;

/// The per-block constants of the tree functions.
#[derive(Clone, Copy)]
struct BtParams {
    /// `btMask = (1 << (chainLog - 1)) - 1`.
    bt_mask: usize,
    /// `window.lowLimit` (`ms.window_low`).
    window_valid: usize,
    /// `1 << windowLog`.
    max_distance: usize,
}

impl BtParams {
    fn of(ms: &MatchState) -> Self {
        BtParams {
            bt_mask: (1usize << (ms.cparams.chain_log - 1)) - 1,
            window_valid: ms.window_low,
            max_distance: 1usize << ms.cparams.window_log,
        }
    }

    /// `curr - windowValid > maxDistance ? curr - maxDistance :
    /// windowValid`, i.e. [`MatchState::lowest_prefix_index`].
    #[inline(always)]
    fn window_low(self, curr: usize) -> usize {
        if curr > self.window_valid + self.max_distance {
            curr - self.max_distance
        } else {
            self.window_valid
        }
    }
}

impl<M: MatchCount, const MLS: u32> BtSearch<M, MLS> {
    /// `ZSTD_updateDUBT`: append `[next_to_update, ip)` to their hash
    /// chains, each marked unsorted.
    ///
    /// # Safety
    /// `ip + HASH_READ_SIZE <= src.len()`; the tables have the sizes of
    /// [`assert_block_bounds`].
    #[inline(always)]
    unsafe fn update_dubt(ms: &mut MatchState, src: &[u8], ip: usize) {
        let hash_log = ms.cparams.hash_log;
        let bt_mask = BtParams::of(ms).bt_mask;
        let (hash_table, bt, _) = ms.ws.tables_mut();
        // `h < 1 << hash_log` and `2 * (idx & bt_mask) + 1 < 1 << chain_log`;
        // every hashed position is `< ip`.
        for idx in ms.next_to_update..ip {
            let h = hash_salted::<MLS>(src, idx, hash_log, 0) as usize;
            let match_index = tget(hash_table, h);
            tset(hash_table, h, idx);
            let node = 2 * (idx & bt_mask);
            tset(bt, node, match_index);
            tset(bt, node + 1, DUBT_UNSORTED_MARK);
        }
        ms.next_to_update = ip;
    }

    /// `ZSTD_insertDUBT1` (`ZSTD_noDict`): sort the unsorted node `curr`
    /// into the tree, comparing at most `nb_compares` candidates and not
    /// descending past `bt_low`.
    ///
    /// # Safety
    /// `curr < iend <= src.len()`; `bt.len() == 2 * (p.bt_mask + 1)`.
    #[inline(always)]
    unsafe fn insert_dubt1(
        self,
        bt: &mut [u32],
        src: &[u8],
        curr: usize,
        iend: usize,
        mut nb_compares: u32,
        bt_low: usize,
    ) {
        let p = self.p;
        let bt_mask = p.bt_mask;
        let ip = curr;
        let mut common_smaller = 0usize;
        let mut common_larger = 0usize;
        let mut smaller_ptr = 2 * (curr & bt_mask);
        let mut larger_ptr = smaller_ptr + 1;
        // The node is unsorted: its first slot links to the next candidate,
        // its second (the unsorted mark's chain) is already saved.
        let mut match_index = tget(bt, smaller_ptr);
        let window_low = p.window_low(curr);
        debug_assert!(window_low < curr);

        // C: `matchIndex > windowLow`; every tree link points below the node
        // holding it, so `match_index < curr` holds and the check only turns
        // a misused state into a miss.
        while nb_compares > 0 && candidate_valid(match_index, window_low + 1, curr) {
            let next = 2 * (match_index & bt_mask);
            let mut match_length = common_smaller.min(common_larger);
            // `match_index + match_length < ip + match_length < iend`: the
            // common lengths are `< iend - ip` (a match reaching `iend`
            // breaks below).
            match_length +=
                self.count
                    .count(src, ip + match_length, match_index + match_length, iend);
            if ip + match_length == iend {
                // equal: no way to know if inf or sup
                break;
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
    }

    /// `ZSTD_DUBT_findBestMatch` (`ZSTD_noDict`).
    ///
    /// # Safety
    /// `ip + HASH_READ_SIZE <= iend <= src.len()`; the tables have the
    /// sizes of [`assert_block_bounds`].
    #[inline(always)]
    unsafe fn find_best_match(
        self,
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        iend: usize,
        off_base: &mut u32,
    ) -> usize {
        let cp = ms.cparams;
        let p = self.p;
        let bt_mask = p.bt_mask;
        let curr = ip;
        let window_low = p.window_low(curr);
        debug_assert!(window_low < curr);
        let bt_low = curr.saturating_sub(bt_mask);
        let unsort_limit = bt_low.max(window_low);
        let mut nb_compares = 1u32 << cp.search_log;
        let mut nb_candidates = nb_compares;
        let mut previous_candidate = 0usize;

        let (hash_table, bt, _) = ms.ws.tables_mut();
        let h = hash_salted::<MLS>(src, ip, cp.hash_log, 0) as usize;
        let mut match_index = tget(hash_table, h);

        // reach end of unsorted candidates list; the chain of unsorted marks
        // becomes a reversed chain back to the head. C tests `matchIndex >
        // unsortLimit` only; hash and tree entries are always `< curr`, and
        // the upper bound keeps the positions `insert_dubt1` reads inside
        // `src` for a misused state too.
        let mut node = 2 * (match_index & bt_mask);
        while candidate_valid(match_index, unsort_limit + 1, curr)
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
        if candidate_valid(match_index, unsort_limit + 1, curr)
            && tget(bt, node + 1) == DUBT_UNSORTED_MARK
        {
            tset(bt, node, 0);
            tset(bt, node + 1, 0);
        }

        // batch sort stacked candidates
        match_index = previous_candidate;
        while match_index != 0 {
            let next_candidate = tget(bt, 2 * (match_index & bt_mask) + 1);
            // Every stacked candidate passed `candidate_valid(.., curr)`
            // above: `match_index < ip < iend`.
            self.insert_dubt1(bt, src, match_index, iend, nb_candidates, unsort_limit);
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

        // As in `insert_dubt1`: C tests `matchIndex > windowLow` only.
        while nb_compares > 0 && candidate_valid(match_index, window_low + 1, curr) {
            let next = 2 * (match_index & bt_mask);
            let mut match_length = common_smaller.min(common_larger);
            // `match_index + match_length < ip + match_length < iend`, as in
            // `insert_dubt1`.
            match_length +=
                self.count
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
                    // equal: no way to know if inf or sup
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

        // skip repetitive patterns
        ms.next_to_update = match_end_idx - 8;
        best_length
    }
}

impl<M: MatchCount, const MLS: u32> Search for BtSearch<M, MLS> {
    const ILIMIT_MARGIN: usize = 8;

    #[inline(always)]
    unsafe fn count(&self, src: &[u8], a: usize, b: usize, limit: usize) -> usize {
        self.count.count(src, a, b, limit)
    }

    #[inline(always)]
    fn refill(&mut self, _ms: &mut MatchState, _src: &[u8], _ilimit: usize) {}

    /// `ZSTD_BtFindBestMatch` (`ZSTD_noDict`); the tree ignores lazy
    /// skipping, as in C.
    #[inline(always)]
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        _lazy_skipping: bool,
    ) -> usize {
        if ip < ms.next_to_update {
            return 0; // skipped area
        }
        // SAFETY: `ip < ilimit` with `ilimit + 8 <= iend <= src.len()`;
        // table sizes asserted per block ([`assert_block_bounds`]).
        unsafe {
            Self::update_dubt(ms, src, ip);
            self.find_best_match(ms, src, ip, iend, off_base)
        }
    }
}

// ---------------------------------------------------------------------------
// Row-based match finder (`ZSTD_RowFindBestMatch`).
// ---------------------------------------------------------------------------

/// `ZSTD_row_getMatchMask` back end, one implementation per SIMD witness.
/// Bit `i` (bit group `i` for NEON) of the result is set iff
/// `row[(i + head) % entries] == tag`; bit 0 therefore names the head byte
/// itself, which the caller skips (`matchPos == 0`).
trait TagMask: Copy {
    /// `ZSTD_row_matchMaskGroupWidth(rowEntries)`.
    fn group_width(row_entries: u32) -> u32;

    /// `ZSTD_row_getMatchMask(tagRow, tag, headGrouped, rowEntries)` with
    /// `row.len() == 1 << ROW_LOG`.
    fn match_mask<const ROW_LOG: u32>(self, row: &[u8], tag: u8, head_grouped: u32) -> u64;
}

/// `ZSTD_rotateRight_U16/U32/U64` for the row width.
#[inline(always)]
fn rotate_mask<const ROW_LOG: u32>(matches: u64, head: u32) -> u64 {
    match ROW_LOG {
        4 => (matches as u16).rotate_right(head) as u64,
        5 => (matches as u32).rotate_right(head) as u64,
        _ => matches.rotate_right(head),
    }
}

/// The generic SWAR arm of `ZSTD_row_getMatchMask` (little-endian, 64-bit
/// chunks).
#[inline(always)]
fn swar_match_mask<const ROW_LOG: u32>(row: &[u8], tag: u8, head: u32) -> u64 {
    const CHUNK: usize = 8;
    const X01: u64 = u64::MAX / 0xFF;
    const X80: u64 = X01 << 7;
    // Multiplying a word whose only set bits are byte MSBs by this constant
    // gathers those MSBs into the top byte without carries.
    const EXTRACT_MAGIC: u64 = (u64::MAX / 0x7F) >> CHUNK;
    assert_eq!(row.len(), 1usize << ROW_LOG);
    let splat = (tag as u64).wrapping_mul(X01);
    let mut matches = 0u64;
    let mut i = (1usize << ROW_LOG) - CHUNK;
    loop {
        // SAFETY: `i + 8 <= row.len()` (asserted above, `i` steps down by 8
        // from `len - 8`).
        let mut chunk = unsafe { read64(row, i) } ^ splat;
        chunk = ((chunk | X80).wrapping_sub(X01) | chunk) & X80; // byte MSB set iff byte != tag
        matches <<= CHUNK;
        matches |= chunk.wrapping_mul(EXTRACT_MAGIC) >> (64 - CHUNK);
        if i == 0 {
            break;
        }
        i -= CHUNK;
    }
    rotate_mask::<ROW_LOG>(!matches, head)
}

impl TagMask for Fallback {
    #[inline(always)]
    fn group_width(_row_entries: u32) -> u32 {
        1
    }

    #[inline(always)]
    fn match_mask<const ROW_LOG: u32>(self, row: &[u8], tag: u8, head_grouped: u32) -> u64 {
        swar_match_mask::<ROW_LOG>(row, tag, head_grouped)
    }
}

/// The x86 tag-compare kernels. Each is a `#[target_feature]` function, so
/// calling one needs the matching fearless_simd witness as proof that the CPU
/// has the feature; the witnesses are only ever constructed after detection.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::{rotate_mask, TagMask};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use fearless_simd::{Avx2, Sse4_2};

    /// `ZSTD_row_getSSEMask(rowEntries / 16, src, tag, head)`: one
    /// `_mm_cmpeq_epi8` + `_mm_movemask_epi8` per 16 entries, chunk `i` at
    /// bits `16 * i ..`, then rotated right by `head`.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE2.
    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn sse_match_mask<const ROW_LOG: u32>(row: &[u8], tag: u8, head: u32) -> u64 {
        assert_eq!(row.len(), 1usize << ROW_LOG);
        let splat = _mm_set1_epi8(tag as i8);
        let mut matches = 0u64;
        for i in (0..row.len() / 16).rev() {
            // SAFETY: `16 * i + 16 <= row.len()` (asserted above); the load
            // is unaligned.
            let chunk = unsafe { _mm_loadu_si128(row.as_ptr().add(16 * i).cast::<__m128i>()) };
            let eq = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, splat)) as u32;
            matches = (matches << 16) | eq as u64;
        }
        rotate_mask::<ROW_LOG>(matches, head)
    }

    /// The same mask from 256-bit compares (`_mm256_cmpeq_epi8` +
    /// `_mm256_movemask_epi8` per 32 entries). libzstd has no AVX2 kernel;
    /// this is bit-identical to [`sse_match_mask`] because movemask bit `j`
    /// is entry `j` in both widths. 16-entry rows use the SSE kernel.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn avx2_match_mask<const ROW_LOG: u32>(row: &[u8], tag: u8, head: u32) -> u64 {
        if ROW_LOG == 4 {
            // SAFETY: AVX2 implies SSE2.
            return unsafe { sse_match_mask::<ROW_LOG>(row, tag, head) };
        }
        assert_eq!(row.len(), 1usize << ROW_LOG);
        let splat = _mm256_set1_epi8(tag as i8);
        let mut matches = 0u64;
        for i in (0..row.len() / 32).rev() {
            // SAFETY: `32 * i + 32 <= row.len()` (asserted above); the load
            // is unaligned.
            let chunk = unsafe { _mm256_loadu_si256(row.as_ptr().add(32 * i).cast::<__m256i>()) };
            let eq = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, splat)) as u32;
            matches = (matches << 32) | eq as u64;
        }
        rotate_mask::<ROW_LOG>(matches, head)
    }

    impl TagMask for Sse4_2 {
        #[inline(always)]
        fn group_width(_row_entries: u32) -> u32 {
            1
        }

        #[inline(always)]
        fn match_mask<const ROW_LOG: u32>(self, row: &[u8], tag: u8, head_grouped: u32) -> u64 {
            // SAFETY: `self` proves SSE4.2, a superset of SSE2.
            unsafe { sse_match_mask::<ROW_LOG>(row, tag, head_grouped) }
        }
    }

    impl TagMask for Avx2 {
        #[inline(always)]
        fn group_width(_row_entries: u32) -> u32 {
            1
        }

        #[inline(always)]
        fn match_mask<const ROW_LOG: u32>(self, row: &[u8], tag: u8, head_grouped: u32) -> u64 {
            // SAFETY: `self` proves AVX2.
            unsafe { avx2_match_mask::<ROW_LOG>(row, tag, head_grouped) }
        }
    }
}

/// The NEON tag-compare kernel (little endian only, like C; Rust has no
/// big-endian aarch64 target). Not compiled or run on the x86_64 machine
/// this was written on.
#[cfg(target_arch = "aarch64")]
mod arm {
    use super::TagMask;
    use core::arch::aarch64::*;
    use fearless_simd::Neon;

    /// `ZSTD_row_getNEONMask(rowEntries, src, tag, headGrouped)`. The mask
    /// has one 4-bit group per entry for 16-entry rows (match flag in the
    /// group's top bit, `& 0x88..`), 2-bit groups for 32 (flag in the bottom
    /// bit, `& 0x55..`) and one bit per entry for 64; the rotation is always
    /// the 64-bit one, by `headGrouped = head * groupWidth`.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn neon_match_mask<const ROW_LOG: u32>(row: &[u8], tag: u8, head_grouped: u32) -> u64 {
        assert_eq!(row.len(), 1usize << ROW_LOG);
        let src = row.as_ptr();
        let dup = vdupq_n_u8(tag);
        match ROW_LOG {
            4 => {
                // vshrn_n_u16 shifts by 4 every u16 and narrows to 8 lower
                // bits, so every nibble of the result is one entry's flag.
                // SAFETY: 16 bytes at `src` (asserted above).
                let chunk = unsafe { vld1q_u8(src) };
                let equal = vreinterpretq_u16_u8(vceqq_u8(chunk, dup));
                let res = vshrn_n_u16::<4>(equal);
                let matches = vget_lane_u64::<0>(vreinterpret_u64_u8(res));
                matches.rotate_right(head_grouped) & 0x8888_8888_8888_8888
            }
            5 => {
                // Same idea with de-interleaved even/odd bytes, then two bits
                // per entry.
                // SAFETY: 32 bytes at `src` (asserted above).
                let chunk = unsafe { vld2q_u16(src.cast::<u16>()) };
                let chunk0 = vreinterpretq_u8_u16(chunk.0);
                let chunk1 = vreinterpretq_u8_u16(chunk.1);
                let t0 = vshrn_n_u16::<6>(vreinterpretq_u16_u8(vceqq_u8(chunk0, dup)));
                let t1 = vshrn_n_u16::<6>(vreinterpretq_u16_u8(vceqq_u8(chunk1, dup)));
                let res = vsli_n_u8::<4>(t0, t1);
                let matches = vget_lane_u64::<0>(vreinterpret_u64_u8(res));
                matches.rotate_right(head_grouped) & 0x5555_5555_5555_5555
            }
            _ => {
                // SAFETY: 64 bytes at `src` (asserted above).
                let chunk = unsafe { vld4q_u8(src) };
                let cmp0 = vceqq_u8(chunk.0, dup);
                let cmp1 = vceqq_u8(chunk.1, dup);
                let cmp2 = vceqq_u8(chunk.2, dup);
                let cmp3 = vceqq_u8(chunk.3, dup);
                let t0 = vsriq_n_u8::<1>(cmp1, cmp0);
                let t1 = vsriq_n_u8::<1>(cmp3, cmp2);
                let t2 = vsriq_n_u8::<2>(t1, t0);
                let t3 = vsriq_n_u8::<4>(t2, t2);
                let t4 = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t3));
                let matches = vget_lane_u64::<0>(vreinterpret_u64_u8(t4));
                matches.rotate_right(head_grouped)
            }
        }
    }

    impl TagMask for Neon {
        /// `ZSTD_row_matchMaskGroupWidth` with `ZSTD_ARCH_ARM_NEON`.
        #[inline(always)]
        fn group_width(row_entries: u32) -> u32 {
            match row_entries {
                16 => 4,
                32 => 2,
                _ => 1,
            }
        }

        #[inline(always)]
        fn match_mask<const ROW_LOG: u32>(self, row: &[u8], tag: u8, head_grouped: u32) -> u64 {
            // SAFETY: `self` proves NEON.
            unsafe { neon_match_mask::<ROW_LOG>(row, tag, head_grouped) }
        }
    }
}

// The row finder's SSE4.2 and NEON levels count with the scalar
// `ZSTD_count`; common.rs has a wider count only for AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl MatchCount for Sse4_2 {
    #[inline(always)]
    unsafe fn count(self, src: &[u8], a: usize, b: usize, limit: usize) -> usize {
        count(src, a, b, limit)
    }
}

#[cfg(target_arch = "aarch64")]
impl MatchCount for Neon {
    #[inline(always)]
    unsafe fn count(self, src: &[u8], a: usize, b: usize, limit: usize) -> usize {
        count(src, a, b, limit)
    }
}

/// `search_rowHash`: rows of `1 << ROW_LOG` hash-table entries selected by
/// the high hash bits, each with a one-byte tag per entry and the row head
/// in the tag row's byte 0.
struct RowSearch<M: TagMask, const MLS: u32, const ROW_LOG: u32> {
    mask: M,
    /// `ms->hashCache`. C keeps it in the match state, but refills it in the
    /// block prologue (`ZSTD_row_fillHashCache`) and after lazy skipping, so
    /// it never carries information across blocks and lives here.
    hash_cache: [u32; ROW_HASH_CACHE_SIZE],
    /// `matchBuffer` of `ZSTD_RowFindBestMatch`: an uninitialised local in
    /// C. Kept here so that it is not zeroed again on every search; only
    /// `..num_matches` is ever read.
    match_buffer: [u32; ROW_HASH_MAX_ENTRIES],
}

/// The row finder's tables and the two scalars it hashes with
/// (`ZSTD_row_nextCachedHash`'s `hashTable`, `tagTable`, `hashLog`,
/// `hashSalt`), bound once per search so that the slices stay in registers
/// through the update loop instead of being re-derived from `ms.ws`.
struct RowTables<'a> {
    hash: &'a mut [u32],
    tag: &'a mut [u8],
    hash_log: u32,
    hash_salt: u64,
}

impl<'a> RowTables<'a> {
    /// Borrow `ms`'s tables; `ms` stays borrowed for as long as the result.
    #[inline(always)]
    fn of(ms: &'a mut MatchState) -> Self {
        let (hash_log, hash_salt) = (ms.cparams.hash_log, ms.hash_salt);
        let (hash, _, tag) = ms.ws.tables_mut();
        Self {
            hash,
            tag,
            hash_log,
            hash_salt,
        }
    }
}

impl<M: TagMask, const MLS: u32, const ROW_LOG: u32> RowSearch<M, MLS, ROW_LOG> {
    const ROW_ENTRIES: usize = 1 << ROW_LOG;
    const ROW_MASK: u32 = (1 << ROW_LOG) - 1;

    fn new(mask: M) -> Self {
        Self {
            mask,
            hash_cache: [0; ROW_HASH_CACHE_SIZE],
            match_buffer: [0; ROW_HASH_MAX_ENTRIES],
        }
    }

    /// `ZSTD_hashPtrSalted(p, rowHashLog + ZSTD_ROW_HASH_TAG_BITS, mls,
    /// hashSalt)` with `rowHashLog = hashLog - rowLog`. The row index
    /// `(hash >> 8) << ROW_LOG` is therefore `< 1 << hash_log`, the size of
    /// the tag and hash tables, and `+ ROW_ENTRIES` stays `<=` it.
    ///
    /// # Safety
    /// `pos + HASH_READ_SIZE <= src.len()`.
    #[inline(always)]
    unsafe fn hash(hash_log: u32, hash_salt: u64, src: &[u8], pos: usize) -> u32 {
        let hbits = hash_log - ROW_LOG + ROW_HASH_TAG_BITS;
        hash_salted::<MLS>(src, pos, hbits, hash_salt)
    }

    /// `&tag_table[rel_row]` (the head byte) for a row index of [`Self::hash`].
    ///
    /// # Safety
    /// `rel_row` came from [`Self::hash`] on a state whose tables are
    /// `1 << hash_log` long (asserted per block).
    #[inline(always)]
    unsafe fn head(tag_table: &mut [u8], rel_row: usize) -> &mut u8 {
        debug_assert!(rel_row + Self::ROW_ENTRIES <= tag_table.len());
        tag_table.get_unchecked_mut(rel_row)
    }

    /// `ZSTD_row_nextIndex`: cycle the head backwards through `1..entries`
    /// (position 0 holds the head itself) and return the new position.
    #[inline(always)]
    fn next_index(head: &mut u8) -> usize {
        let mut next = (*head as u32).wrapping_sub(1) & Self::ROW_MASK;
        if next == 0 {
            next += Self::ROW_MASK; // skip first position
        }
        *head = next as u8;
        next as usize
    }

    /// `ZSTD_row_prefetch`.
    #[inline(always)]
    fn prefetch_row(hash_table: &[u32], tag_table: &[u8], rel_row: usize) {
        prefetch_l1(hash_table, rel_row);
        if ROW_LOG >= 5 {
            prefetch_l1(hash_table, rel_row + 16);
        }
        prefetch_l1(tag_table, rel_row);
        if ROW_LOG == 6 {
            prefetch_l1(tag_table, rel_row + 32);
        }
    }

    /// `ZSTD_row_fillHashCache(ms, base, rowLog, mls, idx, iLimit)`: hash
    /// (and prefetch the rows of) `idx..idx+8`, not beyond `i_limit`.
    ///
    /// # Safety
    /// `i_limit + HASH_READ_SIZE <= src.len()`.
    #[inline(always)]
    unsafe fn fill_hash_cache(&mut self, t: &RowTables, src: &[u8], idx: usize, i_limit: usize) {
        let max_elems = if idx > i_limit { 0 } else { i_limit - idx + 1 };
        let lim = idx + ROW_HASH_CACHE_SIZE.min(max_elems);
        for i in idx..lim {
            // `i <= i_limit`.
            let hash = Self::hash(t.hash_log, t.hash_salt, src, i);
            Self::prefetch_row(
                t.hash,
                t.tag,
                ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize,
            );
            self.hash_cache[i & ROW_HASH_CACHE_MASK] = hash;
        }
    }

    /// `ZSTD_row_nextCachedHash`: return the cached hash of `idx`, replace it
    /// by the hash of `idx + 8` and prefetch that row.
    ///
    /// # Safety
    /// `idx + ROW_HASH_CACHE_SIZE + HASH_READ_SIZE <= src.len()`.
    #[inline(always)]
    unsafe fn next_cached_hash(&mut self, t: &RowTables, src: &[u8], idx: usize) -> u32 {
        let new_hash = Self::hash(t.hash_log, t.hash_salt, src, idx + ROW_HASH_CACHE_SIZE);
        Self::prefetch_row(
            t.hash,
            t.tag,
            ((new_hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize,
        );
        let hash = self.hash_cache[idx & ROW_HASH_CACHE_MASK];
        self.hash_cache[idx & ROW_HASH_CACHE_MASK] = new_hash;
        hash
    }

    /// `ZSTD_row_update_internalImpl`: insert `start..end`.
    ///
    /// # Safety
    /// `end + HASH_READ_SIZE <= src.len()`, plus `end + ROW_HASH_CACHE_SIZE`
    /// in place of `end` when `use_cache`; tables of `1 << hash_log` entries.
    #[inline(always)]
    unsafe fn update_impl(
        &mut self,
        t: &mut RowTables,
        src: &[u8],
        start: usize,
        end: usize,
        use_cache: bool,
    ) {
        for idx in start..end {
            let hash = if use_cache {
                self.next_cached_hash(t, src, idx)
            } else {
                Self::hash(t.hash_log, t.hash_salt, src, idx)
            };
            let rel_row = ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize;
            // `rel_row + pos < rel_row + ROW_ENTRIES <= table len`, see `hash`.
            let pos = Self::next_index(Self::head(t.tag, rel_row));
            *t.tag.get_unchecked_mut(rel_row + pos) = (hash & ROW_HASH_TAG_MASK) as u8;
            tset(t.hash, rel_row + pos, idx);
        }
    }

    /// `ZSTD_row_update_internal`: insert `[next_to_update, ip)`, skipping the
    /// middle of a long gap when the cache is in use, and set
    /// `next_to_update = ip`.
    ///
    /// # Safety
    /// `ip + HASH_READ_SIZE <= src.len()`, plus `ip + ROW_HASH_CACHE_SIZE`
    /// in place of `ip` when `use_cache`; tables of `1 << hash_log` entries.
    #[inline(always)]
    unsafe fn update_internal(
        &mut self,
        t: &mut RowTables,
        next_to_update: &mut usize,
        src: &[u8],
        ip: usize,
        use_cache: bool,
    ) {
        const K_SKIP_THRESHOLD: usize = 384;
        const K_MAX_MATCH_START_POSITIONS_TO_UPDATE: usize = 96;
        const K_MAX_MATCH_END_POSITIONS_TO_UPDATE: usize = 32;
        let mut idx = *next_to_update;
        let target = ip;
        if use_cache && target - idx > K_SKIP_THRESHOLD {
            // Only update a set number of positions at the beginning and end
            // of the match.
            let bound = idx + K_MAX_MATCH_START_POSITIONS_TO_UPDATE;
            self.update_impl(t, src, idx, bound, use_cache);
            idx = target - K_MAX_MATCH_END_POSITIONS_TO_UPDATE;
            self.fill_hash_cache(t, src, idx, ip + 1);
        }
        debug_assert!(target >= idx);
        self.update_impl(t, src, idx, target, use_cache);
        *next_to_update = target;
    }
}

impl<M: TagMask + MatchCount, const MLS: u32, const ROW_LOG: u32> Search
    for RowSearch<M, MLS, ROW_LOG>
{
    const ILIMIT_MARGIN: usize = 8 + ROW_HASH_CACHE_SIZE;

    #[inline(always)]
    unsafe fn count(&self, src: &[u8], a: usize, b: usize, limit: usize) -> usize {
        self.mask.count(src, a, b, limit)
    }

    #[inline(always)]
    fn refill(&mut self, ms: &mut MatchState, src: &[u8], ilimit: usize) {
        let next_to_update = ms.next_to_update;
        let t = RowTables::of(ms);
        // SAFETY: `ilimit + ILIMIT_MARGIN <= iend <= src.len()` with
        // `ILIMIT_MARGIN >= HASH_READ_SIZE`.
        unsafe { self.fill_hash_cache(&t, src, next_to_update, ilimit) }
    }

    /// `ZSTD_RowFindBestMatch` (`ZSTD_noDict`).
    #[inline(always)]
    fn search_max(
        &mut self,
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        iend: usize,
        off_base: &mut u32,
        lazy_skipping: bool,
    ) -> usize {
        let curr = ip;
        let low_limit = ms.lowest_prefix_index(curr);
        // nb of searches is capped at nb entries per row
        let capped_search_log = ms.cparams.search_log.min(ROW_LOG);
        let group_width = M::group_width(Self::ROW_ENTRIES as u32);
        let mut nb_attempts = 1u32 << capped_search_log;
        let mut ml = 4 - 1;
        // One binding of the tables for the whole search.
        let (hash_log, hash_salt) = (ms.cparams.hash_log, ms.hash_salt);
        let (hash, _, tag) = ms.ws.tables_mut();
        let mut t = RowTables {
            hash,
            tag,
            hash_log,
            hash_salt,
        };

        // Update the hashTable and tagTable up to (but not including) ip
        // SAFETY: `ip < ilimit` and `ilimit + 8 + ROW_HASH_CACHE_SIZE <=
        // iend <= src.len()`, so every hashed position (`<= ip + 8`) has 8
        // readable bytes; tables are `1 << hash_log` long (asserted per
        // block).
        let hash = unsafe {
            if !lazy_skipping {
                self.update_internal(&mut t, &mut ms.next_to_update, src, ip, true);
                self.next_cached_hash(&t, src, curr)
            } else {
                // Stop inserting every position when in the lazy skipping mode.
                // The hash cache is also not kept up to date in this mode.
                ms.next_to_update = curr;
                Self::hash(hash_log, hash_salt, src, ip)
            }
        };
        ms.hash_salt_entropy = ms.hash_salt_entropy.wrapping_add(hash); // collect salt entropy

        let rel_row = ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize;
        let tag = (hash & ROW_HASH_TAG_MASK) as u8;
        // SAFETY: `rel_row + ROW_ENTRIES <= tag_table.len()`, see `hash`.
        let tag_row = unsafe { t.tag.get_unchecked(rel_row..rel_row + Self::ROW_ENTRIES) };
        let head_grouped = ((tag_row[0] as u32) & Self::ROW_MASK) * group_width;
        let mut num_matches = 0usize;
        let mut matches = self.mask.match_mask::<ROW_LOG>(tag_row, tag, head_grouped);

        // Cycle through the matches and prefetch
        while matches > 0 && nb_attempts > 0 {
            let match_pos =
                ((head_grouped + matches.trailing_zeros()) / group_width) & Self::ROW_MASK;
            matches &= matches - 1;
            // SAFETY: `match_pos < ROW_ENTRIES`, so `rel_row + match_pos <
            // hash_table.len()`.
            let match_index = unsafe { tget(t.hash, rel_row + match_pos as usize) };
            if match_pos == 0 {
                continue;
            }
            // C tests `matchIndex < lowLimit` only; the upper bound is folded
            // into the same compare so that a stale entry is a miss, not an
            // out-of-bounds read below.
            if !candidate_valid(match_index, low_limit, curr) {
                break;
            }
            prefetch_l1(src, match_index);
            // SAFETY: at most `nb_attempts <= ROW_ENTRIES <= 64` candidates
            // are stored, so `num_matches < match_buffer.len()`.
            unsafe { *self.match_buffer.get_unchecked_mut(num_matches) = match_index as u32 };
            num_matches += 1;
            nb_attempts -= 1;
        }

        // Speed opt: insert current byte into hashtable too. This allows us
        // to avoid one iteration of the loop in update_internal() at the next
        // search.
        // SAFETY: `rel_row + pos < rel_row + ROW_ENTRIES <= table len`.
        unsafe {
            let pos = Self::next_index(Self::head(t.tag, rel_row));
            *t.tag.get_unchecked_mut(rel_row + pos) = tag;
            tset(t.hash, rel_row + pos, ms.next_to_update);
            ms.next_to_update += 1;
        }

        // Return the longest match
        for &match_index in &self.match_buffer[..num_matches] {
            let match_index = match_index as usize;
            debug_assert!(match_index < curr && match_index >= low_limit);
            // read 4B starting from (match + ml + 1 - sizeof(U32))
            // SAFETY: `low_limit <= match_index < ip` (candidate_valid above)
            // and `ip + ml < iend <= src.len()` (`ml` starts at 3 with `ip +
            // 16 <= iend`; a match reaching `iend` ends the loop), so the
            // reads at `+ ml - 3` and the count stay inside `src`.
            let better = unsafe { read32(src, match_index + ml - 3) == read32(src, ip + ml - 3) };
            if better {
                // SAFETY: `match_index < ip <= iend <= src.len()`.
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
        ml
    }
}

// ---------------------------------------------------------------------------
// Block loop.
// ---------------------------------------------------------------------------

/// `ZSTD_compressBlock_lazy_generic(ms, seqStore, rep, src, srcSize,
/// searchMethod, depth, ZSTD_noDict)`. Returns the anchor of the trailing
/// literals.
#[inline(always)]
fn lazy_generic<S: Search>(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
    mut search: S,
) -> usize {
    let istart = block.start;
    let iend = block.end;
    let mut anchor = istart;
    // Below `istart + 1` the loop conditions are false anyway (`ip >= 1`).
    let ilimit = iend.saturating_sub(S::ILIMIT_MARGIN);
    let prefix_lowest = ms.window_low;

    let mut offset_1 = rep[0];
    let mut offset_2 = rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    // C: `ip += (dictAndPrefixLength == 0)` with ip starting at the prefix
    // start; here positions below `window_low` do not exist (block 0).
    let mut ip = istart.max(prefix_lowest);
    if ip == prefix_lowest {
        ip += 1;
    }
    {
        let curr = ip;
        let window_low = ms.lowest_prefix_index(curr);
        let max_rep = (curr - window_low) as u32;
        if offset_2 > max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }

    let mut lazy_skipping = false;
    search.refill(ms, src, ilimit);

    // SAFETY, for every unchecked read below: `ip <= ilimit` implies `ip +
    // ILIMIT_MARGIN <= iend <= src.len()` with `ILIMIT_MARGIN >= 8` (when
    // `ilimit` saturated to 0 no `ip >= 1` passes the tests), so 4-byte reads
    // at `<= ip + 1` and counts starting `<= ip + 5` stay inside `src`. A rep
    // offset is `> 0` and `<= ip - window_low` (clamped by `max_rep` above,
    // or the distance to a candidate `>= low_limit >= window_low >= 1`, and
    // `ip` only grows), so `ip - offset >= 1`.
    while ip < ilimit {
        let mut match_length = 0usize;
        let mut off_base = REPCODE1_TO_OFFBASE;
        let mut start = ip + 1;

        // check repCode
        let mut rep_at_depth0 = false;
        // SAFETY: see the loop header.
        let rep_hit = offset_1 > 0
            && unsafe { read32(src, ip + 1 - offset_1 as usize) == read32(src, ip + 1) };
        if rep_hit {
            // SAFETY: see the loop header.
            match_length =
                unsafe { search.count(src, ip + 1 + 4, ip + 1 + 4 - offset_1 as usize, iend) } + 4;
            if depth == 0 {
                rep_at_depth0 = true; // goto _storeSequence
            }
        }

        if !rep_at_depth0 {
            // first search (depth 0)
            {
                let mut offbase_found = 999_999_999u32;
                let ml2 = search.search_max(ms, src, ip, iend, &mut offbase_found, lazy_skipping);
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
                    // SAFETY: see the loop header (`ip <= ilimit`).
                    let rep_hit = off_base != 0
                        && offset_1 > 0
                        && unsafe { read32(src, ip) == read32(src, ip - offset_1 as usize) };
                    if rep_hit {
                        // SAFETY: see the loop header.
                        let ml_rep =
                            unsafe { search.count(src, ip + 4, ip + 4 - offset_1 as usize, iend) }
                                + 4;
                        let gain2 = (ml_rep * 3) as i32;
                        let gain1 = (match_length * 3) as i32 - highbit32(off_base) as i32 + 1;
                        if ml_rep >= 4 && gain2 > gain1 {
                            match_length = ml_rep;
                            off_base = REPCODE1_TO_OFFBASE;
                            start = ip;
                        }
                    }
                    {
                        let mut ofb_candidate = 999_999_999u32;
                        let ml2 =
                            search.search_max(ms, src, ip, iend, &mut ofb_candidate, lazy_skipping);
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
                        // SAFETY: see the loop header (`ip <= ilimit`).
                        let rep_hit = off_base != 0
                            && offset_1 > 0
                            && unsafe { read32(src, ip) == read32(src, ip - offset_1 as usize) };
                        if rep_hit {
                            // SAFETY: see the loop header.
                            let ml_rep = unsafe {
                                search.count(src, ip + 4, ip + 4 - offset_1 as usize, iend)
                            } + 4;
                            let gain2 = (ml_rep * 4) as i32;
                            let gain1 = (match_length * 4) as i32 - highbit32(off_base) as i32 + 1;
                            if ml_rep >= 4 && gain2 > gain1 {
                                match_length = ml_rep;
                                off_base = REPCODE1_TO_OFFBASE;
                                start = ip;
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
                            let gain1 = (match_length * 4) as i32 - highbit32(off_base) as i32 + 7;
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
                // SAFETY: `1 <= start - 1 - offset < start - 1 < ip < iend <=
                // src.len()` (`start > anchor >= 0`, `prefix_lowest >= 1`).
                while start > anchor
                    && start - offset > prefix_lowest
                    && unsafe { byte(src, start - 1) == byte(src, start - 1 - offset) }
                {
                    start -= 1;
                    match_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset as u32;
            }
        }

        // store sequence
        let lit_length = start - anchor;
        out.store_seq(src, anchor, lit_length, iend, off_base, match_length);
        ip = start + match_length;
        anchor = ip;

        if lazy_skipping {
            // We've found a match, disable lazy skipping mode, and refill the hash cache.
            search.refill(ms, src, ilimit);
            lazy_skipping = false;
        }

        // check immediate repcode
        // SAFETY (both): see the loop header (`ip <= ilimit`).
        while ip <= ilimit
            && offset_2 > 0
            && unsafe { read32(src, ip) == read32(src, ip - offset_2 as usize) }
        {
            let match_length =
                unsafe { search.count(src, ip + 4, ip + 4 - offset_2 as usize, iend) } + 4;
            std::mem::swap(&mut offset_1, &mut offset_2); // swap repcodes
            out.store_seq(src, anchor, 0, iend, REPCODE1_TO_OFFBASE, match_length);
            ip += match_length;
            anchor = ip;
        }
    }

    // If offset_1 started invalid (offsetSaved1 != 0) and became valid
    // (offset_1 != 0), rotate saved offsets.
    if offset_saved1 != 0 && offset_1 != 0 {
        offset_saved2 = offset_saved1;
    }
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
    anchor
}

// ---------------------------------------------------------------------------
// Dispatch.
// ---------------------------------------------------------------------------

/// `searchMethod_e`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMethod {
    /// `search_hashChain`.
    HashChain,
    /// `search_rowHash`.
    RowHash,
    /// `search_binaryTree`, the only method of `BtLazy2`.
    BinaryTree,
}

/// The method `ZSTD_selectBlockCompressor` runs: the binary tree for
/// btlazy2; for greedy/lazy/lazy2 `ZSTD_resolveRowMatchFinderMode(ZSTD_ps_auto,
/// cparams)`, the row finder with `windowLog > 14` and the hash chain
/// otherwise.
pub fn default_search_method(cp: &CParams) -> SearchMethod {
    if cp.strategy == Strategy::BtLazy2 {
        SearchMethod::BinaryTree
    } else if cp.row_match_finder_supported() && cp.window_log > 14 {
        SearchMethod::RowHash
    } else {
        SearchMethod::HashChain
    }
}

/// SIMD level of this machine, detected once.
fn detected_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

/// The row-finder block loop for one tag-mask implementation, specialised on
/// `mls` and `rowLog` like C's `ZSTD_FOR_EACH_MLS_ROWLOG` templates.
#[inline(always)]
fn row_block<M: TagMask + MatchCount>(
    mask: M,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    macro_rules! go {
        ($mls:literal, $row_log:literal) => {
            lazy_generic(
                ms,
                src,
                block,
                rep,
                out,
                depth,
                RowSearch::<M, $mls, $row_log>::new(mask),
            )
        };
    }
    match (mls_of(&ms.cparams), row_log_of(&ms.cparams)) {
        (4, 4) => go!(4, 4),
        (4, 5) => go!(4, 5),
        (4, _) => go!(4, 6),
        (5, 4) => go!(5, 4),
        (5, 5) => go!(5, 5),
        (5, _) => go!(5, 6),
        (_, 4) => go!(6, 4),
        (_, 5) => go!(6, 5),
        _ => go!(6, 6),
    }
}

/// [`row_block`] with the scalar tag compare.
#[inline(never)]
fn row_block_scalar(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    row_block(Fallback::new(), ms, src, block, rep, out, depth)
}

/// [`row_block`] compiled with SSE4.2 enabled, using the SSE tag compare.
///
/// # Safety
///
/// The CPU must support SSE4.2, which `mask` attests.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "sse4.2")]
unsafe fn row_block_sse(
    mask: Sse4_2,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    row_block(mask, ms, src, block, rep, out, depth)
}

/// [`row_block`] compiled with AVX2 enabled, using the AVX2 tag compare.
///
/// # Safety
///
/// The CPU must support AVX2, which `mask` attests.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn row_block_avx2(
    mask: Avx2,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    row_block(mask, ms, src, block, rep, out, depth)
}

/// [`row_block`] compiled with NEON enabled, using the NEON tag compare.
///
/// # Safety
///
/// The CPU must support NEON, which `mask` attests.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon")]
unsafe fn row_block_neon(
    mask: Neon,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    row_block(mask, ms, src, block, rep, out, depth)
}

/// The hash-chain block loop specialised on `mls`.
#[inline(never)]
fn hc_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    match mls_of(&ms.cparams) {
        4 => lazy_generic(ms, src, block, rep, out, depth, HcSearch::<4>),
        5 => lazy_generic(ms, src, block, rep, out, depth, HcSearch::<5>),
        _ => lazy_generic(ms, src, block, rep, out, depth, HcSearch::<6>),
    }
}

/// The binary-tree block loop specialised on `mls`
/// (`ZSTD_compressBlock_btlazy2`) for one [`MatchCount`] level.
#[inline(always)]
fn bt_block<M: MatchCount>(
    count: M,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    let p = BtParams::of(ms);
    match mls_of(&ms.cparams) {
        4 => lazy_generic(
            ms,
            src,
            block,
            rep,
            out,
            depth,
            BtSearch::<M, 4> { count, p },
        ),
        5 => lazy_generic(
            ms,
            src,
            block,
            rep,
            out,
            depth,
            BtSearch::<M, 5> { count, p },
        ),
        _ => lazy_generic(
            ms,
            src,
            block,
            rep,
            out,
            depth,
            BtSearch::<M, 6> { count, p },
        ),
    }
}

/// [`bt_block`] with the scalar `ZSTD_count`.
#[inline(never)]
fn bt_block_scalar(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    bt_block(Fallback::new(), ms, src, block, rep, out, depth)
}

/// [`bt_block`] compiled with AVX2 enabled, using the AVX2 match count.
///
/// # Safety
///
/// The CPU must support AVX2, which `count` attests.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn bt_block_avx2(
    count: Avx2,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    depth: u32,
) -> usize {
    bt_block(count, ms, src, block, rep, out, depth)
}

/// `ZSTD_compressBlock_greedy/lazy/lazy2[_row]/btlazy2` for the strategy in
/// `ms.cparams`, with the match finder of [`default_search_method`] and the
/// SIMD level detected on this machine. Sequences go to `out`, `rep` is
/// updated for the next block; returns the anchor of the trailing literals.
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let method = default_search_method(&ms.cparams);
    compress_block_with(ms, src, block, rep, out, method, detected_level())
}

/// The per-block facts every unchecked access in this module rests on: the
/// block (or prefix) ends inside `src`, the tables have the sizes
/// [`MatchState::new`] gives them, and a row hash fits in 32 bits
/// (`ZSTD_adjustCParams_internal` caps `hashLog` at `rowLog + 24`).
fn assert_block_bounds(ms: &MatchState, src: &[u8], end: usize, method: SearchMethod) {
    let cp = &ms.cparams;
    let (hash_table, chain_table, tag_table) = ms.tables();
    assert!(
        end <= src.len(),
        "block end {end} past src.len() {}",
        src.len()
    );
    assert_eq!(hash_table.len(), 1usize << cp.hash_log, "hash_table size");
    match method {
        SearchMethod::HashChain | SearchMethod::BinaryTree => {
            assert_eq!(
                chain_table.len(),
                1usize << cp.chain_log,
                "chain_table size"
            );
        }
        SearchMethod::RowHash => {
            assert_eq!(tag_table.len(), 1usize << cp.hash_log, "tag_table size");
            let row_log = row_log_of(cp);
            assert!(
                cp.hash_log >= row_log && cp.hash_log - row_log + ROW_HASH_TAG_BITS <= 32,
                "hash_log {} out of range for row_log {row_log}",
                cp.hash_log
            );
        }
    }
}

/// [`compress_block`] with an explicit match finder and SIMD level. Every
/// combination produces the same sequences for the same method; the level
/// only selects the tag-compare and match-count kernels.
pub fn compress_block_with(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    method: SearchMethod,
    level: Level,
) -> usize {
    let depth = depth_of(ms.cparams.strategy);
    assert_block_bounds(ms, src, block.end, method);
    match method {
        SearchMethod::HashChain => hc_block(ms, src, block, rep, out, depth),
        // SAFETY (all four unsafe arms): fearless_simd constructs a witness
        // only after detecting its feature set on this CPU.
        SearchMethod::BinaryTree => match level {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2(w) => unsafe { bt_block_avx2(w, ms, src, block, rep, out, depth) },
            _ => bt_block_scalar(ms, src, block, rep, out, depth),
        },
        SearchMethod::RowHash => match level {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse4_2(w) => unsafe { row_block_sse(w, ms, src, block, rep, out, depth) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2(w) => unsafe { row_block_avx2(w, ms, src, block, rep, out, depth) },
            #[cfg(target_arch = "aarch64")]
            Level::Neon(w) => unsafe { row_block_neon(w, ms, src, block, rep, out, depth) },
            _ => row_block_scalar(ms, src, block, rep, out, depth),
        },
    }
}

/// `ZSTD_loadDictionaryContent`, lazy arm, for the finder of
/// [`default_search_method`]: see [`load_prefix_with`].
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    load_prefix_with(ms, src, range, default_search_method(&ms.cparams))
}

/// `ZSTD_loadDictionaryContent`, lazy and btlazy2 arms: insert every
/// position of `range` up to `end - HASH_READ_SIZE`
/// (`ZSTD_insertAndFindFirstIndex` / `ZSTD_row_update` / `ZSTD_updateTree`
/// at `iend - HASH_READ_SIZE`) and set `next_to_update =
/// end`. Expects the tables of a fresh [`MatchState`] (C zeroes the tag table
/// here; `MatchState::new` already did). The hash width is `BOUNDED(4,
/// minMatch, 6)` as in the block loop, where C's chain loader passes
/// `minMatch` itself; they differ only for `minMatch == 7`, which no level
/// table produces.
pub fn load_prefix_with(
    ms: &mut MatchState,
    src: &[u8],
    range: Range<usize>,
    method: SearchMethod,
) {
    let end = range.end;
    assert_block_bounds(ms, src, end, method);
    let start = ms.next_to_update.max(range.start).max(ms.window_low);
    if end >= start + HASH_READ_SIZE {
        let target = end - HASH_READ_SIZE;
        ms.next_to_update = start;
        // SAFETY (every finder): `target + HASH_READ_SIZE == end <= src.len()`
        // and the table sizes were asserted above.
        match method {
            SearchMethod::HashChain => {
                match mls_of(&ms.cparams) {
                    4 => unsafe {
                        HcSearch::<4>::insert_and_find_first_index(ms, src, target, false)
                    },
                    5 => unsafe {
                        HcSearch::<5>::insert_and_find_first_index(ms, src, target, false)
                    },
                    _ => unsafe {
                        HcSearch::<6>::insert_and_find_first_index(ms, src, target, false)
                    },
                };
            }
            SearchMethod::RowHash => {
                macro_rules! go {
                    ($mls:literal, $row_log:literal) => {
                        unsafe {
                            let mut next_to_update = ms.next_to_update;
                            let mut t = RowTables::of(ms);
                            RowSearch::<Fallback, $mls, $row_log>::new(Fallback::new())
                                .update_internal(&mut t, &mut next_to_update, src, target, false);
                            ms.next_to_update = next_to_update;
                        }
                    };
                }
                match (mls_of(&ms.cparams), row_log_of(&ms.cparams)) {
                    (4, 4) => go!(4, 4),
                    (4, 5) => go!(4, 5),
                    (4, _) => go!(4, 6),
                    (5, 4) => go!(5, 4),
                    (5, 5) => go!(5, 5),
                    (5, _) => go!(5, 6),
                    (_, 4) => go!(6, 4),
                    (_, 5) => go!(6, 5),
                    _ => go!(6, 6),
                }
            }
            // `ZSTD_updateTree(ms, iend - HASH_READ_SIZE, iend)`: "we want
            // the dictionary table fully sorted".
            SearchMethod::BinaryTree => super::bt::update_tree(ms, src, target, end),
        }
    }
    ms.next_to_update = end;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::params::CParams;
    use crate::constants::ZSTD_BLOCKSIZE_MAX;

    fn lazy_params(level: i32, src_len: usize, strategy: Strategy, mls: u32) -> CParams {
        let mut cp = CParams::for_level(level, src_len);
        cp.strategy = strategy;
        cp.min_match = mls;
        cp
    }

    /// Compress `src[job_start..]` block by block on a state whose window
    /// starts at `window_low`, loading `[window_low, job_start)` as prefix,
    /// and check that every block reconstructs byte-exactly.
    fn run_blocks(
        src: &[u8],
        cp: CParams,
        window_low: usize,
        job_start: usize,
        block_size: usize,
        rep0: [u32; 3],
        method: SearchMethod,
    ) -> (usize, usize) {
        let mut ms = MatchState::new(cp, window_low);
        if job_start > window_low {
            load_prefix_with(&mut ms, src, window_low..job_start, method);
            assert_eq!(ms.next_to_update, job_start);
        }
        let mut rep = rep0;
        let mut store = SeqStore::new();
        let mut start = job_start;
        let (mut nseqs, mut nlits) = (0usize, 0usize);
        while start < src.len() {
            let end = (start + block_size).min(src.len());
            store.clear();
            let rep_in = rep;
            let anchor = compress_block_with(
                &mut ms,
                src,
                start..end,
                &mut rep,
                &mut store,
                method,
                Level::fallback(),
            );
            assert!(anchor >= start && anchor <= end);
            store.lits.extend_from_slice(&src[anchor..end]);
            let total: usize = store
                .seqs
                .iter()
                .map(|s| s.lit_len as usize + s.match_len() as usize)
                .sum::<usize>()
                + (end - anchor);
            assert_eq!(total, end - start, "block {start}..{end}");
            let got = store.reconstruct(&src[..start], rep_in);
            assert_eq!(got, &src[start..end], "block {start}..{end} {cp:?}");
            assert!(ms.next_to_update <= end);
            nseqs += store.seqs.len();
            nlits += store.lits.len();
            start = end;
        }
        (nseqs, nlits)
    }

    const METHODS: [SearchMethod; 3] = [
        SearchMethod::HashChain,
        SearchMethod::RowHash,
        SearchMethod::BinaryTree,
    ];

    fn run_m(src: &[u8], cp: CParams, block_size: usize, method: SearchMethod) -> (usize, usize) {
        run_blocks(src, cp, 1, 0, block_size, [1, 4, 8], method)
    }

    /// Both finders from a fresh state; returns the hash-chain statistics.
    fn run(src: &[u8], cp: CParams, block_size: usize) -> (usize, usize) {
        let hc = run_m(src, cp, block_size, SearchMethod::HashChain);
        run_m(src, cp, block_size, SearchMethod::RowHash);
        hc
    }

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Bit `b` set iff `row[(b + head) % row.len()] == tag`.
    fn reference_mask(row: &[u8], tag: u8, head: u32) -> u64 {
        let n = row.len();
        let mut m = 0u64;
        for b in 0..n {
            if row[(b + head as usize) % n] == tag {
                m |= 1 << b;
            }
        }
        m
    }

    /// Rows biased towards few distinct tags so that many bits match.
    fn random_rows(seed: u64, n: usize) -> Vec<(Vec<u8>, u8, u32)> {
        let mut rng = XorShift(seed);
        let mut out = Vec::new();
        for entries in [16usize, 32, 64] {
            for _ in 0..n {
                let alphabet = 1 + (rng.next() % 6) as u8;
                let row: Vec<u8> = (0..entries)
                    .map(|_| (rng.next() % alphabet as u64) as u8 * 37)
                    .collect();
                let tag = (rng.next() % alphabet as u64) as u8 * 37;
                let head = (rng.next() % entries as u64) as u32;
                out.push((row, tag, head));
            }
        }
        out
    }

    fn mask_of<M: TagMask>(m: M, row: &[u8], tag: u8, head: u32) -> u64 {
        match row.len() {
            16 => m.match_mask::<4>(row, tag, head),
            32 => m.match_mask::<5>(row, tag, head),
            _ => m.match_mask::<6>(row, tag, head),
        }
    }

    #[test]
    fn swar_mask_matches_reference() {
        for (row, tag, head) in random_rows(0x9E37_79B9_7F4A_7C15, 400) {
            assert_eq!(
                mask_of(Fallback::new(), &row, tag, head),
                reference_mask(&row, tag, head),
                "{row:?} tag {tag} head {head}"
            );
        }
    }

    /// Every SIMD level available on this machine besides the fallback, with
    /// a name for messages. On x86 an AVX2 machine also gets the SSE4.2
    /// witness so both kernels run.
    fn simd_levels() -> Vec<(&'static str, Level)> {
        let mut out = Vec::new();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                // SAFETY: detected just above.
                out.push(("sse4.2", Level::Sse4_2(unsafe { Sse4_2::new_unchecked() })));
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: detected just above.
                out.push(("avx2", Level::Avx2(unsafe { Avx2::new_unchecked() })));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(w) = Level::new().as_neon() {
                out.push(("neon", Level::Neon(w)));
            }
        }
        out
    }

    /// `reference_mask` in the bit layout of [`TagMask::group_width`]: the
    /// flag of entry `b` sits at bit `b * g + 3` for `g == 4` (C keeps the
    /// nibble's top bit, `& 0x88..`), `2 * b` for `g == 2` (`& 0x55..`) and
    /// `b` for `g == 1`.
    fn reference_mask_grouped(row: &[u8], tag: u8, head: u32, g: u32) -> u64 {
        let plain = reference_mask(row, tag, head);
        let mut m = 0u64;
        for b in 0..row.len() as u32 {
            if plain >> b & 1 == 1 {
                m |= 1 << (b * g + if g == 4 { 3 } else { 0 });
            }
        }
        m
    }

    fn mask_of_level(level: Level, row: &[u8], tag: u8, head: u32) -> u64 {
        match level {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse4_2(w) => mask_of(w, row, tag, head),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2(w) => mask_of(w, row, tag, head),
            #[cfg(target_arch = "aarch64")]
            Level::Neon(w) => mask_of(w, row, tag, head * Neon::group_width(row.len() as u32)),
            _ => mask_of(Fallback::new(), row, tag, head),
        }
    }

    fn group_width_of(level: Level, row_entries: u32) -> u32 {
        match level {
            #[cfg(target_arch = "aarch64")]
            Level::Neon(_) => Neon::group_width(row_entries),
            _ => {
                let _ = row_entries;
                1
            }
        }
    }

    #[test]
    fn simd_masks_match_reference() {
        let levels = simd_levels();
        assert!(
            !levels.is_empty() || matches!(Level::new(), Level::Fallback(_)),
            "Level::new() found SIMD but simd_levels() has no witness for it"
        );
        for (name, level) in levels {
            for (row, tag, head) in random_rows(0x1234_5678_9ABC_DEF0, 400) {
                let g = group_width_of(level, row.len() as u32);
                assert_eq!(
                    mask_of_level(level, &row, tag, head),
                    reference_mask_grouped(&row, tag, head, g),
                    "{name}: {row:?} tag {tag} head {head}"
                );
            }
        }
    }

    /// Sequences, literals, final repcodes and anchors of every block.
    fn collect(src: &[u8], cp: CParams, block_size: usize, level: Level) -> (SeqStore, [u32; 3]) {
        let mut ms = MatchState::new(cp, 1);
        collect_on(&mut ms, src, block_size, default_search_method(&cp), level)
    }

    /// Every block of `src` on `ms` as prepared by the caller: all sequences
    /// and literals concatenated, plus the final repeat offsets.
    fn collect_on(
        ms: &mut MatchState,
        src: &[u8],
        block_size: usize,
        method: SearchMethod,
        level: Level,
    ) -> (SeqStore, [u32; 3]) {
        let mut rep = [1u32, 4, 8];
        let mut all = SeqStore::new();
        let mut store = SeqStore::new();
        let mut start = 0;
        while start < src.len() {
            let end = (start + block_size).min(src.len());
            store.clear();
            let anchor =
                compress_block_with(ms, src, start..end, &mut rep, &mut store, method, level);
            all.seqs.extend_from_slice(&store.seqs);
            all.lits.extend_from_slice(&store.lits);
            all.lits.extend_from_slice(&src[anchor..end]);
            start = end;
        }
        (all, rep)
    }

    /// `MatchState::reset` on a context that compressed something else, with
    /// other table sizes, leaves exactly `MatchState::new`'s state (tables,
    /// update pointer, salt) and therefore the same sequences from both
    /// finders, in either direction of table growth.
    #[test]
    fn reset_context_matches_fresh_context() {
        let a = crate_sources();
        let b = current_exe(300_000);
        // Explicit table sizes: the level tables give both inputs the same
        // hash_log, and the reset must shrink and grow the allocations.
        let mut cp_a = lazy_params(11, a.len(), Strategy::Lazy2, 5);
        cp_a.hash_log = 20;
        cp_a.chain_log = 20;
        let mut cp_b = lazy_params(5, b.len(), Strategy::Greedy, 5);
        cp_b.hash_log = 17;
        cp_b.chain_log = 16;
        let level = Level::new();
        for m in METHODS {
            for (first, cp_first, second, cp_second) in [(&a, cp_a, &b, cp_b), (&b, cp_b, &a, cp_a)]
            {
                let mut reused = MatchState::new(cp_first, 1);
                collect_on(&mut reused, first, 40_000, m, level);
                assert!(reused.tables().0.iter().any(|&e| e != 0));
                reused.reset(cp_second, 1);
                let mut fresh = MatchState::new(cp_second, 1);
                assert!(reused.tables() == fresh.tables());
                assert_eq!(reused.next_to_update, fresh.next_to_update);
                assert_eq!(reused.window_low, fresh.window_low);
                assert_eq!(reused.hash_salt, fresh.hash_salt);
                assert_eq!(reused.hash_salt_entropy, fresh.hash_salt_entropy);
                assert_eq!(reused.cparams, fresh.cparams);
                let (r_store, r_rep) = collect_on(&mut reused, second, 40_000, m, level);
                let (f_store, f_rep) = collect_on(&mut fresh, second, 40_000, m, level);
                assert_eq!(r_store.seqs, f_store.seqs, "{m:?} {cp_second:?}");
                assert_eq!(r_store.lits, f_store.lits);
                assert_eq!(r_rep, f_rep);
                assert!(r_store.seqs.len() > 1000);
            }
        }
    }

    #[test]
    fn simd_and_scalar_produce_identical_sequences() {
        let levels = simd_levels();
        let mut srcs = crate_sources();
        srcs.extend_from_slice(&current_exe(300_000));
        // Cover every rowLog (searchLog clamped to 4..6), not only the level
        // table's; levels 13 and 15 run the tree (AVX2 match count).
        for (level_no, block, search_log) in [
            (5, ZSTD_BLOCKSIZE_MAX, 3),
            (7, 5000, 5),
            (9, ZSTD_BLOCKSIZE_MAX, 6),
            (11, 40_000, 4),
            (13, ZSTD_BLOCKSIZE_MAX, 4),
            (15, 5000, 6),
        ] {
            let mut cp = CParams::for_level(level_no, srcs.len());
            cp.search_log = search_log;
            let (want, want_rep) = collect(&srcs, cp, block, Level::fallback());
            assert!(!want.seqs.is_empty());
            for &(name, level) in &levels {
                let (got, got_rep) = collect(&srcs, cp, block, level);
                assert_eq!(got.seqs, want.seqs, "{name} level {level_no} block {block}");
                assert_eq!(got.lits, want.lits, "{name} level {level_no} block {block}");
                assert_eq!(got_rep, want_rep, "{name} level {level_no} block {block}");
            }
        }
    }

    #[test]
    fn default_search_method_follows_window_log() {
        // 10 KiB: windowLog shrinks to 14 -> hash chain.
        let cp = CParams::for_level(7, 10 << 10);
        assert_eq!(cp.window_log, 14);
        assert_eq!(default_search_method(&cp), SearchMethod::HashChain);
        let cp = CParams::for_level(7, 1 << 20);
        assert_eq!(default_search_method(&cp), SearchMethod::RowHash);
        assert_eq!(
            default_search_method(&CParams::for_level(1, 1 << 20)),
            SearchMethod::HashChain
        );
    }

    #[test]
    fn row_all_row_logs_and_capped_attempts_reconstruct() {
        let text = text_corpus();
        for search_log in [3u32, 4, 5, 6, 7] {
            for strategy in [Strategy::Greedy, Strategy::Lazy2] {
                let mut cp = lazy_params(9, text.len(), strategy, 5);
                cp.search_log = search_log;
                run_m(&text, cp, ZSTD_BLOCKSIZE_MAX, SearchMethod::RowHash);
                run_m(&text, cp, 3000, SearchMethod::RowHash);
            }
        }
    }

    fn text_corpus() -> Vec<u8> {
        let mut text = Vec::new();
        for i in 0..20000u32 {
            text.extend_from_slice(
                format!("line {} of the test corpus {}\n", i, i % 37).as_bytes(),
            );
        }
        text
    }

    fn crate_sources() -> Vec<u8> {
        let mut data = Vec::new();
        for f in [
            "src/compress/lazy.rs",
            "src/compress/fast.rs",
            "src/compress/block.rs",
            "src/compress/mod.rs",
            "src/decode.rs",
        ] {
            if let Ok(bytes) =
                std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/").to_owned() + f)
            {
                data.extend_from_slice(&bytes);
            }
        }
        assert!(data.len() > 100_000, "crate sources missing");
        data
    }

    fn current_exe(limit: usize) -> Vec<u8> {
        let exe = std::env::current_exe().expect("current_exe");
        let mut data = std::fs::read(exe).expect("read exe");
        data.truncate(limit);
        data
    }

    #[test]
    fn all_depths_and_mls_reconstruct() {
        let text = text_corpus();
        let srcs = crate_sources();
        for strategy in [Strategy::Greedy, Strategy::Lazy, Strategy::Lazy2] {
            for mls in [4, 5, 6] {
                let cp = lazy_params(7, text.len(), strategy, mls);
                run(&text, cp, ZSTD_BLOCKSIZE_MAX);
                run(&text, cp, 1000);
                let cp = lazy_params(9, srcs.len(), strategy, mls);
                run(&srcs, cp, ZSTD_BLOCKSIZE_MAX);
            }
        }
    }

    #[test]
    fn current_exe_reconstructs() {
        let exe = current_exe(5 << 17);
        for level in [5, 7, 9, 11] {
            let cp = CParams::for_level(level, exe.len());
            run(&exe, cp, ZSTD_BLOCKSIZE_MAX);
        }
    }

    #[test]
    fn levels_5_to_12_reconstruct() {
        let srcs = crate_sources();
        for level in 5..=12 {
            let cp = CParams::for_level(level, srcs.len());
            let (nseqs, nlits) = run(&srcs, cp, ZSTD_BLOCKSIZE_MAX);
            assert!(nseqs > 0 && nlits < srcs.len() / 2, "level {level}");
        }
    }

    fn run_bt(src: &[u8], cp: CParams, block_size: usize) -> (usize, usize) {
        run_blocks(
            src,
            cp,
            1,
            0,
            block_size,
            [1, 4, 8],
            SearchMethod::BinaryTree,
        )
    }

    #[test]
    fn default_search_method_of_btlazy2_is_the_tree() {
        for (level, len) in [(13, 8 << 20), (15, 8 << 20), (11, 100_000), (9, 10_000)] {
            let cp = CParams::for_level(level, len);
            assert_eq!(cp.strategy, Strategy::BtLazy2, "level {level} len {len}");
            assert_eq!(default_search_method(&cp), SearchMethod::BinaryTree);
        }
    }

    #[test]
    fn levels_13_to_15_reconstruct() {
        let srcs = crate_sources();
        let exe = current_exe(3 << 17);
        for level in 13..=15 {
            for src in [&srcs, &exe] {
                let cp = CParams::for_level(level, src.len());
                assert_eq!(cp.strategy, Strategy::BtLazy2);
                let (nseqs, nlits) = run_bt(src, cp, ZSTD_BLOCKSIZE_MAX);
                assert!(nseqs > 0 && nlits < src.len() / 2, "level {level}");
            }
        }
    }

    /// Every `mls`, search depths from two compares (`ZSTD_SEARCHLOG_MIN`:
    /// the unsorted walk stops after one candidate and nullifies the next)
    /// to 512, and short blocks so that searches reach `iend` and the tree
    /// is left half sorted.
    #[test]
    fn bt_all_mls_and_search_logs_reconstruct() {
        let text = text_corpus();
        let srcs = crate_sources();
        for mls in [4, 5, 6] {
            for search_log in [1u32, 2, 4, 9] {
                let mut cp = lazy_params(13, text.len(), Strategy::BtLazy2, mls);
                cp.search_log = search_log;
                run_bt(&text, cp, ZSTD_BLOCKSIZE_MAX);
                run_bt(&text, cp, 777);
                let mut cp = lazy_params(13, srcs.len(), Strategy::BtLazy2, mls);
                cp.search_log = search_log;
                run_bt(&srcs, cp, ZSTD_BLOCKSIZE_MAX);
            }
        }
    }

    /// A tree much smaller than the input (`btLow > 0`: descents stop at
    /// the oldest node still in the tree) and a window smaller than the
    /// tree (`windowLow` above `btLow`), on repetitive and binary data.
    #[test]
    fn bt_small_tree_and_small_window_reconstruct() {
        let text = text_corpus();
        let exe = current_exe(1 << 20);
        for src in [&text, &exe] {
            for (window_log, chain_log) in [(22, 7), (22, 12), (10, 16), (12, 12)] {
                let mut cp = lazy_params(13, src.len(), Strategy::BtLazy2, 5);
                cp.window_log = window_log;
                cp.chain_log = chain_log;
                run_bt(src, cp, ZSTD_BLOCKSIZE_MAX);
            }
        }
        run_bt(
            &vec![0u8; 300_000],
            CParams::for_level(13, 300_000),
            ZSTD_BLOCKSIZE_MAX,
        );
    }

    #[test]
    fn bt_tiny_inputs() {
        for n in 0..48usize {
            let src: Vec<u8> = (0..n).map(|i| (i % 5) as u8).collect();
            let cp = lazy_params(13, src.len().max(1), Strategy::BtLazy2, 5);
            run_bt(&src, cp, ZSTD_BLOCKSIZE_MAX);
            run_bt(&src, cp, 7);
        }
    }

    #[test]
    fn job_start_with_prefix_and_zero_reps() {
        let text = text_corpus();
        let cp = lazy_params(9, text.len(), Strategy::Lazy2, 5);
        for m in METHODS {
            run_blocks(&text, cp, 5000, 70_000, ZSTD_BLOCKSIZE_MAX, [0, 0, 0], m);
            run_blocks(&text, cp, 5000, 5000, ZSTD_BLOCKSIZE_MAX, [0, 0, 0], m);
            run_blocks(&text, cp, 5000, 5008, 777, [0, 0, 0], m);
            // window_low mid-stream, prefix shorter than HASH_READ_SIZE
            run_blocks(&text, cp, 300, 305, ZSTD_BLOCKSIZE_MAX, [0, 0, 0], m);
        }
    }

    #[test]
    fn tiny_inputs() {
        for strategy in [Strategy::Greedy, Strategy::Lazy, Strategy::Lazy2] {
            for n in 0..48usize {
                let src: Vec<u8> = (0..n).map(|i| (i % 5) as u8).collect();
                let cp = lazy_params(7, src.len().max(1), strategy, 5);
                run(&src, cp, ZSTD_BLOCKSIZE_MAX);
                run(&src, cp, 7);
            }
        }
        run(
            &vec![0u8; 300_000],
            CParams::for_level(7, 300_000),
            ZSTD_BLOCKSIZE_MAX,
        );
    }

    #[test]
    fn rep_disabled_at_block_start_is_restored() {
        let src = vec![7u8; 4000];
        let cp = CParams::for_level(5, src.len());
        let mut ms = MatchState::new(cp, 1);
        let mut store = SeqStore::new();
        let mut rep = [100u32, 4, 8];
        let anchor = compress_block(&mut ms, &src, 0..src.len(), &mut rep, &mut store);
        store.lits.extend_from_slice(&src[anchor..]);
        assert_eq!(store.reconstruct(&[], [100, 4, 8]), src);
        assert_eq!(rep[1], 100);
        assert_eq!(rep[2], 8);
    }

    #[test]
    fn salt_and_hash_match_c_constants() {
        // ZSTD_bitmix(0, 8) ^ ZSTD_bitmix(0, 4), evaluated by hand from the
        // C definition: both terms are pure functions of `len`.
        assert_eq!(initial_hash_salt(), bitmix(0, 8) ^ bitmix(0, 4));
        assert_ne!(initial_hash_salt(), 0);
        let src = b"abcdefghijklmnop";
        // SAFETY: `0 + 8 <= src.len()`.
        let (h4, h5, h6) = unsafe {
            (
                hash_salted::<4>(src, 0, 20, 0),
                hash_salted::<5>(src, 0, 20, 0) as u64,
                hash_salted::<6>(src, 0, 20, 0) as u64,
            )
        };
        // ZSTD_hash4Ptr: (readLE32 * 2654435761) >> (32 - 20)
        let u = u32::from_le_bytes(*b"abcd");
        assert_eq!(h4, u.wrapping_mul(PRIME4) >> 12);
        let u = u64::from_le_bytes(src[..8].try_into().unwrap());
        assert_eq!(h5, (u << 24).wrapping_mul(PRIME5) >> 44);
        assert_eq!(h6, (u << 16).wrapping_mul(PRIME6) >> 44);
    }
}
