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

use super::matchstate::MatchState;
use super::params::{CParams, Strategy};
use super::seqstore::{
    offbase_is_offset, offbase_to_offset, offset_to_offbase, SeqStore, REPCODE1_TO_OFFBASE,
};
use fearless_simd::{Fallback, Level};
use std::ops::Range;
use std::sync::OnceLock;

/// `HASH_READ_SIZE`: bytes a hash may read past a position.
const HASH_READ_SIZE: usize = 8;
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
// Small helpers (private; the fast-path worker has its own copies).
// ---------------------------------------------------------------------------

#[inline(always)]
fn read32(src: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(src[pos..pos + 4].try_into().unwrap())
}

#[inline(always)]
fn read64(src: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(src[pos..pos + 8].try_into().unwrap())
}

/// `ZSTD_highbit32`: index of the highest set bit (`v != 0`).
#[inline(always)]
fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

/// `ZSTD_count(pIn = src[ip..], pMatch = src[mp..], pInLimit = src[limit])`:
/// number of equal bytes, with `mp < ip` and `limit <= src.len()`.
#[inline(always)]
fn count(src: &[u8], ip: usize, mp: usize, limit: usize) -> usize {
    debug_assert!(mp < ip && ip <= limit && limit <= src.len());
    let start = ip;
    let mut ip = ip;
    let mut mp = mp;
    while ip + 8 <= limit {
        let diff = read64(src, ip) ^ read64(src, mp);
        if diff != 0 {
            return ip - start + (diff.trailing_zeros() / 8) as usize;
        }
        ip += 8;
        mp += 8;
    }
    while ip < limit && src[ip] == src[mp] {
        ip += 1;
        mp += 1;
    }
    ip - start
}

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;

/// `ZSTD_hashPtrSalted(src + pos, hbits, mls, salt)` for `mls` 4..=6 and
/// `hbits <= 32` (`ZSTD_hashPtr` is the same with `salt == 0`). `mls == 4`
/// only uses the low 32 bits of the salt, like C.
#[inline(always)]
fn hash_salted<const MLS: u32>(src: &[u8], pos: usize, hbits: u32, salt: u64) -> u32 {
    debug_assert!(hbits <= 32);
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

/// Lazy depth of a strategy: greedy 0, lazy 1, lazy2 2.
#[inline]
fn depth_of(strategy: Strategy) -> u32 {
    match strategy {
        Strategy::Greedy => 0,
        Strategy::Lazy => 1,
        Strategy::Lazy2 => 2,
        Strategy::Fast | Strategy::DFast => unreachable!("not a lazy strategy"),
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

    /// Work done at block start and whenever lazy skipping ends
    /// (`ZSTD_row_fillHashCache` for the row finder; nothing for chains).
    fn refill(&mut self, ms: &MatchState, src: &[u8], ilimit: usize);

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

/// `search_hashChain`: `ZSTD_HcFindBestMatch` over `ms.hash_table` heads and
/// `ms.chain_table` links.
struct HcSearch<const MLS: u32>;

impl<const MLS: u32> HcSearch<MLS> {
    /// `ZSTD_insertAndFindFirstIndex_internal`: insert `[next_to_update, ip)`
    /// (only one position while lazy skipping) and return the chain head of
    /// `ip`'s hash.
    #[inline(always)]
    fn insert_and_find_first_index(
        ms: &mut MatchState,
        src: &[u8],
        ip: usize,
        lazy_skipping: bool,
    ) -> u32 {
        let hash_log = ms.cparams.hash_log;
        let chain_mask = (1usize << ms.cparams.chain_log) - 1;
        let target = ip;
        let mut idx = ms.next_to_update;
        while idx < target {
            let h = hash_salted::<MLS>(src, idx, hash_log, 0) as usize;
            ms.chain_table[idx & chain_mask] = ms.hash_table[h];
            ms.hash_table[h] = idx as u32;
            idx += 1;
            if lazy_skipping {
                break;
            }
        }
        ms.next_to_update = target;
        ms.hash_table[hash_salted::<MLS>(src, ip, hash_log, 0) as usize]
    }
}

impl<const MLS: u32> Search for HcSearch<MLS> {
    const ILIMIT_MARGIN: usize = 8;

    #[inline(always)]
    fn refill(&mut self, _ms: &MatchState, _src: &[u8], _ilimit: usize) {}

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

        let mut match_index =
            Self::insert_and_find_first_index(ms, src, ip, lazy_skipping) as usize;
        // Every candidate is < ip and >= low_limit >= 1; `ip + ml < iend`
        // holds because a match reaching iend ends the loop, so the 4-byte
        // reads at `+ ml - 3` stay inside `src`.
        while match_index >= low_limit && nb_attempts > 0 {
            if read32(src, match_index + ml - 3) == read32(src, ip + ml - 3) {
                let current_ml = count(src, ip, match_index, iend);
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
            match_index = ms.chain_table[match_index & chain_mask] as usize;
            nb_attempts -= 1;
        }
        ml
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
    let splat = (tag as u64).wrapping_mul(X01);
    let mut matches = 0u64;
    let mut i = (1usize << ROW_LOG) - CHUNK;
    loop {
        let mut chunk = read64(row, i) ^ splat;
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

/// `search_rowHash`: rows of `1 << ROW_LOG` hash-table entries selected by
/// the high hash bits, each with a one-byte tag per entry and the row head
/// in the tag row's byte 0.
struct RowSearch<M: TagMask, const MLS: u32, const ROW_LOG: u32> {
    mask: M,
    /// `ms->hashCache`. C keeps it in the match state, but refills it in the
    /// block prologue (`ZSTD_row_fillHashCache`) and after lazy skipping, so
    /// it never carries information across blocks and lives here.
    hash_cache: [u32; ROW_HASH_CACHE_SIZE],
}

impl<M: TagMask, const MLS: u32, const ROW_LOG: u32> RowSearch<M, MLS, ROW_LOG> {
    const ROW_ENTRIES: usize = 1 << ROW_LOG;
    const ROW_MASK: u32 = (1 << ROW_LOG) - 1;

    fn new(mask: M) -> Self {
        Self {
            mask,
            hash_cache: [0; ROW_HASH_CACHE_SIZE],
        }
    }

    /// `ZSTD_hashPtrSalted(p, rowHashLog + ZSTD_ROW_HASH_TAG_BITS, mls,
    /// hashSalt)` with `rowHashLog = hashLog - rowLog`.
    #[inline(always)]
    fn hash(ms: &MatchState, src: &[u8], pos: usize) -> u32 {
        let hbits = ms.cparams.hash_log - ROW_LOG + ROW_HASH_TAG_BITS;
        hash_salted::<MLS>(src, pos, hbits, ms.hash_salt)
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
    fn prefetch_row(ms: &MatchState, rel_row: usize) {
        prefetch_l1(&ms.hash_table, rel_row);
        if ROW_LOG >= 5 {
            prefetch_l1(&ms.hash_table, rel_row + 16);
        }
        prefetch_l1(&ms.tag_table, rel_row);
        if ROW_LOG == 6 {
            prefetch_l1(&ms.tag_table, rel_row + 32);
        }
    }

    /// `ZSTD_row_fillHashCache(ms, base, rowLog, mls, idx, iLimit)`: hash
    /// (and prefetch the rows of) `idx..idx+8`, not beyond `i_limit`.
    #[inline(always)]
    fn fill_hash_cache(&mut self, ms: &MatchState, src: &[u8], idx: usize, i_limit: usize) {
        let max_elems = if idx > i_limit { 0 } else { i_limit - idx + 1 };
        let lim = idx + ROW_HASH_CACHE_SIZE.min(max_elems);
        for i in idx..lim {
            let hash = Self::hash(ms, src, i);
            Self::prefetch_row(ms, ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize);
            self.hash_cache[i & ROW_HASH_CACHE_MASK] = hash;
        }
    }

    /// `ZSTD_row_nextCachedHash`: return the cached hash of `idx`, replace it
    /// by the hash of `idx + 8` and prefetch that row.
    #[inline(always)]
    fn next_cached_hash(&mut self, ms: &MatchState, src: &[u8], idx: usize) -> u32 {
        let new_hash = Self::hash(ms, src, idx + ROW_HASH_CACHE_SIZE);
        Self::prefetch_row(ms, ((new_hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize);
        let hash = self.hash_cache[idx & ROW_HASH_CACHE_MASK];
        self.hash_cache[idx & ROW_HASH_CACHE_MASK] = new_hash;
        hash
    }

    /// `ZSTD_row_update_internalImpl`: insert `start..end`.
    #[inline(always)]
    fn update_impl(
        &mut self,
        ms: &mut MatchState,
        src: &[u8],
        start: usize,
        end: usize,
        use_cache: bool,
    ) {
        for idx in start..end {
            let hash = if use_cache {
                self.next_cached_hash(ms, src, idx)
            } else {
                Self::hash(ms, src, idx)
            };
            let rel_row = ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize;
            let pos = Self::next_index(&mut ms.tag_table[rel_row]);
            ms.tag_table[rel_row + pos] = (hash & ROW_HASH_TAG_MASK) as u8;
            ms.hash_table[rel_row + pos] = idx as u32;
        }
    }

    /// `ZSTD_row_update_internal`: insert `[next_to_update, ip)`, skipping the
    /// middle of a long gap when the cache is in use, and set
    /// `next_to_update = ip`.
    #[inline(always)]
    fn update_internal(&mut self, ms: &mut MatchState, src: &[u8], ip: usize, use_cache: bool) {
        const K_SKIP_THRESHOLD: usize = 384;
        const K_MAX_MATCH_START_POSITIONS_TO_UPDATE: usize = 96;
        const K_MAX_MATCH_END_POSITIONS_TO_UPDATE: usize = 32;
        let mut idx = ms.next_to_update;
        let target = ip;
        if use_cache && target - idx > K_SKIP_THRESHOLD {
            // Only update a set number of positions at the beginning and end
            // of the match.
            let bound = idx + K_MAX_MATCH_START_POSITIONS_TO_UPDATE;
            self.update_impl(ms, src, idx, bound, use_cache);
            idx = target - K_MAX_MATCH_END_POSITIONS_TO_UPDATE;
            self.fill_hash_cache(ms, src, idx, ip + 1);
        }
        debug_assert!(target >= idx);
        self.update_impl(ms, src, idx, target, use_cache);
        ms.next_to_update = target;
    }
}

impl<M: TagMask, const MLS: u32, const ROW_LOG: u32> Search for RowSearch<M, MLS, ROW_LOG> {
    const ILIMIT_MARGIN: usize = 8 + ROW_HASH_CACHE_SIZE;

    #[inline(always)]
    fn refill(&mut self, ms: &MatchState, src: &[u8], ilimit: usize) {
        self.fill_hash_cache(ms, src, ms.next_to_update, ilimit);
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

        // Update the hashTable and tagTable up to (but not including) ip
        let hash = if !lazy_skipping {
            self.update_internal(ms, src, ip, true);
            self.next_cached_hash(ms, src, curr)
        } else {
            // Stop inserting every position when in the lazy skipping mode.
            // The hash cache is also not kept up to date in this mode.
            ms.next_to_update = curr;
            Self::hash(ms, src, ip)
        };
        ms.hash_salt_entropy = ms.hash_salt_entropy.wrapping_add(hash); // collect salt entropy

        let rel_row = ((hash >> ROW_HASH_TAG_BITS) << ROW_LOG) as usize;
        let tag = (hash & ROW_HASH_TAG_MASK) as u8;
        let head_grouped = ((ms.tag_table[rel_row] as u32) & Self::ROW_MASK) * group_width;
        let mut match_buffer = [0u32; ROW_HASH_MAX_ENTRIES];
        let mut num_matches = 0usize;
        let mut matches = self.mask.match_mask::<ROW_LOG>(
            &ms.tag_table[rel_row..rel_row + Self::ROW_ENTRIES],
            tag,
            head_grouped,
        );

        // Cycle through the matches and prefetch
        while matches > 0 && nb_attempts > 0 {
            let match_pos =
                ((head_grouped + matches.trailing_zeros()) / group_width) & Self::ROW_MASK;
            matches &= matches - 1;
            let match_index = ms.hash_table[rel_row + match_pos as usize];
            if match_pos == 0 {
                continue;
            }
            if (match_index as usize) < low_limit {
                break;
            }
            prefetch_l1(src, match_index as usize);
            match_buffer[num_matches] = match_index;
            num_matches += 1;
            nb_attempts -= 1;
        }

        // Speed opt: insert current byte into hashtable too. This allows us
        // to avoid one iteration of the loop in update_internal() at the next
        // search.
        {
            let pos = Self::next_index(&mut ms.tag_table[rel_row]);
            ms.tag_table[rel_row + pos] = tag;
            ms.hash_table[rel_row + pos] = ms.next_to_update as u32;
            ms.next_to_update += 1;
        }

        // Return the longest match
        for &match_index in &match_buffer[..num_matches] {
            let match_index = match_index as usize;
            debug_assert!(match_index < curr && match_index >= low_limit);
            // read 4B starting from (match + ml + 1 - sizeof(U32))
            if read32(src, match_index + ml - 3) == read32(src, ip + ml - 3) {
                let current_ml = count(src, ip, match_index, iend);
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

    while ip < ilimit {
        let mut match_length = 0usize;
        let mut off_base = REPCODE1_TO_OFFBASE;
        let mut start = ip + 1;

        // check repCode
        let mut rep_at_depth0 = false;
        if offset_1 > 0 && read32(src, ip + 1 - offset_1 as usize) == read32(src, ip + 1) {
            match_length = count(src, ip + 1 + 4, ip + 1 + 4 - offset_1 as usize, iend) + 4;
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
                    if off_base != 0
                        && offset_1 > 0
                        && read32(src, ip) == read32(src, ip - offset_1 as usize)
                    {
                        let ml_rep = count(src, ip + 4, ip + 4 - offset_1 as usize, iend) + 4;
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
                        if off_base != 0
                            && offset_1 > 0
                            && read32(src, ip) == read32(src, ip - offset_1 as usize)
                        {
                            let ml_rep = count(src, ip + 4, ip + 4 - offset_1 as usize, iend) + 4;
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
                while start > anchor
                    && start - offset > prefix_lowest
                    && src[start - 1] == src[start - 1 - offset]
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
        out.store_seq(src, anchor, lit_length, off_base, match_length);
        ip = start + match_length;
        anchor = ip;

        if lazy_skipping {
            // We've found a match, disable lazy skipping mode, and refill the hash cache.
            search.refill(ms, src, ilimit);
            lazy_skipping = false;
        }

        // check immediate repcode
        while ip <= ilimit && offset_2 > 0 && read32(src, ip) == read32(src, ip - offset_2 as usize)
        {
            let match_length = count(src, ip + 4, ip + 4 - offset_2 as usize, iend) + 4;
            std::mem::swap(&mut offset_1, &mut offset_2); // swap repcodes
            out.store_seq(src, anchor, 0, REPCODE1_TO_OFFBASE, match_length);
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

/// `searchMethod_e` (without the binary tree, which belongs to `zstd_opt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMethod {
    /// `search_hashChain`.
    HashChain,
    /// `search_rowHash`.
    RowHash,
}

/// `ZSTD_resolveRowMatchFinderMode(ZSTD_ps_auto, cparams)`: the row finder
/// for greedy/lazy/lazy2 with `windowLog > 14`, the hash chain otherwise.
pub fn default_search_method(cp: &CParams) -> SearchMethod {
    if cp.row_match_finder_supported() && cp.window_log > 14 {
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
fn row_block<M: TagMask>(
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

/// `ZSTD_compressBlock_greedy/lazy/lazy2[_row]` for the strategy in
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

/// [`compress_block`] with an explicit match finder and SIMD level. Every
/// combination produces the same sequences for the same method; the level
/// only selects the tag-compare kernel.
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
    limit_update_after_long_match(ms, block.start);
    match method {
        SearchMethod::HashChain => hc_block(ms, src, block, rep, out, depth),
        SearchMethod::RowHash => match level {
            Level::Fallback(_) => row_block_scalar(ms, src, block, rep, out, depth),
            _ => row_block_scalar(ms, src, block, rep, out, depth),
        },
    }
}

/// `ZSTD_buildSeqStore`, "limited update after a very long match": when the
/// previous block left more than 384 positions uninserted (its last match ran
/// past the block end), insert at most the 192 positions before `curr` (fewer
/// while the backlog is under 576) instead of the whole backlog. C applies
/// this in the block driver for every strategy; it is applied here so that
/// the lazy compressors produce C's sequences on their own, and applying it
/// twice is a no-op.
#[inline]
fn limit_update_after_long_match(ms: &mut MatchState, curr: usize) {
    if curr > ms.next_to_update + 384 {
        ms.next_to_update = curr - 192.min(curr - ms.next_to_update - 384);
    }
}

/// `ZSTD_loadDictionaryContent`, lazy arm, for the finder of
/// [`default_search_method`]: see [`load_prefix_with`].
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    load_prefix_with(ms, src, range, default_search_method(&ms.cparams))
}

/// `ZSTD_loadDictionaryContent`, lazy arm: insert every position of `range`
/// up to `end - HASH_READ_SIZE` (`ZSTD_insertAndFindFirstIndex` /
/// `ZSTD_row_update` at `iend - HASH_READ_SIZE`) and set `next_to_update =
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
    let start = ms.next_to_update.max(range.start).max(ms.window_low);
    if end >= start + HASH_READ_SIZE {
        let target = end - HASH_READ_SIZE;
        ms.next_to_update = start;
        match method {
            SearchMethod::HashChain => {
                match mls_of(&ms.cparams) {
                    4 => HcSearch::<4>::insert_and_find_first_index(ms, src, target, false),
                    5 => HcSearch::<5>::insert_and_find_first_index(ms, src, target, false),
                    _ => HcSearch::<6>::insert_and_find_first_index(ms, src, target, false),
                };
            }
            SearchMethod::RowHash => {
                macro_rules! go {
                    ($mls:literal, $row_log:literal) => {
                        RowSearch::<Fallback, $mls, $row_log>::new(Fallback::new())
                            .update_internal(ms, src, target, false)
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

    const METHODS: [SearchMethod; 2] = [SearchMethod::HashChain, SearchMethod::RowHash];

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
    fn limit_update_after_long_match_boundaries() {
        let cp = CParams::for_level(5, 1 << 20);
        let mut ms = MatchState::new(cp, 1);
        // Backlog of exactly 384: untouched.
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1384);
        assert_eq!(ms.next_to_update, 1000);
        // Backlog 385..575: only the excess over 384 gets inserted.
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1385);
        assert_eq!(ms.next_to_update, 1384);
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1575);
        assert_eq!(ms.next_to_update, 1384);
        // Backlog >= 576: insert only the last 192 positions.
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 1576);
        assert_eq!(ms.next_to_update, 1384);
        ms.next_to_update = 1000;
        limit_update_after_long_match(&mut ms, 500_000);
        assert_eq!(ms.next_to_update, 500_000 - 192);
        // Idempotent, so a block driver applying it too changes nothing.
        limit_update_after_long_match(&mut ms, 500_000);
        assert_eq!(ms.next_to_update, 500_000 - 192);
    }

    #[test]
    fn salt_and_hash_match_c_constants() {
        // ZSTD_bitmix(0, 8) ^ ZSTD_bitmix(0, 4), evaluated by hand from the
        // C definition: both terms are pure functions of `len`.
        assert_eq!(initial_hash_salt(), bitmix(0, 8) ^ bitmix(0, 4));
        assert_ne!(initial_hash_salt(), 0);
        let src = b"abcdefghijklmnop";
        // ZSTD_hash4Ptr: (readLE32 * 2654435761) >> (32 - 20)
        let u = u32::from_le_bytes(*b"abcd");
        assert_eq!(
            hash_salted::<4>(src, 0, 20, 0),
            u.wrapping_mul(PRIME4) >> 12
        );
        let u = u64::from_le_bytes(src[..8].try_into().unwrap());
        assert_eq!(
            hash_salted::<5>(src, 0, 20, 0) as u64,
            (u << 24).wrapping_mul(PRIME5) >> 44
        );
        assert_eq!(
            hash_salted::<6>(src, 0, 20, 0) as u64,
            (u << 16).wrapping_mul(PRIME6) >> 44
        );
    }
}
