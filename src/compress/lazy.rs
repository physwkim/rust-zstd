//! Greedy / lazy / lazy2 block compressors: port of `zstd_lazy.c`
//! (`ZSTD_compressBlock_lazy_generic`, no-dictionary mode) with the
//! hash-chain match finder (`ZSTD_HcFindBestMatch`).
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
use std::ops::Range;

/// `HASH_READ_SIZE`: bytes a hash may read past a position.
const HASH_READ_SIZE: usize = 8;
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

/// `BOUNDED(4, minMatch, 6)`.
#[inline]
fn mls_of(cp: &CParams) -> u32 {
    cp.min_match.clamp(4, 6)
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

/// `ZSTD_compressBlock_greedy/lazy/lazy2` (hash chain) for the strategy in
/// `ms.cparams`. Sequences go to `out`, `rep` is updated for the next block;
/// returns the anchor of the trailing literals.
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    let depth = depth_of(ms.cparams.strategy);
    match mls_of(&ms.cparams) {
        4 => lazy_generic(ms, src, block, rep, out, depth, HcSearch::<4>),
        5 => lazy_generic(ms, src, block, rep, out, depth, HcSearch::<5>),
        _ => lazy_generic(ms, src, block, rep, out, depth, HcSearch::<6>),
    }
}

/// `ZSTD_loadDictionaryContent`, lazy arm: insert every position of `range`
/// up to `end - HASH_READ_SIZE` (`ZSTD_insertAndFindFirstIndex(ms, iend -
/// HASH_READ_SIZE)`) and set `next_to_update = end`. Expects the tables of
/// a fresh [`MatchState`]. The hash width is `BOUNDED(4, minMatch, 6)` as in
/// the block loop, where C's dictionary loader would pass `minMatch` itself;
/// they differ only for `minMatch == 7`, which no level table produces.
pub fn load_prefix(ms: &mut MatchState, src: &[u8], range: Range<usize>) {
    let end = range.end;
    let start = ms.next_to_update.max(range.start).max(ms.window_low);
    if end >= start + HASH_READ_SIZE {
        let target = end - HASH_READ_SIZE;
        match mls_of(&ms.cparams) {
            4 => HcSearch::<4>::insert_and_find_first_index(ms, src, target, false),
            5 => HcSearch::<5>::insert_and_find_first_index(ms, src, target, false),
            _ => HcSearch::<6>::insert_and_find_first_index(ms, src, target, false),
        };
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
    ) -> (usize, usize) {
        let mut ms = MatchState::new(cp, window_low);
        if job_start > window_low {
            load_prefix(&mut ms, src, window_low..job_start);
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
            let anchor = compress_block(&mut ms, src, start..end, &mut rep, &mut store);
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

    fn run(src: &[u8], cp: CParams, block_size: usize) -> (usize, usize) {
        run_blocks(src, cp, 1, 0, block_size, [1, 4, 8])
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
    fn hc_all_depths_and_mls_reconstruct() {
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
    fn hc_current_exe_reconstructs() {
        let exe = current_exe(5 << 17);
        for level in [5, 7, 9, 11] {
            let cp = CParams::for_level(level, exe.len());
            run(&exe, cp, ZSTD_BLOCKSIZE_MAX);
        }
    }

    #[test]
    fn hc_levels_5_to_12_reconstruct() {
        let srcs = crate_sources();
        for level in 5..=12 {
            let cp = CParams::for_level(level, srcs.len());
            let (nseqs, nlits) = run(&srcs, cp, ZSTD_BLOCKSIZE_MAX);
            assert!(nseqs > 0 && nlits < srcs.len() / 2, "level {level}");
        }
    }

    #[test]
    fn hc_job_start_with_prefix_and_zero_reps() {
        let text = text_corpus();
        let cp = lazy_params(9, text.len(), Strategy::Lazy2, 5);
        run_blocks(&text, cp, 5000, 70_000, ZSTD_BLOCKSIZE_MAX, [0, 0, 0]);
        run_blocks(&text, cp, 5000, 5000, ZSTD_BLOCKSIZE_MAX, [0, 0, 0]);
        run_blocks(&text, cp, 5000, 5008, 777, [0, 0, 0]);
        // window_low mid-stream, prefix shorter than HASH_READ_SIZE
        run_blocks(&text, cp, 300, 305, ZSTD_BLOCKSIZE_MAX, [0, 0, 0]);
    }

    #[test]
    fn hc_tiny_inputs() {
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
