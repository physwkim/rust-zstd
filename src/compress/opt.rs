//! btopt / btultra / btultra2 block compressors: port of `zstd_opt.c`
//! (`ZSTD_compressBlock_opt_generic`, no-dictionary mode, no LDM) with its
//! price model (`ZSTD_rescaleFreqs`, `ZSTD_setBasePrices`,
//! `ZSTD_updateStats`) and btultra2's first-block statistics pass
//! (`ZSTD_initStats_ultra`). The match finder is `bt_get_all_matches` in
//! [`bt`].
//!
//! Positions follow the [`MatchState`] convention (absolute indices into
//! `src`, `window_low >= 1`); the block loop starts like the lazy one, see
//! [`lazy`](super::lazy).
//!
//! Literals are always entropy-coded for these strategies
//! (`ZSTD_resolveLiteralsCompression` disables them only for negative
//! levels), so `ZSTD_compressedLiterals` is constant true here. Without
//! dictionaries `optPtr->symbolCosts` never holds a valid Huffman table
//! when the statistics are initialized, so that branch of
//! `ZSTD_rescaleFreqs` does not exist: the parser reads nothing produced
//! by the entropy stage.

use super::bt::{self, assert_opt_bounds, bt_get_all_matches, Match, ZSTD_OPT_NUM, ZSTD_OPT_SIZE};
use super::common::{simd_level, HASH_READ_SIZE};
use super::matchstate::MatchState;
use super::params::Strategy;
use super::seqstore::{update_rep, SeqStore};
use crate::constants::{ll_code, ml_code, LL_BITS, MAX_LL, MAX_ML, MAX_OFF, ML_BITS};
use crate::constants::{ZSTD_BLOCKSIZE_MAX, ZSTD_MINMATCH};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Avx2;
use fearless_simd::{Fallback, Level};
use std::ops::Range;

const BITCOST_ACCURACY: u32 = 8;
const BITCOST_MULTIPLIER: u32 = 1 << BITCOST_ACCURACY;
/// `ZSTD_LITFREQ_ADD`: scaling factor for litFreq, so that frequencies
/// adapt faster to new stats.
const ZSTD_LITFREQ_ADD: u32 = 2;
/// `ZSTD_MAX_PRICE`.
const ZSTD_MAX_PRICE: i32 = 1 << 30;
/// `ZSTD_PREDEF_THRESHOLD`: blocks up to this size price symbols from
/// static distributions.
const ZSTD_PREDEF_THRESHOLD: usize = 8;
/// `MaxLit`.
const MAX_LIT: usize = 255;

/// `ZSTD_highbit32`; `v > 0`.
#[inline(always)]
fn highbit32(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

/// `ZSTD_bitWeight`: estimated cost of a stat in full bits only.
#[inline(always)]
fn bit_weight(stat: u32) -> u32 {
    highbit32(stat + 1) * BITCOST_MULTIPLIER
}

/// `ZSTD_fracWeight`: fractional-bit cost of a stat, by linear
/// interpolation.
#[inline(always)]
fn frac_weight(raw_stat: u32) -> u32 {
    let stat = raw_stat + 1;
    let hb = highbit32(stat);
    let b_weight = hb * BITCOST_MULTIPLIER;
    // Fweight was meant for "Fractional weight" but it's effectively a value
    // between 1 and 2 using fixed point arithmetic
    let f_weight = (stat << BITCOST_ACCURACY) >> hb;
    debug_assert!(hb + BITCOST_ACCURACY < 31);
    b_weight + f_weight
}

/// `WEIGHT(stat, optLevel)`: btopt prices whole bits, btultra fractions.
#[inline(always)]
fn weight<const OPT_LEVEL: u32>(stat: u32) -> u32 {
    if OPT_LEVEL != 0 {
        frac_weight(stat)
    } else {
        bit_weight(stat)
    }
}

/// `ZSTD_OptPrice_e`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PriceType {
    /// `zop_dynamic`: prices from the collected statistics.
    Dynamic,
    /// `zop_predef`: static prices for tiny first blocks.
    Predef,
}

/// `ZSTD_optimal_t`: during the forward pass a stretch (a match followed by
/// `litlen` literals) ending at its position; after the backward pass a
/// sequence (`litlen` literals followed by the match).
#[derive(Clone, Copy, Debug, Default)]
struct Optimal {
    price: i32,
    off: u32,
    mlen: u32,
    litlen: u32,
    rep: [u32; 3],
}

/// The statistics half of `optState_t`.
struct Stats {
    lit_freq: [u32; MAX_LIT + 1],
    lit_length_freq: [u32; MAX_LL + 1],
    match_length_freq: [u32; MAX_ML + 1],
    off_code_freq: [u32; MAX_OFF + 1],
    lit_sum: u32,
    lit_length_sum: u32,
    match_length_sum: u32,
    off_code_sum: u32,
    lit_sum_base_price: u32,
    lit_length_sum_base_price: u32,
    match_length_sum_base_price: u32,
    off_code_sum_base_price: u32,
    price_type: PriceType,
}

/// `optState_t`: symbol statistics carried from block to block, plus the
/// parser's match and price tables. Allocated once per [`MatchState`] and
/// kept across resets like libzstd's workspace; a reset only forgets the
/// statistics ([`OptState::invalidate`]).
pub struct OptState {
    stats: Stats,
    matches: Box<[Match; ZSTD_OPT_SIZE]>,
    opt: Box<[Optimal; ZSTD_OPT_SIZE]>,
}

/// `ZSTD_downscaleStats`: `table[s] = base + (table[s] >> shift)` with base
/// `1` (`base_1guaranteed`) or `table[s] > 0` (`base_0possible`); returns
/// the new sum.
fn downscale_stats(table: &mut [u32], shift: u32, base1: bool) -> u32 {
    debug_assert!(shift < 30);
    let mut sum = 0u32;
    for t in table.iter_mut() {
        let base = if base1 { 1 } else { (*t > 0) as u32 };
        let new_stat = base + (*t >> shift);
        sum += new_stat;
        *t = new_stat;
    }
    sum
}

/// `ZSTD_scaleStats`: reduce all frequencies if their sum exceeds
/// `2 << log_target`; returns the resulting sum.
fn scale_stats(table: &mut [u32], log_target: u32) -> u32 {
    let prev_sum: u32 = table.iter().sum();
    let factor = prev_sum >> log_target;
    debug_assert!(log_target < 30);
    if factor <= 1 {
        return prev_sum;
    }
    downscale_stats(table, highbit32(factor), true)
}

impl Stats {
    /// `ZSTD_setBasePrices`.
    fn set_base_prices<const OPT_LEVEL: u32>(&mut self) {
        self.lit_sum_base_price = weight::<OPT_LEVEL>(self.lit_sum);
        self.lit_length_sum_base_price = weight::<OPT_LEVEL>(self.lit_length_sum);
        self.match_length_sum_base_price = weight::<OPT_LEVEL>(self.match_length_sum);
        self.off_code_sum_base_price = weight::<OPT_LEVEL>(self.off_code_sum);
    }

    /// `ZSTD_rescaleFreqs(optPtr, src = block, srcSize, optLevel)`: on the
    /// first block (`litLengthSum == 0`) seed literal statistics from the
    /// block itself and the sequence symbols from baseline tables, else
    /// scale the accumulated statistics down as the next block's seed.
    fn rescale_freqs<const OPT_LEVEL: u32>(&mut self, block: &[u8]) {
        self.price_type = PriceType::Dynamic;

        if self.lit_length_sum == 0 {
            // no literals stats collected -> first block assumed -> init

            // heuristic: use pre-defined stats for too small inputs
            if block.len() <= ZSTD_PREDEF_THRESHOLD {
                self.price_type = PriceType::Predef;
            }

            // first block, no dictionary: base initial cost of literals on
            // direct frequency within src (HIST_count_simple)
            crate::huf::hist_count(&mut self.lit_freq, block);
            self.lit_sum = downscale_stats(&mut self.lit_freq, 8, false);

            const BASE_LL_FREQS: [u32; MAX_LL + 1] = [
                4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
            ];
            self.lit_length_freq = BASE_LL_FREQS;
            self.lit_length_sum = BASE_LL_FREQS.iter().sum();

            self.match_length_freq = [1; MAX_ML + 1];
            self.match_length_sum = (MAX_ML + 1) as u32;

            const BASE_OFC_FREQS: [u32; MAX_OFF + 1] = [
                6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1,
            ];
            self.off_code_freq = BASE_OFC_FREQS;
            self.off_code_sum = BASE_OFC_FREQS.iter().sum();
        } else {
            // new block : scale down accumulated statistics
            self.lit_sum = scale_stats(&mut self.lit_freq, 12);
            self.lit_length_sum = scale_stats(&mut self.lit_length_freq, 11);
            self.match_length_sum = scale_stats(&mut self.match_length_freq, 11);
            self.off_code_sum = scale_stats(&mut self.off_code_freq, 11);
        }

        self.set_base_prices::<OPT_LEVEL>();
    }

    /// `LIT_PRICE(p)`: `ZSTD_rawLiteralsCost(p, 1)`, the price of the one
    /// literal `lit`, without its literal-length symbol.
    #[inline(always)]
    fn lit_price<const OPT_LEVEL: u32>(&self, lit: u8) -> i32 {
        if self.price_type == PriceType::Predef {
            return (6 * BITCOST_MULTIPLIER) as i32; // 6 bit per literal - no statistic used
        }
        // dynamic statistics
        let lit_price_max = self.lit_sum_base_price - BITCOST_MULTIPLIER;
        debug_assert!(self.lit_sum_base_price >= BITCOST_MULTIPLIER);
        let lit_price = weight::<OPT_LEVEL>(self.lit_freq[lit as usize]).min(lit_price_max);
        (self.lit_sum_base_price - lit_price) as i32
    }

    /// `LL_PRICE(l)`: `ZSTD_litLengthPrice`, cost of the literal-length
    /// symbol.
    #[inline(always)]
    fn ll_price<const OPT_LEVEL: u32>(&self, lit_length: u32) -> i32 {
        debug_assert!(lit_length as usize <= ZSTD_BLOCKSIZE_MAX);
        if self.price_type == PriceType::Predef {
            return weight::<OPT_LEVEL>(lit_length) as i32;
        }
        // ZSTD_LLcode() can't compute litLength price for sizes >=
        // ZSTD_BLOCKSIZE_MAX because it isn't representable in the zstd
        // format. So instead just pretend it would cost 1 bit more than
        // ZSTD_BLOCKSIZE_MAX - 1. In such a case, the block would be all
        // literals.
        if lit_length as usize == ZSTD_BLOCKSIZE_MAX {
            return BITCOST_MULTIPLIER as i32
                + self.ll_price::<OPT_LEVEL>(ZSTD_BLOCKSIZE_MAX as u32 - 1);
        }
        // dynamic statistics
        let ll = ll_code(lit_length) as usize;
        (LL_BITS[ll] as u32 * BITCOST_MULTIPLIER + self.lit_length_sum_base_price
            - weight::<OPT_LEVEL>(self.lit_length_freq[ll])) as i32
    }

    /// `LL_INCPRICE(l)`.
    #[inline(always)]
    fn ll_inc_price<const OPT_LEVEL: u32>(&self, lit_length: u32) -> i32 {
        self.ll_price::<OPT_LEVEL>(lit_length) - self.ll_price::<OPT_LEVEL>(lit_length - 1)
    }

    /// `ZSTD_getMatchPrice`: cost of the match part (offset + match length)
    /// of a sequence. With `OPT_LEVEL < 2` long offsets get a handicap
    /// (decompression speed).
    #[inline(always)]
    fn match_price<const OPT_LEVEL: u32>(&self, off_base: u32, match_length: u32) -> u32 {
        let off_code = highbit32(off_base);
        debug_assert!(match_length as usize >= ZSTD_MINMATCH);
        let ml_base = match_length - ZSTD_MINMATCH as u32;

        if self.price_type == PriceType::Predef {
            // fixed scheme, does not use statistics
            return weight::<OPT_LEVEL>(ml_base) + (16 + off_code) * BITCOST_MULTIPLIER;
        }

        // dynamic statistics
        let mut price = off_code * BITCOST_MULTIPLIER
            + (self.off_code_sum_base_price
                - weight::<OPT_LEVEL>(self.off_code_freq[off_code as usize]));
        if OPT_LEVEL < 2 && off_code >= 20 {
            // handicap for long distance offsets, favor decompression speed
            price += (off_code - 19) * 2 * BITCOST_MULTIPLIER;
        }

        // match Length
        let ml = ml_code(ml_base) as usize;
        price += ML_BITS[ml] as u32 * BITCOST_MULTIPLIER
            + (self.match_length_sum_base_price - weight::<OPT_LEVEL>(self.match_length_freq[ml]));

        // heuristic : make matches a bit more costly to favor less sequences
        // -> faster decompression speed
        price + BITCOST_MULTIPLIER / 5
    }

    /// `ZSTD_updateStats`: account one stored sequence.
    #[inline(always)]
    fn update_stats(&mut self, literals: &[u8], off_base: u32, match_length: u32) {
        // literals
        for &l in literals {
            self.lit_freq[l as usize] += ZSTD_LITFREQ_ADD;
        }
        self.lit_sum += literals.len() as u32 * ZSTD_LITFREQ_ADD;

        // literal Length
        self.lit_length_freq[ll_code(literals.len() as u32) as usize] += 1;
        self.lit_length_sum += 1;

        // offset code : follows storeSeq() numeric representation
        let off_code = highbit32(off_base) as usize;
        debug_assert!(off_code <= MAX_OFF);
        self.off_code_freq[off_code] += 1;
        self.off_code_sum += 1;

        // match Length
        let ml = ml_code(match_length - ZSTD_MINMATCH as u32) as usize;
        self.match_length_freq[ml] += 1;
        self.match_length_sum += 1;
    }
}

impl OptState {
    pub fn new() -> Self {
        fn boxed<T: Copy + Default>() -> Box<[T; ZSTD_OPT_SIZE]> {
            vec![T::default(); ZSTD_OPT_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!("length is ZSTD_OPT_SIZE"))
        }
        OptState {
            stats: Stats {
                lit_freq: [0; MAX_LIT + 1],
                lit_length_freq: [0; MAX_LL + 1],
                match_length_freq: [0; MAX_ML + 1],
                off_code_freq: [0; MAX_OFF + 1],
                lit_sum: 0,
                lit_length_sum: 0,
                match_length_sum: 0,
                off_code_sum: 0,
                lit_sum_base_price: 0,
                lit_length_sum_base_price: 0,
                match_length_sum_base_price: 0,
                off_code_sum_base_price: 0,
                price_type: PriceType::Dynamic,
            },
            matches: boxed(),
            opt: boxed(),
        }
    }

    /// `ZSTD_invalidateMatchState`'s `opt.litLengthSum = 0`: the next block
    /// initializes fresh statistics.
    pub fn invalidate(&mut self) {
        self.stats.lit_length_sum = 0;
    }
}

impl Default for OptState {
    fn default() -> Self {
        Self::new()
    }
}

/// `ZSTD_getAllMatchesFn`.
type GetAllMatches = unsafe fn(
    &mut [Match; ZSTD_OPT_SIZE],
    &mut MatchState,
    &mut usize,
    &[u8],
    usize,
    usize,
    &[u32; 3],
    u32,
    u32,
) -> u32;

/// `ZSTD_btGetAllMatches_noDict_<mls>` with the 8-byte count.
///
/// # Safety
/// As [`bt_get_all_matches`].
#[allow(clippy::too_many_arguments)]
unsafe fn get_all_matches_scalar<const MLS: u32>(
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
    bt_get_all_matches::<Fallback, MLS>(
        Fallback::new(),
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

/// `ZSTD_btGetAllMatches_noDict_<mls>` compiled with AVX2, counting 32
/// bytes per step.
///
/// # Safety
/// As [`bt_get_all_matches`]; the CPU must support AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn get_all_matches_avx2<const MLS: u32>(
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
    bt_get_all_matches::<Avx2, MLS>(
        Avx2::new_unchecked(),
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

/// `ZSTD_selectBtGetAllMatches(ms, ZSTD_noDict)`: the finder for
/// `mls = BOUNDED(3, minMatch, 6)` and this CPU's SIMD level.
fn select_get_all_matches(min_match: u32, level: Level) -> GetAllMatches {
    let mls = min_match.clamp(3, 6);
    match level {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(_) => match mls {
            3 => get_all_matches_avx2::<3>,
            4 => get_all_matches_avx2::<4>,
            5 => get_all_matches_avx2::<5>,
            _ => get_all_matches_avx2::<6>,
        },
        _ => match mls {
            3 => get_all_matches_scalar::<3>,
            4 => get_all_matches_scalar::<4>,
            5 => get_all_matches_scalar::<5>,
            _ => get_all_matches_scalar::<6>,
        },
    }
}

/// `ZSTD_compressBlock_btopt` / `_btultra` / `_btultra2`: find the
/// sequences of `src[block]` with the optimal parser and store them into
/// `out`. Returns the anchor: the start of the trailing literals, which the
/// caller appends to `out.lits` (`ZSTD_storeLastLiterals`). `rep` is the
/// repeat-offset history on entry and is updated on exit.
pub fn compress_block(
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    assert_opt_bounds(ms, src, block.end);
    let get_all_matches = select_get_all_matches(ms.cparams.min_match, simd_level());
    let mut state = ms
        .opt
        .take()
        .expect("MatchState::reset allocates OptState for the opt strategies");
    let anchor = match ms.cparams.strategy {
        Strategy::BtOpt => opt_generic::<0>(ms, &mut state, src, block, rep, out, get_all_matches),
        Strategy::BtUltra => {
            opt_generic::<2>(ms, &mut state, src, block, rep, out, get_all_matches)
        }
        Strategy::BtUltra2 => {
            // 2-passes strategy: this strategy makes a first pass over
            // first block to collect statistics in order to seed next
            // round's statistics with it. This can only work if no data has
            // been previously loaded in tables, aka, no dictionary, no
            // prefix, no ldm preprocessing.
            if state.stats.lit_length_sum == 0 // first block
                && out.seqs.is_empty() // no ldm
                && block.start <= ms.window_low // start of frame, nothing loaded nor skipped
                && block.len() > ZSTD_PREDEF_THRESHOLD
            {
                init_stats_ultra(
                    ms,
                    &mut state,
                    src,
                    block.clone(),
                    rep,
                    out,
                    get_all_matches,
                );
            }
            opt_generic::<2>(ms, &mut state, src, block, rep, out, get_all_matches)
        }
        s => unreachable!("opt::compress_block called for {s:?}"),
    };
    ms.opt = Some(state);
    anchor
}

/// `ZSTD_initStats_ultra`: a first compression pass over the first block,
/// only to seed the statistics with more accurate starting values; its
/// sequences and repcodes are dropped.
///
/// libzstd then forgets the pass by moving the window past it (`base -=
/// srcSize`, `dictLimit` and `lowLimit` up by `srcSize`, `nextToUpdate =
/// dictLimit`), which leaves every index the pass inserted below the
/// window. Here positions stay put, so the pass is forgotten instead by
/// emptying the hash buckets it filled (the tables were empty before it)
/// and restoring `next_to_update`. Tree nodes need no clearing: a tree walk
/// only enters a node through a hash bucket or a child link written after
/// that node's own re-insertion, so pass-1 nodes are unreachable.
fn init_stats_ultra(
    ms: &mut MatchState,
    state: &mut OptState,
    src: &[u8],
    block: Range<usize>,
    rep: &[u32; 3],
    out: &mut SeqStore,
    get_all_matches: GetAllMatches,
) {
    let mut tmp_rep = *rep; // updated rep codes will sink here
    debug_assert!(state.stats.lit_length_sum == 0); // first block
    debug_assert!(out.seqs.is_empty()); // no ldm
    debug_assert_eq!(ms.next_to_update, ms.window_low); // no prefix
    let first_update = ms.next_to_update;

    // generate stats into ms.opt
    opt_generic::<2>(
        ms,
        state,
        src,
        block.clone(),
        &mut tmp_rep,
        out,
        get_all_matches,
    );

    // invalidate first scan from history, only keep entropy stats
    out.clear();
    // The pass inserted positions up to its `ilimit = iend - 8`.
    let inserted_end = (block.end + 1)
        .saturating_sub(HASH_READ_SIZE)
        .max(first_update);
    bt::clear_hash_buckets(ms, src, first_update..inserted_end);
    ms.next_to_update = first_update;
}

/// `ZSTD_compressBlock_opt_generic(ms, seqStore, rep, src, srcSize,
/// optLevel, ZSTD_noDict)` without LDM candidates.
fn opt_generic<const OPT_LEVEL: u32>(
    ms: &mut MatchState,
    state: &mut OptState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
    get_all_matches: GetAllMatches,
) -> usize {
    let OptState {
        stats,
        matches,
        opt,
    } = state;
    let istart = block.start;
    let iend = block.end;
    let mut anchor = istart;
    // Below `istart + 1` the loop condition is false anyway (`ip >= 1`).
    let ilimit = iend.saturating_sub(8);
    let cp = ms.cparams;

    let sufficient_len = cp.target_length.min(ZSTD_OPT_NUM as u32 - 1);
    let min_match: u32 = if cp.min_match == 3 { 3 } else { 4 };
    let mut next_to_update3 = ms.next_to_update;

    let mut last_stretch = Optimal::default();

    // init
    stats.rescale_freqs::<OPT_LEVEL>(&src[block.clone()]);
    // C: `ip += (ip == prefixStart)` with ip starting at the block start;
    // here positions below `window_low` do not exist (block 0).
    let prefix_lowest = ms.window_low;
    let mut ip = istart.max(prefix_lowest);
    if ip == prefix_lowest {
        ip += 1;
    }

    // Match Loop
    'series: while ip < ilimit {
        let (mut cur, last_pos) = 'forward: {
            // find first match
            let litlen = (ip - anchor) as u32;
            let ll0 = (litlen == 0) as u32;
            // SAFETY: `ip + 8 < iend <= src.len()`, `ip >= window_low`, the
            // tables passed `assert_opt_bounds` in `compress_block`.
            let nb_matches = unsafe {
                get_all_matches(
                    matches,
                    ms,
                    &mut next_to_update3,
                    src,
                    ip,
                    iend,
                    rep,
                    ll0,
                    min_match,
                )
            } as usize;
            if nb_matches == 0 {
                ip += 1;
                continue 'series;
            }

            // Match found: let's store this solution, and eventually find
            // more candidates. During this forward pass, @opt is used to
            // store stretches, defined as "a match followed by N literals".
            // Note how this is different from a Sequence, which is "N
            // literals followed by a match". Storing stretches allows us to
            // store different match predecessors for each literal position
            // part of a literals run.

            // initialize opt[0]
            opt[0].mlen = 0; // there are only literals so far
            opt[0].litlen = litlen;
            // No need to include the actual price of the literals before the
            // first match because it is static for the duration of the
            // forward pass, and is included in every subsequent price. But,
            // we include the literal length because the cost variation of
            // litlen depends on the value of litlen.
            opt[0].price = stats.ll_price::<OPT_LEVEL>(litlen);
            opt[0].rep = *rep;

            // large match -> immediate encoding
            let max_ml = matches[nb_matches - 1].len;
            let max_off_base = matches[nb_matches - 1].off;
            if max_ml > sufficient_len {
                last_stretch.litlen = 0;
                last_stretch.mlen = max_ml;
                last_stretch.off = max_off_base;
                break 'forward (0, max_ml);
            }

            // set prices for first matches starting position == 0
            debug_assert!(opt[0].price >= 0);
            let mut pos = 1u32;
            while pos < min_match {
                let o = &mut opt[pos as usize];
                o.price = ZSTD_MAX_PRICE;
                o.mlen = 0;
                o.litlen = litlen + pos;
                pos += 1;
            }
            for m in &matches[..nb_matches] {
                let off_base = m.off;
                let end = m.len;
                while pos <= end {
                    let match_price = stats.match_price::<OPT_LEVEL>(off_base, pos) as i32;
                    let sequence_price = opt[0].price + match_price;
                    let o = &mut opt[pos as usize];
                    o.mlen = pos;
                    o.off = off_base;
                    o.litlen = 0; // end of match
                    o.price = sequence_price + stats.ll_price::<OPT_LEVEL>(0);
                    pos += 1;
                }
            }
            let mut last_pos = pos - 1;
            opt[pos as usize].price = ZSTD_MAX_PRICE;

            // check further positions
            let mut cur = 1u32;
            'further: while cur <= last_pos {
                'position: {
                    let c = cur as usize;
                    let inr = ip + c;
                    debug_assert!(c <= ZSTD_OPT_NUM);

                    // Fix current position with one literal if cheaper
                    {
                        let litlen = opt[c - 1].litlen + 1;
                        let price = opt[c - 1].price
                            + stats.lit_price::<OPT_LEVEL>(src[ip + c - 1])
                            + stats.ll_inc_price::<OPT_LEVEL>(litlen);
                        debug_assert!(price < 1_000_000_000); // overflow check
                        if price <= opt[c].price {
                            let prev_match = opt[c];
                            opt[c] = opt[c - 1];
                            opt[c].litlen = litlen;
                            opt[c].price = price;
                            if OPT_LEVEL >= 1 // additional check only for higher modes
                                && prev_match.litlen == 0 // replace a match
                                && stats.ll_inc_price::<OPT_LEVEL>(1) < 0 // ll1 is cheaper than ll0
                                && ip + c < iend
                            {
                                // check next position, in case it would be
                                // cheaper
                                let with1literal = prev_match.price
                                    + stats.lit_price::<OPT_LEVEL>(src[ip + c])
                                    + stats.ll_inc_price::<OPT_LEVEL>(1);
                                let with_more_literals = price
                                    + stats.lit_price::<OPT_LEVEL>(src[ip + c])
                                    + stats.ll_inc_price::<OPT_LEVEL>(litlen + 1);
                                if with1literal < with_more_literals
                                    && with1literal < opt[c + 1].price
                                {
                                    // update offset history - before it
                                    // disappears
                                    debug_assert!(cur >= prev_match.mlen);
                                    let prev = (cur - prev_match.mlen) as usize;
                                    let mut new_reps = opt[prev].rep;
                                    update_rep(
                                        &mut new_reps,
                                        prev_match.off,
                                        opt[prev].litlen == 0,
                                    );
                                    opt[c + 1] = prev_match; // mlen & offbase
                                    opt[c + 1].rep = new_reps;
                                    opt[c + 1].litlen = 1;
                                    opt[c + 1].price = with1literal;
                                    if last_pos < cur + 1 {
                                        last_pos = cur + 1;
                                    }
                                }
                            }
                        }
                    }

                    // Offset history is not updated during match comparison.
                    // Do it here, now that the match is selected and
                    // confirmed.
                    debug_assert!(cur >= opt[c].mlen);
                    if opt[c].litlen == 0 {
                        // just finished a match => alter offset history
                        let prev = (cur - opt[c].mlen) as usize;
                        let mut new_reps = opt[prev].rep;
                        update_rep(&mut new_reps, opt[c].off, opt[prev].litlen == 0);
                        opt[c].rep = new_reps;
                    }

                    // last match must start at a minimum distance of 8 from
                    // oend
                    if inr > ilimit {
                        break 'position;
                    }

                    if cur == last_pos {
                        break 'further;
                    }

                    if OPT_LEVEL == 0
                        && opt[c + 1].price <= opt[c].price + (BITCOST_MULTIPLIER / 2) as i32
                    {
                        // skip unpromising positions; about ~+6% speed, -0.01
                        // ratio
                        break 'position;
                    }

                    debug_assert!(opt[c].price >= 0);
                    let ll0 = (opt[c].litlen == 0) as u32;
                    let previous_price = opt[c].price;
                    let base_price = previous_price + stats.ll_price::<OPT_LEVEL>(0);
                    let cur_rep = opt[c].rep;
                    // SAFETY: `inr <= ilimit` so `inr + 8 <= iend`,
                    // `inr > ip >= window_low`.
                    let nb_matches = unsafe {
                        get_all_matches(
                            matches,
                            ms,
                            &mut next_to_update3,
                            src,
                            inr,
                            iend,
                            &cur_rep,
                            ll0,
                            min_match,
                        )
                    } as usize;

                    if nb_matches == 0 {
                        break 'position;
                    }

                    let longest_ml = matches[nb_matches - 1].len;
                    if longest_ml > sufficient_len
                        || cur + longest_ml >= ZSTD_OPT_NUM as u32
                        || inr + longest_ml as usize >= iend
                    {
                        last_stretch.mlen = longest_ml;
                        last_stretch.off = matches[nb_matches - 1].off;
                        last_stretch.litlen = 0;
                        break 'forward (cur, cur + longest_ml);
                    }

                    // set prices using matches found at position == cur
                    for match_nb in 0..nb_matches {
                        let offset = matches[match_nb].off;
                        let last_ml = matches[match_nb].len;
                        let start_ml = if match_nb > 0 {
                            matches[match_nb - 1].len + 1
                        } else {
                            min_match
                        };
                        let mut mlen = last_ml;
                        while mlen >= start_ml {
                            // scan downward
                            let pos = cur + mlen;
                            let price =
                                base_price + stats.match_price::<OPT_LEVEL>(offset, mlen) as i32;
                            if pos > last_pos || price < opt[pos as usize].price {
                                while last_pos < pos {
                                    // fill empty positions, for future
                                    // comparisons
                                    last_pos += 1;
                                    opt[last_pos as usize].price = ZSTD_MAX_PRICE;
                                    // just needs to be != 0, to mean "not an
                                    // end of match"
                                    opt[last_pos as usize].litlen = 1;
                                }
                                let o = &mut opt[pos as usize];
                                o.mlen = mlen;
                                o.off = offset;
                                o.litlen = 0;
                                o.price = price;
                            } else if OPT_LEVEL == 0 {
                                // early update abort; gets ~+10% speed for
                                // about -0.01 ratio loss
                                break;
                            }
                            mlen -= 1;
                        }
                    }
                    opt[last_pos as usize + 1].price = ZSTD_MAX_PRICE;
                }
                cur += 1;
            }

            last_stretch = opt[last_pos as usize];
            debug_assert!(last_pos >= last_stretch.mlen);
            (last_pos - last_stretch.mlen, last_pos)
        };

        // _shortestPath: cur, last_pos, last_stretch have to be set
        debug_assert!(opt[0].mlen == 0);
        debug_assert!(last_pos >= last_stretch.mlen);
        debug_assert!(cur == last_pos - last_stretch.mlen);

        if last_stretch.mlen == 0 {
            // no solution : all matches have been converted into literals
            debug_assert!(last_stretch.litlen as usize == (ip - anchor) + last_pos as usize);
            ip += last_pos as usize;
            continue;
        }
        debug_assert!(last_stretch.off > 0);

        // Update offset history
        if last_stretch.litlen == 0 {
            // finishing on a match : update offset history
            let mut reps = opt[cur as usize].rep;
            update_rep(&mut reps, last_stretch.off, opt[cur as usize].litlen == 0);
            *rep = reps;
        } else {
            *rep = last_stretch.rep;
            debug_assert!(cur >= last_stretch.litlen);
            cur -= last_stretch.litlen;
        }

        // Let's write the shortest path solution. It is stored in @opt in
        // reverse order, starting from @storeEnd (==cur+2), effectively
        // partially @opt overwriting. Content is changed too:
        // - So far, @opt stored stretches, aka a match followed by literals
        // - Now, it will store sequences, aka literals followed by a match
        let store_end = cur as usize + 2;
        let mut stretch_pos = cur as usize;
        debug_assert!(store_end < ZSTD_OPT_SIZE);
        if last_stretch.litlen > 0 {
            // last "sequence" is unfinished: just a bunch of literals
            opt[store_end].litlen = last_stretch.litlen;
            opt[store_end].mlen = 0;
            opt[store_end - 1] = last_stretch;
        }
        // libzstd 1.5.7 runs this unconditionally (a bare block after the
        // `if`), overwriting the literals-only entry above: trailing
        // literals of the last stretch are not stored and the next series
        // parses them again.
        opt[store_end] = last_stretch; // note: litlen will be fixed
        let mut store_start = store_end;
        loop {
            let next_stretch = opt[stretch_pos];
            opt[store_start].litlen = next_stretch.litlen;
            if next_stretch.mlen == 0 {
                // reaching beginning of segment
                break;
            }
            store_start -= 1;
            opt[store_start] = next_stretch; // note: litlen will be fixed
            debug_assert!((next_stretch.litlen + next_stretch.mlen) as usize <= stretch_pos);
            stretch_pos -= (next_stretch.litlen + next_stretch.mlen) as usize;
        }

        // save sequences
        for store_pos in store_start..=store_end {
            let llen = opt[store_pos].litlen as usize;
            let mlen = opt[store_pos].mlen;
            let off_base = opt[store_pos].off;
            let advance = llen + mlen as usize;

            if mlen == 0 {
                // only literals => must be last "sequence", actually starting
                // a new stream of sequences
                debug_assert!(store_pos == store_end); // must be last sequence
                ip = anchor + llen; // last "sequence" is a bunch of literals => don't progress anchor
                continue; // will finish
            }

            debug_assert!(anchor + llen <= iend);
            stats.update_stats(&src[anchor..anchor + llen], off_base, mlen);
            out.store_seq(src, anchor, llen, iend, off_base, mlen as usize);
            anchor += advance;
            ip = anchor;
        }

        // update all costs
        stats.set_base_prices::<OPT_LEVEL>();
    }

    // Return the last literals size
    anchor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::common::testutil::{roundtrip_blocks, roundtrip_job, Finder};
    use crate::compress::params::CParams;
    use crate::compress::{compress_with, CompressOptions, JOBSIZE_MIN};

    const OPT: Finder = Finder {
        compress_block,
        load_prefix: bt::load_prefix,
    };

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 56) as u8
            })
            .collect()
    }

    /// Text with repeats at every distance class, a noise gap, a long run
    /// (matches longer than `ZSTD_OPT_NUM`, the immediate-encoding path)
    /// and a far repeat of the start.
    fn corpus(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len + 64);
        let mut i = 0u32;
        while v.len() < len / 2 {
            v.extend_from_slice(format!("line {} of the corpus {}\n", i, i % 37).as_bytes());
            i += 1;
        }
        v.extend_from_slice(&noise(len / 8, 11));
        v.extend(std::iter::repeat_n(b'z', len / 8));
        let head = v[..len / 4].to_vec();
        v.extend_from_slice(&head);
        v.truncate(len);
        v
    }

    fn cparams(strategy: Strategy, min_match: u32, src_len: usize) -> CParams {
        let mut cp = CParams::for_level(19, src_len);
        cp.strategy = strategy;
        cp.min_match = min_match;
        cp
    }

    /// Every getAllMatches instantiation (`mls` 3..=6, `min_match` 7 bounded
    /// to 6) of every opt strategy: blocks reconstruct, offsets stay in the
    /// window, with and without a loaded prefix.
    #[test]
    fn every_strategy_and_min_match_reconstructs() {
        let data = corpus(400 << 10);
        for strategy in [Strategy::BtOpt, Strategy::BtUltra, Strategy::BtUltra2] {
            for min_match in 3..=7 {
                let cp = cparams(strategy, min_match, data.len());
                let s = roundtrip_blocks(&OPT, &data, cp, 1 << 17, 1, [1, 4, 8]);
                assert!(s.seqs > 1000, "{strategy:?} mm{min_match}: {s:?}");
                assert!(s.cross_block_matches > 0, "{strategy:?} mm{min_match}");
                let s = roundtrip_job(&OPT, &data, cp, 1 << 17, 5000, 200 << 10, [1, 4, 8]);
                // With min_match 7 libzstd loads the prefix with 7-byte
                // hashes (`ZSTD_updateTree` passes `minMatch` unbounded) but
                // the parser looks up 6-byte ones: the prefix is reachable
                // only through hash collisions.
                assert_eq!(
                    s.prefix_matches > 0,
                    min_match < 7,
                    "{strategy:?} mm{min_match}: {s:?}"
                );
            }
        }
    }

    /// A window smaller than the input: matches never reach past it
    /// (`ZSTD_getLowestMatchIndex`), for the tree, the repcodes and hash3.
    #[test]
    fn small_window_bounds_every_match() {
        let data = corpus(700 << 10);
        for strategy in [Strategy::BtOpt, Strategy::BtUltra2] {
            let mut cp = cparams(strategy, 3, data.len());
            cp.window_log = 17;
            roundtrip_blocks(&OPT, &data, cp, 1 << 17, 1, [1, 4, 8]);
        }
    }

    /// btultra2's statistics pass leaves no trace in the match state: every
    /// hash bucket (and hash3 bucket) it filled is empty again and
    /// `next_to_update` is back at the block start, while the statistics it
    /// collected are kept.
    #[test]
    fn ultra_first_pass_is_forgotten() {
        // Random 16-letter text ending in a repeat that spans the pass's
        // `ilimit = end - 8` and then 5 bytes found nowhere else: the pass
        // must insert `ilimit`, the last position it can insert, whose
        // hashes cover a unique window (so no other cleared position
        // empties its bucket by collision).
        let mut letters: Vec<u8> = noise(1 << 17, 5).iter().map(|b| b'a' + (b & 15)).collect();
        let n = letters.len();
        letters.copy_within(1000..1008, n - 13);
        letters[n - 5..].copy_from_slice(b"QRSTU");
        for (data, min_match) in (3..=6)
            .map(|mm| (corpus(1 << 17), mm))
            .chain((3..=6).map(|mm| (letters.clone(), mm)))
        {
            let cp = cparams(Strategy::BtUltra2, min_match, data.len());
            let mut ms = MatchState::new(cp, 1);
            let mut state = ms.opt.take().unwrap();
            let mut out = SeqStore::new();
            let get_all_matches = select_get_all_matches(cp.min_match, simd_level());
            init_stats_ultra(
                &mut ms,
                &mut state,
                &data,
                0..data.len(),
                &[1, 4, 8],
                &mut out,
                get_all_matches,
            );
            assert!(out.seqs.is_empty() && out.lits.is_empty());
            assert_eq!(ms.next_to_update, 1, "mm{min_match}");
            assert!(state.stats.lit_length_sum > 0, "mm{min_match}");
            let (hash, _, hash3) = ms.ws.opt_tables_mut();
            assert!(hash.iter().all(|&e| e == 0), "mm{min_match}: hash");
            assert_eq!(hash3.is_empty(), min_match != 3);
            assert!(hash3.iter().all(|&e| e == 0), "mm{min_match}: hash3");
        }
    }

    /// L16..L22 frames, single and multi-job, with the default overlap and
    /// with none (every job then runs btultra2's first pass), decode through
    /// both decoders; tiny inputs too (predefined prices).
    #[test]
    fn opt_levels_roundtrip_both_decoders() {
        let small = corpus(200 << 10);
        let big = corpus(1100 << 10);
        let check = |data: &[u8], opts: &CompressOptions| {
            let frame = compress_with(data, opts);
            assert_eq!(crate::decompress(&frame).unwrap(), data, "{opts:?}");
            let theirs = zstd::stream::decode_all(&frame[..]).unwrap();
            assert_eq!(theirs, data, "{opts:?}");
        };
        for level in 16..=22 {
            check(
                &small,
                &CompressOptions {
                    level,
                    ..Default::default()
                },
            );
        }
        for level in [16, 19] {
            for overlap_log in [0, 1] {
                check(
                    &big,
                    &CompressOptions {
                        level,
                        job_size: Some(JOBSIZE_MIN),
                        overlap_log,
                    },
                );
            }
        }
        for len in [0, 1, 7, 8, 9, 20, 64] {
            for level in [16, 18, 19] {
                check(
                    &small[..len],
                    &CompressOptions {
                        level,
                        ..Default::default()
                    },
                );
            }
        }
    }
}
