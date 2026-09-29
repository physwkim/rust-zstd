//! Compression parameters.
//!
//! Port of `ZSTD_defaultCParameters` (clevels.h), `ZSTD_getCParams_internal`
//! and `ZSTD_adjustCParams_internal` (zstd_compress.c) for the
//! no-dictionary case (`dictSize == 0`, `ZSTD_cpm_noAttachDict`).

/// Match-finder strategy. Numeric values follow `ZSTD_strategy`.
///
/// libzstd's `ZSTD_btlazy2`, `ZSTD_btopt`, `ZSTD_btultra` and
/// `ZSTD_btultra2` are not ported: levels whose table row selects one of
/// them keep that row's numeric parameters but run with `Lazy2`, exactly as
/// libzstd built with `ZSTD_EXCLUDE_BTLAZY2_BLOCK_COMPRESSOR` (and the
/// btopt/btultra exclusions) cascades them in `ZSTD_adjustCParams_internal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strategy {
    Fast = 1,
    DFast = 2,
    Greedy = 3,
    Lazy = 4,
    Lazy2 = 5,
}

/// `ZSTD_compressionParameters`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CParams {
    pub window_log: u32,
    pub chain_log: u32,
    pub hash_log: u32,
    pub search_log: u32,
    pub min_match: u32,
    pub target_length: u32,
    pub strategy: Strategy,
}

/// `ZSTD_MAX_CLEVEL`.
pub const ZSTD_MAX_CLEVEL: i32 = 22;
/// `ZSTD_CLEVEL_DEFAULT`: level 0 selects this row.
pub const ZSTD_CLEVEL_DEFAULT: i32 = 3;
/// `ZSTD_WINDOWLOG_MAX` on 64-bit hosts.
pub const ZSTD_WINDOWLOG_MAX: u32 = 31;
/// `ZSTD_WINDOWLOG_ABSOLUTEMIN`.
pub const ZSTD_WINDOWLOG_ABSOLUTEMIN: u32 = 10;
const ZSTD_HASHLOG_MIN: u32 = 6;
const ZSTD_TARGETLENGTH_MAX: i32 = 1 << 17;
const ZSTD_ROW_HASH_TAG_BITS: u32 = 8;

/// Strategy column of `ZSTD_defaultCParameters`, including the unported
/// binary-tree strategies so the table below is a verbatim copy.
#[derive(Clone, Copy)]
enum Strat {
    Fast,
    DFast,
    Greedy,
    Lazy,
    Lazy2,
    BtLazy2,
    BtOpt,
    BtUltra,
    BtUltra2,
}

impl Strat {
    /// Cascade to the strongest ported strategy (see [`Strategy`]).
    fn ported(self) -> Strategy {
        match self {
            Strat::Fast => Strategy::Fast,
            Strat::DFast => Strategy::DFast,
            Strat::Greedy => Strategy::Greedy,
            Strat::Lazy => Strategy::Lazy,
            Strat::Lazy2 | Strat::BtLazy2 | Strat::BtOpt | Strat::BtUltra | Strat::BtUltra2 => {
                Strategy::Lazy2
            }
        }
    }
}

type Row = (u32, u32, u32, u32, u32, u32, Strat);

use Strat::*;

/// `ZSTD_defaultCParameters[4][ZSTD_MAX_CLEVEL+1]`: (W, C, H, S, L, TL, strat).
#[rustfmt::skip]
static DEFAULT_CPARAMETERS: [[Row; ZSTD_MAX_CLEVEL as usize + 1]; 4] = [
    [   /* "default" - for any srcSize > 256 KB */
        (19, 12, 13,  1,  6,  1, Fast    ),  /* base for negative levels */
        (19, 13, 14,  1,  7,  0, Fast    ),  /* level  1 */
        (20, 15, 16,  1,  6,  0, Fast    ),  /* level  2 */
        (21, 16, 17,  1,  5,  0, DFast   ),  /* level  3 */
        (21, 18, 18,  1,  5,  0, DFast   ),  /* level  4 */
        (21, 18, 19,  3,  5,  2, Greedy  ),  /* level  5 */
        (21, 18, 19,  3,  5,  4, Lazy    ),  /* level  6 */
        (21, 19, 20,  4,  5,  8, Lazy    ),  /* level  7 */
        (21, 19, 20,  4,  5, 16, Lazy2   ),  /* level  8 */
        (22, 20, 21,  4,  5, 16, Lazy2   ),  /* level  9 */
        (22, 21, 22,  5,  5, 16, Lazy2   ),  /* level 10 */
        (22, 21, 22,  6,  5, 16, Lazy2   ),  /* level 11 */
        (22, 22, 23,  6,  5, 32, Lazy2   ),  /* level 12 */
        (22, 22, 22,  4,  5, 32, BtLazy2 ),  /* level 13 */
        (22, 22, 23,  5,  5, 32, BtLazy2 ),  /* level 14 */
        (22, 23, 23,  6,  5, 32, BtLazy2 ),  /* level 15 */
        (22, 22, 22,  5,  5, 48, BtOpt   ),  /* level 16 */
        (23, 23, 22,  5,  4, 64, BtOpt   ),  /* level 17 */
        (23, 23, 22,  6,  3, 64, BtUltra ),  /* level 18 */
        (23, 24, 22,  7,  3,256, BtUltra2),  /* level 19 */
        (25, 25, 23,  7,  3,256, BtUltra2),  /* level 20 */
        (26, 26, 24,  7,  3,512, BtUltra2),  /* level 21 */
        (27, 27, 25,  9,  3,999, BtUltra2),  /* level 22 */
    ],
    [   /* for srcSize <= 256 KB */
        (18, 12, 13,  1,  5,  1, Fast    ),  /* base for negative levels */
        (18, 13, 14,  1,  6,  0, Fast    ),  /* level  1 */
        (18, 14, 14,  1,  5,  0, DFast   ),  /* level  2 */
        (18, 16, 16,  1,  4,  0, DFast   ),  /* level  3 */
        (18, 16, 17,  3,  5,  2, Greedy  ),  /* level  4.*/
        (18, 17, 18,  5,  5,  2, Greedy  ),  /* level  5.*/
        (18, 18, 19,  3,  5,  4, Lazy    ),  /* level  6.*/
        (18, 18, 19,  4,  4,  4, Lazy    ),  /* level  7 */
        (18, 18, 19,  4,  4,  8, Lazy2   ),  /* level  8 */
        (18, 18, 19,  5,  4,  8, Lazy2   ),  /* level  9 */
        (18, 18, 19,  6,  4,  8, Lazy2   ),  /* level 10 */
        (18, 18, 19,  5,  4, 12, BtLazy2 ),  /* level 11.*/
        (18, 19, 19,  7,  4, 12, BtLazy2 ),  /* level 12.*/
        (18, 18, 19,  4,  4, 16, BtOpt   ),  /* level 13 */
        (18, 18, 19,  4,  3, 32, BtOpt   ),  /* level 14.*/
        (18, 18, 19,  6,  3,128, BtOpt   ),  /* level 15.*/
        (18, 19, 19,  6,  3,128, BtUltra ),  /* level 16.*/
        (18, 19, 19,  8,  3,256, BtUltra ),  /* level 17.*/
        (18, 19, 19,  6,  3,128, BtUltra2),  /* level 18.*/
        (18, 19, 19,  8,  3,256, BtUltra2),  /* level 19.*/
        (18, 19, 19, 10,  3,512, BtUltra2),  /* level 20.*/
        (18, 19, 19, 12,  3,512, BtUltra2),  /* level 21.*/
        (18, 19, 19, 13,  3,999, BtUltra2),  /* level 22.*/
    ],
    [   /* for srcSize <= 128 KB */
        (17, 12, 12,  1,  5,  1, Fast    ),  /* base for negative levels */
        (17, 12, 13,  1,  6,  0, Fast    ),  /* level  1 */
        (17, 13, 15,  1,  5,  0, Fast    ),  /* level  2 */
        (17, 15, 16,  2,  5,  0, DFast   ),  /* level  3 */
        (17, 17, 17,  2,  4,  0, DFast   ),  /* level  4 */
        (17, 16, 17,  3,  4,  2, Greedy  ),  /* level  5 */
        (17, 16, 17,  3,  4,  4, Lazy    ),  /* level  6 */
        (17, 16, 17,  3,  4,  8, Lazy2   ),  /* level  7 */
        (17, 16, 17,  4,  4,  8, Lazy2   ),  /* level  8 */
        (17, 16, 17,  5,  4,  8, Lazy2   ),  /* level  9 */
        (17, 16, 17,  6,  4,  8, Lazy2   ),  /* level 10 */
        (17, 17, 17,  5,  4,  8, BtLazy2 ),  /* level 11 */
        (17, 18, 17,  7,  4, 12, BtLazy2 ),  /* level 12 */
        (17, 18, 17,  3,  4, 12, BtOpt   ),  /* level 13.*/
        (17, 18, 17,  4,  3, 32, BtOpt   ),  /* level 14.*/
        (17, 18, 17,  6,  3,256, BtOpt   ),  /* level 15.*/
        (17, 18, 17,  6,  3,128, BtUltra ),  /* level 16.*/
        (17, 18, 17,  8,  3,256, BtUltra ),  /* level 17.*/
        (17, 18, 17, 10,  3,512, BtUltra ),  /* level 18.*/
        (17, 18, 17,  5,  3,256, BtUltra2),  /* level 19.*/
        (17, 18, 17,  7,  3,512, BtUltra2),  /* level 20.*/
        (17, 18, 17,  9,  3,512, BtUltra2),  /* level 21.*/
        (17, 18, 17, 11,  3,999, BtUltra2),  /* level 22.*/
    ],
    [   /* for srcSize <= 16 KB */
        (14, 12, 13,  1,  5,  1, Fast    ),  /* base for negative levels */
        (14, 14, 15,  1,  5,  0, Fast    ),  /* level  1 */
        (14, 14, 15,  1,  4,  0, Fast    ),  /* level  2 */
        (14, 14, 15,  2,  4,  0, DFast   ),  /* level  3 */
        (14, 14, 14,  4,  4,  2, Greedy  ),  /* level  4 */
        (14, 14, 14,  3,  4,  4, Lazy    ),  /* level  5.*/
        (14, 14, 14,  4,  4,  8, Lazy2   ),  /* level  6 */
        (14, 14, 14,  6,  4,  8, Lazy2   ),  /* level  7 */
        (14, 14, 14,  8,  4,  8, Lazy2   ),  /* level  8.*/
        (14, 15, 14,  5,  4,  8, BtLazy2 ),  /* level  9.*/
        (14, 15, 14,  9,  4,  8, BtLazy2 ),  /* level 10.*/
        (14, 15, 14,  3,  4, 12, BtOpt   ),  /* level 11.*/
        (14, 15, 14,  4,  3, 24, BtOpt   ),  /* level 12.*/
        (14, 15, 14,  5,  3, 32, BtUltra ),  /* level 13.*/
        (14, 15, 15,  6,  3, 64, BtUltra ),  /* level 14.*/
        (14, 15, 15,  7,  3,256, BtUltra ),  /* level 15.*/
        (14, 15, 15,  5,  3, 48, BtUltra2),  /* level 16.*/
        (14, 15, 15,  6,  3,128, BtUltra2),  /* level 17.*/
        (14, 15, 15,  7,  3,256, BtUltra2),  /* level 18.*/
        (14, 15, 15,  8,  3,256, BtUltra2),  /* level 19.*/
        (14, 15, 15,  8,  3,512, BtUltra2),  /* level 20.*/
        (14, 15, 15,  9,  3,512, BtUltra2),  /* level 21.*/
        (14, 15, 15, 10,  3,999, BtUltra2),  /* level 22.*/
    ],
];

fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

impl CParams {
    /// `ZSTD_getCParams_internal(level, src_size, 0, ZSTD_cpm_noAttachDict)`:
    /// pick the table row by `src_size` (`ZSTD_getCParamRowSize`) and level,
    /// then refine with [`CParams::adjust`]. `level == 0` selects
    /// [`ZSTD_CLEVEL_DEFAULT`]; negative levels use row 0 with
    /// `target_length = -level` (acceleration factor).
    pub fn for_level(level: i32, src_size: usize) -> CParams {
        let r_size = src_size as u64;
        let table_id = (r_size <= 256 << 10) as usize
            + (r_size <= 128 << 10) as usize
            + (r_size <= 16 << 10) as usize;
        let row = if level == 0 {
            ZSTD_CLEVEL_DEFAULT
        } else if level < 0 {
            0
        } else {
            level.min(ZSTD_MAX_CLEVEL)
        } as usize;
        let (w, c, h, s, l, tl, strat) = DEFAULT_CPARAMETERS[table_id][row];
        let mut cp = CParams {
            window_log: w,
            chain_log: c,
            hash_log: h,
            search_log: s,
            min_match: l,
            target_length: tl,
            strategy: strat.ported(),
        };
        if level < 0 {
            // ZSTD_minCLevel() == -ZSTD_TARGETLENGTH_MAX
            let clamped = level.max(-ZSTD_TARGETLENGTH_MAX);
            cp.target_length = (-clamped) as u32;
        }
        cp.adjust(src_size)
    }

    /// `ZSTD_adjustCParams_internal(cp, src_size, 0, ZSTD_cpm_noAttachDict,
    /// ZSTD_ps_auto)`: shrink window/hash/chain logs for small inputs.
    pub fn adjust(mut self, src_size: usize) -> CParams {
        let src_size = src_size as u64;
        let max_window_resize: u64 = 1 << (ZSTD_WINDOWLOG_MAX - 1);

        // resize windowLog if input is small enough, to use less memory
        if src_size <= max_window_resize {
            let t_size = src_size as u32;
            let hash_size_min = 1u32 << ZSTD_HASHLOG_MIN;
            let src_log = if t_size < hash_size_min {
                ZSTD_HASHLOG_MIN
            } else {
                highbit32(t_size - 1) + 1
            };
            if self.window_log > src_log {
                self.window_log = src_log;
            }
        }
        {
            // dictSize == 0: ZSTD_dictAndWindowLog() returns windowLog unchanged.
            let dict_and_window_log = self.window_log;
            // ZSTD_cycleLog(): btScale is 0 for every ported strategy.
            let cycle_log = self.chain_log;
            if self.hash_log > dict_and_window_log + 1 {
                self.hash_log = dict_and_window_log + 1;
            }
            if cycle_log > dict_and_window_log {
                self.chain_log -= cycle_log - dict_and_window_log;
            }
        }
        if self.window_log < ZSTD_WINDOWLOG_ABSOLUTEMIN {
            self.window_log = ZSTD_WINDOWLOG_ABSOLUTEMIN;
        }
        // ZSTD_ps_auto is resolved to ZSTD_ps_enable here; the row matchfinder
        // cannot hash more than 32 bits in total.
        if self.row_match_finder_supported() {
            let row_log = self.search_log.clamp(4, 6);
            let max_row_hash_log = 32 - ZSTD_ROW_HASH_TAG_BITS;
            let max_hash_log = max_row_hash_log + row_log;
            if self.hash_log > max_hash_log {
                self.hash_log = max_hash_log;
            }
        }
        self
    }

    /// `ZSTD_rowMatchFinderSupported`.
    pub fn row_match_finder_supported(&self) -> bool {
        matches!(
            self.strategy,
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2
        )
    }

    /// `ZSTD_minGain(src_size, strategy)`: minimum saving required to emit a
    /// compressed block or a compressed literals section.
    pub fn min_gain(src_size: usize, strategy: Strategy) -> usize {
        // minlog = (strat >= ZSTD_btultra) ? strat - 1 : 6; no bt strategies here.
        let _ = strategy;
        (src_size >> 6) + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level1_large_input_matches_c_table_row() {
        let cp = CParams::for_level(1, 8 << 20);
        assert_eq!(cp.window_log, 19);
        assert_eq!(cp.chain_log, 13);
        assert_eq!(cp.hash_log, 14);
        assert_eq!(cp.min_match, 7);
        assert_eq!(cp.strategy, Strategy::Fast);
    }

    #[test]
    fn small_input_shrinks_window_hash_and_chain() {
        // 1000 bytes: srcLog = highbit32(999)+1 = 10 -> windowLog 10,
        // hashLog <= 11, chainLog <= 10.
        let cp = CParams::for_level(3, 1000);
        assert_eq!(cp.window_log, 10);
        assert_eq!(cp.hash_log, 11);
        assert_eq!(cp.chain_log, 10);
        assert_eq!(cp.strategy, Strategy::DFast);
        // 40 bytes: tSize < 64 -> srcLog = 6 -> windowLog clamped up to 10.
        let cp = CParams::for_level(3, 40);
        assert_eq!(cp.window_log, 10);
        assert_eq!(cp.hash_log, 7);
        assert_eq!(cp.chain_log, 6);
    }

    #[test]
    fn bt_strategies_cascade_to_lazy2() {
        let cp = CParams::for_level(19, 8 << 20);
        assert_eq!(cp.strategy, Strategy::Lazy2);
        assert_eq!(cp.window_log, 23);
        assert_eq!(cp.hash_log, 22);
        assert_eq!(cp.search_log, 7);
    }

    #[test]
    fn level0_and_negative_levels() {
        assert_eq!(
            CParams::for_level(0, 1 << 20),
            CParams::for_level(3, 1 << 20)
        );
        let cp = CParams::for_level(-5, 1 << 20);
        assert_eq!(cp.strategy, Strategy::Fast);
        assert_eq!(cp.target_length, 5);
    }
}
