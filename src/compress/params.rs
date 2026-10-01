//! Compression parameters.
//!
//! Port of `ZSTD_defaultCParameters` (clevels.h), `ZSTD_getCParams_internal`
//! and `ZSTD_adjustCParams_internal` (zstd_compress.c) for the
//! no-dictionary case (`dictSize == 0`, `ZSTD_cpm_noAttachDict`).

use crate::constants::{ZSTD_HASHLOG_MIN, ZSTD_WINDOWLOG_MAX};

/// Match-finder strategy. Numeric values follow `ZSTD_strategy`.
///
/// Every `ZSTD_strategy` is ported; a level's table row selects the
/// strategy it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strategy {
    Fast = 1,
    DFast = 2,
    Greedy = 3,
    Lazy = 4,
    Lazy2 = 5,
    BtLazy2 = 6,
    BtOpt = 7,
    BtUltra = 8,
    BtUltra2 = 9,
}

impl Strategy {
    /// `ZSTD_cycleLog`'s `btScale`: the binary-tree strategies keep two
    /// chain-table entries per position.
    pub fn bt_scale(self) -> u32 {
        match self {
            Strategy::Fast
            | Strategy::DFast
            | Strategy::Greedy
            | Strategy::Lazy
            | Strategy::Lazy2 => 0,
            Strategy::BtLazy2 | Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => 1,
        }
    }

    /// Runs the optimal parser (`ZSTD_compressBlock_btopt` and up).
    pub fn is_opt(self) -> bool {
        match self {
            Strategy::Fast
            | Strategy::DFast
            | Strategy::Greedy
            | Strategy::Lazy
            | Strategy::Lazy2
            | Strategy::BtLazy2 => false,
            Strategy::BtOpt | Strategy::BtUltra | Strategy::BtUltra2 => true,
        }
    }
}

/// `ZSTD_ParamSwitch_e`: a feature left to libzstd's default for the
/// parameters (`Auto`), or forced on or off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParamSwitch {
    #[default]
    Auto,
    Enable,
    Disable,
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
/// `ZSTD_WINDOWLOG_ABSOLUTEMIN`.
pub const ZSTD_WINDOWLOG_ABSOLUTEMIN: u32 = 10;
const ZSTD_TARGETLENGTH_MAX: i32 = 1 << 17;
const ZSTD_ROW_HASH_TAG_BITS: u32 = 8;
/// `ZSTD_HASHLOG3_MAX`: the largest 3-byte hash table, see
/// [`CParams::hash_log3`].
pub const ZSTD_HASHLOG3_MAX: u32 = 17;

/// Strategy column of `ZSTD_defaultCParameters`, including the unported
/// `BtLazy2` so the table below is a verbatim copy.
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
            Strat::Lazy2 => Strategy::Lazy2,
            Strat::BtLazy2 => Strategy::BtLazy2,
            Strat::BtOpt => Strategy::BtOpt,
            Strat::BtUltra => Strategy::BtUltra,
            Strat::BtUltra2 => Strategy::BtUltra2,
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
            // ZSTD_cycleLog(): the binary tree holds two entries per position.
            let cycle_log = self.chain_log - self.strategy.bt_scale();
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
        match self.strategy {
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => true,
            Strategy::Fast
            | Strategy::DFast
            | Strategy::BtLazy2
            | Strategy::BtOpt
            | Strategy::BtUltra
            | Strategy::BtUltra2 => false,
        }
    }

    /// `hashLog3` of `ZSTD_reset_matchState`: the optimal parser's 3-byte
    /// hash table exists only for `min_match == 3`, with
    /// `min(ZSTD_HASHLOG3_MAX, window_log)` bits; `0` means no table.
    pub fn hash_log3(&self) -> u32 {
        if self.strategy.is_opt() && self.min_match == 3 {
            ZSTD_HASHLOG3_MAX.min(self.window_log)
        } else {
            0
        }
    }

    /// `ZSTD_minGain(src_size, strategy)`: minimum saving required to emit a
    /// compressed block or a compressed literals section.
    pub fn min_gain(src_size: usize, strategy: Strategy) -> usize {
        let min_log = if strategy >= Strategy::BtUltra {
            strategy as u32 - 1
        } else {
            6
        };
        (src_size >> min_log) + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window and hash log limits are libzstd's for this target, a
    /// level clamps at `ZSTD_c_compressionLevel`'s bounds (as
    /// `ZSTD_CCtx_setParameter` clamps it), and every level's parameters lie
    /// within `ZSTD_cParam_getBounds` for any input size.
    #[test]
    fn cparam_bounds_match_libzstd() {
        use crate::compress::common::testutil::c_bounds;
        use crate::constants::ZSTD_HASHLOG_MAX;
        use zstd::zstd_safe::zstd_sys::ZSTD_cParameter::*;

        assert_eq!(c_bounds(ZSTD_c_windowLog).1, ZSTD_WINDOWLOG_MAX as i32);
        assert_eq!(
            c_bounds(ZSTD_c_hashLog),
            (ZSTD_HASHLOG_MIN as i32, ZSTD_HASHLOG_MAX as i32)
        );
        let sizes = [
            0,
            1000,
            16 << 10,
            (128 << 10) + 1,
            (256 << 10) + 1,
            1 << 20,
            1 << 29,
            (1 << 29) + 1,
            usize::MAX,
        ];
        let (lo, hi) = c_bounds(ZSTD_c_compressionLevel);
        for size in sizes {
            let cp = |level| CParams::for_level(level, size);
            for (level, bound) in [(i32::MIN, lo), (lo - 1, lo), (hi + 1, hi), (i32::MAX, hi)] {
                assert_eq!(cp(level), cp(bound), "level {level} size {size}");
            }
            assert_ne!(cp(lo), cp(lo + 1), "size {size}");
        }
        assert_ne!(
            CParams::for_level(hi, 1 << 20),
            CParams::for_level(hi - 1, 1 << 20)
        );
        let bounds = [
            ZSTD_c_windowLog,
            ZSTD_c_chainLog,
            ZSTD_c_hashLog,
            ZSTD_c_searchLog,
            ZSTD_c_minMatch,
            ZSTD_c_targetLength,
            ZSTD_c_strategy,
        ]
        .map(|param| (param, c_bounds(param)));
        for level in lo..=hi {
            for size in sizes {
                let cp = CParams::for_level(level, size);
                let values = [
                    cp.window_log,
                    cp.chain_log,
                    cp.hash_log,
                    cp.search_log,
                    cp.min_match,
                    cp.target_length,
                    cp.strategy as u32,
                ];
                for ((param, (min, max)), v) in bounds.into_iter().zip(values) {
                    assert!(
                        (min..=max).contains(&(v as i32)),
                        "level {level} size {size}: {param:?} {v}"
                    );
                }
            }
        }
    }

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
    fn btlazy2_rows_select_btlazy2_with_cycle_log() {
        // Level 13, > 256 KiB row: (22, 22, 22, 4, 5, 32, btlazy2).
        let cp = CParams::for_level(13, 8 << 20);
        assert_eq!(cp.strategy, Strategy::BtLazy2);
        assert_eq!((cp.window_log, cp.chain_log, cp.hash_log), (22, 22, 22));
        // 100 KB, level 12 of the <= 128 KiB row: (17, 18, 17, 7, 4, 12,
        // btlazy2), windowLog 17; cycleLog = chainLog - 1 = 17 fits.
        let cp = CParams::for_level(12, 100_000);
        assert_eq!(cp.strategy, Strategy::BtLazy2);
        assert_eq!((cp.window_log, cp.chain_log), (17, 18));
        // 20 KB: windowLog 15, so the tree (cycleLog 17) shrinks by 2.
        let cp = CParams::for_level(12, 20_000);
        assert_eq!((cp.window_log, cp.chain_log, cp.hash_log), (15, 16, 16));
        // Lazy2 (level 10: chainLog 16) has no btScale: capped at windowLog.
        let cp = CParams::for_level(10, 20_000);
        assert_eq!(cp.strategy, Strategy::Lazy2);
        assert_eq!(cp.chain_log, 15);
    }

    #[test]
    fn opt_levels_select_the_bt_strategies() {
        let strat = |level| CParams::for_level(level, 8 << 20).strategy;
        assert_eq!(strat(16), Strategy::BtOpt);
        assert_eq!(strat(17), Strategy::BtOpt);
        assert_eq!(strat(18), Strategy::BtUltra);
        for level in 19..=22 {
            assert_eq!(strat(level), Strategy::BtUltra2);
        }
        let cp = CParams::for_level(19, 8 << 20);
        assert_eq!(
            (cp.window_log, cp.chain_log, cp.hash_log, cp.search_log),
            (23, 24, 22, 7)
        );
        // Level 22 on 8 MiB: windowLog 23, hashLog 27 -> 24, and the tree's
        // cycleLog 27 - 1 = 26 shrinks chainLog by 3 to 24.
        let cp = CParams::for_level(22, 8 << 20);
        assert_eq!((cp.window_log, cp.chain_log, cp.hash_log), (23, 24, 24));
        assert_eq!(cp.hash_log3(), 17);
        assert_eq!(CParams::for_level(17, 8 << 20).hash_log3(), 0);
        // 1000 bytes at level 19: windowLog 10, cycleLog 18 - 1 -> chainLog 11.
        let cp = CParams::for_level(19, 1000);
        assert_eq!((cp.window_log, cp.chain_log), (10, 11));
        assert_eq!(cp.hash_log3(), 10);
    }

    #[test]
    fn min_gain_follows_strategy() {
        assert_eq!(CParams::min_gain(1 << 17, Strategy::BtOpt), (1 << 11) + 2);
        assert_eq!(CParams::min_gain(1 << 17, Strategy::BtUltra), (1 << 10) + 2);
        assert_eq!(CParams::min_gain(1 << 17, Strategy::BtUltra2), (1 << 9) + 2);
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
