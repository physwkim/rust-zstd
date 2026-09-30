//! Post-sequence block splitter (`ZSTD_c_splitAfterSequences`, the
//! `ZSTD_compressBlock_splitBlock` family of zstd_compress.c): after the
//! match finder has filled a block's [`SeqStore`], cut the sequences where
//! separate entropy tables are estimated to pay for the extra block
//! headers, and emit each partition as a block of its own.
//!
//! A partition may be written RAW or RLE while the finder's sequences
//! after it were built against the repeat offsets of a compressed
//! partition. [`resolve_off_codes`] keeps both histories, as libzstd does:
//! `c_rep` follows the sequences as found, `d_rep` what the decoder will
//! hold, and a repcode that means different offsets in the two is
//! rewritten to the raw offset `c_rep` gives it.

use super::block::{BlockState, ZSTD_BLOCKHEADERSIZE};
use super::params::{CParams, ParamSwitch};
use super::seqstore::{
    offbase_is_repcode, offbase_to_repcode, offset_to_offbase, update_rep, Seq, SeqStore,
    ZSTD_REP_NUM,
};
use crate::fse::{self, EstimateScratch};
use crate::huf;
use std::ops::Range;

/// `MIN_SEQUENCES_BLOCK_SPLITTING`: a chunk with fewer sequences is not
/// split further.
const MIN_SEQUENCES_BLOCK_SPLITTING: usize = 300;
/// `ZSTD_MAX_NB_BLOCK_SPLITS`.
const ZSTD_MAX_NB_BLOCK_SPLITS: usize = 196;
/// `ZSTD_btopt` in `ZSTD_strategy`.
const ZSTD_BTOPT: u32 = 7;

/// `ZSTD_resolveBlockSplitterMode` + `ZSTD_blockSplitterEnabled`: `Auto`
/// splits for `strategy >= ZSTD_btopt` with `window_log >= 17`, which no
/// ported strategy reaches yet.
pub fn block_splitter_enabled(mode: ParamSwitch, cparams: &CParams) -> bool {
    match mode {
        ParamSwitch::Enable => true,
        ParamSwitch::Disable => false,
        ParamSwitch::Auto => cparams.strategy as u32 >= ZSTD_BTOPT && cparams.window_log >= 17,
    }
}

/// `ZSTD_blockSplitCtx`: the split points of the current block and the
/// estimator's buffers.
#[derive(Default)]
pub struct BlockSplitter {
    /// `partitions` without the final `nbSeq`: sequence indices where a
    /// new block starts, ascending.
    splits: Vec<u32>,
    /// The blocks `splits` cut the block into; empty when it stays whole.
    parts: Vec<Partition>,
    /// `lit_start[i]`: literals before sequence `i`; `nb_seq + 1` entries.
    lit_start: Vec<u32>,
    desc: Vec<u8>,
    fse: EstimateScratch,
}

impl BlockSplitter {
    /// `ZSTD_deriveBlockSplits`: the partitions `store`'s block of
    /// `block_len` bytes is cut into, empty when it stays whole. `prev` is
    /// the committed state the partitions would be coded against.
    pub fn derive(
        &mut self,
        store: &SeqStore,
        prev: &BlockState,
        cparams: &CParams,
        block_len: usize,
    ) -> &[Partition] {
        self.splits.clear();
        self.parts.clear();
        let nb_seq = store.seqs.len();
        if nb_seq >= MIN_SEQUENCES_BLOCK_SPLITTING {
            self.lit_start.clear();
            self.lit_start.push(0);
            let mut total = 0u32;
            for seq in &store.seqs {
                total += seq.lit_len;
                self.lit_start.push(total);
            }
            self.derive_helper(store, prev, cparams, 0..nb_seq, None);
        }
        if !self.splits.is_empty() {
            let parts = partitions(store, &self.splits, block_len);
            self.parts.extend(parts);
        }
        &self.parts
    }

    /// `ZSTD_deriveBlockSplitsHelper`: split `chunk` in half when the halves'
    /// estimates sum below the whole's, then recurse into both. `whole` is
    /// the chunk's estimate when the caller already has it (each half's
    /// estimate is its whole in the recursion; the estimate depends only on
    /// the chunk and `prev`, so reusing it changes nothing).
    fn derive_helper(
        &mut self,
        store: &SeqStore,
        prev: &BlockState,
        cparams: &CParams,
        chunk: Range<usize>,
        whole: Option<usize>,
    ) {
        if chunk.len() < MIN_SEQUENCES_BLOCK_SPLITTING
            || self.splits.len() >= ZSTD_MAX_NB_BLOCK_SPLITS
        {
            return;
        }
        let mid = (chunk.start + chunk.end) / 2;
        let whole = whole.or_else(|| self.estimate(store, prev, cparams, chunk.clone()));
        let first = self.estimate(store, prev, cparams, chunk.start..mid);
        let second = self.estimate(store, prev, cparams, mid..chunk.end);
        let (Some(whole), Some(first), Some(second)) = (whole, first, second) else {
            return;
        };
        if first + second < whole {
            self.derive_helper(store, prev, cparams, chunk.start..mid, Some(first));
            self.splits.push(mid as u32);
            self.derive_helper(store, prev, cparams, mid..chunk.end, Some(second));
        }
    }

    /// `ZSTD_buildEntropyStatisticsAndEstimateSubBlockSize` for the
    /// sequences `seqs` of `store` (`ZSTD_deriveSeqStoreChunk`: the chunk
    /// ending the block also carries the trailing literals). `None` where
    /// the C returns an error.
    fn estimate(
        &mut self,
        store: &SeqStore,
        prev: &BlockState,
        cparams: &CParams,
        seqs: Range<usize>,
    ) -> Option<usize> {
        let lits_end = if seqs.end == store.seqs.len() {
            store.lits.len()
        } else {
            self.lit_start[seqs.end] as usize
        };
        let lits = &store.lits[self.lit_start[seqs.start] as usize..lits_end];
        let literals = huf::estimate_literals_section(lits, &prev.huf, cparams, &mut self.desc)?;
        let sequences =
            fse::estimate_sequences_section(&store.seqs[seqs], &prev.fse, cparams, &mut self.fse)?;
        Some(literals + sequences + ZSTD_BLOCKHEADERSIZE)
    }
}

/// One block of a split block: its sequences, its literals in the
/// store, and the source bytes it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub seqs: Range<usize>,
    pub lits: Range<usize>,
    pub src_len: usize,
}

/// The partitions `splits` cut `store` (a block of `block_len` bytes) into,
/// in order: `ZSTD_deriveSeqStoreChunk` for each, with the last one also
/// covering the trailing literals.
fn partitions<'a>(
    store: &'a SeqStore,
    splits: &'a [u32],
    block_len: usize,
) -> impl Iterator<Item = Partition> + 'a {
    let nb_seq = store.seqs.len();
    let ends = splits.iter().map(|&s| s as usize).chain([nb_seq]);
    let (mut seq_start, mut lit_start, mut src_total) = (0usize, 0usize, 0usize);
    ends.map(move |seq_end| {
        let seqs = &store.seqs[seq_start..seq_end];
        let lit_len: usize = seqs.iter().map(|s| s.lit_len as usize).sum();
        let match_len: usize = seqs.iter().map(|s| s.match_len() as usize).sum();
        let last = seq_end == nb_seq;
        let part = Partition {
            seqs: seq_start..seq_end,
            lits: lit_start..if last {
                store.lits.len()
            } else {
                lit_start + lit_len
            },
            src_len: if last {
                block_len - src_total
            } else {
                lit_len + match_len
            },
        };
        seq_start = seq_end;
        lit_start += lit_len;
        src_total += lit_len + match_len;
        part
    })
}

/// `ZSTD_resolveRepcodeToRawOffset`: the offset repcode `off_base` names
/// under `rep` (`rep[0] - 1` may wrap; such a value never matches).
fn resolve_repcode_to_raw_offset(rep: &[u32; 3], off_base: u32, ll0: bool) -> u32 {
    let adjusted = offbase_to_repcode(off_base) - 1 + ll0 as u32;
    if adjusted == ZSTD_REP_NUM {
        rep[0].wrapping_sub(1)
    } else {
        rep[adjusted as usize]
    }
}

/// `ZSTD_seqStore_resolveOffCodes`: walk a partition's sequences with the
/// decoder's history `d_rep` and the finder's `c_rep`, rewriting a repcode
/// whose offset differs between them to the raw offset `c_rep` gives it.
pub fn resolve_off_codes(d_rep: &mut [u32; 3], c_rep: &mut [u32; 3], seqs: &mut [Seq]) {
    for seq in seqs {
        let ll0 = seq.lit_len == 0;
        let off_base = seq.off_base;
        if offbase_is_repcode(off_base) {
            let d_raw = resolve_repcode_to_raw_offset(d_rep, off_base, ll0);
            let c_raw = resolve_repcode_to_raw_offset(c_rep, off_base, ll0);
            if d_raw != c_raw {
                seq.off_base = offset_to_offbase(c_raw);
            }
        }
        update_rep(d_rep, seq.off_base, ll0);
        update_rep(c_rep, off_base, ll0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::seqstore::repcode_to_offbase;

    fn seq(lit_len: u32, off_base: u32) -> Seq {
        Seq {
            lit_len,
            off_base,
            ml_base: 1,
        }
    }

    /// Equal histories: every repcode resolves to the same offset on both
    /// sides, nothing is rewritten and the histories stay equal.
    #[test]
    fn resolve_off_codes_keeps_repcodes_of_equal_histories() {
        let mut seqs = vec![
            seq(5, offset_to_offbase(100)),
            seq(0, repcode_to_offbase(1)),
            seq(3, repcode_to_offbase(3)),
            seq(0, repcode_to_offbase(3)),
            seq(2, repcode_to_offbase(2)),
        ];
        let before = seqs.clone();
        let (mut d_rep, mut c_rep) = ([1, 4, 8], [1, 4, 8]);
        resolve_off_codes(&mut d_rep, &mut c_rep, &mut seqs);
        assert_eq!(seqs, before);
        assert_eq!(d_rep, c_rep);
    }

    /// Diverged histories: a repcode is rewritten to the finder's raw offset
    /// exactly when the two sides resolve it differently, the decoder's
    /// history follows the rewritten sequence and the finder's the original.
    #[test]
    fn resolve_off_codes_rewrites_diverged_repcodes() {
        let mut seqs = vec![
            // ll0 repcode 3 is rep[0] - 1: 6 vs 99, rewritten.
            seq(0, repcode_to_offbase(3)),
            // rep[0]: 99 on both sides now, kept.
            seq(4, repcode_to_offbase(1)),
            // rep[1]: 7 vs 100, rewritten.
            seq(4, repcode_to_offbase(2)),
            // rep[2]: 7 vs 4, rewritten.
            seq(4, repcode_to_offbase(3)),
            // ll0 repcode 1 is rep[1]: 100 on both sides, kept.
            seq(0, repcode_to_offbase(1)),
        ];
        let (mut d_rep, mut c_rep) = ([7, 4, 8], [100, 4, 8]);
        resolve_off_codes(&mut d_rep, &mut c_rep, &mut seqs);
        let off_bases: Vec<u32> = seqs.iter().map(|s| s.off_base).collect();
        assert_eq!(
            off_bases,
            [
                offset_to_offbase(99),
                repcode_to_offbase(1),
                offset_to_offbase(100),
                offset_to_offbase(4),
                repcode_to_offbase(1),
            ]
        );
        // Finder: [99,100,4] -> [100,99,4] -> [4,100,99] -> [100,4,99].
        assert_eq!(c_rep, [100, 4, 99]);
        // Decoder: [99,7,4] -> [100,99,7] -> [4,100,99] -> [100,4,99].
        assert_eq!(d_rep, c_rep);
    }

    /// Every partition covers its literals and matches; the last one also
    /// the trailing literals and the rest of the block.
    #[test]
    fn partitions_cover_the_block() {
        let store = SeqStore {
            lits: vec![0; 5 + 1 + 2 + 9],
            seqs: vec![seq(5, 10), seq(1, 10), seq(2, 10)],
        };
        let block_len = 5 + 1 + 2 + 9 + 3 * 4;
        let parts: Vec<Partition> = partitions(&store, &[1, 2], block_len).collect();
        assert_eq!(
            parts,
            [
                Partition {
                    seqs: 0..1,
                    lits: 0..5,
                    src_len: 9
                },
                Partition {
                    seqs: 1..2,
                    lits: 5..6,
                    src_len: 5
                },
                Partition {
                    seqs: 2..3,
                    lits: 6..17,
                    src_len: 2 + 4 + 9
                },
            ]
        );
    }

    /// `Auto` follows `ZSTD_resolveBlockSplitterMode`: off for every ported
    /// strategy.
    #[test]
    fn auto_mode_is_off_below_btopt() {
        for level in 1..=22 {
            let cp = CParams::for_level(level, 8 << 20);
            assert!(!block_splitter_enabled(ParamSwitch::Auto, &cp), "L{level}");
            assert!(block_splitter_enabled(ParamSwitch::Enable, &cp));
            assert!(!block_splitter_enabled(ParamSwitch::Disable, &cp));
        }
    }
}
