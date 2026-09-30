//! Long distance matching: port of `zstd_ldm.c` (libzstd 1.5.7) without
//! the dictionary paths.
//!
//! [`LdmState::generate_sequences`] (`ZSTD_ldm_generateSequences`) cuts the
//! input at the split points of a gear rolling hash, looks each split's
//! `min_match_length` bytes up in a bucketed hash table of earlier splits
//! and emits a [`RawSeq`] for every long match, extended backwards. The
//! window is the whole input the state has seen, limited to
//! `1 << window_log` bytes. [`block_compress`] (`ZSTD_ldm_blockCompress`)
//! then runs the strategy's block compressor on the literals between those
//! sequences and stores the sequences themselves verbatim.
//!
//! Positions are absolute indices into the input, as everywhere in the
//! compressor. libzstd's window indices are these positions plus a
//! constant, so every comparison is the same; its overflow correction
//! (`ZSTD_ldm_reduceTable`, from 3500 MiB of input on) is not ported, as
//! for the match state.

use super::block;
use super::common::{count, prefetch_l1, HASH_READ_SIZE};
use super::matchstate::MatchState;
use super::opt;
use super::params::{CParams, Strategy};
use super::seqstore::{offset_to_offbase, SeqStore};
use std::ops::Range;

/// `ZSTD_LDM_DEFAULT_WINDOW_LOG` (`ZSTD_WINDOWLOG_LIMIT_DEFAULT`): the
/// window log an explicitly enabled LDM selects before the size adjustment.
pub const LDM_DEFAULT_WINDOW_LOG: u32 = 27;
/// `LDM_BUCKET_SIZE_LOG`: lowest derived bucket size log.
const LDM_BUCKET_SIZE_LOG: u32 = 4;
/// `LDM_MIN_MATCH_LENGTH`: derived minimum match length.
const LDM_MIN_MATCH_LENGTH: u32 = 64;
/// `LDM_BATCH_SIZE`: split points hashed per round.
const LDM_BATCH_SIZE: usize = 64;
/// `kMaxChunkSize` of `ZSTD_ldm_generateSequences`.
const MAX_CHUNK_SIZE: usize = 1 << 20;
/// `ZSTD_HASHLOG_MIN` / `ZSTD_HASHLOG_MAX` (64-bit).
const HASHLOG_MIN: u32 = 6;
const HASHLOG_MAX: u32 = 30;
/// `ZSTD_LDM_MINMATCH_MIN` / `ZSTD_LDM_MINMATCH_MAX`.
const MINMATCH_MIN: u32 = 4;
const MINMATCH_MAX: u32 = 4096;
/// `ZSTD_LDM_BUCKETSIZELOG_MIN` / `ZSTD_LDM_BUCKETSIZELOG_MAX`.
const BUCKETSIZELOG_MIN: u32 = 1;
const BUCKETSIZELOG_MAX: u32 = 8;
/// `ZSTD_LDM_HASHRATELOG_MAX`: `ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN`.
const HASHRATELOG_MAX: u32 = 31 - HASHLOG_MIN;

/// `ldmParams_t` without `enableLdm`. A field left at `0` is derived from
/// the compression parameters by [`LdmParams::adjusted`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LdmParams {
    /// `hashLog`: log2 of the hash table's entry count.
    pub hash_log: u32,
    /// `bucketSizeLog`: log2 of the entries per bucket.
    pub bucket_size_log: u32,
    /// `minMatchLength`: shortest match emitted, and the hashed length.
    pub min_match_length: u32,
    /// `hashRateLog`: a split point every `1 << hash_rate_log` bytes on
    /// average.
    pub hash_rate_log: u32,
    /// `windowLog`: set by [`LdmParams::adjusted`].
    pub window_log: u32,
}

impl LdmParams {
    /// The requested parameters (`ZSTD_c_ldmHashLog`, `ZSTD_c_ldmMinMatch`,
    /// `ZSTD_c_ldmBucketSizeLog`, `ZSTD_c_ldmHashRateLog`), `0` meaning
    /// "derive". Other values outside libzstd's bounds panic, where
    /// `ZSTD_CCtx_setParameter` returns `parameter_outOfBound`.
    pub fn requested(
        hash_log: u32,
        min_match_length: u32,
        bucket_size_log: u32,
        hash_rate_log: u32,
    ) -> Self {
        let check = |name: &str, v: u32, lo: u32, hi: u32| {
            assert!(
                v == 0 || (lo..=hi).contains(&v),
                "{name} {v} out of range {lo}..={hi} (0 derives it)"
            );
        };
        check("ldm_hash_log", hash_log, HASHLOG_MIN, HASHLOG_MAX);
        check(
            "ldm_min_match",
            min_match_length,
            MINMATCH_MIN,
            MINMATCH_MAX,
        );
        check(
            "ldm_bucket_size_log",
            bucket_size_log,
            BUCKETSIZELOG_MIN,
            BUCKETSIZELOG_MAX,
        );
        check("ldm_hash_rate_log", hash_rate_log, 0, HASHRATELOG_MAX);
        Self {
            hash_log,
            bucket_size_log,
            min_match_length,
            hash_rate_log,
            window_log: 0,
        }
    }

    /// `ZSTD_ldm_adjustParameters(params, cParams)` for the frame's final
    /// compression parameters.
    pub fn adjusted(mut self, cparams: &CParams) -> Self {
        let strategy = cparams.strategy as u32;
        self.window_log = cparams.window_log;
        if self.hash_rate_log == 0 {
            if self.hash_log > 0 {
                // derive hashRateLog from hashLog (stays 0 when the window
                // is not larger)
                if self.window_log > self.hash_log {
                    self.hash_rate_log = self.window_log - self.hash_log;
                }
            } else {
                // mapping from [fast, rate7] to [btultra2, rate4]
                self.hash_rate_log = 7 - strategy / 3;
            }
        }
        if self.hash_log == 0 {
            // U32 arithmetic: a rate above the window log wraps to the max.
            self.hash_log = self
                .window_log
                .wrapping_sub(self.hash_rate_log)
                .clamp(HASHLOG_MIN, HASHLOG_MAX);
        }
        if self.min_match_length == 0 {
            self.min_match_length = LDM_MIN_MATCH_LENGTH;
            if cparams.strategy >= Strategy::BtUltra {
                self.min_match_length /= 2;
            }
        }
        if self.bucket_size_log == 0 {
            self.bucket_size_log = strategy.clamp(LDM_BUCKET_SIZE_LOG, BUCKETSIZELOG_MAX);
        }
        self.bucket_size_log = self.bucket_size_log.min(self.hash_log);
        self
    }
}

/// `rawSeq`: `lit_length` literals, then `match_length` bytes at `offset`
/// back. `offset == 0` marks "the rest is literals" in
/// [`RawSeqStore::maybe_split_sequence`]'s result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawSeq {
    pub offset: u32,
    pub lit_length: u32,
    pub match_length: u32,
}

/// `RawSeqStore_t`: generated sequences and the read position of
/// [`block_compress`], which consumes them across blocks.
///
/// The strategies below btopt consume a sequence by shortening it in place
/// ([`RawSeqStore::skip_sequences`]); the optimal parser leaves the
/// sequences intact and counts the bytes consumed of the current one in
/// `pos_in_sequence` ([`RawSeqStore::skip_raw_seq_store_bytes`]). A store
/// serves one frame or job, so one strategy, and never mixes the two.
#[derive(Clone, Debug, Default)]
pub struct RawSeqStore {
    pub seqs: Vec<RawSeq>,
    /// `pos`: the next sequence to consume.
    pub pos: usize,
    /// `posInSequence`: bytes of `seqs[pos]` the optimal parser consumed.
    pub pos_in_sequence: usize,
}

impl RawSeqStore {
    /// Whether every sequence has been consumed (`pos >= size`).
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.seqs.len()
    }

    /// The optimal parser's copy of the store (`ZSTD_optLdm_t::seqStore =
    /// *ms->ldmSeqStore`).
    pub fn view(&self) -> RawSeqView<'_> {
        RawSeqView {
            seqs: &self.seqs,
            pos: self.pos,
            pos_in_sequence: self.pos_in_sequence,
        }
    }

    /// `ZSTD_ldm_skipRawSeqStoreBytes`: see [`RawSeqView::skip_bytes`].
    pub fn skip_raw_seq_store_bytes(&mut self, nb_bytes: usize) {
        let mut view = self.view();
        view.skip_bytes(nb_bytes);
        (self.pos, self.pos_in_sequence) = (view.pos, view.pos_in_sequence);
    }

    /// `ZSTD_ldm_skipSequences`: consume `src_size` bytes of input; a
    /// match cut below `min_match` becomes literals of the next sequence.
    pub fn skip_sequences(&mut self, mut src_size: usize, min_match: u32) {
        debug_assert_eq!(self.pos_in_sequence, 0, "consumed by the optimal parser");
        while src_size > 0 && self.pos < self.seqs.len() {
            let seq = &mut self.seqs[self.pos];
            if src_size <= seq.lit_length as usize {
                // Skip past srcSize literals
                seq.lit_length -= src_size as u32;
                return;
            }
            src_size -= seq.lit_length as usize;
            seq.lit_length = 0;
            if src_size < seq.match_length as usize {
                // Skip past the first srcSize of the match
                seq.match_length -= src_size as u32;
                if seq.match_length < min_match {
                    // The match is too short, omit it
                    let rest = seq.match_length;
                    if let Some(next) = self.seqs.get_mut(self.pos + 1) {
                        next.lit_length += rest;
                    }
                    self.pos += 1;
                }
                return;
            }
            src_size -= seq.match_length as usize;
            seq.match_length = 0;
            self.pos += 1;
        }
    }

    /// `maybeSplitSequence`: the next sequence cut to the `remaining` bytes
    /// of the block; `offset == 0` when the rest of the block is literals.
    fn maybe_split_sequence(&mut self, remaining: u32, min_match: u32) -> RawSeq {
        let mut seq = self.seqs[self.pos];
        debug_assert!(seq.offset > 0);
        // Likely: No partial sequence
        if remaining >= seq.lit_length + seq.match_length {
            self.pos += 1;
            return seq;
        }
        // Cut the sequence short (offset == 0 ==> rest is literals).
        if remaining <= seq.lit_length {
            seq.offset = 0;
        } else {
            seq.match_length = remaining - seq.lit_length;
            if seq.match_length < min_match {
                seq.offset = 0;
            }
        }
        // Skip past `remaining` bytes for the future sequences.
        self.skip_sequences(remaining as usize, min_match);
        seq
    }
}

/// A read position over borrowed sequences: the optimal parser's copy of a
/// [`RawSeqStore`], which it moves through the block while the store
/// itself is moved past the whole block afterwards. The default is
/// `kNullRawSeqStore`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RawSeqView<'a> {
    seqs: &'a [RawSeq],
    pos: usize,
    pos_in_sequence: usize,
}

impl RawSeqView<'_> {
    /// Whether every sequence has been consumed (`size == 0 || pos >=
    /// size`).
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.seqs.len()
    }

    /// The sequence at the read position and the bytes of it consumed.
    pub fn current(&self) -> (RawSeq, usize) {
        (self.seqs[self.pos], self.pos_in_sequence)
    }

    /// `ZSTD_optLdm_skipRawSeqStoreBytes`: move forward by `nb_bytes`,
    /// updating `pos` and `pos_in_sequence`.
    pub fn skip_bytes(&mut self, nb_bytes: usize) {
        let mut curr_pos = self.pos_in_sequence + nb_bytes;
        while curr_pos != 0 && self.pos < self.seqs.len() {
            let curr_seq = self.seqs[self.pos];
            let seq_len = (curr_seq.lit_length + curr_seq.match_length) as usize;
            if curr_pos >= seq_len {
                curr_pos -= seq_len;
                self.pos += 1;
            } else {
                self.pos_in_sequence = curr_pos;
                break;
            }
        }
        if curr_pos == 0 || self.pos == self.seqs.len() {
            self.pos_in_sequence = 0;
        }
    }
}

/// `ldmEntry_t`: a split position and the high half of its hash.
#[derive(Clone, Copy, Debug, Default)]
struct LdmEntry {
    offset: u32,
    checksum: u32,
}

/// `ldmState_t` with its window: the hash table persists across the
/// blocks (or jobs) of a frame.
pub struct LdmState {
    params: LdmParams,
    hash_table: Vec<LdmEntry>,
    bucket_offsets: Vec<u8>,
    /// `window.lowLimit` (== `dictLimit`) as a position: entries at or
    /// below it are stale, and a match may extend backwards down to it.
    low: usize,
    /// `zc->ldmSequences`: a block's sequences on the single-context path.
    block_seqs: RawSeqStore,
}

impl LdmState {
    /// A state for a frame whose first byte is at `first`: see
    /// [`LdmState::reset`].
    pub fn new(params: LdmParams, first: usize) -> Self {
        let mut state = Self {
            params,
            hash_table: Vec::new(),
            bucket_offsets: Vec::new(),
            low: first,
            block_seqs: RawSeqStore::default(),
        };
        state.reset(params, first);
        state
    }

    /// `ZSTD_resetCCtx_internal` / `ZSTDMT_serialState_reset`: zeroed
    /// tables for `params` (kept allocations are reused) and an empty
    /// window starting at `first`.
    pub fn reset(&mut self, params: LdmParams, first: usize) {
        debug_assert!(params.window_log != 0, "LdmParams::adjusted not applied");
        self.params = params;
        self.hash_table.clear();
        self.hash_table
            .resize(1 << params.hash_log, LdmEntry::default());
        self.bucket_offsets.clear();
        self.bucket_offsets
            .resize(1 << (params.hash_log - params.bucket_size_log), 0);
        self.low = first;
        self.block_seqs = RawSeqStore {
            seqs: std::mem::take(&mut self.block_seqs.seqs),
            ..Default::default()
        };
    }

    pub fn params(&self) -> &LdmParams {
        &self.params
    }

    /// `ZSTD_ldm_generateSequences`: replace `out` with the sequences of
    /// `src[range]`, which must follow the data this state has seen.
    /// `max_seqs` is the store's capacity (`ZSTD_ldm_getMaxNbSeq` of the
    /// block or job size): no chunk starts once it is reached.
    pub fn generate_sequences(
        &mut self,
        src: &[u8],
        range: Range<usize>,
        max_seqs: usize,
        out: &mut RawSeqStore,
    ) {
        out.seqs.clear();
        out.pos = 0;
        out.pos_in_sequence = 0;
        let max_dist = 1usize << self.params.window_log;
        let mut leftover = 0usize;
        let mut chunk_start = range.start;
        // The input could be very large (in zstdmt), so it must be broken
        // up into chunks to enforce the maximum distance.
        while chunk_start < range.end && out.seqs.len() < max_seqs {
            let chunk_end = range.end.min(chunk_start + MAX_CHUNK_SIZE);
            // ZSTD_window_enforceMaxDist(&ldmState->window, chunkEnd, ...)
            self.low = self.low.max(chunk_end.saturating_sub(max_dist));
            let prev = out.seqs.len();
            let new_leftover =
                self.generate_sequences_internal(src, chunk_start..chunk_end, &mut out.seqs);
            // Prepend the leftover literals from the last call
            if prev < out.seqs.len() {
                out.seqs[prev].lit_length += leftover as u32;
                leftover = new_leftover;
            } else {
                debug_assert_eq!(new_leftover, chunk_end - chunk_start);
                leftover += chunk_end - chunk_start;
            }
            chunk_start = chunk_end;
        }
    }

    /// [`LdmState::generate_sequences`] for one block on the
    /// single-context path (`ZSTD_buildSeqStore`'s `enableLdm` branch):
    /// the sequences go to a store owned by the state.
    pub fn generate_block_sequences(
        &mut self,
        src: &[u8],
        block: Range<usize>,
    ) -> &mut RawSeqStore {
        // maxNbLdmSeq = ZSTD_ldm_getMaxNbSeq(ldmParams, blockSize)
        let block_size_max = crate::constants::ZSTD_BLOCKSIZE_MAX.min(1 << self.params.window_log);
        let max_seqs = block_size_max / self.params.min_match_length as usize;
        let mut seqs = std::mem::take(&mut self.block_seqs);
        self.generate_sequences(src, block, max_seqs, &mut seqs);
        self.block_seqs = seqs;
        &mut self.block_seqs
    }

    /// `ZSTD_ldm_insertEntry`.
    #[inline]
    fn insert_entry(&mut self, hash: usize, entry: LdmEntry) {
        let bsl = self.params.bucket_size_log;
        let offset = self.bucket_offsets[hash];
        self.hash_table[(hash << bsl) + offset as usize] = entry;
        self.bucket_offsets[hash] = ((offset as u32 + 1) & ((1 << bsl) - 1)) as u8;
    }

    /// `ZSTD_ldm_generateSequences_internal` over one chunk: append its
    /// sequences to `seqs` and return the length of the trailing literals.
    fn generate_sequences_internal(
        &mut self,
        src: &[u8],
        chunk: Range<usize>,
        seqs: &mut Vec<RawSeq>,
    ) -> usize {
        let params = self.params;
        let min_match = params.min_match_length as usize;
        let ents_per_bucket = 1usize << params.bucket_size_log;
        let h_bits = params.hash_log - params.bucket_size_log;
        let lowest = self.low;
        let (istart, iend) = (chunk.start, chunk.end);
        // Below `istart + min_match` the loop condition is false anyway.
        let ilimit = iend.saturating_sub(HASH_READ_SIZE);
        let mut anchor = istart;

        if chunk.len() < min_match {
            return iend - anchor;
        }

        // Initialize the rolling hash state with the first minMatchLength
        // bytes (ZSTD_ldm_gear_reset leaves the state as it is).
        let mut hash_state = GearState::new(&params);
        let mut ip = istart + min_match;
        let mut splits = [0usize; LDM_BATCH_SIZE];
        let mut candidates = [(0usize, 0usize, 0u32); LDM_BATCH_SIZE];

        while ip < ilimit {
            let (hashed, num_splits) = hash_state.feed(&src[ip..ilimit], &mut splits);

            for (cand, &split_n) in candidates.iter_mut().zip(&splits[..num_splits]) {
                let split = ip + split_n - min_match;
                let xxhash = xxh64(&src[split..split + min_match]);
                let hash = (xxhash as u32 & ((1u32 << h_bits) - 1)) as usize;
                // ZSTD_ldm_getBucket
                prefetch_l1(&self.hash_table, hash << params.bucket_size_log);
                *cand = (split, hash, (xxhash >> 32) as u32);
            }

            for &(split, hash, checksum) in &candidates[..num_splits] {
                let new_entry = LdmEntry {
                    offset: split as u32,
                    checksum,
                };

                // If a split point would generate a sequence overlapping
                // with the previous one, we merely register it in the hash
                // table and move on
                if split < anchor {
                    self.insert_entry(hash, new_entry);
                    continue;
                }

                let bucket_start = hash << params.bucket_size_log;
                let bucket = &self.hash_table[bucket_start..bucket_start + ents_per_bucket];
                let mut best: Option<(u32, usize, usize)> = None;
                let mut best_length = 0;
                for cur in bucket {
                    if cur.checksum != checksum || cur.offset as usize <= lowest {
                        continue;
                    }
                    let p_match = cur.offset as usize;
                    // SAFETY: every entry is an earlier split, so
                    // `p_match < split < iend <= src.len()`.
                    let cur_forward = unsafe { count(src, split, p_match, iend) };
                    if cur_forward < min_match {
                        continue;
                    }
                    let cur_backward = count_backwards(src, split, anchor, p_match, lowest);
                    let cur_total = cur_forward + cur_backward;
                    if cur_total > best_length {
                        best_length = cur_total;
                        best = Some((cur.offset, cur_forward, cur_backward));
                    }
                }

                // No match found -- insert an entry into the hash table
                // and process the next candidate match
                let Some((best_offset, forward, backward)) = best else {
                    self.insert_entry(hash, new_entry);
                    continue;
                };

                // Match found
                seqs.push(RawSeq {
                    offset: split as u32 - best_offset,
                    lit_length: (split - backward - anchor) as u32,
                    match_length: (forward + backward) as u32,
                });

                // Insert the current entry into the hash table --- it must
                // be done after the previous block to avoid clobbering
                // bestEntry
                self.insert_entry(hash, new_entry);

                anchor = split + forward;

                // A match that ends after the hashed data is a repeating,
                // overlapping pattern: skip over it (continue the outer
                // loop at anchor; ip + hashed == anchor).
                if anchor > ip + hashed {
                    ip = anchor - hashed;
                    break;
                }
            }

            ip += hashed;
        }

        iend - anchor
    }
}

/// `ZSTD_ldm_countBackwardsMatch`: bytes equal before `p_in` and
/// `p_match`, with `p_in` not going below `anchor` nor `p_match` below
/// `low`.
fn count_backwards(
    src: &[u8],
    mut p_in: usize,
    anchor: usize,
    mut p_match: usize,
    low: usize,
) -> usize {
    let mut len = 0;
    while p_in > anchor && p_match > low && src[p_in - 1] == src[p_match - 1] {
        p_in -= 1;
        p_match -= 1;
        len += 1;
    }
    len
}

/// `ZSTD_ldm_limitTableUpdate`: after a long match, index at most the
/// last 512 positions (fewer while the backlog is under 1536) of the
/// backlog before `anchor`.
fn limit_table_update(ms: &mut MatchState, anchor: usize) {
    if anchor > ms.next_to_update + 1024 {
        ms.next_to_update = anchor - 512.min(anchor - ms.next_to_update - 1024);
    }
}

/// `ZSTD_ldm_fillFastTables`: index the positions from
/// `ms.next_to_update` to `end` into the fast and double-fast tables; the
/// other strategies index inside their block compressors.
fn fill_fast_tables(ms: &mut MatchState, src: &[u8], end: usize) {
    match ms.cparams.strategy {
        Strategy::Fast => super::fast::fill_hash_table_to(ms, src, end),
        Strategy::DFast => super::dfast::fill_double_hash_table_to(ms, src, end),
        Strategy::Greedy
        | Strategy::Lazy
        | Strategy::Lazy2
        | Strategy::BtLazy2
        | Strategy::BtOpt
        | Strategy::BtUltra
        | Strategy::BtUltra2 => {}
    }
}

/// `ZSTD_ldm_blockCompress`: from btopt on, run the optimal parser on
/// `src[block]` with the long matches of `seqs` as extra candidates; below
/// it, store the long matches that fall in the block (cut at the block
/// end) and run the strategy's block compressor on the literals between
/// them. Either way `seqs` moves past the block. Returns the anchor of the
/// block's trailing literals.
pub fn block_compress(
    seqs: &mut RawSeqStore,
    ms: &mut MatchState,
    src: &[u8],
    block: Range<usize>,
    rep: &mut [u32; 3],
    out: &mut SeqStore,
) -> usize {
    // If using opt parser, use LDMs only as candidates rather than always
    // accepting them
    if ms.cparams.strategy >= Strategy::BtOpt {
        let block_len = block.len();
        let anchor = opt::compress_block(ms, src, block, rep, out, seqs.view());
        seqs.skip_raw_seq_store_bytes(block_len);
        return anchor;
    }

    let min_match = ms.cparams.min_match;
    let iend = block.end;
    let mut ip = block.start;
    // Loop through each sequence and apply the block compressor to the
    // literals
    while !seqs.is_exhausted() && ip < iend {
        let seq = seqs.maybe_split_sequence((iend - ip) as u32, min_match);
        // End signal
        if seq.offset == 0 {
            break;
        }
        debug_assert!(ip + (seq.lit_length + seq.match_length) as usize <= iend);

        // Fill tables for block compressor
        limit_table_update(ms, ip);
        fill_fast_tables(ms, src, ip);
        // Run the block compressor
        let lit_end = ip + seq.lit_length as usize;
        let anchor = block::run_block_compressor(ms, src, ip..lit_end, rep, out);
        ip = lit_end;
        // Update the repcodes
        rep[2] = rep[1];
        rep[1] = rep[0];
        rep[0] = seq.offset;
        // Store the sequence
        out.store_seq(
            src,
            anchor,
            lit_end - anchor,
            iend,
            offset_to_offbase(seq.offset),
            seq.match_length as usize,
        );
        ip += seq.match_length as usize;
    }
    // Fill the tables for the block compressor
    limit_table_update(ms, ip);
    fill_fast_tables(ms, src, ip);
    // Compress the last literals
    block::run_block_compressor(ms, src, ip..iend, rep, out)
}

/// `ldmRollingHashState_t`: the gear hash.
struct GearState {
    rolling: u64,
    stop_mask: u64,
}

impl GearState {
    /// `ZSTD_ldm_gear_init`: a split every `1 << hash_rate_log` bytes on
    /// average, tested on the highest bits that still depend only on the
    /// last `min_match_length` bytes.
    fn new(params: &LdmParams) -> Self {
        let max_bits_in_mask = params.min_match_length.min(64);
        let hash_rate_log = params.hash_rate_log;
        let stop_mask = if hash_rate_log > 0 && hash_rate_log <= max_bits_in_mask {
            ((1u64 << hash_rate_log) - 1) << (max_bits_in_mask - hash_rate_log)
        } else {
            // In this degenerate case we simply honor the hash rate.
            (1u64 << hash_rate_log) - 1
        };
        Self {
            rolling: u32::MAX as u64,
            stop_mask,
        }
    }

    /// `ZSTD_ldm_gear_feed`: record in `splits` the end offset of every
    /// split point in `data`, stopping after `LDM_BATCH_SIZE` of them.
    /// Returns the number of bytes processed and of splits recorded.
    ///
    /// Not inlined, and unrolled by four as in C: inlined into
    /// `generate_sequences_internal`, the rolling hash was kept in a stack
    /// slot, putting a store-to-load round trip on its per-byte chain.
    #[inline(never)]
    fn feed(&mut self, data: &[u8], splits: &mut [usize; LDM_BATCH_SIZE]) -> (usize, usize) {
        let mut hash = self.rolling;
        let mask = self.stop_mask;
        let mut n = 0;
        let mut num_splits = 0;
        macro_rules! gear_iter_once {
            ($done:lifetime) => {
                hash = gear_step(hash, GEAR_TAB[data[n] as usize]);
                n += 1;
                if hash & mask == 0 {
                    splits[num_splits] = n;
                    num_splits += 1;
                    if num_splits == LDM_BATCH_SIZE {
                        break $done;
                    }
                }
            };
        }
        'feed: {
            while n + 3 < data.len() {
                gear_iter_once!('feed);
                gear_iter_once!('feed);
                gear_iter_once!('feed);
                gear_iter_once!('feed);
            }
            while n < data.len() {
                gear_iter_once!('feed);
            }
        }
        self.rolling = hash;
        (n, num_splits)
    }
}

/// One gear round, `(hash << 1) + gear`, as the single `lea` libzstd
/// compiles it to on x86-64: LLVM folds the table load into a second
/// dependent `add` instead, two cycles per input byte on the rolling
/// hash's chain rather than one.
#[inline(always)]
fn gear_step(hash: u64, gear: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut hash = hash;
        // SAFETY: register arithmetic only.
        unsafe {
            std::arch::asm!(
                "lea {h}, [{g} + {h} * 2]",
                h = inout(reg) hash,
                g = in(reg) gear,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        hash
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (hash << 1).wrapping_add(gear)
    }
}

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

/// `XXH64_round`.
#[inline(always)]
fn xxh64_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

/// `XXH64_mergeRound`.
#[inline(always)]
fn xxh64_merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ xxh64_round(0, val))
        .wrapping_mul(PRIME64_1)
        .wrapping_add(PRIME64_4)
}

#[inline(always)]
fn le64(p: &[u8]) -> u64 {
    u64::from_le_bytes(p[..8].try_into().unwrap())
}

/// `XXH64(data, len, 0)` (xxhash.h).
fn xxh64(data: &[u8]) -> u64 {
    let len = data.len();
    let mut p = data;
    let mut h64 = if len >= 32 {
        let mut v = [
            PRIME64_1.wrapping_add(PRIME64_2),
            PRIME64_2,
            0,
            0u64.wrapping_sub(PRIME64_1),
        ];
        while p.len() >= 32 {
            for (i, lane) in v.iter_mut().enumerate() {
                *lane = xxh64_round(*lane, le64(&p[8 * i..]));
            }
            p = &p[32..];
        }
        let mut h = v[0]
            .rotate_left(1)
            .wrapping_add(v[1].rotate_left(7))
            .wrapping_add(v[2].rotate_left(12))
            .wrapping_add(v[3].rotate_left(18));
        for lane in v {
            h = xxh64_merge_round(h, lane);
        }
        h
    } else {
        PRIME64_5
    };
    h64 = h64.wrapping_add(len as u64);
    // XXH64_finalize
    while p.len() >= 8 {
        h64 ^= xxh64_round(0, le64(p));
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(PRIME64_1)
            .wrapping_add(PRIME64_4);
        p = &p[8..];
    }
    if p.len() >= 4 {
        h64 ^= (u32::from_le_bytes(p[..4].try_into().unwrap()) as u64).wrapping_mul(PRIME64_1);
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(PRIME64_2)
            .wrapping_add(PRIME64_3);
        p = &p[4..];
    }
    for &b in p {
        h64 ^= (b as u64).wrapping_mul(PRIME64_5);
        h64 = h64.rotate_left(11).wrapping_mul(PRIME64_1);
    }
    // XXH64_avalanche
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(PRIME64_2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(PRIME64_3);
    h64 ^= h64 >> 32;
    h64
}

/// `ZSTD_ldm_gearTab` (zstd_ldm_geartab.h).
#[rustfmt::skip]
static GEAR_TAB: [u64; 256] = [
    0xf5b8f72c5f77775c, 0x84935f266b7ac412, 0xb647ada9ca730ccc,
    0xb065bb4b114fb1de, 0x34584e7e8c3a9fd0, 0x4e97e17c6ae26b05,
    0x3a03d743bc99a604, 0xcecd042422c4044f, 0x76de76c58524259e,
    0x9c8528f65badeaca, 0x86563706e2097529, 0x2902475fa375d889,
    0xafb32a9739a5ebe6, 0xce2714da3883e639, 0x21eaf821722e69e,
    0x37b628620b628, 0x49a8d455d88caf5, 0x8556d711e6958140,
    0x4f7ae74fc605c1f, 0x829f0c3468bd3a20, 0x4ffdc885c625179e,
    0x8473de048a3daf1b, 0x51008822b05646b2, 0x69d75d12b2d1cc5f,
    0x8c9d4a19159154bc, 0xc3cc10f4abbd4003, 0xd06ddc1cecb97391,
    0xbe48e6e7ed80302e, 0x3481db31cee03547, 0xacc3f67cdaa1d210,
    0x65cb771d8c7f96cc, 0x8eb27177055723dd, 0xc789950d44cd94be,
    0x934feadc3700b12b, 0x5e485f11edbdf182, 0x1e2e2a46fd64767a,
    0x2969ca71d82efa7c, 0x9d46e9935ebbba2e, 0xe056b67e05e6822b,
    0x94d73f55739d03a0, 0xcd7010bdb69b5a03, 0x455ef9fcd79b82f4,
    0x869cb54a8749c161, 0x38d1a4fa6185d225, 0xb475166f94bbe9bb,
    0xa4143548720959f1, 0x7aed4780ba6b26ba, 0xd0ce264439e02312,
    0x84366d746078d508, 0xa8ce973c72ed17be, 0x21c323a29a430b01,
    0x9962d617e3af80ee, 0xab0ce91d9c8cf75b, 0x530e8ee6d19a4dbc,
    0x2ef68c0cf53f5d72, 0xc03a681640a85506, 0x496e4e9f9c310967,
    0x78580472b59b14a0, 0x273824c23b388577, 0x66bf923ad45cb553,
    0x47ae1a5a2492ba86, 0x35e304569e229659, 0x4765182a46870b6f,
    0x6cbab625e9099412, 0xddac9a2e598522c1, 0x7172086e666624f2,
    0xdf5003ca503b7837, 0x88c0c1db78563d09, 0x58d51865acfc289d,
    0x177671aec65224f1, 0xfb79d8a241e967d7, 0x2be1e101cad9a49a,
    0x6625682f6e29186b, 0x399553457ac06e50, 0x35dffb4c23abb74,
    0x429db2591f54aade, 0xc52802a8037d1009, 0x6acb27381f0b25f3,
    0xf45e2551ee4f823b, 0x8b0ea2d99580c2f7, 0x3bed519cbcb4e1e1,
    0xff452823dbb010a, 0x9d42ed614f3dd267, 0x5b9313c06257c57b,
    0xa114b8008b5e1442, 0xc1fe311c11c13d4b, 0x66e8763ea34c5568,
    0x8b982af1c262f05d, 0xee8876faaa75fbb7, 0x8a62a4d0d172bb2a,
    0xc13d94a3b7449a97, 0x6dbbba9dc15d037c, 0xc786101f1d92e0f1,
    0xd78681a907a0b79b, 0xf61aaf2962c9abb9, 0x2cfd16fcd3cb7ad9,
    0x868c5b6744624d21, 0x25e650899c74ddd7, 0xba042af4a7c37463,
    0x4eb1a539465a3eca, 0xbe09dbf03b05d5ca, 0x774e5a362b5472ba,
    0x47a1221229d183cd, 0x504b0ca18ef5a2df, 0xdffbdfbde2456eb9,
    0x46cd2b2fbee34634, 0xf2aef8fe819d98c3, 0x357f5276d4599d61,
    0x24a5483879c453e3, 0x88026889192b4b9, 0x28da96671782dbec,
    0x4ef37c40588e9aaa, 0x8837b90651bc9fb3, 0xc164f741d3f0e5d6,
    0xbc135a0a704b70ba, 0x69cd868f7622ada, 0xbc37ba89e0b9c0ab,
    0x47c14a01323552f6, 0x4f00794bacee98bb, 0x7107de7d637a69d5,
    0x88af793bb6f2255e, 0xf3c6466b8799b598, 0xc288c616aa7f3b59,
    0x81ca63cf42fca3fd, 0x88d85ace36a2674b, 0xd056bd3792389e7,
    0xe55c396c4e9dd32d, 0xbefb504571e6c0a6, 0x96ab32115e91e8cc,
    0xbf8acb18de8f38d1, 0x66dae58801672606, 0x833b6017872317fb,
    0xb87c16f2d1c92864, 0xdb766a74e58b669c, 0x89659f85c61417be,
    0xc8daad856011ea0c, 0x76a4b565b6fe7eae, 0xa469d085f6237312,
    0xaaf0365683a3e96c, 0x4dbb746f8424f7b8, 0x638755af4e4acc1,
    0x3d7807f5bde64486, 0x17be6d8f5bbb7639, 0x903f0cd44dc35dc,
    0x67b672eafdf1196c, 0xa676ff93ed4c82f1, 0x521d1004c5053d9d,
    0x37ba9ad09ccc9202, 0x84e54d297aacfb51, 0xa0b4b776a143445,
    0x820d471e20b348e, 0x1874383cb83d46dc, 0x97edeec7a1efe11c,
    0xb330e50b1bdc42aa, 0x1dd91955ce70e032, 0xa514cdb88f2939d5,
    0x2791233fd90db9d3, 0x7b670a4cc50f7a9b, 0x77c07d2a05c6dfa5,
    0xe3778b6646d0a6fa, 0xb39c8eda47b56749, 0x933ed448addbef28,
    0xaf846af6ab7d0bf4, 0xe5af208eb666e49, 0x5e6622f73534cd6a,
    0x297daeca42ef5b6e, 0x862daef3d35539a6, 0xe68722498f8e1ea9,
    0x981c53093dc0d572, 0xfa09b0bfbf86fbf5, 0x30b1e96166219f15,
    0x70e7d466bdc4fb83, 0x5a66736e35f2a8e9, 0xcddb59d2b7c1baef,
    0xd6c7d247d26d8996, 0xea4e39eac8de1ba3, 0x539c8bb19fa3aff2,
    0x9f90e4c5fd508d8, 0xa34e5956fbaf3385, 0x2e2f8e151d3ef375,
    0x173691e9b83faec1, 0xb85a8d56bf016379, 0x8382381267408ae3,
    0xb90f901bbdc0096d, 0x7c6ad32933bcec65, 0x76bb5e2f2c8ad595,
    0x390f851a6cf46d28, 0xc3e6064da1c2da72, 0xc52a0c101cfa5389,
    0xd78eaf84a3fbc530, 0x3781b9e2288b997e, 0x73c2f6dea83d05c4,
    0x4228e364c5b5ed7, 0x9d7a3edf0da43911, 0x8edcfeda24686756,
    0x5e7667a7b7a9b3a1, 0x4c4f389fa143791d, 0xb08bc1023da7cddc,
    0x7ab4be3ae529b1cc, 0x754e6132dbe74ff9, 0x71635442a839df45,
    0x2f6fb1643fbe52de, 0x961e0a42cf7a8177, 0xf3b45d83d89ef2ea,
    0xee3de4cf4a6e3e9b, 0xcd6848542c3295e7, 0xe4cee1664c78662f,
    0x9947548b474c68c4, 0x25d73777a5ed8b0b, 0xc915b1d636b7fc,
    0x21c2ba75d9b0d2da, 0x5f6b5dcf608a64a1, 0xdcf333255ff9570c,
    0x633b922418ced4ee, 0xc136dde0b004b34a, 0x58cc83b05d4b2f5a,
    0x5eb424dda28e42d2, 0x62df47369739cd98, 0xb4e0b42485e4ce17,
    0x16e1f0c1f9a8d1e7, 0x8ec3916707560ebf, 0x62ba6e2df2cc9db3,
    0xcbf9f4ff77d83a16, 0x78d9d7d07d2bbcc4, 0xef554ce1e02c41f4,
    0x8d7581127eccf94d, 0xa9b53336cb3c8a05, 0x38c42c0bf45c4f91,
    0x640893cdf4488863, 0x80ec34bc575ea568, 0x39f324f5b48eaa40,
    0xe9d9ed1f8eff527f, 0x9224fc058cc5a214, 0xbaba00b04cfe7741,
    0x309a9f120fcf52af, 0xa558f3ec65626212, 0x424bec8b7adabe2f,
    0x41622513a6aea433, 0xb88da2d5324ca798, 0xd287733b245528a4,
    0x9a44697e6d68aec3, 0x7b1093be2f49bb28, 0x50bbec632e3d8aad,
    0x6cd90723e1ea8283, 0x897b9e7431b02bf3, 0x219efdcb338a7047,
    0x3b0311f0a27c0656, 0xdb17bf91c0db96e7, 0x8cd4fd6b4e85a5b2,
    0xfab071054ba6409d, 0x40d6fe831fa9dfd9, 0xaf358debad7d791e,
    0xeb8d0e25a65e3e58, 0xbbcbd3df14e08580, 0xcf751f27ecdab2b,
    0x2b4da14f2613d8f4,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh64_reference_values() {
        assert_eq!(xxh64(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"a"), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(xxh64(b"abc"), 0x44BC_2CF5_AD77_0999);
        assert_eq!(
            xxh64(b"Nobody inspects the spammish repetition"),
            0xFBCE_A83C_8A37_8BF1
        );
    }

    fn cparams(strategy: Strategy, window_log: u32) -> CParams {
        let mut cp = CParams::for_level(1, 1 << 20);
        cp.strategy = strategy;
        cp.window_log = window_log;
        cp
    }

    fn params(hash_log: u32, bucket_size_log: u32, hash_rate_log: u32) -> LdmParams {
        LdmParams {
            hash_log,
            bucket_size_log,
            min_match_length: 64,
            hash_rate_log,
            window_log: 27,
        }
    }

    /// Every branch of `ZSTD_ldm_adjustParameters`.
    #[test]
    fn adjusted_derives_like_zstd() {
        let derive = LdmParams::requested(0, 0, 0, 0);
        // fast: rate 7 - 1/3, hash log 27 - 7, bucket size log 1 raised to 4
        assert_eq!(
            derive.adjusted(&cparams(Strategy::Fast, 27)),
            params(20, 4, 7)
        );
        // lazy2 (5): rate 7 - 5/3, bucket size log 5
        assert_eq!(
            derive.adjusted(&cparams(Strategy::Lazy2, 27)),
            params(21, 5, 6)
        );
        // btopt (7): rate 7 - 7/3, bucket size log 7
        assert_eq!(
            derive.adjusted(&cparams(Strategy::BtOpt, 27)),
            params(22, 7, 5)
        );
        // btultra2 (9): rate 4, bucket size log capped at 8, and from
        // btultra on half the minimum match length
        assert_eq!(
            derive.adjusted(&cparams(Strategy::BtUltra2, 27)),
            LdmParams {
                min_match_length: 32,
                ..params(23, 8, 4)
            }
        );
        // an explicit hash log sets the rate to the window log above it...
        let fast27 = cparams(Strategy::Fast, 27);
        assert_eq!(
            LdmParams::requested(16, 0, 0, 0).adjusted(&fast27),
            params(16, 4, 11)
        );
        // ... and leaves it 0 when the window log is not above it
        assert_eq!(
            LdmParams::requested(27, 0, 0, 0).adjusted(&fast27),
            params(27, 4, 0)
        );
        // a rate above the window log wraps in U32 and clamps to the max
        assert_eq!(
            LdmParams::requested(0, 0, 0, 25)
                .adjusted(&cparams(Strategy::Fast, 20))
                .hash_log,
            HASHLOG_MAX
        );
        // a small difference clamps to the min
        assert_eq!(
            LdmParams::requested(0, 0, 0, 24).adjusted(&fast27).hash_log,
            HASHLOG_MIN
        );
        // the bucket size log never exceeds the hash log
        assert_eq!(
            LdmParams::requested(6, 0, 8, 0).adjusted(&fast27),
            params(6, 6, 21)
        );
        // explicit values stay
        let explicit = LdmParams::requested(20, 32, 3, 4).adjusted(&fast27);
        assert_eq!((explicit.min_match_length, explicit.hash_rate_log), (32, 4));
    }

    /// Each parameter at both bounds is accepted, one past either panics.
    #[test]
    fn requested_enforces_zstd_bounds() {
        LdmParams::requested(6, 4, 1, 1);
        LdmParams::requested(30, 4096, 8, 25);
        for (h, m, b, r) in [
            (5, 0, 0, 0),
            (31, 0, 0, 0),
            (0, 3, 0, 0),
            (0, 4097, 0, 0),
            (0, 0, 9, 0),
            (0, 0, 0, 26),
        ] {
            let caught = std::panic::catch_unwind(|| LdmParams::requested(h, m, b, r));
            assert!(caught.is_err(), "({h}, {m}, {b}, {r}) accepted");
        }
    }

    fn store(seqs: &[(u32, u32, u32)]) -> RawSeqStore {
        RawSeqStore {
            seqs: seqs
                .iter()
                .map(|&(offset, lit_length, match_length)| RawSeq {
                    offset,
                    lit_length,
                    match_length,
                })
                .collect(),
            ..Default::default()
        }
    }

    /// `maybeSplitSequence` at every boundary of a 10-literal, 70-byte
    /// match followed by one more sequence, with `min_match` 4.
    #[test]
    fn maybe_split_sequence_boundaries() {
        let seqs = [(100, 10, 70), (200, 5, 80)];
        let seq = |offset, lit_length, match_length| RawSeq {
            offset,
            lit_length,
            match_length,
        };
        // (remaining, returned, store afterwards, pos afterwards)
        let cases = [
            // the whole sequence fits
            (80, seq(100, 10, 70), seq(200, 5, 80), 1),
            // the block ends in the literals: rest is literals
            (6, seq(0, 6, 70), seq(100, 4, 70), 0),
            // ... or right after them
            (10, seq(0, 10, 70), seq(100, 0, 70), 0),
            // a cut match under min_match is dropped
            (13, seq(0, 10, 3), seq(100, 0, 67), 0),
            // a cut match of min_match is kept
            (14, seq(100, 10, 4), seq(100, 0, 66), 0),
            // a remainder under min_match becomes the next literals
            (77, seq(100, 10, 67), seq(200, 8, 80), 1),
        ];
        for (remaining, returned, after, pos) in cases {
            let mut s = store(&seqs);
            let got = s.maybe_split_sequence(remaining, 4);
            assert_eq!(got.offset == 0, returned.offset == 0, "{remaining}");
            if got.offset != 0 {
                assert_eq!(got, returned, "{remaining}");
            } else {
                assert_eq!(got.lit_length, 10, "{remaining}");
            }
            assert_eq!(s.pos, pos, "{remaining}");
            assert_eq!(s.seqs[pos], after, "{remaining}");
        }
    }

    /// `ZSTD_ldm_skipRawSeqStoreBytes` at each boundary: inside a sequence,
    /// at its end, past the last one, and with nothing to skip.
    #[test]
    fn skip_raw_seq_store_bytes_boundaries() {
        // lengths 80 and 85
        let seqs = [(100, 10, 70), (200, 5, 80)];
        for (from, nb_bytes, to) in [
            ((0, 0), 0, (0, 0)),
            ((0, 10), 0, (0, 10)),
            ((0, 0), 30, (0, 30)),
            ((0, 30), 50, (1, 0)),
            ((0, 30), 51, (1, 1)),
            ((1, 84), 1, (2, 0)),
            ((1, 84), 7, (2, 0)),
            ((0, 0), 1000, (2, 0)),
            ((2, 0), 5, (2, 0)),
        ] {
            let mut s = store(&seqs);
            (s.pos, s.pos_in_sequence) = from;
            s.skip_raw_seq_store_bytes(nb_bytes);
            assert_eq!((s.pos, s.pos_in_sequence), to, "{from:?} + {nb_bytes}");
        }
    }

    /// From btopt on, [`block_compress`] offers the sequences to the
    /// optimal parser as candidates (`ZSTD_optLdm_*`): a repeat whose source
    /// the binary tree never indexed is found through them alone, cut at
    /// the block end, and resumed by the next block from `pos_in_sequence`.
    /// The parser's lookahead moves its copy past the sequence it loads, so
    /// (as in libzstd) a store's last sequence is offered only where it
    /// covers the block start: without the trailing sequence the second
    /// block gets no candidate at its start and reaches the repeat one byte
    /// later as repcode 1 (rep[0] is the offset since the first block).
    #[test]
    fn opt_parser_takes_ldm_candidates() {
        let n = 96 << 10;
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut src = vec![0u8]; // the window starts at 1
        src.extend((0..n).map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 56) as u8
        }));
        src.extend_from_within(1..1 + n);
        let split = 1 + n + (64 << 10);
        let run = |seqs: &[(u32, u32, u32)]| {
            let cp = CParams::for_level(16, src.len());
            assert!(cp.strategy >= Strategy::BtOpt);
            let mut ms = MatchState::new(cp, 1);
            // The first copy is never inserted into the binary tree.
            ms.next_to_update = 1 + n;
            let mut seqs = store(seqs);
            let mut rep = [1, 4, 8];
            let mut blocks = vec![];
            for block in [1 + n..split, split..src.len()] {
                let mut out = SeqStore::new();
                let anchor =
                    block_compress(&mut seqs, &mut ms, &src, block.clone(), &mut rep, &mut out);
                let found: Vec<_> = out
                    .seqs
                    .iter()
                    .map(|s| (s.lit_len, s.match_len(), s.off_base))
                    .collect();
                blocks.push((found, block.end - anchor, seqs.pos, seqs.pos_in_sequence));
            }
            blocks
        };
        let n32 = n as u32;
        let off_base = offset_to_offbase(n32);
        let first = (vec![(0, 64 << 10, off_base)], 0, 0, 64 << 10);
        let trailer = (1, 1 << 20, 4);
        assert_eq!(
            run(&[(n32, 0, n32), trailer]),
            [first.clone(), (vec![(0, 32 << 10, off_base)], 0, 1, 0)]
        );
        assert_eq!(
            run(&[(n32, 0, n32)]),
            [first, (vec![(1, (32 << 10) - 1, 1)], 0, 1, 0)]
        );
        // Without candidates only short chance matches are found.
        for (found, ..) in run(&[]) {
            assert!(found.iter().all(|&(_, ml, _)| ml < 16), "{found:?}");
        }
    }

    /// Positions far enough back are rejected once the window moves past
    /// them, per chunk (`ZSTD_window_enforceMaxDist`); matches inside the
    /// window extend backwards to `low` but not below.
    #[test]
    fn generate_sequences_respects_window_and_low() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut noise = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 56) as u8
                })
                .collect()
        };
        let a = noise(64 << 10);
        let mut src = a.clone();
        src.extend_from_slice(&noise(200 << 10));
        src.extend_from_slice(&a);
        let p = LdmParams::requested(0, 0, 0, 0).adjusted(&cparams(Strategy::Fast, 27));
        let run = |window_log: u32, first: usize| {
            let mut state = LdmState::new(LdmParams { window_log, ..p }, first);
            let mut out = RawSeqStore::default();
            state.generate_sequences(&src, first..src.len(), usize::MAX, &mut out);
            out.seqs
        };
        let repeat = (264 << 10) as u32;
        // window 1 MiB: `a` repeats as matches at offset 264 KiB covering
        // it all but the first split's backward reach is capped at `low`
        let seqs = run(20, 0);
        assert!(!seqs.is_empty());
        assert!(seqs.iter().all(|s| s.offset == repeat));
        let matched: u32 = seqs.iter().map(|s| s.match_length).sum();
        assert!(matched > (60 << 10), "{matched}");
        // window 256 KiB: the repeat is beyond it
        assert!(run(18, 0).is_empty());
        // starting at 1 (the harness convention) changes only `low`
        assert_eq!(run(20, 1).len(), seqs.len());
    }
}
