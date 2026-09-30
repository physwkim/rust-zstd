//! `zstd_preSplit.c`: pick where to end a full 128 KiB block before match
//! finding, from how much byte statistics change along it
//! (`ZSTD_splitBlock`).
//!
//! A fingerprint is a histogram of hashed 2-byte events (single bytes when
//! the hash log is 8) sampled every `RATE` positions. Two fingerprints are
//! "too different" when their normalized L1 distance reaches a threshold
//! that starts high and relaxes as more chunks agree.

use super::params::Strategy;

/// `THRESHOLD_PENALTY_RATE`.
const THRESHOLD_PENALTY_RATE: u64 = 16;
/// `THRESHOLD_BASE`.
const THRESHOLD_BASE: u64 = THRESHOLD_PENALTY_RATE - 2;
/// `THRESHOLD_PENALTY`.
const THRESHOLD_PENALTY: u64 = 3;
/// `HASHLOG_MAX`.
const HASHLOG_MAX: u32 = 10;
/// `KNUTH`.
const KNUTH: u32 = 0x9e37_79b9;
/// `CHUNKSIZE` of `ZSTD_splitBlock_byChunks`.
const CHUNK_SIZE: usize = 8 << 10;
/// `SEGMENT_SIZE` of `ZSTD_splitBlock_fromBorders`.
const SEGMENT_SIZE: usize = 512;
/// The only block size `ZSTD_splitBlock` accepts.
pub const SPLIT_BLOCK_SIZE: usize = 128 << 10;

/// `Fingerprint`: event counts over the first `1 << hash_log` entries.
#[derive(Clone)]
struct Fingerprint {
    events: [u32; 1 << HASHLOG_MAX],
    nb_events: u64,
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self {
            events: [0; 1 << HASHLOG_MAX],
            nb_events: 0,
        }
    }
}

/// `hash2`: the byte itself for `HASH_LOG == 8`, else a multiplicative hash
/// of 2 little-endian bytes.
#[inline(always)]
fn hash2<const HASH_LOG: u32>(src: &[u8], n: usize) -> usize {
    if HASH_LOG == 8 {
        src[n] as usize
    } else {
        ((u16::from_le_bytes([src[n], src[n + 1]]) as u32).wrapping_mul(KNUTH) >> (32 - HASH_LOG))
            as usize
    }
}

impl Fingerprint {
    /// `recordFingerprint_generic`: `addEvents_generic` over `src` into a
    /// cleared fingerprint. `nb_events` counts `(len - 1) / RATE`, one less
    /// than the samples taken when `RATE` divides `len - 1` unevenly, as in
    /// libzstd.
    fn record<const RATE: usize, const HASH_LOG: u32>(&mut self, src: &[u8]) {
        self.events[..1 << HASH_LOG].fill(0);
        // HASHLENGTH - 1 positions at the end start no event.
        let limit = src.len() - 1;
        let mut n = 0;
        while n < limit {
            self.events[hash2::<HASH_LOG>(src, n)] += 1;
            n += RATE;
        }
        self.nb_events = (limit / RATE) as u64;
    }

    /// `HIST_add` into a cleared byte histogram, with `nb_events` the byte
    /// count.
    fn record_bytes(&mut self, src: &[u8]) {
        self.events[..256].fill(0);
        for &b in src {
            self.events[b as usize] += 1;
        }
        self.nb_events = src.len() as u64;
    }

    /// `mergeEvents`, over the entries `hash_log` can reach (the rest stay
    /// zero in both).
    fn merge(&mut self, other: &Fingerprint, hash_log: u32) {
        for (a, b) in self.events[..1 << hash_log]
            .iter_mut()
            .zip(&other.events[..1 << hash_log])
        {
            *a += b;
        }
        self.nb_events += other.nb_events;
    }

    /// `fpDistance`: `sum |a[n] * nb_b - b[n] * nb_a|`.
    fn distance(&self, other: &Fingerprint, hash_log: u32) -> u64 {
        self.events[..1 << hash_log]
            .iter()
            .zip(&other.events[..1 << hash_log])
            .map(|(&a, &b)| {
                (a as i64 * other.nb_events as i64 - b as i64 * self.nb_events as i64)
                    .unsigned_abs()
            })
            .sum()
    }

    /// `compareFingerprints`: whether `new` is "too different" from `self`.
    fn differs(&self, new: &Fingerprint, penalty: u64, hash_log: u32) -> bool {
        debug_assert!(self.nb_events > 0 && new.nb_events > 0);
        let p50 = self.nb_events * new.nb_events;
        let threshold = p50 * (THRESHOLD_BASE + penalty) / THRESHOLD_PENALTY_RATE;
        self.distance(new, hash_log) >= threshold
    }
}

/// `ZSTD_optimalBlockSize`'s `splitLevels[strat]` and `ZSTD_c_blockSplitterLevel`:
/// the `ZSTD_splitBlock` level for `block_splitter_level` (0 auto by
/// strategy, 1 off, 2..=6 fixed), `None` when off.
pub fn split_level(block_splitter_level: u8, strategy: Strategy) -> Option<u8> {
    assert!(
        block_splitter_level <= 6,
        "block_splitter_level {block_splitter_level} out of range 0..=6"
    );
    match block_splitter_level {
        0 => Some(match strategy {
            Strategy::Fast => 0,
            Strategy::DFast => 1,
            Strategy::Greedy | Strategy::Lazy => 2,
            Strategy::Lazy2 => 3,
        }),
        1 => None,
        n => Some(n - 2),
    }
}

/// `FPStats` plus `fromBorders`' middle segment: the workspace of
/// [`PreSplitter::split_block`], kept across blocks.
#[derive(Clone, Default)]
pub struct PreSplitter {
    past: Fingerprint,
    new: Fingerprint,
    middle: Fingerprint,
}

impl PreSplitter {
    /// `ZSTD_splitBlock`: the size of the block to cut from the start of
    /// `block` (exactly [`SPLIT_BLOCK_SIZE`] bytes), at `level` `0..=4`:
    /// 0 compares the borders, 1..=4 compare 8 KiB chunks sampled every
    /// 43, 11, 5 and 1 positions.
    pub fn split_block(&mut self, block: &[u8], level: u8) -> usize {
        assert_eq!(block.len(), SPLIT_BLOCK_SIZE);
        match level {
            0 => self.by_borders(block),
            1 => self.by_chunks::<43, 8>(block),
            2 => self.by_chunks::<11, 9>(block),
            3 => self.by_chunks::<5, 10>(block),
            4 => self.by_chunks::<1, 10>(block),
            _ => panic!("ZSTD_splitBlock level {level} out of range 0..=4"),
        }
    }

    /// `ZSTD_splitBlock_byChunks`: end the block at the first 8 KiB chunk
    /// too different from everything before it.
    fn by_chunks<const RATE: usize, const HASH_LOG: u32>(&mut self, block: &[u8]) -> usize {
        let mut penalty = THRESHOLD_PENALTY;
        self.past.record::<RATE, HASH_LOG>(&block[..CHUNK_SIZE]);
        let mut pos = CHUNK_SIZE;
        while pos <= block.len() - CHUNK_SIZE {
            self.new
                .record::<RATE, HASH_LOG>(&block[pos..pos + CHUNK_SIZE]);
            if self.past.differs(&self.new, penalty, HASH_LOG) {
                return pos;
            }
            self.past.merge(&self.new, HASH_LOG);
            penalty = penalty.saturating_sub(1);
            pos += CHUNK_SIZE;
        }
        block.len()
    }

    /// `ZSTD_splitBlock_fromBorders`: when the first and last 512 bytes
    /// differ, end the block at 64 KiB if the middle is as far from both,
    /// else at 32 KiB when the middle is closer to the end, 96 KiB when
    /// closer to the start.
    fn by_borders(&mut self, block: &[u8]) -> usize {
        let len = block.len();
        self.past.record_bytes(&block[..SEGMENT_SIZE]);
        self.new.record_bytes(&block[len - SEGMENT_SIZE..]);
        if !self.past.differs(&self.new, 0, 8) {
            return len;
        }
        let mid = len / 2 - SEGMENT_SIZE / 2;
        self.middle.record_bytes(&block[mid..mid + SEGMENT_SIZE]);
        let from_begin = self.past.distance(&self.middle, 8);
        let from_end = self.new.distance(&self.middle, 8);
        let min_distance = (SEGMENT_SIZE * SEGMENT_SIZE / 3) as u64;
        if from_begin.abs_diff(from_end) < min_distance {
            64 << 10
        } else if from_begin > from_end {
            32 << 10
        } else {
            96 << 10
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, mut x: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    /// `addEvents_generic` counts `(len - 1) / RATE` events although it
    /// samples `ceil((len - 1) / RATE)` positions.
    #[test]
    fn record_counts_floor_of_limit_over_rate() {
        let src = noise(CHUNK_SIZE, 1);
        let mut fp = Fingerprint::default();
        fp.record::<43, 8>(&src);
        assert_eq!(fp.nb_events, 190);
        assert_eq!(fp.events[..256].iter().sum::<u32>(), 191);
        fp.record::<1, 10>(&src);
        assert_eq!(fp.nb_events, 8191);
        assert_eq!(fp.events.iter().sum::<u32>(), 8191);
        // hash2 with a hash log above 8 hashes 2 little-endian bytes.
        let h = hash2::<10>(&[0x34, 0x12], 0);
        assert_eq!(h, (0x1234u32.wrapping_mul(KNUTH) >> 22) as usize);
    }

    #[test]
    fn split_level_maps_auto_off_and_fixed() {
        use Strategy::*;
        let auto: Vec<_> = [Fast, DFast, Greedy, Lazy, Lazy2]
            .iter()
            .map(|&s| split_level(0, s))
            .collect();
        assert_eq!(auto, [Some(0), Some(1), Some(2), Some(2), Some(3)]);
        assert_eq!(split_level(1, Fast), None);
        assert_eq!(split_level(2, Lazy2), Some(0));
        assert_eq!(split_level(6, Fast), Some(4));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn split_level_above_six_panics() {
        split_level(7, Strategy::Fast);
    }

    fn text(len: usize) -> Vec<u8> {
        (0..len).map(|i| b"the quick brown foxes"[i % 21]).collect()
    }

    /// A block of period 21 (coprime to every sampling rate) stays whole at
    /// every level; a switch to noise at 40 KiB cuts the chunked levels at
    /// that chunk.
    #[test]
    fn by_chunks_cuts_at_the_first_differing_chunk() {
        let mut ps = PreSplitter::default();
        let same = text(SPLIT_BLOCK_SIZE);
        for level in 0..=4 {
            assert_eq!(ps.split_block(&same, level), SPLIT_BLOCK_SIZE, "L{level}");
        }
        let mut mixed = same.clone();
        mixed[40 << 10..].copy_from_slice(&noise(SPLIT_BLOCK_SIZE - (40 << 10), 7));
        for level in 1..=4 {
            assert_eq!(ps.split_block(&mixed, level), 40 << 10, "L{level}");
        }
    }

    /// `fromBorders` with differing borders: the middle segment decides
    /// between 32, 64 and 96 KiB.
    #[test]
    fn from_borders_picks_the_side_the_middle_resembles() {
        let text = |len: usize| -> Vec<u8> { (0..len).map(|i| b"abcdefgh"[i % 8]).collect() };
        let mut ps = PreSplitter::default();
        let at = |cut: usize, ps: &mut PreSplitter| {
            let mut block = noise(cut, 3);
            block.extend(text(SPLIT_BLOCK_SIZE - cut));
            ps.split_block(&block, 0)
        };
        // Middle is text like the end: cut early.
        assert_eq!(at(16 << 10, &mut ps), 32 << 10);
        // Middle is noise like the start: cut late.
        assert_eq!(at(112 << 10, &mut ps), 96 << 10);
        // Middle segment half noise, half text: equally far from both.
        assert_eq!(at(64 << 10, &mut ps), 64 << 10);
    }
}
