//! Per-block sequence store: port of `SeqDef` / `SeqStore_t` /
//! `ZSTD_storeSeq` (zstd_compress_internal.h).
//!
//! Offsets use libzstd's `offBase` sum type:
//! `1..=3` is a repcode (`REPCODE_TO_OFFBASE`), any larger value is
//! `offset + ZSTD_REP_NUM` (`OFFSET_TO_OFFBASE`). With `lit_len == 0` the
//! decoder shifts repcodes by one (repcode 1 means `rep[1]`, 3 means
//! `rep[0] - 1`); the match finders emit exactly what the decoder expects, so
//! this store never reinterprets them.

use crate::constants::ZSTD_MINMATCH;

/// `ZSTD_REP_NUM`: number of repeat offsets.
pub const ZSTD_REP_NUM: u32 = 3;
/// `WILDCOPY_OVERLENGTH`: bytes [`SeqStore::store_seq`] may read past the
/// literals and write past the literal buffer's end; the buffer is oversized
/// by as much (`ZSTD_resetCCtx_internal`: `litStart` is
/// `blockSize + WILDCOPY_OVERLENGTH` bytes).
pub const WILDCOPY_OVERLENGTH: usize = 32;

/// `ZSTD_copy16`.
///
/// # Safety
/// 16 bytes readable at `src` and writable at `dst`, not overlapping.
#[inline(always)]
unsafe fn copy16(dst: *mut u8, src: *const u8) {
    std::ptr::copy_nonoverlapping(src, dst, 16);
}
/// `REPCODE1_TO_OFFBASE`.
pub const REPCODE1_TO_OFFBASE: u32 = 1;
/// `REPCODE2_TO_OFFBASE`.
pub const REPCODE2_TO_OFFBASE: u32 = 2;
/// `REPCODE3_TO_OFFBASE`.
pub const REPCODE3_TO_OFFBASE: u32 = 3;

/// `OFFSET_TO_OFFBASE(o)`: `o` is a raw back-reference distance (`> 0`).
#[inline]
pub fn offset_to_offbase(offset: u32) -> u32 {
    debug_assert!(offset > 0);
    offset + ZSTD_REP_NUM
}

/// `REPCODE_TO_OFFBASE(r)`: `r` is a repcode id in `1..=3`.
#[inline]
pub fn repcode_to_offbase(repcode: u32) -> u32 {
    debug_assert!((1..=ZSTD_REP_NUM).contains(&repcode));
    repcode
}

/// `OFFBASE_IS_OFFSET(o)`.
#[inline]
pub fn offbase_is_offset(off_base: u32) -> bool {
    off_base > ZSTD_REP_NUM
}

/// `OFFBASE_IS_REPCODE(o)`.
#[inline]
pub fn offbase_is_repcode(off_base: u32) -> bool {
    (1..=ZSTD_REP_NUM).contains(&off_base)
}

/// `OFFBASE_TO_OFFSET(o)`.
#[inline]
pub fn offbase_to_offset(off_base: u32) -> u32 {
    debug_assert!(offbase_is_offset(off_base));
    off_base - ZSTD_REP_NUM
}

/// `OFFBASE_TO_REPCODE(o)`: returns the id `1..=3`.
#[inline]
pub fn offbase_to_repcode(off_base: u32) -> u32 {
    debug_assert!(offbase_is_repcode(off_base));
    off_base
}

/// `ZSTD_updateRep`: update the repeat-offset history after a sequence with
/// `off_base` and `ll0 == (lit_len == 0)`.
#[inline]
pub fn update_rep(rep: &mut [u32; 3], off_base: u32, ll0: bool) {
    if offbase_is_offset(off_base) {
        rep[2] = rep[1];
        rep[1] = rep[0];
        rep[0] = offbase_to_offset(off_base);
    } else {
        let rep_code = offbase_to_repcode(off_base) - 1 + ll0 as u32;
        if rep_code > 0 {
            let current = if rep_code == ZSTD_REP_NUM {
                rep[0] - 1
            } else {
                rep[rep_code as usize]
            };
            rep[2] = if rep_code >= 2 { rep[1] } else { rep[2] };
            rep[1] = rep[0];
            rep[0] = current;
        }
    }
}

/// `SeqDef`: one sequence. `ml_base == match_len - ZSTD_MINMATCH`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seq {
    pub lit_len: u32,
    pub off_base: u32,
    pub ml_base: u32,
}

impl Seq {
    /// Actual match length.
    #[inline]
    pub fn match_len(&self) -> u32 {
        self.ml_base + ZSTD_MINMATCH as u32
    }
}

/// `SeqStore_t`: literals and sequences of one block. Trailing literals
/// (after the last sequence) are appended to `lits` by the driver via
/// `ZSTD_storeLastLiterals` semantics.
#[derive(Clone, Debug, Default)]
pub struct SeqStore {
    pub lits: Vec<u8>,
    pub seqs: Vec<Seq>,
}

impl SeqStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-size for a block of `block_size` bytes: `block_size +
    /// WILDCOPY_OVERLENGTH` literals, so that [`SeqStore::store_seq`] never
    /// grows the buffer.
    pub fn with_capacity(block_size: usize) -> Self {
        let mut store = Self::new();
        store.reserve(block_size);
        store
    }

    /// Grow to [`SeqStore::with_capacity`]'s sizes for `block_size` if
    /// smaller; existing allocations are kept.
    pub fn reserve(&mut self, block_size: usize) {
        let lits = block_size + WILDCOPY_OVERLENGTH;
        self.lits
            .reserve_exact(lits.saturating_sub(self.lits.len()));
        let seqs = block_size / 4 + 1;
        self.seqs
            .reserve_exact(seqs.saturating_sub(self.seqs.len()));
    }

    /// `ZSTD_storeSeq(seqStore, litLength, literals = src + anchor, litLimit
    /// = src + lit_limit, offBase, matchLength)`: copy `lit_len` literals
    /// starting at `src[anchor]` and append the sequence. `lit_limit` is the
    /// block end; literals ending `WILDCOPY_OVERLENGTH` before it are copied
    /// 16 bytes at a time (`ZSTD_copy16` + `ZSTD_wildcopy`, over-reading and
    /// over-writing up to `WILDCOPY_OVERLENGTH` bytes), later ones
    /// byte-exact (`ZSTD_safecopyLiterals`). `match_len >= ZSTD_MINMATCH`,
    /// `off_base >= 1`, `anchor + lit_len <= lit_limit <= src.len()`.
    #[inline(always)]
    pub fn store_seq(
        &mut self,
        src: &[u8],
        anchor: usize,
        lit_len: usize,
        lit_limit: usize,
        off_base: u32,
        match_len: usize,
    ) {
        debug_assert!(off_base >= 1);
        debug_assert!(match_len >= ZSTD_MINMATCH);
        debug_assert!(anchor + lit_len <= lit_limit && lit_limit <= src.len());
        let len = self.lits.len();
        let lit_end = anchor + lit_len;
        // Common case we can use wildcopy: the reads end before `lit_end +
        // WILDCOPY_OVERLENGTH <= src.len()`, the writes before `len +
        // lit_len + WILDCOPY_OVERLENGTH <= capacity` (a store from
        // `with_capacity` always has that room).
        if lit_end + WILDCOPY_OVERLENGTH <= lit_limit.min(src.len())
            && self.lits.capacity() - len >= lit_len + WILDCOPY_OVERLENGTH
        {
            // SAFETY: the bounds just tested; `src` and `lits` are distinct
            // allocations.
            unsafe {
                let mut ip = src.as_ptr().add(anchor);
                let mut op = self.lits.as_mut_ptr().add(len);
                copy16(op, ip);
                if lit_len > 16 {
                    // ZSTD_wildcopy(lit + 16, literals + 16, litLength - 16,
                    // ZSTD_no_overlap)
                    let oend = op.add(lit_len);
                    op = op.add(16);
                    ip = ip.add(16);
                    copy16(op, ip);
                    if lit_len - 16 > 16 {
                        op = op.add(16);
                        ip = ip.add(16);
                        loop {
                            copy16(op, ip);
                            copy16(op.add(16), ip.add(16));
                            op = op.add(32);
                            ip = ip.add(32);
                            if op >= oend {
                                break;
                            }
                        }
                    }
                }
                // The first `lit_len` bytes after `len` were written above.
                self.lits.set_len(len + lit_len);
            }
        } else {
            self.store_literals_safe(src, anchor, lit_len);
        }
        self.seqs.push(Seq {
            lit_len: lit_len as u32,
            off_base,
            ml_base: (match_len - ZSTD_MINMATCH) as u32,
        });
    }

    /// `ZSTD_safecopyLiterals`, plus the growth a store built without
    /// `with_capacity` needs; kept out of the match finders' loops.
    #[cold]
    #[inline(never)]
    fn store_literals_safe(&mut self, src: &[u8], anchor: usize, lit_len: usize) {
        self.lits.reserve(lit_len + WILDCOPY_OVERLENGTH);
        self.lits.extend_from_slice(&src[anchor..anchor + lit_len]);
    }

    /// `ZSTD_resetSeqStore`.
    pub fn clear(&mut self) {
        self.lits.clear();
        self.seqs.clear();
    }

    /// Execute the stored sequences exactly like the decoder and return the
    /// block content they reproduce. `src_history` is the already-decoded
    /// window preceding the block, `rep` the repeat-offset history in force
    /// at the block start. Panics on an offset the decoder would reject.
    #[cfg(test)]
    pub fn reconstruct(&self, src_history: &[u8], mut rep: [u32; 3]) -> Vec<u8> {
        let mut out = src_history.to_vec();
        let mut lit_pos = 0usize;
        for (i, s) in self.seqs.iter().enumerate() {
            let ll = s.lit_len as usize;
            out.extend_from_slice(&self.lits[lit_pos..lit_pos + ll]);
            lit_pos += ll;
            let offset = if s.off_base > ZSTD_REP_NUM {
                let o = s.off_base - ZSTD_REP_NUM;
                rep = [o, rep[0], rep[1]];
                o
            } else {
                assert!(s.off_base >= 1, "seq {i}: off_base 0");
                let idx = (s.off_base - 1) as usize + (ll == 0) as usize;
                let o = match idx {
                    0 => rep[0],
                    3 => {
                        assert!(rep[0] > 1, "seq {i}: rep[0]-1 with rep[0] = {}", rep[0]);
                        rep[0] - 1
                    }
                    k => rep[k],
                };
                if idx > 0 {
                    rep = [o, rep[0], if idx >= 2 { rep[1] } else { rep[2] }];
                }
                o
            };
            assert!(
                offset >= 1 && offset as usize <= out.len(),
                "seq {i}: offset {offset} outside history of {} bytes",
                out.len()
            );
            let start = out.len() - offset as usize;
            for k in 0..s.match_len() as usize {
                let b = out[start + k];
                out.push(b);
            }
        }
        out.extend_from_slice(&self.lits[lit_pos..]);
        out.split_off(src_history.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_rep_and_offset_semantics() {
        // "abcabcabc" + "xyz": (ll=3, raw offset 3, ml=6) then literals.
        let src = b"abcabcabcxyz";
        let mut st = SeqStore::new();
        st.store_seq(src, 0, 3, src.len(), offset_to_offbase(3), 6);
        st.lits.extend_from_slice(b"xyz");
        assert_eq!(st.reconstruct(&[], [1, 4, 8]), src);

        // History supplies the window; every step picks bytes that differ
        // between the right and the wrong repcode interpretation.
        let hist = b"0123456789";
        let mut st = SeqStore::new();
        let mut rep = [1u32, 4, 8];
        // "AB" + repcode 2 (rep[1] == 4), ml 3: reaches into history -> "89A"
        st.store_seq(b"AB", 0, 2, b"AB".len(), REPCODE2_TO_OFFBASE, 3);
        update_rep(&mut rep, REPCODE2_TO_OFFBASE, false);
        assert_eq!(rep, [4, 1, 8]);
        // ll == 0, repcode 1 -> rep[1] == 1: "AAA" (offset 4 would give "B89")
        st.store_seq(b"", 0, 0, b"".len(), REPCODE1_TO_OFFBASE, 3);
        update_rep(&mut rep, REPCODE1_TO_OFFBASE, true);
        assert_eq!(rep, [1, 4, 8]);
        // "C" + repcode 3 (rep[2] == 8): "B89"
        st.store_seq(b"C", 0, 1, b"C".len(), REPCODE3_TO_OFFBASE, 3);
        update_rep(&mut rep, REPCODE3_TO_OFFBASE, false);
        assert_eq!(rep, [8, 1, 4]);
        // ll == 0, repcode 3 -> rep[0] - 1 == 7: "AAAC" (offset 4 would give "CB89")
        st.store_seq(b"", 0, 0, b"".len(), REPCODE3_TO_OFFBASE, 4);
        update_rep(&mut rep, REPCODE3_TO_OFFBASE, true);
        assert_eq!(rep, [7, 8, 1]);
        let out = st.reconstruct(hist, [1, 4, 8]);
        assert_eq!(&out, b"AB89AAAACB89AAAC" as &[u8]);
        update_rep(&mut rep, offset_to_offbase(100), false);
        assert_eq!(rep, [100, 7, 8]);
    }
}
