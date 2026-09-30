//! Helpers shared by the block compressors: ports of `ZSTD_hashPtr`,
//! `ZSTD_count` and the unaligned little-endian reads of
//! `zstd_compress_internal.h` / `mem.h` (libzstd 1.5.7).

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::Avx2;
use fearless_simd::{Fallback, Level};
use std::marker::PhantomData;
use std::sync::OnceLock;

/// `HASH_READ_SIZE`: the hash functions read up to 8 bytes.
pub const HASH_READ_SIZE: usize = 8;
/// `kSearchStrength`.
pub const K_SEARCH_STRENGTH: u32 = 8;

/// The input as the match finders address it: `window.base`. Byte `i` of
/// the view is at index `i + lo()`, the index [`MatchState`] assigns to
/// it (the window's first byte is `window_low`); indices below `lo()` or
/// from `end()` on are not readable. The finders get their
/// view from [`MatchState::view`], so every index they compute is in the
/// state's index space.
///
/// [`MatchState`]: super::matchstate::MatchState
/// [`MatchState::view`]: super::matchstate::MatchState::view
#[derive(Clone, Copy)]
pub struct Src<'a> {
    /// Address of index 0; outside the input, hence `wrapping_add` only.
    base: *const u8,
    lo: usize,
    end: usize,
    data: PhantomData<&'a [u8]>,
}

impl<'a> Src<'a> {
    /// `data[from..]` with `data[from]` at index `lo`.
    #[inline]
    pub(super) fn new(data: &'a [u8], from: usize, lo: usize) -> Self {
        assert!(from <= data.len());
        Src {
            base: data.as_ptr().wrapping_add(from).wrapping_sub(lo),
            lo,
            end: lo + (data.len() - from),
            data: PhantomData,
        }
    }

    /// The same bytes with every index `shift` higher (`base -= shift`).
    #[inline]
    pub(super) fn rebased(self, shift: usize) -> Self {
        Src {
            base: self.base.wrapping_sub(shift),
            lo: self.lo + shift,
            end: self.end + shift,
            data: PhantomData,
        }
    }

    /// The lowest readable index.
    #[inline(always)]
    pub fn lo(self) -> usize {
        self.lo
    }

    /// The index past the last readable byte.
    #[inline(always)]
    pub fn end(self) -> usize {
        self.end
    }

    /// `base + idx`.
    #[inline(always)]
    pub fn ptr(self, idx: usize) -> *const u8 {
        self.base.wrapping_add(idx)
    }

    /// The bytes at indices `a..b`.
    #[inline]
    pub fn slice(self, a: usize, b: usize) -> &'a [u8] {
        assert!(self.lo <= a && a <= b && b <= self.end);
        // SAFETY: `lo..end` maps onto the borrowed input.
        unsafe { std::slice::from_raw_parts(self.ptr(a), b - a) }
    }

    /// The byte at index `idx`.
    #[inline]
    pub fn at(self, idx: usize) -> u8 {
        assert!(self.lo <= idx && idx < self.end);
        // SAFETY: as in `slice`.
        unsafe { *self.ptr(idx) }
    }
}

/// `MEM_read16`.
///
/// # Safety
/// `src.lo() <= pos` and `pos + 2 <= src.end()`.
#[inline(always)]
pub unsafe fn read16(src: Src, pos: usize) -> u16 {
    debug_assert!(src.lo <= pos && pos + 2 <= src.end);
    u16::from_le_bytes(*(src.ptr(pos) as *const [u8; 2]))
}

/// `MEM_read32` / `MEM_readLE32`.
///
/// # Safety
/// `src.lo() <= pos` and `pos + 4 <= src.end()`.
#[inline(always)]
pub unsafe fn read32(src: Src, pos: usize) -> u32 {
    debug_assert!(src.lo <= pos && pos + 4 <= src.end);
    u32::from_le_bytes(*(src.ptr(pos) as *const [u8; 4]))
}

/// `MEM_read64` / `MEM_readLE64`.
///
/// # Safety
/// `src.lo() <= pos` and `pos + 8 <= src.end()`.
#[inline(always)]
pub unsafe fn read64(src: Src, pos: usize) -> u64 {
    debug_assert!(src.lo <= pos && pos + 8 <= src.end);
    u64::from_le_bytes(*(src.ptr(pos) as *const [u8; 8]))
}

/// The byte at index `pos`.
///
/// # Safety
/// `src.lo() <= pos < src.end()`.
#[inline(always)]
pub unsafe fn byte(src: Src, pos: usize) -> u8 {
    debug_assert!(src.lo <= pos && pos < src.end);
    *src.ptr(pos)
}

/// `table[h]`.
///
/// # Safety
/// `h < table.len()`.
#[inline(always)]
pub unsafe fn tget(table: &[u32], h: usize) -> usize {
    debug_assert!(h < table.len());
    *table.get_unchecked(h) as usize
}

/// `table[h] = v`.
///
/// # Safety
/// `h < table.len()`.
#[inline(always)]
pub unsafe fn tset(table: &mut [u32], h: usize, v: usize) {
    debug_assert!(h < table.len());
    *table.get_unchecked_mut(h) = v as u32;
}

/// Is the table entry `idx` a usable candidate for position `cur`, i.e.
/// `low <= idx < cur`? `low` is the lowest valid index (`low <= cur`).
///
/// The upper bound is always true for a table filled by this module and
/// only makes a stale entry of a misused `MatchState` a miss instead of an
/// out-of-bounds read. It is folded into one unsigned compare on purpose:
/// with two compares LLVM turns the branch-free candidate select of the
/// callers (the C `ZSTD_selectAddr` idiom) into a branch, which costs the
/// double-fast finder 18% throughput.
#[inline(always)]
pub fn candidate_valid(idx: usize, low: usize, cur: usize) -> bool {
    debug_assert!(low <= cur);
    idx.wrapping_sub(low) < cur - low
}

/// `PREFETCH_L1(base + pos)`: a cache hint, no-op on targets without the
/// intrinsic. Like C's, it is unbounded: the hint never faults, so `pos`
/// may lie outside `src`, and a bound check here costs fast's search loop
/// its register allocation.
#[inline(always)]
pub fn prefetch(src: Src, pos: usize) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is a hint that never faults, whatever the address;
    // `Src::ptr` keeps the address computation defined.
    unsafe {
        core::arch::x86_64::_mm_prefetch(src.ptr(pos) as *const i8, core::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (src, pos);
    }
}

/// `PREFETCH_L1(&slice[idx])`. `idx` may point past the end: C prefetches
/// `base + matchIndex` and table rows the same way, and the hint never
/// faults.
#[inline(always)]
pub fn prefetch_l1<T>(slice: &[T], idx: usize) {
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

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;
const PRIME7: u64 = 58295818150454627;
const PRIME8: u64 = 0xCF1BBCDCB7A56463;

/// `ZSTD_hashPtr(p, hBits, mls)`: hash of the `MLS` bytes at index `pos`,
/// `MLS` in `4..=8`; any other value hashes 4 bytes like libzstd's
/// `default` arm. The result is `< 1 << hbits` by construction
/// (`1 <= hbits <= 32`).
///
/// # Safety
/// `src.lo() <= pos` and `pos + HASH_READ_SIZE <= src.end()` (`MLS >= 5`
/// reads 8 bytes).
#[inline(always)]
pub unsafe fn hash_ptr<const MLS: u32>(src: Src, pos: usize, hbits: u32) -> usize {
    debug_assert!((1..=32).contains(&hbits));
    match MLS {
        5 => (((read64(src, pos) << (64 - 40)).wrapping_mul(PRIME5)) >> (64 - hbits)) as usize,
        6 => (((read64(src, pos) << (64 - 48)).wrapping_mul(PRIME6)) >> (64 - hbits)) as usize,
        7 => (((read64(src, pos) << (64 - 56)).wrapping_mul(PRIME7)) >> (64 - hbits)) as usize,
        8 => ((read64(src, pos).wrapping_mul(PRIME8)) >> (64 - hbits)) as usize,
        _ => (read32(src, pos).wrapping_mul(PRIME4) >> (32 - hbits)) as usize,
    }
}

/// `ZSTD_count(pIn, pMatch, pInLimit)`: length of the common prefix of
/// the bytes at indices `a..limit` and `b..`.
///
/// # Safety
/// `src.lo() <= b < a <= limit <= src.end()`.
#[inline]
pub unsafe fn count(src: Src, a: usize, b: usize, limit: usize) -> usize {
    debug_assert!(src.lo <= b && b < a && a <= limit && limit <= src.end);
    let start = a;
    let (mut a, mut b) = (a, b);
    // pIn < pInLimit - 7
    if a + 8 <= limit {
        let diff = read64(src, b) ^ read64(src, a);
        if diff != 0 {
            return (diff.trailing_zeros() >> 3) as usize;
        }
        a += 8;
        b += 8;
        while a + 8 <= limit {
            let diff = read64(src, b) ^ read64(src, a);
            if diff == 0 {
                a += 8;
                b += 8;
                continue;
            }
            a += (diff.trailing_zeros() >> 3) as usize;
            return a - start;
        }
    }
    if a + 4 <= limit && read32(src, b) == read32(src, a) {
        a += 4;
        b += 4;
    }
    if a + 2 <= limit && read16(src, b) == read16(src, a) {
        a += 2;
        b += 2;
    }
    if a < limit && byte(src, b) == byte(src, a) {
        a += 1;
    }
    a - start
}

/// The SIMD level of this machine, detected once (`Level::new`).
pub fn simd_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

/// [`count`] for one fearless_simd level. A block compressor is
/// monomorphized over the implementor and compiled with its target
/// features, so the choice is made once per block, never per call.
pub trait MatchCount: Copy {
    /// `ZSTD_count`: as [`count`], same result.
    ///
    /// # Safety
    /// As [`count`]; the witness `self` proves the CPU has the features.
    unsafe fn count(self, src: Src, a: usize, b: usize, limit: usize) -> usize;
}

impl MatchCount for Fallback {
    #[inline(always)]
    unsafe fn count(self, src: Src, a: usize, b: usize, limit: usize) -> usize {
        count(src, a, b, limit)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl MatchCount for Avx2 {
    #[inline(always)]
    unsafe fn count(self, src: Src, a: usize, b: usize, limit: usize) -> usize {
        // SAFETY: `self` proves AVX2; the bounds are the caller's.
        count_avx2(src, a, b, limit)
    }
}

/// [`count`] with a 32-byte loop between C's first 8-byte step and its
/// 8-byte loop: `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8`, the first
/// clear bit is the first differing byte. libzstd has no such path; the
/// length is the same by construction, and a match that ends in the first
/// 8 bytes costs what it does in [`count`].
///
/// # Safety
/// As [`count`], and the CPU must support AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn count_avx2(src: Src, a: usize, b: usize, limit: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    debug_assert!(src.lo <= b && b < a && a <= limit && limit <= src.end);
    let start = a;
    let (mut a, mut b) = (a, b);
    // SAFETY (every read): `lo <= b < a` and each read ends at or before
    // `limit <= src.end()`; the 32-byte loads are unaligned.
    unsafe {
        if a + 8 <= limit {
            let diff = read64(src, b) ^ read64(src, a);
            if diff != 0 {
                return (diff.trailing_zeros() >> 3) as usize;
            }
            a += 8;
            b += 8;
            while a + 32 <= limit {
                let va = _mm256_loadu_si256(src.ptr(a).cast::<__m256i>());
                let vb = _mm256_loadu_si256(src.ptr(b).cast::<__m256i>());
                let eq = _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32;
                if eq != u32::MAX {
                    return a + (!eq).trailing_zeros() as usize - start;
                }
                a += 32;
                b += 32;
            }
            while a + 8 <= limit {
                let diff = read64(src, b) ^ read64(src, a);
                if diff != 0 {
                    return a + (diff.trailing_zeros() >> 3) as usize - start;
                }
                a += 8;
                b += 8;
            }
        }
        if a + 4 <= limit && read32(src, b) == read32(src, a) {
            a += 4;
            b += 4;
        }
        if a + 2 <= limit && read16(src, b) == read16(src, a) {
            a += 2;
            b += 2;
        }
        if a < limit && byte(src, b) == byte(src, a) {
            a += 1;
        }
    }
    a - start
}

/// `ZSTD_count` at `level`: the per-call dispatch of [`MatchCount`], for
/// callers that are not monomorphized over the level.
///
/// # Safety
/// As [`count`].
#[inline]
pub unsafe fn count_with(level: Level, src: Src, a: usize, b: usize, limit: usize) -> usize {
    match level {
        // SAFETY: fearless_simd constructs the witness only after detecting
        // AVX2 on this CPU.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(w) => w.count(src, a, b, limit),
        _ => count(src, a, b, limit),
    }
}

/// Block-finder test harness shared by the fast and double-fast tests:
/// drives a finder block by block on a persistent [`MatchState`], checks
/// every offset against the window, and reconstructs each block from its
/// [`SeqStore`] against the real history.
#[cfg(test)]
pub mod testutil {
    use super::Src;
    use crate::compress::matchstate::{Block, EnteredPrefix, MatchState};
    use crate::compress::params::CParams;
    use crate::compress::seqstore::{SeqStore, ZSTD_REP_NUM};
    use std::ops::Range;
    use zstd::zstd_safe::zstd_sys as sys;

    /// `ZSTD_cParam_getBounds(param)` of the linked libzstd, which is built
    /// for the same target as this crate.
    pub fn c_bounds(param: sys::ZSTD_cParameter) -> (i32, i32) {
        // SAFETY: reads no memory of ours.
        let bounds = unsafe { sys::ZSTD_cParam_getBounds(param) };
        assert_eq!(bounds.error, 0, "{param:?}");
        (bounds.lowerBound, bounds.upperBound)
    }

    /// Whether `ZSTD_CCtx_setParameter(param, value)` succeeds.
    pub fn c_accepts(param: sys::ZSTD_cParameter, value: i32) -> bool {
        // SAFETY: the context is used only here.
        unsafe {
            let cctx = sys::ZSTD_createCCtx();
            let r = sys::ZSTD_CCtx_setParameter(cctx, param, value);
            sys::ZSTD_freeCCtx(cctx);
            sys::ZSTD_isError(r) == 0
        }
    }

    pub type BlockFn = fn(&mut MatchState, Src, Block, &mut [u32; 3], &mut SeqStore) -> usize;
    pub type PrefixFn = fn(&mut MatchState, Src, EnteredPrefix);

    /// Run the finder `f` on positions `block` of `data`, returning the
    /// anchor position.
    pub fn run_block(
        f: BlockFn,
        ms: &mut MatchState,
        data: &[u8],
        block: Range<usize>,
        rep: &mut [u32; 3],
        store: &mut SeqStore,
    ) -> usize {
        let entered = ms.enter_block(block);
        let (src, block) = ms.start_block(data, entered);
        let anchor = f(ms, src, block, rep, store);
        ms.pos(anchor)
    }

    pub struct Finder {
        pub compress_block: BlockFn,
        pub load_prefix: PrefixFn,
    }

    #[derive(Debug, Default)]
    pub struct Stats {
        pub seqs: usize,
        /// Sequences whose match starts before the current block.
        pub cross_block_matches: usize,
        /// Sequences whose match starts before the job (in the loaded prefix).
        pub prefix_matches: usize,
    }

    /// Decoder-side offset resolution (`ZSTD_execSequence` semantics).
    fn resolve_offset(rep: &mut [u32; 3], off_base: u32, lit_len: u32) -> u32 {
        if off_base > ZSTD_REP_NUM {
            let o = off_base - ZSTD_REP_NUM;
            *rep = [o, rep[0], rep[1]];
            return o;
        }
        assert!(off_base >= 1, "off_base 0");
        let idx = (off_base - 1) as usize + (lit_len == 0) as usize;
        let o = match idx {
            0 => rep[0],
            3 => rep[0] - 1,
            k => rep[k],
        };
        if idx > 0 {
            *rep = [o, rep[0], if idx >= 2 { rep[1] } else { rep[2] }];
        }
        o
    }

    /// Compress `src[job_start..]` in blocks of `block_size` on a
    /// `MatchState` whose window starts at `origin`, after loading
    /// `src[origin..job_start]` as the prefix. Every block is reconstructed
    /// and compared.
    pub fn roundtrip_job(
        f: &Finder,
        src: &[u8],
        cp: CParams,
        block_size: usize,
        origin: usize,
        job_start: usize,
        rep: [u32; 3],
    ) -> Stats {
        let mut ms = MatchState::new(cp, origin);
        if let Some(prefix) = ms.enter_prefix(origin..job_start) {
            let view = ms.view(src);
            (f.load_prefix)(&mut ms, view, prefix);
        }
        let mut rep = rep;
        let mut store = SeqStore::new();
        let mut stats = Stats::default();
        let mut start = job_start;
        while start < src.len() {
            let end = (start + block_size).min(src.len());
            store.clear();
            let rep_in = rep;
            let anchor = run_block(
                f.compress_block,
                &mut ms,
                src,
                start..end,
                &mut rep,
                &mut store,
            );
            assert!(
                anchor >= start && anchor <= end,
                "anchor {anchor} outside {start}..{end}"
            );
            store.lits.extend_from_slice(&src[anchor..end]);

            let mut pos = start;
            let mut r = rep_in;
            for (i, s) in store.seqs.iter().enumerate() {
                pos += s.lit_len as usize;
                let off = resolve_offset(&mut r, s.off_base, s.lit_len) as usize;
                assert!(
                    off >= 1 && off <= pos && pos - off >= origin,
                    "block {start}..{end} seq {i}: offset {off} at {pos} reaches below the origin {origin}"
                );
                assert!(
                    off <= 1usize << cp.window_log,
                    "block {start}..{end} seq {i}: offset {off} exceeds the window"
                );
                if pos - off < start {
                    stats.cross_block_matches += 1;
                }
                if pos - off < job_start {
                    stats.prefix_matches += 1;
                }
                pos += s.match_len() as usize;
                assert!(
                    pos <= end,
                    "block {start}..{end} seq {i}: runs past the block"
                );
                stats.seqs += 1;
            }
            let got = store.reconstruct(&src[..start], rep_in);
            assert!(
                got == src[start..end],
                "block {start}..{end}: reconstruction differs (origin {origin}, level params {cp:?})"
            );
            // The fast strategies track offset_1/offset_2 only and never
            // emit repcode 3, so rep[2] is left untouched, as in C.
            assert_eq!(
                r[..2],
                rep[..2],
                "block {start}..{end}: rep history returned to the caller differs from the decoder's"
            );
            start = end;
        }
        stats
    }

    /// [`roundtrip_job`] from position 0 with no prefix.
    pub fn roundtrip_blocks(
        f: &Finder,
        src: &[u8],
        cp: CParams,
        block_size: usize,
        rep: [u32; 3],
    ) -> Stats {
        roundtrip_job(f, src, cp, block_size, 0, 0, rep)
    }

    /// The crate's own `.rs` sources, concatenated in path order.
    pub fn crate_sources() -> Vec<u8> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut paths = Vec::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    paths.push(path);
                }
            }
        }
        paths.sort();
        let mut data = Vec::new();
        for p in paths {
            data.extend_from_slice(&std::fs::read(p).unwrap());
        }
        assert!(data.len() > 200_000);
        data
    }

    /// The running test binary (an ELF).
    pub fn current_exe_bytes() -> Vec<u8> {
        let data = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        assert_eq!(&data[..4], b"\x7fELF");
        data
    }

    /// Deterministic pseudo-text: words from a small vocabulary, numbers
    /// and occasional binary runs, so that matches of every offset class
    /// (short, long, repcode, cross-block) occur.
    pub fn synthetic_text(len: usize, seed: u64) -> Vec<u8> {
        const WORDS: [&str; 24] = [
            "the",
            "block",
            "compressor",
            "hash",
            "table",
            "match",
            "offset",
            "literal",
            "sequence",
            "window",
            "prefix",
            "repcode",
            "length",
            "anchor",
            "position",
            "entropy",
            "huffman",
            "fse",
            "stream",
            "frame",
            "zstd",
            "rust",
            "port",
            "faithful",
        ];
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut out = Vec::with_capacity(len + 64);
        while out.len() < len {
            let r = next();
            match r % 16 {
                0 => out.extend_from_slice(format!("{} ", r % 100_000).as_bytes()),
                1 => {
                    let n = (r >> 8) % 40 + 4;
                    let b = (r >> 16) as u8;
                    out.extend(std::iter::repeat_n(b, n as usize));
                }
                2 => out.extend_from_slice(&(r >> 3).to_le_bytes()[..3]),
                3 => out.extend_from_slice(b".\n"),
                _ => {
                    out.extend_from_slice(WORDS[(r >> 4) as usize % WORDS.len()].as_bytes());
                    out.push(b' ');
                }
            }
        }
        out.truncate(len);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_libzstd_formulas() {
        let src: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37) ^ 0x5a).collect();
        let v32 = u32::from_le_bytes(src[3..7].try_into().unwrap());
        let v64 = u64::from_le_bytes(src[3..11].try_into().unwrap());
        // SAFETY: 3 + 8 <= 32.
        let (h3, h4, h5, h6, h7, h8) = unsafe {
            (
                hash_ptr::<3>(Src::new(&src, 0, 0), 3, 14),
                hash_ptr::<4>(Src::new(&src, 0, 0), 3, 14),
                hash_ptr::<5>(Src::new(&src, 0, 0), 3, 17),
                hash_ptr::<6>(Src::new(&src, 0, 0), 3, 16),
                hash_ptr::<7>(Src::new(&src, 0, 0), 3, 14),
                hash_ptr::<8>(Src::new(&src, 0, 0), 3, 17),
            )
        };
        assert_eq!(h4, (v32.wrapping_mul(2654435761) >> 18) as usize);
        assert_eq!(
            h5,
            (((v64 << 24).wrapping_mul(889523592379)) >> 47) as usize
        );
        assert_eq!(
            h6,
            (((v64 << 16).wrapping_mul(227718039650203)) >> 48) as usize
        );
        assert_eq!(
            h7,
            (((v64 << 8).wrapping_mul(58295818150454627)) >> 50) as usize
        );
        assert_eq!(h8, ((v64.wrapping_mul(0xCF1BBCDCB7A56463)) >> 47) as usize);
        // mls 3 falls into the 4-byte arm
        assert_eq!(h3, h4);
    }

    /// Every [`MatchCount`] level of this CPU and [`count_with`] agree with
    /// [`count`] for every match length, alignment and `limit` across the
    /// 8- and 32-byte step boundaries.
    #[test]
    fn count_levels_match_scalar() {
        let mut levels = vec![Level::fallback()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: detected just above.
            levels.push(Level::Avx2(unsafe { Avx2::new_unchecked() }));
        }
        let base: Vec<u8> = (0..300u32).map(|i| (i * 131 + 7) as u8).collect();
        for off in [1usize, 3, 8, 31, 32, 33, 100] {
            for n in [
                0usize, 1, 7, 8, 9, 15, 16, 31, 32, 33, 39, 40, 41, 63, 64, 65, 72, 100, 150,
            ] {
                let a = 140;
                let mut src = base.clone();
                for i in 0..n.min(src.len() - a) {
                    src[a + i] = src[a - off + i];
                }
                if a + n < src.len() {
                    src[a + n] = src[a - off + n].wrapping_add(1);
                }
                for limit in a..=src.len() {
                    // SAFETY: `a - off < a <= limit <= src.len()`.
                    let want = unsafe { count(Src::new(&src, 0, 0), a, a - off, limit) };
                    assert_eq!(want, n.min(limit - a), "off={off} n={n} limit={limit}");
                    for &level in &levels {
                        // SAFETY: as above.
                        let got =
                            unsafe { count_with(level, Src::new(&src, 0, 0), a, a - off, limit) };
                        assert_eq!(got, want, "{level:?} off={off} n={n} limit={limit}");
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if let Level::Avx2(w) = level {
                            // SAFETY: as above; `w` was built after detection.
                            let got = unsafe { w.count(Src::new(&src, 0, 0), a, a - off, limit) };
                            assert_eq!(got, want, "avx2 off={off} n={n} limit={limit}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn count_every_length_and_tail_shape() {
        // src = pattern P at 0 and a copy at 100 that diverges after n bytes.
        for n in 0..40usize {
            let mut src = vec![0u8; 200];
            for (i, b) in src.iter_mut().enumerate().take(100) {
                *b = (i * 7 + 3) as u8;
            }
            for i in 0..n {
                src[100 + i] = src[i];
            }
            src[100 + n] = src[n].wrapping_add(1);
            // SAFETY: 0 < 100 <= limit <= 200 == src.len().
            unsafe {
                assert_eq!(count(Src::new(&src, 0, 0), 100, 0, 200), n, "n={n}");
                // limit cuts the match short at every tail size
                for limit in 100..=100 + n {
                    assert_eq!(
                        count(Src::new(&src, 0, 0), 100, 0, limit),
                        limit - 100,
                        "n={n} limit={limit}"
                    );
                }
            }
        }
    }
}
