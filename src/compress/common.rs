//! Helpers shared by the block compressors: ports of `ZSTD_hashPtr`,
//! `ZSTD_count` and the unaligned little-endian reads of
//! `zstd_compress_internal.h` / `mem.h` (libzstd 1.5.7).

/// `HASH_READ_SIZE`: the hash functions read up to 8 bytes.
pub const HASH_READ_SIZE: usize = 8;
/// `kSearchStrength`.
pub const K_SEARCH_STRENGTH: u32 = 8;

/// `MEM_read16`.
#[inline(always)]
pub fn read16(src: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes(src[pos..pos + 2].try_into().unwrap())
}

/// `MEM_read32` / `MEM_readLE32`.
#[inline(always)]
pub fn read32(src: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(src[pos..pos + 4].try_into().unwrap())
}

/// `MEM_read64` / `MEM_readLE64`.
#[inline(always)]
pub fn read64(src: &[u8], pos: usize) -> u64 {
    u64::from_le_bytes(src[pos..pos + 8].try_into().unwrap())
}

/// `PREFETCH_L1(src + pos)`: a cache hint, no-op when `pos` is outside
/// `src` or on targets without the intrinsic.
#[inline(always)]
pub fn prefetch(src: &[u8], pos: usize) {
    #[cfg(target_arch = "x86_64")]
    if pos < src.len() {
        // SAFETY: `pos < src.len()`, so the pointer is inside `src`; the
        // prefetch instruction never faults and does not access memory
        // in a way visible to the program.
        unsafe {
            core::arch::x86_64::_mm_prefetch(
                src.as_ptr().add(pos) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            )
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (src, pos);
    }
}

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;
const PRIME7: u64 = 58295818150454627;
const PRIME8: u64 = 0xCF1BBCDCB7A56463;

/// `ZSTD_hashPtr(p, hBits, mls)`: hash of the `MLS` bytes at `src[pos]`,
/// `MLS` in `4..=8`; any other value hashes 4 bytes like libzstd's
/// `default` arm. `MLS >= 5` reads 8 bytes, so `pos + 8 <= src.len()` is
/// required. The result is `< 1 << hbits` by construction.
#[inline(always)]
pub fn hash_ptr<const MLS: u32>(src: &[u8], pos: usize, hbits: u32) -> usize {
    match MLS {
        5 => (((read64(src, pos) << (64 - 40)).wrapping_mul(PRIME5)) >> (64 - hbits)) as usize,
        6 => (((read64(src, pos) << (64 - 48)).wrapping_mul(PRIME6)) >> (64 - hbits)) as usize,
        7 => (((read64(src, pos) << (64 - 56)).wrapping_mul(PRIME7)) >> (64 - hbits)) as usize,
        8 => ((read64(src, pos).wrapping_mul(PRIME8)) >> (64 - hbits)) as usize,
        _ => (read32(src, pos).wrapping_mul(PRIME4) >> (32 - hbits)) as usize,
    }
}

/// `ZSTD_count(pIn, pMatch, pInLimit)`: length of the common prefix of
/// `src[a..limit]` and `src[b..]`, `b < a <= limit`.
#[inline]
pub fn count(src: &[u8], a: usize, b: usize, limit: usize) -> usize {
    debug_assert!(b < a && a <= limit && limit <= src.len());
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
    if a < limit && src[b] == src[a] {
        a += 1;
    }
    a - start
}

/// Block-finder test harness shared by the fast and double-fast tests:
/// drives a finder block by block on a persistent [`MatchState`], checks
/// every offset against the window, and reconstructs each block from its
/// [`SeqStore`] against the real history.
#[cfg(test)]
pub mod testutil {
    use crate::compress::matchstate::MatchState;
    use crate::compress::params::CParams;
    use crate::compress::seqstore::{SeqStore, ZSTD_REP_NUM};
    use std::ops::Range;

    pub type BlockFn =
        fn(&mut MatchState, &[u8], Range<usize>, &mut [u32; 3], &mut SeqStore) -> usize;
    pub type PrefixFn = fn(&mut MatchState, &[u8], Range<usize>);

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
    /// `MatchState` with `window_low`, after loading `src[window_low..
    /// job_start]` as the prefix. Every block is reconstructed and compared.
    pub fn roundtrip_job(
        f: &Finder,
        src: &[u8],
        cp: CParams,
        block_size: usize,
        window_low: usize,
        job_start: usize,
        rep: [u32; 3],
    ) -> Stats {
        let mut ms = MatchState::new(cp, window_low);
        if job_start > window_low {
            (f.load_prefix)(&mut ms, src, window_low..job_start);
        }
        let mut rep = rep;
        let mut store = SeqStore::new();
        let mut stats = Stats::default();
        let mut start = job_start;
        while start < src.len() {
            let end = (start + block_size).min(src.len());
            store.clear();
            let rep_in = rep;
            let anchor = (f.compress_block)(&mut ms, src, start..end, &mut rep, &mut store);
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
                    off >= 1 && off <= pos && pos - off >= window_low,
                    "block {start}..{end} seq {i}: offset {off} at {pos} reaches below window_low {window_low}"
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
                "block {start}..{end}: reconstruction differs (window_low {window_low}, level params {cp:?})"
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
        window_low: usize,
        rep: [u32; 3],
    ) -> Stats {
        roundtrip_job(f, src, cp, block_size, window_low, 0, rep)
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
        assert_eq!(
            hash_ptr::<4>(&src, 3, 14),
            (v32.wrapping_mul(2654435761) >> 18) as usize
        );
        assert_eq!(
            hash_ptr::<5>(&src, 3, 17),
            (((v64 << 24).wrapping_mul(889523592379)) >> 47) as usize
        );
        assert_eq!(
            hash_ptr::<6>(&src, 3, 16),
            (((v64 << 16).wrapping_mul(227718039650203)) >> 48) as usize
        );
        assert_eq!(
            hash_ptr::<7>(&src, 3, 14),
            (((v64 << 8).wrapping_mul(58295818150454627)) >> 50) as usize
        );
        assert_eq!(
            hash_ptr::<8>(&src, 3, 17),
            ((v64.wrapping_mul(0xCF1BBCDCB7A56463)) >> 47) as usize
        );
        // mls 3 falls into the 4-byte arm
        assert_eq!(hash_ptr::<3>(&src, 3, 14), hash_ptr::<4>(&src, 3, 14));
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
            assert_eq!(count(&src, 100, 0, 200), n, "n={n}");
            // limit cuts the match short at every tail size
            for limit in 100..=100 + n {
                assert_eq!(
                    count(&src, 100, 0, limit),
                    limit - 100,
                    "n={n} limit={limit}"
                );
            }
        }
    }
}
