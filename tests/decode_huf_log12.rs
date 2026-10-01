//! Literals coded with 12-bit Huffman codes, which RFC 8878 §4.2.1 caps
//! at 11 bits and libzstd 1.5.7 decodes (HUF_TABLELOG_MAX = 12): frames
//! built from libzstd's own Huffman coder, with compressed and treeless
//! literals of one and four streams, are rejected at both SIMD levels,
//! serial and MT, when any section describes a 12-bit code (R2-7), and
//! otherwise decode as libzstd decodes them.

mod common;

use rust_zstd::decode::{decompress_with_options, DecodeOptions};
use zstd::zstd_safe::zstd_sys as sys;

extern "C" {
    // lib/common/huf.h; linked from zstd-sys's static libzstd.
    fn HUF_buildCTable_wksp(
        tree: *mut usize,
        count: *const u32,
        max_symbol_value: u32,
        max_nb_bits: u32,
        workspace: *mut u64,
        wksp_size: usize,
    ) -> usize;
    fn HUF_writeCTable_wksp(
        dst: *mut u8,
        max_dst_size: usize,
        ctable: *const usize,
        max_symbol_value: u32,
        huff_log: u32,
        workspace: *mut u64,
        wksp_size: usize,
    ) -> usize;
    fn HUF_compress1X_usingCTable(
        dst: *mut u8,
        dst_size: usize,
        src: *const u8,
        src_size: usize,
        ctable: *const usize,
        flags: i32,
    ) -> usize;
    fn HUF_compress4X_usingCTable(
        dst: *mut u8,
        dst_size: usize,
        src: *const u8,
        src_size: usize,
        ctable: *const usize,
        flags: i32,
    ) -> usize;
    fn HUF_selectDecoder(dst_size: usize, c_src_size: usize) -> u32;
}

fn is_error(r: usize) -> bool {
    // SAFETY: a pure function of its argument.
    unsafe { sys::ZSTD_isError(r) != 0 }
}

/// Deterministic LCG (Knuth MMIX constants); returns the high 32 bits.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// A Huffman code libzstd built for the histogram of `symbols`, and the
/// symbols it codes.
struct Code {
    ctable: [usize; 258],
    /// Table log: the longest code length.
    log: u32,
    /// The tree description (HUF_writeCTable).
    desc: Vec<u8>,
    symbols: Vec<u8>,
}

impl Code {
    fn new(symbols: Vec<u8>, max_bits: u32) -> Code {
        let mut counts = [0u32; 256];
        for &s in &symbols {
            counts[usize::from(s)] += 1;
        }
        let max_symbol = counts.iter().rposition(|&c| c > 0).unwrap() as u32;
        let mut ctable = [0usize; 258];
        let mut wksp = [0u64; 2048];
        let mut desc = [0u8; 256];
        // SAFETY: `ctable` holds HUF_CTABLE_SIZE_ST(255) entries, `wksp`
        // exceeds HUF_WORKSPACE_SIZE and the lengths are the buffers' own.
        let (log, n) = unsafe {
            let log = HUF_buildCTable_wksp(
                ctable.as_mut_ptr(),
                counts.as_ptr(),
                max_symbol,
                max_bits,
                wksp.as_mut_ptr(),
                wksp.len() * 8,
            );
            assert!(!is_error(log));
            let n = HUF_writeCTable_wksp(
                desc.as_mut_ptr(),
                desc.len(),
                ctable.as_ptr(),
                max_symbol,
                log as u32,
                wksp.as_mut_ptr(),
                wksp.len() * 8,
            );
            assert!(!is_error(n) && n > 0);
            (log as u32, n)
        };
        assert_eq!(log, max_bits, "the code is shorter than asked");
        Code {
            ctable,
            log,
            desc: desc[..n].to_vec(),
            symbols,
        }
    }

    /// `len` of the coded symbols, from `at` on.
    fn take(&self, at: &mut usize, len: usize) -> Vec<u8> {
        if *at + len > self.symbols.len() {
            *at = 0;
        }
        *at += len;
        self.symbols[*at - len..*at].to_vec()
    }
}

/// Mostly short codes: symbol `k` with probability `2^-(k+1)`, so that
/// the rarest symbols are longer than 12 bits before length limiting.
fn narrow_symbols(rng: &mut Lcg, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let k = (rng.next() | 1 << 8).leading_zeros() as u8;
            k.wrapping_mul(41).wrapping_add(7)
        })
        .collect()
}

/// Nearly flat over 200 symbols, plus ten rare ones: incompressible
/// enough for the single-symbol decoder, and with codes of 12 bits.
fn wide_symbols(rng: &mut Lcg, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let r = rng.next();
            if r % 4096 < 2 {
                200 + (r >> 16) as u8 % 10
            } else {
                (r >> 8) as u8 % 200
            }
        })
        .collect()
}

/// The decoder a literals section reaches: (table log, double-symbol,
/// four streams).
type Path = (u32, bool, bool);

/// Literals section for `lit` coded with `code`: with its description
/// (Compressed_Literals_Block) or reusing the previous table (Treeless),
/// in one stream or four. Also returns whether a description picks the
/// double-symbol decoder (HUF_selectDecoder).
fn literals_section(code: &Code, lit: &[u8], treeless: bool, four: bool) -> (Vec<u8>, bool) {
    let mut streams = vec![0u8; lit.len() + 1024];
    // SAFETY: the buffers and lengths are each other's; `ctable` came
    // from HUF_buildCTable_wksp over a histogram in which every symbol of
    // `lit` occurs.
    let n = unsafe {
        let compress = if four {
            HUF_compress4X_usingCTable
        } else {
            HUF_compress1X_usingCTable
        };
        compress(
            streams.as_mut_ptr(),
            streams.len(),
            lit.as_ptr(),
            lit.len(),
            code.ctable.as_ptr(),
            0,
        )
    };
    assert!(!is_error(n) && n > 0, "HUF_compress of {} bytes", lit.len());
    streams.truncate(n);
    let mut body = if treeless {
        Vec::new()
    } else {
        code.desc.clone()
    };
    body.extend_from_slice(&streams);
    let (regen, comp) = (lit.len() as u64, body.len() as u64);
    // SAFETY: a pure function of its arguments.
    let x2 = four && unsafe { HUF_selectDecoder(lit.len(), body.len()) } == 1;
    let ty = if treeless { 3 } else { 2 };
    let header = if !four {
        assert!(regen < 1 << 10 && comp < 1 << 10);
        (ty | regen << 4 | comp << 14).to_le_bytes()[..3].to_vec()
    } else if regen < 1 << 10 && comp < 1 << 10 {
        (ty | 1 << 2 | regen << 4 | comp << 14).to_le_bytes()[..3].to_vec()
    } else if regen < 1 << 14 && comp < 1 << 14 {
        (ty | 2 << 2 | regen << 4 | comp << 18).to_le_bytes()[..4].to_vec()
    } else {
        assert!(regen < 1 << 18 && comp < 1 << 18);
        (ty | 3 << 2 | regen << 4 | comp << 22).to_le_bytes()[..5].to_vec()
    };
    ([header, body].concat(), x2)
}

/// Frame of compressed blocks, each one literals section and no sequences.
/// Its 1 MiB window lets a block's compressed size exceed the content
/// size, which caps a single-segment frame's blocks.
fn frame(sections: &[Vec<u8>], content_size: usize) -> Vec<u8> {
    let mut f = 0xFD2F_B528u32.to_le_bytes().to_vec();
    // 4-byte Frame_Content_Size; Window_Descriptor 2^(10 + 10).
    f.push(2 << 6);
    f.push(10 << 3);
    f.extend_from_slice(&(content_size as u32).to_le_bytes());
    for (i, s) in sections.iter().enumerate() {
        let last = u32::from(i + 1 == sections.len());
        // The section and the zero Number_of_Sequences byte.
        let size = s.len() as u32 + 1;
        f.extend_from_slice(&(last | 2 << 1 | size << 3).to_le_bytes()[..3]);
        f.extend_from_slice(s);
        f.push(0);
    }
    f
}

/// Build a frame from `plan` (code index, literal count, treeless, four
/// streams), check that libzstd decodes it to the literals, and decode it
/// at both levels, serial and MT: a frame with a 12-bit code is rejected,
/// any other one decodes as libzstd does. Returns the decoders checked.
fn check_frame(
    codes: &[Code],
    cursors: &mut [usize],
    plan: &[(usize, usize, bool, bool)],
) -> Vec<Path> {
    let mut sections = Vec::new();
    let mut want = Vec::new();
    let mut paths = Vec::new();
    let mut table: Option<(usize, bool)> = None;
    for &(c, len, treeless, four) in plan {
        let c = if treeless { table.unwrap().0 } else { c };
        let lit = codes[c].take(&mut cursors[c], len);
        let (section, x2) = literals_section(&codes[c], &lit, treeless, four);
        if !treeless {
            table = Some((c, x2));
        }
        paths.push((codes[c].log, table.unwrap().1, four));
        sections.push(section);
        want.extend_from_slice(&lit);
    }
    let f = frame(&sections, want.len());
    let theirs =
        zstd::bulk::decompress(&f, want.len()).unwrap_or_else(|e| panic!("{plan:?}: libzstd: {e}"));
    assert!(theirs == want, "libzstd's decode differs from the literals");
    let log12 = paths.iter().any(|p| p.0 == 12);
    for simd in [false, true] {
        for min_parallel_blocks in [usize::MAX, 1] {
            let options = DecodeOptions {
                min_parallel_blocks,
                simd,
            };
            let what = format!("{plan:?} simd={simd} mt={min_parallel_blocks}");
            match decompress_with_options(&f, &options) {
                Ok(_) if log12 => panic!("{what}: a 12-bit code is accepted"),
                Ok(ours) => assert!(ours == theirs, "{what}: output differs"),
                Err(e) if !log12 => panic!("{what}: {e}"),
                Err(_) => {}
            }
        }
    }
    // A rejected frame's 11-bit sections were not checked.
    if log12 {
        paths.retain(|p| p.0 == 12);
    }
    paths
}

#[test]
fn twelve_bit_literals_are_rejected() {
    let mut rng = Lcg(0x5DEE_CE66_D1CE_4E5B);
    let narrow = narrow_symbols(&mut rng, 1 << 20);
    let wide = wide_symbols(&mut rng, 1 << 20);
    let codes = [
        Code::new(narrow.clone(), 12),
        Code::new(wide.clone(), 12),
        Code::new(narrow, 11),
        Code::new(wide, 11),
    ];
    let mut cursors = [0usize; 4];
    let mut paths = Vec::new();

    // Every decoder at log 11, each table reused by treeless sections.
    let (n12, w12, n11, w11) = (0, 1, 2, 3);
    paths.extend(check_frame(
        &codes,
        &mut cursors,
        &[
            (n11, 100_000, false, true),
            (0, 900, true, false),
            (0, 30_000, true, true),
            (w11, 60_000, false, true),
            (0, 700, true, false),
            (0, 20_000, true, true),
            (n11, 700, false, false),
            (0, 10_000, true, true),
            (w11, 12, false, true),
            (0, 1, true, false),
        ],
    ));

    // Each 12-bit description alone, and among 11-bit tables.
    for c in [n12, w12] {
        for (len, four) in [(700, false), (12, true), (100_000, true)] {
            paths.extend(check_frame(&codes, &mut cursors, &[(c, len, false, four)]));
        }
    }
    paths.extend(check_frame(
        &codes,
        &mut cursors,
        &[
            (n12, 100_000, false, true),
            (0, 900, true, false),
            (0, 30_000, true, true),
            (w12, 60_000, false, true),
            (0, 700, true, false),
            (0, 20_000, true, true),
            (n11, 50_000, false, true),
            (n12, 700, false, false),
            (0, 10_000, true, true),
            (n12, 200, false, true),
            (0, 65_000, true, true),
            (w11, 40_000, false, true),
            (0, 900, true, false),
            (w12, 12, false, true),
            (0, 1, true, false),
        ],
    ));

    // Random plans: sizes across the segment and stream-length boundaries
    // of both loops; half of them with 11-bit codes only.
    for f in 0..40 {
        let mut plan = Vec::new();
        for b in 0..1 + rng.next() % 10 {
            let treeless = b > 0 && rng.next().is_multiple_of(2);
            let four = !rng.next().is_multiple_of(3);
            let c = if f % 2 == 0 {
                n11 + rng.next() as usize % 2
            } else {
                rng.next() as usize % 4
            };
            let len = match (four, rng.next() % 3) {
                (false, _) => 1 + rng.next() as usize % 700,
                (true, 0) => 12 + rng.next() as usize % 256,
                (true, 1) => 12 + rng.next() as usize % 4096,
                (true, _) => 12 + rng.next() as usize % 100_000,
            };
            plan.push((c, len, treeless, four));
        }
        paths.extend(check_frame(&codes, &mut cursors, &plan));
    }

    for x2 in [false, true] {
        for four in [false, true] {
            for log in [11, 12] {
                let n = paths.iter().filter(|&&p| p == (log, x2, four)).count();
                assert!(n > 0, "no section reached log {log} x2={x2} four={four}");
            }
        }
    }
}
