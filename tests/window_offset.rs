//! The encoder never emits an offset of Window_Size or more: every offset
//! "must be smaller than Window_Size" (RFC 8878 §3.1.1.4,
//! rfc8878.txt:1204-1206), though the decoder accepts exactly Window_Size
//! (tests/decode_window.rs). libzstd's finders reach exactly the window
//! back; ours stop one byte short (`Window::lowest_match_index`).
//!
//! The input repeats a random window-sized block exactly one window
//! after itself, so every byte of the copy has a candidate at distance
//! Window_Size; a parser of the frame's sequences checks every offset.

mod common;

use rust_zstd::compress::block::{build_seq_store, BlockLdm, BlockState};
use rust_zstd::compress::lazy::{default_search_method, SearchMethod};
use rust_zstd::compress::matchstate::MatchState;
use rust_zstd::compress::{CParams, CompressOptions, ParamSwitch, SeqStore, Strategy};
use rust_zstd::constants::{
    LL_BASE, LL_BITS, LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG, LL_FSE_LOG, ML_BASE, ML_BITS,
    ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG, ML_FSE_LOG, OFF_FSE_LOG, OF_DEFAULT_NORM,
    OF_DEFAULT_NORM_LOG,
};
use rust_zstd::decode::parse_fse_header;

/// An FSE decoding table: per state, the symbol, the bits to read and the
/// base of the next state.
struct Fse {
    log: u32,
    cells: Vec<(u8, u32, usize)>,
}

impl Fse {
    /// RFC 8878 §4.1.1: spread the symbols of the normalized counts `norm`
    /// (`-1` is "less than 1") over `1 << log` states.
    fn new(norm: &[i32], log: u32) -> Self {
        let size = 1usize << log;
        let mut symbols = vec![0u8; size];
        let mut next = vec![0u32; norm.len()];
        let mut high = size;
        for (s, &n) in norm.iter().enumerate() {
            if n == -1 {
                high -= 1;
                symbols[high] = s as u8;
                next[s] = 1;
            } else {
                next[s] = n as u32;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mut pos = 0;
        for (s, &n) in norm.iter().enumerate() {
            for _ in 0..n.max(0) {
                symbols[pos] = s as u8;
                pos = (pos + step) & (size - 1);
                while pos >= high {
                    pos = (pos + step) & (size - 1);
                }
            }
        }
        assert_eq!(pos, 0, "counts do not fill the table");
        let cells = symbols
            .iter()
            .map(|&s| {
                let n = next[s as usize];
                next[s as usize] += 1;
                let bits = log - (31 - n.leading_zeros());
                (s, bits, ((n << bits) as usize) - size)
            })
            .collect();
        Fse { log, cells }
    }

    fn default(norm: &[i16], log: u32) -> Self {
        Fse::new(&norm.iter().map(|&n| i32::from(n)).collect::<Vec<_>>(), log)
    }

    fn rle(symbol: u8) -> Self {
        Fse {
            log: 0,
            cells: vec![(symbol, 0, 0)],
        }
    }
}

/// A backward bitstream (RFC 8878 §4.1): read from the last byte's highest
/// set bit down.
struct Bits<'a> {
    buf: &'a [u8],
    left: usize,
}

impl<'a> Bits<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let last = *buf.last().expect("empty bitstream");
        assert_ne!(last, 0, "bitstream without its end mark");
        let left = 8 * (buf.len() - 1) + 7 - last.leading_zeros() as usize;
        Bits { buf, left }
    }

    fn read(&mut self, n: u32) -> u64 {
        assert!(n as usize <= self.left, "bitstream overrun");
        let mut v = 0;
        for _ in 0..n {
            self.left -= 1;
            v = (v << 1) | u64::from(self.buf[self.left / 8] >> (self.left % 8) & 1);
        }
        v
    }
}

/// A frame's Window_Size, the offset of every sequence (repeat offsets
/// resolved, RFC 8878 §3.1.1.5) and the number of bytes it decodes to.
struct Parsed {
    window_size: u64,
    offsets: Vec<u64>,
    decoded: u64,
}

/// The offset of a sequence of Offset_Value `offset_value` after `ll`
/// literals, updating the repeat offsets `rep` (RFC 8878 §3.1.1.5).
fn resolve_offset(rep: &mut [u64; 3], offset_value: u64, ll: u64) -> u64 {
    if offset_value > 3 {
        *rep = [offset_value - 3, rep[0], rep[1]];
        return rep[0];
    }
    let idx = offset_value as usize - 1 + usize::from(ll == 0);
    let o = if idx == 3 { rep[0] - 1 } else { rep[idx] };
    if idx > 0 {
        *rep = [o, rep[0], if idx > 1 { rep[1] } else { rep[2] }];
    }
    o
}

/// Parse one frame without dictionary.
fn parse_frame(f: &[u8]) -> Parsed {
    assert_eq!(f[..4], [0x28, 0xb5, 0x2f, 0xfd]);
    let fhd = f[4];
    assert_eq!(fhd & 3, 0, "dictionary id");
    let single_segment = fhd & 0x20 != 0;
    let mut p = 5;
    let mut window_size = 0;
    if !single_segment {
        let base = 1u64 << (10 + (f[p] >> 3));
        window_size = base + base / 8 * u64::from(f[p] & 7);
        p += 1;
    }
    let fcs_len = [usize::from(single_segment), 2, 4, 8][usize::from(fhd >> 6)];
    let mut fcs = (0..fcs_len).fold(0u64, |v, i| v | u64::from(f[p + i]) << (8 * i));
    if fcs_len == 2 {
        fcs += 256;
    }
    p += fcs_len;
    if single_segment {
        window_size = fcs;
    }

    let mut parsed = Parsed {
        window_size,
        offsets: Vec::new(),
        decoded: 0,
    };
    let mut rep = [1u64, 4, 8];
    let mut tables: [Option<Fse>; 3] = [None, None, None];
    loop {
        let h = u32::from_le_bytes([f[p], f[p + 1], f[p + 2], 0]);
        p += 3;
        let size = (h >> 3) as usize;
        match (h >> 1) & 3 {
            0 => p += size,
            1 => p += 1,
            2 => {
                parse_block(&f[p..p + size], &mut rep, &mut tables, &mut parsed);
                p += size;
            }
            _ => panic!("reserved block type"),
        }
        if (h >> 1) & 3 != 2 {
            parsed.decoded += size as u64;
        }
        if h & 1 != 0 {
            break;
        }
    }
    if fhd & 4 != 0 {
        p += 4;
    }
    assert_eq!(p, f.len(), "bytes after the frame");
    parsed
}

/// Parse a compressed block's sequences (RFC 8878 §3.1.1.3), appending
/// their offsets.
fn parse_block(b: &[u8], rep: &mut [u64; 3], tables: &mut [Option<Fse>; 3], out: &mut Parsed) {
    let (lit_type, size_format) = (b[0] & 3, b[0] >> 2 & 3);
    let le = |n: usize| (0..n).fold(0u64, |v, i| v | u64::from(b[i]) << (8 * i));
    let (literals, mut q) = if lit_type < 2 {
        let (regenerated, header) = match size_format {
            0 | 2 => (le(1) >> 3, 1),
            1 => (le(2) >> 4, 2),
            _ => (le(3) >> 4, 3),
        };
        let body = if lit_type == 0 {
            regenerated as usize
        } else {
            1
        };
        (regenerated, header + body)
    } else {
        let (header, bits) = match size_format {
            0 | 1 => (3, 10),
            2 => (4, 14),
            _ => (5, 18),
        };
        let v = le(header);
        let mask = (1 << bits) - 1;
        (v >> 4 & mask, header + (v >> (4 + bits) & mask) as usize)
    };

    let nb_seq = match b[q] {
        n @ 0..=127 => {
            q += 1;
            usize::from(n)
        }
        n @ 128..=254 => {
            q += 2;
            (usize::from(n - 128) << 8) + usize::from(b[q - 1])
        }
        255 => {
            q += 3;
            usize::from(b[q - 2]) + (usize::from(b[q - 1]) << 8) + 0x7f00
        }
    };
    if nb_seq == 0 {
        assert_eq!(q, b.len());
        out.decoded += literals;
        return;
    }
    let modes = b[q];
    q += 1;
    // Literals_Lengths, Offsets, Match_Lengths, in this order.
    let defaults = [
        (&LL_DEFAULT_NORM[..], LL_DEFAULT_NORM_LOG, LL_FSE_LOG),
        (&OF_DEFAULT_NORM[..], OF_DEFAULT_NORM_LOG, OFF_FSE_LOG),
        (&ML_DEFAULT_NORM[..], ML_DEFAULT_NORM_LOG, ML_FSE_LOG),
    ];
    for (k, (norm, log, max_log)) in defaults.into_iter().enumerate() {
        match modes >> (6 - 2 * k) & 3 {
            0 => tables[k] = Some(Fse::default(norm, log)),
            1 => {
                tables[k] = Some(Fse::rle(b[q]));
                q += 1;
            }
            2 => {
                let (log, norm, n) = parse_fse_header(&b[q..], max_log as u8).unwrap();
                tables[k] = Some(Fse::new(&norm, u32::from(log)));
                q += n;
            }
            _ => assert!(tables[k].is_some(), "Repeat_Mode without a table"),
        }
    }
    let [Some(ll_t), Some(of_t), Some(ml_t)] = &*tables else {
        unreachable!()
    };

    let mut bits = Bits::new(&b[q..]);
    let mut ll_s = bits.read(ll_t.log) as usize;
    let mut of_s = bits.read(of_t.log) as usize;
    let mut ml_s = bits.read(ml_t.log) as usize;
    let mut literals_used = 0;
    for i in 0..nb_seq {
        let of_code = u32::from(of_t.cells[of_s].0);
        let ml_code = usize::from(ml_t.cells[ml_s].0);
        let ll_code = usize::from(ll_t.cells[ll_s].0);
        let offset_value = (1u64 << of_code) + bits.read(of_code);
        let ml = u64::from(ML_BASE[ml_code]) + bits.read(u32::from(ML_BITS[ml_code]));
        let ll = u64::from(LL_BASE[ll_code]) + bits.read(u32::from(LL_BITS[ll_code]));
        if i + 1 < nb_seq {
            for (t, s) in [(ll_t, &mut ll_s), (ml_t, &mut ml_s), (of_t, &mut of_s)] {
                let (_, n, base) = t.cells[*s];
                *s = base + bits.read(n) as usize;
            }
        }
        let offset = resolve_offset(rep, offset_value, ll);
        out.decoded += ll;
        literals_used += ll;
        assert!(
            (1..=out.decoded).contains(&offset),
            "offset {offset} at {}",
            out.decoded
        );
        out.offsets.push(offset);
        out.decoded += ml;
    }
    assert_eq!(bits.left, 0, "bitstream not consumed");
    out.decoded += literals - literals_used;
}

/// A random block of `window` bytes, the same block again one window
/// later, 4 KiB of it again `window - 1` bytes after its copy, then 4 KiB
/// of text, whose short repeats every level matches.
fn winedge(window: usize) -> Vec<u8> {
    let block = common::lcg_bytes(window, 77);
    let mut data = block.clone();
    data.extend_from_slice(&block);
    data.extend_from_slice(&block[1..4097]);
    data.extend(b"window offsets stay below Window_Size. ".repeat(105));
    data
}

/// Compress `data` with `opts`, check both decoders return it, and return
/// the largest offset after checking every offset is below Window_Size.
fn max_offset(what: &str, data: &[u8], opts: &CompressOptions) -> u64 {
    let frame = rust_zstd::compress_with(data, opts);
    common::assert_round_trip(what, data, &frame);
    let parsed = parse_frame(&frame);
    assert_eq!(parsed.decoded, data.len() as u64, "{what}: parser");
    assert!(!parsed.offsets.is_empty(), "{what}: no sequences");
    let max = *parsed.offsets.iter().max().unwrap();
    assert!(
        max < parsed.window_size,
        "{what}: offset {max} of Window_Size {}",
        parsed.window_size
    );
    max
}

/// Every block compressor, with both hash chain and row finders where the
/// strategy has them, at an 8 KiB window: the blocks' sequences, as the
/// frame loop drives a `MatchState`, never reach the window size, and
/// from level 5 on reach the repeat one byte short of it. Cheap enough
/// for debug builds, unlike the frame-level tests below.
#[test]
fn finders_stop_short_of_the_window() {
    let window_log = 13;
    let w = 1usize << window_log;
    let data = winedge(w);
    for level in 1..=22 {
        let mut cp = CParams::for_level(level, data.len());
        cp.window_log = window_log;
        let methods = match cp.strategy {
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => {
                vec![SearchMethod::HashChain, SearchMethod::RowHash]
            }
            _ => vec![default_search_method(&cp)],
        };
        for method in methods {
            let mut ms = MatchState::new_for(cp, 0, method);
            let mut rep = BlockState::initial().rep;
            let mut decoder_rep = [1, 4, 8];
            let mut store = SeqStore::new();
            let mut max = 0;
            for start in (0..data.len()).step_by(w) {
                let end = (start + w).min(data.len());
                let block = ms.enter_block(start..end);
                let Some(next) = build_seq_store(
                    &mut ms,
                    &data,
                    block,
                    rep,
                    &mut store,
                    &mut BlockLdm::Off,
                    None,
                ) else {
                    continue;
                };
                rep = next;
                let mut pos = start as u64;
                for seq in &store.seqs {
                    pos += u64::from(seq.lit_len);
                    let offset =
                        resolve_offset(&mut decoder_rep, seq.off_base.into(), seq.lit_len.into());
                    assert!(
                        offset <= pos && offset < w as u64,
                        "L{level} {method:?}: offset {offset} at {pos}"
                    );
                    max = max.max(offset);
                    pos += u64::from(seq.match_len());
                }
            }
            if level >= 5 {
                assert_eq!(max, w as u64 - 1, "L{level} {method:?}");
            }
        }
    }
}

/// Levels 1-22, serial and as ZSTDMT jobs of one window whose overlap is
/// the whole window, so that a job's first candidates lie exactly one
/// window back in its prefix. Serial, every level from 5 on reaches the
/// repeat one byte short of the window (a job's loaded prefix leaves the
/// hash chain and row finders none of it). Level 22's input enables long
/// distance matching.
#[test]
#[ignore = "about 4 minutes in release; inputs up to 256 MiB"]
fn offsets_stay_below_window_size() {
    for level in 1..=22 {
        let w = 1usize << CParams::for_level(level, 1 << 30).window_log;
        let data = winedge(w);
        for (job, job_size, overlap_log) in [("", None, 0), (" MT", Some(w), 9)] {
            let what = format!("L{level}{job}");
            let opts = CompressOptions {
                level,
                job_size,
                overlap_log,
                ..CompressOptions::default()
            };
            let max = max_offset(&what, &data, &opts);
            if level >= 5 && job_size.is_none() {
                assert_eq!(max, w as u64 - 1, "{what}: the repeat one byte short");
            }
        }
    }
}

/// Long distance matching, serial and ZSTDMT, at a level of every block
/// compressor family it hands literals to.
#[test]
#[ignore = "about a minute in release; 256 MiB inputs"]
fn ldm_offsets_stay_below_window_size() {
    let data = winedge(1 << 27);
    for level in [1, 3, 6, 13, 16] {
        for job_size in [None, Some(0)] {
            let opts = CompressOptions {
                level,
                job_size,
                ldm: ParamSwitch::Enable,
                ..CompressOptions::default()
            };
            max_offset(&format!("L{level} LDM {job_size:?}"), &data, &opts);
        }
    }
}
