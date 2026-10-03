//! Dictionary compression against libzstd 1.5.7 on small-sample corpora,
//! the case dictionaries are for: a structured dictionary trained by
//! libzstd (`ZDICT_trainFromBuffer`) and a raw-content one, on held-out
//! samples at levels 1, 3, 9 and 19, and on a grid of levels and input
//! sizes across the sizes where libzstd changes how it uses a dictionary.
//!
//! The gate on every frame: libzstd and our decoder decode it with the
//! dictionary, and it is at most [`common::size_limit`] of libzstd's frame
//! with the same parameters: `ZSTD_compress2` with `ZSTD_CCtx_refCDict`,
//! whose semantics [`CompressOptions::dict`] follows, and the same
//! `ZSTD_c_forceAttachDict`: its default, which attaches a dictionary's
//! tables for small inputs and searches them in place, for our
//! [`DictAttach::Auto`], and `ZSTD_dictForceCopy` for our
//! [`DictAttach::Copy`]. `ZSTD_compress_usingDict` and
//! `ZSTD_compress_usingCDict` frames are reported beside ours. Raw
//! prefixes are gated against `ZSTD_CCtx_refPrefix`. Streams with a
//! dictionary or a prefix are our one-shot frames when their size is
//! pledged, and pass the same gate against libzstd's stream either way,
//! without its input buffering ([`common::c_stream_unbuffered`]).

mod common;

use rust_zstd::compress::{
    compress_with_dict, compress_with_prefix, CompressDict, CompressError, CompressOptions,
    Compressor, DictAttach, Encoder, EndDirective, ParamSwitch, JOBSIZE_MIN,
};
use rust_zstd::decode::{decompress_with_dict, DecodeDict};
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use zstd::zstd_safe::zstd_sys::{self as sys, ZSTD_cParameter as P};
use zstd::zstd_safe::{self, CCtx, CParameter, DCtx};

/// The levels of the held-out sample gate.
const LEVELS: [i32; 4] = [1, 3, 9, 19];

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

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// JSON records of a web service, one to four per sample.
fn json_samples(n: usize) -> Vec<Vec<u8>> {
    let mut rng = Lcg(0x4a53_4f4e);
    let names = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];
    let cities = ["Seoul", "Busan", "Lisbon", "Oslo", "Austin", "Kyoto"];
    let tags = ["red", "green", "blue", "beta", "admin", "trial", "eu"];
    (0..n)
        .map(|_| {
            let records: Vec<String> = (0..1 + rng.below(4))
                .map(|_| {
                    let name = rng.pick(&names);
                    format!(
                        "{{\"id\":{},\"user\":\"{name}{}\",\"email\":\"{name}@example.com\",\
                         \"created\":\"2026-{:02}-{:02}T{:02}:{:02}:{:02}Z\",\
                         \"tags\":[\"{}\",\"{}\"],\"active\":{},\"score\":{}.{},\
                         \"address\":{{\"city\":\"{}\",\"zip\":\"{:05}\"}}}}",
                        rng.below(1_000_000),
                        rng.below(100),
                        1 + rng.below(12),
                        1 + rng.below(28),
                        rng.below(24),
                        rng.below(60),
                        rng.below(60),
                        rng.pick(&tags),
                        rng.pick(&tags),
                        rng.below(2) == 0,
                        rng.below(100),
                        rng.below(10),
                        rng.pick(&cities),
                        rng.below(100_000),
                    )
                })
                .collect();
            format!("[{}]", records.join(",")).into_bytes()
        })
        .collect()
}

/// Service log excerpts of 3 to 40 lines.
fn log_samples(n: usize) -> Vec<Vec<u8>> {
    let mut rng = Lcg(0x4c4f_4753);
    let levels = ["INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"];
    let methods = ["GET", "GET", "POST", "PUT", "DELETE"];
    let paths = [
        "/api/v1/items",
        "/api/v1/users",
        "/api/v2/orders",
        "/healthz",
        "/static/app.js",
    ];
    (0..n)
        .map(|_| {
            let mut s = String::new();
            for _ in 0..3 + rng.below(38) {
                s += &format!(
                    "2026-09-{:02} {:02}:{:02}:{:02}.{:03} {:5} [worker-{}] request_id={:08x} \
                     method={} path={}/{} status={} latency_ms={}\n",
                    1 + rng.below(30),
                    rng.below(24),
                    rng.below(60),
                    rng.below(60),
                    rng.below(1000),
                    rng.pick(&levels),
                    rng.below(16),
                    rng.next(),
                    rng.pick(&methods),
                    rng.pick(&paths),
                    rng.below(5000),
                    [200, 200, 200, 201, 304, 404, 500][rng.below(7)],
                    rng.below(900),
                );
            }
            s.into_bytes()
        })
        .collect()
}

/// This crate's sources cut into pieces of 200 bytes to 6 KiB.
fn source_samples() -> Vec<Vec<u8>> {
    fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    files.sort();
    let mut rng = Lcg(0x5352_4321);
    let mut out = Vec::new();
    for file in files {
        let text = std::fs::read(file).unwrap();
        let mut at = 0;
        while at < text.len() {
            let len = (200 + rng.below(6 << 10)).min(text.len() - at);
            out.push(text[at..at + len].to_vec());
            at += len;
        }
    }
    out
}

struct Corpus {
    name: &'static str,
    train: Vec<Vec<u8>>,
    test: Vec<Vec<u8>>,
}

/// The corpora, each with a training set and held-out samples; the last
/// held-out input joins samples to over 128 KiB.
fn corpora() -> Vec<Corpus> {
    let split = |name, mut samples: Vec<Vec<u8>>, n_test: usize| {
        let test_samples = samples.split_off(samples.len() - n_test);
        let mut test: Vec<Vec<u8>> = test_samples.iter().step_by(3).cloned().collect();
        let mut big = Vec::new();
        for s in samples.iter().rev().cycle() {
            if big.len() > 200 << 10 {
                break;
            }
            big.extend_from_slice(s);
        }
        test.push(big);
        Corpus {
            name,
            train: samples,
            test,
        }
    };
    vec![
        split("json", json_samples(1500), 60),
        split("log", log_samples(800), 60),
        split("source", source_samples(), 60),
    ]
}

/// The dictionaries of a corpus: one libzstd trains from it, and raw
/// content of the training samples.
fn dictionaries(corpus: &Corpus) -> Vec<(&'static str, Vec<u8>)> {
    let trained = zstd::dict::from_samples(&corpus.train, 16 << 10).expect("train");
    let mut raw: Vec<u8> = corpus.train.iter().rev().flatten().copied().collect();
    raw.truncate(16 << 10);
    vec![("trained", trained), ("raw", raw)]
}

/// `ZSTD_dictAttachPref_e`, for `ZSTD_c_forceAttachDict`.
#[derive(Clone, Copy, Debug)]
enum Attach {
    /// `ZSTD_dictDefaultAttach`.
    Default = 0,
    /// `ZSTD_dictForceCopy`.
    ForceCopy = 2,
}

impl Attach {
    /// Our preference of the same meaning.
    fn ours(self) -> DictAttach {
        match self {
            Attach::Default => DictAttach::Auto,
            Attach::ForceCopy => DictAttach::Copy,
        }
    }
}

/// Options with `dict` and the preference `attach` means.
fn dict_opts(dict: &Arc<CompressDict>, attach: Attach) -> CompressOptions {
    CompressOptions {
        dict: Some(dict.clone()),
        dict_attach: attach.ours(),
        ..Default::default()
    }
}

/// Our frame of `src` with `dict` and the preference `attach` means.
fn ours_frame(src: &[u8], dict: &Arc<CompressDict>, attach: Attach) -> Vec<u8> {
    Compressor::new(dict_opts(dict, attach)).compress_to_vec(src)
}

/// A frame libzstd's `f` writes into a buffer of `ZSTD_compressBound(len)`
/// bytes on a fresh context.
fn lib_frame(f: impl FnOnce(*mut sys::ZSTD_CCtx, &mut [u8]) -> usize, len: usize) -> Vec<u8> {
    // SAFETY: the context is used only here.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let mut out = vec![0u8; sys::ZSTD_compressBound(len)];
        let n = f(cctx, &mut out);
        assert_eq!(sys::ZSTD_isError(n), 0, "libzstd fails to compress");
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

/// `ZSTD_CCtx_setParameter(cctx, param, value)`, which must succeed.
///
/// # Safety
///
/// `cctx` is a live context.
unsafe fn set(cctx: *mut sys::ZSTD_CCtx, param: P, value: i32) {
    let r = unsafe { sys::ZSTD_CCtx_setParameter(cctx, param, value) };
    // SAFETY: a pure function of its argument.
    assert_eq!(
        unsafe { sys::ZSTD_isError(r) },
        0,
        "set parameter {param:?}"
    );
}

/// libzstd's `ZSTD_CDict` of a dictionary at a level.
struct LibCDict(*mut sys::ZSTD_CDict);

impl LibCDict {
    fn new(dict: &[u8], level: i32) -> Self {
        // SAFETY: ZSTD_createCDict copies `dict`.
        let cdict = unsafe { sys::ZSTD_createCDict(dict.as_ptr().cast(), dict.len(), level) };
        assert!(!cdict.is_null(), "libzstd rejects the dictionary");
        Self(cdict)
    }

    /// `ZSTD_compress2` with `ZSTD_CCtx_refCDict`, `attach` and `params`.
    fn compress(&self, src: &[u8], attach: Attach, params: &[(P, i32)]) -> Vec<u8> {
        lib_frame(
            // SAFETY: a live context and CDict, `out` of its length.
            |cctx, out| unsafe {
                assert_eq!(sys::ZSTD_isError(sys::ZSTD_CCtx_refCDict(cctx, self.0)), 0);
                // ZSTD_c_forceAttachDict
                set(cctx, P::ZSTD_c_experimentalParam4, attach as i32);
                for &(param, value) in params {
                    set(cctx, param, value);
                }
                let (dst, cap) = (out.as_mut_ptr().cast(), out.len());
                sys::ZSTD_compress2(cctx, dst, cap, src.as_ptr().cast(), src.len())
            },
            src.len(),
        )
    }

    /// `ZSTD_compress_usingCDict`.
    fn compress_using(&self, src: &[u8]) -> Vec<u8> {
        lib_frame(
            // SAFETY: a live context and CDict, `out` of its length.
            |cctx, out| unsafe {
                let (dst, cap) = (out.as_mut_ptr().cast(), out.len());
                let (s, n) = (src.as_ptr().cast(), src.len());
                sys::ZSTD_compress_usingCDict(cctx, dst, cap, s, n, self.0)
            },
            src.len(),
        )
    }
}

impl Drop for LibCDict {
    fn drop(&mut self) {
        // SAFETY: created by ZSTD_createCDict and freed once.
        unsafe { sys::ZSTD_freeCDict(self.0) };
    }
}

/// `ZSTD_compress_usingDict`.
fn lib_using_dict(src: &[u8], dict: &[u8], level: i32) -> Vec<u8> {
    lib_frame(
        // SAFETY: a live context, `out` of its length.
        |cctx, out| unsafe {
            let (dst, cap) = (out.as_mut_ptr().cast(), out.len());
            let (s, n) = (src.as_ptr().cast(), src.len());
            let (d, dn) = (dict.as_ptr().cast(), dict.len());
            sys::ZSTD_compress_usingDict(cctx, dst, cap, s, n, d, dn, level)
        },
        src.len(),
    )
}

/// `ZSTD_compress2` at `level` after `ZSTD_CCtx_refPrefix(prefix)`.
fn lib_ref_prefix(src: &[u8], prefix: &[u8], level: i32) -> Vec<u8> {
    let mut cctx = CCtx::create();
    let mut out = Vec::with_capacity(zstd_safe::compress_bound(src.len()));
    cctx.set_parameter(CParameter::CompressionLevel(level))
        .unwrap();
    cctx.ref_prefix(prefix).unwrap();
    cctx.compress2(&mut out, src).expect("libzstd compresses");
    out
}

/// `ZSTD_compress2` at `level` with `params` after
/// `ZSTD_CCtx_refPrefix(prefix)`.
fn lib_prefix_with(src: &[u8], prefix: &[u8], level: i32, params: &[(P, i32)]) -> Vec<u8> {
    lib_frame(
        // SAFETY: a live context, `out` of its length.
        |cctx, out| unsafe {
            set(cctx, P::ZSTD_c_compressionLevel, level);
            for &(param, value) in params {
                set(cctx, param, value);
            }
            let (p, n) = (prefix.as_ptr().cast(), prefix.len());
            assert_eq!(sys::ZSTD_isError(sys::ZSTD_CCtx_refPrefix(cctx, p, n)), 0);
            let (dst, cap) = (out.as_mut_ptr().cast(), out.len());
            sys::ZSTD_compress2(cctx, dst, cap, src.as_ptr().cast(), src.len())
        },
        src.len(),
    )
}

/// libzstd's decode of `frame` with `dict` (`ZSTD_decompress_usingDict`)
/// and ours ([`assert_ours_decodes`]) are `src`.
fn assert_decodes(what: &str, frame: &[u8], dict: &[u8], src: &[u8]) {
    let mut dctx = DCtx::create();
    let mut out = Vec::with_capacity(src.len());
    dctx.decompress_using_dict(&mut out, frame, dict)
        .unwrap_or_else(|e| {
            panic!(
                "{what}: libzstd rejects the frame: {}",
                zstd_safe::get_error_name(e)
            )
        });
    assert!(out == src, "{what}: libzstd decodes another input");
    assert_ours_decodes(what, frame, dict, src);
}

/// Our decode of `frame` with `dict` is `src`.
fn assert_ours_decodes(what: &str, frame: &[u8], dict: &[u8], src: &[u8]) {
    let dict = DecodeDict::new(dict).expect("our decoder parses the dictionary");
    let out = decompress_with_dict(frame, &dict)
        .unwrap_or_else(|e| panic!("{what}: our decoder rejects the frame: {e}"));
    assert!(out == src, "{what}: our decoder decodes another input");
}

/// One dictionary frame through the gate with libzstd's default and its
/// force-copy preference: our frame of `src` with `ours` and the same
/// preference round trips through libzstd and our decoder with `dict`,
/// carries libzstd's dictionary ID, and is at most [`common::size_limit`]
/// of libzstd's frame with `lib` (else the error is pushed to `failures`).
/// Returns our size and libzstd's with the default, then with force-copy.
fn gate(
    what: &str,
    src: &[u8],
    dict: &[u8],
    ours: &Arc<CompressDict>,
    lib: &LibCDict,
    failures: &mut Vec<String>,
) -> [[usize; 2]; 2] {
    [Attach::Default, Attach::ForceCopy].map(|attach| {
        let what = format!("{what} {attach:?}");
        let frame = ours_frame(src, ours, attach);
        assert_decodes(&what, &frame, dict, src);
        let lib = lib.compress(src, attach, &[]);
        assert_eq!(
            zstd_safe::get_dict_id_from_frame(&frame),
            zstd_safe::get_dict_id_from_frame(&lib),
            "{what}: dictionary ID"
        );
        if let Err(e) = common::check_size(&what, frame.len(), lib.len()) {
            failures.push(e);
        }
        [frame.len(), lib.len()]
    })
}

#[test]
fn dictionary_frames_pass_the_gate() {
    let mut failures = Vec::new();
    println!(
        "{:<7} {:<8} {:>3} {:>8} {:>8} {:>8} {:>8} {:>9} {:>10}  worst vs default, vs copy",
        "corpus", "dict", "lvl", "ours", "default", "ourscopy", "copy", "usingDict", "usingCDict"
    );
    for corpus in corpora() {
        for (kind, dict) in dictionaries(&corpus) {
            for level in LEVELS {
                let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
                let lib = LibCDict::new(&dict, level);
                let mut sums = [0usize; 6];
                // (largest ours - lib, its input size), default and copy.
                let mut worst = [(isize::MIN, 0); 2];
                for (i, src) in corpus.test.iter().enumerate() {
                    let what = format!(
                        "{} {kind} L{level} input {i} ({} bytes)",
                        corpus.name,
                        src.len()
                    );
                    let sizes = gate(&what, src, &dict, &ours, &lib, &mut failures);
                    let using_dict = lib_using_dict(src, &dict, level).len();
                    let using_cdict = lib.compress_using(src).len();
                    for (sum, n) in sums.iter_mut().zip(
                        sizes
                            .as_flattened()
                            .iter()
                            .chain(&[using_dict, using_cdict]),
                    ) {
                        *sum += n;
                    }
                    for (w, [n, lib]) in worst.iter_mut().zip(sizes) {
                        *w = (*w).max((n as isize - lib as isize, src.len()));
                    }
                }
                println!(
                    "{:<7} {:<8} {:>3} {:>8} {:>8} {:>8} {:>8} {:>9} {:>10}  {:+} ({}), {:+} ({})",
                    corpus.name,
                    kind,
                    level,
                    sums[0],
                    sums[1],
                    sums[2],
                    sums[3],
                    sums[4],
                    sums[5],
                    worst[0].0,
                    worst[0].1,
                    worst[1].0,
                    worst[1].1
                );
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The held-out samples of the generated corpora at `levels` are
/// libzstd's frames with the preference `attach` byte for byte, a stricter
/// check than the gate's size bound. The crate source corpus is left out:
/// it changes with every edit, and fast, double-fast and btlazy2 may match
/// the first byte of the content where libzstd's do not
/// (`dictStartIndex < matchIndex`; `Window::lowest_match_index` is one
/// inclusive bound), which some versions of it reach.
fn frames_equal(attach: Attach, levels: &[i32]) {
    let mut failures = Vec::new();
    for corpus in corpora().into_iter().filter(|c| c.name != "source") {
        for (kind, dict) in dictionaries(&corpus) {
            for &level in levels {
                let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
                let lib = LibCDict::new(&dict, level);
                let differ: Vec<usize> = (0..corpus.test.len())
                    .filter(|&i| {
                        let src = &corpus.test[i];
                        ours_frame(src, &ours, attach) != lib.compress(src, attach, &[])
                    })
                    .collect();
                if !differ.is_empty() {
                    failures.push(format!(
                        "{} {kind} L{level}: inputs {differ:?} differ",
                        corpus.name
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// With the dictionary's tables copied, at a level of every strategy: the
/// finders follow libzstd's `ZSTD_extDict` rules at the end of the
/// dictionary content.
#[test]
fn dictionary_frames_equal_force_copy() {
    frames_equal(
        Attach::ForceCopy,
        &[-1, 1, 3, 4, 5, 6, 9, 11, 12, 13, 16, 19],
    );
}

/// With libzstd's default, which attaches the dictionary's tables for every
/// held-out sample but the last, at a level of every strategy: the finders
/// follow libzstd's `ZSTD_dictMatchState` rules.
#[test]
fn dictionary_frames_equal_default() {
    frames_equal(Attach::Default, &[-1, 1, 3, 4, 5, 6, 9, 11, 12, 13, 16, 19]);
}

/// Levels across every strategy, and input sizes either side of each
/// strategy's attach cutoff (8, 16 and 32 KiB), of 128 KiB and of six times
/// a 32 KiB dictionary (the dictionary's tables or the frame's own),
/// through the gate; also prints, per dictionary, our size minus libzstd's
/// with its default and with force-copy for every cell.
fn grid(levels: &[i32]) {
    let corpus = corpora().swap_remove(1);
    let text: Vec<u8> = corpus.train.concat();
    let trained = zstd::dict::from_samples(&corpus.train, 16 << 10).expect("train");
    let raw = text[..32 << 10].to_vec();
    let text = &text[text.len() - (192 << 10)..];
    let sizes = [
        0,
        1,
        1 << 10,
        4 << 10,
        8 << 10,
        (8 << 10) + 1,
        16 << 10,
        (16 << 10) + 1,
        32 << 10,
        (32 << 10) + 1,
        64 << 10,
        (128 << 10) - 1,
        128 << 10,
        (192 << 10) - 1,
        192 << 10,
    ];
    let mut failures = Vec::new();
    for (kind, dict) in [("trained 16K", &trained), ("raw 32K", &raw)] {
        println!("{kind}: ours - libzstd, default / force-copy (bytes) per input size");
        print!("{:>5}", "level");
        for size in sizes {
            print!(" {size:>11}");
        }
        println!();
        for &level in levels {
            let ours = Arc::new(CompressDict::new(dict, level).unwrap());
            let lib = LibCDict::new(dict, level);
            print!("{level:>5}");
            for size in sizes {
                let src = &text[text.len() - size..];
                let what = format!("{kind} L{level} {size} bytes");
                let [[n, default], [n_copy, copy]] =
                    gate(&what, src, dict, &ours, &lib, &mut failures);
                let cell = format!(
                    "{:+}/{:+}",
                    n as isize - default as isize,
                    n_copy as isize - copy as isize
                );
                print!(" {cell:>11}");
            }
            println!();
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn grid_passes_the_gate() {
    grid(&[-5, -1, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 15, 16]);
}

/// [`grid`] at the btopt, btultra and btultra2 levels, slow in a debug
/// build: `cargo test --release --test dict_vs_libzstd -- --ignored`.
#[test]
#[ignore]
fn grid_passes_the_gate_high_levels() {
    grid(&[17, 18, 19, 20, 21, 22]);
}

#[test]
fn prefix_frames_pass_the_gate() {
    let mut failures = Vec::new();
    for corpus in corpora() {
        let (_, prefix) = dictionaries(&corpus).pop().unwrap();
        for level in [-5, 1, 3, 9, 19] {
            let opts = CompressOptions {
                level,
                ..Default::default()
            };
            let (mut ours_total, mut lib_total) = (0, 0);
            for (i, src) in corpus.test.iter().enumerate() {
                let what = format!(
                    "{} prefix L{level} input {i} ({} bytes)",
                    corpus.name,
                    src.len()
                );
                let ours = compress_with_prefix(src, &prefix, &opts);
                let mut dctx = DCtx::create();
                dctx.ref_prefix(&prefix).unwrap();
                let mut out = Vec::with_capacity(src.len());
                dctx.decompress(&mut out, &ours)
                    .unwrap_or_else(|e| panic!("{what}: {}", zstd_safe::get_error_name(e)));
                assert!(out == *src, "{what}: libzstd decodes another input");
                assert_ours_decodes(&what, &ours, &prefix, src);
                let lib = lib_ref_prefix(src, &prefix, level);
                if let Err(e) = common::check_size(&what, ours.len(), lib.len()) {
                    failures.push(e);
                }
                ours_total += ours.len();
                lib_total += lib.len();
            }
            println!(
                "{:<7} prefix L{level:<2} ours {ours_total:>9} refPrefix {lib_total:>9}",
                corpus.name
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A [`CompressDict`] gives the same frames used once or reused, around
/// frames with a prefix and without a dictionary on the same contexts, and
/// a dictionary leaves nothing behind for those.
#[test]
fn reused_dictionaries_are_byte_stable() {
    let corpus = corpora().swap_remove(0);
    let (_, trained) = dictionaries(&corpus).swap_remove(0);
    for level in [-5, 1, 3, 9, 19] {
        let dict = Arc::new(CompressDict::new(&trained, level).unwrap());
        let one_shot: Vec<Vec<u8>> = corpus
            .test
            .iter()
            .map(|s| compress_with_dict(s, &dict))
            .collect();
        let opts = CompressOptions {
            level,
            ..Default::default()
        };
        let mut with_dict = Compressor::new(CompressOptions {
            dict: Some(dict.clone()),
            ..opts.clone()
        });
        let mut plain = Compressor::new(opts.clone());
        for round in 0..2 {
            for (i, src) in corpus.test.iter().enumerate() {
                let what = format!("L{level} round {round} input {i}");
                assert!(
                    with_dict.compress_to_vec(src) == one_shot[i],
                    "{what}: reused dictionary"
                );
                let prefixed = compress_with_prefix(src, &trained, &opts);
                let mut out = Vec::new();
                with_dict.compress_with_prefix(src, &trained, &mut out);
                assert!(out == prefixed, "{what}: prefix after a dictionary");
                let fresh = rust_zstd::compress::compress_with(src, &opts);
                assert!(plain.compress_to_vec(src) == fresh, "{what}: no dictionary");
                let mut out = Vec::new();
                plain.compress_with_prefix(src, &trained, &mut out);
                assert!(out == prefixed, "{what}: prefix after no dictionary");
                assert!(
                    plain.compress_to_vec(src) == fresh,
                    "{what}: no dictionary after a prefix"
                );
            }
        }
    }
}

/// Long distance matching and a checksum with a dictionary or a prefix,
/// on an input of 100 KiB, for which the frame copies the dictionary's
/// tables, and of 1 MiB, for which it loads its own, raw content into
/// the long distance matcher too.
#[test]
fn dictionary_with_ldm_and_checksum() {
    let corpus = corpora().swap_remove(2);
    let mut src = Vec::new();
    for s in corpus.test.iter().chain(&corpus.train).cycle() {
        if src.len() > 1 << 20 {
            break;
        }
        src.extend_from_slice(s);
    }
    let params = [
        (P::ZSTD_c_enableLongDistanceMatching, 1),
        (P::ZSTD_c_checksumFlag, 1),
    ];
    let mut failures = Vec::new();
    for (kind, dict) in dictionaries(&corpus) {
        for level in [3, 19] {
            let ours_dict = Arc::new(CompressDict::new(&dict, level).unwrap());
            let lib_dict = LibCDict::new(&dict, level);
            for src in [&src[..100 << 10], &src] {
                let mut frames = Vec::new();
                for attach in [Attach::Default, Attach::ForceCopy] {
                    let ours = Compressor::new(CompressOptions {
                        ldm: ParamSwitch::Enable,
                        checksum: true,
                        ..dict_opts(&ours_dict, attach)
                    })
                    .compress_to_vec(src);
                    let lib = lib_dict.compress(src, attach, &params);
                    frames.push((format!("{kind} {attach:?}"), ours, lib));
                }
                if kind == "raw" {
                    let opts = CompressOptions {
                        level,
                        ldm: ParamSwitch::Enable,
                        checksum: true,
                        ..Default::default()
                    };
                    let ours = compress_with_prefix(src, &dict, &opts);
                    let lib = lib_prefix_with(src, &dict, level, &params);
                    frames.push(("prefix".to_string(), ours, lib));
                }
                for (what, ours, lib) in frames {
                    let what = format!("source {what} L{level} {} bytes ldm+checksum", src.len());
                    assert_decodes(&what, &ours, &dict, src);
                    println!("{what}: ours {} libzstd {}", ours.len(), lib.len());
                    if let Err(e) = common::check_size(&what, ours.len(), lib.len()) {
                        failures.push(e);
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `f`'s frame, which with the parallel feature must be the same on rayon
/// pools of 1 and 4 threads.
fn same_on_pools(what: &str, f: impl Fn() -> Vec<u8> + Sync) -> Vec<u8> {
    let frame = f();
    #[cfg(feature = "parallel")]
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert!(
            pool.install(&f) == frame,
            "{what}: another frame on a {threads}-thread pool"
        );
    }
    #[cfg(not(feature = "parallel"))]
    let _ = what;
    frame
}

/// Frames with a dictionary and a job size, over two jobs of 512 KiB, pass
/// the gate against libzstd's ZSTDMT with the same CDict or prefix
/// (`ZSTD_c_nbWorkers` 2, the same `ZSTD_c_jobSize`), with and without
/// long distance matching. Only the first job starts from the dictionary,
/// so a frame of several jobs is not the one-job frame.
#[test]
fn multithreaded_dictionary_frames_pass_the_gate() {
    let corpus = corpora().swap_remove(1);
    let src: Vec<u8> = corpus.train.concat();
    assert!(src.len() > 2 * JOBSIZE_MIN);
    let mut failures = Vec::new();
    for (kind, dict) in dictionaries(&corpus) {
        for level in [-5, 3, 19] {
            let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
            let lib = LibCDict::new(&dict, level);
            for (job_size, ldm) in [
                (JOBSIZE_MIN, ParamSwitch::Auto),
                (JOBSIZE_MIN, ParamSwitch::Enable),
                (0, ParamSwitch::Auto),
            ] {
                let mut params = vec![
                    (P::ZSTD_c_nbWorkers, 2),
                    (P::ZSTD_c_jobSize, job_size as i32),
                ];
                if ldm == ParamSwitch::Enable {
                    params.push((P::ZSTD_c_enableLongDistanceMatching, 1));
                }
                for attach in [Attach::Default, Attach::ForceCopy] {
                    let what = format!("{kind} L{level} job {job_size} ldm {ldm:?} {attach:?}");
                    let opts = CompressOptions {
                        job_size: Some(job_size),
                        ldm,
                        ..dict_opts(&ours, attach)
                    };
                    let frame = same_on_pools(&what, || {
                        Compressor::new(opts.clone()).compress_to_vec(&src)
                    });
                    // At level 19 (btultra2) the overlap is the window,
                    // which the job size grows to: one job.
                    if job_size == JOBSIZE_MIN && ldm == ParamSwitch::Auto && level < 19 {
                        let one_job = ours_frame(&src, &ours, attach);
                        assert!(frame != one_job, "{what}: the one-job frame");
                    }
                    assert_decodes(&what, &frame, &dict, &src);
                    let lib = lib.compress(&src, attach, &params);
                    if let Err(e) = common::check_size(&what, frame.len(), lib.len()) {
                        failures.push(e);
                    }
                }
                if kind != "raw" {
                    continue;
                }
                let what = format!("prefix L{level} job {job_size} ldm {ldm:?}");
                let opts = CompressOptions {
                    level,
                    job_size: Some(job_size),
                    ldm,
                    ..Default::default()
                };
                let frame = same_on_pools(&what, || compress_with_prefix(&src, &dict, &opts));
                assert_decodes(&what, &frame, &dict, &src);
                let lib = lib_prefix_with(&src, &dict, level, &params);
                if let Err(e) = common::check_size(&what, frame.len(), lib.len()) {
                    failures.push(e);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The input chunk sizes of the streaming gate: 1, 7 and 64 KiB, and the
/// whole input in one call.
const STREAM_CHUNKS: [usize; 4] = [1, 7, 64 << 10, usize::MAX];

/// The calls a stream of `len` bytes is cut into: `chunk`-byte `Continue`
/// calls, `Flush` after each input position in `flushes`, then `End`.
fn stream_calls(len: usize, chunk: usize, flushes: &[usize]) -> Vec<(Range<usize>, EndDirective)> {
    let mut cuts: Vec<usize> = (chunk..len).step_by(chunk).collect();
    cuts.extend_from_slice(flushes);
    cuts.push(len);
    cuts.sort_unstable();
    cuts.dedup();
    let mut calls = Vec::new();
    let mut start = 0;
    for cut in cuts {
        calls.push((start..cut, EndDirective::Continue));
        if flushes.contains(&cut) {
            calls.push((cut..cut, EndDirective::Flush));
        }
        start = cut;
    }
    calls.push((len..len, EndDirective::End));
    calls
}

/// Our frame of `src` through `cctx.compress_stream`, the size pledged
/// first if `pledged`, cut as [`stream_calls`] says, into a 64 KiB output
/// buffer drained after every call.
fn ours_stream(
    cctx: &mut Compressor,
    src: &[u8],
    pledged: bool,
    chunk: usize,
    flushes: &[usize],
) -> Vec<u8> {
    if pledged {
        cctx.set_pledged_src_size(Some(src.len() as u64)).unwrap();
    }
    let mut frame = Vec::new();
    let mut dst = vec![0u8; 1 << 16];
    for (range, op) in stream_calls(src.len(), chunk, flushes) {
        let piece = &src[range];
        let mut pos = 0;
        loop {
            let mut dst_pos = 0;
            let left = cctx
                .compress_stream(piece, &mut pos, &mut dst, &mut dst_pos, op)
                .expect("compress_stream");
            frame.extend_from_slice(&dst[..dst_pos]);
            let done = match op {
                EndDirective::Continue => pos == piece.len() && dst_pos < dst.len(),
                EndDirective::Flush | EndDirective::End => left == 0,
            };
            if done {
                break;
            }
        }
    }
    frame
}

impl LibCDict {
    /// [`common::c_stream_unbuffered`] with `ZSTD_CCtx_refCDict` and
    /// `attach`.
    fn stream(&self, src: &[u8], attach: Attach, pledged: bool, flushes: &[usize]) -> Vec<u8> {
        // SAFETY: a live context and CDict.
        common::c_stream_unbuffered(src, pledged, flushes, |cctx| unsafe {
            assert_eq!(sys::ZSTD_isError(sys::ZSTD_CCtx_refCDict(cctx, self.0)), 0);
            set(cctx, P::ZSTD_c_experimentalParam4, attach as i32);
        })
    }
}

/// [`common::c_stream_unbuffered`] at `level` with
/// `ZSTD_CCtx_refPrefix(prefix)`.
fn lib_stream_prefix(src: &[u8], prefix: &[u8], level: i32, pledged: bool) -> Vec<u8> {
    // SAFETY: a live context; `prefix` outlives the stream.
    common::c_stream_unbuffered(src, pledged, &[], |cctx| unsafe {
        set(cctx, P::ZSTD_c_compressionLevel, level);
        let r = sys::ZSTD_CCtx_refPrefix(cctx, prefix.as_ptr().cast(), prefix.len());
        assert_eq!(sys::ZSTD_isError(r), 0);
    })
}

/// `src` streamed with `cctx` at each of `chunks`: with the size pledged,
/// `one_shot` byte for byte; without, the same frame at every chunk size,
/// which `check` accepts and which is returned.
fn check_streams(
    what: &str,
    cctx: &mut Compressor,
    src: &[u8],
    chunks: &[usize],
    one_shot: &[u8],
    check: impl Fn(&str, &[u8]),
) -> Vec<u8> {
    let mut unpledged: Option<Vec<u8>> = None;
    for &chunk in chunks {
        let what = format!("{what} chunk {chunk}");
        let frame = ours_stream(cctx, src, true, chunk, &[]);
        assert!(
            frame == one_shot,
            "{what}: pledged stream != one-shot frame"
        );
        let frame = ours_stream(cctx, src, false, chunk, &[]);
        match &unpledged {
            None => {
                check(&what, &frame);
                unpledged = Some(frame);
            }
            Some(first) => assert!(&frame == first, "{what}: unpledged stream differs"),
        }
    }
    unpledged.unwrap()
}

/// Without flushes, a dictionary stream with its size pledged is the
/// one-shot frame byte for byte however it is cut, the dictionary attached
/// or copied (small inputs) or loaded (the joined one over 128 KiB); one
/// of unknown size does not depend on the cut and round trips. One
/// `Compressor` streams them all, so each frame also starts from the
/// contexts and dictionary the last one left.
#[test]
fn dictionary_streams_are_the_one_shot_frame() {
    for corpus in corpora() {
        for (kind, dict) in dictionaries(&corpus) {
            for level in LEVELS {
                let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
                for attach in [Attach::Default, Attach::ForceCopy] {
                    let mut cctx = Compressor::new(dict_opts(&ours, attach));
                    for src in [&corpus.test[0], corpus.test.last().unwrap()] {
                        let what = format!(
                            "{} {kind} L{level} {attach:?} {} bytes",
                            corpus.name,
                            src.len()
                        );
                        let one_shot = ours_frame(src, &ours, attach);
                        let check =
                            |what: &str, frame: &[u8]| assert_decodes(what, frame, &dict, src);
                        check_streams(&what, &mut cctx, src, &STREAM_CHUNKS, &one_shot, check);
                    }
                }
            }
        }
    }
}

/// The streaming gate against libzstd's stream with the same CDict and
/// preference ([`common::c_stream_unbuffered`]), with and without the size
/// pledged: every frame, however it is cut, round trips through libzstd
/// and our decoder, carries the dictionary ID, and is at most
/// [`common::size_limit`] of libzstd's. Prints the totals and how many
/// frames are libzstd's byte for byte.
#[test]
fn dictionary_streams_pass_the_gate() {
    let mut failures = Vec::new();
    for corpus in corpora() {
        for (kind, dict) in dictionaries(&corpus) {
            for level in LEVELS {
                let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
                let lib = LibCDict::new(&dict, level);
                for attach in [Attach::Default, Attach::ForceCopy] {
                    let mut cctx = Compressor::new(dict_opts(&ours, attach));
                    for pledged in [true, false] {
                        let (mut ours_total, mut lib_total, mut same, mut n) = (0, 0, 0, 0);
                        for (i, src) in corpus.test.iter().enumerate() {
                            // Every chunk size for the first and last
                            // inputs, the whole input for the rest.
                            let chunks: &[usize] = if i == 0 || i + 1 == corpus.test.len() {
                                &STREAM_CHUNKS
                            } else {
                                &[usize::MAX]
                            };
                            let lib = lib.stream(src, attach, pledged, &[]);
                            for &chunk in chunks {
                                let what = format!(
                                    "{} {kind} L{level} {attach:?} pledged {pledged} input {i} \
                                     ({} bytes) chunk {chunk}",
                                    corpus.name,
                                    src.len()
                                );
                                let frame = ours_stream(&mut cctx, src, pledged, chunk, &[]);
                                assert_decodes(&what, &frame, &dict, src);
                                assert_eq!(
                                    zstd_safe::get_dict_id_from_frame(&frame),
                                    zstd_safe::get_dict_id_from_frame(&lib),
                                    "{what}: dictionary ID"
                                );
                                if let Err(e) = common::check_size(&what, frame.len(), lib.len()) {
                                    failures.push(e);
                                }
                                ours_total += frame.len();
                                lib_total += lib.len();
                                same += usize::from(frame == lib);
                                n += 1;
                            }
                        }
                        println!(
                            "{:<7} {kind:<8} L{level:<2} {attach:<9?} pledged {pledged:<5} \
                             ours {ours_total:>8} libzstd {lib_total:>8} identical {same}/{n}",
                            corpus.name
                        );
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Flushes in a dictionary stream end blocks early: the frame round trips
/// and passes the size gate against libzstd flushing at the same points.
#[test]
fn flushed_dictionary_streams_pass_the_gate() {
    let corpus = corpora().swap_remove(1);
    let src = corpus.test.last().unwrap();
    let flushes: Vec<usize> = [1, 17, 4000, 50_000, 100_000, 150_000]
        .into_iter()
        .filter(|&p| p < src.len())
        .collect();
    for (kind, dict) in dictionaries(&corpus) {
        for level in [1, 3, 9] {
            let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
            let lib = LibCDict::new(&dict, level);
            for attach in [Attach::Default, Attach::ForceCopy] {
                for pledged in [true, false] {
                    let what = format!("log {kind} L{level} {attach:?} pledged {pledged} flushed");
                    let mut cctx = Compressor::new(dict_opts(&ours, attach));
                    let frame = ours_stream(&mut cctx, src, pledged, 4096, &flushes);
                    assert_decodes(&what, &frame, &dict, src);
                    let lib = lib.stream(src, attach, pledged, &flushes);
                    common::check_size(&what, frame.len(), lib.len()).unwrap();
                }
            }
        }
    }
}

/// A dictionary stream over inputs much larger than the window, so the
/// buffer moves its last window down many times: with the size pledged
/// the dictionary is loaded into the frame's tables, its content first in
/// the buffer, and the stream is the one-shot frame; of unknown size it is
/// attached (default) or copied with libzstd's small parameters for an
/// unknown input with a dictionary, its content again first in the
/// buffer, and the frame does not depend on the cut, round trips and
/// passes the gate against libzstd's stream.
#[test]
fn sliding_buffer_with_a_dictionary() {
    let corpus = corpora().swap_remove(2);
    let (_, dict) = dictionaries(&corpus).swap_remove(0);
    let data = common::datasets().swap_remove(0).data;
    let src = &data[..3 << 20];
    for level in [1, 3, 9] {
        let ours = Arc::new(CompressDict::new(&dict, level).unwrap());
        let lib = LibCDict::new(&dict, level);
        for attach in [Attach::Default, Attach::ForceCopy] {
            let what = format!("source 3M L{level} {attach:?}");
            let mut cctx = Compressor::new(dict_opts(&ours, attach));
            let one_shot = ours_frame(src, &ours, attach);
            let lib = lib.stream(src, attach, false, &[]);
            let check = |what: &str, frame: &[u8]| {
                assert_decodes(what, frame, &dict, src);
                println!(
                    "{what} unpledged: ours {} libzstd {} identical {}",
                    frame.len(),
                    lib.len(),
                    frame == lib
                );
                common::check_size(what, frame.len(), lib.len()).unwrap();
            };
            let chunks = [1013, 64 << 10, usize::MAX];
            check_streams(&what, &mut cctx, src, &chunks, &one_shot, check);
        }
    }
}

/// A prefix set with `set_prefix` is the next frame's alone, streamed or
/// one-shot: pledged, the stream is `compress_with_prefix`'s frame however
/// it is cut; of unknown size, it round trips with the prefix and passes
/// the gate against libzstd's stream after `ZSTD_CCtx_refPrefix`. The
/// frame after it has no prefix again, and the dictionary of the options
/// comes back.
#[test]
fn prefix_streams() {
    let mut failures = Vec::new();
    for corpus in corpora() {
        let (_, prefix) = dictionaries(&corpus).pop().unwrap();
        for level in [-5, 1, 3, 9, 19] {
            let opts = CompressOptions {
                level,
                ..Default::default()
            };
            let mut cctx = Compressor::new(opts.clone());
            for src in [&corpus.test[0], corpus.test.last().unwrap()] {
                let what = format!("{} prefix L{level} {} bytes", corpus.name, src.len());
                let one_shot = compress_with_prefix(src, &prefix, &opts);
                let mut unpledged: Option<Vec<u8>> = None;
                for chunk in STREAM_CHUNKS {
                    let what = format!("{what} chunk {chunk}");
                    cctx.set_prefix(&prefix).unwrap();
                    let frame = ours_stream(&mut cctx, src, true, chunk, &[]);
                    assert!(
                        frame == one_shot,
                        "{what}: pledged stream != one-shot frame"
                    );
                    cctx.set_prefix(&prefix).unwrap();
                    let frame = ours_stream(&mut cctx, src, false, chunk, &[]);
                    match &unpledged {
                        None => {
                            let mut dctx = DCtx::create();
                            dctx.ref_prefix(&prefix).unwrap();
                            let mut out = Vec::with_capacity(src.len());
                            dctx.decompress(&mut out, &frame).unwrap_or_else(|e| {
                                panic!("{what}: {}", zstd_safe::get_error_name(e))
                            });
                            assert!(out == **src, "{what}: libzstd decodes another input");
                            assert_ours_decodes(&what, &frame, &prefix, src);
                            unpledged = Some(frame);
                        }
                        Some(first) => {
                            assert!(&frame == first, "{what}: unpledged stream differs")
                        }
                    }
                }
                for pledged in [true, false] {
                    let ours = if pledged {
                        &one_shot
                    } else {
                        unpledged.as_ref().unwrap()
                    };
                    let lib = lib_stream_prefix(src, &prefix, level, pledged);
                    let what = format!("{what} pledged {pledged}");
                    if let Err(e) = common::check_size(&what, ours.len(), lib.len()) {
                        failures.push(e);
                    }
                }
                // Single usage: the next frame has no prefix.
                let plain = rust_zstd::compress::compress_with(src, &opts);
                assert!(
                    ours_stream(&mut cctx, src, true, 4096, &[]) == plain,
                    "{what}: the frame after the prefix's"
                );
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // The prefix replaces the options' dictionary for one frame.
    let corpus = corpora().swap_remove(0);
    let (_, trained) = dictionaries(&corpus).swap_remove(0);
    let (_, prefix) = dictionaries(&corpus).pop().unwrap();
    let src = &corpus.test[0];
    let dict = Arc::new(CompressDict::new(&trained, 3).unwrap());
    let mut cctx = Compressor::new(dict_opts(&dict, Attach::Default));
    cctx.set_prefix(&prefix).unwrap();
    let frame = ours_stream(&mut cctx, src, true, 7, &[]);
    assert!(frame == compress_with_prefix(src, &prefix, &CompressOptions::default()));
    assert!(ours_stream(&mut cctx, src, true, 7, &[]) == compress_with_dict(src, &dict));
}

/// `set_prefix` only before a frame starts; the session reset keeps a
/// prefix not yet used, as `ZSTD_CCtx_reset(ZSTD_reset_session_only)`
/// keeps `prefixDict`; `Compressor::compress` uses it as the next frame;
/// `compress_with_prefix` replaces it; an empty prefix clears it.
#[test]
fn set_prefix_stages() {
    let corpus = corpora().swap_remove(1);
    let (_, prefix) = dictionaries(&corpus).pop().unwrap();
    let src = corpus.test.last().unwrap();
    let opts = CompressOptions::default();
    let with_prefix = compress_with_prefix(src, &prefix, &opts);
    let plain = rust_zstd::compress::compress_with(src, &opts);
    let mut cctx = Compressor::new(opts.clone());
    let mut dst = vec![0u8; 1 << 16];
    let (mut src_pos, mut dst_pos) = (0, 0);
    cctx.compress_stream(
        &src[..1000],
        &mut src_pos,
        &mut dst,
        &mut dst_pos,
        EndDirective::Continue,
    )
    .unwrap();
    assert_eq!(cctx.set_prefix(&prefix), Err(CompressError::StageWrong));
    cctx.reset_stream();
    cctx.set_prefix(&prefix).unwrap();
    cctx.reset_stream();
    assert!(ours_stream(&mut cctx, src, true, 64 << 10, &[]) == with_prefix);
    cctx.set_prefix(&prefix).unwrap();
    assert!(cctx.compress_to_vec(src) == with_prefix);
    assert!(cctx.compress_to_vec(src) == plain);
    cctx.set_prefix(&src[..100]).unwrap();
    let mut out = Vec::new();
    cctx.compress_with_prefix(src, &prefix, &mut out);
    assert!(out == with_prefix);
    assert!(cctx.compress_to_vec(src) == plain);
    cctx.set_prefix(&prefix).unwrap();
    cctx.set_prefix(&[]).unwrap();
    assert!(cctx.compress_to_vec(src) == plain);
    // A first End call is the one-shot frame with the prefix.
    cctx.set_prefix(&prefix).unwrap();
    let mut dst = vec![0u8; 1 << 18];
    let (mut src_pos, mut dst_pos) = (0, 0);
    let left = cctx
        .compress_stream(src, &mut src_pos, &mut dst, &mut dst_pos, EndDirective::End)
        .unwrap();
    assert_eq!(left, 0);
    assert!(dst[..dst_pos] == with_prefix[..]);
}

/// The dictionary is in the options, so it outlives the session: a frame
/// abandoned by `reset_stream` or failed on a wrong pledged size leaves the
/// next stream the one-shot frame; a first `End` call is the one-shot
/// frame; `Encoder` streams with the dictionary.
#[test]
fn dictionary_streams_across_resets_and_encoder() {
    let corpus = corpora().swap_remove(0);
    let (_, trained) = dictionaries(&corpus).swap_remove(0);
    let src = corpus.test.last().unwrap();
    for attach in [Attach::Default, Attach::ForceCopy] {
        let dict = Arc::new(CompressDict::new(&trained, 3).unwrap());
        let one_shot = ours_frame(src, &dict, attach);
        let mut cctx = Compressor::new(dict_opts(&dict, attach));
        let mut dst = vec![0u8; 1 << 18];
        let (mut src_pos, mut dst_pos) = (0, 0);
        cctx.compress_stream(
            &src[..5000],
            &mut src_pos,
            &mut dst,
            &mut dst_pos,
            EndDirective::Continue,
        )
        .unwrap();
        cctx.reset_stream();
        assert!(ours_stream(&mut cctx, src, true, 999, &[]) == one_shot);

        cctx.set_pledged_src_size(Some(10)).unwrap();
        let (mut src_pos, mut dst_pos) = (0, 0);
        let r = cctx.compress_stream(
            src,
            &mut src_pos,
            &mut dst,
            &mut dst_pos,
            EndDirective::Continue,
        );
        assert!(matches!(r, Err(CompressError::SrcSizeWrong { .. })));
        cctx.reset_stream();
        assert!(ours_stream(&mut cctx, src, true, 999, &[]) == one_shot);

        let (mut src_pos, mut dst_pos) = (0, 0);
        let left = cctx
            .compress_stream(src, &mut src_pos, &mut dst, &mut dst_pos, EndDirective::End)
            .unwrap();
        assert_eq!(left, 0);
        assert!(dst[..dst_pos] == one_shot[..]);

        let mut encoder = Encoder::new(Vec::new(), dict_opts(&dict, attach));
        for piece in src.chunks(1000) {
            encoder.write_all(piece).unwrap();
        }
        let frame = encoder.finish().unwrap();
        let unpledged = ours_stream(&mut cctx, src, false, 1000, &[]);
        assert!(frame == unpledged, "{attach:?}: Encoder frame");
        assert_decodes(&format!("{attach:?} Encoder"), &frame, &trained, src);
        let mut pledged = Compressor::new(dict_opts(&dict, attach));
        pledged
            .set_pledged_src_size(Some(src.len() as u64))
            .unwrap();
        let mut encoder = Encoder::with_compressor(Vec::new(), pledged);
        encoder.write_all(src).unwrap();
        assert!(
            encoder.finish().unwrap() == one_shot,
            "{attach:?}: pledged Encoder"
        );
    }
}

/// A dictionary stream with a job size and over `JOBSIZE_MIN` is
/// multithreaded streaming, which is not implemented, unless its first call
/// ends it, which writes the one-shot frame; long distance matching and a
/// checksum stream with a dictionary as they compress one-shot.
#[test]
fn dictionary_streams_with_job_size_and_ldm() {
    let corpus = corpora().swap_remove(2);
    let (_, trained) = dictionaries(&corpus).swap_remove(0);
    let mut src = Vec::new();
    for s in corpus.test.iter().chain(&corpus.train).cycle() {
        if src.len() > 1 << 20 {
            break;
        }
        src.extend_from_slice(s);
    }
    let dict = Arc::new(CompressDict::new(&trained, 3).unwrap());
    let opts = CompressOptions {
        job_size: Some(JOBSIZE_MIN),
        ..dict_opts(&dict, Attach::Default)
    };
    let one_shot = Compressor::new(opts.clone()).compress_to_vec(&src);
    let mut cctx = Compressor::new(opts);
    let mut dst = vec![0u8; src.len() + (1 << 16)];
    for pledged in [true, false] {
        if pledged {
            cctx.set_pledged_src_size(Some(src.len() as u64)).unwrap();
        }
        let (mut pos, mut dst_pos) = (0, 0);
        let r = cctx.compress_stream(
            &src,
            &mut pos,
            &mut dst,
            &mut dst_pos,
            EndDirective::Continue,
        );
        assert!(
            matches!(r, Err(CompressError::Unsupported(_))),
            "pledged {pledged}: {r:?}"
        );
        cctx.reset_stream();
        if pledged {
            cctx.set_pledged_src_size(Some(src.len() as u64)).unwrap();
        }
        let (mut pos, mut dst_pos) = (0, 0);
        let left = cctx
            .compress_stream(&src, &mut pos, &mut dst, &mut dst_pos, EndDirective::End)
            .unwrap();
        assert_eq!(left, 0);
        assert!(
            dst[..dst_pos] == one_shot,
            "pledged {pledged}: first-call End"
        );
    }
    let opts = CompressOptions {
        ldm: ParamSwitch::Enable,
        checksum: true,
        ..dict_opts(&dict, Attach::Default)
    };
    let one_shot = Compressor::new(opts.clone()).compress_to_vec(&src);
    let mut cctx = Compressor::new(opts);
    for chunk in [7, 64 << 10] {
        let frame = ours_stream(&mut cctx, &src, true, chunk, &[]);
        assert!(frame == one_shot, "ldm chunk {chunk}: != one-shot frame");
        let frame = ours_stream(&mut cctx, &src, false, chunk, &[]);
        assert_decodes("ldm", &frame, &trained, &src);
    }
}
