//! Dictionary decoding against libzstd: frames compressed with formatted
//! dictionaries decode byte-exact on every decoder path, dictionaries
//! libzstd rejects are rejected, and frames with the wrong or no
//! dictionary are errors.

mod common;

use common::lcg_bytes;
use rust_zstd::decode::{
    decompress_with_dict, decompress_with_dict_options, DecodeDict, DecodeOptions,
};
use zstd::zstd_safe::zstd_sys as sys;

use sys::ZSTD_cParameter as P;

/// Records sharing field names and a small vocabulary, like the payloads
/// dictionaries are trained for.
fn record(seed: u64) -> Vec<u8> {
    const WORDS: [&str; 12] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima",
    ];
    let r = lcg_bytes(16, seed);
    let w = |i: usize| WORDS[usize::from(r[i]) % WORDS.len()];
    format!(
        r#"{{"id":{},"user":"{}_{}","tags":["{}","{}","{}"],"score":{}.{},"active":{},"note":"{} {} {}"}}"#,
        u32::from_le_bytes(r[0..4].try_into().unwrap()) % 100_000,
        w(4),
        r[5],
        w(6),
        w(7),
        w(8),
        r[9] % 100,
        r[10],
        r[11] & 1 == 0,
        w(12),
        w(13),
        w(14),
    )
    .into_bytes()
}

fn records(n: usize, seed: u64) -> Vec<u8> {
    (0..n as u64)
        .flat_map(|i| record(seed * 1_000_003 + i))
        .collect()
}

/// A dictionary trained by ZDICT_trainFromBuffer on 2000 records.
fn trained_dict() -> Vec<u8> {
    let samples: Vec<Vec<u8>> = (0..2000).map(|i| record(7_000_000 + i)).collect();
    let flat: Vec<u8> = samples.concat();
    let sizes: Vec<usize> = samples.iter().map(Vec::len).collect();
    let mut dict = vec![0u8; 16 * 1024];
    // SAFETY: buffers and sample sizes describe `flat` and `dict`.
    let n = unsafe {
        sys::ZDICT_trainFromBuffer(
            dict.as_mut_ptr().cast(),
            dict.len(),
            flat.as_ptr().cast(),
            sizes.as_ptr(),
            sizes.len() as u32,
        )
    };
    assert_eq!(unsafe { sys::ZDICT_isError(n) }, 0, "train dictionary");
    dict.truncate(n);
    dict
}

/// A formatted dictionary with tables ZDICT_finalizeDictionary fits to
/// the records and random content, which record frames find no matches
/// in: frames compressed with it use the tables and repeat offsets only.
fn entropy_dict() -> Vec<u8> {
    let samples: Vec<Vec<u8>> = (0..2000).map(|i| record(8_000_000 + i)).collect();
    let flat: Vec<u8> = samples.concat();
    let sizes: Vec<usize> = samples.iter().map(Vec::len).collect();
    let content = lcg_bytes(4096, 5);
    let mut dict = vec![0u8; 8 * 1024];
    let params = sys::ZDICT_params_t {
        compressionLevel: 3,
        notificationLevel: 0,
        dictID: 0x1234_5678,
    };
    // SAFETY: buffers and sample sizes describe `flat`, `content` and
    // `dict`.
    let n = unsafe {
        sys::ZDICT_finalizeDictionary(
            dict.as_mut_ptr().cast(),
            dict.len(),
            content.as_ptr().cast(),
            content.len(),
            flat.as_ptr().cast(),
            sizes.as_ptr(),
            sizes.len() as u32,
            params,
        )
    };
    assert_eq!(unsafe { sys::ZDICT_isError(n) }, 0, "finalize dictionary");
    dict.truncate(n);
    dict
}

/// The dictionaries every decode test runs with.
fn dicts() -> Vec<(Vec<u8>, &'static str)> {
    vec![(entropy_dict(), "entropy-only")]
}

/// libzstd's frame for `src` with `dict` loaded (auto content type) and
/// `params` set, through ZSTD_compress2.
fn c_compress_dict(src: &[u8], dict: &[u8], params: &[(P, i32)]) -> Vec<u8> {
    // SAFETY: the context is used only here; buffers are sized by
    // ZSTD_compressBound.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        for &(p, v) in params {
            let r = sys::ZSTD_CCtx_setParameter(cctx, p, v);
            assert_eq!(sys::ZSTD_isError(r), 0, "set parameter {p:?}");
        }
        let r = sys::ZSTD_CCtx_loadDictionary(cctx, dict.as_ptr().cast(), dict.len());
        assert_eq!(sys::ZSTD_isError(r), 0, "load dictionary");
        let mut out = vec![0u8; sys::ZSTD_compressBound(src.len())];
        let n = sys::ZSTD_compress2(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            src.as_ptr().cast(),
            src.len(),
        );
        assert_eq!(sys::ZSTD_isError(n), 0, "compress");
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

/// ZSTD_compress_usingDict at `level`.
fn c_compress_using_dict(src: &[u8], dict: &[u8], level: i32) -> Vec<u8> {
    // SAFETY: as `c_compress_dict`.
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let mut out = vec![0u8; sys::ZSTD_compressBound(src.len())];
        let n = sys::ZSTD_compress_usingDict(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            src.as_ptr().cast(),
            src.len(),
            dict.as_ptr().cast(),
            dict.len(),
            level,
        );
        assert_eq!(sys::ZSTD_isError(n), 0, "compress level {level}");
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

/// One-shot libzstd decode with `dict` (ZSTD_decompress_usingDict).
fn c_decompress_dict(frame: &[u8], dict: &[u8], capacity: usize) -> Option<Vec<u8>> {
    // SAFETY: the context is used only here; `out` has `capacity` bytes.
    unsafe {
        let dctx = sys::ZSTD_createDCtx();
        let mut out = vec![0u8; capacity];
        let n = sys::ZSTD_decompress_usingDict(
            dctx,
            out.as_mut_ptr().cast(),
            out.len(),
            frame.as_ptr().cast(),
            frame.len(),
            dict.as_ptr().cast(),
            dict.len(),
        );
        sys::ZSTD_freeDCtx(dctx);
        (sys::ZSTD_isError(n) == 0).then(|| {
            out.truncate(n);
            out
        })
    }
}

/// Whether libzstd's ZSTD_createDDict (auto content type) loads `dict`.
fn c_loads_dict(dict: &[u8]) -> bool {
    // SAFETY: `dict` is read during the call only.
    unsafe {
        let d = sys::ZSTD_createDDict(dict.as_ptr().cast(), dict.len());
        let ok = !d.is_null();
        sys::ZSTD_freeDDict(d);
        ok
    }
}

/// Every decoder path: serial and MT (every frame, any block count), each
/// with the detected SIMD level and portable.
fn paths() -> [(DecodeOptions, &'static str); 4] {
    let o = |min_parallel_blocks, simd| DecodeOptions {
        min_parallel_blocks,
        simd,
    };
    [
        (o(usize::MAX, true), "serial"),
        (o(usize::MAX, false), "serial portable"),
        (o(1, true), "mt"),
        (o(1, false), "mt portable"),
    ]
}

/// `frame` decodes to `src` with `dict` on every path.
fn assert_decodes(what: &str, frame: &[u8], dict: &DecodeDict, src: &[u8]) {
    for (opts, path) in paths() {
        match decompress_with_dict_options(frame, Some(dict), &opts) {
            Ok(out) => assert!(out == src, "{what} ({path}): wrong output"),
            Err(e) => panic!("{what} ({path}): {e}"),
        }
    }
}

/// `frame` fails with `dict` on every path, with an error containing
/// `want`.
fn assert_rejects(what: &str, frame: &[u8], dict: Option<&DecodeDict>, want: &str) {
    for (opts, path) in paths() {
        match decompress_with_dict_options(frame, dict, &opts) {
            Ok(_) => panic!("{what} ({path}): decoded"),
            Err(e) => assert!(e.contains(want), "{what} ({path}): {e}"),
        }
    }
}

const LEVELS: [i32; 10] = [-5, 1, 2, 3, 5, 7, 12, 16, 19, 22];

#[test]
fn formatted_dicts_parse_like_libzstd() {
    for raw in [trained_dict(), entropy_dict()] {
        let dict = DecodeDict::new(&raw).unwrap();
        // SAFETY: `raw` is read during the calls only.
        let (id, header) = unsafe {
            (
                sys::ZDICT_getDictID(raw.as_ptr().cast(), raw.len()),
                sys::ZDICT_getDictHeaderSize(raw.as_ptr().cast(), raw.len()),
            )
        };
        assert_ne!(id, 0);
        assert_eq!(dict.id(), id);
        assert_eq!(dict.content(), &raw[header..]);
    }
}

#[test]
fn dict_frames_decode() {
    for (raw, name) in dicts() {
        dict_frames_decode_with(&raw, name);
    }
}

fn dict_frames_decode_with(raw: &[u8], name: &str) {
    let dict = DecodeDict::new(raw).unwrap();
    for (n, seed) in [(1, 1), (4, 2), (60, 3), (1500, 4)] {
        let src = records(n, seed);
        for level in LEVELS {
            let frame = c_compress_using_dict(&src, raw, level);
            let what = format!("{name}: {} bytes level {level}", src.len());
            assert_eq!(
                c_decompress_dict(&frame, raw, src.len()).as_deref(),
                Some(&src[..]),
                "{what}: libzstd"
            );
            assert_decodes(&what, &frame, &dict, &src);
            assert_eq!(decompress_with_dict(&frame, &dict).unwrap(), src, "{what}");
        }
    }
}

/// Blocks of at most 1 KiB with Huffman literals forced, so that later
/// blocks repeat the dictionary's tables after earlier blocks, and the MT
/// path plans `START` definitions across many blocks.
#[test]
fn dict_small_blocks_decode() {
    for (raw, name) in dicts() {
        dict_small_blocks_decode_with(&raw, name);
    }
}

fn dict_small_blocks_decode_with(raw: &[u8], name: &str) {
    let dict = DecodeDict::new(raw).unwrap();
    let src = records(400, 9);
    for level in [1, 3, 9, 19] {
        let frame = c_compress_dict(
            &src,
            raw,
            &[
                (P::ZSTD_c_compressionLevel, level),
                // ZSTD_c_maxBlockSize.
                (P::ZSTD_c_experimentalParam18, 1024),
                // ZSTD_c_literalCompressionMode = ZSTD_ps_enable.
                (P::ZSTD_c_experimentalParam5, 1),
                (P::ZSTD_c_checksumFlag, 1),
            ],
        );
        let what = format!("{name}: small blocks level {level}");
        assert_decodes(&what, &frame, &dict, &src);
    }
}

/// Every frame of a concatenation starts from the dictionary again.
#[test]
fn concatenated_dict_frames_decode() {
    for (raw, name) in dicts() {
        let dict = DecodeDict::new(&raw).unwrap();
        let a = records(3, 21);
        let b = records(200, 22);
        let mut frames = c_compress_using_dict(&a, &raw, 3);
        frames.extend(c_compress_using_dict(&b, &raw, 19));
        frames.extend(c_compress_using_dict(&a, &raw, 1));
        let want = [&a[..], &b, &a].concat();
        assert_decodes(&format!("{name}: three frames"), &frames, &dict, &want);
    }
}

/// A frame without a Dictionary_ID is decoded with the dictionary
/// supplied (ZSTD_decompress_usingDict).
#[test]
fn frame_without_dict_id_uses_supplied_dict() {
    for (raw, name) in dicts() {
        let dict = DecodeDict::new(&raw).unwrap();
        let src = records(30, 31);
        let frame = c_compress_dict(
            &src,
            &raw,
            &[(P::ZSTD_c_compressionLevel, 3), (P::ZSTD_c_dictIDFlag, 0)],
        );
        // SAFETY: `frame` is read during the call only.
        assert_eq!(
            unsafe { sys::ZSTD_getDictID_fromFrame(frame.as_ptr().cast(), frame.len()) },
            0
        );
        assert_decodes(&format!("{name}: no dict id"), &frame, &dict, &src);
        assert!(rust_zstd::decompress(&frame).map_or(true, |out| out != src));
    }
}

#[test]
fn wrong_or_missing_dict_is_an_error() {
    let raw = trained_dict();
    let dict = DecodeDict::new(&raw).unwrap();
    let src = records(20, 51);
    let frame = c_compress_using_dict(&src, &raw, 3);

    // Same tables and content, another Dictionary_ID.
    let mut other = raw.clone();
    other[4] ^= 1;
    let other = DecodeDict::new(&other).unwrap();
    assert_rejects("wrong id", &frame, Some(&other), "dictionary");
    // A raw-content dictionary has Dictionary_ID 0.
    let content = DecodeDict::new(dict.content()).unwrap();
    assert_rejects("raw dict", &frame, Some(&content), "dictionary");
    assert_rejects("no dict", &frame, None, "none is loaded");
}

/// Every truncation of a trained dictionary loads or fails as with
/// ZSTD_createDDict, through the entropy section and repeat offsets.
#[test]
fn truncated_dict_verdict_matches_libzstd() {
    let raw = trained_dict();
    // SAFETY: `raw` is read during the call only.
    let header = unsafe { sys::ZDICT_getDictHeaderSize(raw.as_ptr().cast(), raw.len()) };
    let mut rejected = 0;
    for len in 0..header + 64 {
        let cut = &raw[..len];
        let ours = DecodeDict::new(cut);
        assert_eq!(
            ours.is_ok(),
            c_loads_dict(cut),
            "length {len}: {:?}",
            ours.err()
        );
        rejected += usize::from(ours.is_err());
    }
    // Lengths 8..header + 12-ish are the corrupt ones.
    assert!(rejected > header - 16, "{rejected} rejected");
}

/// Single-byte corruptions of the entropy section load or fail as with
/// ZSTD_createDDict, and the dictionaries that load decode what libzstd
/// decodes with them.
#[test]
fn corrupt_entropy_verdict_matches_libzstd() {
    let raw = entropy_dict();
    // SAFETY: `raw` is read during the call only.
    let header = unsafe { sys::ZDICT_getDictHeaderSize(raw.as_ptr().cast(), raw.len()) };
    let src = records(40, 61);
    let frame = c_compress_using_dict(&src, &raw, 3);
    let (mut loads, mut fails) = (0, 0);
    for at in 8..header {
        for x in [0x01u8, 0x10, 0x80, 0xFF] {
            let mut bad = raw.clone();
            bad[at] ^= x;
            let ours = DecodeDict::new(&bad);
            assert_eq!(
                ours.is_ok(),
                c_loads_dict(&bad),
                "byte {at} ^ {x:#x}: {:?}",
                ours.as_ref().err()
            );
            let Ok(dict) = ours else {
                fails += 1;
                continue;
            };
            loads += 1;
            let lib = c_decompress_dict(&frame, &bad, src.len() + 1024);
            for (opts, path) in paths() {
                let out = decompress_with_dict_options(&frame, Some(&dict), &opts).ok();
                assert_eq!(out, lib, "byte {at} ^ {x:#x} ({path})");
            }
        }
    }
    assert!(loads > 0 && fails > 0, "{loads} load, {fails} fail");
}
