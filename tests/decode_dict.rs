//! Dictionary decoding against libzstd: frames compressed with trained
//! (formatted) and raw-content dictionaries decode byte-exact on every
//! decoder path, dictionaries libzstd rejects are rejected, and frames
//! with the wrong or no dictionary, or offsets reaching past the
//! dictionary or the window, are errors.

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
    vec![
        (trained_dict(), "trained"),
        (entropy_dict(), "entropy-only"),
    ]
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
fn raw_content_dict_frames_decode() {
    let content = records(150, 41);
    assert_ne!(&content[..4], &0xEC30_A437u32.to_le_bytes());
    let dict = DecodeDict::new(&content).unwrap();
    assert_eq!(dict.id(), 0);
    assert_eq!(dict.content(), &content[..]);
    for (n, seed) in [(2, 42), (50, 43), (1000, 44)] {
        let src = records(n, seed);
        for level in LEVELS {
            let frame = c_compress_using_dict(&src, &content, level);
            assert_decodes(
                &format!("raw {n} records level {level}"),
                &frame,
                &dict,
                &src,
            );
        }
    }
}

/// Raw content shorter than 8 bytes is history too, as in libzstd.
#[test]
fn short_raw_content_dict_is_history() {
    let content = b"abcdefg";
    let dict = DecodeDict::new(content).unwrap();
    assert_eq!(dict.content(), content);
    // One compressed block: literal "xyz", then a match of 7 bytes at
    // offset 10 that copies the whole dictionary. LL, OF and ML in RLE
    // mode: LL code 3, OF code 3 (offset value 10 + 3 = 13 = 8 + 5,
    // 3 extra bits), ML code 4 (match length 7).
    let block_body = [
        0x18, b'x', b'y', b'z', // raw literals, size 3
        0x01, // one sequence
        0x54, // LL, OF, ML RLE
        3, 3, 4,    // the RLE codes
        0x0D, // bitstream: offset extra bits 5 (101), end mark
    ];
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0x20, 10];
    let header = (block_body.len() as u32) << 3 | 2 << 1 | 1;
    frame.extend_from_slice(&header.to_le_bytes()[..3]);
    frame.extend_from_slice(&block_body);
    let want = b"xyzabcdefg";
    assert_eq!(
        c_decompress_dict(&frame, content, 64).as_deref(),
        Some(&want[..])
    );
    assert_decodes("7-byte dict", &frame, &dict, want);
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

/// An offset reaching one byte before the dictionary content is an
/// error, for libzstd too.
#[test]
fn offset_before_dict_start_is_an_error() {
    let content = lcg_bytes(4096, 71);
    let src = [&lcg_bytes(100, 72)[..], &content[..2000]].concat();
    let frame = c_compress_using_dict(&src, &content, 3);
    assert_decodes(
        "whole dict",
        &frame,
        &DecodeDict::new(&content).unwrap(),
        &src,
    );
    let short = &content[1..];
    assert_eq!(c_decompress_dict(&frame, short, src.len() + 1024), None);
    assert_rejects(
        "dict one byte short",
        &frame,
        Some(&DecodeDict::new(short).unwrap()),
        "before the frame start",
    );
}

/// RFC 8878 lines 1838-1844: an offset past Window_Size may reach the
/// dictionary while the frame has decoded at most Window_Size bytes, and
/// not after. The frame is libzstd's with a 256 KiB window, relabelled to
/// 128 KiB; its first block is the `fresh` bytes up to 128 KiB, and the
/// second starts with what is left of them then a match into the
/// dictionary. libzstd enforces no window on offsets and decodes both.
#[test]
fn dict_reach_ends_at_window_size() {
    let content = lcg_bytes(64 * 1024, 81);
    let dict = DecodeDict::new(&content).unwrap();
    for (fresh, ok) in [(128 * 1024, true), (128 * 1024 + 1, false)] {
        let src = [&lcg_bytes(fresh, 82)[..], &content[..4000]].concat();
        let mut frame = c_compress_dict(
            &src,
            &content,
            &[
                (P::ZSTD_c_compressionLevel, 3),
                (P::ZSTD_c_windowLog, 18),
                (P::ZSTD_c_contentSizeFlag, 0),
            ],
        );
        assert_decodes("256 KiB window", &frame, &dict, &src);
        // Frame_Header_Descriptor without Single_Segment_Flag, then the
        // Window_Descriptor: 2^(10 + 8) -> 2^(10 + 7).
        assert_eq!(frame[4] & 0x20, 0);
        assert_eq!(frame[5], 8 << 3);
        frame[5] = 7 << 3;
        assert_eq!(
            c_decompress_dict(&frame, &content, src.len()).as_deref(),
            Some(&src[..])
        );
        if ok {
            assert_decodes("match at Window_Size", &frame, &dict, &src);
        } else {
            assert_rejects(
                "match past Window_Size",
                &frame,
                Some(&dict),
                "exceeds Window_Size",
            );
        }
    }
}
