//! Dictionary decoding against libzstd: frames compressed with trained
//! (formatted) and raw-content dictionaries decode byte-exact on every
//! decoder path, dictionaries libzstd rejects are rejected, and frames
//! with the wrong or no dictionary, or offsets reaching past the
//! dictionary or the window, are errors. A reused `Decompressor` gives
//! the same verdicts, and so does one holding the dictionary, streaming
//! the input in pieces of any size.

mod common;

use common::{
    c_streaming, check_streamed, is_window_limit, lcg_bytes, read_all, stream_room, stream_with,
    STREAM_CHUNKS,
};
use rust_zstd::compress::{compress_with_dict, CompressDict};
use rust_zstd::decode::{
    decompress_with_dict, decompress_with_dict_options, DecodeDict, DecodeOptions, Decompressor,
};
use std::cell::RefCell;
use zstd::zstd_safe::zstd_sys as sys;

use sys::ZSTD_cParameter as P;

thread_local! {
    /// The bytes of each dictionary `ddict` parsed, by the address of its
    /// content, which its clones share, for libzstd to decode with.
    static DICT_BYTES: RefCell<Vec<(*const u8, Vec<u8>)>> = const { RefCell::new(Vec::new()) };
}

/// `DecodeDict::new(raw)`, which must succeed, remembered for `dict_bytes`.
fn ddict(raw: &[u8]) -> DecodeDict {
    let dict = DecodeDict::new(raw).unwrap();
    DICT_BYTES.with_borrow_mut(|d| d.push((dict.content().as_ptr(), raw.to_vec())));
    dict
}

/// The bytes `ddict` parsed `dict` from. A live dictionary's content was
/// registered last at its address: one registered there before was freed.
fn dict_bytes(dict: &DecodeDict) -> Vec<u8> {
    DICT_BYTES.with_borrow(|d| {
        d.iter()
            .rev()
            .find(|(at, _)| *at == dict.content().as_ptr())
            .map(|(_, raw)| raw.clone())
            .expect("a dictionary made by `ddict`")
    })
}

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

/// ZSTD_compress_usingCDict with a CDict of `dict` at `level`.
fn c_compress_using_cdict(src: &[u8], dict: &[u8], level: i32) -> Vec<u8> {
    // SAFETY: as `c_compress_dict`; the CDict outlives its use.
    unsafe {
        let cdict = sys::ZSTD_createCDict(dict.as_ptr().cast(), dict.len(), level);
        assert!(!cdict.is_null(), "create CDict");
        let cctx = sys::ZSTD_createCCtx();
        let mut out = vec![0u8; sys::ZSTD_compressBound(src.len())];
        let n = sys::ZSTD_compress_usingCDict(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            src.as_ptr().cast(),
            src.len(),
            cdict,
        );
        assert_eq!(sys::ZSTD_isError(n), 0, "compress level {level}");
        sys::ZSTD_freeCCtx(cctx);
        sys::ZSTD_freeCDict(cdict);
        out.truncate(n);
        out
    }
}

/// libzstd's verdict on `input` with `dict`, which its one-shot decoder
/// (ZSTD_decompress_usingDDict, with room for `capacity` bytes) and its
/// streaming one (ZSTD_DCtx_refDDict, then ZSTD_decompressStream until the
/// input ends between frames) must agree on: the content, or None.
fn c_decode_ddict(input: &[u8], dict: &[u8], capacity: usize) -> Option<Vec<u8>> {
    // SAFETY: each context is used only here, with the DDict, which
    // outlives it, and with buffers whose sizes it is given.
    unsafe {
        let ddict = sys::ZSTD_createDDict(dict.as_ptr().cast(), dict.len());
        assert!(!ddict.is_null(), "create DDict");
        let dctx = sys::ZSTD_createDCtx();
        let mut out = vec![0u8; capacity];
        let n = sys::ZSTD_decompress_usingDDict(
            dctx,
            out.as_mut_ptr().cast(),
            out.len(),
            input.as_ptr().cast(),
            input.len(),
            ddict,
        );
        sys::ZSTD_freeDCtx(dctx);
        let one_shot = (sys::ZSTD_isError(n) == 0).then(|| {
            out.truncate(n);
            out
        });

        let dctx = sys::ZSTD_createDCtx();
        assert_eq!(sys::ZSTD_isError(sys::ZSTD_DCtx_refDDict(dctx, ddict)), 0);
        let mut content = Vec::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut src = sys::ZSTD_inBuffer {
            src: input.as_ptr().cast(),
            size: input.len(),
            pos: 0,
        };
        // No input is no frame, which ends between frames.
        let mut hint = 0;
        let streamed = loop {
            if src.pos == src.size && hint == 0 {
                break Some(content);
            }
            let mut dst = sys::ZSTD_outBuffer {
                dst: buf.as_mut_ptr().cast(),
                size: buf.len(),
                pos: 0,
            };
            hint = sys::ZSTD_decompressStream(dctx, &mut dst, &mut src);
            if sys::ZSTD_isError(hint) != 0 {
                break None;
            }
            content.extend_from_slice(&buf[..dst.pos]);
            if src.pos == src.size && hint != 0 && dst.pos < dst.size {
                // Mid-frame, with all the content written out.
                break None;
            }
        };
        sys::ZSTD_freeDCtx(dctx);
        sys::ZSTD_freeDDict(ddict);
        assert!(
            one_shot == streamed,
            "libzstd one-shot {:?}, streaming {:?}",
            one_shot.as_ref().map(Vec::len),
            streamed.as_ref().map(Vec::len)
        );
        one_shot
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
        window_log_max: 0,
    };
    [
        (o(usize::MAX, true), "serial"),
        (o(usize::MAX, false), "serial portable"),
        (o(1, true), "mt"),
        (o(1, false), "mt portable"),
    ]
}

thread_local! {
    /// A decompressor for each of `paths()`, and a buffer for its
    /// `decompress_into` calls, reused by every check of a test, so that
    /// each call follows others with other dictionaries, none, or errors.
    static REUSED: RefCell<Vec<(Decompressor, Vec<u8>)>> = RefCell::new(
        paths()
            .iter()
            .map(|(opts, _)| (Decompressor::with_options(opts), Vec::new()))
            .collect(),
    );
}

/// `frame` decoded with `dict` by the reused decompressor of path `k`.
fn reused(k: usize, frame: &[u8], dict: Option<&DecodeDict>) -> Result<Vec<u8>, String> {
    REUSED.with_borrow_mut(|r| match dict {
        Some(dict) => r[k].0.decompress_with_dict(frame, dict),
        None => r[k].0.decompress(frame),
    })
}

/// `reused`, decoded into the reused buffer of path `k`.
fn reused_into(k: usize, frame: &[u8], dict: Option<&DecodeDict>) -> Result<Vec<u8>, String> {
    REUSED.with_borrow_mut(|r| {
        let (d, dst) = &mut r[k];
        let got = match dict {
            Some(dict) => d.decompress_into_with_dict(frame, dict, dst),
            None => d.decompress_into(frame, dst),
        };
        common::into_result(got, dst)
    })
}

thread_local! {
    /// A serial decompressor for each SIMD level, which each check gives
    /// its dictionary by `set_dict`, so that each streams after others with
    /// other dictionaries, none, or errors.
    static HOLDING: RefCell<[Decompressor; 2]> = RefCell::new([false, true].map(|simd| {
        Decompressor::with_options(&DecodeOptions {
            min_parallel_blocks: usize::MAX,
            simd,
            window_log_max: 0,
        })
    }));
}

fn outcome(r: &Result<Vec<u8>, String>) -> String {
    match r {
        Ok(content) => format!("{} bytes", content.len()),
        Err(e) => format!("error {e:?}"),
    }
}

/// A decompressor holding `dict`, or none, gives the outcome of the
/// one-shot `decompress_with_dict_options` (serial) on `input` at both
/// SIMD levels, the same content or the same error: streaming at each of
/// `STREAM_CHUNKS`, with as much output room up to 64 KiB, but for the
/// window limit's refusals that libzstd's streaming decoder shares
/// (`check_streamed`), and through its `decompress`. So does
/// `DecompressReader::with_dict`, in pieces of 7 bytes and whole, where it
/// refuses by the window limit only input libzstd refuses so in pieces of
/// a byte.
fn assert_streams(what: &str, input: &[u8], dict: Option<&DecodeDict>) {
    let raw = dict.map(dict_bytes);
    let dict_and_bytes = dict.zip(raw.as_deref());
    for simd in [false, true] {
        let opts = DecodeOptions {
            min_parallel_blocks: usize::MAX,
            simd,
            window_log_max: 0,
        };
        let want = decompress_with_dict_options(input, dict, &opts);
        HOLDING.with_borrow_mut(|h| {
            let d = &mut h[usize::from(simd)];
            for chunk in STREAM_CHUNKS {
                d.set_dict(dict);
                let got = stream_with(d, input, chunk, stream_room(chunk));
                let at = format!("{what} simd={simd} chunk {chunk}");
                check_streamed(&at, input, chunk, simd, dict_and_bytes, &got, &want);
            }
            let got = d.decompress(input);
            assert!(
                got == want,
                "{what} simd={simd}: decompress with the dictionary held gives {} where \
                 one-shot gives {}",
                outcome(&got),
                outcome(&want)
            );
        });
        if !simd {
            // `DecompressReader` decodes at the detected level.
            continue;
        }
        for piece in [7, usize::MAX] {
            let got = read_all(input, piece, 4096, dict).map_err(|e| {
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{what}");
                e.to_string()
            });
            if is_window_limit(&got) {
                assert_eq!(
                    c_streaming(input, 1, 1, 0, raw.as_deref()),
                    Err(sys::ZSTD_ErrorCode::ZSTD_error_frameParameter_windowTooLarge),
                    "{what} piece {piece}: the reader refuses a window libzstd takes"
                );
                continue;
            }
            assert!(
                got == want,
                "{what} piece {piece}: the reader gives {} where one-shot gives {}",
                outcome(&got),
                outcome(&want)
            );
        }
    }
}

/// `frame` decodes to `src` with `dict` on every path, fresh and reused,
/// and streaming with the dictionary held (`assert_streams`).
fn assert_decodes(what: &str, frame: &[u8], dict: &DecodeDict, src: &[u8]) {
    for (k, (opts, path)) in paths().into_iter().enumerate() {
        for (got, how) in [
            (decompress_with_dict_options(frame, Some(dict), &opts), ""),
            (reused(k, frame, Some(dict)), ", reused"),
            (reused_into(k, frame, Some(dict)), ", reused into"),
        ] {
            match got {
                Ok(out) => assert!(out == src, "{what} ({path}{how}): wrong output"),
                Err(e) => panic!("{what} ({path}{how}): {e}"),
            }
        }
    }
    assert_streams(what, frame, Some(dict));
}

/// `frame` fails with `dict` on every path, fresh and reused, with an
/// error containing `want`, and streaming with the dictionary held
/// (`assert_streams`).
fn assert_rejects(what: &str, frame: &[u8], dict: Option<&DecodeDict>, want: &str) {
    for (k, (opts, path)) in paths().into_iter().enumerate() {
        for (got, how) in [
            (decompress_with_dict_options(frame, dict, &opts), ""),
            (reused(k, frame, dict), ", reused"),
            (reused_into(k, frame, dict), ", reused into"),
        ] {
            match got {
                Ok(_) => panic!("{what} ({path}{how}): decoded"),
                Err(e) => assert!(e.contains(want), "{what} ({path}{how}): {e}"),
            }
        }
    }
    assert_streams(what, frame, dict);
}

const LEVELS: [i32; 10] = [-5, 1, 2, 3, 5, 7, 12, 16, 19, 22];

#[test]
fn formatted_dicts_parse_like_libzstd() {
    for raw in [trained_dict(), entropy_dict()] {
        let dict = ddict(&raw);
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
    let dict = ddict(raw);
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
    let dict = ddict(raw);
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
        let dict = ddict(&raw);
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
        let dict = ddict(&raw);
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

/// `src` through `decompress_stream`, with room for all of it, then
/// `finish`.
fn stream(d: &mut Decompressor, src: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; 1 << 20];
    let (mut src_pos, mut dst_pos) = (0, 0);
    while src_pos < src.len() {
        d.decompress_stream(src, &mut src_pos, &mut out, &mut dst_pos)?;
    }
    d.finish()?;
    out.truncate(dst_pos);
    Ok(out)
}

/// A reused `Decompressor` decodes each call with the dictionary that
/// call gives, or none: the dictionary of an earlier call, which may have
/// failed, carries over to neither `decompress`, `decompress_stream` nor a
/// call with another dictionary.
#[test]
fn reused_decompressor_takes_each_calls_dict() {
    let (raw_a, raw_b) = (trained_dict(), entropy_dict());
    let a = ddict(&raw_a);
    let b = ddict(&raw_b);
    let content = ddict(a.content());
    let src = records(60, 61);
    let fa = c_compress_using_dict(&src, &raw_a, 3);
    let fb = c_compress_using_dict(&src, &raw_b, 19);
    let fc = c_compress_using_dict(&src, a.content(), 3);
    // Needs `a` but names no dictionary.
    let fa_anon = c_compress_dict(
        &src,
        &raw_a,
        &[(P::ZSTD_c_compressionLevel, 3), (P::ZSTD_c_dictIDFlag, 0)],
    );
    let plain = zstd::bulk::compress(&src, 3).unwrap();
    let anon_alone = rust_zstd::decompress(&fa_anon);
    assert!(anon_alone.as_ref() != Ok(&src), "fa_anon decodes alone");
    let want = Ok(src.clone());
    for (opts, path) in paths() {
        let mut d = Decompressor::with_options(&opts);
        assert_eq!(d.decompress_with_dict(&fa, &a), want, "{path}: a");
        assert_eq!(d.decompress(&plain), want, "{path}: plain after a");
        assert_eq!(d.decompress(&fa_anon), anon_alone, "{path}: anon after a");
        assert_eq!(d.decompress_with_dict(&fa_anon, &a), want, "{path}: anon");
        let e = d.decompress(&fa).unwrap_err();
        assert!(e.contains("none is loaded"), "{path}: a without: {e}");
        let e = d.decompress_with_dict(&fa, &b).unwrap_err();
        assert!(e.contains("dictionary"), "{path}: a with b: {e}");
        assert_eq!(d.decompress_with_dict(&fb, &b), want, "{path}: b");
        assert_eq!(d.decompress_with_dict(&fc, &content), want, "{path}: c");
        assert_eq!(d.decompress_with_dict(&fa, &a), want, "{path}: a again");
        let e = stream(&mut d, &fa).unwrap_err();
        assert!(e.contains("none is loaded"), "{path}: a streamed: {e}");
        assert_eq!(
            d.decompress_with_dict(&fa, &a),
            want,
            "{path}: a after stream"
        );
        assert_eq!(stream(&mut d, &plain), want, "{path}: plain streamed");
    }
}

#[test]
fn raw_content_dict_frames_decode() {
    let content = records(150, 41);
    assert_ne!(&content[..4], &0xEC30_A437u32.to_le_bytes());
    let dict = ddict(&content);
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
    let dict = ddict(content);
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
    let dict = ddict(&raw);
    let src = records(20, 51);
    let frame = c_compress_using_dict(&src, &raw, 3);

    // Same tables and content, another Dictionary_ID.
    let mut other = raw.clone();
    other[4] ^= 1;
    let other = ddict(&other);
    assert_rejects("wrong id", &frame, Some(&other), "dictionary");
    // A raw-content dictionary has Dictionary_ID 0.
    let content = ddict(dict.content());
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
    assert_decodes("whole dict", &frame, &ddict(&content), &src);
    let short = &content[1..];
    assert_eq!(c_decompress_dict(&frame, short, src.len() + 1024), None);
    assert_rejects(
        "dict one byte short",
        &frame,
        Some(&ddict(short)),
        "before the dictionary start",
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
    let dict = ddict(&content);
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

/// A decompressor holding a dictionary starts every frame from it, across
/// frames, `reset` and its `decompress`, until `set_dict` changes it;
/// `decompress_with_dict` names another for its call only. `set_dict`
/// mid-frame starts the stream over.
#[test]
fn held_dict_starts_every_frame() {
    let (raw_a, raw_b) = (trained_dict(), entropy_dict());
    let a = ddict(&raw_a);
    let b = ddict(&raw_b);
    let src = records(60, 101);
    let fa = c_compress_using_dict(&src, &raw_a, 3);
    let fb = c_compress_using_dict(&src, &raw_b, 19);
    let fa2 = [&fa[..], &fa].concat();
    let want = Ok(src.clone());
    for (opts, path) in paths() {
        let mut d = Decompressor::with_options(&opts);
        d.set_dict(Some(&a));
        assert_eq!(stream(&mut d, &fa), want, "{path}: a");
        assert_eq!(
            stream(&mut d, &fa2),
            Ok([&src[..], &src].concat()),
            "{path}: a twice"
        );
        d.reset();
        assert_eq!(stream(&mut d, &fa), want, "{path}: a after reset");
        assert_eq!(d.decompress(&fa), want, "{path}: decompress a");
        assert_eq!(d.decompress_with_dict(&fb, &b), want, "{path}: b once");
        assert_eq!(d.decompress(&fa), want, "{path}: a after b once");
        let e = stream(&mut d, &fb).unwrap_err();
        assert!(e.contains("dictionary"), "{path}: b streamed with a: {e}");

        // Half of a frame, then `set_dict`.
        let (mut src_pos, mut dst_pos) = (0, 0);
        let mut out = vec![0u8; 1 << 20];
        d.set_dict(Some(&b));
        d.decompress_stream(&fb[..fb.len() / 2], &mut src_pos, &mut out, &mut dst_pos)
            .unwrap();
        assert!(d.finish().is_err(), "{path}: mid-frame");
        d.set_dict(Some(&a));
        assert_eq!(stream(&mut d, &fa), want, "{path}: a after half of b");

        d.set_dict(None);
        let e = stream(&mut d, &fa).unwrap_err();
        assert!(e.contains("none is loaded"), "{path}: a without: {e}");
        let e = d.decompress(&fa).unwrap_err();
        assert!(
            e.contains("none is loaded"),
            "{path}: decompress a without: {e}"
        );
    }
    assert_eq!(Decompressor::with_dict(&a).decompress(&fa), want);
    assert_eq!(stream(&mut Decompressor::with_dict(&b), &fb), want);
}

/// Frames of our encoder and of ZSTD_compress_usingCDict, which libzstd
/// decodes, decode with the dictionary on every path and streaming.
#[test]
fn our_and_cdict_frames_decode() {
    for (raw, name) in dicts() {
        let dict = ddict(&raw);
        for (n, seed) in [(1, 111), (40, 112), (800, 113)] {
            let src = records(n, seed);
            for level in [-5, 1, 3, 9, 19] {
                let ours = compress_with_dict(&src, &CompressDict::new(&raw, level).unwrap());
                let cdict = c_compress_using_cdict(&src, &raw, level);
                for (frame, by) in [(ours, "ours"), (cdict, "CDict")] {
                    let what = format!("{name}: {by} {} bytes level {level}", src.len());
                    assert_eq!(
                        c_decode_ddict(&frame, &raw, src.len() + 1024).as_deref(),
                        Some(&src[..]),
                        "{what}: libzstd"
                    );
                    assert_decodes(&what, &frame, &dict, &src);
                }
            }
        }
    }
}

/// Frames with windows of 1 KiB to 128 KiB over 250 KB of records, with
/// and without Frame_Content_Size: streaming, the round buffer starts over
/// many times, the first segment after the dictionary and the others after
/// the segment before.
#[test]
fn dict_frames_past_small_windows() {
    let src = records(2000, 121);
    for (raw, name) in dicts() {
        let dict = ddict(&raw);
        for window_log in [10, 12, 15, 17] {
            for content_size in [0, 1] {
                let frame = c_compress_dict(
                    &src,
                    &raw,
                    &[
                        (P::ZSTD_c_compressionLevel, 3),
                        (P::ZSTD_c_windowLog, window_log),
                        (P::ZSTD_c_contentSizeFlag, content_size),
                    ],
                );
                let what = format!("{name}: window log {window_log} fcs {content_size}");
                assert_eq!(
                    c_decode_ddict(&frame, &raw, src.len()).as_deref(),
                    Some(&src[..]),
                    "{what}: libzstd"
                );
                assert_decodes(&what, &frame, &dict, &src);
            }
        }
    }
}

/// Skippable frame of `size` bytes of User_Data, `fill` each.
fn skippable(size: u32, fill: u8) -> Vec<u8> {
    let mut f = 0x184D_2A53u32.to_le_bytes().to_vec();
    f.extend_from_slice(&size.to_le_bytes());
    f.resize(8 + size as usize, fill);
    f
}

/// Frames with and without a dictionary, and skippable ones, one after
/// another: with the dictionary held every frame starts from it, the plain
/// ones too, as libzstd decodes the input with ZSTD_DCtx_refDDict; those
/// decode to their content with a raw-content dictionary, which they never
/// reach. Without one, the first frame naming it fails, after the content
/// before it is written out.
#[test]
fn interleaved_dict_and_plain_frames() {
    let mut cases = dicts();
    cases.push((records(150, 131), "raw content"));
    for (raw, name) in cases {
        let dict = ddict(&raw);
        let (a, b, c) = (records(5, 132), records(300, 133), records(40, 134));
        let parts = [
            (zstd::bulk::compress(&b, 3).unwrap(), &b[..], "plain"),
            (c_compress_using_dict(&a, &raw, 3), &a, "dict"),
            (skippable(100, 7), &[], "skippable"),
            (zstd::bulk::compress(&c, 19).unwrap(), &c, "plain"),
            (
                compress_with_dict(&c, &CompressDict::new(&raw, 1).unwrap()),
                &c,
                "ours",
            ),
            (
                c_compress_dict(
                    &a,
                    &raw,
                    &[(P::ZSTD_c_compressionLevel, 3), (P::ZSTD_c_dictIDFlag, 0)],
                ),
                &a,
                "no id",
            ),
            (c_compress_using_cdict(&b, &raw, 12), &b, "CDict"),
            (zstd::bulk::compress(&a, 1).unwrap(), &a, "plain"),
        ];
        let input: Vec<u8> = parts.iter().flat_map(|(f, _, _)| f.clone()).collect();
        let content: Vec<u8> = parts.iter().flat_map(|(_, c, _)| c.to_vec()).collect();
        let what = format!("{name}: {}", parts.map(|(_, _, k)| k).join(", "));

        let lib = c_decode_ddict(&input, &raw, content.len() + 1024);
        let ours = decompress_with_dict(&input, &dict);
        assert_eq!(ours.as_ref().ok(), lib.as_ref(), "{what}: libzstd");
        if dict.id() == 0 {
            assert!(ours.as_ref() == Ok(&content), "{what}: content");
        }
        assert_streams(&what, &input, Some(&dict));

        assert_streams(&what, &input, None);
        if dict.id() != 0 {
            let e = rust_zstd::decompress(&input).unwrap_err();
            assert!(e.contains("none is loaded"), "{what}: {e}");
            let mut d = Decompressor::new();
            let (mut src_pos, mut dst_pos) = (0, 0);
            let mut out = vec![0u8; 1 << 20];
            let mut got = Err(String::new());
            for end in (7..input.len()).step_by(7).chain([input.len()]) {
                got = d.decompress_stream(&input[..end], &mut src_pos, &mut out, &mut dst_pos);
                if got.is_err() {
                    break;
                }
            }
            assert_eq!(got, Err(e), "{what}: streamed without");
            assert!(out[..dst_pos] == b[..], "{what}: content before");
        }
    }
}

/// A dictionary frame for the truncation and corruption sweeps, with its
/// dictionary and content.
struct SweepFrame {
    name: String,
    frame: Vec<u8>,
    raw: Vec<u8>,
    src: Vec<u8>,
}

/// Dictionary frames of each kind: a one-block frame with a checksum, many
/// blocks of at most 1 KiB with Huffman literals forced, and a frame of our
/// encoder; then one raw-content dictionary frame.
fn sweep_frames() -> Vec<SweepFrame> {
    let mut cases = Vec::new();
    for (raw, name) in dicts() {
        let src = records(4, 141);
        let frame = c_compress_dict(
            &src,
            &raw,
            &[(P::ZSTD_c_compressionLevel, 3), (P::ZSTD_c_checksumFlag, 1)],
        );
        cases.push(SweepFrame {
            name: format!("{name}: one block"),
            frame,
            raw: raw.clone(),
            src,
        });
        let src = records(12, 142);
        let frame = c_compress_dict(
            &src,
            &raw,
            &[
                (P::ZSTD_c_compressionLevel, 19),
                (P::ZSTD_c_experimentalParam18, 1024),
                (P::ZSTD_c_experimentalParam5, 1),
            ],
        );
        cases.push(SweepFrame {
            name: format!("{name}: small blocks"),
            frame,
            raw: raw.clone(),
            src,
        });
        let src = records(6, 143);
        let frame = compress_with_dict(&src, &CompressDict::new(&raw, 3).unwrap());
        cases.push(SweepFrame {
            name: format!("{name}: ours"),
            frame,
            raw,
            src,
        });
    }
    let raw = records(30, 144);
    let src = records(5, 145);
    cases.push(SweepFrame {
        name: "raw content".to_string(),
        frame: c_compress_using_dict(&src, &raw, 3),
        raw,
        src,
    });
    cases
}

/// `input` gets libzstd's verdict with `raw` on every path, and its
/// content when accepted, but where RFC 8878 decides otherwise; and it
/// streams as it decodes in one shot.
///
/// RFC 8878 rejects two kinds of corrupt frames libzstd accepts with a
/// DDict. One has a Huffman stream that is not consumed exactly, which
/// libzstd's X2 decoder lets pass at its last symbol (HUF_decodeLastSymbolX2
/// clamps the bits consumed); ours rejects it, as for every table
/// (R1-6, R2-3). The other has an offset into the dictionary's header or
/// entropy tables: only the content is history (RFC 8878 lines 1835-1837),
/// as for ZSTD_decompress_usingDict, which rejects it, but a DDict's
/// history starts at the dictionary's first byte (ZSTD_copyDDictParameters).
fn assert_libzstd_verdict(what: &str, input: &[u8], raw: &[u8], dict: &DecodeDict) {
    let lib = c_decode_ddict(input, raw, 1 << 20);
    for (opts, path) in paths() {
        let ours = decompress_with_dict_options(input, Some(dict), &opts);
        if ours.as_ref().ok() == lib.as_ref() {
            continue;
        }
        let rfc = match &ours {
            Err(e) if lib.is_some() && e == "Huffman stream not fully consumed" => true,
            Err(e) if lib.is_some() && e.contains("before the dictionary start") => {
                c_decompress_dict(input, raw, 1 << 20).is_none()
            }
            _ => false,
        };
        assert!(
            rfc,
            "{what} ({path}): libzstd {:?}, ours {}",
            lib.as_ref().map(Vec::len),
            outcome(&ours)
        );
    }
    assert_streams(what, input, Some(dict));
}

/// A dictionary frame cut short anywhere is an error, as for libzstd, and
/// streams to the same error at every chunk size; only no input at all
/// decodes, to nothing.
#[test]
fn truncated_dict_frames_verdict_matches_libzstd() {
    for SweepFrame {
        name,
        frame,
        raw,
        src,
    } in sweep_frames()
    {
        let dict = ddict(&raw);
        assert_eq!(decompress_with_dict(&frame, &dict), Ok(src), "{name}");
        for cut in 0..frame.len() {
            let what = format!("{name} cut {cut}");
            assert_libzstd_verdict(&what, &frame[..cut], &raw, &dict);
            assert_eq!(decompress_with_dict(&frame[..cut], &dict).is_ok(), cut == 0);
        }
    }
}

/// A dictionary frame with any byte altered gets libzstd's verdict, and
/// its content when accepted, and streams as it decodes in one shot.
#[test]
fn corrupt_dict_frames_verdict_matches_libzstd() {
    for SweepFrame {
        name, frame, raw, ..
    } in sweep_frames()
    {
        let dict = ddict(&raw);
        for at in 0..frame.len() {
            for x in [0x01u8, 0x80, 0xFF, 0x55] {
                let mut bad = frame.clone();
                bad[at] ^= x;
                assert_libzstd_verdict(&format!("{name} byte {at} ^ {x:#x}"), &bad, &raw, &dict);
            }
        }
    }
}
