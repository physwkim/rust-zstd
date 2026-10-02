//! The multi-threaded frame decoder (`parallel` feature) against libzstd
//! and against the serial decoder: byte-exact output on every dataset, and
//! the same `Err`-vs-`Ok` outcome (and the same output when `Ok`) on
//! truncated and corrupted multi-block frames.

#![cfg(feature = "parallel")]

mod common;

use common::{
    assert_lockstep, datasets, frame_blocks, lcg_bytes, zstd_bulk, zstd_stream, LEVELS, MIB,
    PARALLEL_ROOM,
};
use rust_zstd::decode::{decompress_with_options, DecodeOptions};
use rust_zstd::Decompressor;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use zstd::zstd_safe::zstd_sys as sys;

/// Every frame through the multi-threaded path, however few its blocks.
fn decode_mt(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_with_options(
        data,
        &DecodeOptions {
            min_parallel_blocks: 1,
            min_parallel_bytes: 0,
            simd: true,
            window_log_max: 0,
        },
    )
}

/// Every frame through the fused serial path.
fn decode_serial(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_with_options(
        data,
        &DecodeOptions {
            min_parallel_blocks: usize::MAX,
            min_parallel_bytes: 0,
            simd: true,
            window_log_max: 0,
        },
    )
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

/// libzstd frame with blocks of at most `max_block` bytes and Huffman
/// literals forced on, so that small inputs give many blocks, most of them
/// with treeless literals and repeat-mode FSE tables.
fn zstd_small_blocks(data: &[u8], level: i32, max_block: i32) -> Vec<u8> {
    zstd_small_blocks_with(data, level, max_block, &[])
}

/// `zstd_small_blocks` with the compression parameters `params` set too.
fn zstd_small_blocks_with(
    data: &[u8],
    level: i32,
    max_block: i32,
    params: &[(sys::ZSTD_cParameter, i32)],
) -> Vec<u8> {
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let set = |p, v| {
            let r = sys::ZSTD_CCtx_setParameter(cctx, p, v);
            assert_eq!(sys::ZSTD_isError(r), 0, "set parameter {:?}", p);
        };
        set(sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level);
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam18, max_block);
        // ZSTD_c_literalCompressionMode = ZSTD_ps_enable.
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam5, 1);
        for &(p, v) in params {
            set(p, v);
        }
        let mut out = vec![0u8; sys::ZSTD_compressBound(data.len())];
        let n = sys::ZSTD_compress2(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            data.as_ptr().cast(),
            data.len(),
        );
        assert_eq!(sys::ZSTD_isError(n), 0);
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

#[test]
fn mt_decodes_libzstd_streams_byte_exact() {
    let pool = pool(4);
    for ds in datasets() {
        for &level in &LEVELS {
            for (kind, compressed) in [
                ("bulk", zstd_bulk(&ds.data, level)),
                ("stream", zstd_stream(&ds.data, level)),
            ] {
                let decoded = pool
                    .install(|| decode_mt(&compressed))
                    .unwrap_or_else(|e| panic!("{} L{} {}: {}", ds.name, level, kind, e));
                assert!(
                    decoded == ds.data,
                    "{} L{} {}: output differs",
                    ds.name,
                    level,
                    kind
                );
            }
        }
    }
}

/// Multi-job frames from this crate's encoder: every job boundary starts
/// fresh entropy tables, and later jobs match into earlier jobs' output.
#[test]
fn mt_decodes_multi_job_frames_from_our_encoder() {
    let pool = pool(3);
    for ds in datasets() {
        let data = &ds.data[..ds.data.len().min(2 * MIB)];
        for &level in &LEVELS {
            let opts = rust_zstd::CompressOptions {
                level,
                job_size: Some(512 * 1024),
                ..rust_zstd::CompressOptions::default()
            };
            let compressed = rust_zstd::compress_with(data, &opts);
            let decoded = pool
                .install(|| decode_mt(&compressed))
                .unwrap_or_else(|e| panic!("{} L{}: {}", ds.name, level, e));
            assert!(decoded == data, "{} L{}: output differs", ds.name, level);
        }
    }
}

/// Frames of one to a few blocks, which `decompress` leaves to the serial
/// path, forced through the multi-threaded one; also a pool of one thread.
#[test]
fn mt_path_forced_on_small_frames() {
    let text = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
    let mut mixed = lcg_bytes(3000, 5);
    mixed.extend_from_slice(&[0u8; 5000]);
    mixed.extend_from_slice(&text);
    let cases = [
        ("one block", zstd_bulk(&text, 3), text.clone()),
        ("empty", zstd_bulk(&[], 3), Vec::new()),
        (
            "small blocks",
            zstd_small_blocks(&mixed, 3, 1024),
            mixed.clone(),
        ),
        (
            "small blocks L19",
            zstd_small_blocks(&text, 19, 1024),
            text.clone(),
        ),
    ];
    for threads in [1, 2, 8] {
        let pool = pool(threads);
        for (name, compressed, want) in &cases {
            let got = pool.install(|| decode_mt(compressed)).unwrap();
            assert!(
                &got == want,
                "{} ({} threads): output differs",
                name,
                threads
            );
        }
    }
    // Concatenated frames keep separate offset histories and tables.
    let mut two = cases[2].1.clone();
    two.extend_from_slice(&cases[3].1);
    let mut want = mixed;
    want.extend_from_slice(&text);
    assert!(pool(2).install(|| decode_mt(&two)).unwrap() == want);
}

fn same_outcome(name: &str, input: &[u8]) {
    let serial = std::panic::catch_unwind(|| decode_serial(input))
        .unwrap_or_else(|_| panic!("{}: serial path panicked", name));
    let mt = std::panic::catch_unwind(|| decode_mt(input))
        .unwrap_or_else(|_| panic!("{}: multi-threaded path panicked", name));
    match (&serial, &mt) {
        (Ok(a), Ok(b)) => assert!(a == b, "{}: outputs differ", name),
        (Err(_), Err(_)) => {}
        _ => panic!(
            "{}: serial {:?} vs multi-threaded {:?}",
            name,
            serial.as_ref().map(Vec::len),
            mt.as_ref().map(Vec::len)
        ),
    }
}

fn small_block_inputs() -> Vec<(String, Vec<u8>)> {
    let mut data = b"The quick brown fox jumps over the lazy dog. ".repeat(60);
    data.extend_from_slice(&lcg_bytes(1500, 11));
    data.extend_from_slice(&b"abcabcabd".repeat(200));
    let mut out = Vec::new();
    for &level in &[1, 3, 19] {
        let c = zstd_small_blocks(&data, level, 1024);
        assert_eq!(decode_mt(&c).unwrap(), data);
        out.push((format!("L{}", level), c));
    }
    out
}

/// Truncation at every offset of multi-block frames: `Err` (never a panic)
/// through the multi-threaded path exactly when the serial path errs.
#[test]
fn mt_truncation_matches_serial_outcome() {
    pool(4).install(|| {
        for (name, c) in small_block_inputs() {
            for cut in 0..c.len() {
                same_outcome(&format!("{} cut {}", name, cut), &c[..cut]);
                if cut > 0 {
                    assert!(decode_mt(&c[..cut]).is_err(), "{} cut {}", name, cut);
                }
            }
        }
    });
}

/// Byte corruptions of multi-block frames: same outcome as the serial path.
#[test]
fn mt_corruption_matches_serial_outcome() {
    pool(4).install(|| {
        for (name, c) in small_block_inputs() {
            for pos in 0..c.len() {
                for flip in [0x01u8, 0x80, 0xFF, 0x55] {
                    let mut bad = c.clone();
                    bad[pos] ^= flip;
                    same_outcome(&format!("{} byte {} ^ {:#x}", name, pos, flip), &bad);
                }
            }
        }
    });
}

fn decompressor(min_parallel_blocks: usize) -> Decompressor {
    Decompressor::with_options(&DecodeOptions {
        min_parallel_blocks,
        min_parallel_bytes: 0,
        simd: true,
        window_log_max: 0,
    })
}

/// Streams of libzstd frames of full blocks, with output room for batches
/// of them: every call decoding them in parallel reads, writes and returns
/// what the serial one does, on pools of two and eight threads.
#[test]
fn mt_streams_match_serial_streams() {
    let mut cases = Vec::new();
    for ds in datasets() {
        let data = &ds.data[..ds.data.len().min(2 * MIB)];
        for level in [1, 19] {
            for (kind, c) in [
                ("bulk", zstd_bulk(data, level)),
                ("stream", zstd_stream(data, level)),
            ] {
                cases.push((format!("{} L{level} {kind}", ds.name), c, data.len()));
            }
        }
    }
    for threads in [2, 8] {
        pool(threads).install(|| {
            for (name, c, len) in &cases {
                for chunk in [100_000, 300_000, usize::MAX] {
                    for room in [PARALLEL_ROOM, 3 * MIB / 2, len + 1] {
                        // From one block on, and from the default count on.
                        for mut parallel in [decompressor(1), Decompressor::new()] {
                            let what = format!("{name} {threads} threads");
                            let mut serial = decompressor(usize::MAX);
                            assert_lockstep(&what, &mut serial, &mut parallel, c, chunk, room);
                        }
                    }
                }
            }
        });
    }
}

/// Truncated and corrupted frames of small blocks, streamed: every call
/// decoding them in parallel reads, writes and returns what the serial one
/// does, failing ones included. The frames are ones the parallel decoder
/// takes: one whose Frame_Content_Size, checked after its blocks, exceeds
/// its 1 KiB window, and one without. With room for every block, a call
/// decodes them in one scope; with room for two, the pipeline decodes the
/// blocks past it ahead, a failing one among them.
#[test]
fn mt_stream_verdicts_match_serial() {
    use sys::ZSTD_cParameter::{ZSTD_c_checksumFlag, ZSTD_c_contentSizeFlag, ZSTD_c_windowLog};
    // Room for every block of these frames.
    const ROOM: usize = 1 << 16;
    let mut data = b"The quick brown fox jumps over the lazy dog. ".repeat(60);
    data.extend_from_slice(&lcg_bytes(1500, 11));
    data.extend_from_slice(&b"abcabcabd".repeat(200));
    let mut inputs = Vec::new();
    for level in [1, 19] {
        for (kind, params) in [
            ("sized", [(ZSTD_c_windowLog, 10), (ZSTD_c_checksumFlag, 1)]),
            (
                "unsized",
                [(ZSTD_c_contentSizeFlag, 0), (ZSTD_c_checksumFlag, 0)],
            ),
        ] {
            let c = zstd_small_blocks_with(&data, level, 1024, &params);
            assert_eq!(decode_mt(&c).unwrap(), data);
            inputs.push((format!("L{level} {kind}"), c));
        }
    }
    let lockstep = |what: &str, input: &[u8], (chunk, room)| {
        let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
        assert_lockstep(what, &mut serial, &mut parallel, input, chunk, room);
    };
    pool(4).install(|| {
        for (name, c) in inputs {
            for at in [(700, ROOM), (usize::MAX, ROOM), (usize::MAX, 2 << 10)] {
                for cut in 0..c.len() {
                    lockstep(&format!("{name} cut {cut}"), &c[..cut], at);
                }
                for pos in 0..c.len() {
                    for flip in [0x01u8, 0x80, 0xFF] {
                        let mut bad = c.clone();
                        bad[pos] ^= flip;
                        lockstep(&format!("{name} byte {pos} ^ {flip:#x}"), &bad, at);
                    }
                }
            }
        }
    });
}

/// Frames of 1 KiB blocks, in a 1 KiB window so that the parallel decoder
/// takes them, whose batches hold no compressed block, one fewer than
/// `min_parallel_blocks`, or that many of one byte fewer than
/// `min_parallel_bytes` (all decoded one after another), or exactly that
/// many bytes (on the pool), among raw and RLE blocks: whole,
/// truncated and corrupted, every call reads, writes and returns what the
/// serial decoder does.
#[test]
fn mt_batches_either_side_of_the_gate() {
    use sys::ZSTD_cParameter::ZSTD_c_windowLog;
    const MIN: usize = 3;
    // Room for every block of these frames.
    const ROOM: usize = 1 << 16;
    let text = b"The quick brown fox jumps over the lazy dog. ".repeat(24);
    let mut cases = Vec::new();
    // Block types: `r` raw (0), `z` RLE (1), `c` compressed (2).
    for kinds in ["rzrzr", "crzcz", "crczc"] {
        let mut data = Vec::new();
        for (i, kind) in kinds.bytes().enumerate() {
            data.extend(match kind {
                b'c' => text[i..][..1024].to_vec(),
                b'r' => lcg_bytes(1024, i as u64),
                _ => vec![i as u8; 1024],
            });
        }
        let c = zstd_small_blocks_with(&data, 3, 1024, &[(ZSTD_c_windowLog, 10)]);
        let blocks = frame_blocks(&c, data.len()).0;
        let types: String = blocks
            .iter()
            .map(|b| ['r', 'z', 'c'][b.ty as usize])
            .collect();
        assert_eq!(types, kinds, "block types");
        // The Block_Size of the compressed blocks.
        let bytes: usize = blocks
            .iter()
            .filter(|b| b.ty == 2)
            .map(|b| b.c_size - 3)
            .sum();
        if kinds == "crczc" {
            for min_bytes in [bytes, bytes + 1] {
                let name = format!("{kinds} min_parallel_bytes {min_bytes} of {bytes}");
                cases.push((name, c.clone(), data.clone(), min_bytes));
            }
        } else {
            cases.push((kinds.to_string(), c, data, 0));
        }
    }
    let opts = |min_parallel_bytes| DecodeOptions {
        min_parallel_blocks: MIN,
        min_parallel_bytes,
        simd: true,
        window_log_max: 0,
    };
    let lockstep = |what: &str, input: &[u8], min_bytes| {
        let mut serial = decompressor(usize::MAX);
        let mut parallel = Decompressor::with_options(&opts(min_bytes));
        assert_lockstep(what, &mut serial, &mut parallel, input, usize::MAX, ROOM);
    };
    for threads in [1, 4] {
        pool(threads).install(|| {
            for (name, c, data, min_bytes) in &cases {
                let got = decompress_with_options(c, &opts(*min_bytes)).unwrap();
                assert!(got == *data, "{name}");
                lockstep(name, c, *min_bytes);
            }
        });
    }
    pool(4).install(|| {
        for (name, c, _, min_bytes) in &cases {
            for cut in 0..c.len() {
                lockstep(&format!("{name} cut {cut}"), &c[..cut], *min_bytes);
            }
            for pos in 0..c.len() {
                for flip in [0x01u8, 0x80, 0xFF] {
                    let mut bad = c.clone();
                    bad[pos] ^= flip;
                    let what = format!("{name} byte {pos} ^ {flip:#x}");
                    lockstep(&what, &bad, *min_bytes);
                }
            }
        }
    });
}

/// `assert_lockstep` at the output room `room`, the first call reading
/// `first` and the others `then`, from where the calls before stopped
/// reading; returns the content written.
fn lockstep_switching(
    what: &str,
    serial: &mut Decompressor,
    parallel: &mut Decompressor,
    [first, then]: [&[u8]; 2],
    room: usize,
) -> Vec<u8> {
    let (mut a, mut b) = (vec![0u8; room], vec![0u8; room]);
    let (mut pos, mut content) = (0, Vec::new());
    for call in 0.. {
        let src = &[first, then][usize::from(call != 0)][pos..];
        let (mut read, mut written, mut b_read, mut b_written) = (0, 0, 0, 0);
        let hint = serial.decompress_stream(src, &mut read, &mut a, &mut written);
        let b_hint = parallel.decompress_stream(src, &mut b_read, &mut b, &mut b_written);
        assert!(
            (&hint, read, &a[..written]) == (&b_hint, b_read, &b[..b_written]),
            "{what}, call {call}: serial gives {hint:?}, reading {read} and writing \
             {written}; parallel {b_hint:?}, reading {b_read} and writing {b_written}"
        );
        content.extend_from_slice(&a[..written]);
        pos += read;
        if hint.is_err() || read == 0 && written == 0 && hint != Ok(0) {
            break;
        }
    }
    assert_eq!(serial.finish(), parallel.finish(), "{what}: finish");
    content
}

/// A stream whose input after the first call is another frame's, the same
/// up to a block that the first call's batch decoded ahead, past its room,
/// or one after: the parallel decoder takes the blocks decoded ahead up to
/// that one, and decodes the rest of the other frame as the serial one does.
#[test]
fn mt_stream_takes_blocks_decoded_ahead_only_where_input_matches() {
    use sys::ZSTD_cParameter::{ZSTD_c_checksumFlag, ZSTD_c_windowLog};
    // A few blocks of 1 KiB, in a 1 KiB window.
    const ROOM: usize = 4 << 10;
    const BLOCK: usize = 1 << 10;
    let words: Vec<&[u8]> = [
        &b"quick "[..],
        b"brown ",
        b"fox ",
        b"jumps ",
        b"over ",
        b"lazy ",
    ]
    .to_vec();
    let mut data = Vec::new();
    for (i, r) in lcg_bytes(8000, 5).into_iter().enumerate() {
        data.extend_from_slice(words[usize::from(r) % words.len()]);
        data.push(b'a' + (i % 26) as u8);
    }
    data.truncate(40 * BLOCK);
    let params = [(ZSTD_c_windowLog, 10), (ZSTD_c_checksumFlag, 1)];
    let a = zstd_small_blocks_with(&data, 3, BLOCK as i32, &params);
    let (blocks, _) = frame_blocks(&a, data.len());
    // Where each block starts in `a`, after the frame header, and in the
    // content.
    let mut starts = vec![(
        a.len() - 4 - blocks.iter().map(|b| b.c_size).sum::<usize>(),
        0,
    )];
    for b in &blocks {
        let &(c, d) = starts.last().unwrap();
        starts.push((c + b.c_size, d + b.size));
    }
    // The first unread block after the first call.
    let mut serial = decompressor(usize::MAX);
    let (mut read, mut written) = (0, 0);
    serial
        .decompress_stream(&a, &mut read, &mut vec![0u8; ROOM], &mut written)
        .unwrap();
    let first = starts.iter().position(|&(c, _)| c == read).unwrap();
    pool(4).install(|| {
        // Pools of four threads decode up to eight blocks ahead.
        for d in first..first + 9 {
            let mut other = data.clone();
            other[starts[d].1 + BLOCK / 2] ^= 0x20;
            let b = zstd_small_blocks_with(&other, 3, BLOCK as i32, &params);
            let (start, end) = (starts[d].0, starts[d + 1].0);
            assert_eq!(a[..start], b[..start], "block {d}");
            assert_ne!(a[..end], b[..end], "block {d}");
            let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
            let what = format!("changed in block {d}");
            let got = lockstep_switching(&what, &mut serial, &mut parallel, [&a, &b], ROOM);
            assert!(got == other, "{what}");
        }
    });
}

/// Streams of 1 KiB blocks whose Huffman and FSE tables change every few
/// blocks, at rooms that stop batches a block or a few in, so that the
/// next call's batch takes blocks decoded ahead with tables the frame has
/// since replaced, or decodes them again: every call reads, writes and
/// returns what the serial decoder does.
#[test]
fn mt_streams_change_tables_between_batches() {
    use sys::ZSTD_cParameter::ZSTD_c_windowLog;
    const BLOCK: usize = 1 << 10;
    let data = changing_words_data(9);
    let mut cases = Vec::new();
    for level in [1, 19] {
        let c = zstd_small_blocks_with(&data, level, BLOCK as i32, &[(ZSTD_c_windowLog, 10)]);
        assert_eq!(decode_mt(&c).unwrap(), data);
        cases.push((format!("L{level}"), c));
    }
    for threads in [2, 3, 4, 8] {
        pool(threads).install(|| {
            for (name, c) in &cases {
                for chunk in [300, 700, 1000, 1500, 2000, 3000, 5000, usize::MAX] {
                    for room in [1, 2, 3, 4, 5, 6, 7, 9, 13].map(|n| n * BLOCK) {
                        let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
                        let what = format!("{name} {threads} threads");
                        assert_lockstep(&what, &mut serial, &mut parallel, c, chunk, room);
                    }
                }
            }
        });
    }
}

/// 40 KiB of words, each followed by one of its letters, whose alphabet
/// changes every two to three blocks of 1 KiB: in blocks of 1 KiB, their
/// Huffman and FSE tables change every few blocks.
fn changing_words_data(seed: u64) -> Vec<u8> {
    const BLOCK: usize = 1 << 10;
    let alphabets: [&[&[u8]]; 3] = [
        &[b"quick ", b"brown ", b"fox ", b"jumps ", b"over ", b"lazy "],
        &[b"0123 ", b"4567, ", b"89. ", b"1000 ", b"-42 "],
        &[b"ALPHA ", b"BETA; ", b"GAMMA ", b"DELTA! ", b"ZETA "],
    ];
    let mut data = Vec::new();
    let noise = lcg_bytes(40 * BLOCK / 5, seed + 1);
    for (i, r) in lcg_bytes(40 * BLOCK / 5, seed).into_iter().enumerate() {
        let words = alphabets[(i / 500 + i / 1300) % 3];
        let word = words[usize::from(r) % words.len()];
        data.extend_from_slice(word);
        // Literals of the alphabet's letters, for Huffman tables.
        data.push(word[usize::from(noise[i]) % word.len()]);
    }
    data.truncate(40 * BLOCK);
    data
}

/// `changing_words_data` as a frame of 1 KiB blocks in a 1 KiB window, so
/// that a stream call with a few KiB of room leaves blocks for the
/// pipeline to decode ahead, with tables the frame replaces later.
fn changing_words(seed: u64) -> (Vec<u8>, Vec<u8>) {
    use sys::ZSTD_cParameter::{ZSTD_c_checksumFlag, ZSTD_c_windowLog};
    let data = changing_words_data(seed);
    let params = [(ZSTD_c_windowLog, 10), (ZSTD_c_checksumFlag, 1)];
    let c = zstd_small_blocks_with(&data, 3, 1 << 10, &params);
    (data, c)
}

/// Run `f` on a pool of `threads` threads, all but the one running it held
/// until the sender it gets sends or drops: tasks spawned meanwhile stay
/// queued, unless the thread running `f` takes them.
fn on_held_pool<R: Send>(threads: usize, f: impl FnOnce(mpsc::Sender<()>) -> R + Send) -> R {
    pool(threads).install(|| {
        let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
        let release_rx = Arc::new(Mutex::new(release.1));
        for _ in 1..threads {
            let (held, release_rx) = (held.0.clone(), release_rx.clone());
            rayon::spawn(move || {
                held.send(()).unwrap();
                let _ = release_rx.lock().unwrap().recv();
            });
        }
        for _ in 1..threads {
            held.1.recv().unwrap();
        }
        f(release.0)
    })
}

/// `assert_lockstep` at chunk `usize::MAX`, calling `between` with the
/// number of the call before each call but the first.
fn lockstep_between(
    what: &str,
    serial: &mut Decompressor,
    parallel: &mut Decompressor,
    input: &[u8],
    room: usize,
    mut between: impl FnMut(usize),
) -> Vec<u8> {
    let (mut a, mut b) = (vec![0u8; room], vec![0u8; room]);
    let (mut pos, mut content) = (0, Vec::new());
    for call in 0.. {
        if call != 0 {
            between(call);
        }
        let src = &input[pos..];
        let (mut read, mut written, mut b_read, mut b_written) = (0, 0, 0, 0);
        let hint = serial.decompress_stream(src, &mut read, &mut a, &mut written);
        let b_hint = parallel.decompress_stream(src, &mut b_read, &mut b, &mut b_written);
        assert!(
            (&hint, read, &a[..written]) == (&b_hint, b_read, &b[..b_written]),
            "{what}, room {room}, call {call}: serial gives {hint:?}, reading {read} and \
             writing {written}; parallel {b_hint:?}, reading {b_read} and writing {b_written}"
        );
        content.extend_from_slice(&a[..written]);
        pos += read;
        if hint.is_err() || read == 0 && written == 0 && hint != Ok(0) {
            break;
        }
    }
    assert_eq!(serial.finish(), parallel.finish(), "{what}: finish");
    content
}

/// Blocks the pipeline decoded ahead whose tasks finish before the next
/// call (a pause between calls), during it (no pause), or only once it
/// has started (the pool held through the first call, then released):
/// every call reads, writes and returns what the serial decoder does.
#[test]
fn mt_stream_takes_tasks_finished_before_or_during_the_next_call() {
    let (data, c) = changing_words(9);
    for room in [2 << 10, 5 << 10] {
        for threads in [2, 4] {
            for when in ["before", "during"] {
                pool(threads).install(|| {
                    let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
                    let what = format!("finished {when}, {threads} threads");
                    let got = lockstep_between(&what, &mut serial, &mut parallel, &c, room, |_| {
                        if when == "before" {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    });
                    assert!(got == data, "{what}");
                });
            }
            on_held_pool(threads, |release| {
                let mut release = Some(release);
                let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
                let what = format!("started after the first call, {threads} threads");
                let got = lockstep_between(&what, &mut serial, &mut parallel, &c, room, |_| {
                    release.take();
                });
                assert!(got == data, "{what}");
            });
        }
    }
}

/// A stream reset, or a decompressor dropped, while the blocks the
/// pipeline planned past the first call's room are queued on a held pool:
/// the next frame, decoded while the pool is still held or once it has
/// run them, gives what the serial decoder gives, call by call.
#[test]
fn mt_stream_resets_with_blocks_in_flight() {
    const ROOM: usize = 2 << 10;
    let (a_data, a) = changing_words(9);
    let (b_data, b) = changing_words(10);
    for end in ["reset", "drop"] {
        for released in [false, true] {
            on_held_pool(4, |release| {
                let what = format!("{end}, pool released {released}");
                let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
                let (mut out, mut read, mut written) = (vec![0u8; ROOM], 0, 0);
                parallel
                    .decompress_stream(&a, &mut read, &mut out, &mut written)
                    .unwrap();
                assert!(written != 0 && read < a.len(), "{what}: a first call");
                assert!(out[..written] == a_data[..written], "{what}: a first call");
                match end {
                    "reset" => parallel.reset(),
                    _ => parallel = decompressor(1),
                }
                if released {
                    drop(release);
                    std::thread::sleep(Duration::from_millis(20));
                    let got = lockstep_between(&what, &mut serial, &mut parallel, &b, ROOM, |_| {});
                    assert!(got == b_data, "{what}");
                } else {
                    let got = lockstep_between(&what, &mut serial, &mut parallel, &b, ROOM, |_| {});
                    assert!(got == b_data, "{what}");
                    drop(release);
                }
            });
        }
    }
}

/// A stream whose input after the first call is the same frame with one
/// compressed block that the first call's pipeline planned ahead, past its
/// room, made raw: the blocks after it are the frame's own, whose Treeless
/// and Repeat references now resolve to the tables in use before it. The
/// pipeline plans them again from those tables, and every call reads,
/// writes and returns what the serial decoder does.
#[test]
fn mt_stream_replans_from_the_tables_before_a_changed_block() {
    const ROOM: usize = 2 << 10;
    let (data, a) = changing_words(9);
    let (blocks, _) = frame_blocks(&a, data.len());
    // Where each block starts in `a`, after the frame header, and in the
    // content.
    let mut starts = vec![(
        a.len() - 4 - blocks.iter().map(|b| b.c_size).sum::<usize>(),
        0,
    )];
    for b in &blocks {
        let &(c, d) = starts.last().unwrap();
        starts.push((c + b.c_size, d + b.size));
    }
    // The first unread block after the first call.
    let mut serial = decompressor(usize::MAX);
    let (mut read, mut written) = (0, 0);
    serial
        .decompress_stream(&a, &mut read, &mut vec![0u8; ROOM], &mut written)
        .unwrap();
    let first = starts.iter().position(|&(c, _)| c == read).unwrap();
    let mut changed = 0;
    pool(4).install(|| {
        // Pools of four threads decode up to eight blocks ahead.
        for d in first..first + 9 {
            if blocks[d].ty != 2 {
                continue;
            }
            let ((c, o), last) = (starts[d], d + 1 == blocks.len());
            let raw = u32::from(last) | (blocks[d].size as u32) << 3;
            let mut b = a[..c].to_vec();
            b.extend_from_slice(&raw.to_le_bytes()[..3]);
            b.extend_from_slice(&data[o..o + blocks[d].size]);
            b.extend_from_slice(&a[starts[d + 1].0..]);
            let [mut serial, mut parallel] = [usize::MAX, 1].map(decompressor);
            let what = format!("block {d} raw");
            lockstep_switching(&what, &mut serial, &mut parallel, [&a, &b], ROOM);
            changed += 1;
        }
    });
    assert!(changed >= 4, "{changed} compressed blocks made raw");
}
