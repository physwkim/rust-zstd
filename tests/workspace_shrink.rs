//! A reused context frees an oversized workspace when libzstd's does:
//! `ZSTD_resetCCtx_internal` resizes the workspace when it is too small
//! for the reset, or when three times the reset's need has been free for
//! more than `ZSTD_WORKSPACETOOLARGE_MAXDURATION` (128) resets
//! (`workspaceWasteful`). The size libzstd keeps is `ZSTD_sizeof_CCtx`
//! less a fresh context's, which must equal the `Compressor`'s after every
//! frame, and every frame must equal libzstd's and a fresh `Compressor`'s.

use rust_zstd::compress::{CompressOptions, Compressor, ParamSwitch};
use zstd::zstd_safe::zstd_sys as sys;

/// A reused `ZSTD_CCtx` with fixed parameters.
struct CCtx {
    cctx: *mut sys::ZSTD_CCtx,
    /// `ZSTD_sizeof_CCtx` before any frame: the context without workspace.
    empty: usize,
}

impl CCtx {
    fn new(params: &[(sys::ZSTD_cParameter, i32)]) -> Self {
        unsafe {
            let cctx = sys::ZSTD_createCCtx();
            for &(param, value) in params {
                let r = sys::ZSTD_CCtx_setParameter(cctx, param, value);
                assert_eq!(sys::ZSTD_isError(r), 0, "set {param:?}");
            }
            let empty = sys::ZSTD_sizeof_CCtx(cctx);
            Self { cctx, empty }
        }
    }

    /// `ZSTD_compress2`.
    fn compress(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; unsafe { sys::ZSTD_compressBound(data.len()) }];
        let n = unsafe {
            sys::ZSTD_compress2(
                self.cctx,
                out.as_mut_ptr().cast(),
                out.len(),
                data.as_ptr().cast(),
                data.len(),
            )
        };
        assert_eq!(unsafe { sys::ZSTD_isError(n) }, 0);
        out.truncate(n);
        out
    }

    /// `ZSTD_cwksp_sizeof` of the context's workspace.
    fn workspace(&self) -> usize {
        unsafe { sys::ZSTD_sizeof_CCtx(self.cctx) - self.empty }
    }
}

impl Drop for CCtx {
    fn drop(&mut self) {
        unsafe { sys::ZSTD_freeCCtx(self.cctx) };
    }
}

/// `len` xorshift64 bytes.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as u8
        })
        .collect()
}

/// `len` bytes repeating 4 KiB of noise: long matches, so even the opt
/// strategies run through it fast.
fn periodic(len: usize) -> Vec<u8> {
    noise(4096, 7).into_iter().cycle().take(len).collect()
}

/// Up to 64 bytes of noise repeated with small edits: a few matches.
fn small(len: usize, seed: u64) -> Vec<u8> {
    let period = noise(16 + seed as usize % 48, seed);
    let edits = noise(len, !seed);
    (0..len)
        .map(|i| match edits[i] % 16 {
            0 => edits[i],
            _ => period[i % period.len()],
        })
        .collect()
}

/// Compress `frames` in order on one `Compressor` and one `ZSTD_CCtx`,
/// both with `opts`/`params`, checking after every frame that the frame
/// equals libzstd's and a fresh `Compressor`'s and that the workspace
/// sizes agree. Returns the workspace size after every frame.
fn run(
    opts: &CompressOptions,
    params: &[(sys::ZSTD_cParameter, i32)],
    frames: &[Vec<u8>],
) -> Vec<usize> {
    let mut ours = Compressor::new(opts.clone());
    let mut c = CCtx::new(params);
    let mut sizes = Vec::new();
    for (i, data) in frames.iter().enumerate() {
        let frame = ours.compress_to_vec(data);
        assert!(frame == c.compress(data), "frame {i}: != libzstd");
        assert!(
            frame == Compressor::new(opts.clone()).compress_to_vec(data),
            "frame {i}: != fresh Compressor"
        );
        let size = c.workspace();
        assert_eq!(ours.workspace_sizes(), [size], "frame {i}");
        sizes.push(size);
    }
    sizes
}

/// The frames after which the workspace changed size.
fn resizes(sizes: &[usize]) -> Vec<usize> {
    (1..sizes.len())
        .filter(|&i| sizes[i] != sizes[i - 1])
        .collect()
}

fn level(level: i32) -> (CompressOptions, Vec<(sys::ZSTD_cParameter, i32)>) {
    let opts = CompressOptions {
        level,
        ..CompressOptions::default()
    };
    (
        opts,
        vec![(sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level)],
    )
}

fn with_ldm(
    (mut opts, mut params): (CompressOptions, Vec<(sys::ZSTD_cParameter, i32)>),
) -> (CompressOptions, Vec<(sys::ZSTD_cParameter, i32)>) {
    opts.ldm = ParamSwitch::Enable;
    params.push((sys::ZSTD_cParameter::ZSTD_c_enableLongDistanceMatching, 1));
    (opts, params)
}

/// A 64 MiB frame, then 200 frames of 1 KiB: the workspace grows for the
/// big frame and shrinks at the 129th small one, the first reset past 128
/// since the resize (libzstd counts every reset, not only oversized ones).
fn big_then_small(opts: &CompressOptions, params: &[(sys::ZSTD_cParameter, i32)]) {
    let mut frames = vec![periodic(64 << 20)];
    frames.extend((0..200).map(|i| small(1024, i)));
    let sizes = run(opts, params, &frames);
    assert_eq!(resizes(&sizes), [129]);
    assert!(
        sizes[129] * 3 <= sizes[128],
        "{} -> {}",
        sizes[128],
        sizes[129]
    );
}

#[test]
fn l19_shrinks_after_128_small_frames() {
    let (opts, params) = level(19);
    big_then_small(&opts, &params);
}

#[test]
fn l19_ldm_shrinks_after_128_small_frames() {
    let (opts, params) = with_ldm(level(19));
    big_then_small(&opts, &params);
}

/// Frames of one size need the workspace they have: it never shrinks,
/// however many follow.
#[test]
fn same_size_frames_never_shrink() {
    for (opts, params) in [level(1), level(12), level(19), with_ldm(level(16))] {
        let frames: Vec<_> = (0..300).map(|i| small(20_000, i)).collect();
        let sizes = run(&opts, &params, &frames);
        assert_eq!(resizes(&sizes), [] as [usize; 0], "L{}", opts.level);
    }
}

/// A fresh context's workspace for one frame, by level, size (`0` too:
/// `ZSTD_compress2` resets a context for the empty frame) and long
/// distance matching: the need of [`needed_space`] is libzstd's
/// `ZSTD_estimateCCtxSize_usingCCtxParams_internal`.
#[test]
fn fresh_workspace_matches_libzstd() {
    let sizes = [0, 1, 100, 1024, 1025, 16 << 10, (16 << 10) + 1, 128 << 10];
    let sizes = sizes
        .into_iter()
        .chain([(128 << 10) + 1, 256 << 10, 1 << 20]);
    let sizes: Vec<usize> = sizes.chain([(256 << 10) + 1]).collect();
    for lvl in (-5..=22).filter(|&l| l != 0) {
        for &len in &sizes {
            let frames = [periodic(len)];
            let (opts, params) = level(lvl);
            run(&opts, &params, &frames);
            let (opts, params) = with_ldm(level(lvl));
            run(&opts, &params, &frames);
        }
    }
}

/// A frame needing a larger workspace grows it at once; one needing less
/// than a third keeps it for 128 more resets; then sizes in between keep
/// whatever is three times their need or less.
#[test]
fn mixed_sizes_follow_libzstd() {
    let (opts, params) = level(9);
    let lens = [1 << 20, 300, 64 << 10, 1 << 20, 10, 5000, 0, 200_000];
    let frames: Vec<_> = (0..400)
        .map(|i| small(lens[(i * 7 + i / 3) % lens.len()], i as u64))
        .collect();
    run(&opts, &params, &frames);
}
