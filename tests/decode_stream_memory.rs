//! Frames far larger than their window, with long-distance matches across
//! all of it, stream in memory bounded by the window, with or without
//! Frame_Content_Size: the decoder ends up holding about Window_Size bytes,
//! and while its buffer grows to that (from what the frame has decoded),
//! the old and the new buffer take at most twice as much. One test, so
//! that nothing else allocates while it measures.

mod common;

use common::{c_compress2, datasets, MIB};
use rust_zstd::Decompressor;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use zstd::zstd_safe::zstd_sys::ZSTD_cParameter::{
    ZSTD_c_compressionLevel, ZSTD_c_contentSizeFlag, ZSTD_c_enableLongDistanceMatching,
    ZSTD_c_windowLog,
};

/// The system allocator, counting the bytes allocated and their peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: forwards every call to `System` unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted as both blocks, as a move would hold them.
        grew(new_size);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// 24 MiB of `period`-byte runs of source code, each with every 4093rd
/// byte changed, so that long-distance matches `period` back cover most of
/// it.
fn repeats(period: usize) -> Vec<u8> {
    let src = datasets()
        .into_iter()
        .find(|d| d.name == "rust_src_8m")
        .unwrap()
        .data;
    let mut data = Vec::with_capacity(24 * MIB);
    for i in 0.. {
        if data.len() == 24 * MIB {
            break;
        }
        let n = period.min(24 * MIB - data.len());
        data.extend(
            src[..n]
                .iter()
                .enumerate()
                .map(|(j, &b)| if j % 4093 == 0 { b ^ i as u8 } else { b }),
        );
    }
    data
}

/// Stream `frame` with 64 KiB of input and of output room per call,
/// checking the content against `data` as it comes; returns the bytes
/// allocated at the end and at their peak.
fn stream_memory(frame: &[u8], data: &[u8]) -> (usize, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let mut d = Decompressor::new();
    let mut out = vec![0u8; 1 << 16];
    let (mut pos, mut done) = (0usize, 0usize);
    loop {
        let src = &frame[pos..frame.len().min(pos + (1 << 16))];
        let (mut read, mut written) = (0, 0);
        let hint = d
            .decompress_stream(src, &mut read, &mut out, &mut written)
            .unwrap();
        assert!(
            out[..written] == data[done..done + written],
            "content at {done}"
        );
        pos += read;
        done += written;
        if hint == 0 {
            break;
        }
    }
    assert_eq!((pos, done), (frame.len(), data.len()));
    d.finish().unwrap();
    let held = LIVE.load(Ordering::Relaxed) - base;
    drop((d, out));
    (held, PEAK.load(Ordering::Relaxed) - base)
}

#[test]
fn long_window_frames_stream_in_window_memory() {
    // (window log, period of the repeats, level)
    for (window_log, period, level) in [(23, 6 * MIB, 3), (21, 3 * MIB / 2, 12)] {
        let data = repeats(period);
        let window = 1usize << window_log;
        for content_size in [true, false] {
            let frame = c_compress2(
                &data,
                &[
                    (ZSTD_c_compressionLevel, level),
                    (ZSTD_c_windowLog, window_log),
                    (ZSTD_c_enableLongDistanceMatching, 1),
                    (ZSTD_c_contentSizeFlag, content_size as i32),
                ],
            );
            let at = format!("window log {window_log} level {level} content size {content_size}");
            assert!(
                frame.len() < data.len() / 4,
                "{at}: {} bytes, long matches missed",
                frame.len()
            );
            let (held, peak) = stream_memory(&frame, &data);
            eprintln!(
                "{at}: frame {} bytes, held {held}, peak {peak}",
                frame.len()
            );
            // The window and its margin of eight blocks, the decoder's
            // tables and buffers, the output room.
            assert!(
                held < window + 2 * MIB,
                "{at}: holds {held} bytes for a {window} byte window"
            );
            assert!(
                peak < 2 * window + 2 * MIB,
                "{at}: peak {peak} bytes for a {window} byte window"
            );
        }
    }
}
