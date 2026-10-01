//! Frame headers that claim more than their blocks hold, in window or in
//! content size: the decoder's largest single allocation stays far below
//! the claim, and the accept / reject outcome is libzstd 1.5.7's one-shot
//! one (ZSTD_decompressDCtx), serial and MT at both SIMD levels, and
//! streaming. One test, so that nothing else allocates while it measures.

mod common;

use rust_zstd::decode::{decompress_with_options, DecodeOptions};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The system allocator, recording the largest size it is asked for.
struct Largest;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System` unchanged.
unsafe impl GlobalAlloc for Largest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Largest = Largest;

/// Well above what a frame of a few small blocks needs (its output, a
/// compressed block's 128 KiB + 64 of slack, the decoder's tables).
const SMALL: usize = 1 << 20;

const MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Frame of Window_Descriptor `wd` and the 8-byte Frame_Content_Size
/// `fcs`, holding `blocks` (block type, size field, content).
fn frame(wd: u8, fcs: u64, blocks: &[(u32, u32, &[u8])]) -> Vec<u8> {
    let mut f = MAGIC.to_vec();
    f.extend_from_slice(&[3 << 6, wd]);
    f.extend_from_slice(&fcs.to_le_bytes());
    for (i, &(ty, size, content)) in blocks.iter().enumerate() {
        let h = u32::from(i + 1 == blocks.len()) | ty << 1 | size << 3;
        f.extend_from_slice(&h.to_le_bytes()[..3]);
        f.extend_from_slice(content);
    }
    f
}

/// The largest allocation of decoding `f` with each path, after checking
/// the outcome against libzstd's.
fn largest_allocation(name: &str, f: &[u8]) -> usize {
    let want = zstd::bulk::decompress(f, SMALL).ok();
    let mut largest = 0;
    for simd in [false, true] {
        for min_parallel_blocks in [usize::MAX, 1] {
            let opts = DecodeOptions {
                min_parallel_blocks,
                simd,
            };
            LARGEST.store(0, Ordering::Relaxed);
            let got = decompress_with_options(f, &opts).ok();
            largest = largest.max(LARGEST.load(Ordering::Relaxed));
            assert!(
                got == want,
                "{name} simd={simd} min_parallel_blocks={min_parallel_blocks}: \
                 {} where libzstd {}",
                if got.is_some() {
                    "accepted"
                } else {
                    "rejected"
                },
                if want.is_some() { "accepts" } else { "rejects" },
            );
        }
    }
    LARGEST.store(0, Ordering::Relaxed);
    common::assert_stream_parity(name, f);
    largest.max(LARGEST.load(Ordering::Relaxed))
}

#[test]
fn claimed_sizes_allocate_what_the_blocks_hold() {
    // A raw block; an RLE and a raw one; a compressed block of 5 raw
    // literals and an RLE one.
    let literals: &[u8] = &[5 << 3, b'h', b'e', b'l', b'l', b'o', 0];
    let blocks: [&[(u32, u32, &[u8])]; 3] = [
        &[(0, 16, &[b'r'; 16])],
        &[(1, 1000, b"x"), (0, 3, b"abc")],
        &[(2, literals.len() as u32, literals), (1, 7, b"y")],
    ];
    let held = [16u64, 1003, 12];
    // Window logs 20, 31 (2 GiB), 31 at mantissa 7 (3.75 GiB) and 32.
    for wd in [10 << 3, 21 << 3, 21 << 3 | 7, 22 << 3] {
        for (blocks, held) in blocks.iter().zip(held) {
            // The exact size, then ever larger claims.
            for fcs in [held, u64::from(u32::MAX), 1 << 32, 1 << 40, 1 << 62] {
                let name = format!(
                    "window descriptor {wd:#x}, {} blocks holding {held}, content size {fcs}",
                    blocks.len()
                );
                let largest = largest_allocation(&name, &frame(wd, fcs, blocks));
                assert!(largest < SMALL, "{name}: allocated {largest}");
            }
        }
    }
}
