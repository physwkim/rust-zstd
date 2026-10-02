# rust-zstd

Pure Rust implementation of the [Zstandard](https://facebook.github.io/zstd/) compression format ([RFC 8878](https://www.rfc-editor.org/rfc/rfc8878)). A full codec — one-shot, streaming, dictionaries, multithreaded compression — with no C dependencies, ported from and verified against libzstd 1.5.7.

## What is Zstandard?

Zstandard (zstd) is a lossless data compression algorithm developed by Yann Collet at Meta. It targets real-time compression scenarios, offering a wide range of compression/speed trade-offs while being backed by an extremely fast decoder.

### How it works

Zstandard combines three classical compression techniques in a layered pipeline:

```
Input bytes
  │
  ▼
┌─────────────────────────────────────────────────┐
│  1. LZ77 Match Finding                          │
│     Slide a window over the input and find      │
│     repeated byte sequences ("matches").        │
│     Each match is encoded as a back-reference:  │
│       (literal_length, offset, match_length)    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  2. Huffman Coding (Literals)                   │
│     Bytes that don't belong to any match are    │
│     called "literals". They are compressed with │
│     Huffman coding — frequent bytes get shorter │
│     binary codes, rare bytes get longer ones.   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  3. FSE — Finite State Entropy (Sequences)      │
│     The sequence of (literal_length, offset,    │
│     match_length) triples is encoded with FSE,  │
│     a tANS-family entropy coder that approaches │
│     the Shannon limit while decoding at table-  │
│     lookup speed.                               │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
Zstandard frame (header + blocks + optional checksum,
each block ≤ 128 KB decompressed). Blocks can be:
  • Raw — stored uncompressed
  • RLE — one byte repeated
  • Compressed — the pipeline above
```

## Features

- **Pure Rust** — no C bindings, no build.rs, no `libc`. `unsafe` is confined to the performance-critical inner loops (entropy decoding, match finding); everything else is safe Rust.
- **Full codec** — one-shot, reusable contexts, streaming with `std::io` adapters, dictionaries, and multithreaded compression (ZSTDMT-equivalent jobs via rayon, enabled by the default `parallel` feature).
- **libzstd-matched encoder** — all of libzstd 1.5.7's default machinery is ported: the parameter rows for levels 1–22 (and the accelerated fast strategy for negative levels), all five match finders (fast, double-fast, lazy row hashing, binary tree, optimal parser), long-distance matching, and both block splitters. On the test corpus the emitted frames are byte-identical to libzstd 1.5.7's at every level, for single-shot and multithreaded job layouts alike.
- **Spec-compliant decoder** — accepts and rejects frames per RFC 8878; verified against libzstd across levels, window sizes, truncation and corruption sweeps.
- **Checksums, skippable frames, concatenated frames** — as in libzstd.

## Installation

```toml
[dependencies]
rust-zstd = "0.3"
```

Parallel compression is enabled by default. To disable it (single-threaded, no rayon dependency):

```toml
[dependencies]
rust-zstd = { version = "0.3", default-features = false }
```

## API

### One-shot

```rust
use rust_zstd::{compress, decompress};

// Levels follow libzstd: 1..=22 (higher = smaller output, slower),
// 0 = the default level (3), negative = the fast strategy accelerated by -level.
let frame = compress(b"Hello, World!", 3);
let original = decompress(&frame).expect("valid zstd frame");
assert_eq!(original, b"Hello, World!");
```

`compress_with` takes `CompressOptions` for everything beyond the level — a content checksum, multithreaded job size, long-distance matching and block-splitter knobs:

```rust
use rust_zstd::{compress_with, CompressOptions};

let opts = CompressOptions {
    level: 19,
    checksum: true,
    job_size: Some(0), // ZSTDMT with libzstd's automatic job size
    ..Default::default()
};
let frame = compress_with(data, &opts);
```

The frame for a given job size is identical with and without the `parallel` feature; threads change speed, never bytes.

### Reusable contexts

`Compressor` and `Decompressor` keep their tables and buffers across calls, like libzstd's `ZSTD_CCtx`/`ZSTD_DCtx` — this is the fast path when processing many small inputs:

```rust
use rust_zstd::{Compressor, CompressOptions, Decompressor};

let mut cctx = Compressor::new(CompressOptions { level: 3, ..Default::default() });
let mut dctx = Decompressor::new();
let mut out = Vec::new();

for input in inputs {
    cctx.compress(input, &mut out);
    let back = dctx.decompress(&out).unwrap();
    assert_eq!(back, *input);
}
```

`Decompressor::decompress_into(&frame, &mut buf)` reuses the output buffer too, avoiding the per-call allocation.

### Streaming

`Encoder` wraps any `std::io::Write`; `DecompressReader` wraps any `std::io::Read`:

```rust
use rust_zstd::{Encoder, DecompressReader, CompressOptions};
use std::io::{Read, Write};

let file = std::fs::File::create("data.zst")?;
let mut enc = Encoder::new(file, CompressOptions { level: 3, ..Default::default() });
enc.write_all(&data)?;
enc.finish()?; // ends the frame, returns the writer

let file = std::fs::File::open("data.zst")?;
let mut dec = DecompressReader::new(file);
let mut data = Vec::new();
dec.read_to_end(&mut data)?;
```

Under the adapters sit libzstd-style push state machines — `Compressor::compress_stream` (with `Continue`/`Flush`/`End` directives, `ZSTD_compressStream2` shape) and `Decompressor::decompress_stream` — for callers that manage their own buffers. Streaming decompression holds the window and at most 1 MiB past it, so arbitrarily large frames decode in bounded memory.

### Dictionaries

Dictionaries trained with the zstd CLI (`zstd --train`) and raw content prefixes both work, on both sides:

```rust
use rust_zstd::compress::{CompressDict, compress_with_dict};
use rust_zstd::decode::{DecodeDict, decompress_with_dict};

let cdict = CompressDict::new(&dict_bytes, 3).unwrap();
let frame = compress_with_dict(data, &cdict);

let ddict = DecodeDict::new(&dict_bytes).unwrap();
let original = decompress_with_dict(&frame, &ddict).unwrap();
```

`Decompressor::decompress_with_dict` reuses a context, and `compress_with_prefix` / `Compressor::compress_with_prefix` take a raw prefix (`ZSTD_c_prefix` equivalent). Streaming compression (`compress_stream`, `Encoder`) uses `CompressOptions::dict` and a prefix set with `Compressor::set_prefix`, as `ZSTD_compressStream2` uses `ZSTD_CCtx_refCDict` and `ZSTD_CCtx_refPrefix`; streaming decompression with a dictionary is not yet supported.

## Performance

Measured against libzstd 1.5.7 on x86-64 (Zen 4), 8 MiB real-data corpora (ELF binary, Rust source, English text and word lists, scientific doubles), one reused context per codec, interleaved timing rounds, median reported:

- **Compressed size** — byte-identical to libzstd 1.5.7 at every level 1–22 on the test corpus, so the ratio is libzstd's exactly.
- **Decompression** — 1.0–1.1x libzstd's speed on the AVX2 path across the corpus and levels. The portable (no-SIMD) path and non-BMI2 targets are a few percent slower.
- **Compression** — within a few percent of libzstd across levels 1–22 on real data, both single-threaded and multithreaded. Degenerate constant input (all zeros) is the known exception: both codecs exceed 4 GB/s there, but libzstd's RLE fast path is several times faster still.
- **Small inputs** — with a reused `Decompressor`, decoding 120 B–5 KB frames is at or above libzstd's reused-`DCtx` speed (dictionary frames under ~500 B remain slower).

## Architecture

```
src/
├── lib.rs            # Public API
├── compress/         # Encoder
│   ├── fast.rs, dfast.rs, lazy.rs, bt.rs, opt.rs   # Match finders, as in libzstd
│   ├── ldm.rs        # Long-distance matching
│   ├── presplit.rs, split.rs                       # Block splitters
│   ├── block.rs, seqstore.rs, params.rs            # Block emission, parameter rows
│   ├── stream.rs     # Compressor, Encoder, ZSTDMT job pipeline
│   └── dict.rs       # CompressDict
├── decode.rs         # Decoder core: frames, blocks, sequence execution
├── decode/           # Streaming ring, DecodeDict
├── huf.rs, fse.rs    # Entropy coders (encode side)
├── bitstream.rs      # Forward/backward bit writers
└── xxhash.rs         # XXH64 for content checksums
```

## Testing

```bash
cargo test                          # or: cargo nextest run
cargo test --no-default-features    # without the parallel feature
```

The suites round-trip every level across real-data corpora, compare frames byte-for-byte against libzstd 1.5.7 (via the `zstd` dev-dependency), decode libzstd- and rust-zstd-produced frames with both decoders plus ruzstd, and sweep truncated and corrupted inputs for matching accept/reject verdicts.

## Acknowledgments

The decoder began as a port of [ruzstd](https://github.com/KillingSpark/zstd-rs) 0.8.2 by Moritz Borcherding, used under the MIT license. The encoder was developed with reference to [zstd](https://github.com/facebook/zstd) 1.5.7 by Meta Platforms, Inc., whose BSD 3-Clause license is reproduced in `src/LICENSE-ZSTD`.

## License

BSD-3-Clause AND MIT — the encoder follows zstd (BSD 3-Clause), the decoder module retains ruzstd's MIT license. See `src/LICENSE-ZSTD` and the header of `src/decode.rs`.
