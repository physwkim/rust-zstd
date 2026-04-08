# rust-zstd

Pure Rust implementation of the [Zstandard](https://facebook.github.io/zstd/) compression format (RFC 8878). Compress and decompress with zero external C dependencies.

## What is Zstandard?

Zstandard (zstd) is a lossless data compression algorithm developed by Yann Collet at Meta. It targets real-time compression scenarios, offering a wide range of compression/speed trade-offs while being backed by an extremely fast decoder.

### How it works

Zstandard combines three classical compression techniques in a layered pipeline:

```
Input bytes
  │
  ▼
┌─────────────────────────────────────────────────┐
│  1. LZ77 Match Finding                         │
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
│     Huffman coding — frequent bytes get shorter  │
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
┌─────────────────────────────────────────────────┐
│  4. Frame / Block Structure                     │
│     Compressed data is organized into frames,   │
│     each containing one or more blocks (≤128KB  │
│     decompressed). Blocks can be:               │
│       • Raw — stored uncompressed               │
│       • RLE — single repeated byte              │
│       • Compressed — Huffman literals + FSE     │
│         sequences                               │
└─────────────────────────────────────────────────┘
```

The decoder reverses this pipeline: parse frame/block headers, decode FSE sequences, decode Huffman literals, then execute the match copy operations to reconstruct the original data.

## Features

- **Pure Rust** — no C bindings, no `unsafe`, no `libc`
- **Compress + Decompress** — full codec, not decode-only
- **Spec-compliant** — output is decodable by any standard zstd decoder (C `libzstd`, Python `zstandard`, etc.)
- **Compression levels 0–11** — from raw storage to deep lazy matching
- **Parallel compression** — optional rayon-based block encoding (enabled by default)
- **Competitive ratios** — within 0–5% of C zstd, better on some workloads
- **Fast decoder** — 1.05–68x faster than C zstd across tested datasets
- **~5400 lines** of Rust (vs ~30,000 lines in C zstd)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
zstd-rs = { git = "https://github.com/physwkim/rust-zstd.git" }
```

Parallel compression is enabled by default. To disable it:

```toml
[dependencies]
zstd-rs = { git = "https://github.com/physwkim/rust-zstd.git", default-features = false }
```

## API

### Compress

```rust
use zstd_rs::compress;

// Compress with a specific level (0-11)
let compressed = compress(b"Hello, World!", 3);

// Level guide:
//   0     — no compression (raw blocks, fastest)
//   1-2   — greedy matching (fast)
//   3-5   — lazy matching (balanced)
//   6-8   — lazy matching + deeper search
//   9-11  — lazy matching + deepest search (best ratio)
```

```rust
use zstd_rs::compress_to_vec;

// Convenience wrapper — compresses at level 1
let compressed = compress_to_vec(b"Hello, World!");
```

### Decompress

```rust
use zstd_rs::decompress;

let original = decompress(&compressed).expect("valid zstd frame");
```

`decompress` supports:
- Single and concatenated frames
- Skippable frames (silently skipped)
- All standard block types (Raw, RLE, Compressed)
- Huffman and FSE entropy coding
- Repeat offsets and all sequence modes

### Roundtrip example

```rust
use zstd_rs::{compress, decompress};

fn main() {
    let data = b"The quick brown fox jumps over the lazy dog. \
                  The quick brown fox jumps over the lazy dog.";

    let compressed = compress(data, 3);
    println!(
        "compressed {} bytes -> {} bytes ({:.1}x)",
        data.len(),
        compressed.len(),
        data.len() as f64 / compressed.len() as f64
    );

    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(&decompressed, data);
    println!("roundtrip OK");
}
```

## Performance

Benchmarked against C zstd 1.5.6 on Apple M4, 128 KB test data per dataset:

### Compression ratio (vs C zstd)

| Dataset | Level 1 | Level 3 | Level 7 | Level 11 |
|---------|---------|---------|---------|----------|
| zeros   | 0.75x   | 0.76x   | 0.77x   | 0.77x    |
| text    | 1.04x   | 1.05x   | 1.05x   | 1.05x    |
| f64     | 1.03x   | 1.01x   | 1.01x   | 1.01x    |
| mixed   | 1.00x   | 1.00x   | 1.02x   | 1.02x    |

> Values < 1.0 = smaller output than C (better). All datasets within 5% of C zstd.

### Decompression speed

| Dataset | Rust (MB/s) | C (MB/s) | Speedup |
|---------|------------|----------|---------|
| zeros   | 33,653     | 492      | 68x     |
| text    | 1,328      | 484      | 2.7x    |
| f64     | 494        | 363      | 1.4x    |
| mixed   | 385        | 368      | 1.05x   |

### Compression speed

| Dataset | Level 1 | Level 3 | Level 7 | Level 11 |
|---------|---------|---------|---------|----------|
| zeros   | 57%     | 66%     | 33%     | 19%      |
| text    | 74%     | 68%     | 44%     | 18%      |
| f64     | 34%     | 36%     | 67%     | 131%     |
| mixed   | 23%     | 21%     | 14%     | 20%      |

> Percentage of C zstd compression speed. Match finding is the main bottleneck at lower levels.

## Architecture

```
src/
├── lib.rs          # Public API: compress, compress_to_vec, decompress
├── compress.rs     # Encoder: match finding, Huffman, FSE, block encoding
├── decode.rs       # Decoder: frame/block parsing, entropy decoding, sequence execution
├── fse.rs          # FSE compression tables and sequence encoder
├── bitstream.rs    # Forward/backward bit writers
└── constants.rs    # Zstd format constants, predefined tables, code mappings
```

### Compression pipeline

1. **Match finding** — hash-based (greedy or lazy) across the entire input with cross-block window
2. **Block splitting** — sequences partitioned into ≤128 KB decompressed blocks
3. **Repeat offset resolution** — zstd's 3-offset history per RFC 8878 §3.1.2.5
4. **Literal encoding** — Huffman with canonical codes, Treeless mode for sequential blocks
5. **Sequence encoding** — FSE tables (predefined or custom) for literal lengths, match lengths, and offsets
6. **Block assembly** — compressed block if smaller than raw, otherwise raw/RLE fallback

## Testing

```bash
# Run all tests (144 roundtrip tests across 12 levels × 12 datasets)
cargo test

# Without parallel feature
cargo test --no-default-features
```

## Acknowledgments

The decoder (`decode.rs`) is ported from [ruzstd](https://github.com/KillingSpark/zstd-rs) 0.8.2 by Moritz Borcherding, used under the MIT license. The encoder and all other modules are original implementations based on the [Zstandard specification](https://www.rfc-editor.org/rfc/rfc8878) and C zstd's [educational decoder](https://github.com/facebook/zstd/tree/dev/doc/educational_decoder).

## License

BSD-3-Clause

The decoder module (`src/decode.rs`) retains its original MIT license from ruzstd. See the file header for the full license text.
