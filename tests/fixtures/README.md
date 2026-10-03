# Test datasets

`tests/common::datasets` decodes these `zstd -19` frames, so every machine
and CI gate the same bytes. They are left out of the crate package.

- `rust_src_8m.zst`: the `.rs` files of crates.io crates, sorted by their
  path below the registry source directory component by component and
  concatenated, the first 8 MiB: bytemuck 1.25.2, cc 1.4.0,
  crossbeam-deque 0.8.6, crossbeam-epoch 0.9.18, crossbeam-utils 0.8.21,
  either 1.15.0, fearless_simd 0.3.0, find-msvc-tools 0.1.9, jobserver
  0.1.35, libc 0.2.189, pkg-config 0.3.33, rayon 1.11.0 and the first
  28,716 bytes of rayon-core 1.13.0. MIT license, see `LICENSE-MIT`.
- `elf_8m.zst`: the first 8 MiB of the x86-64 Linux debug `zstd_ratio`
  test binary of this crate at d2b9c80 (rustc 1.98.0), its source paths
  remapped and the manifest directory literal overwritten with a path of
  the same length. It holds compiled code of this crate, the Rust standard
  library and the crates of `LICENSE-MIT`, and libzstd 1.5.7
  (`LICENSE-zstd`).
- `words_1m.zst`: 1 MiB of words of SCOWL's american-english list (Debian
  wamerican 2020.12.07-2, `len` = 104,334 non-empty lines), each followed
  by a space. From `s = 0x9E3779B97F4A7C15`, each step sets
  `s = s * 6364136223846793005 + 1442695040888963407` (mod 2^64) and takes
  word `(s >> 33) % len`. See `LICENSE-SCOWL`.
