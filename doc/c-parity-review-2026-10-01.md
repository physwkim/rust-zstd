# rust-zstd vs libzstd 1.5.7 parity review, 2026-10-01

Reference: libzstd 1.5.7 as vendored by zstd-sys 2.0.16
(`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/zstd-sys-2.0.16+zstd.1.5.7/zstd/lib`),
checked against RFC 8878 for reference-side bugs.

## Accepted divergences (user decisions, not findings)

- Decoder accept/reject follows one-shot `ZSTD_decompressDCtx` with an
  ample dst buffer (Huffman log 12, raw/RLE past 128 KiB, any window up to
  `ZSTD_WINDOWLOG_MAX`), not RFC-strict and not streaming `maxWindowSize`.
  The output `Vec` grows, so capacity-dependent one-shot limits are modeled
  as a large buffer.
- `CompressOptions::default()` is single-threaded `ZSTD_compress2`
  (one job); `job_size: Some(n)` is ZSTDMT at that job size, `Some(0)` its
  automatic size. Frames depend on job size, never on thread count.
- Levels `<= 0` follow libzstd (0 = 3, negative = fast acceleration).
- Match-finder indices continue across frames and jobs on a reused context
  (`ZSTD_resetCCtx_internal` continue policy); the hash salt advances on
  every reset.
- 32-bit limits follow `MEM_32bits()`.
- Not ported, no output effect: the tiny-block LDM skip.
- Huffman streams keep the strict end-mark and end-of-stream verdict of
  libzstd without its BMI2 fast path (R1-4).
- No legacy v0.1-v0.7 frame decoding (R1-7).
- libzstd bugs that change frames or verdicts are copied for parity (R1-6,
  R1-11, R1-17, R1-18, R1-22), except R1-16: an LDM hash rate log
  above the adjusted window log derives ZSTD_HASHLOG_MIN, not libzstd's
  wrapped 30.

## Open Findings

### R1-1: Content checksum is never verified

Severity: High

Class: unimplemented feature

Rust: `src/decode.rs:3804` — `// Skip the checksum if present; this decoder does not verify it.` then `*pos += 4;`. No XXH64 state is kept and the stored value is never compared. Probe P1: a level-3 frame with its checksum flag set and the last checksum byte XORed with 0x55 gives Rust `Ok(1300 bytes)`, while libzstd one-shot and streaming both give `Err(Restored data doesn't match checksum)`.

C reference: `decompress/zstd_decompress.c:1049-1058` — when `checksumFlag` is set and `!forceIgnoreChecksum` (the default), it computes `(U32)XXH64_digest(&dctx->xxhState)` and runs `RETURN_ERROR_IF(checkRead != checkCalc, checksum_wrong, "")`. The xxh state is reset at `:719-720`.

Impact: A frame that asks for verification is accepted with corrupted content or a corrupted checksum, so any corruption that still decodes cleanly passes unnoticed. The decoder never gives a `checksum_wrong` result.

### R1-2: A frame-header error after the first frame is ignored when earlier frames produced output, so trailing garbage and a truncated next frame return Ok

Severity: High

Class: reference-independent defect

Rust: `src/decode.rs:178-181` — on any `parse_frame_header` error, `if !output.is_empty() { break; }` returns `Ok(output)`. Probe P4 on a valid 5-byte raw frame followed by:
- 10 garbage bytes: Rust Ok.
- 1 byte: Rust Ok.
- A bare zstd magic (a truncated second frame): Rust Ok.
- A 5-byte truncated skippable header: Rust Ok.

libzstd one-shot rejects all four with `Src size is incorrect`. When the first frame is empty (P4e, `FCS=0` plus garbage), Rust errors, so the verdict depends on whether earlier output is non-empty. The `break` also swallows every other `parse_frame_header` error on a later frame, including the R1-5 reserved bit once that check is added.

C reference: `decompress/zstd_decompress.c:1087-1166` — `ZSTD_decompressMultiFrame` loops while `srcSize >= ZSTD_startingInputLength`. A `prefix_unknown` error after a finished frame becomes `srcSize_wrong` (`:1145-1155`), any other error is returned (`:1156`), and leftover bytes fail with `RETURN_ERROR_IF(srcSize, srcSize_wrong, "input not entirely consumed")` (`:1166`).

Impact: If input is cut within the header of a second or later frame, the caller gets Ok with that frame's data silently missing. Garbage appended after valid frames is accepted, and the same trailing bytes are accepted or rejected depending on whether the earlier output was empty.

### R1-3: A non-zero Dictionary_ID is accepted although no dictionary is loaded

Severity: Medium

Class: interop-contract gap

Rust: `src/decode.rs:2571-2578` — `// We don't support dictionaries, but we still need to skip these bytes`. The dictID bytes are skipped and never compared. Probe P3: a raw-block frame with `FHD=0x21` and `dictID=7` gives Rust `Ok(5 bytes)`, while libzstd one-shot and streaming give `Err(Dictionary mismatch)`. With `dictID=0` in a 1-byte field, both accept.

C reference: `decompress/zstd_decompress.c:717-718` — `RETURN_ERROR_IF(dctx->fParams.dictID && (dctx->dictID != dctx->fParams.dictID), dictionary_wrong, "")`. With no dictionary loaded, `dctx->dictID` is 0, so any non-zero frame dictID is rejected.

Impact: A frame compressed with a dictionary is decoded without it. When it only uses raw blocks or its own tables and no dictionary offsets, Rust returns Ok instead of `dictionary_wrong`. By reading (not probed): a first sequence that uses a repcode set by the dictionary would run with the default repcodes 1/4/8 and silently produce wrong bytes whenever that offset lands inside the output. The caller gets no signal that a dictionary was needed.

### R1-5: Must-be-zero reserved bits are not checked in the Frame_Header_Descriptor or in Symbol_Compression_Modes

Severity: Low

Class: reference-faithful gap

Rust: two sites.
- `src/decode.rs:2402-2445` — `FrameDescriptor` has no accessor for bit 3 (0x08), and `parse_frame_header` (`:2528`) never tests it.
- `src/decode.rs:2266-2274` — `CompressionModes` reads only bits 7-2. The modes byte is stored at `:2310/:2329/:2341`, and bits 1-0 are never tested.

Probes:
- P2: `FHD=0x28` gives Rust `Ok(5 bytes)` and libzstd `Err(Unsupported frame parameter)`. `FHD=0x30` (unused bit 4) is accepted by both.
- P6: a level-1 frame with `nbSeq=4`, modes byte `|= 1/2/3`, gives Rust `Ok(200)` and libzstd `Err(Data corruption detected)` for all three.

C reference: `decompress/zstd_decompress.c:511-512` — `RETURN_ERROR_IF((fhdByte & 0x08) != 0, frameParameter_unsupported, "reserved bits, must be zero")`. `decompress/zstd_decompress_block.c:730` — `RETURN_ERROR_IF(*ip & 3, corruption_detected, "")`.

Impact: Frames that libzstd rejects as `frameParameter_unsupported` or `corruption_detected` decode to Ok in Rust. The data is the same as when the bits are clear, but the accept set is wider than the reference.

### R1-8: [libzstd] Skippable Frame_Size values 0xFFFFFFF8..0xFFFFFFFF are rejected on 64-bit by a 32-bit wrap check (our port does not copy this)

Severity: Low

Class: libzstd bug

Rust: `src/decode.rs:169-175` — `pos.checked_add(SKIPPABLE_FRAME_HEADER_LEN).and_then(|p| p.checked_add(skip_len as usize)).filter(|&end| end <= data.len())` accepts every u32 `Frame_Size` whose bytes are present. This is from reading only; a probe would need at least 4 GiB of input.

C reference: `decompress/zstd_decompress.c:595-596` — `RETURN_ERROR_IF((U32)(sizeU32 + ZSTD_SKIPPABLEHEADERSIZE) < sizeU32, frameParameter_unsupported, "")` does the wrap test in U32 on every build, although the next line computes `skippableHeaderSize + sizeU32` in `size_t`, which cannot overflow on 64-bit. The same function is used by `ZSTD_decompressMultiFrame` (`:1125`) and `ZSTD_findFrameSizeInfo` (`:746`).

Impact: On 64-bit, libzstd rejects a complete skippable frame with User_Data of 4 GiB−8 or more; Rust skips it and continues. Under the one-shot-parity decision Rust should reject it too, while the Frame_Size field itself allows any 32-bit value.

Decided 2026-10-01: copy libzstd, reject Frame_Size >= 0xFFFFFFF8 on every build.

### R1-9: `load_prefix` and blocks under 7 bytes skip the window update, so `next_src` lags the indexed bytes and a reused MT `Compressor` emits thread-count-dependent frames

Severity: High

Class: reference-independent defect

Rust: `src/compress/matchstate.rs:746-747` — `MatchState::start_block` is the only place that runs `correct_overflow_if_needed` and `window.extend_to(positions.end)`, and it is reached only through `build_seq_store` (`src/compress/block.rs:243`). Two paths bypass it:
- `load_prefix` (`src/compress/block.rs:160`) indexes `origin..job.start` into the tables without `Window::extend_to`.
- Blocks of 6 bytes or less skip `build_seq_store` through the `attempts_compression` gate (`block.rs:273`, used at `:521/:778/:874`).

After a non-first job of 1-6 bytes, `next_src` is still at `origin` while the tables hold prefix indices up to `index(job.start)`. The next `MatchState::reset` → `Window::continue_at` (`matchstate.rs:100-104`) sets `low = index(next_src)`, below those stale entries, so they count as live candidates in the next frame. `ContextPool::with_context` (`src/compress/mod.rs:301`) hands contexts out LIFO, so which job inherits the stale context depends on timing. `too_close_to_max` (`matchstate.rs:116`) reads the same stale `next_src`.

C reference: `zstd_compress.c:4950` — `ZSTD_loadDictionaryContent` calls `ZSTD_window_update` for the prefix. `zstd_compress.c:4817` — `ZSTD_compressContinue_internal` runs it over every chunk, before the `MIN_CBLOCK_SIZE` raw shortcut. `zstd_compress.c:4627` — `ZSTD_compress_frameChunk` calls `ZSTD_overflowCorrectIfNeeded` for every block, before `ZSTD_compressBlock_internal`'s `srcSize < 7` exit. So `nextSrc` covers every indexed byte and `ZSTD_window_clear` puts all old entries below `lowLimit`.

Impact, probe-proven (scratchpad `probe/src/bin/reuse4.rs`, `reuse5.rs`; overflow probe against libzstd built with `ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY=1`):
- Reused `Compressor`, level 1, `job_size: Some(512 << 10)`, frame 1 of 524,289 B, frame 2 of 1,048,579 B: frame 2 is 108,806 B at 1 thread, 108,990 B at 2, and {108,806 ×5, 108,990 ×15, 109,049 ×10} at 8. A fresh `Compressor` and libzstd `NbWorkers(2)` give 109,049 B. A last job of 512K+6 B differs, 512K+7 B matches.
- Level 16 with debug assertions: frame 2 panics at `src/compress/bt.rs:133` (`assertion failed: match_index < curr`).
- 40 reused frames of 128 KiB + 3 B with `overflow_correct_frequently`: `Compressor::overflow_corrections()` differs from libzstd's `nbOverflowCorrections` after 6/40 frames at level 1 and 11/40 at level 3 (frame bytes identical); a 7-byte tail gives 0 differing rows.

This breaks the accepted contract "frames depend on job size, never on thread count".

### R1-12: The port handles libzstd's minMatch-7 loader bug two ways: the hash-chain loader fixes it, the binary-tree loader copies it

Severity: Low

Class: reference-faithful gap

Rust: `src/compress/lazy.rs:1775-1787` — `mls_of(&ms.cparams)` (6 for minMatch 7) in the `SearchMethod::HashChain` arm of `load_prefix`. `src/compress/lazy.rs:1813` — the `SearchMethod::BinaryTree` arm calls `super::bt::update_tree`, which hashes with `min_match` itself (`bt.rs:524-528`).

C reference: `zstd_compress.c:5010` / `:5028` — both loaders take `cParams.minMatch` unbounded (see R1-11).

Impact: At minMatch 7, greedy/lazy/lazy2 on the hash chain with a prefix (an MT job after the first) would emit different sequences from libzstd: we match into the prefix and libzstd does not. btlazy2 and btopt+ would stay byte-identical. One policy should apply to both loaders; the project's byte-identity rule points to copying C. Unreachable through the public API (see R1-11). Found by reading, not probed.

Decided 2026-10-01: the hash-chain loader hashes minMatch like libzstd, as the binary-tree loader does.

### R1-13: A prefix of HASH_READ_SIZE bytes or less sets `next_to_update` to the prefix end; libzstd leaves it at the prefix start and skips the tag-table memset

Severity: Low

Class: reference-faithful gap

Rust: `src/compress/fast.rs:474`, `src/compress/dfast.rs:433`, `src/compress/lazy.rs:1818` — `ms.next_to_update = end` whatever the prefix length. `src/compress/lazy.rs:1769` — the guard `if end >= start + HASH_READ_SIZE` includes exactly 8 bytes, so a row prefix of exactly 8 bytes still zeroes the tag table (`:1789`). `src/compress/block.rs:164` — only overflow correction is skipped for short prefixes.

C reference: `zstd_compress.c:4971` sets `ms->nextToUpdate = ip - base` (prefix start). Then `:4975` `if (srcSize <= HASH_READ_SIZE) return 0;` returns before the tag memset (`:5006`) and before `nextToUpdate = iend`. For a prefix under 8 bytes, `ZSTD_compress_insertDictionary` (`zstd_compress.c:5205`, `dictSize<8`) returns before `ZSTD_window_update`, so the prefix is not part of the window at all.

Impact: For an 8-byte prefix, libzstd's first lazy search inserts the prefix positions into the chain or row, and a later LDM `ZSTD_ldm_fillFastTables` fills them for fast/dfast. We never insert them, so matches into the prefix differ. For a prefix under 8 bytes, libzstd cannot reference it at all, while our window starts at the prefix. Not reachable today: `overlap_size` (`mod.rs:762`) is either 0 or at least `1 << (window_log - 7)`, and two jobs of at least `JOBSIZE_MIN` give window_log ≥ 19, i.e. at least 4 KiB (at least 2 KiB with LDM). Found by reading, not probed.

### R1-20: No content checksum (`ZSTD_c_checksumFlag`) on the compression side

Severity: Low

Class: unimplemented feature

Rust: `src/compress/mod.rs:801` — `let descriptor = ((single_segment as u8) << 5) | (fcs_code << 6);`, so `Content_Checksum_flag` (bit 2) is never set. `CompressOptions` (`mod.rs:62`) has no checksum option. No XXH64 runs over the frame input, nothing is written after the last block, and the MT path has no serial checksum.

C reference:
- `zstd_compress.c:838-841` — `ZSTD_c_checksumFlag` sets `fParams.checksumFlag`.
- `zstd_compress.c:4702/4708` — the flag goes into the frame header descriptor.
- `zstd_compress.c:4607-4608` — `XXH64_update` runs per chunk.
- `zstd_compress.c:5372` — `ZSTD_writeEpilogue` appends the 4-byte `XXH64_digest`.
- MT: `zstdmt_compress.c:614-615` (serial `XXH64_update`), `:716` (checksum cleared for jobs other than 0), `:1435` (`frameChecksumNeeded`) and `:1526` (the digest after the last job).

Impact: A caller asking for libzstd's `-C` / `checksumFlag=1` output cannot get it. Every Rust frame lacks the 4-byte trailer and flag bit, so decoders cannot detect corruption, and the frames are not byte-identical to libzstd with checksums on (the zstd CLI default). Evidence: reading.

Decided 2026-10-01: add a checksum option, default off as in ZSTD_compress2.

## libzstd bugs

Upstream reports. Where the port copies one, its row says so.

### R1-4: [libzstd] The 4-stream Huffman fast path skips the end-of-stream and end-mark checks, so the one-shot verdict depends on the host CPU (our port does not copy this)

Severity: Medium

Class: libzstd bug

Rust: `src/decode.rs:1573-1576` and `src/decode.rs:1818-1823` (4X2: `2016-2021`) — the fast path rejects a stream whose last byte is 0 (`"Huffman stream has no end mark"`) and checks `br.is_finished()` on every stream (`"Huffman stream not fully consumed"`), the same as the body path (`:1872`, `:2076`).

Probe P7/P8 on 4-stream literal sections, with three kinds of change:
1. An extra byte inserted before stream 1 or stream 3.
2. Stream 1's last byte set to 0.
3. The regenerated size lowered by 1 so stream 4 holds one extra symbol.

Verdicts:
- Stock zstd-sys on this BMI2 host (one-shot and streaming): Ok for every case, same output.
- libzstd with `CFLAGS=-DHUF_DISABLE_FAST_DECODE` (the body path, which is what a non-BMI2 CPU runs): `Data corruption detected` for every case.
- Rust: Err for every case, except P8 seed 20 (that is R1-6).

C reference: `decompress/huf_decompress.c:150-157` — `HUF_initFastDStream` treats `lastByte==0` as `bitsConsumed=0` instead of rejecting it. `:281-303` — `HUF_initRemainingDStream` sets `bit->start = args->ilowest`, so a stream's tail may read into the bytes before it. `:876-888` (4X2 `:1697-1712`) — `HUF_decodeStreamX1/X2` is not followed by `BIT_endOfDStream`, and `args.op[i] != segmentEnd` always holds. `:903-913` — with `DYNAMIC_BMI2`, a CPU without BMI2 goes straight to `fallbackFn`, whose body checks `BIT_endOfDStream` on all four streams (`:692-693`, `:1495`). The fast path also needs `dtLog == 11` and all streams at least 8 bytes.

Impact: "One-shot libzstd" is not a single accept/reject function here: the same frame is accepted on BMI2 CPUs and rejected on CPUs without BMI2, with Huffman log 12, or with a stream under 8 bytes. Rust follows the non-BMI2 verdict, so on BMI2 hosts it rejects frames that libzstd accepts. Which verdict to follow is a user decision.

Decided 2026-10-01: the port keeps the strict verdict.

### R1-6: [libzstd] X2's last-symbol step consumes both symbols' bits of a 2-symbol cell, so a stream carrying one extra symbol passes the end check that X1 fails (our port copied this)

Severity: Low

Class: libzstd bug

Rust: `src/decode.rs:1897-1914` — `huf_decode_last_symbol_x2` writes one symbol but runs `br.skip_bits(entry.nb_bits)` for a `length == 2` cell and clamps `bits_consumed` to 64. X1 or X2 is picked by `huf_select_x2(dst_size, source.len())` (`:871`). Probe P8 lowers the regenerated size of a 4-stream section by 1, which leaves stream 4 with one encoded symbol that is never decoded:
- Seed 20 (X2 selected, 8220→8219): libzstd body path `Ok(8219)`, Rust `Ok(8219)`.
- All 10 X1-selected rows: both reject.
- X2 seeds 3, 10 and 12, where the final two symbols do not fit one cell: both reject.

C reference: `decompress/huf_decompress.c:1275-1290` — `HUF_decodeLastSymbolX2` runs `BIT_skipBits(DStream, dt[val].nbBits)` for `length != 1`, and clamps to 64 with the comment "ugly hack; works only because it's the last symbol". `BIT_endOfDStream` then passes (`:1373`, `:1495`). X1's `HUF_decodeSymbolX1` consumes only the symbol's own bits, and the X1 body rejects the same shape (`:592`, `:692`). The choice between the two comes from the timing heuristic `HUF_selectDecoder` (`:1821-1843`, used at `:1930`).

Impact: Whether a Huffman stream with one undecoded trailing symbol is corrupt depends on a speed heuristic (the compressed/regenerated size ratio) and on the data, not on the stream itself. Rust copies this exactly, so there is no parity divergence. It is a libzstd accept-set defect that the port inherited.

Decided 2026-10-01: the port keeps copying libzstd.

### R1-10: [libzstd] `ZSTD_deriveSeqStoreChunk` keeps a long length that sits exactly at the chunk end

Severity: Low

Class: libzstd bug

Rust: `src/compress/seqstore.rs` — `Seq` stores the full u32 literal and match lengths, so there is no `longLengthPos` and no chunk re-basing. The port does not copy the bug (found by reading).

C reference: `zstd_compress.c:4013` — the test is `if (originalSeqStore->longLengthPos < startIdx || originalSeqStore->longLengthPos > endIdx)`, but `endIdx` is exclusive, so it should read `>= endIdx`. When `longLengthPos == endIdx`, the chunk keeps `longLengthType` with `longLengthPos = nbSeq` of the chunk. `ZSTD_seqToCodes` (`zstd_compress.c:2715-2718`) then writes `llCodeTable[nbSeq]` / `mlCodeTable[nbSeq]`, one entry past the chunk.

Impact: in libzstd itself this is harmless. The stray write lands inside the array, on the next chunk's first sequence, which is the long one, and it writes MaxLL/MaxML, that sequence's correct code. It is still out-of-contract indexing that a stricter port would reproduce as an out-of-bounds write. There is no effect on our frames.

### R1-11: [libzstd] With minMatch 7, the hash-chain and binary-tree prefix loaders hash 7 bytes but the searches hash 6, so a loaded prefix is never matched

Severity: Low

Class: libzstd bug

Rust: `src/compress/bt.rs:527` — `7 => update_tree_internal::<_, 7>(...)`: our btlazy2 and btopt+ prefix loader copies the bug. `src/compress/lazy.rs:1775-1787` (`HcSearch::<mls_of>` inside `load_prefix`) does not copy it: it hashes with `BOUNDED(4, minMatch, 6)`. That deviation is stated in the doc comment at `lazy.rs:1757-1759`.

C reference: `zstd_lazy.c:661`: `ZSTD_insertAndFindFirstIndex` passes `ms->cParams.minMatch`, which is `ZSTD_hashPtr` case 7. `zstd_opt.c:584`: `ZSTD_updateTree` does the same. Both are called from `ZSTD_loadDictionaryContent` (`zstd_compress.c:5010`, `:5028`). The block searches use `mls = BOUNDED(4, minMatch, 6)` (`zstd_lazy.c:1531`, `:1955`), so they hash 6 bytes. The row loader is consistent: `ZSTD_row_update` uses `MIN(minMatch, 6)` (`zstd_lazy.c:952`).

Impact: Proven with a probe against libzstd 1.5.7 (scratch `probe2`). The setup was a 100000-byte random prefix set with `ZSTD_CCtx_refPrefix`, the same bytes as input, `ZSTD_c_windowLog` 14 and hashLog/chainLog 16. With minMatch 5 or 6 the output is 83656 bytes; with minMatch 7 it is 100031 bytes (no match at all) for greedy, lazy and lazy2 on the hash chain, btlazy2 and btopt. The row finder is unaffected (34490 bytes at minMatch 5, 6 and 7). Every prefix or dictionary match is lost, which violates no RFC rule but is a pure ratio bug. Our port is unreachable today: there is no `min_match` option, and the greedy through btlazy2 rows of the level tables use minMatch 4 or 5 only.

Decided 2026-10-01: the port copies libzstd in both loaders (R1-12).

### R1-14: [libzstd] The fast and row-lazy finders form pointers outside the input object (C11 6.5.6p8 undefined behaviour); the attribute on them only silences UBSan

Severity: Low

Class: libzstd bug

Rust: `src/compress/fast.rs:111`, `:146`, `:244`, `src/compress/dfast.rs:72`, `src/compress/lazy.rs:1218` — the same limits are computed on `usize` indices with `saturating_sub`; `Src::ptr` uses `wrapping_add` (`common.rs:73`). Our port does not copy the bug.

C reference:
- `zstd_fast.c:254` / `:338`: `ip2 = ip0 + step`, and `:250` `nextStep = ip0 + kStepIncr`. `step` is `targetLength + 1`, up to 131073 at `ZSTD_minCLevel()`, so `ip2`/`ip3` point far past one-past-the-end of a block at most 128 KiB long before the `ip3 >= ilimit` test.
- `zstd_lazy.c:1527`: `ilimit = iend - 8 - ZSTD_ROW_HASH_CACHE_SIZE`. For an 8..15-byte block that starts at the beginning of the buffer (for example a streamed flush at the start of `inBuff`), this points before the object.
- All of these functions carry `ZSTD_ALLOW_POINTER_OVERFLOW_ATTR` (`common/compiler.h:326-337`), which is `no_sanitize("pointer-overflow")` and does not make the arithmetic defined.

Impact: No frame difference; the compiler may assume these pointers stay in bounds. Our negative-level probe (levels -131072, -131071, -100000, -65536, -1000, -257..-255, -130..-127, -64, -33..-31, -9, -8, with block tails of 0..100 bytes) was byte-identical to libzstd, so the UB has no observed effect on output in this build.

### R1-15: [libzstd] On 32-bit, ldmHashLog 29/30 wraps the LDM hash-table size to 0 bytes, so libzstd writes out of bounds and segfaults (port did not copy the bug; it panics)

Severity: High

Class: libzstd bug

Rust: `src/compress/ldm.rs:338-339` — `self.hash_table.resize(1 << params.hash_log, LdmEntry::default())`. On i686, explicit hash logs 28, 29 and 30 end in a `capacity overflow` panic from `raw_vec/mod.rs:28`. The port does not corrupt memory. At 28 it panics where libzstd returns `memory_allocation`.

C reference: `lib/zstd.h:1263,1267,1296` — under MEM_32bits, `ZSTD_LDM_HASHLOG_MAX = ZSTD_HASHLOG_MAX = 30`, so ZSTD_c_ldmHashLog 29 and 30 pass the bounds check. `lib/compress/zstd_ldm.c:171-175` computes `ldmHSize * sizeof(ldmEntry_t)`, which is 2^32 or 2^33 and wraps to 0 in a 32-bit size_t. The workspace estimate therefore reserves nothing for the table. `lib/compress/zstd_compress.c:2224-2226` then reserves and memsets a 0-byte `hashTable`, and ZSTD_ldm_insertEntry indexes up to 2^hashLog entries into it.

Impact: i686 libzstd 1.5.7 with LDM enabled, level 3 and a 64 KiB input:
- `ldmHashLog=29` segfaults (exit 139).
- `ldmHashLog=30` also segfaults.
- `ldmHashRateLog=24` on a 1 KiB input (derived hashLog 30) returns a 1034-byte frame after unchecked writes outside the workspace.
- `ldmHashLog=28` returns error −64 (memory_allocation).

The port panics in the three explicit cases; in the rate case it derives hash log 6 (see Accepted divergences). At hashLog 20 both produce the same 33175-byte frame. Proven by probe: a scratch i686 build of zstd-sys 2.0.16 against the port (not committed).

### R1-17: [libzstd] ZSTD_ldm_gear_reset never stores its hash, so LDM split points after a chunk start or a match skip come from a stale rolling state (port copied the bug)

Severity: Low

Class: libzstd bug

Rust: `src/compress/ldm.rs:463-464` — "Initialize the rolling hash state with the first minMatchLength bytes (ZSTD_ldm_gear_reset leaves the state as it is)". Nothing is fed before `ip += min_match`. At `ldm.rs:542-548` the overlapping-match skip sets `ip = anchor - hashed` without re-hashing `[anchor - minMatch, anchor)`.

C reference: `lib/compress/zstd_ldm.c:60-85` — the contract says it "feeds [data, data + minMatchLength) into the hash … effectively resets the hash state". The body computes a local `hash` from `state->rolling` but never writes `state->rolling = hash`. It is called at `:378` (after `gear_init` sets `rolling = ~0` at `:37`) and at `:501` (after a skip).

Impact: for the first `minMatchLength - 1` bytes (up to 63) after each 1 MiB chunk start and each skip, the stopMask bits depend on `~0` or on pre-skip bytes rather than on the preceding minMatch bytes. Split points, and so inserted entries and found LDM matches, differ from the documented design. Frames stay valid; only ratio is affected.

The bug itself was found by reading. That the port copied it is supported by a probe: 3 MiB input with LDM enabled gives identical frames at L3 (1579911 B) and L19 (1578082 B).

Decided 2026-10-01: the port keeps copying libzstd.

### R1-18: [libzstd] A missing `else` in the ZSTD_compressBlock_opt_generic backtrack discards the literals-only final entry, so trailing literals of the last stretch are parsed again (port copied the bug)

Severity: Low

Class: libzstd bug

Rust: `src/compress/opt.rs:1047-1057` — the `if last_stretch.litlen > 0 { … opt[store_end - 1] = last_stretch; }` block is followed unconditionally by `opt[store_end] = last_stretch; let mut store_start = store_end;`, with a comment saying libzstd 1.5.7 does the same. The `mlen == 0` branch at `opt.rs:1079-1085` (`ip = anchor + llen`) is unreachable.

C reference: `lib/compress/zstd_opt.c:1385-1394` — `if (lastStretch.litlen > 0) { …storeStart = storeEnd-1; opt[storeStart] = lastStretch; } { opt[storeEnd] = lastStretch; storeStart = storeEnd; }`. This is a bare block where `else` was meant. It overwrites the literals-only entry, and the `mlen==0` store branch at `:1420-1424` becomes dead.

Impact: when a series ends with trailing literals (`lastStretch.litlen > 0`), `ip` restarts at the end of the last match instead of after the literals. The next series re-runs match finding and pricing over those positions. The effect is CPU cost and a different parse at levels 16-22 (btopt/btultra/btultra2); frames stay valid.

The bug was found by reading. That the port copied it is supported by the L19 byte-identical probe in R1-17.

Decided 2026-10-01: the port keeps copying libzstd.

### R1-19: [libzstd] `iend - 8` / `iend - HASH_READ_SIZE` form a pointer before `istart` for inputs under 8 bytes (undefined behaviour); port did not copy it

Severity: Low

Class: libzstd bug

Rust: `src/compress/opt.rs:726` — `let ilimit = iend.saturating_sub(8);` — and `src/compress/ldm.rs:456` — `let ilimit = iend.saturating_sub(HASH_READ_SIZE);`. Both use index arithmetic, so neither can form an out-of-object position.

C reference:
- `lib/compress/zstd_opt.c:1089` — `const BYTE* const ilimit = iend - 8;`. A 7-byte block passes the `< MIN_CBLOCK_SIZE+ZSTD_blockHeaderSize+1+1` (7) gate in ZSTD_buildSeqStore, so this computes `istart - 1`.
- `lib/compress/zstd_ldm.c:362` — `ilimit = iend - HASH_READ_SIZE` is computed before the `srcSize < minMatchLength` return at `:373`. Any chunk under 8 bytes reaches it: a 7-byte block, or a short final chunk of an MT job.
- This is pointer arithmetic outside the array object (C11 6.5.6p8). `ZSTD_ALLOW_POINTER_OVERFLOW_ATTR` (`zstd_opt.c:1075`, `zstd_ldm.c:341`) only suppresses the UBSan report.

Impact: a 7-byte ZSTD_compress2 at levels 16-22, or any LDM-enabled compression of a sub-8-byte chunk, executes UB in libzstd. It is benign with current compilers, since the pointer is only compared and the loop does not run. The port's frames are unaffected. Found by reading, not probed.

### R1-21: [libzstd] `ZSTD_window_update` / `ZSTD_window_correctOverflow` form pointers outside the source object (C11 6.5.6p8)

Severity: Low

Class: libzstd bug

Rust: `src/compress/matchstate.rs:88` and `:102` — the window base is a `usize` position, `base: origin.wrapping_sub(WINDOW_START_INDEX)` and `self.base = origin.wrapping_sub(end)`. This is integer arithmetic, so the port does not copy the UB.

C reference:
- `zstd_compress_internal.h:1374` — `window->base = ip - distanceFromBase;`. On a fresh CCtx `distanceFromBase` is 2, so `base` points 2 bytes before the caller's buffer. With continued indices it can be up to about 3.5 GiB before it.
- `zstd_compress_internal.h:1205-1206` — `window->base += correction; window->dictBase += correction;` moves the same made-up pointers.
- Pointer arithmetic that leaves the array object is undefined behaviour under C11 6.5.6p8. libzstd acknowledges this by tagging both functions `ZSTD_ALLOW_POINTER_OVERFLOW_ATTR` (`zstd_compress_internal.h:1158`, `:1353`), which expands to `no_sanitize("pointer-overflow")` in `common/compiler.h:326-337`. That suppresses the sanitizer; it does not remove the UB.

Impact: It works only on a flat address space where the compiler does not exploit the UB. A source buffer near the bottom of the address space, combined with a large continued index, makes `ip - distanceFromBase` wrap. Frames and state do not differ from the port; this is a note on reference soundness. Evidence: reading.

### R1-22: [libzstd] `addEvents_generic` counts fewer events than it samples

Severity: Low

Class: libzstd bug

Rust: `src/compress/presplit.rs:71` — `self.nb_events = (limit / RATE) as u64;` after sampling positions `0, RATE, 2·RATE, … < limit`. The test `record_counts_floor_of_limit_over_rate` (`presplit.rs:225`) pins nb_events = 190 against an event total of 191 at rate 43. The port copies the bug deliberately, for byte parity.

C reference: `zstd_preSplit.c:66` — `fp->nbEvents += limit/samplingRate;`. The loop `for (n = 0; n < limit; n += samplingRate)` makes ceil(limit/samplingRate) increments, so `nbEvents` is one short whenever the rate does not divide `limit`. For an 8 KiB chunk (limit 8191) that is every level that samples: rates 43, 11 and 5 give 190/191, 744/745 and 1638/1639. `fpDistance` and `compareFingerprints` normalise the histograms by these `nbEvents`, and `mergeEvents` accumulates the shortfall.

Impact: The pre-splitter's distance and threshold are slightly biased, which can move or suppress a split point versus a correct count. Frames stay valid, since this is heuristic only. Fixing it on our side would break byte identity with libzstd, so the current behaviour is correct for parity. Evidence: reading, plus the existing Rust unit test.

Decided 2026-10-01: the port keeps copying libzstd.

## Review Log

2026-10-01, round R1 (main 56c6282). Five read-only panels: decode,
entropy encoding, fast/dfast/lazy/row, bt/opt/LDM, driver/params/MT. 22
findings. Beyond the corpus, probes found no frame difference from libzstd
(entropy: 43,000 random cases; finders: levels -131072..22 at edge sizes;
driver: 870 frames at block and job boundaries). The port's own defects
cluster in two places. The decoder's accept set is wider than one-shot
libzstd's where the frame is bad (checksum, trailing bytes, dictID, reserved
bits). The window owner is bypassed by the prefix loader and sub-7-byte
blocks. Thirteen findings are libzstd bugs: five the port does not share,
and eight where following libzstd is a decision (Decision lines). RFC 8878 was not
available offline, so the libzstd findings are argued from the C source.
