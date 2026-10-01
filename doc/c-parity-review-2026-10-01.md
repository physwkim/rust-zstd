# rust-zstd vs libzstd 1.5.7 parity review, 2026-10-01

Reference: libzstd 1.5.7 as vendored by zstd-sys 2.0.16
(`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/zstd-sys-2.0.16+zstd.1.5.7/zstd/lib`),
checked against RFC 8878 for reference-side bugs.

## Accepted divergences (user decisions, not findings)

- Decoder accept/reject follows RFC 8878, not libzstd (user decision
  2026-10-01, replacing one-shot `ZSTD_decompressDCtx` parity). Frames the
  RFC allows are accepted even where libzstd rejects them; frames it calls
  invalid are rejected even where libzstd decodes them. Offset ==
  Window_Size is accepted (§3.1.1, "up to a distance of Window_Size").
  The resource limit the RFC leaves to the decoder is window log <=
  `ZSTD_WINDOWLOG_MAX`.
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
- No `ZSTD_c_literalCompressionMode` (experimental in libzstd); literals
  follow its auto rule, so default frames are unaffected (R2-16).
- Empty input decodes to empty output, as one-shot libzstd does, although
  RFC 8878 says compressed data has at least one frame (R3-3).
- libzstd bugs are not copied (user decision 2026-10-01). Frames may
  therefore differ from libzstd's. The merge gate is a round trip through
  rust-zstd, libzstd and ruzstd plus a size no larger than stock libzstd's
  within a small tolerance; byte comparison with libzstd is a diagnostic.

## Open Findings

## libzstd bugs

Upstream reports. Where the port still shares one, its Decided or Port
line says what the port does; that line goes when the port fix lands.

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

### R1-6: [libzstd] X2's last-symbol step consumes both symbols' bits of a 2-symbol cell, so a stream carrying one extra symbol passes the end check that X1 fails

Severity: Low

Class: libzstd bug

C reference: `decompress/huf_decompress.c:1275-1290` — `HUF_decodeLastSymbolX2` runs `BIT_skipBits(DStream, dt[val].nbBits)` for `length != 1`, and clamps to 64 with the comment "ugly hack; works only because it's the last symbol". `BIT_endOfDStream` then passes (`:1373`, `:1495`). X1's `HUF_decodeSymbolX1` consumes only the symbol's own bits, and the X1 body rejects the same shape (`:592`, `:692`). The choice between the two comes from the timing heuristic `HUF_selectDecoder` (`:1821-1843`, used at `:1930`).

Impact: Whether a Huffman stream with one undecoded trailing symbol is corrupt depends on a speed heuristic (the compressed/regenerated size ratio) and on the data, not on the stream itself. Rust copies this exactly, so there is no parity divergence. It is a libzstd accept-set defect that the port inherited.

### R1-10: [libzstd] `ZSTD_deriveSeqStoreChunk` keeps a long length that sits exactly at the chunk end

Severity: Low

Class: libzstd bug

Rust: `src/compress/seqstore.rs` — `Seq` stores the full u32 literal and match lengths, so there is no `longLengthPos` and no chunk re-basing. The port does not copy the bug (found by reading).

C reference: `zstd_compress.c:4013` — the test is `if (originalSeqStore->longLengthPos < startIdx || originalSeqStore->longLengthPos > endIdx)`, but `endIdx` is exclusive, so it should read `>= endIdx`. When `longLengthPos == endIdx`, the chunk keeps `longLengthType` with `longLengthPos = nbSeq` of the chunk. `ZSTD_seqToCodes` (`zstd_compress.c:2715-2718`) then writes `llCodeTable[nbSeq]` / `mlCodeTable[nbSeq]`, one entry past the chunk.

Impact: in libzstd itself this is harmless. The stray write lands inside the array, on the next chunk's first sequence, which is the long one, and it writes MaxLL/MaxML, that sequence's correct code. It is still out-of-contract indexing that a stricter port would reproduce as an out-of-bounds write. There is no effect on our frames.

### R1-11: [libzstd] With minMatch 7, the hash-chain and binary-tree prefix loaders hash 7 bytes but the searches hash 6, so a loaded prefix is never matched

Severity: Low

Class: libzstd bug

C reference: `zstd_lazy.c:661`: `ZSTD_insertAndFindFirstIndex` passes `ms->cParams.minMatch`, which is `ZSTD_hashPtr` case 7. `zstd_opt.c:584`: `ZSTD_updateTree` does the same. Both are called from `ZSTD_loadDictionaryContent` (`zstd_compress.c:5010`, `:5028`). The block searches use `mls = BOUNDED(4, minMatch, 6)` (`zstd_lazy.c:1531`, `:1955`), so they hash 6 bytes. The row loader is consistent: `ZSTD_row_update` uses `MIN(minMatch, 6)` (`zstd_lazy.c:952`).

Impact: Proven with a probe against libzstd 1.5.7 (scratch `probe2`). The setup was a 100000-byte random prefix set with `ZSTD_CCtx_refPrefix`, the same bytes as input, `ZSTD_c_windowLog` 14 and hashLog/chainLog 16. With minMatch 5 or 6 the output is 83656 bytes; with minMatch 7 it is 100031 bytes (no match at all) for greedy, lazy and lazy2 on the hash chain, btlazy2 and btopt. The row finder is unaffected (34490 bytes at minMatch 5, 6 and 7). Every prefix or dictionary match is lost, which violates no RFC rule but is a pure ratio bug. Our port is unreachable today: there is no `min_match` option, and the greedy through btlazy2 rows of the level tables use minMatch 4 or 5 only.

### R1-14: [libzstd] The fast and row-lazy finders form pointers outside the input object (C11 6.5.6p8 undefined behaviour); the attribute on them only silences UBSan

Severity: Low

Class: libzstd bug

Rust: `src/compress/fast.rs:111`, `:146`, `:244`, `src/compress/dfast.rs:72`, `src/compress/lazy.rs:1218` — the same limits are computed on `usize` indices with `saturating_sub`; `Src::ptr` uses `wrapping_add` (`common.rs:73`). Our port does not copy the bug.

C reference:
- `zstd_fast.c:254` / `:338`: `ip2 = ip0 + step`, and `:250` `nextStep = ip0 + kStepIncr`. `step` is `targetLength + 1`, up to 131073 at `ZSTD_minCLevel()`, so `ip2`/`ip3` point far past one-past-the-end of a block at most 128 KiB long before the `ip3 >= ilimit` test.
- `zstd_lazy.c:1527`: `ilimit = iend - 8 - ZSTD_ROW_HASH_CACHE_SIZE`. For an 8..15-byte block that starts at the beginning of the buffer (for example a streamed flush at the start of `inBuff`), this points before the object.
- All of these functions carry `ZSTD_ALLOW_POINTER_OVERFLOW_ATTR` (`common/compiler.h:326-337`), which is `no_sanitize("pointer-overflow")` and does not make the arithmetic defined.

Impact: No frame difference; the compiler may assume these pointers stay in bounds. Our negative-level probe (levels -131072, -131071, -100000, -65536, -1000, -257..-255, -130..-127, -64, -33..-31, -9, -8, with block tails of 0..100 bytes) was byte-identical to libzstd, so the UB has no observed effect on output in this build.

### R1-15: [libzstd] On 32-bit, ldmHashLog 29/30 wraps the LDM hash-table size to 0 bytes, so libzstd writes out of bounds and segfaults (port did not copy the bug)

Severity: High

Class: libzstd bug

C reference: `lib/zstd.h:1263,1267,1296` — under MEM_32bits, `ZSTD_LDM_HASHLOG_MAX = ZSTD_HASHLOG_MAX = 30`, so ZSTD_c_ldmHashLog 29 and 30 pass the bounds check. `lib/compress/zstd_ldm.c:171-175` computes `ldmHSize * sizeof(ldmEntry_t)`, which is 2^32 or 2^33 and wraps to 0 in a 32-bit size_t. The workspace estimate therefore reserves nothing for the table. `lib/compress/zstd_compress.c:2224-2226` then reserves and memsets a 0-byte `hashTable`, and ZSTD_ldm_insertEntry indexes up to 2^hashLog entries into it.

Impact: i686 libzstd 1.5.7 with LDM enabled, level 3 and a 64 KiB input:
- `ldmHashLog=29` segfaults (exit 139).
- `ldmHashLog=30` also segfaults.
- `ldmHashRateLog=24` on a 1 KiB input (derived hashLog 30) returns a 1034-byte frame after unchecked writes outside the workspace.
- `ldmHashLog=28` returns error −64 (memory_allocation).

The port panics in the three explicit cases; in the rate case it derives hash log 6 (see Accepted divergences). At hashLog 20 both produce the same 33175-byte frame. Proven by probe: a scratch i686 build of zstd-sys 2.0.16 against the port (not committed).

### R1-17: [libzstd] ZSTD_ldm_gear_reset never stores its hash, so LDM split points after a chunk start or a match skip come from a stale rolling state

Severity: Low

Class: libzstd bug

C reference: `lib/compress/zstd_ldm.c:60-85` — the contract says it "feeds [data, data + minMatchLength) into the hash … effectively resets the hash state". The body computes a local `hash` from `state->rolling` but never writes `state->rolling = hash`. It is called at `:378` (after `gear_init` sets `rolling = ~0` at `:37`) and at `:501` (after a skip).

Impact: for the first `minMatchLength - 1` bytes (up to 63) after each 1 MiB chunk start and each skip, the stopMask bits depend on `~0` or on pre-skip bytes rather than on the preceding minMatch bytes. Split points, and so inserted entries and found LDM matches, differ from the documented design. Frames stay valid; only ratio is affected.

The bug itself was found by reading. That the port copied it is supported by a probe: 3 MiB input with LDM enabled gives identical frames at L3 (1579911 B) and L19 (1578082 B).

### R1-18: [libzstd] A missing `else` in the ZSTD_compressBlock_opt_generic backtrack discards the literals-only final entry, so trailing literals of the last stretch are parsed again

Severity: Low

Class: libzstd bug

C reference: `lib/compress/zstd_opt.c:1385-1394` — `if (lastStretch.litlen > 0) { …storeStart = storeEnd-1; opt[storeStart] = lastStretch; } { opt[storeEnd] = lastStretch; storeStart = storeEnd; }`. This is a bare block where `else` was meant. It overwrites the literals-only entry, and the `mlen==0` store branch at `:1420-1424` becomes dead.

Impact: when a series ends with trailing literals (`lastStretch.litlen > 0`), `ip` restarts at the end of the last match instead of after the literals. The next series re-runs match finding and pricing over those positions. The effect is CPU cost and a different parse at levels 16-22 (btopt/btultra/btultra2); frames stay valid.

The bug was found by reading. That the port copied it is supported by the L19 byte-identical probe in R1-17.

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

C reference: `zstd_preSplit.c:66` — `fp->nbEvents += limit/samplingRate;`. The loop `for (n = 0; n < limit; n += samplingRate)` makes ceil(limit/samplingRate) increments, so `nbEvents` is one short whenever the rate does not divide `limit`. For an 8 KiB chunk (limit 8191) that is every level that samples: rates 43, 11 and 5 give 190/191, 744/745 and 1638/1639. `fpDistance` and `compareFingerprints` normalise the histograms by these `nbEvents`, and `mergeEvents` accumulates the shortfall.

Impact: The pre-splitter's distance and threshold are slightly biased, which can move or suppress a split point versus a correct count. Frames stay valid, since this is heuristic only. Fixing it on our side would break byte identity with libzstd, so the current behaviour is correct for parity. Evidence: reading, plus the existing Rust unit test.

### R2-2: [libzstd] One-shot decoding never limits a block's decoded size to Block_Maximum_Size, but streaming does

Severity: Low

Class: libzstd bug

C reference: `decompress/zstd_decompress.c:1016-1026` — one-shot `ZSTD_decompressFrame` bounds raw and RLE blocks only by `oend-op`, and compressed blocks only by where the literals sit (bsm+32, or unbounded per R2-1). Streaming `ZSTD_decompressContinue` rejects `cBlockSize > blockSizeMax` (`:1315`) and, for every block type, `rSize > blockSizeMax` (`:1367`, "Decompressed Block Size Exceeds Maximum"). RFC 8878 §3.1.1.2.3 says "Block_Size is limited by Block_Maximum_Size". §3.1.1.2.4 says that maximum "is applicable to both the decompressed size and the compressed size of any block".

Impact: proven by probe (`p3`, `p1`).
- **1 KiB window:** an RLE block of 200000, a raw block of 2000 and a raw block of 131073 are each `Ok` in one-shot libzstd and in all Rust paths. `ZSTD_decompressStream` rejects them.
- **1 MiB window:** a compressed block decoding to 131078 bytes (bsm+6) is `Ok` in one-shot and in Rust. Streaming returns `Data corruption detected`.

libzstd gives one RFC-invalid frame two verdicts depending on the API, and a 4-byte RLE block expands to 2 MiB instead of the RFC's 128 KiB.

### R2-3: [libzstd] X2's last-symbol step accepts a Huffman stream that ran out one symbol early, and the made-up byte depends on how libzstd was built

Severity: Medium

Class: libzstd bug

C reference: `decompress/huf_decompress.c:1275-1290` — `HUF_decodeLastSymbolX2` skips nothing when `bitsConsumed == sizeof(bitContainer)*8`. `common/bitstream.h:346-351` — `BIT_lookBitsFast` masks the shift with `regMask`, so it looks up `bitContainer << 0`, which is 32 or 64 bits wide depending on the build. `:449-452` — `BIT_endOfDStream` then holds. The BMI2 fast path is different: it finishes the tail with `bit->start = args->ilowest` (`:281-303`, `:1697-1712`), so it reads the previous stream's bytes, and it runs no end check. RFC 8878 §4.2.2: a bitstream "not entirely and exactly consumed ... is considered faulty".

Impact: proven by probe on fuzz case `fuzz_ex_1.zst`, a 4-stream X2 section of 40686 literals whose stream 3 has 0 bits left at its last symbol (literal 30515):
- X1-forced libzstd returns `Data corruption detected`.
- libzstd's X2 body path and all four Rust paths return `Ok` with byte 0x62.
- Stock x86-64 libzstd (asm loop and C fast loop) returns `Ok` with byte 0x47.

On i686, 900k mutations compared the Rust build (whose output equals the x86-64 port's) with i686 libzstd. Totals per class:
- **88 cases:** libzstd returns Err, Rust returns Ok.
- **103 cases:** libzstd returns Ok, Rust returns Err.
- **68 cases:** both return Ok but the bytes differ.

The one sampled case per class has this shape: the same exhausted stream lands on a 2-symbol cell at one container width (accepted) and a 1-symbol cell at the other (overflow, rejected). The other cases were not classified.

So libzstd gives one frame three outcomes: BMI2 x86-64, non-BMI2 x86-64, and 32-bit. The port matches only the second, so on BMI2 hosts and on i686 it returns different bytes, or the opposite verdict, from the system libzstd. This is the missing-symbol counterpart of R1-6 (one extra symbol). R1-4's verdict split does not cover the case where both return Ok with different bytes.

### R2-4: [libzstd] Match offsets at or beyond Window_Size are accepted as long as they stay inside the frame

Severity: Low

Class: libzstd bug

C reference: `decompress/zstd_decompress_block.c:1052-1054` (also `:930-932`, `:979-981`) — one-shot checks `sequence.offset > oLitEnd - prefixStart`, then against `virtualStart`. Without a dictionary both are the frame start, and there is no window term. RFC 8878 §3.1.1.4 says "all offsets leading to previously decoded data must be smaller than Window_Size", and §3.1.1.3 only requires "previous decoded data, up to a distance of Window_Size".

Impact: proven by probe (`p2`). Setup: Window_Size 1 KiB (and 2 KiB), two raw blocks of 1000 bytes, then one sequence with offset 1024, 1025, 1900 or 2000. One-shot libzstd, streaming libzstd and all Rust paths return `Ok(2010)`; only offset 2001 (before the frame start) is rejected. A frame that relies on this decodes in libzstd and the port, but is invalid for a decoder that keeps only Window_Size bytes, which the RFC allows. `p10` confirms that offsets into a previous frame are rejected on every path.

### R2-5: [libzstd] ZSTD_decompressStream accepts a Compressed_Block with Block_Size 0 as an empty block; one-shot rejects it (the port follows one-shot)

Severity: Low

Class: libzstd bug

Rust: `src/decode.rs:3994` — `split_block` parses a literals header from an empty slice and errors ("Cannot read 2 bits, only 0 remaining") on all four paths. The port does not copy the streaming behaviour.

C reference: `decompress/zstd_decompress.c:1319-1336` — when `cBlockSize == 0`, `ZSTD_decompressContinue` takes the "empty block" path for any block type, compressed included, and moves on to the next header or the checksum. One-shot `ZSTD_decompressFrame` passes the block to `ZSTD_decodeLiteralsBlock`, which rejects `srcSize < MIN_CBLOCK_SIZE` (`decompress/zstd_decompress_block.c:139`). Under RFC 8878 a 0-byte compressed block is invalid: §3.1.1.3 says it consists of a Literals_Section (1-5 byte header, §3.1.1.3.1.1) and a Sequences_Section (Number_of_Sequences is 1-3 bytes, §3.1.1.3.2.1).

Impact: proven by probe (`p4`) on a frame whose only block is a last `Compressed` block of size 0. One-shot libzstd returns `Data corruption detected`, streaming libzstd returns `Ok(0)`, and Rust returns Err. Only the streaming API goes against the RFC.

### R2-6: [libzstd] 4-stream Huffman literals with Regenerated_Size 4 are rejected, although the RFC's (1,1,1,1) split is valid

Severity: Low

Class: libzstd bug

C reference: `decompress/zstd_decompress_block.c:187-190` — `litSize < MIN_LITERALS_FOR_4_STREAMS` (6, `common/zstd_internal.h:92`) returns `literals_headerWrong`. `decompress/huf_decompress.c:609` and `:1390` apply the same limit (`dstSize < 6`). RFC 8878 §3.1.1.3.1.6 says each stream decodes `(Regenerated_Size+3)/4` bytes, "except for the last stream, which may be up to 3 bytes smaller". So Regenerated_Size 4 splits as 1,1,1,1, which is valid. Size 5 is invalid, so the RFC's lower bound is "4, or 6 and above", not 6.

Impact: proven by probe (`p9`). Size_Format 01 with Regenerated_Size 4 and four 1-byte streams, each holding one 1-bit symbol, is rejected by libzstd (one-shot and streaming) and by Rust. Sizes 6, 7 and 9 (last stream 0 or 1 byte) are `Ok` everywhere. An RFC-valid frame of this shape from another encoder is rejected by both.

### R2-7: [libzstd] Huffman trees with a 12-bit maximum code length are accepted; RFC 8878 caps codes at 11 bits

Severity: Low

Class: libzstd bug

C reference: `common/huf.h:37` — `HUF_TABLELOG_MAX 12`. `common/entropy_common.c:280` and `:288` — `HUF_readStats_body` only rejects weights or a tableLog above 12. RFC 8878 §4.2.1: "This specification limits the maximum code length to 11 bits."

Impact: proven by probe (`p6`). Direct weights 12..1 with an implied last weight of 1 (sum 4096, Max_Number_of_Bits 12) decode to `Ok(7)` in one-shot libzstd, streaming libzstd and all Rust paths, the same as the 11-bit control. This row is only for the upstream list; the port side is already covered by a user decision.

### R2-10: [libzstd] `ZSTD_resetCCtx_internal` subtracts two NULL pointers on a CCtx's first use

Severity: Low

Class: libzstd bug

Rust: `src/compress/matchstate.rs` — `Window::new` sets up real base/next indices before `too_close_to_max` is ever read. The port did not copy the bug.

C reference: `$Z/compress/zstd_compress.c:2142` — `indexTooClose = ZSTD_indexTooCloseToMax(zc->blockState.matchState.window)` runs before `!zc->initialized` is checked at `:2145`. On a CCtx straight from `ZSTD_initCCtx` (`:102-112`, memset 0), `ZSTD_indexTooCloseToMax` (`:2081`) computes `w.nextSrc - w.base` with both pointers NULL. C11 6.5.6p9 defines pointer subtraction only for pointers into the same array object, so this is undefined behavior. The result is discarded only because `!zc->initialized` is ORed in afterwards.

Impact: no output effect with the usual compilers, but UBSan's pointer-subtract checks flag it, and an optimizer may assume the subtraction cannot happen. Found by reading.

### R2-11: [libzstd] the ZSTDMT round-buffer size wraps on 32-bit targets

Severity: Low

Class: libzstd bug

Rust: `src/compress/mod.rs` — the jobs are slices of the caller's input, with no round buffer, so nothing in the port corresponds and the port did not copy the bug.

C reference: `$Z/compress/zstdmt_compress.c:1326` — `sectionsSize = mtctx->targetSectionSize * nbWorkers` is computed in `size_t`. On 32-bit, `ZSTD_c_jobSize` may be up to 512 MiB, and with `nbWorkers` 8 the product is 2^32, which wraps to 0. `capacity = MAX(windowSize, sectionsSize) + slackSize` (`:1327`) then has room for fewer whole sections than there are workers.

Impact: the frame is unchanged; ZSTDMT has fewer jobs in flight than requested (a smaller buffer means it waits for earlier jobs to finish), so throughput drops. Unsigned wraparound is defined behavior, so this is wrong sizing, not UB. Found by reading; not probed on a 32-bit target.

### R2-12: [libzstd] lazy/row/opt match finders emit offset == Window_Size, which RFC 8878 §3.1.1.4 forbids

Severity: Low

Class: libzstd bug

Two finders exclude it: btlazy2/DUBT (`lazy.rs:381`, `window_low + 1`) and fast/dfast, whose low bound is taken at the block end. The port copied this behaviour: frames are byte-identical to libzstd.

C reference: these all allow `offset == maxDistance`:
- `zstd_compress_internal.h:1395-1399` and `:1412-1416`: `withinWindow = curr - maxDistance`.
- `zstd_lazy.c:683-687,708` (HC `matchIndex>=lowLimit`).
- `zstd_lazy.c:1159-1163,1234` (row).
- `zstd_lazy.c:1554-1555` (`maxRep`).
- `zstd_opt.c:619-620,657,724`.

DUBT uses a strict bound, `zstd_lazy.c:98,106` (`matchIndex > windowLow`). libzstd's own decoder contract is inclusive: `zstd_decompress_block.c:1389` has `assert(seq.offset <= windowSize)`. RFC 8878 §3.1.1.4 (rfc8878.txt:1204-1206) says "all offsets leading to previously decoded data must be smaller than Window_Size". §3.1.1 (line 592) says "up to a distance of Window_Size", which reads as inclusive and conflicts with §3.1.1.4.

Impact: Probe-proven (`scratchpad/probe/src/bin/winedge2.rs`). Input: W random bytes, the same W bytes repeated at distance W−1, W or W+1, then a 4 KiB tail. All frames have the single-segment flag clear, so Window_Size = W.
- Matched at distance exactly W, by both libzstd and ours: L5, L9, L12 (wlog 21/22), L16 (wlog 22) and L19 (wlog 23). Example: L9 gives 4,198,894 B instead of the 8,392,909 B it costs when there is no match.
- Never matched at W+1.
- L13 and L15 (btlazy2) match at W−1 but not at W.
- L1, L3 and L4 (fast/dfast) do not match even at W−1.

Both decoders accept these frames (our round trip passed). A decoder that enforces §3.1.1.4 literally would reject frames from levels 5–12 and 16–22 whenever data repeats exactly one window back. No such decoder exists locally (ruzstd keeps more than W bytes and accepts them).

### R2-14: [libzstd] Finders form pointers outside the input buffer (C11 6.5.6p8 undefined behaviour); the port does not copy it

Severity: Low

Class: libzstd bug

Rust: `src/compress/common.rs:39-40,73` — `Src::new`, `rebased` and `ptr` build `base` and `base + idx` with `wrapping_add`/`wrapping_sub`. The only non-wrapping `.add` calls in the finders index the in-bounds `bt` and row tables (`bt.rs:124-169,376-417` and `lazy.rs:643,695,723`, all masked). The port did not copy the undefined behaviour.

C reference:
- `zstd_compress_internal.h:1374` — `window->base = ip - distanceFromBase`, which points `distanceFromBase` bytes before the caller's buffer (at least 2 bytes, `ZSTD_WINDOW_START_INDEX`, more once indices continue).
- `zstd_fast.c:292,317,380` — `base + matchIdx` is computed before `matchIdx >= prefixStartIndex` is checked. With the cmov path, `matchIdx` = 0 from an empty table already forms an out-of-object pointer.
- Upstream knows: `compiler.h:322-333` adds `ZSTD_ALLOW_POINTER_OVERFLOW_ATTR` (`no_sanitize("pointer-overflow")`), used at `zstd_compress_internal.h:1158,1353` and in fast, dfast, lazy, opt and ldm.

Impact: Found by reading. In libzstd this is undefined behaviour that the build hides from UBSan rather than fixes; it matters only to a compiler that optimizes on pointer provenance. There is no defect in the port, and frames are unaffected.

### R2-15: [libzstd] BIT_initCStream computes `endPtr` past the buffer before it checks capacity

Severity: Low

Class: libzstd bug

Rust: `src/bitstream.rs:76` — `BitCStream::new(buf: &mut [u8])` works on a slice and asserts `len > 8`, and its callers (`src/fse.rs:319`, `src/fse.rs:387`) size the buffer with `capacity_for`, so no pointer is ever formed outside the buffer. The port did not copy the bug. Found by reading.

C reference: `common/bitstream.h:157-158` — `bitC->endPtr = bitC->startPtr + dstCapacity - sizeof(bitC->bitContainer);` runs before `if (dstCapacity <= sizeof(bitC->bitContainer)) return ERROR(dstSize_tooSmall);`. When dstCapacity < 8 and startPtr is near the start of an object, the pointer lands before the object, which is UB under C11 6.5.6p8. One way to reach it is a small `dstCapacity` flowing through `HUF_compressWeights` → `FSE_compress_usingCTable` (for example `ZSTD_compressBlock` with dstCapacity 4..7).

Impact: The function still returns the right error, so no frame or state changes. The problem is the UB itself: a sanitizer (`-fsanitize=pointer-overflow`) or an aggressive optimizer can act on it. The fix is to move the capacity check above the `endPtr` line.

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

2026-10-01, round R2 (main 870301e). Five read-only panels, each on code
it did not write: decode; compression driver; cross-cutting state and the
public API; all match finders; entropy, bitstreams and XXH64. RFC 8878 was
available this time. 16 findings. Probes against libzstd stayed
byte-identical (2292 driver frames, 2160 finder frames, a 330-config option
matrix, 1560 edge lengths, XXH64 on 2441 inputs, 1.6M decode mutations
without a path disagreement). The port's own gaps: one-shot libzstd's
unbounded decoded size for in-place raw literals (R2-1), the missing
`ZSTD_window_enforceMaxDist` state (R2-8), an option checked after the
header is written (R2-9), and tables that never shrink (R2-13). The rest
are libzstd bugs; six are RFC 8878 violations that the port shares
(`[libzstd+port]`): block size limits only in streaming, X2's exhausted
stream, offsets at or past Window_Size in both directions, 4-stream size 4,
and 12-bit Huffman codes.

2026-10-01, round R3 (main 3d94b0f). Four panels: decode verdicts against
RFC 8878; finder window and index rules; LDM and the MT pipeline; entropy
coding and the pre-splitter. Three Low findings, all from the decode
panel (R3-1 to R3-3, probe-confirmed on all four decode paths and against
libzstd). The other three panels found nothing: the window panel probed
reused versus fresh contexts, thread counts and overflow correction; the
LDM panel ran 1,078 LDM config sequence comparisons against libzstd; the
entropy panel read the code side by side with libzstd.
