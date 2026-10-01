// Fixture generator for tests/overflow_correction.rs's window low end
// case: libzstd 1.5.7 with ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY set to 1
// compresses gen(len) (overflow_frequent.c's input) FRAMES times on one
// context and prints, for every block ZSTD_buildSeqStore compresses, the
// match state's window as ZSTD_compress_frameChunk left it: after
// ZSTD_overflowCorrectIfNeeded and ZSTD_window_enforceMaxDist. A sequence
// producer that always fails reads it; with the fallback enabled the block
// compressor runs as without one (the frames equal overflow_frequent.txt's).
//
//   Z=~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/zstd-sys-2.0.16+zstd.1.5.7/zstd/lib
//   D="-DZSTD_DISABLE_ASM -DZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY=1"
//   gcc -O2 $D -I$Z -I$Z/common -o window_low_frequent tests/data/window_low_frequent.c $Z/common/*.c $Z/compress/*.c
//   ./window_low_frequent > tests/data/window_low_frequent.txt
//
// Output: a `case` row (level, input length, frames, then the applied
// windowLog, chainLog and strategy), then per block a row of its offset in
// the input and size, the index of its first byte, lowLimit and
// nbOverflowCorrections, and a `frame` row (size, FNV-1a 64) per frame.
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
#include "compress/zstd_compress_internal.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* level, input MiB, frames */
static const int CASES[][3] = {
    {1, 20, 1}, {3, 20, 2}, {13, 20, 1}, {16, 20, 1}, {19, 20, 1},
};

static uint64_t lcg;

static uint64_t next(void) {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg >> 33;
}

static uint64_t below(uint64_t n) { return next() % n; }

/* overflow_frequent.c's gen(). */
static unsigned char* gen(size_t len) {
    unsigned char* out = malloc(len + 512);
    size_t n = 0, last = 1;
    lcg = 1;
    while (n < len) {
        uint64_t k = below(8), m;
        if (k < 3 || n < 64) {
            for (m = 1 + below(32); m; m--) out[n++] = (unsigned char)next();
        } else if (k == 3) {
            for (m = 1 + below(64); m; m--) out[n++] = 0;
        } else {
            size_t dist, start, i;
            if (k == 7) {
                dist = last;
            } else {
                size_t max = k == 4 ? 256 : k == 5 ? 65536 : n;
                dist = 1 + below(max < n ? max : n);
            }
            last = dist;
            start = n - dist;
            m = k == 6 ? 16 + below(240) : 4 + below(60);
            for (i = 0; i < m; i++) out[n + i] = out[start + i];
            n += m;
        }
    }
    return out;
}

static uint64_t fnv64(const unsigned char* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (n--) h = (h ^ *p++) * 0x100000001b3ULL;
    return h;
}

static const unsigned char* input;

static size_t observe(void* state, ZSTD_Sequence* outSeqs, size_t outSeqsCapacity,
                      const void* src, size_t srcSize, const void* dict, size_t dictSize,
                      int compressionLevel, size_t windowSize) {
    const ZSTD_window_t* w = &((ZSTD_CCtx*)state)->blockState.matchState.window;
    const unsigned char* ip = src;
    (void)outSeqs; (void)outSeqsCapacity; (void)dict; (void)dictSize;
    (void)compressionLevel; (void)windowSize;
    if (w->dictLimit != w->lowLimit) {
        fprintf(stderr, "dictLimit %u != lowLimit %u\n", w->dictLimit, w->lowLimit);
        exit(1);
    }
    printf("%zu %zu %u %u %u\n", (size_t)(ip - input), srcSize, (unsigned)(ip - w->base),
           w->lowLimit, w->nbOverflowCorrections);
    return ZSTD_SEQUENCE_PRODUCER_ERROR;
}

int main(void) {
    size_t c;
    for (c = 0; c < sizeof CASES / sizeof CASES[0]; c++) {
        int level = CASES[c][0], frames = CASES[c][2], f;
        size_t len = (size_t)CASES[c][1] << 20;
        unsigned char* src = gen(len);
        size_t cap = ZSTD_compressBound(len);
        unsigned char* dst = malloc(cap);
        ZSTD_CCtx* cx = ZSTD_createCCtx();
        ZSTD_compressionParameters cp = ZSTD_getCParams(level, len, 0);
        input = src;
        ZSTD_CCtx_setParameter(cx, ZSTD_c_compressionLevel, level);
        ZSTD_CCtx_setParameter(cx, ZSTD_c_enableSeqProducerFallback, 1);
        ZSTD_registerSequenceProducer(cx, cx, observe);
        printf("case %d %zu %d %u %u %d\n", level, len, frames, cp.windowLog, cp.chainLog,
               (int)cp.strategy);
        for (f = 0; f < frames; f++) {
            size_t r = ZSTD_compress2(cx, dst, cap, src, len);
            if (ZSTD_isError(r)) {
                fprintf(stderr, "L%d: %s\n", level, ZSTD_getErrorName(r));
                return 1;
            }
            printf("frame %zu %016llx\n", r, (unsigned long long)fnv64(dst, r));
        }
        fflush(stdout);
        ZSTD_freeCCtx(cx);
        free(dst);
        free(src);
    }
    return 0;
}
