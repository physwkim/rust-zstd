// Fixture generator for tests/overflow_correction.rs's reused-context
// case: libzstd 1.5.7 with ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY set to 1
// compresses gen(len) (overflow_frequent.c's input) FRAMES times on one
// context and prints the match state's nbOverflowCorrections after each
// frame, one row per level and length. A last block under 7 bytes still
// gets its overflow check (ZSTD_compress_frameChunk).
//
//   Z=~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/zstd-sys-2.0.16+zstd.1.5.7/zstd/lib
//   D="-DZSTD_DISABLE_ASM -DZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY=1"
//   gcc -O2 $D -I$Z -I$Z/common -o overflow_frequent_reuse tests/data/overflow_frequent_reuse.c $Z/common/*.c $Z/compress/*.c
//   ./overflow_frequent_reuse > tests/data/overflow_frequent_reuse.txt
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
#include "compress/zstd_compress_internal.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define FRAMES 40

/* level, input length: 128 KiB plus a 3-byte block, and plus 7 */
static const int CASES[][2] = {
    {1, 131075}, {1, 131079}, {3, 131075}, {3, 131079},
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

int main(void) {
    size_t c;
    for (c = 0; c < sizeof CASES / sizeof CASES[0]; c++) {
        int level = CASES[c][0], f;
        size_t len = (size_t)CASES[c][1];
        unsigned char* src = gen(len);
        size_t cap = ZSTD_compressBound(len);
        unsigned char* dst = malloc(cap);
        ZSTD_CCtx* cx = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cx, ZSTD_c_compressionLevel, level);
        printf("L%d %zu:", level, len);
        for (f = 0; f < FRAMES; f++) {
            size_t r = ZSTD_compress2(cx, dst, cap, src, len);
            if (ZSTD_isError(r)) {
                fprintf(stderr, "L%d: %s\n", level, ZSTD_getErrorName(r));
                return 1;
            }
            printf(" %u", cx->blockState.matchState.window.nbOverflowCorrections);
        }
        printf("\n");
        ZSTD_freeCCtx(cx);
        free(dst);
        free(src);
    }
    return 0;
}
