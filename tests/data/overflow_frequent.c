// Fixture generator for tests/overflow_correction.rs: the frames libzstd
// 1.5.7 writes with ZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY set to 1, one
// row per case of CASES (keep it in sync with the test's), on inputs
// built by gen() (the test's `input`).
//
//   Z=~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/zstd-sys-2.0.16+zstd.1.5.7/zstd/lib
//   D="-DZSTD_MULTITHREAD -DZSTD_DISABLE_ASM -DZSTD_WINDOW_OVERFLOW_CORRECT_FREQUENTLY=1"
//   gcc -O2 -pthread $D -I$Z -I$Z/common -o overflow_frequent tests/data/overflow_frequent.c $Z/common/*.c $Z/compress/*.c
//   ./overflow_frequent > tests/data/overflow_frequent.txt
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* level, long distance matching (0 auto, 1 enabled), job MiB (0: one
 * job), input MiB */
static const int CASES[][4] = {
    {-5, 0, 0, 20}, {-1, 0, 0, 20}, {1, 0, 0, 20}, {2, 0, 0, 20},
    {3, 0, 0, 20}, {4, 0, 0, 20}, {5, 0, 0, 20}, {6, 0, 0, 20},
    {7, 0, 0, 20}, {8, 0, 0, 20}, {9, 0, 0, 20}, {10, 0, 0, 20},
    {11, 0, 0, 20}, {12, 0, 0, 20}, {13, 0, 0, 20}, {14, 0, 0, 20},
    {15, 0, 0, 20}, {16, 0, 0, 20}, {17, 0, 0, 20}, {18, 0, 0, 20},
    {19, 0, 0, 20},
    {1, 0, 4, 20}, {3, 0, 4, 20}, {9, 0, 16, 20}, {16, 0, 16, 20},
    {19, 0, 18, 20},
    {20, 0, 0, 56}, {21, 0, 0, 104}, {22, 0, 0, 200},
    {1, 1, 0, 136}, {3, 1, 0, 136}, {9, 1, 0, 136}, {16, 1, 0, 136},
    {19, 1, 0, 136}, {3, 1, 8, 136},
};

static uint64_t lcg;

static uint64_t next(void) {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg >> 33;
}

static uint64_t below(uint64_t n) { return next() % n; }

/* Noise literals, zero runs and copies from 1..256, 1..64 Ki and 1..len
 * bytes back or at the last distance. */
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

int main(void) {
    size_t c;
    for (c = 0; c < sizeof CASES / sizeof CASES[0]; c++) {
        int level = CASES[c][0], ldm = CASES[c][1], job = CASES[c][2];
        size_t len = (size_t)CASES[c][3] << 20;
        unsigned char* src = gen(len);
        size_t cap = ZSTD_compressBound(len), r;
        unsigned char* dst = malloc(cap);
        char jobs[16] = "def";
        ZSTD_CCtx* cx = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cx, ZSTD_c_compressionLevel, level);
        if (ldm) ZSTD_CCtx_setParameter(cx, ZSTD_c_enableLongDistanceMatching, ZSTD_ps_enable);
        if (job) {
            ZSTD_CCtx_setParameter(cx, ZSTD_c_nbWorkers, 1);
            ZSTD_CCtx_setParameter(cx, ZSTD_c_jobSize, job << 20);
            snprintf(jobs, sizeof jobs, "%dM", job);
        }
        r = ZSTD_compress2(cx, dst, cap, src, len);
        if (ZSTD_isError(r)) {
            fprintf(stderr, "L%d: %s\n", level, ZSTD_getErrorName(r));
            return 1;
        }
        printf("L%d %s %s %dM %zu %016llx %016llx\n", level, ldm ? "ldm" : "auto", jobs,
               CASES[c][3], r, (unsigned long long)fnv64(dst, r),
               (unsigned long long)fnv64(src, len));
        fflush(stdout);
        ZSTD_freeCCtx(cx);
        free(dst);
        free(src);
    }
    return 0;
}
