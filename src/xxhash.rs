//! XXH64, the hash of zstd's Content_Checksum: the low 32 bits of the
//! seed-0 XXH64 of a frame's decoded content.
//!
//! Port of the XXH64 streaming functions of libzstd 1.5.7's
//! `lib/common/xxhash.h` (`XXH64_reset` with seed 0, `XXH64_update`,
//! `XXH64_digest`).

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

/// XXH64_round.
#[inline(always)]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

/// XXH64_mergeRound.
fn merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val))
        .wrapping_mul(PRIME64_1)
        .wrapping_add(PRIME64_4)
}

#[inline(always)]
fn read64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

/// One 32-byte stripe into the four lanes.
#[inline(always)]
fn stripe(v: &mut [u64; 4], s: &[u8; 32]) {
    v[0] = round(v[0], read64(&s[0..]));
    v[1] = round(v[1], read64(&s[8..]));
    v[2] = round(v[2], read64(&s[16..]));
    v[3] = round(v[3], read64(&s[24..]));
}

/// XXH64_state_t.
#[derive(Clone)]
pub(crate) struct Xxh64 {
    v: [u64; 4],
    total_len: u64,
    /// The input past the last whole stripe, `total_len % 32` bytes.
    mem: [u8; 32],
    mem_size: usize,
}

impl Xxh64 {
    /// XXH64_reset with seed 0.
    pub(crate) fn new() -> Self {
        Self {
            v: [
                PRIME64_1.wrapping_add(PRIME64_2),
                PRIME64_2,
                0,
                0u64.wrapping_sub(PRIME64_1),
            ],
            total_len: 0,
            mem: [0; 32],
            mem_size: 0,
        }
    }

    /// XXH64_update.
    pub(crate) fn update(&mut self, mut input: &[u8]) {
        self.total_len += input.len() as u64;
        if self.mem_size + input.len() < 32 {
            self.mem[self.mem_size..self.mem_size + input.len()].copy_from_slice(input);
            self.mem_size += input.len();
            return;
        }
        if self.mem_size > 0 {
            let (head, rest) = input.split_at(32 - self.mem_size);
            self.mem[self.mem_size..].copy_from_slice(head);
            stripe(&mut self.v, &self.mem);
            input = rest;
            self.mem_size = 0;
        }
        let (stripes, tail) = input.as_chunks::<32>();
        let mut v = self.v;
        for s in stripes {
            stripe(&mut v, s);
        }
        self.v = v;
        self.mem[..tail.len()].copy_from_slice(tail);
        self.mem_size = tail.len();
    }

    /// XXH64_digest.
    pub(crate) fn digest(&self) -> u64 {
        let mut h = if self.total_len >= 32 {
            let [v1, v2, v3, v4] = self.v;
            let h = v1
                .rotate_left(1)
                .wrapping_add(v2.rotate_left(7))
                .wrapping_add(v3.rotate_left(12))
                .wrapping_add(v4.rotate_left(18));
            self.v.iter().fold(h, |h, &v| merge_round(h, v))
        } else {
            // v[2] is the seed.
            self.v[2].wrapping_add(PRIME64_5)
        };
        h = h.wrapping_add(self.total_len);

        // XXH64_finalize.
        let mut tail = &self.mem[..self.mem_size];
        while tail.len() >= 8 {
            h ^= round(0, read64(tail));
            h = h
                .rotate_left(27)
                .wrapping_mul(PRIME64_1)
                .wrapping_add(PRIME64_4);
            tail = &tail[8..];
        }
        if tail.len() >= 4 {
            let k = u32::from_le_bytes(tail[..4].try_into().unwrap());
            h ^= u64::from(k).wrapping_mul(PRIME64_1);
            h = h
                .rotate_left(23)
                .wrapping_mul(PRIME64_2)
                .wrapping_add(PRIME64_3);
            tail = &tail[4..];
        }
        for &b in tail {
            h ^= u64::from(b).wrapping_mul(PRIME64_5);
            h = h.rotate_left(11).wrapping_mul(PRIME64_1);
        }

        // XXH64_avalanche.
        h ^= h >> 33;
        h = h.wrapping_mul(PRIME64_2);
        h ^= h >> 29;
        h = h.wrapping_mul(PRIME64_3);
        h ^ (h >> 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes the reference vectors hash: an xorshift64 stream, the
    /// high half's low byte of each state.
    fn input(len: usize) -> Vec<u8> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 32) as u8
            })
            .collect()
    }

    fn xxh64(data: &[u8]) -> u64 {
        let mut h = Xxh64::new();
        h.update(data);
        h.digest()
    }

    /// Values of `XXH64(input(len), len, 0)` from libzstd 1.5.7's
    /// `lib/common/xxhash.h` (compiled with `XXH_INLINE_ALL`): every tail
    /// path of XXH64_finalize, with and without whole stripes.
    #[test]
    fn matches_libzstd_xxhash_h() {
        let want: [(usize, u64); 14] = [
            (0, 0xef46db3751d8e999),
            (1, 0xb50c2974c97c6663),
            (3, 0x3d134528ea614554),
            (4, 0xdbcdf4f000009046),
            (7, 0x96349dbd1be0919a),
            (8, 0xa6e910fc0360f935),
            (12, 0x67b21a2a369bc08d),
            (31, 0x3f14b54c0ef2773a),
            (32, 0xe48dae740176fd82),
            (33, 0x6c4ce80297800e46),
            (63, 0xcdcf42c6c54f70be),
            (64, 0xa1e3e55045147c9d),
            (100, 0x48e7eb1624f84f3a),
            (1000, 0x9e2a3d6dd5fa3ed5),
        ];
        let data = input(1000);
        for (len, hash) in want {
            assert_eq!(xxh64(&data[..len]), hash, "length {len}");
        }
        assert_eq!(xxh64(b"abc"), 0x44bc2cf5ad770999);
    }

    /// Any split of the input into updates gives the one-update digest.
    #[test]
    fn split_updates_match_one_update() {
        let data = input(1000);
        for len in [0, 5, 31, 32, 33, 64, 95, 300, 1000] {
            let whole = xxh64(&data[..len]);
            for step in [1, 3, 8, 31, 32, 33, 100] {
                let mut h = Xxh64::new();
                for piece in data[..len].chunks(step) {
                    h.update(piece);
                }
                h.update(&[]);
                assert_eq!(h.digest(), whole, "length {len}, pieces of {step}");
            }
        }
    }

    /// The Content_Checksum libzstd writes is the low half of the digest.
    #[test]
    fn low_half_is_libzstds_content_checksum() {
        let data = input(5000);
        let mut cctx = zstd::bulk::Compressor::new(1).unwrap();
        cctx.include_checksum(true).unwrap();
        for len in (0..=260).chain([1000, 4096, 5000]) {
            let frame = cctx.compress(&data[..len]).unwrap();
            let stored = u32::from_le_bytes(frame[frame.len() - 4..].try_into().unwrap());
            assert_eq!(xxh64(&data[..len]) as u32, stored, "length {len}");
        }
    }
}
