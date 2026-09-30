//! Differential test of the decoder's FSE normalized-count reader
//! (`rust_zstd::decode::parse_fse_header`) against libzstd's
//! `FSE_readNCount`, on valid headers, every truncation of them, and
//! single-bit corruptions.

use zstd::zstd_safe::zstd_sys as sys;

extern "C" {
    // lib/common/fse.h; linked from zstd-sys's static libzstd.
    fn FSE_readNCount(
        normalized_counter: *mut i16,
        max_symbol_value: *mut u32,
        table_log: *mut u32,
        rbuffer: *const u8,
        rbuff_size: usize,
    ) -> usize;
}

/// libzstd's result: (table log, counts of symbols 0..=maxSV, bytes).
fn c_read(src: &[u8]) -> Option<(u8, Vec<i32>, usize)> {
    let mut norm = [0i16; 256];
    let mut max_sv = 255u32;
    let mut log = 0u32;
    let r = unsafe {
        FSE_readNCount(
            norm.as_mut_ptr(),
            &mut max_sv,
            &mut log,
            src.as_ptr(),
            src.len(),
        )
    };
    if unsafe { sys::ZSTD_isError(r) } != 0 {
        return None;
    }
    let counts = norm[..=max_sv as usize]
        .iter()
        .map(|&c| i32::from(c))
        .collect();
    Some((log as u8, counts, r))
}

fn check(src: &[u8]) {
    let ours = rust_zstd::decode::parse_fse_header(src, 15).ok();
    let theirs = c_read(src);
    assert_eq!(ours, theirs, "input {:02x?}", src);
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// A valid header for a random sparse histogram, or None when the
/// histogram normalizes to a single symbol.
fn random_header(rng: &mut Lcg) -> Option<Vec<u8>> {
    let max_symbol = (rng.next() % 256) as usize;
    let mut counts = vec![0u32; max_symbol + 1];
    // Mix dense and very sparse histograms so that zero runs of every
    // length (including the >= 12 repeat-code path) occur.
    let density = 1 + rng.next() % 16;
    for c in counts.iter_mut() {
        if rng.next() % 16 < density {
            *c = 1 + rng.next() % (1 << (rng.next() % 12));
        }
    }
    counts[max_symbol] = counts[max_symbol].max(1);
    let total: usize = counts.iter().map(|&c| c as usize).sum();
    let table_log = 5 + rng.next() % 8;
    let mut norm = vec![0i16; max_symbol + 1];
    let log = rust_zstd::fse::normalize_count(
        &mut norm,
        table_log,
        &counts,
        total,
        max_symbol,
        rng.next().is_multiple_of(2),
    )
    .ok()?;
    if log == 0 {
        return None;
    }
    let mut out = Vec::new();
    rust_zstd::fse::write_ncount(&mut out, &norm, max_symbol, log).ok()?;
    Some(out)
}

#[test]
fn ncount_matches_libzstd_on_valid_truncated_and_flipped_headers() {
    let mut rng = Lcg(0x5EED_F5E0_0001);
    let mut valid = 0;
    for _ in 0..3000 {
        let Some(header) = random_header(&mut rng) else {
            continue;
        };
        valid += 1;
        // Trailing bytes after the header must not change the result.
        let mut padded = header.clone();
        padded.extend((0..rng.next() % 12).map(|_| rng.next() as u8));
        check(&padded);
        assert!(rust_zstd::decode::parse_fse_header(&padded, 15).is_ok());
        for len in 0..header.len() {
            check(&header[..len]);
        }
        for bit in 0..header.len() * 8 {
            let mut bad = header.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            check(&bad);
        }
    }
    assert!(valid > 1500, "only {} valid headers generated", valid);
}

#[test]
fn ncount_matches_libzstd_on_random_bytes() {
    let mut rng = Lcg(0xC0FF_EE00_0002);
    for _ in 0..200_000 {
        let len = (rng.next() % 40) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        check(&bytes);
    }
}
