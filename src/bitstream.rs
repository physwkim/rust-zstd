//! Bit-level stream writers for zstd encoding.
//!
//! `BitWriter` — forward bitstream.
//! `BitCStream` — `BIT_CStream_t`, the backward bitstream (FSE sequences,
//! FSE-compressed Huffman weights).

/// Forward bitstream writer (Huffman literal streams).
pub struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32,
    current: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(256),
            bit_pos: 0,
            current: 0,
        }
    }

    pub fn write_bits(&mut self, value: u64, nbits: u32) {
        let mut val = value;
        let mut bits = nbits;
        while bits > 0 {
            let space = 8 - self.bit_pos;
            let take = std::cmp::min(space, bits);
            let mask = (1u64 << take) - 1;
            self.current |= ((val & mask) as u8) << self.bit_pos;
            val >>= take;
            bits -= take;
            self.bit_pos += take;
            if self.bit_pos == 8 {
                self.buf.push(self.current);
                self.current = 0;
                self.bit_pos = 0;
            }
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buf.push(self.current);
        }
        self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len() * 8 + self.bit_pos as usize
    }
}

/// `BIT_CStream_t`: backward bitstream writer over a caller-provided
/// region. Bits accumulate LSB-first in a 64-bit container; `flush_bits`
/// stores the whole container as 8 little-endian bytes and advances by the
/// complete bytes only, so the region must be sized with `capacity_for`
/// for the bits the caller will add. The decoder reads the result from
/// its end (`BitReaderReversed`).
pub struct BitCStream<'a> {
    bit_container: u64,
    bit_pos: u32,
    buf: &'a mut [u8],
    ptr: usize,
    end_ptr: usize,
}

impl<'a> BitCStream<'a> {
    /// `BIT_initCStream`: `buf.len()` must exceed 8.
    pub fn new(buf: &'a mut [u8]) -> Self {
        assert!(buf.len() > 8, "BIT_initCStream: dstSize_tooSmall");
        let end_ptr = buf.len() - 8;
        Self {
            bit_container: 0,
            bit_pos: 0,
            buf,
            ptr: 0,
            end_ptr,
        }
    }

    /// Region length that lets `close` succeed after at most `bits` bits
    /// (the endmark included): `BIT_closeCStream` reports overflow once
    /// `ptr` reaches `endPtr = capacity - 8`.
    pub const fn capacity_for(bits: usize) -> usize {
        bits.div_ceil(8) + 9
    }

    /// `BIT_addBits`: add the low `nb_bits` (< 32) of `value`.
    #[inline]
    pub fn add_bits(&mut self, value: u64, nb_bits: u32) {
        debug_assert!(nb_bits < 32);
        debug_assert!(nb_bits + self.bit_pos < 64);
        self.bit_container |= (value & ((1u64 << nb_bits) - 1)) << self.bit_pos;
        self.bit_pos += nb_bits;
    }

    /// `BIT_addBitsFast`: `value` has no bit set above `nb_bits`.
    #[inline]
    pub fn add_bits_fast(&mut self, value: u64, nb_bits: u32) {
        debug_assert_eq!(value >> nb_bits, 0);
        debug_assert!(nb_bits + self.bit_pos < 64);
        self.bit_container |= value << self.bit_pos;
        self.bit_pos += nb_bits;
    }

    /// `BIT_flushBits` (the checked variant): store the container, advance
    /// by the complete bytes, and stop at `end_ptr` on overflow, which
    /// `close` then reports.
    #[inline]
    pub fn flush_bits(&mut self) {
        debug_assert!(self.bit_pos < 64);
        let nb_bytes = (self.bit_pos >> 3) as usize;
        debug_assert!(self.ptr <= self.end_ptr);
        // SAFETY: `ptr` starts at 0 and every advance below clamps it to
        // `end_ptr = buf.len() - 8`, so the 8-byte store ends inside `buf`.
        unsafe {
            self.buf
                .as_mut_ptr()
                .add(self.ptr)
                .cast::<u64>()
                .write_unaligned(self.bit_container.to_le());
        }
        self.ptr = (self.ptr + nb_bytes).min(self.end_ptr);
        self.bit_pos &= 7;
        self.bit_container >>= nb_bytes * 8;
    }

    /// `BIT_closeCStream`: add the endmark and return the stream size in
    /// bytes, or 0 when it did not fit in the region.
    pub fn close(mut self) -> usize {
        self.add_bits_fast(1, 1);
        self.flush_bits();
        if self.ptr >= self.end_ptr {
            return 0;
        }
        self.ptr + (self.bit_pos > 0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_writer_basic() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b1100, 4);
        w.write_bits(0b1, 1);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xE5]);
    }

    #[test]
    fn cstream_sentinel_only() {
        // the smallest region BIT_initCStream accepts holds the endmark
        let mut buf = [0u8; 9];
        let w = BitCStream::new(&mut buf);
        assert_eq!(w.close(), 1);
        assert_eq!(buf[0], 0x01);
    }

    #[test]
    fn cstream_c_layout() {
        let mut buf = [0u8; BitCStream::capacity_for(17)];
        let mut w = BitCStream::new(&mut buf);
        w.add_bits(0xFF, 8);
        w.flush_bits();
        w.add_bits(0x1AB, 8); // bit 8 of the value is masked off
        let size = w.close();
        // flush: [0xFF], then 0xAB + endmark -> container 0x1AB, bitPos 9
        // flush 1 byte: [0xAB], remaining 0x01
        assert_eq!(size, 3);
        assert_eq!(&buf[..3], &[0xFF, 0xAB, 0x01]);
    }

    #[test]
    fn cstream_overflow_reports_zero() {
        let mut buf = [0u8; 10];
        let mut w = BitCStream::new(&mut buf);
        for _ in 0..8 {
            w.add_bits(0x5555, 16);
            w.flush_bits();
        }
        assert_eq!(w.close(), 0);
    }
}
