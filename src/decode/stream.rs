//! Streaming decompression (ZSTD_decompressStream): the frame decoder of
//! `decompress`, fed input in pieces of any size and writing its output
//! into buffers of any size.

use super::*;
use std::io::{self, Read};

/// A zstd decompressor that takes the input and gives the content in pieces
/// of any size (ZSTD_decompressStream). It decodes with the same
/// `FrameDecoder` as `decompress`, so it gives the same verdict on every
/// input, keeping at most one unit of input (a block) and one window of
/// output: a frame decodes in memory bounded by its Window_Size, by its
/// Frame_Content_Size when that is smaller, and by what it has decoded
/// to so far.
///
/// Its `decompress` and `decompress_with_dict` take whole input, as the
/// functions of those names do, and keep its tables and buffers for the
/// next call (ZSTD_decompressDCtx, ZSTD_decompress_usingDDict).
pub struct Decompressor {
    dec: FrameDecoder,
    /// The start of a unit that came in pieces.
    unit: Vec<u8>,
    ring: Ring,
    /// A frame has ended: once its content is written out,
    /// `decompress_stream` returns 0.
    frame_ended: bool,
    /// The error that stopped decoding, which every later call returns.
    failed: Option<String>,
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor {
    pub fn new() -> Self {
        Self::with_options(&DecodeOptions::default())
    }

    /// A decompressor on the paths `opts` picks. `decompress_stream`
    /// decodes on the current thread whatever `opts.min_parallel_blocks`
    /// says.
    #[doc(hidden)]
    pub fn with_options(opts: &DecodeOptions) -> Self {
        Decompressor {
            dec: FrameDecoder::new(opts),
            unit: Vec::new(),
            ring: Ring::default(),
            frame_ended: false,
            failed: None,
        }
    }

    /// Decompress `src`, whole, as the function `decompress` does, with the
    /// tables and buffers this decompressor keeps from call to call
    /// (ZSTD_decompressDCtx). With the `parallel` feature, frames of four
    /// or more blocks decode on the current rayon pool if the pool `new`
    /// found had more than one thread.
    ///
    /// It resets the streaming state, as `reset` does, before decoding and
    /// again after, whatever the result: input and output that
    /// `decompress_stream` holds, and its error, are dropped, and its next
    /// call starts on a new frame.
    pub fn decompress(&mut self, src: &[u8]) -> Result<Vec<u8>, String> {
        self.decompress_whole(src, None)
    }

    /// `decompress` with dictionary `dict`, as the function
    /// `decompress_with_dict` decodes (ZSTD_decompress_usingDDict). Only
    /// this call uses `dict`: the next one starts from the dictionary it is
    /// given, or none.
    ///
    /// It resets the streaming state before decoding and again after, as
    /// `decompress` does.
    pub fn decompress_with_dict(
        &mut self,
        src: &[u8],
        dict: &DecodeDict,
    ) -> Result<Vec<u8>, String> {
        self.decompress_whole(src, Some(dict))
    }

    fn decompress_whole(
        &mut self,
        src: &[u8],
        dict: Option<&DecodeDict>,
    ) -> Result<Vec<u8>, String> {
        self.reset();
        let content = decompress_frames(&mut self.dec, src, dict);
        // An error leaves the frame decoder inside a frame.
        self.reset();
        content
    }

    /// Decode the input at `src[*src_pos..]` into `dst[*dst_pos..]`,
    /// advancing both positions past what it reads and writes. It takes
    /// frames, skippable ones included, one after another, as `decompress`
    /// does.
    ///
    /// Returns 0 once a frame has been decoded and its content written out;
    /// the next call starts on the next frame. Otherwise it stopped for
    /// more input or for room in `dst`, and returns a positive hint: how
    /// many input bytes it takes to finish the current unit, or 1 when only
    /// output remains to be written.
    ///
    /// After an error the decompressor is stopped: every later call returns
    /// the same error, until `reset`.
    ///
    /// # Panics
    /// If `*src_pos > src.len()` or `*dst_pos > dst.len()`.
    pub fn decompress_stream(
        &mut self,
        src: &[u8],
        src_pos: &mut usize,
        dst: &mut [u8],
        dst_pos: &mut usize,
    ) -> Result<usize, String> {
        if let Some(e) = &self.failed {
            return Err(e.clone());
        }
        self.run(src, src_pos, dst, dst_pos).inspect_err(|e| {
            self.failed = Some(e.clone());
        })
    }

    /// The verdict on the input ending after what `decompress_stream` has
    /// read: Ok when that ends between frames and all the content has been
    /// written out; while content is left to write out, an error saying so;
    /// otherwise the error `decompress` gives on the same input.
    pub fn finish(&self) -> Result<(), String> {
        if let Some(e) = &self.failed {
            return Err(e.clone());
        }
        if self.ring.pending() != 0 {
            return Err(format!(
                "{} bytes of content left to write out",
                self.ring.pending()
            ));
        }
        self.dec.end_of_input(&self.unit)
    }

    /// Drop the input and output in hand, and any error, for input that
    /// starts with a new frame.
    pub fn reset(&mut self) {
        self.dec.stage = Stage::FrameHeader;
        self.unit.clear();
        self.ring.flushed = self.ring.end;
        self.frame_ended = false;
        self.failed = None;
    }

    fn run(
        &mut self,
        src: &[u8],
        src_pos: &mut usize,
        dst: &mut [u8],
        dst_pos: &mut usize,
    ) -> Result<usize, String> {
        loop {
            // The ring is written out before the next unit: a block may
            // start a new segment over it.
            *dst_pos += self.ring.flush(&mut dst[*dst_pos..]);
            if self.ring.pending() != 0 {
                return Ok(if self.frame_ended { 1 } else { self.hint() });
            }
            if self.frame_ended {
                self.frame_ended = false;
                return Ok(0);
            }
            let (skipped, ended) = self.dec.skip(src.len() - *src_pos);
            *src_pos += skipped;
            if ended {
                self.frame_ended = true;
                continue;
            }
            if self.dec.skipping() {
                return Ok(self.hint());
            }

            let input = &src[*src_pos..];
            let unit = if self.unit.is_empty() && self.dec.unit_len(input) <= input.len() {
                let len = self.dec.unit_len(input);
                *src_pos += len;
                &input[..len]
            } else {
                // The unit's length may grow as its first bytes come in.
                loop {
                    let need = self.dec.unit_len(&self.unit) - self.unit.len();
                    if need == 0 {
                        break;
                    }
                    let input = &src[*src_pos..];
                    if input.is_empty() {
                        return Ok(need);
                    }
                    let take = need.min(input.len());
                    self.unit.extend_from_slice(&input[..take]);
                    *src_pos += take;
                }
                &self.unit[..]
            };
            let event = self.dec.process(unit, &mut self.ring, None)?;
            self.unit.clear();
            self.frame_ended = event == Event::FrameEnded;
        }
    }

    /// The input bytes it takes to finish the current unit, at least 1.
    fn hint(&self) -> usize {
        (self.dec.unit_len(&self.unit) - self.unit.len()).max(1)
    }
}

/// An `io::Read` of the content of the frames `inner` reads, decoded one
/// after another by a `Decompressor`. It fails with `InvalidData` and the
/// `Decompressor`'s error where `decompress` fails on the same input,
/// truncated input included.
pub struct DecompressReader<R> {
    inner: R,
    dec: Decompressor,
    /// Input from `inner`, `buf[pos..len]` yet to be decoded.
    buf: Box<[u8]>,
    pos: usize,
    len: usize,
    /// `inner` has ended.
    eof: bool,
}

impl<R: Read> DecompressReader<R> {
    pub fn new(inner: R) -> Self {
        DecompressReader {
            inner,
            dec: Decompressor::new(),
            // ZSTD_DStreamInSize: a block with its header.
            buf: vec![0; BLOCK_HEADER_LEN + MAX_BLOCK_SIZE].into_boxed_slice(),
            pos: 0,
            len: 0,
            eof: false,
        }
    }

    /// The reader of the frames. Input it has read and not decoded yet is
    /// dropped.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Read for DecompressReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let invalid = |e| io::Error::new(io::ErrorKind::InvalidData, e);
        loop {
            if self.pos == self.len && !self.eof {
                self.len = self.inner.read(&mut self.buf)?;
                self.pos = 0;
                self.eof = self.len == 0;
            }
            let mut written = 0;
            self.dec
                .decompress_stream(&self.buf[..self.len], &mut self.pos, out, &mut written)
                .map_err(invalid)?;
            if written != 0 {
                return Ok(written);
            }
            if self.eof {
                self.dec.finish().map_err(invalid)?;
                return Ok(0);
            }
        }
    }
}

/// Room a block's destination has past its start (the `Dst` contract).
const BLOCK_ROOM: usize = MAX_BLOCK_SIZE + WILDCOPY_OVERLENGTH;

/// The window of the frame being decoded, as a round buffer (ZSTD_DStream's
/// `outBuff`): blocks decode one after another into a segment that starts
/// at 0, until the next block might not fit; the next segment then starts
/// at 0 again, with the previous one as its `ExtHistory`.
///
/// The buffer grows with what the frame decodes to, doubling, up to
/// `full`: `reach + WILDCOPY_OVERLENGTH + BLOCK_ROOM` bytes, `reach` being
/// the frame's Window_Size or its smaller Frame_Content_Size. So the frame
/// header's claims allocate nothing, and a frame never takes more than
/// `full`, nor more than what it has decoded and a block's room.
///
/// A segment ends only in a buffer of `full` bytes or more, after more than
/// `reach + WILDCOPY_OVERLENGTH` bytes. A frame that decodes to `reach`
/// bytes at most has one segment, so its `reach` is its window when it has
/// two. A match at `avail` bytes into the current segment then copies
/// from at most `window` bytes back, so from no earlier than
/// `WILDCOPY_OVERLENGTH + 1` bytes past `avail` in the previous segment,
/// ahead of every byte the current segment's copies wrote over it,
/// overshoot included.
#[derive(Default)]
struct Ring {
    buf: Vec<u8>,
    /// The frame's Window_Size.
    window: usize,
    /// The size past which the buffer does not grow.
    full: usize,
    /// The end of the current segment, where the next block decodes to.
    end: usize,
    /// The end of the previous segment, 0 before the frame has one.
    ext_end: usize,
    /// How much of the current segment has been written out.
    flushed: usize,
}

impl Ring {
    /// Write out what it can of the decoded content into `dst`, returning
    /// how much.
    fn flush(&mut self, dst: &mut [u8]) -> usize {
        let n = self.pending().min(dst.len());
        dst[..n].copy_from_slice(&self.buf[self.flushed..self.flushed + n]);
        self.flushed += n;
        n
    }

    /// Decoded content not written out yet.
    fn pending(&self) -> usize {
        self.end - self.flushed
    }

    /// Grow the buffer so that the next block fits, or as near as `full`
    /// allows, keeping the segment. Only for a frame with one segment.
    fn grow(&mut self) -> Result<(), String> {
        let len = self
            .buf
            .len()
            .saturating_mul(2)
            .max(self.end + BLOCK_ROOM)
            .min(self.full);
        self.buf
            .try_reserve_exact(len - self.buf.len())
            .map_err(|e| format!("Cannot allocate a {} byte window buffer: {}", len, e))?;
        self.buf.resize(len, 0);
        Ok(())
    }
}

impl FrameOut for Ring {
    fn start(&mut self, window: usize, content_size: Option<u64>) {
        let reach = content_size
            .and_then(|n| usize::try_from(n).ok())
            .map_or(window, |n| n.min(window));
        self.window = window;
        self.full = reach.saturating_add(WILDCOPY_OVERLENGTH + BLOCK_ROOM);
        self.end = 0;
        self.ext_end = 0;
        self.flushed = 0;
    }

    fn block_dst(&mut self) -> Result<(Dst, ExtHistory), String> {
        debug_assert_eq!(self.pending(), 0, "the ring is written out");
        if self.end + BLOCK_ROOM > self.buf.len() && self.buf.len() < self.full {
            debug_assert_eq!(self.ext_end, 0, "a full buffer for a second segment");
            self.grow()?;
        }
        if self.end + BLOCK_ROOM > self.buf.len() {
            // With `self.buf.len() >= self.full`, `self.end` is past
            // `reach + WILDCOPY_OVERLENGTH`.
            self.ext_end = self.end;
            self.end = 0;
            self.flushed = 0;
        }
        let base = self.buf.as_mut_ptr();
        let ext = match self.ext_end {
            0 => ExtHistory::NONE,
            len => ExtHistory {
                // SAFETY: `ext_end` is within the buffer.
                end: unsafe { base.add(len) },
                len,
                dict: false,
            },
        };
        let dst = Dst {
            base,
            op: self.end,
            window: self.window,
        };
        Ok((dst, ext))
    }

    unsafe fn commit(&mut self, end: usize) -> &[u8] {
        let start = self.end;
        self.end = end;
        &self.buf[start..end]
    }
}
