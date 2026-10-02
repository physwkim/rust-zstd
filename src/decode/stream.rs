//! Streaming decompression (ZSTD_decompressStream): the frame decoder of
//! `decompress`, fed input in pieces of any size and writing its output
//! into buffers of any size.

use super::*;
use std::io::{self, Read};

/// A zstd decompressor that takes the input and gives the content in pieces
/// of any size (ZSTD_decompressStream). It decodes with the same
/// `FrameDecoder` as `decompress`, so it gives the same verdict on every
/// input, keeping at most one unit of input (a block) and one window of
/// output, with a margin of eight blocks, at most 1 MiB, for a frame that
/// decodes past its window: a frame decodes in memory bounded by its
/// Window_Size and that margin, by its Frame_Content_Size when that is
/// smaller, and by what it has decoded to so far. The one exception is libzstd's: a frame whose Window_Size
/// is above the limit `set_window_log_max` sets, by default `(1 << 27) + 1`,
/// is refused unless one call gets it whole. With the `parallel` feature it
/// also keeps, once a call has decoded blocks of a frame on the rayon pool
/// and until the frame ends, the buffers they decoded into, up to four per
/// thread of the pool, and copies of the blocks it planned past the call's
/// room, up to two per thread, which pool tasks decode ahead for the next
/// call, with a copy of the tables they started from.
///
/// One made `with_dict`, or given a dictionary by `set_dict`, starts every
/// frame from it (ZSTD_DCtx_refDDict), as `decompress_with_dict` does: the
/// frames of `decompress_stream` and of `decompress`, one after another,
/// until `set_dict` changes it.
///
/// Its `decompress` and `decompress_with_dict` take whole input, as the
/// functions of those names do, and keep its tables and buffers for the
/// next call (ZSTD_decompressDCtx, ZSTD_decompress_usingDDict);
/// `decompress_into` and `decompress_into_with_dict` decode into a buffer
/// the caller keeps too.
pub struct Decompressor {
    dec: FrameDecoder,
    /// The Window_Size above which `decompress_stream` refuses a frame
    /// (`DecodeOptions::window_log_max`).
    window_max: u64,
    /// The dictionary frames start from, unless a call names another.
    dict: Option<DecodeDict>,
    /// The buffer `decompress` and `decompress_with_dict` decode into: it
    /// keeps its room from call to call while what they return is copied
    /// out of it (`take_output`).
    out: Vec<u8>,
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

    /// A decompressor on the paths `opts` picks, with the window limit it
    /// sets (`set_window_log_max`).
    ///
    /// # Panics
    /// Where `set_window_log_max` panics on `opts.window_log_max`.
    #[doc(hidden)]
    pub fn with_options(opts: &DecodeOptions) -> Self {
        Decompressor {
            dec: FrameDecoder::new(opts),
            window_max: window_max(opts.window_log_max),
            dict: None,
            out: Vec::new(),
            unit: Vec::new(),
            ring: Ring::default(),
            frame_ended: false,
            failed: None,
        }
    }

    /// A decompressor whose frames start from `dict`, as `set_dict` gives
    /// it.
    pub fn with_dict(dict: &DecodeDict) -> Self {
        let mut d = Self::new();
        d.set_dict(Some(dict));
        d
    }

    /// Start every frame from `dict` from now on, or from no dictionary
    /// (ZSTD_DCtx_refDDict): the frames `decompress_stream` and `decompress`
    /// decode, until the next `set_dict`. The dictionary is shared, not
    /// copied.
    ///
    /// A frame in progress started from the dictionary before, so it first
    /// resets the streaming state, as `reset` does.
    pub fn set_dict(&mut self, dict: Option<&DecodeDict>) {
        self.reset();
        self.dict = dict.cloned();
    }

    /// `ZSTD_DCtx_setParameter(ZSTD_d_windowLogMax)`: `decompress_stream`
    /// refuses a frame whose Window_Size is above `1 << log`, which bounds
    /// the memory a frame takes, unless one call has all of the frame and
    /// room for its Frame_Content_Size. `0` restores the default, the limit
    /// of a new ZSTD_DCtx: `(1 << 27) + 1`, one byte past `log` 27. One-shot
    /// decoding (`decompress` and the like) takes no window buffer and no
    /// limit, as ZSTD_decompressDCtx.
    ///
    /// The limit holds from the next frame header `decompress_stream`
    /// completes: a frame already started keeps the one it started under,
    /// where libzstd refuses the call (`stage_wrong`) until the frame ends.
    ///
    /// # Panics
    /// If `log` is neither 0 nor in `10..=31` (`10..=30` where `usize` is 32
    /// bits), where libzstd returns `parameter_outOfBound`.
    pub fn set_window_log_max(&mut self, log: u32) {
        self.window_max = window_max(log);
    }

    /// Decompress `src`, whole, as the function `decompress` does, or as
    /// `decompress_with_dict` does with this decompressor's dictionary if it
    /// has one, with the tables and buffers it keeps from call to call
    /// (ZSTD_decompressDCtx). With the `parallel` feature, frames of four
    /// or more blocks decode on the current rayon pool if the pool `new`
    /// found had more than one thread.
    ///
    /// It resets the streaming state, as `reset` does, before decoding and
    /// again after, whatever the result: input and output that
    /// `decompress_stream` holds, and its error, are dropped, and its next
    /// call starts on a new frame.
    pub fn decompress(&mut self, src: &[u8]) -> Result<Vec<u8>, String> {
        self.decompress_vec(src, None)
    }

    /// `decompress` into `dst`: it clears `dst`, then fills it with what
    /// `decompress` returns, or leaves it empty where `decompress` fails.
    /// The room `dst` has is kept, so a `dst` reused from call to call
    /// takes no allocation once it has held the largest content.
    pub fn decompress_into(&mut self, src: &[u8], dst: &mut Vec<u8>) -> Result<(), String> {
        Ok(self.decompress_whole(src, None, dst)?)
    }

    /// `decompress` with dictionary `dict` in place of the decompressor's
    /// own, as the function `decompress_with_dict` decodes
    /// (ZSTD_decompress_usingDDict). Only this call uses `dict`: the next
    /// one starts from the dictionary it is given, or the decompressor's.
    ///
    /// It resets the streaming state before decoding and again after, as
    /// `decompress` does.
    pub fn decompress_with_dict(
        &mut self,
        src: &[u8],
        dict: &DecodeDict,
    ) -> Result<Vec<u8>, String> {
        self.decompress_vec(src, Some(dict))
    }

    /// `decompress_with_dict` into `dst`, which it clears and fills as
    /// `decompress_into` does.
    pub fn decompress_into_with_dict(
        &mut self,
        src: &[u8],
        dict: &DecodeDict,
        dst: &mut Vec<u8>,
    ) -> Result<(), String> {
        Ok(self.decompress_whole(src, Some(dict), dst)?)
    }

    /// `decompress_whole` into the decompressor's buffer, returning the
    /// content as `take_output` gives it.
    fn decompress_vec(&mut self, src: &[u8], dict: Option<&DecodeDict>) -> Result<Vec<u8>, String> {
        let mut buf = std::mem::take(&mut self.out);
        let result = self.decompress_whole(src, dict, &mut buf);
        let content = result.map(|()| take_output(&mut buf));
        self.out = buf;
        Ok(content?)
    }

    /// Decode `src` into `dst`, each frame from `dict`, or without one from
    /// the decompressor's own dictionary.
    #[inline(always)]
    fn decompress_whole(
        &mut self,
        src: &[u8],
        dict: Option<&DecodeDict>,
        dst: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        self.reset();
        dst.clear();
        let dict = dict.or(self.dict.as_ref());
        let result = decompress_frames(&mut self.dec, src, dict, dst);
        // An error leaves the frame decoder inside a frame, and `dst` with
        // the content before it.
        self.reset();
        if result.is_err() {
            dst.clear();
        }
        result
    }

    /// Decode the input at `src[*src_pos..]` into `dst[*dst_pos..]`,
    /// advancing both positions past what it reads and writes. It takes
    /// frames, skippable ones included, one after another, as `decompress`
    /// does, each from the decompressor's dictionary if it has one.
    ///
    /// Returns 0 once a frame has been decoded and its content written out;
    /// the next call starts on the next frame. Otherwise it stopped for
    /// more input or for room in `dst`, and returns a positive hint: how
    /// many input bytes it takes to finish the current unit, or 1 when only
    /// output remains to be written.
    ///
    /// A frame whose Window_Size is above the decompressor's limit fails at
    /// its header, unless this call has all of it and room for its
    /// Frame_Content_Size (`set_window_log_max`).
    ///
    /// With the `parallel` feature, when this call's input holds
    /// `min_parallel_blocks` or more whole blocks of a frame (from `new`,
    /// four on a pool of more than one thread) and `dst` has room for the
    /// most they decode to, they decode on the current rayon pool if that
    /// many of them are compressed, of `min_parallel_bytes` or more in all
    /// (8 KiB from `new`). The pool decodes up to twice its threads' worth
    /// of blocks more, which the next call takes where its input starts
    /// with the same bytes. Every call reads, writes and returns what it
    /// would decoding the blocks one after another.
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
        self.run(src, src_pos, dst, dst_pos).map_err(|e| {
            let e = String::from(e);
            self.failed = Some(e.clone());
            e
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
        Ok(self.dec.end_of_input(&self.unit)?)
    }

    /// Drop the input and output in hand, and any error, for input that
    /// starts with a new frame. The dictionary stays.
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
    ) -> Result<usize, DecodeError> {
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

            // Whole blocks in the input decode in parallel, each written
            // out before the next as at the top of this loop, the last one
            // there.
            #[cfg(feature = "parallel")]
            if self.unit.is_empty() {
                let dict = self.dict.as_ref();
                let mut out = RingOut {
                    ring: &mut self.ring,
                    dict: dict.map_or(&[], DecodeDict::content),
                };
                let room = dst.len() - *dst_pos;
                let mut read = 0;
                let decoded = self.dec.decode_blocks_parallel(
                    &src[*src_pos..],
                    dict,
                    &mut out,
                    room,
                    &mut read,
                    |out| {
                        *dst_pos += out.ring.flush(&mut dst[*dst_pos..]);
                        out.ring.pending() == 0
                    },
                );
                *src_pos += read;
                if let Some(event) = decoded? {
                    self.frame_ended = event == Event::FrameEnded;
                    continue;
                }
            }

            let input = &src[*src_pos..];
            let whole = self.unit.is_empty() && self.dec.unit_len(input) <= input.len();
            let unit = if whole {
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
            let dict = self.dict.as_ref();
            let mut out = RingOut {
                ring: &mut self.ring,
                dict: dict.map_or(&[], DecodeDict::content),
            };
            // The content of a block whose header came first, as one that
            // straddles two calls' input does, stays in the frame's chain,
            // with the whole blocks after it.
            #[cfg(feature = "parallel")]
            {
                let mut read = 0;
                let decoded = self.dec.decode_held_block_parallel(
                    unit,
                    &src[*src_pos..],
                    dict,
                    &mut out,
                    &mut read,
                    |out| {
                        *dst_pos += out.ring.flush(&mut dst[*dst_pos..]);
                        out.ring.pending() == 0
                    },
                );
                *src_pos += read;
                if let Some(event) = decoded? {
                    self.unit.clear();
                    self.frame_ended = event == Event::FrameEnded;
                    continue;
                }
            }
            let event = self.dec.process(unit, &mut out, dict)?;
            self.unit.clear();
            if event == Event::FrameStarted {
                // A header that came whole starts the call's input: a call
                // returns once a frame ends.
                self.admit(if whole { input } else { &[] }, dst.len() - *dst_pos)?;
            }
            self.frame_ended = event == Event::FrameEnded;
        }
    }

    /// ZSTD_decompressStream's window limit, on the frame just started:
    /// refuse a Window_Size above `window_max`, unless the frame decodes in
    /// one pass, as libzstd decodes it with ZSTD_decompress_usingDDict
    /// then: its Frame_Content_Size fits in `room`, the output the call
    /// has, and `input`, the call's input from the frame's start, holds the
    /// whole frame (`holds_frame`). `input` is empty for a header that came
    /// in pieces, where libzstd's walk starts inside the header and fails.
    /// The verdict comes before the frame's first block, so before the ring
    /// takes memory for it.
    fn admit(&mut self, input: &[u8], room: usize) -> Result<(), DecodeError> {
        let (frame, _) = self.dec.frame_start();
        let window = frame.window as u64;
        if window <= self.window_max
            || frame.content_size.is_some_and(|fcs| fcs <= room as u64) && holds_frame(input)
        {
            return Ok(());
        }
        Err(format!(
            "Window size {} too large, the streaming limit is {}",
            window, self.window_max
        )
        .into())
    }

    /// The input bytes it takes to finish the current unit, at least 1.
    fn hint(&self) -> usize {
        (self.dec.unit_len(&self.unit) - self.unit.len()).max(1)
    }
}

/// `ZSTD_WINDOWLOG_ABSOLUTEMIN`, the least `ZSTD_d_windowLogMax`.
const ZSTD_WINDOWLOG_ABSOLUTEMIN: u32 = 10;
/// `ZSTD_WINDOWLOG_LIMIT_DEFAULT`.
const ZSTD_WINDOWLOG_LIMIT_DEFAULT: u32 = 27;
/// `ZSTD_MAXWINDOWSIZE_DEFAULT`, the window limit of a new ZSTD_DCtx and of
/// `window_log_max` 0.
const ZSTD_MAXWINDOWSIZE_DEFAULT: u64 = (1 << ZSTD_WINDOWLOG_LIMIT_DEFAULT) + 1;

/// The Window_Size limit of `ZSTD_d_windowLogMax` `log`
/// (`Decompressor::set_window_log_max`).
fn window_max(log: u32) -> u64 {
    match log {
        0 => ZSTD_MAXWINDOWSIZE_DEFAULT,
        ZSTD_WINDOWLOG_ABSOLUTEMIN..=ZSTD_WINDOWLOG_MAX => 1 << log,
        _ => panic!(
            "window_log_max {log} out of range: 0 or \
             {ZSTD_WINDOWLOG_ABSOLUTEMIN}..={ZSTD_WINDOWLOG_MAX}"
        ),
    }
}

/// Whether `input` holds the whole frame whose header it starts with, as
/// ZSTD_findFrameCompressedSize walks it: the header, the blocks up to the
/// last by their headers alone, and the Content_Checksum. A reserved block
/// type ends the walk; a Block_Size past the frame's maximum does not.
fn holds_frame(input: &[u8]) -> bool {
    let mut pos = frame_header_len(input);
    if pos > input.len() {
        return false;
    }
    let checksum = FrameDescriptor(input[4]).content_checksum_flag();
    loop {
        let Some(&[b0, b1, b2]) = input.get(pos..pos + BLOCK_HEADER_LEN) else {
            return false;
        };
        let header = u32::from_le_bytes([b0, b1, b2, 0]);
        let size = match (header >> 1) & 3 {
            // RLE: one byte of content.
            1 => 1,
            // Reserved: corruption_detected.
            3 => return false,
            _ => header as usize >> 3,
        };
        pos += BLOCK_HEADER_LEN + size;
        if header & 1 != 0 {
            break;
        }
    }
    pos + if checksum { CHECKSUM_LEN } else { 0 } <= input.len()
}

/// An `io::Read` of the content of the frames `inner` reads, decoded one
/// after another by a `Decompressor`, from a dictionary if made `with_dict`.
/// It fails with `InvalidData` and the `Decompressor`'s error where
/// `decompress` (or `decompress_with_dict`) fails on the same input,
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

    /// The reader of `inner`'s frames, each decoded from `dict`
    /// (`Decompressor::with_dict`).
    pub fn with_dict(inner: R, dict: &DecodeDict) -> Self {
        let mut r = Self::new(inner);
        r.dec.set_dict(Some(dict));
        r
    }

    /// The window limit of the frames it reads from now on, as
    /// `Decompressor::set_window_log_max` sets it: a frame above it fails
    /// with `InvalidData`, unless the reader got all of it in one read of
    /// `inner` and the `read` has room for its Frame_Content_Size.
    ///
    /// # Panics
    /// Where `Decompressor::set_window_log_max` panics.
    pub fn set_window_log_max(&mut self, log: u32) {
        self.dec.set_window_log_max(log);
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

/// Blocks of its Block_Maximum_Size, `min(window, MAX_BLOCK_SIZE)`, that a
/// frame decoding past its window holds in a segment of the round buffer
/// past Window_Size and a block's room, so at most 896 KiB. A block that
/// starts that far into a segment reaches nothing before it, and decodes
/// without the extDict checks and copies.
const RING_MARGIN_BLOCKS: usize = 7;

/// The window of the frame being decoded, as a round buffer (ZSTD_DStream's
/// `outBuff`): blocks decode one after another into a segment that starts
/// at 0, until the next block might not fit; the next segment then starts
/// at 0 again, with the previous one as its `ExtHistory`.
///
/// The buffer grows with what the frame decodes to, doubling, up to
/// `full`: `reach + WILDCOPY_OVERLENGTH + BLOCK_ROOM` bytes, `reach` being
/// the frame's Window_Size or its smaller Frame_Content_Size, and the
/// `RING_MARGIN_BLOCKS` margin unless Frame_Content_Size is at most
/// Window_Size. So the frame header's claims allocate nothing, and a frame
/// never takes more than `full`, at most Window_Size and 1 MiB and 64
/// bytes, nor more than what it has decoded and a block's room.
///
/// A segment ends only in a buffer of `full` bytes or more, after more than
/// `full - BLOCK_ROOM` bytes, at least `reach + WILDCOPY_OVERLENGTH`. A
/// frame that decodes to `reach` bytes at most has one segment, so its
/// `reach` is its window when it has two. A match at `avail` bytes into
/// the current segment then copies from at most `window` bytes back, so
/// from no earlier than `WILDCOPY_OVERLENGTH + 1` bytes past `avail` in the
/// previous segment, ahead of every byte the current segment's copies wrote
/// over it, overshoot included. A block that starts more than `window`
/// bytes into its segment has no `ExtHistory`: no match reaches past the
/// segment's start from there, an offset past Window_Size being refused
/// whatever the history (a dictionary's too, the frame having decoded more
/// than Window_Size bytes), and offset 0 with none.
///
/// The `ExtHistory` of the first segment is the content of the dictionary
/// the frame started from (`RingOut`), which a match may reach while the
/// frame has decoded at most Window_Size bytes. A second segment starts
/// only after more than `window + WILDCOPY_OVERLENGTH` bytes, past that
/// reach, so no later segment has the dictionary before it.
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
    fn grow(&mut self) -> Result<(), DecodeError> {
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

/// A `Decompressor`'s `Ring` as the `FrameOut` of its frames, with the
/// content of the dictionary they start from, empty without one.
struct RingOut<'a> {
    ring: &'a mut Ring,
    dict: &'a [u8],
}

impl FrameOut for RingOut<'_> {
    fn start(&mut self, window: usize, content_size: Option<u64>) {
        let ring = &mut *self.ring;
        let reach = content_size
            .and_then(|n| usize::try_from(n).ok())
            .map_or(window, |n| n.min(window));
        let margin = if content_size.is_some_and(|n| n <= window as u64) {
            0
        } else {
            RING_MARGIN_BLOCKS * window.min(MAX_BLOCK_SIZE)
        };
        ring.window = window;
        ring.full = reach.saturating_add(WILDCOPY_OVERLENGTH + BLOCK_ROOM + margin);
        ring.end = 0;
        ring.ext_end = 0;
        ring.flushed = 0;
    }

    fn block_dst(&mut self) -> Result<(Dst, ExtHistory), DecodeError> {
        let ring = &mut *self.ring;
        debug_assert_eq!(ring.pending(), 0, "the ring is written out");
        if ring.end + BLOCK_ROOM > ring.buf.len() && ring.buf.len() < ring.full {
            debug_assert_eq!(ring.ext_end, 0, "a full buffer for a second segment");
            ring.grow()?;
        }
        if ring.end + BLOCK_ROOM > ring.buf.len() {
            // With `ring.buf.len() >= ring.full`, `ring.end` is past
            // `ring.full - BLOCK_ROOM`.
            ring.ext_end = ring.end;
            ring.end = 0;
            ring.flushed = 0;
        }
        let base = ring.buf.as_mut_ptr();
        let ext = match ring.ext_end {
            _ if ring.end > ring.window => ExtHistory::NONE,
            0 => ExtHistory::dict(self.dict),
            len => ExtHistory {
                // SAFETY: `ext_end` is within the buffer.
                end: unsafe { base.add(len) },
                len,
                dict: false,
            },
        };
        let dst = Dst {
            base,
            op: ring.end,
            window: ring.window,
        };
        Ok((dst, ext))
    }

    unsafe fn commit(&mut self, end: usize) -> &[u8] {
        let ring = &mut *self.ring;
        let start = ring.end;
        ring.end = end;
        &ring.buf[start..end]
    }
}
