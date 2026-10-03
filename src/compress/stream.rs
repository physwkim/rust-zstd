//! Streaming compression: `ZSTD_compressStream2` over the one-shot frame
//! driver.
//!
//! A streaming frame is one job on one context, started by the same
//! [`Context::reset`] and [`begin_job`] as a one-shot frame's job 0 and
//! cut into blocks by the same resumable [`compress_blocks`]. Its input
//! collects in a buffer that holds the window before the next block plus
//! the input not yet compressed; `Continue` compresses only the blocks no
//! later input can change, so a frame without `Flush` is the one-shot
//! frame of the same input and pledged size, however the input is cut.
//! `Flush` ends a chunk (`ZSTD_compressContinue`): no block crosses it.
//!
//! libzstd keeps `windowSize + blockSize` of input in a ring (`inBuff`)
//! and reaches the bytes before the wrap as an `extDict` segment. The
//! finders here address one contiguous slice, so the buffer instead keeps
//! up to twice that and, when full, moves its last window down to the
//! start ([`Context::rebase`] keeps every index): one extra copy of the
//! input, amortized, and the same matches as the one-shot frame.
//!
//! A frame with a dictionary starts from it as the one-shot frame does
//! ([`FrameDict`]), sized for the pledged size or an unknown one: a copied
//! or loaded dictionary's content starts the buffer, where the one-shot
//! frame joins it before the input, and stays until the buffer first
//! moves, which is after the input passes the window size and the content
//! leaves the window; an attached dictionary is searched in place by every
//! block.
//!
//! [`compress_blocks`]: super::block::compress_blocks

use super::block::{self, InputEnd, JobBlocks};
use super::dict::FrameDict;
use super::{
    begin_job, block_sizing, default_search_method, multithreaded, split, write_epilogue,
    write_frame_header, write_raw_block, CompressDict, CompressError, CompressOptions, Compressor,
    Context, JobLdm, JobStart,
};
use crate::constants::ZSTD_BLOCKSIZE_MAX;
use crate::xxhash::Xxh64;
use std::io::{self, Write};
use std::sync::Arc;

/// `ZSTD_EndDirective`: what a [`Compressor::compress_stream`] call does
/// once its input is consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndDirective {
    /// `ZSTD_e_continue`: compress what can be, buffer the rest.
    Continue,
    /// `ZSTD_e_flush`: compress and emit everything buffered, ending the
    /// current block early.
    Flush,
    /// `ZSTD_e_end`: flush and end the frame (last block, checksum); the
    /// next call starts a new frame.
    End,
}

/// The streaming session of a [`Compressor`].
#[derive(Default)]
pub(super) struct Session {
    /// `pledgedSrcSizePlusOne - 1` for the next frame; reset when a frame
    /// ends.
    pledged: Option<u64>,
    stage: Stage,
    /// Compressed bytes `out[flushed..]` not yet copied to the caller.
    out: Vec<u8>,
    flushed: usize,
}

#[derive(Default)]
enum Stage {
    /// `zcss_init`: no frame; the next call starts one.
    #[default]
    Idle,
    /// A frame is in progress.
    Frame(Box<Frame>),
    /// The frame is complete; `out` may still hold its tail.
    Ended,
    /// A call failed; the frame is abandoned until
    /// [`Compressor::reset_stream`].
    Failed,
}

impl Session {
    /// Copy pending output into `dst[*dst_pos..]`; whether all of it fit.
    fn flush_to(&mut self, dst: &mut [u8], dst_pos: &mut usize) -> bool {
        let n = (dst.len() - *dst_pos).min(self.out.len() - self.flushed);
        dst[*dst_pos..*dst_pos + n].copy_from_slice(&self.out[self.flushed..self.flushed + n]);
        *dst_pos += n;
        self.flushed += n;
        if self.flushed < self.out.len() {
            return false;
        }
        self.out.clear();
        self.flushed = 0;
        true
    }

    fn pending(&self) -> usize {
        self.out.len() - self.flushed
    }
}

/// `pledgedSrcSize` as the workspace sizing and the ZSTDMT check read it:
/// `ZSTD_CONTENTSIZE_UNKNOWN`, larger than any input, for an unknown size.
fn frame_size(pledged: Option<u64>) -> usize {
    pledged.map_or(usize::MAX, |p| usize::try_from(p).unwrap_or(usize::MAX))
}

/// One streaming frame: its context and job state, and the input buffer.
struct Frame {
    ctx: Box<Context>,
    /// The dictionary the finders search in place, when attached
    /// ([`FrameDict::dict_match_state`]).
    attached: Option<Arc<CompressDict>>,
    /// Single-context long distance matching is on.
    ldm: bool,
    split: bool,
    blocks: JobBlocks,
    checksum: Option<Xxh64>,
    pledged: Option<u64>,
    consumed: u64,
    /// The frame header until the first block is written.
    header: Option<Vec<u8>>,
    /// Input: the window before `blocks.next_start()`, then the input not
    /// yet compressed. A copied or loaded dictionary's content comes first
    /// until the buffer first moves.
    buf: Vec<u8>,
    /// The window size, kept before the next block when the buffer moves.
    keep: usize,
    /// The buffer length that moves it: room for a dictionary's content in
    /// the buffer, which matches may reference whole until the input passes
    /// the window size, and twice the window and a block.
    cap: usize,
    block_size_max: usize,
}

impl Frame {
    /// `ZSTD_CCtx_init_compressStream2` on `ctx` for `pledged` bytes, or an
    /// unknown size, with `prefix` if given, else `opts.dict` if set: the
    /// parameters and dictionary use `ZSTD_compressBegin_internal` resolves
    /// for it, which a one-shot frame of that size resolves too. A copied
    /// or loaded dictionary's content starts the buffer, as it starts the
    /// one-shot frame's joined input.
    fn begin(
        opts: &CompressOptions,
        pledged: Option<u64>,
        prefix: Option<&[u8]>,
        mut ctx: Box<Context>,
    ) -> Self {
        let size = frame_size(pledged);
        let cdict = opts.dict.clone();
        let dict = match (prefix, &cdict) {
            (Some(prefix), _) => Some(FrameDict::prefix(prefix, pledged, opts)),
            (None, Some(cdict)) => Some(FrameDict::of(cdict, pledged, opts)),
            (None, None) => None,
        };
        let dict = dict.as_ref();
        debug_assert!(!multithreaded(opts, size));
        let (frame_cparams, cparams, ldm_params) = match dict {
            Some(dict) => dict.params(false),
            None => {
                let (cparams, ldm) = opts.frame_params(pledged);
                (cparams, cparams, ldm)
            }
        };
        let mut header = Vec::new();
        let id = dict.map_or(0, FrameDict::id);
        write_frame_header(&mut header, pledged, cparams.window_log, opts.checksum, id);
        let sizing = block_sizing(opts, &cparams, false, header.len());
        let frequently = opts.overflow_correct_frequently;
        let method = dict.map_or_else(|| default_search_method(&cparams), FrameDict::search_method);
        let buf = dict.map_or_else(Vec::new, |dict| dict.content().to_vec());
        let ldm = ldm_params.map_or(JobLdm::Off, |params| {
            let start = dict.map_or(0..0, |dict| dict.ldm_content(false));
            JobLdm::Internal(params, &buf, start)
        });
        let (ms, scratch, state, _) = ctx.reset(cparams, method, 0, ldm, size, frequently);
        let start = JobStart::First(dict);
        let blocks = begin_job(ms, scratch, state, &buf, 0..buf.len(), start, sizing, true);
        let attaches = dict.and_then(FrameDict::dict_match_state).is_some();
        let keep = 1usize << cparams.window_log;
        Self {
            ctx,
            attached: cdict.filter(|_| attaches),
            ldm: ldm_params.is_some(),
            split: split::block_splitter_enabled(opts.split_after_sequences, &frame_cparams),
            blocks,
            checksum: opts.checksum.then(Xxh64::new),
            pledged,
            consumed: 0,
            header: Some(header),
            cap: buf.len() + 2 * (keep + sizing.block_size_max),
            buf,
            keep,
            block_size_max: sizing.block_size_max,
        }
    }

    /// Buffer as much of `input` as fits and return how much; a full
    /// buffer first moves its last window down.
    fn accept(&mut self, input: &[u8]) -> usize {
        if self.buf.len() == self.cap {
            // Only blocks no later input can change are left: at most
            // `block_size_max + 1` bytes follow the next block start.
            let shift = self.blocks.next_start() - self.keep;
            self.buf.copy_within(shift.., 0);
            self.buf.truncate(self.buf.len() - shift);
            self.ctx.rebase(shift, self.ldm);
            self.blocks.rebase(shift);
        }
        let take = input.len().min(self.cap - self.buf.len());
        let input = &input[..take];
        if self.buf.capacity() < self.buf.len() + take {
            let want = (2 * self.buf.capacity()).clamp(self.buf.len() + take, self.cap);
            self.buf.reserve_exact(want - self.buf.len());
        }
        self.buf.extend_from_slice(input);
        if let Some(checksum) = &mut self.checksum {
            checksum.update(input);
        }
        self.consumed += take as u64;
        take
    }

    /// Whether buffered input is not yet compressed.
    fn holds_input(&self) -> bool {
        self.blocks.next_start() < self.buf.len()
    }

    /// `ZSTD_nextInputSizeHint`: the input that would complete the next
    /// block.
    fn input_hint(&self) -> usize {
        let held = self.buf.len() - self.blocks.next_start();
        (self.block_size_max + 1).saturating_sub(held).max(1)
    }

    /// Compress the blocks of the buffered input that `input` makes ready,
    /// the frame header before the first. As a one-shot frame's, they go to
    /// rayon only when the pipelined loop overlaps some of them, and then
    /// all at once: overlapping from outside the pool, every overlap was a
    /// thread hop, and the match state went to whichever worker took it.
    fn compress(&mut self, input: InputEnd, out: &mut Vec<u8>) {
        if !self.blocks.has_ready(input) {
            return;
        }
        if let Some(header) = self.header.take() {
            out.extend_from_slice(&header);
        }
        #[cfg(feature = "parallel")]
        if self.blocks.overlaps(input) {
            rayon::scope(|_| {
                let _in_job = super::InJob::enter();
                self.compress_blocks(input, out, true);
            });
            return;
        }
        self.compress_blocks(input, out, false);
    }

    /// [`block::compress_blocks`] over the buffered input up to `input`.
    fn compress_blocks(&mut self, input: InputEnd, out: &mut Vec<u8>, pipelined: bool) {
        let (ms, scratch, state, mut ldm) = self.ctx.resume(self.ldm);
        block::compress_blocks(
            ms,
            &self.buf,
            &mut self.blocks,
            input,
            self.split,
            state,
            scratch,
            &mut ldm,
            self.attached.as_deref().map(CompressDict::dict_match_state),
            out,
            pipelined,
        );
    }

    /// `ZSTD_compressEnd`'s size control: the input is not the pledged
    /// size.
    fn size_wrong(&self) -> Option<CompressError> {
        let pledged = self.pledged.filter(|&p| p != self.consumed)?;
        let consumed = self.consumed;
        Some(CompressError::SrcSizeWrong { pledged, consumed })
    }

    /// `ZSTD_compressEnd`: the buffered input as the frame's last blocks,
    /// or, with none left, `ZSTD_writeEpilogue`'s empty last block; then
    /// the checksum. Returns the context.
    fn finish(mut self, out: &mut Vec<u8>) -> Box<Context> {
        let end = InputEnd::JobEnd(self.buf.len());
        if self.blocks.has_ready(end) {
            self.compress(end, out);
        } else {
            if let Some(header) = self.header.take() {
                out.extend_from_slice(&header);
            }
            write_raw_block(out, &[], true);
        }
        write_epilogue(out, self.checksum);
        self.ctx
    }
}

impl Compressor {
    /// `ZSTD_CCtx_setPledgedSrcSize`: the next frame's input size, written
    /// to its header and checked as it arrives; `None` (the default) is
    /// unknown. Only before a frame starts, else
    /// [`CompressError::StageWrong`].
    pub fn set_pledged_src_size(&mut self, size: Option<u64>) -> Result<(), CompressError> {
        match self.stream.stage {
            Stage::Idle => {
                self.stream.pledged = size;
                Ok(())
            }
            _ => Err(CompressError::StageWrong),
        }
    }

    /// `ZSTD_CCtx_refPrefix`: `prefix`, copied, is the raw-content prefix
    /// of the next frame alone, whether [`Compressor::compress`] or
    /// [`Compressor::compress_stream`] starts it, in place of
    /// [`CompressOptions::dict`]; see [`Compressor::compress_with_prefix`]
    /// for what the frame is. An empty `prefix` clears it, and
    /// [`Compressor::reset_stream`] keeps it, as libzstd's session reset
    /// does. Only before a frame starts, else [`CompressError::StageWrong`].
    pub fn set_prefix(&mut self, prefix: &[u8]) -> Result<(), CompressError> {
        match self.stream.stage {
            Stage::Idle => {
                self.prefix = (!prefix.is_empty()).then(|| prefix.to_vec());
                Ok(())
            }
            _ => Err(CompressError::StageWrong),
        }
    }

    /// `ZSTD_CCtx_reset(ZSTD_reset_session_only)`: abandon the streaming
    /// frame in progress, its pending output and the pledged size; the
    /// options stay.
    pub fn reset_stream(&mut self) {
        let session = std::mem::take(&mut self.stream);
        if let Stage::Frame(frame) = session.stage {
            self.contexts.give_back(frame.ctx);
        }
    }

    /// `ZSTD_compressStream2`: consume `src[*src_pos..]` and write
    /// compressed bytes to `dst[*dst_pos..]`, advancing both positions.
    /// The first call of a frame starts it with the pledged size
    /// ([`Compressor::set_pledged_src_size`]); a first call with
    /// [`EndDirective::End`] compresses its whole input as one frame, as
    /// [`Compressor::compress`] does, and that is the pledged size.
    ///
    /// Returns, for `Flush` and `End`, a lower bound of the bytes still to
    /// write: `0` once the flush, or the frame, is complete, else the call
    /// is to be repeated with room in `dst`. For `Continue`, the input that
    /// would complete the next block.
    ///
    /// Without `Flush`, the frame is the one [`Compressor::compress`]
    /// writes for the same input when its size was pledged, however the
    /// input is cut into calls; without a pledged size, the header has no
    /// content size and the parameters, and how a dictionary is used, are
    /// those of an unknown size. A frame takes the prefix
    /// [`Compressor::set_prefix`] left, else [`CompressOptions::dict`].
    ///
    /// Errors: [`CompressError::SrcSizeWrong`] when the input passes the
    /// pledged size (the call consumes none of it) or ends short of it;
    /// [`CompressError::Unsupported`] for a `job_size` frame that is not
    /// one first `End` call or pledged at most `JOBSIZE_MIN`
    /// (multithreaded streaming is not implemented). After an error every
    /// call returns [`CompressError::StageWrong`] until
    /// [`Compressor::reset_stream`].
    ///
    /// Panics if a position is past its buffer's end.
    pub fn compress_stream(
        &mut self,
        src: &[u8],
        src_pos: &mut usize,
        dst: &mut [u8],
        dst_pos: &mut usize,
        end_op: EndDirective,
    ) -> Result<usize, CompressError> {
        assert!(*src_pos <= src.len(), "src_pos past src");
        assert!(*dst_pos <= dst.len(), "dst_pos past dst");
        let result = self.stream_loop(src, src_pos, dst, dst_pos, end_op);
        if result.is_err() {
            if let Stage::Frame(frame) = std::mem::replace(&mut self.stream.stage, Stage::Failed) {
                self.contexts.give_back(frame.ctx);
            }
        }
        result
    }

    fn stream_loop(
        &mut self,
        src: &[u8],
        src_pos: &mut usize,
        dst: &mut [u8],
        dst_pos: &mut usize,
        end_op: EndDirective,
    ) -> Result<usize, CompressError> {
        loop {
            if !self.stream.flush_to(dst, dst_pos) {
                break;
            }
            match &mut self.stream.stage {
                Stage::Failed => return Err(CompressError::StageWrong),
                Stage::Ended => {
                    // ZSTD_CCtx_reset(zcs, ZSTD_reset_session_only)
                    self.stream.stage = Stage::Idle;
                    self.stream.pledged = None;
                    if end_op == EndDirective::End && *src_pos == src.len() {
                        return Ok(0);
                    }
                }
                Stage::Idle if end_op == EndDirective::End => {
                    // The first call ends the frame: ZSTD_compressEnd over
                    // the whole input, the one-shot frame (ZSTD_compress2).
                    let rest = &src[*src_pos..];
                    if let Some(pledged) = self.stream.pledged.filter(|&p| p != rest.len() as u64) {
                        let consumed = rest.len() as u64;
                        return Err(CompressError::SrcSizeWrong { pledged, consumed });
                    }
                    let mut out = std::mem::take(&mut self.stream.out);
                    self.compress_next(rest, &mut out);
                    self.stream.out = out;
                    *src_pos = src.len();
                    self.stream.stage = Stage::Ended;
                }
                Stage::Idle => {
                    if multithreaded(&self.opts, frame_size(self.stream.pledged)) {
                        let what = "job_size streaming over JOBSIZE_MIN or of unknown size";
                        return Err(CompressError::Unsupported(what));
                    }
                    self.contexts.expand(1);
                    let ctx = self.contexts.take();
                    let prefix = self.prefix.take();
                    let frame =
                        Frame::begin(&self.opts, self.stream.pledged, prefix.as_deref(), ctx);
                    self.stream.stage = Stage::Frame(Box::new(frame));
                }
                Stage::Frame(frame) => {
                    let out = &mut self.stream.out;
                    if *src_pos < src.len() {
                        let rest = src.len() - *src_pos;
                        if let Some(pledged) =
                            frame.pledged.filter(|&p| frame.consumed + rest as u64 > p)
                        {
                            let consumed = frame.consumed + rest as u64;
                            return Err(CompressError::SrcSizeWrong { pledged, consumed });
                        }
                        *src_pos += frame.accept(&src[*src_pos..]);
                        frame.compress(InputEnd::Open(frame.buf.len()), out);
                        continue;
                    }
                    match end_op {
                        EndDirective::Continue => break,
                        EndDirective::Flush if frame.holds_input() => {
                            frame.compress(InputEnd::Chunk(frame.buf.len()), out);
                        }
                        EndDirective::Flush => break,
                        EndDirective::End => {
                            if let Some(e) = frame.size_wrong() {
                                return Err(e);
                            }
                            let Stage::Frame(frame) =
                                std::mem::replace(&mut self.stream.stage, Stage::Ended)
                            else {
                                unreachable!()
                            };
                            let ctx = frame.finish(&mut self.stream.out);
                            self.contexts.give_back(ctx);
                        }
                    }
                }
            }
        }
        let pending = self.stream.pending();
        let unconsumed = *src_pos < src.len();
        Ok(match (&self.stream.stage, end_op) {
            (Stage::Frame(frame), EndDirective::Continue) if pending == 0 => frame.input_hint(),
            (Stage::Frame(_), EndDirective::End) => pending.max(1),
            (Stage::Frame(frame), EndDirective::Flush) if unconsumed || frame.holds_input() => {
                pending.max(1)
            }
            _ if unconsumed => pending.max(1),
            _ => pending,
        })
    }
}

/// `ZSTD_CStreamOutSize`: room for one whole block and its headers.
const OUT_SIZE: usize = ZSTD_BLOCKSIZE_MAX + (ZSTD_BLOCKSIZE_MAX >> 8) + 64;

/// An [`io::Write`] adapter over [`Compressor::compress_stream`]: `write`
/// is `Continue`, `flush` is `Flush` then the writer's flush, and
/// [`Encoder::finish`] is `End`. Dropped unfinished, the frame is left
/// incomplete.
pub struct Encoder<W: Write> {
    writer: W,
    cctx: Compressor,
    out: Vec<u8>,
}

impl<W: Write> Encoder<W> {
    /// A frame compressed with `opts` into `writer`, of unknown size.
    pub fn new(writer: W, opts: CompressOptions) -> Self {
        Self::with_compressor(writer, Compressor::new(opts))
    }

    /// A frame compressed with `cctx`, whose pledged size it keeps, into
    /// `writer`.
    pub fn with_compressor(writer: W, cctx: Compressor) -> Self {
        Self {
            writer,
            cctx,
            out: vec![0; OUT_SIZE],
        }
    }

    /// The underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.writer
    }

    /// Run `end_op` over `src` until it is consumed and, for `Flush` and
    /// `End`, complete, writing every output buffer.
    fn run(&mut self, src: &[u8], end_op: EndDirective) -> io::Result<()> {
        let mut src_pos = 0;
        loop {
            let mut dst_pos = 0;
            let left = self
                .cctx
                .compress_stream(src, &mut src_pos, &mut self.out, &mut dst_pos, end_op)
                .map_err(io::Error::other)?;
            self.writer.write_all(&self.out[..dst_pos])?;
            let done = match end_op {
                EndDirective::Continue => src_pos == src.len() && dst_pos < self.out.len(),
                EndDirective::Flush | EndDirective::End => left == 0,
            };
            if done {
                return Ok(());
            }
        }
    }

    /// End the frame and return the writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.run(&[], EndDirective::End)?;
        Ok(self.writer)
    }
}

impl<W: Write> Write for Encoder<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.run(buf, EndDirective::Continue)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.run(&[], EndDirective::Flush)?;
        self.writer.flush()
    }
}
