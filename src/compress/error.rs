//! The compressor's error type.

use std::fmt;

/// Why a compression call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompressError {
    /// `srcSize_wrong`: the frame's input did not match the pledged size.
    /// `consumed` is the input the frame would have had: past `pledged`
    /// when a call brought more, below it when the frame ended short.
    SrcSizeWrong { pledged: u64, consumed: u64 },
    /// `stage_wrong`: the call is not valid at this point of the stream,
    /// e.g. pledging a size while a frame is in progress, or any call
    /// after an error before [`Compressor::reset_stream`].
    ///
    /// [`Compressor::reset_stream`]: super::Compressor::reset_stream
    StageWrong,
    /// The option combination is not supported by this call.
    Unsupported(&'static str),
    /// `dictionary_corrupted`: a structured dictionary's entropy tables or
    /// repeat offsets do not parse, see [`CompressDict::new`].
    ///
    /// [`CompressDict::new`]: super::CompressDict::new
    DictionaryCorrupted,
    /// `dictionary_wrong`: [`DictContentType::FullDict`] for a dictionary
    /// that is not structured.
    ///
    /// [`DictContentType::FullDict`]: super::DictContentType::FullDict
    DictionaryWrong,
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressError::SrcSizeWrong { pledged, consumed } => write!(
                f,
                "pledged source size {pledged} but the frame's input is {consumed} bytes"
            ),
            CompressError::StageWrong => f.write_str("call not valid at this stream stage"),
            CompressError::Unsupported(what) => write!(f, "unsupported: {what}"),
            CompressError::DictionaryCorrupted => f.write_str("dictionary is corrupted"),
            CompressError::DictionaryWrong => {
                f.write_str("dictionary is not a structured zstd dictionary")
            }
        }
    }
}

impl std::error::Error for CompressError {}
