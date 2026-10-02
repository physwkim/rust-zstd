#![allow(clippy::needless_range_loop, clippy::len_without_is_empty)]
//! Pure Rust Zstandard codec — compress + decompress, zero external dependencies.

/// Stable stand-ins for unstable `std::hint` items.
///
/// One place to delete when the feature stabilizes, rather than a shape the
/// call sites have to carry.
pub(crate) mod hint {
    /// `std::hint::cold_path()` is stable only from 1.95.0 (rust#136873), so
    /// calling it put a 1.95 floor on a crate that declares no `rust-version`.
    /// `#[cold]` on a real call carries no such floor and says the same thing;
    /// the call is to an empty function, and every site below already sits
    /// inside the branch it marks, so the instruction it costs is one the hot
    /// path does not execute.
    #[cold]
    #[inline(never)]
    pub(crate) fn cold_path() {}
}

pub mod bitstream;
pub mod compress;
pub mod constants;
pub mod decode;
pub mod fse;
pub mod huf;
mod xxhash;

pub use compress::{
    compress, compress_to_vec, compress_with, CompressError, CompressOptions, Compressor, Encoder,
    EndDirective, ParamSwitch,
};
pub use decode::{decompress, DecompressReader, Decompressor};
