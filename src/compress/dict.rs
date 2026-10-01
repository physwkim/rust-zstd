//! Dictionaries: `ZSTD_CDict` ([`CompressDict`]), the dictionary parse of
//! `ZSTD_compress_insertDictionary`, and how a frame starts from a
//! dictionary (`FrameDict`).
//!
//! Layout. A frame with a dictionary compresses `content ++ input` as its
//! one job: the dictionary's content (all of a raw-content dictionary, the
//! rest of a structured one after its entropy tables and repeat offsets)
//! at positions `0..content.len()`, the window's origin, and the input
//! after it. In index space the content ends where the input begins, as in
//! libzstd, where `ZSTD_loadDictionaryContent`'s `ZSTD_window_update` puts
//! the content in the window and the input's then follows it, so every
//! distance `current - matchIndex`, into the dictionary too, is libzstd's.
//! The content enters the window through [`MatchState::enter_dict`], which
//! records where it ends (`loadedDictEnd`): matches may then reference all
//! of it, offsets above the window size included, until the input passes
//! the window size (RFC 8878 §5, [`Window::lowest_match_index`]).
//!
//! A [`CompressDict`] hashes its content once, into tables of its own
//! parameters, at those same positions from a fresh window: the content
//! at [`WINDOW_START_INDEX`], as a CDict's `ZSTD_reset_matchState` and
//! load lay it out. A frame that uses its tables copies them and the
//! window as they are ([`MatchState::copy_dict`],
//! `ZSTD_resetCCtx_byCopyingCDict`), and never hashes the content again.
//! libzstd attaches the tables of a dictionary large for its input instead
//! (`ZSTD_resetCCtx_byAttachingCDict`), to search them in place; here they
//! are copied for every input.
//!
//! Cost: the joined buffer copies the content and the input once per
//! frame, `content.len() + input.len()` bytes of allocation and memcpy. In
//! exchange every finder searches one contiguous buffer; libzstd instead
//! searches the dictionary where it lies (`ZSTD_extDict` and
//! `ZSTD_dictMatchState` variants of every finder). Its copied tables
//! keep the content in the `dictBase` segment, so the finders here follow
//! the `ZSTD_extDict` rules where the content ends ([`Window::dict_limit`]).
//!
//! [`Window::lowest_match_index`]: super::matchstate::Window::lowest_match_index
//! [`Window::dict_limit`]: super::matchstate::Window::dict_limit
//! [`WINDOW_START_INDEX`]: super::matchstate::WINDOW_START_INDEX

use super::block::{self, BlockState, TableLoad};
use super::lazy::{default_search_method, SearchMethod};
use super::ldm::LdmParams;
use super::matchstate::MatchState;
use super::opt::DictStats;
use super::params::{CParamMode, CParams, Strategy, ZSTD_CLEVEL_DEFAULT};
use super::{CompressError, CompressOptions};
use crate::constants::{LL_FSE_LOG, MAX_LL, MAX_ML, MAX_OFF, ML_FSE_LOG, OFF_FSE_LOG};
use crate::fse::{FseCTable, FseState, FseTableState};
use crate::huf::{HufState, HufTable, HUF_TABLELOG_DEFAULT, HUF_TABLELOG_MAX};
use std::fmt;

/// `ZSTD_MAGIC_DICTIONARY`: the first four bytes of a structured
/// dictionary (RFC 8878 §5).
pub const ZSTD_MAGIC_DICTIONARY: u32 = 0xEC30_A437;

/// `ZSTD_USE_CDICT_PARAMS_SRCSIZE_CUTOFF`: an input below this size uses a
/// [`CompressDict`]'s tables.
const USE_CDICT_PARAMS_SRCSIZE_CUTOFF: u64 = 128 << 10;
/// `ZSTD_USE_CDICT_PARAMS_DICTSIZE_MULTIPLIER`: so does an input below this
/// many times the dictionary's size.
const USE_CDICT_PARAMS_DICTSIZE_MULTIPLIER: u64 = 6;

/// `ZSTD_dictContentType_e`: how a dictionary's bytes are read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DictContentType {
    /// `ZSTD_dct_auto`: a dictionary that starts with
    /// [`ZSTD_MAGIC_DICTIONARY`] is structured, any other is raw content.
    #[default]
    Auto,
    /// `ZSTD_dct_rawContent`: all of it is content.
    RawContent,
    /// `ZSTD_dct_fullDict`: it must be structured.
    FullDict,
}

/// `ZSTD_CDict`: a dictionary parsed and its content hashed once, at a
/// compression level, for any number of frames (see the module docs). A
/// frame compressed with it ([`CompressOptions::dict`],
/// [`compress_with_dict`](super::compress_with_dict)) follows
/// `ZSTD_compress2` with `ZSTD_CCtx_refCDict`: the dictionary's level
/// supersedes the options' `level`, and the frame header carries its ID.
pub struct CompressDict {
    /// The content frames see before their input.
    content: Vec<u8>,
    /// The size of the whole dictionary (the CDict's `dictContentSize`),
    /// which the parameters are chosen for.
    dict_size: usize,
    /// `dictID`: 0 for raw content.
    id: u32,
    /// `compressionLevel`, 0 resolved to [`ZSTD_CLEVEL_DEFAULT`].
    level: i32,
    /// `cBlockState`: the repeat offsets and entropy tables frames start
    /// from.
    entropy: BlockState,
    /// The optimal parser's first statistics, from `entropy`.
    opt_stats: Option<DictStats>,
    /// `matchState`: the content hashed with the dictionary's parameters
    /// (`ZSTD_dtlm_full`).
    ms: MatchState,
}

impl CompressDict {
    /// `ZSTD_createCDict(dict, level)`: [`CompressDict::with_content_type`]
    /// with [`DictContentType::Auto`].
    pub fn new(dict: &[u8], level: i32) -> Result<Self, CompressError> {
        Self::with_content_type(dict, level, DictContentType::Auto)
    }

    /// `ZSTD_createCDict` reading `dict` as `content_type`: the parameters
    /// for `level` (`0` the default level 3) and a dictionary of
    /// `dict.len()` bytes before an input of unknown size
    /// (`ZSTD_cpm_createCDict`), and the dictionary loaded into them
    /// (`ZSTD_initCDict_internal`). A dictionary under 8 bytes is ignored
    /// unless `content_type` is [`DictContentType::FullDict`].
    ///
    /// # Errors
    ///
    /// [`CompressError::DictionaryWrong`] where `FullDict` meets a
    /// dictionary that is not structured, and
    /// [`CompressError::DictionaryCorrupted`] for a structured dictionary
    /// whose entropy tables or repeat offsets do not parse
    /// (`ZSTD_loadCEntropy`), or whose Huffman table is deeper than RFC
    /// 8878's 11 bits.
    pub fn with_content_type(
        dict: &[u8],
        level: i32,
        content_type: DictContentType,
    ) -> Result<Self, CompressError> {
        let level = if level == 0 {
            ZSTD_CLEVEL_DEFAULT
        } else {
            level
        };
        let cparams = CParams::for_level_with(level, None, dict.len(), CParamMode::CreateCDict);
        let (id, entropy, content) = insert_dictionary(dict, content_type)?;
        let content = content.to_vec();
        let mut ms = MatchState::new_for(cparams, 0, default_search_method(&cparams));
        if !content.is_empty() {
            block::load_dict(&mut ms, &content, 0..content.len(), TableLoad::Full);
        }
        Ok(Self {
            content,
            dict_size: dict.len(),
            id,
            level,
            opt_stats: DictStats::of(&entropy),
            entropy,
            ms,
        })
    }

    /// `ZSTD_getDictID_fromCDict`: the ID frames carry, 0 for raw content.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// The compression level frames use.
    pub fn level(&self) -> i32 {
        self.level
    }

    /// `ZSTD_getCParamsFromCDict`: the parameters of the hashed content.
    pub fn cparams(&self) -> CParams {
        self.ms.cparams
    }
}

impl fmt::Debug for CompressDict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressDict")
            .field("id", &self.id)
            .field("level", &self.level)
            .field("dict_size", &self.dict_size)
            .field("content_len", &self.content.len())
            .field("cparams", &self.ms.cparams)
            .finish_non_exhaustive()
    }
}

/// Where a frame's tables get the dictionary's content from.
enum Tables<'a> {
    /// `ZSTD_resetCCtx_byCopyingCDict`: a [`CompressDict`]'s.
    Copy(&'a MatchState),
    /// `ZSTD_loadDictionaryContent` with `ZSTD_dtlm_fast`, hashing the
    /// content with the frame's parameters.
    Load,
}

/// One frame's dictionary, resolved for its input: the parameters, where
/// its tables come from and the block state it starts from. The frame
/// driver takes it as an `Option<&FrameDict>`: the content goes before the
/// input (see the module docs), [`FrameDict::preload`] starts the job's
/// match state, and [`FrameDict::id`] goes in the frame header.
pub(super) struct FrameDict<'a> {
    content: &'a [u8],
    id: u32,
    /// The frame's parameters (`ZSTD_getCParamsFromCCtxParams`), which
    /// resolve the post-block splitter and long distance matching.
    frame: CParams,
    /// The parameters the frame is compressed with: `frame`, or a copied
    /// dictionary's with `frame`'s window log.
    applied: CParams,
    ldm: Option<LdmParams>,
    tables: Tables<'a>,
    /// The block state the frame starts from; `None` is `repStartValue`
    /// without entropy tables.
    entropy: Option<&'a BlockState>,
    opt_stats: Option<&'a DictStats>,
}

/// `attachDictSizeCutoffs`: the input size up to which libzstd attaches a
/// dictionary's tables of `strategy` rather than copying them, which
/// sizes the frame's parameters for the input alone
/// (`ZSTD_cpm_attachDict`).
fn attach_dict_size_cutoff(strategy: Strategy) -> u64 {
    match strategy {
        Strategy::Fast | Strategy::BtUltra | Strategy::BtUltra2 => 8 << 10,
        Strategy::DFast => 16 << 10,
        Strategy::Greedy
        | Strategy::Lazy
        | Strategy::Lazy2
        | Strategy::BtLazy2
        | Strategy::BtOpt => 32 << 10,
    }
}

impl<'a> FrameDict<'a> {
    /// `ZSTD_compress2` with `ZSTD_CCtx_refCDict(dict)` for an input of
    /// `src_size` bytes: `opts` at the dictionary's level, sized for the
    /// input and the dictionary (`ZSTD_getCParamMode`), then the
    /// dictionary's tables with the frame's window log for an input below
    /// 128 KiB or six times the dictionary's size, else the frame's own
    /// tables loaded with the content (`ZSTD_compressBegin_internal`). The
    /// dictionary's entropy tables and repeat offsets either way.
    pub(super) fn of(dict: &'a CompressDict, src_size: usize, opts: &CompressOptions) -> Self {
        let pledged = src_size as u64;
        let mode = if pledged <= attach_dict_size_cutoff(dict.ms.cparams.strategy) {
            CParamMode::AttachDict
        } else {
            CParamMode::NoAttachDict
        };
        let (frame, ldm) = opts.frame_cparams(dict.level, src_size, dict.dict_size, mode);
        let use_tables = dict.dict_size > 0
            && (pledged < USE_CDICT_PARAMS_SRCSIZE_CUTOFF
                || pledged < dict.dict_size as u64 * USE_CDICT_PARAMS_DICTSIZE_MULTIPLIER);
        let (applied, tables) = if use_tables {
            let applied = CParams {
                window_log: frame.window_log,
                ..dict.ms.cparams
            };
            (applied, Tables::Copy(&dict.ms))
        } else {
            (frame, Tables::Load)
        };
        Self {
            content: &dict.content,
            id: dict.id,
            frame,
            applied,
            ldm: ldm.map(|requested| requested.adjusted(&applied)),
            tables,
            entropy: Some(&dict.entropy),
            opt_stats: dict.opt_stats.as_ref(),
        }
    }

    /// `ZSTD_compress2` with `ZSTD_CCtx_refPrefix(prefix)` for an input of
    /// `src_size` bytes: `opts` sized for the input and a dictionary of
    /// `prefix.len()` bytes, the prefix loaded as raw content
    /// (`ZSTD_dct_rawContent`), unless it is under 8 bytes, without an ID,
    /// entropy tables or repeat offsets of its own.
    pub(super) fn prefix(prefix: &'a [u8], src_size: usize, opts: &CompressOptions) -> Self {
        let (frame, ldm) =
            opts.frame_cparams(opts.level, src_size, prefix.len(), CParamMode::NoAttachDict);
        let (_, _, content) = insert_dictionary(prefix, DictContentType::RawContent)
            .expect("raw content never fails to load");
        Self {
            content,
            id: 0,
            frame,
            applied: frame,
            ldm: ldm.map(|requested| requested.adjusted(&frame)),
            tables: Tables::Load,
            entropy: None,
            opt_stats: None,
        }
    }

    /// The content that goes before the input.
    pub(super) fn content(&self) -> &'a [u8] {
        self.content
    }

    /// The frame header's `Dictionary_ID`.
    pub(super) fn id(&self) -> u32 {
        self.id
    }

    /// The frame's parameters, those it is compressed with, and its long
    /// distance matching parameters (see [`FrameDict`]).
    pub(super) fn params(&self) -> (CParams, CParams, Option<LdmParams>) {
        (self.frame, self.applied, self.ldm)
    }

    /// The lazy finder the frame's tables are for: a copied dictionary's
    /// (`useRowMatchFinder` from the CDict), else the frame's default.
    pub(super) fn search_method(&self) -> SearchMethod {
        match self.tables {
            Tables::Copy(dict) => dict.search_method,
            Tables::Load => default_search_method(&self.applied),
        }
    }

    /// Start `ms`, just reset for the frame's one job with the window at
    /// the content's start in `data` (the content then the input), from
    /// the dictionary: copy its tables or load the content, seed the
    /// optimal parser's first statistics, and return the block state the
    /// first block starts from.
    pub(super) fn preload(&self, ms: &mut MatchState, data: &[u8]) -> BlockState {
        debug_assert_eq!(&data[..self.content.len()], self.content);
        match self.tables {
            Tables::Copy(dict) => ms.copy_dict(dict),
            Tables::Load if self.content.is_empty() => {}
            Tables::Load => block::load_dict(ms, data, 0..self.content.len(), TableLoad::Fast),
        }
        if let (Some(stats), Some(opt)) = (self.opt_stats, ms.opt.as_mut()) {
            opt.seed_dict(stats);
        }
        self.entropy.cloned().unwrap_or_else(BlockState::initial)
    }
}

/// `ZSTD_compress_insertDictionary` before the content load: the
/// dictionary's ID, the block state frames start from, and the content to
/// load. A dictionary under 8 bytes is ignored, raw content (`RawContent`,
/// or `Auto` without the magic number) loads whole with the initial block
/// state (`ZSTD_reset_compressedBlockState`), and a structured dictionary
/// gives its ID, entropy tables and repeat offsets ([`load_entropy`]) and
/// the rest as content.
fn insert_dictionary(
    dict: &[u8],
    content_type: DictContentType,
) -> Result<(u32, BlockState, &[u8]), CompressError> {
    let initial = || BlockState::initial();
    if dict.len() < 8 {
        return match content_type {
            DictContentType::FullDict => Err(CompressError::DictionaryWrong),
            _ => Ok((0, initial(), &[])),
        };
    }
    let structured = read_le32(dict, 0) == ZSTD_MAGIC_DICTIONARY;
    match (content_type, structured) {
        (DictContentType::RawContent, _) | (DictContentType::Auto, false) => {
            Ok((0, initial(), dict))
        }
        (DictContentType::FullDict, false) => Err(CompressError::DictionaryWrong),
        (_, true) => {
            let (entropy, header_len) = load_entropy(dict)?;
            Ok((read_le32(dict, 4), entropy, &dict[header_len..]))
        }
    }
}

fn read_le32(src: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(src[at..at + 4].try_into().unwrap())
}

/// `ZSTD_highbit32`.
fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

/// `ZSTD_loadCEntropy`: the block state of a structured dictionary (past
/// its magic number and ID) and the length of its header. The Huffman
/// table is `Valid`, reusable without a check, when it codes all 256
/// symbols; a sequence table when it codes every symbol up to the largest
/// (`ZSTD_dictNCountRepeat`), for offsets the largest a first block can
/// need, `dictContentSize + 128 KiB`. Else they are `Check`.
///
/// # Errors
///
/// [`CompressError::DictionaryCorrupted`] where libzstd returns
/// `dictionary_corrupted`: a table that does not parse, a sequence table
/// log above its maximum (8 for offsets, 9 for lengths), fewer than 12
/// bytes of repeat offsets, or a repeat offset of 0 or above the content's
/// size. Also for a Huffman table deeper than 11 bits, which libzstd
/// accepts up to 12 (`HUF_TABLELOG_MAX`): RFC 8878 §4.2.1 limits literal
/// codes to 11 bits, and a decoder held to it rejects the frames such a
/// table codes.
fn load_entropy(dict: &[u8]) -> Result<(BlockState, usize), CompressError> {
    const CORRUPTED: CompressError = CompressError::DictionaryCorrupted;
    let mut ip = 8;
    let (huf, huf_valid, len) = read_huf_ctable(&dict[ip..]).ok_or(CORRUPTED)?;
    ip += len;
    let (of_norm, of_max, of_log, len) =
        read_ncount(&dict[ip..], MAX_OFF, OFF_FSE_LOG).ok_or(CORRUPTED)?;
    // fill all offset symbols to avoid garbage at end of table
    let of_table = FseCTable::build(&of_norm, MAX_OFF, of_log);
    ip += len;
    let (ml_norm, ml_max, ml_log, len) =
        read_ncount(&dict[ip..], MAX_ML, ML_FSE_LOG).ok_or(CORRUPTED)?;
    let ml = repeat(
        FseCTable::build(&ml_norm, ml_max, ml_log),
        ncount_covers(&ml_norm, ml_max, MAX_ML),
    );
    ip += len;
    let (ll_norm, ll_max, ll_log, len) =
        read_ncount(&dict[ip..], MAX_LL, LL_FSE_LOG).ok_or(CORRUPTED)?;
    let ll = repeat(
        FseCTable::build(&ll_norm, ll_max, ll_log),
        ncount_covers(&ll_norm, ll_max, MAX_LL),
    );
    ip += len;

    let reps = dict.get(ip..ip + 12).ok_or(CORRUPTED)?;
    let rep: [u32; 3] = std::array::from_fn(|i| read_le32(reps, 4 * i));
    ip += 12;
    let content_size = dict.len() - ip;
    // The maximum offset that must be supported
    let of_code_max = u32::try_from(content_size + (128 << 10)).map_or(MAX_OFF, |max_offset| {
        (highbit32(max_offset) as usize).min(MAX_OFF)
    });
    let of = repeat(of_table, ncount_covers(&of_norm, of_max, of_code_max));
    // All repCodes must be <= dictContentSize and != 0
    if rep.iter().any(|&r| r == 0 || r as usize > content_size) {
        return Err(CORRUPTED);
    }
    let huf = if huf_valid {
        HufState::Valid(huf)
    } else {
        HufState::Check(huf)
    };
    let entropy = BlockState {
        rep,
        huf,
        fse: FseState { ll, of, ml },
    };
    Ok((entropy, ip))
}

fn repeat(table: FseCTable, valid: bool) -> FseTableState {
    if valid {
        FseTableState::Valid(table)
    } else {
        FseTableState::Check(table)
    }
}

/// `ZSTD_dictNCountRepeat(norm, dict_max, max) == FSE_repeat_valid`: the
/// table codes every symbol up to `max`.
fn ncount_covers(norm: &[i16], dict_max: usize, max: usize) -> bool {
    dict_max >= max && norm[..=max].iter().all(|&n| n != 0)
}

/// `FSE_readNCount` with `maxSymbolValue` `max`, then the dictionary's
/// table log check: the normalized counts over `0..=max`, zero past the
/// last one read (as `FSE_readNCount` clears them), the last symbol read,
/// the table log, and the header's length. `None` for a header that does
/// not parse, codes a symbol above `max` or needs more than `max_log`
/// bits.
fn read_ncount(src: &[u8], max: usize, max_log: u32) -> Option<(Vec<i16>, usize, u32, usize)> {
    let (log, counts, len) = crate::decode::parse_fse_header(src, max_log as u8).ok()?;
    let last = counts.len().checked_sub(1).filter(|&last| last <= max)?;
    let mut norm = vec![0i16; max + 1];
    for (n, &c) in norm.iter_mut().zip(&counts) {
        *n = c as i16;
    }
    Some((norm, last, u32::from(log), len))
}

/// `HUF_readCTable` with `maxSymbolValue` 255: the table a tree
/// description codes, whether it codes all 256 symbols
/// (`!hasZeroWeights && maxSymbolValue == 255`), and the description's
/// length. `None` where `HUF_readStats` fails, and for codes over 11 bits
/// (see [`load_entropy`]).
fn read_huf_ctable(src: &[u8]) -> Option<(HufTable, bool, usize)> {
    let (&header, rest) = src.split_first()?;
    let len = 1 + if header >= 128 {
        (usize::from(header) - 127).div_ceil(2)
    } else {
        usize::from(header)
    };
    let mut weights = crate::decode::decode_huf_weights_from_fse(rest, header).ok()?;
    // HUF_readStats: collect weight stats
    let mut rank_count = [0u32; HUF_TABLELOG_MAX as usize + 1];
    let mut weight_total = 0u32;
    for &w in &weights {
        if u32::from(w) > HUF_TABLELOG_MAX {
            return None;
        }
        rank_count[usize::from(w)] += 1;
        weight_total += (1 << w) >> 1;
    }
    if weight_total == 0 {
        return None;
    }
    // get last non-null symbol weight (implied, total must be 2^n)
    let table_log = highbit32(weight_total) + 1;
    if table_log > HUF_TABLELOG_MAX {
        return None;
    }
    let rest = (1 << table_log) - weight_total;
    if !rest.is_power_of_two() {
        return None;
    }
    let last_weight = highbit32(rest) + 1;
    weights.push(last_weight as u8);
    rank_count[last_weight as usize] += 1;
    // check tree construction validity: at least 2 elts of rank 1, must be
    // even
    if rank_count[1] < 2 || rank_count[1] & 1 != 0 {
        return None;
    }
    if table_log > HUF_TABLELOG_DEFAULT {
        return None;
    }

    // HUF_readCTable: fill nbBits, then the values, by rank in symbol order
    let mut elts = [0u64; 256];
    let mut nb_per_rank = [0u16; HUF_TABLELOG_MAX as usize + 2];
    for (elt, &w) in elts.iter_mut().zip(&weights) {
        let nb_bits = if w == 0 {
            0
        } else {
            table_log + 1 - u32::from(w)
        };
        *elt = u64::from(nb_bits);
        nb_per_rank[nb_bits as usize] += 1;
    }
    let mut val_per_rank = [0u16; HUF_TABLELOG_MAX as usize + 2];
    let mut min = 0u16;
    for n in (1..=table_log as usize).rev() {
        val_per_rank[n] = min; // get starting value within each rank
        min += nb_per_rank[n];
        min >>= 1;
    }
    for elt in &mut elts[..weights.len()] {
        let nb_bits = (*elt & 0xFF) as usize;
        let value = val_per_rank[nb_bits];
        val_per_rank[nb_bits] += 1;
        if nb_bits > 0 {
            *elt |= u64::from(value) << (64 - nb_bits);
        }
    }
    let table = HufTable {
        elts,
        table_log: table_log as u8,
        max_symbol: (weights.len() - 1) as u8,
    };
    let valid = rank_count[0] == 0 && weights.len() == 256;
    Some((table, valid, len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fse::write_ncount;
    use crate::huf::{build_ctable, write_ctable};

    const ID: u32 = 0x1234;
    const REP: [u32; 3] = [1, 4, 8];

    /// Normalized counts over `n` symbols, none zero, summing to
    /// `1 << log`.
    fn flat(n: usize, log: u32) -> Vec<i16> {
        let total = 1usize << log;
        (0..n)
            .map(|s| (total / n + usize::from(s < total % n)) as i16)
            .collect()
    }

    /// A tree description of `weights` in the direct (4 bits each) form.
    fn direct(weights: &[u8]) -> Vec<u8> {
        let mut out = vec![127 + weights.len() as u8];
        out.extend(
            weights
                .chunks(2)
                .map(|w| w[0] << 4 | w.get(1).unwrap_or(&0)),
        );
        out
    }

    /// Literal counts of 256 symbols, all different (a table of equal
    /// weights has no description).
    fn skewed() -> [u32; 256] {
        std::array::from_fn(|s| 1 + s as u32)
    }

    /// The Huffman table `HUF_buildCTable` builds for `count` and its
    /// description.
    fn huf_of(count: &[u32; 256]) -> (Vec<u8>, HufTable) {
        let table = build_ctable(count, 255, HUF_TABLELOG_DEFAULT).unwrap();
        let mut out = Vec::new();
        write_ctable(&mut out, &table, 255, u32::from(table.table_log)).unwrap();
        (out, table)
    }

    struct Parts {
        huf: Vec<u8>,
        /// Offset, match length and literal length counts and logs.
        seq: [(Vec<i16>, u32); 3],
        rep: [u32; 3],
        content: Vec<u8>,
    }

    impl Parts {
        /// Tables that code every symbol a first block can need, and 64
        /// bytes of content.
        fn valid() -> Self {
            Self {
                huf: huf_of(&skewed()).0,
                seq: [(flat(18, 5), 5), (flat(53, 6), 6), (flat(36, 6), 6)],
                rep: REP,
                content: (0..64).collect(),
            }
        }

        fn dict(&self) -> Vec<u8> {
            let mut d = ZSTD_MAGIC_DICTIONARY.to_le_bytes().to_vec();
            d.extend(ID.to_le_bytes());
            d.extend(&self.huf);
            for (norm, log) in &self.seq {
                write_ncount(&mut d, norm, norm.len() - 1, *log).unwrap();
            }
            d.extend(self.rep.iter().flat_map(|r| r.to_le_bytes()));
            d.extend(&self.content);
            d
        }

        fn load(&self) -> Result<BlockState, CompressError> {
            let d = self.dict();
            let (entropy, len) = load_entropy(&d)?;
            assert_eq!(&d[len..], self.content);
            Ok(entropy)
        }
    }

    #[test]
    fn insert_dictionary_follows_content_type() {
        use DictContentType::*;
        let structured = Parts::valid().dict();
        let raw = b"0123456789abcdef";
        let wrong = Err(CompressError::DictionaryWrong);
        assert_eq!(insert_dictionary(&raw[..7], FullDict).map(|r| r.0), wrong);
        assert_eq!(insert_dictionary(raw, FullDict).map(|r| r.0), wrong);
        for (dict, ct) in [
            (&raw[..7], Auto),
            (&raw[..7], RawContent),
            (raw, Auto),
            (raw, RawContent),
            (&structured[..], RawContent),
        ] {
            let (id, entropy, content) = insert_dictionary(dict, ct).unwrap();
            assert_eq!(id, 0);
            assert_eq!(entropy.rep, BlockState::initial().rep);
            assert!(matches!(entropy.huf, HufState::None));
            let whole = if dict.len() < 8 { &[][..] } else { dict };
            assert_eq!(content, whole);
        }
        for ct in [Auto, FullDict] {
            let (id, entropy, content) = insert_dictionary(&structured, ct).unwrap();
            assert_eq!((id, entropy.rep), (ID, REP));
            assert_eq!(content, Parts::valid().content);
        }
    }

    #[test]
    fn complete_tables_are_valid() {
        let entropy = Parts::valid().load().unwrap();
        assert!(matches!(&entropy.huf, HufState::Valid(t) if *t == huf_of(&skewed()).1));
        let fse = &entropy.fse;
        assert!(matches!(fse.of, FseTableState::Valid(_)));
        assert!(matches!(fse.ml, FseTableState::Valid(_)));
        assert!(matches!(fse.ll, FseTableState::Valid(_)));
    }

    #[test]
    fn tables_short_of_a_symbol_are_check() {
        let check_huf = |huf| {
            let entropy = Parts {
                huf,
                ..Parts::valid()
            }
            .load()
            .unwrap();
            assert!(matches!(entropy.huf, HufState::Check(_)));
        };
        // Two literals.
        check_huf(direct(&[1]));
        // A literal of weight 0.
        let mut count = skewed();
        count[7] = 0;
        check_huf(huf_of(&count).0);

        let fse = |seq| {
            Parts {
                seq,
                ..Parts::valid()
            }
            .load()
            .unwrap()
            .fse
        };
        let mut ml_gap = flat(52, 6);
        ml_gap.insert(9, 0);
        let f = fse([(flat(18, 5), 5), (ml_gap, 6), (flat(35, 6), 6)]);
        assert!(matches!(f.ml, FseTableState::Check(_)));
        assert!(matches!(f.ll, FseTableState::Check(_)));
        assert!(matches!(f.of, FseTableState::Valid(_)));
        // Offset codes up to 17 cover 64 bytes of content and 128 KiB.
        let f = fse([(flat(17, 5), 5), (flat(53, 6), 6), (flat(36, 6), 6)]);
        assert!(matches!(f.of, FseTableState::Check(_)));
        let big = Parts {
            content: vec![0; 128 << 10],
            ..Parts::valid()
        };
        assert!(matches!(
            big.load().unwrap().fse.of,
            FseTableState::Check(_)
        ));
    }

    /// Our verdict on `dict`, and whether libzstd accepts it
    /// (`ZSTD_createCDict`).
    fn verdicts(dict: &[u8]) -> (Result<(), CompressError>, bool) {
        let ours = load_entropy(dict).map(|_| ());
        (ours, zstd::zstd_safe::CDict::try_create(dict, 3).is_some())
    }

    #[test]
    fn corrupted_dictionaries_are_rejected() {
        let ok = (Ok(()), true);
        let corrupted = (Err(CompressError::DictionaryCorrupted), false);

        let rep = |rep| {
            verdicts(
                &Parts {
                    rep,
                    ..Parts::valid()
                }
                .dict(),
            )
        };
        assert_eq!(rep([1, 4, 64]), ok);
        for bad in [[0, 4, 8], [1, 0, 8], [1, 4, 0], [1, 65, 8]] {
            assert_eq!(rep(bad), corrupted, "repeat offsets {bad:?}");
        }
        let d = Parts {
            content: Vec::new(),
            ..Parts::valid()
        }
        .dict();
        assert_eq!(verdicts(&d[..d.len() - 1]), corrupted);

        let seq = |seq| {
            verdicts(
                &Parts {
                    seq,
                    ..Parts::valid()
                }
                .dict(),
            )
        };
        let (of, ml, ll) = ((flat(18, 5), 5), (flat(53, 6), 6), (flat(36, 6), 6));
        assert_eq!(seq([(flat(18, 8), 8), ml.clone(), ll.clone()]), ok);
        assert_eq!(seq([(flat(18, 9), 9), ml.clone(), ll.clone()]), corrupted);
        assert_eq!(seq([(flat(33, 6), 6), ml.clone(), ll.clone()]), corrupted);
        assert_eq!(seq([of.clone(), (flat(53, 9), 9), ll.clone()]), ok);
        assert_eq!(seq([of.clone(), (flat(53, 10), 10), ll.clone()]), corrupted);
        assert_eq!(seq([of.clone(), (flat(54, 6), 6), ll.clone()]), corrupted);
        assert_eq!(seq([of.clone(), ml.clone(), (flat(36, 9), 9)]), ok);
        assert_eq!(seq([of.clone(), ml.clone(), (flat(36, 10), 10)]), corrupted);
        assert_eq!(seq([of, ml, (flat(37, 6), 6)]), corrupted);

        let huf = |huf| {
            verdicts(
                &Parts {
                    huf,
                    ..Parts::valid()
                }
                .dict(),
            )
        };
        // Weights 11..=1 and an implied 1: an 11-bit code.
        let chain: Vec<u8> = (1..=11).rev().collect();
        assert_eq!(huf(direct(&chain)), ok);
        // Weights 12..=1: a 12-bit code, which libzstd accepts.
        let chain: Vec<u8> = (1..=12).rev().collect();
        assert_eq!(
            huf(direct(&chain)),
            (Err(CompressError::DictionaryCorrupted), true)
        );
        // 4 + 1 leaves 3 to a power of two: no implied last weight.
        assert_eq!(huf(direct(&[3, 1])), corrupted);
    }

    #[test]
    fn frame_header_carries_the_dictionary_id() {
        let src = b"a frame with a dictionary ID".repeat(4);
        for (id, code) in [(0, 0), (1, 1), (255, 1), (256, 2), (65535, 2), (65536, 3)] {
            let mut d = Parts::valid().dict();
            d[4..8].copy_from_slice(&u32::to_le_bytes(id));
            let frame = super::super::compress_with_dict(&src, &CompressDict::new(&d, 3).unwrap());
            let fhd = frame[4];
            assert_eq!(fhd & 3, code, "dictionary ID {id}");
            let at = 6 - usize::from(fhd >> 5 & 1);
            let len = [0, 1, 2, 4][usize::from(code)];
            let mut field = [0u8; 4];
            field[..len].copy_from_slice(&frame[at..at + len]);
            assert_eq!(u32::from_le_bytes(field), id);
        }
    }
}
