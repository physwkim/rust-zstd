//! Decoder dictionaries (RFC 8878 §5; libzstd zstd_ddict.c and
//! ZSTD_loadDEntropy of zstd_decompress.c).

use super::{
    DecoderScratch, FSEScratch, FSETable, HuffmanScratch, HuffmanTable, ModeType, SEQ_TABLES,
};
use std::sync::Arc;

/// Magic_Number of a formatted dictionary (RFC 8878 line 1809).
const DICT_MAGIC: u32 = 0xEC30_A437;

/// A dictionary parsed once for any number of decodes (libzstd ZSTD_DDict).
///
/// A dictionary of at least 8 bytes that starts with Magic_Number
/// 0xEC30A437 is a formatted dictionary (RFC 8878 §5): its entropy tables and repeat offsets start every frame decoded
/// with it, and the bytes after them are its content. Any other byte
/// string is a raw-content dictionary of Dictionary_ID 0: all of it is
/// content (ZSTD_decompress_insertDictionary).
///
/// Two RFC 8878 §5 constraints on dictionaries are not enforced, as in
/// libzstd: a raw-content dictionary may be shorter than the 8 bytes of
/// lines 1786-1787 (it is plain history; libzstd's encoder ignores one
/// below 8 bytes, ZSTD_compress_insertDictionary), and a formatted
/// dictionary may have Dictionary_ID 0 against lines 1812-1813 (0 only
/// means "no dictionary" in a frame header). These resolve which history
/// and tables a frame gets, not whether a frame is well-formed.
pub struct DecodeDict {
    id: u32,
    content: Vec<u8>,
    entropy: Option<Arc<DictEntropy>>,
}

/// The entropy section of a formatted dictionary, as the tables a frame's
/// first blocks may repeat (ZSTD_entropyDTables_t).
pub(super) struct DictEntropy {
    /// Huffman table for Treeless literals, double-symbol as libzstd
    /// builds it (HUF_readDTableX2_wksp).
    pub(super) huf: HuffmanScratch,
    /// Tables for Repeat mode sequences.
    pub(super) fse: FSEScratch,
    /// Starting repeat offsets, each in 1..=content length.
    pub(super) rep: [u32; 3],
}

impl DecodeDict {
    /// Parse `dict`. A formatted dictionary whose entropy section libzstd
    /// rejects (dictionary_corrupted) is an error.
    pub fn new(dict: &[u8]) -> Result<DecodeDict, String> {
        let magic = dict
            .get(..4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        if dict.len() < 8 || magic != Some(DICT_MAGIC) {
            return Ok(DecodeDict {
                id: 0,
                content: dict.to_vec(),
                entropy: None,
            });
        }
        let id = u32::from_le_bytes(dict[4..8].try_into().unwrap());
        let (entropy, used) =
            load_entropy(dict).map_err(|e| format!("Dictionary corrupted: {e}"))?;
        Ok(DecodeDict {
            id,
            content: dict[used..].to_vec(),
            entropy: Some(Arc::new(entropy)),
        })
    }

    /// Dictionary_ID: a frame naming another nonzero one is refused; 0 for
    /// a raw-content dictionary.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// The content: what follows the repeat offsets of a formatted
    /// dictionary, or all of a raw-content one.
    pub fn content(&self) -> &[u8] {
        &self.content
    }

    pub(super) fn entropy(&self) -> Option<&Arc<DictEntropy>> {
        self.entropy.as_ref()
    }
}

/// ZSTD_loadDEntropy: the Huffman table, then the OF, ML, LL tables, then
/// the three repeat offsets, of the formatted dictionary `dict`. Returns
/// them with the length of everything before the content.
fn load_entropy(dict: &[u8]) -> Result<(DictEntropy, usize), String> {
    if dict.len() <= 8 {
        return Err("no entropy tables".to_string());
    }
    let mut pos = 8;
    let mut huf = HuffmanTable::new();
    let (huf_len, nb_weights) = huf.read_weights(&dict[pos..])?;
    huf.weight_stats(nb_weights)?;
    huf.fill(true);
    pos += huf_len;

    let scratch = DecoderScratch::new();
    let mut fse = scratch.fse;
    // RFC 8878 lines 1826-1828: offsets, match lengths, literals lengths;
    // ZSTD_buildSeqTable's checks on an FSE_Compressed_Mode table.
    for t in [1, 2, 0] {
        pos += super::build_sequence_table(
            ModeType::FSECompressed,
            &dict[pos..],
            fse.table_mut(t),
            &SEQ_TABLES[t],
        )?;
    }

    let reps = dict
        .get(pos..pos + 12)
        .ok_or_else(|| "repeat offsets truncated".to_string())?;
    pos += 12;
    // RFC 8878 lines 1832-1833: each less than the dictionary size;
    // libzstd's bound, at most the content size, is the tighter one.
    let content_len = dict.len() - pos;
    let mut rep = [0u32; 3];
    for (r, b) in rep.iter_mut().zip(reps.as_chunks::<4>().0) {
        *r = u32::from_le_bytes(*b);
        if *r == 0 || *r as usize > content_len {
            return Err(format!(
                "repeat offset {} outside content of {} bytes",
                *r, content_len
            ));
        }
    }
    Ok((
        DictEntropy {
            huf: HuffmanScratch { table: huf },
            fse,
            rep,
        },
        pos,
    ))
}

impl DecoderScratch {
    /// Start a frame from the dictionary's tables and repeat offsets
    /// (ZSTD_copyDDictParameters with entropyPresent): its tables stay in
    /// use until a block builds its own.
    pub(super) fn load_dict(&mut self, e: &Arc<DictEntropy>) {
        self.dict = Some(Arc::clone(e));
        self.huf_from_dict = true;
        self.fse_from_dict = [true; 3];
        self.offset_hist = e.rep;
    }

    /// The Huffman table Treeless literals would use now.
    pub(super) fn huf_table(&self) -> &HuffmanTable {
        match &self.dict {
            Some(d) if self.huf_from_dict => &d.huf.table,
            _ => &self.huf.table,
        }
    }

    /// The `SEQ_TABLES[t]` table Repeat mode would use now.
    pub(super) fn fse_table(&self, t: usize) -> &FSETable {
        match &self.dict {
            Some(d) if self.fse_from_dict[t] => d.fse.table(t),
            _ => self.fse.table(t),
        }
    }
}
