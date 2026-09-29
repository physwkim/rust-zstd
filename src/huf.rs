//! Huffman literals encoder.
//!
//! Port of libzstd 1.5.7 `lib/compress/huf_compress.c` (the
//! `HUF_compress1X_repeat` / `HUF_compress4X_repeat` path) and
//! `lib/compress/zstd_compress_literals.c` (`ZSTD_compressLiterals`).

use super::fse;
use crate::compress::{CParams, Strategy};
use crate::constants::*;

/// `HUF_TABLELOG_MAX`.
pub const HUF_TABLELOG_MAX: u32 = 12;
/// `HUF_TABLELOG_DEFAULT` == `LitHufLog`.
pub const HUF_TABLELOG_DEFAULT: u32 = 11;
/// `HUF_SYMBOLVALUE_MAX`.
pub const HUF_SYMBOLVALUE_MAX: usize = 255;
/// `MAX_FSE_TABLELOG_FOR_HUFF_HEADER`.
const MAX_FSE_TABLELOG_FOR_HUFF_HEADER: u32 = 6;

/// A built Huffman table: `HUF_CElt[256]` plus the `HUF_CTableHeader`.
///
/// Each element is packed as in C: `nb_bits` in the low 8 bits and the code
/// value left-aligned in the top `nb_bits` bits, so an encoder can OR it
/// straight into a 64-bit container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HufTable {
    pub elts: [u64; 256],
    /// `HUF_CTableHeader.tableLog`: the longest code in the table.
    pub table_log: u8,
    /// `HUF_CTableHeader.maxSymbolValue`: the alphabet the table was built for.
    pub max_symbol: u8,
}

impl HufTable {
    /// `HUF_getNbBits`.
    #[inline]
    pub fn nb_bits(&self, symbol: usize) -> u32 {
        (self.elts[symbol] & 0xFF) as u32
    }

    /// The code value of `symbol`, right-aligned.
    #[inline]
    pub fn value(&self, symbol: usize) -> u32 {
        let nb = self.nb_bits(symbol);
        if nb == 0 {
            0
        } else {
            (self.elts[symbol] >> (64 - nb)) as u32
        }
    }
}

/// `HUF_repeat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HufRepeat {
    None,
    Check,
    Valid,
}

/// `ZSTD_hufCTables_t`: the Huffman table the decoder currently holds and its
/// `HUF_repeat` mode. `None` (`HUF_repeat_none`): no table. `Check`
/// (`HUF_repeat_check`): the table may be reused if it covers every symbol.
/// `Valid` (`HUF_repeat_valid`): known to cover the symbols (dictionaries
/// only; never produced here).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum HufState {
    #[default]
    None,
    Check(HufTable),
    Valid(HufTable),
}

impl HufState {
    pub fn table(&self) -> Option<&HufTable> {
        match self {
            HufState::None => None,
            HufState::Check(t) | HufState::Valid(t) => Some(t),
        }
    }

    pub fn repeat(&self) -> HufRepeat {
        match self {
            HufState::None => HufRepeat::None,
            HufState::Check(_) => HufRepeat::Check,
            HufState::Valid(_) => HufRepeat::Valid,
        }
    }
}

fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

// =========================================================================
// Histogram
// =========================================================================

/// `HIST_count_wksp` (`HIST_count_parallel_wksp`): count every byte of
/// `src` into `counts` and return `(largest count, max symbol present)`.
/// Four interleaved histograms over 16-byte stripes break the dependency
/// chain between increments. `counts` is cleared first; `src` must not be
/// empty.
pub fn hist_count(counts: &mut [u32; 256], src: &[u8]) -> (u32, usize) {
    let mut c1 = [0u32; 256];
    let mut c2 = [0u32; 256];
    let mut c3 = [0u32; 256];
    let mut c4 = [0u32; 256];
    let (stripes, rest) = src.as_chunks::<16>();
    for stripe in stripes {
        for word in stripe.as_chunks::<4>().0 {
            let c = u32::from_le_bytes(*word);
            c1[(c & 0xFF) as usize] += 1;
            c2[((c >> 8) & 0xFF) as usize] += 1;
            c3[((c >> 16) & 0xFF) as usize] += 1;
            c4[(c >> 24) as usize] += 1;
        }
    }
    for &b in rest {
        c1[b as usize] += 1;
    }
    let mut max = 0;
    for s in 0..256 {
        let total = c1[s] + c2[s] + c3[s] + c4[s];
        counts[s] = total;
        max = max.max(total);
    }
    let mut max_symbol = 255;
    while counts[max_symbol] == 0 {
        max_symbol -= 1;
    }
    (max, max_symbol)
}

// =========================================================================
// HUF_buildCTable_wksp: HUF_sort, HUF_buildTree, HUF_setMaxHeight,
// HUF_buildCTableFromTree
// =========================================================================

/// `nodeElt`.
#[derive(Clone, Copy, Debug, Default)]
struct NodeElt {
    count: u32,
    parent: u16,
    byte: u8,
    nb_bits: u8,
}

/// `huffNodeTable`: `huffNode[i]` lives at index `i + 1`; index 0 is the
/// barrier entry the C code reaches through `huffNode[-1]`.
const HUFF_NODE_TABLE_SIZE: usize = 2 * (HUF_SYMBOLVALUE_MAX + 1);
/// `STARTNODE`.
const STARTNODE: isize = (HUF_SYMBOLVALUE_MAX + 1) as isize;

const RANK_POSITION_TABLE_SIZE: usize = 192;
const RANK_POSITION_LOG_BUCKETS_BEGIN: u32 = (RANK_POSITION_TABLE_SIZE as u32 - 1) - 32 - 1; // 158
const RANK_POSITION_DISTINCT_COUNT_CUTOFF: u32 = RANK_POSITION_LOG_BUCKETS_BEGIN + 7; // 166

/// `HUF_getIndex`.
fn huf_get_index(count: u32) -> usize {
    if count < RANK_POSITION_DISTINCT_COUNT_CUTOFF {
        count as usize
    } else {
        (highbit32(count) + RANK_POSITION_LOG_BUCKETS_BEGIN) as usize
    }
}

/// `HUF_insertionSort` over `nodes[low..=high]`.
fn huf_insertion_sort(nodes: &mut [NodeElt]) {
    for i in 1..nodes.len() {
        let key = nodes[i];
        let mut j = i as isize - 1;
        while j >= 0 && nodes[j as usize].count < key.count {
            nodes[j as usize + 1] = nodes[j as usize];
            j -= 1;
        }
        nodes[(j + 1) as usize] = key;
    }
}

/// `HUF_quickSortPartition`.
fn huf_quick_sort_partition(arr: &mut [NodeElt], low: isize, high: isize) -> isize {
    let pivot = arr[high as usize].count;
    let mut i = low - 1;
    for j in low..high {
        if arr[j as usize].count > pivot {
            i += 1;
            arr.swap(i as usize, j as usize);
        }
    }
    arr.swap((i + 1) as usize, high as usize);
    i + 1
}

/// `HUF_simpleQuickSort`: descending by count, not stable; the exact element
/// order it produces decides which equal-count symbol gets which code, so it
/// is ported step for step.
fn huf_simple_quick_sort(arr: &mut [NodeElt], mut low: isize, mut high: isize) {
    const K_INSERTION_SORT_THRESHOLD: isize = 8;
    if high - low < K_INSERTION_SORT_THRESHOLD {
        if high > low {
            huf_insertion_sort(&mut arr[low as usize..=high as usize]);
        }
        return;
    }
    while low < high {
        let idx = huf_quick_sort_partition(arr, low, high);
        if idx - low < high - idx {
            huf_simple_quick_sort(arr, low, idx - 1);
            low = idx + 1;
        } else {
            huf_simple_quick_sort(arr, idx + 1, high);
            high = idx - 1;
        }
    }
}

/// `HUF_sort`: bucket sort of symbols `0..=max_symbol` by descending count
/// into `huffNode[0..]` (`nodes[1..]`).
fn huf_sort(nodes: &mut [NodeElt; HUFF_NODE_TABLE_SIZE], count: &[u32], max_symbol: usize) {
    let huff_node = &mut nodes[1..];
    let max_symbol1 = max_symbol + 1;
    // (base, curr)
    let mut rank_position = [(0u16, 0u16); RANK_POSITION_TABLE_SIZE];
    for &c in &count[..max_symbol1] {
        rank_position[huf_get_index(c)].0 += 1;
    }
    for n in (1..RANK_POSITION_TABLE_SIZE).rev() {
        rank_position[n - 1].0 += rank_position[n].0;
        rank_position[n - 1].1 = rank_position[n - 1].0;
    }
    for (n, &c) in count[..max_symbol1].iter().enumerate() {
        let r = huf_get_index(c) + 1;
        let pos = rank_position[r].1 as usize;
        rank_position[r].1 += 1;
        huff_node[pos].count = c;
        huff_node[pos].byte = n as u8;
    }
    for rp in rank_position
        .iter()
        .take(RANK_POSITION_TABLE_SIZE - 1)
        .skip(RANK_POSITION_DISTINCT_COUNT_CUTOFF as usize)
    {
        let bucket_size = (rp.1 - rp.0) as usize;
        let start = rp.0 as usize;
        if bucket_size > 1 {
            huf_simple_quick_sort(
                &mut huff_node[start..start + bucket_size],
                0,
                bucket_size as isize - 1,
            );
        }
    }
}

/// `HUF_buildTree`: returns `nonNullRank`.
fn huf_build_tree(nodes: &mut [NodeElt; HUFF_NODE_TABLE_SIZE], max_symbol: usize) -> usize {
    // `huffNode[i]` is `nodes[i + 1]`; `i == -1` is the barrier.
    macro_rules! hn {
        ($i:expr) => {
            nodes[($i + 1) as usize]
        };
    }
    let mut non_null_rank = max_symbol as isize;
    while hn!(non_null_rank).count == 0 {
        non_null_rank -= 1;
    }
    let mut low_s = non_null_rank;
    let mut node_nb = STARTNODE;
    let node_root = node_nb + low_s - 1;
    let mut low_n = node_nb;
    hn!(node_nb).count = hn!(low_s).count + hn!(low_s - 1).count;
    hn!(low_s).parent = node_nb as u16;
    hn!(low_s - 1).parent = node_nb as u16;
    node_nb += 1;
    low_s -= 2;
    for n in node_nb..=node_root {
        hn!(n).count = 1 << 30;
    }
    hn!(-1).count = 1 << 31; // fake entry, strong barrier
    while node_nb <= node_root {
        let n1 = if hn!(low_s).count < hn!(low_n).count {
            low_s -= 1;
            low_s + 1
        } else {
            low_n += 1;
            low_n - 1
        };
        let n2 = if hn!(low_s).count < hn!(low_n).count {
            low_s -= 1;
            low_s + 1
        } else {
            low_n += 1;
            low_n - 1
        };
        hn!(node_nb).count = hn!(n1).count + hn!(n2).count;
        hn!(n1).parent = node_nb as u16;
        hn!(n2).parent = node_nb as u16;
        node_nb += 1;
    }
    hn!(node_root).nb_bits = 0;
    for n in (STARTNODE..node_root).rev() {
        let p = hn!(n).parent as isize;
        hn!(n).nb_bits = hn!(p).nb_bits + 1;
    }
    for n in 0..=non_null_rank {
        let p = hn!(n).parent as isize;
        hn!(n).nb_bits = hn!(p).nb_bits + 1;
    }
    non_null_rank as usize
}

/// `HUF_setMaxHeight`: limit every code to `target_nb_bits`, repaying the
/// Kraft cost from the cheapest ranks. Returns the resulting max depth.
fn huf_set_max_height(
    nodes: &mut [NodeElt; HUFF_NODE_TABLE_SIZE],
    last_non_null: usize,
    target_nb_bits: u32,
) -> u32 {
    macro_rules! hn {
        ($i:expr) => {
            nodes[($i + 1) as usize]
        };
    }
    let largest_bits = hn!(last_non_null as isize).nb_bits as u32;
    if largest_bits <= target_nb_bits {
        return largest_bits;
    }
    let mut total_cost: i32 = 0;
    let base_cost: i32 = 1 << (largest_bits - target_nb_bits);
    let mut n = last_non_null as isize;
    while hn!(n).nb_bits as u32 > target_nb_bits {
        total_cost += base_cost - (1 << (largest_bits - hn!(n).nb_bits as u32));
        hn!(n).nb_bits = target_nb_bits as u8;
        n -= 1;
    }
    debug_assert!(hn!(n).nb_bits as u32 <= target_nb_bits);
    while hn!(n).nb_bits as u32 == target_nb_bits {
        n -= 1;
    }
    debug_assert!((total_cost as u32 & (base_cost as u32 - 1)) == 0);
    total_cost >>= largest_bits - target_nb_bits;
    debug_assert!(total_cost > 0);

    const NO_SYMBOL: u32 = 0xF0F0_F0F0;
    let mut rank_last = [NO_SYMBOL; HUF_TABLELOG_MAX as usize + 2];
    {
        let mut current_nb_bits = target_nb_bits;
        let mut pos = n;
        while pos >= 0 {
            if hn!(pos).nb_bits as u32 >= current_nb_bits {
                pos -= 1;
                continue;
            }
            current_nb_bits = hn!(pos).nb_bits as u32; // < targetNbBits
            rank_last[(target_nb_bits - current_nb_bits) as usize] = pos as u32;
            pos -= 1;
        }
    }

    while total_cost > 0 {
        let mut nb_bits_to_decrease = highbit32(total_cost as u32) + 1;
        while nb_bits_to_decrease > 1 {
            let high_pos = rank_last[nb_bits_to_decrease as usize];
            let low_pos = rank_last[nb_bits_to_decrease as usize - 1];
            if high_pos == NO_SYMBOL {
                nb_bits_to_decrease -= 1;
                continue;
            }
            if low_pos == NO_SYMBOL {
                break;
            }
            let high_total = hn!(high_pos as isize).count;
            let low_total = 2 * hn!(low_pos as isize).count;
            if high_total <= low_total {
                break;
            }
            nb_bits_to_decrease -= 1;
        }
        debug_assert!(
            rank_last[nb_bits_to_decrease as usize] != NO_SYMBOL || nb_bits_to_decrease == 1
        );
        while nb_bits_to_decrease <= HUF_TABLELOG_MAX
            && rank_last[nb_bits_to_decrease as usize] == NO_SYMBOL
        {
            nb_bits_to_decrease += 1;
        }
        debug_assert!(rank_last[nb_bits_to_decrease as usize] != NO_SYMBOL);
        total_cost -= 1 << (nb_bits_to_decrease - 1);
        let idx = rank_last[nb_bits_to_decrease as usize];
        hn!(idx as isize).nb_bits += 1;
        if rank_last[nb_bits_to_decrease as usize - 1] == NO_SYMBOL {
            rank_last[nb_bits_to_decrease as usize - 1] = idx;
        }
        if idx == 0 {
            // special case, reached largest symbol
            rank_last[nb_bits_to_decrease as usize] = NO_SYMBOL;
        } else {
            let prev = idx - 1;
            rank_last[nb_bits_to_decrease as usize] = prev;
            if hn!(prev as isize).nb_bits as u32 != target_nb_bits - nb_bits_to_decrease {
                rank_last[nb_bits_to_decrease as usize] = NO_SYMBOL; // this rank is now empty
            }
        }
    }

    while total_cost < 0 {
        // Sometimes, cost correction overshoot
        if rank_last[1] == NO_SYMBOL {
            while hn!(n).nb_bits as u32 == target_nb_bits {
                n -= 1;
            }
            hn!(n + 1).nb_bits -= 1;
            debug_assert!(n >= 0);
            rank_last[1] = (n + 1) as u32;
            total_cost += 1;
            continue;
        }
        hn!(rank_last[1] as isize + 1).nb_bits -= 1;
        rank_last[1] += 1;
        total_cost += 1;
    }
    target_nb_bits
}

/// `HUF_buildCTableFromTree`.
fn huf_build_ctable_from_tree(
    nodes: &[NodeElt; HUFF_NODE_TABLE_SIZE],
    non_null_rank: usize,
    max_symbol: usize,
    max_nb_bits: u32,
) -> HufTable {
    let huff_node = &nodes[1..];
    let mut nb_per_rank = [0u16; HUF_TABLELOG_MAX as usize + 1];
    let mut val_per_rank = [0u16; HUF_TABLELOG_MAX as usize + 1];
    let alphabet_size = max_symbol + 1;
    for node in &huff_node[..=non_null_rank] {
        nb_per_rank[node.nb_bits as usize] += 1;
    }
    {
        let mut min = 0u16;
        for n in (1..=max_nb_bits as usize).rev() {
            val_per_rank[n] = min; // get starting value within each rank
            min += nb_per_rank[n];
            min >>= 1;
        }
    }
    let mut elts = [0u64; 256];
    for node in &huff_node[..alphabet_size] {
        // push nbBits per symbol, symbol order
        elts[node.byte as usize] = node.nb_bits as u64;
    }
    for elt in elts.iter_mut().take(alphabet_size) {
        // assign value within rank, symbol order
        let nb_bits = (*elt & 0xFF) as usize;
        let value = val_per_rank[nb_bits];
        val_per_rank[nb_bits] += 1;
        if nb_bits > 0 {
            debug_assert!((value as u64 >> nb_bits) == 0);
            *elt |= (value as u64) << (64 - nb_bits);
        }
    }
    HufTable {
        elts,
        table_log: max_nb_bits as u8,
        max_symbol: max_symbol as u8,
    }
}

/// `HUF_buildCTable_wksp`: build a length-limited canonical Huffman table
/// for `count[..=max_symbol]`. Returns `None` for the C `GENERIC` error when
/// the depth cannot be limited to `HUF_TABLELOG_MAX`.
pub fn build_ctable(count: &[u32], max_symbol: usize, max_nb_bits: u32) -> Option<HufTable> {
    let max_nb_bits = if max_nb_bits == 0 {
        HUF_TABLELOG_DEFAULT
    } else {
        max_nb_bits
    };
    debug_assert!(max_symbol <= HUF_SYMBOLVALUE_MAX);
    let mut nodes = [NodeElt::default(); HUFF_NODE_TABLE_SIZE];
    huf_sort(&mut nodes, count, max_symbol);
    let non_null_rank = huf_build_tree(&mut nodes, max_symbol);
    let max_nb_bits = huf_set_max_height(&mut nodes, non_null_rank, max_nb_bits);
    if max_nb_bits > HUF_TABLELOG_MAX {
        return None;
    }
    Some(huf_build_ctable_from_tree(
        &nodes,
        non_null_rank,
        max_symbol,
        max_nb_bits,
    ))
}

/// `HUF_estimateCompressedSize`: bytes `count[..=max_symbol]` would take
/// under `table`, header excluded.
pub fn estimate_compressed_size(table: &HufTable, count: &[u32], max_symbol: usize) -> usize {
    let mut nb_bits = 0usize;
    for (s, &c) in count[..=max_symbol].iter().enumerate() {
        nb_bits += table.nb_bits(s) as usize * c as usize;
    }
    nb_bits >> 3
}

/// `HUF_validateCTable`: does `table` have a code for every symbol present
/// in `count[..=max_symbol]`?
pub fn validate_ctable(table: &HufTable, count: &[u32], max_symbol: usize) -> bool {
    if (table.max_symbol as usize) < max_symbol {
        return false;
    }
    let mut bad = false;
    for (s, &c) in count[..=max_symbol].iter().enumerate() {
        bad |= (c != 0) & (table.nb_bits(s) == 0);
    }
    !bad
}

// =========================================================================
// HUF_writeCTable_wksp / HUF_compressWeights
// =========================================================================

/// `HUF_compressWeights`: FSE-compress `weights` (values `0..=12`) and append
/// the result to `out`. Returns the byte size, `Some(0)` when not
/// compressible, `Some(1)` when every weight is equal, or `None` for the C
/// error cases.
pub fn compress_weights(out: &mut Vec<u8>, weights: &[u8]) -> Option<usize> {
    let wt_size = weights.len();
    if wt_size <= 1 {
        return Some(0); // Not compressible
    }
    let mut count = [0u32; HUF_TABLELOG_MAX as usize + 1];
    let mut max_symbol = 0usize;
    for &w in weights {
        count[w as usize] += 1;
        max_symbol = max_symbol.max(w as usize);
    }
    let max_count = *count[..=max_symbol].iter().max().unwrap() as usize;
    if max_count == wt_size {
        return Some(1); // only a single symbol in src : rle
    }
    if max_count == 1 {
        return Some(0); // each symbol present maximum once => not compressible
    }
    let table_log = fse::optimal_table_log(MAX_FSE_TABLELOG_FOR_HUFF_HEADER, wt_size, max_symbol);
    let mut norm = [0i16; HUF_TABLELOG_MAX as usize + 1];
    fse::normalize_count(&mut norm, table_log, &count, wt_size, max_symbol, false).ok()?;
    let start = out.len();
    fse::write_ncount(out, &norm, max_symbol, table_log).ok()?;
    let ct = fse::FseCTable::build(&norm, max_symbol, table_log);
    let c_size = fse::compress_using_ctable(out, weights, &ct);
    if c_size == 0 {
        out.truncate(start);
        return Some(0);
    }
    Some(out.len() - start)
}

/// `HUF_writeCTable_wksp`: append the Huffman tree description for
/// `table` (symbols `0..max_symbol`, depth `huff_log`) to `out` and return
/// its byte size. `None` for the C error cases.
pub fn write_ctable(
    out: &mut Vec<u8>,
    table: &HufTable,
    max_symbol: usize,
    huff_log: u32,
) -> Option<usize> {
    debug_assert_eq!(table.max_symbol as usize, max_symbol);
    debug_assert_eq!(table.table_log as u32, huff_log);
    if max_symbol > HUF_SYMBOLVALUE_MAX {
        return None;
    }
    // bitsToWeight[0] = 0; bitsToWeight[n] = huffLog + 1 - n
    let mut huff_weight = [0u8; HUF_SYMBOLVALUE_MAX + 1];
    for (n, w) in huff_weight[..max_symbol].iter_mut().enumerate() {
        let nb = table.nb_bits(n);
        *w = if nb == 0 {
            0
        } else {
            (huff_log + 1 - nb) as u8
        };
    }
    let start = out.len();
    out.push(0);
    let h_size = compress_weights(out, &huff_weight[..max_symbol])?;
    if h_size > 1 && h_size < max_symbol / 2 {
        // FSE compressed
        out[start] = h_size as u8;
        return Some(h_size + 1);
    }
    out.truncate(start + 1);
    if max_symbol > 256 - 128 {
        // should not happen : likely means source cannot be compressed
        out.truncate(start);
        return None;
    }
    out[start] = (128 + (max_symbol - 1)) as u8;
    huff_weight[max_symbol] = 0;
    for n in (0..max_symbol).step_by(2) {
        out.push((huff_weight[n] << 4) + huff_weight[n + 1]);
    }
    Some(max_symbol.div_ceil(2) + 1)
}

// =========================================================================
// HUF_compress1X_usingCTable / HUF_compress4X_usingCTable
// =========================================================================

/// `HUF_CStream_t`: two 64-bit containers so that the encoder can fill
/// them independently (`idx` 0 and 1) and merge, writing whole 8-byte words
/// into `buf` at `ptr`. Bits are packed at the top of a container and land
/// in the output least-significant first, i.e. a backward bitstream.
struct HufCStream<'a> {
    bit_container: [u64; 2],
    bit_pos: [u32; 2],
    buf: &'a mut [u8],
    ptr: usize,
}

impl HufCStream<'_> {
    /// `HUF_addBits` with the C `kFast == 0` masking (`HUF_getValue`).
    #[inline(always)]
    fn add_bits(&mut self, elt: u64, idx: usize) {
        let nb_bits = (elt & 0xFF) as u32;
        self.bit_container[idx] >>= nb_bits;
        self.bit_container[idx] |= elt & !0xFF;
        self.bit_pos[idx] += nb_bits;
    }

    /// `HUF_zeroIndex1`.
    #[inline(always)]
    fn zero_index1(&mut self) {
        self.bit_container[1] = 0;
        self.bit_pos[1] = 0;
    }

    /// `HUF_mergeIndex1`.
    #[inline(always)]
    fn merge_index1(&mut self) {
        debug_assert!(self.bit_pos[1] < 64);
        self.bit_container[0] >>= self.bit_pos[1];
        self.bit_container[0] |= self.bit_container[1];
        self.bit_pos[0] += self.bit_pos[1];
    }

    /// `HUF_flushBits`: store the container's top `bit_pos` bits as one
    /// little-endian word and advance by the whole bytes among them.
    #[inline(always)]
    fn flush_bits(&mut self) {
        let nb_bits = self.bit_pos[0];
        debug_assert!(nb_bits > 0 && nb_bits <= 64);
        let nb_bytes = (nb_bits >> 3) as usize;
        let word = self.bit_container[0] >> (64 - nb_bits);
        self.bit_pos[0] &= 7;
        self.buf[self.ptr..self.ptr + 8].copy_from_slice(&word.to_le_bytes());
        self.ptr += nb_bytes;
    }

    /// `HUF_closeCStream`: end mark, final flush, byte size.
    fn close(mut self) -> usize {
        const END_MARK: u64 = (1 << 63) | 1; // HUF_endMark: value 1, 1 bit
        self.add_bits(END_MARK, 0);
        self.flush_bits();
        self.ptr + (self.bit_pos[0] > 0) as usize
    }
}

/// `HUF_compress1X_usingCTable_internal_body_loop`: `K_UNROLL` symbols per
/// container, two containers per outer iteration.
#[inline(always)]
fn compress_1x_loop<const K_UNROLL: usize>(bit_c: &mut HufCStream, ip: &[u8], ct: &[u64; 256]) {
    let mut n = ip.len();
    let rem = n % K_UNROLL;
    if rem > 0 {
        for _ in 0..rem {
            n -= 1;
            bit_c.add_bits(ct[ip[n] as usize], 0);
        }
        bit_c.flush_bits();
    }
    debug_assert_eq!(n % K_UNROLL, 0);
    if !n.is_multiple_of(2 * K_UNROLL) {
        for u in 1..K_UNROLL {
            bit_c.add_bits(ct[ip[n - u] as usize], 0);
        }
        bit_c.add_bits(ct[ip[n - K_UNROLL] as usize], 0);
        bit_c.flush_bits();
        n -= K_UNROLL;
    }
    debug_assert_eq!(n % (2 * K_UNROLL), 0);
    while n > 0 {
        for u in 1..K_UNROLL {
            bit_c.add_bits(ct[ip[n - u] as usize], 0);
        }
        bit_c.add_bits(ct[ip[n - K_UNROLL] as usize], 0);
        bit_c.flush_bits();
        bit_c.zero_index1();
        for u in 1..K_UNROLL {
            bit_c.add_bits(ct[ip[n - K_UNROLL - u] as usize], 1);
        }
        bit_c.add_bits(ct[ip[n - 2 * K_UNROLL] as usize], 1);
        bit_c.merge_index1();
        bit_c.flush_bits();
        n -= 2 * K_UNROLL;
    }
    debug_assert_eq!(n, 0);
}

/// `HUF_tightCompressBound`: every symbol takes at most `table_log` bits;
/// the 8 spare bytes absorb the last whole-word store.
fn tight_compress_bound(src_size: usize, table_log: usize) -> usize {
    ((src_size * table_log) >> 3) + 8
}

/// `HUF_compress1X_usingCTable_internal`: one backward Huffman bitstream for
/// `src`, appended to `out`. Returns its byte size (never 0 here: the C
/// early-outs are all output-capacity checks and `out` is unbounded).
pub fn compress_1x_using_ctable(out: &mut Vec<u8>, src: &[u8], table: &HufTable) -> usize {
    let table_log = table.table_log as usize;
    let start = out.len();
    out.resize(start + tight_compress_bound(src.len(), table_log), 0);
    let mut bit_c = HufCStream {
        bit_container: [0; 2],
        bit_pos: [0; 2],
        buf: &mut out[start..],
        ptr: 0,
    };
    // kUnroll / kLastFast per tableLog as in the 64-bit C switch; every
    // add here uses the masked (non-fast) form, which yields the same bits.
    match table_log {
        11 | 10 => compress_1x_loop::<5>(&mut bit_c, src, &table.elts),
        9 => compress_1x_loop::<6>(&mut bit_c, src, &table.elts),
        8 => compress_1x_loop::<7>(&mut bit_c, src, &table.elts),
        7 => compress_1x_loop::<8>(&mut bit_c, src, &table.elts),
        _ => compress_1x_loop::<9>(&mut bit_c, src, &table.elts),
    }
    let size = bit_c.close();
    out.truncate(start + size);
    size
}

/// `HUF_compress4X_usingCTable_internal`: jump table plus four streams.
/// Returns the byte size, or 0 when `src` is too small or a stream does not
/// fit the 16-bit jump table (`out` is restored).
pub fn compress_4x_using_ctable(out: &mut Vec<u8>, src: &[u8], table: &HufTable) -> usize {
    let src_size = src.len();
    let segment_size = src_size.div_ceil(4); // first 3 segments
    if src_size < 12 {
        return 0; // no saving possible : too small input
    }
    let start = out.len();
    out.extend_from_slice(&[0u8; 6]); // jumpTable
    let mut ip = 0usize;
    for i in 0..4 {
        let end = if i < 3 { ip + segment_size } else { src_size };
        let c_size = compress_1x_using_ctable(out, &src[ip..end], table);
        if c_size == 0 || c_size > 65535 {
            out.truncate(start);
            return 0;
        }
        if i < 3 {
            out[start + 2 * i..start + 2 * i + 2].copy_from_slice(&(c_size as u16).to_le_bytes());
        }
        ip = end;
    }
    out.len() - start
}

/// `HUF_compressCTable_internal`: encode `src` after whatever header is
/// already in `out[ostart..]`; returns the total size from `ostart`, or 0
/// when that would not beat `src.len() - 1` (`out` is restored to `ostart`).
fn compress_ctable_internal(
    out: &mut Vec<u8>,
    ostart: usize,
    src: &[u8],
    single_stream: bool,
    table: &HufTable,
) -> usize {
    let c_size = if single_stream {
        compress_1x_using_ctable(out, src, table)
    } else {
        compress_4x_using_ctable(out, src, table)
    };
    if c_size == 0 || out.len() - ostart >= src.len() - 1 {
        out.truncate(ostart);
        return 0;
    }
    out.len() - ostart
}

// =========================================================================
// HUF_compress_internal
// =========================================================================

const SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE: usize = 4096;
const SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO: usize = 10;

/// `HUF_compress_internal` for `HUF_compress1X_repeat` /
/// `HUF_compress4X_repeat` with `maxSymbolValue = 255` and
/// `huffLog = LitHufLog`. Appends the tree description (if any) and the
/// streams to `out`; `table`/`repeat` are the C `oldHufTable`/`repeat`
/// in-out parameters. Returns `Some(0)` for "not compressible", `Some(1)`
/// for a single-symbol input (nothing appended), `None` for a C error.
fn compress_internal(
    out: &mut Vec<u8>,
    src: &[u8],
    single_stream: bool,
    table: &mut Option<HufTable>,
    repeat: &mut HufRepeat,
    prefer_repeat: bool,
    suspect_uncompressible: bool,
) -> Option<usize> {
    let src_size = src.len();
    let ostart = out.len();
    if src_size == 0 {
        return Some(0); // Uncompressed
    }
    if src_size > ZSTD_BLOCKSIZE_MAX {
        return None;
    }
    let huff_log = HUF_TABLELOG_DEFAULT;

    if prefer_repeat && *repeat == HufRepeat::Valid {
        let old = table.as_ref()?;
        return Some(compress_ctable_internal(
            out,
            ostart,
            src,
            single_stream,
            old,
        ));
    }

    let mut count = [0u32; 256];
    if suspect_uncompressible
        && src_size >= SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE * SUSPECT_INCOMPRESSIBLE_SAMPLE_RATIO
    {
        let (largest_begin, _) = hist_count(&mut count, &src[..SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE]);
        let (largest_end, _) = hist_count(
            &mut count,
            &src[src_size - SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE..],
        );
        let largest_total = largest_begin as usize + largest_end as usize;
        if largest_total <= ((2 * SUSPECT_INCOMPRESSIBLE_SAMPLE_SIZE) >> 7) + 4 {
            return Some(0); // heuristic : probably not compressible enough
        }
    }

    let (largest, max_symbol_value) = hist_count(&mut count, src);
    if largest as usize == src_size {
        return Some(1); // single symbol, rle
    }
    if largest as usize <= (src_size >> 7) + 4 {
        return Some(0); // heuristic : probably not compressible enough
    }

    if *repeat == HufRepeat::Check
        && !table
            .as_ref()
            .is_some_and(|t| validate_ctable(t, &count, max_symbol_value))
    {
        *repeat = HufRepeat::None;
    }
    if prefer_repeat && *repeat != HufRepeat::None {
        let old = table.as_ref()?;
        return Some(compress_ctable_internal(
            out,
            ostart,
            src,
            single_stream,
            old,
        ));
    }

    // HUF_optimalTableLog without HUF_flags_optimalDepth
    let huff_log = fse::optimal_table_log_internal(huff_log, src_size, max_symbol_value, 1);
    let new_table = build_ctable(&count, max_symbol_value, huff_log)?;
    let huff_log = new_table.table_log as u32;

    let h_size = write_ctable(out, &new_table, max_symbol_value, huff_log)?;
    if *repeat != HufRepeat::None {
        let old = table.as_ref()?;
        let old_size = estimate_compressed_size(old, &count, max_symbol_value);
        let new_size = estimate_compressed_size(&new_table, &count, max_symbol_value);
        if old_size <= h_size + new_size || h_size + 12 >= src_size {
            out.truncate(ostart);
            return Some(compress_ctable_internal(
                out,
                ostart,
                src,
                single_stream,
                old,
            ));
        }
    }
    if h_size + 12 >= src_size {
        out.truncate(ostart);
        return Some(0);
    }
    *repeat = HufRepeat::None;
    *table = Some(new_table);
    let new = table.as_ref().unwrap();
    Some(compress_ctable_internal(
        out,
        ostart,
        src,
        single_stream,
        new,
    ))
}

// =========================================================================
// ZSTD_compressLiterals
// =========================================================================

/// `ZSTD_noCompressLiterals`: Raw_Literals_Block.
pub fn encode_literals_raw(out: &mut Vec<u8>, literals: &[u8]) {
    let size = literals.len();
    if size <= 31 {
        out.push(LIT_TYPE_RAW | ((size as u8) << 3));
    } else if size <= 4095 {
        let h = (LIT_TYPE_RAW as u16) | (1 << 2) | ((size as u16) << 4);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        let h = (LIT_TYPE_RAW as u32) | (3 << 2) | ((size as u32) << 4);
        out.extend_from_slice(&h.to_le_bytes()[..3]);
    }
    out.extend_from_slice(literals);
}

/// `ZSTD_compressRleLiteralsBlock`: RLE_Literals_Block.
pub fn encode_literals_rle(out: &mut Vec<u8>, byte: u8, size: usize) {
    if size <= 31 {
        out.push(LIT_TYPE_RLE | ((size as u8) << 3));
    } else if size <= 4095 {
        let h = (LIT_TYPE_RLE as u16) | (1 << 2) | ((size as u16) << 4);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        let h = (LIT_TYPE_RLE as u32) | (3 << 2) | ((size as u32) << 4);
        out.extend_from_slice(&h.to_le_bytes()[..3]);
    }
    out.push(byte);
}

/// `ZSTD_minLiteralsToCompress`.
fn min_literals_to_compress(strategy: Strategy, repeat: HufRepeat) -> usize {
    let shift = (9 - strategy as i32).min(3);
    if repeat == HufRepeat::Valid {
        6
    } else {
        8usize << shift
    }
}

/// `SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO`.
const SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO: usize = 20;

/// `ZSTD_literalsCompressionIsDisabled` with `literalCompressionMode ==
/// ZSTD_ps_auto`: negative levels (`target_length > 0` on `ZSTD_fast`)
/// skip literal compression.
fn literals_compression_is_disabled(cparams: &CParams) -> bool {
    cparams.strategy == Strategy::Fast && cparams.target_length > 0
}

/// `ZSTD_compressLiterals`: write the literals section for one block and
/// return the Huffman state the decoder holds afterwards (`nextHuf`).
/// `nb_seq` is the block's sequence count, from which the caller-side
/// `suspectUncompressible` flag is derived as in
/// `ZSTD_entropyCompressSeqStore_internal`; `cparams` supplies the
/// strategy and the `disableLiteralCompression` decision.
pub fn compress_literals_with(
    out: &mut Vec<u8>,
    literals: &[u8],
    nb_seq: usize,
    prev: &HufState,
    cparams: &CParams,
) -> HufState {
    let strategy = cparams.strategy;
    let src_size = literals.len();
    let lh_size = 3 + (src_size >= 1024) as usize + (src_size >= 16384) as usize;
    let mut single_stream = src_size < 256;
    let mut h_type = LIT_TYPE_COMPRESSED;

    if literals_compression_is_disabled(cparams) {
        encode_literals_raw(out, literals);
        return prev.clone();
    }

    // if too small, don't even attempt compression (speed opt)
    if src_size < min_literals_to_compress(strategy, prev.repeat()) {
        encode_literals_raw(out, literals);
        return prev.clone();
    }

    let suspect_uncompressible =
        nb_seq == 0 || src_size / nb_seq >= SUSPECT_UNCOMPRESSIBLE_LITERAL_RATIO;
    let prefer_repeat = strategy < Strategy::Lazy && src_size <= 1024;
    let (mut table, mut repeat) = match prev {
        HufState::None => (None, HufRepeat::None),
        HufState::Check(t) => (Some(t.clone()), HufRepeat::Check),
        HufState::Valid(t) => (Some(t.clone()), HufRepeat::Valid),
    };
    if repeat == HufRepeat::Valid && lh_size == 3 {
        single_stream = true;
    }
    let ostart = out.len();
    out.resize(ostart + lh_size, 0);
    let c_lit_size = compress_internal(
        out,
        literals,
        single_stream,
        &mut table,
        &mut repeat,
        prefer_repeat,
        suspect_uncompressible,
    );
    if repeat != HufRepeat::None {
        // reused the existing table
        h_type = LIT_TYPE_TREELESS;
    }

    let min_gain = CParams::min_gain(src_size, strategy);
    let c_lit_size = match c_lit_size {
        Some(c) if c != 0 && c < src_size - min_gain => c,
        _ => {
            out.truncate(ostart);
            encode_literals_raw(out, literals);
            return prev.clone();
        }
    };
    if c_lit_size == 1 && (src_size >= 8 || literals.iter().all(|&b| b == literals[0])) {
        out.truncate(ostart);
        encode_literals_rle(out, literals[0], src_size);
        return prev.clone();
    }
    debug_assert_eq!(out.len(), ostart + lh_size + c_lit_size);

    let next = if h_type == LIT_TYPE_COMPRESSED {
        // using a newly constructed table
        HufState::Check(table.expect("a new table was written"))
    } else {
        prev.clone()
    };

    // Build header
    let h_type = h_type as u32;
    match lh_size {
        3 => {
            // 2 - 2 - 10 - 10
            let lhc = h_type
                | ((!single_stream as u32) << 2)
                | ((src_size as u32) << 4)
                | ((c_lit_size as u32) << 14);
            out[ostart..ostart + 3].copy_from_slice(&lhc.to_le_bytes()[..3]);
        }
        4 => {
            // 2 - 2 - 14 - 14
            let lhc = h_type | (2 << 2) | ((src_size as u32) << 4) | ((c_lit_size as u32) << 18);
            out[ostart..ostart + 4].copy_from_slice(&lhc.to_le_bytes());
        }
        _ => {
            // 2 - 2 - 18 - 18
            let lhc = h_type | (3 << 2) | ((src_size as u32) << 4) | ((c_lit_size as u32) << 22);
            out[ostart..ostart + 4].copy_from_slice(&lhc.to_le_bytes());
            out[ostart + 4] = (c_lit_size >> 10) as u8;
        }
    }
    next
}
