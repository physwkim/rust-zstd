mod common;

use common::{datasets, MIB};
use zstd::zstd_safe::zstd_sys as sys;

#[test]
fn compression_ratios_by_level() {
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("zeros_64K", vec![0u8; 65536]),
        (
            "text_50K",
            b"The quick brown fox jumps over the lazy dog. ".repeat(1100),
        ),
        (
            "seq_f64_32K",
            (0..4096u64)
                .flat_map(|i| (i as f64 * 0.5).to_le_bytes())
                .collect(),
        ),
    ];

    for (name, data) in &patterns {
        eprintln!("\n{} ({} bytes):", name, data.len());
        for level in [0, 1, 3, 7, 11, 13, 14, 15] {
            let compressed = rust_zstd::compress(data, level);
            let decompressed = rust_zstd::decompress(&compressed)
                .unwrap_or_else(|e| panic!("{} level {} decompress: {}", name, level, e));
            assert_eq!(decompressed.len(), data.len(), "{} level {}", name, level);
            assert_eq!(&decompressed, data, "{} level {}", name, level);
            let ratio = data.len() as f64 / compressed.len() as f64;
            eprintln!(
                "  level {:2}: {:>8} -> {:>8}  ({:.2}x)",
                level,
                data.len(),
                compressed.len(),
                ratio
            );
        }
    }
}

/// libzstd single-threaded frame at `level` with both block splitters off
/// (`ZSTD_c_blockSplitterLevel` = 1, `ZSTD_c_splitAfterSequences` =
/// `ZSTD_ps_disable`): this crate has neither, so with them off libzstd
/// compresses the same 128 KiB blocks.
fn zstd_nosplit(data: &[u8], level: i32) -> Vec<u8> {
    unsafe {
        let cctx = sys::ZSTD_createCCtx();
        let set = |p, v| {
            let r = sys::ZSTD_CCtx_setParameter(cctx, p, v);
            assert_eq!(sys::ZSTD_isError(r), 0, "set parameter {:?}", p);
        };
        set(sys::ZSTD_cParameter::ZSTD_c_compressionLevel, level);
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam20, 1);
        set(sys::ZSTD_cParameter::ZSTD_c_experimentalParam13, 2);
        let mut out = vec![0u8; sys::ZSTD_compressBound(data.len())];
        let n = sys::ZSTD_compress2(
            cctx,
            out.as_mut_ptr().cast(),
            out.len(),
            data.as_ptr().cast(),
            data.len(),
        );
        assert_eq!(sys::ZSTD_isError(n), 0);
        sys::ZSTD_freeCCtx(cctx);
        out.truncate(n);
        out
    }
}

/// Levels 13-15 on inputs above 256 KiB run btlazy2 in both codecs, so with
/// libzstd's splitters off the sequences and therefore the frame sizes are
/// the same: no tolerance. With its splitters on libzstd is up to 1.3%
/// smaller on these inputs (none on words); that gap is printed, not
/// asserted, because it belongs to the splitters this crate lacks.
#[test]
fn btlazy2_sizes_equal_libzstd_without_splitters() {
    for ds in datasets() {
        if !matches!(ds.name, "rust_src_8m" | "elf_8m" | "words_1m") {
            continue;
        }
        let data = &ds.data[..ds.data.len().min(MIB)];
        for level in [13, 14, 15] {
            let ours = rust_zstd::compress(data, level);
            assert_eq!(
                rust_zstd::decompress(&ours).as_deref(),
                Ok(data),
                "{} level {level}",
                ds.name
            );
            let c = zstd_nosplit(data, level);
            let c_default = zstd::bulk::compress(data, level).unwrap();
            eprintln!(
                "{} level {level}: ours {} libzstd no-split {} default {}",
                ds.name,
                ours.len(),
                c.len(),
                c_default.len()
            );
            assert_eq!(ours.len(), c.len(), "{} level {level}", ds.name);
        }
    }
}
