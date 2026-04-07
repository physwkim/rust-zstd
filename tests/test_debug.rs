#[test]
fn debug_f64_sequences() {
    let data: Vec<u8> = (0..131072u64).flat_map(|i| (i as f64).to_le_bytes()).collect();
    let c = zstd_rs::compress(&data, 1);
    eprintln!("f64 level1: {} -> {}", data.len(), c.len());
    // Verify roundtrip
    let d = zstd_rs::decompress(&c).unwrap();
    assert_eq!(d.len(), data.len());
    assert_eq!(d, data);
}
