//! Regression test for issue #1: 0.1.0's encoder emitted frames its own
//! decoder and libzstd rejected, and panicked on one input (FSE normalized
//! counts did not sum to 1 << table_log). The three inputs below are the
//! reporter's, verbatim from the issue body.

/// 325 bytes over the alphabet {a, b, 0, space}.
///
/// `rust_zstd::compress` returns a frame that the reference decoder rejects.
pub const CORRUPT: &str = "\
abbbabbbbabb0ababbbbbaaaaaaabbbbbabababaabbbabbaabababbaaababbabbbabbbaabaab\
abaa0babbabbaabbbbabaaaaaabaabaabababbbbaaaa0bababbabababbaaaaaab0aaaaaaabbb\
abbbbaaabbaaaaababbbbabbabaabbbabaaa0aaaaabaababbabbbaabbab aaaaababbbaaaabb\
bbaabbabbbabababbabaaabbabbabaaab0aaabaaabaaab0abbabbbaabbaabaa0aaaabab0aaab\
abaaaaabaabbbbbabbbba";

/// 724 bytes over the alphabet {a..e, 0, space}.
///
/// `rust_zstd::compress` panics with an index-out-of-bounds inside FSE table
/// construction.
pub const PANIC: &str = "\
cbacecaeecaeaba0cbeadeaeaadcaedeedeedaedbebaaabdcdbdaecdbbbebdbeeacdedbdaaa0\
dadeae0aa0abbacdaeeeaa0bbcdeaacbacaaa0aa00aa0 eaedb0bbacddaaeabaaa aadaaebdc\
ddbeba0aaccadeeacdcdbbbcddaaaa0dcacecbbacdaa00cebaabeca0a0eeeacebaaa0aaaaead\
daa0 aaaa 0a0adcadadadeaaadabeabbebdabbedaeeaeccaab0bdcdddcbebaceccddaaeebbd\
daacaccdcbeb0a00a0a dcaceeaa a0acbeadbbcde0accbae00aadbdaeabdcda aaabbbdacbc\
dbccdebeaaaa0aeaddccbccecaceedabeacddaa0aaebacdabbbdaaecbbaaeecaaedbebbaabed\
edbdcbacaebacd0aa0edcaddbbaacdeeacaaeebaa00acbacccbaeabaaaabaa0ba0abccdeaa0e\
daccaabdacbceeaecdcaddaaaa0cbac0eeeaaaaeacaacbeaeaadcebbddba0adaecbcdbacdebe\
ededecaccdadaecaaaaaabeeaca0aa  aedcadcbacdaeeacaccdcddbecaebaededeeaedbacca\
daacdcadaabecaaccaaaaa a a000aeeea abcaa";

/// 482 bytes over the alphabet {a, b, 0}.
///
/// Same failure as [`CORRUPT`] but at level 3, to show the bug is not confined to
/// the fastest level.
pub const CORRUPT_L3: &str = "\
babbababbabbbaaabbbaaababbababbbbaababaaaabaaaabbaabbbaaaaabbaabababaababbbb\
aabaaabaaaaabbaab0ababbbaaaaaaaaabaabaabbaabaaabbbbaaabaaaabbabbbabababaaaba\
baabaaaaaabaaabbaabbbabbaaaba0aabbabbbabbaabbbbaaaabaabbaabababaaaabaaababba\
aaab0aaaaaabaabaaababaaaa0babbabaaabaaabbabbaaabbaaaaaaaabbaa0babbabaaaaaaaa\
bbababbaaababbaabbbbaabbbaababaabbbbbaaaaaabababaaab0aa0aabbbabbaabbabbbbaaa\
abaaaaabbbababbbbaba0aaabbbaa0aabaabaaaaabbaaababbabbbbbbbabbabb0bbaababbaab\
aabbaabbbbbabbbbbbbaaaaaaa";

fn round_trips(name: &str, input: &str) {
    for level in [1, 3, 9, 19] {
        let frame = rust_zstd::compress(input.as_bytes(), level);
        let ours = rust_zstd::decompress(&frame)
            .unwrap_or_else(|e| panic!("{name} L{level}: our decoder rejected our frame: {e}"));
        assert_eq!(ours, input.as_bytes(), "{name} L{level}: bytes differ");
        let lib = zstd::stream::decode_all(&frame[..])
            .unwrap_or_else(|e| panic!("{name} L{level}: libzstd rejected our frame: {e}"));
        assert_eq!(
            lib,
            input.as_bytes(),
            "{name} L{level}: libzstd bytes differ"
        );
    }
}

#[test]
fn issue_1_inputs_round_trip() {
    round_trips("CORRUPT", CORRUPT);
    round_trips("PANIC", PANIC);
    round_trips("CORRUPT_L3", CORRUPT_L3);
}
