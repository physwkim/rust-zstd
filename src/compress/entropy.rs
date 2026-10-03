//! The entropy tables the decoder holds across blocks
//! (`ZSTD_entropyCTables_t`).

/// `HUF_repeat` / `FSE_repeat`: whether the next block may reference the
/// table the decoder holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Repeat {
    /// `*_repeat_none`: no table can be referenced.
    #[default]
    None,
    /// `*_repeat_check`: a table the next block may reference after
    /// checking that it covers its symbols.
    Check,
    /// `*_repeat_valid`: usable without checks (dictionaries only).
    Valid,
}
