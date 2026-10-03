//! The entropy tables the decoder holds across blocks
//! (`ZSTD_entropyCTables_t`), double-buffered per table.
//!
//! libzstd codes a block into `nextCBlock`, copying into it each table the
//! block keeps from `prevCBlock`, and swaps the two when the block is
//! written COMPRESSED. Here each table has two slots instead ([`Slots`]):
//! the committed one, which the decoder holds, and a spare one. A coder
//! reads the committed table and builds a new one in the spare through a
//! [`TableRef`], and returns a [`Next`] saying what the block does to the
//! table; the owner applies it ([`Slots::commit`]) if the block is written
//! COMPRESSED and drops it otherwise. Applying is a mode change and at
//! most a slot flip: no table is ever copied from block to block.

use crate::fse::FseCTable;
use crate::huf::HufTable;

/// `HUF_repeat` / `FSE_repeat`: whether the next block may reference the
/// table the decoder holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
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

// SAFETY: `None` is 0.
unsafe impl bytemuck::Zeroable for Repeat {}

/// The table the decoder holds, with its [`Repeat`] mode: there is a table
/// exactly when one may be referenced.
#[derive(Debug)]
pub enum Held<'a, T> {
    None,
    Check(&'a T),
    Valid(&'a T),
}

impl<T> Clone for Held<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Held<'_, T> {}

impl<'a, T> Held<'a, T> {
    pub fn table(self) -> Option<&'a T> {
        match self {
            Held::None => None,
            Held::Check(t) | Held::Valid(t) => Some(t),
        }
    }

    pub fn repeat(self) -> Repeat {
        match self {
            Held::None => Repeat::None,
            Held::Check(_) => Repeat::Check,
            Held::Valid(_) => Repeat::Valid,
        }
    }
}

/// A coder's access to one table: the table the decoder holds, and the
/// spare slot a table the block describes or encodes with is built in.
pub struct TableRef<'a, T> {
    pub held: Held<'a, T>,
    pub spare: &'a mut T,
}

/// What a coded section does to one table once its block is committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// The decoder keeps its table and mode: the section repeated the
    /// table (`set_repeat`, treeless literals) or did not touch it (raw
    /// or RLE literals, no sequences).
    Keep,
    /// No table can be referenced any more (`set_basic`, `set_rle`).
    None,
    /// The section described a new table, the one in the spare slot,
    /// which the next block may reference after checking it
    /// (`set_compressed`).
    New,
}

/// One table, double-buffered: the slot the decoder's table is in and its
/// mode, and a spare slot. What the spare holds is never read through the
/// slots until [`Slots::commit`] makes it the decoder's table.
#[derive(Clone, Debug, Default)]
pub struct Slots<T> {
    slots: [T; 2],
    /// The slot of the decoder's table.
    cur: usize,
    mode: Repeat,
}

// SAFETY: every field is `Zeroable`.
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for Slots<T> {}

impl<T> Slots<T> {
    /// `table` held by the decoder with `mode`, which is not
    /// [`Repeat::None`].
    pub fn holding(table: T, mode: Repeat) -> Self
    where
        T: Default,
    {
        debug_assert!(mode != Repeat::None);
        Self {
            slots: [table, T::default()],
            cur: 0,
            mode,
        }
    }

    /// The table the decoder holds.
    pub fn held(&self) -> Held<'_, T> {
        let table = &self.slots[self.cur];
        match self.mode {
            Repeat::None => Held::None,
            Repeat::Check => Held::Check(table),
            Repeat::Valid => Held::Valid(table),
        }
    }

    /// The table the decoder holds and the spare slot, for a coder.
    pub fn split(&mut self) -> TableRef<'_, T> {
        let [first, second] = &mut self.slots;
        let (table, spare) = if self.cur == 0 {
            (&*first, second)
        } else {
            (&*second, first)
        };
        let held = match self.mode {
            Repeat::None => Held::None,
            Repeat::Check => Held::Check(table),
            Repeat::Valid => Held::Valid(table),
        };
        TableRef { held, spare }
    }

    /// Apply what a committed block did to the table.
    pub fn commit(&mut self, next: Next) {
        match next {
            Next::Keep => {}
            Next::None => self.mode = Repeat::None,
            Next::New => {
                self.cur ^= 1;
                self.mode = Repeat::Check;
            }
        }
    }

    /// A `Valid` table becomes `Check`.
    pub fn demote_valid(&mut self) {
        if self.mode == Repeat::Valid {
            self.mode = Repeat::Check;
        }
    }

    /// No table can be referenced.
    pub fn clear(&mut self) {
        self.mode = Repeat::None;
    }

    /// Hold a copy of the table `from` holds, with its mode.
    pub fn load(&mut self, from: &Slots<T>)
    where
        T: Clone,
    {
        self.mode = from.mode;
        if from.mode != Repeat::None {
            self.slots[self.cur].clone_from(&from.slots[from.cur]);
        }
    }
}

/// `ZSTD_fseCTables_t`: the literal-length, offset and match-length
/// tables.
#[derive(Clone, Debug, Default)]
pub struct FseSlots {
    pub ll: Slots<FseCTable>,
    pub of: Slots<FseCTable>,
    pub ml: Slots<FseCTable>,
}

// SAFETY: every field is `Zeroable`.
unsafe impl bytemuck::Zeroable for FseSlots {}

/// The three sequence tables the decoder holds.
#[derive(Clone, Copy, Debug)]
pub struct FseHeld<'a> {
    pub ll: Held<'a, FseCTable>,
    pub of: Held<'a, FseCTable>,
    pub ml: Held<'a, FseCTable>,
}

/// A sequences section coder's access to the three tables.
pub struct FseTables<'a> {
    pub ll: TableRef<'a, FseCTable>,
    pub of: TableRef<'a, FseCTable>,
    pub ml: TableRef<'a, FseCTable>,
}

/// What a sequences section does to the three tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FseNext {
    pub ll: Next,
    pub of: Next,
    pub ml: Next,
}

impl FseNext {
    /// A section without sequences keeps every table.
    pub const KEEP: Self = Self {
        ll: Next::Keep,
        of: Next::Keep,
        ml: Next::Keep,
    };
}

impl FseSlots {
    pub fn held(&self) -> FseHeld<'_> {
        FseHeld {
            ll: self.ll.held(),
            of: self.of.held(),
            ml: self.ml.held(),
        }
    }

    pub fn split(&mut self) -> FseTables<'_> {
        FseTables {
            ll: self.ll.split(),
            of: self.of.split(),
            ml: self.ml.split(),
        }
    }

    pub fn commit(&mut self, next: FseNext) {
        self.ll.commit(next.ll);
        self.of.commit(next.of);
        self.ml.commit(next.ml);
    }
}

/// `ZSTD_entropyCTables_t`: the Huffman table and the sequence tables.
#[derive(Clone, Debug, Default)]
pub struct EntropyTables {
    pub huf: Slots<HufTable>,
    pub fse: FseSlots,
}

// SAFETY: every field is `Zeroable`.
unsafe impl bytemuck::Zeroable for EntropyTables {}

impl EntropyTables {
    /// No table can be referenced (`ZSTD_reset_compressedBlockState`).
    pub fn clear(&mut self) {
        self.huf.clear();
        self.fse.ll.clear();
        self.fse.of.clear();
        self.fse.ml.clear();
    }

    /// Hold copies of the tables `from` holds.
    pub fn load(&mut self, from: &EntropyTables) {
        self.huf.load(&from.huf);
        self.fse.ll.load(&from.fse.ll);
        self.fse.of.load(&from.fse.of);
        self.fse.ml.load(&from.fse.ml);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per [`Next`], what the decoder holds after a commit: `Keep` leaves
    /// table and mode, `None` drops the mode, `New` makes the spare the
    /// table, `Check`. A spare written for a block that is not committed
    /// is never held.
    #[test]
    fn commit_applies_each_next() {
        let mut slots = Slots::holding(1u32, Repeat::Valid);
        let t = slots.split();
        assert!(matches!(t.held, Held::Valid(&1)));
        *t.spare = 2;
        // the block that built 2 was not committed
        assert!(matches!(slots.held(), Held::Valid(&1)));
        slots.commit(Next::Keep);
        assert!(matches!(slots.held(), Held::Valid(&1)));
        *slots.split().spare = 3;
        slots.commit(Next::New);
        assert!(matches!(slots.held(), Held::Check(&3)));
        *slots.split().spare = 4;
        slots.commit(Next::New);
        assert!(matches!(slots.held(), Held::Check(&4)));
        slots.commit(Next::None);
        assert!(matches!(slots.held(), Held::None));
        assert!(matches!(slots.split().held, Held::None));
    }

    #[test]
    fn demote_valid_only_lowers_valid() {
        let mut slots = Slots::holding(1u32, Repeat::Valid);
        slots.demote_valid();
        assert!(matches!(slots.held(), Held::Check(&1)));
        slots.demote_valid();
        assert!(matches!(slots.held(), Held::Check(&1)));
        slots.clear();
        slots.demote_valid();
        assert!(matches!(slots.held(), Held::None));
    }

    /// `load` copies the held table into the slot the loader holds,
    /// whichever slot either side holds it in.
    #[test]
    fn load_copies_the_held_table() {
        let mut from = Slots::holding(5u32, Repeat::Check);
        *from.split().spare = 6;
        from.commit(Next::New);
        let mut to = Slots::<u32>::default();
        to.load(&from);
        assert!(matches!(to.held(), Held::Check(&6)));
        let mut none = Slots::<u32>::default();
        to.load(&none);
        assert!(matches!(to.held(), Held::None));
        none.load(&from);
        assert!(matches!(none.held(), Held::Check(&6)));
    }
}
