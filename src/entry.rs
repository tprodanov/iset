use core::{
    fmt,
    ops::Range,
};
use super::{
    IntervalMap, Interval,
    ix::{IndexType, DefaultIx},
};

/// A view into a vacant entry in an `IntervalMap`. It is part of the [Entry](enum.Entry.html) enum.
pub struct VacantEntry<'a, T, V, Ix: IndexType = DefaultIx> {
    tree: &'a mut IntervalMap<T, V, Ix>,
    parent: Ix,
    left_side: bool,
    interval: Interval<T>,
}

impl<'a, T, V, Ix: IndexType> VacantEntry<'a, T, V, Ix>
where T: Copy,
{
    pub(super) fn new(tree: &'a mut IntervalMap<T, V, Ix>, parent: Ix, left_side: bool, interval: Interval<T>) -> Self {
        Self { tree, parent, left_side, interval }
    }

    /// Returns the interval that was used for the search.
    pub fn interval(&self) -> Range<T> {
        self.interval.to_range()
    }
}

impl<'a, T, V, Ix: IndexType> VacantEntry<'a, T, V, Ix>
where T: Copy + PartialOrd,
{
    /// Inserts a new value in the IntervalMap, and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        self.tree.insert_at(self.parent, self.left_side, self.interval, value)
    }
}

impl<'a, T, V, Ix: IndexType> fmt::Debug for VacantEntry<'a, T, V, Ix>
where T: PartialOrd + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VacantEntry({:?})", self.interval.to_range())
    }
}

/// A view into an occupied entry in an `IntervalMap`. It is part of the [Entry](enum.Entry.html) enum.
pub struct OccupiedEntry<'a, T, V, Ix: IndexType = DefaultIx> {
    tree: &'a mut IntervalMap<T, V, Ix>,
    index: Ix,
}

impl<'a, T, V, Ix: IndexType> OccupiedEntry<'a, T, V, Ix>
where T: Copy,
{
    pub(super) fn new(tree: &'a mut IntervalMap<T, V, Ix>, index: Ix) -> Self {
        Self { tree, index }
    }

    /// Returns the interval that was used for the search.
    pub fn interval(&self) -> Range<T> {
        self.tree.nodes[self.index.get()].interval.to_range()
    }

    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        &self.tree.nodes[self.index.get()].value
    }

    /// Gets a mutable reference to the value in the entry.
    /// If you need a reference to the OccupiedEntry that may outlive the destruction of the Entry value,
    /// see [into_mut](#method.into_mut).
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.tree.nodes[self.index.get()].value
    }

    /// Converts the entry into a mutable reference to its value.
    /// If you need multiple references to the OccupiedEntry, see [get_mut](#method.get_mut).
    pub fn into_mut(self) -> &'a mut V {
        &mut self.tree.nodes[self.index.get()].value
    }

    /// Replaces the value of the entry, and returns the entry's old value.
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(&mut self.tree.nodes[self.index.get()].value, value)
    }
}

impl<'a, T, V, Ix: IndexType> fmt::Debug for OccupiedEntry<'a, T, V, Ix>
where T: PartialOrd + Copy + fmt::Debug,
      V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let node = &self.tree.nodes[self.index.get()];
        write!(f, "OccupiedEntry({:?} => {:?})", node.interval.to_range(), node.value)
    }
}

impl<'a, T, V, Ix: IndexType> OccupiedEntry<'a, T, V, Ix>
where T: Copy + PartialOrd,
{
    /// Removes the entry from the map, and returns the removed value.
    pub fn remove(self) -> V {
        self.tree.remove_at(self.index).expect("Cannot be None")
    }
}

/// A view into a single entry in an [IntervalMap](../struct.IntervalMap.html), which may either be vacant or occupied.
/// This enum is constructed from the [entry](../struct.IntervalMap.html#method.entry).
pub enum Entry<'a, T, V, Ix: IndexType = DefaultIx> {
    Vacant(VacantEntry<'a, T, V, Ix>),
    Occupied(OccupiedEntry<'a, T, V, Ix>)
}

impl<'a, T, V, Ix: IndexType> Entry<'a, T, V, Ix>
where T: Copy,
{
    /// Returns the interval that was used for the search.
    pub fn interval(&self) -> Range<T> {
        match self {
            Entry::Occupied(entry) => entry.interval(),
            Entry::Vacant(entry) => entry.interval(),
        }
    }
}

impl<'a, T, V, Ix: IndexType> Entry<'a, T, V, Ix>
where T: Copy + PartialOrd,
{
    /// If value is missing, initializes it with the `default` value.
    /// In any case, returns a mutable reference to the value.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// If value is missing, initializes it with a `default()` function call.
    /// In any case, returns a mutable reference to the value.
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// If value is missing, initializes it with a `default(interval)`.
    /// In any case, returns a mutable reference to the value.
    pub fn or_insert_with_interval<F>(self, default: F) -> &'a mut V
    where F: FnOnce(Range<T>) -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let val = default(entry.interval());
                entry.insert(val)
            }
        }
    }

    /// If the entry is occupied, modifies the value with `f` function, and returns a new entry.
    /// Does nothing if the value was vacant.
    pub fn and_modify<F>(self, f: F) -> Self
    where F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }
}

impl<'a, T, V, Ix: IndexType> Entry<'a, T, V, Ix>
where T: Copy + PartialOrd,
      V: Default,
{
    /// If value is missing, initializes it with `V::default()`.
    /// In any case, returns a mutable reference to the value.
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(V::default()),
        }
    }
}

impl<'a, T, V, Ix: IndexType> fmt::Debug for Entry<'a, T, V, Ix>
where T: PartialOrd + Copy + fmt::Debug,
      V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Entry::Occupied(entry) => entry.fmt(f),
            Entry::Vacant(entry) => entry.fmt(f),
        }
    }
}
