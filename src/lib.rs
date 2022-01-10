//! This crates implements map and set with interval keys (ranges `x..y`).

//! [IntervalMap](struct.IntervalMap.html) is implemented using red-black binary tree, where each node contains
//! information about the smallest start and largest end in its subtree.
//! The tree takes *O(N)* space and allows insertion, removal and search in *O(log N)*.
//! [IntervalMap](struct.IntervalMap.html) allows to search for all entries overlapping a query (interval or a point,
//! output would be sorted by keys). Search takes *O(log N + K)* where *K* is the size of the output.
//! Additionally, you can extract the smallest/largest interval and their values in *O(log N)*.
//!
//! [IntervalSet](struct.IntervalSet.html) is a newtype over [IntervalMap](struct.IntervalMap.html) with empty values.
//!
//! Any iterator that goes over an [IntervalMap](struct.IntervalMap.html) or [IntervalSet](struct.IntervalSet.html)
//! returns intervals/values sorted lexicographically by intervals.
//!
//! This crate allows to write interval maps and sets to .dot files
//! (see [IntervalMap::write_dot](struct.IntervalMap.html#method.write_dot),
//! [IntervalMap::write_dot_without_values](struct.IntervalMap.html#method.write_dot_without_values) and
//! [IntervalSet::write_dot](struct.IntervalSet.html#method.write_dot)).
//! You can disable this feature using `cargo build --no-default-features`,
//! in that case the crate supports `no_std` environments.
//!
//! This crate supports serialization/deserialization using an optional feature `serde`.

// TODO:
// - union, split
// - unsorted_iter
// - remove by value
// - macro with index type

#![no_std]

#[cfg(feature = "dot")]
#[macro_use]
extern crate std;
extern crate alloc;
extern crate bit_vec;

pub mod ix;
pub mod iter;
pub mod set;
#[cfg(test)]
mod tests;
mod tree_rm;

use alloc::vec::Vec;
use core::ops::{Range, RangeFull, RangeInclusive, RangeBounds, Bound};
use core::fmt::{self, Debug, Display, Formatter};
use core::cmp::Ordering;
#[cfg(feature = "dot")]
use std::io::{self, Write};
#[cfg(feature = "serde")]
use {
    core::marker::PhantomData,
    serde::{Serialize, Serializer, Deserialize, Deserializer},
    serde::ser::{SerializeTuple, SerializeSeq},
    serde::de::{Visitor, SeqAccess},
};

use ix::IndexType;
pub use ix::DefaultIx;
use iter::*;
pub use set::IntervalSet;

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
struct Interval<T: PartialOrd + Copy> {
    start: T,
    end: T,
}

impl<T: PartialOrd + Copy + Display> Display for Interval<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl<T: PartialOrd + Copy> Interval<T> {
    fn new(range: &Range<T>) -> Self {
        Interval {
            start: range.start,
            end: range.end,
        }
    }

    fn to_range(&self) -> Range<T> {
        self.start..self.end
    }

    fn intersects_range<R: RangeBounds<T>>(&self, range: &R) -> bool {
        // Each match returns bool
        (match range.end_bound() {
            Bound::Included(value) => self.start <= *value,
            Bound::Excluded(value) => self.start < *value,
            Bound::Unbounded => true,
        })
            &&
        (match range.start_bound() {
            Bound::Included(value) | Bound::Excluded(value) => self.end > *value,
            Bound::Unbounded => true,
        })
    }

    fn extend(&mut self, other: &Interval<T>) {
        if other.start < self.start {
            self.start = other.start;
        }
        if other.end > self.end {
            self.end = other.end;
        }
    }
}

#[cfg(feature = "serde")]
impl<T: PartialOrd + Copy + Serialize> Serialize for Interval<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (self.start, self.end).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: PartialOrd + Copy + Deserialize<'de>> Deserialize<'de> for Interval<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (start, end) = <(T, T)>::deserialize(deserializer)?;
        Ok(Interval { start, end })
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct Node<T: PartialOrd + Copy, V, Ix: IndexType> {
    interval: Interval<T>,
    subtree_interval: Interval<T>,
    value: V,
    left: Ix,
    right: Ix,
    parent: Ix,
    red_color: bool,
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> Node<T, V, Ix> {
    fn new(range: Range<T>, value: V) -> Self {
        Node {
            interval: Interval::new(&range),
            subtree_interval: Interval::new(&range),
            value,
            left: Ix::MAX,
            right: Ix::MAX,
            parent: Ix::MAX,
            red_color: true,
        }
    }

    #[inline(always)]
    fn is_red(&self) -> bool {
        self.red_color
    }

    #[inline(always)]
    fn is_black(&self) -> bool {
        !self.red_color
    }

    #[inline(always)]
    fn set_red(&mut self) {
        self.red_color = true;
    }

    #[inline(always)]
    fn set_black(&mut self) {
        self.red_color = false;
    }

    /// Swaps values and intervals between two mutable nodes.
    fn swap_with(&mut self, other: &mut Self) {
        core::mem::swap(&mut self.value, &mut other.value);
        core::mem::swap(&mut self.interval, &mut other.interval);
        core::mem::swap(&mut self.subtree_interval, &mut other.subtree_interval);
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, V: Display, Ix: IndexType> Node<T, V, Ix> {
    fn write_dot<W: Write>(&self, index: Ix, mut writer: W) -> io::Result<()> {
        writeln!(writer, "    {} [label=\"i={}\\n{}: {}\\nsubtree: {}\", fillcolor={}, style=filled]",
            index, index, self.interval, self.value, self.subtree_interval,
            if self.is_red() { "salmon" } else { "grey65" })?;
        if self.left.defined() {
            writeln!(writer, "    {} -> {} [label=\"L\"]", index, self.left)?;
        }
        if self.right.defined() {
            writeln!(writer, "    {} -> {} [label=\"R\"]", index, self.right)?;
        }
        Ok(())
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, V, Ix: IndexType> Node<T, V, Ix> {
    fn write_dot_without_values<W: Write>(&self, index: Ix, mut writer: W) -> io::Result<()> {
        writeln!(writer, "    {} [label=\"i={}: {}\\nsubtree: {}\", fillcolor={}, style=filled]",
            index, index, self.interval, self.subtree_interval, if self.is_red() { "salmon" } else { "grey65" })?;
        if self.left.defined() {
            writeln!(writer, "    {} -> {} [label=\"L\"]", index, self.left)?;
        }
        if self.right.defined() {
            writeln!(writer, "    {} -> {} [label=\"R\"]", index, self.right)?;
        }
        Ok(())
    }
}

fn check_interval<T: PartialOrd + Copy>(start: T, end: T) {
    if start < end {
        assert!(end > start, "Interval cannot be ordered (`start < end` but not `end > start`)");
    } else if end <= start {
        panic!("Interval is empty (`start >= end`)");
    } else {
        panic!("Interval cannot be ordered (not `start < end` and not `end <= start`)");
    }
}

fn check_interval_incl<T: PartialOrd + Copy>(start: T, end: T) {
    if start <= end {
        assert!(end >= start, "Interval cannot be ordered (`start < end` but not `end > start`)");
    } else if end < start {
        panic!("Interval is empty (`start > end`)");
    } else {
        panic!("Interval cannot be ordered (not `start <= end` and not `end < start`)");
    }
}

/// Map with interval keys (ranges `x..y`).
///
/// Range bounds should implement `PartialOrd` and `Copy`, for example any
/// integer or float types. However, you cannot use values that cannot be used in comparison (such as `NAN`).
/// There are no restrictions on values.
///
/// ```rust
/// let mut map = iset::IntervalMap::new();
/// map.insert(20..30, "a");
/// map.insert(15..25, "b");
/// map.insert(10..20, "c");
///
/// let a: Vec<_> = map.iter(..).collect();
/// assert_eq!(a, &[(10..20, &"c"), (15..25, &"b"), (20..30, &"a")]);
///
/// // Iterate over intervals that overlap query (..20 here). Output is sorted.
/// let b: Vec<_> = map.intervals(..20).collect();
/// assert_eq!(b, &[10..20, 15..25]);
///
/// // Iterate over values that overlap query (20.. here). Output is sorted by intervals.
/// let c: Vec<_> = map.values(20..).collect();
/// assert_eq!(c, &[&"b", &"a"]);
/// ```
///
/// Insertion takes *O(log N)* and search takes *O(log N + K)* where *K* is the size of the output.
///
/// You can [insert](#method.insert) and [remove](#method.remove) intervals of type `x..y`.
/// Next, you can get all intervals that overlap a query (`x..y`, `x..=y`, `x..`, `..`, etc) using methods
/// [iter](#method.iter), [intervals](#method.intervals) and [values](#method.values).
/// Methods [overlap](#method.overlap), [intervals_overlap](#method.intervals_overlap) and
/// [values_overlap](#method.values_overlap) allow to search for intervals/values that overlap a single point
/// (same as `x..=x`). It is possible to extract value by the interval in `O(log N)` using
/// [get](#method.get) and [get_mut](#method.get_mut).
///
/// Additionally, you can use mutable iterators [iter_mut](#method.iter_mut), [values_mut](#method.values_mut)
/// as well as [overlap_mut](#method.overlap_mut) and [values_overlap_mut](#method.values_overlap_mut).
///
/// You can get a value that corresponds to the [smallest](#method.smallest) or [largest](#method.largest)
/// interval in *O(log N)*.
///
/// You can construct [IntervalMap](struct.IntervalMap.html) using `collect()`:
/// ```rust
/// let map: iset::IntervalMap<_, _> = vec![(10..20, "a"), (0..20, "b")]
///                                        .into_iter().collect();
/// ```
///
/// You can also construct [IntervalMap](struct.IntervalMap.html) using [interval_map](macro.interval_map.html) macro:
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let map = interval_map!{ 0..10 => "a", 5..15 => "b", -5..20 => "c" };
/// let a: Vec<_> = map.iter(..).collect();
/// assert_eq!(a, &[(-5..20, &"c"), (0..10, &"a"), (5..15, &"b")]);
/// ```
///
/// # Index types:
/// You can specify [index type](ix/trait.IndexType.html) (`u8`, `u16`, `u32` and `u64`) used in the inner
/// representation of `IntervalMap` ([iset::DefaultIx](ix/type.DefaultIx.html) = u32`).
///
/// Method [new](#method.new), macro [interval_map](macro.interval_map.html) or function
/// `collect()` create `IntervalMap` with index type `u32`. If you wish to use another index type you can use
/// methods [default](#method.default) or [with_capacity](#method.with_capacity). For example:
/// ```rust
/// let mut map: iset::IntervalMap<_, _, u16> = iset::IntervalMap::default();
/// map.insert(10..20, "a");
/// ```
///
/// Finally, you can construct an `IntervalMap` from a sorted iterator in *O(N)* (requires index type):
/// ```rust
/// let map: iset::IntervalMap<_, _, u32> = iset::IntervalMap::from_sorted(
///     vec![(10..20, "a"), (15..20, "b")].into_iter());
/// ```
#[derive(Clone)]
pub struct IntervalMap<T: PartialOrd + Copy, V, Ix: IndexType = DefaultIx> {
    nodes: Vec<Node<T, V, Ix>>,
    root: Ix,
}

impl<T: PartialOrd + Copy, V> IntervalMap<T, V> {
    /// Creates an empty [IntervalMap](struct.IntervalMap.html).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> Default for IntervalMap<T, V, Ix> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            root: Ix::MAX,
        }
    }
}

// #[cfg(not(std))]
fn calculate_max_depth(mut n: usize) -> u16 {
    // Same as `((n + 1) as f64).log2().ceil() as u16`, but without std.
    let mut depth = 0;
    while n > 0 {
        n >>= 1;
        depth += 1;
    }
    depth
}

#[cfg(std)]
fn calculate_max_depth(n: usize) -> u16 {
    ((n + 1) as f64).log2().ceil() as u16
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> IntervalMap<T, V, Ix> {
    /// Creates an empty [IntervalMap](struct.IntervalMap.html) with `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            root: Ix::MAX,
        }
    }

    /// Initializes map within indices [start, end) in case of sorted nodes.
    /// rev_depth: inverse depth (top recursion call has high rev_depth, lowest recursion call has rev_depth == 1).
    fn init_from_sorted(&mut self, start: usize, end: usize, rev_depth: u16) -> Ix {
        debug_assert!(start < end);
        if start + 1 == end {
            if rev_depth > 1 {
                self.nodes[start].set_black();
            }
            return Ix::new(start).unwrap();
        }

        let center = (start + end) / 2;
        self.nodes[center].set_black();
        let center_ix = Ix::new(center).unwrap();
        if start < center {
            let left_ix = self.init_from_sorted(start, center, rev_depth - 1);
            self.nodes[center].left = left_ix;
            self.nodes[left_ix.get()].parent = center_ix;
        }
        if center + 1 < end {
            let right_ix = self.init_from_sorted(center + 1, end, rev_depth - 1);
            self.nodes[center].right = right_ix;
            self.nodes[right_ix.get()].parent = center_ix;
        }
        self.update_subtree_interval(center_ix);
        center_ix
    }

    /// Creates an interval map from a sorted iterator over pairs `(range, value)`. Takes *O(N)*.
    pub fn from_sorted<I: Iterator<Item = (Range<T>, V)>>(iter: I) -> Self {
        let mut map = Self {
            nodes: iter.map(|(range, value)| Node::new(range, value)).collect(),
            root: Ix::MAX,
        };
        let n = map.nodes.len();
        for i in 1..n {
            assert!(map.nodes[i - 1].interval <= map.nodes[i].interval,
                "Cannot construct interval map from sorted nodes: intervals at positions {} and {} are unordered!",
                i, i + 1);
        }
        if n > 0 {
            let max_depth = calculate_max_depth(n);
            map.root = map.init_from_sorted(0, n, max_depth);
        }
        map
    }

    /// Returns the number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the map contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clears the map, removing all values. This method has no effect on the allocated capacity.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = Ix::MAX;
    }

    /// Shrinks inner contents.
    pub fn shrink_to_fit(&mut self) {
        self.nodes.shrink_to_fit();
    }

    fn update_subtree_interval(&mut self, index: Ix) {
        let node = &self.nodes[index.get()];
        let mut subtree_interval = node.interval.clone();
        if node.left.defined() {
            subtree_interval.extend(&self.nodes[node.left.get()].subtree_interval);
        }
        if node.right.defined() {
            subtree_interval.extend(&self.nodes[node.right.get()].subtree_interval);
        }
        self.nodes[index.get()].subtree_interval = subtree_interval;
    }

    fn sibling(&self, index: Ix) -> Ix {
        let parent = self.nodes[index.get()].parent;
        if !parent.defined() {
            Ix::MAX
        } else if self.nodes[parent.get()].left == index {
            self.nodes[parent.get()].right
        } else {
            self.nodes[parent.get()].left
        }
    }

    fn rotate_left(&mut self, index: Ix) {
        let prev_parent = self.nodes[index.get()].parent;
        let prev_right = self.nodes[index.get()].right;
        debug_assert!(prev_right.defined());

        let new_right = self.nodes[prev_right.get()].left;
        self.nodes[index.get()].right = new_right;
        if new_right.defined() {
            self.nodes[new_right.get()].parent = index;
        }
        self.update_subtree_interval(index);

        self.nodes[prev_right.get()].left = index;
        self.nodes[index.get()].parent = prev_right;
        self.update_subtree_interval(prev_right);

        if prev_parent.defined() {
            if self.nodes[prev_parent.get()].left == index {
                self.nodes[prev_parent.get()].left = prev_right;
            } else {
                self.nodes[prev_parent.get()].right = prev_right;
            }
            self.nodes[prev_right.get()].parent = prev_parent;
            self.update_subtree_interval(prev_parent);
        } else {
            self.root = prev_right;
            self.nodes[prev_right.get()].parent = Ix::MAX;
        }
    }

    fn rotate_right(&mut self, index: Ix) {
        let prev_parent = self.nodes[index.get()].parent;
        let prev_left = self.nodes[index.get()].left;
        debug_assert!(prev_left.defined());

        let new_left = self.nodes[prev_left.get()].right;
        self.nodes[index.get()].left = new_left;
        if new_left.defined() {
            self.nodes[new_left.get()].parent = index;
        }
        self.update_subtree_interval(index);

        self.nodes[prev_left.get()].right = index;
        self.nodes[index.get()].parent = prev_left;
        self.update_subtree_interval(prev_left);

        if prev_parent.defined() {
            if self.nodes[prev_parent.get()].right == index {
                self.nodes[prev_parent.get()].right = prev_left;
            } else {
                self.nodes[prev_parent.get()].left = prev_left;
            }
            self.nodes[prev_left.get()].parent = prev_parent;
            self.update_subtree_interval(prev_parent);
        } else {
            self.root = prev_left;
            self.nodes[prev_left.get()].parent = Ix::MAX;
        }
    }

    fn insert_repair(&mut self, mut index: Ix) {
        loop {
            debug_assert!(self.nodes[index.get()].is_red());
            if index == self.root {
                self.nodes[index.get()].set_black();
                return;
            }

            // parent should be defined
            let parent = self.nodes[index.get()].parent;
            if self.nodes[parent.get()].is_black() {
                return;
            }

            // parent is red
            // grandparent should be defined
            let grandparent = self.nodes[parent.get()].parent;
            let uncle = self.sibling(parent);

            if uncle.defined() && self.nodes[uncle.get()].is_red() {
                self.nodes[parent.get()].set_black();
                self.nodes[uncle.get()].set_black();
                self.nodes[grandparent.get()].set_red();
                index = grandparent;
                continue;
            }

            if index == self.nodes[parent.get()].right && parent == self.nodes[grandparent.get()].left {
                self.rotate_left(parent);
                index = self.nodes[index.get()].left;
            } else if index == self.nodes[parent.get()].left && parent == self.nodes[grandparent.get()].right {
                self.rotate_right(parent);
                index = self.nodes[index.get()].right;
            }

            let parent = self.nodes[index.get()].parent;
            let grandparent = self.nodes[parent.get()].parent;
            if index == self.nodes[parent.get()].left {
                self.rotate_right(grandparent);
            } else {
                self.rotate_left(grandparent);
            }
            self.nodes[parent.get()].set_black();
            self.nodes[grandparent.get()].set_red();
            return;
        }
    }

    /// Inserts an interval `x..y` and its value. Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`)
    /// or contains a value that cannot be compared (such as `NAN`).
    pub fn insert(&mut self, interval: Range<T>, value: V) {
        check_interval(interval.start, interval.end);
        let new_ind = Ix::new(self.nodes.len()).unwrap_or_else(|e| panic!("{}", e));
        let mut new_node = Node::new(interval, value);
        if !self.root.defined() {
            self.root = Ix::new(0).unwrap();
            new_node.set_black();
            self.nodes.push(new_node);
            return;
        }

        let mut current = self.root;
        loop {
            self.nodes[current.get()].subtree_interval.extend(&new_node.interval);
            let child = if new_node.interval <= self.nodes[current.get()].interval {
                &mut self.nodes[current.get()].left
            } else {
                &mut self.nodes[current.get()].right
            };
            if !child.defined() {
                *child = new_ind;
                new_node.parent = current;
                break;
            } else {
                current = *child;
            }
        }
        self.nodes.push(new_node);
        self.insert_repair(new_ind);
    }

    fn find_index(&self, interval: &Range<T>) -> Ix {
        check_interval(interval.start, interval.end);
        let interval = Interval::new(interval);
        let mut index = self.root;
        while index.defined() {
            let node = &self.nodes[index.get()];
            match interval.partial_cmp(&node.interval) {
                Some(Ordering::Less) => index = node.left,
                Some(Ordering::Greater) => index = node.right,
                Some(Ordering::Equal) => return index,
                None => panic!("Cannot order intervals"),
            }
        }
        index
    }

    /// Check if the interval map contains `interval` (exact match).
    pub fn contains(&self, interval: Range<T>) -> bool {
        self.find_index(&interval).defined()
    }

    /// Returns value associated with `interval` (exact match).
    /// If there are multiple matches, returns a value associated with any of them (order is unspecified).
    /// If there is no such interval, returns `None`.
    pub fn get(&self, interval: Range<T>) -> Option<&V> {
        let index = self.find_index(&interval);
        if index.defined() {
            Some(&self.nodes[index.get()].value)
        } else {
            None
        }
    }

    /// Returns mutable value associated with `interval` (exact match).
    /// If there are multiple matches, returns a value associated with any of them (order is unspecified).
    /// If there is no such interval, returns `None`.
    pub fn get_mut(&mut self, interval: Range<T>) -> Option<&mut V> {
        let index = self.find_index(&interval);
        if index.defined() {
            Some(&mut self.nodes[index.get()].value)
        } else {
            None
        }
    }

    /// Removes an entry, associated with `interval` (exact match is required), takes *O(log N)*.
    /// Returns value if the interval was present in the map, and None otherwise.
    /// If several intervals match the query interval, removes any one of them (order is unspecified).
    pub fn remove(&mut self, interval: Range<T>) -> Option<V> {
        self.remove_at(self.find_index(&interval))
    }

    /// Returns the pair `(x..y, &value)` with the smallest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn smallest(&self) -> Option<(Range<T>, &V)> {
        let mut index = self.root;
        if !index.defined() {
            return None;
        }
        while self.nodes[index.get()].left.defined() {
            index = self.nodes[index.get()].left;
        }
        Some((self.nodes[index.get()].interval.to_range(), &self.nodes[index.get()].value))
    }

    /// Returns the pair `(x..y, &mut value)` with the smallest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn smallest_mut(&mut self) -> Option<(Range<T>, &mut V)> {
        let mut index = self.root;
        if !index.defined() {
            return None;
        }
        while self.nodes[index.get()].left.defined() {
            index = self.nodes[index.get()].left;
        }
        Some((self.nodes[index.get()].interval.to_range(), &mut self.nodes[index.get()].value))
    }

    /// Returns the pair `(x..y, &value)` with the largest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn largest(&self) -> Option<(Range<T>, &V)> {
        let mut index = self.root;
        if !index.defined() {
            return None;
        }
        while self.nodes[index.get()].right.defined() {
            index = self.nodes[index.get()].right;
        }
        Some((self.nodes[index.get()].interval.to_range(), &self.nodes[index.get()].value))
    }

    /// Returns the pair `(x..y, &mut value)` with the largest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn largest_mut(&mut self) -> Option<(Range<T>, &mut V)> {
        let mut index = self.root;
        if !index.defined() {
            return None;
        }
        while self.nodes[index.get()].right.defined() {
            index = self.nodes[index.get()].right;
        }
        Some((self.nodes[index.get()].interval.to_range(), &mut self.nodes[index.get()].value))
    }

    /// Iterates over pairs `(x..y, &value)` that overlap the `query`.
    /// Takes *O(log N + K)* where *K* is the size of the output.
    /// Output is sorted by intervals, but not by values.
    ///
    /// Panics if `interval` is empty or contains a value that cannot be compared (such as `NAN`).
    pub fn iter<'a, R: RangeBounds<T>>(&'a self, query: R) -> Iter<'a, T, V, R, Ix> {
        Iter::new(self, query)
    }

    /// Iterates over intervals `x..y` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn intervals<'a, R: RangeBounds<T>>(&'a self, query: R) -> Intervals<'a, T, V, R, Ix> {
        Intervals::new(self, query)
    }

    /// Iterates over values that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn values<'a, R: RangeBounds<T>>(&'a self, query: R) -> Values<'a, T, V, R, Ix> {
        Values::new(self, query)
    }

    /// Iterator over pairs `(x..y, &mut value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn iter_mut<'a, R: RangeBounds<T>>(&'a mut self, query: R) -> IterMut<'a, T, V, R, Ix> {
        IterMut::new(self, query)
    }

    /// Iterator over *mutable* values that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn values_mut<'a, R: RangeBounds<T>>(&'a mut self, query: R) -> ValuesMut<'a, T, V, R, Ix> {
        ValuesMut::new(self, query)
    }

    /// Consumes [IntervalMap](struct.IntervalMap.html) and
    /// iterates over pairs `(x..y, value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn into_iter<R: RangeBounds<T>>(self, query: R) -> IntoIter<T, V, R, Ix> {
        IntoIter::new(self, query)
    }

    /// Iterates over pairs `(x..y, &value)` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap<'a>(&'a self, point: T) -> Iter<'a, T, V, RangeInclusive<T>, Ix> {
        Iter::new(self, point..=point)
    }

    /// Iterates over intervals `x..y` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn intervals_overlap<'a>(&'a self, point: T) -> Intervals<'a, T, V, RangeInclusive<T>, Ix> {
        Intervals::new(self, point..=point)
    }

    /// Iterates over values that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn values_overlap<'a>(&'a self, point: T) -> Values<'a, T, V, RangeInclusive<T>, Ix> {
        Values::new(self, point..=point)
    }

    /// Iterator over pairs `(x..y, &mut value)` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap_mut<'a>(&'a mut self, point: T) -> IterMut<'a, T, V, RangeInclusive<T>, Ix> {
        IterMut::new(self, point..=point)
    }

    /// Iterates over *mutable* values that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn values_overlap_mut<'a>(&'a mut self, point: T) -> ValuesMut<'a, T, V, RangeInclusive<T>, Ix> {
        ValuesMut::new(self, point..=point)
    }
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> core::iter::IntoIterator for IntervalMap<T, V, Ix> {
    type IntoIter = IntoIter<T, V, RangeFull, Ix>;
    type Item = (Range<T>, V);

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self, ..)
    }
}

/// Construct [IntervalMap](struct.IntervalMap.html) from pairs `(x..y, value)`.
impl<T: PartialOrd + Copy, V> core::iter::FromIterator<(Range<T>, V)> for IntervalMap<T, V> {
    fn from_iter<I: IntoIterator<Item = (Range<T>, V)>>(iter: I) -> Self {
        let mut map = IntervalMap::new();
        for (range, value) in iter {
            map.insert(range, value);
        }
        map
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, V: Display, Ix: IndexType> IntervalMap<T, V, Ix> {
    /// Writes dot file to `writer`. `T` and `V` should implement `Display`.
    pub fn write_dot<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "digraph {{")?;
        for i in 0..self.nodes.len() {
            self.nodes[i].write_dot(Ix::new(i).unwrap(), &mut writer)?;
        }
        writeln!(writer, "}}")
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, V, Ix: IndexType> IntervalMap<T, V, Ix> {
    /// Writes dot file to `writer` without values. `T` should implement `Display`.
    pub fn write_dot_without_values<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "digraph {{")?;
        for i in 0..self.nodes.len() {
            self.nodes[i].write_dot_without_values(Ix::new(i).unwrap(), &mut writer)?;
        }
        writeln!(writer, "}}")
    }
}

impl<T: PartialOrd + Copy + Debug, V: Debug, Ix: IndexType> Debug for IntervalMap<T, V, Ix> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{{")?;
        let mut need_comma = false;
        for (interval, value) in self.iter(..) {
            if need_comma {
                write!(f, ", ")?;
            } else {
                need_comma = true;
            }
            write!(f, "{:?}: {:?}", interval, value)?;
        }
        write!(f, "}}")
    }
}

#[cfg(feature = "serde")]
impl<T, V, Ix> Serialize for IntervalMap<T, V, Ix>
    where
        T: PartialOrd + Copy + Serialize,
        V: Serialize,
        Ix: IndexType + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // For some reason, Vec<Node> does not support serialization. Because of that we create a newtype.
        struct NodeVecSer<'a, T, V, Ix>(&'a Vec<Node<T, V, Ix>>)
            where
                T: PartialOrd + Copy + Serialize,
                V: Serialize,
                Ix: IndexType + Serialize;

        impl<'a, T, V, Ix> Serialize for NodeVecSer<'a, T, V, Ix>
            where
                T: PartialOrd + Copy + Serialize,
                V: Serialize,
                Ix: IndexType + Serialize,
        {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
                for node in self.0 {
                    seq.serialize_element(node)?;
                }
                seq.end()
            }
        }

        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&NodeVecSer(&self.nodes))?;
        tup.serialize_element(&self.root)?;
        tup.end()
    }
}

// For some reason, Vec<Node> does not support deserialization. Because of that we create a newtype.
#[cfg(feature = "serde")]
struct NodeVecDe<T: PartialOrd + Copy, V, Ix: IndexType>(Vec<Node<T, V, Ix>>);

#[cfg(feature = "serde")]
impl<T: PartialOrd + Copy, V, Ix: IndexType> NodeVecDe<T, V, Ix> {
    fn take(self) -> Vec<Node<T, V, Ix>> {
        self.0
    }
}

#[cfg(feature = "serde")]
impl<'de, T, V, Ix> Deserialize<'de> for NodeVecDe<T, V, Ix>
    where
        T: PartialOrd + Copy + Deserialize<'de>,
        V: Deserialize<'de>,
        Ix: IndexType + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NodeVecVisitor<T: PartialOrd + Copy, V, Ix: IndexType> {
            marker: PhantomData<(T, V, Ix)>,
        }

        impl<'de, T, V, Ix> Visitor<'de> for NodeVecVisitor<T, V, Ix>
        where
            T: PartialOrd + Copy + Deserialize<'de>,
            V: Deserialize<'de>,
            Ix: IndexType + Deserialize<'de>,
        {
            type Value = NodeVecDe<T, V, Ix>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of Node<T, V, Ix>")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(NodeVecDe(values))
            }
        }

        let visitor = NodeVecVisitor {
            marker: PhantomData,
        };
        deserializer.deserialize_seq(visitor)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, V, Ix> Deserialize<'de> for IntervalMap<T, V, Ix>
where
    T: PartialOrd + Copy + Deserialize<'de>,
    V: Deserialize<'de>,
    Ix: IndexType + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (node_vec, root) = <(NodeVecDe<T, V, Ix>, Ix)>::deserialize(deserializer)?;
        Ok(IntervalMap {
            nodes: node_vec.take(),
            root,
        })
    }
}

/// Macros for [IntervalMap](struct.IntervalMap.html) creation.
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let map = interval_map!{ 0..10 => "a", 5..15 => "b", -5..20 => "c" };
/// let a: Vec<_> = map.iter(..).collect();
/// assert_eq!(a, &[(-5..20, &"c"), (0..10, &"a"), (5..15, &"b")]);
/// ```
#[macro_export]
macro_rules! interval_map {
    ( () ) => ( $crate::IntervalMap::new() );
    ( $( $k:expr => $v:expr ),* $(,)? ) => {
        {
            let mut _temp_map = $crate::IntervalMap::new();
            $(
                _temp_map.insert($k, $v);
            )*
            _temp_map
        }
    };
}

/// Macros for [IntervalSet](struct.IntervalSet.html) creation.
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let set = interval_set!{ 100..210, 50..150 };
/// let a: Vec<_> = set.iter(..).collect();
/// assert_eq!(a, &[50..150, 100..210]);
/// ```
#[macro_export]
macro_rules! interval_set {
    ( () ) => ( $crate::IntervalSet::new() );
    ( $( $k:expr ),* $(,)? ) => {
        {
            let mut _temp_set = $crate::IntervalSet::new();
            $(
                _temp_set.insert($k);
            )*
            _temp_set
        }
    };
}
