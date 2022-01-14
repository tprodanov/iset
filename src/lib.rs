//! This crates implements map and set with interval keys (ranges `x..y`).
//!
//! [IntervalMap](struct.IntervalMap.html) is implemented using red-black binary tree, where each node contains
//! information about the smallest start and largest end in its subtree.
//! The tree takes *O(N)* space and allows insertion, removal and search in *O(log N)*.
//! [IntervalMap](struct.IntervalMap.html) allows to search for all entries overlapping a query (interval or a point,
//! output would be sorted by keys) in *O(log N + K)* where *K* is the size of the output.
//!
//! [IntervalSet](struct.IntervalSet.html) is a newtype over [IntervalMap](struct.IntervalMap.html) with empty values.
//!
//! Iterators that goes over an [IntervalMap](struct.IntervalMap.html) or [IntervalSet](struct.IntervalSet.html)
//! returns intervals/values sorted lexicographically by intervals.
//!
//! Optional feature `dot` allows to write interval maps and sets to .dot files
//! (see [IntervalMap::write_dot](struct.IntervalMap.html#method.write_dot)).
//! Additionally, this crate supports serialization/deserialization using an optional feature `serde`.
//! Without `dot` or `serde` features, this crate supports `no_std` environments.
//!
//! See [IntervalMap](struct.IntervalMap.html) for more extensive documentation and examples.

// TODO: has_overlap, overlap_len, overlap_count.

#![no_std]

#[cfg(feature = "dot")]
#[macro_use]
extern crate std;
extern crate alloc;

pub mod ix;
pub mod iter;
pub mod set;
mod tree_rm;
mod bitvec;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;
use core::ops::{Range, RangeFull, RangeInclusive, RangeBounds, Bound, AddAssign, Sub, Index};
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{FromIterator, IntoIterator};
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
use bitvec::BitVec;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
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
        check_interval(range.start, range.end);
        Interval {
            start: range.start,
            end: range.end,
        }
    }

    #[inline]
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

impl<T: PartialOrd + Copy> Ord for Interval<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Implement cmp by ourselves because T can be PartialOrd.
        if self.start < other.start {
            Ordering::Less
        } else if self.start == other.start {
            if self.end < other.end {
                Ordering::Less
            } else if self.end == other.end {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        } else {
            Ordering::Greater
        }
    }
}

impl<T: PartialOrd + Copy> Eq for Interval<T> { }

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
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> Node<T, V, Ix> {
    fn new(interval: Interval<T>, value: V) -> Self {
        Node {
            interval: interval.clone(),
            subtree_interval: interval,
            value,
            left: Ix::MAX,
            right: Ix::MAX,
            parent: Ix::MAX,
        }
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
    fn write_dot<W: Write>(&self, index: usize, is_red: bool, mut writer: W) -> io::Result<()> {
        writeln!(writer, "    {} [label=\"i={}\\n{}: {}\\nsubtree: {}\", fillcolor={}, style=filled]",
            index, index, self.interval, self.value, self.subtree_interval, if is_red { "salmon" } else { "grey65" })?;
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
    fn write_dot_without_values<W: Write>(&self, index: usize, is_red: bool, mut writer: W) -> io::Result<()> {
        writeln!(writer, "    {} [label=\"i={}: {}\\nsubtree: {}\", fillcolor={}, style=filled]",
            index, index, self.interval, self.subtree_interval, if is_red { "salmon" } else { "grey65" })?;
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

/// Map with interval keys (`x..y`).
///
/// Range bounds should implement `PartialOrd` and `Copy`, for example any
/// integer or float types. However, you cannot use values that cannot be used in comparison (such as `NAN`),
/// although infinity is allowed.
/// There are no restrictions on values.
///
/// # Example.
///```rust
/// #[macro_use] extern crate iset;
///
/// let mut map = interval_map!{ 20..30 => 'a', 15..25 => 'b', 10..20 => 'c' };
/// assert_eq!(map.insert(10..20, 'd'), Some('c'));
/// assert_eq!(map.insert(5..15, 'e'), None);
///
/// // Iterator over all pairs (range, value). Output is sorted.
/// let a: Vec<_> = map.iter(..).collect();
/// assert_eq!(a, &[(5..15, &'e'), (10..20, &'d'), (15..25, &'b'), (20..30, &'a')]);
///
/// // Iterate over intervals that overlap query (..20 here). Output is sorted.
/// let b: Vec<_> = map.intervals(..20).collect();
/// assert_eq!(b, &[5..15, 10..20, 15..25]);
///
/// assert_eq!(map[15..25], 'b');
/// // Replace 15..25 => 'b' into 'z'.
/// *map.get_mut(15..25).unwrap() = 'z';
///
/// // Iterate over values that overlap query (20.. here). Output is sorted by intervals.
/// let c: Vec<_> = map.values(20..).collect();
/// assert_eq!(c, &[&'z', &'a']);
///
/// // Remove 10..20 => 'd'.
/// assert_eq!(map.remove(10..20), Some('d'));
/// ```
///
/// # Insertion, search and removal.
///
/// All three operations take *O(log N)*.
/// By default, this crate does not allow duplicate keys, [insert](#method.insert) replaces and returns the old
/// value if the interval was already present in the map.
/// Note, that the key is not updated even if the value is replaced.
/// This matters for types that can be `==` without being identical.
///
/// Search operations [contains](#method.contains), [get](#method.get) and [get_mut](#method.get_mut) is usually faster
/// than insertion or removal, as the tree does not need to be rebalanced.
///
/// You can remove nodes from the tree using [remove](#method.remove) method given the interval key.
/// Currently, it is not feasible to have a method that removes multiple nodes at once
/// (for example based on a predicate).
///
/// It is possible to store entries with equal intervals by calling [force_insert](#method.force_insert).
/// This method should be used with care, as methods [get](#method.get), [get_mut](#method.get_mut) and
/// [remove](#method.remove) only return/remove a single entry (see [force_insert](#method.force_insert) for more details).
///
/// Additionally, it is possible to get or remove the entry with the smallest/largest interval in the map
/// (in lexicographical order), see [smallest](#method.smallest), [largest](#method.largest), etc.
/// These methods take *O(log N)* as well.
///
/// Method [range](#method.range) allows to extract the range that covers all intervals in *O(1)*.
///
/// # Iteration.
///
/// Interval map allows to quickly find all intervals that overlap a query interval in *O(log N + K)* where *K* is
/// the size of the output. As a bonus, returned items are always sorted by intervals (in lexicographical order).
/// Iteration methods include:
/// - [iter](#method.iter): iterate over pairs `(x..y, &value)`,
/// - [intervals](#method.intervals): iterate over interval keys `x..y`,
/// - [values](#method.values): iterate over values `&value`,
/// - Mutable iterators [iter_mut](#method.iter_mut) and [values_mut](#method.values_mut),
/// - Into iterators [into_iter](#method.into_iter), [into_intervals](#method.into_intervals) and
/// [into_values](#method.into_values).
///
/// Additionally, all above methods have their `unsorted_` counterparts
/// (for example [unsorted_iter](#method.unsorted_iter)).
/// These iterators traverse the whole map in an arbitrary (unsorted) order.
/// Although both `map.iter(..)` and `map.unsorted_iter()` both output all entries in the map and both take *O(N)*,
/// unsorted iterator is slightly faster as it reads the memory consecutively instead of traversing the tree.
///
/// Methods `iter`, `intervals`, `values`, `iter_mut` and `values_mut` have alternatives [overlap](#method.overlap),
/// [overlap_intervals](#method.overlap_intervals), ..., that allow to iterate over all entries that
/// cover a single point `x` (same as `x..=x`).
///
/// # Index types.
///
/// Every node in the tree stores three indices (to the parent and two children), and as a result, memory usage can be
/// reduced by reducing index sizes. In most cases, number of items in the map does not exceed `u32::MAX`, therefore
/// we store indices as `u32` numbers by default (`iset::DefaultIx = u32`).
/// You can use four integer types (`u8`, `u16`, `u32` or `u64`) as index types.
/// Number of elements in the interval map cannot exceed `IndexType::MAX - 1`: for example a map with `u8` indices
/// can store up to 255 items.
///
/// Using smaller index types saves memory and may reduce running time.
///
/// # Interval map creation.
///
/// An interval map can be created using the following methods:
/// ```rust
/// #[macro_use] extern crate iset;
/// use iset::IntervalMap;
///
/// // Creates an empty interval map with the default index type (u32):
/// let mut map = IntervalMap::new();
/// map.insert(10..20, 'a');
///
/// // Creates an empty interval map and specifies index type (u16 here):
/// let mut map = IntervalMap::<_, _, u16>::default();
/// map.insert(10..20, 'a');
///
/// let mut map = IntervalMap::<_, _, u16>::with_capacity(10);
/// map.insert(10..20, 'a');
///
/// // Creates an interval map with the default index type:
/// let map = interval_map!{ 0..10 => 'a', 5..15 => 'b' };
///
/// // Creates an interval map and specifies index type:
/// let map = interval_map!{ [u16] 0..10 => 'a', 5..15 => 'b' };
///
/// // Creates an interval map from a sorted iterator, takes O(N):
/// let vec = vec![(0..10, 'b'), (5..15, 'a')];
/// let map = IntervalMap::<_, _, u32>::from_sorted(vec.into_iter());
///
/// // Alternatively, you can use `.collect()` method that creates an interval map
/// // with the default index size. `Collect` does not require sorted intervals,
/// // but takes O(N log N).
/// let vec = vec![(5..15, 'a'), (0..10, 'b')];
/// let map: IntervalMap<_, _> = vec.into_iter().collect();
/// ```
///
/// # Implementation, merge and split.
///
/// To allow for fast retrieval of all intervals overlapping a query, we store the range of the subtree in each node
/// of the tree. Additionally, each node stores indices to the parent and to two children.
/// As a result, size of the map is approximately `4 * sizeof(T) + sizeof(V) + 3 * sizeof(Ix)`.
///
/// In order to reduce number of heap allocations and access memory consecutively, we store tree nodes in a vector.
/// This does not impact time complexity of all methods except for *merge* and *split*.
/// In a heap-allocated tree, merge takes *O(M log (N / M + 1))* where *M* is the size of the smaller tree.
/// Here, we are required to merge sorted iterators and construct a tree using the sorted iterator as input,
/// which takes *O(N + M)*.
///
/// Because of that, this crate does not implement merge or split, however, these procedures can be emulated using
/// [from_sorted](#method.from_sorted), [itertools::merge](https://docs.rs/itertools/latest/itertools/fn.merge.html)
/// and [Iterator::partition](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.partition) in linear time.
#[derive(Clone)]
pub struct IntervalMap<T, V, Ix = DefaultIx>
where T: PartialOrd + Copy,
      Ix: IndexType,
{
    nodes: Vec<Node<T, V, Ix>>,
    // true if the node is red, false if black.
    colors: BitVec,
    root: Ix,
}

impl<T: PartialOrd + Copy, V> IntervalMap<T, V> {
    /// Creates an empty [IntervalMap](struct.IntervalMap.html)
    /// with default index type [DefaultIx](ix/type.DefaultIx.html).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> Default for IntervalMap<T, V, Ix> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            colors: BitVec::new(),
            root: Ix::MAX,
        }
    }
}

#[cfg(not(std))]
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
            colors: BitVec::with_capacity(capacity),
            root: Ix::MAX,
        }
    }

    /// Initializes map within indices [start, end) in case of sorted nodes.
    /// rev_depth: inverse depth (top recursion call has high rev_depth, lowest recursion call has rev_depth == 1).
    fn init_from_sorted(&mut self, start: usize, end: usize, rev_depth: u16) -> Ix {
        debug_assert!(start < end);
        if start + 1 == end {
            if rev_depth == 1 {
                // Set red.
                self.colors.set1(start);
            }
            return Ix::new(start).unwrap_or_else(|error| panic!("{}", error));
        }

        let center = (start + end) / 2;
        let center_ix = Ix::new(center).unwrap_or_else(|error| panic!("{}", error));
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
    ///
    /// Panics if the intervals are not sorted or if there are equal intervals.
    pub fn from_sorted<I>(iter: I) -> Self
    where I: Iterator<Item = (Range<T>, V)>,
    {
        let nodes: Vec<_> = iter.map(|(range, value)| Node::new(Interval::new(&range), value)).collect();
        let n = nodes.len();
        let mut map = Self {
            nodes,
            colors: BitVec::from_elem(n, false), // Start with all black nodes.
            root: Ix::MAX,
        };
        for i in 1..n {
            assert!(map.nodes[i - 1].interval < map.nodes[i].interval,
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
        self.colors.clear();
        self.root = Ix::MAX;
    }

    /// Shrinks inner contents.
    pub fn shrink_to_fit(&mut self) {
        self.nodes.shrink_to_fit();
        self.colors.shrink_to_fit();
    }

    #[inline]
    fn is_red(&self, ix: Ix) -> bool {
        self.colors.get(ix.get())
    }

    #[inline]
    fn is_black(&self, ix: Ix) -> bool {
        !self.colors.get(ix.get())
    }

    #[inline]
    fn is_black_or_nil(&self, ix: Ix) -> bool {
        !ix.defined() || !self.colors.get(ix.get())
    }

    #[inline]
    fn set_red(&mut self, ix: Ix) {
        self.colors.set1(ix.get());
    }

    #[inline]
    fn set_black(&mut self, ix: Ix) {
        self.colors.set0(ix.get());
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
            debug_assert!(self.is_red(index));
            if index == self.root {
                self.set_black(index);
                return;
            }

            // parent should be defined
            let parent = self.nodes[index.get()].parent;
            if self.is_black(parent) {
                return;
            }

            // parent is red
            // grandparent should be defined
            let grandparent = self.nodes[parent.get()].parent;
            let uncle = self.sibling(parent);

            if uncle.defined() && self.is_red(uncle) {
                self.set_black(parent);
                self.set_black(uncle);
                self.set_red(grandparent);
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
            self.set_black(parent);
            self.set_red(grandparent);
            return;
        }
    }

    fn fix_intervals_up(&mut self, mut ix: Ix) {
        while ix.defined() {
            self.update_subtree_interval(ix);
            ix = self.nodes[ix.get()].parent;
        }
    }

    /// Insert pair `(interval, value)`.
    /// If both `replace` and `interval` was already in the map, replacing the value and returns the old value.
    /// Otherwise, inserts a new node and returns None.
    fn insert_inner(&mut self, interval: Range<T>, value: V, replace: bool) -> Option<V> {
        let interval = Interval::new(&interval);
        let mut current = self.root;
        let new_index = Ix::new(self.nodes.len()).unwrap_or_else(|error| panic!("{}", error));

        if !current.defined() {
            self.root = new_index;
            self.nodes.push(Node::new(interval, value));
            // New node should be black.
            self.colors.push(false);
            return None;
        }

        loop {
            let node = &mut self.nodes[current.get()];
            let child = match interval.cmp(&node.interval) {
                Ordering::Less => &mut node.left,
                Ordering::Equal if replace => {
                    let old_val = core::mem::replace(&mut node.value, value);
                    self.fix_intervals_up(current);
                    return Some(old_val);
                }
                // If Ordering::Greater or if Equal and there is no replacing.
                _ => &mut node.right,
            };
            if child.defined() {
                current = *child;
            } else {
                let mut new_node = Node::new(interval, value);
                *child = new_index;
                new_node.parent = current;
                self.nodes.push(new_node);
                self.colors.push(true);
                self.fix_intervals_up(current);
                self.insert_repair(new_index);
                return None;
            }
        }
    }

    /// Inserts an interval `x..y` and its value into the map. Takes *O(log N)*.
    ///
    /// If the map did not contain the interval, returns `None`. Otherwise returns the old value.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    #[inline]
    pub fn insert(&mut self, interval: Range<T>, value: V) -> Option<V> {
        self.insert_inner(interval, value, true)
    }

    /// Inserts an interval `x..y` and its value into the map even if there was an entry with matching interval.
    /// Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    ///
    /// <div class="example-wrap" style="display:inline-block"><pre class="compile_fail" style="white-space:normal;font:inherit;">
    ///
    /// **Warning:** After `force_insert`, the map can contain several entries with equal intervals.
    /// Calling [get](#method.get), [get_mut](#method.get_mut) or [remove](#method.remove)
    /// will arbitrarily
    /// return/remove only one of the entries.
    ///
    /// Various iterators will output all appropriate intervals, however the order of entries with equal intervals
    /// will be arbitrary.
    /// </pre></div>
    ///
    /// ```rust
    /// let mut map = iset::interval_map!{};
    /// map.force_insert(10..20, 1);
    /// map.force_insert(15..25, 2);
    /// map.force_insert(20..30, 3);
    /// map.force_insert(15..25, 4);
    ///
    /// // Returns either 2 or 4.
    /// assert!(map.get(15..25).unwrap() % 2 == 0);
    /// // Removes either 15..25 => 2 or 15..25 => 4.
    /// assert!(map.remove(15..25).unwrap() % 2 == 0);
    /// println!("{:?}", map);
    /// // {10..20 => 1, 15..25 => 4, 20..30 => 3} OR
    /// // {10..20 => 1, 15..25 => 2, 20..30 => 3}
    /// ```
    #[inline]
    pub fn force_insert(&mut self, interval: Range<T>, value: V) {
        assert!(self.insert_inner(interval, value, false).is_none(), "Force insert should always return None");
    }

    fn find_index(&self, interval: &Range<T>) -> Ix {
        let interval = Interval::new(interval);
        let mut index = self.root;
        while index.defined() {
            let node = &self.nodes[index.get()];
            match interval.cmp(&node.interval) {
                Ordering::Less => index = node.left,
                Ordering::Greater => index = node.right,
                Ordering::Equal => return index,
            }
        }
        index
    }

    /// Check if the interval map contains `interval` (exact match).
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    pub fn contains(&self, interval: Range<T>) -> bool {
        self.find_index(&interval).defined()
    }

    /// Returns value associated with `interval` (exact match).
    /// If there is no such interval, returns `None`.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    pub fn get(&self, interval: Range<T>) -> Option<&V> {
        let index = self.find_index(&interval);
        if index.defined() {
            Some(&self.nodes[index.get()].value)
        } else {
            None
        }
    }

    /// Returns mutable value associated with `interval` (exact match).
    /// If there is no such interval, returns `None`.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
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
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    pub fn remove(&mut self, interval: Range<T>) -> Option<V> {
        self.remove_at(self.find_index(&interval))
    }

    /// Returns a range of interval keys in the map, takes *O(1)*. Returns `None` if the map is empty.
    /// `out.start` is the minimal start of all intervals in the map,
    /// and `out.end` is the maximal end of all intervals in the map.
    pub fn range(&self) -> Option<Range<T>> {
        if self.root.defined() {
            Some(self.nodes[self.root.get()].subtree_interval.to_range())
        } else {
            None
        }
    }

    fn smallest_index(&self) -> Ix {
        let mut index = self.root;
        while self.nodes[index.get()].left.defined() {
            index = self.nodes[index.get()].left;
        }
        index
    }

    fn largest_index(&self) -> Ix {
        let mut index = self.root;
        while self.nodes[index.get()].right.defined() {
            index = self.nodes[index.get()].right;
        }
        index
    }

    /// Returns the pair `(x..y, &value)` with the smallest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn smallest(&self) -> Option<(Range<T>, &V)> {
        if !self.root.defined() {
            None
        } else {
            let node = &self.nodes[self.smallest_index().get()];
            Some((node.interval.to_range(), &node.value))
        }
    }

    /// Returns the pair `(x..y, &mut value)` with the smallest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn smallest_mut(&mut self) -> Option<(Range<T>, &mut V)> {
        if !self.root.defined() {
            None
        } else {
            let index = self.smallest_index();
            let node = &mut self.nodes[index.get()];
            Some((node.interval.to_range(), &mut node.value))
        }
    }

    /// Removes the smallest interval `x..y` (in lexicographical order) from the map and returns pair `(x..y, value)`.
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn remove_smallest(&mut self) -> Option<(Range<T>, V)> {
        if !self.root.defined() {
            None
        } else {
            let index = self.smallest_index();
            let range = self.nodes[index.get()].interval.to_range();
            Some((range, self.remove_at(index).unwrap()))
        }
    }

    /// Returns the pair `(x..y, &value)` with the largest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn largest(&self) -> Option<(Range<T>, &V)> {
        if !self.root.defined() {
            None
        } else {
            let node = &self.nodes[self.largest_index().get()];
            Some((node.interval.to_range(), &node.value))
        }
    }

    /// Returns the pair `(x..y, &mut value)` with the largest interval `x..y` (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn largest_mut(&mut self) -> Option<(Range<T>, &mut V)> {
        if !self.root.defined() {
            None
        } else {
            let index = self.largest_index();
            let node = &mut self.nodes[index.get()];
            Some((node.interval.to_range(), &mut node.value))
        }
    }

    /// Removes the largest interval `x..y` (in lexicographical order) from the map and returns pair `(x..y, value)`.
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn remove_largest(&mut self) -> Option<(Range<T>, V)> {
        if !self.root.defined() {
            None
        } else {
            let index = self.largest_index();
            let range = self.nodes[index.get()].interval.to_range();
            Some((range, self.remove_at(index).unwrap()))
        }
    }

    /// Iterates over pairs `(x..y, &value)` that overlap the `query`.
    /// Takes *O(log N + K)* where *K* is the size of the output.
    /// Output is sorted by intervals, but not by values.
    ///
    /// Panics if `interval` is empty or contains a value that cannot be compared (such as `NAN`).
    pub fn iter<'a, R>(&'a self, query: R) -> Iter<'a, T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        Iter::new(self, query)
    }

    /// Iterates over intervals `x..y` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn intervals<'a, R>(&'a self, query: R) -> Intervals<'a, T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        Intervals::new(self, query)
    }

    /// Iterates over values that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn values<'a, R>(&'a self, query: R) -> Values<'a, T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        Values::new(self, query)
    }

    /// Iterator over pairs `(x..y, &mut value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn iter_mut<'a, R>(&'a mut self, query: R) -> IterMut<'a, T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        IterMut::new(self, query)
    }

    /// Iterator over *mutable* values that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn values_mut<'a, R>(&'a mut self, query: R) -> ValuesMut<'a, T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        ValuesMut::new(self, query)
    }

    /// Consumes [IntervalMap](struct.IntervalMap.html) and
    /// iterates over pairs `(x..y, value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn into_iter<R>(self, query: R) -> IntoIter<T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        IntoIter::new(self, query)
    }

    /// Consumes [IntervalMap](struct.IntervalMap.html) and
    /// iterates over pairs `(x..y, value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn into_intervals<R>(self, query: R) -> IntoIntervals<T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        IntoIntervals::new(self, query)
    }

    /// Consumes [IntervalMap](struct.IntervalMap.html) and
    /// iterates over values, for which intervals that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn into_values<R>(self, query: R) -> IntoValues<T, V, R, Ix>
    where R: RangeBounds<T>,
    {
        IntoValues::new(self, query)
    }

    /// Iterates over pairs `(x..y, &value)` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    #[inline]
    pub fn overlap<'a>(&'a self, point: T) -> Iter<'a, T, V, RangeInclusive<T>, Ix> {
        Iter::new(self, point..=point)
    }

    /// Iterates over intervals `x..y` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    #[inline]
    pub fn intervals_overlap<'a>(&'a self, point: T) -> Intervals<'a, T, V, RangeInclusive<T>, Ix> {
        Intervals::new(self, point..=point)
    }

    /// Iterates over values that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    #[inline]
    pub fn values_overlap<'a>(&'a self, point: T) -> Values<'a, T, V, RangeInclusive<T>, Ix> {
        Values::new(self, point..=point)
    }

    /// Iterator over pairs `(x..y, &mut value)` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    #[inline]
    pub fn overlap_mut<'a>(&'a mut self, point: T) -> IterMut<'a, T, V, RangeInclusive<T>, Ix> {
        IterMut::new(self, point..=point)
    }

    /// Iterates over *mutable* values that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    #[inline]
    pub fn values_overlap_mut<'a>(&'a mut self, point: T) -> ValuesMut<'a, T, V, RangeInclusive<T>, Ix> {
        ValuesMut::new(self, point..=point)
    }

    /// Creates an unsorted iterator over all pairs `(x..y, &value)`.
    /// Slightly faster than the sorted iterator, although both take *O(N)*.
    pub fn unsorted_iter<'a>(&'a self) -> UnsIter<'a, T, V, Ix> {
        UnsIter::new(self)
    }

    /// Creates an unsorted iterator over all intervals `x..y`.
    pub fn unsorted_intervals<'a>(&'a self) -> UnsIntervals<'a, T, V, Ix> {
        UnsIntervals::new(self)
    }

    /// Creates an unsorted iterator over all values `&value`.
    pub fn unsorted_values<'a>(&'a self) -> UnsValues<'a, T, V, Ix> {
        UnsValues::new(self)
    }

    /// Creates an unsorted iterator over all pairs `(x..y, &mut value)`.
    pub fn unsorted_iter_mut<'a>(&'a mut self) -> UnsIterMut<'a, T, V, Ix> {
        UnsIterMut::new(self)
    }

    /// Creates an unsorted iterator over all mutable values `&mut value`.
    pub fn unsorted_values_mut<'a>(&'a mut self) -> UnsValuesMut<'a, T, V, Ix> {
        UnsValuesMut::new(self)
    }

    /// Consumes `IntervalMap` and creates an unsorted iterator over all pairs `(x..y, value)`.
    pub fn unsorted_into_iter(self) -> UnsIntoIter<T, V, Ix> {
        UnsIntoIter::new(self)
    }

    /// Consumes `IntervalMap` and creates an unsorted iterator over all intervals `x..y`.
    pub fn unsorted_into_intervals(self) -> UnsIntoIntervals<T, V, Ix> {
        UnsIntoIntervals::new(self)
    }

    /// Consumes `IntervalMap` and creates an unsorted iterator over all values.
    pub fn unsorted_into_values(self) -> UnsIntoValues<T, V, Ix> {
        UnsIntoValues::new(self)
    }
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> IntoIterator for IntervalMap<T, V, Ix> {
    type IntoIter = IntoIter<T, V, RangeFull, Ix>;
    type Item = (Range<T>, V);

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self, ..)
    }
}

/// Construct [IntervalMap](struct.IntervalMap.html) from pairs `(x..y, value)`.
///
/// Panics, if the iterator contains duplicate intervals.
impl<T: PartialOrd + Copy, V> FromIterator<(Range<T>, V)> for IntervalMap<T, V> {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (Range<T>, V)>
    {
        let mut map = IntervalMap::new();
        for (range, value) in iter {
            assert!(map.insert(range, value).is_none(), "Cannot collect IntervalMap with duplicate intervals!");
        }
        map
    }
}

impl<T: PartialOrd + Copy, V, Ix: IndexType> Index<Range<T>> for IntervalMap<T, V, Ix> {
    type Output = V;

    fn index(&self, range: Range<T>) -> &Self::Output {
        self.get(range).expect("No entry found for range")
    }
}

impl<T, V, Ix> IntervalMap<T, V, Ix>
where T: PartialOrd + Copy + Default + AddAssign + Sub<Output = T>,
      Ix: IndexType,
{
    /// Calculates the total length of the `query` that is covered by intervals in the map.
    /// Takes *O(1)* memory and *O(log N + K)* time where *K* is the number of intervals that overlap `query`.
    ///
    /// This method makes two assumptions:
    /// - `T::default()` is equivalent to 0, which is true for numeric types,
    /// - Single-point intersections are irrelevant, for example intersection between *[0, 1]* and *[1, 2]* is zero,
    /// This also means that the size of the interval *(0, 1)* will be 1 even for integer types.
    pub fn covered_len<R>(&self, query: R) -> T
    where R: RangeBounds<T>,
    {
        let mut res = T::default();
        let start_bound = query.start_bound().cloned();
        let end_bound = query.end_bound().cloned();

        let mut started = false;
        let mut curr_start = res; // T::default(), will not be used.
        let mut curr_end = res;
        for interval in self.intervals(query) {
            let start = match start_bound {
                Bound::Unbounded => interval.start,
                Bound::Included(a) | Bound::Excluded(a) if a >= interval.start
                  => a,
                _ => interval.start,
            };
            let end = match end_bound {
                Bound::Unbounded => interval.end,
                Bound::Included(b) | Bound::Excluded(b) if b <= interval.end
                  => b,
                _ => interval.end,
            };
            debug_assert!(end >= start);

            if started {
                if start > curr_end {
                    res += curr_end - curr_start;
                    curr_start = start;
                    curr_end = end;
                } else if end > curr_end {
                    curr_end = end;
                }
            } else {
                curr_start = start;
                curr_end = end;
                started = true;
            }
        }
        if started {
            res += curr_end - curr_start;
        }
        res
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, V: Display, Ix: IndexType> IntervalMap<T, V, Ix> {
    /// Writes dot file to `writer`. `T` and `V` should implement `Display`.
    pub fn write_dot<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "digraph {{")?;
        for i in 0..self.nodes.len() {
            self.nodes[i].write_dot(i, self.colors.get(i), &mut writer)?;
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
            self.nodes[i].write_dot_without_values(i, self.colors.get(i), &mut writer)?;
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
            write!(f, "{:?} => {:?}", interval, value)?;
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
                for node in self.0.iter() {
                    seq.serialize_element(node)?;
                }
                seq.end()
            }
        }

        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&NodeVecSer(&self.nodes))?;
        tup.serialize_element(&self.colors)?;
        tup.serialize_element(&self.root)?;
        tup.end()
    }
}

// For some reason, Vec<Node> does not support deserialization. Because of that we create a newtype.
#[cfg(feature = "serde")]
struct NodeVecDe<T: PartialOrd + Copy, V, Ix: IndexType>(Vec<Node<T, V, Ix>>);

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
                let mut nodes = Vec::new();
                while let Some(node) = seq.next_element()? {
                    nodes.push(node);
                }
                Ok(NodeVecDe(nodes))
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
        let (node_vec, colors, root) = <(NodeVecDe<T, V, Ix>, BitVec, Ix)>::deserialize(deserializer)?;
        Ok(IntervalMap {
            nodes: node_vec.0,
            colors,
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
///
/// // Creates an interval map with `u8` index type (up to 255 values in the map).
/// let set = interval_map!{ [u8] 0..10 => "a", 5..15 => "b", -5..20 => "c" };
/// ```
///
/// Panics if there are duplicate intervals.
#[macro_export]
macro_rules! interval_map {
    // Create an empty interval map given the index type.
    ( [$ix:ty] $(,)? ) => ( $crate::IntervalMap::<_, _, $ix>::default() );

    // Create an empty interval map given the default index type.
    ( () ) => ( $crate::IntervalMap::new() );

    // Create a filled interval map given the index type.
    ( [$ix:ty] $(,)? $( $k:expr => $v:expr ),* $(,)? ) => {
        {
            let mut _temp_map = $crate::IntervalMap::<_, _, $ix>::default();
            $(
                assert!(_temp_map.insert($k, $v).is_none(), "Cannot use interval_map!{ ... } with duplicate intervals");
            )*
            _temp_map
        }
    };

    // Create a filled interval map with the default index type.
    ( $( $k:expr => $v:expr ),* $(,)? ) => {
        {
            let mut _temp_map = $crate::IntervalMap::new();
            $(
                assert!(_temp_map.insert($k, $v).is_none(), "Cannot use interval_map!{ ... } with duplicate intervals");
            )*
            _temp_map
        }
    };
}

/// Macros for [IntervalSet](set/struct.IntervalSet.html) creation.
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let set = interval_set!{ 100..210, 50..150 };
/// let a: Vec<_> = set.iter(..).collect();
/// assert_eq!(a, &[50..150, 100..210]);
///
/// // Creates an interval set with `u8` index type (up to 255 values in the set).
/// let set = interval_set!{ [u8] 100..210, 50..150 };
/// ```
#[macro_export]
macro_rules! interval_set {
    // Create an empty interval set given the index type.
    ( [$ix:ty] $(,)? ) => ( $crate::IntervalSet::<_, $ix>::default() );

    // Create an empty interval set given with the default index type.
    ( () ) => ( $crate::IntervalSet::new() );

    // Create a filled interval set given the index type.
    ( [$ix:ty] $(,)? $( $k:expr ),* $(,)? ) => {
        {
            let mut _temp_set = $crate::IntervalSet::<_, $ix>::default();
            $(
                _temp_set.insert($k);
            )*
            _temp_set
        }
    };

    // Create a filled interval set with the default index type.
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
