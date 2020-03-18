//! This crates implements map and set with interval keys (ranges `x..y`).

//! [IntervalMap](struct.IntervalMap.html) is implemented using red-black binary tree, where each node contains
//! information about the smallest start and largest end in its subtree.
//! The tree takes *O(N)* space and allows insertion in *O(log N)*.
//! [IntervalMap](struct.IntervalMap.html) allows to search for all entries overlapping a query (interval or a point,
//! output would be sorted by keys). Search takes *O(log N + K)* where *K* is the size of the output.
//! Additionally, you can extract smallest/largest interval with its value in *O(log N)*.
//!
//! [IntervalSet](struct.IntervalSet.html) is a newtype over [IntervalMap](struct.IntervalMap.html) with empty values.
//!
//! Any iterator that goes over the [IntervalMap](struct.IntervalMap.html) or [IntervalSet](struct.IntervalSet.html)
//! returns intervals/values sorted lexicographically by intervals.

// TODO:
// - smallest, largest
// - deletion
// - union, split
// - index newtype
// - macro new
// - iter_mut

extern crate bit_vec;

pub mod iter;
#[cfg(test)]
mod tests;

use std::ops::{Range, RangeFull, RangeInclusive, RangeBounds, Bound};
use std::fmt::{self, Debug, Display, Formatter};
use std::io::{self, Write};

pub use iter::*;

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

const UNDEFINED: usize = std::usize::MAX;

#[derive(Debug, Clone)]
struct Node<T: PartialOrd + Copy, V> {
    interval: Interval<T>,
    subtree_interval: Interval<T>,
    value: V,
    left: usize,
    right: usize,
    parent: usize,
    red_color: bool,
}

impl<T: PartialOrd + Copy, V> Node<T, V> {
    fn new(range: Range<T>, value: V) -> Self {
        Node {
            interval: Interval::new(&range),
            subtree_interval: Interval::new(&range),
            value,
            left: UNDEFINED,
            right: UNDEFINED,
            parent: UNDEFINED,
            red_color: true,
        }
    }

    #[inline]
    fn is_red(&self) -> bool {
        self.red_color
    }

    #[inline]
    fn is_black(&self) -> bool {
        !self.red_color
    }

    #[inline]
    fn set_red(&mut self) {
        self.red_color = true;
    }

    #[inline]
    fn set_black(&mut self) {
        self.red_color = false;
    }
}

impl<T: PartialOrd + Copy + Display, V: Display> Node<T, V> {
    fn write_dot<W: Write>(&self, index: usize, mut writer: W) -> io::Result<()> {
        writeln!(writer, "    {} [label=\"i={}\\n{}: {}\\n{}\", fillcolor={}, style=filled]",
            index, index, self.interval, self.value, self.subtree_interval,
            if self.is_red() { "salmon" } else { "grey65" })?;
        if self.left != UNDEFINED {
            writeln!(writer, "    {} -> {} [label=\"L\"]", index, self.left)?;
        }
        if self.right != UNDEFINED {
            writeln!(writer, "    {} -> {} [label=\"R\"]", index, self.right)?;
        }
        Ok(())
    }
}

fn check_interval<T: PartialOrd>(start: &T, end: &T) {
    if start < end {
        assert!(end > start, "Interval cannot be ordered (`start < end` but not `end > start`)");
    } else if end <= start {
        panic!("Interval is empty (`start >= `end`)");
    } else {
        panic!("Interval cannot be ordered (`start < end` but not `end <= start`)");
    }
}

fn check_interval_incl<T: PartialOrd>(start: &T, end: &T) {
    if start <= end {
        assert!(end >= start, "Interval cannot be ordered (`start < end` but not `end > start`)");
    } else if end < start {
        panic!("Interval is empty (`start > `end`)");
    } else {
        panic!("Interval cannot be ordered (`start <= end` but not `end < start`)");
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
/// You can use insert only intervals of type `x..y` but you can search using any query that implements `RangeBounds`,
/// for example (`x..y`, `x..=y`, `x..`, `..=y` and so on). Functions
/// [overlap](#method.overlap), [intervals_overlap](#method.intervals_overlap) and
/// [values_overlap](#method.values_overlap) allow to search for intervals/values that overlap a single point
/// (same as `x..=x`).
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
#[derive(Clone)]
pub struct IntervalMap<T: PartialOrd + Copy, V> {
    nodes: Vec<Node<T, V>>,
    root: usize,
}

impl<T: PartialOrd + Copy, V> IntervalMap<T, V> {
    /// Creates an empty [IntervalMap](struct.IntervalMap.html).
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: UNDEFINED,
        }
    }

    /// Creates an empty [IntervalMap](struct.IntervalMap.html) with `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            root: UNDEFINED,
        }
    }

    /// Shrinks inner contents.
    pub fn shrink_to_fit(&mut self) {
        self.nodes.shrink_to_fit();
    }

    fn update_subtree_interval(&mut self, index: usize) {
        let left = self.nodes[index].left;
        let right = self.nodes[index].right;

        let mut new_interval = self.nodes[index].interval.clone();
        if left != UNDEFINED {
            new_interval.extend(&self.nodes[left].subtree_interval);
        }
        if right != UNDEFINED {
            new_interval.extend(&self.nodes[right].subtree_interval);
        }
        self.nodes[index].subtree_interval = new_interval;
    }

    fn sibling(&self, index: usize) -> usize {
        let parent = self.nodes[index].parent;
        if parent == UNDEFINED {
            UNDEFINED
        } else if self.nodes[parent].left == index {
            self.nodes[parent].right
        } else {
            self.nodes[parent].left
        }
    }

    fn rotate_left(&mut self, index: usize) {
        let prev_parent = self.nodes[index].parent;
        let prev_right = self.nodes[index].right;
        debug_assert!(prev_right != UNDEFINED);

        let new_right = self.nodes[prev_right].left;
        self.nodes[index].right = new_right;
        if new_right != UNDEFINED {
            self.nodes[new_right].parent = index;
        }
        self.update_subtree_interval(index);

        self.nodes[prev_right].left = index;
        self.nodes[index].parent = prev_right;
        self.update_subtree_interval(prev_right);

        if prev_parent != UNDEFINED {
            if self.nodes[prev_parent].left == index {
                self.nodes[prev_parent].left = prev_right;
            } else {
                self.nodes[prev_parent].right = prev_right;
            }
            self.nodes[prev_right].parent = prev_parent;
            self.update_subtree_interval(prev_parent);
        } else {
            self.root = prev_right;
            self.nodes[prev_right].parent = UNDEFINED;
        }
    }

    fn rotate_right(&mut self, index: usize) {
        let prev_parent = self.nodes[index].parent;
        let prev_left = self.nodes[index].left;
        debug_assert!(prev_left != UNDEFINED);

        let new_left = self.nodes[prev_left].right;
        self.nodes[index].left = new_left;
        if new_left != UNDEFINED {
            self.nodes[new_left].parent = index;
        }
        self.update_subtree_interval(index);

        self.nodes[prev_left].right = index;
        self.nodes[index].parent = prev_left;
        self.update_subtree_interval(prev_left);

        if prev_parent != UNDEFINED {
            if self.nodes[prev_parent].right == index {
                self.nodes[prev_parent].right = prev_left;
            } else {
                self.nodes[prev_parent].left = prev_left;
            }
            self.nodes[prev_left].parent = prev_parent;
            self.update_subtree_interval(prev_parent);
        } else {
            self.root = prev_left;
            self.nodes[prev_left].parent = UNDEFINED;
        }
    }

    fn insert_repair(&mut self, mut index: usize) {
        loop {
            debug_assert!(self.nodes[index].is_red());
            if index == self.root {
                self.nodes[index].set_black();
                return;
            }

            // parent should be defined
            let parent = self.nodes[index].parent;
            if self.nodes[parent].is_black() {
                return;
            }

            // parent is red
            // grandparent should be defined
            let grandparent = self.nodes[parent].parent;
            let uncle = self.sibling(parent);

            if uncle != UNDEFINED && self.nodes[uncle].is_red() {
                self.nodes[parent].set_black();
                self.nodes[uncle].set_black();
                self.nodes[grandparent].set_red();
                index = grandparent;
                continue;
            }

            if index == self.nodes[parent].right && parent == self.nodes[grandparent].left {
                self.rotate_left(parent);
                index = self.nodes[index].left;
            } else if index == self.nodes[parent].left && parent == self.nodes[grandparent].right {
                self.rotate_right(parent);
                index = self.nodes[index].right;
            }

            let parent = self.nodes[index].parent;
            let grandparent = self.nodes[parent].parent;
            if index == self.nodes[parent].left {
                self.rotate_right(grandparent);
            } else {
                self.rotate_left(grandparent);
            }
            self.nodes[parent].set_black();
            self.nodes[grandparent].set_red();
            return;
        }
    }

    /// Inserts an interval `x..y` and its value. Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`)
    /// or contains a value that cannot be compared (such as `NAN`).
    pub fn insert(&mut self, interval: Range<T>, value: V) {
        check_interval(&interval.start, &interval.end);
        let new_ind = self.nodes.len();
        let mut new_node = Node::new(interval, value);
        if self.root == UNDEFINED {
            self.root = 0;
            new_node.set_black();
            self.nodes.push(new_node);
            return;
        }

        let mut current = self.root;
        loop {
            self.nodes[current].subtree_interval.extend(&new_node.interval);
            let child = if new_node.interval <= self.nodes[current].interval {
                &mut self.nodes[current].left
            } else {
                &mut self.nodes[current].right
            };
            if *child == UNDEFINED {
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

    #[allow(dead_code)]
    fn change_index(&mut self, old: usize, new: usize) {
        let left = self.nodes[old].left;
        let right = self.nodes[old].right;
        let parent = self.nodes[old].parent;
        if left != UNDEFINED {
            self.nodes[left].parent = new;
        }
        if right != UNDEFINED {
            self.nodes[right].parent = new;
        }
        if parent != UNDEFINED {
            if self.nodes[parent].left == old {
                self.nodes[parent].left = new;
            } else {
                self.nodes[parent].right = new;
            }
        }
    }

    // fn remove_element(&mut self, i: usize) -> V {
    //     let left = self.nodes[i].left;
    //     let right = self.nodes[i].right;
    //     let parent = self.nodes[i].parent;


    // }

    // pub fn remove(&mut self, interval: Range<T>) -> Option<V> {
    //     let interval = Interval::new(&interval);
    //     let mut i = self.root;
    //     while i != UNDEFINED {
    //         if self.nodes[i].interval == interval {
    //             return Some(self.remove_element(i));
    //         }
    //         i = if interval <= self.nodes[i].interval {
    //             self.nodes[i].left
    //         } else {
    //             self.nodes[i].right
    //         };
    //     }
    //     None
    // }

    /// Iterates over pairs `(x..y, &value)` that overlap the `query`.
    /// Takes *O(log N + K)* where *K* is the size of the output.
    /// Output is sorted by intervals, but not by values.
    ///
    /// Panics if `interval` is empty or contains a value that cannot be compared (such as `NAN`).
    pub fn iter<'a, R: RangeBounds<T>>(&'a self, query: R) -> Iter<'a, T, V, R> {
        Iter::new(self, query)
    }

    /// Iterates over intervals `x..y` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn intervals<'a, R: RangeBounds<T>>(&'a self, query: R) -> Intervals<'a, T, V, R> {
        Intervals::new(self, query)
    }

    /// Iterates over values that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn values<'a, R: RangeBounds<T>>(&'a self, query: R) -> Values<'a, T, V, R> {
        Values::new(self, query)
    }

    /// Iterator over pairs `(x..y, &mut value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn iter_mut<'a, R: RangeBounds<T>>(&'a mut self, query: R) -> IterMut<'a, T, V, R> {
        IterMut::new(self, query)
    }

    /// Iterator over *mutable* values that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn values_mut<'a, R: RangeBounds<T>>(&'a mut self, query: R) -> ValuesMut<'a, T, V, R> {
        ValuesMut::new(self, query)
    }

    /// Consumes [IntervalMap](struct.IntervalMap.html) and
    /// iterates over pairs `(x..y, value)` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn into_iter<R: RangeBounds<T>>(self, query: R) -> IntoIter<T, V, R> {
        IntoIter::new(self, query)
    }

    /// Iterates over pairs `(x..y, &value)` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap<'a>(&'a self, point: T) -> Iter<'a, T, V, RangeInclusive<T>> {
        Iter::new(self, point..=point)
    }

    /// Iterates over intervals `x..y` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn intervals_overlap<'a>(&'a self, point: T) -> Intervals<'a, T, V, RangeInclusive<T>> {
        Intervals::new(self, point..=point)
    }

    /// Iterates over values that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn values_overlap<'a>(&'a self, point: T) -> Values<'a, T, V, RangeInclusive<T>> {
        Values::new(self, point..=point)
    }

    /// Iterator over pairs `(x..y, &mut value)` that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap_mut<'a>(&'a mut self, point: T) -> IterMut<'a, T, V, RangeInclusive<T>> {
        IterMut::new(self, point..=point)
    }

    /// Iterates over *mutable* values that overlap the `point`.
    /// See [iter](#method.iter) for more details.
    pub fn values_overlap_mut<'a>(&'a mut self, point: T) -> ValuesMut<'a, T, V, RangeInclusive<T>> {
        ValuesMut::new(self, point..=point)
    }

    /// Returns the pair `(x..y, &value)` with the smallest `x..y` (intervals are sorted lexicographically).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn smallest(&self) -> Option<(Range<T>, &V)> {
        let mut index = self.root;
        if index == UNDEFINED {
            return None;
        }
        while self.nodes[index].left != UNDEFINED {
            index = self.nodes[index].left;
        }
        Some((self.nodes[index].interval.to_range(), &self.nodes[index].value))
    }

    /// Returns the pair `(x..y, &mut value)` with the smallest `x..y` (intervals are sorted lexicographically).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn smallest_mut(&mut self) -> Option<(Range<T>, &mut V)> {
        let mut index = self.root;
        if index == UNDEFINED {
            return None;
        }
        while self.nodes[index].left != UNDEFINED {
            index = self.nodes[index].left;
        }
        Some((self.nodes[index].interval.to_range(), &mut self.nodes[index].value))
    }

    /// Returns the pair `(x..y, &value)` with the largest `x..y` (intervals are sorted lexicographically).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn largest(&self) -> Option<(Range<T>, &V)> {
        let mut index = self.root;
        if index == UNDEFINED {
            return None;
        }
        while self.nodes[index].right != UNDEFINED {
            index = self.nodes[index].right;
        }
        Some((self.nodes[index].interval.to_range(), &self.nodes[index].value))
    }

    /// Returns the pair `(x..y, &mut value)` with the largest `x..y` (intervals are sorted lexicographically).
    /// Takes *O(log N)*. Returns `None` if the map is empty.
    pub fn largest_mut(&mut self) -> Option<(Range<T>, &mut V)> {
        let mut index = self.root;
        if index == UNDEFINED {
            return None;
        }
        while self.nodes[index].right != UNDEFINED {
            index = self.nodes[index].right;
        }
        Some((self.nodes[index].interval.to_range(), &mut self.nodes[index].value))
    }
}

impl<T: PartialOrd + Copy, V> std::iter::IntoIterator for IntervalMap<T, V> {
    type IntoIter = IntoIter<T, V, RangeFull>;
    type Item = (Range<T>, V);

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self, ..)
    }
}

/// Construct [IntervalMap](struct.IntervalMap.html) from pairs `(x..y, value)`.
impl<T: PartialOrd + Copy, V> std::iter::FromIterator<(Range<T>, V)> for IntervalMap<T, V> {
    fn from_iter<I: IntoIterator<Item = (Range<T>, V)>>(iter: I) -> Self {
        let mut map = IntervalMap::new();
        for (range, value) in iter {
            map.insert(range, value);
        }
        map
    }
}

impl<T: PartialOrd + Copy + Display, V: Display> IntervalMap<T, V> {
    /// Write dot file to `writer`. `T` and `V` should implement `Display`.
    pub fn write_dot<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "digraph {{")?;
        for i in 0..self.nodes.len() {
            self.nodes[i].write_dot(i, &mut writer)?;
        }
        writeln!(writer, "}}")
    }
}

impl<T: PartialOrd + Copy + Debug, V: Debug> Debug for IntervalMap<T, V> {
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

/// Set with interval keys (ranges `x..y`). Newtype over `IntervalMap<T, ()>`.
///
/// See [IntervalMap](struct.IntervalMap.html) for more information.
///
/// ```rust
/// let mut set = iset::IntervalSet::new();
/// set.insert(0.4..1.5);
/// set.insert(0.1..0.5);
/// set.insert(-1.0..0.2);
///
/// // Iterate over intervals that overlap `0.2..0.8`.
/// let a: Vec<_> = set.iter(0.2..0.8).collect();
/// assert_eq!(a, &[0.1..0.5, 0.4..1.5]);
///
/// // Iterate over intervals that overlap a point 0.5.
/// let b: Vec<_> = set.overlap(0.5).collect();
/// assert_eq!(b, &[0.4..1.5]);
///
/// // Will panic:
/// // set.insert(0.0..std::f32::NAN);
/// // set.overlap(std::f32::NAN);
///
/// // It is still possible to use infinity.
/// let inf = std::f32::INFINITY;
/// set.insert(0.0..inf);
/// let c: Vec<_> = set.overlap(0.5).collect();
/// assert_eq!(c, &[0.0..inf, 0.4..1.5]);
///
/// println!("{:?}", set);
/// // {-1.0..0.2, 0.0..inf, 0.1..0.5, 0.4..1.5}
/// ```
///
/// There are no mutable iterators over [IntervalSet](struct.IntervalSet.html) as keys should be immutable.
///
/// You can get [smallest](#method.smallest) and [largest](#method.largest) intervals in *O(log N)*.
///
/// You can construct [IntervalSet](struct.IntervalSet.html) using `collect()`:
/// ```rust
/// let set: iset::IntervalSet<_> = vec![10..20, 0..20].into_iter().collect();
/// ```
///
/// You can also construct [IntervalSet](struct.IntervalSet.html) using [interval_set](macro.interval_set.html) macro:
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let set = interval_set!{ 100..210, 50..150 };
/// let a: Vec<_> = set.iter(..).collect();
/// assert_eq!(a, &[50..150, 100..210]);
/// ```
#[derive(Clone)]
pub struct IntervalSet<T: PartialOrd + Copy> {
    inner: IntervalMap<T, ()>,
}

impl<T: PartialOrd + Copy> IntervalSet<T> {
    /// Creates an empty [IntervalSet](struct.IntervalSet.html).
    pub fn new() -> Self {
        Self {
            inner: IntervalMap::new(),
        }
    }

    /// Creates an empty [IntervalSet](struct.IntervalSet.html) with `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: IntervalMap::with_capacity(capacity),
        }
    }

    /// Shrinks inner contents.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// Inserts an interval `x..y`. Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`)
    /// or contains a value that cannot be compared (such as `NAN`).
    pub fn insert(&mut self, interval: Range<T>) {
        self.inner.insert(interval, ());
    }

    /// Iterates over intervals `x..y` that overlap the `query`.
    /// Takes *O(log N + K)* where *K* is the size of the output.
    /// Output is sorted by intervals.
    ///
    /// Panics if `interval` is empty or contains a value that cannot be compared (such as `NAN`).
    pub fn iter<'a, R: RangeBounds<T>>(&'a self, query: R) -> Intervals<'a, T, (), R> {
        self.inner.intervals(query)
    }

    /// Iterates over intervals `x..y` that overlap the `point`. Same as `iter(point..=point)`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap<'a>(&'a self, point: T) -> Intervals<'a, T, (), RangeInclusive<T>> {
        self.inner.intervals(point..=point)
    }

    /// Returns the smallest interval in the set (intervals are sorted lexicographically).
    /// Takes *O(log N)*. Returns `None` if the set is empty.
    pub fn smallest(&self) -> Option<Range<T>> {
        self.inner.smallest().map(|(interval, _)| interval)
    }

    /// Returns the largest interval in the set (intervals are sorted lexicographically).
    /// Takes *O(log N)*. Returns `None` if the set is empty.
    pub fn largest(&self) -> Option<Range<T>> {
        self.inner.largest().map(|(interval, _)| interval)
    }
}

impl<T: PartialOrd + Copy> std::iter::IntoIterator for IntervalSet<T> {
    type IntoIter = IntoIterSet<T>;
    type Item = Range<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterSet::new(self.inner)
    }
}

/// Construct [IntervalSet](struct.IntervalSet.html) from ranges `x..y`.
impl<T: PartialOrd + Copy> std::iter::FromIterator<Range<T>> for IntervalSet<T> {
    fn from_iter<I: IntoIterator<Item = Range<T>>>(iter: I) -> Self {
        let mut set = IntervalSet::new();
        for range in iter {
            set.insert(range);
        }
        set
    }
}

impl<T: PartialOrd + Copy + Debug> Debug for IntervalSet<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{{")?;
        let mut need_comma = false;
        for interval in self.iter(..) {
            if need_comma {
                write!(f, ", ")?;
            } else {
                need_comma = true;
            }
            write!(f, "{:?}", interval)?;
        }
        write!(f, "}}")
    }
}

/// Creates [IntervalMap](struct.IntervalMap.html) containing the arguments:
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let map = interval_map!{ 0..10 => "a", 5..15 => "b", -5..20 => "c" };
/// let a: Vec<_> = map.iter(..).collect();
/// assert_eq!(a, &[(-5..20, &"c"), (0..10, &"a"), (5..15, &"b")]);
/// ```
#[macro_export]
macro_rules! interval_map {
    ( $( $k:expr => $v:expr ),* ) => {
        {
            let mut _temp_map = iset::IntervalMap::new();
            $(
                _temp_map.insert($k, $v);
            )*
            _temp_map
        }
    };
}

/// Creates [IntervalSet](struct.IntervalSet.html) containing the arguments:
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let set = interval_set!{ 100..210, 50..150 };
/// let a: Vec<_> = set.iter(..).collect();
/// assert_eq!(a, &[50..150, 100..210]);
/// ```
#[macro_export]
macro_rules! interval_set {
    ( $( $k:expr ),* ) => {
        {
            let mut _temp_set = iset::IntervalSet::new();
            $(
                _temp_set.insert($k);
            )*
            _temp_set
        }
    };
}
