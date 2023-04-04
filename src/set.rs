//! `IntervalSet` implementation.

#[cfg(feature = "dot")]
use core::fmt::Display;
use core::fmt::{self, Debug, Formatter};
use core::iter::{FromIterator, IntoIterator};
use core::ops::{AddAssign, Range, RangeBounds, RangeFull, RangeInclusive, Sub};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "dot")]
use std::io::{self, Write};

use super::iter::*;
use super::ix::{DefaultIx, IndexType};
use super::IntervalMap;

/// Set with interval keys (ranges `x..y`). Newtype over `IntervalMap<T, ()>`.
/// See [IntervalMap](../struct.IntervalMap.html) for more information.
///
/// ```rust
/// #[macro_use] extern crate iset;
///
/// let mut set = interval_set!{ 0.4..1.5, 0.1..0.5, 5.0..7.0 };
/// assert!(set.insert(-1.0..0.2));
/// // false because the interval is already in the set.
/// assert!(!set.insert(0.1..0.5));
///
/// assert!(set.contains(5.0..7.0));
/// assert!(set.remove(5.0..7.0));
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
/// // set.insert(0.0..core::f64::NAN);
/// // set.overlap(core::f64::NAN);
///
/// // It is still possible to use infinity.
/// const INF: f64 = core::f64::INFINITY;
/// set.insert(0.0..INF);
/// let c: Vec<_> = set.overlap(0.5).collect();
/// assert_eq!(c, &[0.0..INF, 0.4..1.5]);
///
/// println!("{:?}", set);
/// // {-1.0..0.2, 0.0..inf, 0.1..0.5, 0.4..1.5}
/// assert_eq!(set.range().unwrap(), -1.0..INF);
/// assert_eq!(set.smallest().unwrap(), -1.0..0.2);
/// assert_eq!(set.largest().unwrap(), 0.4..1.5);
/// ```
///
/// There are no mutable iterators over [IntervalSet](struct.IntervalSet.html) as keys should be immutable.
///
/// In contrast to the [IntervalMap](../struct.IntervalMap.html), `IntervalSet` does not have a
/// [force_insert](../struct.IntervalMap.html#method.force_insert), and completely forbids duplicate intervals.
#[derive(Clone)]
pub struct IntervalSet<T, Ix = DefaultIx>
where
    T: PartialOrd + Copy,
    Ix: IndexType,
{
    inner: IntervalMap<T, (), Ix>,
}

impl<T: PartialOrd + Copy> IntervalSet<T> {
    /// Creates an empty [IntervalSet](struct.IntervalSet.html)
    /// with default index type [DefaultIx](../ix/type.DefaultIx.html).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: PartialOrd + Copy, Ix: IndexType> Default for IntervalSet<T, Ix> {
    fn default() -> Self {
        Self {
            inner: IntervalMap::default(),
        }
    }
}

impl<T: PartialOrd + Copy, Ix: IndexType> IntervalSet<T, Ix> {
    /// Creates an empty [IntervalSet](struct.IntervalSet.html) with `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: IntervalMap::with_capacity(capacity),
        }
    }

    /// Creates an interval set from a sorted iterator over intervals. Takes *O(N)*.
    ///
    /// Panics if the intervals are not sorted or if there are equal intervals.
    pub fn from_sorted<I>(iter: I) -> Self
    where
        I: Iterator<Item = Range<T>>,
    {
        Self {
            inner: IntervalMap::from_sorted(iter.map(|range| (range, ()))),
        }
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clears the set, removing all values. This method has no effect on the allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Shrinks inner contents.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    /// Inserts an interval `x..y` to the set. If the set did not have this interval present, true is returned.
    /// Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    pub fn insert(&mut self, interval: Range<T>) -> bool {
        self.inner.insert(interval, ()).is_none()
    }

    /// Check if the interval set contains `interval` (exact match). Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    pub fn contains(&self, interval: Range<T>) -> bool {
        self.inner.contains(interval)
    }

    /// Removes the interval from the set. Returns true if the interval was present in the set. Takes *O(log N)*.
    ///
    /// Panics if `interval` is empty (`start >= end`) or contains a value that cannot be compared (such as `NAN`).
    pub fn remove(&mut self, interval: Range<T>) -> bool {
        self.inner.remove(interval).is_some()
    }

    /// Returns a range of interval keys in the set, takes *O(1)*. Returns `None` if the set is empty.
    /// `out.start` is the minimal start of all intervals in the set,
    /// and `out.end` is the maximal end of all intervals in the set.
    #[inline]
    pub fn range(&self) -> Option<Range<T>> {
        self.inner.range()
    }

    /// Returns the smallest interval in the set (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the set is empty.
    pub fn smallest(&self) -> Option<Range<T>> {
        self.inner.smallest().map(|(interval, _)| interval)
    }

    /// Removes and returns the smallest interval in the set (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the set is empty.
    pub fn remove_smallest(&mut self) -> Option<Range<T>> {
        self.inner.remove_smallest().map(|(interval, _)| interval)
    }

    /// Returns the largest interval in the set (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the set is empty.
    pub fn largest(&self) -> Option<Range<T>> {
        self.inner.largest().map(|(interval, _)| interval)
    }

    /// Removes and returns the largest interval in the set (in lexicographical order).
    /// Takes *O(log N)*. Returns `None` if the set is empty.
    pub fn remove_largest(&mut self) -> Option<Range<T>> {
        self.inner.remove_largest().map(|(interval, _)| interval)
    }

    /// Checks, if the query overlaps any intervals in the interval set.
    /// Equivalent to `set.iter(query).next().is_some()`, but much faster.
    #[inline]
    pub fn has_overlap<R>(&self, query: R) -> bool
    where
        R: RangeBounds<T>,
    {
        self.inner.has_overlap(query)
    }

    /// Iterates over intervals `x..y` that overlap the `query`.
    /// Takes *O(log N + K)* where *K* is the size of the output.
    /// Output is sorted by intervals.
    ///
    /// Panics if `interval` is empty or contains a value that cannot be compared (such as `NAN`).
    pub fn iter<R>(&self, query: R) -> Intervals<'_, T, (), R, Ix>
    where
        R: RangeBounds<T>,
    {
        self.inner.intervals(query)
    }

    /// Iterates over intervals `x..y` that overlap the `point`. Same as `iter(point..=point)`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap(&self, point: T) -> Intervals<'_, T, (), RangeInclusive<T>, Ix> {
        self.inner.intervals(point..=point)
    }

    /// Consumes [IntervalSet](struct.IntervalSet.html) and iterates over intervals `x..y` that overlap the `query`.
    /// See [iter](#method.iter) for more details.
    pub fn into_iter<R>(self, query: R) -> IntoIntervals<T, (), R, Ix>
    where
        R: RangeBounds<T>,
    {
        IntoIntervals::new(self.inner, query)
    }

    /// Creates an unsorted iterator over all intervals `x..y`.
    /// Slightly faster than the sorted iterator, although both take *O(N)*.
    pub fn unsorted_iter(&self) -> UnsIntervals<'_, T, (), Ix> {
        UnsIntervals::new(&self.inner)
    }

    /// Consumes `IntervalSet` and creates an unsorted iterator over all intervals `x..y`.
    pub fn unsorted_into_iter(self) -> UnsIntoIntervals<T, (), Ix> {
        UnsIntoIntervals::new(self.inner)
    }
}

impl<T: PartialOrd + Copy, Ix: IndexType> IntoIterator for IntervalSet<T, Ix> {
    type IntoIter = IntoIntervals<T, (), RangeFull, Ix>;
    type Item = Range<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIntervals::new(self.inner, ..)
    }
}

/// Construct [IntervalSet](struct.IntervalSet.html) from ranges `x..y`.
impl<T: PartialOrd + Copy> FromIterator<Range<T>> for IntervalSet<T> {
    fn from_iter<I: IntoIterator<Item = Range<T>>>(iter: I) -> Self {
        let mut set = IntervalSet::new();
        for range in iter {
            set.insert(range);
        }
        set
    }
}

impl<T, Ix> IntervalSet<T, Ix>
where
    T: PartialOrd + Copy + Default + AddAssign + Sub<Output = T>,
    Ix: IndexType,
{
    /// Calculates the total length of the `query` that is covered by intervals in the map.
    /// Takes *O(log N + K)* where *K* is the number of intervals that overlap `query`.
    ///
    /// See [IntervalMap::covered_len](../struct.IntervalMap.html#method.covered_len) for more details.
    #[inline]
    pub fn covered_len<R>(&self, query: R) -> T
    where
        R: RangeBounds<T>,
    {
        self.inner.covered_len(query)
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, Ix: IndexType> IntervalSet<T, Ix> {
    /// Writes dot file to `writer`. `T` should implement `Display`.
    pub fn write_dot<W: Write>(&self, writer: W) -> io::Result<()> {
        self.inner.write_dot_without_values(writer)
    }
}

impl<T: PartialOrd + Copy + Debug, Ix: IndexType> Debug for IntervalSet<T, Ix> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{{")?;
        let mut need_comma = false;
        for interval in self.iter(..) {
            if need_comma {
                write!(f, ", ")?;
            } else {
                need_comma = true;
            }
            write!(f, "{interval:?}")?;
        }
        write!(f, "}}")
    }
}

#[cfg(feature = "serde")]
impl<T, Ix> Serialize for IntervalSet<T, Ix>
where
    T: PartialOrd + Copy + Serialize,
    Ix: IndexType + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, Ix> Deserialize<'de> for IntervalSet<T, Ix>
where
    T: PartialOrd + Copy + Deserialize<'de>,
    Ix: IndexType + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let inner = <IntervalMap<T, (), Ix>>::deserialize(deserializer)?;
        Ok(IntervalSet { inner })
    }
}
