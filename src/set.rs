use core::ops::{Range, RangeInclusive, RangeBounds};
use core::fmt::{self, Debug, Display, Formatter};
#[cfg(feature = "dot")]
use std::io::{self, Write};
#[cfg(feature = "serde")]
use {
    core::marker::PhantomData,
    serde::{Serialize, Serializer, Deserialize, Deserializer},
    serde::ser::{SerializeTuple, SerializeSeq},
    serde::de::{Visitor, SeqAccess},
};

use super::IntervalMap;
use super::ix::{IndexType, DefaultIx};
use super::iter::*;

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
/// // set.insert(0.0..core::f32::NAN);
/// // set.overlap(core::f32::NAN);
///
/// // It is still possible to use infinity.
/// let inf = core::f32::INFINITY;
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
///
/// # Index types:
/// You can specify [index type](trait.IndexType.html) (for example `u32` and `u64`) used in the inner
/// representation of `IntervalSet`.
///
/// Method [new](#method.new), macro [interval_map](macro.interval_map.html) or function
/// `collect()` create `IntervalSet` with index type `u32`. If you wish to use another index type you can use
/// methods [default](#method.default) or [with_capacity](#method.with_capacity). For example:
/// ```rust
/// let mut set: iset::IntervalSet<_, u64> = iset::IntervalSet::default();
/// set.insert(10..20);
/// ```
/// See [IndexType](trait.IndexType.html) for details.
#[derive(Clone)]
pub struct IntervalSet<T: PartialOrd + Copy, Ix: IndexType = DefaultIx> {
    inner: IntervalMap<T, (), Ix>,
}

impl<T: PartialOrd + Copy> IntervalSet<T> {
    /// Creates an empty [IntervalSet](struct.IntervalSet.html).
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

    /// Creates an interval set from a sorted iterator over intervals. Takes $O(n)$.
    pub fn from_sorted<I: Iterator<Item = Range<T>>>(iter: I) -> Self {
        Self {
            inner: IntervalMap::from_sorted(iter.map(|range| (range, ()))),
        }
    }

    /// Returns number of elements in the set.
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
    pub fn iter<'a, R: RangeBounds<T>>(&'a self, query: R) -> Intervals<'a, T, (), R, Ix> {
        self.inner.intervals(query)
    }

    /// Iterates over intervals `x..y` that overlap the `point`. Same as `iter(point..=point)`.
    /// See [iter](#method.iter) for more details.
    pub fn overlap<'a>(&'a self, point: T) -> Intervals<'a, T, (), RangeInclusive<T>, Ix> {
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

    /// Check if the interval set contains `interval` (exact match).
    pub fn contains(&self, interval: Range<T>) -> bool {
        self.inner.contains(interval)
    }
}

impl<T: PartialOrd + Copy, Ix: IndexType> core::iter::IntoIterator for IntervalSet<T, Ix> {
    type IntoIter = IntoIterSet<T, Ix>;
    type Item = Range<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterSet::new(self.inner)
    }
}

/// Construct [IntervalSet](struct.IntervalSet.html) from ranges `x..y`.
impl<T: PartialOrd + Copy> core::iter::FromIterator<Range<T>> for IntervalSet<T> {
    fn from_iter<I: IntoIterator<Item = Range<T>>>(iter: I) -> Self {
        let mut set = IntervalSet::new();
        for range in iter {
            set.insert(range);
        }
        set
    }
}

#[cfg(feature = "dot")]
impl<T: PartialOrd + Copy + Display, Ix: IndexType> IntervalSet<T, Ix> {
    /// Write dot file to `writer`. `T` should implement `Display`.
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
            write!(f, "{:?}", interval)?;
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