//! Module with various iterators over `IntervalMap` and `IntervalSet`.

use alloc::vec::Vec;
use core::iter::FusedIterator;
use core::mem;
use core::ops::{Bound, RangeBounds, RangeInclusive};

use super::{check_ordered, BitVec, IndexType, Interval, IntervalMap, Node};

fn should_go_left<T, V, Ix>(nodes: &[Node<T, V, Ix>], index: Ix, start_bound: Bound<&T>) -> bool
where
    T: PartialOrd + Copy,
    Ix: IndexType,
{
    if !nodes[index.get()].left.defined() {
        return false;
    }
    let left_end = nodes[nodes[index.get()].left.get()].subtree_interval.end;
    match start_bound {
        Bound::Included(&value) | Bound::Excluded(&value) => left_end >= value,
        Bound::Unbounded => true,
    }
}

fn should_go_right<T, V, Ix>(nodes: &[Node<T, V, Ix>], index: Ix, end_bound: Bound<&T>) -> bool
where
    T: PartialOrd + Copy,
    Ix: IndexType,
{
    if !nodes[index.get()].right.defined() {
        return false;
    }
    let right_start = nodes[nodes[index.get()].right.get()].subtree_interval.start;
    match end_bound {
        Bound::Included(&value) => right_start <= value,
        Bound::Excluded(&value) => right_start < value,
        Bound::Unbounded => true,
    }
}

#[derive(Debug, Clone)]
struct ActionStack(BitVec);

impl ActionStack {
    fn new() -> Self {
        Self(BitVec::from_elem(2, false))
    }

    #[inline]
    fn push(&mut self) {
        self.0.push(false);
        self.0.push(false);
    }

    // 00 - just entered
    // 01 - was to the left
    // 10 - returned
    // 11 - was to the right

    #[inline]
    fn can_go_left(&self) -> bool {
        !self.0.get_tail(1) && !self.0.get_tail(0)
    }

    #[inline]
    fn go_left(&mut self) {
        self.0.set1(self.0.len() - 1);
    }

    #[inline]
    fn can_match(&self) -> bool {
        !self.0.get_tail(1)
    }

    #[inline]
    fn make_match(&mut self) {
        self.0.set1(self.0.len() - 2);
        self.0.set0(self.0.len() - 1);
    }

    #[inline]
    fn can_go_right(&self) -> bool {
        !self.0.get_tail(0)
    }

    #[inline]
    fn go_right(&mut self) {
        self.0.set1(self.0.len() - 2);
        self.0.set1(self.0.len() - 1);
    }

    #[inline]
    fn pop(&mut self) {
        self.0.pop();
        self.0.pop();
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn move_to_next<T, V, R, Ix>(
    nodes: &[Node<T, V, Ix>],
    mut index: Ix,
    range: &R,
    stack: &mut ActionStack,
) -> Ix
where
    T: PartialOrd + Copy,
    R: RangeBounds<T>,
    Ix: IndexType,
{
    while index.defined() {
        if stack.can_go_left() {
            while should_go_left(nodes, index, range.start_bound()) {
                stack.go_left();
                stack.push();
                index = nodes[index.get()].left;
            }
            stack.go_left();
        }

        if stack.can_match() {
            stack.make_match();
            if nodes[index.get()].interval.intersects_range(range) {
                return index;
            }
        }

        if stack.can_go_right() && should_go_right(nodes, index, range.end_bound()) {
            stack.go_right();
            stack.push();
            index = nodes[index.get()].right;
        } else {
            stack.pop();
            index = nodes[index.get()].parent;
        }
    }
    index
}

/// Macro that generates Iterator over IntervalMap.
macro_rules! iterator {
    (
        $(#[$outer:meta])*
        struct $name:ident -> $elem:ty,
        $node:ident -> $out:expr, {$( $mut_:tt )?}
    ) => {
        $(#[$outer])*
        pub struct $name<'a, T, V, R, Ix>
        where T: PartialOrd + Copy,
              R: RangeBounds<T>,
              Ix: IndexType,
        {
            pub(crate) index: Ix,
            range: R,
            nodes: &'a $( $mut_ )? [Node<T, V, Ix>],
            stack: ActionStack,
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> $name<'a, T, V, R, Ix> {
            pub(crate) fn new(tree: &'a $( $mut_ )? IntervalMap<T, V, Ix>, range: R) -> Self {
                check_ordered(&range);
                Self {
                    index: tree.root,
                    range,
                    nodes: & $( $mut_ )? tree.nodes,
                    stack: ActionStack::new(),
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> Iterator for $name<'a, T, V, R, Ix> {
            type Item = $elem;

            fn next(&mut self) -> Option<Self::Item> {
                self.index = move_to_next(self.nodes, self.index, &self.range, &mut self.stack);
                if self.index.defined() {
                    let $node = & $( $mut_ )? self.nodes[self.index.get()];
                    Some($out)
                } else {
                    None
                }
            }

            fn size_hint(& self) -> (usize, Option<usize>) {
                // Not optimal implementation, basically, always returns lower bound = 0, upper bound = map.len().
                (0, Some(self.nodes.len()))
            }
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> FusedIterator for $name<'a, T, V, R, Ix> { }
    };
}

iterator! {
    #[doc="Iterator over pairs `(x..y, &value)`."]
    #[derive(Clone, Debug)]
    struct Iter -> (RangeInclusive<T>, &'a V),
    node -> (node.interval.to_range(), &node.value), { /* no mut */ }
}

iterator! {
    #[doc="Iterator over intervals `x..y`."]
    #[derive(Clone, Debug)]
    struct Intervals -> RangeInclusive<T>,
    node -> node.interval.to_range(), { /* no mut */ }
}

#[doc = "Iterator over values."]
#[derive(Clone, Debug)]
pub struct Values<'a, T, V, R, Ix>
where
    T: PartialOrd + Copy,
    R: RangeBounds<T>,
    Ix: IndexType,
{
    pub(crate) index: Ix,
    range: R,
    nodes: &'a [Node<T, V, Ix>],
    stack: ActionStack,
}
impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> Values<'a, T, V, R, Ix> {
    pub(crate) fn new(tree: &'a IntervalMap<T, V, Ix>, range: R) -> Self {
        check_ordered(&range);
        Self {
            index: tree.root,
            range,
            nodes: &tree.nodes,
            stack: ActionStack::new(),
        }
    }
}
impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> Iterator
    for Values<'a, T, V, R, Ix>
{
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.index = move_to_next(self.nodes, self.index, &self.range, &mut self.stack);
        if self.index.defined() {
            let node = &self.nodes[self.index.get()];
            Some((&node.value))
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.nodes.len()))
    }
}
impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> FusedIterator
    for Values<'a, T, V, R, Ix>
{
}

iterator! {
    #[doc="Iterator over pairs `(x..y, &mut value)`."]
    #[derive(Debug)]
    struct IterMut -> (RangeInclusive<T>, &'a mut V),
    node -> (node.interval.to_range(), unsafe { &mut *(&mut node.value as *mut V) }), { mut }
}

iterator! {
    #[doc="Iterator over mutable values."]
    #[derive(Debug)]
    struct ValuesMut -> &'a mut V,
    node -> unsafe { &mut *(&mut node.value as *mut V) }, { mut }
}

/// Macro that generates IntoIterator over IntervalMap.
macro_rules! into_iterator {
    (
        $(#[$outer:meta])*
        struct $name:ident -> $elem:ty,
        $node:ident -> $out:expr
    ) => {
        $(#[$outer])*
        pub struct $name<T, V, R, Ix>
        where T: PartialOrd + Copy,
              R: RangeBounds<T>,
              Ix: IndexType,
        {
            index: Ix,
            range: R,
            nodes: Vec<Node<T, V, Ix>>,
            stack: ActionStack,
        }

        impl<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> $name<T, V, R, Ix> {
            pub(crate) fn new(tree: IntervalMap<T, V, Ix>, range: R) -> Self {
                check_ordered(&range);
                Self {
                    index: tree.root,
                    range,
                    nodes: tree.nodes,
                    stack: ActionStack::new(),
                }
            }
        }

        impl<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> Iterator for $name<T, V, R, Ix> {
            type Item = $elem;

            fn next(&mut self) -> Option<Self::Item> {
                self.index = move_to_next(&self.nodes, self.index, &self.range, &mut self.stack);
                if self.index.defined() {
                    let $node = &mut self.nodes[self.index.get()];
                    Some($out)
                } else {
                    None
                }
            }

            fn size_hint(& self) -> (usize, Option<usize>) {
                // Not optimal implementation, basically, always returns lower bound = 0, upper bound = map.len().
                (0, Some(self.nodes.len()))
            }
        }

        impl<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> FusedIterator for $name<T, V, R, Ix> { }
    };
}

into_iterator! {
    #[doc="Iterator over pairs `(x..y, value)`. Takes ownership of the interval map/set."]
    #[derive(Debug)]
    struct IntoIter -> (RangeInclusive<T>, V),
    node -> (node.interval.to_range(), mem::replace(&mut node.value, unsafe { mem::zeroed() }))
}

into_iterator! {
    #[doc="Iterator over intervals `x..y`. Takes ownership of the interval map/set."]
    #[derive(Debug)]
    struct IntoIntervals -> RangeInclusive<T>,
    node -> node.interval.to_range()
}

into_iterator! {
    #[doc="Iterator over values. Takes ownership of the interval map/set."]
    #[derive(Debug)]
    struct IntoValues -> V,
    node -> mem::replace(&mut node.value, unsafe { mem::zeroed() })
}

/// Macro that generates unsorted iterator over IntervalMap.
macro_rules! unsorted_iterator {
    (
        $(#[$outer:meta])*
        struct $name:ident -> $elem:ty,
        ($get_iter:ident -> $iter_type:ty),
        $node:ident -> $out:expr, {$( $mut_:tt )?}
    ) => {
        $(#[$outer])*
        pub struct $name<'a, T, V, Ix>($iter_type)
        where T: PartialOrd + Copy,
              Ix: IndexType;

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> $name<'a, T, V, Ix> {
            pub(crate) fn new(tree: &'a $( $mut_ )? IntervalMap<T, V, Ix>) -> Self {
                Self(tree.nodes.$get_iter())
            }
        }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> Iterator for $name<'a, T, V, Ix> {
            type Item = $elem;

            fn next(&mut self) -> Option<Self::Item> {
                match self.0.next() {
                    Some($node) => Some($out),
                    None => None,
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> DoubleEndedIterator for $name<'a, T, V, Ix> {
            fn next_back(&mut self) -> Option<Self::Item> {
                match self.0.next_back() {
                    Some($node) => Some($out),
                    None => None,
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> FusedIterator for $name<'a, T, V, Ix> { }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> ExactSizeIterator for $name<'a, T, V, Ix> {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }
    };
}

unsorted_iterator! {
    #[doc="Unsorted iterator over pairs `(x..y, &value)`."]
    #[derive(Clone, Debug)]
    struct UnsIter -> (RangeInclusive<T>, &'a V),
    (iter -> alloc::slice::Iter<'a, Node<T, V, Ix>>),
    node -> (node.interval.to_range(), &node.value), { /* no mut */ }
}

unsorted_iterator! {
    #[doc="Unsorted iterator over intervals `x..y`."]
    #[derive(Clone, Debug)]
    struct UnsIntervals -> RangeInclusive<T>,
    (iter -> alloc::slice::Iter<'a, Node<T, V, Ix>>),
    node -> node.interval.to_range(), { /* no mut */ }
}

unsorted_iterator! {
    #[doc="Unsorted iterator over values `&V`."]
    #[derive(Clone, Debug)]
    struct UnsValues -> &'a V,
    (iter -> alloc::slice::Iter<'a, Node<T, V, Ix>>),
    node -> &node.value, { /* no mut */ }
}

unsorted_iterator! {
    #[doc="Unsorted iterator over pairs `(x..y, &mut V)`."]
    #[derive(Debug)]
    struct UnsIterMut -> (RangeInclusive<T>, &'a mut V),
    (iter_mut -> alloc::slice::IterMut<'a, Node<T, V, Ix>>),
    node -> (node.interval.to_range(), &mut node.value), { mut }
}

unsorted_iterator! {
    #[doc="Unsorted iterator over mutable values `&mut V`."]
    #[derive(Debug)]
    struct UnsValuesMut -> &'a mut V,
    (iter_mut -> alloc::slice::IterMut<'a, Node<T, V, Ix>>),
    node -> &mut node.value, { mut }
}

/// Macro that generates unsorted IntoIterator over IntervalMap.
macro_rules! unsorted_into_iterator {
    (
        $(#[$outer:meta])*
        struct $name:ident -> $elem:ty,
        $node:ident -> $out:expr
    ) => {
        $(#[$outer])*
        pub struct $name<T, V, Ix>(alloc::vec::IntoIter<Node<T, V, Ix>>)
        where T: PartialOrd + Copy,
              Ix: IndexType;

        impl<T: PartialOrd + Copy, V, Ix: IndexType> $name<T, V, Ix> {
            pub(crate) fn new(tree: IntervalMap<T, V, Ix>) -> Self {
                Self(tree.nodes.into_iter())
            }
        }

        impl<T: PartialOrd + Copy, V, Ix: IndexType> Iterator for $name<T, V, Ix> {
            type Item = $elem;

            fn next(&mut self) -> Option<Self::Item> {
                match self.0.next() {
                    Some($node) => Some($out),
                    None => None,
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }

        impl<T: PartialOrd + Copy, V, Ix: IndexType> FusedIterator for $name<T, V, Ix> { }

        impl<T: PartialOrd + Copy, V, Ix: IndexType> ExactSizeIterator for $name<T, V, Ix> {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }
    };
}

unsorted_into_iterator! {
    #[doc="Unsorted IntoIterator over pairs `(x..y, V)`. Takes ownership of the interval map/set."]
    #[derive(Debug)]
    struct UnsIntoIter -> (RangeInclusive<T>, V),
    node -> (node.interval.to_range(), node.value)
}

unsorted_into_iterator! {
    #[doc="Unsorted IntoIterator over intervals `x..y`. Takes ownership of the interval map/set."]
    #[derive(Debug)]
    struct UnsIntoIntervals -> RangeInclusive<T>,
    node -> node.interval.to_range()
}

unsorted_into_iterator! {
    #[doc="Unsorted IntoIterator over intervals `x..y`. Takes ownership of the interval map/set."]
    #[derive(Debug)]
    struct UnsIntoValues -> V,
    node -> node.value
}

fn should_go_left_exact<T, V, Ix>(nodes: &[Node<T, V, Ix>], index: Ix, query: &Interval<T>) -> bool
where
    T: PartialOrd + Copy,
    Ix: IndexType,
{
    let node = &nodes[index.get()];
    let left_index = nodes[index.get()].left;
    left_index.defined()
        && query <= &node.interval
        && nodes[left_index.get()].subtree_interval.contains(query)
}

fn should_go_right_exact<T, V, Ix>(nodes: &[Node<T, V, Ix>], index: Ix, query: &Interval<T>) -> bool
where
    T: PartialOrd + Copy,
    Ix: IndexType,
{
    let node = &nodes[index.get()];
    let right_index = nodes[index.get()].right;
    right_index.defined()
        && query >= &node.interval
        && nodes[right_index.get()].subtree_interval.contains(query)
}

fn move_to_next_exact<T, V, Ix>(
    nodes: &[Node<T, V, Ix>],
    mut index: Ix,
    query: &Interval<T>,
    stack: &mut ActionStack,
) -> Ix
where
    T: PartialOrd + Copy,
    Ix: IndexType,
{
    while !stack.is_empty() && index.defined() {
        if stack.can_go_left() {
            while should_go_left_exact(nodes, index, query) {
                stack.go_left();
                stack.push();
                index = nodes[index.get()].left;
            }
            stack.go_left();
        }

        if stack.can_match() {
            stack.make_match();
            if query == &nodes[index.get()].interval {
                return index;
            }
        }

        if stack.can_go_right() && should_go_right_exact(nodes, index, query) {
            stack.go_right();
            stack.push();
            index = nodes[index.get()].right;
        } else {
            stack.pop();
            index = nodes[index.get()].parent;
        }
    }
    Ix::MAX
}

/// Macro that generates iterator over exactly matching intervals in the IntervalMap.
macro_rules! iterator_exact {
    (
        $(#[$outer:meta])*
        struct $name:ident -> $elem:ty,
        $node:ident -> $out:expr, {$( $mut_:tt )?}
    ) => {
        $(#[$outer])*
        pub struct $name<'a, T, V, Ix>
        where T: PartialOrd + Copy,
              Ix: IndexType,
        {
            pub(crate) index: Ix,
            query: Interval<T>,
            nodes: &'a $( $mut_ )? [Node<T, V, Ix>],
            stack: ActionStack,
        }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> $name<'a, T, V, Ix> {
            pub(crate) fn new(tree: &'a $( $mut_ )? IntervalMap<T, V, Ix>, query: Interval<T>) -> Self {
                Self {
                    index: tree.find_index(&query),
                    query,
                    nodes: & $( $mut_ )? tree.nodes,
                    stack: ActionStack::new(),
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> Iterator for $name<'a, T, V, Ix> {
            type Item = $elem;

            fn next(&mut self) -> Option<Self::Item> {
                self.index = move_to_next_exact(self.nodes, self.index, &self.query, &mut self.stack);
                if self.index.defined() {
                    let $node = & $( $mut_ )? self.nodes[self.index.get()];
                    Some($out)
                } else {
                    None
                }
            }

            fn size_hint(& self) -> (usize, Option<usize>) {
                // Not optimal implementation, basically, always returns lower bound = 0, upper bound = map.len().
                (0, Some(self.nodes.len()))
            }
        }

        impl<'a, T: PartialOrd + Copy, V, Ix: IndexType> FusedIterator for $name<'a, T, V, Ix> { }
    };
}

iterator_exact! {
    #[doc="Iterator over values `&V` for exact matches with the query."]
    #[derive(Clone, Debug)]
    struct ValuesExact -> &'a V,
    node -> &node.value, { /* no mut */ }
}

iterator_exact! {
    #[doc="Iterator over mutable values `&mut V` for exact matches with the query."]
    #[derive(Debug)]
    struct ValuesExactMut -> &'a mut V,
    node -> unsafe { &mut *(&mut node.value as *mut V) }, { mut }
}
