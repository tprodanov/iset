//! Module with various iterators over `IntervalMap` and `IntervalSet`.

use core::ops::{Range, RangeFull, RangeBounds, Bound};
use core::iter::FusedIterator;
use core::mem;
use bit_vec::BitVec;

use super::{IntervalMap, Node, IndexType, check_interval, check_interval_incl};

fn check_ordered<T: PartialOrd, R: RangeBounds<T>>(range: &R) {
    match (range.start_bound(), range.end_bound()) {
        (_, Bound::Unbounded) | (Bound::Unbounded, _) => {},
        (Bound::Included(a), Bound::Included(b)) => check_interval_incl(a, b),
        (Bound::Included(a), Bound::Excluded(b))
        | (Bound::Excluded(a), Bound::Included(b))
        | (Bound::Excluded(a), Bound::Excluded(b)) => check_interval(a, b),
    }
}

fn should_go_left<T, V, Ix>(nodes: &[Node<T, V, Ix>], index: Ix, start_bound: Bound<&T>) -> bool
where T: PartialOrd + Copy,
      Ix: IndexType,
{
    if !nodes[index.get()].left.defined() {
        return false;
    }
    let left_end = nodes[nodes[index.get()].left.get()].subtree_interval.end;
    match start_bound {
        Bound::Included(value) | Bound::Excluded(value) => left_end >= *value,
        Bound::Unbounded => true,
    }
}

fn should_go_right<T, V, Ix>(nodes: &[Node<T, V, Ix>], index: Ix, end_bound: Bound<&T>) -> bool
where T: PartialOrd + Copy,
      Ix: IndexType,
{
    if !nodes[index.get()].right.defined() {
        return false;
    }
    let right_start = nodes[nodes[index.get()].right.get()].subtree_interval.start;
    match end_bound {
        Bound::Included(value) => right_start <= *value,
        Bound::Excluded(value) => right_start < *value,
        Bound::Unbounded => true,
    }
}

#[derive(Debug)]
struct ActionStack(BitVec);

impl ActionStack {
    fn new() -> Self {
        Self(BitVec::from_elem(2, false))
    }

    #[inline]
    fn push(& mut self) {
        self.0.push(false);
        self.0.push(false);
    }

    // 00 - just entered
    // 01 - was to the left
    // 10 - returned
    // 11 - was to the right

    #[inline]
    fn can_go_left(&self) -> bool {
        !self.0[self.0.len() - 2] && !self.0[self.0.len() - 1]
    }

    #[inline]
    fn go_left(&mut self) {
        self.0.set(self.0.len() - 1, true);
    }

    #[inline]
    fn can_match(&self) -> bool {
        !self.0[self.0.len() - 2]
    }

    #[inline]
    fn make_match(&mut self) {
        self.0.set(self.0.len() - 2, true);
        self.0.set(self.0.len() - 1, false);
    }

    #[inline]
    fn can_go_right(&self) -> bool {
        !self.0[self.0.len() - 1]
    }

    #[inline]
    fn go_right(&mut self) {
        self.0.set(self.0.len() - 2, true);
        self.0.set(self.0.len() - 1, true);
    }

    #[inline]
    fn pop(&mut self) {
        self.0.pop();
        self.0.pop();
    }
}

fn move_to_next<T, V, R, Ix>(nodes: &[Node<T, V, Ix>], mut index: Ix, range: &R, stack: &mut ActionStack) -> Ix
where T: PartialOrd + Copy,
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
        struct $name:ident -> $item:ty,
        $self:ident -> $out:expr, {$( $mut_:tt )*}
    ) => {
        $(#[$outer])*
        pub struct $name<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> {
            index: Ix,
            range: R,
            nodes: &'a $( $mut_ )* [Node<T, V, Ix>],
            stack: ActionStack,
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> $name<'a, T, V, R, Ix> {
            pub(crate) fn new(tree: &'a $( $mut_ )* IntervalMap<T, V, Ix>, range: R) -> Self {
                check_ordered(&range);
                Self {
                    index: tree.root,
                    range,
                    nodes: & $( $mut_ )* tree.nodes,
                    stack: ActionStack::new(),
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> Iterator for $name<'a, T, V, R, Ix> {
            type Item = $item;

            fn next(&mut $self) -> Option<Self::Item> {
                $self.index = move_to_next($self.nodes, $self.index, &$self.range, &mut $self.stack);
                if !$self.index.defined() {
                    None
                } else {
                    Some($out)
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> FusedIterator for $name<'a, T, V, R, Ix> { }
    };
}

iterator! {
    #[doc="Iterator over pairs `(x..y, &value)`."]
    struct Iter -> (Range<T>, &'a V),
    self -> (self.nodes[self.index.get()].interval.to_range(), &self.nodes[self.index.get()].value), { /* no mut */ }
}

iterator! {
    #[doc="Iterator over intervals `x..y`."]
    struct Intervals -> Range<T>,
    self -> self.nodes[self.index.get()].interval.to_range(), { /* no mut */ }
}

iterator! {
    #[doc="Iterator over values."]
    struct Values -> &'a V,
    self -> &self.nodes[self.index.get()].value, { /* no mut */ }
}

iterator! {
    #[doc="Iterator over pairs `(x..y, &mut value)`."]
    struct IterMut -> (Range<T>, &'a mut V),
    self -> (self.nodes[self.index.get()].interval.to_range(),
        unsafe { &mut *(&mut self.nodes[self.index.get()].value as *mut V) }), { mut }
}

iterator! {
    #[doc="Iterator over mutable values."]
    struct ValuesMut -> &'a mut V,
    self -> unsafe { &mut *(&mut self.nodes[self.index.get()].value as *mut V) }, { mut }
}

/// Iterator over pairs `(x..y, value)`. Takes ownership of `IntervalMap`.
pub struct IntoIter<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> {
    index: Ix,
    range: R,
    nodes: Vec<Node<T, V, Ix>>,
    stack: ActionStack,
}

impl<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> IntoIter<T, V, R, Ix> {
    pub(crate) fn new(tree: IntervalMap<T, V, Ix>, range: R) -> Self {
        check_ordered(&range);
        let index = tree.root;
        Self {
            index,
            range,
            nodes: tree.nodes,
            stack: ActionStack::new(),
        }
    }
}

impl<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> Iterator for IntoIter<T, V, R, Ix> {
    type Item = (Range<T>, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.index = move_to_next(&self.nodes, self.index, &self.range, &mut self.stack);
        if !self.index.defined() {
            None
        } else {
            // Replace value with zeroed value, it must not be accessed anymore.
            let value = mem::replace(&mut self.nodes[self.index.get()].value, unsafe { mem::zeroed() });
            Some((self.nodes[self.index.get()].interval.to_range(), value))
        }
    }
}

impl<T: PartialOrd + Copy, V, R: RangeBounds<T>, Ix: IndexType> FusedIterator for IntoIter<T, V, R, Ix> { }

/// Iterator over pairs `x..y`. Takes ownership of `IntervalSet`.
pub struct IntoIterSet<T: PartialOrd + Copy, Ix: IndexType> {
    inner: IntoIter<T, (), RangeFull, Ix>,
}

impl<T: PartialOrd + Copy, Ix: IndexType> IntoIterSet<T, Ix> {
    pub(crate) fn new(tree: IntervalMap<T, (), Ix>) -> Self {
        Self {
            inner: IntoIter::new(tree, ..),
        }
    }
}

impl<T: PartialOrd + Copy, Ix: IndexType> Iterator for IntoIterSet<T, Ix> {
    type Item = Range<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(range, _)| range)
    }
}

impl<T: PartialOrd + Copy, Ix: IndexType> FusedIterator for IntoIterSet<T, Ix> { }
