
use std::ops::{Range, RangeFull, RangeBounds, Bound};
use std::iter::FusedIterator;
use std::mem;
use bit_vec::BitVec;

use super::{IntervalMap, Node, UNDEFINED};

fn check_ordered<T: PartialOrd, R: RangeBounds<T>>(range: &R) -> bool {
    match (range.start_bound(), range.end_bound()) {
        (_, Bound::Unbounded) | (Bound::Unbounded, _) => true,
        (Bound::Included(a), Bound::Included(b)) => a <= b,
        (Bound::Included(a), Bound::Excluded(b))
        | (Bound::Excluded(a), Bound::Included(b))
        | (Bound::Excluded(a), Bound::Excluded(b)) => a < b,
    }
}

fn should_go_left<T: PartialOrd + Copy, V>(nodes: &[Node<T, V>], index: usize, start_bound: Bound<&T>) -> bool {
    if nodes[index].left == UNDEFINED {
        return false;
    }
    let left_end = nodes[nodes[index].left].subtree_interval.end;
    match start_bound {
        Bound::Included(value) | Bound::Excluded(value) => left_end >= *value,
        Bound::Unbounded => true,
    }
}

fn should_go_right<T: PartialOrd + Copy, V>(nodes: &[Node<T, V>], index: usize, end_bound: Bound<&T>) -> bool {
    if nodes[index].right == UNDEFINED {
        return false;
    }
    let right_start = nodes[nodes[index].right].subtree_interval.start;
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

fn move_to_next<T, V, R>(nodes: &[Node<T, V>], mut index: usize, range: &R, stack: &mut ActionStack) -> usize
where T: PartialOrd + Copy,
      R: RangeBounds<T>,
{
    while index != UNDEFINED {
        if stack.can_go_left() {
            while should_go_left(nodes, index, range.start_bound()) {
                stack.go_left();
                stack.push();
                index = nodes[index].left;
            }
            stack.go_left();
        }

        if stack.can_match() {
            stack.make_match();
            if nodes[index].interval.intersects_range(range) {
                return index;
            }
        }

        if stack.can_go_right() && should_go_right(nodes, index, range.end_bound()) {
            stack.go_right();
            stack.push();
            index = nodes[index].right;
        } else {
            stack.pop();
            index = nodes[index].parent;
        }
    }
    index
}


/// Macro that generates Iterator over IntervalMap.
/// Arguments: (name of the struct, Iterator::Type, self (because of the macros hygeine), output expression)
macro_rules! map_iterator {
    ($name:ident, $item:ty, $sel:ident, $out:expr) => {
        pub struct $name<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> {
            index: usize,
            range: R,
            nodes: &'a [Node<T, V>],
            stack: ActionStack,
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> $name<'a, T, V, R> {
            pub(crate) fn new(tree: &'a IntervalMap<T, V>, range: R) -> Self {
                assert!(check_ordered(&range), "Cannot iterate with an empty query");
                Self {
                    index: tree.root,
                    range,
                    nodes: &tree.nodes,
                    stack: ActionStack::new(),
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> Iterator for $name<'a, T, V, R> {
            type Item = $item;

            fn next(&mut $sel) -> Option<Self::Item> {
                $sel.index = move_to_next($sel.nodes, $sel.index, &$sel.range, &mut $sel.stack);
                if $sel.index == UNDEFINED {
                    None
                } else {
                    Some($out)
                }
            }
        }

        impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> FusedIterator for $name<'a, T, V, R> { }
    }
}

map_iterator!(Iter, (Range<T>, &'a V), self,
    (self.nodes[self.index].interval.to_range(), &self.nodes[self.index].value));
map_iterator!(Intervals, Range<T>, self, self.nodes[self.index].interval.to_range());
map_iterator!(Values, &'a V, self, &self.nodes[self.index].value);


pub struct IntoIter<T: PartialOrd + Copy, V, R: RangeBounds<T>> {
    index: usize,
    range: R,
    nodes: Vec<Node<T, V>>,
    stack: ActionStack,
}

impl<T: PartialOrd + Copy, V, R: RangeBounds<T>> IntoIter<T, V, R> {
    pub(crate) fn new(tree: IntervalMap<T, V>, range: R) -> Self {
        assert!(check_ordered(&range), "Cannot iterate with an empty query");
        let index = tree.root;
        Self {
            index,
            range,
            nodes: tree.nodes,
            stack: ActionStack::new(),
        }
    }
}

impl<T: PartialOrd + Copy, V, R: RangeBounds<T>> Iterator for IntoIter<T, V, R> {
    type Item = (Range<T>, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.index = move_to_next(&self.nodes, self.index, &self.range, &mut self.stack);
        if self.index == UNDEFINED {
            None
        } else {
            // Replace value with zeroed value, it must not be accessed anymore.
            let value = mem::replace(&mut self.nodes[self.index].value, unsafe { mem::zeroed() });
            Some((self.nodes[self.index].interval.to_range(), value))
        }
    }
}

impl<T: PartialOrd + Copy, V, R: RangeBounds<T>> FusedIterator for IntoIter<T, V, R> { }

pub struct IntoIterSet<T: PartialOrd + Copy> {
    inner: IntoIter<T, (), RangeFull>,
}

impl<T: PartialOrd + Copy> IntoIterSet<T> {
    pub(crate) fn new(tree: IntervalMap<T, ()>) -> Self {
        Self {
            inner: IntoIter::new(tree, ..),
        }
    }
}

impl<T: PartialOrd + Copy> Iterator for IntoIterSet<T> {
    type Item = Range<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(range, _)| range)
    }
}

impl<T: PartialOrd + Copy> FusedIterator for IntoIterSet<T> { }