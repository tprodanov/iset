extern crate bit_vec;

use std::ops::{Range, RangeBounds, Bound};
use std::cmp::{min, max, Ordering};
use std::fmt::{self, Debug, Display, Formatter};
use std::io::{self, Write};
use bit_vec::BitVec;

#[derive(Clone, Copy, Debug)]
struct CheckedOrd<T: PartialOrd>(T);

impl<T: PartialOrd> PartialEq for CheckedOrd<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.partial_cmp(&other.0).expect("partial_cmp produced None") == Ordering::Equal
    }
}

impl<T: PartialOrd> PartialEq<T> for CheckedOrd<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.0.partial_cmp(other).expect("partial_cmp produced None") == Ordering::Equal
    }
}

impl<T: PartialOrd> PartialOrd for CheckedOrd<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.partial_cmp(&other.0).expect("partial_cmp produced None"))
    }
}

impl<T: PartialOrd> PartialOrd<T> for CheckedOrd<T> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Some(self.0.partial_cmp(other).expect("partial_cmp produced None"))
    }
}

impl<T: PartialOrd> Eq for CheckedOrd<T> { }

impl<T: PartialOrd> Ord for CheckedOrd<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).expect("partial_cmp produced None")
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
struct Interval<T: PartialOrd + Copy> {
    start: CheckedOrd<T>,
    end: CheckedOrd<T>,
}

impl<T: PartialOrd + Copy + Display> Display for Interval<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}..{}", self.start.0, self.end.0)
    }
}

impl<T: PartialOrd + Copy> Interval<T> {
    fn new(range: &Range<T>) -> Self {
        Interval {
            start: CheckedOrd(range.start),
            end: CheckedOrd(range.end),
        }
    }

    fn to_range(&self) -> Range<T> {
        self.start.0..self.end.0
    }

    fn intersects(&self, other: &Interval<T>) -> bool {
        self.start < other.end && other.start < self.end
    }

    fn contains(&self, pos: T) -> bool {
        self.start <= pos && self.end > pos
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
        self.start = min(self.start, other.start);
        self.end = max(self.end, other.end);
    }
}

fn check_ordered<T: PartialOrd, R: RangeBounds<T>>(range: &R) -> bool {
    match (range.start_bound(), range.end_bound()) {
        (_, Bound::Unbounded) | (Bound::Unbounded, _) => true,
        (Bound::Included(a), Bound::Included(b)) => a <= b,
        (Bound::Included(a), Bound::Excluded(b))
        | (Bound::Excluded(a), Bound::Included(b))
        | (Bound::Excluded(a), Bound::Excluded(b)) => a < b,
    }
}

const UNDEFINED: usize = std::usize::MAX;

#[derive(Debug)]
struct Node<T: PartialOrd + Copy, V> {
    interval: Interval<T>,
    subtree_interval: Interval<T>,
    value: V,
    left: usize,
    right: usize,
    parent: usize,
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
        }
    }
}

impl<T: PartialOrd + Copy + Display, V: Display> Node<T, V> {
    fn write_dot<W: Write>(&self, index: usize, mut writer: W) -> io::Result<()> {
        writeln!(writer, "    {} [label=\"i={}\\n{}: {}\\n{}\"]",
            index, index, self.interval, self.value, self.subtree_interval)?;
        if self.left != UNDEFINED {
            writeln!(writer, "    {} -> {} [label=\"L\"]", index, self.left)?;
        }
        if self.right != UNDEFINED {
            writeln!(writer, "    {} -> {} [label=\"R\"]", index, self.right)?;
        }
        Ok(())
    }
}

pub struct IntervalMap<T: PartialOrd + Copy, V> {
    nodes: Vec<Node<T, V>>,
    root: usize,
}

impl<T: PartialOrd + Copy, V> IntervalMap<T, V> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: UNDEFINED,
        }
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

    pub fn insert(&mut self, interval: Range<T>, value: V) {
        assert!(interval.start < interval.end, "Cannot insert an empty interval");
        let j = self.nodes.len();
        let mut new_node = Node::new(interval, value);
        if self.root == UNDEFINED {
            self.root = 0;
            self.nodes.push(new_node);
            return;
        }

        let mut i = self.root;
        loop {
            self.nodes[i].subtree_interval.extend(&new_node.interval);
            let child = if new_node.interval <= self.nodes[i].interval {
                &mut self.nodes[i].left
            } else {
                &mut self.nodes[i].right
            };
            if *child == UNDEFINED {
                *child = j;
                new_node.parent = i;
                break;
            } else {
                i = *child;
            }
        }
        self.nodes.push(new_node);
    }

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

    pub fn iter<'a, R: RangeBounds<T>>(&'a self, range: R) -> Iter<'a, T, V, R> {
        assert!(check_ordered(&range), "Cannot search with an empty query");
        Iter {
            index: self.root,
            range,
            nodes: &self.nodes,
            stack: ActionStack::new(),
        }
    }
}

impl<T: PartialOrd + Copy + Display, V: Display> IntervalMap<T, V> {
    pub fn write_dot<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "digraph {{")?;
        for i in 0..self.nodes.len() {
            self.nodes[i].write_dot(i, &mut writer)?;
        }
        writeln!(writer, "}}")
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

pub struct Iter<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> {
    index: usize,
    range: R,
    nodes: &'a [Node<T, V>],
    stack: ActionStack,
}

impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> Iterator for Iter<'a, T, V, R> {
    type Item = (Range<T>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.index = move_to_next(self.nodes, self.index, &self.range, &mut self.stack);
        if self.index == UNDEFINED {
            None
        } else {
            Some((self.nodes[self.index].interval.to_range(), &self.nodes[self.index].value))
        }
    }
}

impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> std::iter::FusedIterator for Iter<'a, T, V, R> { }
