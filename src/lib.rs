extern crate bit_vec;

pub mod iter;
#[cfg(test)]
mod tests;

use std::ops::{Range, RangeFull, RangeBounds, Bound};
use std::cmp::{min, max, Ordering};
use std::fmt::{self, Debug, Display, Formatter};

pub use iter::*;

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

const UNDEFINED: usize = std::usize::MAX;

#[derive(Debug)]
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

#[cfg(test)]
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

    pub fn insert(&mut self, interval: Range<T>, value: V) {
        assert!(interval.start < interval.end, "Cannot insert an empty interval");
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

    pub fn iter<'a, R: RangeBounds<T>>(&'a self, range: R) -> Iter<'a, T, V, R> {
        Iter::new(self, range)
    }

    pub fn intervals<'a, R: RangeBounds<T>>(&'a self, range: R) -> Intervals<'a, T, V, R> {
        Intervals::new(self, range)
    }

    pub fn values<'a, R: RangeBounds<T>>(&'a self, range: R) -> Values<'a, T, V, R> {
        Values::new(self, range)
    }

    pub fn into_iter_within<R: RangeBounds<T>>(self, range: R) -> IntoIter<T, V, R> {
        IntoIter::new(self, range)
    }
}

impl<T: PartialOrd + Copy, V> std::iter::IntoIterator for IntervalMap<T, V> {
    type IntoIter = IntoIter<T, V, RangeFull>;
    type Item = (Range<T>, V);

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self, ..)
    }
}

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
    #[cfg(test)]
    pub(crate) fn write_dot<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "digraph {{")?;
        for i in 0..self.nodes.len() {
            self.nodes[i].write_dot(i, &mut writer)?;
        }
        writeln!(writer, "}}")
    }
}
