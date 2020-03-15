use std::ops::{Range, RangeBounds, Bound};
use std::cmp::{max, Ordering};
use std::mem::drop;

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
}

const UNDEFINED: usize = std::usize::MAX;

struct Node<T: PartialOrd + Copy, V> {
    interval: Interval<T>,
    value: V,
    left: usize,
    right: usize,
    parent: usize,
    biggest_end: CheckedOrd<T>,
}

impl<T: PartialOrd + Copy, V> Node<T, V> {
    fn new(range: Range<T>, value: V) -> Self {
        Node {
            interval: Interval::new(&range),
            value,
            left: UNDEFINED,
            right: UNDEFINED,
            parent: UNDEFINED,
            biggest_end: CheckedOrd(range.end),
        }
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

    fn update_biggest_end(&mut self, node_ind: usize) {
        let mut node = &self.nodes[node_ind];
        let mut biggest_end = node.interval.end;
        if node.left != UNDEFINED {
            biggest_end = max(biggest_end, self.nodes[node.left].biggest_end);
        }
        if node.right != UNDEFINED {
            biggest_end = max(biggest_end, self.nodes[node.right].biggest_end);
        }
        drop(node);
        self.nodes[node_ind].biggest_end = biggest_end;
    }

    pub fn insert(&mut self, interval: Range<T>, value: V) {
        let i = self.nodes.len();
        let mut new_node = Node::new(interval, value);
        if self.root == UNDEFINED {
            self.root = 0;
            return;
        }

        let mut j = self.root;
        loop {
            let mut child = if self.nodes[i].interval <= self.nodes[j].interval {
                &mut self.nodes[j].left
            } else {
                &mut self.nodes[j].right
            };
            if *child == UNDEFINED {
                *child = i;
                new_node.parent = j;
                break;
            } else {
                j = *child;
            }
        }
        self.nodes.push(new_node);
        self.update_biggest_end(i);
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
        let mut iter = Iter {
            index: self.root,
            range: range,
            nodes: &self.nodes,
        };
        iter.move_to_next();
        iter
    }
}

pub struct Iter<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> {
    index: usize,
    range: R,
    nodes: &'a [Node<T, V>],
}

impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> Iter<'a, T, V, R> {
    fn should_go_left(&self) -> bool {
        if self.nodes[self.index].left == UNDEFINED {
            return false;
        }
        let left_end = self.nodes[self.nodes[self.index].left].biggest_end;
        match self.range.start_bound() {
            Bound::Included(value) | Bound::Excluded(value) => left_end >= *value,
            Bound::Unbounded => true,
        }
    }

    fn should_go_right(&self) -> bool {
        if self.nodes[self.index].right == UNDEFINED {
            return false;
        }
        let right_start = self.nodes[self.nodes[self.index].right].interval.start;
        match self.range.end_bound() {
            Bound::Included(value) => right_start <= *value,
            Bound::Excluded(value) => right_start < *value,
            Bound::Unbounded => true,
        }
    }

    fn move_to_next(&mut self) {
        let mut was_left = false;
        while self.index == UNDEFINED {
            while !was_left && self.should_go_left() {
                self.index = self.nodes[self.index].left;
            }
            if self.nodes[self.index].interval.intersects_range(&self.range) {
                return;
            }
            if self.should_go_right() {
                was_left = false;
                self.index = self.nodes[self.index].right;
            } else {
                was_left = true;
                self.index = self.nodes[self.index].parent;
            }
        }
    }
}

impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> Iterator for Iter<'a, T, V, R> {
    type Item = (Range<T>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == UNDEFINED {
            return None;
        }
        let res = (self.nodes[self.index].interval.to_range(), &self.nodes[self.index].value);
        self.move_to_next();
        Some(res)
    }
}

impl<'a, T: PartialOrd + Copy, V, R: RangeBounds<T>> std::iter::FusedIterator for Iter<'a, T, V, R> { }
