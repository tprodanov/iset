use std::ops::Range;
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
}

const UNDEFINED: usize = std::usize::MAX;

struct Node<T: PartialOrd + Copy, V> {
    interval: Interval<T>,
    value: V,
    left: usize,
    right: usize,
    biggest_end: CheckedOrd<T>,
}

impl<T: PartialOrd + Copy, V> Node<T, V> {
    fn new(range: Range<T>, value: V) -> Self {
        Node {
            interval: Interval::new(&range),
            value,
            left: UNDEFINED,
            right: UNDEFINED,
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
        self.nodes.push(Node::new(interval, value));
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
                break;
            } else {
                j = *child;
            }
        }
        self.update_biggest_end(i);
    }
}
