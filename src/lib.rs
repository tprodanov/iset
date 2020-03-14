
use std::ops::Range;
use std::cmp::{max, Ordering};

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

type OptionNode<T, V> = Option<Box<Node<T, V>>>;

struct Node<T: PartialOrd + Copy, V> {
    interval: Interval<T>,
    value: V,
    left: OptionNode<T, V>,
    right: OptionNode<T, V>,
    max_end: CheckedOrd<T>,
}

impl<T: PartialOrd + Copy, V> Node<T, V> {
    fn new(range: Range<T>, value: V) -> Self {
        Node {
            interval: Interval::new(&range),
            value,
            left: None,
            right: None,
            max_end: CheckedOrd(range.end),
        }
    }

    fn insert(&mut self, node: Box<Node<T, V>>) {
        let inner_max_end = if node.interval <= self.interval {
            if let Some(mut child) = self.left.as_mut() {
                child.insert(node);
            } else {
                self.left = Some(node);
            }
            self.left.as_ref().unwrap().max_end

        } else {
            if let Some(mut child) = self.right.as_mut() {
                child.insert(node);
            } else {
                self.right = Some(node);
            }
            self.right.as_ref().unwrap().max_end
        };
        self.max_end = max(inner_max_end, self.max_end);
    }
}

pub struct IntervalMap<T: PartialOrd + Copy, V> {
    root: OptionNode<T, V>,
}
