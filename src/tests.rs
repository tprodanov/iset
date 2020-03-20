extern crate rand;

use std::ops::{self, Range, RangeBounds, Bound};
use std::fmt::{Debug, Write};
use std::fs::File;
use rand::prelude::*;
use bit_vec::BitVec;

use super::*;

/// Returns distance to leaves (only black nodes).
fn check_tree_recursive<T, V, Ix>(tree: &IntervalMap<T, V, Ix>, index: Ix, upper_interval: &mut Interval<T>,
    visited: &mut BitVec) -> u32
where T: PartialOrd + Copy,
      Ix: IndexType,
{
    assert!(!visited[index.get()], "The tree contains a cycle: node {} was visited twice", index);
    visited.set(index.get(), true);

    let node = &tree.nodes[index.get()];
    let mut down_interval = node.interval.clone();
    let left = node.left;
    let right = node.right;

    let left_depth = if left.defined() {
        if node.is_red() {
            assert!(tree.nodes[left.get()].is_black(), "Red node {} has a red child {}", index, left);
        }
        Some(check_tree_recursive(tree, left, &mut down_interval, visited))
    } else {
        None
    };
    let right_depth = if right.defined() {
        if node.is_red() {
            assert!(tree.nodes[right.get()].is_black(), "Red node {} has a red child {}", index, right);
        }
        Some(check_tree_recursive(tree, right, &mut down_interval, visited))
    } else {
        None
    };
    assert!(down_interval == node.subtree_interval, "Interval != subtree interval for node {}", index);
    upper_interval.extend(&down_interval);

    match (left_depth, right_depth) {
        (Some(x), Some(y)) => assert!(x == y, "Node {} has different depths to leaves: {} != {}", index, x, y),
        _ => {},
    }
    let depth = left_depth.or(right_depth).unwrap_or(0);
    if node.is_black() {
        depth + 1
    } else {
        depth
    }
}

fn check<T: PartialOrd + Copy, V, Ix: IndexType>(tree: &IntervalMap<T, V, Ix>) {
    if !tree.root.defined() {
        assert!(tree.nodes.is_empty(), "Non empty nodes with an empty root");
        return;
    }
    for i in 0..tree.nodes.len() {
        if i == tree.root.get() {
            assert!(!tree.nodes[i].parent.defined(), "Root {} has a parent {}", i, tree.nodes[i].parent);
        } else {
            assert!(tree.nodes[i].parent.defined(), "Non-root {} has an empty parent (root is {})", i, tree.root);
        }
    }

    let node = &tree.nodes[tree.root.get()];
    let mut interval = node.interval.clone();
    let mut visited = BitVec::from_elem(tree.nodes.len(), false);
    check_tree_recursive(tree, tree.root, &mut interval, &mut visited);
    assert!(interval == node.subtree_interval, "Interval != subtree interval for node {}", tree.root);
}

struct NaiveIntervalMap<T: PartialOrd + Copy, V> {
    nodes: Vec<(Range<T>, V)>,
}

fn intersects<T: PartialOrd, R: RangeBounds<T>>(range: &Range<T>, query: &R) -> bool {
    (match query.end_bound() {
        Bound::Included(value) => value >= &range.start,
        Bound::Excluded(value) => value > &range.start,
        Bound::Unbounded => true,
    })
        &&
    (match query.start_bound() {
        Bound::Included(value) | Bound::Excluded(value) => value < &range.end,
        Bound::Unbounded => true,
    })
}

impl<T: PartialOrd + Copy, V> NaiveIntervalMap<T, V> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    fn insert(&mut self, range: Range<T>, value: V) {
        self.nodes.push((range, value));
    }

    fn iter<'a, R: 'a + RangeBounds<T>>(&'a self, query: R) -> impl Iterator<Item = (Range<T>, &V)> + 'a {
        self.nodes.iter().filter(move |(range, _value)| intersects(range, &query))
            .map(|(range, value)| (range.clone(), value))
    }
}

fn generate_ordered_pair<T: PartialOrd + Copy, F: FnMut() -> T>(generator: &mut F, forbid_eq: bool) -> (T, T) {
    let a = generator();
    let mut b = generator();
    while forbid_eq && a == b {
        b = generator();
    }
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn modify_maps<T, F>(naive: &mut NaiveIntervalMap<T, u32>, tree: &mut IntervalMap<T, u32>, n_inserts: u32,
        mut generator: F) -> String
where T: PartialOrd + Copy + Debug,
      F: FnMut() -> T,
{
    let mut history = String::new();
    for i in 0..n_inserts {
        let (a, b) = generate_ordered_pair(&mut generator, true);
        let range = a..b;
        writeln!(history, "insert({:?})", range).unwrap();
        naive.insert(range.clone(), i);
        tree.insert(range.clone(), i);
    }
    history
}

fn save_iter<'a, T, I>(history: &mut String, prefix: &str, iter: I) -> Vec<(Range<T>, u32)>
where T: PartialOrd + Copy + Debug,
      I: Iterator<Item = (Range<T>, &'a u32)>,
{
    let mut res: Vec<_> = iter.map(|(range, value)| (range, *value)).collect();
    res.sort_by(|a, b| (a.0.start, a.0.end, a.1).partial_cmp(&(b.0.start, b.0.end, b.1)).unwrap());
    writeln!(history, "{}{:?}", prefix, res).unwrap();
    res
}

fn generate_int(range: Range<i32>) -> impl (FnMut() -> i32) {
    let mut rng = thread_rng();
    move || rng.gen_range(range.start, range.end)
}

fn generate_float(range: Range<f64>) -> impl (FnMut() -> f64) {
    let mut rng = thread_rng();
    move || rng.gen_range(range.start, range.end)
}

fn generate_range<T: PartialOrd + Copy + Debug, F: FnMut() -> T>(mut generator: F)
        -> impl (FnMut() -> ops::Range<T>) {
    move || {
        let (a, b) = generate_ordered_pair(&mut generator, true);
        a..b
    }
}

fn generate_range_from<T: PartialOrd + Copy + Debug, F: FnMut() -> T>(mut generator: F)
        -> impl (FnMut() -> ops::RangeFrom<T>) {
    move || generator()..
}

fn generate_range_full() -> ops::RangeFull {
    ..
}

fn generate_range_incl<T: PartialOrd + Copy + Debug, F: FnMut() -> T>(mut generator: F)
        -> impl (FnMut() -> ops::RangeInclusive<T>) {
    move || {
        let (a, b) = generate_ordered_pair(&mut generator, false);
        a..=b
    }
}

fn generate_range_to<T: PartialOrd + Copy + Debug, F: FnMut() -> T>(mut generator: F)
        -> impl (FnMut() -> ops::RangeTo<T>) {
    move || ..generator()
}

fn generate_range_to_incl<T: PartialOrd + Copy + Debug, F: FnMut() -> T>(mut generator: F)
        -> impl (FnMut() -> ops::RangeToInclusive<T>) {
    move || ..=generator()
}

fn search_rand<T, R, F>(naive: &mut NaiveIntervalMap<T, u32>, tree: &mut IntervalMap<T, u32>, n_searches: u32,
        mut range_generator: F, history: &str)
where T: PartialOrd + Copy + Debug,
      R: RangeBounds<T> + Debug + Clone,
      F: FnMut() -> R,
{
    for _ in 0..n_searches {
        let range = range_generator();
        let mut query = format!("search({:?})", range);
        let vec_a = save_iter(&mut query, "    naive: ", naive.iter(range.clone()));
        let vec_b = save_iter(&mut query, "    tree:  ", tree.iter(range.clone()));
        if vec_a != vec_b {
            println!("{}", history);
            println!();
            println!("{}", query);
            assert!(false);
        }
    }
}

fn compare_extremums<T>(naive: &NaiveIntervalMap<T, u32>, tree: &IntervalMap<T, u32>, history: &str)
where T: PartialOrd + Copy + Debug
{
    let smallest_a = naive.nodes.iter()
        .min_by(|a, b| (a.0.start, a.0.end, a.1).partial_cmp(&(b.0.start, b.0.end, b.1)).unwrap())
        .map(|(interval, _)| interval.clone());
    let smallest_b = tree.smallest().map(|(interval, _)| interval);
    if smallest_a != smallest_b {
        println!("{}", history);
        println!();
        assert_eq!(smallest_a, smallest_b);
    }

    let largest_a = naive.nodes.iter()
        .max_by(|a, b| (a.0.start, a.0.end, a.1).partial_cmp(&(b.0.start, b.0.end, b.1)).unwrap())
        .map(|(interval, _)| interval.clone());
    let largest_b = tree.largest().map(|(interval, _)| interval);
    if largest_a != largest_b {
        println!("{}", history);
        println!();
        assert_eq!(largest_a, largest_b);
    }
}

#[test]
fn test_int_inserts() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let history = modify_maps(&mut naive, &mut tree, COUNT, generate_int(0..100));

    let f = File::create("tests/data/int.dot").unwrap();
    tree.write_dot(f).unwrap();
    check(&tree);
    compare_extremums(&naive, &tree, &history);

    search_rand(&mut naive, &mut tree, COUNT, generate_range(generate_int(0..100)), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_from(generate_int(0..100)), &history);
    search_rand(&mut naive, &mut tree, 1, generate_range_full, &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_incl(generate_int(0..100)), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to(generate_int(0..100)), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to_incl(generate_int(0..100)), &history);
}

#[test]
fn test_float_inserts() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let history = modify_maps(&mut naive, &mut tree, COUNT, generate_float(0.0..1.0));

    let f = File::create("tests/data/float.dot").unwrap();
    tree.write_dot(f).unwrap();
    check(&tree);
    compare_extremums(&naive, &tree, &history);

    search_rand(&mut naive, &mut tree, COUNT, generate_range(generate_float(0.0..1.0)), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_from(generate_float(0.0..1.0)), &history);
    search_rand(&mut naive, &mut tree, 1, generate_range_full, &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_incl(generate_float(0.0..1.0)), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to(generate_float(0.0..1.0)), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to_incl(generate_float(0.0..1.0)), &history);
}
