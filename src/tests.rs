extern crate rand;
#[cfg(feature = "serde")]
extern crate serde_json;

use std::string::String;
use std::ops::{self, Range, RangeBounds, Bound};
use std::fmt::{Debug, Write};
use std::fs::File;
use std::path::Path;
use rand::prelude::*;
use bit_vec::BitVec;

use super::*;

/// Returns distance to leaves (only black nodes).
fn validate_tree_recursive<T, V, Ix>(tree: &IntervalMap<T, V, Ix>, index: Ix, upper_interval: &mut Interval<T>,
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
        Some(validate_tree_recursive(tree, left, &mut down_interval, visited))
    } else {
        None
    };
    let right_depth = if right.defined() {
        if node.is_red() {
            assert!(tree.nodes[right.get()].is_black(), "Red node {} has a red child {}", index, right);
        }
        Some(validate_tree_recursive(tree, right, &mut down_interval, visited))
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

fn validate<T: PartialOrd + Copy, V, Ix: IndexType>(tree: &IntervalMap<T, V, Ix>, size: usize) {
    assert_eq!(size, tree.len(), "Tree sizes do not match");
    assert_eq!(size > 0, tree.root.defined(), "Tree root != size");

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
    validate_tree_recursive(tree, tree.root, &mut interval, &mut visited);
    assert!(interval == node.subtree_interval, "Interval != subtree interval for node {}", tree.root);

    for i in 0..tree.len() {
        assert!(visited[i], "The tree is disjoint: node {} has no connection to the root", i);
    }
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

fn range_eq<T: PartialOrd>(a: &Range<T>, b: &Range<T>) -> bool {
    a.start_bound() == b.start_bound() && a.end_bound() == b.end_bound()
}

struct NaiveIntervalMap<T: PartialOrd + Copy, V> {
    nodes: Vec<(Range<T>, V)>,
}

impl<T: PartialOrd + Copy, V> NaiveIntervalMap<T, V> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn insert(&mut self, range: Range<T>, value: V) {
        self.nodes.push((range, value));
    }

    fn iter<'a, R: 'a + RangeBounds<T>>(&'a self, query: R) -> impl Iterator<Item = (Range<T>, &V)> + 'a {
        self.nodes.iter().filter(move |(range, _value)| intersects(range, &query))
            .map(|(range, value)| (range.clone(), value))
    }

    fn all_matching<'a>(&'a self, query: Range<T>) -> impl Iterator<Item = &V> + 'a {
        self.nodes.iter().filter(move |(range, _value)| range_eq(&range, &query)).map(|(_range, value)| value)
    }

    fn remove_random(&mut self, rng: &mut impl Rng) -> (Range<T>, V) {
        self.nodes.swap_remove(rng.gen_range(0..self.nodes.len()))
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
      F: FnMut() -> Range<T>,
{
    let mut history = String::new();
    for i in 0..n_inserts {
        let range = generator();
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

fn generate_int(a: i32, b: i32) -> impl (FnMut() -> i32) {
    let mut rng = thread_rng();
    move || rng.gen_range(a..b)
}

fn generate_float(range: f64) -> impl (FnMut() -> f64) {
    let mut rng = thread_rng();
    move || rng.gen::<f64>() * range
}

fn generate_float_rounding() -> impl (FnMut() -> f64) {
    const MULT: f64 = 1e8;
    let mut rng = thread_rng();
    move || (rng.gen::<f64>() * MULT).round() / MULT
}

fn generate_range<T: PartialOrd + Copy + Debug, F: FnMut() -> T>(mut generator: F)
        -> impl (FnMut() -> Range<T>) {
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

fn compare_match_results<T>(naive: &NaiveIntervalMap<T, u32>, tree: &IntervalMap<T, u32>, history: &str, range: Range<T>)
where T: PartialOrd + Copy + Clone + Debug
{
    let values: Vec<u32> = naive.all_matching(range.clone()).map(|v| *v).collect();
    let mut correct = true;
    correct &= values.is_empty() != tree.contains(range.clone());
    correct &= match tree.get(range.clone()) {
        Some(value) => values.contains(&value),
        None => values.is_empty(),
    };
    if !correct {
        println!("Range: {:?},   values: {:?},   tree.get: {:?}", range, values, tree.get(range.clone()));
        println!("{}", history);
        println!();
        panic!();
    }
}

fn compare_exact_matching<T, F>(naive: &NaiveIntervalMap<T, u32>, tree: &IntervalMap<T, u32>, history: &str,
    mut generator: F)
where T: PartialOrd + Copy + Debug,
      F: FnMut() -> Range<T>,
{
    for (range, _value) in &naive.nodes {
        compare_match_results(naive, tree, history, range.clone());
    }

    for _ in 0..naive.nodes.len() {
        compare_match_results(naive, tree, history, generator());
    }
}

#[cfg(feature = "serde")]
fn compare_iterators<T, I, J>(mut iter1: I, mut iter2: J, history: &str)
where
    T: PartialEq + Debug,
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
{
    loop {
        match (iter1.next(), iter2.next()) {
            (None, None) => break,
            (x, y) => if x != y {
                println!("{}", history);
                println!();
                assert_eq!(x, y);
            },
        }
    }
}

#[test]
fn test_int_inserts() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let mut generator = generate_int(0, 100);
    let history = modify_maps(&mut naive, &mut tree, COUNT, generate_range(&mut generator));

    let output_path = Path::new("tests/data/int.dot");
    let folders = output_path.parent().unwrap();
    std::fs::create_dir_all(folders).unwrap();
    let f = File::create(output_path).unwrap();
    tree.write_dot(f).unwrap();
    validate(&tree, naive.len());
    compare_extremums(&naive, &tree, &history);
    compare_exact_matching(&naive, &tree, &history, generate_range(&mut generator));

    search_rand(&mut naive, &mut tree, COUNT, generate_range(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_from(&mut generator), &history);
    search_rand(&mut naive, &mut tree, 1, generate_range_full, &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_incl(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to_incl(&mut generator), &history);
}

#[test]
fn test_float_inserts() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let mut generator = generate_float(1000.0);
    let history = modify_maps(&mut naive, &mut tree, COUNT, generate_range(&mut generator));

    let output_path = Path::new("tests/data/float.dot");
    let folders = output_path.parent().unwrap();
    std::fs::create_dir_all(folders).unwrap();
    let f = File::create(output_path).unwrap();
    tree.write_dot(f).unwrap();
    validate(&tree, naive.len());
    compare_extremums(&naive, &tree, &history);

    search_rand(&mut naive, &mut tree, COUNT, generate_range(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_from(&mut generator), &history);
    search_rand(&mut naive, &mut tree, 1, generate_range_full, &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_incl(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to_incl(&mut generator), &history);
}

#[test]
fn test_from_sorted() {
    const COUNT: u32 = 1000;
    let mut vec = Vec::new();
    let mut map: IntervalMap<_, _, u32> = IntervalMap::from_sorted(vec.clone().into_iter());
    validate(&map, 0);

    for i in 0..COUNT {
        vec.push((i..i+1, i));
        map = IntervalMap::from_sorted(vec.clone().into_iter());
        assert_eq!(map.len(), vec.len());
        validate(&map, vec.len());
    }

    for (range, value) in vec {
        assert_eq!(map.get(range), Some(&value));
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree: IntervalMap<i32, u32> = IntervalMap::new();
    let history = modify_maps(&mut naive, &mut tree, COUNT, generate_range(generate_int(0, 10000)));

    let json_path = Path::new("tests/data/serde.json");
    let folders = json_path.parent().unwrap();
    std::fs::create_dir_all(folders).unwrap();
    let output = File::create(json_path).unwrap();
    serde_json::to_writer_pretty(output, &tree).unwrap();

    let input = File::open(json_path).unwrap();
    let tree2: IntervalMap<i32, u32> = serde_json::from_reader(input).unwrap();
    compare_iterators(tree.iter(..), tree2.iter(..), &history);
}

fn removal_with_insert_chance(insert_chance: f64, count: u32) {
    let mut range_generator = generate_range(generate_float_rounding());
    let mut naive = NaiveIntervalMap::<f64, u32>::new();
    let mut tree = IntervalMap::<f64, u32>::new();

    let mut rng = thread_rng();
    for i in 0..count {
        let r = rng.gen::<f64>();
        if naive.len() == 0 || r <= insert_chance {
            let range = range_generator();
            println!("map.insert({:?}, {});", range, i);
            naive.insert(range.clone(), i);
            tree.insert(range, i);
        } else {
            let (range, value) = naive.remove_random(&mut rng);
            println!("map.remove({:?}); // -> {}", range, value);
            let rm_val = tree.remove(range.clone());
            assert_eq!(rm_val, Some(value));
            validate(&tree, naive.len());
        }
    }
    println!("-------------")
}

#[test]
fn test_removal() {
    for _ in 0..10 {
        removal_with_insert_chance(0.4, 10000);
    }
    for _ in 0..10 {
        removal_with_insert_chance(0.6, 10000);
    }
    for _ in 0..10 {
        removal_with_insert_chance(0.8, 10000);
    }
}
