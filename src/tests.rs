// NOTE: Run tests with `cargo test --features std,dot,serde`.

use std::{
    println,
    string::String,
    ops::{self, Range, RangeBounds, Bound},
    fmt::{Debug, Write},
};
use rand::prelude::*;
#[cfg(feature = "serde")]
use std::{
    path::Path,
    fs::File,
};

use super::*;

/// Returns distance to leaves (only black nodes).
fn validate_tree_recursive<T, V, Ix>(tree: &IntervalMap<T, V, Ix>, index: Ix, upper_interval: &mut Interval<T>,
    visited: &mut BitVec) -> u32
where T: PartialOrd + Copy,
      Ix: IndexType,
{
    assert!(!visited.get(index.get()), "The tree contains a cycle: node {} was visited twice", index);
    visited.set(index.get(), true);

    let node = &tree.nodes[index.get()];
    let mut down_interval = node.interval.clone();
    let left = node.left;
    let right = node.right;

    let left_depth = if left.defined() {
        if tree.is_red(index) {
            assert!(tree.is_black(left), "Red node {} has a red child {}", index, left);
        }
        Some(validate_tree_recursive(tree, left, &mut down_interval, visited))
    } else {
        None
    };
    let right_depth = if right.defined() {
        if tree.is_red(index) {
            assert!(tree.is_black(right), "Red node {} has a red child {}", index, right);
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
    if tree.is_black(index) {
        depth + 1
    } else {
        depth
    }
}

fn validate<T: PartialOrd + Copy, V, Ix: IndexType>(tree: &IntervalMap<T, V, Ix>, size: usize) {
    assert_eq!(size, tree.len(), "Tree sizes do not match");
    assert_eq!(size > 0, tree.root.defined(), "Tree root != size");
    assert_eq!(tree.len(), tree.colors.len(), "Number of nodes != number of colors");

    if !tree.root.defined() {
        assert!(tree.nodes.is_empty(), "Non empty nodes with an empty root");
        return;
    }
    assert!(tree.is_black(tree.root), "Tree root needs to be black");
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
        assert!(visited.get(i), "The tree is disjoint: node {} has no connection to the root", i);
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

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
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

impl<V> NaiveIntervalMap<i32, V> {
    fn covered_len<R: RangeBounds<i32> + Clone>(&self, query: R) -> i32 {
        let mut endpoints = Vec::new();
        for (range, _value) in self.iter(query.clone()) {
            let start = match query.start_bound() {
                Bound::Unbounded => range.start,
                Bound::Included(&a) | Bound::Excluded(&a) => std::cmp::max(a, range.start),
            };
            endpoints.push((start, false));

            let end = match query.end_bound() {
                Bound::Unbounded => range.end,
                Bound::Included(&b) | Bound::Excluded(&b) => std::cmp::min(b, range.end),
            };
            endpoints.push((end, true));
        }
        endpoints.sort();
        let mut len = 0;
        let mut curr_start = std::i32::MIN;
        let mut curr_cov = 0;
        for (pos, is_end) in endpoints.into_iter() {
            assert!(pos >= curr_start);
            if is_end {
                assert!(curr_cov > 0);
                curr_cov -= 1;
                if curr_cov == 0 {
                    len += pos - curr_start;
                }
            } else {
                curr_cov += 1;
                if curr_cov == 1 {
                    curr_start = pos;
                }
            }
        }
        assert!(curr_cov == 0);
        assert!(len >= 0);
        len
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

fn random_inserts<T, F>(
    naive: &mut NaiveIntervalMap<T, u32>,
    tree: &mut IntervalMap<T, u32>,
    n_inserts: u32,
    mut generator: F,
) -> String
where T: PartialOrd + Copy + Debug,
      F: FnMut() -> Range<T>,
{
    let mut history = String::new();
    for i in 0..n_inserts {
        let range = generator();
        writeln!(history, "insert({:?})", range).unwrap();
        naive.insert(range.clone(), i);
        if let Some(value) = tree.insert(range.clone(), i) {
            let i = naive.nodes.iter().position(|(range2, _value2)| range == *range2).unwrap();
            assert_eq!(naive.nodes[i].1, value);
            naive.nodes.swap_remove(i);
        }
    }
    history
}

fn random_force_inserts<T, F>(
    naive: &mut NaiveIntervalMap<T, u32>,
    tree: &mut IntervalMap<T, u32>,
    n_inserts: u32,
    mut generator: F,
) -> String
where T: PartialOrd + Copy + Debug,
      F: FnMut() -> Range<T>,
{
    let mut history = String::new();
    for i in 0..n_inserts {
        let range = generator();
        writeln!(history, "insert({:?})", range).unwrap();
        naive.insert(range.clone(), i);
        tree.force_insert(range, i);
    }
    history
}

fn save_iter<'a, T, I>(iter: I) -> Vec<(Range<T>, u32)>
where T: PartialOrd + Copy,
      I: Iterator<Item = (Range<T>, &'a u32)>,
{
    let mut res: Vec<_> = iter.map(|(range, value)| (range, *value)).collect();
    res.sort_by(|a, b| (a.0.start, a.0.end, a.1).partial_cmp(&(b.0.start, b.0.end, b.1)).unwrap());
    res
}

fn generate_int(a: i32, b: i32) -> impl (FnMut() -> i32) {
    let mut rng = thread_rng();
    move || rng.gen_range(a..b)
}

fn generate_float(a: f64, b: f64) -> impl (FnMut() -> f64) {
    let mut rng = thread_rng();
    move || rng.gen_range(a..b)
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
        let vec_a = save_iter(naive.iter(range.clone()));
        let vec_b = save_iter(tree.iter(range.clone()));
        if vec_a != vec_b {
            println!("{}", history);
            println!();
            println!("iter({:?})", range);
            assert!(false);
        }
        if vec_a.is_empty() == tree.has_overlap(range.clone()) {
            println!("{}", history);
            println!();
            println!("has_overlap({:?})", range);
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

fn compare_match_results<T, G>(
    naive: &NaiveIntervalMap<T, u32>,
    tree: &IntervalMap<T, u32>,
    history: &str,
    range: Range<T>,
    getter: &mut G,
)
where T: PartialOrd + Copy + Clone + Debug,
      G: FnMut(&IntervalMap<T, u32>, Range<T>) -> Vec<u32>,
{
    let mut naive_vals: Vec<_> = naive.all_matching(range.clone()).map(|v| *v).collect();
    let mut tree_vals = getter(tree, range.clone());
    naive_vals.sort_unstable();
    tree_vals.sort_unstable();

    if naive_vals != tree_vals {
        println!("Range: {:?},   naive: {:?},   tree: {:?}", range, naive_vals, tree_vals);
        println!("{}", history);
        println!();
        panic!();
    }
}

fn compare_exact_matching<T, F, G>(
    naive: &NaiveIntervalMap<T, u32>,
    tree: &IntervalMap<T, u32>,
    history: &str,
    mut generator: F,
    mut getter: G,
)
where T: PartialOrd + Copy + Debug,
      F: FnMut() -> Range<T>,
      G: FnMut(&IntervalMap<T, u32>, Range<T>) -> Vec<u32>,
{
    for (range, _value) in &naive.nodes {
        compare_match_results(naive, tree, history, range.clone(), &mut getter);
    }

    for _ in 0..naive.nodes.len() {
        compare_match_results(naive, tree, history, generator(), &mut getter);
    }
}

fn check_covered_len<V, R, F>(naive: &NaiveIntervalMap<i32, V>, tree: &IntervalMap<i32, V>,
    count: u32, mut generator: F, history: &str)
where R: RangeBounds<i32> + Clone + Debug,
      F: FnMut() -> R,
{
    for _ in 0..count {
        let query = generator();
        let len1 = naive.covered_len(query.clone());
        let len2 = tree.covered_len(query.clone());
        if len1 != len2 {
            println!("{}", history);
            println!();
            println!("Query = {:?},   naive len = {},   map len = {}", query, len1, len2);
            panic!();
        }
        assert_eq!(len1, len2);
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
    let history = random_inserts(&mut naive, &mut tree, COUNT, generate_range(generate_int(20, 120)));

    validate(&tree, naive.len());
    compare_extremums(&naive, &tree, &history);

    let mut generator = generate_int(0, 140);
    compare_exact_matching(&naive, &tree, &history,
        generate_range(&mut generator), |tree, range| tree.get(range).into_iter().cloned().collect());
    search_rand(&mut naive, &mut tree, COUNT, generate_range(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_from(&mut generator), &history);
    search_rand(&mut naive, &mut tree, 1, generate_range_full, &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_incl(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to(&mut generator), &history);
    search_rand(&mut naive, &mut tree, COUNT, generate_range_to_incl(&mut generator), &history);
}

#[test]
fn test_covered_len() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let history = random_inserts(&mut naive, &mut tree, COUNT, generate_range(generate_int(-500, 500)));
    validate(&tree, naive.len());

    let mut generator = generate_int(-510, 510);
    check_covered_len(&mut naive, &mut tree, COUNT, generate_range(&mut generator), &history);
    check_covered_len(&mut naive, &mut tree, COUNT, generate_range_from(&mut generator), &history);
    check_covered_len(&mut naive, &mut tree, 1, generate_range_full, &history);
    check_covered_len(&mut naive, &mut tree, COUNT, generate_range_incl(&mut generator), &history);
    check_covered_len(&mut naive, &mut tree, COUNT, generate_range_to(&mut generator), &history);
    check_covered_len(&mut naive, &mut tree, COUNT, generate_range_to_incl(&mut generator), &history);
}

#[test]
fn test_float_inserts() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let history = random_inserts(&mut naive, &mut tree, COUNT, generate_range(generate_float(0.0, 1000.0)));

    validate(&tree, naive.len());
    compare_extremums(&naive, &tree, &history);

    let mut generator = generate_float(-50.0, 1050.0);
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
    let mut map: IntervalMap<u32, u32, u32> = IntervalMap::from_sorted([]);
    validate(&map, 0);

    let mut vec = Vec::new();
    for i in 0..COUNT {
        vec.push((i..i+1, i));
        map = IntervalMap::from_sorted(vec.clone());
        validate(&map, vec.len());
    }
    for (range, value) in vec {
        assert_eq!(map.get(range), Some(&value));
    }
}

#[test]
fn test_exact_iterators() {
    const COUNT: u32 = 5000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree = IntervalMap::new();
    let history = random_force_inserts(&mut naive, &mut tree, COUNT, generate_range(generate_int(10, 25)));
    validate(&tree, COUNT as usize);

    compare_exact_matching(&naive, &tree, &history,
        generate_range(generate_int(5, 30)),
        |tree, range| tree.values_at(range).cloned().collect());
}

#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    const COUNT: u32 = 1000;
    let mut naive = NaiveIntervalMap::new();
    let mut tree: IntervalMap<i32, u32> = IntervalMap::new();
    let history = random_inserts(&mut naive, &mut tree, COUNT, generate_range(generate_int(0, 10000)));

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
        if naive.is_empty() || r <= insert_chance {
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
