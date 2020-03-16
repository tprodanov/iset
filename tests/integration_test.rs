extern crate iset;
extern crate rand;

use std::ops::{Range, RangeBounds, Bound};
use std::fmt::{Debug, Write};
use std::fs::File;
use rand::prelude::*;

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

fn generate_ordered_pair<T: PartialOrd + Copy, F: FnMut() -> T>(generator: &mut F) -> (T, T) {
    let a = generator();
    let mut b = generator();
    while a == b {
        b = generator();
    }
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn modify_maps<T, F>(naive: &mut NaiveIntervalMap<T, u32>, tree: &mut iset::IntervalMap<T, u32>, n_inserts: u32,
        mut generator: F) -> String
where T: PartialOrd + Copy + Debug,
      F: FnMut() -> T,
{
    let mut history = String::new();
    for i in 0..n_inserts {
        let (a, b) = generate_ordered_pair(&mut generator);
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

fn generate_int(range: Range<u32>) -> impl (FnMut() -> u32) {
    let mut rng = thread_rng();
    move || rng.gen_range(range.start, range.end)
}

fn search_rand<T, F>(naive: &mut NaiveIntervalMap<T, u32>, tree: &mut iset::IntervalMap<T, u32>, n_searches: u32,
        mut generator: F, history: &mut String)
where T: PartialOrd + Copy + Debug,
F: FnMut() -> T,
{
    for _ in 0..n_searches {
        let (a, b) = generate_ordered_pair(&mut generator);
        let range = a..b;
        writeln!(history, "search({:?})", range).unwrap();
        let vec_a = save_iter(history, "    naive: ", naive.iter(range.clone()));
        let vec_b = save_iter(history, "    tree:  ", tree.iter(range.clone()));
        if vec_a != vec_b {
            println!("{}", history);
            assert!(false);
        }
    }
}

#[test]
fn test_inserts() {
    let mut naive = NaiveIntervalMap::new();
    let mut tree = iset::IntervalMap::new();
    let mut history = modify_maps(&mut naive, &mut tree, 100, generate_int(0..100));
    let f = File::create("tests/data/out.dot").unwrap();
    tree.write_dot(f).unwrap();
    search_rand(&mut naive, &mut tree, 10, generate_int(0..100), &mut history);
}
