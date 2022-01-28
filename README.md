This crates implements map and set with interval keys (ranges `x..y`).
`IntervalMap` is implemented using red-black binary tree,
where each node contains information about the smallest start and largest end in its subtree.
The tree takes *O(N)* space and allows insertion, removal and search in *O(log N)*.
`IntervalMap` allows to search for all entries overlapping a query (interval or a point, output would be sorted by keys)
in *O(log N + K)* where *K* is the size of the output.

`IntervalSet` is a newtype over `IntervalMap` with empty values.

## Usage

The following code constructs a small interval map and search for intervals/values overlapping various queries.

```rust
#[macro_use] extern crate iset;

let mut map = interval_map!{ 20..30 => 'a', 15..25 => 'b', 10..20 => 'c' };
assert_eq!(map.insert(10..20, 'd'), Some('c'));
assert_eq!(map.insert(5..15, 'e'), None);

// Iterator over all pairs (range, value). Output is sorted.
let a: Vec<_> = map.iter(..).collect();
assert_eq!(a, &[(5..15, &'e'), (10..20, &'d'), (15..25, &'b'), (20..30, &'a')]);

// Iterate over intervals that overlap query (..20 here). Output is sorted.
let b: Vec<_> = map.intervals(..20).collect();
assert_eq!(b, &[5..15, 10..20, 15..25]);

assert_eq!(map[15..25], 'b');
// Replace 15..25 => 'b' into 'z'.
*map.get_mut(15..25).unwrap() = 'z';

// Iterate over values that overlap query (20.. here). Output is sorted by intervals.
let c: Vec<_> = map.values(20..).collect();
assert_eq!(c, &[&'z', &'a']);

// Remove 10..20 => 'd'.
assert_eq!(map.remove(10..20), Some('d'));

println!("{:?}", map);
// {5..15 => 'e', 15..25 => 'z', 20..30 => 'a'}
```

You can find more detailed usage [here](https://docs.rs/iset).

## Changelog
You can find changelog [here](https://github.com/tprodanov/iset/releases).

## Issues
Please submit issues [here](https://github.com/tprodanov/iset/issues) or send them to `timofey.prodanov[at]gmail.com`.
