This crates implements map and set with interval keys (ranges `x..y`).

`IntervalMap` is implemented using red-black binary tree, where each node contains
information about the smallest start and largest end in its subtree.
The tree takes *O(N)* space and allows insertion in *O(log N)*.
`IntervalMap` allows to search for all entries overlapping a query (interval or a point,
output would be sorted by keys). Search takes *O(log N + K)* where *K* is the size of the output.
Additionally, you can extract smallest/largest interval with its value in *O(log N)*.

`IntervalSet` is a newtype over `IntervalMap` with empty values.

Any iterator that goes over the `IntervalMap` or `IntervalSet`
returns intervals/values sorted lexicographically by intervals.

## Usage

The following code constructs a small interval map and search for intervals/values overlapping various queries.

```rust
let mut map = iset::IntervalMap::new();
map.insert(20..30, "a");
map.insert(15..25, "b");
map.insert(10..20, "c");

// Iterate over (interval, &value) pairs that overlap query (.. here).
// Output is sorted by intervals.
let a: Vec<_> = map.iter(..).collect();
assert_eq!(a, &[(10..20, &"c"), (15..25, &"b"), (20..30, &"a")]);

// Iterate over intervals that overlap query (..20 here). Output is sorted.
let b: Vec<_> = map.intervals(..20).collect();
assert_eq!(b, &[10..20, 15..25]);

// Iterate over values that overlap query (20.. here). Output is sorted by intervals.
let c: Vec<_> = map.values(20..).collect();
assert_eq!(c, &[&"b", &"a"]);

println!("{:?}", map);
// {10..20: "c", 15..25: "b", 20..30: "a"}
```

You can find more detailed usage [here](https://docs.rs/iset).

## Changelog
You can find changelog [here](https://gitlab.com/tprodanov/iset/-/releases).

## Issues
Please submit issues [here](https://gitlab.com/tprodanov/iset/issues) or send them to
`timofey.prodanov[at]gmail.com`.
