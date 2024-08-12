//! Wrapper around integer types, used as indices within `IntervalMap` and `IntervalSet`.

use core::fmt::Display;
use core::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};

macro_rules! index_error {
    (u64) => {
        "Failed to insert a new element into IntervalMap/Set: number of elements is too large for u64."
    };
    ($name:ident) => {
        concat!(
            "Failed to insert a new element into IntervalMap/Set: number of elements is too large for ",
            stringify!($name),
            ", try using u64.")
    };
}

/// Default index type.
pub type DefaultIx = u32;

/// Trait for index types: used in the inner representation of [IntervalMap](../struct.IntervalMap.html) and
/// [IntervalSet](../set/struct.IntervalSet.html).
///
/// Implemented for `u8`, `u16`, `u32`, `u64` and `usize`,
/// `u32` is used by default ([DefaultIx](type.DefaultIx.html)).
///
/// `IntervalMap` or `IntervalSet` can store up to `Ix::MAX - 1` elements
/// (for example `IntervalMap<_, _, u8>` can store up to 255 items).
///
/// Using smaller index types saves memory and slightly reduces running time.
pub trait IndexType: Copy + Display + Sized + Eq {
    type T: Copy + Display + Sized + Eq;

    fn new(val: usize) -> Self::T;

    fn get(val: Self::T) -> usize;
}

macro_rules! impl_index {
    ($primitive:ident, $nonzero:ident) => {
        impl IndexType for $primitive {
            type T = $nonzero;

            /// Creates a new index. Panics if `val` is too big.
            fn new(val: usize) -> Self::T {
                val.try_into().ok()
                    .and_then(|v: $primitive| Self::T::new(v ^ $primitive::MAX))
                    .expect(index_error!($primitive))
            }

            /// Converts index into `usize`.
            fn get(val: Self::T) -> usize {
                (val.get() ^ $primitive::MAX)
                    .try_into()
                    .expect(concat!("usize is too small for ", stringify!($primitive), " type."))
            }
        }
    };
}

impl_index!(u8, NonZeroU8);
impl_index!(u16, NonZeroU16);
impl_index!(u32, NonZeroU32);
impl_index!(u64, NonZeroU64);
impl_index!(usize, NonZeroUsize);
