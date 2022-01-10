//! Wrapper around integer types, used as indices within `IntervalMap` and `IntervalSet`.

use core::fmt::Display;

/// Trait for index types: used in the inner representation of [IntervalMap](../struct.IntervalMap.html) and
/// [IntervalSet](../set/struct.IntervalSet.html).
///
/// Implemented for `u8`, `u16`, `u32` and `u64`,
/// `u32` is used by default ([DefaultIx](type.DefaultIx.html)).
///
/// `IntervalMap` or `IntervalSet` can store up to `Ix::MAX - 1` elements
/// (for example `IntervalMap<_, _, u8>` can store up to 255 items).
///
/// Using smaller index types saves memory and slightly reduces running time.
pub trait IndexType: Copy + Display + Sized + Eq + Ord {
    /// Undefined index. There can be no indices higher than MAX.
    const MAX: Self;

    /// Converts index into `usize`.
    fn get(self) -> usize;

    /// Creates a new index. Returns error if the `elemen_num` is too big.
    fn new(element_num: usize) -> Result<Self, &'static str>;

    /// Returns `true` if the index is defined.
    #[inline(always)]
    fn defined(self) -> bool {
        self != Self::MAX
    }
}

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

macro_rules! impl_index {
    ($type:ident) => {
        impl IndexType for $type {
            const MAX: Self = core::$type::MAX;

            #[inline(always)]
            fn get(self) -> usize {
                self as usize
            }

            #[inline]
            fn new(element_num: usize) -> Result<Self, &'static str> {
                let element_num = element_num as $type;
                if element_num == core::$type::MAX {
                    Err(index_error!($type))
                } else {
                    Ok(element_num as $type)
                }
            }
        }
    };
}

impl_index!(u8);
impl_index!(u16);
impl_index!(u32);
impl_index!(u64);
/// Default index type.
pub type DefaultIx = u32;
