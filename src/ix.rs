//! Wrapper around integer types, used as indices within `IntervalMap` and `IntervalSet`.

use core::convert::TryInto;
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
    /// Converts index into `usize`.
    fn get(self) -> usize;

    /// Creates a new index. Returns error if the `elemen_num` is too big.
    fn new(element_num: usize) -> Result<Self, &'static str>;
}

macro_rules! impl_index {
    ($type:ident) => {
        impl IndexType for core::num::$type {
            #[inline(always)]
            fn get(self) -> usize {
                (self.get() - 1) as usize
            }

            #[inline]
            fn new(element_num: usize) -> Result<Self, &'static str> {
                const ERROR_STR :&'static str=
                    concat!(
                        "Failed to insert a new element into IntervalMap/Set: number of elements is too large for ",
                        stringify!($type),
                        ", try using NonZeroU64.");
                let nonzero = core::num::$type::new(
                    element_num
                        .checked_add(1)
                        .ok_or(ERROR_STR)?
                        .try_into()
                        .map_err(|_| ERROR_STR)?,
                )
                .ok_or(ERROR_STR)?;
                Ok(nonzero)
            }
        }
    };
}

impl_index!(NonZeroU8);
impl_index!(NonZeroU16);
impl_index!(NonZeroU32);
impl_index!(NonZeroU64);
impl_index!(NonZeroUsize);
/// Default index type.
pub type DefaultIx = core::num::NonZeroU32;
