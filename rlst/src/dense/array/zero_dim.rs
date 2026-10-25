//! Zero-dimensional arrays and their scalar accessors.

use super::{Array, DynArray};
use crate::{RandomAccessByRef, RandomAccessMut, c32, c64};

/// A zero-dimensional array owning one scalar.
///
/// This is [`DynArray<Item, 0>`]: its shape and stride are `[]`, and it holds exactly one
/// element, so `len()` is one and `is_empty()` is false. It supports every operation of
/// a rank-generic array, including indexing with the empty multi-index `[]` and iteration over
/// its one element. Besides indexing, access the value with [`Self::scalar`],
/// [`Self::scalar_mut`], or [`Self::into_scalar`]. Concrete conversions in both directions are
/// available for `f32`, `f64`, [`c32`], [`c64`], and every signed and unsigned integer type,
/// including `isize`, `usize`, `i128`, and `u128`.
///
/// A zero-dimensional array is distinct from an empty array of shape `[0]`, which holds no
/// element.
///
/// ```
/// use rlst::ZeroDimArray;
///
/// let mut value = ZeroDimArray::from_scalar(3.0_f64);
/// assert_eq!(value.shape(), [0usize; 0]);
/// assert_eq!(value.number_of_elements(), 1);
/// *value.scalar_mut() = 4.0;
/// assert_eq!(value[[]], 4.0);
/// value[[]] = 5.0;
/// assert_eq!(value.into_scalar(), 5.0);
/// ```
///
/// Code generic over the rank covers rank zero without special cases:
///
/// ```
/// use rlst::{DynArray, EvaluateArray};
///
/// fn doubled<const NDIM: usize>(shape: [usize; NDIM]) -> DynArray<f64, NDIM> {
///     let mut array = DynArray::<f64, NDIM>::from_shape(shape);
///     array.fill_with_value(1.5);
///     (2.0_f64 * array).eval()
/// }
///
/// assert_eq!(doubled([2, 3]).iter_value().sum::<f64>(), 18.0);
/// assert_eq!(doubled([]).into_scalar(), 3.0);
/// ```
pub type ZeroDimArray<Item> = DynArray<Item, 0>;

impl<Item: Copy + Default> ZeroDimArray<Item> {
    /// Construct a zero-dimensional array owning `value`.
    pub fn from_scalar(value: Item) -> Self {
        Self::from_shape_and_vec([], vec![value])
    }

    /// Consume the array and return its scalar.
    pub fn into_scalar(self) -> Item {
        self[[]]
    }
}

impl<ArrayImpl: RandomAccessByRef<0>> Array<ArrayImpl, 0> {
    /// Borrow the one element of a zero-dimensional array or view.
    pub fn scalar(&self) -> &ArrayImpl::Item {
        self.get([])
            .expect("a zero-dimensional array has exactly one element")
    }
}

impl<ArrayImpl: RandomAccessMut<0>> Array<ArrayImpl, 0> {
    /// Mutably borrow the one element of a zero-dimensional array or view.
    pub fn scalar_mut(&mut self) -> &mut ArrayImpl::Item {
        self.get_mut([])
            .expect("a zero-dimensional array has exactly one element")
    }
}

macro_rules! impl_zero_dim_scalar_conversions {
    ($($item:ty),+ $(,)?) => {
        $(
            impl From<$item> for ZeroDimArray<$item> {
                fn from(value: $item) -> Self {
                    Self::from_scalar(value)
                }
            }

            impl From<ZeroDimArray<$item>> for $item {
                fn from(value: ZeroDimArray<$item>) -> Self {
                    value.into_scalar()
                }
            }
        )+
    };
}

impl_zero_dim_scalar_conversions!(
    f32, f64, c32, c64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize,
);

#[cfg(test)]
mod tests {
    use super::ZeroDimArray;
    use crate::{DynArray, Shape, c32, c64};

    #[test]
    fn scalar_shape_and_borrows() {
        let mut array = ZeroDimArray::from_scalar(7_i32);
        assert_eq!(array.shape(), [0usize; 0]);
        assert_eq!(array.imp().shape(), [0usize; 0]);
        assert_eq!(array.stride(), [0usize; 0]);
        assert_eq!(array.len(), 1);
        assert_eq!(array.number_of_elements(), 1);
        assert!(!array.is_empty());

        let scalar: &i32 = array.scalar();
        assert_eq!(*scalar, 7);
        let scalar: &mut i32 = array.scalar_mut();
        *scalar = -11;
        assert_eq!(*array.scalar(), -11);
        assert_eq!(*array.r().scalar(), -11);
        *array.r_mut().scalar_mut() = 5;
        assert_eq!(array[[]], 5);
        assert_eq!(array.into_scalar(), 5);
    }

    #[test]
    fn zero_dim_array_is_rank_zero_dyn_array() {
        let array: DynArray<f64, 0> = ZeroDimArray::from_scalar(2.5);
        assert_eq!(array.data().unwrap(), &[2.5]);
    }

    #[test]
    fn concrete_conversion_roundtrips() {
        fn roundtrip<Item>(value: Item)
        where
            Item: Copy + Default + PartialEq + std::fmt::Debug + From<ZeroDimArray<Item>>,
            ZeroDimArray<Item>: From<Item>,
        {
            let array: ZeroDimArray<Item> = value.into();
            assert_eq!(*array.scalar(), value);
            let recovered: Item = array.into();
            assert_eq!(recovered, value);
        }

        roundtrip(1.25_f32);
        roundtrip(-2.5_f64);
        roundtrip(c32::new(1.25, -3.5));
        roundtrip(c64::new(-2.5, 7.25));
        roundtrip(i8::MIN);
        roundtrip(i16::MIN);
        roundtrip(i32::MIN);
        roundtrip(i64::MIN);
        roundtrip(i128::MIN);
        roundtrip(isize::MIN);
        roundtrip(u8::MAX);
        roundtrip(u16::MAX);
        roundtrip(u32::MAX);
        roundtrip(u64::MAX);
        roundtrip(u128::MAX);
        roundtrip(usize::MAX);
    }

    #[test]
    fn custom_copy_default_item() {
        #[derive(Clone, Copy, Default)]
        struct CustomItem {
            value: u32,
        }

        fn wrap<Item: Copy + Default>(value: Item) -> ZeroDimArray<Item> {
            ZeroDimArray::from_scalar(value)
        }

        let mut array = wrap(CustomItem { value: 42 });
        assert_eq!(array.scalar().value, 42);
        array.scalar_mut().value = 73;
        assert_eq!(array.into_scalar().value, 73);
        assert_eq!(wrap(CustomItem::default()).into_scalar().value, 0);
    }
}
