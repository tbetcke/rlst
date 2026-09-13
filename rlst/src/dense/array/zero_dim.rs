//! Dedicated scalar storage for zero-dimensional arrays.

use super::Array;
use crate::{BaseItem, Shape, c32, c64};

/// Storage for a single scalar with no axes.
///
/// Construct the public wrapper with [`ZeroDimArray::from_scalar`]. This storage
/// implements only [`BaseItem`] and [`Shape<0>`]; scalar access is provided by
/// the wrapper's dedicated methods.
pub struct ScalarArrayImpl<Item: Copy + Default> {
    scalar: Item,
}

/// A zero-dimensional array owning one scalar.
///
/// Its shape is `[]` and its number of elements is one. Access the value with
/// [`Self::scalar`], [`Self::scalar_mut`], or [`Self::into_scalar`]. Concrete
/// conversions in both directions are available for `f32`, `f64`, [`c32`],
/// [`c64`], and every signed and unsigned integer type, including `isize`,
/// `usize`, `i128`, and `u128`.
///
/// ```
/// use rlst::ZeroDimArray;
///
/// let mut value = ZeroDimArray::from_scalar(3.0_f64);
/// assert_eq!(value.shape(), []);
/// assert_eq!(value.number_of_elements(), 1);
/// *value.scalar_mut() = 4.0;
/// assert_eq!(value.into_scalar(), 4.0);
/// ```
///
/// Random indexing is unavailable, even with an empty multi-index:
///
/// ```compile_fail
/// use rlst::dense::array::zero_dim::ZeroDimArray;
///
/// let value = ZeroDimArray::from_scalar(3.0_f64);
/// let _ = value[[]];
/// ```
///
/// Checked random access is unavailable:
///
/// ```compile_fail
/// use rlst::dense::array::zero_dim::ZeroDimArray;
/// use rlst::RandomAccessByValue;
///
/// let value = ZeroDimArray::from_scalar(3.0_f64);
/// let _ = value.get_value([]);
/// ```
///
/// Linear access is also unavailable:
///
/// ```compile_fail
/// use rlst::dense::array::zero_dim::ZeroDimArray;
/// use rlst::UnsafeRandom1DAccessByValue;
///
/// let value = ZeroDimArray::from_scalar(3.0_f64);
/// let _ = unsafe { value.get_value_1d_unchecked(0) };
/// ```
pub type ZeroDimArray<Item> = Array<ScalarArrayImpl<Item>, 0>;

impl<Item: Copy + Default> BaseItem for ScalarArrayImpl<Item> {
    type Item = Item;
}

impl<Item: Copy + Default> Shape<0> for ScalarArrayImpl<Item> {
    fn shape(&self) -> [usize; 0] {
        []
    }
}

impl<Item: Copy + Default> ZeroDimArray<Item> {
    /// Construct a zero-dimensional array owning `value`.
    pub fn from_scalar(value: Item) -> Self {
        Array::new(ScalarArrayImpl { scalar: value })
    }

    /// Consume the array and return its scalar.
    pub fn into_scalar(self) -> Item {
        self.into_imp().scalar
    }

    /// Borrow the scalar.
    pub fn scalar(&self) -> &Item {
        &self.imp().scalar
    }

    /// Mutably borrow the scalar.
    pub fn scalar_mut(&mut self) -> &mut Item {
        &mut self.imp_mut().scalar
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
    use crate::{Shape, c32, c64};

    #[test]
    fn scalar_shape_and_borrows() {
        let mut array = ZeroDimArray::from_scalar(7_i32);
        assert_eq!(array.shape(), []);
        assert_eq!(array.imp().shape(), []);
        assert_eq!(array.len(), 1);
        assert_eq!(array.number_of_elements(), 1);
        assert!(!array.is_empty());

        let scalar: &i32 = array.scalar();
        assert_eq!(*scalar, 7);
        let scalar: &mut i32 = array.scalar_mut();
        *scalar = -11;
        assert_eq!(*array.scalar(), -11);
        assert_eq!(array.into_scalar(), -11);
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
