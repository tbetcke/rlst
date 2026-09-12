//! Lazy axis-permutation array view.

use crate::{
    ContainerType,
    dense::{array::Array, layout::convert_1d_nd_from_shape},
    traits::{
        accessors::{
            UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
            UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, Shape},
    },
};

/// Array implementation that presents source axes in a new order.
pub struct ArrayTranspose<ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    permutation: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<ArrayImpl, const NDIM: usize> ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create an axis-permuted view.
    ///
    /// For a valid permutation, `permutation[output_axis]` names the corresponding source axis.
    ///
    /// # Panics
    ///
    /// Panics if an entry is outside `0..NDIM` or if an axis occurs more than once.
    /// Every source axis must occur exactly once.
    pub fn new(arr: Array<ArrayImpl, NDIM>, permutation: [usize; NDIM]) -> Self {
        let mut seen = [false; NDIM];
        let mut shape = [0; NDIM];
        let source_shape = arr.shape();

        for (output_axis, &source_axis) in permutation.iter().enumerate() {
            assert!(source_axis < NDIM, "axis permutation entry out of range");
            assert!(
                !seen[source_axis],
                "axis permutation contains repeated axes"
            );
            seen[source_axis] = true;
            shape[output_axis] = source_shape[source_axis];
        }
        Self {
            arr,
            permutation,
            shape,
        }
    }
}

impl<ArrayImpl, const NDIM: usize> ContainerType for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<Item: Copy + Default, ArrayImpl, const NDIM: usize> BaseItem
    for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe {
            self.arr
                .get_value_unchecked(inverse_permute_multi_index(multi_index, self.permutation))
        }
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByRef<NDIM> for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe {
            self.arr
                .get_unchecked(inverse_permute_multi_index(multi_index, self.permutation))
        }
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessMut<NDIM> for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        unsafe {
            self.arr
                .get_unchecked_mut(inverse_permute_multi_index(multi_index, self.permutation))
        }
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayTranspose<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByRef for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.get_unchecked(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessMut for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

fn inverse_permute_multi_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    permutation: [usize; NDIM],
) -> [usize; NDIM] {
    let mut result = [0; NDIM];

    for (index, &p) in permutation.iter().enumerate() {
        result[p] = multi_index[index];
    }

    result
}

// SAFETY: the constructor enforces a bijection of axes, so distinct output indices
// map to distinct source indices. The source guarantees disjoint mutable access.
unsafe impl<ArrayImpl, const NDIM: usize> crate::UnsafeDisjointRandom1DAccessMut
    for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: crate::UnsafeDisjointRandom1DAccessMut + UnsafeRandomAccessMut<NDIM> + Shape<NDIM>,
{
}

#[cfg(test)]
mod tests {
    use super::ArrayTranspose;
    use crate::DynArray;

    #[test]
    #[should_panic(expected = "axis permutation contains repeated axes")]
    fn constructor_rejects_repeated_axes() {
        let array = DynArray::<f64, 2>::from_shape([2, 3]);
        let _ = ArrayTranspose::new(array, [0, 0]);
    }

    #[test]
    #[should_panic(expected = "axis permutation contains repeated axes")]
    fn permute_axes_rejects_repeated_axes() {
        let _ = DynArray::<f64, 2>::from_shape([2, 3]).permute_axes([1, 1]);
    }

    #[test]
    #[should_panic(expected = "axis permutation contains repeated axes")]
    fn empty_array_rejects_repeated_axes() {
        let _ = DynArray::<f64, 2>::from_shape([0, 3]).permute_axes([0, 0]);
    }

    #[test]
    #[should_panic(expected = "axis permutation entry out of range")]
    fn constructor_rejects_out_of_range_axis() {
        let array = DynArray::<f64, 2>::from_shape([2, 3]);
        let _ = ArrayTranspose::new(array, [0, 2]);
    }

    #[test]
    fn cyclic_permutation_maps_shape_and_elements() {
        let mut array = DynArray::<f64, 3>::from_shape([2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    array[[i, j, k]] = (100 * i + 10 * j + k) as f64;
                }
            }
        }
        let view = array.permute_axes([2, 0, 1]);
        assert_eq!(view.shape(), [4, 2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(view[[k, i, j]], (100 * i + 10 * j + k) as f64);
                }
            }
        }
    }

    #[test]
    fn mutable_iteration_yields_disjoint_elements() {
        let mut array = DynArray::<f64, 3>::from_shape([2, 3, 4]);
        {
            let mut view = array.r_mut().permute_axes([2, 0, 1]);
            // Keep every reference alive at once, as the disjoint-access contract permits.
            let elements: Vec<_> = view.iter_mut().collect();
            for (index, element) in elements.into_iter().enumerate() {
                *element = index as f64;
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(array[[i, j, k]], (k + 4 * i + 8 * j) as f64);
                }
            }
        }
    }

    #[test]
    fn identity_and_empty_permutations_are_valid() {
        let view = DynArray::<f64, 3>::from_shape([2, 3, 4]).permute_axes([0, 1, 2]);
        assert_eq!(view.shape(), [2, 3, 4]);
        let empty = DynArray::<f64, 2>::from_shape([0, 3]).permute_axes([1, 0]);
        assert_eq!(empty.shape(), [3, 0]);
        assert_eq!(empty.iter_value().count(), 0);
        let vector = DynArray::<f64, 1>::from_shape([3]).permute_axes([0]);
        assert_eq!(vector.shape(), [3]);
    }
}
