//! Definition of [StridedBaseArray], a container for array data with custom stride.
//!
//! A [StridedBaseArray] combines a data container with explicit shape and stride metadata.

use std::collections::HashSet;

use crate::{
    dense::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape, convert_nd_raw},
    traits::{
        accessors::{
            RawAccess, RawAccessMut, StableRawAccessMut, StridedRawAccess, StridedRawAccessMut,
            UnsafeDisjointRandom1DAccessMut, UnsafeRandom1DAccessByRef,
            UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
            UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, Shape, Stride},
        data_container::{
            ContainerType, DataContainer, DisjointRefDataContainerMut,
            MutableRawAccessDataContainer, RawAccessDataContainer, RefDataContainer,
            RefDataContainerMut, StableRawAccessDataContainer, ValueDataContainer,
        },
    },
};

/// Array implementation with explicit non-negative strides.
///
/// A logical index maps to the storage offset `sum(index[axis] * stride[axis])`. The layout may
/// contain gaps, but distinct logical indices may not alias.
pub struct StridedBaseArray<Data, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
    stride: [usize; NDIM],
}

impl<Data: ContainerType, const NDIM: usize> ContainerType for StridedBaseArray<Data, NDIM> {
    type Type = Data::Type;
}

/// Return the storage span required by a shape and stride.
pub(crate) fn required_storage_len<const NDIM: usize>(
    shape: [usize; NDIM],
    stride: [usize; NDIM],
) -> usize {
    if shape.iter().product::<usize>() == 0 {
        0
    } else {
        1 + shape
            .iter()
            .zip(stride)
            .map(|(&extent, step)| (extent - 1) * step)
            .sum::<usize>()
    }
}

impl<Data: DataContainer, const NDIM: usize> StridedBaseArray<Data, NDIM> {
    /// Combine a data container with a shape and stride.
    ///
    /// The container may be longer than the reachable storage span.
    ///
    /// # Panics
    ///
    /// Panics if `NDIM` is zero, the computed data requirement exceeds the container length, or two computed
    /// logical offsets are equal. Shape/stride arithmetic uses ordinary `usize` operations, so
    /// callers must provide metadata whose products, offsets, and storage span are representable.
    pub fn new(data: Data, shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
        assert!(
            NDIM > 0,
            "StridedBaseArray requires a positive rank; use ZeroDimArray for scalar storage"
        );
        let number_of_elements = shape.iter().product::<usize>();
        let required_span = required_storage_len(shape, stride);
        assert!(
            data.number_of_elements() >= required_span,
            "Strided layout requires {required_span} storage elements but `data` has {} elements",
            data.number_of_elements()
        );

        let mut offsets = HashSet::with_capacity(number_of_elements);
        for index in 0..number_of_elements {
            let offset = convert_nd_raw(convert_1d_nd_from_shape(index, shape), stride);
            assert!(
                offsets.insert(offset),
                "Strided layout maps multiple logical elements to storage offset {offset}"
            );
        }

        Self {
            data,
            shape,
            stride,
        }
    }
}

impl<Data: DataContainer, const NDIM: usize> BaseItem for StridedBaseArray<Data, NDIM> {
    type Item = Data::Item;
}

impl<Data: DataContainer, const NDIM: usize> Shape<NDIM> for StridedBaseArray<Data, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.data
                .get_unchecked(convert_nd_raw(multi_index, self.stride))
        }
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.data
                .get_unchecked_value(convert_nd_raw(multi_index, self.stride))
        }
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandom1DAccessByValue
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape)) }
    }
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandom1DAccessByRef
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.get_unchecked(convert_1d_nd_from_shape(index, self.shape)) }
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape)) }
    }
}

// SAFETY: `new` rejects layouts that alias logical indices, and the shape and stride
// remain immutable for the lifetime of this value.
unsafe impl<Data: DisjointRefDataContainerMut, const NDIM: usize> UnsafeDisjointRandom1DAccessMut
    for StridedBaseArray<Data, NDIM>
{
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.data
                .get_unchecked_mut(convert_nd_raw(multi_index, self.stride))
        }
    }
}

impl<Data: RawAccessDataContainer, const NDIM: usize> RawAccess for StridedBaseArray<Data, NDIM> {
    fn data(&self) -> Option<&[Self::Item]> {
        self.data.data()
    }
}

unsafe impl<Data: StableRawAccessDataContainer, const NDIM: usize> StableRawAccessMut
    for StridedBaseArray<Data, NDIM>
{
}

impl<Data: MutableRawAccessDataContainer, const NDIM: usize> RawAccessMut
    for StridedBaseArray<Data, NDIM>
{
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.data.data_mut()
    }
}

// The container may hold gaps and be longer than the span. Strides are non-negative, so the
// first element is at offset 0.
impl<Data: RawAccessDataContainer, const NDIM: usize> StridedRawAccess
    for StridedBaseArray<Data, NDIM>
{
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        Some((self.data.data()?, 0))
    }
}

impl<Data: MutableRawAccessDataContainer, const NDIM: usize> StridedRawAccessMut
    for StridedBaseArray<Data, NDIM>
{
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)> {
        Some((self.data.data_mut()?, 0))
    }
}

impl<Data, const NDIM: usize> Stride<NDIM> for StridedBaseArray<Data, NDIM> {
    fn stride(&self) -> [usize; NDIM] {
        self.stride
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        StridedDynArray,
        dense::array::{
            StridedSliceArray, StridedSliceArrayMut,
            storage_test_support::{
                assert_storage_addresses_elements, assert_storage_consistent_with_data,
                assert_storage_mut_writes_elements,
            },
        },
    };

    #[test]
    fn strided_arrays_address_their_elements_at_offset_zero() {
        let row_major = StridedDynArray::<f64, 3>::row_major([3, 4, 2]);
        assert_eq!(assert_storage_addresses_elements(&row_major), (24, 0));

        let custom = StridedDynArray::<f64, 2>::from_shape_and_stride([3, 2], [2, 7]);
        assert_eq!(assert_storage_addresses_elements(&custom), (12, 0));

        // A strided view of a slice longer than its span.
        let values = [0.0; 40];
        let longer = StridedSliceArray::from_shape_and_stride(&values, [3, 4], [2, 6]);
        assert_eq!(assert_storage_addresses_elements(&longer), (40, 0));
    }

    #[test]
    fn strided_arrays_write_through_mutable_storage() {
        let mut row_major = StridedDynArray::<f64, 3>::row_major([3, 4, 2]);
        assert_storage_mut_writes_elements(&mut row_major);

        let mut custom = StridedDynArray::<f64, 2>::from_shape_and_stride([3, 2], [2, 7]);
        assert_storage_mut_writes_elements(&mut custom);

        let mut values = [0.0; 40];
        let mut longer = StridedSliceArrayMut::from_shape_and_stride(&mut values, [3, 4], [2, 6]);
        assert_storage_mut_writes_elements(&mut longer);
    }

    #[test]
    fn strided_storage_is_consistent_with_data() {
        let row_major = StridedDynArray::<f64, 2>::row_major([3, 4]);
        assert_eq!(assert_storage_consistent_with_data(&row_major), (12, 0));

        let values = [0.0; 40];
        let longer = StridedSliceArray::from_shape_and_stride(&values, [3, 4], [2, 6]);
        assert_eq!(assert_storage_consistent_with_data(&longer), (40, 0));
    }

    #[test]
    fn padded_dynamic_layout_allocates_required_span() {
        let array = StridedDynArray::<f64, 2>::from_shape_and_stride([2, 3], [1, 4]);
        assert_eq!(array.data().unwrap().len(), 10);
    }

    #[test]
    fn mutable_iteration_yields_simultaneously_live_disjoint_references() {
        let mut array = StridedDynArray::<i32, 2>::from_shape_and_stride([2, 2], [1, 3]);
        let mut references = array.iter_mut().collect::<Vec<_>>();
        *references[0] = 1;
        *references[1] = 2;
        *references[2] = 3;
        *references[3] = 4;
        drop(references);
        assert_eq!(array.iter_value().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "requires 10 storage elements")]
    fn rejects_insufficient_storage_span() {
        let mut data = [0.0; 9];
        let _ = StridedSliceArrayMut::from_shape_and_stride(&mut data, [2, 3], [1, 4]);
    }

    #[test]
    #[should_panic(expected = "maps multiple logical elements")]
    fn rejects_overlapping_layout() {
        let mut data = [0.0; 4];
        let _ = StridedSliceArrayMut::from_shape_and_stride(&mut data, [2, 2], [1, 1]);
    }
}
