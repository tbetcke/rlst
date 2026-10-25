//! Definition of [BaseArray], a container for array data.
//!
//! A [BaseArray] combines a contiguous data container with a shape and column-major
//! n-dimensional indexing.

use crate::{
    dense::layout::{check_multi_index_in_bounds, col_major_stride_from_shape},
    traits::{
        accessors::{
            RawAccess, RawAccessMut, StableRawAccessMut, StridedRawAccess, StridedRawAccessMut,
            UnsafeDisjointRandom1DAccessMut, UnsafeRandom1DAccessByRef,
            UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
            UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, ResizeInPlace, Shape, Stride},
        data_container::{
            ContainerType, DataContainer, DisjointRefDataContainerMut,
            MutableRawAccessDataContainer, RawAccessDataContainer, RefDataContainer,
            RefDataContainerMut, ResizeableDataContainer, StableRawAccessDataContainer,
            ValueDataContainer,
        },
    },
};

/// Contiguous column-major array implementation backed by `Data`.
///
/// The container length is exactly the product of `shape`; indexing uses the stride derived from
/// that shape. Rank zero is allowed: shape `[]` holds exactly one element.
pub struct BaseArray<Data, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
}

impl<Data: ContainerType, const NDIM: usize> ContainerType for BaseArray<Data, NDIM> {
    type Type = Data::Type;
}

impl<Data: DataContainer, const NDIM: usize> BaseArray<Data, NDIM> {
    /// Combine a data container and shape.
    ///
    /// # Panics
    ///
    /// Panics if the container length is not exactly the product of `shape` (one for rank zero).
    pub fn new(data: Data, shape: [usize; NDIM]) -> Self {
        assert_eq!(
            data.number_of_elements(),
            shape.iter().product::<usize>(),
            "Expected {} elements but `data` has {} elements",
            shape.iter().product::<usize>(),
            data.number_of_elements()
        );

        Self { data, shape }
    }
}

impl<Data, const NDIM: usize> Shape<NDIM> for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Data: DataContainer, const NDIM: usize> BaseItem for BaseArray<Data, NDIM> {
    type Item = Data::Item;
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.data
                .get_unchecked(compute_col_major_index(multi_index, self.shape))
        }
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandom1DAccessByValue
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.data.get_unchecked_value(index) }
    }
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandom1DAccessByRef
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

// SAFETY: column-major linear indices map one-to-one to container elements and
// the shape does not change while mutable references are outstanding.
unsafe impl<Data: DisjointRefDataContainerMut, const NDIM: usize> UnsafeDisjointRandom1DAccessMut
    for BaseArray<Data, NDIM>
{
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.data
                .get_unchecked_value(compute_col_major_index(multi_index, self.shape))
        }
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.data
                .get_unchecked_mut(compute_col_major_index(multi_index, self.shape))
        }
    }
}

impl<Data: RawAccessDataContainer, const NDIM: usize> RawAccess for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.data.data()
    }
}

unsafe impl<Data: StableRawAccessDataContainer, const NDIM: usize> StableRawAccessMut
    for BaseArray<Data, NDIM>
{
}

impl<Data: MutableRawAccessDataContainer, const NDIM: usize> RawAccessMut
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.data.data_mut()
    }
}

// The container holds exactly the elements, column-major from its start.
impl<Data: RawAccessDataContainer, const NDIM: usize> StridedRawAccess for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        Some((self.data.data()?, 0))
    }
}

impl<Data: MutableRawAccessDataContainer, const NDIM: usize> StridedRawAccessMut
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)> {
        Some((self.data.data_mut()?, 0))
    }
}

impl<Data, const NDIM: usize> Stride<NDIM> for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        col_major_stride_from_shape(self.shape)
    }
}

impl<Data: ResizeableDataContainer, const NDIM: usize> ResizeInPlace<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        let new_len = shape.iter().product();
        self.data.resize(new_len);
        self.shape = shape;
    }
}

#[inline(always)]
fn compute_col_major_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    shape: [usize; NDIM],
) -> usize {
    let mut acc = 0;
    let mut shape_prod = 1;
    for index in 0..NDIM {
        acc += multi_index[index] * shape_prod;
        shape_prod *= shape[index];
    }
    acc
}

#[cfg(test)]
mod tests {
    use crate::{
        DynArray, FFTW_ALIGNED, SliceArray, SliceArrayMut,
        dense::{
            array::{
                AlignedDynArray, StaticArray,
                storage_test_support::{
                    assert_storage_addresses_elements, assert_storage_consistent_with_data,
                    assert_storage_mut_writes_elements,
                },
            },
            base_array::BaseArray,
            data_container::ArrayContainer,
        },
    };

    #[test]
    fn owned_arrays_address_their_elements_at_offset_zero() {
        let dyn_array = DynArray::<f64, 3>::from_shape([3, 4, 2]);
        assert_eq!(assert_storage_addresses_elements(&dyn_array), (24, 0));

        let aligned = AlignedDynArray::<f64, 2, FFTW_ALIGNED>::from_shape([5, 3]);
        assert_eq!(assert_storage_addresses_elements(&aligned), (15, 0));

        let static_array =
            StaticArray::new(BaseArray::new(ArrayContainer::<f64, 12>::new(), [3, 4]));
        assert_eq!(assert_storage_addresses_elements(&static_array), (12, 0));

        let values = [0.0; 6];
        let slice_array = SliceArray::from_shape(&values, [2, 3]);
        assert_eq!(assert_storage_addresses_elements(&slice_array), (6, 0));

        let mut values = [0.0; 6];
        let slice_array_mut = SliceArrayMut::from_shape(&mut values, [3, 2]);
        assert_eq!(assert_storage_addresses_elements(&slice_array_mut), (6, 0));
    }

    #[test]
    fn owned_arrays_write_through_mutable_storage() {
        let mut dyn_array = DynArray::<f64, 3>::from_shape([3, 4, 2]);
        assert_storage_mut_writes_elements(&mut dyn_array);

        let mut aligned = AlignedDynArray::<f64, 2, FFTW_ALIGNED>::from_shape([5, 3]);
        assert_storage_mut_writes_elements(&mut aligned);

        let mut static_array =
            StaticArray::new(BaseArray::new(ArrayContainer::<f64, 12>::new(), [3, 4]));
        assert_storage_mut_writes_elements(&mut static_array);

        let mut values = [0.0; 6];
        let mut slice_array_mut = SliceArrayMut::from_shape(&mut values, [3, 2]);
        assert_storage_mut_writes_elements(&mut slice_array_mut);
    }

    #[test]
    fn owned_storage_is_consistent_with_data() {
        let dyn_array = DynArray::<f64, 2>::from_shape([4, 3]);
        assert_eq!(assert_storage_consistent_with_data(&dyn_array), (12, 0));

        let aligned = AlignedDynArray::<f64, 2, FFTW_ALIGNED>::from_shape([4, 3]);
        assert_eq!(assert_storage_consistent_with_data(&aligned), (12, 0));

        let values = [0.0; 6];
        let slice_array = SliceArray::from_shape(&values, [2, 3]);
        assert_eq!(assert_storage_consistent_with_data(&slice_array), (6, 0));
    }

    #[test]
    fn empty_owned_array_has_empty_storage_at_offset_zero() {
        let empty = DynArray::<f64, 2>::from_shape([0, 3]);
        assert_eq!(assert_storage_consistent_with_data(&empty), (0, 0));
        let (storage, offset) = empty.storage().unwrap();
        assert!(storage.is_empty());
        assert_eq!(offset, 0);
    }
}
