//! A multislice allows to slice across several dimensions at once.

use itertools::{Itertools, izip};

use crate::{
    Array, BaseItem, ContainerType, IsSmallerThan, MemoryLayout, NumberDifference, NumberType,
    RawAccess, Shape, Stride, SubtractEqual, Unknown, UnsafeRandom1DAccessByRef,
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut, dense::layout::convert_1d_nd_from_shape,
};

/// A multislice is an array implementation representing a slice across multiple dimensions
pub struct Multislice<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
{
    arr: Array<ArrayImpl, ADIM>,
    offset: [usize; ADIM],
    mask: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize>
    Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: Shape<ADIM>,
{
    /// Create a new multislice instance
    pub fn new(arr: Array<ArrayImpl, ADIM>, axes: [usize; SDIM], indices: [usize; SDIM]) -> Self {
        // We compute the mask.
        let mut offset = [0; ADIM];
        let mut mask = [0; NDIM];
        let mut mask_count = 0;
        for (axis_index, o) in offset.iter_mut().enumerate() {
            if let Some(pos) = axes.iter().position(|&elem| elem == axis_index) {
                // The index exists. We have to add it to the offset
                *o = indices[pos];
            } else {
                // The index does not exist. Hence, it is added to the mask
                mask[mask_count] = axis_index;
                mask_count += 1;
            }
        }

        // Now get the shape
        let orig_shape = arr.shape();
        let mut shape = [0; NDIM];
        for (&mask_index, s) in izip!(mask.iter(), shape.iter_mut()) {
            *s = orig_shape[mask_index];
        }

        Self {
            arr,
            offset,
            mask,
            shape,
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> ContainerType
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
{
    type Type = Unknown;
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> Shape<NDIM>
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: Shape<ADIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> Stride<NDIM>
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: Stride<ADIM>,
{
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        let orig_stride = self.arr.stride();
        let mut stride = [0; NDIM];
        for (&mask_index, s) in izip!(self.mask.iter(), stride.iter_mut()) {
            *s = orig_stride[mask_index];
        }
        stride
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> BaseItem
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe {
            self.arr
                .get_value_unchecked(multi_index_to_orig(multi_index, self.offset, self.mask))
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: UnsafeRandomAccessByRef<ADIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe {
            self.arr
                .get_unchecked(multi_index_to_orig(multi_index, self.offset, self.mask))
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: UnsafeRandomAccessMut<ADIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        unsafe {
            self.arr
                .get_unchecked_mut(multi_index_to_orig(multi_index, self.offset, self.mask))
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> UnsafeRandom1DAccessByValue
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM> + Shape<ADIM>,
{
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> UnsafeRandom1DAccessByRef
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: UnsafeRandomAccessByRef<ADIM> + Shape<ADIM>,
{
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.get_unchecked(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> UnsafeRandom1DAccessMut
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: UnsafeRandomAccessMut<ADIM> + Shape<ADIM>,
{
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<Item, ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize> RawAccess
    for Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
    ArrayImpl: RawAccess<Item = Item> + Stride<ADIM> + Shape<ADIM>,
{
    fn data(&self) -> Option<&[Self::Item]> {
        // We first check whether the data is contiguous

        // Asser that the mask is contiguous
        if !self
            .mask
            .iter()
            .tuple_windows()
            .all(|(&elem1, &elem2)| elem2 == 1 + elem1)
        {
            return None;
        }

        let stride = self.arr.stride();

        // First Case: We are row-major
        // For a row-major the mask should point to the last elements
        // of the original multi-index
        // If ADIM = 7 and SDIM = 3 the mask should be [4, 5, 6].
        // The mask is already contiguous. So check that mask[0] = ADIM - SDIM
        // and check that the original array is row-major
        if self.mask[0] == ADIM - SDIM && matches!(self.arr.memory_layout(), MemoryLayout::RowMajor)
        {
            // Compute the start index.
            let start_index = izip!(stride.iter(), self.offset.iter())
                .take(SDIM)
                .map(|(s, i)| s * i)
                .sum::<usize>();
            // Compute end index. The end index is the start index + the stride `stride[SDIM - 1]`.
            let end_index = start_index + stride[SDIM - 1];
            Some(&self.arr.data()?[start_index..end_index])
        } else if self.mask[0] == 0 && matches!(self.arr.memory_layout(), MemoryLayout::ColumnMajor)
        {
            // Compute the start index. We need to get the memory index skip from the last SDIM elements in the stride.
            let start_index = izip!(stride.iter(), self.offset.iter())
                .skip(ADIM - SDIM)
                .map(|(s, i)| s * i)
                .sum::<usize>();
            // Compute end index. The end index is the start index + the stride `stride[SDIM - 1]`.
            let end_index = start_index + stride[ADIM - SDIM];
            Some(&self.arr.data()?[start_index..end_index])
        } else {
            None
        }
    }
}

#[inline(always)]
fn multi_index_to_orig<const ADIM: usize, const SDIM: usize, const NDIM: usize>(
    multi_index: [usize; NDIM],
    offset: [usize; ADIM],
    mask: [usize; NDIM],
) -> [usize; ADIM]
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
{
    let mut orig = [0; ADIM];
    for (index, &value) in multi_index.iter().enumerate() {
        orig[mask[index]] = value;
    }

    for (val1, val2) in izip!(orig.iter_mut(), offset.iter()) {
        *val1 += val2;
    }
    orig
}

#[cfg(test)]
mod test {

    use itertools::izip;

    use crate::{DynArray, StridedDynArray, dense::layout::convert_nd_raw};

    #[test]
    fn test_multislice_shape() {
        let arr = DynArray::<f64, 4>::from_shape([2, 3, 4, 5]);
        let axes = [2, 1];
        let indices = [3, 1];

        let expected_shape = [2, 5];
        let slice = arr.multislice::<2, 2>(axes, indices);
        assert_eq!(slice.shape(), expected_shape);
    }

    #[test]
    fn test_multislice_stride() {
        let arr = DynArray::<f64, 4>::from_shape([2, 3, 4, 5]);
        let axes = [2, 1];
        let indices = [3, 1];

        let stride = arr.stride();

        let expected_stride = [stride[0], stride[3]];
        let slice = arr.multislice::<2, 2>(axes, indices);
        assert_eq!(slice.stride(), expected_stride);
    }

    #[test]
    fn test_multislice_access() {
        let mut arr = DynArray::<f64, 4>::from_shape([2, 3, 4, 5]);
        arr.fill_from_seed_equally_distributed(0);
        let axes = [2, 1];
        let indices = [3, 1];
        let slice = arr.r().multislice::<2, 2>(axes, indices);

        assert_eq!(slice.get_value([1, 2]), arr.get_value([1, 1, 3, 2]));
    }

    #[test]
    fn test_multislice_data_column_major() {
        let mut arr = DynArray::<f64, 4>::from_shape([3, 2, 5, 4]);
        arr.fill_from_seed_equally_distributed(0);

        let axes = [3, 2];
        let indices = [2, 3];

        let slice = arr.r().multislice::<2, 2>(axes, indices);

        let orig_data = arr.data().unwrap();

        let slice_data = slice.data().unwrap();

        // Check that the length of slice_data is correct

        let orig_data_start_index = convert_nd_raw([0, 0, 3, 2], arr.stride());
        let orig_data_last_index = convert_nd_raw([2, 1, 3, 2], arr.stride());

        let expected_length = orig_data_last_index - orig_data_start_index + 1;
        assert_eq!(slice_data.len(), expected_length);

        // Check that the data in both is identical

        for (a, e) in izip!(
            slice_data.iter(),
            orig_data[orig_data_start_index..1 + orig_data_last_index].iter()
        ) {
            assert_eq!(*a, *e)
        }

        // Make sure that for non-contiguous slices we get `None` returned.

        assert!(arr.r().multislice::<2, 2>([1, 3], [0, 1]).data().is_none());
    }

    #[test]
    fn test_multislice_data_row_major() {
        let mut arr = StridedDynArray::<f64, 4>::row_major([3, 2, 5, 4]);
        arr.fill_from_seed_equally_distributed(0);

        let axes = [1, 0];
        let indices = [1, 2];

        let slice = arr.r().multislice::<2, 2>(axes, indices);

        let orig_data = arr.data().unwrap();

        let slice_data = slice.data().unwrap();

        // Check that the length of slice_data is correct

        let orig_data_start_index = convert_nd_raw([2, 1, 0, 0], arr.stride());
        let orig_data_last_index = convert_nd_raw([2, 1, 4, 3], arr.stride());

        let expected_length = orig_data_last_index - orig_data_start_index + 1;
        assert_eq!(slice_data.len(), expected_length);

        // Check that the data in both is identical

        for (a, e) in izip!(
            slice_data.iter(),
            orig_data[orig_data_start_index..1 + orig_data_last_index].iter()
        ) {
            assert_eq!(*a, *e)
        }

        // Make sure that for non-contiguous slices we get `None` returned.

        assert!(arr.r().multislice::<2, 2>([1, 3], [0, 1]).data().is_none());
    }
}
