//! A multislice allows to slice across several dimensions at once.

use itertools::izip;

use crate::{
    Array, BaseItem, ContainerType, IsSmallerThan, NumberDifference, NumberType, Shape, Stride,
    SubtractEqual, Unknown, UnsafeRandomAccessByValue,
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
}

impl<ArrayImpl, const ADIM: usize, const SDIM: usize, const NDIM: usize>
    Multislice<ArrayImpl, ADIM, SDIM, NDIM>
where
    NumberType<SDIM>: IsSmallerThan<ADIM>,
    NumberDifference<ADIM, SDIM>: SubtractEqual<NDIM>,
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
        Self { arr, offset, mask }
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
        let orig_shape = self.arr.shape();
        let mut shape = [0; NDIM];
        for (&mask_index, s) in izip!(self.mask.iter(), shape.iter_mut()) {
            *s = orig_shape[mask_index];
        }
        shape
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

    use crate::DynArray;

    #[test]
    fn test_multislice_shape() {
        let arr = DynArray::<f64, 4>::from_shape([2, 3, 4, 5]);
        let axes = [2, 1];
        let indices = [3, 1];

        let expected_shape = [2, 5];
        let slice = arr.multislice::<2, 2>(axes, indices);
        assert_eq!(slice.shape(), expected_shape);
    }
}
