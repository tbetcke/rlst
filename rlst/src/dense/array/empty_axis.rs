//! Extend an array by a singleton (extent-one) axis at the front or back.

use crate::{
    ContainerType,
    base_types::NumberType,
    traits::{
        accessors::{
            StridedRawAccess, StridedRawAccessMut, UnsafeRandom1DAccessByRef,
            UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
        },
        base_operations::BaseItem,
        number_relations::IsSmallerByOne,
    },
};

use super::{
    Array, RawAccess, RawAccessMut, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Position at which to insert the singleton axis.
#[derive(Clone, Copy)]
pub enum AxisPosition {
    /// Insert axis at the front.
    Front,
    /// Insert axis at the back.
    Back,
}

/// Array view with one additional extent-one axis.
pub struct ArrayAppendAxis<ArrayImpl, const ADIM: usize, const NDIM: usize>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    arr: Array<ArrayImpl, ADIM>,
    axis_position: AxisPosition,
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    /// Create a view with a singleton axis at `axis_position`.
    pub fn new(arr: Array<ArrayImpl, ADIM>, axis_position: AxisPosition) -> Self {
        Self { arr, axis_position }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> ContainerType
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> BaseItem
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe {
            self.arr
                .get_value_unchecked(multi_index_to_orig(multi_index, self.axis_position))
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandom1DAccessByValue,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.arr.imp().get_value_1d_unchecked(index) }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandom1DAccessByRef,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.arr.imp().get_1d_unchecked(index) }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandom1DAccessMut,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.arr.imp_mut().get_1d_unchecked_mut(index) }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandomAccessByRef<ADIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe {
            self.arr
                .get_unchecked(multi_index_to_orig(multi_index, self.axis_position))
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandomAccessMut<ADIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        unsafe {
            self.arr
                .get_unchecked_mut(multi_index_to_orig(multi_index, self.axis_position))
        }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> Shape<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: Shape<ADIM>,
{
    fn shape(&self) -> [usize; NDIM] {
        let mut result = [1; NDIM];
        let orig_shape = self.arr.shape();

        match self.axis_position {
            AxisPosition::Front => result[1..].copy_from_slice(&orig_shape),
            AxisPosition::Back => result[..ADIM].copy_from_slice(&orig_shape),
        }
        result
    }
}

impl<ArrayImpl: RawAccess, const ADIM: usize, const NDIM: usize> RawAccess
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.arr.data()
    }
}

impl<ArrayImpl: RawAccessMut, const ADIM: usize, const NDIM: usize> RawAccessMut
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    #[inline(always)]
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.arr.data_mut()
    }
}

// The inserted axis has extent 1, so its stride addresses no other element.
impl<ArrayImpl: StridedRawAccess, const ADIM: usize, const NDIM: usize> StridedRawAccess
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    #[inline(always)]
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        self.arr.storage()
    }
}

impl<ArrayImpl: StridedRawAccessMut, const ADIM: usize, const NDIM: usize> StridedRawAccessMut
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    #[inline(always)]
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)> {
        self.arr.storage_mut()
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> Stride<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: Stride<ADIM> + Shape<ADIM>,
{
    fn stride(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_stride = self.arr.stride();

        match self.axis_position {
            AxisPosition::Front => {
                result[0] = 1;
                result[1..].copy_from_slice(&orig_stride)
            }
            AxisPosition::Back => {
                // An axis appended to a zero-dimensional array is its only axis.
                result[NDIM - 1] = match ADIM {
                    0 => 1,
                    _ => orig_stride[ADIM - 1] * self.arr.shape()[ADIM - 1],
                };
                result[..ADIM].copy_from_slice(&orig_stride)
            }
        }
        result
    }
}

impl<ArrayImpl, const ADIM: usize> Array<ArrayImpl, ADIM> {
    /// Insert a singleton axis at the front or back.
    ///
    /// Despite the historical method name, the inserted axis has extent one, not zero. The view
    /// owns its array operand and does not allocate or copy element storage.
    pub fn insert_empty_axis<const NDIM: usize>(
        self,
        axis_position: AxisPosition,
    ) -> Array<ArrayAppendAxis<ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsSmallerByOne<NDIM>,
    {
        Array::new(ArrayAppendAxis::new(self, axis_position))
    }
}

/// Map an extended index of dimension NDIM to an index of dimension ADIM with NDIM = ADIM + 1.
fn multi_index_to_orig<const ADIM: usize, const NDIM: usize>(
    multi_index: [usize; NDIM],
    axis_position: AxisPosition,
) -> [usize; ADIM]
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    let mut orig = [0; ADIM];
    match axis_position {
        AxisPosition::Front => orig.copy_from_slice(&multi_index[1..]),
        AxisPosition::Back => orig.copy_from_slice(&multi_index[..ADIM]),
    }

    orig
}

// SAFETY: adding a unit axis does not change the one-dimensional index mapping.
unsafe impl<ArrayImpl, const ADIM: usize, const NDIM: usize> crate::UnsafeDisjointRandom1DAccessMut
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: crate::UnsafeDisjointRandom1DAccessMut,
{
}

#[cfg(test)]
mod tests {
    use super::AxisPosition;
    use crate::{
        DynArray, ZeroDimArray,
        dense::array::storage_test_support::{
            assert_storage_addresses_elements, assert_storage_consistent_with_data,
            assert_storage_mut_writes_elements,
        },
    };

    #[test]
    fn inserted_axis_forwards_the_storage() {
        let mut mat = DynArray::<f64, 2>::from_shape([3, 4]);
        for position in [AxisPosition::Front, AxisPosition::Back] {
            let view = mat.r().insert_empty_axis::<3>(position);
            assert_eq!(assert_storage_addresses_elements(&view), (12, 0));
            assert_eq!(assert_storage_consistent_with_data(&view), (12, 0));

            let view = mat
                .r()
                .into_subview([1, 1], [2, 3])
                .insert_empty_axis::<3>(position);
            assert_eq!(assert_storage_addresses_elements(&view), (12, 4));

            assert_storage_mut_writes_elements(&mut mat.r_mut().insert_empty_axis::<3>(position));
            assert_storage_mut_writes_elements(
                &mut mat
                    .r_mut()
                    .into_subview([1, 1], [2, 3])
                    .insert_empty_axis::<3>(position),
            );
        }
    }

    #[test]
    fn inserting_an_axis_into_a_zero_dimensional_array() {
        let mut scalar = ZeroDimArray::from_scalar(2.5);
        for position in [AxisPosition::Front, AxisPosition::Back] {
            let value = *scalar.scalar();
            let view = scalar.r().insert_empty_axis::<1>(position);
            assert_eq!(view.shape(), [1]);
            assert_eq!(view.stride(), [1]);
            assert_eq!(view[[0]], value);
            assert_eq!(assert_storage_addresses_elements(&view), (1, 0));
            assert_eq!(assert_storage_consistent_with_data(&view), (1, 0));

            assert_storage_mut_writes_elements(
                &mut scalar.r_mut().insert_empty_axis::<1>(position),
            );
        }
    }
}
