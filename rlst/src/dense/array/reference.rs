//! Reference to an array.
//!
//! A reference is an owned struct that holds a reference to an array. It is used to
//! pass arrays to functions without transferring ownership.

use crate::{
    ContainerType,
    dense::array::Array,
    traits::{
        accessors::{
            RawAccess, RawAccessMut, StableRawAccessMut, StridedRawAccess, StridedRawAccessMut,
            UnsafeDisjointRandom1DAccessMut, UnsafeRandom1DAccessByRef,
            UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
            UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, ResizeInPlace, Shape, Stride},
    },
};

/// Array implementation forwarding operations through a shared borrow.
pub struct ArrayRef<'a, ArrayImpl, const NDIM: usize>(&'a Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> ContainerType for ArrayRef<'a, ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<'a, ArrayImpl, const NDIM: usize> BaseItem for ArrayRef<'a, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayRef<'a, ArrayImpl, NDIM> {
    /// Wrap a shared borrow without copying array data.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayRef<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArrayRef<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayRef<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.0.data()
    }
}

impl<ArrayImpl: StridedRawAccess, const NDIM: usize> StridedRawAccess
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        self.0.imp().storage()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe { self.0.get_value_unchecked(multi_index) }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.0.imp().get_value_1d_unchecked(index) }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.0.imp().get_1d_unchecked(index) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe { self.0.get_unchecked(multi_index) }
    }
}

/////////// ArrayRefMut

/// Array implementation forwarding operations through an exclusive borrow.
pub struct ArrayRefMut<'a, ArrayImpl, const NDIM: usize>(&'a mut Array<ArrayImpl, NDIM>);

impl<Item: Copy + Default> crate::dense::macros::direct::Output<0>
    for Array<ArrayRefMut<'_, super::zero_dim::ScalarArrayImpl<Item>, 0>, 0>
{
    unsafe fn output_mut(&mut self, _index: [usize; 0]) -> &mut Item {
        self.imp_mut().0.scalar_mut()
    }
}

impl<'a, ArrayImpl, const NDIM: usize> BaseItem for ArrayRefMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayRefMut<'a, ArrayImpl, NDIM> {
    /// Wrap an exclusive borrow without copying array data.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ContainerType for ArrayRefMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.0.data()
    }
}

unsafe impl<ArrayImpl: StableRawAccessMut, const NDIM: usize> StableRawAccessMut
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.0.data_mut()
    }
}

impl<ArrayImpl: StridedRawAccess, const NDIM: usize> StridedRawAccess
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        self.0.imp().storage()
    }
}

impl<ArrayImpl: StridedRawAccessMut, const NDIM: usize> StridedRawAccessMut
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)> {
        self.0.imp_mut().storage_mut()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe { self.0.get_value_unchecked(multi_index) }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.0.imp().get_value_1d_unchecked(index) }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.0.imp().get_1d_unchecked(index) }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.0.imp_mut().get_1d_unchecked_mut(index) }
    }
}

// SAFETY: the view forwards indices unchanged and its source provides the
// disjointness and stability guarantee.
unsafe impl<ArrayImpl: UnsafeDisjointRandom1DAccessMut, const NDIM: usize>
    UnsafeDisjointRandom1DAccessMut for ArrayRefMut<'_, ArrayImpl, NDIM>
{
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe { self.0.get_unchecked(multi_index) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        unsafe { self.0.get_unchecked_mut(multi_index) }
    }
}

impl<ArrayImpl: ResizeInPlace<NDIM>, const NDIM: usize> ResizeInPlace<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        DynArray, StridedDynArray,
        dense::array::storage_test_support::{
            assert_storage_addresses_elements, assert_storage_consistent_with_data,
            assert_storage_mut_writes_elements,
        },
    };

    #[test]
    fn references_forward_the_storage_of_their_array() {
        let arr = StridedDynArray::<f64, 2>::row_major([3, 4]);
        let pointer = arr.storage().unwrap().0.as_ptr();
        let view = arr.r();
        assert_eq!(assert_storage_addresses_elements(&view), (12, 0));
        assert_eq!(view.storage().unwrap().0.as_ptr(), pointer);

        // The reference of a view keeps the view's offset.
        let sub = arr.r().into_subview([1, 1], [2, 3]);
        assert_eq!(assert_storage_addresses_elements(&sub.r()), (12, 5));

        let mut arr = DynArray::<f64, 3>::from_shape([2, 3, 4]);
        let pointer = arr.storage().unwrap().0.as_ptr();
        let view = arr.r_mut();
        assert_eq!(assert_storage_addresses_elements(&view), (24, 0));
        assert_eq!(view.storage().unwrap().0.as_ptr(), pointer);
    }

    #[test]
    fn mutable_references_write_through_mutable_storage() {
        let mut arr = DynArray::<f64, 3>::from_shape([2, 3, 4]);
        assert_storage_mut_writes_elements(&mut arr.r_mut());

        let mut outer = arr.r_mut();
        assert_storage_mut_writes_elements(&mut outer.r_mut());

        let mut arr = StridedDynArray::<f64, 2>::row_major([3, 4]);
        assert_storage_mut_writes_elements(&mut arr.r_mut());
    }

    #[test]
    fn reference_storage_is_consistent_with_data() {
        let mut arr = DynArray::<f64, 2>::from_shape([4, 3]);
        assert_eq!(assert_storage_consistent_with_data(&arr.r()), (12, 0));
        assert_eq!(assert_storage_consistent_with_data(&arr.r_mut()), (12, 0));

        let mut arr = StridedDynArray::<f64, 2>::row_major([4, 3]);
        assert_eq!(assert_storage_consistent_with_data(&arr.r()), (12, 0));
        assert_eq!(assert_storage_consistent_with_data(&arr.r_mut()), (12, 0));
    }
}
