//! Change the type hint of an array to a different type.

use crate::{
    ArrayFromContainerType, ContainerType, ContainerTypeRepr, RowMajorArrayFromContainerType,
    dense::array::Array,
    traits::{
        accessors::{
            RawAccess, RawAccessMut, StridedRawAccess, StridedRawAccessMut,
            UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
            UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, Shape, Stride},
    },
};

/// A wrapper around an array that coerces its dimension.
pub struct WithEvalType<ArrayImpl, TypeHint, const NDIM: usize>
where
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    /// The array whose type hint is to be changed.
    arr: Array<ArrayImpl, NDIM>,
    _type_hint: std::marker::PhantomData<TypeHint>,
}

impl<ArrayImpl, TypeHint, const NDIM: usize> WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    /// Create a new `WithTypeHint`.
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            _type_hint: std::marker::PhantomData,
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Coerce the array to a specific item type and dimension.
    pub fn with_container_type<
        TypeHint: ContainerTypeRepr + ArrayFromContainerType + RowMajorArrayFromContainerType,
    >(
        self,
    ) -> Array<WithEvalType<ArrayImpl, TypeHint, NDIM>, NDIM> {
        Array::new(WithEvalType::new(self))
    }
}

impl<ArrayImpl, TypeHint: ContainerTypeRepr, const NDIM: usize> ContainerType
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: ContainerType,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    type Type = TypeHint;
}

impl<ArrayImpl, TypeHint, const NDIM: usize> BaseItem for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: BaseItem,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, TypeHint, const NDIM: usize> Shape<NDIM> for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: Shape<NDIM>,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> Stride<NDIM>
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: Stride<NDIM>,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> RawAccess for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: RawAccess,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.arr.data()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> RawAccessMut
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: RawAccessMut,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.arr.data_mut()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> StridedRawAccess
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: StridedRawAccess,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        self.arr.storage()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> StridedRawAccessMut
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: StridedRawAccessMut,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)> {
        self.arr.storage_mut()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe { self.arr.get_value_unchecked(multi_index) }
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM>,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe { self.arr.get_unchecked(multi_index) }
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM>,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        unsafe { self.arr.get_unchecked_mut(multi_index) }
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandom1DAccessByValue
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.arr.imp().get_value_1d_unchecked(index) }
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandom1DAccessByRef
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByRef,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.arr.imp().get_1d_unchecked(index) }
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandom1DAccessMut
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.arr.imp_mut().get_1d_unchecked_mut(index) }
    }
}

// SAFETY: the type hint wrapper forwards one-dimensional indices unchanged.
unsafe impl<ArrayImpl, TypeHint, const NDIM: usize> crate::UnsafeDisjointRandom1DAccessMut
    for WithEvalType<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: crate::UnsafeDisjointRandom1DAccessMut,
    TypeHint: RowMajorArrayFromContainerType + ArrayFromContainerType + ContainerTypeRepr,
{
}

#[cfg(test)]
mod tests {
    use crate::{
        DynArray, Unknown,
        dense::array::storage_test_support::{
            assert_storage_addresses_elements, assert_storage_consistent_with_data,
            assert_storage_mut_writes_elements,
        },
    };

    #[test]
    fn type_hint_forwards_the_storage() {
        let mut mat = DynArray::<f64, 2>::from_shape([3, 4]);
        let view = mat.r().with_container_type::<Unknown>();
        assert_eq!(assert_storage_addresses_elements(&view), (12, 0));
        assert_eq!(assert_storage_consistent_with_data(&view), (12, 0));

        let view = mat
            .r()
            .into_subview([1, 1], [2, 3])
            .with_container_type::<Unknown>();
        assert_eq!(assert_storage_addresses_elements(&view), (12, 4));

        assert_storage_mut_writes_elements(&mut mat.r_mut().with_container_type::<Unknown>());
        assert_storage_mut_writes_elements(
            &mut mat
                .r_mut()
                .into_subview([1, 1], [2, 3])
                .with_container_type::<Unknown>(),
        );
    }
}
