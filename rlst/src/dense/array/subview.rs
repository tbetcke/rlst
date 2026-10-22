//! Subview onto an array
use crate::dense::array::Array;

use crate::dense::layout::{
    check_multi_index_in_bounds, checked_storage_offset, convert_1d_nd_from_shape,
};
use crate::traits::{
    accessors::{
        StridedRawAccess, StridedRawAccessMut, UnsafeDisjointRandom1DAccessMut,
        UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
        UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
    },
    base_operations::{BaseItem, Shape, Stride},
};
use crate::{ContainerType, Unknown};

/// Rectangular, same-rank view into an array.
///
/// The view starts at `offset`, has its own `shape`, and preserves the source strides.
pub struct ArraySubView<ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> ArraySubView<ArrayImpl, NDIM> {
    /// Create a rectangular view with the given origin and shape.
    ///
    /// # Panics
    ///
    /// Panics if the computed end of the view exceeds the source extent on any axis.
    /// Bounds arithmetic uses ordinary `usize` addition, so callers must provide offsets and
    /// extents whose sums are representable.
    pub fn new(arr: Array<ArrayImpl, NDIM>, offset: [usize; NDIM], shape: [usize; NDIM]) -> Self {
        let arr_shape = arr.shape();
        for index in 0..NDIM {
            assert!(
                offset[index] + shape[index] <= arr_shape[index],
                "View out of bounds for dimension {}. {} > {}",
                index,
                offset[index] + shape[index],
                arr_shape[index]
            )
        }
        Self { arr, offset, shape }
    }
}

// Basic traits for ArraySubView

impl<ArrayImpl, const NDIM: usize> ContainerType for ArraySubView<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = Unknown;
}

impl<ArrayImpl, const NDIM: usize> BaseItem for ArraySubView<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const NDIM: usize> Shape<NDIM> for ArraySubView<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArraySubView<ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.arr
                .get_value_unchecked(offset_multi_index(multi_index, self.offset))
        }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        unsafe {
            self.arr
                .get_unchecked(offset_multi_index(multi_index, self.offset))
        }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape)) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.get_unchecked(convert_1d_nd_from_shape(index, self.shape)) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape)) }
    }
}

// SAFETY: adding a fixed offset is injective, so the source array's disjoint
// mapping is preserved by this view.
unsafe impl<ArrayImpl, const NDIM: usize> UnsafeDisjointRandom1DAccessMut
    for ArraySubView<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeDisjointRandom1DAccessMut + UnsafeRandomAccessMut<NDIM>,
{
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(crate::dense::layout::check_multi_index_in_bounds(
            multi_index,
            self.shape()
        ));
        unsafe {
            self.arr
                .get_unchecked_mut(offset_multi_index(multi_index, self.offset))
        }
    }
}

// The origin of the view is its first element; the strides are the parent's.
impl<ArrayImpl: StridedRawAccess + Stride<NDIM>, const NDIM: usize> StridedRawAccess
    for ArraySubView<ArrayImpl, NDIM>
{
    fn storage(&self) -> Option<(&[Self::Item], usize)> {
        let stride = self.arr.stride();
        let (storage, offset) = self.arr.storage()?;
        Some((
            storage,
            checked_storage_offset(offset, self.offset, stride)?,
        ))
    }
}

impl<ArrayImpl: StridedRawAccessMut + Stride<NDIM>, const NDIM: usize> StridedRawAccessMut
    for ArraySubView<ArrayImpl, NDIM>
{
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)> {
        let stride = self.arr.stride();
        let origin = self.offset;
        let (storage, offset) = self.arr.storage_mut()?;
        Some((storage, checked_storage_offset(offset, origin, stride)?))
    }
}

fn offset_multi_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    offset: [usize; NDIM],
) -> [usize; NDIM] {
    let mut output = [0; NDIM];
    for (ind, elem) in output.iter_mut().enumerate() {
        *elem = multi_index[ind] + offset[ind]
    }
    output
}

#[cfg(test)]
mod tests {
    use crate::{
        DynArray, StridedDynArray,
        dense::array::{
            StridedSliceArray,
            storage_test_support::{
                assert_storage_addresses_elements, assert_storage_mut_writes_elements,
            },
        },
    };

    #[test]
    fn subviews_add_the_offset_of_their_origin() {
        // Column-major, strides [1, 5].
        let mat = DynArray::<f64, 2>::from_shape([5, 6]);
        let view = mat.r().into_subview([1, 2], [3, 2]);
        assert_eq!(assert_storage_addresses_elements(&view), (30, 11));

        // Row-major, strides [5, 1].
        let mat = StridedDynArray::<f64, 2>::row_major([4, 5]);
        let view = mat.r().into_subview([1, 2], [2, 3]);
        assert_eq!(assert_storage_addresses_elements(&view), (20, 7));
    }

    #[test]
    fn nested_and_mixed_views_compose() {
        // Column-major, strides [1, 6].
        let mat = DynArray::<f64, 2>::from_shape([6, 6]);
        let view = mat
            .r()
            .into_subview([1, 1], [4, 4])
            .into_subview([1, 2], [2, 2]);
        assert_eq!(assert_storage_addresses_elements(&view), (36, 2 + 3 * 6));

        // Column-major, strides [1, 4, 20].
        let arr = DynArray::<f64, 3>::from_shape([4, 5, 6]);
        let slice_of_subview = arr.r().into_subview([1, 1, 1], [3, 3, 3]).slice::<2>(2, 1);
        assert_eq!(
            assert_storage_addresses_elements(&slice_of_subview),
            (120, 1 + 4 + 2 * 20)
        );

        // The slice has shape [4, 6] and strides [1, 20] at offset 2 * 4.
        let subview_of_slice = arr.r().slice::<2>(1, 2).into_subview([1, 2], [2, 3]);
        assert_eq!(
            assert_storage_addresses_elements(&subview_of_slice),
            (120, 8 + 1 + 2 * 20)
        );

        // A subview of a subview of a row-major array, strides [30, 6, 1].
        let arr = StridedDynArray::<f64, 3>::row_major([4, 5, 6]);
        let view = arr
            .r()
            .into_subview([1, 1, 1], [3, 4, 5])
            .into_subview([1, 0, 2], [2, 2, 2]);
        assert_eq!(
            assert_storage_addresses_elements(&view),
            (120, 2 * 30 + 6 + 3)
        );
    }

    #[test]
    fn views_write_through_mutable_storage() {
        let mut mat = DynArray::<f64, 2>::from_shape([6, 6]);
        assert_storage_mut_writes_elements(&mut mat.r_mut().into_subview([1, 2], [3, 2]));
        assert_storage_mut_writes_elements(
            &mut mat
                .r_mut()
                .into_subview([1, 1], [4, 4])
                .into_subview([1, 2], [2, 2]),
        );

        let mut arr = DynArray::<f64, 3>::from_shape([4, 5, 6]);
        assert_storage_mut_writes_elements(
            &mut arr
                .r_mut()
                .into_subview([1, 1, 1], [3, 3, 3])
                .slice::<2>(2, 1),
        );
        assert_storage_mut_writes_elements(
            &mut arr.r_mut().slice::<2>(1, 2).into_subview([1, 2], [2, 3]),
        );

        let mut mat = StridedDynArray::<f64, 2>::row_major([4, 5]);
        assert_storage_mut_writes_elements(&mut mat.r_mut().into_subview([1, 2], [2, 3]));
    }

    #[test]
    fn offset_of_an_empty_subview_may_lie_past_the_storage() {
        // The subview of shape [0, 0] at the far corner of a column-major [4, 3] array.
        let mat = DynArray::<f64, 2>::from_shape([4, 3]);
        let view = mat.r().into_subview([4, 3], [0, 0]);
        let (storage, offset) = view.storage().unwrap();
        assert_eq!((storage.len(), offset), (12, 16));
    }

    #[test]
    fn subview_whose_offset_overflows_has_no_storage() {
        // An array without elements accepts any stride.
        let values: [f64; 0] = [];
        let arr = StridedSliceArray::from_shape_and_stride(&values, [0, 5], [1, usize::MAX]);
        assert!(arr.storage().is_some());
        let view = arr.r().into_subview([0, 4], [0, 1]);
        assert!(view.storage().is_none());
    }
}
