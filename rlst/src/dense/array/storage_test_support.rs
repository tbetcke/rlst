//! Checks of [`StridedRawAccess`] and [`StridedRawAccessMut`] shared by the tests of the array
//! and view implementations.

use crate::{
    dense::{
        array::Array,
        layout::{convert_1d_nd_from_shape, convert_nd_raw},
    },
    traits::{
        accessors::{RawAccess, StridedRawAccess, StridedRawAccessMut, UnsafeRandomAccessByRef},
        base_operations::{BaseItem, Shape, Stride},
    },
};

/// The multi-indices of `shape`, in column-major order.
fn multi_indices<const NDIM: usize>(shape: [usize; NDIM]) -> impl Iterator<Item = [usize; NDIM]> {
    (0..shape.iter().product()).map(move |index| convert_1d_nd_from_shape(index, shape))
}

/// Assert that, for every multi-index `i` of `arr`, `storage[offset + Σ i[k] · stride[k]]` is
/// the element `arr[i]` itself, and return the storage length and the offset.
pub(crate) fn assert_storage_addresses_elements<ArrayImpl, const NDIM: usize>(
    arr: &Array<ArrayImpl, NDIM>,
) -> (usize, usize)
where
    ArrayImpl: StridedRawAccess + UnsafeRandomAccessByRef<NDIM> + Shape<NDIM> + Stride<NDIM>,
{
    let shape = arr.shape();
    assert!(
        shape.iter().product::<usize>() > 0,
        "the check needs a view with elements, shape {shape:?}"
    );
    let (storage, offset) = arr.storage().expect("the array exposes its storage");
    let stride = arr.stride();
    for index in multi_indices(shape) {
        let position = offset + convert_nd_raw(index, stride);
        assert!(
            std::ptr::eq(&storage[position], arr.get(index).unwrap()),
            "element {index:?} is not at storage position {position} \
             (offset {offset}, stride {stride:?}, shape {shape:?})"
        );
    }
    (storage.len(), offset)
}

/// Assert that `storage_mut` returns the slice and offset of `storage`, and that a write at the
/// storage position of each element of `arr` is seen at that element.
pub(crate) fn assert_storage_mut_writes_elements<ArrayImpl, const NDIM: usize>(
    arr: &mut Array<ArrayImpl, NDIM>,
) where
    ArrayImpl: StridedRawAccessMut
        + BaseItem<Item = f64>
        + UnsafeRandomAccessByRef<NDIM>
        + Shape<NDIM>
        + Stride<NDIM>,
{
    let shape = arr.shape();
    assert!(
        shape.iter().product::<usize>() > 0,
        "the check needs a view with elements, shape {shape:?}"
    );
    let stride = arr.stride();
    let (pointer, len, offset) = {
        let (storage, offset) = arr.storage().expect("the array exposes its storage");
        (storage.as_ptr(), storage.len(), offset)
    };
    for (count, index) in multi_indices(shape).enumerate() {
        let value = 1000.0 + count as f64;
        let (storage, storage_offset) = arr
            .storage_mut()
            .expect("the array exposes its mutable storage");
        assert_eq!(
            (storage.as_ptr(), storage.len(), storage_offset),
            (pointer, len, offset),
            "storage_mut() differs from storage()"
        );
        storage[storage_offset + convert_nd_raw(index, stride)] = value;
        assert_eq!(
            arr[index], value,
            "a write at element {index:?} of the storage is not seen in the view"
        );
    }
}

/// Assert the consistency rule of [`StridedRawAccess`]: `data()` returns `Some(d)`, and
/// `storage()` returns `Some((s, offset))` with `d` starting at `s[offset]` and ending within
/// `s`. Return the storage length and the offset.
pub(crate) fn assert_storage_consistent_with_data<ArrayImpl, const NDIM: usize>(
    arr: &Array<ArrayImpl, NDIM>,
) -> (usize, usize)
where
    ArrayImpl: RawAccess + StridedRawAccess,
{
    let data = arr.data().expect("data() returns the elements");
    let (storage, offset) = arr.storage().expect("the array exposes its storage");
    assert!(
        offset + data.len() <= storage.len(),
        "data() of length {} at offset {offset} exceeds a storage of length {}",
        data.len(),
        storage.len()
    );
    assert_eq!(
        storage[offset..].as_ptr(),
        data.as_ptr(),
        "data() does not start at the offset {offset} of the storage"
    );
    (storage.len(), offset)
}
