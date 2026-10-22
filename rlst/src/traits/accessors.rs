//! Accessor traits for n-dimensional array-like structures.
//!
//! The traits in this module provide indexed access to array-like structures
//! of the form `arr[multi_index]` with `multi_index` an array of type `[usize; NDIM]`.
//! `NDIM` is the number of dimensions in the array. For example `arr[[3, 4, 8]]` returns
//! the element at position `[3, 4, 8]`.
//!
//! If an array structure is associated with an underlying memory region one can return
//! references to the underlying elements. However, this may not always be the case. The traits
//! in this module therefore distinguish between access by reference and access by value. For access
//! by value the underlying value type needs to support the `Copy` trait.
//!
//! The traits in this module are available as unsafe variant and as safe variant.
//! If the unsafe variant and the [Shape] trait are implemented the safe variant
//! is automatically implemented by checking whether the provided multi index is within the shape bounds.
//!
//! The traits are also available as 1d index variants. This is mainly useful for the implementation of iterators.
//! By convention in this library a 1d index is always assumed to be derived from a column-major enumeration of the
//! elements. The 1d indexed traits are only available as unsafe variants.
//!
//! For direct memory access the traits [RawAccess] and [RawAccessMut] are provided. These return the raw memory slices
//! of underlying data. [StridedRawAccess] and [StridedRawAccessMut] generalise them to views whose elements are not one
//! contiguous slice, such as subviews: they return the backing storage and the offset of the view's first element.

use super::base_operations::{BaseItem, Shape};

/// This trait provides unsafe access by value to the underlying data.
pub trait UnsafeRandomAccessByValue<const NDIM: usize>: BaseItem {
    /// Return the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item;
}

/// This trait provides unsafe access by value to the underlying data using a 1d index.
///
/// 1d indexing is always in column-major order.
///
/// # Safety
/// `index` must not be out of bounds.
pub trait UnsafeRandom1DAccessByValue: BaseItem {
    /// Return the element at position determined by `index`.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item;
}

/// This trait provides unsafe access by reference to the underlying data.
pub trait UnsafeRandomAccessByRef<const NDIM: usize>: BaseItem {
    /// Return a reference to the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item;
}

/// Provides unchecked reference access by logical one-dimensional index.
///
/// The index is the column-major enumeration of the logical shape, not necessarily a physical
/// offset into backing storage.
pub trait UnsafeRandom1DAccessByRef: BaseItem {
    /// Return a reference to the element at position determined by `index`.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item;
}

/// This trait provides unsafe mutable access to the underlying data.
pub trait UnsafeRandomAccessMut<const NDIM: usize>: BaseItem {
    /// Return a mutable reference to the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item;
}

/// Provides unchecked mutable access by logical one-dimensional index.
///
/// Implementing this trait alone does not promise that references returned for distinct indices
/// can coexist; that stronger contract is [`UnsafeDisjointRandom1DAccessMut`].
pub trait UnsafeRandom1DAccessMut: BaseItem {
    /// Return a mutable reference to the element at position determined by `index`.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
}

/// Marker for one-dimensional mutable access that is stable and pairwise disjoint.
///
/// # Safety
/// Implementors must guarantee that, while the array is mutably borrowed, every valid
/// one-dimensional index always identifies the same element and distinct indices identify
/// non-overlapping elements. Repeated calls must remain valid while references returned by
/// earlier calls are live. If the implementation also implements [`UnsafeRandomAccessMut`],
/// its multidimensional accessor must identify the same element as the corresponding
/// column-major one-dimensional index. This permits views to forward this capability and an
/// iterator to keep several returned mutable references alive simultaneously.
pub unsafe trait UnsafeDisjointRandom1DAccessMut: UnsafeRandom1DAccessMut {}

/// This trait provides bounds checked access to the underlying data by value.
pub trait RandomAccessByValue<const NDIM: usize>: BaseItem {
    /// Return the element at position determined by `multi_index`.
    /// If the value is out of bounds `None` is returned.
    fn get_value(&self, multi_index: [usize; NDIM]) -> Option<Self::Item>;
}

/// This trait provides bounds checked access to the underlying data by reference.
pub trait RandomAccessByRef<const NDIM: usize>: BaseItem {
    /// Return a reference to the element at position determined by `multi_index`.
    /// If the value is out of bounds `None` is returned.
    fn get(&self, multi_index: [usize; NDIM]) -> Option<&Self::Item>;
}

/// This trait provides bounds checked mutable access to the underlying data.
pub trait RandomAccessMut<const NDIM: usize>: BaseItem {
    /// Return a mutable reference to the element at position determined by `multi_index`.
    /// If the value is out of bounds `None` is returned.
    fn get_mut(&mut self, multi_index: [usize; NDIM]) -> Option<&mut Self::Item>;
}

/// Get raw access to the underlying data.
pub trait RawAccess: BaseItem {
    /// Get a slice of the complete backing storage.
    ///
    /// Returns `None` when the implementation cannot expose its storage as one contiguous slice.
    /// The slice follows physical storage order, not logical column-major iteration order.
    fn data(&self) -> Option<&[Self::Item]>;
}

/// Get mutable raw access to the underlying data.
pub trait RawAccessMut: RawAccess {
    /// Get a mutable slice of the whole data.
    ///
    /// Returns `None` when the implementation cannot expose its complete backing storage as one
    /// mutable contiguous slice. The slice is in physical storage order.
    fn data_mut(&mut self) -> Option<&mut [Self::Item]>;
}

/// Strided access to the backing storage of an array or view.
///
/// [`storage`](StridedRawAccess::storage) returns a slice of backing storage and the offset,
/// in elements, of the view's first element in it. With the strides of the type's
/// [`Stride`](super::base_operations::Stride) implementation, the element at multi-index `i`
/// is `storage[offset + Σ i[k] · stride[k]]`. The slice may hold elements that do not belong
/// to the view, before, between and after its own, for example the other rows of a matrix
/// around a subview.
///
/// This is a weaker promise than [`RawAccess`], whose slice holds the array's elements and
/// starts with the first one, and it is available for more views: subviews and every slice
/// of an array with strided storage. A type that implements both keeps them consistent:
/// whenever [`RawAccess::data`] returns `Some(d)`, `storage` returns `Some((s, offset))` with
/// `&s[offset..offset + d.len()]` the same memory as `d`. For an owned array that is
/// `(d, 0)`.
///
/// For a view without elements, such as a subview of shape `[0, 0]` at the far corner of its
/// parent, the offset may lie past the end of the slice; it must not be used to index it.
///
/// The trait is safe to implement: callers check the elements they address against the
/// length of the slice.
///
/// # Implementations
///
/// - Owned and borrowed arrays, [`BaseArray`](crate::dense::base_array::BaseArray) and
///   [`StridedBaseArray`](crate::dense::strided_base_array::StridedBaseArray) (`DynArray`,
///   `AlignedDynArray`, `StaticArray`, `StridedDynArray`, `SliceArray`, `StridedSliceArray` and
///   their mutable forms): their whole container at offset 0.
/// - The references of `r()` and `r_mut()`: their array's storage and offset.
/// - `slice`, `row`, `col`, `multislice` and `into_subview`: their parent's storage, with the
///   offset of their own first element added, so that nested and mixed views compose.
/// - `with_container_type`, `coerce_dim` and `insert_empty_axis`: the wrapped array's storage
///   and offset.
///
/// A view whose offset does not fit in `usize`, which only a view without elements can have,
/// returns `None`.
///
/// Not implemented by lazy expressions, which have no storage; by `transpose()`,
/// `permute_axes()`, `reverse_axis()` and `into_flat()`, which have no [`Stride`]
/// implementation; by the shared references of `into_shared` and `into_shared_atomic`, whose
/// storage sits behind a lock; and by `ZeroDimArray`.
///
/// [`Stride`]: super::base_operations::Stride
///
/// # Example
///
/// ```
/// use rlst::{DynArray, StridedRawAccess};
///
/// let mut arr = DynArray::<f64, 2>::from_shape([4, 3]);
/// arr.fill_from_iter((0..12).map(|value| value as f64));
///
/// // The 2 x 2 subview whose first element is `arr[[1, 1]]`.
/// let view = arr.r().into_subview([1, 1], [2, 2]);
/// let (storage, offset) = view.imp().storage().unwrap();
/// let stride = view.stride();
///
/// // The slice is the whole storage of `arr`, and the offset that of `arr[[1, 1]]`.
/// assert_eq!(storage.len(), 12);
/// assert_eq!(offset, 1 + 1 * 4);
/// assert_eq!(stride, [1, 4]);
/// assert_eq!(storage[offset + 1 * stride[0] + 1 * stride[1]], view[[1, 1]]);
/// assert_eq!(view[[1, 1]], arr[[2, 2]]);
/// ```
pub trait StridedRawAccess: BaseItem {
    /// The backing storage and the offset of the first element of the view in it, or `None`
    /// if the storage cannot be exposed as a slice.
    fn storage(&self) -> Option<(&[Self::Item], usize)>;
}

/// Mutable strided access to the backing storage of an array or view.
///
/// The mutable counterpart of [`StridedRawAccess`], with the same layout of the slice. A
/// mutable view hands out its parent's whole storage, elements outside the view included;
/// this is memory-safe, since the view holds the parent's exclusive borrow, but callers
/// should write only the view's elements.
///
/// It is implemented by the types that implement [`StridedRawAccess`] and give mutable access:
/// the arrays with a mutable container, `r_mut()`, and the views and wrappers of those.
/// `r()` holds a shared borrow and does not implement it.
///
/// # Example
///
/// ```
/// use rlst::{DynArray, StridedRawAccessMut};
///
/// let mut arr = DynArray::<f64, 2>::from_shape([4, 3]);
///
/// let mut view = arr.r_mut().into_subview([1, 1], [2, 2]);
/// let stride = view.stride();
/// let (storage, offset) = view.imp_mut().storage_mut().unwrap();
/// // Write the element at multi-index [1, 0] of the view.
/// storage[offset + 1 * stride[0]] = 5.0;
///
/// assert_eq!(view[[1, 0]], 5.0);
/// assert_eq!(arr[[2, 1]], 5.0);
/// ```
pub trait StridedRawAccessMut: StridedRawAccess {
    /// The mutable backing storage and the offset of the first element of the view in it, or
    /// `None` if the storage cannot be exposed as a slice.
    fn storage_mut(&mut self) -> Option<(&mut [Self::Item], usize)>;
}

/// Raw mutable storage whose allocation and extent remain stable while borrowed by an owner.
///
/// # Safety
/// Calls to [`RawAccessMut::data_mut`] must return the same allocation and extent for the
/// lifetime of the value, and that storage must remain consistent with its shape and stride.
pub unsafe trait StableRawAccessMut: RawAccessMut {}

/// Check if `multi_index` not out of bounds with respect to `shape`.
#[inline]
fn check_dimension<const NDIM: usize>(multi_index: [usize; NDIM], shape: [usize; NDIM]) -> bool {
    multi_index
        .iter()
        .zip(shape.iter())
        .fold(true, |acc, (ind, s)| acc && (ind < s))
}

impl<Mat: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>, const NDIM: usize>
    RandomAccessByValue<NDIM> for Mat
{
    fn get_value(&self, multi_index: [usize; NDIM]) -> Option<Self::Item> {
        if check_dimension(multi_index, self.shape()) {
            Some(unsafe { self.get_value_unchecked(multi_index) })
        } else {
            None
        }
    }
}

impl<Item, Mat: UnsafeRandomAccessMut<NDIM, Item = Item> + Shape<NDIM>, const NDIM: usize>
    RandomAccessMut<NDIM> for Mat
{
    fn get_mut(&mut self, multi_index: [usize; NDIM]) -> Option<&mut Self::Item> {
        if check_dimension(multi_index, self.shape()) {
            unsafe { Some(self.get_unchecked_mut(multi_index)) }
        } else {
            None
        }
    }
}

impl<Item, Mat: UnsafeRandomAccessByRef<NDIM, Item = Item> + Shape<NDIM>, const NDIM: usize>
    RandomAccessByRef<NDIM> for Mat
{
    fn get(&self, multi_index: [usize; NDIM]) -> Option<&Self::Item> {
        if check_dimension(multi_index, self.shape()) {
            unsafe { Some(self.get_unchecked(multi_index)) }
        } else {
            None
        }
    }
}
