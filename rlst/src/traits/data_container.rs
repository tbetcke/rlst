//! Traits for data containers.
//!
//! A [DataContainer] is a low-level interface to physical data. Its derived
//! traits [ValueDataContainer], [RefDataContainer], and [RefDataContainerMut]
//! provides value-based access, reference based access, or mutable reference based access.
//! The [ModifiableDataContainer] defines a setter routine that can modify elements by value.
//! By default a data container is not resizeable. To define a resizeable data container implement
//! the [ResizeableDataContainer] trait. For raw access to the underlying data the traits
//! [RawAccessDataContainer] and [MutableRawAccessDataContainer] are provided.
//! The [ContainerType] trait attaches an associated type that can be used to distinguish between containers
//! on a type level.

use crate::{Array, MutableArrayImpl, RawAccessMut, Stride};

/// Base trait for a data container. It defines the type of the underlying data
/// and provides a method to return the number of elements.
pub trait DataContainer {
    /// Item type
    type Item: Copy + Default;

    /// Return the number of elements in the container.
    fn number_of_elements(&self) -> usize;
}

/// A data container that provides value based access.
pub trait ValueDataContainer: DataContainer {
    /// Access the container unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item;
}

/// A data container that provides reference based access.
pub trait RefDataContainer: DataContainer {
    /// Access the container by reference
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item;
}

/// A data container that provides access by mutable reference.
pub trait RefDataContainerMut: DataContainer {
    /// Access the container mutably unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
}

/// Mutable container whose distinct indices are stable, disjoint elements.
///
/// # Safety
/// Every valid index must identify the same element for the duration of a mutable borrow,
/// and distinct indices must identify non-overlapping storage. Implementations must permit
/// references to several distinct indices to remain live simultaneously.
pub unsafe trait DisjointRefDataContainerMut: RefDataContainerMut {}

/// A data container that can modify data by value.
pub trait ModifiableDataContainer: DataContainer {
    /// Set the value at a given index.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn set_unchecked_value(&mut self, index: usize, value: Self::Item);
}

/// A resizeable data container that can change its size at runtime.
pub trait ResizeableDataContainer: DataContainer {
    /// Resize the data container by providing the new length in `new_len`.
    fn resize(&mut self, new_len: usize);
}

/// A data container that provides raw access to the data slice.
pub trait RawAccessDataContainer: DataContainer {
    /// Return the complete container storage as a slice.
    ///
    /// Returns `None` if storage is not available as one contiguous slice.
    fn data(&self) -> Option<&[Self::Item]>;
}

/// A data container that provides mutable raw access to the data slice.
pub trait MutableRawAccessDataContainer: RawAccessDataContainer {
    /// Return the complete container storage as a mutable slice.
    ///
    /// Returns `None` if storage is not available as one contiguous slice.
    fn data_mut(&mut self) -> Option<&mut [Self::Item]>;
}

/// Raw-access container with stable allocation and extent.
///
/// # Safety
/// The slice exposed by raw access must retain its allocation and length for the lifetime of
/// the container unless an operation requiring exclusive ownership explicitly resizes it.
pub unsafe trait StableRawAccessDataContainer: MutableRawAccessDataContainer {}

/// Stores the type of a container.
pub trait ContainerType {
    /// The type for the container.
    type Type: ContainerTypeRepr + RowMajorArrayFromContainerType + ArrayFromContainerType;

    /// Returns the type hint for the container as string ref.
    fn type_as_str(&self) -> &'static str {
        Self::Type::STR
    }
}

/// Associate a string representation with a container type.
pub trait ContainerTypeRepr {
    /// The string representation of the container type.
    const STR: &str;
}

/// Create a new array from a given container type.
pub trait ArrayFromContainerType {
    /// Array Output implementation type
    type OutputImpl<Item, const NDIM: usize>: MutableArrayImpl<Item, NDIM>
        + RawAccessMut<Item = Item>
        + Stride<NDIM>
    where
        Item: Copy + Default;

    /// Given the shape, create a new column-major array
    fn new_arr<Item, const NDIM: usize>(
        shape: [usize; NDIM],
    ) -> Array<Self::OutputImpl<Item, NDIM>, NDIM>
    where
        Item: Copy + Default;
}

/// Create a new row-major array from a given container type.
pub trait RowMajorArrayFromContainerType {
    /// Array Output type
    type OutputImpl<Item, const NDIM: usize>: MutableArrayImpl<Item, NDIM>
        + RawAccessMut<Item = Item>
        + Stride<NDIM>
    where
        Item: Copy + Default;

    /// Given the shape, create a new row-major array.
    fn new_row_major_arr<Item, const NDIM: usize>(
        shape: [usize; NDIM],
    ) -> Array<Self::OutputImpl<Item, NDIM>, NDIM>
    where
        Item: Copy + Default;
}

/// Selects a container type based on two other container types.
pub trait ContainerTypeSelector<U: ContainerTypeRepr, V: ContainerTypeRepr> {
    /// Select the container type.
    type Type: ContainerTypeRepr + ArrayFromContainerType + RowMajorArrayFromContainerType;
}
