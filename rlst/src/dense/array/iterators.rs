//! Various iterator implementations.

use std::marker::PhantomData;
use std::ops::Add;

use itertools::Itertools;
use typenum::{Const, NonZero, PInt, ToUInt, U, Unsigned};

use crate::dense::array::multislice::Multislice;
use crate::dense::array::reference::ArrayRef;
use crate::dense::array::{Array, Shape};
use crate::dense::layout::{col_major_stride_from_shape, convert_1d_nd_from_shape, convert_nd_raw};
use crate::traits::accessors::{
    UnsafeDisjointRandom1DAccessMut, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::traits::{base_operations::BaseItem, iterators::AsMultiIndex};
use crate::{IsSmallerThan, NumberType, UnsafeRandom1DAccessByRef, UnsafeRandomAccessByRef};

use super::reference::{self, ArrayRefMut};
use super::slice::ArraySlice;

/// Default column major iterator.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIteratorByValue<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

/// Default column major iterator by reference.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIteratorByRef<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

/// Mutable default iterator. Like [ArrayDefaultIteratorByRef] but with mutable access.
pub struct ArrayDefaultIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

/// A multi-index iterator returns the corrent element and the corresponding multi-index.
pub struct MultiIndexIterator<I, const NDIM: usize> {
    shape: [usize; NDIM],
    iter: I,
}

impl<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> Iterator
    for MultiIndexIterator<I, NDIM>
{
    type Item = ([usize; NDIM], T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((index, value)) = self.iter.next() {
            Some((convert_1d_nd_from_shape(index, self.shape), value))
        } else {
            None
        }
    }
}

impl<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> AsMultiIndex<T, I, NDIM> for I {
    fn multi_index(self, shape: [usize; NDIM]) -> MultiIndexIterator<I, NDIM> {
        MultiIndexIterator::<I, NDIM> { shape, iter: self }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorByValue<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new default iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            pos: 0,
            nelements: arr.len(),
        }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorByRef<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new default iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            pos: 0,
            nelements: arr.len(),
        }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new mutable default iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        let nelements = arr.len();
        Self {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIteratorByValue<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { Some(self.arr.imp().get_value_1d_unchecked(self.pos)) };
            self.pos += 1;
            value
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIteratorByRef<'a, ArrayImpl, NDIM>
{
    type Item = &'a ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { Some(self.arr.imp().get_1d_unchecked(self.pos)) };
            self.pos += 1;
            value
        }
    }
}

// In the following have to use transmute to manually change the lifetime of the data
// obtained by `get_mut` to the lifetime 'a of the matrix. The borrow checker cannot see
// that the reference obtained by get_mut is bound to the lifetime of the iterator due
// to the mutable reference in its initialization.
// See also: https://stackoverflow.com/questions/62361624/lifetime-parameter-problem-in-custom-iterator-over-mutable-references
// And also: https://users.rust-lang.org/t/when-is-transmuting-lifetimes-useful/56140

impl<'a, ArrayImpl: UnsafeDisjointRandom1DAccessMut, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
{
    type Item = &'a mut ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe {
                std::mem::transmute::<
                    &mut <ArrayImpl as BaseItem>::Item,
                    &'a mut <ArrayImpl as BaseItem>::Item,
                >(self.arr.imp_mut().get_1d_unchecked_mut(self.pos))
            };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Row iterator
pub struct RowIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

impl<'a, ArrayImpl> RowIterator<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, 2>) -> Self {
        let nrows = arr.shape()[0];
        RowIterator {
            arr,
            nrows,
            current_row: 0,
        }
    }
}

/// Mutable row iterator
pub struct RowIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

impl<'a, ArrayImpl> RowIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, 2>) -> Self {
        let nrows = arr.shape()[0];
        RowIteratorMut {
            arr,
            nrows,
            current_row: 0,
        }
    }
}

impl<'a, ArrayImpl> std::iter::Iterator for RowIterator<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    type Item = Array<ArraySlice<reference::ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.r().slice(0, self.current_row);
        self.current_row += 1;
        Some(slice)
    }
}

impl<'a, ArrayImpl> std::iter::Iterator for RowIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    type Item = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.r_mut().slice(0, self.current_row);
        self.current_row += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<ArraySlice<ArrayRefMut<'_, ArrayImpl, 2>, 2, 1>, 1>,
                Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

/// Column iterator
pub struct ColIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'a, ArrayImpl> ColIterator<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, 2>) -> Self {
        let ncols = arr.shape()[1];
        ColIterator {
            arr,
            ncols,
            current_col: 0,
        }
    }
}

/// Mutable column iterator
pub struct ColIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'a, ArrayImpl> ColIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, 2>) -> Self {
        let ncols = arr.shape()[1];
        ColIteratorMut {
            arr,
            ncols,
            current_col: 0,
        }
    }
}

impl<'a, ArrayImpl: Shape<2>> std::iter::Iterator for ColIterator<'a, ArrayImpl, 2> {
    type Item = Array<ArraySlice<reference::ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.r().slice(1, self.current_col);
        self.current_col += 1;
        Some(slice)
    }
}

impl<'a, ArrayImpl> std::iter::Iterator for ColIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    type Item = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.r_mut().slice(1, self.current_col);
        self.current_col += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<ArraySlice<ArrayRefMut<'_, ArrayImpl, 2>, 2, 1>, 1>,
                Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

/// The number of diagonal elements: the smallest extent, or one for a zero-dimensional array,
/// whose one element is indexed by `[]`.
fn diagonal_length<const NDIM: usize>(shape: [usize; NDIM]) -> usize {
    shape.into_iter().min().unwrap_or(1)
}

/// Diagonal iterator.
pub struct ArrayDiagIteratorByValue<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIteratorByValue<'a, ArrayImpl, NDIM> {
    /// Create a new diagonal iterator.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        let nelements = diagonal_length(arr.shape());
        ArrayDiagIteratorByValue {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIteratorByValue<'a, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { self.arr.get_value_unchecked([self.pos; NDIM]) };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Diagonal iterator.
pub struct ArrayDiagIteratorByRef<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIteratorByRef<'a, ArrayImpl, NDIM> {
    /// Create a new diagonal iterator.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        let nelements = diagonal_length(arr.shape());
        ArrayDiagIteratorByRef {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIteratorByRef<'a, ArrayImpl, NDIM>
{
    type Item = &'a ArrayImpl::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { self.arr.get_unchecked([self.pos; NDIM]) };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Mutable diagonal iterator.
pub struct ArrayDiagIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIteratorMut<'a, ArrayImpl, NDIM> {
    /// Create a new mutable diagonal iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        let nelements = diagonal_length(arr.shape());
        ArrayDiagIteratorMut {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIteratorMut<'a, ArrayImpl, NDIM>
{
    type Item = &'a mut ArrayImpl::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe {
                std::mem::transmute::<&mut ArrayImpl::Item, Self::Item>(
                    self.arr.get_unchecked_mut([self.pos; NDIM]),
                )
            };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Multislice iterator.
pub struct MultisliceIterator<
    'a,
    Item,
    ArrayImpl,
    const NDIM: usize,
    const SDIM: usize,
    const OUTDIM: usize,
> where
    Const<NDIM>: ToUInt,
    Const<SDIM>: ToUInt,
    Const<OUTDIM>: ToUInt,
    <Const<NDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<SDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<OUTDIM> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<SDIM>: IsSmallerThan<NDIM>,
    PInt<U<OUTDIM>>: Add<PInt<U<SDIM>>, Output = PInt<U<NDIM>>>,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>,
{
    arr: &'a Array<ArrayImpl, NDIM>,
    axes: [usize; SDIM],
    axes_shape: [usize; SDIM],
    nindices: usize,
    indices: [usize; SDIM],
    is_finished: bool,
    _marker: PhantomData<Item>,
}

impl<'a, Item, ArrayImpl, const NDIM: usize, const SDIM: usize, const OUTDIM: usize>
    MultisliceIterator<'a, Item, ArrayImpl, NDIM, SDIM, OUTDIM>
where
    Const<NDIM>: ToUInt,
    Const<SDIM>: ToUInt,
    Const<OUTDIM>: ToUInt,
    <Const<NDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<SDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<OUTDIM> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<SDIM>: IsSmallerThan<NDIM>,
    PInt<U<OUTDIM>>: Add<PInt<U<SDIM>>, Output = PInt<U<NDIM>>>,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>,
{
    /// Create a new multislice iterator.
    ///
    /// See [Array::multislice_iter] for details.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>, axes: [usize; SDIM]) -> Self {
        if !axes
            .iter()
            .tuple_windows()
            .all(|(elem1, elem2)| elem2 > elem1)
        {
            panic!("`axes` must be sorted in ascending order.");
        }

        let shape = arr.shape();
        let axes_shape = {
            let mut tmp = [0; SDIM];

            for (index, elem) in tmp.iter_mut().enumerate() {
                *elem = shape[axes[index]];
            }
            tmp
        };

        let nindices = axes_shape.iter().product::<usize>();
        Self {
            arr,
            axes,
            axes_shape,
            nindices,
            indices: [0; SDIM],
            is_finished: nindices == 0,
            _marker: PhantomData,
        }
    }
}

impl<'a, Item, ArrayImpl, const NDIM: usize, const SDIM: usize, const OUTDIM: usize> Iterator
    for MultisliceIterator<'a, Item, ArrayImpl, NDIM, SDIM, OUTDIM>
where
    Const<NDIM>: ToUInt,
    Const<SDIM>: ToUInt,
    Const<OUTDIM>: ToUInt,
    <Const<NDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<SDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<OUTDIM> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<SDIM>: IsSmallerThan<NDIM>,
    PInt<U<OUTDIM>>: Add<PInt<U<SDIM>>, Output = PInt<U<NDIM>>>,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>
        + UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>,
{
    type Item = (
        [usize; SDIM],
        Array<Multislice<ArrayRef<'a, ArrayImpl, NDIM>, NDIM, SDIM, OUTDIM>, OUTDIM>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_finished {
            None
        } else {
            let out = (
                self.indices,
                self.arr.r().multislice(self.axes, self.indices),
            );

            let index_1d =
                convert_nd_raw(self.indices, col_major_stride_from_shape(self.axes_shape));

            if 1 + index_1d < self.nindices {
                self.indices = convert_1d_nd_from_shape(1 + index_1d, self.axes_shape);
            } else {
                self.is_finished = true;
            }

            Some(out)
        }
    }
}

/// Mutable Multislice iterator.
pub struct MultisliceIteratorMut<
    'a,
    Item,
    ArrayImpl,
    const NDIM: usize,
    const SDIM: usize,
    const OUTDIM: usize,
> where
    Const<NDIM>: ToUInt,
    Const<SDIM>: ToUInt,
    Const<OUTDIM>: ToUInt,
    <Const<NDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<SDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<OUTDIM> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<SDIM>: IsSmallerThan<NDIM>,
    PInt<U<OUTDIM>>: Add<PInt<U<SDIM>>, Output = PInt<U<NDIM>>>,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item>
        + UnsafeRandomAccessMut<NDIM, Item = Item>
        + Shape<NDIM>,
{
    arr: &'a mut Array<ArrayImpl, NDIM>,
    axes: [usize; SDIM],
    axes_shape: [usize; SDIM],
    nindices: usize,
    indices: [usize; SDIM],
    is_finished: bool,
    _marker: PhantomData<Item>,
}

impl<'a, Item, ArrayImpl, const NDIM: usize, const SDIM: usize, const OUTDIM: usize>
    MultisliceIteratorMut<'a, Item, ArrayImpl, NDIM, SDIM, OUTDIM>
where
    Const<NDIM>: ToUInt,
    Const<SDIM>: ToUInt,
    Const<OUTDIM>: ToUInt,
    <Const<NDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<SDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<OUTDIM> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<SDIM>: IsSmallerThan<NDIM>,
    PInt<U<OUTDIM>>: Add<PInt<U<SDIM>>, Output = PInt<U<NDIM>>>,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item>
        + UnsafeRandomAccessMut<NDIM, Item = Item>
        + Shape<NDIM>,
{
    /// Create a new mutable multislice iterator.
    ///
    /// See [Array::multislice_iter_mut] for details.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>, axes: [usize; SDIM]) -> Self {
        if !axes
            .iter()
            .tuple_windows()
            .all(|(elem1, elem2)| elem2 > elem1)
        {
            panic!("`axes` must be sorted in ascending order.");
        }

        let shape = arr.shape();
        let axes_shape = {
            let mut tmp = [0; SDIM];

            for (index, elem) in tmp.iter_mut().enumerate() {
                *elem = shape[axes[index]];
            }
            tmp
        };

        let nindices = axes_shape.iter().product::<usize>();
        Self {
            arr,
            axes,
            axes_shape,
            nindices,
            indices: [0; SDIM],
            is_finished: nindices == 0,
            _marker: PhantomData,
        }
    }
}

impl<'a, Item, ArrayImpl, const NDIM: usize, const SDIM: usize, const OUTDIM: usize> Iterator
    for MultisliceIteratorMut<'a, Item, ArrayImpl, NDIM, SDIM, OUTDIM>
where
    Const<NDIM>: ToUInt,
    Const<SDIM>: ToUInt,
    Const<OUTDIM>: ToUInt,
    <Const<NDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<SDIM> as ToUInt>::Output: Unsigned + NonZero,
    <Const<OUTDIM> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<SDIM>: IsSmallerThan<NDIM>,
    PInt<U<OUTDIM>>: Add<PInt<U<SDIM>>, Output = PInt<U<NDIM>>>,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item>
        + UnsafeRandomAccessMut<NDIM, Item = Item>
        + Shape<NDIM>,
{
    type Item = (
        [usize; SDIM],
        Array<Multislice<ArrayRefMut<'a, ArrayImpl, NDIM>, NDIM, SDIM, OUTDIM>, OUTDIM>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_finished {
            None
        } else {
            let out = (self.indices, unsafe {
                std::mem::transmute::<
                    Array<Multislice<ArrayRefMut<'_, ArrayImpl, NDIM>, NDIM, SDIM, OUTDIM>, OUTDIM>,
                    Array<Multislice<ArrayRefMut<'a, ArrayImpl, NDIM>, NDIM, SDIM, OUTDIM>, OUTDIM>,
                >(self.arr.r_mut().multislice(self.axes, self.indices))
            });

            let index_1d =
                convert_nd_raw(self.indices, col_major_stride_from_shape(self.axes_shape));

            if 1 + index_1d < self.nindices {
                self.indices = convert_1d_nd_from_shape(1 + index_1d, self.axes_shape);
            } else {
                self.is_finished = true;
            }

            Some(out)
        }
    }
}
