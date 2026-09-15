//! Growable frames and rank-adaptive pivoted QR.
//!
//! [`ElementFrame`] forwards operations to a [`FrameImpl`]. [`GenericFrameImpl`]
//! stores owned elements of any linear space; [`ArrayFrameImpl`] stores columns
//! in one column-major matrix. Use [`crate::InnerProductSpace::frame`] to obtain a
//! space's preferred implementation, erased behind [`SpaceFrame`]. QR never
//! modifies the input frame.

mod array;
mod dynamic;
mod generic;
mod qr;

pub use array::ArrayFrameImpl;
pub use dynamic::{DynFrame, ErasedQrDecomposition, SpaceFrame};
pub use generic::GenericFrameImpl;
pub use qr::{FrameQr, FrameQrDecomposition, FrameQrOptions};

use std::{marker::PhantomData, ops::Neg};

use crate::operator::element::Element;
use crate::{
    Array, FrameApply, FrameImpl, InnerProductSpace, LinearSpace, RlstError, RlstNum, RlstResult,
    RlstScalar, Shape, SliceArray, SliceArrayMut, UnsafeRandom1DAccessByValue,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod forwarding_tests;

/// A frame that owns an implementation and forwards all operations to it.
///
/// The implementation determines storage and supported capabilities. Application
/// requires [`FrameApply`], and QR requires [`FrameQr`]. Elements and QR results
/// borrow the underlying space for `'a`, independently of the wrapper's lifetime.
pub struct ElementFrame<'a, ElementFrameImpl: FrameImpl<'a>> {
    imp: ElementFrameImpl,
    _space: PhantomData<&'a ElementFrameImpl::Space>,
}

impl<'a, I: FrameImpl<'a>> ElementFrame<'a, I> {
    /// Wrap an existing implementation without copying its storage.
    pub fn new(imp: I) -> Self {
        Self {
            imp,
            _space: PhantomData,
        }
    }

    /// Borrow the implementation.
    pub fn imp(&self) -> &I {
        &self.imp
    }

    /// Mutably borrow the implementation.
    pub fn imp_mut(&mut self) -> &mut I {
        &mut self.imp
    }

    /// Consume the wrapper and return its implementation without copying.
    pub fn into_imp(self) -> I {
        self.imp
    }

    /// Return the space associated with the columns.
    pub fn space(&self) -> &'a I::Space {
        self.imp.space()
    }

    /// Copy a column into an owned element, or return `None` out of bounds.
    pub fn element(&self, index: usize) -> Option<Element<'a, I::Space>> {
        self.imp.element(index)
    }

    /// Replace a column, delegating bounds and compatibility checks.
    pub fn set_element(&mut self, index: usize, elem: &Element<I::Space>) -> RlstResult<()> {
        self.imp.set_element(index, elem)
    }

    /// Number of active columns.
    pub fn len(&self) -> usize {
        self.imp.len()
    }

    /// Whether there are no active columns.
    pub fn is_empty(&self) -> bool {
        self.imp.is_empty()
    }

    /// Number of columns that fit without reallocating storage.
    pub fn capacity(&self) -> usize {
        self.imp.capacity()
    }

    /// Reserve room for at least `additional` further columns.
    pub fn reserve(&mut self, additional: usize) {
        self.imp.reserve(additional);
    }

    /// Copy an element into a new column, delegating compatibility checks.
    pub fn push(&mut self, elem: &Element<I::Space>) -> RlstResult<()> {
        self.imp.push(elem)
    }

    /// Remove the last column, or return `None` if empty.
    pub fn pop(&mut self) -> Option<Element<'a, I::Space>> {
        self.imp.pop()
    }

    /// Remove all columns, retaining capacity.
    pub fn clear(&mut self) {
        self.imp.clear();
    }
}

impl<'a, I: FrameImpl<'a>> FrameImpl<'a> for ElementFrame<'a, I> {
    type Space = I::Space;

    fn space(&self) -> &'a Self::Space {
        self.imp.space()
    }
    fn element(&self, index: usize) -> Option<Element<'a, Self::Space>> {
        self.imp.element(index)
    }
    fn set_element(&mut self, index: usize, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.imp.set_element(index, elem)
    }
    fn len(&self) -> usize {
        self.imp.len()
    }
    fn is_empty(&self) -> bool {
        self.imp.is_empty()
    }
    fn capacity(&self) -> usize {
        self.imp.capacity()
    }
    fn reserve(&mut self, additional: usize) {
        self.imp.reserve(additional);
    }
    fn push(&mut self, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.imp.push(elem)
    }
    fn pop(&mut self) -> Option<Element<'a, Self::Space>> {
        self.imp.pop()
    }
    fn clear(&mut self) {
        self.imp.clear();
    }
}

impl<'a, I: FrameApply<'a>> ElementFrame<'a, I> {
    /// Form a linear combination using the implementation's application routine.
    ///
    /// Accepts strided arrays and lazy expressions. The coefficient length must
    /// equal [`Self::len`]; an empty combination is the space's zero element.
    pub fn apply<A>(&self, coefficients: &Array<A, 1>) -> RlstResult<Element<'a, I::Space>>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = <I::Space as LinearSpace>::F>,
    {
        self.imp.apply(coefficients)
    }

    /// Compute `output = alpha * self * coefficients + beta * output`.
    ///
    /// Delegates to the implementation's [`FrameApply::apply_into`], preserving
    /// its validation and optimized application. A zero `beta` discards the old
    /// output. Shape errors leave the output unchanged.
    pub fn apply_into<A>(
        &self,
        alpha: <I::Space as LinearSpace>::F,
        coefficients: &Array<A, 1>,
        beta: <I::Space as LinearSpace>::F,
        output: &mut Element<I::Space>,
    ) -> RlstResult<()>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = <I::Space as LinearSpace>::F>,
        <I::Space as LinearSpace>::F: num::Zero,
    {
        self.imp.apply_into(alpha, coefficients, beta, output)
    }
}

impl<'a, I: FrameApply<'a>> FrameApply<'a> for ElementFrame<'a, I> {
    fn apply<A>(&self, coefficients: &Array<A, 1>) -> RlstResult<Element<'a, Self::Space>>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = <Self::Space as LinearSpace>::F>,
    {
        self.imp.apply(coefficients)
    }

    fn apply_into<A>(
        &self,
        alpha: <Self::Space as LinearSpace>::F,
        coefficients: &Array<A, 1>,
        beta: <Self::Space as LinearSpace>::F,
        output: &mut Element<Self::Space>,
    ) -> RlstResult<()>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = <Self::Space as LinearSpace>::F>,
        <Self::Space as LinearSpace>::F: num::Zero,
    {
        self.imp.apply_into(alpha, coefficients, beta, output)
    }
}

impl<'a, I: FrameQr<'a>> ElementFrame<'a, I>
where
    I::Space: InnerProductSpace,
    <I::Space as LinearSpace>::F: RlstScalar,
{
    /// Compute pivoted QR with the implementation's algorithm and wrap Q.
    ///
    /// The result satisfies `self[:, permutation] ≈ q * r` with the same
    /// implementation type for Q. See [`FrameQrOptions`] for rank selection.
    pub fn qr_pivoted(
        &self,
        options: FrameQrOptions<<<I::Space as LinearSpace>::F as RlstScalar>::Real>,
    ) -> RlstResult<FrameQrDecomposition<Self, <I::Space as LinearSpace>::F>> {
        let FrameQrDecomposition { q, r, permutation } = self.imp.qr_pivoted(options)?;
        Ok(FrameQrDecomposition {
            q: Self::new(q),
            r,
            permutation,
        })
    }
}

impl<'a, I: FrameQr<'a>> FrameQr<'a> for ElementFrame<'a, I>
where
    I::Space: InnerProductSpace,
    <I::Space as LinearSpace>::F: RlstScalar,
{
    fn qr_pivoted(
        &self,
        options: FrameQrOptions<<<Self::Space as LinearSpace>::F as RlstScalar>::Real>,
    ) -> RlstResult<FrameQrDecomposition<Self, <Self::Space as LinearSpace>::F>> {
        Self::qr_pivoted(self, options)
    }
}

impl<'a, Space: LinearSpace> ElementFrame<'a, GenericFrameImpl<'a, Space>> {
    /// Iterate over stored elements without copying them.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Element<'a, Space>> {
        self.imp.iter()
    }
}

impl<'a, Item: RlstNum + Neg<Output = Item>> ElementFrame<'a, ArrayFrameImpl<'a, Item>> {
    /// Borrow the active matrix without copying.
    pub fn matrix(&self) -> SliceArray<'_, Item, 2> {
        self.imp.matrix()
    }

    /// Mutably borrow active entries without permitting storage resizing.
    pub fn matrix_mut(&mut self) -> SliceArrayMut<'_, Item, 2> {
        self.imp.matrix_mut()
    }

    /// Borrow an active column, or return `None` out of bounds.
    pub fn column(&self, index: usize) -> Option<SliceArray<'_, Item, 1>> {
        self.imp.column(index)
    }

    /// Mutably borrow an active column without permitting storage resizing.
    pub fn column_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Item, 1>> {
        self.imp.column_mut(index)
    }
}

fn index_error() -> RlstError {
    RlstError::IndexLayoutError("Frame column index out of bounds".into())
}
