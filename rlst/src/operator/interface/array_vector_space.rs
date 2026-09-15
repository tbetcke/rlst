//! Implementation of operator concepts for dense arrays.
//!
//! This module defines an [ArrayVectorSpace], a concrete implementation of [LinearSpace] based on
//! vectors represented through a 1d [Array](crate::dense::array::Array). A [ArrayVectorSpace] is also
//! an [InnerProductSpace] and a [NormedSpace](crate::NormedSpace). For the inner product the second vector
//! is taken as complex conjugate.

use std::{marker::PhantomData, ops::Neg};

use crate::{
    ArrayFrameImpl, ElementFrame, EvaluateArray, Gemm, InnerProductSpace, Lapack, LinearSpace,
    RlstNum, RlstScalar, SpaceFrame, dense::array::DynArray, operator::element::Element,
};

/// Finite-dimensional vector space backed by dynamic one-dimensional arrays.
///
/// Elements created by this space have the configured dimension. Arithmetic
/// assumes both operands came from a compatible space instance; mismatched
/// array lengths are rejected by the underlying array operations.
pub struct ArrayVectorSpace<Item> {
    dimension: usize,
    _marker: PhantomData<Item>,
}

impl<Item> ArrayVectorSpace<Item> {
    /// Create a vector space whose elements contain `dimension` scalars.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            _marker: PhantomData,
        }
    }
}

impl<Item: RlstNum + Neg<Output = Item>> ArrayVectorSpace<Item> {
    /// Construct an empty matrix-backed frame borrowing this space.
    ///
    /// [`InnerProductSpace::frame`] returns the same storage with the backend
    /// erased. Use this method instead when the zero-copy matrix and column
    /// views of [`ArrayFrameImpl`] are needed.
    pub fn array_frame(&self) -> ElementFrame<'_, ArrayFrameImpl<'_, Item>> {
        ElementFrame::new(ArrayFrameImpl::new(self))
    }
}

impl<Item> LinearSpace for ArrayVectorSpace<Item>
where
    Item: Copy
        + Default
        + 'static
        + std::ops::Add<Output = Item>
        + std::ops::Sub<Output = Item>
        + std::ops::Neg<Output = Item>
        + std::ops::AddAssign<Item>
        + std::ops::SubAssign<Item>
        + std::ops::Mul<Item, Output = Item>
        + std::ops::MulAssign<Item>,
{
    type F = Item;

    type Impl = DynArray<Item, 1>;

    fn zero(&self) -> crate::operator::element::Element<'_, Self> {
        Element::new(self, DynArray::<Item, 1>::from_shape([self.dimension]))
    }

    fn add(
        &self,
        x: &crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<'_, Self> {
        Element::new(self, (x.imp().r() + y.imp().r()).eval())
    }

    fn sub(
        &self,
        x: &crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<'_, Self> {
        Element::new(self, (x.imp().r() - y.imp().r()).eval())
    }

    fn scalar_mul(
        &self,
        scalar: &Self::F,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<'_, Self> {
        Element::new(
            self,
            DynArray::<_, 1>::new_from(&x.imp().r().scalar_mul(*scalar)),
        )
    }

    fn neg(
        &self,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<'_, Self> {
        Element::new(self, x.imp().r().neg().eval())
    }

    fn sum_inplace(
        &self,
        x: &mut crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) {
        *x.imp_mut() += y.imp().r();
    }

    fn sub_inplace(
        &self,
        x: &mut crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) {
        *x.imp_mut() -= y.imp().r();
    }

    fn scale_inplace(&self, scalar: &Self::F, x: &mut crate::operator::element::Element<Self>) {
        *x.imp_mut() *= *scalar;
    }

    fn copy_from(
        &self,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<'_, Self> {
        Element::new(self, x.imp().eval())
    }
}

impl<Item> crate::IndexableSpace for ArrayVectorSpace<Item>
where
    Self: LinearSpace,
{
    fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<Item> InnerProductSpace for ArrayVectorSpace<Item>
where
    Item: RlstScalar + Lapack + Gemm,
{
    fn inner_product(&self, x: &Element<Self>, other: &Element<Self>) -> Self::F {
        if x.imp().shape() == [0] && other.imp().shape() == [0] {
            return Item::zero();
        }
        x.imp().inner(other.imp()).unwrap()
    }

    /// Select matrix storage, so that linear combinations use BLAS and QR uses
    /// LAPACK. See [`Self::array_frame`] for the unerased backend.
    fn frame(&self) -> ElementFrame<'_, SpaceFrame<'_, Self>>
    where
        Self::F: RlstScalar,
    {
        ElementFrame::new(SpaceFrame::new(ArrayFrameImpl::new(self)))
    }
}
