//! Abstract linear spaces and their elements.

use crate::operator::element::Element;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace: Sized {
    /// Field Type.
    type F;

    /// Concrete type associated with elements of the space.
    type Impl;

    /// Create a new zero element from the space.
    fn zero(&self) -> Element<'_, Self>;

    /// Add two elements of the space.
    fn add(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self>;

    /// Subtract two elements of the space.
    fn sub(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self>;

    /// Multiply an element of the space by a scalar.
    fn scalar_mul(&self, scalar: &Self::F, x: &Element<Self>) -> Element<'_, Self>;

    /// Negate an element of the space.
    fn neg(&self, x: &Element<Self>) -> Element<'_, Self>;

    /// Sum element `y` into  element `x`
    fn sum_inplace(&self, x: &mut Element<Self>, y: &Element<Self>);

    /// Subtract element `y` from element `x`
    fn sub_inplace(&self, x: &mut Element<Self>, y: &Element<Self>);

    /// Multiply with a scalar in place.
    fn scale_inplace(&self, scalar: &Self::F, x: &mut Element<Self>);

    /// Create a new element by copying an existing  one.
    fn copy_from(&self, x: &Element<Self>) -> Element<'_, Self>;

    /// Compute `y += alpha * x`.
    ///
    /// The default implementation forms the scaled element first and is
    /// therefore not allocation-free. Implementations whose elements are backed
    /// by arrays should override this with a fused update; nothing else is
    /// required of an implementation, and the default is always correct.
    fn axpy_inplace(&self, alpha: &Self::F, x: &Element<Self>, y: &mut Element<Self>) {
        self.sum_inplace(y, &self.scalar_mul(alpha, x));
    }
}

/// A dual space
pub trait DualSpace: LinearSpace {
    /// Dual Space
    type DualSpace: LinearSpace<F = Self::F>;

    /// Dual pairing
    fn dual_pairing(&self, x: &Element<Self>, y: &Element<Self::DualSpace>) -> Self::F;
}

/// Indexable space
pub trait IndexableSpace: LinearSpace {
    /// Dimension
    fn dimension(&self) -> usize;
}

/// Inner product space
pub trait InnerProductSpace: LinearSpace {
    /// Inner product
    ///
    /// In spaces over complex numbers it is assumed that the complex conjugate
    /// is taken with respect to the second argument.
    fn inner_product(&self, x: &Element<Self>, y: &Element<Self>) -> Self::F;
}

/// Normed space
pub trait NormedSpace: LinearSpace {
    /// The output type of the norm.
    type Output;

    /// Norm of an element.
    fn norm(&self, x: &Element<Self>) -> Self::Output;
}

/// Implementation of a growable collection of columns in one borrowed space.
///
/// [`ElementFrame`](crate::ElementFrame) forwards its operations to this trait.
/// Extracted elements own their values and borrow the space, not the frame.
/// Unless a backend validates compatibility, callers must only supply elements
/// from compatible space instances (as for ordinary element arithmetic).
pub trait FrameImpl<'a> {
    /// The underlying linear space.
    type Space: LinearSpace + 'a;
    /// The space associated with every column.
    fn space(&self) -> &'a Self::Space;
    /// Copy a column into an owned element, or return `None` out of bounds.
    fn element(&self, index: usize) -> Option<Element<'a, Self::Space>>;
    /// Replace a column. Return an error for an invalid index or incompatible input.
    fn set_element(&mut self, index: usize, elem: &Element<Self::Space>) -> crate::RlstResult<()>;
    /// Number of active columns.
    fn len(&self) -> usize;
    /// Number of columns that fit without reallocating storage.
    fn capacity(&self) -> usize;
    /// Reserve room for at least `additional` further columns.
    fn reserve(&mut self, additional: usize);
    /// Whether there are no active columns.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Copy an element into a new column.
    fn push(&mut self, elem: &Element<Self::Space>) -> crate::RlstResult<()>;
    /// Remove the last column, transferring its value into an owned element.
    fn pop(&mut self) -> Option<Element<'a, Self::Space>>;
    /// Remove all columns, retaining capacity.
    fn clear(&mut self);
}

/// Borrowing access to the columns of an element-backed frame.
///
/// Implementations that store their columns as [`Element`]s can hand out
/// references to them. Packed storages such as
/// [`ArrayFrameImpl`](crate::ArrayFrameImpl) keep their columns in one matrix
/// and have no stored element to borrow, so they do not implement this trait
/// and offer array views instead.
///
/// The mutating accessors expose the stored element directly and can replace it
/// with an incompatible value, for example one of a different dimension.
/// Callers must preserve the space's invariants, as for
/// [`Element::new`](crate::operator::element::Element::new) and
/// [`Element::imp_mut`](crate::operator::element::Element::imp_mut).
pub trait FrameRefAccess<'a>: FrameImpl<'a> {
    /// Borrow a column, or return `None` out of bounds.
    fn element_ref(&self, index: usize) -> Option<&Element<'a, Self::Space>>;

    /// Mutably borrow a column, or return `None` out of bounds.
    fn element_mut<'s>(&'s mut self, index: usize) -> Option<&'s mut Element<'a, Self::Space>>;

    /// Iterate over the columns without copying them.
    fn iter<'s>(&'s self) -> impl ExactSizeIterator<Item = &'s Element<'a, Self::Space>>
    where
        'a: 's;

    /// Mutably iterate over the columns.
    fn iter_mut<'s>(
        &'s mut self,
    ) -> impl ExactSizeIterator<Item = &'s mut Element<'a, Self::Space>>
    where
        'a: 's;
}

/// Accumulate `sum_i coefficients[i] * columns[i]` in place.
///
/// Shared by the [`FrameApply::apply`] default and element-backed frames so the
/// shape check and the allocation-free accumulation live in one place.
/// `columns` may yield owned elements or references to stored ones.
pub(crate) fn combine_columns<'a, S, A, C>(
    space: &'a S,
    len: usize,
    coefficients: &crate::Array<A, 1>,
    columns: impl IntoIterator<Item = C>,
) -> crate::RlstResult<Element<'a, S>>
where
    S: LinearSpace,
    A: crate::Shape<1> + crate::UnsafeRandom1DAccessByValue<Item = S::F>,
    C: std::borrow::Borrow<Element<'a, S>>,
{
    if coefficients.shape()[0] != len {
        return Err(crate::RlstError::SingleDimensionError {
            expected: len,
            actual: coefficients.shape()[0],
        });
    }
    let mut result = space.zero();
    for (coefficient, column) in coefficients.iter_value().zip(columns) {
        space.axpy_inplace(&coefficient, column.borrow(), &mut result);
    }
    Ok(result)
}

/// Linear combinations of frame columns.
///
/// The default operations use only the underlying space. Array frames override
/// these methods with dense matrix multiplication and require a [`crate::Gemm`]
/// scalar; storing columns does not require this additional bound.
pub trait FrameApply<'a>: FrameImpl<'a> {
    /// Form the linear combination of columns with the given coefficients.
    ///
    /// Accepts strided arrays and lazy expressions. An empty combination is zero.
    fn apply<A>(
        &self,
        coefficients: &crate::Array<A, 1>,
    ) -> crate::RlstResult<Element<'a, Self::Space>>
    where
        A: crate::Shape<1>
            + crate::UnsafeRandom1DAccessByValue<Item = <Self::Space as LinearSpace>::F>,
    {
        combine_columns(
            self.space(),
            self.len(),
            coefficients,
            (0..self.len()).map(|index| self.element(index).unwrap()),
        )
    }

    /// Compute the inner products of `x` with every column.
    ///
    /// Entry `i` of the result is `inner_product(x, column_i)`, so in complex
    /// spaces the conjugate is taken on the column and the result is `V^H x`
    /// for the column matrix `V`. When the columns are orthonormal, for example
    /// the Q factor of a pivoted QR or an Arnoldi basis, `adjoint_apply`
    /// followed by [`Self::apply_into`] projects an element onto their span
    /// without copying them. For general columns the result is `V^H x`, not a
    /// projection. The default uses the space's inner product; array frames
    /// override it with a dense matrix-vector product.
    fn adjoint_apply(
        &self,
        x: &Element<Self::Space>,
    ) -> crate::RlstResult<Vec<<Self::Space as LinearSpace>::F>>
    where
        Self::Space: InnerProductSpace,
    {
        let space = self.space();
        Ok((0..self.len())
            .map(|index| space.inner_product(x, &self.element(index).unwrap()))
            .collect())
    }

    /// Compute `output = alpha * self * coefficients + beta * output`.
    ///
    /// A zero `beta` replaces the output without reading its previous value.
    /// Shape errors leave the output unchanged. The output must belong to a
    /// compatible space instance. The default folds each column into `output`
    /// with [`LinearSpace::axpy_inplace`] and forms no temporary element; it
    /// obtains the columns through [`FrameImpl::element`], so element-backed
    /// frames override it to borrow their columns instead.
    fn apply_into<A>(
        &self,
        alpha: <Self::Space as LinearSpace>::F,
        coefficients: &crate::Array<A, 1>,
        beta: <Self::Space as LinearSpace>::F,
        output: &mut Element<Self::Space>,
    ) -> crate::RlstResult<()>
    where
        A: crate::Shape<1>
            + crate::UnsafeRandom1DAccessByValue<Item = <Self::Space as LinearSpace>::F>,
        <Self::Space as LinearSpace>::F: num::Zero + num::One + PartialEq + Copy,
    {
        accumulate_columns(
            self.space(),
            self.len(),
            alpha,
            coefficients,
            beta,
            output,
            (0..self.len()).map(|index| self.element(index).unwrap()),
        )
    }
}

/// Accumulate `output = alpha * sum_i coefficients[i] * columns[i] + beta * output`.
///
/// Shared by the [`FrameApply::apply_into`] default and element-backed frames.
/// The coefficient length is validated before `output` is touched. A zero
/// `beta` replaces `output` by the zero element first, so non-finite previous
/// values are discarded rather than propagated; a unit `beta` skips the scaling
/// pass; any other `beta` scales `output` in place. Each column is then folded
/// in with one [`LinearSpace::axpy_inplace`], so no temporary element is formed.
/// `columns` may yield owned elements or references to stored ones.
pub(crate) fn accumulate_columns<'a, S, A, C>(
    space: &S,
    len: usize,
    alpha: S::F,
    coefficients: &crate::Array<A, 1>,
    beta: S::F,
    output: &mut Element<S>,
    columns: impl IntoIterator<Item = C>,
) -> crate::RlstResult<()>
where
    S: LinearSpace + 'a,
    S::F: num::Zero + num::One + PartialEq + Copy,
    A: crate::Shape<1> + crate::UnsafeRandom1DAccessByValue<Item = S::F>,
    C: std::borrow::Borrow<Element<'a, S>>,
{
    if coefficients.shape()[0] != len {
        return Err(crate::RlstError::SingleDimensionError {
            expected: len,
            actual: coefficients.shape()[0],
        });
    }
    if num::Zero::is_zero(&beta) {
        *output.imp_mut() = space.zero().into_imp();
    } else if !num::One::is_one(&beta) {
        space.scale_inplace(&beta, output);
    }
    for (coefficient, column) in coefficients.iter_value().zip(columns) {
        space.axpy_inplace(&(alpha * coefficient), column.borrow(), output);
    }
    Ok(())
}
