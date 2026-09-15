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
        if coefficients.shape()[0] != self.len() {
            return Err(crate::RlstError::SingleDimensionError {
                expected: self.len(),
                actual: coefficients.shape()[0],
            });
        }
        let space = self.space();
        let mut result = space.zero();
        for (index, coefficient) in coefficients.iter_value().enumerate() {
            let scaled = space.scalar_mul(&coefficient, &self.element(index).unwrap());
            space.sum_inplace(&mut result, &scaled);
        }
        Ok(result)
    }

    /// Compute the inner products of `x` with every column.
    ///
    /// Entry `i` of the result is `inner_product(x, column_i)`, so in complex
    /// spaces the conjugate is taken on the column and the result is `V^H x`
    /// for the column matrix `V`. Together with [`Self::apply_into`] this
    /// projects an element onto the span of the columns without copying them.
    /// The default uses the space's inner product; array frames override it
    /// with a dense matrix-vector product.
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
    /// compatible space instance.
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
        <Self::Space as LinearSpace>::F: num::Zero,
    {
        let space = self.space();
        let mut result = self.apply(coefficients)?;
        space.scale_inplace(&alpha, &mut result);
        if !num::Zero::is_zero(&beta) {
            let previous = space.scalar_mul(&beta, output);
            space.sum_inplace(&mut result, &previous);
        }
        *output.imp_mut() = result.into_imp();
        Ok(())
    }
}
