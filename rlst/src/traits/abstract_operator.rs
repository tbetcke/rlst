//! Abstract linear operator interface.
//!
//! An abstract linear operator is a linear mapping A:X -> Y, where
//! X and Y are arbitrary vector spaces. The space `X` is the `domain`
//! space and the space `Y` is the `range` space. Every abstract linear operator
//! provides an [OperatorBase::apply] method that implements the map `y = Ax` for
//! `x` an element of `X`.
//!
//! To define an abstract linear operator one needs to implement the [OperatorBase] trait on a
//! type `Op_Impl`. The type `Operator<Op_Impl: OperatorBase>` then implements a complete algebra,
//! including [std::ops::Add], [std::ops::Sub], [std::ops::Mul] for a scalar parameter, [std::ops::AddAssign],
//! [std::ops::SubAssign], [std::ops::Neg], and [std::ops::MulAssign] for operators as long as the underlying
//! spaces support the corresponding operations on their elements.

use num::{One, Zero};

use crate::{LinearSpace, operator::element::Element};

/// Interface between an abstract linear operator and its implementation.
///
/// `Domain` and `Range` identify the spaces whose elements are accepted by
/// [`Self::apply`]. Implementations must preserve the BLAS-like algebraic contract
/// `y <- alpha A x + beta y`. Implementations may still evaluate the prior value of `y`
/// when `beta` is zero, so initialize `y` to finite values unless the concrete operator
/// documents otherwise.
pub trait OperatorBase {
    /// The domain space of the operator.
    type Domain: LinearSpace;
    /// The range space of the operator.
    type Range: LinearSpace;

    /// Return a reference to the concrete domain space.
    fn domain(&self) -> &Self::Domain;
    /// Return a reference to the concrete range space.
    fn range(&self) -> &Self::Range;

    /// Apply A in place as `y <- alpha A x + beta y`.
    ///
    /// `x` must belong to this operator's domain and `y` to its range; their
    /// space types enforce this, while implementations may additionally rely
    /// on both values having been created by the returned concrete space
    /// instances.
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut Element<Self::Range>,
    );

    /// Apply the operator `A` to `x` and return `y = Ax`.
    #[inline(always)]
    fn dot(&self, x: &Element<Self::Domain>) -> Element<'_, Self::Range>
    where
        <Self::Range as crate::LinearSpace>::F: One + Zero,
    {
        let mut res = self.range().zero();
        self.apply(One::one(), x, Zero::zero(), &mut res);
        res
    }
}
