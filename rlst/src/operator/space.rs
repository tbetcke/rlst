//! Additional implementations for spaces.

use crate::{DualSpace, Inner, InnerProductSpace, LinearSpace, NormedSpace, RlstScalar};

use super::element::Element;

impl<Item: RlstScalar, S: InnerProductSpace<F = Item>> NormedSpace for S {
    type Output = <Item as RlstScalar>::Real;

    fn norm(&self, x: &super::element::Element<Self>) -> Self::Output {
        x.inner(x).abs().sqrt()
    }
}

/// Create the additive identity belonging to this specific space instance.
///
/// The returned element borrows `space`, preventing it from outliving the
/// object that defines its operations and instance-level metadata.
pub fn zero_element<Space: LinearSpace>(space: &Space) -> Element<'_, Space> {
    space.zero()
}

impl<S: InnerProductSpace> DualSpace for S {
    type DualSpace = Self;

    fn dual_pairing(&self, x: &Element<Self>, y: &Element<Self::DualSpace>) -> Self::F {
        self.inner_product(x, y)
    }
}
