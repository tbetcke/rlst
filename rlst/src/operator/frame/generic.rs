//! Element-backed storage for arbitrary linear spaces.

use super::index_error;
use crate::operator::element::Element;
use crate::traits::linear_space::{accumulate_columns, combine_columns};
use crate::{
    Array, FrameApply, FrameImpl, FrameRefAccess, InnerProductSpace, LinearSpace, RlstResult,
    Shape, UnsafeRandom1DAccessByValue,
};

/// An element-backed frame for an arbitrary linear space.
pub struct GenericFrameImpl<'a, Space: LinearSpace> {
    pub(super) space: &'a Space,
    pub(super) elements: Vec<Element<'a, Space>>,
}

impl<'a, Space: LinearSpace> GenericFrameImpl<'a, Space> {
    /// Construct an empty frame borrowing `space`.
    pub fn new(space: &'a Space) -> Self {
        Self {
            space,
            elements: Vec::new(),
        }
    }
}

impl<'a, Space: LinearSpace> FrameRefAccess<'a> for GenericFrameImpl<'a, Space> {
    fn element_ref(&self, index: usize) -> Option<&Element<'a, Space>> {
        self.elements.get(index)
    }
    fn element_mut<'s>(&'s mut self, index: usize) -> Option<&'s mut Element<'a, Space>> {
        self.elements.get_mut(index)
    }
    fn iter<'s>(&'s self) -> impl ExactSizeIterator<Item = &'s Element<'a, Space>>
    where
        'a: 's,
    {
        self.elements.iter()
    }
    fn iter_mut<'s>(&'s mut self) -> impl ExactSizeIterator<Item = &'s mut Element<'a, Space>>
    where
        'a: 's,
    {
        self.elements.iter_mut()
    }
}

impl<'a, Space: LinearSpace> FrameImpl<'a> for GenericFrameImpl<'a, Space> {
    type Space = Space;

    fn space(&self) -> &'a Space {
        self.space
    }
    fn element(&self, index: usize) -> Option<Element<'a, Space>> {
        self.elements.get(index).cloned()
    }
    fn set_element(&mut self, index: usize, elem: &Element<Space>) -> RlstResult<()> {
        let target = self.elements.get_mut(index).ok_or_else(index_error)?;
        *target = self.space.copy_from(elem);
        Ok(())
    }
    fn len(&self) -> usize {
        self.elements.len()
    }
    fn capacity(&self) -> usize {
        self.elements.capacity()
    }
    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
    }
    fn push(&mut self, elem: &Element<Space>) -> RlstResult<()> {
        self.elements.push(self.space.copy_from(elem));
        Ok(())
    }
    fn pop(&mut self) -> Option<Element<'a, Space>> {
        self.elements.pop()
    }
    fn clear(&mut self) {
        self.elements.clear();
    }
}

impl<'a, Space: LinearSpace> FrameApply<'a> for GenericFrameImpl<'a, Space> {
    fn apply<A>(&self, coefficients: &Array<A, 1>) -> RlstResult<Element<'a, Space>>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = Space::F>,
    {
        // Borrow the stored elements instead of copying them through `element`.
        combine_columns(
            self.space,
            self.elements.len(),
            coefficients,
            self.elements.iter(),
        )
    }

    fn adjoint_apply(&self, x: &Element<Space>) -> RlstResult<Vec<Space::F>>
    where
        Space: InnerProductSpace,
    {
        // Borrow the stored elements instead of copying them through `element`.
        Ok(self
            .elements
            .iter()
            .map(|basis| self.space.inner_product(x, basis))
            .collect())
    }

    fn apply_into<A>(
        &self,
        alpha: Space::F,
        coefficients: &Array<A, 1>,
        beta: Space::F,
        output: &mut Element<Space>,
    ) -> RlstResult<()>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = Space::F>,
        Space::F: num::Zero + num::One + PartialEq + Copy,
    {
        // Borrow the stored elements instead of copying them through `element`.
        accumulate_columns(
            self.space,
            self.elements.len(),
            alpha,
            coefficients,
            beta,
            output,
            self.elements.iter(),
        )
    }
}
