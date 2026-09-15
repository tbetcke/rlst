//! Element-backed storage for arbitrary linear spaces.

use super::index_error;
use crate::operator::element::Element;
use crate::{FrameApply, FrameImpl, InnerProductSpace, LinearSpace, RlstResult};

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

    /// Iterate over stored elements without copying them.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Element<'a, Space>> {
        self.elements.iter()
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
}
