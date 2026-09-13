//! Growable frames and rank-adaptive pivoted QR.
//!
//! [`ElementFrame`] works with any linear space. [`ArrayFrame`] stores all
//! columns in one column-major matrix. Use [`crate::FrameSpace::frame`] to select a
//! space's preferred backend. QR never modifies the input frame.

mod array;
mod qr;

pub use array::ArrayFrame;
pub use qr::{FrameQr, FrameQrDecomposition, FrameQrOptions};

use crate::operator::element::Element;
use crate::{Frame, FrameApply, LinearSpace, RlstError, RlstResult};

#[cfg(test)]
mod tests;

/// An element-backed frame for an arbitrary linear space.
pub struct ElementFrame<'a, Space: LinearSpace> {
    pub(super) space: &'a Space,
    pub(super) elements: Vec<Element<'a, Space>>,
}

impl<'a, Space: LinearSpace> ElementFrame<'a, Space> {
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

impl<'a, Space: LinearSpace> Frame<'a> for ElementFrame<'a, Space> {
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

fn index_error() -> RlstError {
    RlstError::IndexLayoutError("Frame column index out of bounds".into())
}

impl<'a, Space: LinearSpace> FrameApply<'a> for ElementFrame<'a, Space> {}
