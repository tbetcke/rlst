//! Matrix-backed frame storage.

use std::ops::Neg;

use crate::operator::{element::Element, interface::array_vector_space::ArrayVectorSpace};
use crate::{
    Array, AsMatrixApply, DynArray, ElementFrame, FrameApply, FrameImpl, FrameSpace, Gemm,
    IndexableSpace, LinearSpace, RlstError, RlstNum, RlstResult, Shape, SliceArray, SliceArrayMut,
    UnsafeRandom1DAccessByValue,
};

/// A growable column-major matrix of elements of an [`ArrayVectorSpace`].
///
/// The allocated column capacity is separate from the active column count.
/// Matrix and column views expose only active entries and cannot resize storage.
pub struct ArrayFrameImpl<'a, Item> {
    pub(super) space: &'a ArrayVectorSpace<Item>,
    pub(super) data: DynArray<Item, 2>,
    pub(super) len: usize,
}

impl<'a, Item: RlstNum + Neg<Output = Item>> ArrayFrameImpl<'a, Item> {
    /// Construct an empty matrix-backed frame.
    pub fn new(space: &'a ArrayVectorSpace<Item>) -> Self {
        Self {
            space,
            data: DynArray::from_shape([space.dimension(), 0]),
            len: 0,
        }
    }

    /// Borrow the active matrix without copying.
    pub fn matrix(&self) -> SliceArray<'_, Item, 2> {
        let m = self.space.dimension();
        SliceArray::from_shape(&self.data.data().unwrap()[..m * self.len], [m, self.len])
    }

    /// Mutably borrow active entries without permitting storage resizing.
    pub fn matrix_mut(&mut self) -> SliceArrayMut<'_, Item, 2> {
        let m = self.space.dimension();
        SliceArrayMut::from_shape(
            &mut self.data.data_mut().unwrap()[..m * self.len],
            [m, self.len],
        )
    }

    /// Borrow an active column without copying, or return `None` out of bounds.
    pub fn column(&self, index: usize) -> Option<SliceArray<'_, Item, 1>> {
        if index >= self.len {
            return None;
        }
        let m = self.space.dimension();
        Some(SliceArray::from_shape(
            &self.data.data().unwrap()[m * index..m * (index + 1)],
            [m],
        ))
    }

    /// Mutably borrow an active column without permitting storage resizing.
    pub fn column_mut(&mut self, index: usize) -> Option<SliceArrayMut<'_, Item, 1>> {
        if index >= self.len {
            return None;
        }
        let m = self.space.dimension();
        Some(SliceArrayMut::from_shape(
            &mut self.data.data_mut().unwrap()[m * index..m * (index + 1)],
            [m],
        ))
    }

    fn validate(&self, elem: &Element<ArrayVectorSpace<Item>>) -> RlstResult<()> {
        for actual in [elem.space().dimension(), elem.imp().shape()[0]] {
            if actual != self.space.dimension() {
                return Err(RlstError::SingleDimensionError {
                    expected: self.space.dimension(),
                    actual,
                });
            }
        }
        Ok(())
    }
}

impl<'a, Item: RlstNum + Neg<Output = Item>> FrameImpl<'a> for ArrayFrameImpl<'a, Item> {
    type Space = ArrayVectorSpace<Item>;

    fn space(&self) -> &'a Self::Space {
        self.space
    }
    fn element(&self, index: usize) -> Option<Element<'a, Self::Space>> {
        Some(Element::new(
            self.space,
            DynArray::new_from(&self.column(index)?),
        ))
    }
    fn set_element(&mut self, index: usize, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.validate(elem)?;
        self.column_mut(index)
            .ok_or_else(super::index_error)?
            .fill_from(elem.imp());
        Ok(())
    }
    fn len(&self) -> usize {
        self.len
    }
    fn capacity(&self) -> usize {
        self.data.shape()[1]
    }
    fn reserve(&mut self, additional: usize) {
        let required = self
            .len
            .checked_add(additional)
            .expect("Frame capacity overflow");
        if required <= self.capacity() {
            return;
        }
        let capacity = required.max(self.capacity().saturating_mul(2)).max(1);
        let m = self.space.dimension();
        m.checked_mul(capacity)
            .expect("Frame allocation size overflow");
        let mut data = DynArray::from_shape([m, capacity]);
        let active = m * self.len;
        data.data_mut().unwrap()[..active].copy_from_slice(&self.data.data().unwrap()[..active]);
        self.data = data;
    }
    fn push(&mut self, elem: &Element<Self::Space>) -> RlstResult<()> {
        self.validate(elem)?;
        self.reserve(1);
        self.len += 1;
        self.column_mut(self.len - 1).unwrap().fill_from(elem.imp());
        Ok(())
    }
    fn pop(&mut self) -> Option<Element<'a, Self::Space>> {
        let index = self.len.checked_sub(1)?;
        let result = self.element(index);
        self.len = index;
        Some(result.unwrap())
    }
    fn clear(&mut self) {
        self.len = 0;
    }
}

impl<'a, Item: RlstNum + Neg<Output = Item> + Gemm> FrameApply<'a> for ArrayFrameImpl<'a, Item> {
    fn apply<A>(&self, coefficients: &Array<A, 1>) -> RlstResult<Element<'a, Self::Space>>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = Item>,
    {
        let mut result = self.space.zero();
        self.apply_into(Item::one(), coefficients, Item::zero(), &mut result)?;
        Ok(result)
    }

    fn apply_into<A>(
        &self,
        alpha: Item,
        coefficients: &Array<A, 1>,
        beta: Item,
        output: &mut Element<Self::Space>,
    ) -> RlstResult<()>
    where
        A: Shape<1> + UnsafeRandom1DAccessByValue<Item = Item>,
    {
        self.validate(output)?;
        if coefficients.shape()[0] != self.len {
            return Err(RlstError::SingleDimensionError {
                expected: self.len,
                actual: coefficients.shape()[0],
            });
        }
        // Avoid empty native calls and explicitly preserve beta=0 semantics.
        if self.len == 0 || self.space.dimension() == 0 || alpha.is_zero() {
            for value in output.imp_mut().data_mut().unwrap() {
                *value = if beta.is_zero() {
                    Item::zero()
                } else {
                    beta * *value
                };
            }
            return Ok(());
        }
        // Packing O(n) coefficients also supports lazy and strided inputs; the
        // O(m*n) frame data is borrowed directly without copying columns.
        let coefficients = DynArray::new_from(coefficients);
        self.matrix()
            .apply(alpha, &coefficients, beta, output.imp_mut());
        Ok(())
    }
}

impl<Item: RlstNum + Neg<Output = Item>> FrameSpace for ArrayVectorSpace<Item> {
    type FrameImpl<'a> = ArrayFrameImpl<'a, Item>;
    fn frame(&self) -> ElementFrame<'_, Self::FrameImpl<'_>> {
        ElementFrame::new(ArrayFrameImpl::new(self))
    }
}
