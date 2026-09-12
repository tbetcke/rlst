//! Operations for sparse matrices

use std::ops::{Add, AddAssign, Mul, Sub};

use crate::base_types::{c32, c64};
use crate::{BaseItem, FromAij, Shape};

use super::{
    binary_operator::BinaryAijOperator, csr_mat::CsrMatrix, unary_aij_operator::UnaryAijOperator,
};

/// Matrix structure defined through an iterator.
pub struct SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    iter: I,
    shape: [usize; 2],
}

/// Iterator that multiplies each entry  by a scalar.
/// This is necessary to avoid issues with variable capturing in the closure for the iterator.
/// Basically, since we use a closure of type  `fn(Item) -> Out`, we cannot capture the scalar directly
/// as this is not allowed for `fn` style closures.
pub struct ScalarMulIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    iter: I,
    scalar: Item,
}

impl<Item, I> ScalarMulIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    /// Create a new scalar multiplication iterator.
    pub fn new(iter: I, scalar: Item) -> Self {
        Self { iter, scalar }
    }
}

impl<Item, I> Iterator for ScalarMulIterator<Item, I>
where
    Item: Copy + Default + Mul<Item, Output = Item>,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Item = ([usize; 2], Item);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(index, value)| (index, value * self.scalar))
    }
}

impl<Item, I> SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    /// Create a new sparse matrix operation.
    pub fn new(iter: I, shape: [usize; 2]) -> Self {
        Self { iter, shape }
    }

    /// Get the iterator over the sparse matrix entries.
    pub fn iter(&self) -> &I {
        &self.iter
    }

    /// Apply a function to the explicitly stored entries.
    ///
    /// Implicit zeros are not visited. Mapped values equal to `Out::default()`
    /// are omitted, preserving the canonical sparse representation. Use this
    /// method when stored-entry semantics are intended for a function that does
    /// not preserve zero (for example, `op.map_stored(f64::exp)`).
    pub fn map_stored<Out, F>(
        self,
        func: F,
    ) -> SparseMatOpIterator<Out, UnaryAijOperator<Item, I, Out, F>>
    where
        Out: Copy + Default + PartialEq,
        F: Fn(Item) -> Out,
    {
        let shape = self.shape;
        SparseMatOpIterator::new(UnaryAijOperator::new(self.iter, func), shape)
    }

    /// Convert into a Csr matrix.
    pub fn into_csr(self) -> CsrMatrix<Item>
    where
        Item: Copy + Default + PartialEq + AddAssign,
    {
        let shape = self.shape();

        CsrMatrix::from_aij_iter(shape, self.iter)
    }
}

impl<Item, I> Shape<2> for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl<Item, I> BaseItem for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Item = Item;
}

macro_rules! impl_unary_op {
    ($trait_name:ident, $method_name:ident) => {
        impl<Item, I> crate::traits::number_traits::$trait_name for SparseMatOpIterator<Item, I>
        where
            Item: Copy + Default + crate::traits::number_traits::$trait_name,
            <Item as crate::traits::number_traits::$trait_name>::Output: Copy + Default + PartialEq,
            I: Iterator<Item = ([usize; 2], Item)>,
        {
            type Output = SparseMatOpIterator<
                <Item as crate::traits::number_traits::$trait_name>::Output,
                UnaryAijOperator<
                    Item,
                    I,
                    <Item as crate::traits::number_traits::$trait_name>::Output,
                    fn(Item) -> <Item as crate::traits::number_traits::$trait_name>::Output,
                >,
            >;

            fn $method_name(self) -> Self::Output {
                let shape = self.shape();
                SparseMatOpIterator::new(
                    UnaryAijOperator::new(self.iter, |val: Item| val.$method_name()),
                    shape,
                )
            }
        }
    };
}

impl_unary_op!(Abs, abs);
impl_unary_op!(Square, square);
impl_unary_op!(AbsSquare, abs_square);
impl_unary_op!(Sqrt, sqrt);
impl_unary_op!(Sin, sin);
impl_unary_op!(Tan, tan);
impl_unary_op!(Asin, asin);
impl_unary_op!(Atan, atan);
impl_unary_op!(Sinh, sinh);
impl_unary_op!(Tanh, tanh);
impl_unary_op!(Asinh, asinh);
impl_unary_op!(Atanh, atanh);

impl<Item1, Item2, Out, I1, I2> std::ops::Add<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Add<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn add(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        assert_eq!(self.shape(), other.shape());
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a + b),
            shape,
        )
    }
}

impl<Item1, Item2, Out, I1, I2> std::ops::Sub<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Sub<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn sub(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        assert_eq!(self.shape(), other.shape());
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a - b),
            shape,
        )
    }
}

macro_rules! impl_scalar_mult {
    ($scalar_type:ty) => {
        impl<I> std::ops::Mul<SparseMatOpIterator<$scalar_type, I>> for $scalar_type
        where
            I: Iterator<Item = ([usize; 2], $scalar_type)>,
        {
            type Output = SparseMatOpIterator<$scalar_type, ScalarMulIterator<$scalar_type, I>>;

            fn mul(self, other: SparseMatOpIterator<$scalar_type, I>) -> Self::Output {
                let shape = other.shape();
                let new_iter = ScalarMulIterator::new(other.iter, self);
                SparseMatOpIterator::new(new_iter, shape)
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);

impl<Item, I> std::ops::Mul<Item> for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default + Mul<Item, Output = Item>,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Output = SparseMatOpIterator<Item, ScalarMulIterator<Item, I>>;

    fn mul(self, scalar: Item) -> Self::Output {
        let shape = self.shape();
        let new_iter = ScalarMulIterator::new(self.iter, scalar);
        SparseMatOpIterator::new(new_iter, shape)
    }
}

impl<Item1, Item2, Out, I1, I2> std::ops::Mul<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Mul<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn mul(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        assert_eq!(self.shape(), other.shape());
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a * b),
            shape,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AijIteratorByValue, FromAij, Nonzeros};

    #[test]
    fn binary_operations_require_equal_shapes_in_both_orders() {
        let a = CsrMatrix::from_aij([0, 2], &[], &[], &[] as &[f64]);
        let b = CsrMatrix::from_aij([1, 2], &[], &[], &[] as &[f64]);

        assert!(std::panic::catch_unwind(|| a.op() + b.op()).is_err());
        assert!(std::panic::catch_unwind(|| b.op() + a.op()).is_err());
        assert!(std::panic::catch_unwind(|| a.op() - b.op()).is_err());
        assert!(std::panic::catch_unwind(|| a.op() * b.op()).is_err());
    }

    #[test]
    fn map_stored_ignores_implicit_and_elides_mapped_zeros() {
        let matrix = CsrMatrix::from_aij([2, 2], &[0, 0, 1], &[0, 1, 1], &[0.0, 2.0, 3.0]);
        assert_eq!(matrix.nnz(), 2);

        let exponential = matrix.op().map_stored(f64::exp).into_csr();
        assert_eq!(exponential.nnz(), 2);
        assert_eq!(
            exponential.iter_aij_value().collect::<Vec<_>>(),
            vec![([0, 1], 2.0_f64.exp()), ([1, 1], 3.0_f64.exp())]
        );
        assert_eq!(exponential.todense().data().unwrap()[0], 0.0);

        let elided = matrix
            .op()
            .map_stored(|value| if value == 2.0 { 0.0 } else { value })
            .into_csr();
        assert_eq!(elided.nnz(), 1);
        assert_eq!(
            elided.iter_aij_value().collect::<Vec<_>>(),
            vec![([1, 1], 3.0)]
        );
    }

    #[test]
    fn map_stored_filters_many_entries_without_recursion() {
        let count = 100_000;
        let rows = (0..count).collect::<Vec<_>>();
        let cols = vec![0; count];
        let values = vec![1.0; count];
        let matrix = CsrMatrix::from_aij([count, 1], &rows, &cols, &values);
        let mapped = matrix.op().map_stored(|_| 0.0).into_csr();
        assert_eq!(mapped.nnz(), 0);
    }
}
