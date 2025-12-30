//! Basic types.

mod number_traits_impl;
mod rlst_num_impl;

use std::ops::Add;

pub use num::complex::Complex32 as c32;
pub use num::complex::Complex64 as c64;

use thiserror::Error;
use typenum::Cmp;
use typenum::Const;
use typenum::Less;
use typenum::NonZero;
use typenum::PInt;
use typenum::ToUInt;
use typenum::U;
use typenum::Unsigned;

use crate::Array;
use crate::ArrayFromContainerType;
use crate::ContainerTypeRepr;
use crate::ContainerTypeSelector;
use crate::DynArray;
use crate::IsSmallerThan;
use crate::RowMajorArrayFromContainerType;
use crate::StridedDynArray;
use crate::SubtractEqual;
use crate::dense::array::StaticArray;
use crate::dense::base_array::BaseArray;
use crate::dense::data_container::ArrayContainer;
use crate::dense::layout::row_major_stride_from_shape;
use crate::dense::strided_base_array::StridedBaseArray;
use crate::traits::number_relations::{IsGreaterByOne, IsGreaterZero, IsSmallerByOne};

use typenum;

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum RlstError {
    /// Not implemented.
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    /// Operation failed.
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    /// Matrix is empty.
    #[error("Matrix has empty dimension {0:#?}.")]
    MatrixIsEmpty((usize, usize)),
    /// Dimension mismatch.
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Index layout error.
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    /// MPI rank error.
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
    /// Incompatible stride for Lapack.
    #[error("Incompatible stride for Lapack.")]
    IncompatibleStride,
    /// Lapack error.
    #[error("Lapack error: {0}")]
    LapackError(#[from] LapackError),
    /// General error.
    #[error("{0}")]
    GeneralError(String),
    /// I/O error.
    #[error("I/O Error: {0}")]
    IoError(String),
    /// Matrix is not square.
    #[error("Matrix is not square. Dimension: {0}x{1}")]
    MatrixNotSquare(usize, usize),
    /// Matrix is not Hermitian.
    #[error("Matrix is not Hermitian (complex conjugate symmetric).")]
    MatrixNotHermitian,
}

/// Alias for an Rlst Result type.
pub type RlstResult<T> = std::result::Result<T, RlstError>;

/// Transposition Mode.
#[derive(Clone, Copy, PartialEq)]
pub enum TransMode {
    /// No modification of matrix.
    NoTrans,
    /// Complex conjugate of matrix.
    ConjNoTrans,
    /// Transposition of matrix.
    Trans,
    /// Conjugate transpose of matrix.
    ConjTrans,
}

// The following is a workaround to allow for implied trait bounds on associated types.
// A full descripton can be found at: https://docs.rs/imply-hack/latest/imply_hack/index.html

/// Trait to specify associated bounds as super trait bounds.
/// This allows the bounds to be implied and not explicitly stated in the type signature.
/// See the [imply-hack](https://docs.rs/imply-hack/latest/imply_hack/index.html) crate for more
/// details.
pub trait Imply<T>: sealed::ImplyInner<T, Is = T> {}

impl<T, U> Imply<T> for U {}

mod sealed {
    pub trait ImplyInner<T> {
        type Is;
    }

    impl<T, U> ImplyInner<T> for U {
        type Is = T;
    }
}

/// Memory layout of an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Column major.
    ColumnMajor,
    /// Row major.
    RowMajor,
    /// Unknown.
    Unknown,
}

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum LapackError {
    /// Info code from LAPACK.
    #[error("LAPACK error code: {0}")]
    LapackInfoCode(i32),
}

/// Alias for a Lapack Result type.
pub type LapackResult<T> = std::result::Result<T, LapackError>;

/// Empty type for number relations.
pub struct NumberType<const N: usize>;

/// Empty type for difference M - N of numbers;
pub struct NumberDifference<const M: usize, const N: usize>;

// the condition that M = 1 + N.
impl<const M: usize, const N: usize> IsGreaterByOne<N> for NumberType<M>
where
    PInt<U<N>>: Add<typenum::P1, Output = PInt<U<M>>>,
    Const<N>: ToUInt,
    Const<M>: ToUInt,
    <Const<N> as ToUInt>::Output: Unsigned + NonZero,
    <Const<M> as ToUInt>::Output: Unsigned + NonZero,
{
}

// The condition that D + N = M
impl<const D: usize, const M: usize, const N: usize> SubtractEqual<D> for NumberDifference<M, N>
where
    Const<M>: ToUInt,
    Const<N>: ToUInt,
    Const<D>: ToUInt,
    <Const<N> as ToUInt>::Output: Unsigned + NonZero,
    <Const<M> as ToUInt>::Output: Unsigned + NonZero,
    <Const<D> as ToUInt>::Output: Unsigned + NonZero,
    NumberType<N>: IsSmallerThan<M>,
    PInt<U<D>>: Add<PInt<U<N>>, Output = PInt<U<M>>>,
{
    const OUTPUT: usize = D;
}

// the condition that M = N - 1
impl<const M: usize, const N: usize> IsSmallerByOne<N> for NumberType<M> where
    NumberType<N>: IsGreaterByOne<M>
{
}

// Implement the condition that N is not zero.
impl<const N: usize> IsGreaterZero for NumberType<N>
where
    Const<N>: ToUInt,
    <Const<N> as ToUInt>::Output: NonZero,
{
}

// Implement the condition that M < N.
impl<const M: usize, const N: usize> IsSmallerThan<N> for NumberType<M>
where
    Const<N>: ToUInt,
    Const<M>: ToUInt,
    <Const<N> as ToUInt>::Output: NonZero + Unsigned,
    <Const<M> as ToUInt>::Output: NonZero + Unsigned,
    PInt<U<M>>: Cmp<PInt<U<N>>, Output = Less>,
{
}

/// Determine whether a matrix is upper or lower triangular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    /// Upper triangular matrix.
    Upper,
    /// Lower triangular matrix.
    Lower,
}

// Container types.

/// An unknown container type.
pub struct Unknown;

/// A static container with fixed size N.
pub struct Stack<const N: usize>;

impl ContainerTypeRepr for Unknown {
    const STR: &str = "Unknown";
}
impl<const N: usize> ContainerTypeRepr for Stack<N> {
    const STR: &str = "Stack";
}

impl ArrayFromContainerType for Unknown {
    type Output<Item, const NDIM: usize>
        = DynArray<Item, NDIM>
    where
        Item: Copy + Default;

    fn new_arr<Item, const NDIM: usize>(shape: [usize; NDIM]) -> Self::Output<Item, NDIM>
    where
        Item: Copy + Default,
    {
        DynArray::from_shape(shape)
    }
}

impl RowMajorArrayFromContainerType for Unknown {
    type Output<Item, const NDIM: usize>
        = StridedDynArray<Item, NDIM>
    where
        Item: Copy + Default;

    fn new_row_major_arr<Item, const NDIM: usize>(shape: [usize; NDIM]) -> Self::Output<Item, NDIM>
    where
        Item: Copy + Default,
    {
        StridedDynArray::row_major(shape)
    }
}

impl<const N: usize> ArrayFromContainerType for Stack<N> {
    type Output<Item, const NDIM: usize>
        = StaticArray<Item, NDIM, N>
    where
        Item: Copy + Default;

    fn new_arr<Item, const NDIM: usize>(shape: [usize; NDIM]) -> Self::Output<Item, NDIM>
    where
        Item: Copy + Default,
    {
        Array::new(BaseArray::new(ArrayContainer::<Item, N>::new(), shape))
    }
}

impl<const N: usize> RowMajorArrayFromContainerType for Stack<N> {
    type Output<Item, const NDIM: usize>
        = Array<StridedBaseArray<ArrayContainer<Item, N>, NDIM>, NDIM>
    where
        Item: Copy + Default;

    fn new_row_major_arr<Item, const NDIM: usize>(shape: [usize; NDIM]) -> Self::Output<Item, NDIM>
    where
        Item: Copy + Default,
    {
        Array::new(StridedBaseArray::new(
            ArrayContainer::<Item, N>::new(),
            shape,
            row_major_stride_from_shape(shape),
        ))
    }
}

/// Trait to select a container type based on the input types.
pub struct SelectContainerType;

// For combining two arrays with unkown type we select `Unknown`
impl ContainerTypeSelector<Unknown, Unknown> for SelectContainerType {
    type Type = Unknown;
}

// For combining a stack array and an unkown array we select `stack`.
// In this way, evaluating the array gives back a stack size. This makes sense since
// the output size is known as only arrays of the same size can be combined.
impl<const N: usize> ContainerTypeSelector<Stack<N>, Unknown> for SelectContainerType {
    type Type = Stack<N>;
}

// As above.
impl<const N: usize> ContainerTypeSelector<Unknown, Stack<N>> for SelectContainerType {
    type Type = Stack<N>;
}

// Two stack arrays of same size again give a stack array.
impl<const N: usize> ContainerTypeSelector<Stack<N>, Stack<N>> for SelectContainerType {
    type Type = Stack<N>;
}
