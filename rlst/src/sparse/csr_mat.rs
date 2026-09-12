//! Definition of CSR matrices.
//!
//! A CSR matrix consists of three arrays.
//! - `data` - Stores all entries of the CSR matrix.
//! - `indices` - The column indices associated with each entry in `data`.
//! - `indptr` - An array of offsets. Entries for row `i` occupy `indptr[i]..indptr[i + 1]`.
//!
//! The last entry of `indptr` is the number of nonzero elements of the sparse matrix.

use std::ops::{Add, AddAssign, Mul};

use crate::dense::array::reference::{ArrayRef, ArrayRefMut};
use crate::dense::array::slice::ArraySlice;
use crate::sparse::tools::normalize_aij;
use crate::{AijIteratorByValue, BaseItem, Shape, dense::array::DynArray, sparse::SparseMatType};
use crate::{
    AijIteratorMut, Array, AsMatrixApply, FromAij, Nonzeros, SparseMatrixType,
    UnsafeDisjointRandom1DAccessMut, UnsafeRandomAccessByValue, UnsafeRandomAccessMut, empty_array,
};
use itertools::{Itertools, izip};
use num::One;

use super::mat_operations::SparseMatOpIterator;

/// A CSR matrix
pub struct CsrMatrix<Item> {
    /// The `mat_type` denotes the storage type for the sparse matrix.
    mat_type: SparseMatType,
    /// The shape of the sparse matrix.
    shape: [usize; 2],
    /// The array of column indices.
    indices: DynArray<usize, 1>,
    /// The array of index pointers.
    indptr: DynArray<usize, 1>,
    /// The entries of the sparse matrix.
    data: DynArray<Item, 1>,
}

impl<Item> CsrMatrix<Item> {
    fn from_canonical_aij(
        shape: [usize; 2],
        rows: Vec<usize>,
        cols: Vec<usize>,
        data: Vec<Item>,
    ) -> Self
    where
        Item: Copy + Default,
    {
        let nelems = data.len();
        let mut indptr = Vec::<usize>::with_capacity(1 + shape[0]);
        let mut count = 0;
        for row in 0..shape[0] {
            indptr.push(count);
            while count < nelems && rows[count] == row {
                count += 1;
            }
        }
        indptr.push(count);

        Self {
            mat_type: SparseMatType::Csr,
            shape,
            indices: DynArray::from_shape_and_vec([nelems], cols),
            indptr: DynArray::from_shape_and_vec([1 + shape[0]], indptr),
            data: DynArray::from_shape_and_vec([nelems], data),
        }
    }

    /// Create a new CSR matrix from raw CSR arrays.
    ///
    /// The input structure is validated. Entries are then put into canonical
    /// row-major order, duplicate coordinates are summed, and entries equal to
    /// `Item::default()` are removed. Consequently, every matrix constructed by
    /// this function has sorted, unique column indices in each row and stores no
    /// explicit zeros.
    ///
    /// # Panics
    ///
    /// Panics unless `indptr` has `shape[0] + 1` entries, starts at zero, is
    /// nondecreasing, and ends at the common length of `indices` and `data`.
    /// Also panics if any column index is at least `shape[1]`.
    pub fn new(
        shape: [usize; 2],
        indices: DynArray<usize, 1>,
        indptr: DynArray<usize, 1>,
        data: DynArray<Item, 1>,
    ) -> Self
    where
        Item: AddAssign + PartialEq + Copy + Default,
    {
        assert_eq!(indptr.len(), 1 + shape[0]);
        assert_eq!(data.len(), indices.len());
        assert_eq!(indptr.get_value([0]).unwrap(), 0);
        assert_eq!(*indptr.data().unwrap().last().unwrap(), data.len());

        for (first, second) in indptr.iter_value().tuple_windows() {
            assert!(
                first <= second,
                "Elements of indptr not in increasing order {first} > {second}."
            );
        }
        if let Some(&max_col_index) = indices.data().unwrap().iter().max() {
            assert!(max_col_index < shape[1]);
        }

        let mut rows = Vec::with_capacity(data.len());
        for (row, (start, end)) in indptr.iter_value().tuple_windows().enumerate() {
            rows.extend(std::iter::repeat_n(row, end - start));
        }
        let (rows, cols, data) = normalize_aij(
            &rows,
            indices.data().unwrap(),
            data.data().unwrap(),
            SparseMatType::Csr,
        );
        Self::from_canonical_aij(shape, rows, cols, data)
    }

    /// Return the row offsets (`shape[0] + 1` entries).
    pub fn indptr(&self) -> &DynArray<usize, 1> {
        &self.indptr
    }

    /// Return the canonical column indices, sorted within each row.
    pub fn indices(&self) -> &DynArray<usize, 1> {
        &self.indices
    }

    /// Return values in one-to-one correspondence with [`Self::indices`].
    pub fn data(&self) -> &DynArray<Item, 1> {
        &self.data
    }
}

impl<Item> FromAij for CsrMatrix<Item>
where
    Item: AddAssign + PartialEq + Copy + Default,
{
    /// Create a new CSR matrix from arrays `row`, `cols` and `data` that store for
    /// each nonzero entry the associated row, column, and value.
    fn from_aij(shape: [usize; 2], rows: &[usize], cols: &[usize], data: &[Item]) -> Self {
        let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);

        if let Some(&max_col) = cols.iter().max() {
            assert!(
                max_col < shape[1],
                "Maximum column {max_col} must be smaller than `shape.1` {}",
                shape[1]
            );
        }
        if let Some(&max_row) = rows.last() {
            assert!(
                max_row < shape[0],
                "Maximum row {max_row} must be smaller than `shape.0` {}",
                shape[0]
            );
        }

        Self::from_canonical_aij(shape, rows, cols, data)
    }
}

impl<Item> Shape<2> for CsrMatrix<Item> {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl<Item> BaseItem for CsrMatrix<Item>
where
    Item: Copy + Default,
{
    type Item = Item;
}

impl<Item> Nonzeros for CsrMatrix<Item> {
    fn nnz(&self) -> usize {
        self.data.len()
    }
}

impl<Item> SparseMatrixType for CsrMatrix<Item> {
    fn mat_type(&self) -> SparseMatType {
        self.mat_type
    }
}

impl<Item> AijIteratorByValue for CsrMatrix<Item>
where
    Item: Copy + Default,
{
    fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_ {
        self.indptr
            .iter_value()
            .tuple_windows::<(usize, usize)>()
            .enumerate()
            .flat_map(|(row, (start, end))| {
                izip!(
                    self.indices.data().unwrap()[start..end].iter(),
                    self.data.data().unwrap()[start..end].iter()
                )
                .map(|(col, value)| ([row, *col], *value))
                .collect::<Vec<_>>()
            })
    }
}

impl<Item> AijIteratorMut for CsrMatrix<Item>
where
    Item: Copy + Default,
{
    fn iter_aij_mut(&mut self) -> impl Iterator<Item = ([usize; 2], &mut Self::Item)> + '_ {
        self.indptr
            .iter_value()
            .tuple_windows::<(usize, usize)>()
            .enumerate()
            .flat_map(|(row, (start, end))| {
                izip!(
                    self.indices.data().unwrap()[start..end].iter(),
                    self.data.data_mut().unwrap()[start..end]
                        .iter_mut()
                        // Need to convert the mutable reference to the raw pointer
                        // as borrow checker does not allow the mutable reference to leak from FnMut.
                        .map(|v| v as *mut Item)
                )
                .map(|(col, value)| ([row, *col], value))
                .collect::<Vec<_>>()
            })
            .map(|(idx, value)| (idx, unsafe { &mut *value }))
    }
}

impl<Item: Copy + Default> CsrMatrix<Item> {
    /// Return as sparse matrix in iterator form.
    pub fn op(&self) -> SparseMatOpIterator<Item, impl Iterator<Item = ([usize; 2], Item)> + '_> {
        SparseMatOpIterator::new(self.iter_aij_value(), self.shape())
    }
}

impl<Item: Copy + Default> CsrMatrix<Item> {
    /// Convert to a dense matrix.
    pub fn todense(&self) -> DynArray<Item, 2> {
        DynArray::from_iter_aij(self.shape(), self.iter_aij_value())
    }
}

impl<Item: Default + Mul<Output = Item> + AddAssign<Item> + Add<Output = Item> + Copy + One>
    CsrMatrix<Item>
{
    /// Apply the matrix to a vector or dense matrix and allocate the result.
    ///
    /// A vector input has length `self.shape()[1]`. A matrix input has shape
    /// `[self.shape()[1], nrhs]`; the result has shape `[self.shape()[0], nrhs]`.
    ///
    /// # Panics
    ///
    /// Panics for inputs whose rank is not one or two, or whose leading dimension
    /// does not match the matrix column count.
    pub fn dot<ArrayImpl, const NDIM: usize>(
        &self,
        other: &Array<ArrayImpl, NDIM>,
    ) -> DynArray<Item, NDIM>
    where
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    {
        let mut out = empty_array::<Item, NDIM>();

        if NDIM == 1 {
            let mut out = out.r_mut().coerce_dim::<1>().unwrap();
            let other = other.r().coerce_dim::<1>().unwrap();
            out.resize_in_place([self.shape()[0]]);
            self.apply(One::one(), &other, Default::default(), &mut out);
        } else if NDIM == 2 {
            let mut out = out.r_mut().coerce_dim::<2>().unwrap();
            let other = other.r().coerce_dim::<2>().unwrap();
            out.resize_in_place([self.shape()[0], other.shape()[1]]);
            self.apply(One::one(), &other, Default::default(), &mut out);
        } else {
            panic!(
                "Unsupported number of dimensions NDIM = {NDIM}. Only NDIM=1 or NDIM=2 supported."
            );
        }

        out
    }
}

impl<Item, ArrayImplX, ArrayImplY> AsMatrixApply<Array<ArrayImplX, 1>, Array<ArrayImplY, 1>>
    for CsrMatrix<Item>
where
    Item: Default + Mul<Output = Item> + AddAssign<Item> + Add<Output = Item> + Copy + One,
    ArrayImplX: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    ArrayImplY: UnsafeDisjointRandom1DAccessMut<Item = Item> + Shape<1>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &crate::Array<ArrayImplX, 1>,
        beta: Self::Item,
        y: &mut crate::Array<ArrayImplY, 1>,
    ) {
        assert_eq!(y.len(), self.shape()[0]);
        assert_eq!(x.len(), self.shape()[1]);
        for (row, out) in y.iter_mut().enumerate() {
            *out = beta * *out
                + alpha * {
                    let c1 = unsafe { self.indptr.get_value_unchecked([row]) };
                    let c2 = unsafe { self.indptr.get_value_unchecked([1 + row]) };
                    let mut acc = Item::default();

                    for index in c1..c2 {
                        let col = unsafe { self.indices.get_value_unchecked([index]) };
                        acc += unsafe {
                            self.data.get_value_unchecked([index]) * x.get_value_unchecked([col])
                        };
                    }
                    acc
                }
        }
    }
}

impl<Item, ArrayImplX, ArrayImplY> AsMatrixApply<Array<ArrayImplX, 2>, Array<ArrayImplY, 2>>
    for CsrMatrix<Item>
where
    Item: Copy,
    Self: BaseItem<Item = Item>,
    ArrayImplX: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
    ArrayImplY: UnsafeRandomAccessMut<2, Item = Item> + Shape<2>,
    for<'b> Self: AsMatrixApply<
            Array<ArraySlice<ArrayRef<'b, ArrayImplX, 2>, 2, 1>, 1>,
            Array<ArraySlice<ArrayRefMut<'b, ArrayImplY, 2>, 2, 1>, 1>,
        >,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &crate::Array<ArrayImplX, 2>,
        beta: Self::Item,
        y: &mut crate::Array<ArrayImplY, 2>,
    ) {
        assert_eq!(x.shape()[0], self.shape()[1]);
        assert_eq!(y.shape()[0], self.shape()[0]);
        assert_eq!(x.shape()[1], y.shape()[1]);
        for (colx, mut coly) in izip!(x.col_iter(), y.col_iter_mut()) {
            self.apply(alpha, &colx, beta, &mut coly)
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_csr() {
        // We create a simple CSR matrix.
        let rows: Vec<usize> = vec![1, 4, 4];
        let cols: Vec<usize> = vec![2, 5, 6];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];

        let shape = [8, 13];
        let sparse_mat = CsrMatrix::from_aij(shape, &rows, &cols, &data);

        let mut x = DynArray::<f64, 1>::from_shape([shape[1]]);
        x.fill_from_seed_equally_distributed(0);

        let y = crate::dot!(sparse_mat, x);
        let expected = crate::dot!(sparse_mat.todense(), x);

        crate::assert_array_relative_eq!(y, expected, 1E-10);
    }

    #[test]
    fn raw_csr_is_canonicalized() {
        let matrix = CsrMatrix::new(
            [2, 3],
            DynArray::from_shape_and_vec([5], vec![2, 0, 2, 1, 1]),
            DynArray::from_shape_and_vec([3], vec![0, 3, 5]),
            DynArray::from_shape_and_vec([5], vec![1_i32, 4, -1, 2, 3]),
        );

        assert_eq!(matrix.indptr().data().unwrap(), &[0, 1, 2]);
        assert_eq!(matrix.indices().data().unwrap(), &[0, 1]);
        assert_eq!(matrix.data().data().unwrap(), &[4, 5]);
        assert_eq!(
            matrix.iter_aij_value().collect_vec(),
            vec![([0, 0], 4), ([1, 1], 5)]
        );
    }

    #[test]
    fn empty_aij_shapes_are_supported() {
        for shape in [[0, 4], [3, 0], [0, 0]] {
            let matrix = CsrMatrix::<f64>::from_aij(shape, &[], &[], &[]);
            assert_eq!(matrix.shape(), shape);
            assert_eq!(matrix.nnz(), 0);
            assert_eq!(matrix.indptr().len(), shape[0] + 1);
        }
    }

    #[test]
    fn raw_and_aij_duplicate_application_agree() {
        let raw = CsrMatrix::new(
            [2, 3],
            DynArray::from_shape_and_vec([4], vec![2, 0, 2, 1]),
            DynArray::from_shape_and_vec([3], vec![0, 3, 4]),
            DynArray::from_shape_and_vec([4], vec![1.0, 2.0, 3.0, 4.0]),
        );
        let aij = CsrMatrix::from_aij([2, 3], &[0, 0, 0, 1], &[2, 0, 2, 1], &[1.0, 2.0, 3.0, 4.0]);
        let x = DynArray::from_shape_and_vec([3], vec![2.0, 3.0, 5.0]);
        crate::assert_array_relative_eq!(raw.dot(&x), aij.dot(&x), 1E-14);
    }

    #[test]
    fn matrix_rhs_apply_validates_before_mutating() {
        let matrix = CsrMatrix::from_aij([2, 3], &[0, 0, 1], &[0, 2, 1], &[2.0, 1.0, 3.0]);
        let x = DynArray::from_shape_and_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut y = DynArray::from_shape_and_vec([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
        matrix.apply(2.0, &x, 0.5, &mut y);
        assert_eq!(y.data().unwrap(), &[15.0, 22.0, 43.0, 50.0]);

        let bad_x = DynArray::from_shape_and_vec([3, 2], vec![1.0; 6]);
        let mut bad_y = DynArray::from_shape_and_vec([2, 1], vec![7.0, 8.0]);
        let before = bad_y.data().unwrap().to_vec();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.apply(1.0, &bad_x, 1.0, &mut bad_y);
            }))
            .is_err()
        );
        assert_eq!(bad_y.data().unwrap(), before.as_slice());

        let short_x = DynArray::from_shape_and_vec([3, 1], vec![1.0; 3]);
        let mut wide_y = DynArray::from_shape_and_vec([2, 2], vec![9.0; 4]);
        let before = wide_y.data().unwrap().to_vec();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.apply(1.0, &short_x, 1.0, &mut wide_y);
            }))
            .is_err()
        );
        assert_eq!(wide_y.data().unwrap(), before.as_slice());

        let mut wrong_row_y = DynArray::from_shape_and_vec([1, 2], vec![6.0; 2]);
        let before = wrong_row_y.data().unwrap().to_vec();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.apply(1.0, &bad_x, 1.0, &mut wrong_row_y);
            }))
            .is_err()
        );
        assert_eq!(wrong_row_y.data().unwrap(), before.as_slice());

        let bad_rows = DynArray::<f64, 2>::from_shape([2, 0]);
        let mut empty_y = DynArray::<f64, 2>::from_shape([2, 0]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.apply(1.0, &bad_rows, 0.0, &mut empty_y);
            }))
            .is_err()
        );
    }
}
