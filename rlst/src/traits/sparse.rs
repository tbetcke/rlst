//! Sparse matrix traits

#[cfg(feature = "mpi")]
use std::rc::Rc;

#[cfg(feature = "mpi")]
use mpi::traits::Communicator;

#[cfg(feature = "mpi")]
use crate::distributed_tools::IndexLayout;

use crate::sparse::SparseMatType;

use super::{BaseItem, Shape};

/// Return the number of non-zero entries in the sparse matrix.
pub trait Nonzeros {
    /// Return the number of non-zero entries.
    fn nnz(&self) -> usize;
}

/// Return the type of the sparse matrix.
pub trait SparseMatrixType {
    /// Return the type of the sparse matrix.
    fn mat_type(&self) -> SparseMatType;
}

/// Construct a sparse matrix from Aij style slices.
pub trait FromAij
where
    Self: BaseItem + Sized,
{
    /// Create a sparse matrix from parallel `rows`, `cols`, and `data` slices.
    ///
    /// Implementations require equal slice lengths and coordinates within `shape`.
    /// Duplicate-coordinate and explicit-zero handling is implementation-defined;
    /// [`crate::sparse::csr_mat::CsrMatrix`] sums duplicates and removes zeros.
    ///
    /// # Panics
    ///
    /// May panic when slice lengths differ or a coordinate is out of bounds.
    fn from_aij(
        shape: [usize; 2],
        rows: &[usize],
        cols: &[usize],
        data: &[<Self as BaseItem>::Item],
    ) -> Self;

    /// Create a sparse matrix from an iterator of `([row, column], value)` tuples.
    ///
    /// Coordinates use zero-based indexing and are subject to [`Self::from_aij`]'s contract.
    fn from_aij_iter<I>(shape: [usize; 2], iter: I) -> Self
    where
        I: Iterator<Item = ([usize; 2], <Self as BaseItem>::Item)>,
    {
        let (rows, cols, data): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut data = Vec::new();

            for (index, value) in iter {
                rows.push(index[0]);
                cols.push(index[1]);
                data.push(value);
            }

            (rows, cols, data)
        };

        Self::from_aij(shape, &rows, &cols, &data)
    }
}

#[cfg(feature = "mpi")]
#[cfg_attr(docsrs, doc(cfg(feature = "mpi")))]
/// Construct a sparse matrix from Aij style slices for  distributed matrices.
pub trait FromAijDistributed<'a>
where
    Self: BaseItem + Sized,
{
    /// The communicator type used for distributed operations.
    type C: Communicator;
    /// Collectively create a matrix from parallel global-row, global-column, and value slices.
    ///
    /// Every rank must participate with compatible domain/range layouts, including ranks
    /// with empty contributions. Contributions may target rows owned by other ranks;
    /// implementations route them to the range owner. Slice lengths must match, rows must
    /// lie in the range layout, and columns in the domain layout.
    fn from_aij(
        domain_layout: Rc<IndexLayout<'a, Self::C>>,
        range_layout: Rc<IndexLayout<'a, Self::C>>,
        rows: &[usize],
        cols: &[usize],
        data: &[<Self as BaseItem>::Item],
    ) -> Self;

    /// Create a sparse matrix from an iterator of ([i, j], value) tuples.
    fn from_aij_iter<I>(
        domain_layout: Rc<IndexLayout<'a, Self::C>>,
        range_layout: Rc<IndexLayout<'a, Self::C>>,
        iter: I,
    ) -> Self
    where
        I: Iterator<Item = ([usize; 2], <Self as BaseItem>::Item)>,
    {
        let (rows, cols, data): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut data = Vec::new();

            for (index, value) in iter {
                rows.push(index[0]);
                cols.push(index[1]);
                data.push(value);
            }

            (rows, cols, data)
        };

        Self::from_aij(domain_layout, range_layout, &rows, &cols, &data)
    }
}

/// Trait for matrices that have a shape and can return a AijIterator.
pub trait AijIterMat: BaseItem + Shape<2> {
    /// Get an iterator over the matrix in (i, j, value) form.
    fn iter_aij(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_;
}
