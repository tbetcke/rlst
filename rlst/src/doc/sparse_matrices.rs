//! Sparse matrix operations
//!
//! RLST contains a complete sparse matrix algebra. Sparse matrices
//! can defined either on a single process or via MPI distributed across
//! multiple processes. Typical matrix operations such as addition, componentwise
//! operations and matrix vector products are supported for sparse matrices.
//!
//! # The default sparse matrix format
//!
//! RLST uses the CSR format for sparse matrices with `nnz` nonzero entries. A sparse matrix in this format is defined
//! by three arrays.
//!
//! - `indices`: This array contains the column indices of all nonzero elements. It is of length `nnz`.
//! - `indptr`: The column indices of row `i` are contained in `indptr[i]..indptr[i + 1]`. The `indptr` array
//!   has `1 + m` elements, where `m` is the number of rows in the sparse matrix. The last element of `indptr` is
//!   always identical to `nnz`.
//! - `data`: The data entries of the sparse matrix.
//!
//! CSR constructors produce a canonical representation: column indices are sorted within each
//! row, duplicate coordinates are summed, and explicit zeros are removed. Both raw
//! [`CsrMatrix::new`](crate::sparse::csr_mat::CsrMatrix::new) construction and AIJ construction
//! enforce this construction-time invariant. Mutable entry access can subsequently introduce a
//! stored zero; `map_stored` applies its function to every currently stored entry and filters zero
//! results when the mapped matrix is assembled.
//!
//! On a single node the [CsrMatrix](crate::sparse::csr_mat::CsrMatrix) defines this structure. To initialize a new sparse
//! matrix the easiest is to use the [CsrMatrix::from_aij](crate::traits::FromAij::from_aij) method. It creates a sparse
//! matrix from a `rows`, `cols`, and `data` array. Repeated entries for the same row/col index are summed up in the sparse matrix assembly.
//! Zero entries are filtered out. The following example creates a sparse matrix `sparse_mat` and multiplies it with a vector `x`.
//!
//! ```
//! # use rlst;
//! # use rlst::sparse::csr_mat::CsrMatrix;
//! # extern crate blas_src;
//! use rlst::FromAij;
//! let rows: Vec<usize> = vec![1, 4, 4];
//! let cols: Vec<usize> = vec![2, 5, 6];
//! let data: Vec<f64> = vec![1.0, 2.0, 3.0];
//!
//! let shape = [8, 13];
//! let sparse_mat = CsrMatrix::from_aij(shape, &rows, &cols, &data);
//!
//! let mut x = rlst::DynArray::<f64, 1>::from_shape([shape[1]]);
//! x.fill_from_seed_equally_distributed(0);
//!
//! let y = rlst::dot!(sparse_mat, x);
//! # let expected = rlst::dot!(sparse_mat.todense(), x);
//!
//! # rlst::assert_array_relative_eq!(y, expected, 1E-10);
//! ```     
//!
//! # Matrix-vector products
//!
//! The `CsrMatrix` type in RLST supports the [AsMatrixApply](crate::AsMatrixApply) trait for multiplication of sparse matrices with
//! one and two dimensional dense arrays. Alternatively, one can use the [dot](crate::dot) macro, which instantiates
//! a new array to hold the result of a sparse matrix-vector product. Alternatively, one can use the `[dot](crate::dot)` macro to evaluate
//! a matrix-vector product. An example for the use of `dot` is given below.
//!
//! ```
//! # use rlst;
//! # use rlst::sparse::csr_mat::CsrMatrix;
//! # use rlst::FromAij;
//! # extern crate blas_src;
//! # let rows: Vec<usize> = vec![1, 4, 4];
//! # let cols: Vec<usize> = vec![2, 5, 6];
//! # let data: Vec<f64> = vec![1.0, 2.0, 3.0];
//! # let shape = [8, 13];
//! # let sparse_mat = CsrMatrix::from_aij(shape, &rows, &cols, &data);
//! # let mut x = rlst::DynArray::<f64, 1>::from_shape([shape[1]]);
//! # x.fill_from_seed_equally_distributed(0);
//! let y = rlst::dot!(sparse_mat, x);
//! # let expected = rlst::dot!(sparse_mat.todense(), x);
//! # rlst::assert_array_relative_eq!(y, expected, 1E-10);
//! ```     
//!
//! # Componentwise operations on sparse matrices
//!
//! Sparse matrices support addition, subtraction, scalar multiplication,
//! componentwise matrix multiplication, and zero-preserving unary operations
//! such as [Abs](crate::Abs), [Square](crate::Square), [Sqrt](crate::Sqrt), and
//! [Sin](crate::Sin). Binary matrix operations require equal shapes.
//!
//! Functions that do not preserve implicit zero (for example `exp`, `cos`, and
//! `recip`) deliberately do not implement the ordinary numerical traits for
//! sparse matrices. Use
//! [`SparseMatOpIterator::map_stored`](crate::sparse::mat_operations::SparseMatOpIterator::map_stored)
//! when applying such a function only to explicitly stored entries is intended.
//! Sparse matrix/matrix division is also unsupported because division involving
//! implicit zeros generally cannot be represented by a sparse result.
//!
//! To access componentwise operations each sparse matrix supports the [op](crate::sparse::csr_mat::CsrMatrix::op). This returns a
//! [SparseMatOpIterator](crate::sparse::mat_operations::SparseMatOpIterator). Componentwise sparse matrix operations are defined
//! through this iterator. Operations such as adding two sparse matrices or componentwise multiplication of two sparse matrices
//! are executed by merging the corresponding iterators. The following code gives an example.
//!
//!
//! ```
//! # use rlst;
//! # use rlst::sparse::csr_mat::CsrMatrix;
//! use rlst::FromAij;
//! use std::ops::Mul;
//! let rows1: Vec<usize> = vec![1, 4, 4];
//! let cols1: Vec<usize> = vec![2, 5, 6];
//! let data1: Vec<f64> = vec![1.0, 2.0, 3.0];
//!
//! let rows2: Vec<usize> = vec![2, ];
//! let cols2: Vec<usize> = vec![2, ];
//! let data2: Vec<f64> = vec![1.0];
//!
//!
//! let shape = [8, 13];
//! let a = CsrMatrix::from_aij(shape, &rows1, &cols1, &data1);
//! let b = CsrMatrix::from_aij(shape, &rows2, &cols2, &data2);
//!
//! let c = (5.0 * a.op().map_stored(f64::cosh) + b.op()).into_csr();
//!
//! ```
//! In the above example we apply `cosh` to each stored element in `a`, multiply the result by `5.0` and add `b`.
//! The result is then turned back into a sparse matrix. The implementation uses lazy evalution. The operations are only executed
//! when `into_csr` is called.
//!
//!
