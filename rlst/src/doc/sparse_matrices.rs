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
//! - `indptr`: The entries belonging to row `i` occupy the half-open range
//!   `indptr[i]..indptr[i + 1]` in `indices` and `data`. It has `1 + m`
//!   elements for an `m`-row matrix, starts at zero, is nondecreasing, and ends at `nnz`.
//! - `data`: The data entries of the sparse matrix.
//!
//! CSR constructors produce a canonical representation: column indices are sorted within each
//! row, duplicate coordinates are summed, and explicit zeros are removed. Both raw
//! [`CsrMatrix::new`](crate::sparse::csr_mat::CsrMatrix::new) construction and AIJ construction
//! enforce this construction-time invariant. [`CsrMatrix::new`](crate::sparse::csr_mat::CsrMatrix::new)
//! panics if the pointer length, pointer endpoints, pointer ordering, data/index lengths, or column
//! bounds are invalid. AIJ construction requires equally long slices and in-bounds row and column
//! indices. Mutable entry access can subsequently introduce a stored zero; `map_stored` applies its
//! function to every currently stored entry and filters zero results when the mapped matrix is assembled.
//!
//! ```
//! use rlst::{FromAij, Nonzeros};
//! use rlst::sparse::csr_mat::CsrMatrix;
//!
//! // Duplicates are summed, rows are sorted, and explicit zeros are discarded.
//! let matrix = CsrMatrix::from_aij(
//!     [2, 3],
//!     &[1, 0, 0, 0],
//!     &[2, 1, 1, 2],
//!     &[4.0, 2.0, 3.0, 0.0],
//! );
//! assert_eq!(matrix.nnz(), 2);
//! assert_eq!(matrix.indptr().data().unwrap(), &[0, 1, 2]);
//! assert_eq!(matrix.indices().data().unwrap(), &[1, 2]);
//! assert_eq!(matrix.data().data().unwrap(), &[5.0, 4.0]);
//! ```
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
//! # let expected = rlst::dot_blas!(sparse_mat.todense(), x);
//!
//! # rlst::assert_array_relative_eq!(y, expected, 1E-10);
//! ```     
//!
//! # Matrix-vector products
//!
//! The `CsrMatrix` type in RLST supports the [`AsMatrixApply`](crate::AsMatrixApply) trait for
//! multiplication with one- and two-dimensional dense arrays. The [`dot!`](crate::dot) macro
//! allocates the result of a sparse matrix-vector or matrix-matrix product. Its inner dimension
//! must equal the sparse matrix column count.
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
//! # let expected = rlst::dot_blas!(sparse_mat.todense(), x);
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
//! # Direct sparse solves with UMFPACK
//!
//! The companion `rlst-suitesparse` crate added during 0.7 development factorizes real `f64`
//! and complex `c64` CSR matrices with SuiteSparse UMFPACK. Call `factorize()` once and reuse the
//! returned numeric factorization for multiple contiguous vector right-hand sides. Matrices must
//! be nonempty and square; singularity is exposed through the factorization status.
//!
//! This integration requires a native SuiteSparse installation. UMFPACK is GPL-2.0-or-later, so
//! linking it can affect distribution licensing even though the Rust crates are dual-licensed.
//! See the companion crate's [real and complex solve tutorial](https://codeberg.org/linalg-rs/rlst/src/branch/main/rlst-suitesparse/README.md#real-and-complex-solves)
//! for setup, factorization reuse, status handling, and `SUITESPARSE_PATH` discovery.
//!
//! # Matrix Market files
//!
//! [`write_coordinate_mm`](crate::io::matrix_market::write_coordinate_mm) writes stored entries
//! using Matrix Market's one-based coordinate indices. [`read_coordinate_mm`](crate::io::matrix_market::read_coordinate_mm)
//! accepts general real or complex coordinate files and assembles duplicates with the same CSR
//! canonicalization described above. Dense [`write_array_mm`](crate::io::matrix_market::write_array_mm)
//! and [`read_array_mm`](crate::io::matrix_market::read_array_mm) use Matrix Market column-major
//! record order, regardless of the source array's internal layout. Readers reject unsupported
//! integer and symmetric variants, malformed records, count mismatches, and out-of-range indices
//! with an [`RlstError`](crate::RlstError).
//!
//! # Distributed multiplication
//!
//! With the `mpi` feature, `dot!(a, x)` also accepts a distributed CSR matrix and
//! a distributed vector or matrix. The input must share `a.domain_layout()`;
//! the result shares `a.range_layout()` and owns its local dense data. Matrix
//! inputs retain their number of RHS columns. Borrowed views and lazy distributed
//! expressions are supported. The equivalent method call is `a.dot(&x)`.
//!
//! Every rank in the matrix communicator must call the operation, even with no
//! local rows or a globally empty input. Input layouts, local row counts, and
//! agreement on RHS dimensions are checked collectively before ghost exchange.
//!
//! ```no_run
//! # #[cfg(feature = "mpi")]
//! # {
//! use std::rc::Rc;
//! use rlst::{FromAijDistributed, distributed_tools::IndexLayout};
//! use rlst::sparse::distributed_csr_mat::DistributedCsrMatrix;
//! let universe = mpi::initialize().unwrap();
//! let world = universe.world();
//! let layout = Rc::new(IndexLayout::from_local_counts(1, &world));
//! let row = layout.local_range().0;
//! let a = DistributedCsrMatrix::from_aij(
//!     layout.clone(), layout.clone(), &[row], &[row], &[2.0_f64],
//! );
//! let mut x = rlst::dist_vec!(f64, layout.clone());
//! x.local.fill_with_value(3.0);
//! let y = rlst::dot!(a, x.r());
//! assert_eq!(y.local[[0]], 6.0);
//! # }
//! ```
//!
//! # Matrix Market round trip
//!
//! A coordinate-file round trip is:
//!
//! ```no_run
//! use rlst::{FromAij, Nonzeros};
//! use rlst::io::matrix_market::{read_coordinate_mm, write_coordinate_mm};
//! use rlst::sparse::csr_mat::CsrMatrix;
//!
//! let matrix = CsrMatrix::from_aij([2, 2], &[0, 1], &[1, 0], &[2.0_f64, 3.0]);
//! let path = std::env::temp_dir().join("rlst-matrix-market-example.mtx");
//! write_coordinate_mm(&matrix, path.to_str().unwrap())?;
//! let restored = read_coordinate_mm::<f64>(path.to_str().unwrap())?;
//! assert_eq!(restored.nnz(), 2);
//! std::fs::remove_file(path)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
