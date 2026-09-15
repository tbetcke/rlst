# Changelog

## v0.7.0

Changes since [v0.6.1](https://codeberg.org/rlst/rlst/releases/tag/v0.6.1),
ordered from major features and compatibility changes to smaller fixes.
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.7.0.

### Expanded dense linear algebra

- Added generalized eigenvalue decompositions for real and complex matrices,
  including symmetric/Hermitian-definite problems and selection of left, right,
  both, or no eigenvectors for general problems.
- Added generalized Schur (QZ) decompositions, with Schur vectors and homogeneous
  eigenvalue pairs that retain information about infinite eigenvalues.
- Added pivoted symmetric-indefinite LDL factorization and linear-system solves,
  including complex symmetric matrices (using transpose, not Hermitian, symmetry).
- Added LQ factorization, application of its orthogonal/unitary factor, and
  minimum-norm solves for full-row-rank underdetermined systems.
- Added triangular/quasi-triangular Sylvester equation solves, exposing LAPACK's
  scaling and coefficient-perturbation status.
- These new LAPACK-backed facilities support `f32`, `f64`, `c32`, and `c64`.

### SuiteSparse sparse direct solvers

- Added the `rlst-suitesparse` companion crate with UMFPACK factorization of
  `CsrMatrix<f64>` and `CsrMatrix<c64>`, including CSR-to-CSC conversion, symbolic
  analysis, reusable numeric factors, and real/complex solves.
- Right-hand sides and solution buffers use RLST arrays; factorization and solve
  APIs report errors and singular-factorization status explicitly.
- Added native SuiteSparse discovery for Homebrew and Linux installations and a
  `SUITESPARSE_PATH` override. UMFPACK is GPL-2.0-or-later; see the companion
  crate's README for licensing implications.

### Frames of elements

- Added `FrameRefAccess`, a frame capability that borrows stored columns instead
  of copying them: `element_ref`, `element_mut`, `iter`, and `iter_mut`, on both
  the backend and the `ElementFrame` wrapper. `GenericFrameImpl` implements it.
  `ArrayFrameImpl` stores its columns packed in one matrix and has no stored
  element to borrow, so it does not implement the trait and continues to offer
  `column`, `column_mut`, `matrix`, and `matrix_mut`. The mutating accessors
  expose the stored element directly, so callers must preserve the space's
  invariants, as for `Element::new` and `Element::imp_mut`.
- Added `LinearSpace::axpy_inplace`, computing `y += alpha * x`. It has a default
  implementation in terms of `scalar_mul` and `sum_inplace`, so existing spaces
  need no changes; implementations backed by arrays may override it with a fused,
  allocation-free update, as `ArrayVectorSpace` and
  `DistributedArrayVectorSpace` now do.
- Generic frame application no longer copies columns or forms scaled
  temporaries. `GenericFrameImpl::apply` borrows its columns and accumulates
  through `axpy_inplace`, and the `FrameApply::apply_into` default scales the
  output in place rather than allocating a second temporary. Results, error
  conditions, and the zero-`beta` contract are unchanged; array-backed frames
  already used BLAS and are unaffected.

### Dense multiplication and scalar arrays — migration required

- **Use `dot_blas!` in place of the old dense `dot!` for BLAS-backed matrix
  products**, including chains of matrices. It remains the recommended choice
  for general dense matrix multiplication.
- `dot!` now performs direct dense tensor contractions of the left operand's
  last axis with the right operand's first axis. It supports arbitrary positive
  ranks, strided views, lazy expressions, and scalar types without BLAS support.
  Dense output rank must be inferable from context, for example a
  `DynArray<_, R>` annotation. Complex products do not conjugate inputs.
- Added `dot_into!` to overwrite existing output storage without allocation.
  Contraction metadata is validated before writes, and empty contracted axes
  produce zeros.
- Added `ZeroDimArray` for scalar results with shape `[]`, including `scalar`,
  `scalar_mut`, and `into_scalar` accessors. Vector-vector direct contractions
  return this type; a length-one vector is not a scalar output buffer.
- `rlst_dynamic_array!(T, [])` creates scalar storage when rank zero is known at
  the call site. Ordinary and strided array storage require positive rank;
  aligned scalar construction and unresolved generic rank-zero construction
  are rejected.
- Sparse matrices and abstract operators retain their `dot!` dispatch and
  right-to-left product chains. Distributed CSR matrices now support collective
  `dot`/`dot!` products with distributed vectors and multiple right-hand sides.

### Safety and API contracts — migration required

- **MPI is now opt-in:** enable the `mpi` feature to use distributed APIs.
  The default feature set is empty.
- Mutable strided layouts now reject overlapping elements and insufficient
  backing storage. Axis permutations and multislices validate axes and indices
  before constructing views, including empty-array cases.
- Added explicit disjoint-access contracts (`UnsafeDisjointRandom1DAccessMut`
  and `DisjointRefDataContainerMut`) for mutable iteration and operations that
  keep multiple element references alive. Custom array/container implementations
  must satisfy the stronger bounds where required.
- FFTW plans now require stable storage (`StableRawAccessMut` and the related
  container contract). `Fft::borrow_mut` returns `FftMutGuard`, permitting element
  updates through `data_mut` while preventing replacement of the planned array.
- Raw CSR construction now validates and canonicalizes its input: columns are
  sorted, duplicates summed, and explicit zeros removed. `CsrMatrix::new` has
  corresponding scalar bounds, and empty matrix shapes are supported.
- Sparse binary operations require equal shapes. Ordinary sparse numerical
  traits for non-zero-preserving functions (such as `exp`, `ln`, `cos`, and
  `recip`) and sparse matrix/matrix division have been removed. Use `map_stored`
  explicitly when transforming stored entries only is intended; it skips
  implicit zeros and removes zero-valued results.
- Custom `MmIdentifier` implementations must now provide `MM_NFIELDS` and
  `from_mm_fields` in addition to `MMTYPE`.
- Random-number APIs now use `rand` 0.10. Downstream RNGs must use compatible
  versions; import `rand::RngExt` for `random`/`random_range` and use `rand::rng()`
  in place of `thread_rng()`.

### Correctness and robustness fixes

- Hardened distributed layouts, sorting, mapping, ghost communication, gathers,
  scatters, and sparse operations for empty local ranks and globally empty
  inputs. Added collective metadata checks and preserved rank participation
  when rejecting incompatible inputs.
- Matrix Market readers now validate headers, field types, record counts,
  dimensions, and coordinate bounds; malformed, truncated, and excess data
  produce errors. Fixed real/complex parsing and rectangular column-major array
  input handling.
- Conjugate gradients now returns immediately for an exact initial solution,
  uses an absolute residual for a zero right-hand side, and avoids invalid
  updates on zero or non-finite search-direction denominators.
- Fixed pseudo-inverse handling for zero retained rank and all-zero matrices,
  with regression coverage for rank deficiency and truncation thresholds.
- Sparse application with multiple right-hand sides validates all dimensions
  before modifying the output. Stored-entry filtering no longer recurses over
  long runs of removed entries.
- Repaired `rlst_static_type!` expansion and added external-crate coverage.
- Added LAPACK storage/shape validation and numerical edge-case coverage,
  including generalized eigenvector ordering, padded views, empty inputs,
  Schur structure, and Sylvester scaling.

### Dependencies, documentation, and build improvements

- Updated workspace dependencies to their latest stable release series,
  including `rand`/`rand_chacha` 0.10, `rand_distr` 0.6, `itertools` 0.15,
  `syn` 3, `darling` 0.24, `bindgen` 0.73, `git2` 0.21, and `cc` 1.4.
  Updated `bytemuck`, `mpi`, and `mpi-sys` requirements and refreshed compatible
  dependency resolution. Burn integration targets 0.21.
- Fixed system FFTW discovery through `pkg-config` and native macOS Homebrew
  fallbacks, requiring both single- and double-precision libraries and avoiding
  host Homebrew discovery during cross-compilation.
- Refreshed user guides and API documentation for dense/sparse operations,
  decompositions, distributed computation, interpolation, and optional
  integrations. Added this changelog to the crate documentation.
- Updated random-number examples and fixed zero-dimensional shape assertions
  when optional Burn dependencies introduce additional equality implementations.
- Expanded CI with SuiteSparse tests and bounded MPI regressions on 1, 2, and
  4 ranks, including timeout/process-group cleanup tests; refreshed CI runners
  and lint/format compatibility.

## v0.6.1

- Update burn dependency to 0.21

## v0.6.0

- Added an interface to convert Burn tensors into rlst arrays and back

## v0.5.0

- Added the `multislice` and `multislice_iter` method to slice over multiple axes
  and iterate over multiple axes at the same time.
- Added an `interpolation` module for barycentric interpolation in 1, 2, and
  3 dimensions.
- Added a `chebychev` module for the evaluation, differentiation and conversion
  to interpolation data of Chebychev series coefficients in 1, 2, and 3 dimensions.
- Improved the `eval` routines. They now work in most situations where generic
  arrays are provided.
- Small bug fixes, some CI improvements, etc.

