# Changelog

## v0.8.0 (unreleased)

Changes since v0.7.0. All workspace crates (`rlst`, `rlst-proc-macro`, and
`rlst-suitesparse`) use development version `0.8.0-dev`.

### Frames of elements — migration required

The previously unimplemented `Frame` trait has been replaced by `FrameImpl<'a>`.
`ElementFrame<'a, Implementation>` is the public wrapper and forwards storage,
application, and QR operations to its implementation. `GenericFrameImpl` stores
owned elements; `ArrayFrameImpl` uses a column-major matrix. Both support growth,
linear combinations, and rank-adaptive pivoted QR. The former reference-based
`get`, `get_mut`, and iterator requirements have been replaced by owned
`element` extraction and `set_element`; array-backed wrappers provide zero-copy
matrix and column views. `push` borrows and copies an element and returns a result.

The initial development types `ElementFrame<'a, Space>` and `ArrayFrame<'a, Item>`
are now `GenericFrameImpl<'a, Space>` and `ArrayFrameImpl<'a, Item>`. Explicit
construction uses `ElementFrame::new(GenericFrameImpl::new(&space))` or
`ElementFrame::new(ArrayFrameImpl::new(&space))`. `imp`, `imp_mut`, and
`into_imp` provide implementation access.

Implementations opt into `FrameApply` and `FrameQr` for application and QR.
The wrapper provides inherent methods and implements the corresponding traits
by delegation; its QR result contains a wrapped Q with the same implementation
type. See the abstract linear algebra guide for construction and QR examples.

The `FrameSpace` trait has been removed. `frame` is now a method of
`InnerProductSpace` with a default implementation, so every inner-product space
over an `RlstScalar` field has it without writing any code; spaces that
previously implemented `FrameSpace` with `GenericFrameImpl` should delete that
impl. A space selects different storage by overriding the method, as
`ArrayVectorSpace` does, and generic code such as GMRES reaches that override
without Rust specialization. To make one signature serve every space, `frame`
returns `ElementFrame<'_, SpaceFrame<'_, Self>>`, where the new `SpaceFrame`
erases the backend behind the new object-safe `DynFrame` trait and forwards to
it, preserving BLAS application and LAPACK QR at the cost of one virtual call
per frame operation. Backend-specific inherent methods such as `matrix` and
`column` are not available through the erased frame: construct the backend
directly with `ElementFrame::new(ArrayFrameImpl::new(&space))`, which has no
scalar bounds, or use the new `ArrayVectorSpace::array_frame`. The
`InnerProductSpace` implementation for `ArrayVectorSpace<Item>` now requires
`Item: Lapack + Gemm`, which admits the same scalars as the previous
`RlstScalar` bound.

### GMRES for abstract operators

Added `GmresIteration` in `operator::algorithms::gmres`, a generalized minimal
residual solver for operator equations whose operator need not be self-adjoint.
It follows the same builder shape as `CgIteration`, and `run` reports the
residual by the same rules: the relative residual `||b-Ax||/||b||`, or the
absolute residual for a zero right-hand side. The residual is recomputed from
the operator at the end of every restart cycle, so the returned value is the
true residual rather than the estimate from the Givens recursion.

The Krylov basis is stored in an `ElementFrame` obtained from the space through
`InnerProductSpace::frame`, which makes this the first consumer of the frame
machinery. Any inner-product space works, including distributed ones, and
array-backed spaces keep the basis as a single dense matrix. `set_restart`
bounds the size
of that basis; by default GMRES does not restart. Two callbacks are available:
`set_callable` receives the iteration count and the current residual from the
Givens recursion at no extra cost, while `set_element_callable` reconstructs the
approximation and the residual element for every call at the cost of one extra
linear combination and one extra operator application per iteration.

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
