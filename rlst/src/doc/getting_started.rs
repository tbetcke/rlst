//! Getting started with RLST.
//!
//! ## Creating and evaluating an array
//!
//! The dynamic-array macro allocates a column-major array whose number of axes is
//! inferred from the shape expression:
//!
//! ```
//! use rlst::EvaluateArray;
//!
//! let mut a = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! a.fill_from_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter());
//!
//! assert_eq!(a.shape(), [2, 3]);
//! assert_eq!(a.stride(), [1, 2]);
//! assert_eq!(a[[1, 2]], 6.0);
//!
//! // Arithmetic creates a lazy expression. `r()` lets the expression borrow `a`
//! // rather than consume it; `eval()` allocates and evaluates its result.
//! let doubled = (2.0_f64 * a.r()).eval();
//! assert_eq!(doubled[[1, 2]], 12.0);
//! assert_eq!(a[[1, 2]], 6.0);
//! ```
//!
//! See the [dense-array tutorial](crate::doc::dense_linear_algebra) for storage layouts,
//! slice-backed and static arrays, views, iteration, and matrix multiplication.
//!
//! ## Zero-dimensional scalar results
//!
//! A [`ZeroDimArray`](crate::ZeroDimArray) has no axes and owns exactly one scalar.
//! It is useful as a reduction result. Shape `[]` is distinct from shape `[0]`,
//! which describes an empty one-dimensional array. Scalar containers provide
//! dedicated borrowing and consuming accessors instead of random indexing:
//!
//! ```
//! let shape: [usize; 0] = [];
//! let mut result = rlst::rlst_dynamic_array!(f64, shape);
//! assert_eq!(result.shape(), []);
//! assert_eq!(result.number_of_elements(), 1);
//! assert!(!result.is_empty());
//! assert_eq!(*result.scalar(), 0.0);
//! *result.scalar_mut() = 3.5;
//! assert_eq!(result.into_scalar(), 3.5);
//!
//! let result: rlst::ZeroDimArray<f64> = 7.0.into();
//! let scalar: f64 = result.into();
//! assert_eq!(scalar, 7.0);
//! ```
//!
//! `From` conversions in both directions are provided for `f32`, `f64`, `c32`,
//! `c64`, and all primitive signed and unsigned integer types. For generic
//! `Copy + Default` items, use `ZeroDimArray::from_scalar(value)` and
//! `into_scalar()`. Rust's orphan rules prevent a blanket reverse `From`
//! implementation for arbitrary item types.
//!
//! The macro also accepts the literal `[]`. Rank-zero dispatch requires the rank
//! to be known at the macro expansion site: within a const-generic function the
//! macro retains its `DynArray<T, NDIM>` result type and rejects rank zero at
//! runtime. Explicitly aligned rank-zero construction is also rejected.
//!
//! In the unaligned form, use a shape with an independently determined rank or
//! provide an explicit result type. An unconstrained shape such as
//! `Default::default()` otherwise selects rank zero; inference from subsequent
//! array uses is not supported. Annotate the shape (for example,
//! `let shape: [usize; 2] = Default::default()`) or use `DynArray::from_shape`
//! when later uses must determine a positive rank.
//!
//! ## Selecting a BLAS/LAPACK library
//!
//! RLST calls BLAS and LAPACK but does not select a provider for downstream binaries. A
//! binary crate must arrange for compatible native libraries to be linked. For example,
//! a macOS application can add the following dependencies to its `Cargo.toml`:
//!
//! The APIs described here target the in-development `0.7.0-dev` tree. Until a stable 0.7
//! release is published, select that source explicitly; the registry requirement `"0.7"` does
//! not select a prerelease:
//!
//! ```toml
//! [dependencies]
//! rlst = { git = "https://codeberg.org/linalg-rs/rlst", branch = "main" }
//! blas-src = { version = "0.14", features = ["accelerate"] }
//! lapack-src = { version = "0.13", features = ["accelerate"] }
//! ```
//!
//! For reproducible builds, replace `branch` with a tested `rev`. Users of the latest published
//! stable release can instead select its `0.6` registry series, but APIs added during 0.7
//! development will not be available there.
//!
//! On Linux, use the provider appropriate for the installation (commonly system OpenBLAS).
//! Provider crate versions and features change independently, so consult their current
//! documentation. Importing a provider crate (for example `use blas_src as _;`) may be required
//! by that provider to force native linkage; this is provider-specific rather than an RLST API
//! requirement.
//!
//! ## Using the FFT interface
//!
//! RLST optionally uses FFTW for one-, two-, and three-dimensional transforms. **FFTW is GPL
//! licensed**, so distributing a binary linked to FFTW can impose GPL obligations. Enable the
//! `fftw` feature together with exactly one linkage feature:
//!
//! - `fftw_system` links to a system installation of both single- and double-precision FFTW;
//! - `fftw_source` builds and statically links FFTW;
//! - `fftw_mkl` uses Intel MKL's FFTW-compatible interface, subject to MKL's license.
//!
//! With the feature enabled, see the `rlst::fftw` module for transform examples.
