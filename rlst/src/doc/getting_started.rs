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
