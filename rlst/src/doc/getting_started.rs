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
//! ## Zero-dimensional arrays
//!
//! An array of rank zero has no axes and holds exactly one element; its shape and
//! stride are `[]`. [`ZeroDimArray<T>`](crate::ZeroDimArray) is another name for
//! `DynArray<T, 0>`, and is, for example, the result of a vector-vector product.
//! Shape `[]` is distinct from shape `[0]`, which describes an empty one-dimensional
//! array. Index the element with the empty multi-index `[]`, or use the scalar
//! accessors:
//!
//! ```
//! let mut result = rlst::rlst_dynamic_array!(f64, []);
//! assert_eq!(result.shape(), [0usize; 0]);
//! assert_eq!(result.number_of_elements(), 1);
//! assert!(!result.is_empty());
//! assert_eq!(result[[]], 0.0);
//! *result.scalar_mut() = 3.5;
//! assert_eq!(*result.scalar(), 3.5);
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
//! Rank zero is an ordinary rank: code generic over the rank, iteration, lazy
//! expressions, and the aligned, strided, slice-backed and static storages all
//! accept it, and the macro gives the same type in concrete and const-generic code.
//!
//! ```
//! use rlst::EvaluateArray;
//!
//! fn sum_of_doubled<const NDIM: usize>(shape: [usize; NDIM]) -> f64 {
//!     let mut array = rlst::rlst_dynamic_array!(f64, shape);
//!     array.fill_with_value(1.5);
//!     (2.0_f64 * array.r()).eval().iter_value().sum()
//! }
//!
//! assert_eq!(sum_of_doubled([2, 2]), 12.0);
//! assert_eq!(sum_of_doubled([]), 3.0);
//! ```
//!
//! The rank of the macro's result is that of the shape expression, so give
//! `Default::default()` and similar shapes a type when the rank cannot be inferred
//! otherwise, for example `let shape: [usize; 2] = Default::default()`.
//!
//! ## Selecting a BLAS/LAPACK library
//!
//! RLST calls BLAS and LAPACK but does not select a provider for downstream binaries. A
//! binary crate must arrange for compatible native libraries to be linked. For example,
//! a macOS application can add the following dependencies to its `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! rlst = "0.12"
//! blas-src = { version = "0.14", features = ["accelerate"] }
//! lapack-src = { version = "0.13", features = ["accelerate"] }
//! ```
//!
//! See the [changelog](crate::doc::changelog) for new features and migration notes from 0.9.
//!
//! On Linux, use the provider appropriate for the installation (commonly system OpenBLAS).
//! Provider crate versions and features change independently, so consult their current
//! documentation. Importing a provider crate (for example `use blas_src as _;`) may be required
//! by that provider to force native linkage; this is provider-specific rather than an RLST API
//! requirement.
//!
//! ## Controlling BLAS threads
//!
//! [`set_blas_threads`](crate::threading::set_blas_threads) sets the number of threads of the
//! linked BLAS library and returns the resulting state of each backend it found. OpenBLAS, Intel
//! MKL, BLIS, FlexiBLAS, and Apple Accelerate (macOS 15 or later) are supported.
//!
//! ```
//! # extern crate blas_src;
//! use rlst::threading::{get_blas_threads, set_blas_threads};
//!
//! match set_blas_threads(1) {
//!     Ok(settings) => {
//!         for setting in settings {
//!             println!("{:?}: {:?}", setting.backend, setting.threads);
//!         }
//!     }
//!     Err(err) => println!("BLAS threads unchanged: {err}"),
//! }
//! println!("{:?}", get_blas_threads());
//! ```
//!
//! Dynamically linked libraries and Accelerate are found at runtime on Linux and macOS. A
//! statically linked OpenBLAS, MKL, or BLIS requires the matching `openblas_threading`,
//! `mkl_threading`, or `blis_threading` feature. Accelerate only distinguishes single-threaded
//! from multi-threaded execution, and its setting applies to the calling thread only; see the
//! [`threading`](crate::threading) module for details.
//!
//! ## Using the FFT interface
//!
//! RLST optionally uses FFTW for one-, two-, and three-dimensional transforms. **FFTW is GPL
//! licensed**, so distributing a binary linked to FFTW can impose GPL obligations. Enable the
//! `fftw_system` feature, which implies `fftw` and links to a system installation of both
//! single- and double-precision FFTW.
//!
//! With the feature enabled, see the `rlst::fftw` module for transform examples.
