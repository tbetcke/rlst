//! SuiteSparse bindings for RLST sparse matrices.
//!
//! This crate currently provides a safe UMFPACK interface for solving real and
//! complex double-precision systems stored as [`rlst::sparse::csr_mat::CsrMatrix`].
//! SuiteSparse must be installed natively. Set `SUITESPARSE_PATH` to its
//! installation prefix when it is not discoverable in a standard location.
//!
//! ```
//! use rlst::{FromAij, rlst_dynamic_array};
//! use rlst::sparse::csr_mat::CsrMatrix;
//! use rlst_suitesparse::UmfpackFactorize;
//!
//! let matrix = CsrMatrix::from_aij(
//!     [2, 2],
//!     &[0, 0, 1, 1],
//!     &[0, 1, 0, 1],
//!     &[4.0, 1.0, 2.0, 3.0],
//! );
//! let numeric = matrix.factorize()?;
//! let mut rhs = rlst_dynamic_array!(f64, [2]);
//! rhs[[0]] = 1.0;
//! rhs[[1]] = 1.0;
//! let solution = numeric.solve(&rhs)?;
//! assert!((solution[[0]] - 0.2).abs() < 1e-12);
//! assert!((solution[[1]] - 0.2).abs() < 1e-12);
//! # Ok::<(), rlst_suitesparse::UmfpackError>(())
//! ```

#![warn(missing_docs)]

/// Safe UMFPACK factorization and solve support.
pub mod umfpack;

pub use umfpack::{
    UmfpackComplexNumeric, UmfpackError, UmfpackFactorizationStatus, UmfpackFactorize,
    UmfpackNumeric, UmfpackResult,
};
