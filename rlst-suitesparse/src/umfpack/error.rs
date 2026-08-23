use std::error::Error;
use std::fmt::{Display, Formatter};

/// Errors returned by UMFPACK bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UmfpackError {
    /// The input matrix has a zero-sized dimension.
    EmptyMatrix,
    /// UMFPACK's direct solver requires a square matrix.
    NonSquareMatrix {
        /// Number of rows in the matrix.
        rows: usize,
        /// Number of columns in the matrix.
        columns: usize,
    },
    /// The CSR data does not satisfy the expected storage invariants.
    InvalidCsr(&'static str),
    /// A matrix dimension or sparse index cannot be represented by UMFPACK's
    /// 64-bit index type.
    IndexOutOfRange(usize),
    /// A right-hand side or solution has the wrong length.
    DimensionMismatch {
        /// Required vector length.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },
    /// UMFPACK returned a failure status.
    NativeFailure {
        /// Native operation that failed.
        operation: &'static str,
        /// UMFPACK status code.
        status: i32,
    },
    /// UMFPACK reported success but did not return the expected native handle.
    NullHandle(&'static str),
}

impl Display for UmfpackError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(formatter, "UMFPACK does not accept an empty matrix"),
            Self::NonSquareMatrix { rows, columns } => {
                write!(
                    formatter,
                    "UMFPACK requires a square matrix, got {rows}x{columns}"
                )
            }
            Self::InvalidCsr(message) => write!(formatter, "invalid CSR matrix: {message}"),
            Self::IndexOutOfRange(index) => {
                write!(
                    formatter,
                    "sparse index {index} cannot be represented by UMFPACK"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NativeFailure { operation, status } => {
                write!(formatter, "UMFPACK {operation} failed with status {status}")
            }
            Self::NullHandle(operation) => {
                write!(formatter, "UMFPACK {operation} returned a null handle")
            }
        }
    }
}

impl Error for UmfpackError {}

/// Result type returned by this crate.
pub type UmfpackResult<T> = Result<T, UmfpackError>;
