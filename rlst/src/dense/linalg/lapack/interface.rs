//! Checked, scalar-generic wrappers around selected LAPACK routines.
//!
//! These low-level traits operate on flat **column-major** storage. Dimensions
//! and leading dimensions follow LAPACK conventions; wrappers assert their
//! slice-size invariants and query workspaces internally where supported.
//! Mutable matrix arguments are generally overwritten with factors or
//! solutions. A nonzero LAPACK `info` is returned as
//! [`crate::base_types::LapackError`]. Prefer the high-level decomposition
//! traits in [`crate::traits::linalg::decompositions`] unless packed factors or
//! LAPACK-specific controls are required.

use crate::base_types::LapackError;
use crate::base_types::LapackResult;

pub mod ev;
pub mod gees;
pub mod geev;
pub mod gelqf;
pub mod gels;
pub mod geqp3;
pub mod geqrf;
pub mod gesdd;
pub mod gesvd;
pub mod getrf;
pub mod getri;
pub mod getrs;
pub mod gges;
pub mod ggev;
pub mod mlq;
pub mod mqr;
pub mod orglq;
pub mod orgqr;
pub mod posv;
pub mod potrf;
pub mod sygv;
pub mod sytrf;
pub mod sytrs;
pub mod trsm;
pub mod trsyl;

pub(crate) fn lapack_int(value: usize, name: &str) -> i32 {
    i32::try_from(value).unwrap_or_else(|_| panic!("`{name}` {value} exceeds LAPACK integer range"))
}

pub(crate) fn matrix_len(leading_dimension: usize, columns: usize, name: &str) -> usize {
    leading_dimension.checked_mul(columns).unwrap_or_else(|| {
        panic!("`{name}` storage size overflows usize: {leading_dimension} * {columns}")
    })
}

/// Helper function to convert LAPACK info codes to a `LapackResult`.
pub(crate) fn lapack_return<T>(info: i32, result: T) -> LapackResult<T> {
    if info == 0 {
        Ok(result)
    } else {
        Err(LapackError::LapackInfoCode(info))
    }
}

#[cfg(test)]
mod tests {
    use super::{lapack_int, matrix_len};

    #[test]
    #[should_panic(expected = "exceeds LAPACK integer range")]
    fn lapack_int_rejects_overflow() {
        lapack_int(i32::MAX as usize + 1, "dimension");
    }

    #[test]
    #[should_panic(expected = "storage size overflows usize")]
    fn matrix_len_rejects_overflow() {
        matrix_len(usize::MAX, 2, "matrix");
    }
}
