//! Safe UMFPACK factorization and solve support for RLST CSR matrices.

mod complex;
mod csc;
mod error;
mod real;
mod sys;

#[cfg(test)]
mod tests;

pub use complex::{UmfpackComplexNumeric, UmfpackComplexSymbolic};
pub use error::{UmfpackError, UmfpackResult};
pub use real::{UmfpackNumeric, UmfpackRealSymbolic};

const UMFPACK_A: i32 = 0;
const UMFPACK_WARNING_SINGULAR_MATRIX: i32 = 1;

/// Non-error status produced while constructing an UMFPACK factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UmfpackFactorizationStatus {
    /// UMFPACK completed factorization without warnings.
    Success,
    /// UMFPACK produced a factorization for a singular matrix.
    SingularMatrix,
    /// UMFPACK produced an implementation-defined warning status.
    Warning(i32),
}

/// Perform an UMFPACK symbolic analysis of an RLST CSR matrix.
///
/// Import this trait and call [`analyze`](UmfpackSymbolic::analyze) on a
/// [`rlst::sparse::csr_mat::CsrMatrix<f64>`] or
/// [`rlst::sparse::csr_mat::CsrMatrix<rlst::c64>`].
pub trait UmfpackSymbolic {
    /// Concrete symbolic factorization returned for the matrix scalar type.
    type Symbolic;

    /// Convert this matrix from CSR to UMFPACK's CSC representation and perform
    /// symbolic analysis.
    fn analyze(&self) -> UmfpackResult<Self::Symbolic>;
}

fn factorization_status(status: i32) -> UmfpackFactorizationStatus {
    match status {
        0 => UmfpackFactorizationStatus::Success,
        UMFPACK_WARNING_SINGULAR_MATRIX => UmfpackFactorizationStatus::SingularMatrix,
        status => UmfpackFactorizationStatus::Warning(status),
    }
}
