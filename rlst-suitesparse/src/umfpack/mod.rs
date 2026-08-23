//! Safe UMFPACK factorization and solve support for RLST CSR matrices.

use rlst::{Array, MutableRawAccessArrayImpl, RawAccessArrayImpl};

mod complex;
mod csc;
mod error;
mod real;
mod sys;

#[cfg(test)]
mod tests;

pub use complex::UmfpackComplexNumeric;
pub use error::{UmfpackError, UmfpackResult};
pub use real::UmfpackNumeric;

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

/// Factorize an RLST CSR matrix with UMFPACK.
///
/// Import this trait and call [`factorize`](UmfpackFactorize::factorize) on a
/// [`rlst::sparse::csr_mat::CsrMatrix<f64>`] or
/// [`rlst::sparse::csr_mat::CsrMatrix<rlst::c64>`].
pub trait UmfpackFactorize {
    /// Concrete numeric factorization returned for the matrix scalar type.
    type Numeric;

    /// Convert this matrix from CSR to UMFPACK's CSC representation, perform
    /// symbolic analysis, and create a numeric factorization.
    fn factorize(&self) -> UmfpackResult<Self::Numeric>;
}

fn factorization_status(status: i32) -> UmfpackFactorizationStatus {
    match status {
        0 => UmfpackFactorizationStatus::Success,
        UMFPACK_WARNING_SINGULAR_MATRIX => UmfpackFactorizationStatus::SingularMatrix,
        status => UmfpackFactorizationStatus::Warning(status),
    }
}

fn array_data<'a, Item, ArrayImpl>(
    array: &'a Array<ArrayImpl, 1>,
    name: &'static str,
) -> UmfpackResult<&'a [Item]>
where
    ArrayImpl: RawAccessArrayImpl<Item, 1>,
{
    if !array.is_contiguous() {
        return Err(UmfpackError::NonContiguousArray(name));
    }
    array.data().ok_or(UmfpackError::DataUnavailable(name))
}

fn array_data_mut<'a, Item, ArrayImpl>(
    array: &'a mut Array<ArrayImpl, 1>,
    name: &'static str,
) -> UmfpackResult<&'a mut [Item]>
where
    ArrayImpl: MutableRawAccessArrayImpl<Item, 1>,
{
    if !array.is_contiguous() {
        return Err(UmfpackError::NonContiguousArray(name));
    }
    array.data_mut().ok_or(UmfpackError::DataUnavailable(name))
}
