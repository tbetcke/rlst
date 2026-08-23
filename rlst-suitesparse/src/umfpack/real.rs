use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use rlst::sparse::csr_mat::CsrMatrix;

use super::csc::CscMatrix;
use super::sys;
use super::{
    UMFPACK_A, UmfpackError, UmfpackFactorizationStatus, UmfpackResult, UmfpackSymbolic,
    factorization_status,
};

/// Symbolic UMFPACK factorization of a real RLST CSR matrix.
///
/// Construct this handle by calling [`UmfpackSymbolic::analyze`] on a
/// [`CsrMatrix<f64>`], then call [`UmfpackRealSymbolic::factorize`] to obtain a
/// numeric factorization.
pub struct UmfpackRealSymbolic {
    raw: *mut c_void,
    matrix: Arc<CscMatrix<f64>>,
}

impl UmfpackSymbolic for CsrMatrix<f64> {
    type Symbolic = UmfpackRealSymbolic;

    fn analyze(&self) -> UmfpackResult<Self::Symbolic> {
        let matrix = Arc::new(CscMatrix::from_csr(self)?);
        let mut raw = ptr::null_mut();
        let status = unsafe {
            sys::umfpack_dl_symbolic(
                matrix.rows_i64()?,
                matrix.columns_i64()?,
                matrix.column_pointers.as_ptr(),
                matrix.row_indices.as_ptr(),
                matrix.values.as_ptr(),
                &mut raw,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        if status != 0 {
            if !raw.is_null() {
                unsafe { sys::umfpack_dl_free_symbolic(&mut raw) };
            }
            return Err(UmfpackError::NativeFailure {
                operation: "symbolic analysis",
                status,
            });
        }
        if raw.is_null() {
            return Err(UmfpackError::NullHandle("symbolic analysis"));
        }

        Ok(UmfpackRealSymbolic { raw, matrix })
    }
}

impl UmfpackRealSymbolic {
    /// Create a numeric factorization that can solve systems with the analyzed
    /// matrix.
    pub fn factorize(&self) -> UmfpackResult<UmfpackNumeric> {
        let mut raw = ptr::null_mut();
        let status = unsafe {
            sys::umfpack_dl_numeric(
                self.matrix.column_pointers.as_ptr(),
                self.matrix.row_indices.as_ptr(),
                self.matrix.values.as_ptr(),
                self.raw,
                &mut raw,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        if status < 0 {
            if !raw.is_null() {
                unsafe { sys::umfpack_dl_free_numeric(&mut raw) };
            }
            return Err(UmfpackError::NativeFailure {
                operation: "numeric factorization",
                status,
            });
        }
        if raw.is_null() {
            return Err(UmfpackError::NullHandle("numeric factorization"));
        }

        Ok(UmfpackNumeric {
            raw,
            matrix: Arc::clone(&self.matrix),
            status: factorization_status(status),
        })
    }
}

impl Drop for UmfpackRealSymbolic {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { sys::umfpack_dl_free_symbolic(&mut self.raw) };
        }
    }
}

/// Numeric UMFPACK factorization of an RLST CSR matrix.
pub struct UmfpackNumeric {
    raw: *mut c_void,
    matrix: Arc<CscMatrix<f64>>,
    status: UmfpackFactorizationStatus,
}

impl UmfpackNumeric {
    /// Return the non-error status reported while this factorization was
    /// created.
    pub fn status(&self) -> UmfpackFactorizationStatus {
        self.status
    }

    /// Solve `Ax = rhs` and return the solution vector.
    pub fn solve(&self, rhs: &[f64]) -> UmfpackResult<Vec<f64>> {
        let mut solution = vec![0.0; self.matrix.rows];
        self.solve_into(rhs, &mut solution)?;
        Ok(solution)
    }

    /// Solve `Ax = rhs`, writing the result into `solution`.
    pub fn solve_into(&self, rhs: &[f64], solution: &mut [f64]) -> UmfpackResult<()> {
        self.validate_vector_len(rhs.len())?;
        self.validate_vector_len(solution.len())?;

        let status = unsafe {
            sys::umfpack_dl_solve(
                UMFPACK_A,
                self.matrix.column_pointers.as_ptr(),
                self.matrix.row_indices.as_ptr(),
                self.matrix.values.as_ptr(),
                solution.as_mut_ptr(),
                rhs.as_ptr(),
                self.raw,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        if status == 0 {
            Ok(())
        } else {
            Err(UmfpackError::NativeFailure {
                operation: "solve",
                status,
            })
        }
    }

    fn validate_vector_len(&self, actual: usize) -> UmfpackResult<()> {
        if actual == self.matrix.rows {
            Ok(())
        } else {
            Err(UmfpackError::DimensionMismatch {
                expected: self.matrix.rows,
                actual,
            })
        }
    }
}

impl Drop for UmfpackNumeric {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { sys::umfpack_dl_free_numeric(&mut self.raw) };
        }
    }
}
