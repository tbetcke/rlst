use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use rlst::c64;
use rlst::sparse::csr_mat::CsrMatrix;
use rlst::{Array, DynArray, MutableRawAccessArrayImpl, RawAccessArrayImpl, rlst_dynamic_array};

use super::csc::{ComplexCscMatrix, complex_parts};
use super::sys;
use super::{
    UMFPACK_A, UmfpackError, UmfpackFactorizationStatus, UmfpackFactorize, UmfpackResult,
    array_data, array_data_mut, factorization_status,
};

struct UmfpackComplexSymbolic {
    raw: *mut c_void,
    matrix: Arc<ComplexCscMatrix>,
}

impl UmfpackFactorize for CsrMatrix<c64> {
    type Numeric = UmfpackComplexNumeric;

    fn factorize(&self) -> UmfpackResult<Self::Numeric> {
        UmfpackComplexSymbolic::analyze(self)?.factorize()
    }
}

impl UmfpackComplexSymbolic {
    fn analyze(matrix: &CsrMatrix<c64>) -> UmfpackResult<Self> {
        let matrix = Arc::new(ComplexCscMatrix::from_csr(matrix)?);
        let mut raw = ptr::null_mut();
        let status = unsafe {
            sys::umfpack_zl_symbolic(
                matrix.rows_i64()?,
                matrix.columns_i64()?,
                matrix.column_pointers.as_ptr(),
                matrix.row_indices.as_ptr(),
                matrix.real_values.as_ptr(),
                matrix.imaginary_values.as_ptr(),
                &mut raw,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        if status != 0 {
            if !raw.is_null() {
                unsafe { sys::umfpack_zl_free_symbolic(&mut raw) };
            }
            return Err(UmfpackError::NativeFailure {
                operation: "complex symbolic analysis",
                status,
            });
        }
        if raw.is_null() {
            return Err(UmfpackError::NullHandle("complex symbolic analysis"));
        }

        Ok(Self { raw, matrix })
    }

    fn factorize(&self) -> UmfpackResult<UmfpackComplexNumeric> {
        let mut raw = ptr::null_mut();
        let status = unsafe {
            sys::umfpack_zl_numeric(
                self.matrix.column_pointers.as_ptr(),
                self.matrix.row_indices.as_ptr(),
                self.matrix.real_values.as_ptr(),
                self.matrix.imaginary_values.as_ptr(),
                self.raw,
                &mut raw,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        if status < 0 {
            if !raw.is_null() {
                unsafe { sys::umfpack_zl_free_numeric(&mut raw) };
            }
            return Err(UmfpackError::NativeFailure {
                operation: "complex numeric factorization",
                status,
            });
        }
        if raw.is_null() {
            return Err(UmfpackError::NullHandle("complex numeric factorization"));
        }

        Ok(UmfpackComplexNumeric {
            raw,
            matrix: Arc::clone(&self.matrix),
            status: factorization_status(status),
        })
    }
}

impl Drop for UmfpackComplexSymbolic {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { sys::umfpack_zl_free_symbolic(&mut self.raw) };
        }
    }
}

/// Numeric UMFPACK factorization of a complex RLST CSR matrix.
pub struct UmfpackComplexNumeric {
    raw: *mut c_void,
    matrix: Arc<ComplexCscMatrix>,
    status: UmfpackFactorizationStatus,
}

impl UmfpackComplexNumeric {
    /// Return the non-error status reported while this factorization was
    /// created.
    pub fn status(&self) -> UmfpackFactorizationStatus {
        self.status
    }

    /// Solve `Ax = rhs` and return the solution in a new dynamic array.
    pub fn solve<ArrayImpl>(&self, rhs: &Array<ArrayImpl, 1>) -> UmfpackResult<DynArray<c64, 1>>
    where
        ArrayImpl: RawAccessArrayImpl<c64, 1>,
    {
        let mut solution = rlst_dynamic_array!(c64, [self.matrix.rows]);
        self.solve_into(rhs, &mut solution)?;
        Ok(solution)
    }

    /// Solve `Ax = rhs`, writing the result into `solution`.
    pub fn solve_into<RhsArrayImpl, SolutionArrayImpl>(
        &self,
        rhs: &Array<RhsArrayImpl, 1>,
        solution: &mut Array<SolutionArrayImpl, 1>,
    ) -> UmfpackResult<()>
    where
        RhsArrayImpl: RawAccessArrayImpl<c64, 1>,
        SolutionArrayImpl: MutableRawAccessArrayImpl<c64, 1>,
    {
        let rhs = array_data(rhs, "right-hand side")?;
        let solution = array_data_mut(solution, "solution")?;
        self.validate_vector_len(rhs.len())?;
        self.validate_vector_len(solution.len())?;

        let (rhs_real, rhs_imaginary) = complex_parts(rhs);
        let mut solution_real = vec![0.0; solution.len()];
        let mut solution_imaginary = vec![0.0; solution.len()];
        let status = unsafe {
            sys::umfpack_zl_solve(
                UMFPACK_A,
                self.matrix.column_pointers.as_ptr(),
                self.matrix.row_indices.as_ptr(),
                self.matrix.real_values.as_ptr(),
                self.matrix.imaginary_values.as_ptr(),
                solution_real.as_mut_ptr(),
                solution_imaginary.as_mut_ptr(),
                rhs_real.as_ptr(),
                rhs_imaginary.as_ptr(),
                self.raw,
                ptr::null(),
                ptr::null_mut(),
            )
        };

        if status == 0 {
            for (value, (real, imaginary)) in solution
                .iter_mut()
                .zip(solution_real.into_iter().zip(solution_imaginary))
            {
                *value = c64::new(real, imaginary);
            }
            Ok(())
        } else {
            Err(UmfpackError::NativeFailure {
                operation: "complex solve",
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

impl Drop for UmfpackComplexNumeric {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { sys::umfpack_zl_free_numeric(&mut self.raw) };
        }
    }
}
