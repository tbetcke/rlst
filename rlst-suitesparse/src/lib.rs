//! SuiteSparse bindings for RLST sparse matrices.
//!
//! This crate currently provides a safe UMFPACK interface for solving real and
//! complex double-precision systems stored as [`rlst::sparse::csr_mat::CsrMatrix`].
//! SuiteSparse must be installed natively. Set `SUITESPARSE_PATH` to its
//! installation prefix when it is not discoverable in a standard location.
//!
//! ```
//! use rlst::FromAij;
//! use rlst::sparse::csr_mat::CsrMatrix;
//! use rlst_suitesparse::UmfpackSymbolic;
//!
//! let matrix = CsrMatrix::from_aij(
//!     [2, 2],
//!     &[0, 0, 1, 1],
//!     &[0, 1, 0, 1],
//!     &[4.0, 1.0, 2.0, 3.0],
//! );
//! let symbolic = matrix.analyze()?;
//! let numeric = symbolic.factorize()?;
//! let solution = numeric.solve(&[1.0, 1.0])?;
//! assert!((solution[0] - 0.2).abs() < 1e-12);
//! assert!((solution[1] - 0.2).abs() < 1e-12);
//! # Ok::<(), rlst_suitesparse::UmfpackError>(())
//! ```

#![warn(missing_docs)]

use std::error::Error;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::ops::AddAssign;
use std::ptr;
use std::sync::Arc;

use rlst::sparse::csr_mat::CsrMatrix;
use rlst::{Shape, c64};

const UMFPACK_A: i32 = 0;
const UMFPACK_WARNING_SINGULAR_MATRIX: i32 = 1;

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
/// [`CsrMatrix<f64>`] or [`CsrMatrix<c64>`].
pub trait UmfpackSymbolic {
    /// Concrete symbolic factorization returned for the matrix scalar type.
    type Symbolic;

    /// Convert this matrix from CSR to UMFPACK's CSC representation and perform
    /// symbolic analysis.
    fn analyze(&self) -> UmfpackResult<Self::Symbolic>;
}

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

/// Symbolic UMFPACK factorization of a complex RLST CSR matrix.
///
/// Construct this handle by calling [`UmfpackSymbolic::analyze`] on a
/// [`CsrMatrix<c64>`], then call [`UmfpackComplexSymbolic::factorize`] to obtain
/// a numeric factorization.
pub struct UmfpackComplexSymbolic {
    raw: *mut c_void,
    matrix: Arc<ComplexCscMatrix>,
}

impl UmfpackSymbolic for CsrMatrix<c64> {
    type Symbolic = UmfpackComplexSymbolic;

    fn analyze(&self) -> UmfpackResult<Self::Symbolic> {
        let matrix = Arc::new(ComplexCscMatrix::from_csr(self)?);
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

        Ok(UmfpackComplexSymbolic { raw, matrix })
    }
}

impl UmfpackComplexSymbolic {
    /// Create a numeric factorization that can solve systems with the analyzed
    /// matrix.
    pub fn factorize(&self) -> UmfpackResult<UmfpackComplexNumeric> {
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

    /// Solve `Ax = rhs` and return the solution vector.
    pub fn solve(&self, rhs: &[c64]) -> UmfpackResult<Vec<c64>> {
        let mut solution = vec![c64::new(0.0, 0.0); self.matrix.rows];
        self.solve_into(rhs, &mut solution)?;
        Ok(solution)
    }

    /// Solve `Ax = rhs`, writing the result into `solution`.
    pub fn solve_into(&self, rhs: &[c64], solution: &mut [c64]) -> UmfpackResult<()> {
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

fn complex_parts(values: &[c64]) -> (Vec<f64>, Vec<f64>) {
    values.iter().map(|value| (value.re, value.im)).unzip()
}

#[derive(Debug)]
struct CscMatrix<Item> {
    rows: usize,
    columns: usize,
    column_pointers: Vec<i64>,
    row_indices: Vec<i64>,
    values: Vec<Item>,
}

impl<Item> CscMatrix<Item>
where
    Item: AddAssign + Copy + Default,
{
    fn from_csr(matrix: &CsrMatrix<Item>) -> UmfpackResult<Self> {
        let [rows, columns] = matrix.shape();
        if rows == 0 || columns == 0 {
            return Err(UmfpackError::EmptyMatrix);
        }
        if rows != columns {
            return Err(UmfpackError::NonSquareMatrix { rows, columns });
        }
        let indptr = matrix.indptr().data().ok_or(UmfpackError::InvalidCsr(
            "index pointers are not contiguous",
        ))?;
        let indices = matrix.indices().data().ok_or(UmfpackError::InvalidCsr(
            "column indices are not contiguous",
        ))?;
        let data = matrix
            .data()
            .data()
            .ok_or(UmfpackError::InvalidCsr("values are not contiguous"))?;

        validate_csr(rows, columns, indptr, indices, data)?;

        let mut counts = vec![0usize; columns];
        for &column in indices {
            counts[column] = counts[column]
                .checked_add(1)
                .ok_or(UmfpackError::IndexOutOfRange(column))?;
        }

        let mut pointers = vec![0usize; columns + 1];
        for column in 0..columns {
            pointers[column + 1] = pointers[column]
                .checked_add(counts[column])
                .ok_or(UmfpackError::IndexOutOfRange(counts[column]))?;
        }
        let mut offsets = pointers[..columns].to_vec();
        let mut raw_rows = vec![0usize; indices.len()];
        let mut raw_values = vec![Item::default(); data.len()];

        for row in 0..rows {
            for entry in indptr[row]..indptr[row + 1] {
                let column = indices[entry];
                let offset = offsets[column];
                raw_rows[offset] = row;
                raw_values[offset] = data[entry];
                offsets[column] += 1;
            }
        }

        let mut column_pointers = Vec::with_capacity(columns + 1);
        let mut row_indices = Vec::with_capacity(indices.len());
        let mut values = Vec::with_capacity(data.len());
        column_pointers.push(0);
        for column in 0..columns {
            let mut entry = pointers[column];
            while entry < pointers[column + 1] {
                let row = raw_rows[entry];
                let mut value = raw_values[entry];
                entry += 1;
                while entry < pointers[column + 1] && raw_rows[entry] == row {
                    value += raw_values[entry];
                    entry += 1;
                }
                row_indices.push(to_i64(row)?);
                values.push(value);
            }
            column_pointers.push(to_i64(values.len())?);
        }

        Ok(Self {
            rows,
            columns,
            column_pointers,
            row_indices,
            values,
        })
    }

    fn rows_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.rows)
    }

    fn columns_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.columns)
    }
}

#[derive(Debug)]
struct ComplexCscMatrix {
    rows: usize,
    columns: usize,
    column_pointers: Vec<i64>,
    row_indices: Vec<i64>,
    real_values: Vec<f64>,
    imaginary_values: Vec<f64>,
}

impl ComplexCscMatrix {
    fn from_csr(matrix: &CsrMatrix<c64>) -> UmfpackResult<Self> {
        let csc = CscMatrix::from_csr(matrix)?;
        let (real_values, imaginary_values) = complex_parts(&csc.values);

        Ok(Self {
            rows: csc.rows,
            columns: csc.columns,
            column_pointers: csc.column_pointers,
            row_indices: csc.row_indices,
            real_values,
            imaginary_values,
        })
    }

    fn rows_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.rows)
    }

    fn columns_i64(&self) -> UmfpackResult<i64> {
        to_i64(self.columns)
    }
}

fn validate_csr<Item>(
    rows: usize,
    columns: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[Item],
) -> UmfpackResult<()> {
    if indptr.len() != rows + 1 {
        return Err(UmfpackError::InvalidCsr(
            "index pointer length does not match the row count",
        ));
    }
    if indptr.first() != Some(&0) || indptr.last() != Some(&indices.len()) {
        return Err(UmfpackError::InvalidCsr(
            "index pointers do not span the column-index array",
        ));
    }
    if indices.len() != data.len() {
        return Err(UmfpackError::InvalidCsr(
            "column-index and value lengths differ",
        ));
    }
    if indptr.windows(2).any(|window| window[0] > window[1]) {
        return Err(UmfpackError::InvalidCsr(
            "index pointers are not monotonically increasing",
        ));
    }
    if indices.iter().any(|&column| column >= columns) {
        return Err(UmfpackError::InvalidCsr(
            "a column index is outside the matrix shape",
        ));
    }
    Ok(())
}

fn to_i64(value: usize) -> UmfpackResult<i64> {
    i64::try_from(value).map_err(|_| UmfpackError::IndexOutOfRange(value))
}

fn factorization_status(status: i32) -> UmfpackFactorizationStatus {
    match status {
        0 => UmfpackFactorizationStatus::Success,
        UMFPACK_WARNING_SINGULAR_MATRIX => UmfpackFactorizationStatus::SingularMatrix,
        status => UmfpackFactorizationStatus::Warning(status),
    }
}

mod sys {
    use std::ffi::c_void;

    unsafe extern "C" {
        pub fn umfpack_dl_symbolic(
            n_row: i64,
            n_col: i64,
            ap: *const i64,
            ai: *const i64,
            ax: *const f64,
            symbolic: *mut *mut c_void,
            control: *const f64,
            info: *mut f64,
        ) -> i32;
        pub fn umfpack_dl_numeric(
            ap: *const i64,
            ai: *const i64,
            ax: *const f64,
            symbolic: *mut c_void,
            numeric: *mut *mut c_void,
            control: *const f64,
            info: *mut f64,
        ) -> i32;
        pub fn umfpack_dl_solve(
            system: i32,
            ap: *const i64,
            ai: *const i64,
            ax: *const f64,
            solution: *mut f64,
            rhs: *const f64,
            numeric: *mut c_void,
            control: *const f64,
            info: *mut f64,
        ) -> i32;
        pub fn umfpack_dl_free_symbolic(symbolic: *mut *mut c_void);
        pub fn umfpack_dl_free_numeric(numeric: *mut *mut c_void);
        pub fn umfpack_zl_symbolic(
            n_row: i64,
            n_col: i64,
            ap: *const i64,
            ai: *const i64,
            ax: *const f64,
            az: *const f64,
            symbolic: *mut *mut c_void,
            control: *const f64,
            info: *mut f64,
        ) -> i32;
        pub fn umfpack_zl_numeric(
            ap: *const i64,
            ai: *const i64,
            ax: *const f64,
            az: *const f64,
            symbolic: *mut c_void,
            numeric: *mut *mut c_void,
            control: *const f64,
            info: *mut f64,
        ) -> i32;
        pub fn umfpack_zl_solve(
            system: i32,
            ap: *const i64,
            ai: *const i64,
            ax: *const f64,
            az: *const f64,
            solution_real: *mut f64,
            solution_imaginary: *mut f64,
            rhs_real: *const f64,
            rhs_imaginary: *const f64,
            numeric: *mut c_void,
            control: *const f64,
            info: *mut f64,
        ) -> i32;
        pub fn umfpack_zl_free_symbolic(symbolic: *mut *mut c_void);
        pub fn umfpack_zl_free_numeric(numeric: *mut *mut c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlst::{DynArray, FromAij};

    fn test_matrix() -> CsrMatrix<f64> {
        CsrMatrix::from_aij([2, 2], &[0, 0, 1, 1], &[0, 1, 0, 1], &[4.0, 1.0, 2.0, 3.0])
    }

    fn complex_test_matrix() -> CsrMatrix<c64> {
        CsrMatrix::from_aij(
            [2, 2],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[
                c64::new(4.0, 1.0),
                c64::new(1.0, -1.0),
                c64::new(2.0, 1.0),
                c64::new(3.0, -2.0),
            ],
        )
    }

    #[test]
    fn converts_csr_to_csc() {
        let csc = CscMatrix::from_csr(&test_matrix()).unwrap();

        assert_eq!(csc.column_pointers, vec![0, 2, 4]);
        assert_eq!(csc.row_indices, vec![0, 1, 0, 1]);
        assert_eq!(csc.values, vec![4.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn combines_duplicate_csr_entries() {
        let matrix = CsrMatrix::new(
            [2, 2],
            DynArray::from_shape_and_vec([3], vec![0, 0, 1]),
            DynArray::from_shape_and_vec([3], vec![0, 2, 3]),
            DynArray::from_shape_and_vec([3], vec![1.0, 3.0, 2.0]),
        );
        let csc = CscMatrix::from_csr(&matrix).unwrap();

        assert_eq!(csc.column_pointers, vec![0, 1, 2]);
        assert_eq!(csc.row_indices, vec![0, 1]);
        assert_eq!(csc.values, vec![4.0, 2.0]);
    }

    #[test]
    fn solves_a_real_system() {
        let symbolic = test_matrix().analyze().unwrap();
        let numeric = symbolic.factorize().unwrap();
        let solution = numeric.solve(&[1.0, 1.0]).unwrap();

        assert_eq!(numeric.status(), UmfpackFactorizationStatus::Success);
        assert!((solution[0] - 0.2).abs() < 1e-12);
        assert!((solution[1] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn rejects_incompatible_rhs() {
        let symbolic = test_matrix().analyze().unwrap();
        let numeric = symbolic.factorize().unwrap();

        assert_eq!(
            numeric.solve(&[1.0]),
            Err(UmfpackError::DimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn solves_a_complex_system() {
        let symbolic = complex_test_matrix().analyze().unwrap();
        let numeric = symbolic.factorize().unwrap();
        let solution = numeric
            .solve(&[c64::new(4.0, 2.0), c64::new(5.0, -4.0)])
            .unwrap();

        assert_eq!(numeric.status(), UmfpackFactorizationStatus::Success);
        assert!((solution[0] - c64::new(1.0, 1.0)).norm() < 1e-12);
        assert!((solution[1] - c64::new(2.0, -1.0)).norm() < 1e-12);
    }
}
