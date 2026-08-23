//! Raw UMFPACK C API declarations for the 64-bit index variants.

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
