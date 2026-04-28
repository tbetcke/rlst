//! Implementation of ?sygv/?hegv - Generalized Symmetric/Hermitian Eigenvalue Problem.

use lapack::{chegv, dsygv, ssygv, zhegv};

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::lapack_return;
use crate::traits::rlst_num::RlstScalar;
use num::Zero;

use num::complex::ComplexFloat;

/// The job options for the eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JobZ {
    /// Compute eigenvalues only.
    None = b'N',
    /// Compute eigenvalues and eigenvectors.
    ValuesAndVectors = b'V',
}

/// Use upper or lower triangular part of the matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SygvUplo {
    /// Use the upper triangular part of the matrix.
    Upper = b'U',
    /// Use the lower triangular part of the matrix.
    Lower = b'L',
}

/// ?sygv/?hegv - Generalized Symmetric/Hermitian Eigenvalue Problem.
pub trait Sygv: RlstScalar {
    /// Compute generalized eigenvalues and optionally eigenvectors.
    ///
    /// Depending on `itype`, solves:
    /// 1: Ax = λBx
    /// 2: ABx = λx
    /// 3: BAx = λx
    #[allow(clippy::too_many_arguments)]
    fn sygv(
        itype: i32,
        jobz: JobZ,
        uplo: SygvUplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
        w: &mut [Self::Real],
    ) -> LapackResult<()>;
}

macro_rules! implement_sygv_real {
    ($scalar:ty, $sygv:expr) => {
        impl Sygv for $scalar {
            fn sygv(
                itype: i32,
                jobz: JobZ,
                uplo: SygvUplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
                w: &mut [Self::Real],
            ) -> LapackResult<()> {
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n
                );

                assert!(
                    lda >= std::cmp::max(1, n),
                    "Require `lda` {} >= `max(1, n)` {}.",
                    lda,
                    std::cmp::max(1, n)
                );

                assert_eq!(
                    b.len(),
                    ldb * n,
                    "Require `b.len()` {} == `ldb * n` {}.",
                    b.len(),
                    ldb * n
                );

                assert!(
                    ldb >= std::cmp::max(1, n),
                    "Require `ldb` {} >= `max(1, n)` {}.",
                    ldb,
                    std::cmp::max(1, n)
                );

                assert_eq!(w.len(), n, "Require `w.len()` {} == `n` {}.", w.len(), n);

                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];
                let itype = [itype];

                unsafe {
                    $sygv(
                        &itype, jobz as u8, uplo as u8, n as i32, a, lda as i32, b, ldb as i32, w,
                        &mut work, -1, &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0] as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $sygv(
                        &itype,
                        jobz as u8,
                        uplo as u8,
                        n as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        w,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

macro_rules! implement_hegv_complex {
    ($scalar:ty, $hegv:expr) => {
        impl Sygv for $scalar {
            fn sygv(
                itype: i32,
                jobz: JobZ,
                uplo: SygvUplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
                w: &mut [Self::Real],
            ) -> LapackResult<()> {
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n
                );

                assert!(
                    lda >= std::cmp::max(1, n),
                    "Require `lda` {} >= `max(1, n)` {}.",
                    lda,
                    std::cmp::max(1, n)
                );

                assert_eq!(
                    b.len(),
                    ldb * n,
                    "Require `b.len()` {} == `ldb * n` {}.",
                    b.len(),
                    ldb * n
                );

                assert!(
                    ldb >= std::cmp::max(1, n),
                    "Require `ldb` {} >= `max(1, n)` {}.",
                    ldb,
                    std::cmp::max(1, n)
                );

                assert_eq!(w.len(), n, "Require `w.len()` {} == `n` {}.", w.len(), n);

                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];
                let itype = [itype];
                let mut rwork = vec![
                    <<$scalar as RlstScalar>::Real as Zero>::zero();
                    std::cmp::max(1, 3 * n - 2)
                ];

                unsafe {
                    $hegv(
                        &itype, jobz as u8, uplo as u8, n as i32, a, lda as i32, b, ldb as i32, w,
                        &mut work, -1, &mut rwork, &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $hegv(
                        &itype,
                        jobz as u8,
                        uplo as u8,
                        n as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        w,
                        &mut work,
                        lwork as i32,
                        &mut rwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_sygv_real!(f32, ssygv);
implement_sygv_real!(f64, dsygv);
implement_hegv_complex!(c32, chegv);
implement_hegv_complex!(c64, zhegv);
