//! Implementation of ?ggev - Generalized Eigenvalue Problem for general matrices.

use lapack::{cggev, dggev, sggev, zggev};

use crate::base_types::{c32, c64, LapackError, LapackResult};
use crate::dense::linalg::lapack::interface::lapack_return;
use crate::traits::rlst_num::RlstScalar;
use num::Zero;

use num::complex::ComplexFloat;

/// The job options for the eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Job {
    /// Do not compute eigenvectors.
    None = b'N',
    /// Compute eigenvectors.
    Compute = b'V',
}

/// ?ggev - Generalized Eigenvalue Problem for general matrices.
pub trait Ggev: RlstScalar {
    /// Compute generalized eigenvalues and optionally left and/or right eigenvectors.
    #[allow(clippy::too_many_arguments)]
    fn ggev(
        jobvl: Job,
        jobvr: Job,
        n: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
        alphar: &mut [Self::Real],
        alphai: &mut [Self::Real],
        beta: &mut [Self::Real],
        vl: &mut [Self],
        ldvl: usize,
        vr: &mut [Self],
        ldvr: usize,
    ) -> LapackResult<()>;

    /// Compute generalized eigenvalues and optionally left and/or right eigenvectors for complex matrices.
    #[allow(clippy::too_many_arguments)]
    fn ggev_complex(
        jobvl: Job,
        jobvr: Job,
        n: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
        alpha: &mut [Self],
        beta: &mut [Self],
        vl: &mut [Self],
        ldvl: usize,
        vr: &mut [Self],
        ldvr: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_ggev_real {
    ($scalar:ty, $ggev:expr) => {
        impl Ggev for $scalar {
            fn ggev(
                jobvl: Job,
                jobvr: Job,
                n: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
                alphar: &mut [Self::Real],
                alphai: &mut [Self::Real],
                beta: &mut [Self::Real],
                vl: &mut [Self],
                ldvl: usize,
                vr: &mut [Self],
                ldvr: usize,
            ) -> LapackResult<()> {
                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $ggev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        alphar,
                        alphai,
                        beta,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
                        &mut work,
                        -1,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0] as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $ggev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        alphar,
                        alphai,
                        beta,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }

            fn ggev_complex(
                _jobvl: Job,
                _jobvr: Job,
                _n: usize,
                _a: &mut [Self],
                _lda: usize,
                _b: &mut [Self],
                _ldb: usize,
                _alpha: &mut [Self],
                _beta: &mut [Self],
                _vl: &mut [Self],
                _ldvl: usize,
                _vr: &mut [Self],
                _ldvr: usize,
            ) -> LapackResult<()> {
                panic!("ggev_complex not implemented for real types")
            }
        }
    };
}

macro_rules! implement_ggev_complex {
    ($scalar:ty, $ggev:expr) => {
        impl Ggev for $scalar {
            fn ggev(
                _jobvl: Job,
                _jobvr: Job,
                _n: usize,
                _a: &mut [Self],
                _lda: usize,
                _b: &mut [Self],
                _ldb: usize,
                _alphar: &mut [Self::Real],
                _alphai: &mut [Self::Real],
                _beta: &mut [Self::Real],
                _vl: &mut [Self],
                _ldvl: usize,
                _vr: &mut [Self],
                _ldvr: usize,
            ) -> LapackResult<()> {
                panic!("ggev not implemented for complex types")
            }

            fn ggev_complex(
                jobvl: Job,
                jobvr: Job,
                n: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
                alpha: &mut [Self],
                beta: &mut [Self],
                vl: &mut [Self],
                ldvl: usize,
                vr: &mut [Self],
                ldvr: usize,
            ) -> LapackResult<()> {
                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];
                let mut rwork = vec![<<$scalar as RlstScalar>::Real>::zero(); 8 * n];

                unsafe {
                    $ggev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        alpha,
                        beta,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
                        &mut work,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $ggev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        alpha,
                        beta,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
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

implement_ggev_real!(f32, sggev);
implement_ggev_real!(f64, dggev);
implement_ggev_complex!(c32, cggev);
implement_ggev_complex!(c64, zggev);
