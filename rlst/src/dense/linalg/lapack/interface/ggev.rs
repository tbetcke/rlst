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
        alpha: &mut [Self::Complex],
        beta: &mut [Self::Complex],
        vl: &mut [Self::Complex],
        ldvl: usize,
        vr: &mut [Self::Complex],
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
                alpha: &mut [Self::Complex],
                beta: &mut [Self::Complex],
                vl: &mut [Self::Complex],
                ldvl: usize,
                vr: &mut [Self::Complex],
                ldvr: usize,
            ) -> LapackResult<()> {
                let mut alphar = vec![<$scalar>::zero(); n];
                let mut alphai = vec![<$scalar>::zero(); n];
                let mut beta_real = vec![<$scalar>::zero(); n];

                let mut vlr = match jobvl {
                    Job::Compute => vec![<$scalar>::zero(); ldvl * n],
                    Job::None => vec![<$scalar>::zero(); 1],
                };
                let mut vrr = match jobvr {
                    Job::Compute => vec![<$scalar>::zero(); ldvr * n],
                    Job::None => vec![<$scalar>::zero(); 1],
                };

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
                        &mut alphar,
                        &mut alphai,
                        &mut beta_real,
                        &mut vlr,
                        ldvl as i32,
                        &mut vrr,
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
                        &mut alphar,
                        &mut alphai,
                        &mut beta_real,
                        &mut vlr,
                        ldvl as i32,
                        &mut vrr,
                        ldvr as i32,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }

                for i in 0..n {
                    alpha[i] = <<$scalar as RlstScalar>::Complex>::new(alphar[i], alphai[i]);
                    beta[i] = <<$scalar as RlstScalar>::Complex>::from_real(beta_real[i]);
                }

                if jobvl == Job::Compute {
                    let mut count = 0;
                    while count < n {
                        if count < n - 1 && alphai[count] != <$scalar>::zero() {
                            for row in 0..n {
                                vl[row + count * ldvl] = <<$scalar as RlstScalar>::Complex>::new(
                                    vlr[row + count * ldvl],
                                    vlr[row + (1 + count) * ldvl],
                                );
                                vl[row + (1 + count) * ldvl] =
                                    <<$scalar as RlstScalar>::Complex>::new(
                                        vlr[row + count * ldvl],
                                        -vlr[row + (1 + count) * ldvl],
                                    );
                            }
                            count += 2;
                        } else {
                            for row in 0..n {
                                vl[row + count * ldvl] =
                                    <<$scalar as RlstScalar>::Complex>::from_real(
                                        vlr[row + count * ldvl],
                                    );
                            }
                            count += 1;
                        }
                    }
                }

                if jobvr == Job::Compute {
                    let mut count = 0;
                    while count < n {
                        if count < n - 1 && alphai[count] != <$scalar>::zero() {
                            for row in 0..n {
                                vr[row + count * ldvr] = <<$scalar as RlstScalar>::Complex>::new(
                                    vrr[row + count * ldvr],
                                    vrr[row + (1 + count) * ldvr],
                                );
                                vr[row + (1 + count) * ldvr] =
                                    <<$scalar as RlstScalar>::Complex>::new(
                                        vrr[row + count * ldvr],
                                        -vrr[row + (1 + count) * ldvr],
                                    );
                            }
                            count += 2;
                        } else {
                            for row in 0..n {
                                vr[row + count * ldvr] =
                                    <<$scalar as RlstScalar>::Complex>::from_real(
                                        vrr[row + count * ldvr],
                                    );
                            }
                            count += 1;
                        }
                    }
                }

                lapack_return(info, ())
            }
        }
    };
}

macro_rules! implement_ggev_complex {
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
                alpha: &mut [Self::Complex],
                beta: &mut [Self::Complex],
                vl: &mut [Self::Complex],
                ldvl: usize,
                vr: &mut [Self::Complex],
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
