//! Implementation of ?gges - generalized Schur (QZ) factorization without sorting.

use lapack::{cgges, dgges, sgges, zgges};
use num::{Zero, complex::ComplexFloat};

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::gees::JobVs;
use crate::dense::linalg::lapack::interface::{lapack_int, lapack_return, matrix_len};
use crate::traits::rlst_num::RlstScalar;

/// Compute an unsorted generalized Schur factorization.
pub trait Gges: RlstScalar {
    /// Overwrite A/B with S/T and optionally compute left/right Schur vectors.
    #[allow(clippy::too_many_arguments)]
    fn gges(
        jobq: JobVs,
        jobz: JobVs,
        n: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
        alpha: &mut [Self::Complex],
        beta: &mut [Self::Complex],
        q: Option<&mut [Self]>,
        ldq: usize,
        z: Option<&mut [Self]>,
        ldz: usize,
    ) -> LapackResult<()>;
}

fn output<'a, T: Zero>(
    job: JobVs,
    value: Option<&'a mut [T]>,
    ld: usize,
    n: usize,
    dummy: &'a mut [T],
) -> &'a mut [T] {
    match job {
        JobVs::Compute => {
            let value = value.expect("Schur-vector output required");
            assert!(ld >= n.max(1));
            assert_eq!(value.len(), matrix_len(ld, n, "Schur vectors"));
            value
        }
        JobVs::None => {
            assert!(ld >= 1);
            dummy
        }
    }
}

macro_rules! implement_real {
    ($scalar:ty,$gges:expr) => {
        impl Gges for $scalar {
            fn gges(
                jobq: JobVs,
                jobz: JobVs,
                n: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
                alpha: &mut [Self::Complex],
                beta: &mut [Self::Complex],
                q: Option<&mut [Self]>,
                ldq: usize,
                z: Option<&mut [Self]>,
                ldz: usize,
            ) -> LapackResult<()> {
                assert!(lda >= n.max(1));
                assert_eq!(a.len(), matrix_len(lda, n, "a"));
                assert!(ldb >= n.max(1));
                assert_eq!(b.len(), matrix_len(ldb, n, "b"));
                assert_eq!(alpha.len(), n);
                assert_eq!(beta.len(), n);
                let mut qdummy = [0.0];
                let mut zdummy = [0.0];
                let q = output(jobq, q, ldq, n, &mut qdummy);
                let z = output(jobz, z, ldz, n, &mut zdummy);
                if n == 0 {
                    return Ok(());
                }
                let ni = lapack_int(n, "n");
                let mut ar = vec![0.0; n];
                let mut ai = vec![0.0; n];
                let mut br = vec![0.0; n];
                let mut sdim = 0;
                let mut info = 0;
                let mut work = [0.0];
                let mut bwork = [0];
                unsafe {
                    $gges(
                        jobq as u8,
                        jobz as u8,
                        b'N',
                        None,
                        ni,
                        a,
                        lapack_int(lda, "lda"),
                        b,
                        lapack_int(ldb, "ldb"),
                        &mut sdim,
                        &mut ar,
                        &mut ai,
                        &mut br,
                        q,
                        lapack_int(ldq, "ldq"),
                        z,
                        lapack_int(ldz, "ldz"),
                        &mut work,
                        -1,
                        &mut bwork,
                        &mut info,
                    )
                };
                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lw = lapack_int(work[0] as usize, "lwork").max(1);
                let mut work = vec![0.0; lw as usize];
                unsafe {
                    $gges(
                        jobq as u8,
                        jobz as u8,
                        b'N',
                        None,
                        ni,
                        a,
                        lapack_int(lda, "lda"),
                        b,
                        lapack_int(ldb, "ldb"),
                        &mut sdim,
                        &mut ar,
                        &mut ai,
                        &mut br,
                        q,
                        lapack_int(ldq, "ldq"),
                        z,
                        lapack_int(ldz, "ldz"),
                        &mut work,
                        lw,
                        &mut bwork,
                        &mut info,
                    )
                };
                for i in 0..n {
                    alpha[i] = <Self::Complex>::new(ar[i], ai[i]);
                    beta[i] = <Self::Complex>::from_real(br[i]);
                }
                lapack_return(info, ())
            }
        }
    };
}
macro_rules! implement_complex {
    ($scalar:ty,$real:ty,$gges:expr) => {
        impl Gges for $scalar {
            fn gges(
                jobq: JobVs,
                jobz: JobVs,
                n: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
                alpha: &mut [Self::Complex],
                beta: &mut [Self::Complex],
                q: Option<&mut [Self]>,
                ldq: usize,
                z: Option<&mut [Self]>,
                ldz: usize,
            ) -> LapackResult<()> {
                assert!(lda >= n.max(1));
                assert_eq!(a.len(), matrix_len(lda, n, "a"));
                assert!(ldb >= n.max(1));
                assert_eq!(b.len(), matrix_len(ldb, n, "b"));
                assert_eq!(alpha.len(), n);
                assert_eq!(beta.len(), n);
                let mut qdummy = [<$scalar>::zero()];
                let mut zdummy = [<$scalar>::zero()];
                let q = output(jobq, q, ldq, n, &mut qdummy);
                let z = output(jobz, z, ldz, n, &mut zdummy);
                if n == 0 {
                    return Ok(());
                }
                let ni = lapack_int(n, "n");
                let mut sdim = 0;
                let mut info = 0;
                let mut work = [<$scalar>::zero()];
                let mut rwork = vec![<$real>::zero(); matrix_len(8, n, "rwork")];
                let mut bwork = [0];
                unsafe {
                    $gges(
                        jobq as u8,
                        jobz as u8,
                        b'N',
                        None,
                        ni,
                        a,
                        lapack_int(lda, "lda"),
                        b,
                        lapack_int(ldb, "ldb"),
                        &mut sdim,
                        alpha,
                        beta,
                        q,
                        lapack_int(ldq, "ldq"),
                        z,
                        lapack_int(ldz, "ldz"),
                        &mut work,
                        -1,
                        &mut rwork,
                        &mut bwork,
                        &mut info,
                    )
                };
                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lw = lapack_int(work[0].re() as usize, "lwork").max(1);
                let mut work = vec![<$scalar>::zero(); lw as usize];
                unsafe {
                    $gges(
                        jobq as u8,
                        jobz as u8,
                        b'N',
                        None,
                        ni,
                        a,
                        lapack_int(lda, "lda"),
                        b,
                        lapack_int(ldb, "ldb"),
                        &mut sdim,
                        alpha,
                        beta,
                        q,
                        lapack_int(ldq, "ldq"),
                        z,
                        lapack_int(ldz, "ldz"),
                        &mut work,
                        lw,
                        &mut rwork,
                        &mut bwork,
                        &mut info,
                    )
                };
                lapack_return(info, ())
            }
        }
    };
}
implement_real!(f32, sgges);
implement_real!(f64, dgges);
implement_complex!(c32, f32, cgges);
implement_complex!(c64, f64, zgges);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_job_combinations_and_empty() {
        for (jq, jz) in [
            (JobVs::None, JobVs::None),
            (JobVs::Compute, JobVs::None),
            (JobVs::None, JobVs::Compute),
            (JobVs::Compute, JobVs::Compute),
        ] {
            let a_padding = 12345.0;
            let b_padding = 23456.0;
            let q_padding = 34567.0;
            let z_padding = 45678.0;
            let mut a = [1.0_f64, 0.0, a_padding, 0.0, 2.0, a_padding];
            let mut b = [
                1.0_f64, 0.0, b_padding, b_padding, 0.0, 1.0, b_padding, b_padding,
            ];
            let mut alpha = [c64::new(0.0, 0.0); 2];
            let mut beta = alpha;
            let mut q = [0.0, 0.0, q_padding, 0.0, 0.0, q_padding];
            let mut z = [
                0.0, 0.0, z_padding, z_padding, 0.0, 0.0, z_padding, z_padding,
            ];
            f64::gges(
                jq,
                jz,
                2,
                &mut a,
                3,
                &mut b,
                4,
                &mut alpha,
                &mut beta,
                if jq == JobVs::Compute {
                    Some(&mut q)
                } else {
                    None
                },
                if jq == JobVs::Compute { 3 } else { 1 },
                if jz == JobVs::Compute {
                    Some(&mut z)
                } else {
                    None
                },
                if jz == JobVs::Compute { 4 } else { 1 },
            )
            .unwrap();
            assert!((alpha[0].re / beta[0].re - 1.0).abs() < 1e-12);
            assert_eq!([a[2], a[5]], [a_padding; 2]);
            assert_eq!([b[2], b[3], b[6], b[7]], [b_padding; 4]);
            if jq == JobVs::Compute {
                assert_eq!([q[2], q[5]], [q_padding; 2]);
            }
            if jz == JobVs::Compute {
                assert_eq!([z[2], z[3], z[6], z[7]], [z_padding; 4]);
            }
        }
        f64::gges(
            JobVs::None,
            JobVs::None,
            0,
            &mut [],
            1,
            &mut [],
            1,
            &mut [],
            &mut [],
            None,
            1,
            None,
            1,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "Schur-vector output required")]
    fn gges_rejects_missing_requested_output() {
        f64::gges(
            JobVs::Compute,
            JobVs::None,
            1,
            &mut [1.0],
            1,
            &mut [1.0],
            1,
            &mut [c64::new(0.0, 0.0)],
            &mut [c64::new(0.0, 0.0)],
            None,
            1,
            None,
            1,
        )
        .unwrap();
    }

    #[test]
    fn gges_supports_other_scalars() {
        let mut a = [1.0_f32];
        let mut b = [1.0_f32];
        let mut alpha = [c32::new(0.0, 0.0)];
        let mut beta = alpha;
        f32::gges(
            JobVs::None,
            JobVs::None,
            1,
            &mut a,
            1,
            &mut b,
            1,
            &mut alpha,
            &mut beta,
            None,
            1,
            None,
            1,
        )
        .unwrap();
        let mut a = [c32::new(1.0, 0.0)];
        let mut b = [c32::new(1.0, 0.0)];
        let mut alpha = [c32::new(0.0, 0.0)];
        let mut beta = alpha;
        c32::gges(
            JobVs::None,
            JobVs::None,
            1,
            &mut a,
            1,
            &mut b,
            1,
            &mut alpha,
            &mut beta,
            None,
            1,
            None,
            1,
        )
        .unwrap();
        let mut a = [c64::new(1.0, 0.0)];
        let mut b = [c64::new(1.0, 0.0)];
        let mut alpha = [c64::new(0.0, 0.0)];
        let mut beta = alpha;
        c64::gges(
            JobVs::None,
            JobVs::None,
            1,
            &mut a,
            1,
            &mut b,
            1,
            &mut alpha,
            &mut beta,
            None,
            1,
            None,
            1,
        )
        .unwrap();
    }
}
