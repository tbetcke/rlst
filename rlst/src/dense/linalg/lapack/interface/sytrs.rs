//! Implementation of ?sytrs/?hetrs - Solve a system of linear equations using symmetric/Hermitian factorization.

use lapack::{chetrs, dsytrs, ssytrs, zhetrs};

use crate::base_types::{c32, c64};
use crate::dense::linalg::lapack::interface::lapack_return;
use crate::dense::linalg::lapack::interface::sytrf::SytrfUplo;
use crate::{base_types::LapackResult, traits::rlst_num::RlstScalar};

/// ?sytrs/?hetrs - Solve a system of linear equations using symmetric/Hermitian factorization.
pub trait Sytrs: RlstScalar {
    /// Solve from packed factors produced by [`super::sytrf::Sytrf::sytrf`].
    ///
    /// `a` and `ipiv` are unchanged. The column-major `n x nrhs` array B is
    /// overwritten by the solution. `uplo` must match the factorization.
    /// Invalid slice lengths or leading dimensions panic; LAPACK failures are
    /// returned as errors.
    #[allow(clippy::too_many_arguments)]
    fn sytrs(
        uplo: SytrfUplo,
        n: usize,
        nrhs: usize,
        a: &[Self],
        lda: usize,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_sytrs {
    ($scalar:ty, $sytrs:expr) => {
        impl Sytrs for $scalar {
            fn sytrs(
                uplo: SytrfUplo,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> LapackResult<()> {
                let mut info = 0;

                assert!(
                    lda >= std::cmp::max(1, n),
                    "Require `lda` {} >= `max(1, N)` {} .",
                    lda,
                    std::cmp::max(1, n)
                );
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n,
                );

                assert_eq!(
                    ipiv.len(),
                    n,
                    "Require `ipiv.len()` {} == `n` {}.",
                    ipiv.len(),
                    n,
                );

                assert!(
                    ldb >= std::cmp::max(1, n),
                    "Require `ldb` {} >= `max(1, N)` {} .",
                    ldb,
                    std::cmp::max(1, n)
                );
                assert_eq!(
                    b.len(),
                    ldb * nrhs,
                    "Require `b.len()` {} == `ldb * nrhs` {}.",
                    b.len(),
                    ldb * nrhs,
                );

                unsafe {
                    $sytrs(
                        uplo as u8,
                        n as i32,
                        nrhs as i32,
                        a,
                        lda as i32,
                        ipiv,
                        b,
                        ldb as i32,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_sytrs!(f32, ssytrs);
implement_sytrs!(f64, dsytrs);
implement_sytrs!(c32, chetrs);
implement_sytrs!(c64, zhetrs);
