//! Implementation of ?sytrf/?hetrf - Factorization of a symmetric/Hermitian matrix.

use lapack::{chetrf, dsytrf, ssytrf, zhetrf};
use num::Zero;

use crate::base_types::{c32, c64};
use crate::dense::linalg::lapack::interface::lapack_return;
use crate::{base_types::LapackResult, traits::rlst_num::RlstScalar};

/// `Uplo` parameter for `?sytrf` to specify which triangular part of the matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SytrfUplo {
    /// Upper triangular part of the matrix is stored.
    Upper = b'U',
    /// Lower triangular part of the matrix is stored.
    Lower = b'L',
}

/// ?sytrf/?hetrf - Factorization of a symmetric/Hermitian matrix.
pub trait Sytrf: RlstScalar {
    /// Perform the factorization of a symmetric/Hermitian matrix.
    fn sytrf(
        uplo: SytrfUplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [i32],
    ) -> LapackResult<()>;
}

macro_rules! implement_sytrf {
    ($scalar:ty, $sytrf:expr) => {
        impl Sytrf for $scalar {
            fn sytrf(
                uplo: SytrfUplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                ipiv: &mut [i32],
            ) -> LapackResult<()> {
                let mut info = 0;

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
                    ipiv.len(),
                    n,
                    "Require `ipiv.len()` {} == `n` {}.",
                    ipiv.len(),
                    n
                );

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $sytrf(
                        uplo as u8, n as i32, a, lda as i32, ipiv, &mut work, -1, &mut info,
                    );
                }

                if info != 0 {
                    return Err(crate::base_types::LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $sytrf(
                        uplo as u8,
                        n as i32,
                        a,
                        lda as i32,
                        ipiv,
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

implement_sytrf!(f32, ssytrf);
implement_sytrf!(f64, dsytrf);
implement_sytrf!(c32, chetrf);
implement_sytrf!(c64, zhetrf);
