//! Implementation of ?gelqf - LQ factorization.

use lapack::{cgelqf, dgelqf, sgelqf, zgelqf};
use num::{Zero, complex::ComplexFloat};

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::{lapack_int, lapack_return, matrix_len};

/// Compute the packed LQ factorization of an `m` by `n` matrix.
pub trait Gelqf: Sized {
    /// Factor `a` in place, returning the elementary-reflector factors in `tau`.
    fn gelqf(m: usize, n: usize, a: &mut [Self], lda: usize, tau: &mut [Self]) -> LapackResult<()>;
}

macro_rules! implement_gelqf {
    ($scalar:ty, $gelqf:expr) => {
        impl Gelqf for $scalar {
            fn gelqf(
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                tau: &mut [Self],
            ) -> LapackResult<()> {
                assert!(
                    lda >= m.max(1),
                    "Require `lda` {lda} >= `max(1, m)` {}.",
                    m.max(1)
                );
                assert_eq!(a.len(), matrix_len(lda, n, "a"));
                assert_eq!(tau.len(), m.min(n));
                if m == 0 || n == 0 {
                    return Ok(());
                }
                let (m, n, lda) = (
                    lapack_int(m, "m"),
                    lapack_int(n, "n"),
                    lapack_int(lda, "lda"),
                );
                let mut info = 0;
                let mut work = [<$scalar>::zero(); 1];
                unsafe { $gelqf(m, n, a, lda, tau, &mut work, -1, &mut info) };
                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lwork = lapack_int(work[0].re() as usize, "lwork").max(1);
                let mut work = vec![<$scalar>::zero(); lwork as usize];
                unsafe { $gelqf(m, n, a, lda, tau, &mut work, lwork, &mut info) };
                lapack_return(info, ())
            }
        }
    };
}

implement_gelqf!(f32, sgelqf);
implement_gelqf!(f64, dgelqf);
implement_gelqf!(c32, cgelqf);
implement_gelqf!(c64, zgelqf);

#[cfg(test)]
mod tests {
    use super::Gelqf;
    use crate::base_types::{c32, c64};

    #[test]
    fn gelqf_real_padded_and_empty() {
        let padding = 12345.0;
        let mut a = vec![1.0_f64, 2.0, padding, 3.0, 4.0, padding, 5.0, 6.0, padding];
        let mut tau = vec![0.0; 2];
        f64::gelqf(2, 3, &mut a, 3, &mut tau).unwrap();
        f64::gelqf(0, 0, &mut [], 1, &mut []).unwrap();
        assert!(tau.iter().any(|value| *value != 0.0));
        assert_eq!([a[2], a[5], a[8]], [padding; 3]);
    }

    #[test]
    #[should_panic]
    fn gelqf_rejects_invalid_storage() {
        f64::gelqf(2, 2, &mut [0.0; 3], 2, &mut [0.0; 2]).unwrap();
    }

    #[test]
    fn gelqf_supports_all_scalars() {
        let mut a = [1.0_f32];
        let mut tau = [0.0_f32];
        f32::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        let mut a = [c32::new(1.0, 1.0)];
        let mut tau = [c32::new(0.0, 0.0)];
        c32::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        let mut a = [c64::new(1.0, 1.0)];
        let mut tau = [c64::new(0.0, 0.0)];
        c64::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
    }
}
