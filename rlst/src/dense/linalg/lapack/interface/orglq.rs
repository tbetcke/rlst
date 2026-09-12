//! Implementation of ?orglq/?unglq - form Q from an LQ factorization.

use lapack::{cunglq, dorglq, sorglq, zunglq};
use num::{Zero, complex::ComplexFloat};

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::{lapack_int, lapack_return, matrix_len};

/// Form rows of the orthogonal or unitary matrix Q from packed LQ reflectors.
pub trait Orglq: Sized {
    /// Form an `m` by `n` Q with `k` reflectors (`m <= n` and `k <= m`).
    fn orglq(
        m: usize,
        n: usize,
        k: usize,
        a: &mut [Self],
        lda: usize,
        tau: &[Self],
    ) -> LapackResult<()>;
}

macro_rules! implement_orglq {
    ($scalar:ty, $orglq:expr) => {
        impl Orglq for $scalar {
            fn orglq(
                m: usize,
                n: usize,
                k: usize,
                a: &mut [Self],
                lda: usize,
                tau: &[Self],
            ) -> LapackResult<()> {
                assert!(m <= n, "Require `m` {m} <= `n` {n}.");
                assert!(k <= m, "Require `k` {k} <= `m` {m}.");
                assert!(
                    lda >= m.max(1),
                    "Require `lda` {lda} >= `max(1, m)` {}.",
                    m.max(1)
                );
                assert_eq!(a.len(), matrix_len(lda, n, "a"));
                assert_eq!(tau.len(), k);
                if m == 0 || n == 0 {
                    return Ok(());
                }
                let (m, n, k, lda) = (
                    lapack_int(m, "m"),
                    lapack_int(n, "n"),
                    lapack_int(k, "k"),
                    lapack_int(lda, "lda"),
                );
                let mut info = 0;
                let mut work = [<$scalar>::zero(); 1];
                unsafe { $orglq(m, n, k, a, lda, tau, &mut work, -1, &mut info) };
                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lwork = lapack_int(work[0].re() as usize, "lwork").max(1);
                let mut work = vec![<$scalar>::zero(); lwork as usize];
                unsafe { $orglq(m, n, k, a, lda, tau, &mut work, lwork, &mut info) };
                lapack_return(info, ())
            }
        }
    };
}

implement_orglq!(f32, sorglq);
implement_orglq!(f64, dorglq);
implement_orglq!(c32, cunglq);
implement_orglq!(c64, zunglq);

#[cfg(test)]
mod tests {
    use super::Orglq;
    use crate::base_types::{c32, c64};
    use crate::dense::linalg::lapack::interface::gelqf::Gelqf;

    #[test]
    fn orglq_forms_padded_orthonormal_rows() {
        let padding = 12345.0;
        let mut a = vec![1.0_f64, 2.0, padding, 3.0, 4.0, padding, 5.0, 6.0, padding];
        let mut tau = vec![0.0; 2];
        f64::gelqf(2, 3, &mut a, 3, &mut tau).unwrap();
        f64::orglq(2, 3, 2, &mut a, 3, &tau).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let dot: f64 = (0..3).map(|col| a[i + 3 * col] * a[j + 3 * col]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        assert_eq!([a[2], a[5], a[8]], [padding; 3]);
    }

    #[test]
    #[should_panic]
    fn orglq_rejects_invalid_leading_dimension() {
        f64::orglq(2, 2, 2, &mut [0.0; 2], 1, &[0.0; 2]).unwrap();
    }

    #[test]
    fn orglq_supports_other_scalars() {
        let mut a = [1.0_f32];
        let mut tau = [0.0];
        f32::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        f32::orglq(1, 1, 1, &mut a, 1, &tau).unwrap();
        let mut a = [c32::new(1.0, 1.0)];
        let mut tau = [c32::new(0.0, 0.0)];
        c32::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        c32::orglq(1, 1, 1, &mut a, 1, &tau).unwrap();
        let mut a = [c64::new(1.0, 1.0)];
        let mut tau = [c64::new(0.0, 0.0)];
        c64::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        c64::orglq(1, 1, 1, &mut a, 1, &tau).unwrap();
    }
}
