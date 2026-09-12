//! Implementation of ?ormlq/?unmlq - apply Q from an LQ factorization.

use lapack::{cunmlq, dormlq, sormlq, zunmlq};
use num::{Zero, complex::ComplexFloat};

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::mqr::{MqrSide, MqrTransMode};
use crate::dense::linalg::lapack::interface::{lapack_int, lapack_return, matrix_len};

trait MlqTranspose {
    fn as_char(mode: MqrTransMode) -> u8;
}
macro_rules! impl_transpose {
    ($scalar:ty, $adjoint:expr) => {
        impl MlqTranspose for $scalar {
            fn as_char(mode: MqrTransMode) -> u8 {
                match mode {
                    MqrTransMode::NoTranspose => b'N',
                    MqrTransMode::ConjugateTranspose => $adjoint,
                }
            }
        }
    };
}
impl_transpose!(f32, b'T');
impl_transpose!(f64, b'T');
impl_transpose!(c32, b'C');
impl_transpose!(c64, b'C');

/// Apply the orthogonal or unitary Q represented by packed LQ reflectors.
pub trait Mlq: Sized {
    /// Apply Q or Q^H to `c` from the requested side.
    #[allow(clippy::too_many_arguments)]
    fn mlq(
        side: MqrSide,
        trans: MqrTransMode,
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        tau: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_mlq {
    ($scalar:ty, $mlq:expr) => {
        impl Mlq for $scalar {
            fn mlq(
                side: MqrSide,
                trans: MqrTransMode,
                m: usize,
                n: usize,
                k: usize,
                a: &[Self],
                lda: usize,
                tau: &[Self],
                c: &mut [Self],
                ldc: usize,
            ) -> LapackResult<()> {
                let nq = match side {
                    MqrSide::Left => m,
                    MqrSide::Right => n,
                };
                assert!(k <= nq, "Require `k` {k} <= affected dimension {nq}.");
                assert!(
                    lda >= k.max(1),
                    "Require `lda` {lda} >= `max(1, k)` {}.",
                    k.max(1)
                );
                assert_eq!(a.len(), matrix_len(lda, nq, "a"));
                assert_eq!(tau.len(), k);
                assert!(
                    ldc >= m.max(1),
                    "Require `ldc` {ldc} >= `max(1, m)` {}.",
                    m.max(1)
                );
                assert_eq!(c.len(), matrix_len(ldc, n, "c"));
                if m == 0 || n == 0 || k == 0 {
                    return Ok(());
                }
                let (m, n, k, lda, ldc) = (
                    lapack_int(m, "m"),
                    lapack_int(n, "n"),
                    lapack_int(k, "k"),
                    lapack_int(lda, "lda"),
                    lapack_int(ldc, "ldc"),
                );
                let trans = <$scalar as MlqTranspose>::as_char(trans);
                let mut info = 0;
                let mut work = [<$scalar>::zero(); 1];
                unsafe {
                    $mlq(
                        side as u8, trans, m, n, k, a, lda, tau, c, ldc, &mut work, -1, &mut info,
                    )
                };
                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lwork = lapack_int(work[0].re() as usize, "lwork").max(1);
                let mut work = vec![<$scalar>::zero(); lwork as usize];
                unsafe {
                    $mlq(
                        side as u8, trans, m, n, k, a, lda, tau, c, ldc, &mut work, lwork,
                        &mut info,
                    )
                };
                lapack_return(info, ())
            }
        }
    };
}
implement_mlq!(f32, sormlq);
implement_mlq!(f64, dormlq);
implement_mlq!(c32, cunmlq);
implement_mlq!(c64, zunmlq);

#[cfg(test)]
mod tests {
    use super::Mlq;
    use crate::base_types::{c32, c64};
    use crate::dense::linalg::lapack::interface::{
        gelqf::Gelqf,
        mqr::{MqrSide, MqrTransMode},
    };

    #[test]
    fn mlq_applies_and_reverses_q_with_padding() {
        let a_padding = 12345.0;
        let mut a = vec![
            1.0_f64, 2.0, a_padding, 3.0, 4.0, a_padding, 5.0, 6.0, a_padding,
        ];
        let mut tau = vec![0.0; 2];
        f64::gelqf(2, 3, &mut a, 3, &mut tau).unwrap();
        let c_padding = 54321.0;
        let original = [1.0, c_padding, 2.0, c_padding, 3.0, c_padding];
        let mut c = original;
        f64::mlq(
            MqrSide::Right,
            MqrTransMode::NoTranspose,
            1,
            3,
            2,
            &a,
            3,
            &tau,
            &mut c,
            2,
        )
        .unwrap();
        f64::mlq(
            MqrSide::Right,
            MqrTransMode::ConjugateTranspose,
            1,
            3,
            2,
            &a,
            3,
            &tau,
            &mut c,
            2,
        )
        .unwrap();
        for index in [0, 2, 4] {
            assert!((c[index] - original[index]).abs() < 1e-12);
        }
        assert_eq!([a[2], a[5], a[8]], [a_padding; 3]);
        assert_eq!([c[1], c[3], c[5]], [c_padding; 3]);
    }

    #[test]
    #[should_panic]
    fn mlq_rejects_invalid_output_storage() {
        f64::mlq(
            MqrSide::Right,
            MqrTransMode::NoTranspose,
            1,
            2,
            1,
            &[1.0, 0.0],
            1,
            &[0.0],
            &mut [0.0],
            1,
        )
        .unwrap();
    }

    #[test]
    fn mlq_supports_other_scalars() {
        let mut a = [1.0_f32];
        let mut tau = [0.0];
        f32::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        let mut c = [2.0];
        f32::mlq(
            MqrSide::Left,
            MqrTransMode::NoTranspose,
            1,
            1,
            1,
            &a,
            1,
            &tau,
            &mut c,
            1,
        )
        .unwrap();
        let mut a = [c32::new(1.0, 1.0)];
        let mut tau = [c32::new(0.0, 0.0)];
        c32::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        let mut c = [c32::new(2.0, 0.0)];
        c32::mlq(
            MqrSide::Left,
            MqrTransMode::ConjugateTranspose,
            1,
            1,
            1,
            &a,
            1,
            &tau,
            &mut c,
            1,
        )
        .unwrap();
        let mut a = [c64::new(1.0, 1.0)];
        let mut tau = [c64::new(0.0, 0.0)];
        c64::gelqf(1, 1, &mut a, 1, &mut tau).unwrap();
        let mut c = [c64::new(2.0, 0.0)];
        c64::mlq(
            MqrSide::Right,
            MqrTransMode::NoTranspose,
            1,
            1,
            1,
            &a,
            1,
            &tau,
            &mut c,
            1,
        )
        .unwrap();
    }
}
