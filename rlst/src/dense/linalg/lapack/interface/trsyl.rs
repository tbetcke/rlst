//! Implementation of ?trsyl - solve a triangular Sylvester equation.

use lapack::{ctrsyl, dtrsyl, strsyl, ztrsyl};

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::{lapack_int, matrix_len};
use crate::traits::rlst_num::RlstScalar;

/// Operation applied to a Schur-form coefficient matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrsylTranspose {
    /// Use the matrix unchanged.
    NoTranspose,
    /// Use its transpose for real values and conjugate transpose for complex values.
    ConjugateTranspose,
}

/// Sign of the second product in the Sylvester equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrsylSign {
    /// Add `X op(B)`.
    Plus,
    /// Subtract `X op(B)`.
    Minus,
}

/// Whether LAPACK solved the requested equation or a nearby perturbed equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrsylStatus {
    /// No perturbation was reported.
    Success,
    /// Close or common eigenvalues were perturbed.
    Perturbed,
}

/// Result metadata from TRSYL.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrsylOutcome<Real> {
    /// Scale in `op(A) X + sign X op(B) = scale C`.
    pub scale: Real,
    /// LAPACK perturbation status.
    pub status: TrsylStatus,
}

trait TrsylChar {
    fn trans_char(mode: TrsylTranspose) -> u8;
}
macro_rules! impl_char {
    ($t:ty,$c:expr) => {
        impl TrsylChar for $t {
            fn trans_char(mode: TrsylTranspose) -> u8 {
                match mode {
                    TrsylTranspose::NoTranspose => b'N',
                    TrsylTranspose::ConjugateTranspose => $c,
                }
            }
        }
    };
}
impl_char!(f32, b'T');
impl_char!(f64, b'T');
impl_char!(c32, b'C');
impl_char!(c64, b'C');

/// Solve `op(A) X +/- X op(B) = scale C`, overwriting C with X.
///
/// `a` and `b` must be in real Schur form for real values and triangular for complex values.
pub trait Trsyl: RlstScalar {
    /// Solve the triangular Sylvester equation and report scaling and perturbation.
    ///
    /// A is `m x m`, B is `n x n`, and C is `m x n`, all column-major. A and B
    /// are unchanged; C is overwritten by X. For real scalars A and B may have
    /// 2-by-2 diagonal Schur blocks. Invalid storage dimensions panic. LAPACK
    /// status 1 is returned as [`TrsylStatus::Perturbed`], not as an error.
    #[allow(clippy::too_many_arguments)]
    fn trsyl(
        trana: TrsylTranspose,
        tranb: TrsylTranspose,
        sign: TrsylSign,
        m: usize,
        n: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        ldc: usize,
    ) -> LapackResult<TrsylOutcome<Self::Real>>;
}
macro_rules! implement_trsyl {
    ($scalar:ty,$trsyl:expr) => {
        impl Trsyl for $scalar {
            fn trsyl(
                trana: TrsylTranspose,
                tranb: TrsylTranspose,
                sign: TrsylSign,
                m: usize,
                n: usize,
                a: &[Self],
                lda: usize,
                b: &[Self],
                ldb: usize,
                c: &mut [Self],
                ldc: usize,
            ) -> LapackResult<TrsylOutcome<Self::Real>> {
                assert!(lda >= m.max(1));
                assert_eq!(a.len(), matrix_len(lda, m, "a"));
                assert!(ldb >= n.max(1));
                assert_eq!(b.len(), matrix_len(ldb, n, "b"));
                assert!(ldc >= m.max(1));
                assert_eq!(c.len(), matrix_len(ldc, n, "c"));
                if m == 0 || n == 0 {
                    return Ok(TrsylOutcome {
                        scale: 1.0,
                        status: TrsylStatus::Success,
                    });
                }
                let isgn = [match sign {
                    TrsylSign::Plus => 1,
                    TrsylSign::Minus => -1,
                }];
                let mut scale = [0.0];
                let mut info = 0;
                unsafe {
                    $trsyl(
                        <$scalar as TrsylChar>::trans_char(trana),
                        <$scalar as TrsylChar>::trans_char(tranb),
                        &isgn,
                        lapack_int(m, "m"),
                        lapack_int(n, "n"),
                        a,
                        lapack_int(lda, "lda"),
                        b,
                        lapack_int(ldb, "ldb"),
                        c,
                        lapack_int(ldc, "ldc"),
                        &mut scale,
                        &mut info,
                    )
                };
                match info {
                    0 => Ok(TrsylOutcome {
                        scale: scale[0],
                        status: TrsylStatus::Success,
                    }),
                    1 => Ok(TrsylOutcome {
                        scale: scale[0],
                        status: TrsylStatus::Perturbed,
                    }),
                    code => Err(LapackError::LapackInfoCode(code)),
                }
            }
        }
    };
}
implement_trsyl!(f32, strsyl);
implement_trsyl!(f64, dtrsyl);
implement_trsyl!(c32, ctrsyl);
implement_trsyl!(c64, ztrsyl);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RlstScalar;
    use crate::base_types::{c32, c64};
    use paste::paste;
    #[test]
    fn solves_with_padding_and_reports_perturbation() {
        let padding = 12345.0;
        let mut c = [3.0_f64, padding];
        let out = f64::trsyl(
            TrsylTranspose::NoTranspose,
            TrsylTranspose::NoTranspose,
            TrsylSign::Plus,
            1,
            1,
            &[1.0, padding],
            2,
            &[2.0, padding, padding],
            3,
            &mut c,
            2,
        )
        .unwrap();
        assert_eq!(out.status, TrsylStatus::Success);
        assert!((c[0] - 1.0).abs() < 1e-14);
        assert_eq!(c[1], padding);
        let mut c = [1.0];
        let out = f64::trsyl(
            TrsylTranspose::NoTranspose,
            TrsylTranspose::NoTranspose,
            TrsylSign::Minus,
            1,
            1,
            &[1.0],
            1,
            &[1.0],
            1,
            &mut c,
            1,
        )
        .unwrap();
        assert_eq!(out.status, TrsylStatus::Perturbed);
    }

    #[test]
    #[should_panic]
    fn trsyl_rejects_invalid_coefficient_storage() {
        f64::trsyl(
            TrsylTranspose::NoTranspose,
            TrsylTranspose::NoTranspose,
            TrsylSign::Plus,
            2,
            1,
            &[1.0; 3],
            2,
            &[1.0],
            1,
            &mut [1.0; 2],
            2,
        )
        .unwrap();
    }

    macro_rules! operation_tests {
        ($scalar:ty, $tol:expr, $a:expr, $b:expr, $x:expr) => {
            paste! {
                #[test]
                fn [<trsyl_all_operations_and_signs_ $scalar>]() {
                    let a: [$scalar; 4] = $a;
                    let b: [$scalar; 4] = $b;
                    let expected: [$scalar; 4] = $x;
                    for trana in [
                        TrsylTranspose::NoTranspose,
                        TrsylTranspose::ConjugateTranspose,
                    ] {
                        for tranb in [
                            TrsylTranspose::NoTranspose,
                            TrsylTranspose::ConjugateTranspose,
                        ] {
                            for sign in [TrsylSign::Plus, TrsylSign::Minus] {
                                let op_entry = |
                                    matrix: &[$scalar; 4],
                                    row: usize,
                                    col: usize,
                                    trans: TrsylTranspose,
                                | match trans {
                                    TrsylTranspose::NoTranspose => matrix[row + 2 * col],
                                    TrsylTranspose::ConjugateTranspose => {
                                        <$scalar as RlstScalar>::conj(&matrix[col + 2 * row])
                                    }
                                };
                                let mut c = [<$scalar as num::Zero>::zero(); 4];
                                for col in 0..2 {
                                    for row in 0..2 {
                                        for inner in 0..2 {
                                            c[row + 2 * col] += op_entry(&a, row, inner, trana)
                                                * expected[inner + 2 * col];
                                            let xb = expected[row + 2 * inner]
                                                * op_entry(&b, inner, col, tranb);
                                            match sign {
                                                TrsylSign::Plus => c[row + 2 * col] += xb,
                                                TrsylSign::Minus => c[row + 2 * col] -= xb,
                                            }
                                        }
                                    }
                                }
                                let outcome = <$scalar>::trsyl(
                                    trana, tranb, sign, 2, 2, &a, 2, &b, 2, &mut c, 2,
                                )
                                .unwrap();
                                assert_eq!(outcome.status, TrsylStatus::Success);
                                assert_eq!(outcome.scale, 1.0);
                                for index in 0..4 {
                                    assert!((c[index] - expected[index]).abs() < $tol);
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    operation_tests!(
        f32,
        2e-5,
        [4.0, 0.0, -0.75, 6.0],
        [0.5, 0.0, 0.25, 1.5],
        [1.0, -2.0, 0.5, 3.0]
    );
    operation_tests!(
        f64,
        1e-13,
        [4.0, 0.0, -0.75, 6.0],
        [0.5, 0.0, 0.25, 1.5],
        [1.0, -2.0, 0.5, 3.0]
    );
    operation_tests!(
        c32,
        3e-5,
        [
            c32::new(4.0, 0.5),
            c32::new(0.0, 0.0),
            c32::new(-0.75, 0.25),
            c32::new(6.0, -0.5),
        ],
        [
            c32::new(0.5, 0.25),
            c32::new(0.0, 0.0),
            c32::new(0.25, -0.5),
            c32::new(1.5, 0.75),
        ],
        [
            c32::new(1.0, 0.5),
            c32::new(-2.0, 0.25),
            c32::new(0.5, -1.0),
            c32::new(3.0, 0.75),
        ]
    );
    operation_tests!(
        c64,
        2e-13,
        [
            c64::new(4.0, 0.5),
            c64::new(0.0, 0.0),
            c64::new(-0.75, 0.25),
            c64::new(6.0, -0.5),
        ],
        [
            c64::new(0.5, 0.25),
            c64::new(0.0, 0.0),
            c64::new(0.25, -0.5),
            c64::new(1.5, 0.75),
        ],
        [
            c64::new(1.0, 0.5),
            c64::new(-2.0, 0.25),
            c64::new(0.5, -1.0),
            c64::new(3.0, 0.75),
        ]
    );
}
