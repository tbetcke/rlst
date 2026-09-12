//! Dense solver for ordinary Sylvester equations.
//!
//! The returned matrix is deliberately scaled: it satisfies
//! `A X +/- X B = scale C` for an unperturbed solve. Keeping LAPACK's scale
//! avoids an automatic division that could overflow.

use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::{RlstError, RlstResult};
use crate::dense::array::{Array, DynArray, empty_array};
use crate::dense::linalg::lapack::interface::trsyl::{TrsylOutcome, TrsylTranspose};
pub use crate::dense::linalg::lapack::interface::trsyl::{TrsylSign, TrsylStatus};
use crate::traits::base_operations::{EvaluateArray, Shape};
use crate::traits::linalg::base::{Gemm, MultIntoResize};
use crate::traits::linalg::decompositions::{EigenvalueDecomposition, SylvesterSolve};
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;
use num::One;

/// Solution and LAPACK metadata for an ordinary Sylvester equation.
///
/// When [`Self::status`] is [`TrsylStatus::Success`], the matrix satisfies
/// `A X +/- X B = scale C` up to rounding error. A
/// [`TrsylStatus::Perturbed`] result instead corresponds to a nearby problem
/// in which close or common eigenvalues were perturbed.
pub struct SylvesterSolution<Item: RlstScalar> {
    x: DynArray<Item, 2>,
    scale: Item::Real,
    status: TrsylStatus,
}

impl<Item: RlstScalar> SylvesterSolution<Item> {
    /// Return the computed, possibly scaled solution matrix.
    ///
    /// Divide this matrix by [`Self::scale`] only when that operation is known
    /// to be safe from overflow.
    pub fn x(&self) -> &DynArray<Item, 2> {
        &self.x
    }

    /// Return the scale in `A X +/- X B = scale C`.
    ///
    /// The result is deliberately not divided by this value because doing so
    /// can overflow. Callers may unscale when they know that it is safe.
    pub fn scale(&self) -> Item::Real {
        self.scale
    }

    /// Return whether LAPACK perturbed close or common eigenvalues.
    ///
    /// A perturbed result solves a nearby problem rather than the exact input
    /// equation.
    pub fn status(&self) -> TrsylStatus {
        self.status
    }
}

impl<Item, AImpl, BImpl, CImpl> SylvesterSolve<Array<BImpl, 2>, Array<CImpl, 2>> for Array<AImpl, 2>
where
    Item: Lapack + Gemm,
    AImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    BImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    CImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;

    fn solve_sylvester(
        &self,
        b: &Array<BImpl, 2>,
        c: &Array<CImpl, 2>,
        sign: TrsylSign,
    ) -> RlstResult<SylvesterSolution<Item>> {
        let [m, ma] = self.shape();
        let [n, nb] = b.shape();
        if m != ma {
            return Err(RlstError::MatrixNotSquare(m, ma));
        }
        if n != nb {
            return Err(RlstError::MatrixNotSquare(n, nb));
        }
        if c.shape() != [m, n] {
            return Err(RlstError::GeneralError(format!(
                "Sylvester right-hand side must have shape [{m}, {n}], got {:?}",
                c.shape()
            )));
        }
        if m == 0 || n == 0 {
            return Ok(SylvesterSolution {
                x: DynArray::new_from(c),
                scale: Item::Real::one(),
                status: TrsylStatus::Success,
            });
        }

        let (r, u) = self.schur()?;
        let (s, v) = b.schur()?;
        let c = DynArray::new_from(c);
        let uh_c = empty_array().simple_mult_into_resize(u.r().conj().transpose().eval(), c.r());
        let mut y = empty_array().simple_mult_into_resize(uh_c.r(), v.r());
        let TrsylOutcome { scale, status } = Item::trsyl(
            TrsylTranspose::NoTranspose,
            TrsylTranspose::NoTranspose,
            sign,
            m,
            n,
            r.data().unwrap(),
            m,
            s.data().unwrap(),
            n,
            y.data_mut().unwrap(),
            m,
        )?;
        let u_y = empty_array().simple_mult_into_resize(u.r(), y.r());
        let x = empty_array().simple_mult_into_resize(u_y.r(), v.r().conj().transpose().eval());
        Ok(SylvesterSolution { x, scale, status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dot;
    use paste::paste;

    macro_rules! sylvester_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<sylvester_both_signs_rectangular_ $scalar>]() {
                    let (m, n) = (3, 2);
                    let mut a = DynArray::<$scalar, 2>::from_shape([m, m]);
                    let mut b = DynArray::<$scalar, 2>::from_shape([n, n]);
                    let mut expected = DynArray::<$scalar, 2>::from_shape([m, n]);
                    a.fill_from_seed_normally_distributed(31);
                    b.fill_from_seed_normally_distributed(32);
                    expected.fill_from_seed_normally_distributed(33);
                    for i in 0..m { a[[i, i]] += <$scalar as num::One>::one() * <$scalar as num::FromPrimitive>::from_f64(5.0).unwrap(); }
                    for sign in [TrsylSign::Plus, TrsylSign::Minus] {
                        let ax = dot!(a.r(), expected.r());
                        let xb = dot!(expected.r(), b.r());
                        let mut c = ax;
                        for j in 0..n {
                            for i in 0..m {
                                match sign {
                                    TrsylSign::Plus => c[[i, j]] += xb[[i, j]],
                                    TrsylSign::Minus => c[[i, j]] -= xb[[i, j]],
                                }
                            }
                        }
                        let solution = a.solve_sylvester(&b, &c, sign).unwrap();
                        assert_eq!(solution.status(), TrsylStatus::Success);
                        crate::assert_array_abs_diff_eq!(solution.x(), expected, $tol);
                    }
                }
            }
        };
    }

    sylvester_tests!(f32, 3e-4);
    sylvester_tests!(f64, 1e-11);
    sylvester_tests!(c32, 5e-4);
    sylvester_tests!(c64, 1e-11);

    #[test]
    fn sylvester_real_block_views_empty_and_perturbation() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 2]);
        a[[1, 0]] = 1.0;
        a[[0, 1]] = -1.0;
        let mut b = DynArray::<f64, 2>::from_shape([1, 1]);
        b[[0, 0]] = 3.0;
        let mut c = DynArray::<f64, 2>::from_shape([1, 2]);
        c[[0, 0]] = 2.0;
        c[[0, 1]] = -1.0;
        let solution = a
            .r()
            .transpose()
            .solve_sylvester(&b, &c.r().transpose(), TrsylSign::Plus)
            .unwrap();
        let lhs = dot!(a.r().transpose().eval(), solution.x().r());
        let rhs_product = dot!(solution.x().r(), b.r());
        for i in 0..2 {
            assert!(
                (lhs[[i, 0]] + rhs_product[[i, 0]] - solution.scale() * c[[0, i]]).abs() < 1e-12
            );
        }

        let empty_a = DynArray::<f64, 2>::from_shape([0, 0]);
        let empty_c = DynArray::<f64, 2>::from_shape([0, 1]);
        let result = empty_a
            .solve_sylvester(&b, &empty_c, TrsylSign::Plus)
            .unwrap();
        assert_eq!(result.scale(), 1.0);
        assert_eq!(result.status(), TrsylStatus::Success);
        let empty_b = DynArray::<f64, 2>::from_shape([0, 0]);
        let zero_columns = DynArray::<f64, 2>::from_shape([2, 0]);
        let result = a
            .solve_sylvester(&empty_b, &zero_columns, TrsylSign::Plus)
            .unwrap();
        assert_eq!(result.x().shape(), [2, 0]);

        let mut one = DynArray::<f64, 2>::from_shape([1, 1]);
        one[[0, 0]] = 1.0;
        let result = one.solve_sylvester(&one, &one, TrsylSign::Minus).unwrap();
        assert_eq!(result.status(), TrsylStatus::Perturbed);
    }

    #[test]
    fn sylvester_reports_lapack_scaling() {
        let mut a = DynArray::<f64, 2>::from_shape([1, 1]);
        let mut b = DynArray::<f64, 2>::from_shape([1, 1]);
        let mut c = DynArray::<f64, 2>::from_shape([1, 1]);
        a[[0, 0]] = 0.1;
        b[[0, 0]] = 0.1;
        c[[0, 0]] = 1e308;
        let result = a.solve_sylvester(&b, &c, TrsylSign::Plus).unwrap();
        assert_eq!(result.status(), TrsylStatus::Success);
        assert!(result.scale() < 1.0);
        assert!(result.x()[[0, 0]].is_finite());
        let ax = a[[0, 0]] * result.x()[[0, 0]];
        let xb = result.x()[[0, 0]] * b[[0, 0]];
        let scaled_c = result.scale() * c[[0, 0]];
        let normalized_residual = (ax + xb - scaled_c).abs()
            / (ax.abs() + xb.abs() + scaled_c.abs()).max(f64::MIN_POSITIVE);
        assert!(normalized_residual < 1e-14);
    }

    #[test]
    fn sylvester_rejects_shapes() {
        let a = DynArray::<f64, 2>::from_shape([2, 3]);
        let b = DynArray::<f64, 2>::from_shape([1, 1]);
        let c = DynArray::<f64, 2>::from_shape([2, 1]);
        assert!(a.solve_sylvester(&b, &c, TrsylSign::Plus).is_err());
        let a = DynArray::<f64, 2>::from_shape([2, 2]);
        let c = DynArray::<f64, 2>::from_shape([1, 2]);
        assert!(a.solve_sylvester(&b, &c, TrsylSign::Plus).is_err());
    }
}
