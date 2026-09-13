//! Generalized Schur (QZ) decomposition for dense matrix pencils.
//!
//! The unsorted decomposition preserves homogeneous generalized eigenvalues:
//! callers receive `alpha` and `beta` separately and must not assume that
//! `alpha / beta` is finite or defined.

use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::{RlstError, RlstResult};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::gees::JobVs;
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::GeneralizedSchur;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;

/// An unsorted generalized Schur decomposition.
///
/// The factors satisfy `A = Q S Z^H` and `B = Q T Z^H`, where Q and Z are
/// orthogonal for real inputs and unitary for complex inputs. For real inputs,
/// S is quasi-upper triangular and T is upper triangular; for complex inputs,
/// both are upper triangular. No ordering of the diagonal blocks is promised.
///
/// Generalized eigenvalues are retained in homogeneous form `(alpha, beta)`.
/// A zero `beta` can represent an infinite eigenvalue, while `alpha = beta = 0`
/// is indeterminate. The decomposition therefore never divides by `beta`.
pub struct QzDecomposition<Item: RlstScalar> {
    s: DynArray<Item, 2>,
    t: DynArray<Item, 2>,
    q: DynArray<Item, 2>,
    z: DynArray<Item, 2>,
    alpha: DynArray<Item::Complex, 1>,
    beta: DynArray<Item::Complex, 1>,
}

impl<Item: RlstScalar> QzDecomposition<Item> {
    /// Return the generalized Schur form of A.
    pub fn s(&self) -> &DynArray<Item, 2> {
        &self.s
    }

    /// Return the generalized Schur form of B.
    pub fn t(&self) -> &DynArray<Item, 2> {
        &self.t
    }

    /// Return the left Schur vectors.
    pub fn q(&self) -> &DynArray<Item, 2> {
        &self.q
    }

    /// Return the right Schur vectors.
    pub fn z(&self) -> &DynArray<Item, 2> {
        &self.z
    }

    /// Return the homogeneous generalized-eigenvalue numerators.
    ///
    /// Entry `i` and [`Self::beta`] entry `i` jointly represent one
    /// generalized eigenvalue. Their ordering is unspecified.
    pub fn alpha(&self) -> &DynArray<Item::Complex, 1> {
        &self.alpha
    }

    /// Return the homogeneous generalized-eigenvalue denominators.
    ///
    /// A zero entry is valid and can represent an infinite eigenvalue; callers
    /// should check it before forming a quotient.
    pub fn beta(&self) -> &DynArray<Item::Complex, 1> {
        &self.beta
    }
}

impl<Item, ArrayImpl, BArrayImpl> GeneralizedSchur<Array<BArrayImpl, 2>> for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    BArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;

    fn qz(&self, b: &Array<BArrayImpl, 2>) -> RlstResult<QzDecomposition<Item>> {
        let [m, n] = self.shape();
        if m != n {
            return Err(RlstError::MatrixNotSquare(m, n));
        }
        if b.shape() != [n, n] {
            return Err(RlstError::GeneralError(format!(
                "QZ decomposition requires equal square shapes; got A={:?}, B={:?}",
                self.shape(),
                b.shape()
            )));
        }

        let mut s = DynArray::new_from(self);
        let mut t = DynArray::new_from(b);
        let mut q = DynArray::from_shape([n, n]);
        let mut z = DynArray::from_shape([n, n]);
        let mut alpha = DynArray::from_shape([n]);
        let mut beta = DynArray::from_shape([n]);

        Item::gges(
            JobVs::Compute,
            JobVs::Compute,
            n,
            s.data_mut().unwrap(),
            n.max(1),
            t.data_mut().unwrap(),
            n.max(1),
            alpha.data_mut().unwrap(),
            beta.data_mut().unwrap(),
            Some(q.data_mut().unwrap()),
            n.max(1),
            Some(z.data_mut().unwrap()),
            n.max(1),
        )?;

        Ok(QzDecomposition {
            s,
            t,
            q,
            z,
            alpha,
            beta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dot_blas;
    use crate::traits::base_operations::*;
    use paste::paste;

    macro_rules! qz_numerical_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<qz_reconstructs_and_is_unitary_ $scalar>]() {
                    let n = 5;
                    let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                    let mut b = DynArray::<$scalar, 2>::from_shape([n, n]);
                    a.fill_from_seed_normally_distributed(11);
                    b.fill_from_seed_normally_distributed(12);
                    let qz = a.qz(&b).unwrap();
                    let ah = dot_blas!(qz.q().r(), qz.s().r(), qz.z().r().conj().transpose().eval());
                    let bh = dot_blas!(qz.q().r(), qz.t().r(), qz.z().r().conj().transpose().eval());
                    crate::assert_array_abs_diff_eq!(ah, a, $tol);
                    crate::assert_array_abs_diff_eq!(bh, b, $tol);
                    let mut identity = DynArray::<$scalar, 2>::from_shape([n, n]);
                    identity.set_identity();
                    let qhq = dot_blas!(qz.q().r().conj().transpose().eval(), qz.q().r());
                    let zhz = dot_blas!(qz.z().r().conj().transpose().eval(), qz.z().r());
                    crate::assert_array_abs_diff_eq!(qhq, identity, $tol);
                    crate::assert_array_abs_diff_eq!(zhz, identity, $tol);
                }
            }
        };
    }

    qz_numerical_tests!(f32, 2e-4);
    qz_numerical_tests!(f64, 1e-11);
    qz_numerical_tests!(c32, 3e-4);
    qz_numerical_tests!(c64, 1e-11);

    #[test]
    fn qz_retains_homogeneous_values_and_accepts_views_and_empty() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 2]);
        a[[0, 0]] = 2.0;
        a[[1, 1]] = 3.0;
        let mut b = DynArray::<f64, 2>::from_shape([2, 2]);
        b[[1, 1]] = 4.0;
        let qz = a.r().transpose().qz(&b.r().transpose()).unwrap();
        assert!(qz.beta().iter_value().any(|value| value.norm() < 1e-14));
        let finite = qz
            .alpha()
            .iter_value()
            .zip(qz.beta().iter_value())
            .find(|(_, beta)| beta.norm() > 1e-14)
            .unwrap();
        assert!((finite.0.re / finite.1.re - 0.75).abs() < 1e-14);
        assert_eq!(qz.alpha().shape(), [2]);

        let empty = DynArray::<f64, 2>::from_shape([0, 0]);
        let qz = empty.qz(&empty).unwrap();
        assert_eq!(qz.q().shape(), [0, 0]);
    }

    #[test]
    fn qz_real_conjugate_pair_and_one_by_one() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 2]);
        a[[1, 0]] = 1.0;
        a[[0, 1]] = -1.0;
        let mut b = DynArray::<f64, 2>::from_shape([2, 2]);
        b.set_identity();
        let qz = a.qz(&b).unwrap();
        assert!(qz.alpha().iter_value().all(|value| value.im.abs() > 0.5));

        let mut a1 = DynArray::<f64, 2>::from_shape([1, 1]);
        let mut b1 = DynArray::<f64, 2>::from_shape([1, 1]);
        a1[[0, 0]] = 6.0;
        b1[[0, 0]] = 2.0;
        let qz = a1.qz(&b1).unwrap();
        let ratio = qz.alpha()[[0]].re / qz.beta()[[0]].re;
        assert!((ratio - 3.0).abs() < 1e-14);
    }

    #[test]
    fn qz_preserves_real_and_complex_schur_structure() {
        let mut ar = DynArray::<f64, 2>::from_shape([5, 5]);
        let mut br = DynArray::<f64, 2>::from_shape([5, 5]);
        ar.fill_from_seed_normally_distributed(21);
        br.fill_from_seed_normally_distributed(22);
        let real = ar.qz(&br).unwrap();
        for j in 0..5 {
            for i in j + 1..5 {
                assert!(real.t()[[i, j]].abs() < 1e-12);
                if i > j + 1 {
                    assert!(real.s()[[i, j]].abs() < 1e-12);
                }
            }
        }

        let mut ac = DynArray::<c64, 2>::from_shape([5, 5]);
        let mut bc = DynArray::<c64, 2>::from_shape([5, 5]);
        ac.fill_from_seed_normally_distributed(23);
        bc.fill_from_seed_normally_distributed(24);
        let complex = ac.qz(&bc).unwrap();
        for j in 0..5 {
            for i in j + 1..5 {
                assert!(complex.s()[[i, j]].norm() < 1e-12);
                assert!(complex.t()[[i, j]].norm() < 1e-12);
            }
        }
    }

    #[test]
    fn qz_rejects_invalid_shapes() {
        let a = DynArray::<f64, 2>::from_shape([2, 3]);
        let b = DynArray::<f64, 2>::from_shape([2, 2]);
        assert!(a.qz(&b).is_err());
        let a = DynArray::<f64, 2>::from_shape([2, 2]);
        let b = DynArray::<f64, 2>::from_shape([3, 3]);
        assert!(a.qz(&b).is_err());
    }
}
