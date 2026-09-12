//! LQ decomposition and full-row-rank minimum-norm solves.

use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::{RlstError, RlstResult};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::trsm::{TrsmDiag, TrsmSide, TrsmTransA, TrsmUplo};
use crate::traits::base_operations::Shape;
use crate::traits::linalg::{decompositions::Lq, lapack::Lapack};

pub use super::qr::QMode;
pub use crate::dense::linalg::lapack::interface::mqr::{MqrSide, MqrTransMode};

/// Stores the packed reflectors and scalar factors of an LQ decomposition.
///
/// For an `m` by `n` input and `k = min(m, n)`, the decomposition satisfies
/// `A = L Q`. [`Self::l_mat`] has shape `m` by `k`. Compact Q has shape
/// `k` by `n`, while full Q has shape `n` by `n`. Thus compact factors
/// reconstruct A directly; with full Q, use its first `k` rows or pad L with
/// zero columns to shape `m` by `n`.
pub struct LqDecomposition<Item> {
    a: DynArray<Item, 2>,
    tau: Vec<Item>,
}

impl<Item, ArrayImpl> Lq for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;

    fn lq(&self) -> RlstResult<LqDecomposition<Item>> {
        let mut a = DynArray::new_from(self);
        let [m, n] = a.shape();
        let mut tau = vec![Item::zero(); m.min(n)];
        if m != 0 && n != 0 {
            Item::gelqf(m, n, a.data_mut().unwrap(), m, &mut tau)?;
        }
        Ok(LqDecomposition { a, tau })
    }
}

impl<Item> LqDecomposition<Item>
where
    Item: Lapack,
{
    /// Form Q explicitly.
    ///
    /// Compact Q has shape `min(m, n)` by `n`; full Q has shape `n` by `n`.
    /// For an empty reflector set, full Q is the identity matrix.
    pub fn q_mat(&self, mode: QMode) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.a.shape();
        let k = m.min(n);
        let q_rows = match mode {
            QMode::Compact => k,
            QMode::Full => n,
        };
        let mut q = DynArray::<Item, 2>::from_shape([q_rows, n]);
        for row in 0..k {
            for col in row + 1..n {
                q[[row, col]] = self.a[[row, col]];
            }
        }
        if k == 0 {
            if matches!(mode, QMode::Full) {
                q.set_identity();
            }
            return Ok(q);
        }
        Item::orglq(
            q_rows,
            n,
            k,
            q.data_mut().unwrap(),
            q_rows.max(1),
            &self.tau,
        )?;
        Ok(q)
    }

    /// Extract the `m` by `min(m, n)` lower trapezoidal factor L.
    pub fn l_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.a.shape();
        let k = m.min(n);
        let mut l = DynArray::<Item, 2>::from_shape([m, k]);
        for col in 0..k {
            for row in col..m {
                l[[row, col]] = self.a[[row, col]];
            }
        }
        Ok(l)
    }

    /// Apply full Q or Q^H implicitly to a two-dimensional input.
    ///
    /// [`MqrSide::Left`] computes `Q rhs` and
    /// [`MqrSide::Right`] computes `rhs Q`; selecting
    /// [`MqrTransMode::ConjugateTranspose`] substitutes `Q^H` (which is `Q^T`
    /// for real scalars). The affected input dimension must equal the number
    /// of columns of the factorized matrix. The input and packed factorization
    /// are not modified.
    pub fn apply_q<ArrayImpl>(
        &self,
        side: MqrSide,
        trans: MqrTransMode,
        rhs: &Array<ArrayImpl, 2>,
    ) -> RlstResult<DynArray<Item, 2>>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    {
        let [rows, cols] = rhs.shape();
        let n = self.a.shape()[1];
        let affected = match side {
            MqrSide::Left => rows,
            MqrSide::Right => cols,
        };
        if affected != n {
            return Err(RlstError::GeneralError(format!(
                "LQ Q application requires affected dimension {n}, got {affected}."
            )));
        }
        let mut result = DynArray::new_from(rhs);
        if rows == 0 || cols == 0 || self.tau.is_empty() {
            return Ok(result);
        }
        Item::mlq(
            side,
            trans,
            rows,
            cols,
            self.tau.len(),
            self.a.data().unwrap(),
            self.a.shape()[0].max(1),
            &self.tau,
            result.data_mut().unwrap(),
            rows.max(1),
        )?;
        Ok(result)
    }

    /// Solve `A X = B` for the minimum Euclidean/Frobenius-norm solution.
    ///
    /// The factorized matrix must satisfy `m <= n` and have full row rank.
    /// A one-dimensional right-hand side has length `m` and returns length
    /// `n`; a two-dimensional right-hand side has shape `m` by `nrhs` and
    /// returns shape `n` by `nrhs`. The result has minimum Euclidean norm for
    /// a vector and minimum Frobenius norm for a matrix.
    ///
    /// This factorization is not rank revealing: near-rank-deficient inputs
    /// have no reliability guarantee. Only exact zero diagonal entries of the
    /// triangular factor are rejected.
    pub fn solve<ArrayImpl, const NDIM: usize>(
        &self,
        rhs: &Array<ArrayImpl, NDIM>,
    ) -> RlstResult<DynArray<Item, NDIM>>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        if NDIM != 1 && NDIM != 2 {
            return Err(RlstError::GeneralError(
                "LQ solve is only implemented for 1D and 2D arrays.".to_string(),
            ));
        }
        let [m, n] = self.a.shape();
        if m > n {
            return Err(RlstError::GeneralError(
                "LQ minimum-norm solve requires m <= n.".to_string(),
            ));
        }
        if rhs.shape()[0] != m {
            return Err(RlstError::GeneralError(format!(
                "LQ solve requires {m} right-hand-side rows, got {}.",
                rhs.shape()[0]
            )));
        }
        for diagonal in 0..m {
            if self.a[[diagonal, diagonal]] == Item::zero() {
                return Err(RlstError::GeneralError(
                    "LQ minimum-norm solve requires full row rank.".to_string(),
                ));
            }
        }

        let nrhs = if NDIM == 1 { 1 } else { rhs.shape()[1] };
        let mut shape = rhs.shape();
        shape[0] = n;
        let mut solution = DynArray::<Item, NDIM>::from_shape(shape);
        let rhs_copy = DynArray::new_from(rhs);
        for col in 0..nrhs {
            for row in 0..m {
                solution.data_mut().unwrap()[row + n * col] =
                    rhs_copy.data().unwrap()[row + m * col];
            }
        }
        if m == 0 || nrhs == 0 {
            return Ok(solution);
        }

        let mut l11 = DynArray::<Item, 2>::from_shape([m, m]);
        for col in 0..m {
            for row in col..m {
                l11[[row, col]] = self.a[[row, col]];
            }
        }
        Item::trsm(
            TrsmSide::Left,
            TrsmUplo::Lower,
            TrsmTransA::NoTrans,
            TrsmDiag::NonUnit,
            m,
            nrhs,
            Item::one(),
            l11.data_mut().unwrap(),
            m,
            solution.data_mut().unwrap(),
            n,
        )?;
        Item::mlq(
            MqrSide::Left,
            MqrTransMode::ConjugateTranspose,
            n,
            nrhs,
            m,
            self.a.data().unwrap(),
            m.max(1),
            &self.tau,
            solution.data_mut().unwrap(),
            n,
        )?;
        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::{MqrSide, MqrTransMode, QMode};
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::{Lq, MultIntoResize, RlstScalar, empty_array};
    use paste::paste;

    macro_rules! lq_scalar_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<lq_reconstruct_apply_and_solve_ $scalar>]() {
                    let mut a = DynArray::<$scalar, 2>::from_shape([2, 3]);
                    a.fill_from_seed_equally_distributed(17);
                    a[[0, 0]] += <$scalar>::from_real(3.0);
                    a[[1, 1]] += <$scalar>::from_real(3.0);
                    let lq = a.lq().unwrap();
                    let l = lq.l_mat().unwrap();
                    let qc = lq.q_mat(QMode::Compact).unwrap();
                    let reconstructed = empty_array().simple_mult_into_resize(l.r(), qc.r());
                    crate::assert_array_relative_eq!(reconstructed, a, $tol);
                    let qch = DynArray::new_from(&qc.r().conj().transpose());
                    let qcqch = empty_array().simple_mult_into_resize(qc.r(), qch.r());
                    let mut compact_identity = DynArray::<$scalar, 2>::from_shape([2, 2]);
                    compact_identity.set_identity();
                    crate::assert_array_relative_eq!(qcqch, compact_identity, $tol);

                    for shape in [[2, 2], [3, 2]] {
                        let mut other = DynArray::<$scalar, 2>::from_shape(shape);
                        other.fill_from_seed_equally_distributed(31);
                        other[[0, 0]] += <$scalar>::from_real(3.0);
                        other[[1, 1]] += <$scalar>::from_real(3.0);
                        let other_lq = other.lq().unwrap();
                        let other_reconstructed = empty_array().simple_mult_into_resize(
                            other_lq.l_mat().unwrap().r(),
                            other_lq.q_mat(QMode::Compact).unwrap().r(),
                        );
                        crate::assert_array_relative_eq!(other_reconstructed, other, $tol);
                    }

                    let q = lq.q_mat(QMode::Full).unwrap();
                    let qh = DynArray::new_from(&q.r().conj().transpose());
                    let mut identity = DynArray::<$scalar, 2>::from_shape([3, 3]);
                    identity.set_identity();
                    let qqh = empty_array().simple_mult_into_resize(q.r(), qh.r());
                    crate::assert_array_relative_eq!(qqh, identity, $tol);

                    let mut left_rhs = DynArray::<$scalar, 2>::from_shape([3, 2]);
                    left_rhs.fill_from_seed_equally_distributed(41);
                    let left = lq.apply_q(MqrSide::Left, MqrTransMode::NoTranspose, &left_rhs).unwrap();
                    let explicit_left = empty_array().simple_mult_into_resize(q.r(), left_rhs.r());
                    crate::assert_array_relative_eq!(left, explicit_left, $tol);
                    let left_adjoint = lq.apply_q(MqrSide::Left, MqrTransMode::ConjugateTranspose, &left_rhs).unwrap();
                    let explicit_left_adjoint = empty_array().simple_mult_into_resize(qh.r(), left_rhs.r());
                    crate::assert_array_relative_eq!(left_adjoint, explicit_left_adjoint, $tol);
                    let mut right_rhs = DynArray::<$scalar, 2>::from_shape([2, 3]);
                    right_rhs.fill_from_seed_equally_distributed(43);
                    let right = lq.apply_q(MqrSide::Right, MqrTransMode::ConjugateTranspose, &right_rhs).unwrap();
                    let explicit_right = empty_array().simple_mult_into_resize(right_rhs.r(), qh.r());
                    crate::assert_array_relative_eq!(right, explicit_right, $tol);
                    let right_plain = lq.apply_q(MqrSide::Right, MqrTransMode::NoTranspose, &right_rhs).unwrap();
                    let explicit_right_plain = empty_array().simple_mult_into_resize(right_rhs.r(), q.r());
                    crate::assert_array_relative_eq!(right_plain, explicit_right_plain, $tol);
                    let q_again = lq.q_mat(QMode::Full).unwrap();
                    crate::assert_array_relative_eq!(q_again, q, $tol);

                    let mut expected = DynArray::<$scalar, 2>::from_shape([3, 2]);
                    expected.fill_from_seed_equally_distributed(23);
                    let rhs = empty_array().simple_mult_into_resize(a.r(), expected.r());
                    let x = lq.solve(&rhs).unwrap();
                    let residual = empty_array().simple_mult_into_resize(a.r(), x.r());
                    crate::assert_array_relative_eq!(residual, rhs, $tol);
                    let qx = empty_array().simple_mult_into_resize(q.r(), x.r());
                    for col in 0..2 {
                        assert!(qx[[2, col]].abs() < $tol * 10.0);
                    }

                    let mut rhs_vector = DynArray::<$scalar, 1>::from_shape([2]);
                    rhs_vector[[0]] = rhs[[0, 0]];
                    rhs_vector[[1]] = rhs[[1, 0]];
                    let vector_x = lq.solve(&rhs_vector).unwrap();
                    assert_eq!(vector_x.shape(), [3]);
                    for row in 0..3 {
                        assert!((vector_x[[row]] - x[[row, 0]]).abs() < $tol);
                    }
                }
            }
        };
    }

    lq_scalar_tests!(f32, 2e-4);
    lq_scalar_tests!(f64, 2e-12);
    lq_scalar_tests!(c32, 3e-4);
    lq_scalar_tests!(c64, 3e-12);

    #[test]
    fn lq_shapes_views_errors_and_empty_contracts() {
        let mut source = DynArray::<f64, 2>::from_shape([3, 2]);
        source.fill_from_seed_equally_distributed(5);
        let view = source.transpose();
        let lq = view.lq().unwrap();
        assert_eq!(lq.l_mat().unwrap().shape(), [2, 2]);
        assert_eq!(lq.q_mat(QMode::Compact).unwrap().shape(), [2, 3]);
        assert_eq!(lq.q_mat(QMode::Full).unwrap().shape(), [3, 3]);
        assert!(
            lq.apply_q(
                MqrSide::Left,
                MqrTransMode::NoTranspose,
                &DynArray::from_shape([2, 1])
            )
            .is_err()
        );
        assert!(lq.solve(&DynArray::<f64, 1>::from_shape([3])).is_err());
        let rhs_storage = DynArray::<f64, 2>::from_shape([1, 2]);
        assert_eq!(lq.solve(&rhs_storage.transpose()).unwrap().shape(), [3, 1]);
        let apply_storage = DynArray::<f64, 2>::from_shape([1, 3]);
        assert_eq!(
            lq.apply_q(
                MqrSide::Left,
                MqrTransMode::NoTranspose,
                &apply_storage.transpose(),
            )
            .unwrap()
            .shape(),
            [3, 1]
        );

        let tall = DynArray::<f64, 2>::from_shape([3, 2]).lq().unwrap();
        assert!(tall.solve(&DynArray::<f64, 1>::from_shape([3])).is_err());
        let singular = DynArray::<f64, 2>::from_shape([2, 3]).lq().unwrap();
        assert!(
            singular
                .solve(&DynArray::<f64, 1>::from_shape([2]))
                .is_err()
        );

        let empty = DynArray::<f64, 2>::from_shape([0, 3]).lq().unwrap();
        let q = empty.q_mat(QMode::Full).unwrap();
        let mut identity = DynArray::<f64, 2>::from_shape([3, 3]);
        identity.set_identity();
        crate::assert_array_abs_diff_eq!(q, identity, 0.0);
        let rhs = DynArray::<f64, 2>::from_shape([0, 2]);
        let x = empty.solve(&rhs).unwrap();
        assert_eq!(x.shape(), [3, 2]);
        assert!(x.iter_value().all(|value| value == 0.0));
        let zero_columns = lq.solve(&DynArray::<f64, 2>::from_shape([2, 0])).unwrap();
        assert_eq!(zero_columns.shape(), [3, 0]);
        let applied = lq
            .apply_q(
                MqrSide::Left,
                MqrTransMode::NoTranspose,
                &DynArray::<f64, 2>::from_shape([3, 0]),
            )
            .unwrap();
        assert_eq!(applied.shape(), [3, 0]);
        assert!(
            lq.solve(&DynArray::<f64, 3>::from_shape([2, 1, 1]))
                .is_err()
        );
    }
}
