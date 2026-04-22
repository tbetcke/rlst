//! Implementation of Generalized Eigenvalue Decomposition for general matrices.

use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use crate::dense::linalg::lapack::interface::ggev::Job;
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::GeneralizedEigenvalueDecomposition;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;
use crate::UnsafeRandom1DAccessByValue;
use num::Zero;

impl<Item, ArrayImpl, BArrayImpl> GeneralizedEigenvalueDecomposition<Array<BArrayImpl, 2>>
    for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    BArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
{
    type Item = Item;

    #[allow(clippy::type_complexity)]
    fn eig_generalized(
        &self,
        b: &Array<BArrayImpl, 2>,
        mode: EigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Complex, 1>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
    )> {
        let n = self.shape()[0];
        assert_eq!(n, self.shape()[1], "Matrix A must be square.");
        assert_eq!(n, b.shape()[0], "Matrix B must have the same size as A.");
        assert_eq!(n, b.shape()[1], "Matrix B must be square.");

        let mut a_mat = DynArray::new_from(self);
        let mut b_mat = DynArray::new_from(b);

        let jobv = match mode {
            EigMode::EigenvaluesOnly => Job::None,
            _ => Job::Compute,
        };

        let mut vl_mat = match mode {
            EigMode::EigenvaluesOnly => DynArray::<Item, 2>::from_shape([0, 0]),
            _ => DynArray::<Item, 2>::from_shape([n, n]),
        };

        let mut vr_mat = match mode {
            EigMode::EigenvaluesOnly => DynArray::<Item, 2>::from_shape([0, 0]),
            _ => DynArray::<Item, 2>::from_shape([n, n]),
        };

        let mut eigenvalues = DynArray::<<Self::Item as RlstScalar>::Complex, 1>::from_shape([n]);

        if !Item::IS_COMPLEX {
            let mut alphar = vec![<Item as RlstScalar>::Real::zero(); n];
            let mut alphai = vec![<Item as RlstScalar>::Real::zero(); n];
            let mut beta = vec![<Item as RlstScalar>::Real::zero(); n];

            Item::ggev(
                jobv,
                jobv,
                n,
                a_mat.data_mut().unwrap(),
                n,
                b_mat.data_mut().unwrap(),
                n,
                &mut alphar,
                &mut alphai,
                &mut beta,
                vl_mat.data_mut().unwrap(),
                n,
                vr_mat.data_mut().unwrap(),
                n,
            )?;

            for i in 0..n {
                eigenvalues[[i]] = Item::complex(alphar[i] / beta[i], alphai[i] / beta[i]);
            }
        } else {
            let mut alpha = vec![Item::zero(); n];
            let mut beta = vec![Item::zero(); n];

            Item::ggev_complex(
                jobv,
                jobv,
                n,
                a_mat.data_mut().unwrap(),
                n,
                b_mat.data_mut().unwrap(),
                n,
                &mut alpha,
                &mut beta,
                vl_mat.data_mut().unwrap(),
                n,
                vr_mat.data_mut().unwrap(),
                n,
            )?;

            for i in 0..n {
                eigenvalues[[i]] = alpha[i].add_complex(Item::Complex::zero())
                    / beta[i].add_complex(Item::Complex::zero());
            }
        }

        match mode {
            EigMode::EigenvaluesOnly => Ok((eigenvalues, None, None)),
            _ => {
                // For now, we only support eigenvalues.
                Ok((eigenvalues, None, None))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::base_types::{c32, c64};
    use crate::traits::rlst_num::RlstScalar;
    use num::FromPrimitive;
    use paste::paste;

    macro_rules! implement_gen_eig_test {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_gen_eig_$scalar>]() {
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                let mut b = DynArray::<$scalar, 2>::from_shape([n, n]);

                a.fill_from_seed_normally_distributed(0);
                b.fill_from_seed_normally_distributed(1);

                let lam = a.eigenvalues_generalized(&b).unwrap();

                // Check (A - lambda B)v = 0 is harder without eigenvectors.
                // For now, we just check that we get n eigenvalues.
                assert_eq!(lam.len(), n);

                // We can check one eigenvalue by computing det(A - lambda B)
                // but that is also complex.
                // A simpler check is for a diagonal case.
                let mut a_diag = DynArray::<$scalar, 2>::from_shape([n, n]);
                let mut b_diag = DynArray::<$scalar, 2>::from_shape([n, n]);
                for i in 0..n {
                    a_diag[[i, i]] = <$scalar as RlstScalar>::from_real(<$scalar as RlstScalar>::Real::from_f64(i as f64 + 1.0).unwrap());
                    b_diag[[i, i]] = <$scalar as RlstScalar>::from_real(<$scalar as RlstScalar>::Real::from_f64(2.0).unwrap());
                }

                let lam_diag = a_diag.eigenvalues_generalized(&b_diag).unwrap();
                for i in 0..n {
                    let expected: <$scalar as RlstScalar>::Real = num::cast((i as f64 + 1.0) / 2.0).unwrap();
                    // Eigenvalues are not necessarily in order
                    let found = lam_diag.iter_value().any(|l| (l.re() - expected).abs() < $tol);
                    assert!(found, "Eigenvalue {:?} not found", expected);
                }
            }
            }
        };
    }

    implement_gen_eig_test!(f32, 1E-4);
    implement_gen_eig_test!(f64, 1E-10);
    implement_gen_eig_test!(c32, 1E-4);
    implement_gen_eig_test!(c64, 1E-10);
}
