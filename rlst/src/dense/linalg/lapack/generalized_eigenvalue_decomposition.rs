//! Implementation of Generalized Eigenvalue Decomposition for general matrices.

use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use crate::dense::linalg::lapack::interface::ggev::Job;
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::GeneralizedEigenvalueDecomposition;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;
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

        let (jobvl, jobvr, mut vl, mut vr) = match mode {
            EigMode::EigenvaluesOnly => (Job::None, Job::None, None, None),
            EigMode::RightEigenvectors => (
                Job::None,
                Job::Compute,
                None,
                Some(DynArray::<<Item as RlstScalar>::Complex, 2>::from_shape([
                    n, n,
                ])),
            ),
            EigMode::LeftEigenvectors => (
                Job::Compute,
                Job::None,
                Some(DynArray::<<Item as RlstScalar>::Complex, 2>::from_shape([
                    n, n,
                ])),
                None,
            ),
            EigMode::BothEigenvectors => (
                Job::Compute,
                Job::Compute,
                Some(DynArray::<<Item as RlstScalar>::Complex, 2>::from_shape([
                    n, n,
                ])),
                Some(DynArray::<<Item as RlstScalar>::Complex, 2>::from_shape([
                    n, n,
                ])),
            ),
        };

        let mut eigenvalues = DynArray::<<Self::Item as RlstScalar>::Complex, 1>::from_shape([n]);

        let mut alpha = vec![<Item as RlstScalar>::Complex::zero(); n];
        let mut beta = vec![<Item as RlstScalar>::Complex::zero(); n];

        let mut vl_temp = DynArray::<<Item as RlstScalar>::Complex, 2>::from_shape([0, 0]);
        let mut vr_temp = DynArray::<<Item as RlstScalar>::Complex, 2>::from_shape([0, 0]);

        Item::ggev(
            jobvl,
            jobvr,
            n,
            a_mat.data_mut().unwrap(),
            n,
            b_mat.data_mut().unwrap(),
            n,
            &mut alpha,
            &mut beta,
            vl.as_mut()
                .map(|v| v.data_mut().unwrap())
                .unwrap_or(vl_temp.data_mut().unwrap()),
            n,
            vr.as_mut()
                .map(|v| v.data_mut().unwrap())
                .unwrap_or(vr_temp.data_mut().unwrap()),
            n,
        )?;

        for i in 0..n {
            eigenvalues[[i]] = alpha[i] / beta[i];
        }

        Ok((eigenvalues, vr, vl))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::traits::base_operations::*;
    use crate::traits::rlst_num::RlstScalar;
    use itertools::izip;
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

                let (lam2, vr, vl) = a.eig_generalized(&b, EigMode::BothEigenvectors).unwrap();

                crate::assert_array_relative_eq!(lam, lam2, $tol);

                let a_complex = a.r().into_type::<<$scalar as RlstScalar>::Complex>().eval();
                let b_complex = b.r().into_type::<<$scalar as RlstScalar>::Complex>().eval();

                let mut diag = DynArray::from_shape([n, n]);
                izip!(diag.diag_iter_mut(), lam2.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = w_elem;
                });

                // Check right eigenvectors: A * vr = B * vr * diag
                let vr = vr.unwrap();
                let lhs = dot!(a_complex.r(), vr.r());
                let rhs = dot!(b_complex.r(), dot!(vr.r(), diag.r()));
                crate::assert_array_abs_diff_eq!(lhs, rhs, $tol);

                // Check left eigenvectors: vl^H * A = diag * vl^H * B
                let vlh = vl.unwrap().conj().transpose().eval();
                let lhs = dot!(vlh.r(), a_complex.r());
                let rhs = dot!(diag.r(), dot!(vlh.r(), b_complex.r()));
                crate::assert_array_abs_diff_eq!(lhs, rhs, $tol);


            }
            }
        };
    }

    implement_gen_eig_test!(f32, 1E-3);
    implement_gen_eig_test!(f64, 1E-10);
    implement_gen_eig_test!(c32, 1E-3);
    implement_gen_eig_test!(c64, 1E-10);
}
