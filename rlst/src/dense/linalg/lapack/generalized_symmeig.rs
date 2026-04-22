//! Implementation of Generalized Symmetric/Hermitian Eigenvalue Decomposition.

use crate::base_types::{RlstResult, UpLo};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::sygv::{JobZ, SygvUplo};
use crate::dense::linalg::lapack::symmeig::SymmEigMode;
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::{GeneralizedSymmEig, GeneralizedSymmEigType};
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;
use crate::UnsafeRandom1DAccessByValue;

impl<Item, ArrayImpl, BArrayImpl> GeneralizedSymmEig<Array<BArrayImpl, 2>> for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    BArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
{
    type Item = Item;

    fn eigh_generalized(
        &self,
        b: &Array<BArrayImpl, 2>,
        uplo: UpLo,
        mode: SymmEigMode,
        itype: GeneralizedSymmEigType,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        Option<DynArray<Self::Item, 2>>,
    )> {
        let n = self.shape()[0];
        assert_eq!(n, self.shape()[1], "Matrix A must be square.");
        assert_eq!(n, b.shape()[0], "Matrix B must have the same size as A.");
        assert_eq!(n, b.shape()[1], "Matrix B must be square.");

        let mut a_mat = DynArray::new_from(self);
        let mut b_mat = DynArray::new_from(b);
        let mut w = DynArray::<<Self::Item as RlstScalar>::Real, 1>::from_shape([n]);

        let jobz = match mode {
            SymmEigMode::EigenvaluesOnly => JobZ::None,
            SymmEigMode::EigenvaluesAndEigenvectors => JobZ::ValuesAndVectors,
        };

        let sygv_uplo = match uplo {
            UpLo::Upper => SygvUplo::Upper,
            UpLo::Lower => SygvUplo::Lower,
        };

        Item::sygv(
            itype as i32,
            jobz,
            sygv_uplo,
            n,
            a_mat.data_mut().unwrap(),
            n,
            b_mat.data_mut().unwrap(),
            n,
            w.data_mut().unwrap(),
        )?;

        match mode {
            SymmEigMode::EigenvaluesOnly => Ok((w, None)),
            SymmEigMode::EigenvaluesAndEigenvectors => Ok((w, Some(a_mat))),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dot;
    use crate::traits::base_operations::EvaluateArray;
    use crate::traits::rlst_num::RlstScalar;
    use num::FromPrimitive;
    use paste::paste;

    macro_rules! implement_gen_symm_eig_test {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_gen_symm_eig_$scalar>]() {
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                let mut b = DynArray::<$scalar, 2>::from_shape([n, n]);

                a.fill_from_seed_normally_distributed(0);
                b.fill_from_seed_normally_distributed(1);

                // Make A symmetric/Hermitian
                let a = DynArray::new_from(&(a.r() + a.r().conj().transpose()));

                // Make B symmetric/Hermitian positive definite
                let b = DynArray::new_from(&(b.r() + b.r().conj().transpose()));
                // Add identity to B to ensure positive definiteness
                let mut ident = DynArray::<$scalar, 2>::from_shape([n, n]);
                ident.set_identity();
                let scalar_val = <$scalar as RlstScalar>::from_real(<$scalar as RlstScalar>::Real::from_f64(10.0 * n as f64).unwrap());
                let b = DynArray::new_from(&(b.r() + scalar_val * ident.r()));

                let (w, v) = a.eigh_generalized(&b, UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors, GeneralizedSymmEigType::AxEqualLambdaBx).unwrap();
                let v = v.unwrap();

                // Check Ax = lambda Bx
                for i in 0..n {
                    let lambda = w[[i]];
                    let vi = v.r().into_subview([0, i], [n, 1]).eval();

                    let lhs = dot!(a.r(), vi.r());
                    let rhs = dot!(b.r(), (vi.r() * <$scalar as RlstScalar>::from_real(lambda)).eval());

                    crate::assert_array_relative_eq!(lhs, rhs, $tol);
                }
            }
            }
        };
    }

    implement_gen_symm_eig_test!(f32, 1E-4);
    implement_gen_symm_eig_test!(f64, 1E-10);
    implement_gen_symm_eig_test!(c32, 1E-4);
    implement_gen_symm_eig_test!(c64, 1E-10);
}
