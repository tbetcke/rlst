//! Implementation of the pseudo-inverse.

use itertools::izip;

use crate::{
    Shape, UnsafeRandom1DAccessByValue,
    dense::array::{Array, DynArray},
    diag, dot,
    traits::{
        base_operations::EvaluateArray,
        linalg::{base::Gemm, lapack::Lapack},
        rlst_num::RlstScalar,
    },
};

/// An SVD-based (possibly truncated) pseudo-inverse.
///
/// It stores `V`, retained positive singular values, and `U^H`, so it can be
/// applied without first materializing the `n x m` matrix.
pub struct PInv<Item> {
    s: DynArray<Item, 1>,
    ut: DynArray<Item, 2>,
    v: DynArray<Item, 2>,
}

impl<Item: Lapack + Gemm> PInv<Item> {
    /// Create a pseudo-inverse representation from `s`, `U^H`, and V.
    ///
    /// `s` must contain positive retained singular values; `ut` has shape
    /// `s.len() x m` and `v` has shape `n x s.len()`. These invariants are not
    /// checked by the constructor and invalid dimensions will panic when used.
    pub fn new(s: DynArray<Item, 1>, ut: DynArray<Item, 2>, v: DynArray<Item, 2>) -> Self {
        Self { s, ut, v }
    }

    /// Get the singular values.
    pub fn s(&self) -> &DynArray<Item, 1> {
        &self.s
    }

    /// Return the stored `U^H` factor (`U^T` for real scalars).
    pub fn ut(&self) -> &DynArray<Item, 2> {
        &self.ut
    }

    /// Get the V matrix.
    pub fn v(&self) -> &DynArray<Item, 2> {
        &self.v
    }

    /// Materialize the `n x m` matrix `V diag(1/s) U^H`.
    pub fn as_matrix(&self) -> DynArray<Item, 2> {
        if self.s.is_empty() {
            return DynArray::from_shape([self.v.shape()[0], self.ut.shape()[1]]);
        }

        let sinv = self
            .s
            .r()
            .unary_op(|elem| <Item as RlstScalar>::recip(elem));

        dot!(self.v.r(), diag!(sinv), self.ut.r())
    }

    /// Apply the pseudo-inverse to a vector or matrix.
    ///
    /// The input must have first dimension `m`; output has first dimension
    /// `n`, preserving the number of right-hand-side columns.
    ///
    /// # Panics
    /// Panics for dimensions other than one or two, or incompatible shapes.
    pub fn apply<ArrayImpl, const NDIM: usize>(
        &self,
        arr: &Array<ArrayImpl, NDIM>,
    ) -> DynArray<Item, NDIM>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        if NDIM != 1 && NDIM != 2 {
            panic!("`PInv::apply` is only implemented for 1D and 2D arrays.");
        }
        if self.s.is_empty() {
            let mut shape = arr.shape();
            shape[0] = self.v.shape()[0];
            return DynArray::from_shape(shape);
        }

        let sinv = self
            .s
            .r()
            .unary_op(|elem| <Item as RlstScalar>::recip(elem));

        if NDIM == 2 {
            let arr = DynArray::new_from(arr).coerce_dim::<2>().unwrap();
            let mut tmp = dot!(self.ut.r(), arr);
            for mut col in tmp.col_iter_mut() {
                for (elem, si) in izip!(col.iter_mut(), sinv.iter_value()) {
                    // Needed because rust analyzer had trouble identifying the type correctly.
                    let elem: &mut Item = elem;
                    *elem *= si;
                }
            }
            dot!(self.v.r(), tmp).coerce_dim::<NDIM>().unwrap().eval()
        } else if NDIM == 1 {
            let arr = DynArray::new_from(arr).coerce_dim::<1>().unwrap();
            let mut tmp = dot!(self.ut.r(), arr);
            for (elem, si) in izip!(tmp.iter_mut(), sinv.iter_value()) {
                *elem *= si;
            }
            dot!(self.v.r(), tmp).coerce_dim::<NDIM>().unwrap().eval()
        } else {
            panic!("`PInv::apply` is only implemented for 1D and 2D arrays.");
        }
    }
}

#[cfg(test)]
mod test {

    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::traits::linalg::SingularValueDecomposition;

    use paste::paste;

    macro_rules! implement_pinv_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_pseudo_inverse_thin_$scalar>]() {
                let m = 20;
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_normally_distributed(0);

                let pinv = a.pseudo_inverse(None, None).unwrap();

                let pinv_mat = pinv.as_matrix();

                assert_eq!(pinv_mat.shape(), [n, m]);

                let mut ident = DynArray::<$scalar, 2>::from_shape([n, n]);
                ident.set_identity();

                let actual = dot!(pinv_mat.r(), a.r());

                crate::assert_array_abs_diff_eq!(actual, ident, $tol);

                let mut x = DynArray::<$scalar, 2>::from_shape([m, 2]);

                x.fill_from_seed_equally_distributed(1);

                crate::assert_array_relative_eq!(dot!(pinv_mat.r(), x.r()), pinv.apply(&x), $tol);
            }

            #[test]
            fn [<test_pseudo_inverse_thick_$scalar>]() {
                let m = 10;
                let n = 20;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_normally_distributed(0);

                let pinv = a.pseudo_inverse(None, None).unwrap();

                let pinv_mat = pinv.as_matrix();

                assert_eq!(pinv_mat.shape(), [n, m]);

                let mut ident = DynArray::<$scalar, 2>::from_shape([m, m]);
                ident.set_identity();

                let actual = dot!(a.r(), pinv_mat.r());

                crate::assert_array_abs_diff_eq!(actual, ident, $tol);

                let mut x = DynArray::<$scalar, 2>::from_shape([m, 2]);

                x.fill_from_seed_equally_distributed(1);

                crate::assert_array_relative_eq!(dot!(pinv_mat.r(), x.r()), pinv.apply(&x), $tol);
            }

                    }
        };
    }

    implement_pinv_tests!(f32, 1E-4);
    implement_pinv_tests!(f64, 1E-10);
    implement_pinv_tests!(c32, 1E-4);
    implement_pinv_tests!(c64, 1E-10);

    #[test]
    fn test_pseudo_inverse_rank_deficient() {
        let mut a = DynArray::<f64, 2>::from_shape([3, 2]);
        a[[0, 0]] = 1.0;
        a[[1, 0]] = 2.0;
        a[[2, 0]] = 3.0;
        a[[0, 1]] = 2.0;
        a[[1, 1]] = 4.0;
        a[[2, 1]] = 6.0;

        let pinv = a.pseudo_inverse(None, Some(1E-12)).unwrap();
        assert_eq!(pinv.s().len(), 1);

        let pinv_mat = pinv.as_matrix();
        let reconstructed = dot!(a.r(), pinv_mat.r(), a.r());
        crate::assert_array_abs_diff_eq!(reconstructed, a, 1E-12);
    }

    #[test]
    fn test_pseudo_inverse_cutoff_excludes_boundary() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 2]);
        a[[0, 0]] = 4.0;
        a[[1, 1]] = 2.0;

        let pinv = a.pseudo_inverse(None, Some(0.5)).unwrap();
        assert_eq!(pinv.s().len(), 1);

        let pinv_mat = pinv.as_matrix();
        assert!((pinv_mat[[0, 0]] - 0.25).abs() < 1E-12);
        assert!(pinv_mat[[1, 1]].abs() < 1E-12);
    }

    #[test]
    fn test_pseudo_inverse_rank_zero() {
        let mut a = DynArray::<f64, 2>::from_shape([2, 2]);
        a.set_identity();

        let pinv = a.pseudo_inverse(Some(0), None).unwrap();
        assert_eq!(pinv.s().len(), 0);
        crate::assert_array_abs_diff_eq!(
            pinv.as_matrix(),
            DynArray::<f64, 2>::from_shape([2, 2]),
            1E-12
        );
    }

    #[test]
    fn test_pseudo_inverse_all_zero() {
        let a = DynArray::<f64, 2>::from_shape([3, 2]);

        let pinv = a.pseudo_inverse(None, None).unwrap();
        assert_eq!(pinv.s().len(), 0);
        crate::assert_array_abs_diff_eq!(
            pinv.as_matrix(),
            DynArray::<f64, 2>::from_shape([2, 3]),
            1E-12
        );
    }
}
