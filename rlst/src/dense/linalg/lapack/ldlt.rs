//! Lapack Symmetric/Hermitian Factorization.

use crate::base_types::{RlstResult, UpLo};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::sytrf::SytrfUplo;
use crate::traits::linalg::decompositions::Ldl;
use crate::traits::linalg::lapack::Lapack;
use crate::{Shape, UnsafeRandom1DAccessByValue};

impl<Item, ArrayImpl> Ldl for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;
    fn ldl(&self, uplo: UpLo) -> RlstResult<LdlDecomposition<Item>> {
        let mut ldl_mat = DynArray::new_from(self);
        let [m, n] = ldl_mat.shape();
        assert_eq!(m, n, "Matrix must be square for LDL decomposition.");
        let n = m;
        let lda = n;
        let mut ipiv = vec![0_i32; n];

        let lapack_uplo = match uplo {
            UpLo::Upper => SytrfUplo::Upper,
            UpLo::Lower => SytrfUplo::Lower,
        };

        Item::sytrf(lapack_uplo, n, ldl_mat.data_mut().unwrap(), lda, &mut ipiv)?;

        Ok(LdlDecomposition {
            ldl: ldl_mat,
            ipiv,
            uplo,
        })
    }
}

/// Packed pivoted LDL factors of a square symmetric/Hermitian matrix.
///
/// The factors and LAPACK pivot vector are intentionally not exposed; use
/// [`Self::solve`] to solve systems with the original matrix.
pub struct LdlDecomposition<Item> {
    ldl: DynArray<Item, 2>,
    ipiv: Vec<i32>,
    uplo: UpLo,
}

impl<Item> LdlDecomposition<Item>
where
    Item: Lapack,
{
    /// Solve `A x = b` from the stored factors without modifying `b`.
    ///
    /// A vector right-hand side must have length `n`; a matrix right-hand side must have `n`
    /// rows. The underlying LAPACK wrapper checks the resulting flat storage length, but this
    /// does not independently reject every incompatible shape when the number of right-hand
    /// sides is zero. Singular pivots are reported when constructing the factorization.
    pub fn solve<ArrayImpl, const NDIM: usize>(
        &self,
        b: &Array<ArrayImpl, NDIM>,
    ) -> RlstResult<DynArray<Item, NDIM>>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        let mut sol = DynArray::new_from(b);
        let n = self.ldl.shape()[0];
        let nrhs = if NDIM == 1 { 1 } else { sol.shape()[1] };

        let lapack_uplo = match self.uplo {
            UpLo::Upper => SytrfUplo::Upper,
            UpLo::Lower => SytrfUplo::Lower,
        };

        Item::sytrs(
            lapack_uplo,
            n,
            nrhs,
            self.ldl.data().unwrap(),
            n,
            &self.ipiv,
            sol.data_mut().unwrap(),
            n,
        )?;

        Ok(sol)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::MultIntoResize;
    use crate::dense::array::DynArray;
    use crate::empty_array;
    use num::complex::ComplexFloat;

    #[test]
    fn test_sytrf_f64_lower() {
        let n = 5;
        let mut arr = DynArray::<f64, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                arr[[i, j]] = (i + j) as f64 + 1.0;
            }
            arr[[i, i]] += 10.0;
        }

        let ldl = arr.ldl(UpLo::Lower).unwrap();
        let mut x_expected = DynArray::<f64, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = (i + 1) as f64;
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-12);
    }

    #[test]
    fn test_sytrf_f64_upper() {
        let n = 5;
        let mut arr = DynArray::<f64, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                arr[[i, j]] = (i + j) as f64 + 1.0;
            }
            arr[[i, i]] += 10.0;
        }

        let ldl = arr.ldl(UpLo::Upper).unwrap();
        let mut x_expected = DynArray::<f64, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = (i + 1) as f64;
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-12);
    }

    #[test]
    fn test_sytrf_c64_lower() {
        let n = 5;
        let mut arr = DynArray::<crate::base_types::c64, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                let val = (i + j) as f64 + 1.0;
                arr[[i, j]] = crate::base_types::c64::new(val, i as f64 - j as f64);
            }
            arr[[i, i]] = crate::base_types::c64::new(arr[[i, i]].re(), 0.0); // Must be Hermitian
            arr[[i, i]] += 10.0;
        }
        // Ensure Hermitian
        for i in 0..n {
            for j in i + 1..n {
                arr[[i, j]] = arr[[j, i]].conj();
            }
        }

        let ldl = arr.ldl(UpLo::Lower).unwrap();
        let mut x_expected = DynArray::<crate::base_types::c64, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = crate::base_types::c64::new((i + 1) as f64, (i + 1) as f64);
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-12);
    }

    #[test]
    fn test_sytrf_c64_upper() {
        let n = 5;
        let mut arr = DynArray::<crate::base_types::c64, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                let val = (i + j) as f64 + 1.0;
                arr[[i, j]] = crate::base_types::c64::new(val, i as f64 - j as f64);
            }
            arr[[i, i]] = crate::base_types::c64::new(arr[[i, i]].re(), 0.0); // Must be Hermitian
            arr[[i, i]] += 10.0;
        }
        // Ensure Hermitian
        for i in 0..n {
            for j in i + 1..n {
                arr[[i, j]] = arr[[j, i]].conj();
            }
        }

        let ldl = arr.ldl(UpLo::Upper).unwrap();
        let mut x_expected = DynArray::<crate::base_types::c64, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = crate::base_types::c64::new((i + 1) as f64, (i + 1) as f64);
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-12);
    }

    #[test]
    fn test_sytrf_f32_lower() {
        let n = 5;
        let mut arr = DynArray::<f32, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                arr[[i, j]] = (i + j) as f32 + 1.0;
            }
            arr[[i, i]] += 10.0;
        }

        let ldl = arr.ldl(UpLo::Lower).unwrap();
        let mut x_expected = DynArray::<f32, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = (i + 1) as f32;
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-5);
    }

    #[test]
    fn test_sytrf_f32_upper() {
        let n = 5;
        let mut arr = DynArray::<f32, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                arr[[i, j]] = (i + j) as f32 + 1.0;
            }
            arr[[i, i]] += 10.0;
        }

        let ldl = arr.ldl(UpLo::Upper).unwrap();
        let mut x_expected = DynArray::<f32, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = (i + 1) as f32;
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-5);
    }

    #[test]
    fn test_sytrf_c32_lower() {
        let n = 5;
        let mut arr = DynArray::<crate::base_types::c32, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                let val = (i + j) as f32 + 1.0;
                arr[[i, j]] = crate::base_types::c32::new(val, i as f32 - j as f32);
            }
            arr[[i, i]] = crate::base_types::c32::new(arr[[i, i]].re(), 0.0); // Must be Hermitian
            arr[[i, i]] += 10.0;
        }
        // Ensure Hermitian
        for i in 0..n {
            for j in i + 1..n {
                arr[[i, j]] = arr[[j, i]].conj();
            }
        }

        let ldl = arr.ldl(UpLo::Lower).unwrap();
        let mut x_expected = DynArray::<crate::base_types::c32, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = crate::base_types::c32::new((i + 1) as f32, (i + 1) as f32);
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-5);
    }

    #[test]
    fn test_sytrf_c32_upper() {
        let n = 5;
        let mut arr = DynArray::<crate::base_types::c32, 2>::from_shape([n, n]);
        for i in 0..n {
            for j in 0..n {
                let val = (i + j) as f32 + 1.0;
                arr[[i, j]] = crate::base_types::c32::new(val, i as f32 - j as f32);
            }
            arr[[i, i]] = crate::base_types::c32::new(arr[[i, i]].re(), 0.0); // Must be Hermitian
            arr[[i, i]] += 10.0;
        }
        // Ensure Hermitian
        for i in 0..n {
            for j in i + 1..n {
                arr[[i, j]] = arr[[j, i]].conj();
            }
        }

        let ldl = arr.ldl(UpLo::Upper).unwrap();
        let mut x_expected = DynArray::<crate::base_types::c32, 1>::from_shape([n]);
        for i in 0..n {
            x_expected[[i]] = crate::base_types::c32::new((i + 1) as f32, (i + 1) as f32);
        }

        let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());
        let x_actual = ldl.solve(&rhs).unwrap();

        crate::assert_array_relative_eq!(x_actual, x_expected, 1e-5);
    }
}
