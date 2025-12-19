//! Functions for the evaluation of Chebychev series.

use std::ops::{DivAssign, MulAssign};

use num::PrimInt;

use crate::fftw::{FftPlanInplace, R2RPlanKind};
use crate::{Array, fftw::R2RPlan};
use crate::{DynArray, EvaluateObject};

/// Interface to tools for working with Chebychev expansions
pub trait ChebychevCoefficientsToData<const NDIM: usize> {
    /// The associated item type
    type Item;
    /// Compute Chebychev coefficients from values at Chebychev points
    fn chebychev_coeffs_from_data_first_kind(&self) -> DynArray<Self::Item, NDIM>;
    /// Compute data at Chebychev points from series coefficients
    fn chebychev_data_from_coeffs_first_kind(&self) -> DynArray<Self::Item, NDIM>;
}

pub trait EvaluateChebychevCoefficients<const NDIM: usize> {
    type Item;

    /// Evaluate a 1d array assuming that its entries are Chebychev coefficients
    fn evaluate_chebychev_series_1d<ArrayImpl>(
        eval_points: &Array<ArrayImpl, 1>,
    ) -> DynArray<Self::Item, NDIM>;
}

impl<ArrayImpl, const NDIM: usize> ChebychevCoefficientsToData<NDIM> for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: EvaluateObject<Output = DynArray<f64, NDIM>>,
{
    type Item = f64;
    fn chebychev_coeffs_from_data_first_kind(&self) -> DynArray<Self::Item, NDIM> {
        if NDIM > 3 {
            unimplemented!();
        }

        let mut result = self.eval();

        let shape = result.shape();

        // Compute the discrete cosine transform of the data

        let kind = vec![R2RPlanKind::FFTW_REDFT00; NDIM];

        result
            .r_mut()
            .into_fft()
            .r2r_fft(crate::fftw::FftwPlanFlags::Estimate, &kind)
            .unwrap()
            .execute();

        // We scale all values.
        let n_recip = (shape.iter().map(|elem| elem - 1).product::<usize>() as f64).recip();
        result *= n_recip;

        // We now have to deal with the boundary data

        // For NDIM = 1 we just need to adjust the single boundary points
        // For NDIM = 2 and NDIM = 3 we need to adjust the boundary values of the hypercube
        // formed by the values.

        if NDIM == 1 {
            *result.data_mut().unwrap().first_mut().unwrap() /= 2.0;
            *result.data_mut().unwrap().last_mut().unwrap() /= 2.0;
        } else if NDIM == 2 {
            for (axis, dim) in shape.iter().enumerate() {
                for index in [0, dim - 1] {
                    result
                        .r_mut()
                        .coerce_dim::<2>()
                        .unwrap()
                        .slice::<1>(axis, index)
                        .div_assign(2.0);
                }
            }
        } else if NDIM == 3 {
            for (axis, dim) in shape.iter().enumerate() {
                for index in [0, dim - 1] {
                    result
                        .r_mut()
                        .coerce_dim::<3>()
                        .unwrap()
                        .slice::<2>(axis, index)
                        .div_assign(2.0);
                }
            }
        }

        result
    }

    fn chebychev_data_from_coeffs_first_kind(&self) -> DynArray<Self::Item, NDIM> {
        if NDIM > 3 {
            unimplemented!();
        }

        let mut result = self.eval();

        let shape = result.shape();

        // Rescale the values.

        result *= (2.pow(NDIM as u32) as f64).recip();

        if NDIM == 1 {
            *result.data_mut().unwrap().first_mut().unwrap() *= 2.0;
            *result.data_mut().unwrap().last_mut().unwrap() *= 2.0;
        } else if NDIM == 2 {
            for (axis, dim) in shape.iter().enumerate() {
                for index in [0, dim - 1] {
                    result
                        .r_mut()
                        .coerce_dim::<2>()
                        .unwrap()
                        .slice::<1>(axis, index)
                        .mul_assign(2.0);
                }
            }
        } else if NDIM == 3 {
            for (axis, dim) in shape.iter().enumerate() {
                for index in [0, dim - 1] {
                    result
                        .r_mut()
                        .coerce_dim::<3>()
                        .unwrap()
                        .slice::<2>(axis, index)
                        .mul_assign(2.0);
                }
            }
        }

        // Compute the discrete cosine transform of the data

        let kind = vec![R2RPlanKind::FFTW_REDFT00; NDIM];

        result
            .r_mut()
            .into_fft()
            .r2r_fft(crate::fftw::FftwPlanFlags::Estimate, &kind)
            .unwrap()
            .execute();

        // This neutralises the boundary scaling and leaves only the interior values divided by two.
        // This is easier than trying to scale only interior values.

        result
    }
}

#[cfg(test)]
mod test {
    use crate::{
        dense::tools::PrettyPrint,
        interpolation::{Kind, chebychev_points},
    };

    use super::*;

    #[test]
    pub fn test_data_to_coeffs_1d() {
        let n = 3;

        let points = chebychev_points::<f64>(Kind::Second, n);
        let mut one = DynArray::<f64, _>::from_shape([n]);
        one.fill_with_value(1.0);

        let data = 2.0_f64 * points.r() * points.r() + points.r();

        let coeffs = data.chebychev_coeffs_from_data_first_kind();

        let data_out = coeffs.chebychev_data_from_coeffs_first_kind();

        crate::assert_array_relative_eq!(coeffs, one, 1E-10);
        crate::assert_array_relative_eq!(data, data_out, 1E-10);
    }

    #[test]
    pub fn test_data_to_coeffs_2d() {
        let n = 3;

        let points = chebychev_points::<f64>(Kind::Second, n);
        let mut data = DynArray::<f64, 2>::from_shape([n, n]);

        for (i, pi) in points.iter_value().enumerate() {
            for (j, pj) in points.iter_value().enumerate() {
                data[[i, j]] = pi * pj + pi + pj + 2.0 * pi * pi + 2.0 * pj * pj - 1.0;
            }
        }

        let mut coeffs_expected = DynArray::<f64, 2>::from_shape([3, 3]);
        // All values are one except the very last entry
        coeffs_expected[[0, 0]] = 1.0;
        coeffs_expected[[1, 0]] = 1.0;
        coeffs_expected[[2, 0]] = 1.0;
        coeffs_expected[[0, 1]] = 1.0;
        coeffs_expected[[1, 1]] = 1.0;
        coeffs_expected[[0, 2]] = 1.0;

        let coeffs = data.chebychev_coeffs_from_data_first_kind();

        coeffs.pretty_print();

        crate::assert_array_relative_eq!(coeffs_expected, coeffs, 1E-10);

        let data_out = coeffs.chebychev_data_from_coeffs_first_kind();

        crate::assert_array_abs_diff_eq!(data, data_out, 1E-10);
    }

    #[test]
    pub fn test_data_to_coeffs_3d() {
        let n = 3;

        let points = chebychev_points::<f64>(Kind::Second, n);
        let mut data = DynArray::<f64, 3>::from_shape([n, n, n]);

        for (i, pi) in points.iter_value().enumerate() {
            for (j, pj) in points.iter_value().enumerate() {
                for (k, pk) in points.iter_value().enumerate() {
                    data[[i, j, k]] =
                        pi * pj + pi + pj + 2.0 * pi * pi + 2.0 * pj * pj - 1.0 + 3.0 * pk;
                }
            }
        }

        let mut coeffs_expected = DynArray::<f64, 3>::from_shape([3, 3, 3]);
        // All values are one except the very last entry
        coeffs_expected[[0, 0, 0]] = 1.0;
        coeffs_expected[[1, 0, 0]] = 1.0;
        coeffs_expected[[2, 0, 0]] = 1.0;
        coeffs_expected[[0, 1, 0]] = 1.0;
        coeffs_expected[[1, 1, 0]] = 1.0;
        coeffs_expected[[0, 2, 0]] = 1.0;
        coeffs_expected[[0, 0, 1]] = 3.0;

        let coeffs = data.chebychev_coeffs_from_data_first_kind();

        crate::assert_array_relative_eq!(coeffs_expected, coeffs, 1E-10);

        let data_out = coeffs.chebychev_data_from_coeffs_first_kind();

        crate::assert_array_abs_diff_eq!(data, data_out, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_1d() {
        let n = 11;

        let mut expected = DynArray::<f64, _>::from_shape([n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_first_kind();

        let actual = coeffs.chebychev_data_from_coeffs_first_kind();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_2d() {
        let n = 5;
        let m = 7;

        let mut expected = DynArray::<f64, _>::from_shape([m, n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_first_kind();

        let actual = coeffs.chebychev_data_from_coeffs_first_kind();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_3d() {
        let m = 7;
        let n = 5;
        let k = 8;

        let mut expected = DynArray::<f64, _>::from_shape([m, n, k]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_first_kind();

        let actual = coeffs.chebychev_data_from_coeffs_first_kind();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }
}
