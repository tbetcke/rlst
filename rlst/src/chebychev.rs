//! Functions for the evaluation of Chebychev series.

use std::ops::{DivAssign, MulAssign};

use itertools::izip;
use num::PrimInt;

use crate::fftw::{FftPlanInplace, R2RPlanKind};
use crate::{Array, fftw::R2RPlan};
use crate::{
    DynArray, EvaluateObject, EvaluateRowMajorArray, MutableArrayImpl, RlstScalar, Shape,
    UnsafeRandomAccessByValue, ValueArrayImpl,
};

/// Interface to tools for working with Chebychev expansions
pub trait ChebychevCoefficientsToData<Item, const NDIM: usize> {
    /// Compute Chebychev coefficients from values at Chebychev points
    fn chebychev_coeffs_from_data_second_kind(&self) -> DynArray<Item, NDIM>;
    /// Compute data at Chebychev points from series coefficients
    fn chebychev_data_from_coeffs_second_kind(&self) -> DynArray<Item, NDIM>;
}

/// Evaluation of Chebychev Coefficients in one dimension
pub trait EvaluateChebychevCoefficients1d<Item> {
    /// Evaluate a 1d array assuming that its entries are Chebychev coefficients
    ///
    /// The implementation uses the [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm).
    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 1>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<Item, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 1>;
}

/// Evaluation of Chebychev Coefficients in two dimensions
pub trait EvaluateChebychevCoefficients2d<Item> {
    /// Evaluate a 2d array assuming that its entries are Chebychev coefficients
    ///
    /// The implementation uses the [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm).
    fn tensor_evaluate_chebychev_series_second_kind<
        ArrayImplPointsX,
        ArrayImplPointsY,
        ArrayImplResult,
    >(
        &self,
        eval_points_x: &Array<ArrayImplPointsX, 1>,
        eval_points_y: &Array<ArrayImplPointsY, 1>,
        result: &mut Array<ArrayImplResult, 2>,
    ) where
        ArrayImplPointsX: ValueArrayImpl<Item, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 2>;
}

/// Evaluation of Chebychev coefficients in three dimensions
pub trait EvaluateChebychevCoefficients3d<Item> {
    /// Evaluate a 3d array assuming that its entries are Chebychev coefficients
    ///
    /// The implementation uses the [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm).
    fn tensor_evaluate_chebychev_series_second_kind<
        ArrayImplPointsX,
        ArrayImplPointsY,
        ArrayImplPointsZ,
        ArrayImplResult,
    >(
        &self,
        eval_points_x: &Array<ArrayImplPointsX, 1>,
        eval_points_y: &Array<ArrayImplPointsY, 1>,
        eval_points_z: &Array<ArrayImplPointsZ, 1>,
        result: &mut Array<ArrayImplResult, 3>,
    ) where
        ArrayImplPointsX: ValueArrayImpl<Item, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item, 1>,
        ArrayImplPointsZ: ValueArrayImpl<Item, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 3>;
}

impl<Item, ArrayImpl> EvaluateChebychevCoefficients1d<Item> for Array<ArrayImpl, 1>
where
    Item: RlstScalar,
    ArrayImpl: ValueArrayImpl<Item, 1>,
{
    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 1>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<Item, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 1>,
    {
        // Make sure that the arrays make sense
        assert!(!self.is_empty());
        assert_eq!(eval_points.len(), result.len());

        // The degree of the Chebychev expansion
        let n = self.len() - 1;
        let mut bkp1;
        let mut bkp2;
        let a0 = self.get_value([0]).unwrap();

        let two = Item::from_f64(2.0).unwrap();

        for (x, r) in izip!(eval_points.iter_value(), result.iter_mut()) {
            bkp1 = Item::default();
            bkp2 = Item::default();
            // We only do the coefficients a_n up to a_1.
            // The coefficient a_0 is treated separately
            for cheb_coeff in self.r().reverse_axis(0).iter_value().take(n) {
                (bkp1, bkp2) = (cheb_coeff + two * x * bkp1 - bkp2, bkp1);
            }
            *r = a0 + x * bkp1 - bkp2;
        }
    }
}

impl<Item, ArrayImpl> EvaluateChebychevCoefficients2d<Item> for Array<ArrayImpl, 2>
where
    Item: RlstScalar,
    ArrayImpl: ValueArrayImpl<Item, 2>,
{
    fn tensor_evaluate_chebychev_series_second_kind<
        ArrayImplPointsX,
        ArrayImplPointsY,
        ArrayImplResult,
    >(
        &self,
        eval_points_x: &Array<ArrayImplPointsX, 1>,
        eval_points_y: &Array<ArrayImplPointsY, 1>,
        result: &mut Array<ArrayImplResult, 2>,
    ) where
        ArrayImplPointsX: ValueArrayImpl<Item, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 2>,
    {
        // Check that the dimensions are correct

        let [mx, my] = result.shape();

        let [_nx, ny] = self.shape();

        assert_eq!(mx, eval_points_x.len());
        assert_eq!(my, eval_points_y.len());

        // We need to create a temporary matrix that holds the intermediate result
        // of transforming along one direction.

        let mut tmp = DynArray::<Item, 2>::from_shape([mx, ny]);

        // Now iterate through the columns of the input matrix and evaluate each column into tmp

        for (col_in, mut col_tmp) in izip!(self.col_iter(), tmp.col_iter_mut()) {
            col_in.evaluate_chebychev_series_second_kind(eval_points_x, &mut col_tmp);
        }

        // And now we iterate through the rows and evaluate along y to get the result for each row.
        // This is stored in the output array.

        for (row_tmp, mut row_res) in izip!(tmp.row_iter(), result.row_iter_mut()) {
            row_tmp.evaluate_chebychev_series_second_kind(eval_points_y, &mut row_res);
        }
    }
}

impl<ArrayImpl, const NDIM: usize> ChebychevCoefficientsToData<f64, NDIM> for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: EvaluateObject<Output = DynArray<f64, NDIM>>,
{
    fn chebychev_coeffs_from_data_second_kind(&self) -> DynArray<f64, NDIM> {
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

    fn chebychev_data_from_coeffs_second_kind(&self) -> DynArray<f64, NDIM> {
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
        Linspace, assert_array_relative_eq,
        dense::tools::PrettyPrint,
        interpolation::{
            Kind, barycentric_chebychev_weights, chebychev_points, tensor_barycentric_evaluate_2d,
        },
    };

    use super::*;

    #[test]
    pub fn test_data_to_coeffs_1d() {
        let n = 3;

        let points = chebychev_points::<f64>(Kind::Second, n);
        let mut one = DynArray::<f64, _>::from_shape([n]);
        one.fill_with_value(1.0);

        let data = 2.0_f64 * points.r() * points.r() + points.r();

        let coeffs = data.chebychev_coeffs_from_data_second_kind();

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind();

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

        let coeffs = data.chebychev_coeffs_from_data_second_kind();

        coeffs.pretty_print();

        crate::assert_array_relative_eq!(coeffs_expected, coeffs, 1E-10);

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind();

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

        let coeffs = data.chebychev_coeffs_from_data_second_kind();

        crate::assert_array_relative_eq!(coeffs_expected, coeffs, 1E-10);

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind();

        crate::assert_array_abs_diff_eq!(data, data_out, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_1d() {
        let n = 11;

        let mut expected = DynArray::<f64, _>::from_shape([n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_2d() {
        let n = 5;
        let m = 7;

        let mut expected = DynArray::<f64, _>::from_shape([m, n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_3d() {
        let m = 7;
        let n = 5;
        let k = 8;

        let mut expected = DynArray::<f64, _>::from_shape([m, n, k]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_eval_cheb_series_1d_second_kind() {
        let n = 3;
        let m = 11;
        let mut eval_points = DynArray::<f64, 1>::from_shape([m]);
        eval_points.linspace(-1.0, 1.0);

        let mut coeffs = DynArray::<f64, 1>::from_shape([n]);

        // This defines the polynomial p(x) = 5.0x^2 + 2x - 1.5

        coeffs[[0]] = 1.0;
        coeffs[[1]] = 2.0;
        coeffs[[2]] = 2.5;

        let mut result = DynArray::<f64, 1>::from_shape([m]);
        coeffs.evaluate_chebychev_series_second_kind(&eval_points, &mut result);

        let mut expected = eval_points.eval();
        expected
            .iter_mut()
            .for_each(|x| *x = 5.0 * *x * *x + 2.0 * *x - 1.5);

        crate::assert_array_relative_eq!(result, expected, 1E-13);
    }

    #[test]
    pub fn test_eval_cheb_series_2d_second_kind() {
        let nx = 5;
        let ny = 10;
        let mx = 60;
        let my = 70;

        let mut eval_points_x = DynArray::<f64, 1>::from_shape([mx]);
        let mut eval_points_y = DynArray::<f64, 1>::from_shape([my]);
        eval_points_x.linspace(-1.0, 1.0);
        eval_points_y.linspace(-1.0, 1.0);

        let mut coeffs = DynArray::<f64, 2>::from_shape([nx, ny]);
        coeffs.fill_from_seed_equally_distributed(0);

        // We fill the coefficients with random values. We then compare the direct evaluation with
        // the evaluation obtained by a DCT to physical space, followed by barycentric evaluation.

        let mut actual = DynArray::<f64, 2>::from_shape([mx, my]);
        coeffs.tensor_evaluate_chebychev_series_second_kind(
            &eval_points_x,
            &eval_points_y,
            &mut actual,
        );

        // We now compute the same by doing a DFT followed by barycentric evaluation.

        let data = coeffs.chebychev_data_from_coeffs_second_kind();

        let mut expected = DynArray::<f64, 2>::from_shape([mx, my]);

        let cheb_points_x = chebychev_points(Kind::Second, nx);
        let cheb_points_y = chebychev_points(Kind::Second, ny);
        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);

        tensor_barycentric_evaluate_2d(
            &eval_points_x,
            &eval_points_y,
            &cheb_points_x,
            &cheb_points_y,
            &weights_x,
            &weights_y,
            &data,
            &mut expected,
        );

        assert_array_relative_eq!(actual, expected, 1E-5);
    }
}
