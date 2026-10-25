//! Functions for the evaluation and conversion of Chebychev series
//!
//! This module contains functions to
//! - Evaluate a Chebychev series
//! - Convert between data at Chebychev points and series coefficients
//! - Take the derivative of Chebychev expansions
//!
//! Evaluation and derivatives are available for 1, 2, or 3-dimensional Chebychev expansions,
//! the conversions of [`ChebychevCoefficientsToData`] for any number of dimensions. In more
//! than one dimension the Chebychev expansions are assumed to be defined as tensors of direct
//! products of 1-dimensional expansions.
//!
//! The conversions use FFTW's DCT-I through [`rlst::fftw`](crate::fftw), which is why this
//! module needs the feature `fftw`.
//!
//! ```
//! use rlst::chebychev::{
//!     ChebychevCoefficientsToData, DerivativeOfChebychevSeries1d, EvaluateChebychevCoefficients1d,
//! };
//! use rlst::interpolation::{Kind, chebychev_points};
//! use rlst::rlst_dynamic_array;
//!
//! // Sample p(x) = x^3 at 4 Chebychev points of the second kind.
//! let points = chebychev_points::<f64>(Kind::Second, 4);
//! let mut data = rlst_dynamic_array!(f64, [4]);
//! for (value, x) in data.iter_mut().zip(points.iter_value()) {
//!     *value = x * x * x;
//! }
//!
//! // The coefficients of p' = 3 x^2, evaluated at x = 0.5.
//! let mut coeffs = data.chebychev_coeffs_from_data_second_kind()?;
//! coeffs.chebychev_derivative_second_kind(1);
//! let mut x = rlst_dynamic_array!(f64, [1]);
//! x[[0]] = 0.5;
//! let mut value = rlst_dynamic_array!(f64, [1]);
//! coeffs.evaluate_chebychev_series_second_kind(&x, &mut value);
//! assert!((value[[0]] - 0.75).abs() < 1e-12);
//! # Ok::<(), rlst::fftw::FftwError>(())
//! ```

use itertools::izip;

use crate::dense::base_array::BaseArray;
use crate::dense::data_container::ArrayContainer;
use crate::fftw::{FftwError, FftwReal, Layout, Planner, R2rKind};
use crate::{Array, DynArray, MutableArrayImpl, RlstScalar, ValueArrayImpl};
use crate::{c32, c64};

/// Convert between tensor-grid data and Chebychev coefficients.
///
/// Implemented for arrays of `f32`, `f64`, `c32` and `c64` of any number of dimensions,
/// including zero, for which both conversions return a copy. Every axis uses second-kind
/// points and its array extent determines the number of points.
///
/// Along each axis of extent `n > 1` the conversion is one DCT-I, FFTW's
/// [`R2rKind::Redft00`], along that axis and batched over the others, applied axis by axis,
/// and scaled. Along an axis of extent 1, one Chebychev point gives a constant interpolant, so
/// data and coefficients are equal along that axis. An array with an axis of extent 0 gives an
/// empty array of the same shape without calling FFTW. Complex arrays are converted as their
/// real and imaginary parts.
///
/// Each call plans its transforms with [`Rigor::Estimate`](crate::fftw::Rigor::Estimate), on
/// the array it returns, which FFTW then leaves untouched. The results agree with those of
/// RLST 0.10 to round-off; one-dimensional plans in turn are not the same plans as one
/// multidimensional plan, so the last bits can differ.
///
/// # Errors
///
/// Both methods return the [`FftwError`] of a plan that FFTW refuses or of its execution
/// instead of panicking. None is expected: the one that a working installation can return is
/// [`FftwError::VersionTooOld`], when the FFTW linked at run time is older than 3.3.4 (3.3.6
/// with the feature `fftw_threads`).
///
/// ```
/// use rlst::chebychev::ChebychevCoefficientsToData;
/// use rlst::rlst_dynamic_array;
///
/// // The constant 1 on a 3 x 4 grid has the single coefficient 1.
/// let mut data = rlst_dynamic_array!(f64, [3, 4]);
/// data.fill_with_value(1.0);
/// let coeffs = data.chebychev_coeffs_from_data_second_kind()?;
/// assert!((coeffs[[0, 0]] - 1.0).abs() < 1e-14);
/// assert!(coeffs.iter_value().skip(1).all(|c| c.abs() < 1e-14));
///
/// // And back to the data.
/// let back = coeffs.chebychev_data_from_coeffs_second_kind()?;
/// assert!(back.iter_value().all(|v| (v - 1.0).abs() < 1e-14));
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
pub trait ChebychevCoefficientsToData<Item, const NDIM: usize> {
    /// Compute Chebychev coefficients from values at Chebychev points
    ///
    /// This function interprets the contents of the array as data at Chebychev points of the
    /// second kind, `cos(π j / (n - 1))` for `j = 0, …, n - 1` along an axis of extent `n`, as
    /// [`chebychev_points`](crate::interpolation::chebychev_points) returns them, and computes
    /// the associated series expansion coefficients. See the
    /// [trait](ChebychevCoefficientsToData) for the extents 1 and 0 and for the errors.
    ///
    /// ```
    /// use rlst::chebychev::ChebychevCoefficientsToData;
    /// use rlst::interpolation::{Kind, chebychev_points};
    /// use rlst::rlst_dynamic_array;
    ///
    /// // p(x) = 2 x^2 + x = T_0(x) + T_1(x) + T_2(x).
    /// let points = chebychev_points::<f64>(Kind::Second, 3);
    /// let mut data = rlst_dynamic_array!(f64, [3]);
    /// for (value, x) in data.iter_mut().zip(points.iter_value()) {
    ///     *value = 2.0 * x * x + x;
    /// }
    /// let coeffs = data.chebychev_coeffs_from_data_second_kind()?;
    /// assert!(coeffs.iter_value().all(|c| (c - 1.0).abs() < 1e-14));
    ///
    /// // One point along the first axis: the data are constant along it.
    /// let mut row = rlst_dynamic_array!(f64, [1, 3]);
    /// row.fill_from_iter(data.iter_value());
    /// let coeffs = row.chebychev_coeffs_from_data_second_kind()?;
    /// assert!(coeffs.iter_value().all(|c| (c - 1.0).abs() < 1e-14));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    fn chebychev_coeffs_from_data_second_kind(&self) -> Result<DynArray<Item, NDIM>, FftwError>;
    /// Compute data at Chebychev points from series coefficients
    ///
    /// This function interprets the contents of the array as Chebychev expansion coefficients
    /// and computes the associated values at Chebychev points of the second kind, in the
    /// order of [`chebychev_coeffs_from_data_second_kind`](Self::chebychev_coeffs_from_data_second_kind),
    /// whose inverse it is. See the [trait](ChebychevCoefficientsToData) for the extents 1
    /// and 0 and for the errors.
    ///
    /// ```
    /// use rlst::chebychev::ChebychevCoefficientsToData;
    /// use rlst::interpolation::{Kind, chebychev_points};
    /// use rlst::{c64, rlst_dynamic_array};
    ///
    /// // p(x) = T_0(x) + i T_2(x) = 1 + i (2 x^2 - 1).
    /// let mut coeffs = rlst_dynamic_array!(c64, [3]);
    /// coeffs[[0]] = c64::new(1.0, 0.0);
    /// coeffs[[2]] = c64::new(0.0, 1.0);
    /// let data = coeffs.chebychev_data_from_coeffs_second_kind()?;
    ///
    /// let points = chebychev_points::<f64>(Kind::Second, 3);
    /// for (value, x) in data.iter_value().zip(points.iter_value()) {
    ///     assert!((value - c64::new(1.0, 2.0 * x * x - 1.0)).norm() < 1e-14);
    /// }
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    fn chebychev_data_from_coeffs_second_kind(&self) -> Result<DynArray<Item, NDIM>, FftwError>;
}

/// Evaluation of Chebychev Coefficients in one dimension
///
/// ```
/// use rlst::chebychev::EvaluateChebychevCoefficients1d;
/// use rlst::{Linspace, rlst_dynamic_array};
///
/// // p(x) = T_0(x) + 2 T_1(x) + 2.5 T_2(x) = 5 x^2 + 2 x - 1.5.
/// let mut coeffs = rlst_dynamic_array!(f64, [3]);
/// coeffs[[0]] = 1.0;
/// coeffs[[1]] = 2.0;
/// coeffs[[2]] = 2.5;
///
/// let mut points = rlst_dynamic_array!(f64, [5]);
/// points.linspace(-1.0, 1.0);
/// let mut values = rlst_dynamic_array!(f64, [5]);
/// coeffs.evaluate_chebychev_series_second_kind(&points, &mut values);
/// for (value, x) in values.iter_value().zip(points.iter_value()) {
///     assert!((value - (5.0 * x * x + 2.0 * x - 1.5)).abs() < 1e-13);
/// }
/// ```
pub trait EvaluateChebychevCoefficients1d<Item>
where
    Item: RlstScalar,
{
    /// Evaluate a 1d array assuming that its entries are Chebychev coefficients
    ///
    /// The implementation uses the [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm).
    ///
    /// ```
    /// use rlst::chebychev::EvaluateChebychevCoefficients1d;
    /// use rlst::{c64, rlst_dynamic_array};
    ///
    /// // p(x) = i T_1(x) = i x, with complex coefficients and real points.
    /// let mut coeffs = rlst_dynamic_array!(c64, [2]);
    /// coeffs[[1]] = c64::new(0.0, 1.0);
    ///
    /// let mut points = rlst_dynamic_array!(f64, [2]);
    /// points[[0]] = 0.25;
    /// points[[1]] = -0.5;
    /// let mut values = rlst_dynamic_array!(c64, [2]);
    /// coeffs.evaluate_chebychev_series_second_kind(&points, &mut values);
    /// assert!((values[[0]] - c64::new(0.0, 0.25)).norm() < 1e-15);
    /// assert!((values[[1]] - c64::new(0.0, -0.5)).norm() < 1e-15);
    /// ```
    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 1>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<Item::Real, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 1>;
}

/// Evaluation of Chebychev Coefficients in two dimensions
///
/// ```
/// use rlst::chebychev::EvaluateChebychevCoefficients2d;
/// use rlst::rlst_dynamic_array;
///
/// // p(x, y) = T_1(x) T_1(y) = x y.
/// let mut coeffs = rlst_dynamic_array!(f64, [2, 2]);
/// coeffs[[1, 1]] = 1.0;
///
/// // The points (0.5, 0.5) and (-0.25, 1.0), as columns.
/// let mut points = rlst_dynamic_array!(f64, [2, 2]);
/// points.fill_from_iter([0.5, 0.5, -0.25, 1.0].into_iter());
/// let mut values = rlst_dynamic_array!(f64, [2]);
/// coeffs.evaluate_chebychev_series_second_kind(&points, &mut values);
/// assert!((values[[0]] - 0.25).abs() < 1e-15);
/// assert!((values[[1]] + 0.25).abs() < 1e-15);
/// ```
pub trait EvaluateChebychevCoefficients2d<Item>
where
    Item: RlstScalar,
{
    /// Evaluate a 2d Chebychev expansion at an array of points
    ///
    /// The points are stored as columns of the `(2, npts)` array `eval_points`.
    /// The output array is a one-dimensional array of length `npts`.
    ///
    /// ```
    /// use rlst::chebychev::EvaluateChebychevCoefficients2d;
    /// use rlst::rlst_dynamic_array;
    ///
    /// // p(x, y) = T_0(x) T_0(y) + T_2(x) T_0(y) = 2 x^2.
    /// let mut coeffs = rlst_dynamic_array!(f64, [3, 1]);
    /// coeffs[[0, 0]] = 1.0;
    /// coeffs[[2, 0]] = 1.0;
    ///
    /// // The point (0.5, -0.75).
    /// let mut points = rlst_dynamic_array!(f64, [2, 1]);
    /// points.fill_from_iter([0.5, -0.75].into_iter());
    /// let mut values = rlst_dynamic_array!(f64, [1]);
    /// coeffs.evaluate_chebychev_series_second_kind(&points, &mut values);
    /// assert!((values[[0]] - 0.5).abs() < 1e-15);
    /// ```
    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 2>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<Item::Real, 2>,
        ArrayImplResult: MutableArrayImpl<Item, 1>;
    /// Evaluate a 2d Chebychev expansion at the two-dimensional tensor product of 1d coordinates
    ///
    /// The evaluation points are assumed to be direct products of the points `eval_points_x` and
    /// `eval_points_y` in the x and y directions.
    /// The implementation uses the [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm).
    ///
    /// ```
    /// use rlst::chebychev::EvaluateChebychevCoefficients2d;
    /// use rlst::{Linspace, rlst_dynamic_array};
    ///
    /// // p(x, y) = T_1(x) T_1(y) = x y.
    /// let mut coeffs = rlst_dynamic_array!(f64, [2, 2]);
    /// coeffs[[1, 1]] = 1.0;
    ///
    /// let mut x = rlst_dynamic_array!(f64, [3]);
    /// let mut y = rlst_dynamic_array!(f64, [4]);
    /// x.linspace(-1.0, 1.0);
    /// y.linspace(0.0, 1.0);
    /// let mut values = rlst_dynamic_array!(f64, [3, 4]);
    /// coeffs.tensor_evaluate_chebychev_series_second_kind(&x, &y, &mut values);
    /// for (i, xi) in x.iter_value().enumerate() {
    ///     for (j, yj) in y.iter_value().enumerate() {
    ///         assert!((values[[i, j]] - xi * yj).abs() < 1e-15);
    ///     }
    /// }
    /// ```
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
        ArrayImplPointsX: ValueArrayImpl<Item::Real, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item::Real, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 2>;
}

/// Evaluation of Chebychev coefficients in three dimensions
///
/// ```
/// use rlst::chebychev::EvaluateChebychevCoefficients3d;
/// use rlst::rlst_dynamic_array;
///
/// // p(x, y, z) = T_1(x) T_0(y) T_1(z) = x z.
/// let mut coeffs = rlst_dynamic_array!(f64, [2, 1, 2]);
/// coeffs[[1, 0, 1]] = 1.0;
///
/// // The point (0.5, 0.1, -0.5), as a column.
/// let mut points = rlst_dynamic_array!(f64, [3, 1]);
/// points.fill_from_iter([0.5, 0.1, -0.5].into_iter());
/// let mut values = rlst_dynamic_array!(f64, [1]);
/// coeffs.evaluate_chebychev_series_second_kind(&points, &mut values);
/// assert!((values[[0]] + 0.25).abs() < 1e-15);
/// ```
pub trait EvaluateChebychevCoefficients3d<Item>
where
    Item: RlstScalar,
{
    /// Evaluate a 3d Chebychev expansion at an array of points
    ///
    /// The points are stored as columns of the `(3, npts)` array `eval_points`.
    /// The output array is a one-dimensional array of length `npts`.
    ///
    /// ```
    /// use rlst::chebychev::EvaluateChebychevCoefficients3d;
    /// use rlst::rlst_dynamic_array;
    ///
    /// // p(x, y, z) = T_1(x) T_1(y) T_1(z) = x y z.
    /// let mut coeffs = rlst_dynamic_array!(f64, [2, 2, 2]);
    /// coeffs[[1, 1, 1]] = 1.0;
    ///
    /// // The points (0.5, 0.5, 0.5) and (1, -1, 0.25), as columns.
    /// let mut points = rlst_dynamic_array!(f64, [3, 2]);
    /// points.fill_from_iter([0.5, 0.5, 0.5, 1.0, -1.0, 0.25].into_iter());
    /// let mut values = rlst_dynamic_array!(f64, [2]);
    /// coeffs.evaluate_chebychev_series_second_kind(&points, &mut values);
    /// assert!((values[[0]] - 0.125).abs() < 1e-15);
    /// assert!((values[[1]] + 0.25).abs() < 1e-15);
    /// ```
    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 2>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<Item::Real, 2>,
        ArrayImplResult: MutableArrayImpl<Item, 1>;

    /// Evaluate a 3d Chebychev expansion at the three-dimensional tensor product of 1d coordinates
    ///
    /// The implementation uses the [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm).
    ///
    /// ```
    /// use rlst::chebychev::EvaluateChebychevCoefficients3d;
    /// use rlst::{Linspace, rlst_dynamic_array};
    ///
    /// // p(x, y, z) = T_1(x) T_1(y) T_1(z) = x y z.
    /// let mut coeffs = rlst_dynamic_array!(f64, [2, 2, 2]);
    /// coeffs[[1, 1, 1]] = 1.0;
    ///
    /// let mut x = rlst_dynamic_array!(f64, [3]);
    /// let mut y = rlst_dynamic_array!(f64, [3]);
    /// let mut z = rlst_dynamic_array!(f64, [4]);
    /// x.linspace(-1.0, 1.0);
    /// y.linspace(-0.5, 0.5);
    /// z.linspace(0.0, 1.0);
    /// let mut values = rlst_dynamic_array!(f64, [3, 3, 4]);
    /// coeffs.tensor_evaluate_chebychev_series_second_kind(&x, &y, &z, &mut values);
    /// assert!((values[[2, 2, 3]] - 0.5).abs() < 1e-15);
    /// assert!((values[[0, 0, 2]] - 1.0 / 3.0).abs() < 1e-15);
    /// ```
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
        ArrayImplPointsX: ValueArrayImpl<Item::Real, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item::Real, 1>,
        ArrayImplPointsZ: ValueArrayImpl<Item::Real, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 3>;
}

/// Compute the series coefficients of the derivative of a Chebychev series
///
/// ```
/// use rlst::chebychev::DerivativeOfChebychevSeries1d;
/// use rlst::rlst_dynamic_array;
///
/// // x^2 = (T_0(x) + T_2(x)) / 2, whose derivative is 2 x = 2 T_1(x).
/// let mut coeffs = rlst_dynamic_array!(f64, [3]);
/// coeffs.fill_from_iter([0.5, 0.0, 0.5].into_iter());
/// coeffs.chebychev_derivative_second_kind(1);
/// for (c, expected) in coeffs.iter_value().zip([0.0, 2.0, 0.0]) {
///     assert!((c - expected).abs() < 1e-15);
/// }
/// ```
pub trait DerivativeOfChebychevSeries1d<Item> {
    /// Compute the series coefficients of the derivative of a chebychev series of the second kind.
    ///
    /// The coefficients of the derivative overwrite the original array without changing its shape.
    /// Coefficients above the differentiated polynomial's degree become zero.
    /// The parameter `order` determines how many derivatives are taken. `order=1` computes
    /// the first derivative and `order=0` leaves the coefficients unchanged.
    ///
    /// ```
    /// use rlst::chebychev::DerivativeOfChebychevSeries1d;
    /// use rlst::rlst_dynamic_array;
    ///
    /// // The second derivative of x^2 = (T_0(x) + T_2(x)) / 2 is 2 = 2 T_0(x).
    /// let mut coeffs = rlst_dynamic_array!(f64, [3]);
    /// coeffs.fill_from_iter([0.5, 0.0, 0.5].into_iter());
    /// coeffs.chebychev_derivative_second_kind(2);
    /// for (c, expected) in coeffs.iter_value().zip([2.0, 0.0, 0.0]) {
    ///     assert!((c - expected).abs() < 1e-15);
    /// }
    /// ```
    fn chebychev_derivative_second_kind(&mut self, order: usize);
}

/// Compute the series coefficients of the derivative of a two-dimensional Chebychev series
///
/// ```
/// use rlst::chebychev::DerivativeOfChebychevSeries2d;
/// use rlst::rlst_dynamic_array;
///
/// // x y^2 = T_1(x) (T_0(y) + T_2(y)) / 2; its derivative in x is y^2.
/// let mut coeffs = rlst_dynamic_array!(f64, [2, 3]);
/// coeffs[[1, 0]] = 0.5;
/// coeffs[[1, 2]] = 0.5;
/// coeffs.chebychev_derivative_second_kind([1, 0]);
/// assert!((coeffs[[0, 0]] - 0.5).abs() < 1e-15);
/// assert!((coeffs[[0, 2]] - 0.5).abs() < 1e-15);
/// assert!((coeffs.iter_value().map(f64::abs).sum::<f64>() - 1.0).abs() < 1e-15);
/// ```
pub trait DerivativeOfChebychevSeries2d<Item> {
    /// Compute the series coefficients of the derivative of a two dimensional Chebychev series of the second kind.
    ///
    /// The coefficients of the derivative overwrite the original array.
    /// The parameter `order` determines how many derivatives are taken. `order=[0, 2]` means
    /// that the output has no derivative along the x-axis and two derivatives
    /// along the y-axis.
    ///
    /// ```
    /// use rlst::chebychev::DerivativeOfChebychevSeries2d;
    /// use rlst::rlst_dynamic_array;
    ///
    /// // The derivative in x and y of x y^2 = T_1(x) (T_0(y) + T_2(y)) / 2 is 2 y = 2 T_1(y).
    /// let mut coeffs = rlst_dynamic_array!(f64, [2, 3]);
    /// coeffs[[1, 0]] = 0.5;
    /// coeffs[[1, 2]] = 0.5;
    /// coeffs.chebychev_derivative_second_kind([1, 1]);
    /// assert!((coeffs[[0, 1]] - 2.0).abs() < 1e-15);
    /// assert!((coeffs.iter_value().map(f64::abs).sum::<f64>() - 2.0).abs() < 1e-15);
    /// ```
    fn chebychev_derivative_second_kind(&mut self, order: [usize; 2]);
}

/// Compute the series coefficients of the derivative of a three-dimensional Chebychev series
///
/// ```
/// use rlst::chebychev::DerivativeOfChebychevSeries3d;
/// use rlst::rlst_dynamic_array;
///
/// // The derivative in z of x y z = T_1(x) T_1(y) T_1(z) is x y.
/// let mut coeffs = rlst_dynamic_array!(f64, [2, 2, 2]);
/// coeffs[[1, 1, 1]] = 1.0;
/// coeffs.chebychev_derivative_second_kind([0, 0, 1]);
/// assert!((coeffs[[1, 1, 0]] - 1.0).abs() < 1e-15);
/// assert!((coeffs.iter_value().map(f64::abs).sum::<f64>() - 1.0).abs() < 1e-15);
/// ```
pub trait DerivativeOfChebychevSeries3d<Item> {
    /// Compute the series coefficients of the derivative of a three-dimensional Chebychev series of the second kind.
    ///
    /// The coefficients of the derivative overwrite the original array.
    /// The parameter `order` determines how many derivatives are taken. `order=[0, 2, 1]` means
    /// that the output has no derivative along the x-axis, two derivatives
    /// along the y-axis and one derivative along the z axis.
    ///
    /// ```
    /// use rlst::chebychev::DerivativeOfChebychevSeries3d;
    /// use rlst::rlst_dynamic_array;
    ///
    /// // The derivative in x, y and z of x y z = T_1(x) T_1(y) T_1(z) is 1.
    /// let mut coeffs = rlst_dynamic_array!(f64, [2, 2, 2]);
    /// coeffs[[1, 1, 1]] = 1.0;
    /// coeffs.chebychev_derivative_second_kind([1, 1, 1]);
    /// assert!((coeffs[[0, 0, 0]] - 1.0).abs() < 1e-15);
    /// assert!((coeffs.iter_value().map(f64::abs).sum::<f64>() - 1.0).abs() < 1e-15);
    /// ```
    fn chebychev_derivative_second_kind(&mut self, order: [usize; 3]);
}

impl<Item, ArrayImpl> DerivativeOfChebychevSeries1d<Item> for Array<ArrayImpl, 1>
where
    Item: RlstScalar,
    ArrayImpl: MutableArrayImpl<Item, 1>,
{
    #[inline(always)]
    fn chebychev_derivative_second_kind(&mut self, order: usize) {
        let n = self.len();

        // If the order is zero there is nothing to do
        if order == 0 {
            return;
        }

        // We repeat the derivative loop `order` times

        for _ in 0..order {
            let mut bkp1 = Item::default(); // The series element k + 1;
            let mut dp1 = Item::default(); // The cofficient d_{k+1} of the derivative series.
            let mut dp2 = Item::default(); // The coefficient d_{k+2} of the derivative series.
            let two = Item::from_f64(2.0).unwrap();
            let half = two.recip();

            for (index, d) in self
                .r_mut()
                .reverse_axis(0)
                .iter_mut()
                .take(n - 1)
                .enumerate()
            {
                // The coefficients up to 1
                let kp1 = Item::from_usize(n - index).unwrap();
                let tmp = *d;
                *d = dp2 + two * kp1 * bkp1;
                bkp1 = tmp;
                (dp1, dp2) = (*d, dp1); // We need this in the next iteration
            }

            // Finally, the zero coefficient.

            self[[0]] = bkp1 + half * dp2;
        }
    }
}

impl<Item, ArrayImpl> DerivativeOfChebychevSeries2d<Item> for Array<ArrayImpl, 2>
where
    Item: RlstScalar,
    ArrayImpl: MutableArrayImpl<Item, 2>,
{
    fn chebychev_derivative_second_kind(&mut self, order: [usize; 2]) {
        // We first compute the derivatives in the x-direction.
        // The axes parameter `multislice` is 1 for this since we evaluate for each fixed y-value the derivative over all x-values

        for (_, mut slice) in self.multislice_iter_mut::<1, 1>([1]) {
            slice.chebychev_derivative_second_kind(order[0]);
        }

        // And now the derivatives in the y-direction
        for (_, mut slice) in self.multislice_iter_mut::<1, 1>([0]) {
            slice.chebychev_derivative_second_kind(order[1]);
        }
    }
}

impl<Item, ArrayImpl> DerivativeOfChebychevSeries3d<Item> for Array<ArrayImpl, 3>
where
    Item: RlstScalar,
    ArrayImpl: MutableArrayImpl<Item, 3>,
{
    fn chebychev_derivative_second_kind(&mut self, order: [usize; 3]) {
        // We first compute the derivatives in the x-direction.
        // The axes parameter `multislice` is 1 for this since we evaluate for each fixed y-value the derivative over all x-values

        for (_, mut slice) in self.multislice_iter_mut::<2, 1>([1, 2]) {
            slice.chebychev_derivative_second_kind(order[0]);
        }

        // And now the derivatives in the y-direction
        for (_, mut slice) in self.multislice_iter_mut::<2, 1>([0, 2]) {
            slice.chebychev_derivative_second_kind(order[1]);
        }

        // And finally in the z direction
        for (_, mut slice) in self.multislice_iter_mut::<2, 1>([0, 1]) {
            slice.chebychev_derivative_second_kind(order[2]);
        }
    }
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
        ArrayImplPoints: ValueArrayImpl<Item::Real, 1>,
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
            let x = Item::from_real(x);
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
    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 2>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<Item::Real, 2>,
        ArrayImplResult: MutableArrayImpl<Item, 1>,
    {
        // We check that the dimensions of `eval_points` and `result` are compatible.

        let points_shape = eval_points.shape();
        let npoints = points_shape[1];
        let points_dim = points_shape[0];

        assert_eq!(npoints, result.len());
        assert_eq!(points_dim, 2);

        // Now evaluate the points successively along each dimension

        let mut p0 =
            Array::<_, 1>::new(BaseArray::new(ArrayContainer::<Item::Real, 1>::new(), [1]));
        let mut p1 =
            Array::<_, 1>::new(BaseArray::new(ArrayContainer::<Item::Real, 1>::new(), [1]));
        let mut res = Array::<_, 2>::new(BaseArray::new(ArrayContainer::<Item, 1>::new(), [1, 1]));

        // For each point we evaluate along all dimensions

        for (point, r) in izip!(eval_points.col_iter(), result.iter_mut()) {
            p0[[0]] = point.get_value([0]).unwrap();
            p1[[0]] = point.get_value([1]).unwrap();
            self.tensor_evaluate_chebychev_series_second_kind(&p0, &p1, &mut res);

            *r = res[[0, 0]];
        }
    }

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
        ArrayImplPointsX: ValueArrayImpl<Item::Real, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item::Real, 1>,
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

impl<Item, ArrayImpl> EvaluateChebychevCoefficients3d<Item> for Array<ArrayImpl, 3>
where
    Item: RlstScalar,
    ArrayImpl: ValueArrayImpl<Item, 3>,
{
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
        ArrayImplPointsX: ValueArrayImpl<Item::Real, 1>,
        ArrayImplPointsY: ValueArrayImpl<Item::Real, 1>,
        ArrayImplPointsZ: ValueArrayImpl<Item::Real, 1>,
        ArrayImplResult: MutableArrayImpl<Item, 3>,
    {
        // Check that the dimensions are correct

        let [mx, my, mz] = result.shape();

        let [_nx, ny, nz] = self.shape();

        assert_eq!(mx, eval_points_x.len());
        assert_eq!(my, eval_points_y.len());
        assert_eq!(mz, eval_points_z.len());

        // We first evaluate along the x direction.

        let mut tmp_x = DynArray::<Item, 3>::from_shape([mx, ny, nz]);

        for (indices, slice) in self.r().multislice_iter::<_, 1>([1, 2]) {
            slice.evaluate_chebychev_series_second_kind(
                eval_points_x,
                &mut tmp_x.r_mut().multislice([1, 2], indices),
            );
        }

        // Now we evaluate along the y direction

        let mut tmp_y = DynArray::<Item, 3>::from_shape([mx, my, nz]);

        for (indices, slice) in tmp_x.r().multislice_iter::<_, 1>([0, 2]) {
            slice.evaluate_chebychev_series_second_kind(
                eval_points_y,
                &mut tmp_y.r_mut().multislice([0, 2], indices),
            );
        }

        // And now along the z direction

        for (indices, slice) in tmp_y.r().multislice_iter::<_, 1>([0, 1]) {
            slice.evaluate_chebychev_series_second_kind(
                eval_points_z,
                &mut result.r_mut().multislice([0, 1], indices),
            );
        }
    }

    fn evaluate_chebychev_series_second_kind<ArrayImplPoints, ArrayImplResult>(
        &self,
        eval_points: &Array<ArrayImplPoints, 2>,
        result: &mut Array<ArrayImplResult, 1>,
    ) where
        ArrayImplPoints: ValueArrayImpl<<Item as RlstScalar>::Real, 2>,
        ArrayImplResult: MutableArrayImpl<Item, 1>,
    {
        // We check that the dimensions of `eval_points` and `result` are compatible.

        let points_shape = eval_points.shape();
        let npoints = points_shape[1];
        let points_dim = points_shape[0];

        assert_eq!(npoints, result.len());
        assert_eq!(points_dim, 3);

        // Now evaluate the points successively along each dimension

        let mut p0 =
            Array::<_, 1>::new(BaseArray::new(ArrayContainer::<Item::Real, 1>::new(), [1]));
        let mut p1 =
            Array::<_, 1>::new(BaseArray::new(ArrayContainer::<Item::Real, 1>::new(), [1]));
        let mut p2 =
            Array::<_, 1>::new(BaseArray::new(ArrayContainer::<Item::Real, 1>::new(), [1]));
        let mut res =
            Array::<_, 3>::new(BaseArray::new(ArrayContainer::<Item, 1>::new(), [1, 1, 1]));

        // For each point we evaluate along all dimensions

        for (point, r) in izip!(eval_points.col_iter(), result.iter_mut()) {
            p0[[0]] = point.get_value([0]).unwrap();
            p1[[0]] = point.get_value([1]).unwrap();
            p2[[0]] = point.get_value([2]).unwrap();
            self.tensor_evaluate_chebychev_series_second_kind(&p0, &p1, &p2, &mut res);

            *r = res[[0, 0, 0]];
        }
    }
}

macro_rules! impl_cheb_coeffs_complex {
    ($dtype:ty) => {
        impl<ArrayImpl, const NDIM: usize> ChebychevCoefficientsToData<$dtype, NDIM>
            for Array<ArrayImpl, NDIM>
        where
            ArrayImpl: ValueArrayImpl<$dtype, NDIM>,
        {
            fn chebychev_coeffs_from_data_second_kind(
                &self,
            ) -> Result<DynArray<$dtype, NDIM>, FftwError> {
                let shape = self.shape();
                let mut re = DynArray::<<$dtype as RlstScalar>::Real, NDIM>::from_shape(shape);
                let mut im = DynArray::<<$dtype as RlstScalar>::Real, NDIM>::from_shape(shape);

                re.fill_from(&self.r().unary_op(|elem: $dtype| elem.re()));
                im.fill_from(&self.r().unary_op(|elem: $dtype| elem.im()));

                let re_coeffs = re.chebychev_coeffs_from_data_second_kind()?;
                let im_coeffs = im.chebychev_coeffs_from_data_second_kind()?;

                let mut out = DynArray::<$dtype, NDIM>::from_shape(shape);
                for (o, r, i) in izip!(
                    out.iter_mut(),
                    re_coeffs.iter_value(),
                    im_coeffs.iter_value()
                ) {
                    *o = <$dtype>::new(r, i);
                }
                Ok(out)
            }

            fn chebychev_data_from_coeffs_second_kind(
                &self,
            ) -> Result<DynArray<$dtype, NDIM>, FftwError> {
                let shape = self.shape();
                let mut re = DynArray::<<$dtype as RlstScalar>::Real, NDIM>::from_shape(shape);
                let mut im = DynArray::<<$dtype as RlstScalar>::Real, NDIM>::from_shape(shape);

                re.fill_from(&self.r().unary_op(|elem: $dtype| elem.re()));
                im.fill_from(&self.r().unary_op(|elem: $dtype| elem.im()));

                let re_data = re.chebychev_data_from_coeffs_second_kind()?;
                let im_data = im.chebychev_data_from_coeffs_second_kind()?;

                let mut out = DynArray::<$dtype, NDIM>::from_shape(shape);
                for (o, r, i) in izip!(out.iter_mut(), re_data.iter_value(), im_data.iter_value()) {
                    *o = <$dtype>::new(r, i);
                }
                Ok(out)
            }
        }
    };
}

impl_cheb_coeffs_complex!(c64);
impl_cheb_coeffs_complex!(c32);

/// A DCT-I (`REDFT00`) along `axis` of `data`, in place, batched over the other axes.
///
/// The plan is made on `data` with `Rigor::Estimate`, the planner's default, so FFTW does not
/// write `data` while planning, and the plan is made for the alignment class of its buffer,
/// which a `DynArray` does not fix. `axis` has an extent above 1: REDFT00 is not defined for
/// one point.
fn redft00_along_axis<T: FftwReal, const NDIM: usize>(
    data: &mut DynArray<T, NDIM>,
    axis: usize,
) -> Result<(), FftwError> {
    let layout = Layout::along_axis_inplace(data, axis)?;
    let plan = Planner::<T>::new().r2r_inplace_on(data, &layout, &[R2rKind::Redft00])?;
    plan.execute(data)
}

/// For each element of a column-major array of `shape`, in memory order, the number of axes of
/// extent above 1 along which it is at the first or the last index.
///
/// `shape` has no extent 0.
fn boundary_counts<const NDIM: usize>(shape: [usize; NDIM]) -> impl Iterator<Item = usize> {
    (0..shape.iter().product()).map(move |linear| {
        let mut rest = linear;
        let mut count = 0;
        for n in shape {
            let index = rest % n;
            rest /= n;
            if n > 1 && (index == 0 || index == n - 1) {
                count += 1;
            }
        }
        count
    })
}

macro_rules! impl_cheb_coeffs {
    ($dtype:ty) => {
        impl<ArrayImpl, const NDIM: usize> ChebychevCoefficientsToData<$dtype, NDIM>
            for Array<ArrayImpl, NDIM>
        where
            ArrayImpl: ValueArrayImpl<$dtype, NDIM>,
        {
            fn chebychev_coeffs_from_data_second_kind(
                &self,
            ) -> Result<DynArray<$dtype, NDIM>, FftwError> {
                let mut result = DynArray::<$dtype, NDIM>::from_shape(self.shape());
                result.fill_from(self);

                let shape = result.shape();

                // An empty array has nothing to transform.
                if shape.contains(&0) {
                    return Ok(result);
                }

                // Compute the discrete cosine transform of the data along each axis. An axis of
                // extent 1 is left as it is: one Chebychev point gives a constant interpolant.
                let mut denominator = 1;
                for (axis, &n) in shape.iter().enumerate() {
                    if n > 1 {
                        redft00_along_axis(&mut result, axis)?;
                        denominator *= n - 1;
                    }
                }

                // We scale all values by 1 / ∏(n - 1) over the axes of extent above 1, and
                // halve the boundary values of each such axis.
                let n_recip = (denominator as $dtype).recip();
                for (value, count) in izip!(result.data_mut().unwrap(), boundary_counts(shape)) {
                    *value *= n_recip;
                    for _ in 0..count {
                        *value /= 2.0;
                    }
                }

                Ok(result)
            }

            fn chebychev_data_from_coeffs_second_kind(
                &self,
            ) -> Result<DynArray<$dtype, NDIM>, FftwError> {
                let mut result = DynArray::<$dtype, NDIM>::from_shape(self.shape());
                result.fill_from(self);

                let shape = result.shape();

                // An empty array has nothing to transform.
                if shape.contains(&0) {
                    return Ok(result);
                }

                // Halve the interior values along each axis of extent above 1. The discrete
                // cosine transform then gives the boundary values once and the interior ones
                // twice, which the halving neutralises. This is easier than trying to scale
                // only the transform's interior terms.
                let axes = shape.iter().filter(|&&n| n > 1).count();
                for (value, count) in izip!(result.data_mut().unwrap(), boundary_counts(shape)) {
                    for _ in count..axes {
                        *value /= 2.0;
                    }
                }

                // Compute the discrete cosine transform of the data along each axis of extent
                // above 1; along an axis of extent 1, data and coefficient are equal.
                for (axis, &n) in shape.iter().enumerate() {
                    if n > 1 {
                        redft00_along_axis(&mut result, axis)?;
                    }
                }

                Ok(result)
            }
        }
    };
}

impl_cheb_coeffs!(f64);
impl_cheb_coeffs!(f32);

#[cfg(test)]
mod test {
    use num::Complex;
    use rand_distr::{Distribution, StandardNormal, StandardUniform};

    use crate::{
        EvaluateArray, Linspace, RandScalar, ZeroDimArray, assert_array_relative_eq,
        dense::tools::PrettyPrint,
        interpolation::{
            Kind, barycentric_chebychev_weights, barycentric_evaluate_2d, barycentric_evaluate_3d,
            chebychev_points, tensor_barycentric_evaluate_2d, tensor_barycentric_evaluate_3d,
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

        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        crate::assert_array_relative_eq!(coeffs, one, 1E-10);
        crate::assert_array_relative_eq!(data, data_out, 1E-10);
    }

    #[test]
    pub fn test_data_to_coeffs_1d_f32() {
        let n = 3;

        let points = chebychev_points::<f32>(Kind::Second, n);
        let mut one = DynArray::<f32, _>::from_shape([n]);
        one.fill_with_value(1.0);

        let data = 2.0_f32 * points.r() * points.r() + points.r();

        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        crate::assert_array_relative_eq!(coeffs, one, 1E-5);
        crate::assert_array_relative_eq!(data, data_out, 1E-5);
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

        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();

        coeffs.pretty_print();

        crate::assert_array_relative_eq!(coeffs_expected, coeffs, 1E-10);

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

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

        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();

        crate::assert_array_relative_eq!(coeffs_expected, coeffs, 1E-10);

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        crate::assert_array_abs_diff_eq!(data, data_out, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_1d() {
        let n = 11;

        let mut expected = DynArray::<f64, _>::from_shape([n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind().unwrap();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_complex_random_data_to_coefficients_and_back_1d() {
        let n = 11;

        let mut expected = DynArray::<c64, _>::from_shape([n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind().unwrap();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_2d() {
        let n = 5;
        let m = 7;

        let mut expected = DynArray::<f64, _>::from_shape([m, n]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind().unwrap();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        crate::assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_random_data_to_coefficients_and_back_3d() {
        let m = 7;
        let n = 5;
        let k = 8;

        let mut expected = DynArray::<f64, _>::from_shape([m, n, k]);
        expected.fill_from_seed_normally_distributed(0);

        let coeffs = expected.chebychev_coeffs_from_data_second_kind().unwrap();

        let actual = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

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
    pub fn test_tensor_eval_cheb_series_2d_second_kind() {
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

        // we fill the coefficients with random values. we then compare the direct evaluation with
        // the evaluation obtained by a dct to physical space, followed by barycentric evaluation.

        let mut actual = DynArray::<f64, 2>::from_shape([mx, my]);
        coeffs.tensor_evaluate_chebychev_series_second_kind(
            &eval_points_x,
            &eval_points_y,
            &mut actual,
        );

        // we now compute the same by doing a dft followed by barycentric evaluation.

        let data = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

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

        assert_array_relative_eq!(actual, expected, 1e-10);
    }

    #[test]
    pub fn test_eval_cheb_series_2d_second_kind() {
        let nx = 5;
        let ny = 10;
        let m = 60;

        let mut eval_points = DynArray::<f64, 2>::from_shape([2, m]);
        eval_points.fill_from_seed_equally_distributed(0);
        for value in eval_points.iter_mut() {
            // Shift random values to [-1, 1]
            *value = 2.0 * *value - 1.0
        }

        let mut coeffs = DynArray::<f64, 2>::from_shape([nx, ny]);
        coeffs.fill_from_seed_normally_distributed(0);

        // we fill the coefficients with random values. we then compare the direct evaluation with
        // the evaluation obtained by a dct to physical space, followed by barycentric evaluation.

        let mut actual = DynArray::<f64, 1>::from_shape([m]);
        coeffs.evaluate_chebychev_series_second_kind(&eval_points, &mut actual);
        let data = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        let mut expected = DynArray::<f64, 1>::from_shape([m]);

        let cheb_points_x = chebychev_points(Kind::Second, nx);
        let cheb_points_y = chebychev_points(Kind::Second, ny);
        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);

        barycentric_evaluate_2d(
            &eval_points,
            &cheb_points_x,
            &cheb_points_y,
            &weights_x,
            &weights_y,
            &data,
            &mut expected,
        );

        assert_array_relative_eq!(actual, expected, 1e-10);
    }

    #[test]
    pub fn test_tensor_eval_cheb_series_3d_second_kind() {
        let nx = 5;
        let ny = 10;
        let nz = 13;
        let mx = 18;
        let my = 19;
        let mz = 14;

        let mut eval_points_x = DynArray::<f64, 1>::from_shape([mx]);
        let mut eval_points_y = DynArray::<f64, 1>::from_shape([my]);
        let mut eval_points_z = DynArray::<f64, 1>::from_shape([mz]);
        eval_points_x.linspace(-1.0, 1.0);
        eval_points_y.linspace(-1.0, 1.0);
        eval_points_z.linspace(-1.0, 1.0);

        let mut coeffs = DynArray::<f64, 3>::from_shape([nx, ny, nz]);
        coeffs.fill_from_seed_equally_distributed(0);

        // we fill the coefficients with random values. we then compare the direct evaluation with
        // the evaluation obtained by a dct to physical space, followed by barycentric evaluation.

        let mut actual = DynArray::<f64, 3>::from_shape([mx, my, mz]);
        coeffs.tensor_evaluate_chebychev_series_second_kind(
            &eval_points_x,
            &eval_points_y,
            &eval_points_z,
            &mut actual,
        );

        // We now compute the same by doing a DFT followed by barycentric evaluation.

        let data = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        let mut expected = DynArray::<f64, 3>::from_shape([mx, my, mz]);

        let cheb_points_x = chebychev_points(Kind::Second, nx);
        let cheb_points_y = chebychev_points(Kind::Second, ny);
        let cheb_points_z = chebychev_points(Kind::Second, nz);
        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);
        let weights_z = barycentric_chebychev_weights::<f64>(Kind::Second, nz);

        tensor_barycentric_evaluate_3d(
            &eval_points_x,
            &eval_points_y,
            &eval_points_z,
            &cheb_points_x,
            &cheb_points_y,
            &cheb_points_z,
            &weights_x,
            &weights_y,
            &weights_z,
            &data,
            &mut expected,
        );

        assert_array_relative_eq!(actual, expected, 1E-10);
    }

    #[test]
    pub fn test_eval_cheb_series_3d_second_kind() {
        let nx = 5;
        let ny = 10;
        let nz = 13;
        let m = 60;

        let mut eval_points = DynArray::<f64, 2>::from_shape([3, m]);
        eval_points.fill_from_seed_equally_distributed(0);
        for value in eval_points.iter_mut() {
            // Shift random values to [-1, 1]
            *value = 2.0 * *value - 1.0
        }

        let mut coeffs = DynArray::<f64, 3>::from_shape([nx, ny, nz]);
        coeffs.fill_from_seed_normally_distributed(0);

        // we fill the coefficients with random values. we then compare the direct evaluation with
        // the evaluation obtained by a dct to physical space, followed by barycentric evaluation.

        let mut actual = DynArray::<f64, 1>::from_shape([m]);
        coeffs.evaluate_chebychev_series_second_kind(&eval_points, &mut actual);
        let data = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();

        let mut expected = DynArray::<f64, 1>::from_shape([m]);

        let cheb_points_x = chebychev_points(Kind::Second, nx);
        let cheb_points_y = chebychev_points(Kind::Second, ny);
        let cheb_points_z = chebychev_points(Kind::Second, nz);
        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);
        let weights_z = barycentric_chebychev_weights::<f64>(Kind::Second, nz);

        barycentric_evaluate_3d(
            &eval_points,
            &cheb_points_x,
            &cheb_points_y,
            &cheb_points_z,
            &weights_x,
            &weights_y,
            &weights_z,
            &data,
            &mut expected,
        );

        assert_array_relative_eq!(actual, expected, 1e-10);
    }

    #[test]
    fn test_derivative_1d() {
        // We sample the sine function from -1 to 1, compute the coefficients, take the derivative
        // and transform back. The result is compared with the cosine function.

        let n = 20; // Order of the Chebychev expansion

        let points = chebychev_points::<f64>(Kind::Second, n);

        // We fill the data array
        let mut values = points.eval();
        values.iter_mut().for_each(|elem| *elem = elem.sin());

        // and the array of expected values.
        let mut expected = points.eval();
        expected.iter_mut().for_each(|elem| *elem = elem.cos());

        // We compute the coefficients of the expansion
        let coeffs = values.chebychev_coeffs_from_data_second_kind().unwrap();

        // Now compute the derivative of the coefficients
        let mut deriv = coeffs.eval();
        deriv.chebychev_derivative_second_kind(1);

        // We transform back
        let values_deriv = deriv.chebychev_data_from_coeffs_second_kind().unwrap();

        // We compare with the sine of the original values.

        crate::assert_array_relative_eq!(values_deriv, expected, 1E-10);

        // Now check the second derivative
        let mut deriv_2 = coeffs.eval();
        deriv_2.chebychev_derivative_second_kind(2);

        let values_deriv_2 = deriv_2.chebychev_data_from_coeffs_second_kind().unwrap();

        // The result should be close to -sin(x)
        let mut expected = points.eval();
        expected.iter_mut().for_each(|elem| *elem = -elem.sin());

        crate::assert_array_relative_eq!(values_deriv_2, expected, 1E-10);
    }

    #[test]
    fn test_derivative_complex_1d() {
        // We sample the sine function from -1 to 1, compute the coefficients, take the derivative
        // and transform back. The result is compared with the cosine function.

        let n = 30; // Order of the Chebychev expansion

        let points = chebychev_points::<f64>(Kind::Second, n);

        // We fill the data array
        let mut values = points.r().cast::<c64>().eval();
        values
            .iter_mut()
            .for_each(|elem| *elem = Complex::new(elem.re().sin(), (5.0_f64 * elem.re()).cos()));

        // and the array of expected values.
        let mut expected = points.r().cast::<c64>().eval();
        expected.iter_mut().for_each(|elem| {
            *elem = Complex::new(elem.re().cos(), -5.0_f64 * (5.0_f64 * elem.re()).sin())
        });

        // We compute the coefficients of the expansion
        let coeffs = values.chebychev_coeffs_from_data_second_kind().unwrap();

        // Now compute the derivative of the coefficients
        let mut deriv = coeffs.eval();
        deriv.chebychev_derivative_second_kind(1);

        // We transform back
        let values_deriv = deriv.chebychev_data_from_coeffs_second_kind().unwrap();

        // We compare with the sine of the original values.

        crate::assert_array_relative_eq!(values_deriv, expected, 1E-10);
    }

    #[test]
    fn test_derivative_2d() {
        // We sample sin(x * y) in the unit cube, compute the coefficients, take two derivatives
        // with respect to y and one derivative with respect to x
        // and transform back. The result is compared with the analytical derivative.

        let nx = 20; // Order of the Chebychev expansion
        let ny = 30; // Order of the Chebychev expansion

        let points_x = chebychev_points::<f64>(Kind::Second, nx);
        let points_y = chebychev_points::<f64>(Kind::Second, ny);

        // We fill the data array
        let mut values = DynArray::from_shape([nx, ny]);
        for (indx, px) in points_x.iter_value().enumerate() {
            for (indy, py) in points_y.iter_value().enumerate() {
                values[[indx, indy]] = f64::sin(px * py);
            }
        }

        // and the array of expected values.
        let mut expected = DynArray::from_shape([nx, ny]);
        for (indx, px) in points_x.iter_value().enumerate() {
            for (indy, py) in points_y.iter_value().enumerate() {
                expected[[indx, indy]] =
                    // 2 derivatives with respect to y and one derivative with respect to x
                    -px * px * py * f64::cos(px * py) - 2.0 * px * f64::sin(px * py);
            }
        }

        // We compute the coefficients of the expansion
        let coeffs = values.chebychev_coeffs_from_data_second_kind().unwrap();

        // Now compute the derivative of the coefficients
        let mut deriv = coeffs.eval();
        deriv.chebychev_derivative_second_kind([1, 2]);

        // We transform back
        let values_deriv = deriv.chebychev_data_from_coeffs_second_kind().unwrap();

        // We compare with the sine of the original values.

        crate::assert_array_relative_eq!(values_deriv, expected, 1E-8);
    }

    #[test]
    fn test_derivative_3d() {
        // We sample sin(x * y * z) in the unit cube, compute the coefficients, take two derivatives
        // with respect to y, one derivative with respect to x, and 3 derivatives with
        // respect to z and transform back. The result is compared with the analytical derivative.

        let nx = 14; // Order of the Chebychev expansion
        let ny = 15; // Order of the Chebychev expansion
        let nz = 16; // Order of the Chebychev expansion

        let points_x = chebychev_points::<f64>(Kind::Second, nx);
        let points_y = chebychev_points::<f64>(Kind::Second, ny);
        let points_z = chebychev_points::<f64>(Kind::Second, nz);

        // We fill the data array
        let mut values = DynArray::from_shape([nx, ny, nz]);
        for (indx, px) in points_x.iter_value().enumerate() {
            for (indy, py) in points_y.iter_value().enumerate() {
                for (indz, pz) in points_z.iter_value().enumerate() {
                    values[[indx, indy, indz]] = f64::sin(px * py * pz);
                }
            }
        }

        // and the array of expected values.
        let mut expected = DynArray::from_shape([nx, ny, nz]);
        for (indx, px) in points_x.iter_value().enumerate() {
            for (indy, py) in points_y.iter_value().enumerate() {
                for (indz, pz) in points_z.iter_value().enumerate() {
                    expected[[indx, indy, indz]] = -18.0 * px * px * py * f64::cos(px * py * pz)
                        + px * px
                            * px
                            * px
                            * py
                            * py
                            * py
                            * 11.0
                            * pz
                            * pz
                            * f64::cos(px * py * pz)
                        + px * px * px * py * py * 30.0 * pz * f64::sin(px * py * pz)
                        - px * px
                            * px
                            * px
                            * px
                            * py
                            * py
                            * py
                            * py
                            * pz
                            * pz
                            * pz
                            * f64::sin(px * py * pz);
                }
                // 2 derivatives with respect to y and one derivative with respect to x
            }
        }

        // We compute the coefficients of the expansion
        let coeffs = values.chebychev_coeffs_from_data_second_kind().unwrap();

        // Now compute the derivative of the coefficients
        let mut deriv = coeffs.eval();
        deriv.chebychev_derivative_second_kind([1, 2, 3]);

        // We transform back
        let values_deriv = deriv.chebychev_data_from_coeffs_second_kind().unwrap();

        // We compare with the sine of the original values.

        crate::assert_array_abs_diff_eq!(values_deriv, expected, 1E-4);
    }

    /// The largest absolute difference of two arrays, relative to the largest absolute value
    /// of `expected`.
    fn relative_difference<T: RlstScalar, const NDIM: usize>(
        actual: &DynArray<T, NDIM>,
        expected: &DynArray<T, NDIM>,
    ) -> T::Real {
        assert_eq!(actual.shape(), expected.shape());
        let mut difference = T::Real::default();
        let mut scale = T::Real::default();
        for (a, e) in izip!(actual.iter_value(), expected.iter_value()) {
            if (a - e).abs() > difference {
                difference = (a - e).abs();
            }
            if e.abs() > scale {
                scale = e.abs();
            }
        }
        difference / scale
    }

    /// Both conversions of an array with an axis of extent 1 equal those of the array without
    /// that axis, `squeezed`, which has the same elements in the same order.
    fn check_extent_one<T, const NDIM: usize, const M: usize>(
        shape: [usize; NDIM],
        squeezed: [usize; M],
        tol: T::Real,
    ) where
        T: RlstScalar + RandScalar,
        StandardNormal: Distribution<T::Real>,
        StandardUniform: Distribution<T::Real>,
        DynArray<T, NDIM>: ChebychevCoefficientsToData<T, NDIM>,
        DynArray<T, M>: ChebychevCoefficientsToData<T, M>,
    {
        let mut data = DynArray::<T, NDIM>::from_shape(shape);
        data.fill_from_seed_normally_distributed(1);
        let mut data_squeezed = DynArray::<T, M>::from_shape(squeezed);
        data_squeezed.fill_from_iter(data.iter_value());

        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        let coeffs_squeezed = data_squeezed
            .chebychev_coeffs_from_data_second_kind()
            .unwrap();
        let mut expected = DynArray::<T, NDIM>::from_shape(shape);
        expected.fill_from_iter(coeffs_squeezed.iter_value());
        assert!(
            relative_difference(&coeffs, &expected) <= tol,
            "coefficients for shape {shape:?}"
        );

        let back = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();
        let back_squeezed = coeffs_squeezed
            .chebychev_data_from_coeffs_second_kind()
            .unwrap();
        let mut expected = DynArray::<T, NDIM>::from_shape(shape);
        expected.fill_from_iter(back_squeezed.iter_value());
        assert!(
            relative_difference(&back, &expected) <= tol,
            "data for shape {shape:?}"
        );
        assert!(
            relative_difference(&back, &data) <= tol,
            "round trip for shape {shape:?}"
        );
    }

    #[test]
    fn test_extent_one_1d() {
        // One point: data and coefficient are equal.
        let mut data = DynArray::<f64, 1>::from_shape([1]);
        data[[0]] = 0.75;
        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        assert_eq!(coeffs[[0]], 0.75);
        let back = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();
        assert_eq!(back[[0]], 0.75);

        let mut data = DynArray::<c32, 1>::from_shape([1]);
        data[[0]] = c32::new(0.5, -2.0);
        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        assert_eq!(coeffs[[0]], c32::new(0.5, -2.0));
        let back = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();
        assert_eq!(back[[0]], c32::new(0.5, -2.0));
    }

    #[test]
    fn test_extent_one_2d() {
        // Data constant along the axis of extent 1: the coefficients of the other axis.
        let points = chebychev_points::<f64>(Kind::Second, 3);
        let mut data = DynArray::<f64, 2>::from_shape([1, 3]);
        for (value, x) in izip!(data.iter_mut(), points.iter_value()) {
            *value = 2.0 * x * x + x;
        }
        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        let mut one = DynArray::<f64, 2>::from_shape([1, 3]);
        one.fill_with_value(1.0);
        crate::assert_array_relative_eq!(coeffs, one, 1E-14);

        for (shape, squeezed) in [([1, 5], [5]), ([5, 1], [5]), ([1, 1], [1])] {
            check_extent_one::<f64, 2, 1>(shape, squeezed, 1E-14);
            check_extent_one::<f32, 2, 1>(shape, squeezed, 1E-6);
            check_extent_one::<c64, 2, 1>(shape, squeezed, 1E-14);
            check_extent_one::<c32, 2, 1>(shape, squeezed, 1E-6);
        }
    }

    #[test]
    fn test_extent_one_3d() {
        for (shape, squeezed) in [
            ([1, 5, 4], [5, 4]),
            ([5, 1, 4], [5, 4]),
            ([5, 4, 1], [5, 4]),
            ([1, 1, 4], [1, 4]),
        ] {
            check_extent_one::<f64, 3, 2>(shape, squeezed, 1E-14);
            check_extent_one::<f32, 3, 2>(shape, squeezed, 1E-6);
            check_extent_one::<c64, 3, 2>(shape, squeezed, 1E-14);
            check_extent_one::<c32, 3, 2>(shape, squeezed, 1E-6);
        }
    }

    #[test]
    fn test_rank_zero() {
        // No axis: both conversions return a copy, without calling FFTW.
        let data = ZeroDimArray::from_scalar(0.75_f64);
        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        assert_eq!(coeffs.into_scalar(), 0.75);
        let back = data.chebychev_data_from_coeffs_second_kind().unwrap();
        assert_eq!(back.into_scalar(), 0.75);

        let data = ZeroDimArray::from_scalar(c32::new(0.5, -2.0));
        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        assert_eq!(coeffs.into_scalar(), c32::new(0.5, -2.0));
        let back = data.chebychev_data_from_coeffs_second_kind().unwrap();
        assert_eq!(back.into_scalar(), c32::new(0.5, -2.0));

        // An axis of extent 1 is the same as no axis.
        check_extent_one::<f64, 1, 0>([1], [], 0.0);
        check_extent_one::<f32, 1, 0>([1], [], 0.0);
        check_extent_one::<c64, 1, 0>([1], [], 0.0);
        check_extent_one::<c32, 1, 0>([1], [], 0.0);
    }

    #[test]
    fn test_extent_zero() {
        fn check<T, const NDIM: usize>(shape: [usize; NDIM])
        where
            T: RlstScalar,
            DynArray<T, NDIM>: ChebychevCoefficientsToData<T, NDIM>,
        {
            let data = DynArray::<T, NDIM>::from_shape(shape);
            let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
            assert_eq!(coeffs.shape(), shape);
            let back = data.chebychev_data_from_coeffs_second_kind().unwrap();
            assert_eq!(back.shape(), shape);
        }

        check::<f64, 1>([0]);
        check::<f32, 2>([3, 0]);
        check::<c64, 3>([0, 4, 2]);
        check::<c32, 3>([1, 0, 1]);
        check::<f64, 4>([2, 3, 0, 1]);
    }

    #[test]
    fn test_data_to_coeffs_4d() {
        // f = T_2(x_0) + T_1(x_1) T_1(x_2) + 3 T_3(x_3).
        let shape = [3, 2, 2, 4];
        let points = shape.map(|n| chebychev_points::<f64>(Kind::Second, n));
        let mut data = DynArray::<f64, 4>::from_shape(shape);
        for (i, x0) in points[0].iter_value().enumerate() {
            for (j, x1) in points[1].iter_value().enumerate() {
                for (k, x2) in points[2].iter_value().enumerate() {
                    for (l, x3) in points[3].iter_value().enumerate() {
                        data[[i, j, k, l]] =
                            (2.0 * x0 * x0 - 1.0) + x1 * x2 + 3.0 * (4.0 * x3 * x3 * x3 - 3.0 * x3);
                    }
                }
            }
        }

        let mut coeffs_expected = DynArray::<f64, 4>::from_shape(shape);
        coeffs_expected[[2, 0, 0, 0]] = 1.0;
        coeffs_expected[[0, 1, 1, 0]] = 1.0;
        coeffs_expected[[0, 0, 0, 3]] = 3.0;

        let coeffs = data.chebychev_coeffs_from_data_second_kind().unwrap();
        crate::assert_array_abs_diff_eq!(coeffs, coeffs_expected, 1E-14);

        let data_out = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();
        crate::assert_array_abs_diff_eq!(data_out, data, 1E-14);
    }

    #[test]
    fn test_random_data_to_coefficients_and_back_4d_and_5d() {
        let mut expected = DynArray::<f64, 4>::from_shape([4, 1, 5, 3]);
        expected.fill_from_seed_normally_distributed(0);
        let coeffs = expected.chebychev_coeffs_from_data_second_kind().unwrap();
        let actual = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();
        crate::assert_array_relative_eq!(actual, expected, 1E-10);

        let mut expected = DynArray::<c32, 5>::from_shape([3, 2, 4, 2, 3]);
        expected.fill_from_seed_normally_distributed(0);
        let coeffs = expected.chebychev_coeffs_from_data_second_kind().unwrap();
        let actual = coeffs.chebychev_data_from_coeffs_second_kind().unwrap();
        assert!(relative_difference(&actual, &expected) < 1E-5);
    }
}
