//! Functions to evaluate interpolation polynomials.
//!
//! This module defines functions for the stable evaluation of Lagrange polynomials
//! using the barycentric interpolation formula. The following gives a simple example
//! in which we sample the function `cos(x)` at Chebychev points and then evaluate the
//! corresponding interpolation polynomial at a set of random points.
//!
//! ```
//! # use rand_chacha::ChaCha8Rng;
//! # use rand::{Rng, SeedableRng};
//! # use rlst::{
//! #     DynArray,
//! #     interpolation::{
//! #         Kind, barycentric_chebychev_weights, barycentric_evaluate_1d, barycentric_weights,
//! #         chebychev_points,
//! #     },
//! # };
//! // We interpolate at n=20 points, corresponding to a degree 19 interpolation polynomial.
//! let n = 20;
//!
//! // We want to evaluate the result at 1000 points.
//! let m = 1000;
//!
//! // Initialise the random number generator for the evaluation points.
//! let mut rng = ChaCha8Rng::seed_from_u64(0);
//!
//! // This returns the Chebychev points of the second kind.
//! let cheb_points = chebychev_points::<f64>(Kind::Second, n);
//!
//! // Sample the interpolation values at the Chebychev points.
//! let interp_values: DynArray<f64, 1> = cheb_points
//!     .iter_value()
//!     .map(|point| point.cos())
//!     .collect::<Vec<_>>()
//!     .into();
//!
//! // The following two lines create `m` random points in the interval [-1, 1]
//! // at which we will evaluate the interpolation polynomial.
//! let mut eval_points = DynArray::<f64, 1>::from_shape([m]);
//! eval_points
//!     .iter_mut()
//!     .for_each(|point| *point = 2.0 * rng.random::<f64>() - 1.0);
//!
//! // The barycentric formula requires barycentric weights.
//! // These are computed for Chebycev points as follows.
//! let weights = barycentric_chebychev_weights(Kind::Second, n);
//!
//! // Create an output array for the evaluation values.
//! let mut eval_values = DynArray::<f64, _>::from_shape([m]);
//!
//! // Evaluate the polynomial at the given points and write the result to `eval_values`.
//! barycentric_evaluate_1d(
//!     &eval_points,
//!     &cheb_points,
//!     &interp_values,
//!     &weights,
//!     &mut eval_values,
//! );
//!
//! ```
//! The core of this module is the routine [barycentric_evaluate_1d], which performs the
//! barycentric evaluation along a single dimension. Barycentric evaluation requires a vector
//! of weights associated with the interpolation points. For Chebychev points these are returned
//! in the function [barycentric_chebychev_weights]. For other point distributions the weights
//! can be computed through the function [barycentric_weights].
//!
//! The complexity of evaluating a Chebychev interpolation with `n` values at `m` evaluation points
//! has a complexity of `O(n * m)`. The direct computation of the weights with [barycentric_weights]
//! costs `O(n^2)`, while the generation of the Chebychev weights with `barycentric_chebycev_weights]`
//! costs only `O(n)`.

use itertools::izip;

use crate::{
    Array, DynArray, RlstScalar, Shape, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

use num::{One, traits::FloatConst};

/// Define whether to use Chebychev points of the first kind or second kind
pub enum Kind {
    /// Choose Chebychev points of the first kind.
    First,
    /// Choose Chebychev points of the second kind.
    Second,
}

/// Return the n Chebychev points of type `kind` in [-1, 1].
///
/// - The Chebychev points are returned in descending order.
/// - Chebychev points of the first kind include the end points. Chebychev points of the second kind don't.
/// - For Chebychev points of the second kind require `n > 1`.
pub fn chebychev_points<Item: RlstScalar>(kind: Kind, n: usize) -> DynArray<Item, 1> {
    let mut output = DynArray::<Item, 1>::from_shape([n]);
    let indices = (0..n).rev();

    let pi = num::cast::<_, Item>(f64::PI()).unwrap();

    match kind {
        Kind::First => {
            assert!(n > 0);
            let pi_div_two_n = pi / num::cast::<_, Item>(2 * n).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = Item::cos(num::cast::<_, Item>(2 * index + 1).unwrap() * pi_div_two_n);
            }
        }
        Kind::Second => {
            assert!(n > 1);
            let pi_div_nm1 = pi / num::cast::<_, Item>(n - 1).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = Item::cos(num::cast::<_, Item>(index).unwrap() * pi_div_nm1);
            }
        }
    }

    output
}

/// Compute a set of general barycentric weights given interpolation points
pub fn barycentric_weights<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
>(
    interp_points: &Array<ArrayImpl, 1>,
) -> DynArray<Item, 1> {
    let n = interp_points.len();

    let mut result = DynArray::<Item, 1>::from_shape([n]);

    unsafe {
        for j in 0..n {
            let xj = interp_points.get_value_unchecked([j]);
            let mut res: Item = Item::one();
            for i in 0..n {
                if i == j {
                    continue;
                } else {
                    res *= xj - interp_points.get_value_unchecked([i]);
                }
            }
            *result.get_unchecked_mut([j]) = res.recip();
        }
    }

    result
}

/// Return the barycentric weights required for barycentric interpolation
pub fn barycentric_chebychev_weights<T: RlstScalar>(kind: Kind, n: usize) -> DynArray<T, 1> {
    let mut output = DynArray::<T, 1>::from_shape([n]);
    match kind {
        Kind::First => {
            assert!(n > 0);
            let indices = (0..n).rev();
            let mut pm_one = <T as One>::one();
            let pi = num::cast::<_, T>(f64::PI()).unwrap();
            let pi_div_two_n = pi / num::cast::<_, T>(2 * n).unwrap();
            for (value, index) in izip!(output.iter_mut(), indices) {
                *value = pm_one
                    * <T as RlstScalar>::sin(
                        num::cast::<_, T>(2 * index + 1).unwrap() * pi_div_two_n,
                    );
                pm_one = -pm_one;
            }
        }
        Kind::Second => {
            assert!(n > 1);
            let mut pm_one = <T as One>::one();
            for value in output.iter_mut() {
                *value = pm_one;
                pm_one = -pm_one;
            }
            *output.data_mut().unwrap().first_mut().unwrap() *= num::cast(0.5).unwrap();
            *output.data_mut().unwrap().last_mut().unwrap() *= num::cast(0.5).unwrap();
        }
    }
    output
}

/// Barycentric evaluation in one dimension.
///
/// Evaluate an interpolation polynomial at a given set of evaluation points.
///
/// # Arguments
/// - `eval_points`: The points at which to evaluate the interpolation polynomial.
/// - `interp_nodes`: The interpolation nodes.
/// - `interp_values`: The values of the interpolation polynomial at the nodes.
/// - `interp_weights`: The barycentric weights of the interpolation formula.
///
/// The following routines are provides to compute the interpolation weights.
/// - [barycentric_weights]: Given an array of interpolation points directly compute the associated weights.
/// - [barycentric_chebychev_weights]: Use an analytic formula to compute the weights for Chebychev points of the first or second kind.
///
/// The output `result` is an array with the same length as the array `eval_points`.
pub fn barycentric_evaluate_1d<
    Item,
    ArrayImplEvalPoints,
    ArrayImplInterpNodes,
    ArrayImplInterpValues,
    ArrayImplInterpWeights,
    ArrayImplResult,
>(
    eval_points: &Array<ArrayImplEvalPoints, 1>,
    interp_nodes: &Array<ArrayImplInterpNodes, 1>,
    interp_values: &Array<ArrayImplInterpValues, 1>,
    interp_weights: &Array<ArrayImplInterpWeights, 1>,
    result: &mut Array<ArrayImplResult, 1>,
) where
    ArrayImplEvalPoints: UnsafeRandom1DAccessByValue<Item = Item::Real> + Shape<1>,
    ArrayImplInterpNodes: UnsafeRandom1DAccessByValue<Item = Item::Real> + Shape<1>,
    ArrayImplInterpValues: UnsafeRandomAccessByValue<1, Item = Item>
        + UnsafeRandom1DAccessByValue<Item = Item>
        + Shape<1>,
    ArrayImplInterpWeights: UnsafeRandom1DAccessByValue<Item = Item::Real> + Shape<1>,
    ArrayImplResult:
        UnsafeRandomAccessMut<1, Item = Item> + UnsafeRandom1DAccessMut<Item = Item> + Shape<1>,
    Item: RlstScalar,
{
    let npoints = eval_points.len();
    let nnodes = interp_nodes.len();

    assert_eq!(result.len(), npoints);
    assert_eq!(nnodes, interp_values.len());
    assert_eq!(nnodes, interp_weights.len());

    let mut nominator = vec![Item::zero(); npoints];
    let mut denominator = vec![Item::zero(); npoints];
    let mut exact: Vec<i32> = vec![-1; npoints];

    for (interp_node, interp_weight, (interp_value_index, interp_value)) in izip!(
        interp_nodes.iter_value(),
        interp_weights.iter_value(),
        interp_values.iter_value().enumerate()
    ) {
        for (nom, denom, e, eval_point) in izip!(
            nominator.iter_mut(),
            denominator.iter_mut(),
            exact.iter_mut(),
            eval_points.iter_value()
        ) {
            let diff = Item::from_real(eval_point - interp_node);
            if diff == Item::zero() {
                *e = interp_value_index as i32;
            }
            let inv_diff = Item::from_real(interp_weight) / diff;
            *nom += inv_diff * interp_value;
            *denom += inv_diff;
        }
    }
    for (&nom, &denom, res) in izip!(nominator.iter(), denominator.iter(), result.iter_mut()) {
        *res = nom / denom;
    }

    for (index, &val_index) in exact.iter().enumerate().filter(|(_, v)| **v != -1) {
        *result.get_mut([index]).unwrap() = interp_values.get_value([val_index as usize]).unwrap();
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use itertools::izip;
    use num::traits::FloatConst;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        DynArray,
        interpolation::{
            Kind, barycentric_chebychev_weights, barycentric_evaluate_1d, barycentric_weights,
            chebychev_points,
        },
    };

    #[test]
    fn test_cheb_points_first_kind() {
        let n = 5;

        let points = chebychev_points::<f64>(Kind::First, n);

        for (index, &value) in points.data().unwrap().iter().rev().enumerate() {
            assert_relative_eq!(
                value,
                f64::cos((2 * index + 1) as f64 * (f64::PI()) / (2.0 * n as f64)),
                epsilon = 1E-13
            );
        }
    }

    #[test]
    fn test_cheb_points_second_kind() {
        let n = 5;

        let values = chebychev_points::<f64>(Kind::Second, n);

        for (index, &value) in values.data().unwrap().iter().rev().enumerate() {
            assert_relative_eq!(
                value,
                f64::cos((index) as f64 * (f64::PI()) / (n - 1) as f64),
                epsilon = 1E-13
            );
        }
    }

    #[test]
    fn test_barycentric_weights_first_kind() {
        let n = 5;

        let points = chebychev_points::<f64>(Kind::First, n);

        let weights = barycentric_weights(&points);
        let actual_weights = barycentric_chebychev_weights::<f64>(Kind::First, n);

        // The normalisation is different, so make sure it fits.
        let factor = actual_weights[[0]] / weights[[0]];

        crate::assert_array_relative_eq!(factor * weights.r(), actual_weights, 1E-10);
    }

    #[test]
    fn test_barycentric_weights_second_kind() {
        let n = 5;

        let points = chebychev_points::<f64>(Kind::Second, n);

        let weights = barycentric_weights(&points);
        let actual_weights = barycentric_chebychev_weights::<f64>(Kind::Second, n);
        // The normalisation is different, so make sure it fits.
        let factor = actual_weights[[0]] / weights[[0]];

        crate::assert_array_relative_eq!(factor * weights.r(), actual_weights, 1E-10);
    }

    #[test]
    fn test_barycentric_interp_with_barycentric_first_kind() {
        let n = 20;
        let m = 1000;

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let cheb_points = chebychev_points::<f64>(Kind::First, n);
        let interp_values: DynArray<f64, 1> = cheb_points
            .iter_value()
            .map(|point| point.cos())
            .collect::<Vec<_>>()
            .into();

        let mut eval_points = DynArray::<f64, 1>::from_shape([m]);

        eval_points
            .iter_mut()
            .for_each(|point| *point = 2.0 * rng.random::<f64>() - 1.0);

        let weights = barycentric_chebychev_weights(Kind::First, n);
        let mut eval_values = DynArray::<f64, _>::from_shape([m]);

        barycentric_evaluate_1d(
            &eval_points,
            &cheb_points,
            &interp_values,
            &weights,
            &mut eval_values,
        );

        let max_error = izip!(eval_points.iter_value(), eval_values.iter_value())
            .map(|(point, value)| {
                let c = point.cos();
                (value - c).abs() / c.abs()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        assert!(max_error < 1E-9);
    }

    #[test]
    fn test_barycentric_interp_with_barycentric_second_kind() {
        let n = 20;
        let m = 1000;

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let cheb_points = chebychev_points::<f64>(Kind::Second, n);
        let interp_values: DynArray<f64, 1> = cheb_points
            .iter_value()
            .map(|point| point.cos())
            .collect::<Vec<_>>()
            .into();

        let mut eval_points = DynArray::<f64, 1>::from_shape([m]);

        eval_points
            .iter_mut()
            .for_each(|point| *point = 2.0 * rng.random::<f64>() - 1.0);

        let weights = barycentric_chebychev_weights(Kind::Second, n);
        let mut eval_values = DynArray::<f64, _>::from_shape([m]);

        barycentric_evaluate_1d(
            &eval_points,
            &cheb_points,
            &interp_values,
            &weights,
            &mut eval_values,
        );

        let max_error = izip!(eval_points.iter_value(), eval_values.iter_value())
            .map(|(point, value)| {
                let c = point.cos();
                (value - c).abs() / c.abs()
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        assert!(max_error < 1E-9);
    }
}
