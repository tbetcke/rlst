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
//!     &weights,
//!     &interp_values,
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
//! costs `O(n^2)`, while generation with [`barycentric_chebychev_weights`]
//! costs only `O(n)`.

use itertools::izip;

use crate::{
    Array, DynArray, MutableArrayImpl, RlstScalar, ScalarAsArray, Shape,
    UnsafeDisjointRandom1DAccessMut, UnsafeRandom1DAccessByValue, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut, ValueArrayImpl,
};

use num::{One, traits::FloatConst};

/// Define whether to use Chebychev points of the first kind or second kind
pub enum Kind {
    /// Choose Chebychev points of the first kind.
    First,
    /// Choose Chebychev points of the second kind.
    Second,
}

/// Return `n` Chebychev points of `kind` in `[-1, 1]`, in descending order.
///
/// First-kind points are interior roots and require `n > 0`. Second-kind points
/// include both endpoints and require `n > 1`.
///
/// # Panics
///
/// Panics when the selected kind's requirement on `n` is not met.
pub fn chebychev_points<Item: RlstScalar>(kind: Kind, n: usize) -> DynArray<Item, 1> {
    let mut output = DynArray::<Item, 1>::from_shape([n]);
    let indices = 0..n;

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

/// Compute general barycentric weights for distinct interpolation points.
///
/// Returns weights `w[j] = 1 / product(i != j, x[j] - x[i])`. Repeated points
/// cause division by zero; an empty input produces an empty output.
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

/// Return `n` analytic weights for `n` Chebychev interpolation points.
///
/// Overall scaling of barycentric weights is immaterial. Use the same `kind` and
/// `n` passed to [`chebychev_points`].
///
/// # Panics
///
/// Panics for first-kind `n == 0` or second-kind `n <= 1`.
pub fn barycentric_chebychev_weights<T: RlstScalar>(kind: Kind, n: usize) -> DynArray<T, 1> {
    let mut output = DynArray::<T, 1>::from_shape([n]);
    match kind {
        Kind::First => {
            assert!(n > 0);
            let indices = 0..n;
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
/// - `interp_weights`: The barycentric weights of the interpolation formula.
/// - `interp_values`: The values of the interpolation polynomial at the nodes.
/// - `result`: Output storage, with one entry per evaluation point.
///
/// The following routines are provides to compute the interpolation weights.
/// - [barycentric_weights]: Given an array of interpolation points directly compute the associated weights.
/// - [barycentric_chebychev_weights]: Use an analytic formula to compute the weights for Chebychev points of the first or second kind.
///
/// The output `result` is an array with the same length as `eval_points`. Evaluation
/// points exactly equal to a node receive that node's value directly.
///
/// # Panics
///
/// Panics unless node, weight, and value lengths match and `result` has the same
/// length as `eval_points`.
#[inline(always)]
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
    interp_weights: &Array<ArrayImplInterpWeights, 1>,
    interp_values: &Array<ArrayImplInterpValues, 1>,
    result: &mut Array<ArrayImplResult, 1>,
) where
    ArrayImplEvalPoints: UnsafeRandom1DAccessByValue<Item = Item::Real> + Shape<1>,
    ArrayImplInterpNodes: UnsafeRandom1DAccessByValue<Item = Item::Real> + Shape<1>,
    ArrayImplInterpValues: UnsafeRandomAccessByValue<1, Item = Item>
        + UnsafeRandom1DAccessByValue<Item = Item>
        + Shape<1>,
    ArrayImplInterpWeights: UnsafeRandom1DAccessByValue<Item = Item::Real> + Shape<1>,
    ArrayImplResult: UnsafeRandomAccessMut<1, Item = Item>
        + UnsafeDisjointRandom1DAccessMut<Item = Item>
        + Shape<1>,
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

/// Barycentric tensor evaluation in two dimension.
///
/// Evaluate an interpolation polynomial on a tensor grid defined by points in the x and y directions.
///
/// # Arguments
/// - `eval_points_x`: The coordinates of the points along the x-direction.
/// - `eval_points_y`: The coordinates of the points along the y-direction.
/// - `interp_nodes_x`: The interpolation nodes in the x-direction
/// - `interp_nodes_y`: The interpolation nodes in the y-direction
/// - `interp_values`: A `[interp_nodes_x.len(), interp_nodes_y.len()]` array of values at the tensor product of the interpolation nodes.
///   The values of the interpolation polynomial at the nodes.
/// - `interp_weights_x`: The interpolation weights in the x-direction.
/// - `interp_weights_y`: The interpolation weights in the y-direction.
///
/// The following routines are provides to compute the interpolation weights.
/// - [barycentric_weights]: Given an array of interpolation points directly compute the associated weights.
/// - [barycentric_chebychev_weights]: Use an analytic formula to compute the weights for Chebychev points of the first or second kind.
///
/// The output `result` is an array of dimension `[eval_points_x.len(), eval_points_y.len()]` and contains the
/// evaluation of the interpolation polynomial at the tensor grid defined by the direct product of `eval_points_x` and `eval_points_y`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn tensor_barycentric_evaluate_2d<
    Item,
    ArrayImplEvalPointsX,
    ArrayImplEvalPointsY,
    ArrayImplInterpNodesX,
    ArrayImplInterpNodesY,
    ArrayImplInterpValues,
    ArrayImplInterpWeightsX,
    ArrayImplInterpWeightsY,
    ArrayImplResult,
>(
    eval_points_x: &Array<ArrayImplEvalPointsX, 1>,
    eval_points_y: &Array<ArrayImplEvalPointsY, 1>,
    interp_nodes_x: &Array<ArrayImplInterpNodesX, 1>,
    interp_nodes_y: &Array<ArrayImplInterpNodesY, 1>,
    interp_weights_x: &Array<ArrayImplInterpWeightsX, 1>,
    interp_weights_y: &Array<ArrayImplInterpWeightsY, 1>,
    interp_values: &Array<ArrayImplInterpValues, 2>,
    result: &mut Array<ArrayImplResult, 2>,
) where
    Item: RlstScalar,
    ArrayImplEvalPointsX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplEvalPointsY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpValues: ValueArrayImpl<Item, 2>,
    ArrayImplResult: MutableArrayImpl<Item, 2>,
{
    // Make sure that the dimensions make sense

    let [nx, ny] = interp_values.shape();
    let mx = eval_points_x.len();
    let my = eval_points_y.len();

    assert_eq!(nx, interp_nodes_x.len());
    assert_eq!(ny, interp_nodes_y.len());
    assert_eq!(nx, interp_weights_x.len());
    assert_eq!(ny, interp_weights_y.len());

    assert_eq!([mx, my], result.shape());

    // We first evaluate for yeach y-point along the x-direction.

    let mut tmp = DynArray::<Item, 2>::from_shape([mx, ny]);

    for (col_values, mut col_tmp) in izip!(interp_values.col_iter(), tmp.col_iter_mut()) {
        barycentric_evaluate_1d(
            eval_points_x,
            interp_nodes_x,
            interp_weights_x,
            &col_values,
            &mut col_tmp,
        );
    }

    // Now we can evaluate tmp into result

    for (row_tmp, mut row_result) in izip!(tmp.row_iter(), result.row_iter_mut()) {
        barycentric_evaluate_1d(
            eval_points_y,
            interp_nodes_y,
            interp_weights_y,
            &row_tmp,
            &mut row_result,
        );
    }
}

/// Barycentric tensor evaluation in three dimension.
///
/// Evaluate an interpolation polynomial on a tensor grid defined by points in the x, y and z directions.
///
/// # Arguments
/// - `eval_points_x`: The coordinates of the points along the x-direction.
/// - `eval_points_y`: The coordinates of the points along the y-direction.
/// - `eval_points_z`: The coordinates of the points along the z-direction.
/// - `interp_nodes_x`: The interpolation nodes in the x-direction
/// - `interp_nodes_y`: The interpolation nodes in the y-direction
/// - `interp_nodes_z`: The interpolation nodes in the z-direction
/// - `interp_values`: A `[interp_nodes_x.len(), interp_nodes_y.len(), interp_nodes_z.len()]` array of values at the tensor product of the interpolation nodes.
///   The values of the interpolation polynomial at the nodes.
/// - `interp_weights_x`: The interpolation weights in the x-direction.
/// - `interp_weights_y`: The interpolation weights in the y-direction.
/// - `interp_weights_z`: The interpolation weights in the z-direction.
///
/// The following routines are provides to compute the interpolation weights.
/// - [barycentric_weights]: Given an array of interpolation points directly compute the associated weights.
/// - [barycentric_chebychev_weights]: Use an analytic formula to compute the weights for Chebychev points of the first or second kind.
///
/// The output `result` is an array of dimension `[eval_points_x.len(), eval_points_y.len(), eval_points_z.len()]` and contains the
/// evaluation of the interpolation polynomial at the tensor grid defined by the direct product of `eval_points_x`, `eval_points_y` and `eval_points_z`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn tensor_barycentric_evaluate_3d<
    Item,
    ArrayImplEvalPointsX,
    ArrayImplEvalPointsY,
    ArrayImplEvalPointsZ,
    ArrayImplInterpNodesX,
    ArrayImplInterpNodesY,
    ArrayImplInterpNodesZ,
    ArrayImplInterpValues,
    ArrayImplInterpWeightsX,
    ArrayImplInterpWeightsY,
    ArrayImplInterpWeightsZ,
    ArrayImplResult,
>(
    eval_points_x: &Array<ArrayImplEvalPointsX, 1>,
    eval_points_y: &Array<ArrayImplEvalPointsY, 1>,
    eval_points_z: &Array<ArrayImplEvalPointsZ, 1>,
    interp_nodes_x: &Array<ArrayImplInterpNodesX, 1>,
    interp_nodes_y: &Array<ArrayImplInterpNodesY, 1>,
    interp_nodes_z: &Array<ArrayImplInterpNodesZ, 1>,
    interp_weights_x: &Array<ArrayImplInterpWeightsX, 1>,
    interp_weights_y: &Array<ArrayImplInterpWeightsY, 1>,
    interp_weights_z: &Array<ArrayImplInterpWeightsZ, 1>,
    interp_values: &Array<ArrayImplInterpValues, 3>,
    result: &mut Array<ArrayImplResult, 3>,
) where
    Item: RlstScalar,
    ArrayImplEvalPointsX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplEvalPointsY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplEvalPointsZ: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesZ: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsZ: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpValues: ValueArrayImpl<Item, 3>,
    ArrayImplResult: MutableArrayImpl<Item, 3>,
{
    // Make sure that the dimensions make sense

    let [nx, ny, nz] = interp_values.shape();
    let mx = eval_points_x.len();
    let my = eval_points_y.len();
    let mz = eval_points_z.len();

    assert_eq!(nx, interp_nodes_x.len());
    assert_eq!(ny, interp_nodes_y.len());
    assert_eq!(nz, interp_nodes_z.len());
    assert_eq!(nx, interp_weights_x.len());
    assert_eq!(ny, interp_weights_y.len());
    assert_eq!(nz, interp_weights_z.len());

    assert_eq!([mx, my, mz], result.shape());

    // We first evaluate for each (y,z)-point along the x-direction.

    let mut tmp_x = DynArray::<Item, 3>::from_shape([mx, ny, nz]);

    for (indices, interp_slice) in interp_values.multislice_iter::<_, 1>([1, 2]) {
        barycentric_evaluate_1d(
            eval_points_x,
            interp_nodes_x,
            interp_weights_x,
            &interp_slice,
            &mut tmp_x.r_mut().multislice([1, 2], indices),
        )
    }

    // We now evaluate for each (x, z) value along the y-direction

    let mut tmp_y = DynArray::<Item, 3>::from_shape([mx, my, nz]);

    for (indices, interp_slice) in tmp_x.multislice_iter::<_, 1>([0, 2]) {
        barycentric_evaluate_1d(
            eval_points_y,
            interp_nodes_y,
            interp_weights_y,
            &interp_slice,
            &mut tmp_y.r_mut().multislice([0, 2], indices),
        )
    }

    // Finally, we evaluate for each (x, y) value along the z direction.
    // The results are directly written into `result`

    for (indices, interp_slice) in tmp_y.multislice_iter::<_, 1>([0, 1]) {
        barycentric_evaluate_1d(
            eval_points_z,
            interp_nodes_z,
            interp_weights_z,
            &interp_slice,
            &mut result.r_mut().multislice([0, 1], indices),
        )
    }
}

/// Barycentric evaluation in two dimensions
///
/// Evaluate an interpolation polynomial on a given array of points.
///
/// # Arguments
/// - `eval_points`: The coordinates of the points as an `(2, npts)` array.
/// - `interp_nodes_x`: The interpolation nodes in the x-direction
/// - `interp_nodes_y`: The interpolation nodes in the y-direction
/// - `interp_values`: A `[interp_nodes_x.len(), interp_nodes_y.len()]` array of values at the tensor product of the interpolation nodes.
///   The values of the interpolation polynomial at the nodes.
/// - `interp_weights_x`: The interpolation weights in the x-direction.
/// - `interp_weights_y`: The interpolation weights in the y-direction.
///
/// The following routines are provides to compute the interpolation weights.
/// - [barycentric_weights]: Given an array of interpolation points directly compute the associated weights.
/// - [barycentric_chebychev_weights]: Use an analytic formula to compute the weights for Chebychev points of the first or second kind.
///
/// The output `result` is an array of dimension `[npts]` and contains the
/// evaluation of the interpolation polynomial at the `eval_points`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn barycentric_evaluate_2d<
    Item,
    ArrayImplEvalPoints,
    ArrayImplInterpNodesX,
    ArrayImplInterpNodesY,
    ArrayImplInterpValues,
    ArrayImplInterpWeightsX,
    ArrayImplInterpWeightsY,
    ArrayImplResult,
>(
    eval_points: &Array<ArrayImplEvalPoints, 2>,
    interp_nodes_x: &Array<ArrayImplInterpNodesX, 1>,
    interp_nodes_y: &Array<ArrayImplInterpNodesY, 1>,
    interp_weights_x: &Array<ArrayImplInterpWeightsX, 1>,
    interp_weights_y: &Array<ArrayImplInterpWeightsY, 1>,
    interp_values: &Array<ArrayImplInterpValues, 2>,
    result: &mut Array<ArrayImplResult, 1>,
) where
    Item: RlstScalar,
    ArrayImplEvalPoints: ValueArrayImpl<Item::Real, 2>,
    ArrayImplInterpNodesX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpValues: ValueArrayImpl<Item, 2>,
    ArrayImplResult: MutableArrayImpl<Item, 1>,
{
    let shape_points = eval_points.shape();
    let npoints = shape_points[1];
    let dim = shape_points[0];

    assert_eq!(npoints, result.len());
    assert_eq!(interp_nodes_x.len(), interp_weights_x.len());
    assert_eq!(interp_nodes_y.len(), interp_weights_y.len());

    assert_eq!(dim, 2);

    let mut res = Array::one_element_array();

    for (point, r) in izip!(eval_points.col_iter(), result.iter_mut()) {
        let p0 = point.get_value([0]).unwrap().as_array();
        let p1 = point.get_value([1]).unwrap().as_array();

        tensor_barycentric_evaluate_2d(
            &p0,
            &p1,
            interp_nodes_x,
            interp_nodes_y,
            interp_weights_x,
            interp_weights_y,
            interp_values,
            &mut res,
        );

        *r = res.get_value([0, 0]).unwrap();
    }
}

/// Barycentric evaluation in three dimensions
///
/// Evaluate an interpolation polynomial on a given array of points.
///
/// # Arguments
/// - `eval_points`: Coordinates as a `(3, npts)` array.
/// - `interp_nodes_x`, `interp_nodes_y`, `interp_nodes_z`: Nodes on each axis.
/// - `interp_values`: Values on the tensor grid, with shape equal to the three node lengths.
/// - `interp_weights_x`, `interp_weights_y`, `interp_weights_z`: Weights on each axis.
///
/// The following routines are provided to compute the interpolation weights.
/// - [barycentric_weights]: Given an array of interpolation points directly compute the associated weights.
/// - [barycentric_chebychev_weights]: Use an analytic formula to compute the weights for Chebychev points of the first or second kind.
///
/// The output `result` is an array of dimension `[npts]` and contains the
/// evaluation of the interpolation polynomial at the `eval_points`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn barycentric_evaluate_3d<
    Item,
    ArrayImplEvalPoints,
    ArrayImplInterpNodesX,
    ArrayImplInterpNodesY,
    ArrayImplInterpNodesZ,
    ArrayImplInterpValues,
    ArrayImplInterpWeightsX,
    ArrayImplInterpWeightsY,
    ArrayImplInterpWeightsZ,
    ArrayImplResult,
>(
    eval_points: &Array<ArrayImplEvalPoints, 2>,
    interp_nodes_x: &Array<ArrayImplInterpNodesX, 1>,
    interp_nodes_y: &Array<ArrayImplInterpNodesY, 1>,
    interp_nodes_z: &Array<ArrayImplInterpNodesZ, 1>,
    interp_weights_x: &Array<ArrayImplInterpWeightsX, 1>,
    interp_weights_y: &Array<ArrayImplInterpWeightsY, 1>,
    interp_weights_z: &Array<ArrayImplInterpWeightsZ, 1>,
    interp_values: &Array<ArrayImplInterpValues, 3>,
    result: &mut Array<ArrayImplResult, 1>,
) where
    Item: RlstScalar,
    ArrayImplEvalPoints: ValueArrayImpl<Item::Real, 2>,
    ArrayImplInterpNodesX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpNodesZ: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsX: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsY: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpWeightsZ: ValueArrayImpl<Item::Real, 1>,
    ArrayImplInterpValues: ValueArrayImpl<Item, 3>,
    ArrayImplResult: MutableArrayImpl<Item, 1>,
{
    let shape_points = eval_points.shape();
    let npoints = shape_points[1];
    let dim = shape_points[0];

    assert_eq!(npoints, result.len());
    assert_eq!(interp_nodes_x.len(), interp_weights_x.len());
    assert_eq!(interp_nodes_y.len(), interp_weights_y.len());
    assert_eq!(interp_nodes_z.len(), interp_weights_z.len());

    assert_eq!(dim, 3);

    let mut res = Array::one_element_array();

    for (point, r) in izip!(eval_points.col_iter(), result.iter_mut()) {
        let p0 = point.get_value([0]).unwrap().as_array();
        let p1 = point.get_value([1]).unwrap().as_array();
        let p2 = point.get_value([2]).unwrap().as_array();

        tensor_barycentric_evaluate_3d(
            &p0,
            &p1,
            &p2,
            interp_nodes_x,
            interp_nodes_y,
            interp_nodes_z,
            interp_weights_x,
            interp_weights_y,
            interp_weights_z,
            interp_values,
            &mut res,
        );

        *r = res.get_value([0, 0, 0]).unwrap();
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
        DynArray, assert_array_relative_eq,
        interpolation::{
            Kind, barycentric_chebychev_weights, barycentric_evaluate_1d, barycentric_evaluate_2d,
            barycentric_evaluate_3d, barycentric_weights, chebychev_points,
            tensor_barycentric_evaluate_2d, tensor_barycentric_evaluate_3d,
        },
        linspace,
    };

    #[test]
    fn test_cheb_points_first_kind() {
        let n = 5;

        let points = chebychev_points::<f64>(Kind::First, n);

        for (index, &value) in points.data().unwrap().iter().enumerate() {
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

        for (index, &value) in values.data().unwrap().iter().enumerate() {
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
    fn test_barycentric_interp_with_chebychev_first_kind() {
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
            &weights,
            &interp_values,
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
    fn test_barycentric_interp_with_chebychev_second_kind_1d() {
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
            &weights,
            &interp_values,
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
    fn test_barycentric_tensor_interp_with_chebychev_second_kind_2d() {
        let nx = 20;
        let ny = 25;

        let mx = 50;
        let my = 60;

        let cheb_x = chebychev_points::<f64>(Kind::Second, nx);
        let cheb_y = chebychev_points::<f64>(Kind::Second, ny);

        let mut values = DynArray::<f64, 2>::from_shape([nx, ny]);

        for (xind, px) in cheb_x.iter_value().enumerate() {
            for (yind, py) in cheb_y.iter_value().enumerate() {
                values[[xind, yind]] = f64::exp(px * py);
            }
        }

        let eval_points_x = linspace!(f64, -1.0, 1.0, mx);
        let eval_points_y = linspace!(f64, -1.0, 1.0, my);

        let mut expected = DynArray::<f64, 2>::from_shape([mx, my]);

        for (xind, px) in eval_points_x.iter_value().enumerate() {
            for (yind, py) in eval_points_y.iter_value().enumerate() {
                expected[[xind, yind]] = f64::exp(px * py);
            }
        }

        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);

        let mut result = DynArray::<f64, 2>::from_shape([mx, my]);

        tensor_barycentric_evaluate_2d(
            &eval_points_x,
            &eval_points_y,
            &cheb_x,
            &cheb_y,
            &weights_x,
            &weights_y,
            &values,
            &mut result,
        );

        assert_array_relative_eq!(expected, result, 1E-10);
    }

    #[test]
    fn test_barycentric_tensor_interp_with_chebychev_second_kind_3d() {
        let nx = 20;
        let ny = 25;
        let nz = 30;

        let mx = 50;
        let my = 60;
        let mz = 70;

        let cheb_x = chebychev_points::<f64>(Kind::Second, nx);
        let cheb_y = chebychev_points::<f64>(Kind::Second, ny);
        let cheb_z = chebychev_points::<f64>(Kind::Second, nz);

        let mut values = DynArray::<f64, 3>::from_shape([nx, ny, nz]);

        for (xind, px) in cheb_x.iter_value().enumerate() {
            for (yind, py) in cheb_y.iter_value().enumerate() {
                for (zind, pz) in cheb_z.iter_value().enumerate() {
                    values[[xind, yind, zind]] = f64::exp(2.0 * px + 3.0 * py + 4.0 * pz);
                }
            }
        }

        let eval_points_x = linspace!(f64, -1.0, 1.0, mx);
        let eval_points_y = linspace!(f64, -1.0, 1.0, my);
        let eval_points_z = linspace!(f64, -1.0, 1.0, mz);

        let mut expected = DynArray::<f64, 3>::from_shape([mx, my, mz]);

        for (xind, px) in eval_points_x.iter_value().enumerate() {
            for (yind, py) in eval_points_y.iter_value().enumerate() {
                for (zind, pz) in eval_points_z.iter_value().enumerate() {
                    expected[[xind, yind, zind]] = f64::exp(2.0 * px + 3.0 * py + 4.0 * pz);
                }
            }
        }

        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);
        let weights_z = barycentric_chebychev_weights::<f64>(Kind::Second, nz);

        let mut result = DynArray::<f64, 3>::from_shape([mx, my, mz]);

        tensor_barycentric_evaluate_3d(
            &eval_points_x,
            &eval_points_y,
            &eval_points_z,
            &cheb_x,
            &cheb_y,
            &cheb_z,
            &weights_x,
            &weights_y,
            &weights_z,
            &values,
            &mut result,
        );

        assert_array_relative_eq!(expected, result, 1E-10);
    }

    #[test]
    fn test_barycentric_interp_with_chebychev_second_kind_2d() {
        let nx = 20;
        let ny = 25;

        let m = 60;

        let cheb_x = chebychev_points::<f64>(Kind::Second, nx);
        let cheb_y = chebychev_points::<f64>(Kind::Second, ny);

        let mut values = DynArray::<f64, 2>::from_shape([nx, ny]);

        for (xind, px) in cheb_x.iter_value().enumerate() {
            for (yind, py) in cheb_y.iter_value().enumerate() {
                values[[xind, yind]] = f64::exp(px * py);
            }
        }

        let mut eval_points = DynArray::<f64, 2>::from_shape([2, m]);
        eval_points.fill_from_seed_equally_distributed(0);
        for value in eval_points.iter_mut() {
            // Shift random values to [-1, 1]
            *value = 2.0 * *value - 1.0
        }

        let mut expected = DynArray::<f64, 1>::from_shape([m]);

        for (point, value) in izip!(eval_points.col_iter(), expected.iter_mut()) {
            *value = f64::exp(point.get([0]).unwrap() * point.get([1]).unwrap());
        }

        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);

        let mut result = DynArray::<f64, 1>::from_shape([m]);

        barycentric_evaluate_2d(
            &eval_points,
            &cheb_x,
            &cheb_y,
            &weights_x,
            &weights_y,
            &values,
            &mut result,
        );

        assert_array_relative_eq!(expected, result, 1E-10);
    }

    #[test]
    fn test_barycentric_interp_with_chebychev_second_kind_3d() {
        let nx = 20;
        let ny = 25;
        let nz = 30;

        let m = 60;

        let cheb_x = chebychev_points::<f64>(Kind::Second, nx);
        let cheb_y = chebychev_points::<f64>(Kind::Second, ny);
        let cheb_z = chebychev_points::<f64>(Kind::Second, nz);

        let mut values = DynArray::<f64, 3>::from_shape([nx, ny, nz]);

        for (xind, px) in cheb_x.iter_value().enumerate() {
            for (yind, py) in cheb_y.iter_value().enumerate() {
                for (zind, pz) in cheb_z.iter_value().enumerate() {
                    values[[xind, yind, zind]] = f64::exp(px * py * pz);
                }
            }
        }

        let mut eval_points = DynArray::<f64, 2>::from_shape([3, m]);
        eval_points.fill_from_seed_equally_distributed(0);
        for value in eval_points.iter_mut() {
            // Shift random values to [-1, 1]
            *value = 2.0 * *value - 1.0
        }

        let mut expected = DynArray::<f64, 1>::from_shape([m]);

        for (point, value) in izip!(eval_points.col_iter(), expected.iter_mut()) {
            *value = f64::exp(
                point.get([0]).unwrap() * point.get([1]).unwrap() * point.get([2]).unwrap(),
            );
        }

        let weights_x = barycentric_chebychev_weights::<f64>(Kind::Second, nx);
        let weights_y = barycentric_chebychev_weights::<f64>(Kind::Second, ny);
        let weights_z = barycentric_chebychev_weights::<f64>(Kind::Second, nz);

        let mut result = DynArray::<f64, 1>::from_shape([m]);

        barycentric_evaluate_3d(
            &eval_points,
            &cheb_x,
            &cheb_y,
            &cheb_z,
            &weights_x,
            &weights_y,
            &weights_z,
            &values,
            &mut result,
        );

        assert_array_relative_eq!(expected, result, 1E-10);
    }
}
