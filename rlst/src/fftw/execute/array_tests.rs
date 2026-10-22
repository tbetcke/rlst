//! Execution on RLST arrays and views (task F2; design Sections 7.2 to 7.4, c2c).
//!
//! Every plan here is made with `Rigor::Estimate`, depends on no wisdom and assumes no SIMD.
//! Results are compared with the logical reference of `test_support`, which reads the input
//! view by multi-index and never builds a `Layout`, and every view's parent outside the view
//! is checked to hold its sentinel afterwards.

use std::fmt;

use num::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::dense::array::{StridedSliceArray, StridedSliceArrayMut};
use crate::fftw::test_support::{
    PolicyKind, SEED, TestReal, assert_gaps_unchanged, assert_matches_reference, c2c_tolerance,
    fill_view, filled, logical_reference, multi_indices, relative_error, sentinel, sentinel_parent,
    storage_bits, widen,
};
use crate::fftw::{AlignmentClass, Direction, FftwError, Layout, Planner, Side};
use crate::{
    Array, BaseItem, DynArray, Shape, Stride, StridedDynArray, StridedRawAccess,
    StridedRawAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessMut, c64,
};

/// The arrays and views the tests execute on.
trait View<T: TestReal, const N: usize>:
    BaseItem<Item = Complex<T>>
    + StridedRawAccessMut
    + Shape<N>
    + Stride<N>
    + UnsafeRandomAccessByRef<N>
    + UnsafeRandomAccessMut<N>
{
}

impl<T: TestReal, A, const N: usize> View<T, N> for A where
    A: BaseItem<Item = Complex<T>>
        + StridedRawAccessMut
        + Shape<N>
        + Stride<N>
        + UnsafeRandomAccessByRef<N>
        + UnsafeRandomAccessMut<N>
{
}

/// The memory orders and views of the correctness matrix (F2 brief, work item 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// An owned column-major array at `FFTW_ALIGNED`.
    ColumnMajor,
    /// An owned row-major array (`StridedDynArray::row_major`), `Vec`-backed.
    RowMajor,
    /// A strided view with steps above 1 over aligned storage.
    Strided,
    /// A subview of a larger parent.
    Subview,
    /// A subview of a subview.
    NestedSubview,
    /// A subview of a slice of a parent of higher rank.
    SubviewOfSlice,
    /// A slice of a parent of higher rank.
    Slice,
    /// A multislice of a parent of rank two higher.
    Multislice,
}

impl Order {
    const ALL: [Order; 8] = [
        Order::ColumnMajor,
        Order::RowMajor,
        Order::Strided,
        Order::Subview,
        Order::NestedSubview,
        Order::SubviewOfSlice,
        Order::Slice,
        Order::Multislice,
    ];
}

/// One cell of the correctness matrix on arrays.
#[derive(Clone, Debug)]
struct Cell<const N: usize> {
    order: Order,
    shape: [usize; N],
    axes: Vec<usize>,
    policy: PolicyKind,
    direction: Direction,
    seed: u64,
}

impl<const N: usize> fmt::Display for Cell<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}, shape {:?}, axes {:?}, {:?}, {:?} (seed {:#x})",
            self.order, self.shape, self.axes, self.policy, self.direction, self.seed
        )
    }
}

/// `values`, given in column-major order of `shape`, widened and multiplied by `factor`.
fn scaled<T: TestReal, const N: usize>(
    shape: [usize; N],
    values: &[Complex<T>],
    factor: f64,
) -> DynArray<Complex<f64>, N> {
    let mut result = DynArray::<Complex<f64>, N>::from_shape(shape);
    for (index, value) in multi_indices(shape).zip(widen(values)) {
        result[index] = value * factor;
    }
    result
}

/// Execute an out-of-place plan of `policy` from `input` to `output`.
fn execute_out_of_place<T, AIn, AOut, const N: usize>(
    planner: &Planner<T>,
    layout: &Layout,
    direction: Direction,
    policy: PolicyKind,
    input: &mut Array<AIn, N>,
    output: &mut Array<AOut, N>,
) where
    T: TestReal,
    AIn: View<T, N>,
    AOut: View<T, N>,
{
    match policy {
        PolicyKind::Preserving => planner
            .c2c(layout, direction)
            .unwrap()
            .execute(&*input, output)
            .unwrap(),
        PolicyKind::Destroying => planner
            .clone()
            .may_destroy_input()
            .c2c(layout, direction)
            .unwrap()
            .execute(input, output)
            .unwrap(),
        PolicyKind::InPlace => unreachable!("an out-of-place policy"),
    }
}

/// Check an out-of-place cell: the transform against the logical reference, the input's
/// storage for a preserving plan, the gaps of both views, and for a forward cell the round
/// trip back into the input.
fn run_out_of_place<T, AIn, AOut, const N: usize>(
    cell: &Cell<N>,
    input: &mut Array<AIn, N>,
    output: &mut Array<AOut, N>,
) where
    T: TestReal,
    AIn: View<T, N>,
    AOut: View<T, N>,
{
    let values = fill_view(input, cell.seed);
    let before = storage_bits(input);
    let layout = Layout::over_axes(input, output, &cell.axes).unwrap();
    // Plan for the classes of these arrays: row-major arrays are `Vec`-backed, and views at
    // an offset change class.
    let planner =
        Planner::<T>::new().alignment(AlignmentClass::of(&*input), AlignmentClass::of(&*output));
    execute_out_of_place(
        &planner,
        &layout,
        cell.direction,
        cell.policy,
        input,
        output,
    );
    if cell.policy == PolicyKind::Preserving {
        assert!(
            storage_bits(input) == before,
            "{} {cell}: a preserving plan changed its input's storage",
            T::NAME
        );
    }
    assert_gaps_unchanged(input, cell);
    assert_gaps_unchanged(output, cell);
    let reference = logical_reference(cell.shape, &values, &cell.axes, cell.direction);
    assert_matches_reference(output, &reference, &cell.axes, cell);

    if cell.direction == Direction::Forward {
        // The backward transform of the output, into the input, is N times the input.
        let layout = Layout::over_axes(output, input, &cell.axes).unwrap();
        let planner = Planner::<T>::new()
            .alignment(AlignmentClass::of(&*output), AlignmentClass::of(&*input));
        execute_out_of_place(
            &planner,
            &layout,
            Direction::Backward,
            cell.policy,
            output,
            input,
        );
        assert_gaps_unchanged(input, cell);
        assert_gaps_unchanged(output, cell);
        let expected = scaled(cell.shape, &values, layout.logical_size() as f64);
        let context = format!("{cell}, round trip");
        assert_matches_reference(input, &expected, &cell.axes, &context);
    }
}

/// Check an in-place cell, and for a forward cell the round trip.
fn run_in_place<T, A, const N: usize>(cell: &Cell<N>, data: &mut Array<A, N>)
where
    T: TestReal,
    A: View<T, N>,
{
    let values = fill_view(data, cell.seed);
    let layout = Layout::over_axes_inplace(data, &cell.axes).unwrap();
    let class = AlignmentClass::of(&*data);
    let planner = Planner::<T>::new().alignment(class, class);
    let plan = planner.c2c_inplace(&layout, cell.direction).unwrap();
    plan.execute(data).unwrap();
    assert_gaps_unchanged(data, cell);
    let reference = logical_reference(cell.shape, &values, &cell.axes, cell.direction);
    assert_matches_reference(data, &reference, &cell.axes, cell);

    if cell.direction == Direction::Forward {
        let plan = planner.c2c_inplace(&layout, Direction::Backward).unwrap();
        plan.execute(data).unwrap();
        assert_gaps_unchanged(data, cell);
        let expected = scaled(cell.shape, &values, layout.logical_size() as f64);
        let context = format!("{cell}, round trip");
        assert_matches_reference(data, &expected, &cell.axes, &context);
    }
}

fn run_cell<T, AIn, AOut, const N: usize>(
    cell: &Cell<N>,
    input: &mut Array<AIn, N>,
    output: &mut Array<AOut, N>,
) where
    T: TestReal,
    AIn: View<T, N>,
    AOut: View<T, N>,
{
    if cell.policy == PolicyKind::InPlace {
        run_in_place(cell, input);
    } else {
        run_out_of_place(cell, input, output);
    }
}

/// `shape` with `first` in front.
fn prepend<const N: usize, const M: usize>(first: usize, shape: [usize; N]) -> [usize; M] {
    assert_eq!(M, N + 1);
    let mut result = [first; M];
    result[1..].copy_from_slice(&shape);
    result
}

/// `shape` with `last` at the back.
fn append<const N: usize, const M: usize>(shape: [usize; N], last: usize) -> [usize; M] {
    assert_eq!(M, N + 1);
    let mut result = [last; M];
    result[..N].copy_from_slice(&shape);
    result
}

/// The strides of a strided view with steps of 2: `[2, 2 n0, 2 n0 n1, …]`.
fn steps_of_two<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut stride = [2; N];
    for axis in 1..N {
        stride[axis] = stride[axis - 1] * shape[axis - 1];
    }
    stride
}

/// Build the views of `cell.order` and run the cell on them. Every view's parent is two or
/// three elements larger on each axis, or has more axes, and the view is at a non-zero
/// origin; the parents are filled with the sentinel.
macro_rules! run_views {
    ($t:ty, $n:literal, $n1:literal, $n2:literal, $cell:expr) => {{
        let cell: &Cell<$n> = $cell;
        let shape = cell.shape;
        let grow = |by: usize| shape.map(|n| n + by);
        let ones = [1_usize; $n];
        let twos = [2_usize; $n];
        match cell.order {
            Order::ColumnMajor => {
                let mut input = sentinel_parent::<$t, $n>(shape);
                let mut output = sentinel_parent::<$t, $n>(shape);
                run_cell::<$t, _, _, $n>(cell, &mut input, &mut output);
            }
            Order::RowMajor => {
                let mut input = StridedDynArray::<Complex<$t>, $n>::row_major(shape);
                let mut output = StridedDynArray::<Complex<$t>, $n>::row_major(shape);
                run_cell::<$t, _, _, $n>(cell, &mut input, &mut output);
            }
            Order::Strided => {
                let stride = steps_of_two(shape);
                let len = 2 * shape.iter().product::<usize>();
                let mut input = filled(sentinel::<$t>(), len);
                let mut output = filled(sentinel::<$t>(), len);
                run_cell::<$t, _, _, $n>(
                    cell,
                    &mut StridedSliceArrayMut::from_shape_and_stride(&mut input[..], shape, stride),
                    &mut StridedSliceArrayMut::from_shape_and_stride(
                        &mut output[..],
                        shape,
                        stride,
                    ),
                );
            }
            Order::Subview => {
                let mut input = sentinel_parent::<$t, $n>(grow(3));
                let mut output = sentinel_parent::<$t, $n>(grow(2));
                run_cell::<$t, _, _, $n>(
                    cell,
                    &mut input.r_mut().into_subview(twos, shape),
                    &mut output.r_mut().into_subview(ones, shape),
                );
            }
            Order::NestedSubview => {
                let mut input = sentinel_parent::<$t, $n>(grow(3));
                let mut output = sentinel_parent::<$t, $n>(grow(3));
                run_cell::<$t, _, _, $n>(
                    cell,
                    &mut input
                        .r_mut()
                        .into_subview(ones, grow(2))
                        .into_subview(ones, shape),
                    &mut output
                        .r_mut()
                        .into_subview(twos, grow(1))
                        .into_subview(ones, shape),
                );
            }
            Order::SubviewOfSlice => {
                let mut input = sentinel_parent::<$t, $n1>(append(grow(2), 3));
                let mut output = sentinel_parent::<$t, $n1>(prepend(2, grow(3)));
                run_cell::<$t, _, _, $n>(
                    cell,
                    &mut input.r_mut().slice::<$n>($n, 1).into_subview(ones, shape),
                    &mut output.r_mut().slice::<$n>(0, 1).into_subview(twos, shape),
                );
            }
            Order::Slice => {
                let mut input = sentinel_parent::<$t, $n1>(prepend(3, shape));
                let mut output = sentinel_parent::<$t, $n1>(append(shape, 3));
                run_cell::<$t, _, _, $n>(
                    cell,
                    &mut input.r_mut().slice::<$n>(0, 1),
                    &mut output.r_mut().slice::<$n>($n, 2),
                );
            }
            Order::Multislice => {
                let parent: [usize; $n2] = append::<$n1, $n2>(prepend(2, shape), 3);
                let mut input = sentinel_parent::<$t, $n2>(parent);
                let mut output = sentinel_parent::<$t, $n2>(parent);
                run_cell::<$t, _, _, $n>(
                    cell,
                    &mut input.r_mut().multislice::<2, $n>([0, $n1], [1, 2]),
                    &mut output.r_mut().multislice::<2, $n>([$n1, 0], [1, 0]),
                );
            }
        }
    }};
}

fn run_rank1<T: TestReal>(cell: &Cell<1>) {
    run_views!(T, 1, 2, 3, cell)
}

fn run_rank2<T: TestReal>(cell: &Cell<2>) {
    run_views!(T, 2, 3, 4, cell)
}

fn run_rank3<T: TestReal>(cell: &Cell<3>) {
    run_views!(T, 3, 4, 5, cell)
}

/// The axes of the fixed cells of rank `rank`: along each axis, over all axes, and for rank
/// 3 over two axes in four orders.
fn fixed_axes(rank: usize) -> Vec<Vec<usize>> {
    match rank {
        1 => vec![vec![0]],
        2 => vec![vec![0], vec![1], vec![0, 1]],
        3 => vec![
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1, 2],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![2, 0],
        ],
        _ => unreachable!(),
    }
}

/// Run the fixed cells of `order` in precision `T` and `direction`: shapes `[97]`, `[8, 6]`,
/// `[17, 4]` and `[6, 5, 4]`, the axes of [`fixed_axes`] and the three policies.
fn run_fixed_cells<T: TestReal>(order: Order, direction: Direction) {
    let mut seed = SEED ^ ((Order::ALL.iter().position(|&o| o == order).unwrap() as u64) << 32);
    let mut next_seed = || {
        seed += 1;
        seed
    };
    for policy in PolicyKind::ALL {
        for axes in fixed_axes(1) {
            run_rank1::<T>(&Cell {
                order,
                shape: [97],
                axes,
                policy,
                direction,
                seed: next_seed(),
            });
        }
        for shape in [[8, 6], [17, 4]] {
            for axes in fixed_axes(2) {
                run_rank2::<T>(&Cell {
                    order,
                    shape,
                    axes,
                    policy,
                    direction,
                    seed: next_seed(),
                });
            }
        }
        for axes in fixed_axes(3) {
            run_rank3::<T>(&Cell {
                order,
                shape: [6, 5, 4],
                axes,
                policy,
                direction,
                seed: next_seed(),
            });
        }
    }
}

fn run_all_directions_and_precisions(order: Order) {
    for direction in [Direction::Forward, Direction::Backward] {
        run_fixed_cells::<f64>(order, direction);
        run_fixed_cells::<f32>(order, direction);
    }
}

#[test]
fn test_array_cells_column_major() {
    run_all_directions_and_precisions(Order::ColumnMajor);
}

#[test]
fn test_array_cells_subview() {
    run_all_directions_and_precisions(Order::Subview);
}

#[test]
fn test_array_cells_other_orders() {
    for order in Order::ALL {
        if order != Order::ColumnMajor && order != Order::Subview {
            run_fixed_cells::<f64>(order, Direction::Forward);
        }
    }
}

#[test]
fn test_array_cells_sampled() {
    // 24 cells from the product of memory orders, shapes with extents up to 12, axes,
    // policies, directions and precisions, drawn with the fixed seed.
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    for _ in 0..24 {
        let order = Order::ALL[rng.random_range(0..Order::ALL.len())];
        let rank = rng.random_range(1..=3_usize);
        let mut axes: Vec<usize> = (0..rank).filter(|_| rng.random_bool(0.5)).collect();
        if axes.is_empty() {
            axes.push(rng.random_range(0..rank));
        }
        if axes.len() > 1 && rng.random_bool(0.5) {
            axes.reverse();
        }
        let policy = PolicyKind::ALL[rng.random_range(0..3)];
        let direction = if rng.random_bool(0.5) {
            Direction::Forward
        } else {
            Direction::Backward
        };
        let single = rng.random_bool(0.5);
        let seed = rng.random();
        let mut extent = || rng.random_range(1..=12_usize);
        macro_rules! sampled {
            ($run:ident, $shape:expr) => {{
                let cell = Cell {
                    order,
                    shape: $shape,
                    axes,
                    policy,
                    direction,
                    seed,
                };
                if single {
                    $run::<f32>(&cell);
                } else {
                    $run::<f64>(&cell);
                }
            }};
        }
        match rank {
            1 => sampled!(run_rank1, [extent()]),
            2 => sampled!(run_rank2, [extent(), extent()]),
            _ => sampled!(run_rank3, [extent(), extent(), extent()]),
        }
    }
}

// Identities (design Section 7.3).

fn bits<T: TestReal>(values: impl IntoIterator<Item = Complex<T>>) -> Vec<(u64, u64)> {
    values
        .into_iter()
        .map(|value| (value.re.to_f64().to_bits(), value.im.to_f64().to_bits()))
        .collect()
}

fn values_of<T, A, const N: usize>(view: &Array<A, N>) -> Vec<Complex<T>>
where
    T: TestReal,
    A: BaseItem<Item = Complex<T>> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    multi_indices(view.shape())
        .map(|index| *view.get(index).unwrap())
        .collect()
}

fn check_batched_against_single_plans<T: TestReal>() {
    // Along axis 0 of [n, b]: one batched plan against one plan per column. n is a multiple
    // of 8, so every column starts at class 0 in both precisions.
    let (n, b) = (24, 5);
    let mut input = sentinel_parent::<T, 2>([n, b]);
    let mut batched = sentinel_parent::<T, 2>([n, b]);
    let mut single = sentinel_parent::<T, 2>([n, b]);
    fill_view(&mut input, SEED);
    let layout = Layout::along_axis(&input, &batched, 0).unwrap();
    let plan = Planner::<T>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    plan.execute(&input, &mut batched).unwrap();
    let layout = Layout::along_axis(&input.r().col(0), &single.r().col(0), 0).unwrap();
    let plan = Planner::<T>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    for j in 0..b {
        plan.execute(&input.r().col(j), &mut single.r_mut().col(j))
            .unwrap();
        let error = relative_error(
            &widen(&values_of(&batched.r().col(j))),
            &widen(&values_of(&single.r().col(j))),
        );
        let tolerance = c2c_tolerance::<T>(n);
        assert!(
            error <= tolerance,
            "{} column {j}: batched and single plans differ by {error:e}, above {tolerance:e}",
            T::NAME
        );
    }

    // Over [0, 1] of [n0, n1, b]: one batched plan against one plan per slice. n0 n1 is a
    // multiple of 8, so every slice starts at class 0.
    let (n0, n1, b) = (8, 6, 3);
    let mut input = sentinel_parent::<T, 3>([n0, n1, b]);
    let mut batched = sentinel_parent::<T, 3>([n0, n1, b]);
    let mut single = sentinel_parent::<T, 3>([n0, n1, b]);
    fill_view(&mut input, SEED + 1);
    let layout = Layout::over_axes(&input, &batched, &[0, 1]).unwrap();
    let plan = Planner::<T>::new()
        .c2c(&layout, Direction::Backward)
        .unwrap();
    plan.execute(&input, &mut batched).unwrap();
    let slice_layout = Layout::over_axes(
        &input.r().slice::<2>(2, 0),
        &single.r().slice::<2>(2, 0),
        &[0, 1],
    )
    .unwrap();
    let plan = Planner::<T>::new()
        .c2c(&slice_layout, Direction::Backward)
        .unwrap();
    for k in 0..b {
        plan.execute(
            &input.r().slice::<2>(2, k),
            &mut single.r_mut().slice::<2>(2, k),
        )
        .unwrap();
        let error = relative_error(
            &widen(&values_of(&batched.r().slice::<2>(2, k))),
            &widen(&values_of(&single.r().slice::<2>(2, k))),
        );
        let tolerance = c2c_tolerance::<T>(n0 * n1);
        assert!(
            error <= tolerance,
            "{} slice {k}: batched and single plans differ by {error:e}, above {tolerance:e}",
            T::NAME
        );
    }
}

#[test]
fn test_batched_against_single_plans() {
    check_batched_against_single_plans::<f64>();
    check_batched_against_single_plans::<f32>();
}

#[test]
fn test_extent_one_axis_gives_the_same_plan_and_bits() {
    // An array of shape [n, 1] and one of shape [n] give equal layouts, so one plan runs on
    // both, with equal bits.
    let n = 97;
    let mut column = sentinel_parent::<f64, 2>([n, 1]);
    let mut vector = sentinel_parent::<f64, 1>([n]);
    let values = fill_view(&mut vector, SEED);
    for (i, &value) in values.iter().enumerate() {
        column[[i, 0]] = value;
    }
    let mut column_out = sentinel_parent::<f64, 2>([n, 1]);
    let mut vector_out = sentinel_parent::<f64, 1>([n]);
    let layout = Layout::along_axis(&column, &column_out, 0).unwrap();
    assert_eq!(layout, Layout::along_axis(&vector, &vector_out, 0).unwrap());
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    plan.execute(&column, &mut column_out).unwrap();
    plan.execute(&vector, &mut vector_out).unwrap();
    assert_eq!(bits(values_of(&column_out)), bits(values_of(&vector_out)));
}

// nd-fmm's fixed-batch unit (design Sections 2.5 and 5 rule 6).

#[test]
fn test_fixed_batch_unit() {
    // Boxes of shape [n_phi, rings] as slices of a column-major parent [n_phi, rings, boxes]
    // of c64 at FFTW_ALIGNED. n_phi is a multiple of 4, so every box starts at class 0.
    let (n_phi, rings, boxes) = (140, 5, 3);
    let mut input = sentinel_parent::<f64, 3>([n_phi, rings, boxes]);
    let mut output = sentinel_parent::<f64, 3>([n_phi, rings, boxes]);
    let mut own_output = sentinel_parent::<f64, 3>([n_phi, rings, boxes]);
    fill_view(&mut input, SEED);

    let layout =
        Layout::along_axis(&input.r().slice::<2>(2, 0), &output.r().slice::<2>(2, 0), 0).unwrap();
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Backward)
        .unwrap();
    for k in 0..boxes {
        let context = format!("box {k} (seed {SEED:#x})");
        plan.execute(
            &input.r().slice::<2>(2, k),
            &mut output.r_mut().slice::<2>(2, k),
        )
        .unwrap();
        let result = values_of(&output.r().slice::<2>(2, k));

        // The same plan on a standalone aligned copy of the box.
        let box_values = values_of(&input.r().slice::<2>(2, k));
        let mut copy = sentinel_parent::<f64, 2>([n_phi, rings]);
        for (index, &value) in multi_indices([n_phi, rings]).zip(&box_values) {
            copy[index] = value;
        }
        let mut copy_output = sentinel_parent::<f64, 2>([n_phi, rings]);
        plan.execute(&copy, &mut copy_output).unwrap();
        assert_eq!(
            bits(result.iter().copied()),
            bits(values_of(&copy_output)),
            "{context}: the plan on a standalone copy differs"
        );

        // A plan made for this box with `c2c_on`.
        let own_plan = Planner::<f64>::new()
            .c2c_on(
                &mut input.r_mut().slice::<2>(2, k),
                &mut own_output.r_mut().slice::<2>(2, k),
                &layout,
                Direction::Backward,
            )
            .unwrap();
        own_plan
            .execute(
                &input.r().slice::<2>(2, k),
                &mut own_output.r_mut().slice::<2>(2, k),
            )
            .unwrap();
        assert_eq!(
            bits(result.iter().copied()),
            bits(values_of(&own_output.r().slice::<2>(2, k))),
            "{context}: the plan made for this box differs"
        );

        let reference = logical_reference([n_phi, rings], &box_values, &[0], Direction::Backward);
        assert_matches_reference(&output.r().slice::<2>(2, k), &reference, &[0], &context);
    }
}

// Negative tests (design Section 7.4). Each output's parent is a sentinel parent, unchanged
// afterwards.

fn assert_all_sentinel<T: TestReal, const N: usize>(
    parent: &crate::dense::array::AlignedDynArray<Complex<T>, N, { crate::FFTW_ALIGNED }>,
) {
    let sentinel = sentinel::<T>();
    assert!(
        bits(parent.data().unwrap().iter().copied())
            == bits(std::iter::repeat_n(sentinel, parent.number_of_elements())),
        "{}: the output's parent changed",
        T::NAME
    );
}

fn assert_mismatch(error: FftwError, side: Side) {
    assert!(
        matches!(error, FftwError::LayoutMismatch { side: found, .. } if found == side),
        "expected a LayoutMismatch on the {side} side, found {error}"
    );
}

#[test]
fn test_wrong_shape_and_strides() {
    let mut input = sentinel_parent::<f64, 2>([8, 3]);
    fill_view(&mut input, SEED);
    let mut parent = sentinel_parent::<f64, 2>([8, 5]);
    let layout = Layout::along_axis(&input, &parent.r().into_subview([0, 1], [8, 3]), 0).unwrap();
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();

    // A wrong shape: one column more.
    let error = plan
        .execute(&input, &mut parent.r_mut().into_subview([0, 1], [8, 4]))
        .unwrap_err();
    assert_mismatch(error.clone(), Side::Output);
    let message = error.to_string();
    assert!(
        message.contains("an array of shape [8, 4] and strides [1, 8]"),
        "{message}"
    );
    assert!(
        message.contains(
            "transform dimensions [(n 8, stride 1)] and batch dimensions [(n 3, stride 8)]"
        ),
        "{message}"
    );
    assert_all_sentinel(&parent);

    // Wrong strides: a row-major input against a plan from a column-major one.
    let mut row_major = StridedDynArray::<c64, 2>::row_major([8, 3]);
    fill_view(&mut row_major, SEED);
    let error = Planner::<f64>::new()
        .unaligned()
        .c2c(&layout, Direction::Forward)
        .unwrap()
        .execute(&row_major, &mut parent.r_mut().into_subview([0, 1], [8, 3]))
        .unwrap_err();
    assert_mismatch(error, Side::Input);
    assert_all_sentinel(&parent);
}

fn check_alignment_mismatch<T: TestReal>(class: usize) {
    // Views of a [9, 3] parent with strides [1, 9]: at [0, 0] class 0, at [1, 0] one element
    // further on.
    let mut input = sentinel_parent::<T, 2>([9, 3]);
    fill_view(&mut input, SEED);
    let mut parent = sentinel_parent::<T, 2>([9, 3]);
    let layout = Layout::along_axis(
        &input.r().into_subview([0, 0], [8, 3]),
        &parent.r().into_subview([0, 0], [8, 3]),
        0,
    )
    .unwrap();
    let plan = Planner::<T>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    let error = plan
        .execute(
            &input.r().into_subview([1, 0], [8, 3]),
            &mut parent.r_mut().into_subview([0, 0], [8, 3]),
        )
        .unwrap_err();
    assert_eq!(
        error,
        FftwError::AlignmentMismatch {
            side: Side::Input,
            expected: AlignmentClass::default(),
            found: AlignmentClass::from_offset(class),
        }
    );
    assert_all_sentinel(&parent);
}

#[test]
fn test_alignment_mismatch_on_views() {
    check_alignment_mismatch::<f64>(16);
    check_alignment_mismatch::<f32>(8);
}

#[test]
fn test_placement_mismatch_from_arrays() {
    let mut input = sentinel_parent::<f64, 2>([8, 3]);
    let mut output = sentinel_parent::<f64, 2>([8, 3]);
    let in_place = Layout::along_axis_inplace(&input, 0).unwrap();
    let out_of_place = Layout::along_axis(&input, &output, 0).unwrap();
    let error = Planner::<f64>::new()
        .c2c(&in_place, Direction::Forward)
        .unwrap_err();
    assert!(
        matches!(error, FftwError::PlacementMismatch { .. }),
        "{error}"
    );
    let error = Planner::<f64>::new()
        .c2c_on(&mut input, &mut output, &in_place, Direction::Forward)
        .unwrap_err();
    assert!(
        matches!(error, FftwError::PlacementMismatch { .. }),
        "{error}"
    );
    assert_all_sentinel(&output);
    let error = Planner::<f64>::new()
        .c2c_inplace(&out_of_place, Direction::Forward)
        .unwrap_err();
    assert!(
        matches!(error, FftwError::PlacementMismatch { .. }),
        "{error}"
    );
}

#[test]
fn test_overlapping_views() {
    // Two strided views, of 8 elements 2 apart, whose spans of 15 elements overlap.
    let mut buffer = filled(sentinel::<f64>(), 32);
    let layout = {
        let (left, right) = buffer.split_at(16);
        Layout::along_axis(
            &StridedSliceArray::from_shape_and_stride(left, [8], [2]),
            &StridedSliceArray::from_shape_and_stride(right, [8], [2]),
            0,
        )
        .unwrap()
    };
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    let base = buffer.as_mut_ptr();
    {
        // Input and output overlap, which safe borrows rule out. They are built from one
        // allocation only to test the check, as in F1's test on slices: the check reads the
        // views' addresses, shapes and strides and nothing else, and on the error no element
        // is read or written through either view. The output starts 8 elements, 128 bytes,
        // in, so its class is 0 like the input's.
        // SAFETY: both ranges lie inside `buffer`, which outlives the views.
        let input_slice = unsafe { std::slice::from_raw_parts(base.cast_const(), 16) };
        // SAFETY: as above.
        let output_slice = unsafe { std::slice::from_raw_parts_mut(base.add(8), 16) };
        let input = StridedSliceArray::from_shape_and_stride(input_slice, [8], [2]);
        let mut output = StridedSliceArrayMut::from_shape_and_stride(output_slice, [8], [2]);
        let error = plan.execute(&input, &mut output).unwrap_err();
        assert!(matches!(error, FftwError::Overlap { .. }), "{error}");
    }
    let sentinel = sentinel::<f64>();
    assert!(
        buffer
            .iter()
            .all(|value| value.re.to_bits() == sentinel.re.to_bits())
    );
}

#[test]
fn test_scratch_cap_on_a_column_of_a_row_major_array() {
    // A column of a row-major [4, 1000] array: 4 elements 1000 apart span 3001 elements.
    let mut input = StridedDynArray::<c64, 2>::row_major([4, 1000]);
    let mut output = StridedDynArray::<c64, 2>::row_major([4, 1000]);
    output.fill_with_value(sentinel::<f64>());
    let values = fill_view(&mut input.r_mut().col(7), SEED);
    let layout = Layout::along_axis(&input.r().col(7), &output.r().col(7), 0).unwrap();
    assert_eq!(layout.input_span(), 3001);
    let planner = Planner::<f64>::new().max_scratch(1024);
    assert_eq!(
        planner.c2c(&layout, Direction::Forward).unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: 2 * 3001 * 16,
            bound: 1024,
        }
    );

    // The same layout plans on the arrays themselves.
    let plan = planner
        .c2c_on(
            &mut input.r_mut().col(7),
            &mut output.r_mut().col(7),
            &layout,
            Direction::Forward,
        )
        .unwrap();
    plan.execute(&input.r().col(7), &mut output.r_mut().col(7))
        .unwrap();
    let context = format!("column 7 (seed {SEED:#x})");
    assert_gaps_unchanged(&output.r().col(7), &context);
    let reference = logical_reference([4], &values, &[0], Direction::Forward);
    assert_matches_reference(&output.r().col(7), &reference, &[0], &context);
}

/// A test-only array type whose storage is shorter than its shape and strides claim.
struct ShortStorage {
    data: Vec<c64>,
    claimed: usize,
}

impl BaseItem for ShortStorage {
    type Item = c64;
}

impl Shape<1> for ShortStorage {
    fn shape(&self) -> [usize; 1] {
        [self.claimed]
    }
}

impl Stride<1> for ShortStorage {
    fn stride(&self) -> [usize; 1] {
        [1]
    }
}

impl StridedRawAccess for ShortStorage {
    fn storage(&self) -> Option<(&[c64], usize)> {
        Some((&self.data, 0))
    }
}

impl StridedRawAccessMut for ShortStorage {
    fn storage_mut(&mut self) -> Option<(&mut [c64], usize)> {
        Some((&mut self.data, 0))
    }
}

#[test]
fn test_storage_shorter_than_claimed() {
    // Shape and strides match the layout, but the storage holds 8 of the 16 elements, which
    // execution finds before calling FFTW.
    let mut input = sentinel_parent::<f64, 1>([16]);
    let mut output = sentinel_parent::<f64, 1>([16]);
    fill_view(&mut input, SEED);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let plan = Planner::<f64>::new()
        .unaligned()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    let short = || {
        Array::new(ShortStorage {
            data: vec![sentinel::<f64>(); 8],
            claimed: 16,
        })
    };

    let error = plan.execute(&short(), &mut output).unwrap_err();
    assert_mismatch(error.clone(), Side::Input);
    let message = error.to_string();
    assert!(
        message.contains("at least 16 storage elements from the first element"),
        "{message}"
    );
    assert!(
        message.contains("a storage of 8 elements with the first element at offset 0"),
        "{message}"
    );
    assert_all_sentinel(&output);

    let mut short_output = short();
    let error = plan.execute(&input, &mut short_output).unwrap_err();
    assert_mismatch(error, Side::Output);
    assert!(
        short_output
            .imp()
            .data
            .iter()
            .all(|&value| value == sentinel::<f64>())
    );
}

/// A test-only array type whose mutable storage is shorter than its shared storage.
struct ShorterWhenMutable {
    data: Vec<c64>,
}

impl BaseItem for ShorterWhenMutable {
    type Item = c64;
}

impl Shape<1> for ShorterWhenMutable {
    fn shape(&self) -> [usize; 1] {
        [self.data.len()]
    }
}

impl Stride<1> for ShorterWhenMutable {
    fn stride(&self) -> [usize; 1] {
        [1]
    }
}

impl StridedRawAccess for ShorterWhenMutable {
    fn storage(&self) -> Option<(&[c64], usize)> {
        Some((&self.data, 0))
    }
}

impl StridedRawAccessMut for ShorterWhenMutable {
    fn storage_mut(&mut self) -> Option<(&mut [c64], usize)> {
        let half = self.data.len() / 2;
        Some((&mut self.data[..half], 0))
    }
}

#[test]
fn test_mutable_storage_shorter_than_shared_storage() {
    // The shared storage fits the layout, the mutable one does not: execution checks the
    // storage behind the pointer it passes to FFTW, and rejects the output.
    let mut input = sentinel_parent::<f64, 1>([16]);
    let output = sentinel_parent::<f64, 1>([16]);
    fill_view(&mut input, SEED);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let mut inconsistent = Array::new(ShorterWhenMutable {
        data: vec![sentinel::<f64>(); 16],
    });
    let error = Planner::<f64>::new()
        .unaligned()
        .c2c(&layout, Direction::Forward)
        .unwrap()
        .execute(&input, &mut inconsistent)
        .unwrap_err();
    assert_mismatch(error.clone(), Side::Output);
    assert!(
        error.to_string().contains("a storage of 8 elements"),
        "{error}"
    );
    let error = Planner::<f64>::new()
        .unaligned()
        .c2c_on(&mut input, &mut inconsistent, &layout, Direction::Forward)
        .unwrap_err();
    assert_mismatch(error, Side::Output);
    assert!(
        inconsistent
            .imp()
            .data
            .iter()
            .all(|&value| value == sentinel::<f64>())
    );
}

#[test]
fn test_axes_pair_by_index() {
    // The example of DC15: a plan from a column-major input [4, 2, 3] and an output of the
    // same shape with strides [1, 12, 4]. Column-major arrays of shape [4, 3, 2] fit each
    // side, but input column [:, 1, 0] would land in output column [:, 0, 1].
    let input = DynArray::<c64, 3>::from_shape([4, 2, 3]);
    let output = StridedDynArray::<c64, 3>::from_shape_and_stride([4, 2, 3], [1, 12, 4]);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();

    let mut other_input = sentinel_parent::<f64, 3>([4, 3, 2]);
    let mut other_output = sentinel_parent::<f64, 3>([4, 3, 2]);
    fill_view(&mut other_input, SEED);
    let error = plan.execute(&other_input, &mut other_output).unwrap_err();
    assert_mismatch(error.clone(), Side::Output);
    assert!(error.to_string().contains("pair by index"), "{error}");
    assert_all_sentinel(&other_output);

    // The same memory as slices is checked by span alone, and accepted, as documented.
    plan.execute(
        other_input.data().unwrap(),
        other_output.data_mut().unwrap(),
    )
    .unwrap();

    // A slice and an array are checked each on its own side, and accepted.
    plan.execute(other_input.data().unwrap(), &mut other_output)
        .unwrap();
}

#[test]
fn test_planning_on_views() {
    // `c2c_on` and `c2c_inplace_on` on subviews give plans for their classes that run on
    // them.
    let mut input = sentinel_parent::<f32, 2>([10, 6]);
    let mut output = sentinel_parent::<f32, 2>([10, 6]);
    let values = fill_view(&mut input.r_mut().into_subview([1, 2], [8, 3]), SEED);
    let layout = Layout::along_axis(
        &input.r().into_subview([1, 2], [8, 3]),
        &output.r().into_subview([2, 1], [8, 3]),
        0,
    )
    .unwrap();
    let plan = Planner::<f32>::new()
        .c2c_on(
            &mut input.r_mut().into_subview([1, 2], [8, 3]),
            &mut output.r_mut().into_subview([2, 1], [8, 3]),
            &layout,
            Direction::Forward,
        )
        .unwrap();
    assert_eq!(
        plan.alignment(),
        Some((
            AlignmentClass::of(&input.r().into_subview([1, 2], [8, 3])),
            AlignmentClass::of(&output.r().into_subview([2, 1], [8, 3]))
        ))
    );
    plan.execute(
        &input.r().into_subview([1, 2], [8, 3]),
        &mut output.r_mut().into_subview([2, 1], [8, 3]),
    )
    .unwrap();
    let context = format!("c2c_on (seed {SEED:#x})");
    let view = output.r().into_subview([2, 1], [8, 3]);
    assert_gaps_unchanged(&view, &context);
    let reference = logical_reference([8, 3], &values, &[0], Direction::Forward);
    assert_matches_reference(&view, &reference, &[0], &context);

    let mut data = sentinel_parent::<f32, 2>([10, 6]);
    let values = fill_view(&mut data.r_mut().into_subview([1, 2], [8, 3]), SEED);
    let layout = Layout::along_axis_inplace(&data.r().into_subview([1, 2], [8, 3]), 0).unwrap();
    let plan = Planner::<f32>::new()
        .c2c_inplace_on(
            &mut data.r_mut().into_subview([1, 2], [8, 3]),
            &layout,
            Direction::Backward,
        )
        .unwrap();
    plan.execute(&mut data.r_mut().into_subview([1, 2], [8, 3]))
        .unwrap();
    let context = format!("c2c_inplace_on (seed {SEED:#x})");
    let view = data.r().into_subview([1, 2], [8, 3]);
    assert_gaps_unchanged(&view, &context);
    let reference = logical_reference([8, 3], &values, &[0], Direction::Backward);
    assert_matches_reference(&view, &reference, &[0], &context);
}
