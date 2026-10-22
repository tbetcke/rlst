//! Execution of split complex-to-complex plans (task F3; design Sections 7.2 to 7.4).
//!
//! Every plan here is made with `Rigor::Estimate`, depends on no wisdom and assumes no SIMD.
//! A split view's complex values are read by logical multi-index from its two parts and
//! compared with the logical reference of `test_support`, which never builds a `Layout`, and
//! every view's parent outside the view, the gaps between the planes included, is checked to
//! hold its sentinel afterwards.

use std::fmt;

use num::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::dense::array::{AlignedDynArray, StridedSliceArray, StridedSliceArrayMut};
use crate::fftw::test_support::{
    PolicyKind, SEED, TestReal, assert_gaps_unchanged, assert_matches_reference, c2c_tolerance,
    fill_split_view, fill_view, filled, fixed_sizes, insert_axis, logical_reference, multi_indices,
    real_sentinel, real_sentinel_parent, relative_error, sentinel_parent, split_values,
    storage_bits, widen,
};
use crate::fftw::{AlignmentClass, Dim, Direction, FftwError, Layout, Placement, Planner, Side};
use crate::{
    Array, BaseItem, DynArray, FFTW_ALIGNED, Shape, SliceArray, SliceArrayMut, Stride,
    StridedRawAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessMut,
};

/// The real arrays and views the tests execute on.
trait View<T: TestReal, const N: usize>:
    BaseItem<Item = T>
    + StridedRawAccessMut
    + Shape<N>
    + Stride<N>
    + UnsafeRandomAccessByRef<N>
    + UnsafeRandomAccessMut<N>
{
}

impl<T: TestReal, A, const N: usize> View<T, N> for A where
    A: BaseItem<Item = T>
        + StridedRawAccessMut
        + Shape<N>
        + Stride<N>
        + UnsafeRandomAccessByRef<N>
        + UnsafeRandomAccessMut<N>
{
}

/// Where the part axis is and which views carry it (F3 brief, work item 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arrangement {
    /// Owned column-major arrays with the part axis outermost: nd-fmm's planes.
    Outermost,
    /// The part axis innermost: the memory of interleaved storage read as split.
    Innermost,
    /// The part axis after the first logical axis.
    Middle,
    /// The input's part axis outermost, the output's innermost.
    Mixed,
    /// Subviews of larger parents, part axis outermost; the input's parent has a third,
    /// sentinel plane.
    Subview,
    /// Views with steps of 2 over aligned storage, part axis outermost.
    Strided,
    /// Slices of parents of higher rank, part axis outermost.
    Slice,
    /// Row-major strides over aligned storage, part axis first, so the planes are
    /// contiguous.
    RowMajor,
}

impl Arrangement {
    const ALL: [Arrangement; 8] = [
        Arrangement::Outermost,
        Arrangement::Innermost,
        Arrangement::Middle,
        Arrangement::Mixed,
        Arrangement::Subview,
        Arrangement::Strided,
        Arrangement::Slice,
        Arrangement::RowMajor,
    ];

    /// The part axes of input and output for logical rank `m`.
    fn parts(self, m: usize) -> (usize, usize) {
        match self {
            Arrangement::Innermost | Arrangement::RowMajor => (0, 0),
            Arrangement::Middle => (1.min(m), 1.min(m)),
            Arrangement::Mixed => (m, 0),
            _ => (m, m),
        }
    }
}

/// One cell of the split correctness matrix, with the logical shape `shape`.
#[derive(Clone, Debug)]
struct Cell<const M: usize> {
    arrangement: Arrangement,
    shape: [usize; M],
    axes: Vec<usize>,
    policy: PolicyKind,
    direction: Direction,
    seed: u64,
    round_trip: bool,
}

impl<const M: usize> fmt::Display for Cell<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (part_in, part_out) = self.arrangement.parts(M);
        write!(
            f,
            "{:?}, logical shape {:?}, part axes ({part_in}, {part_out}), logical axes {:?}, \
             {:?}, {:?} (seed {:#x})",
            self.arrangement, self.shape, self.axes, self.policy, self.direction, self.seed
        )
    }
}

/// `values`, given in column-major order of `shape`, widened and multiplied by `factor`.
fn scaled<T: TestReal, const M: usize>(
    shape: [usize; M],
    values: &[Complex<T>],
    factor: f64,
) -> DynArray<Complex<f64>, M> {
    let mut result = DynArray::<Complex<f64>, M>::from_shape(shape);
    for (index, value) in multi_indices(shape).zip(widen(values)) {
        result[index] = value * factor;
    }
    result
}

/// Execute an out-of-place split plan of `policy` from `input` to `output`.
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
            .split_c2c(layout, direction)
            .unwrap()
            .execute(&*input, output)
            .unwrap(),
        PolicyKind::Destroying => planner
            .clone()
            .may_destroy_input()
            .split_c2c(layout, direction)
            .unwrap()
            .execute(input, output)
            .unwrap(),
        PolicyKind::InPlace => unreachable!("an out-of-place policy"),
    }
}

/// Assert the complex values of the split view `view` against `expected`.
fn assert_split_matches<T, A, const M: usize, const N: usize>(
    view: &Array<A, N>,
    part: usize,
    shape: [usize; M],
    expected: &DynArray<Complex<f64>, M>,
    axes: &[usize],
    context: &dyn fmt::Display,
) where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    let actual = split_values(view, part, shape);
    assert_matches_reference(&actual, expected, axes, context);
}

/// Check a cell on `input` and `output` with their part axes: the transform against the
/// logical reference, the input's storage for a preserving plan, the gaps of both views and,
/// if the cell asks for it, the round trip.
fn run_cell<T, AIn, AOut, const M: usize, const N: usize>(
    cell: &Cell<M>,
    (input, part_in): (&mut Array<AIn, N>, usize),
    (output, part_out): (&mut Array<AOut, N>, usize),
) where
    T: TestReal,
    AIn: View<T, N>,
    AOut: View<T, N>,
{
    let values = fill_split_view(input, part_in, cell.shape, cell.seed);
    let before = storage_bits(input);
    let reference = logical_reference(cell.shape, &values, &cell.axes, cell.direction);
    let factor = cell
        .axes
        .iter()
        .map(|&axis| cell.shape[axis])
        .product::<usize>() as f64;
    let expected = scaled(cell.shape, &values, factor);

    if cell.policy == PolicyKind::InPlace {
        let layout = Layout::split_over_axes_inplace(input, part_in, &cell.axes).unwrap();
        let class = AlignmentClass::of(&*input);
        let planner = Planner::<T>::new().alignment(class, class);
        let plan = planner.split_c2c_inplace(&layout, cell.direction).unwrap();
        plan.execute(input).unwrap();
        assert_gaps_unchanged(input, cell);
        assert_split_matches(input, part_in, cell.shape, &reference, &cell.axes, cell);
        if cell.round_trip {
            let inverse = match cell.direction {
                Direction::Forward => Direction::Backward,
                Direction::Backward => Direction::Forward,
            };
            let plan = planner.split_c2c_inplace(&layout, inverse).unwrap();
            plan.execute(input).unwrap();
            assert_gaps_unchanged(input, cell);
            let context = format!("{cell}, round trip");
            assert_split_matches(input, part_in, cell.shape, &expected, &cell.axes, &context);
        }
        return;
    }

    let layout = Layout::split_over_axes(input, part_in, output, part_out, &cell.axes).unwrap();
    // Plan for the classes of these arrays: views at an offset change class.
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
    assert_split_matches(output, part_out, cell.shape, &reference, &cell.axes, cell);

    if cell.round_trip {
        // The inverse transform of the output, into the input, is N times the input.
        let inverse = match cell.direction {
            Direction::Forward => Direction::Backward,
            Direction::Backward => Direction::Forward,
        };
        let layout = Layout::split_over_axes(output, part_out, input, part_in, &cell.axes).unwrap();
        let planner = Planner::<T>::new()
            .alignment(AlignmentClass::of(&*output), AlignmentClass::of(&*input));
        execute_out_of_place(&planner, &layout, inverse, cell.policy, output, input);
        assert_gaps_unchanged(input, cell);
        assert_gaps_unchanged(output, cell);
        let context = format!("{cell}, round trip");
        assert_split_matches(input, part_in, cell.shape, &expected, &cell.axes, &context);
    }
}

/// `shape` with `first` in front.
fn prepend<const N: usize, const M: usize>(first: usize, shape: [usize; N]) -> [usize; M] {
    insert_axis(shape, 0, first)
}

/// `shape` with `last` at the back.
fn append<const N: usize, const M: usize>(shape: [usize; N], last: usize) -> [usize; M] {
    insert_axis(shape, N, last)
}

/// The strides of a strided view with steps of 2: `[2, 2 n0, 2 n0 n1, …]`.
fn steps_of_two<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut stride = [2; N];
    for axis in 1..N {
        stride[axis] = stride[axis - 1] * shape[axis - 1];
    }
    stride
}

/// Row-major strides of `shape`.
fn row_major<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut stride = [1; N];
    for axis in (0..N.saturating_sub(1)).rev() {
        stride[axis] = stride[axis + 1] * shape[axis + 1];
    }
    stride
}

/// Build the views of `cell.arrangement` for logical rank `$m` (array rank `$n`, parent rank
/// `$n1` for slices) and run the cell on them. Parents are filled with the sentinel.
macro_rules! run_views {
    ($t:ty, $m:literal, $n:literal, $n1:literal, $cell:expr) => {{
        let cell: &Cell<$m> = $cell;
        let logical = cell.shape;
        let (part_in, part_out) = cell.arrangement.parts($m);
        let shape_in: [usize; $n] = insert_axis(logical, part_in, 2);
        let shape_out: [usize; $n] = insert_axis(logical, part_out, 2);
        let sentinel = real_sentinel::<$t>();
        match cell.arrangement {
            Arrangement::Outermost
            | Arrangement::Innermost
            | Arrangement::Middle
            | Arrangement::Mixed => {
                let mut input = real_sentinel_parent::<$t, $n>(shape_in);
                let mut output = real_sentinel_parent::<$t, $n>(shape_out);
                run_cell::<$t, _, _, $m, $n>(cell, (&mut input, part_in), (&mut output, part_out));
            }
            Arrangement::Subview => {
                // Three planes in the input's parent, the third left as sentinel; two in the
                // output's.
                let mut input = real_sentinel_parent::<$t, $n>(append(logical.map(|n| n + 3), 3));
                let mut output = real_sentinel_parent::<$t, $n>(append(logical.map(|n| n + 2), 2));
                run_cell::<$t, _, _, $m, $n>(
                    cell,
                    (
                        &mut input.r_mut().into_subview(append([2; $m], 0), shape_in),
                        part_in,
                    ),
                    (
                        &mut output.r_mut().into_subview(append([1; $m], 0), shape_out),
                        part_out,
                    ),
                );
            }
            Arrangement::Strided => {
                let len = 2 * shape_in.iter().product::<usize>();
                let mut input = filled(sentinel, len);
                let mut output = filled(sentinel, len);
                run_cell::<$t, _, _, $m, $n>(
                    cell,
                    (
                        &mut StridedSliceArrayMut::from_shape_and_stride(
                            &mut input[..],
                            shape_in,
                            steps_of_two(shape_in),
                        ),
                        part_in,
                    ),
                    (
                        &mut StridedSliceArrayMut::from_shape_and_stride(
                            &mut output[..],
                            shape_out,
                            steps_of_two(shape_out),
                        ),
                        part_out,
                    ),
                );
            }
            Arrangement::Slice => {
                let mut input = real_sentinel_parent::<$t, $n1>(prepend(3, shape_in));
                let mut output = real_sentinel_parent::<$t, $n1>(append(shape_out, 3));
                run_cell::<$t, _, _, $m, $n>(
                    cell,
                    (&mut input.r_mut().slice::<$n>(0, 1), part_in),
                    (&mut output.r_mut().slice::<$n>($n, 2), part_out),
                );
            }
            Arrangement::RowMajor => {
                let len = shape_in.iter().product::<usize>();
                let mut input = filled(sentinel, len);
                let mut output = filled(sentinel, len);
                run_cell::<$t, _, _, $m, $n>(
                    cell,
                    (
                        &mut StridedSliceArrayMut::from_shape_and_stride(
                            &mut input[..],
                            shape_in,
                            row_major(shape_in),
                        ),
                        part_in,
                    ),
                    (
                        &mut StridedSliceArrayMut::from_shape_and_stride(
                            &mut output[..],
                            shape_out,
                            row_major(shape_out),
                        ),
                        part_out,
                    ),
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

/// The seeds of the fixed cells: distinct per arrangement and cell.
struct Seeds(u64);

impl Seeds {
    fn new(arrangement: Arrangement) -> Self {
        let index = Arrangement::ALL
            .iter()
            .position(|&a| a == arrangement)
            .unwrap() as u64;
        Self(SEED ^ ((index + 16) << 32))
    }

    fn next(&mut self) -> u64 {
        self.0 += 1;
        self.0
    }
}

/// The fixed cells of an arrangement in precision `T` and `direction`, for the three
/// policies: `[n, 3]` and `[n]` along axis 0 with n in `sizes`, and `[6, 5, 4]` over two and
/// three logical axes. The round trip runs once, on the first `[n, 3]` cell of each policy.
fn run_fixed_cells<T: TestReal>(arrangement: Arrangement, direction: Direction, sizes: &[usize]) {
    let mut seeds = Seeds::new(arrangement);
    for policy in PolicyKind::ALL {
        for (index, &n) in sizes.iter().enumerate() {
            run_rank2::<T>(&Cell {
                arrangement,
                shape: [n, 3],
                axes: vec![0],
                policy,
                direction,
                seed: seeds.next(),
                round_trip: index == 0,
            });
            if arrangement != Arrangement::Middle {
                run_rank1::<T>(&Cell {
                    arrangement,
                    shape: [n],
                    axes: vec![0],
                    policy,
                    direction,
                    seed: seeds.next(),
                    round_trip: false,
                });
            }
        }
        for axes in [vec![0, 1], vec![2, 0], vec![0, 1, 2]] {
            run_rank3::<T>(&Cell {
                arrangement,
                shape: [6, 5, 4],
                axes,
                policy,
                direction,
                seed: seeds.next(),
                round_trip: false,
            });
        }
    }
}

/// nd-fmm's sizes for the planes; the other arrangements use two of them.
const PLANE_SIZES: [usize; 5] = [1, 2, 17, 97, 140];
const OTHER_SIZES: [usize; 2] = [17, 2];

fn run_both_precisions_and_directions(arrangement: Arrangement, sizes: &[usize]) {
    for direction in [Direction::Forward, Direction::Backward] {
        run_fixed_cells::<f64>(arrangement, direction, sizes);
        run_fixed_cells::<f32>(arrangement, direction, sizes);
    }
}

#[test]
fn test_split_cells_planes() {
    run_both_precisions_and_directions(Arrangement::Outermost, &PLANE_SIZES);
}

#[test]
fn test_split_cells_interleaved_memory() {
    run_both_precisions_and_directions(Arrangement::Innermost, &OTHER_SIZES);
}

#[test]
fn test_split_cells_part_axis_in_the_middle() {
    run_both_precisions_and_directions(Arrangement::Middle, &OTHER_SIZES);
}

#[test]
fn test_split_cells_mixed_part_axes() {
    run_both_precisions_and_directions(Arrangement::Mixed, &OTHER_SIZES);
}

#[test]
fn test_split_cells_views() {
    for arrangement in [
        Arrangement::Subview,
        Arrangement::Strided,
        Arrangement::Slice,
        Arrangement::RowMajor,
    ] {
        run_both_precisions_and_directions(arrangement, &OTHER_SIZES[..1]);
    }
}

#[test]
fn test_split_cells_sampled() {
    // 16 cells from the product of arrangements, F1's sizes up to 588, batches, logical
    // ranks, axes, policies, directions and precisions, drawn with the fixed seed.
    let sizes: Vec<usize> = fixed_sizes().into_iter().filter(|&n| n <= 588).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x5b1);
    for _ in 0..16 {
        let arrangement = Arrangement::ALL[rng.random_range(0..Arrangement::ALL.len())];
        let rank = rng.random_range(1..=3_usize);
        let policy = PolicyKind::ALL[rng.random_range(0..3)];
        let direction = if rng.random_bool(0.5) {
            Direction::Forward
        } else {
            Direction::Backward
        };
        let single = rng.random_bool(0.5);
        let seed = rng.random();
        macro_rules! sampled {
            ($run:ident, $shape:expr, $axes:expr) => {{
                let cell = Cell {
                    arrangement,
                    shape: $shape,
                    axes: $axes,
                    policy,
                    direction,
                    seed,
                    round_trip: false,
                };
                if single {
                    $run::<f32>(&cell);
                } else {
                    $run::<f64>(&cell);
                }
            }};
        }
        let n = sizes[rng.random_range(0..sizes.len())];
        match rank {
            1 => sampled!(run_rank1, [n], vec![0]),
            2 => {
                let b = rng.random_range(1..=4_usize);
                let axes = match rng.random_range(0..3) {
                    0 => vec![0],
                    1 => vec![1, 0],
                    _ => vec![0, 1],
                };
                // Two-dimensional transforms of the largest sizes would make the reference
                // slow in debug builds.
                let n = if axes.len() == 2 { n.min(64) } else { n };
                sampled!(run_rank2, [n, b], axes)
            }
            _ => {
                let mut extent = || rng.random_range(1..=8_usize);
                let shape = [extent(), extent(), extent()];
                let mut axes: Vec<usize> = (0..3).filter(|_| rng.random_bool(0.5)).collect();
                if axes.is_empty() {
                    axes.push(rng.random_range(0..3));
                }
                if axes.len() > 1 && rng.random_bool(0.5) {
                    axes.reverse();
                }
                sampled!(run_rank3, shape, axes)
            }
        }
    }
}

// Identities (design Section 7.3).

fn bits<T: TestReal>(values: impl IntoIterator<Item = T>) -> Vec<u64> {
    values
        .into_iter()
        .map(|value| value.to_f64().to_bits())
        .collect()
}

fn reals_of<T, A, const N: usize>(view: &Array<A, N>) -> Vec<T>
where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    multi_indices(view.shape())
        .map(|index| *view.get(index).unwrap())
        .collect()
}

#[test]
fn test_split_against_interleaved() {
    // nd-fmm's units [n_phi, 9]: the split plan on planes [n_phi, 9, 2] against the
    // interleaved plan on the same values. The plans differ (design, Section 3 item 7), so
    // they agree to round-off, not bit for bit.
    const RINGS: usize = 9;
    for n_phi in [140, 588] {
        for direction in [Direction::Forward, Direction::Backward] {
            let context = format!("n_phi {n_phi}, {direction:?} (seed {SEED:#x})");
            let shape = [n_phi, RINGS];
            let mut planes = real_sentinel_parent::<f64, 3>([n_phi, RINGS, 2]);
            let mut planes_out = real_sentinel_parent::<f64, 3>([n_phi, RINGS, 2]);
            let values = fill_split_view(&mut planes, 2, shape, SEED);
            let layout = Layout::split_along_axis(&planes, 2, &planes_out, 2, 0).unwrap();
            Planner::<f64>::new()
                .split_c2c(&layout, direction)
                .unwrap()
                .execute(&planes, &mut planes_out)
                .unwrap();
            let split = split_values(&planes_out, 2, shape);

            let mut interleaved = sentinel_parent::<f64, 2>(shape);
            let mut interleaved_out = sentinel_parent::<f64, 2>(shape);
            for (index, &value) in multi_indices(shape).zip(&values) {
                interleaved[index] = value;
            }
            let layout = Layout::along_axis(&interleaved, &interleaved_out, 0).unwrap();
            Planner::<f64>::new()
                .c2c(&layout, direction)
                .unwrap()
                .execute(&interleaved, &mut interleaved_out)
                .unwrap();

            let tolerance = c2c_tolerance::<f64>(n_phi);
            for ring in 0..RINGS {
                let split: Vec<Complex<f64>> = (0..n_phi).map(|i| split[[i, ring]]).collect();
                let interleaved: Vec<Complex<f64>> =
                    (0..n_phi).map(|i| interleaved_out[[i, ring]]).collect();
                let error = relative_error(&split, &interleaved);
                assert!(
                    error <= tolerance,
                    "{context}, ring {ring}: split and interleaved differ by {error:e}, above \
                     {tolerance:e}"
                );
            }
        }
    }
}

#[test]
fn test_one_split_plan_many_units() {
    // One plan of nd-fmm's box [140, 9, 2] executed on every box `slice(3, k)` of a parent
    // [140, 9, 2, 4]: the bits of a plan made with `split_c2c_on` for each box and of the same
    // plan on a standalone copy. A box is 2520 reals, 20160 bytes, a multiple of 64, so every
    // box is at class 0.
    let (n_phi, rings, boxes) = (140, 9, 4);
    let shape = [n_phi, rings];
    let mut input = real_sentinel_parent::<f64, 4>([n_phi, rings, 2, boxes]);
    let mut output = real_sentinel_parent::<f64, 4>([n_phi, rings, 2, boxes]);
    let mut own_output = real_sentinel_parent::<f64, 4>([n_phi, rings, 2, boxes]);
    for k in 0..boxes {
        fill_split_view(
            &mut input.r_mut().slice::<3>(3, k),
            2,
            shape,
            SEED + k as u64,
        );
    }
    let layout = Layout::split_along_axis(
        &input.r().slice::<3>(3, 0),
        2,
        &output.r().slice::<3>(3, 0),
        2,
        0,
    )
    .unwrap();
    let plan = Planner::<f64>::new()
        .split_c2c(&layout, Direction::Backward)
        .unwrap();
    for k in 0..boxes {
        let context = format!("box {k} (seed {:#x})", SEED + k as u64);
        plan.execute(
            &input.r().slice::<3>(3, k),
            &mut output.r_mut().slice::<3>(3, k),
        )
        .unwrap();
        let result = reals_of(&output.r().slice::<3>(3, k));

        // The same plan on a standalone aligned copy of the box.
        let mut copy = real_sentinel_parent::<f64, 3>([n_phi, rings, 2]);
        for index in multi_indices([n_phi, rings, 2]) {
            copy[index] = input.r().slice::<3>(3, k)[index];
        }
        let mut copy_output = real_sentinel_parent::<f64, 3>([n_phi, rings, 2]);
        plan.execute(&copy, &mut copy_output).unwrap();
        assert_eq!(
            bits(result.iter().copied()),
            bits(reals_of(&copy_output)),
            "{context}: the plan on a standalone copy differs"
        );

        // A plan made for this box with `split_c2c_on`.
        let own_plan = Planner::<f64>::new()
            .split_c2c_on(
                &mut input.r_mut().slice::<3>(3, k),
                &mut own_output.r_mut().slice::<3>(3, k),
                &layout,
                Direction::Backward,
            )
            .unwrap();
        own_plan
            .execute(
                &input.r().slice::<3>(3, k),
                &mut own_output.r_mut().slice::<3>(3, k),
            )
            .unwrap();
        assert_eq!(
            bits(result.iter().copied()),
            bits(reals_of(&own_output.r().slice::<3>(3, k))),
            "{context}: the plan made for this box differs"
        );

        let box_values = split_values(&input.r().slice::<3>(3, k), 2, shape);
        let values: Vec<Complex<f64>> = multi_indices(shape)
            .map(|index| box_values[index])
            .collect();
        let reference = logical_reference(shape, &values, &[0], Direction::Backward);
        assert_split_matches(
            &output.r().slice::<3>(3, k),
            2,
            shape,
            &reference,
            &[0],
            &context,
        );
    }
}

#[test]
fn test_split_on_slices() {
    // A raw split layout on slices: two transforms of length 12, real plane then imaginary
    // plane, the output interleaved, against the reference.
    let layout = Layout::new_split(
        &[Dim {
            n: 12,
            is: 1,
            os: 2,
        }],
        &[Dim {
            n: 2,
            is: 12,
            os: 24,
        }],
        (24, 1),
        Placement::OutOfPlace,
    )
    .unwrap();
    assert_eq!((layout.input_span(), layout.output_span()), (48, 48));
    let mut planes = real_sentinel_parent::<f32, 3>([12, 2, 2]);
    let values = fill_split_view(&mut planes, 2, [12, 2], SEED);
    let mut output = filled(real_sentinel::<f32>(), 48);
    Planner::<f32>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap()
        .execute(planes.data().unwrap(), &mut output[..])
        .unwrap();
    let interleaved = SliceArray::from_shape(&output[..], [2, 12, 2]);
    let reference = logical_reference([12, 2], &values, &[0], Direction::Forward);
    assert_split_matches(&interleaved, 0, [12, 2], &reference, &[0], &"slices");

    // A slice one real short of the span.
    let error = Planner::<f32>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap()
        .execute(planes.data().unwrap(), &mut output[..47])
        .unwrap_err();
    assert_mismatch(error, Side::Output);
}

// Negative tests (design Section 7.4). Each output's parent is a sentinel parent, unchanged
// afterwards.

fn assert_all_sentinel<T: TestReal, const N: usize>(
    parent: &AlignedDynArray<T, N, FFTW_ALIGNED>,
    context: &str,
) {
    assert!(
        bits(parent.data().unwrap().iter().copied())
            == bits(std::iter::repeat_n(
                real_sentinel::<T>(),
                parent.number_of_elements()
            )),
        "{} {context}: the output's parent changed",
        T::NAME
    );
}

fn assert_mismatch(error: FftwError, side: Side) {
    assert!(
        matches!(error, FftwError::LayoutMismatch { side: found, .. } if found == side),
        "expected a LayoutMismatch on the {side} side, found {error}"
    );
}

fn assert_part_axis(error: FftwError) {
    assert!(
        matches!(&error, FftwError::InvalidLayout { dimension, .. } if dimension == "part axis"),
        "expected an InvalidLayout naming the part axis, found {error}"
    );
}

#[test]
fn test_wrong_gap_and_shape() {
    let mut input = real_sentinel_parent::<f64, 3>([8, 3, 2]);
    fill_split_view(&mut input, 2, [8, 3], SEED);
    let layout = Layout::split_along_axis(&input, 2, &input, 2, 0).unwrap();
    let plan = Planner::<f64>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap();

    // A wrong gap: the planes of a box with one ring more, of which the view takes three.
    let mut parent = real_sentinel_parent::<f64, 3>([8, 4, 2]);
    let error = plan
        .execute(
            &input,
            &mut parent.r_mut().into_subview([0, 0, 0], [8, 3, 2]),
        )
        .unwrap_err();
    assert_mismatch(error.clone(), Side::Output);
    let message = error.to_string();
    assert!(
        message.contains("and a part axis (n 2, stride 24)"),
        "{message}"
    );
    assert!(
        message.contains("an array of shape [8, 3, 2] and strides [1, 8, 32]"),
        "{message}"
    );
    assert_all_sentinel(&parent, "wrong gap");

    // A wrong logical shape with the right gap: two rings of the planes of three.
    let mut planes = real_sentinel_parent::<f64, 3>([8, 3, 2]);
    let error = plan
        .execute(
            &input,
            &mut planes.r_mut().into_subview([0, 0, 0], [8, 2, 2]),
        )
        .unwrap_err();
    assert_mismatch(error, Side::Output);
    assert_all_sentinel(&planes, "wrong logical shape");

    // A part axis of extent 3 has no part axis of extent 2.
    let mut three = real_sentinel_parent::<f64, 3>([8, 3, 3]);
    let error = plan
        .execute(&input.r().into_subview([0, 0, 0], [8, 3, 2]), &mut three)
        .unwrap_err();
    assert_mismatch(error, Side::Output);
    assert_all_sentinel(&three, "part extent 3");

    // The planes swapped into interleaved storage fit each side's gap only on their own
    // side: an interleaved output for a plan of planes.
    let mut interleaved = real_sentinel_parent::<f64, 3>([2, 8, 3]);
    let error = plan.execute(&input, &mut interleaved).unwrap_err();
    assert_mismatch(error, Side::Output);
    assert_all_sentinel(&interleaved, "interleaved output");
}

#[test]
fn test_split_layout_construction_errors() {
    let a = real_sentinel_parent::<f64, 3>([8, 3, 3]);
    assert_part_axis(Layout::split_along_axis(&a, 2, &a, 2, 0).unwrap_err());
    let b = real_sentinel_parent::<f64, 3>([8, 3, 2]);
    assert_part_axis(Layout::split_along_axis(&b, 3, &b, 2, 0).unwrap_err());
    let error = Layout::split_along_axis(&b, 2, &b, 2, 2).unwrap_err();
    assert!(
        matches!(&error, FftwError::InvalidLayout { dimension, reason }
            if dimension == "axes" && reason.contains("logical axis 2")),
        "{error}"
    );
    let c = real_sentinel_parent::<f64, 3>([8, 4, 2]);
    let error = Layout::split_along_axis(&b, 2, &c, 2, 0).unwrap_err();
    assert!(
        matches!(&error, FftwError::InvalidLayout { dimension, .. }
            if dimension == "logical axis 1"),
        "{error}"
    );
}

fn check_alignment_mismatch<T: TestReal>(class: usize) {
    // Views of a [9, 3, 2] parent: at [0, 0, 0] class 0, at [1, 0, 0] one real further on.
    let mut input = real_sentinel_parent::<T, 3>([9, 3, 2]);
    fill_split_view(
        &mut input.r_mut().into_subview([1, 0, 0], [8, 3, 2]),
        2,
        [8, 3],
        SEED,
    );
    let mut parent = real_sentinel_parent::<T, 3>([9, 3, 2]);
    let layout = Layout::split_along_axis(
        &input.r().into_subview([0, 0, 0], [8, 3, 2]),
        2,
        &parent.r().into_subview([0, 0, 0], [8, 3, 2]),
        2,
        0,
    )
    .unwrap();
    let plan = Planner::<T>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap();
    let error = plan
        .execute(
            &input.r().into_subview([1, 0, 0], [8, 3, 2]),
            &mut parent.r_mut().into_subview([0, 0, 0], [8, 3, 2]),
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
    assert_all_sentinel(&parent, "another class");

    // A plan for that class runs on it; the classes of the imaginary parts follow.
    let plan = Planner::<T>::new()
        .alignment(
            AlignmentClass::from_offset(class),
            AlignmentClass::default(),
        )
        .split_c2c(&layout, Direction::Forward)
        .unwrap();
    plan.execute(
        &input.r().into_subview([1, 0, 0], [8, 3, 2]),
        &mut parent.r_mut().into_subview([0, 0, 0], [8, 3, 2]),
    )
    .unwrap();

    // A class that is not a multiple of the real's alignment is rejected before planning.
    let error = Planner::<T>::new()
        .alignment(AlignmentClass::from_offset(2), AlignmentClass::default())
        .split_c2c(&layout, Direction::Forward)
        .unwrap_err();
    assert!(
        matches!(&error, FftwError::InvalidLayout { dimension, .. } if dimension == "alignment"),
        "{error}"
    );
}

#[test]
fn test_split_alignment_mismatch() {
    check_alignment_mismatch::<f64>(8);
    check_alignment_mismatch::<f32>(4);
}

#[test]
fn test_split_placement_mismatch() {
    let mut input = real_sentinel_parent::<f64, 3>([8, 3, 2]);
    let mut output = real_sentinel_parent::<f64, 3>([8, 3, 2]);
    let in_place = Layout::split_along_axis_inplace(&input, 2, 0).unwrap();
    let out_of_place = Layout::split_along_axis(&input, 2, &output, 2, 0).unwrap();
    let placement = |error: FftwError| {
        assert!(
            matches!(error, FftwError::PlacementMismatch { .. }),
            "{error}"
        )
    };
    placement(
        Planner::<f64>::new()
            .split_c2c(&in_place, Direction::Forward)
            .unwrap_err(),
    );
    placement(
        Planner::<f64>::new()
            .split_c2c_on(&mut input, &mut output, &in_place, Direction::Forward)
            .unwrap_err(),
    );
    placement(
        Planner::<f64>::new()
            .split_c2c_inplace(&out_of_place, Direction::Forward)
            .unwrap_err(),
    );
    placement(
        Planner::<f64>::new()
            .split_c2c_inplace_on(&mut output, &out_of_place, Direction::Forward)
            .unwrap_err(),
    );
    assert_all_sentinel(&output, "placement");
}

#[test]
fn test_overlapping_planes() {
    // Two split views of [8, 2] whose spans of 16 reals overlap by 8.
    let mut buffer = filled(real_sentinel::<f64>(), 32);
    let layout = {
        let (left, right) = buffer.split_at(16);
        Layout::split_along_axis(
            &SliceArray::from_shape(left, [8, 2]),
            1,
            &SliceArray::from_shape(right, [8, 2]),
            1,
            0,
        )
        .unwrap()
    };
    let plan = Planner::<f64>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap();
    let base = buffer.as_mut_ptr();
    {
        // Input and output overlap, which safe borrows rule out. They are built from one
        // allocation only to test the check, as in F1's and F2's tests: the check reads the
        // views' addresses, shapes and strides and nothing else, and on the error no element
        // is read or written through either view. The output starts 8 reals, 64 bytes, in,
        // so its class is 0 like the input's.
        // SAFETY: both ranges lie inside `buffer`, which outlives the views.
        let input_slice = unsafe { std::slice::from_raw_parts(base.cast_const(), 16) };
        // SAFETY: as above.
        let output_slice = unsafe { std::slice::from_raw_parts_mut(base.add(8), 16) };
        let input = SliceArray::from_shape(input_slice, [8, 2]);
        let mut output = SliceArrayMut::from_shape(output_slice, [8, 2]);
        let error = plan.execute(&input, &mut output).unwrap_err();
        assert!(matches!(error, FftwError::Overlap { .. }), "{error}");
    }
    assert!(
        buffer
            .iter()
            .all(|value| value.to_bits() == real_sentinel::<f64>().to_bits())
    );
}

#[test]
fn test_layouts_of_the_other_storage() {
    // A split layout given to the interleaved planner methods, and an interleaved layout to
    // the split ones (DC28), before FFTW is called.
    let split_out = Layout::new_split(
        &[Dim { n: 8, is: 1, os: 1 }],
        &[],
        (8, 8),
        Placement::OutOfPlace,
    )
    .unwrap();
    let split_in = Layout::new_split(
        &[Dim { n: 8, is: 1, os: 1 }],
        &[],
        (8, 8),
        Placement::InPlace,
    )
    .unwrap();
    let interleaved_out =
        Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace).unwrap();
    let interleaved_in =
        Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::InPlace).unwrap();
    let planner = Planner::<f64>::new();

    let error = planner.c2c(&split_out, Direction::Forward).unwrap_err();
    assert_part_axis(error.clone());
    assert!(error.to_string().contains("split_c2c"), "{error}");
    assert_part_axis(
        planner
            .c2c_inplace(&split_in, Direction::Forward)
            .unwrap_err(),
    );
    let mut complex_in = sentinel_parent::<f64, 1>([16]);
    let mut complex_out = sentinel_parent::<f64, 1>([16]);
    assert_part_axis(
        planner
            .c2c_on(
                &mut complex_in,
                &mut complex_out,
                &split_out,
                Direction::Forward,
            )
            .unwrap_err(),
    );
    assert_part_axis(
        planner
            .c2c_inplace_on(&mut complex_in, &split_in, Direction::Forward)
            .unwrap_err(),
    );

    let error = planner
        .split_c2c(&interleaved_out, Direction::Forward)
        .unwrap_err();
    assert_part_axis(error.clone());
    assert!(error.to_string().contains("new_split"), "{error}");
    assert_part_axis(
        planner
            .split_c2c_inplace(&interleaved_in, Direction::Forward)
            .unwrap_err(),
    );
    let mut real_in = real_sentinel_parent::<f64, 1>([16]);
    let mut real_out = real_sentinel_parent::<f64, 1>([16]);
    assert_part_axis(
        planner
            .split_c2c_on(
                &mut real_in,
                &mut real_out,
                &interleaved_out,
                Direction::Forward,
            )
            .unwrap_err(),
    );
    assert_part_axis(
        planner
            .split_c2c_inplace_on(&mut real_in, &interleaved_in, Direction::Forward)
            .unwrap_err(),
    );
    assert_all_sentinel(&real_out, "other storage");
}

#[test]
fn test_split_scratch_cap() {
    // A column of row-major planes [4, 1000, 2]: 4 elements 2000 reals apart, with a gap of 1,
    // span 6002 reals.
    let shape = [4, 1000, 2];
    let (origin, column) = ([0, 7, 0], [4, 1, 2]);
    let mut input_buffer = filled(real_sentinel::<f64>(), 8000);
    let mut output_buffer = filled(real_sentinel::<f64>(), 8000);
    let mut input =
        StridedSliceArrayMut::from_shape_and_stride(&mut input_buffer[..], shape, row_major(shape));
    let mut output = StridedSliceArrayMut::from_shape_and_stride(
        &mut output_buffer[..],
        shape,
        row_major(shape),
    );
    let values = fill_split_view(
        &mut input.r_mut().into_subview(origin, column),
        2,
        [4, 1],
        SEED,
    );
    let layout = Layout::split_along_axis(
        &input.r().into_subview(origin, column),
        2,
        &output.r().into_subview(origin, column),
        2,
        0,
    )
    .unwrap();
    assert_eq!(layout.input_span(), 6002);
    let planner = Planner::<f64>::new().max_scratch(1024);
    assert_eq!(
        planner.split_c2c(&layout, Direction::Forward).unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: 2 * 6002 * 8,
            bound: 1024,
        }
    );

    // The same layout plans on the arrays themselves.
    let plan = planner
        .split_c2c_on(
            &mut input.r_mut().into_subview(origin, column),
            &mut output.r_mut().into_subview(origin, column),
            &layout,
            Direction::Forward,
        )
        .unwrap();
    plan.execute(
        &input.r().into_subview(origin, column),
        &mut output.r_mut().into_subview(origin, column),
    )
    .unwrap();
    let context = format!("column 7 (seed {SEED:#x})");
    let view = output.r().into_subview(origin, column);
    assert_gaps_unchanged(&view, &context);
    let reference = logical_reference([4, 1], &values, &[0], Direction::Forward);
    assert_split_matches(&view, 2, [4, 1], &reference, &[0], &context);

    // In place, the scratch is one span.
    let layout =
        Layout::split_along_axis_inplace(&input.r().into_subview(origin, column), 2, 0).unwrap();
    assert_eq!(
        planner
            .split_c2c_inplace(&layout, Direction::Forward)
            .unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: 6002 * 8,
            bound: 1024,
        }
    );
}

#[test]
fn test_split_plans_on_strided_slices() {
    // `split_c2c_inplace_on` on a strided view with steps of 2, and its plan's class.
    let mut buffer = filled(real_sentinel::<f32>(), 2 * 2 * 16 * 3);
    let shape = [16, 3, 2];
    let mut view =
        StridedSliceArrayMut::from_shape_and_stride(&mut buffer[..], shape, steps_of_two(shape));
    let values = fill_split_view(&mut view, 2, [16, 3], SEED);
    let layout = Layout::split_along_axis_inplace(&view, 2, 0).unwrap();
    let plan = Planner::<f32>::new()
        .split_c2c_inplace_on(&mut view, &layout, Direction::Backward)
        .unwrap();
    let class = AlignmentClass::of(&view);
    assert_eq!(plan.alignment(), Some((class, class)));
    assert_eq!(plan.direction(), Direction::Backward);
    plan.execute(&mut view).unwrap();
    let context = format!("split_c2c_inplace_on (seed {SEED:#x})");
    assert_gaps_unchanged(&view, &context);
    let reference = logical_reference([16, 3], &values, &[0], Direction::Backward);
    assert_split_matches(&view, 2, [16, 3], &reference, &[0], &context);
    let view = StridedSliceArray::from_shape_and_stride(&buffer[..], shape, steps_of_two(shape));
    assert_split_matches(&view, 2, [16, 3], &reference, &[0], &context);
}

#[test]
fn test_split_plans_are_send_and_sync_and_described() {
    fn send_and_sync<P: Send + Sync>() {}
    send_and_sync::<crate::fftw::Plan<f64, crate::fftw::SplitC2c, crate::fftw::Preserving>>();
    send_and_sync::<crate::fftw::Plan<f32, crate::fftw::SplitC2c, crate::fftw::Destroying>>();
    send_and_sync::<crate::fftw::Plan<f64, crate::fftw::SplitC2c, crate::fftw::InPlace>>();

    let layout = Layout::new_split(
        &[Dim {
            n: 16,
            is: 1,
            os: 1,
        }],
        &[],
        (16, 16),
        Placement::OutOfPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap();
    assert!(!plan.description().is_empty());
    let again = Planner::<f64>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap();
    assert_eq!(plan.fingerprint(), again.fingerprint());
    let debug = format!("{plan:?}");
    assert!(debug.contains("SplitC2c"), "{debug}");
    assert!(
        debug.contains("storage: (Split { gap: 16 }, Split { gap: 16 })"),
        "{debug}"
    );
    assert_eq!(plan.layout(), &layout);
    // Plans are destroyed under the lock when dropped, from any thread.
    std::thread::spawn(move || drop(again)).join().unwrap();
}

#[test]
fn test_interleaved_and_split_views_of_one_buffer() {
    // An interleaved array of c64 and the same memory read as split with part axis 0 give
    // the same transform, to round-off: fill_view writes complex values, the split view reads
    // them back as its two parts.
    let n = 24;
    let mut complex = sentinel_parent::<f64, 1>([n]);
    let values = fill_view(&mut complex, SEED);
    let reals: Vec<f64> = complex
        .data()
        .unwrap()
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect();
    let mut data = real_sentinel_parent::<f64, 2>([2, n]);
    data.data_mut().unwrap().copy_from_slice(&reals);
    assert_eq!(
        split_values(&data, 0, [n])[[5]],
        values[5],
        "the split view reads the interleaved values"
    );
    let layout = Layout::split_along_axis_inplace(&data, 0, 0).unwrap();
    Planner::<f64>::new()
        .split_c2c_inplace(&layout, Direction::Forward)
        .unwrap()
        .execute(&mut data)
        .unwrap();
    let reference = logical_reference([n], &values, &[0], Direction::Forward);
    assert_split_matches(&data, 0, [n], &reference, &[0], &"interleaved memory");
}
