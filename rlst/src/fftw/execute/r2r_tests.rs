//! Real-to-real transforms with kinds per axis (task F5b; design Sections 7.1 to 7.4).
//!
//! Every plan here is made with `Rigor::Estimate`, depends on no wisdom and assumes no SIMD.
//! Results are compared with the r2r reference of `test_support`, which applies the O(n²)
//! definition of each kind along its own axis by multi-index and never builds a `Layout`, and
//! every view's parent outside the view is checked to hold its sentinel afterwards.

use std::fmt;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::fftw::test_support::{
    PolicyKind, SEED, TestElement, TestReal, aligned, assert_gaps_unchanged, filled, multi_indices,
    random_reals, real_sentinel, real_sentinel_parent, reference_r2r, storage_bits,
};
use crate::fftw::{
    AlignmentClass, Destroying, Dim, Direction, FftwError, InPlace, Layout, Placement, Plan,
    Planner, Preserving, R2r, R2rKind, Scale, Side, Storage,
};
use crate::{
    Array, BaseItem, DynArray, Shape, Stride, StridedDynArray, StridedRawAccessMut,
    UnsafeRandomAccessByRef, UnsafeRandomAccessMut,
};

/// The constant c of the r2r tolerance c · ε · (1 + log₂ N), fixed for the family (design
/// Section 7.1).
const R2R_TOLERANCE_FACTOR: f64 = 4.0;

/// The tolerance for the logical size `n`.
fn tolerance<T: TestReal>(n: usize) -> f64 {
    R2R_TOLERANCE_FACTOR * T::EPSILON * (1.0 + (n as f64).log2())
}

/// The real arrays and views the tests execute on.
trait RealView<T: TestReal, const N: usize>:
    BaseItem<Item = T>
    + StridedRawAccessMut
    + Shape<N>
    + Stride<N>
    + UnsafeRandomAccessByRef<N>
    + UnsafeRandomAccessMut<N>
{
}

impl<T: TestReal, A, const N: usize> RealView<T, N> for A where
    A: BaseItem<Item = T>
        + StridedRawAccessMut
        + Shape<N>
        + Stride<N>
        + UnsafeRandomAccessByRef<N>
        + UnsafeRandomAccessMut<N>
{
}

/// The inverse of each kind: the kind that, after it, multiplies the data by the logical size.
fn inverse(kind: R2rKind) -> R2rKind {
    match kind {
        R2rKind::R2hc => R2rKind::Hc2r,
        R2rKind::Hc2r => R2rKind::R2hc,
        R2rKind::Redft10 => R2rKind::Redft01,
        R2rKind::Redft01 => R2rKind::Redft10,
        R2rKind::Rodft10 => R2rKind::Rodft01,
        R2rKind::Rodft01 => R2rKind::Rodft10,
        other => other,
    }
}

/// The memory orders of the r2r cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// An owned column-major array at `FFTW_ALIGNED`.
    ColumnMajor,
    /// An owned row-major array (`StridedDynArray::row_major`), `Vec`-backed.
    RowMajor,
    /// A subview of a larger parent, at a non-zero origin.
    Subview,
}

impl Order {
    const ALL: [Order; 3] = [Order::ColumnMajor, Order::RowMajor, Order::Subview];
}

/// The input and reference of one shape, its axes and kinds, in one precision.
struct Case<T, const N: usize> {
    shape: [usize; N],
    axes: Vec<usize>,
    kinds: Vec<R2rKind>,
    seed: u64,
    values: Vec<T>,
    expected: DynArray<f64, N>,
}

impl<T: TestReal, const N: usize> Case<T, N> {
    fn new(shape: [usize; N], axes: &[usize], kinds: &[R2rKind], seed: u64) -> Self {
        let values = random_reals::<T>(seed, shape.iter().product());
        let expected = reference_r2r(shape, &values, axes, kinds);
        Self {
            shape,
            axes: axes.to_vec(),
            kinds: kinds.to_vec(),
            seed,
            values,
            expected,
        }
    }

    /// The product of the kinds' logical sizes.
    fn logical_size(&self) -> usize {
        self.axes
            .iter()
            .zip(&self.kinds)
            .map(|(&axis, kind)| kind.logical_size(self.shape[axis]))
            .product()
    }

    fn inverse_kinds(&self) -> Vec<R2rKind> {
        self.kinds.iter().map(|&kind| inverse(kind)).collect()
    }
}

/// One cell: a case, a memory order and a policy.
struct Cell<'a, T, const N: usize> {
    case: &'a Case<T, N>,
    order: Order,
    policy: PolicyKind,
}

impl<T: TestReal, const N: usize> fmt::Display for Cell<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kinds: Vec<String> = self.case.kinds.iter().map(R2rKind::to_string).collect();
        write!(
            f,
            "{} r2r {:?}, shape {:?}, axes {:?}, kinds [{}], {:?} (seed {:#x})",
            T::NAME,
            self.order,
            self.case.shape,
            self.case.axes,
            kinds.join(", "),
            self.policy,
            self.case.seed
        )
    }
}

/// Write `values`, in column-major order of the view, into the view by multi-index.
fn fill<T: Copy, A, const N: usize>(view: &mut Array<A, N>, values: &[T])
where
    A: BaseItem<Item = T> + UnsafeRandomAccessMut<N> + Shape<N>,
{
    for (index, &value) in multi_indices(view.shape()).zip(values) {
        *view.get_mut(index).unwrap() = value;
    }
}

/// The values of a real view, widened, by multi-index.
fn values_of<T, A, const N: usize>(view: &Array<A, N>) -> DynArray<f64, N>
where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    let mut values = DynArray::<f64, N>::from_shape(view.shape());
    for index in multi_indices(view.shape()) {
        values[index] = TestReal::to_f64(*view.get(index).unwrap());
    }
    values
}

/// `values`, in column-major order of `shape`, widened and multiplied by `factor`.
fn scaled<T: TestReal, const N: usize>(
    shape: [usize; N],
    values: &[T],
    factor: f64,
) -> DynArray<f64, N> {
    let mut result = DynArray::<f64, N>::from_shape(shape);
    for (index, &value) in multi_indices(shape).zip(values) {
        result[index] = factor * TestReal::to_f64(value);
    }
    result
}

/// The relative L2 error of `actual` against `expected`, or the absolute one for `expected`
/// of norm 0.
fn relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    let difference: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).powi(2))
        .sum();
    let norm: f64 = expected.iter().map(|e| e * e).sum();
    if norm == 0.0 {
        difference.sqrt()
    } else {
        (difference / norm).sqrt()
    }
}

/// Assert that `actual` equals `expected` to `tolerance` in the relative L2 norm of each
/// transform: the elements that share their indices off `axes`.
fn assert_close<const N: usize>(
    actual: &DynArray<f64, N>,
    expected: &DynArray<f64, N>,
    axes: &[usize],
    tolerance: f64,
    context: &dyn fmt::Display,
) {
    assert_eq!(actual.shape(), expected.shape(), "{context}");
    let mut transforms = std::collections::BTreeMap::<Vec<usize>, (Vec<_>, Vec<_>)>::new();
    for index in multi_indices(actual.shape()) {
        let batch: Vec<usize> = (0..N)
            .map(|axis| if axes.contains(&axis) { 0 } else { index[axis] })
            .collect();
        let (found, wanted) = transforms.entry(batch).or_default();
        found.push(actual[index]);
        wanted.push(expected[index]);
    }
    for (batch, (found, wanted)) in transforms {
        let error = relative_error(&found, &wanted);
        assert!(
            error <= tolerance,
            "{context}, transform at {batch:?}: relative error {error:e} above {tolerance:e}"
        );
    }
}

/// An out-of-place cell: the transform against the reference, the input's storage for a
/// preserving plan, the gaps of both views, and the round trip back into the input with the
/// inverse kinds, which is N times the input.
fn run_out_of_place<T, A, B, const N: usize>(
    cell: &Cell<T, N>,
    input: &mut Array<A, N>,
    output: &mut Array<B, N>,
) where
    T: TestReal,
    A: RealView<T, N>,
    B: RealView<T, N>,
{
    let case = cell.case;
    fill(input, &case.values);
    let before = storage_bits(input);
    let layout = Layout::over_axes(&*input, &*output, &case.axes).unwrap();
    assert_eq!(layout.storage(), (Storage::Real, Storage::Real), "{cell}");
    let planner =
        Planner::<T>::new().alignment(AlignmentClass::of(&*input), AlignmentClass::of(&*output));
    match cell.policy {
        PolicyKind::Preserving => match planner.r2r(&layout, &case.kinds) {
            Ok(plan) => {
                plan.execute(&*input, output).unwrap();
                assert!(
                    storage_bits(input) == before,
                    "{cell}: a preserving plan changed its input's storage"
                );
            }
            // FFTW may have no input-preserving algorithm with an HC2R kind (DC34).
            Err(FftwError::PlanCreationFailed { hint, .. }) => {
                assert!(case.kinds.contains(&R2rKind::Hc2r), "{cell}");
                assert!(hint.is_some(), "{cell}: no hint");
                return;
            }
            Err(error) => panic!("{cell}: {error}"),
        },
        PolicyKind::Destroying => {
            let plan = planner
                .clone()
                .may_destroy_input()
                .r2r(&layout, &case.kinds)
                .unwrap();
            plan.execute(input, output).unwrap();
        }
        PolicyKind::InPlace => unreachable!("an out-of-place policy"),
    }
    assert_gaps_unchanged(input, cell);
    assert_gaps_unchanged(output, cell);
    let n = case.logical_size();
    assert_close(
        &values_of(output),
        &case.expected,
        &case.axes,
        tolerance::<T>(n),
        cell,
    );

    // The inverse kinds, from the output back into the input, give N times the input.
    let layout = Layout::over_axes(&*output, &*input, &case.axes).unwrap();
    let plan = Planner::<T>::new()
        .alignment(AlignmentClass::of(&*output), AlignmentClass::of(&*input))
        .may_destroy_input()
        .r2r(&layout, &case.inverse_kinds())
        .unwrap();
    plan.execute(output, input).unwrap();
    assert_gaps_unchanged(input, cell);
    let context = format!("{cell}, round trip");
    assert_close(
        &values_of(input),
        &scaled(case.shape, &case.values, n as f64),
        &case.axes,
        2.0 * tolerance::<T>(n),
        &context,
    );
}

/// An in-place cell: the transform against the reference, the gaps, and the in-place round
/// trip with the inverse kinds.
fn run_in_place<T, A, const N: usize>(cell: &Cell<T, N>, data: &mut Array<A, N>)
where
    T: TestReal,
    A: RealView<T, N>,
{
    let case = cell.case;
    fill(data, &case.values);
    let layout = Layout::over_axes_inplace(&*data, &case.axes).unwrap();
    let class = AlignmentClass::of(&*data);
    let planner = Planner::<T>::new().alignment(class, class);
    let plan = planner.r2r_inplace(&layout, &case.kinds).unwrap();
    plan.execute(data).unwrap();
    assert_gaps_unchanged(data, cell);
    let n = case.logical_size();
    assert_close(
        &values_of(data),
        &case.expected,
        &case.axes,
        tolerance::<T>(n),
        cell,
    );
    let plan = planner.r2r_inplace(&layout, &case.inverse_kinds()).unwrap();
    plan.execute(data).unwrap();
    assert_gaps_unchanged(data, cell);
    let context = format!("{cell}, round trip");
    assert_close(
        &values_of(data),
        &scaled(case.shape, &case.values, n as f64),
        &case.axes,
        2.0 * tolerance::<T>(n),
        &context,
    );
}

fn run_cell<T, A, B, const N: usize>(
    cell: &Cell<T, N>,
    input: &mut Array<A, N>,
    output: &mut Array<B, N>,
) where
    T: TestReal,
    A: RealView<T, N>,
    B: RealView<T, N>,
{
    if cell.policy == PolicyKind::InPlace {
        run_in_place(cell, input);
    } else {
        run_out_of_place(cell, input, output);
    }
}

/// Build the views of `cell.order` and run the cell on them; subviews have parents two or
/// three elements larger on each axis and non-zero origins, filled with the sentinel.
macro_rules! run_views {
    ($t:ty, $n:literal, $cell:expr) => {{
        let cell: &Cell<$t, $n> = $cell;
        let shape = cell.case.shape;
        match cell.order {
            Order::ColumnMajor => {
                let mut input = real_sentinel_parent::<$t, $n>(shape);
                let mut output = real_sentinel_parent::<$t, $n>(shape);
                run_cell(cell, &mut input, &mut output);
            }
            Order::RowMajor => {
                let mut input = StridedDynArray::<$t, $n>::row_major(shape);
                let mut output = StridedDynArray::<$t, $n>::row_major(shape);
                run_cell(cell, &mut input, &mut output);
            }
            Order::Subview => {
                let mut input = real_sentinel_parent::<$t, $n>(shape.map(|n| n + 3));
                let mut output = real_sentinel_parent::<$t, $n>(shape.map(|n| n + 2));
                run_cell(
                    cell,
                    &mut input.r_mut().into_subview([2; $n], shape),
                    &mut output.r_mut().into_subview([1; $n], shape),
                );
            }
        }
    }};
}

fn run_rank1<T: TestReal>(cell: &Cell<T, 1>) {
    run_views!(T, 1, cell)
}

fn run_rank2<T: TestReal>(cell: &Cell<T, 2>) {
    run_views!(T, 2, cell)
}

fn run_rank3<T: TestReal>(cell: &Cell<T, 3>) {
    run_views!(T, 3, cell)
}

/// The seed of a case.
fn case_seed(shape: &[usize], axes: &[usize], kinds: &[R2rKind]) -> u64 {
    let kinds = kinds
        .iter()
        .map(|kind| R2rKind::ALL.iter().position(|k| k == kind).unwrap());
    shape
        .iter()
        .copied()
        .chain([usize::MAX])
        .chain(axes.iter().copied())
        .chain(kinds)
        .fold(SEED, |code, value| {
            code.wrapping_mul(31).wrapping_add(value as u64)
        })
}

/// Every kind on `[n]` with n in 1, 2, 3, 17 and 97 (F5 brief, work item 7), the three
/// policies, column-major; `Redft00` with n = 1 is an `InvalidKind` before FFTW is called.
fn check_every_kind<T: TestReal>() {
    for kind in R2rKind::ALL {
        for n in [1, 2, 3, 17, 97] {
            if kind == R2rKind::Redft00 && n == 1 {
                check_redft00_of_one::<T>();
                continue;
            }
            let shape = [n];
            let case = Case::<T, 1>::new(shape, &[0], &[kind], case_seed(&shape, &[0], &[kind]));
            for policy in PolicyKind::ALL {
                run_rank1(&Cell {
                    case: &case,
                    order: Order::ColumnMajor,
                    policy,
                });
            }
        }
    }
}

/// `Redft00` on a dimension of extent 1, on scratch and on the caller's buffers, out of place
/// and in place: `InvalidKind`, with the buffers unchanged.
fn check_redft00_of_one<T: TestReal>() {
    let mut input = real_sentinel_parent::<T, 1>([1]);
    let mut output = real_sentinel_parent::<T, 1>([1]);
    let before = storage_bits(&output);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let in_place = Layout::along_axis_inplace(&input, 0).unwrap();
    let kinds = [R2rKind::Redft00];
    let planner = Planner::<T>::new();
    for error in [
        planner.r2r(&layout, &kinds).unwrap_err(),
        planner
            .r2r_on(&mut input, &mut output, &layout, &kinds)
            .unwrap_err(),
        planner.r2r_inplace(&in_place, &kinds).unwrap_err(),
        planner
            .r2r_inplace_on(&mut output, &in_place, &kinds)
            .unwrap_err(),
    ] {
        assert!(
            matches!(&error, FftwError::InvalidKind { reason } if reason.contains("REDFT00")),
            "{}: {error}",
            T::NAME
        );
    }
    assert!(storage_bits(&output) == before);
}

#[test]
fn test_every_kind_f64() {
    check_every_kind::<f64>();
}

#[test]
fn test_every_kind_f32() {
    check_every_kind::<f32>();
}

/// The mixed kinds of the fixed cells: over all axes of `[6, 5, 4]` in two orders, and over
/// two axes.
fn mixed_kinds() -> [(Vec<usize>, Vec<R2rKind>); 3] {
    [
        (
            vec![0, 1, 2],
            vec![R2rKind::Redft10, R2rKind::Rodft00, R2rKind::Dht],
        ),
        (
            vec![2, 0, 1],
            vec![R2rKind::R2hc, R2rKind::Redft11, R2rKind::Rodft01],
        ),
        (vec![1, 2], vec![R2rKind::Hc2r, R2rKind::Redft00]),
    ]
}

fn check_mixed_kinds<T: TestReal>() {
    let shape = [6, 5, 4];
    for (axes, kinds) in mixed_kinds() {
        let case = Case::<T, 3>::new(shape, &axes, &kinds, case_seed(&shape, &axes, &kinds));
        for order in Order::ALL {
            for policy in PolicyKind::ALL {
                run_rank3(&Cell {
                    case: &case,
                    order,
                    policy,
                });
            }
        }
    }
}

#[test]
fn test_mixed_kinds() {
    check_mixed_kinds::<f64>();
    check_mixed_kinds::<f32>();
}

#[test]
fn test_kinds_apply_to_their_own_axes() {
    // The regression of design Section 2.2 item 9: the 0.10 interface reversed the shape of
    // column-major arrays but not the kinds, and applied kind 0 to the last axis. Different
    // kinds on the two axes of column-major and row-major arrays apply each kind to its own
    // axis, and the two memory orders give the same values.
    let shape = [6, 5];
    let (axes, kinds) = ([0, 1], [R2rKind::Redft10, R2rKind::Rodft10]);
    let values = random_reals::<f64>(SEED, 30);
    let expected = reference_r2r(shape, &values, &axes, &kinds);
    let swapped = reference_r2r(shape, &values, &axes, &[kinds[1], kinds[0]]);
    let n = kinds[0].logical_size(6) * kinds[1].logical_size(5);

    let mut column = real_sentinel_parent::<f64, 2>(shape);
    let mut column_out = real_sentinel_parent::<f64, 2>(shape);
    fill(&mut column, &values);
    let layout = Layout::over_axes(&column, &column_out, &axes).unwrap();
    Planner::<f64>::new()
        .r2r(&layout, &kinds)
        .unwrap()
        .execute(&column, &mut column_out)
        .unwrap();

    let mut row = StridedDynArray::<f64, 2>::row_major(shape);
    let mut row_out = StridedDynArray::<f64, 2>::row_major(shape);
    fill(&mut row, &values);
    let layout = Layout::over_axes(&row, &row_out, &axes).unwrap();
    Planner::<f64>::new()
        .alignment(AlignmentClass::of(&row), AlignmentClass::of(&row_out))
        .r2r(&layout, &kinds)
        .unwrap()
        .execute(&row, &mut row_out)
        .unwrap();

    for (name, output) in [
        ("column-major", values_of(&column_out)),
        ("row-major", values_of(&row_out)),
    ] {
        assert_close(&output, &expected, &axes, tolerance::<f64>(n), &name);
        // The kinds applied to the other axes would give other values.
        let error = relative_error(
            &multi_indices(shape).map(|i| output[i]).collect::<Vec<_>>(),
            &multi_indices(shape).map(|i| swapped[i]).collect::<Vec<_>>(),
        );
        assert!(
            error > 0.1,
            "{name}: the swapped kinds give the same values"
        );
    }
}

#[test]
fn test_r2r_sampled_cells() {
    // 12 cells from the product of kinds per axis, memory orders, shapes with extents up to
    // 12, axes, policies and precisions, drawn with the fixed seed.
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xf5b);
    for _ in 0..12 {
        let order = Order::ALL[rng.random_range(0..Order::ALL.len())];
        let policy = PolicyKind::ALL[rng.random_range(0..3)];
        let rank = rng.random_range(1..=3_usize);
        let mut axes: Vec<usize> = (0..rank).filter(|_| rng.random_bool(0.5)).collect();
        if axes.is_empty() {
            axes.push(rng.random_range(0..rank));
        }
        if axes.len() > 1 && rng.random_bool(0.5) {
            axes.reverse();
        }
        let single = rng.random_bool(0.5);
        let seed = rng.random();
        let mut extents = [0_usize; 3];
        for extent in &mut extents {
            *extent = rng.random_range(1..=12_usize);
        }
        // Any kind, except `Redft00` on an axis of extent 1.
        let kinds: Vec<R2rKind> = axes
            .iter()
            .map(|&axis| {
                loop {
                    let kind = R2rKind::ALL[rng.random_range(0..R2rKind::ALL.len())];
                    if kind != R2rKind::Redft00 || extents[axis] > 1 {
                        break kind;
                    }
                }
            })
            .collect();
        macro_rules! sampled {
            ($run:ident, $n:literal) => {{
                let shape: [usize; $n] = std::array::from_fn(|axis| extents[axis]);
                if single {
                    let case = Case::<f32, $n>::new(shape, &axes, &kinds, seed);
                    $run(&Cell {
                        case: &case,
                        order,
                        policy,
                    });
                } else {
                    let case = Case::<f64, $n>::new(shape, &axes, &kinds, seed);
                    $run(&Cell {
                        case: &case,
                        order,
                        policy,
                    });
                }
            }};
        }
        match rank {
            1 => sampled!(run_rank1, 1),
            2 => sampled!(run_rank2, 2),
            _ => sampled!(run_rank3, 3),
        }
    }
}

// Identities (design Section 7.3).

fn check_multidimensional_against_successive<T: TestReal>() {
    // A two-dimensional r2r plan against one-dimensional plans along each axis in turn, to
    // round-off: different plans, the same transform.
    let shape = [12, 7];
    let kinds = [R2rKind::Redft10, R2rKind::Rodft11];
    let values = random_reals::<T>(SEED + 5, 84);
    let mut input = real_sentinel_parent::<T, 2>(shape);
    let mut both = real_sentinel_parent::<T, 2>(shape);
    let mut first = real_sentinel_parent::<T, 2>(shape);
    let mut second = real_sentinel_parent::<T, 2>(shape);
    fill(&mut input, &values);
    Planner::<T>::new()
        .r2r(&Layout::over_axes(&input, &both, &[0, 1]).unwrap(), &kinds)
        .unwrap()
        .execute(&input, &mut both)
        .unwrap();
    Planner::<T>::new()
        .r2r(&Layout::along_axis(&input, &first, 0).unwrap(), &kinds[..1])
        .unwrap()
        .execute(&input, &mut first)
        .unwrap();
    Planner::<T>::new()
        .r2r(
            &Layout::along_axis(&first, &second, 1).unwrap(),
            &kinds[1..],
        )
        .unwrap()
        .execute(&first, &mut second)
        .unwrap();
    let n = kinds[0].logical_size(12) * kinds[1].logical_size(7);
    let context = format!("{} multidimensional r2r against successive ones", T::NAME);
    assert_close(
        &values_of(&both),
        &values_of(&second),
        &[0, 1],
        2.0 * tolerance::<T>(n),
        &context,
    );
}

#[test]
fn test_multidimensional_against_successive() {
    check_multidimensional_against_successive::<f64>();
    check_multidimensional_against_successive::<f32>();
}

// Defaults and policies (design Section 7.4).

#[test]
fn test_defaults_and_policies() {
    let input = real_sentinel_parent::<f64, 2>([8, 6]);
    let output = real_sentinel_parent::<f64, 2>([8, 6]);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let kinds = [R2rKind::Redft10];
    let _: Plan<f64, R2r, Preserving> = Planner::new().r2r(&layout, &kinds).unwrap();
    let _: Plan<f64, R2r, Preserving> = Planner::new()
        .preserve_input()
        .r2r(&layout, &kinds)
        .unwrap();
    let _: Plan<f64, R2r, Destroying> = Planner::new()
        .may_destroy_input()
        .r2r(&layout, &kinds)
        .unwrap();
    let in_place = Layout::along_axis_inplace(&input, 0).unwrap();
    let _: Plan<f64, R2r, InPlace> = Planner::new().r2r_inplace(&in_place, &kinds).unwrap();

    // An r2r plan with an HC2R kind is `Preserving`: it leaves its input unchanged, bit for
    // bit, or, if FFTW has no input-preserving algorithm, planning returns
    // `PlanCreationFailed` with the hint to `may_destroy_input()`.
    for (axes, kinds) in [
        (&[0][..], &[R2rKind::Hc2r][..]),
        (&[0, 1], &[R2rKind::Hc2r, R2rKind::Hc2r]),
        (&[1, 0], &[R2rKind::Redft10, R2rKind::Hc2r]),
    ] {
        let mut input = real_sentinel_parent::<f64, 2>([8, 6]);
        let mut output = real_sentinel_parent::<f64, 2>([8, 6]);
        fill(&mut input, &random_reals::<f64>(SEED + 6, 48));
        let layout = Layout::over_axes(&input, &output, axes).unwrap();
        match Planner::<f64>::new().r2r(&layout, kinds) {
            Ok(plan) => {
                let before = storage_bits(&input);
                plan.execute(&input, &mut output).unwrap();
                assert!(storage_bits(&input) == before, "HC2R over {axes:?}");
            }
            Err(FftwError::PlanCreationFailed { hint, .. }) => {
                let hint = hint.expect("a preserving HC2R plan names the remedy");
                assert!(hint.contains("may_destroy_input"), "{hint}");
            }
            Err(error) => panic!("HC2R over {axes:?}: {error}"),
        }
        // The destroying plan exists.
        Planner::<f64>::new()
            .may_destroy_input()
            .r2r(&layout, kinds)
            .unwrap();
    }
}

#[test]
fn test_plan_creation_failed_shows_its_hint() {
    let error = FftwError::PlanCreationFailed {
        rigor: crate::fftw::Rigor::Estimate,
        flags: 0,
        layout: "a layout".to_string(),
        hint: Some("try this"),
    };
    assert!(
        error
            .to_string()
            .ends_with("FFTW gives no reason; try this"),
        "{error}"
    );
    let error = FftwError::PlanCreationFailed {
        rigor: crate::fftw::Rigor::Estimate,
        flags: 0,
        layout: "a layout".to_string(),
        hint: None,
    };
    assert!(
        error.to_string().ends_with("FFTW gives no reason"),
        "{error}"
    );
}

#[test]
fn test_plan_properties() {
    let input = DynArray::<f64, 2>::from_shape([5, 4]);
    let layout = Layout::over_axes_inplace(&input, &[1, 0]).unwrap();
    let kinds = [R2rKind::Rodft00, R2rKind::Redft00];
    let plan = Planner::<f64>::new().r2r_inplace(&layout, &kinds).unwrap();
    assert_eq!(plan.kinds(), &kinds);
    // Rodft00 along axis 1 (n = 4): 2 (n + 1) = 10; Redft00 along axis 0 (n = 5): 2 (n − 1)
    // = 8. The layout's product of extents is not the r2r logical size.
    assert_eq!(plan.logical_size(), 80);
    assert_eq!(plan.layout().logical_size(), 20);
    assert_eq!(plan.scale_factor(Scale::Inverse), 1.0 / 80.0);
    let debug = format!("{plan:?}");
    assert!(
        debug.contains("R2r") && debug.contains("Rodft00"),
        "{debug}"
    );
    assert!(!plan.description().is_empty());
    for kind in R2rKind::ALL {
        assert_eq!(kind.logical_size(4), inverse(kind).logical_size(4));
    }
}

// Negative tests (design Section 7.4). Each returns its error and leaves the output's parent
// holding the sentinel.

#[test]
fn test_wrong_number_of_kinds() {
    let mut input = real_sentinel_parent::<f64, 2>([8, 3]);
    let mut output = real_sentinel_parent::<f64, 2>([8, 3]);
    let before = storage_bits(&output);
    let layout = Layout::over_axes(&input, &output, &[0, 1]).unwrap();
    let in_place = Layout::over_axes_inplace(&input, &[0, 1]).unwrap();
    let planner = Planner::<f64>::new();
    for kinds in [&[][..], &[R2rKind::Dht], &[R2rKind::Dht; 3]] {
        for error in [
            planner.r2r(&layout, kinds).unwrap_err(),
            planner
                .r2r_on(&mut input, &mut output, &layout, kinds)
                .unwrap_err(),
            planner.r2r_inplace(&in_place, kinds).unwrap_err(),
            planner
                .r2r_inplace_on(&mut output, &in_place, kinds)
                .unwrap_err(),
        ] {
            assert!(
                matches!(&error, FftwError::InvalidKind { reason } if reason.contains("2 transform dimensions")),
                "{error}"
            );
        }
    }
    assert!(storage_bits(&output) == before);
}

#[test]
fn test_layouts_of_other_families() {
    // The r2r methods reject layouts of every other family before FFTW is called, and the
    // other methods reject r2r layouts (see also the F5a tests).
    let dims = [Dim { n: 8, is: 1, os: 1 }];
    let make = |storage, placement| Layout::with_storage(&dims, &[], storage, placement).unwrap();
    let out = Placement::OutOfPlace;
    let planner = Planner::<f64>::new();
    let kinds = [R2rKind::Dht];
    let named = |error: FftwError| match error {
        FftwError::InvalidLayout { dimension, reason } => (dimension, reason),
        other => panic!("expected InvalidLayout, found {other}"),
    };
    for (storage, dimension) in [
        ((Storage::Complex, Storage::Complex), "layout"),
        ((Storage::Real, Storage::Complex), "layout"),
        ((Storage::Complex, Storage::Real), "layout"),
        (
            (Storage::Split { gap: 8 }, Storage::Split { gap: 8 }),
            "part axis",
        ),
        ((Storage::Real, Storage::Split { gap: 5 }), "part axis"),
    ] {
        let (found, reason) = named(planner.r2r(&make(storage, out), &kinds).unwrap_err());
        assert_eq!(found, dimension, "{reason}");
        assert!(reason.contains("real-to-real plan needs"), "{reason}");
    }
    let r2r = make((Storage::Real, Storage::Real), out);
    let (found, reason) = named(planner.c2c(&r2r, Direction::Forward).unwrap_err());
    assert_eq!(found, "layout");
    assert!(reason.contains("`r2r` planner methods"), "{reason}");
    let (found, _) = named(
        planner
            .r2r_inplace(
                &make((Storage::Complex, Storage::Complex), Placement::InPlace),
                &kinds,
            )
            .unwrap_err(),
    );
    assert_eq!(found, "layout");

    let mut reals = filled(real_sentinel::<f64>(), 16);
    let mut other = reals.clone();
    let sentinel = real_sentinel::<f64>().bits();
    let (found, _) = named(
        planner
            .r2r_on(
                &mut reals[..],
                &mut other[..],
                &make((Storage::Real, Storage::Complex), out),
                &kinds,
            )
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    assert!(other.iter().all(|value| value.bits() == sentinel));
    // A layout of the wrong placement.
    assert!(matches!(
        planner.r2r(
            &make((Storage::Real, Storage::Real), Placement::InPlace),
            &kinds
        ),
        Err(FftwError::PlacementMismatch { .. })
    ));
}

#[test]
fn test_r2r_alignment_and_mismatch() {
    let layout = Layout::with_storage(
        &[Dim {
            n: 16,
            is: 1,
            os: 1,
        }],
        &[],
        (Storage::Real, Storage::Real),
        Placement::OutOfPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new()
        .r2r(&layout, &[R2rKind::Redft11])
        .unwrap();
    let input = aligned(&random_reals::<f64>(SEED, 17));
    let mut output = filled(real_sentinel::<f64>(), 17);
    let sentinel = real_sentinel::<f64>().bits();
    let error = plan.execute(&input[1..], &mut output[..16]).unwrap_err();
    assert_eq!(
        error,
        FftwError::AlignmentMismatch {
            side: Side::Input,
            expected: AlignmentClass::default(),
            found: AlignmentClass::from_offset(8),
        }
    );
    let error = plan.execute(&input[..16], &mut output[..15]).unwrap_err();
    assert!(
        matches!(
            error,
            FftwError::LayoutMismatch {
                side: Side::Output,
                ..
            }
        ),
        "{error}"
    );
    assert!(output.iter().all(|value| value.bits() == sentinel));

    // An array of another shape.
    let mut array = real_sentinel_parent::<f64, 1>([15]);
    let error = plan.execute(&input[..16], &mut array).unwrap_err();
    assert!(matches!(error, FftwError::LayoutMismatch { .. }), "{error}");
}

#[test]
fn test_r2r_scratch_cap() {
    // A column of a row-major [8, 1000]: 8 reals 1000 apart span 7001 reals.
    let mut input = StridedDynArray::<f64, 2>::row_major([8, 1000]);
    let mut output = StridedDynArray::<f64, 2>::row_major([8, 1000]);
    let values = random_reals::<f64>(SEED, 8);
    fill(&mut input.r_mut().col(3), &values);
    let layout = Layout::along_axis(&input.r().col(3), &output.r().col(3), 0).unwrap();
    let kinds = [R2rKind::Rodft10];
    let planner = Planner::<f64>::new().max_scratch(1024);
    assert_eq!(
        planner.r2r(&layout, &kinds).unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: 2 * 7001 * 8,
            bound: 1024,
        }
    );
    let plan = planner
        .r2r_on(
            &mut input.r_mut().col(3),
            &mut output.r_mut().col(3),
            &layout,
            &kinds,
        )
        .unwrap();
    plan.execute(&input.r().col(3), &mut output.r_mut().col(3))
        .unwrap();
    assert_close(
        &values_of(&output.r().col(3)),
        &reference_r2r([8], &values, &[0], &kinds),
        &[0],
        tolerance::<f64>(16),
        &"scratch cap",
    );
    assert_eq!(output[[0, 2]], 0.0);
}
