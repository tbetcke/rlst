//! Real-to-complex and complex-to-real transforms (task F5a; design Sections 7.1 to 7.4).
//!
//! Every plan here is made with `Rigor::Estimate`, depends on no wisdom and assumes no SIMD.
//! Results are compared with the references of `test_support`, which read values by
//! multi-index and never build a `Layout`, and every view's parent outside the view is
//! checked to hold its sentinel afterwards. The inputs and references of a shape and its axes
//! are computed once and shared by the cells of every memory order and policy.

use std::fmt;

use num::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::dense::array::StridedSliceArrayMut;
use crate::fftw::test_support::{
    PolicyKind, SEED, TestElement, TestReal, aligned, assert_gaps_unchanged, filled, halved_shape,
    in_place_real_offsets, multi_indices, random_reals, real_sentinel, real_sentinel_parent,
    reference_c2r, reference_r2c, relative_error, sentinel, sentinel_parent, storage_bits, widen,
};
use crate::fftw::{
    AlignmentClass, C2c, C2r, Destroying, Dim, Direction, FftwError, InPlace, Layout, Placement,
    Plan, Planner, Preserving, R2c, Scale, Side, SplitC2c, Storage, as_reals, as_reals_mut,
};
use crate::{
    Array, BaseItem, DynArray, Shape, Stride, StridedDynArray, StridedRawAccessMut,
    UnsafeRandomAccessByRef, UnsafeRandomAccessMut, c64,
};

/// The constant c of the r2c tolerance c · ε · (1 + log₂ N), fixed for the family (design
/// Section 7.1).
const R2C_TOLERANCE_FACTOR: f64 = 4.0;

/// The constant c of the c2r tolerance, fixed for the family.
const C2R_TOLERANCE_FACTOR: f64 = 4.0;

/// The tolerance c · ε · (1 + log₂ N) for the logical size `n`.
fn tolerance<T: TestReal>(factor: f64, n: usize) -> f64 {
    factor * T::EPSILON * (1.0 + (n as f64).log2())
}

/// The tolerance of a round trip, an r2c and a c2r transform in turn.
fn round_trip_tolerance<T: TestReal>(n: usize) -> f64 {
    tolerance::<T>(R2C_TOLERANCE_FACTOR + C2R_TOLERANCE_FACTOR, n)
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

/// The complex arrays and views the tests execute on.
trait ComplexView<T: TestReal, const N: usize>:
    BaseItem<Item = Complex<T>>
    + StridedRawAccessMut
    + Shape<N>
    + Stride<N>
    + UnsafeRandomAccessByRef<N>
    + UnsafeRandomAccessMut<N>
{
}

impl<T: TestReal, A, const N: usize> ComplexView<T, N> for A where
    A: BaseItem<Item = Complex<T>>
        + StridedRawAccessMut
        + Shape<N>
        + Stride<N>
        + UnsafeRandomAccessByRef<N>
        + UnsafeRandomAccessMut<N>
{
}

/// The memory orders and views of the fixed cells (F5 brief, work item 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// Owned column-major arrays at `FFTW_ALIGNED`.
    ColumnMajor,
    /// Owned row-major arrays (`StridedDynArray::row_major`), `Vec`-backed.
    RowMajor,
    /// Subviews of larger parents, at non-zero origins.
    Subview,
    /// Views with steps of 2 over aligned storage.
    Strided,
}

impl Order {
    const ALL: [Order; 4] = [
        Order::ColumnMajor,
        Order::RowMajor,
        Order::Subview,
        Order::Strided,
    ];
}

/// The two families under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    R2c,
    C2r,
}

/// The inputs and references of one real shape and its axes in one precision.
struct Case<T, const N: usize> {
    /// The real shape; the complex shape is halved along the last entry of `axes`.
    shape: [usize; N],
    axes: Vec<usize>,
    seed: u64,
    /// The input of r2c, in column-major order of `shape`.
    reals: Vec<T>,
    /// The r2c reference of `reals`.
    spectrum: DynArray<Complex<f64>, N>,
    /// The input of c2r: `spectrum` rounded to `T`, Hermitian by construction, in
    /// column-major order of the complex shape.
    half: Vec<Complex<T>>,
    /// The c2r reference of `half`.
    inverse: DynArray<f64, N>,
}

impl<T: TestReal, const N: usize> Case<T, N> {
    fn new(shape: [usize; N], axes: &[usize], seed: u64) -> Self {
        let reals = random_reals::<T>(seed, shape.iter().product());
        let spectrum = reference_r2c(shape, &reals, axes);
        let complex_shape = halved_shape(shape, *axes.last().unwrap());
        let half: Vec<Complex<T>> = multi_indices(complex_shape)
            .map(|index| {
                let value = spectrum[index];
                Complex::new(T::round_from(value.re), T::round_from(value.im))
            })
            .collect();
        let inverse = reference_c2r(shape, &array_of(complex_shape, &widen(&half)), axes);
        Self {
            shape,
            axes: axes.to_vec(),
            seed,
            reals,
            spectrum,
            half,
            inverse,
        }
    }

    fn halved(&self) -> usize {
        *self.axes.last().unwrap()
    }

    fn complex_shape(&self) -> [usize; N] {
        halved_shape(self.shape, self.halved())
    }

    fn logical_size(&self) -> usize {
        self.axes.iter().map(|&axis| self.shape[axis]).product()
    }
}

/// One cell: a case, a family, a memory order and a policy.
struct Cell<'a, T, const N: usize> {
    case: &'a Case<T, N>,
    family: Family,
    order: Order,
    policy: PolicyKind,
}

impl<T: TestReal, const N: usize> fmt::Display for Cell<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:?} {:?}, real shape {:?}, axes {:?}, {:?} (seed {:#x})",
            T::NAME,
            self.family,
            self.order,
            self.case.shape,
            self.case.axes,
            self.policy,
            self.case.seed
        )
    }
}

/// `values`, in column-major order of `shape`, as an array.
fn array_of<E: Copy + Default, const N: usize>(shape: [usize; N], values: &[E]) -> DynArray<E, N> {
    let mut array = DynArray::<E, N>::from_shape(shape);
    for (index, &value) in multi_indices(shape).zip(values) {
        array[index] = value;
    }
    array
}

/// Write `values`, in column-major order of the view, into the view by multi-index.
fn fill<E: Copy, A, const N: usize>(view: &mut Array<A, N>, values: &[E])
where
    A: BaseItem<Item = E> + UnsafeRandomAccessMut<N> + Shape<N>,
{
    for (index, &value) in multi_indices(view.shape()).zip(values) {
        *view.get_mut(index).unwrap() = value;
    }
}

/// The values of a complex view, widened, by multi-index.
fn complex_values<T, A, const N: usize>(view: &Array<A, N>) -> DynArray<Complex<f64>, N>
where
    T: TestReal,
    A: BaseItem<Item = Complex<T>> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    let values: Vec<Complex<T>> = multi_indices(view.shape())
        .map(|index| *view.get(index).unwrap())
        .collect();
    array_of(view.shape(), &widen(&values))
}

/// The values of a real view, widened to complex values with imaginary part 0.
fn real_values<T, A, const N: usize>(view: &Array<A, N>) -> DynArray<Complex<f64>, N>
where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    let mut values = DynArray::<Complex<f64>, N>::from_shape(view.shape());
    for index in multi_indices(view.shape()) {
        values[index] = Complex::new(TestReal::to_f64(*view.get(index).unwrap()), 0.0);
    }
    values
}

/// Reals as complex values with imaginary part 0, multiplied by `factor`.
fn as_complex<const N: usize>(reals: &DynArray<f64, N>, factor: f64) -> DynArray<Complex<f64>, N> {
    let mut values = DynArray::<Complex<f64>, N>::from_shape(reals.shape());
    for index in multi_indices(reals.shape()) {
        values[index] = Complex::new(factor * reals[index], 0.0);
    }
    values
}

/// `values` of `T`, in column-major order of `shape`, widened and multiplied by `factor`.
fn scaled_reals<T: TestReal, const N: usize>(
    shape: [usize; N],
    values: &[T],
    factor: f64,
) -> DynArray<Complex<f64>, N> {
    let mut result = DynArray::<Complex<f64>, N>::from_shape(shape);
    for (index, value) in multi_indices(shape).zip(values) {
        result[index] = Complex::new(factor * TestReal::to_f64(*value), 0.0);
    }
    result
}

/// Assert that `actual` equals `expected` to `tolerance` in the relative L2 norm of each
/// transform: the elements that share their indices off `axes`.
fn assert_close<const N: usize>(
    actual: &DynArray<Complex<f64>, N>,
    expected: &DynArray<Complex<f64>, N>,
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

/// The reals of the in-place real side at `offsets` of the complex view's storage, as complex
/// values with imaginary part 0, by multi-index of `shape`.
fn in_place_reals<T, A, const N: usize>(
    complex: &Array<A, N>,
    offsets: &[usize],
    shape: [usize; N],
) -> DynArray<Complex<f64>, N>
where
    T: TestReal,
    A: BaseItem<Item = Complex<T>> + StridedRawAccessMut,
{
    let (storage, _) = complex.storage().unwrap();
    let reals = as_reals(storage);
    let values: Vec<Complex<f64>> = offsets
        .iter()
        .map(|&offset| Complex::new(TestReal::to_f64(reals[offset]), 0.0))
        .collect();
    array_of(shape, &values)
}

/// An out-of-place r2c cell: the transform against the reference, the input's storage for a
/// preserving plan, the gaps of both views, and the round trip back into the input with the
/// default c2r plan.
fn run_r2c<T, R, C, const N: usize>(
    cell: &Cell<T, N>,
    real: &mut Array<R, N>,
    complex: &mut Array<C, N>,
) where
    T: TestReal,
    R: RealView<T, N>,
    C: ComplexView<T, N>,
{
    let case = cell.case;
    fill(real, &case.reals);
    let before = storage_bits(real);
    let layout = Layout::over_axes(&*real, &*complex, &case.axes).unwrap();
    assert_eq!(
        layout.storage(),
        (Storage::Real, Storage::Complex),
        "{cell}"
    );
    // Plan for the classes of these arrays: row-major arrays are `Vec`-backed, and views at
    // an offset change class.
    let planner =
        Planner::<T>::new().alignment(AlignmentClass::of(&*real), AlignmentClass::of(&*complex));
    match cell.policy {
        PolicyKind::Preserving => {
            let plan = planner.r2c(&layout).unwrap();
            plan.execute(&*real, complex).unwrap();
            assert!(
                storage_bits(real) == before,
                "{cell}: a preserving plan changed its input's storage"
            );
        }
        PolicyKind::Destroying => {
            let plan = planner.clone().may_destroy_input().r2c(&layout).unwrap();
            plan.execute(real, complex).unwrap();
        }
        PolicyKind::InPlace => unreachable!("an out-of-place policy"),
    }
    assert_gaps_unchanged(real, cell);
    assert_gaps_unchanged(complex, cell);
    let n = case.logical_size();
    assert_close(
        &complex_values(complex),
        &case.spectrum,
        &case.axes,
        tolerance::<T>(R2C_TOLERANCE_FACTOR, n),
        cell,
    );

    // The default c2r plan of the output, into the input, is N times the input.
    let layout = Layout::over_axes(&*complex, &*real, &case.axes).unwrap();
    let plan = Planner::<T>::new()
        .alignment(AlignmentClass::of(&*complex), AlignmentClass::of(&*real))
        .c2r(&layout)
        .unwrap();
    plan.execute(complex, real).unwrap();
    assert_gaps_unchanged(real, cell);
    assert_gaps_unchanged(complex, cell);
    let context = format!("{cell}, round trip");
    assert_close(
        &real_values(real),
        &scaled_reals(case.shape, &case.reals, n as f64),
        &case.axes,
        round_trip_tolerance::<T>(n),
        &context,
    );
}

/// An out-of-place c2r cell: the transform of the Hermitian input against the reference, the
/// input's storage for a preserving plan, and the gaps of both views. A preserving plan over
/// more than one transform dimension is `PreserveInputUnsupported`.
fn run_c2r<T, R, C, const N: usize>(
    cell: &Cell<T, N>,
    real: &mut Array<R, N>,
    complex: &mut Array<C, N>,
) where
    T: TestReal,
    R: RealView<T, N>,
    C: ComplexView<T, N>,
{
    let case = cell.case;
    fill(complex, &case.half);
    let before = storage_bits(complex);
    let real_before = storage_bits(real);
    let layout = Layout::over_axes(&*complex, &*real, &case.axes).unwrap();
    assert_eq!(
        layout.storage(),
        (Storage::Complex, Storage::Real),
        "{cell}"
    );
    let planner =
        Planner::<T>::new().alignment(AlignmentClass::of(&*complex), AlignmentClass::of(&*real));
    match cell.policy {
        PolicyKind::Preserving => match planner.clone().preserve_input().c2r(&layout) {
            Ok(plan) => {
                assert_eq!(case.axes.len(), 1, "{cell}");
                plan.execute(&*complex, real).unwrap();
                assert!(
                    storage_bits(complex) == before,
                    "{cell}: a preserving plan changed its input's storage"
                );
            }
            Err(error) => {
                assert_eq!(
                    error,
                    FftwError::PreserveInputUnsupported {
                        rank: case.axes.len()
                    },
                    "{cell}"
                );
                assert!(case.axes.len() > 1, "{cell}");
                assert!(storage_bits(real) == real_before, "{cell}");
                return;
            }
        },
        PolicyKind::Destroying => {
            // The default c2r plan destroys its input.
            let plan: Plan<T, C2r, Destroying> = planner.c2r(&layout).unwrap();
            plan.execute(complex, real).unwrap();
        }
        PolicyKind::InPlace => unreachable!("an out-of-place policy"),
    }
    assert_gaps_unchanged(real, cell);
    assert_gaps_unchanged(complex, cell);
    assert_close(
        &real_values(real),
        &as_complex(&case.inverse, 1.0),
        &case.axes,
        tolerance::<T>(C2R_TOLERANCE_FACTOR, case.logical_size()),
        cell,
    );
}

/// Whether the halved axis of `complex` is contiguous, as an in-place layout needs. If not,
/// assert that `r2c_inplace` and `c2r_inplace` reject the view naming that axis.
fn contiguous_halved_axis<T, C, const N: usize>(cell: &Cell<T, N>, complex: &Array<C, N>) -> bool
where
    T: TestReal,
    C: ComplexView<T, N>,
{
    let case = cell.case;
    let halved = case.halved();
    if complex.shape()[halved] == 1 || complex.stride()[halved] == 1 {
        return true;
    }
    let n_last = case.shape[halved];
    for result in [
        Layout::r2c_inplace(complex, &case.axes, n_last),
        Layout::c2r_inplace(complex, &case.axes, n_last),
    ] {
        assert!(
            matches!(&result, Err(FftwError::InvalidLayout { dimension, .. }) if *dimension == format!("axis {halved}")),
            "{cell}: expected an InvalidLayout naming the halved axis, found {result:?}"
        );
    }
    false
}

/// An in-place r2c cell on the complex view: the reals written into its storage, the
/// transform against the reference, the gaps, and the in-place round trip.
fn run_r2c_in_place<T, C, const N: usize>(cell: &Cell<T, N>, complex: &mut Array<C, N>)
where
    T: TestReal,
    C: ComplexView<T, N>,
{
    if !contiguous_halved_axis(cell, complex) {
        return;
    }
    let case = cell.case;
    let n_last = case.shape[case.halved()];
    let layout = Layout::r2c_inplace(&*complex, &case.axes, n_last).unwrap();
    let offsets = in_place_real_offsets(&*complex, case.halved(), case.shape);
    {
        let (storage, _) = complex.storage_mut().unwrap();
        let reals = as_reals_mut(storage);
        for (&offset, &value) in offsets.iter().zip(&case.reals) {
            reals[offset] = value;
        }
    }
    let class = AlignmentClass::of(&*complex);
    let planner = Planner::<T>::new().alignment(class, class);
    planner
        .r2c_inplace(&layout)
        .unwrap()
        .execute(complex)
        .unwrap();
    assert_gaps_unchanged(complex, cell);
    let n = case.logical_size();
    assert_close(
        &complex_values(complex),
        &case.spectrum,
        &case.axes,
        tolerance::<T>(R2C_TOLERANCE_FACTOR, n),
        cell,
    );

    let layout = Layout::c2r_inplace(&*complex, &case.axes, n_last).unwrap();
    planner
        .c2r_inplace(&layout)
        .unwrap()
        .execute(complex)
        .unwrap();
    assert_gaps_unchanged(complex, cell);
    let context = format!("{cell}, round trip");
    assert_close(
        &in_place_reals(complex, &offsets, case.shape),
        &scaled_reals(case.shape, &case.reals, n as f64),
        &case.axes,
        round_trip_tolerance::<T>(n),
        &context,
    );
}

/// An in-place c2r cell on the complex view: the transform of the Hermitian input, read from
/// the reals of its storage, against the reference, and the gaps.
fn run_c2r_in_place<T, C, const N: usize>(cell: &Cell<T, N>, complex: &mut Array<C, N>)
where
    T: TestReal,
    C: ComplexView<T, N>,
{
    if !contiguous_halved_axis(cell, complex) {
        return;
    }
    let case = cell.case;
    fill(complex, &case.half);
    let layout = Layout::c2r_inplace(&*complex, &case.axes, case.shape[case.halved()]).unwrap();
    let class = AlignmentClass::of(&*complex);
    let plan = Planner::<T>::new()
        .alignment(class, class)
        .c2r_inplace(&layout)
        .unwrap();
    plan.execute(complex).unwrap();
    assert_gaps_unchanged(complex, cell);
    let offsets = in_place_real_offsets(&*complex, case.halved(), case.shape);
    assert_close(
        &in_place_reals(complex, &offsets, case.shape),
        &as_complex(&case.inverse, 1.0),
        &case.axes,
        tolerance::<T>(C2R_TOLERANCE_FACTOR, case.logical_size()),
        cell,
    );
}

fn run_cell<T, R, C, const N: usize>(
    cell: &Cell<T, N>,
    real: &mut Array<R, N>,
    complex: &mut Array<C, N>,
) where
    T: TestReal,
    R: RealView<T, N>,
    C: ComplexView<T, N>,
{
    match (cell.family, cell.policy) {
        (Family::R2c, PolicyKind::InPlace) => run_r2c_in_place(cell, complex),
        (Family::C2r, PolicyKind::InPlace) => run_c2r_in_place(cell, complex),
        (Family::R2c, _) => run_r2c(cell, real, complex),
        (Family::C2r, _) => run_c2r(cell, real, complex),
    }
}

/// The strides of a strided view with steps of 2: `[2, 2 n0, 2 n0 n1, …]`.
fn steps_of_two<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut stride = [2; N];
    for axis in 1..N {
        stride[axis] = stride[axis - 1] * shape[axis - 1];
    }
    stride
}

/// Build the real and complex views of `cell.order` and run the cell on them. Subviews have
/// parents two or three elements larger on each axis and non-zero origins; the parents and
/// the storage of strided views are filled with the sentinel.
macro_rules! run_views {
    ($t:ty, $n:literal, $cell:expr) => {{
        let cell: &Cell<$t, $n> = $cell;
        let shape = cell.case.shape;
        let complex_shape = cell.case.complex_shape();
        let grow = |shape: [usize; $n], by: usize| shape.map(|n| n + by);
        match cell.order {
            Order::ColumnMajor => {
                let mut real = real_sentinel_parent::<$t, $n>(shape);
                let mut complex = sentinel_parent::<$t, $n>(complex_shape);
                run_cell(cell, &mut real, &mut complex);
            }
            Order::RowMajor => {
                let mut real = StridedDynArray::<$t, $n>::row_major(shape);
                let mut complex = StridedDynArray::<Complex<$t>, $n>::row_major(complex_shape);
                run_cell(cell, &mut real, &mut complex);
            }
            Order::Subview => {
                let mut real = real_sentinel_parent::<$t, $n>(grow(shape, 3));
                let mut complex = sentinel_parent::<$t, $n>(grow(complex_shape, 2));
                run_cell(
                    cell,
                    &mut real.r_mut().into_subview([2; $n], shape),
                    &mut complex.r_mut().into_subview([1; $n], complex_shape),
                );
            }
            Order::Strided => {
                let real_len = 2 * shape.iter().product::<usize>();
                let complex_len = 2 * complex_shape.iter().product::<usize>();
                let mut real = filled(real_sentinel::<$t>(), real_len);
                let mut complex = filled(sentinel::<$t>(), complex_len);
                run_cell(
                    cell,
                    &mut StridedSliceArrayMut::from_shape_and_stride(
                        &mut real[..],
                        shape,
                        steps_of_two(shape),
                    ),
                    &mut StridedSliceArrayMut::from_shape_and_stride(
                        &mut complex[..],
                        complex_shape,
                        steps_of_two(complex_shape),
                    ),
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

/// Run the cells of both families, the memory orders `orders` and every policy on `case`.
fn run_case<T: TestReal, const N: usize>(
    case: &Case<T, N>,
    orders: &[Order],
    run: fn(&Cell<T, N>),
) {
    for family in [Family::R2c, Family::C2r] {
        for &order in orders {
            for policy in PolicyKind::ALL {
                run(&Cell {
                    case,
                    family,
                    order,
                    policy,
                });
            }
        }
    }
}

/// The seed of the case of `shape` and `axes`.
fn case_seed(shape: &[usize], axes: &[usize]) -> u64 {
    let code = shape
        .iter()
        .chain([&usize::MAX])
        .chain(axes)
        .fold(0_u64, |code, &value| {
            code.wrapping_mul(31).wrapping_add(value as u64)
        });
    SEED ^ code
}

/// The fixed cells (F5 brief, work item 7): sizes 1, 2, 3, 17, 97, 140, 588 and 64 along the
/// halved axis, `[n]` and `[n, 3]` along each axis, `[6, 5, 4]` over two and three axes in
/// two orders, every policy. Column-major arrays run every size; row-major arrays, subviews
/// and strided views, whose cells cost as much and check the same layouts at other strides,
/// run the sizes up to 17 and `[6, 5, 4]`, and the sampled cells cover the rest.
fn check_fixed_cells<T: TestReal>() {
    for n in [1, 2, 3, 17, 97, 140, 588, 64] {
        let orders: &[Order] = if n <= 17 {
            &Order::ALL
        } else {
            &[Order::ColumnMajor]
        };
        let shape = [n];
        let case = Case::<T, 1>::new(shape, &[0], case_seed(&shape, &[0]));
        run_case(&case, orders, run_rank1::<T>);
        let shape = [n, 3];
        for axis in [0, 1] {
            let case = Case::<T, 2>::new(shape, &[axis], case_seed(&shape, &[axis]));
            run_case(&case, orders, run_rank2::<T>);
        }
    }
    let shape = [6, 5, 4];
    for axes in [&[0, 1][..], &[1, 0], &[0, 1, 2], &[2, 1, 0]] {
        let case = Case::<T, 3>::new(shape, axes, case_seed(&shape, axes));
        run_case(&case, &Order::ALL, run_rank3::<T>);
    }
}

#[test]
fn test_real_fixed_cells_f64() {
    check_fixed_cells::<f64>();
}

#[test]
fn test_real_fixed_cells_f32() {
    check_fixed_cells::<f32>();
}

#[test]
fn test_real_sampled_cells() {
    // 16 cells from the product of families, memory orders, shapes with extents up to 12,
    // axes, policies and precisions, drawn with the fixed seed.
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xf5a);
    for _ in 0..16 {
        let family = if rng.random_bool(0.5) {
            Family::R2c
        } else {
            Family::C2r
        };
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
        let mut extent = || rng.random_range(1..=12_usize);
        macro_rules! sampled {
            ($run:ident, $n:literal, $shape:expr) => {{
                let shape: [usize; $n] = $shape;
                if single {
                    let case = Case::<f32, $n>::new(shape, &axes, seed);
                    $run(&Cell {
                        case: &case,
                        family,
                        order,
                        policy,
                    });
                } else {
                    let case = Case::<f64, $n>::new(shape, &axes, seed);
                    $run(&Cell {
                        case: &case,
                        family,
                        order,
                        policy,
                    });
                }
            }};
        }
        match rank {
            1 => sampled!(run_rank1, 1, [extent()]),
            2 => sampled!(run_rank2, 2, [extent(), extent()]),
            _ => sampled!(run_rank3, 3, [extent(), extent(), extent()]),
        }
    }
}

// Identities (design Section 7.3).

fn check_round_trip_beyond_the_reference<T: TestReal>() {
    // 2^16, beyond the reference's range: c2r after r2c is N times the input, out of place
    // and in place.
    let n = 1 << 16;
    let values = random_reals::<T>(SEED, n);
    let mut real = aligned(&values);
    let mut complex = filled(sentinel::<T>(), n / 2 + 1);
    let layout = Layout::with_storage(
        &[Dim {
            n: n as isize,
            is: 1,
            os: 1,
        }],
        &[],
        (Storage::Real, Storage::Complex),
        Placement::OutOfPlace,
    )
    .unwrap();
    let inverse = Layout::with_storage(
        layout.dims(),
        &[],
        (Storage::Complex, Storage::Real),
        Placement::OutOfPlace,
    )
    .unwrap();
    Planner::<T>::new()
        .r2c(&layout)
        .unwrap()
        .execute(&real[..], &mut complex[..])
        .unwrap();
    Planner::<T>::new()
        .c2r(&inverse)
        .unwrap()
        .execute(&mut complex[..], &mut real[..])
        .unwrap();
    let expected: Vec<Complex<f64>> = values
        .iter()
        .map(|&x| Complex::new(n as f64 * TestReal::to_f64(x), 0.0))
        .collect();
    let actual: Vec<Complex<f64>> = real
        .iter()
        .map(|&x| Complex::new(TestReal::to_f64(x), 0.0))
        .collect();
    let error = relative_error(&actual, &expected);
    assert!(
        error <= round_trip_tolerance::<T>(n),
        "{} 2^16 round trip: relative error {error:e}",
        T::NAME
    );

    let mut data = filled(sentinel::<T>(), n / 2 + 1);
    as_reals_mut(&mut data[..])[..n].copy_from_slice(&values);
    let layout = Layout::with_storage(
        layout.dims(),
        &[],
        (Storage::Real, Storage::Complex),
        Placement::InPlace,
    )
    .unwrap();
    let inverse = Layout::with_storage(
        layout.dims(),
        &[],
        (Storage::Complex, Storage::Real),
        Placement::InPlace,
    )
    .unwrap();
    let planner = Planner::<T>::new();
    planner
        .r2c_inplace(&layout)
        .unwrap()
        .execute(&mut data[..])
        .unwrap();
    planner
        .c2r_inplace(&inverse)
        .unwrap()
        .execute(&mut data[..])
        .unwrap();
    let actual: Vec<Complex<f64>> = as_reals(&data[..])[..n]
        .iter()
        .map(|&x| Complex::new(TestReal::to_f64(x), 0.0))
        .collect();
    let error = relative_error(&actual, &expected);
    assert!(
        error <= round_trip_tolerance::<T>(n),
        "{} 2^16 in-place round trip: relative error {error:e}",
        T::NAME
    );
}

#[test]
fn test_round_trip_beyond_the_reference() {
    check_round_trip_beyond_the_reference::<f64>();
    check_round_trip_beyond_the_reference::<f32>();
}

fn check_r2c_against_c2c<T: TestReal>() {
    // The r2c transform of real data is the first n / 2 + 1 entries of the c2c transform of
    // the same data with imaginary part 0, along the halved axis.
    for (shape, axes) in [([97, 4], &[0][..]), ([6, 8], &[0, 1]), ([7, 5], &[1, 0])] {
        let values = random_reals::<T>(case_seed(&shape, axes), shape[0] * shape[1]);
        let complex_shape = halved_shape(shape, *axes.last().unwrap());
        let mut real = real_sentinel_parent::<T, 2>(shape);
        let mut spectrum = sentinel_parent::<T, 2>(complex_shape);
        let mut data = sentinel_parent::<T, 2>(shape);
        let mut full = sentinel_parent::<T, 2>(shape);
        fill(&mut real, &values);
        let zero = T::round_from(0.0);
        let as_complex: Vec<Complex<T>> = values.iter().map(|&x| Complex::new(x, zero)).collect();
        fill(&mut data, &as_complex);
        Planner::<T>::new()
            .r2c(&Layout::over_axes(&real, &spectrum, axes).unwrap())
            .unwrap()
            .execute(&real, &mut spectrum)
            .unwrap();
        Planner::<T>::new()
            .c2c(
                &Layout::over_axes(&data, &full, axes).unwrap(),
                Direction::Forward,
            )
            .unwrap()
            .execute(&data, &mut full)
            .unwrap();
        let full = complex_values(&full);
        let mut cut = DynArray::<Complex<f64>, 2>::from_shape(complex_shape);
        for index in multi_indices(complex_shape) {
            cut[index] = full[index];
        }
        let context = format!(
            "{} r2c against c2c, shape {shape:?}, axes {axes:?}",
            T::NAME
        );
        let n = axes.iter().map(|&axis| shape[axis]).product();
        assert_close(
            &complex_values(&spectrum),
            &cut,
            axes,
            2.0 * tolerance::<T>(R2C_TOLERANCE_FACTOR, n),
            &context,
        );
    }
}

#[test]
fn test_r2c_against_c2c() {
    check_r2c_against_c2c::<f64>();
    check_r2c_against_c2c::<f32>();
}

fn check_linearity_and_parseval<T: TestReal>() {
    let shape = [12, 7];
    let axes = [1, 0];
    let len = shape[0] * shape[1];
    let (a, b) = (T::round_from(0.75), T::round_from(-1.5));
    let x = random_reals::<T>(SEED + 1, len);
    let y = random_reals::<T>(SEED + 2, len);
    let combined: Vec<T> = x.iter().zip(&y).map(|(&x, &y)| a * x + b * y).collect();
    let complex_shape = halved_shape(shape, 0);
    let transform = |values: &[T]| {
        let mut real = real_sentinel_parent::<T, 2>(shape);
        let mut complex = sentinel_parent::<T, 2>(complex_shape);
        fill(&mut real, values);
        Planner::<T>::new()
            .r2c(&Layout::over_axes(&real, &complex, &axes).unwrap())
            .unwrap()
            .execute(&real, &mut complex)
            .unwrap();
        complex_values(&complex)
    };
    let (tx, ty, tc) = (transform(&x), transform(&y), transform(&combined));
    let mut expected = DynArray::<Complex<f64>, 2>::from_shape(complex_shape);
    for index in multi_indices(complex_shape) {
        expected[index] = tx[index] * TestReal::to_f64(a) + ty[index] * TestReal::to_f64(b);
    }
    let n = len;
    let context = format!("{} linearity", T::NAME);
    assert_close(
        &tc,
        &expected,
        &axes,
        3.0 * tolerance::<T>(R2C_TOLERANCE_FACTOR, n),
        &context,
    );

    // Parseval: Σ |x|² = (1/N) Σ w_k |X_k|² over the half, with w_k = 2 for the entries
    // whose mirror is not in the half: along the halved axis of extent 12, all but 0 and 6.
    let energy: f64 = x.iter().map(|&x| TestReal::to_f64(x).powi(2)).sum();
    let spectrum: f64 = multi_indices(complex_shape)
        .map(|index| {
            let weight = if index[0] == 0 || index[0] == 6 {
                1.0
            } else {
                2.0
            };
            weight * tx[index].norm_sqr()
        })
        .sum();
    let error = (spectrum / n as f64 - energy).abs() / energy;
    assert!(
        error <= 2.0 * tolerance::<T>(R2C_TOLERANCE_FACTOR, n),
        "{} Parseval: relative error {error:e}",
        T::NAME
    );
}

#[test]
fn test_linearity_and_parseval() {
    check_linearity_and_parseval::<f64>();
    check_linearity_and_parseval::<f32>();
}

fn check_batched_against_single_plans<T: TestReal>() {
    // Along axis 0 of [n, b]: one batched plan against one plan per column, to round-off,
    // r2c and c2r. Each single plan is made for its column's classes.
    let (n, b) = (40, 6);
    let h = n / 2 + 1;
    let values = random_reals::<T>(SEED + 3, n * b);
    let mut real = real_sentinel_parent::<T, 2>([n, b]);
    let mut batched = sentinel_parent::<T, 2>([h, b]);
    let mut single = sentinel_parent::<T, 2>([h, b]);
    fill(&mut real, &values);
    Planner::<T>::new()
        .r2c(&Layout::along_axis(&real, &batched, 0).unwrap())
        .unwrap()
        .execute(&real, &mut batched)
        .unwrap();
    for j in 0..b {
        let layout = Layout::along_axis(&real.r().col(j), &single.r().col(j), 0).unwrap();
        Planner::<T>::new()
            .alignment(
                AlignmentClass::of(&real.r().col(j)),
                AlignmentClass::of(&single.r().col(j)),
            )
            .r2c(&layout)
            .unwrap()
            .execute(&real.r().col(j), &mut single.r_mut().col(j))
            .unwrap();
    }
    let context = format!("{} batched r2c against single plans", T::NAME);
    assert_close(
        &complex_values(&batched),
        &complex_values(&single),
        &[0],
        2.0 * tolerance::<T>(R2C_TOLERANCE_FACTOR, n),
        &context,
    );

    // The same computation gives the same bits: one plan, made for the columns at class 0
    // (columns 0 and 8 of real [8, b] and complex [5, b], in both precisions), on column 8 of
    // the parent and on a standalone aligned copy of it.
    let (n, b) = (8, 9);
    let values = random_reals::<T>(SEED + 4, n * b);
    let mut real = real_sentinel_parent::<T, 2>([n, b]);
    let mut output = sentinel_parent::<T, 2>([5, b]);
    fill(&mut real, &values);
    let layout = Layout::along_axis(&real.r().col(0), &output.r().col(0), 0).unwrap();
    let plan = Planner::<T>::new().r2c(&layout).unwrap();
    plan.execute(&real.r().col(8), &mut output.r_mut().col(8))
        .unwrap();
    let mut copy = real_sentinel_parent::<T, 1>([n]);
    fill(&mut copy, &values[8 * n..9 * n]);
    let mut copy_output = sentinel_parent::<T, 1>([5]);
    plan.execute(&copy, &mut copy_output).unwrap();
    let column = complex_values(&output.r().col(8));
    let alone = complex_values(&copy_output);
    assert!(
        multi_indices([5]).all(|index| column[index] == alone[index]),
        "{}: one plan on a column and on its copy differs",
        T::NAME
    );
}

#[test]
fn test_batched_against_single_plans() {
    check_batched_against_single_plans::<f64>();
    check_batched_against_single_plans::<f32>();
}

// Defaults and policies (design Section 7.4).

#[test]
fn test_defaults_and_policies() {
    let real = real_sentinel_parent::<f64, 2>([8, 3]);
    let mut spectrum = sentinel_parent::<f64, 2>([5, 3]);
    let r2c = Layout::along_axis(&real, &spectrum, 0).unwrap();
    let c2r = Layout::along_axis(&spectrum, &real, 0).unwrap();

    // The policies of DC1, by the types of the plans.
    let _: Plan<f64, R2c, Preserving> = Planner::new().r2c(&r2c).unwrap();
    let _: Plan<f64, R2c, Preserving> = Planner::new().preserve_input().r2c(&r2c).unwrap();
    let _: Plan<f64, R2c, Destroying> = Planner::new().may_destroy_input().r2c(&r2c).unwrap();
    let _: Plan<f64, C2r, Destroying> = Planner::new().c2r(&c2r).unwrap();
    let _: Plan<f64, C2r, Destroying> = Planner::new().may_destroy_input().c2r(&c2r).unwrap();
    let preserving: Plan<f64, C2r, Preserving> = Planner::new().preserve_input().c2r(&c2r).unwrap();

    // A preserving one-axis c2r leaves its input bit for bit.
    let half: Vec<Complex<f64>> = (0..15)
        .map(|k| Complex::new(k as f64, if k % 5 == 0 { 0.0 } else { -0.5 }))
        .collect();
    fill(&mut spectrum, &half);
    let before = storage_bits(&spectrum);
    let mut output = real_sentinel_parent::<f64, 2>([8, 3]);
    preserving.execute(&spectrum, &mut output).unwrap();
    assert!(storage_bits(&spectrum) == before);

    // Over two axes FFTW cannot preserve the input; the planner says so before FFTW is
    // called, on scratch and on the caller's buffers.
    let mut input = sentinel_parent::<f64, 2>([3, 5]);
    let mut output = real_sentinel_parent::<f64, 2>([3, 8]);
    let layout = Layout::over_axes(&input, &output, &[0, 1]).unwrap();
    let unsupported = FftwError::PreserveInputUnsupported { rank: 2 };
    assert_eq!(
        Planner::<f64>::new()
            .preserve_input()
            .c2r(&layout)
            .unwrap_err(),
        unsupported
    );
    let output_before = storage_bits(&output);
    assert_eq!(
        Planner::<f64>::new()
            .preserve_input()
            .c2r_on(&mut input, &mut output, &layout)
            .unwrap_err(),
        unsupported
    );
    assert!(storage_bits(&output) == output_before);
    assert!(unsupported.to_string().contains("over 2 dimensions"));
    // The default plan over two axes exists.
    Planner::<f64>::new().c2r(&layout).unwrap();
}

#[test]
fn test_plan_properties() {
    let real = DynArray::<f64, 2>::from_shape([6, 8]);
    let complex = DynArray::<c64, 2>::from_shape([6, 5]);
    let plan = Planner::<f64>::new()
        .r2c(&Layout::over_axes(&real, &complex, &[0, 1]).unwrap())
        .unwrap();
    assert_eq!(plan.logical_size(), 48);
    assert_eq!(plan.scale_factor(Scale::None), 1.0);
    assert_eq!(plan.scale_factor(Scale::Inverse), 1.0 / 48.0);
    assert_eq!(plan.scale_factor(Scale::InverseSqrt), 1.0 / 48.0_f64.sqrt());
    assert!(format!("{plan:?}").contains("R2c"));
    assert!(!plan.description().is_empty());

    // Along axis 1, halved: 5 = 8 / 2 + 1.
    let plan = Planner::<f32>::new()
        .c2r(&Layout::along_axis(&complex, &real, 1).unwrap())
        .unwrap();
    assert_eq!(plan.logical_size(), 8);
    assert_eq!(plan.scale_factor(Scale::Inverse), 0.125_f32);
    assert!(format!("{plan:?}").contains("C2r"));
    // A c2c plan's logical size is its layout's.
    let layout = Layout::new(
        &[Dim {
            n: 12,
            is: 1,
            os: 1,
        }],
        &[],
        Placement::OutOfPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    assert_eq!(plan.logical_size(), 12);
    assert_eq!(plan.scale_factor(Scale::InverseSqrt), 1.0 / 12.0_f64.sqrt());
}

// Negative tests (design Section 7.4). Each returns its error and leaves the output's parent
// holding the sentinel.

fn assert_mismatch(error: FftwError, side: Side) {
    assert!(
        matches!(error, FftwError::LayoutMismatch { side: found, .. } if found == side),
        "expected a LayoutMismatch on the {side} side, found {error}"
    );
}

#[test]
fn test_real_layout_mismatch() {
    let mut real = real_sentinel_parent::<f64, 2>([8, 3]);
    let mut complex = sentinel_parent::<f64, 2>([5, 3]);
    let layout = Layout::along_axis(&real, &complex, 0).unwrap();
    let plan = Planner::<f64>::new().r2c(&layout).unwrap();

    // A wrong extent on the halved axis of the output, and a column more.
    for shape in [[4, 3], [6, 3], [5, 4]] {
        let mut output = sentinel_parent::<f64, 2>(shape);
        let before = storage_bits(&output);
        assert_mismatch(plan.execute(&real, &mut output).unwrap_err(), Side::Output);
        assert!(storage_bits(&output) == before);
    }
    // The real input of the complex shape.
    let short = real_sentinel_parent::<f64, 2>([5, 3]);
    assert_mismatch(plan.execute(&short, &mut complex).unwrap_err(), Side::Input);
    // A slice shorter than the complex span.
    let mut output = filled(sentinel::<f64>(), 14);
    assert_mismatch(
        plan.execute(&real, &mut output[..]).unwrap_err(),
        Side::Output,
    );
    assert!(
        output
            .iter()
            .all(|value| value.bits() == sentinel::<f64>().bits())
    );

    // Axes that fit each side alone but do not pair (DC15): a plan from a real [4, 2, 3]
    // and a complex [3, 2, 3] whose batch is transposed in memory accepts column-major
    // arrays [4, 3, 2] and [3, 3, 2] side by side, but not as a pair.
    let input = real_sentinel_parent::<f64, 3>([4, 2, 3]);
    let transposed = StridedDynArray::<c64, 3>::from_shape_and_stride([3, 2, 3], [1, 9, 3]);
    let plan = Planner::<f64>::new()
        .r2c(&Layout::along_axis(&input, &transposed, 0).unwrap())
        .unwrap();
    let other_input = real_sentinel_parent::<f64, 3>([4, 3, 2]);
    let mut other_output = sentinel_parent::<f64, 3>([3, 3, 2]);
    let before = storage_bits(&other_output);
    let error = plan.execute(&other_input, &mut other_output).unwrap_err();
    assert_mismatch(error.clone(), Side::Output);
    assert!(error.to_string().contains("pair by index"), "{error}");
    assert!(storage_bits(&other_output) == before);

    // In place, a complex buffer of another extent along the halved axis.
    let mut data = sentinel_parent::<f64, 2>([5, 3]);
    let layout = Layout::r2c_inplace(&data, &[0], 8).unwrap();
    let plan = Planner::<f64>::new().r2c_inplace(&layout).unwrap();
    let mut other = sentinel_parent::<f64, 2>([4, 3]);
    let before = storage_bits(&other);
    assert_mismatch(plan.execute(&mut other).unwrap_err(), Side::Output);
    assert!(storage_bits(&other) == before);
    plan.execute(&mut data).unwrap();
    let _ = &mut real;
}

#[test]
fn test_layouts_of_other_families() {
    // Every planner method rejects a layout of another family before FFTW is called: naming
    // the part axis when split storage is involved on one side only, the layout otherwise.
    let dims = [Dim { n: 8, is: 1, os: 1 }];
    let out = Placement::OutOfPlace;
    let make = |storage, placement| Layout::with_storage(&dims, &[], storage, placement).unwrap();
    let complex = (Storage::Complex, Storage::Complex);
    let split = (Storage::Split { gap: 8 }, Storage::Split { gap: 8 });
    let r2c = (Storage::Real, Storage::Complex);
    let c2r = (Storage::Complex, Storage::Real);
    let r2r = (Storage::Real, Storage::Real);
    let split_r2c = (Storage::Real, Storage::Split { gap: 5 });
    let in_place = Placement::InPlace;

    let named = |error: FftwError| match error {
        FftwError::InvalidLayout { dimension, reason } => (dimension, reason),
        other => panic!("expected InvalidLayout, found {other}"),
    };
    let planner = Planner::<f64>::new();
    let direction = Direction::Forward;

    // c2c and split c2c methods given real layouts.
    for (storage, dimension) in [(r2c, "layout"), (c2r, "layout"), (r2r, "layout")] {
        let (found, reason) = named(planner.c2c(&make(storage, out), direction).unwrap_err());
        assert_eq!(found, dimension);
        assert!(reason.contains("complex-to-complex plan needs"), "{reason}");
        let (found, _) = named(
            planner
                .split_c2c(&make(storage, out), direction)
                .unwrap_err(),
        );
        assert_eq!(found, "part axis");
    }
    let (found, _) = named(planner.c2c(&make(split_r2c, out), direction).unwrap_err());
    assert_eq!(found, "part axis");
    let (found, _) = named(
        planner
            .c2c_inplace(&make(r2c, in_place), direction)
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let (found, _) = named(
        planner
            .split_c2c_inplace(&make(r2r, in_place), direction)
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");

    // r2c and c2r methods given layouts of every other family.
    for (storage, dimension) in [
        (complex, "layout"),
        (c2r, "layout"),
        (r2r, "layout"),
        (split, "part axis"),
        (split_r2c, "part axis"),
    ] {
        let (found, reason) = named(planner.r2c(&make(storage, out)).unwrap_err());
        assert_eq!(found, dimension, "{reason}");
        assert!(reason.contains("real-to-complex plan needs"), "{reason}");
    }
    for (storage, dimension) in [
        (complex, "layout"),
        (r2c, "layout"),
        (r2r, "layout"),
        (split, "part axis"),
        (split_r2c, "part axis"),
    ] {
        let (found, reason) = named(planner.c2r(&make(storage, out)).unwrap_err());
        assert_eq!(found, dimension, "{reason}");
        assert!(reason.contains("complex-to-real plan needs"), "{reason}");
    }
    for storage in [complex, c2r] {
        let (found, _) = named(planner.r2c_inplace(&make(storage, in_place)).unwrap_err());
        assert_eq!(found, "layout");
    }
    for storage in [complex, r2c] {
        let (found, _) = named(planner.c2r_inplace(&make(storage, in_place)).unwrap_err());
        assert_eq!(found, "layout");
    }
    let (found, reason) = named(planner.r2c(&make(complex, out)).unwrap_err());
    assert_eq!(found, "layout");
    assert!(reason.contains("`c2c` planner methods"), "{reason}");

    // The `*_on` methods, on buffers that keep their sentinel.
    let mut reals = filled(real_sentinel::<f64>(), 32);
    let mut values = filled(sentinel::<f64>(), 32);
    let real_bits: Vec<_> = reals.iter().map(|value| value.bits()).collect();
    let complex_bits: Vec<_> = values.iter().map(|value| value.bits()).collect();
    let (found, _) = named(
        planner
            .r2c_on(&mut reals[..], &mut values[..], &make(c2r, out))
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let (found, _) = named(
        planner
            .c2r_on(&mut values[..], &mut reals[..], &make(r2c, out))
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let (found, _) = named(
        planner
            .r2c_inplace_on(&mut values[..], &make(complex, in_place))
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let (found, _) = named(
        planner
            .c2r_inplace_on(&mut values[..], &make(r2c, in_place))
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let mut other_values = values.clone();
    let (found, _) = named(
        planner
            .c2c_on(
                &mut values[..],
                &mut other_values[..],
                &make(r2c, out),
                direction,
            )
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let (found, _) = named(
        planner
            .c2c_inplace_on(&mut values[..], &make(c2r, in_place), direction)
            .unwrap_err(),
    );
    assert_eq!(found, "layout");
    let mut other_reals = reals.clone();
    let (found, _) = named(
        planner
            .split_c2c_on(
                &mut reals[..],
                &mut other_reals[..],
                &make(r2c, out),
                direction,
            )
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");
    let (found, _) = named(
        planner
            .split_c2c_inplace_on(&mut reals[..], &make(r2r, in_place), direction)
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");
    assert!(reals.iter().map(|value| value.bits()).eq(real_bits));
    assert!(values.iter().map(|value| value.bits()).eq(complex_bits));
    // The plan types exist for every family.
    let _ = (
        std::marker::PhantomData::<Plan<f64, C2c, InPlace>>,
        std::marker::PhantomData::<Plan<f64, SplitC2c, InPlace>>,
    );
}

#[test]
fn test_real_alignment_mismatch() {
    let layout = Layout::with_storage(
        &[Dim {
            n: 16,
            is: 1,
            os: 1,
        }],
        &[],
        (Storage::Real, Storage::Complex),
        Placement::OutOfPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new().r2c(&layout).unwrap();
    let reals = aligned(&random_reals::<f64>(SEED, 17));
    let mut output = filled(sentinel::<f64>(), 10);
    let sentinel_bits = sentinel::<f64>().bits();

    // One real off class 0: class 8.
    let error = plan.execute(&reals[1..], &mut output[..9]).unwrap_err();
    assert_eq!(
        error,
        FftwError::AlignmentMismatch {
            side: Side::Input,
            expected: AlignmentClass::default(),
            found: AlignmentClass::from_offset(8),
        }
    );
    // One complex value off class 0: class 16.
    let error = plan.execute(&reals[..16], &mut output[1..]).unwrap_err();
    assert_eq!(
        error,
        FftwError::AlignmentMismatch {
            side: Side::Output,
            expected: AlignmentClass::default(),
            found: AlignmentClass::from_offset(16),
        }
    );
    assert!(output.iter().all(|value| value.bits() == sentinel_bits));

    // A plan for class 8 on the input runs there, and classes that are not multiples of the
    // element's alignment are rejected before planning.
    let plan = Planner::<f64>::new()
        .alignment(AlignmentClass::from_offset(8), AlignmentClass::default())
        .r2c(&layout)
        .unwrap();
    plan.execute(&reals[1..], &mut output[..9]).unwrap();
    let error = Planner::<f32>::new()
        .alignment(AlignmentClass::from_offset(2), AlignmentClass::default())
        .r2c(&layout)
        .unwrap_err();
    assert!(
        matches!(error, FftwError::InvalidLayout { ref dimension, .. } if dimension == "alignment"),
        "{error}"
    );

    // In place, one complex value off class 0.
    let layout = Layout::with_storage(
        &[Dim {
            n: 16,
            is: 1,
            os: 1,
        }],
        &[],
        (Storage::Complex, Storage::Real),
        Placement::InPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new().c2r_inplace(&layout).unwrap();
    let mut data = filled(sentinel::<f64>(), 10);
    let error = plan.execute(&mut data[1..]).unwrap_err();
    assert!(
        matches!(
            error,
            FftwError::AlignmentMismatch {
                side: Side::Input,
                ..
            }
        ),
        "{error}"
    );
    assert!(data.iter().all(|value| value.bits() == sentinel_bits));
    // Unequal classes in place are rejected before planning.
    let error = Planner::<f64>::new()
        .alignment(AlignmentClass::default(), AlignmentClass::from_offset(16))
        .c2r_inplace(&layout)
        .unwrap_err();
    assert!(matches!(error, FftwError::InvalidLayout { .. }), "{error}");
}

#[test]
fn test_real_overlap() {
    // An r2c input of 16 reals and an output of 9 complex values that start 64 bytes into
    // the input: their byte ranges overlap.
    let layout = Layout::with_storage(
        &[Dim {
            n: 16,
            is: 1,
            os: 1,
        }],
        &[],
        (Storage::Real, Storage::Complex),
        Placement::OutOfPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new().r2c(&layout).unwrap();
    let mut buffer = filled(sentinel::<f64>(), 16);
    let base = buffer.as_mut_ptr();
    {
        // Input and output overlap, which safe borrows rule out. They are built from one
        // allocation only to test the check: the check reads the slices' addresses and
        // lengths and nothing else, and on the error no element is read or written through
        // either slice. Both start at class 0, so the alignment check passes.
        // SAFETY: the 16 reals are the first 8 complex values of `buffer`, and the output's
        // 9 complex values are its elements 4 to 12; both lie inside `buffer`, which
        // outlives the slices, and `Complex<f64>` is two `f64`s.
        let input = unsafe { std::slice::from_raw_parts(base.cast::<f64>().cast_const(), 16) };
        // SAFETY: as above.
        let output = unsafe { std::slice::from_raw_parts_mut(base.add(4), 9) };
        let error = plan.execute(input, output).unwrap_err();
        assert!(matches!(error, FftwError::Overlap { .. }), "{error}");
        if let FftwError::Overlap { input, output } = error {
            assert_eq!(input.end - input.start, 16 * 8);
            assert_eq!(output.end - output.start, 9 * 16);
        }
    }
    let sentinel_bits = sentinel::<f64>().bits();
    assert!(buffer.iter().all(|value| value.bits() == sentinel_bits));
}

#[test]
fn test_real_scratch_cap() {
    // A column of a row-major real [8, 1000] and of a complex [5, 1000]: 8 reals and 5
    // complex values 1000 apart, spanning 7001 reals and 4001 complex values.
    let mut real = StridedDynArray::<f64, 2>::row_major([8, 1000]);
    let mut complex = StridedDynArray::<c64, 2>::row_major([5, 1000]);
    let values = random_reals::<f64>(SEED, 8);
    fill(&mut real.r_mut().col(7), &values);
    let layout = Layout::along_axis(&real.r().col(7), &complex.r().col(7), 0).unwrap();
    let planner = Planner::<f64>::new().max_scratch(1024);
    assert_eq!(
        planner.r2c(&layout).unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: 7001 * 8 + 4001 * 16,
            bound: 1024,
        }
    );
    // The same layout plans on the arrays themselves.
    let plan = planner
        .r2c_on(
            &mut real.r_mut().col(7),
            &mut complex.r_mut().col(7),
            &layout,
        )
        .unwrap();
    plan.execute(&real.r().col(7), &mut complex.r_mut().col(7))
        .unwrap();
    let expected = reference_r2c([8], &values, &[0]);
    let actual = complex_values(&complex.r().col(7));
    assert_close(
        &actual,
        &expected,
        &[0],
        tolerance::<f64>(R2C_TOLERANCE_FACTOR, 8),
        &"scratch cap",
    );
    // Other columns are not written.
    assert_eq!(complex[[0, 6]], c64::new(0.0, 0.0));

    // In place, the scratch is the complex span, the larger one.
    let layout = Layout::with_storage(
        &[Dim { n: 8, is: 1, os: 1 }],
        &[Dim {
            n: 100,
            is: 10,
            os: 5,
        }],
        (Storage::Real, Storage::Complex),
        Placement::InPlace,
    )
    .unwrap();
    assert_eq!(
        planner.r2c_inplace(&layout).unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: 500 * 16,
            bound: 1024,
        }
    );
}
