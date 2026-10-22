//! Concurrency of complex-to-complex plans (tasks F4 and F3; design Sections 5 and 7.5).
//!
//! The tests run as threads of the test process, beside the other tests, and plan with
//! `Rigor::Estimate` only, so no wisdom of a greater rigor exists in the process and plans of
//! one layout made in different threads are the same plan (design, Section 3 item 4). Threads
//! are started with `std::thread::scope` in fixed numbers and overlap through `Barrier`s;
//! a panic in a thread fails the test when its scope ends. The tests that need a fresh process
//! are in the harness-free binary `tests/fftw_determinism.rs`.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, mpsc};
use std::time::Duration;

use num::Complex;

use crate::dense::array::StridedSliceArrayMut;
use crate::fftw::lock::with_fftw;
use crate::fftw::test_support::{
    PolicyKind, SEED, TestReal, aligned, assert_matches_reference, fill_view, logical_reference,
    multi_indices, random_values, real_sentinel, real_sentinel_parent, sentinel, sentinel_parent,
};
use crate::fftw::{
    AlignmentClass, C2CDirection, C2CPlan, C2c, Destroying, Direction, FftBufferMut,
    FftPlanInplace, FftwPlanFlags, FftwReal, InPlace, Layout, Plan, Planner, Preserving, SplitC2c,
};
use crate::{
    Array, BaseItem, DynArray, Shape, SliceArray, SliceArrayMut, Stride, StridedRawAccessMut,
    UnsafeRandomAccessByRef, UnsafeRandomAccessMut, c64,
};

/// nd-fmm's unit: `N_PHI` points on each of `RINGS` rings (design, Section 2.5).
const UNIT_SHAPE: [usize; 2] = [140, 8];

/// The elements of a unit: 1120 complex values, a multiple of 64 bytes in both precisions,
/// so that consecutive units of an aligned parent all have alignment class 0.
const UNIT: usize = 140 * 8;

/// The threads that execute one plan at once.
const THREADS: usize = 8;

/// The executions of each thread.
const ITERATIONS: usize = 8;

/// A complex-to-complex plan of one of the three input policies.
enum AnyPlan<T: FftwReal> {
    Preserving(Plan<T, C2c, Preserving>),
    Destroying(Plan<T, C2c, Destroying>),
    InPlace(Plan<T, C2c, InPlace>),
}

impl<T: TestReal> AnyPlan<T> {
    /// A plan of `policy` for `layout` (out of place) or `inplace` (in place), on scratch.
    fn new(
        planner: &Planner<T>,
        policy: PolicyKind,
        layout: &Layout,
        inplace: &Layout,
        direction: Direction,
    ) -> Self {
        match policy {
            PolicyKind::Preserving => Self::Preserving(planner.c2c(layout, direction).unwrap()),
            PolicyKind::Destroying => Self::Destroying(
                planner
                    .clone()
                    .may_destroy_input()
                    .c2c(layout, direction)
                    .unwrap(),
            ),
            PolicyKind::InPlace => Self::InPlace(planner.c2c_inplace(inplace, direction).unwrap()),
        }
    }

    /// A plan of `policy` for nd-fmm's unit, along axis 0, at alignment class 0.
    fn unit(policy: PolicyKind, direction: Direction) -> Self {
        let scratch = [Complex::<T>::default(); UNIT];
        let unit = SliceArray::from_shape(&scratch[..], UNIT_SHAPE);
        let layout = Layout::along_axis(&unit, &unit, 0).unwrap();
        let inplace = Layout::along_axis_inplace(&unit, 0).unwrap();
        Self::new(&Planner::new(), policy, &layout, &inplace, direction)
    }

    /// Execute from `input` into `output` out of place, or on `input` in place, where the
    /// result then is.
    fn execute<I, O>(&self, input: &mut I, output: &mut O)
    where
        I: FftBufferMut<Complex<T>> + ?Sized,
        O: FftBufferMut<Complex<T>> + ?Sized,
    {
        match self {
            Self::Preserving(plan) => plan.execute(&*input, output),
            Self::Destroying(plan) => plan.execute(input, output),
            Self::InPlace(plan) => plan.execute(input),
        }
        .unwrap()
    }

    fn description(&self) -> String {
        match self {
            Self::Preserving(plan) => plan.description(),
            Self::Destroying(plan) => plan.description(),
            Self::InPlace(plan) => plan.description(),
        }
    }

    fn fingerprint(&self) -> u64 {
        match self {
            Self::Preserving(plan) => plan.fingerprint(),
            Self::Destroying(plan) => plan.fingerprint(),
            Self::InPlace(plan) => plan.fingerprint(),
        }
    }
}

/// The bits of complex values, for comparisons bit for bit.
fn bits<T: TestReal>(values: &[Complex<T>]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|value| (value.re.to_f64().to_bits(), value.im.to_f64().to_bits()))
        .collect()
}

/// The seed of the input of unit `k`.
fn unit_seed(k: usize) -> u64 {
    SEED + k as u64
}

/// The context of a message from the threads of a test.
struct Context<'a> {
    precision: &'static str,
    policy: PolicyKind,
    seed: u64,
    thread: usize,
    iteration: usize,
    test: &'a str,
}

impl fmt::Display for Context<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} {:?}, seed {:#x}, thread {}, iteration {}",
            self.test, self.precision, self.policy, self.seed, self.thread, self.iteration
        )
    }
}

/// The units of an executing thread: its input and output, its input values, and the bits
/// of its result from the sequential execution.
struct Unit<'a, T> {
    input: &'a mut [Complex<T>],
    output: &'a mut [Complex<T>],
    values: Vec<Complex<T>>,
    expected: Vec<(u64, u64)>,
    seed: u64,
}

impl<T: TestReal> Unit<'_, T> {
    /// Restore the input, fill the output with the sentinel, execute `plan` once on the unit
    /// and assert that the result equals the sequential one, bit for bit.
    fn execute_and_check(&mut self, plan: &AnyPlan<T>, policy: PolicyKind, context: &Context) {
        self.input.copy_from_slice(&self.values);
        self.output.fill(sentinel::<T>());
        plan.execute(
            &mut SliceArrayMut::from_shape(&mut *self.input, UNIT_SHAPE),
            &mut SliceArrayMut::from_shape(&mut *self.output, UNIT_SHAPE),
        );
        let result = match policy {
            PolicyKind::InPlace => &*self.input,
            _ => &*self.output,
        };
        assert!(
            bits(result) == self.expected,
            "{context}: the result differs from the sequential one"
        );
    }
}

/// Split the parents `input` and `output` into units, one per thread, with seeded input from
/// `unit_seed(first_seed + k)`, and compute the result of `plan` on each unit sequentially.
fn units<'a, T: TestReal>(
    plan: &AnyPlan<T>,
    policy: PolicyKind,
    input: &'a mut [Complex<T>],
    output: &'a mut [Complex<T>],
    first_seed: usize,
) -> Vec<Unit<'a, T>> {
    // The units are disjoint chunks of the parents, as `chunks_exact_mut(UNIT)` gives them.
    let (inputs, _) = input.as_chunks_mut::<UNIT>();
    let (outputs, _) = output.as_chunks_mut::<UNIT>();
    inputs
        .iter_mut()
        .zip(outputs)
        .enumerate()
        .map(|(k, (input, output))| {
            let seed = unit_seed(first_seed + k);
            let mut unit = Unit {
                input: &mut input[..],
                output: &mut output[..],
                values: random_values::<T>(seed, UNIT),
                expected: Vec::new(),
                seed,
            };
            unit.input.copy_from_slice(&unit.values);
            plan.execute(
                &mut SliceArrayMut::from_shape(&mut *unit.input, UNIT_SHAPE),
                &mut SliceArrayMut::from_shape(&mut *unit.output, UNIT_SHAPE),
            );
            unit.expected = bits(match policy {
                PolicyKind::InPlace => &*unit.input,
                _ => &*unit.output,
            });
            unit
        })
        .collect()
}

/// Parents of `count` units at `FFTW_ALIGNED`, filled with the sentinel.
fn parents<T: TestReal>(
    count: usize,
) -> (
    crate::dense::array::AlignedDynArray<Complex<T>, 1, { crate::FFTW_ALIGNED }>,
    crate::dense::array::AlignedDynArray<Complex<T>, 1, { crate::FFTW_ALIGNED }>,
) {
    (
        sentinel_parent::<T, 1>([count * UNIT]),
        sentinel_parent::<T, 1>([count * UNIT]),
    )
}

#[test]
fn test_plans_are_send_and_sync() {
    fn send_and_sync<P: Send + Sync>() {}
    send_and_sync::<Plan<f64, C2c, Preserving>>();
    send_and_sync::<Plan<f64, C2c, Destroying>>();
    send_and_sync::<Plan<f64, C2c, InPlace>>();
    send_and_sync::<Plan<f32, C2c, Preserving>>();
    send_and_sync::<Plan<f32, C2c, Destroying>>();
    send_and_sync::<Plan<f32, C2c, InPlace>>();
    send_and_sync::<Plan<f64, SplitC2c, Preserving>>();
    send_and_sync::<Plan<f64, SplitC2c, Destroying>>();
    send_and_sync::<Plan<f64, SplitC2c, InPlace>>();
    send_and_sync::<Plan<f32, SplitC2c, Preserving>>();
    send_and_sync::<Plan<f32, SplitC2c, Destroying>>();
    send_and_sync::<Plan<f32, SplitC2c, InPlace>>();
}

/// One plan of `policy` executed by `THREADS` threads at once, `ITERATIONS` times each, on
/// disjoint units of one parent (design, Section 5 rule 2).
fn check_one_plan_many_threads<T: TestReal>(policy: PolicyKind) {
    let plan = AnyPlan::<T>::unit(policy, Direction::Forward);
    let (mut input, mut output) = parents::<T>(THREADS);
    let units = units(
        &plan,
        policy,
        input.data_mut().unwrap(),
        output.data_mut().unwrap(),
        0,
    );
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for (thread, mut unit) in units.into_iter().enumerate() {
            let (plan, barrier) = (&plan, &barrier);
            scope.spawn(move || {
                for iteration in 0..ITERATIONS {
                    let context = Context {
                        precision: T::NAME,
                        policy,
                        seed: unit.seed,
                        thread,
                        iteration,
                        test: "one plan, many threads",
                    };
                    // Every thread starts each execution together with the others.
                    barrier.wait();
                    unit.execute_and_check(plan, policy, &context);
                }
            });
        }
    });
}

#[test]
fn test_one_plan_many_threads() {
    for policy in PolicyKind::ALL {
        check_one_plan_many_threads::<f64>(policy);
        check_one_plan_many_threads::<f32>(policy);
    }
}

/// nd-fmm's box as split planes (task F3): `UNIT_SHAPE` with the part axis outermost.
const SPLIT_SHAPE: [usize; 3] = [140, 8, 2];

/// The reals of a split box: 2240, a multiple of 64 bytes in both precisions, so that
/// consecutive boxes of an aligned parent all have alignment class 0.
const SPLIT_UNIT: usize = 2 * UNIT;

/// A split complex-to-complex plan of one of the three input policies.
enum AnySplitPlan<T: FftwReal> {
    Preserving(Plan<T, SplitC2c, Preserving>),
    Destroying(Plan<T, SplitC2c, Destroying>),
    InPlace(Plan<T, SplitC2c, InPlace>),
}

impl<T: TestReal> AnySplitPlan<T> {
    /// A plan of `policy` for nd-fmm's split box, along logical axis 0, at class 0.
    fn unit(policy: PolicyKind, direction: Direction) -> Self {
        let scratch = [T::default(); SPLIT_UNIT];
        let unit = SliceArray::from_shape(&scratch[..], SPLIT_SHAPE);
        let planner = Planner::<T>::new();
        match policy {
            PolicyKind::Preserving => {
                let layout = Layout::split_along_axis(&unit, 2, &unit, 2, 0).unwrap();
                Self::Preserving(planner.split_c2c(&layout, direction).unwrap())
            }
            PolicyKind::Destroying => {
                let layout = Layout::split_along_axis(&unit, 2, &unit, 2, 0).unwrap();
                Self::Destroying(
                    planner
                        .may_destroy_input()
                        .split_c2c(&layout, direction)
                        .unwrap(),
                )
            }
            PolicyKind::InPlace => {
                let layout = Layout::split_along_axis_inplace(&unit, 2, 0).unwrap();
                Self::InPlace(planner.split_c2c_inplace(&layout, direction).unwrap())
            }
        }
    }

    /// Execute from the box `input` into the box `output` out of place, or on `input` in
    /// place.
    fn execute(&self, input: &mut [T], output: &mut [T]) {
        let mut input = SliceArrayMut::from_shape(input, SPLIT_SHAPE);
        let mut output = SliceArrayMut::from_shape(output, SPLIT_SHAPE);
        match self {
            Self::Preserving(plan) => plan.execute(&input, &mut output),
            Self::Destroying(plan) => plan.execute(&mut input, &mut output),
            Self::InPlace(plan) => plan.execute(&mut input),
        }
        .unwrap()
    }
}

/// The bits of reals, for comparisons bit for bit.
fn real_bits<T: TestReal>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|&value| TestReal::to_f64(value).to_bits())
        .collect()
}

/// One split plan of `policy` executed by `THREADS` threads at once, `ITERATIONS` times
/// each, on disjoint boxes of one parent, against the sequential results bit for bit.
fn check_one_split_plan_many_threads<T: TestReal>(policy: PolicyKind) {
    let plan = AnySplitPlan::<T>::unit(policy, Direction::Forward);
    let mut input = real_sentinel_parent::<T, 1>([THREADS * SPLIT_UNIT]);
    let mut output = real_sentinel_parent::<T, 1>([THREADS * SPLIT_UNIT]);
    let (inputs, _) = input.data_mut().unwrap().as_chunks_mut::<SPLIT_UNIT>();
    let (outputs, _) = output.data_mut().unwrap().as_chunks_mut::<SPLIT_UNIT>();
    // Each box with its seeded input and its sequential result.
    let boxes: Vec<_> = inputs
        .iter_mut()
        .zip(outputs)
        .enumerate()
        .map(|(k, (input, output))| {
            let seed = unit_seed(k);
            let values: Vec<T> = random_values::<T>(seed, UNIT)
                .into_iter()
                .flat_map(|value| [value.re, value.im])
                .collect();
            input.copy_from_slice(&values);
            plan.execute(input, output);
            let expected = real_bits(match policy {
                PolicyKind::InPlace => &input[..],
                _ => &output[..],
            });
            (input, output, values, expected, seed)
        })
        .collect();
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for (thread, (input, output, values, expected, seed)) in boxes.into_iter().enumerate() {
            let (plan, barrier) = (&plan, &barrier);
            scope.spawn(move || {
                for iteration in 0..ITERATIONS {
                    let context = Context {
                        precision: T::NAME,
                        policy,
                        seed,
                        thread,
                        iteration,
                        test: "one split plan, many threads",
                    };
                    input.copy_from_slice(&values);
                    output.fill(real_sentinel::<T>());
                    // Every thread starts each execution together with the others.
                    barrier.wait();
                    plan.execute(input, output);
                    let result = match policy {
                        PolicyKind::InPlace => &input[..],
                        _ => &output[..],
                    };
                    assert!(
                        real_bits(result) == expected,
                        "{context}: the result differs from the sequential one"
                    );
                }
            });
        }
    });
}

#[test]
fn test_one_split_plan_many_threads() {
    for policy in PolicyKind::ALL {
        check_one_split_plan_many_threads::<f64>(policy);
        check_one_split_plan_many_threads::<f32>(policy);
    }
}

/// The arrays and views of the planning threads (F2's memory orders), all over aligned
/// storage so that their alignment classes are the same in every thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    ColumnMajor,
    RowMajor,
    Subview,
    Slice,
}

/// The shape of the planning threads' problems, transformed along axis 0.
const PLANNING_SHAPE: [usize; 2] = [12, 5];

/// Make, describe, execute once and drop a plan of `policy` for `input` and `output`; assert
/// the result against the logical reference, and return the plan's fingerprint.
fn plan_and_execute<T, AIn, AOut>(
    policy: PolicyKind,
    input: &mut Array<AIn, 2>,
    output: &mut Array<AOut, 2>,
    context: &str,
) -> u64
where
    T: TestReal,
    AIn: BaseItem<Item = Complex<T>>
        + StridedRawAccessMut
        + Shape<2>
        + Stride<2>
        + UnsafeRandomAccessByRef<2>
        + UnsafeRandomAccessMut<2>,
    AOut: BaseItem<Item = Complex<T>>
        + StridedRawAccessMut
        + Shape<2>
        + Stride<2>
        + UnsafeRandomAccessByRef<2>
        + UnsafeRandomAccessMut<2>,
{
    let values = fill_view(input, SEED);
    let layout = Layout::along_axis(input, output, 0).unwrap();
    let inplace = Layout::along_axis_inplace(input, 0).unwrap();
    // Subviews and slices at an offset have other classes than their parents; in place,
    // the input is also the output.
    let input_class = AlignmentClass::of(&*input);
    let output_class = match policy {
        PolicyKind::InPlace => input_class,
        _ => AlignmentClass::of(&*output),
    };
    let planner = Planner::<T>::new().alignment(input_class, output_class);
    let plan = AnyPlan::new(&planner, policy, &layout, &inplace, Direction::Forward);
    assert!(!plan.description().is_empty(), "{context}: no description");
    let fingerprint = plan.fingerprint();
    plan.execute(input, output);
    drop(plan);
    let reference = logical_reference(PLANNING_SHAPE, &values, &[0], Direction::Forward);
    match policy {
        PolicyKind::InPlace => assert_matches_reference(&*input, &reference, &[0], &context),
        _ => assert_matches_reference(&*output, &reference, &[0], &context),
    }
    fingerprint
}

/// Plan, execute and drop plans for every order and policy in precision `T`; return each
/// problem's label and fingerprint.
fn plan_orders<T: TestReal>(thread: usize) -> Vec<(String, u64)> {
    let shape = PLANNING_SHAPE;
    let mut fingerprints = Vec::new();
    for order in [
        Order::ColumnMajor,
        Order::RowMajor,
        Order::Subview,
        Order::Slice,
    ] {
        for policy in PolicyKind::ALL {
            let label = format!("{} {order:?} {policy:?}", T::NAME);
            let context = format!("planning while executing: {label}, thread {thread}");
            let fingerprint = match order {
                Order::ColumnMajor => {
                    let mut input = sentinel_parent::<T, 2>(shape);
                    let mut output = sentinel_parent::<T, 2>(shape);
                    plan_and_execute(policy, &mut input, &mut output, &context)
                }
                Order::RowMajor => {
                    let stride = [shape[1], 1];
                    let len = shape[0] * shape[1];
                    let mut input = aligned(&vec![sentinel::<T>(); len]);
                    let mut output = aligned(&vec![sentinel::<T>(); len]);
                    plan_and_execute(
                        policy,
                        &mut StridedSliceArrayMut::from_shape_and_stride(
                            &mut input[..],
                            shape,
                            stride,
                        ),
                        &mut StridedSliceArrayMut::from_shape_and_stride(
                            &mut output[..],
                            shape,
                            stride,
                        ),
                        &context,
                    )
                }
                Order::Subview => {
                    let mut input = sentinel_parent::<T, 2>([15, 8]);
                    let mut output = sentinel_parent::<T, 2>([14, 7]);
                    plan_and_execute(
                        policy,
                        &mut input.r_mut().into_subview([2, 2], shape),
                        &mut output.r_mut().into_subview([1, 1], shape),
                        &context,
                    )
                }
                Order::Slice => {
                    let mut input = sentinel_parent::<T, 3>([3, shape[0], shape[1]]);
                    let mut output = sentinel_parent::<T, 3>([shape[0], shape[1], 3]);
                    plan_and_execute(
                        policy,
                        &mut input.r_mut().slice::<2>(0, 1),
                        &mut output.r_mut().slice::<2>(2, 2),
                        &context,
                    )
                }
            };
            fingerprints.push((label, fingerprint));
        }
    }
    fingerprints
}

/// Plan and execute with the 0.10 module, which shares the FFTW lock, and assert the results.
fn plan_with_the_old_module() {
    for k in 0..4 {
        let shape = [16, 3 + k];
        let mut array = DynArray::<c64, 2>::from_shape(shape);
        let values = random_values::<f64>(SEED + k as u64, shape[0] * shape[1]);
        for (index, &value) in multi_indices(shape).zip(&values) {
            array[index] = value;
        }
        let fft = array.into_fft();
        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();
        let reference = logical_reference(shape, &values, &[0, 1], Direction::Forward);
        let context = format!("planning while executing: the 0.10 module, shape {shape:?}");
        assert_matches_reference(&fft.borrow().unwrap().r(), &reference, &[0, 1], &context);
    }
}

/// Counts a finished planning thread when dropped, during unwinding as well, so that the
/// executing threads stop even if a planning thread panics.
struct Finished<'a>(&'a AtomicUsize);

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn test_planning_while_executing() {
    const EXECUTORS: usize = 4;
    const PLANNERS: usize = 4;
    // At least this many executions per executing thread, and more until every planning
    // thread has finished.
    const MIN_ITERATIONS: usize = 4;

    let policy = PolicyKind::Preserving;
    let plan = AnyPlan::<f64>::unit(policy, Direction::Forward);
    let (mut input, mut output) = parents::<f64>(EXECUTORS);
    let units = units(
        &plan,
        policy,
        input.data_mut().unwrap(),
        output.data_mut().unwrap(),
        THREADS,
    );
    // The planning threads and the thread of the 0.10 module.
    let planning = PLANNERS + 1;
    let finished = AtomicUsize::new(0);
    let barrier = Barrier::new(EXECUTORS + planning);
    let fingerprints = std::thread::scope(|scope| {
        for (thread, mut unit) in units.into_iter().enumerate() {
            let (plan, barrier, finished) = (&plan, &barrier, &finished);
            scope.spawn(move || {
                barrier.wait();
                let mut iteration = 0;
                while iteration < MIN_ITERATIONS || finished.load(Ordering::SeqCst) < planning {
                    let context = Context {
                        precision: f64::NAME,
                        policy,
                        seed: unit.seed,
                        thread,
                        iteration,
                        test: "planning while executing",
                    };
                    unit.execute_and_check(plan, policy, &context);
                    iteration += 1;
                }
            });
        }
        let planners: Vec<_> = (0..PLANNERS)
            .map(|thread| {
                let (barrier, finished) = (&barrier, &finished);
                scope.spawn(move || {
                    let _finished = Finished(finished);
                    barrier.wait();
                    let mut fingerprints = plan_orders::<f64>(thread);
                    fingerprints.extend(plan_orders::<f32>(thread));
                    fingerprints
                })
            })
            .collect();
        {
            let (barrier, finished) = (&barrier, &finished);
            scope.spawn(move || {
                let _finished = Finished(finished);
                barrier.wait();
                plan_with_the_old_module();
            });
        }
        planners
            .into_iter()
            .map(|planner| planner.join().unwrap())
            .collect::<Vec<_>>()
    });

    // Plans of the same layout made in different threads are the same plan.
    let (first, others) = fingerprints.split_first().unwrap();
    for (thread, other) in others.iter().enumerate() {
        for ((label, fingerprint), (other_label, other_fingerprint)) in first.iter().zip(other) {
            assert_eq!(label, other_label);
            assert_eq!(
                fingerprint,
                other_fingerprint,
                "planning while executing: {label}: thread {} and thread 0 made different plans",
                thread + 1
            );
        }
    }
}

#[test]
fn test_descriptions_from_many_threads() {
    let plan = AnyPlan::<f64>::unit(PolicyKind::Preserving, Direction::Backward);
    let (description, fingerprint) = (plan.description(), plan.fingerprint());
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let (plan, barrier, description) = (&plan, &barrier, &description);
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(
                    &plan.description(),
                    description,
                    "descriptions from many threads: thread {thread}"
                );
                assert_eq!(
                    plan.fingerprint(),
                    fingerprint,
                    "descriptions from many threads: thread {thread}"
                );
            });
        }
    });
}

#[test]
fn test_execution_takes_no_lock() {
    // Made before the lock is held: planning takes the lock.
    let plan = AnyPlan::<f64>::unit(PolicyKind::Preserving, Direction::Forward);
    let (mut input, mut output) = parents::<f64>(1);
    let mut units = units(
        &plan,
        PolicyKind::Preserving,
        input.data_mut().unwrap(),
        output.data_mut().unwrap(),
        0,
    );
    let mut unit = units.pop().unwrap();
    let (locked, wait_for_lock) = mpsc::channel();
    let (executed, wait_for_execution) = mpsc::channel();
    let holder = std::thread::scope(|scope| {
        let holder = scope.spawn(move || {
            with_fftw(|_| {
                locked.send(()).unwrap();
                // Bounded, so that execution that takes the lock fails the test rather than
                // deadlocking it; the lock is released before the result is asserted.
                wait_for_execution.recv_timeout(Duration::from_secs(10))
            })
        });
        let plan = &plan;
        scope.spawn(move || {
            wait_for_lock
                .recv_timeout(Duration::from_secs(10))
                .expect("execution takes no lock: the lock was not acquired");
            let context = Context {
                precision: f64::NAME,
                policy: PolicyKind::Preserving,
                seed: unit.seed,
                thread: 1,
                iteration: 0,
                test: "execution takes no lock",
            };
            unit.execute_and_check(plan, PolicyKind::Preserving, &context);
            // The holder may have timed out and dropped its receiver.
            let _ = executed.send(());
        });
        holder.join().unwrap()
    });
    assert!(
        holder.is_ok(),
        "execution takes no lock: no execution within 10 s while the FFTW lock was held"
    );
}

#[test]
fn test_plans_move_between_threads() {
    let (mut input, mut output) = parents::<f64>(1);
    let values = random_values::<f64>(unit_seed(0), UNIT);
    input.data_mut().unwrap().copy_from_slice(&values);
    let plan =
        std::thread::spawn(|| AnyPlan::<f64>::unit(PolicyKind::Preserving, Direction::Forward))
            .join()
            .unwrap();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            plan.execute(
                &mut SliceArrayMut::from_shape(input.data_mut().unwrap(), UNIT_SHAPE),
                &mut SliceArrayMut::from_shape(output.data_mut().unwrap(), UNIT_SHAPE),
            );
            drop(plan);
        });
    });
    let reference = logical_reference(UNIT_SHAPE, &values, &[0], Direction::Forward);
    let output = SliceArray::from_shape(output.data().unwrap(), UNIT_SHAPE);
    assert_matches_reference(&output, &reference, &[0], &"plans move between threads");
}
