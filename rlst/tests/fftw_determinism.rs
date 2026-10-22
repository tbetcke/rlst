//! Determinism of FFTW plans across fresh processes (task F4; design Sections 5 and 7.6).
//!
//! FFTW's wisdom is global to the process, so these tests cannot run as threads of one test
//! process. This executable has `harness = false` and is both parent and child:
//!
//! - with `RLST_FFTW_DETERMINISM_CASE` set to the name of a case, it is a child: it runs that
//!   case, as the first FFTW work of its process, prints one result line per problem to
//!   standard output, and exits with status 0, or panics;
//! - without it, it is the parent: it runs the scenarios, each spawning fresh children of
//!   this executable, compares their outputs, prints a summary, and exits with status 1 if
//!   any scenario failed. It ignores its arguments, which `cargo test` passes on.
//!
//! A result line names the problem, the plan's fingerprint and a 64-bit FNV-1a hash of the
//! output's bits. Lines are compared as text, and only between processes of one machine and
//! build: fingerprints and bits depend on the FFTW build and the CPU's SIMD support. Every
//! buffer whose bits are compared is at alignment class 0, from an aligned allocation.
//!
//! Later tasks add cases to [`CASES`] and scenarios to [`SCENARIOS`].

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::io::Read;
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use num::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rlst::dense::array::{AlignedDynArray, StridedSliceArrayMut};
use rlst::fftw::{
    C2CDirection, C2CPlan, Direction, FftBufferMut, FftPlanInplace, FftwError, FftwPlanFlags,
    FftwReal, Layout, Planner, Rigor,
};
use rlst::{DynArray, SliceArray, SliceArrayMut, c32, c64};

/// The environment variable that makes the executable a child running one case.
const CASE_VARIABLE: &str = "RLST_FFTW_DETERMINISM_CASE";

/// The time a child may take before the parent kills it.
const DEADLINE: Duration = Duration::from_secs(120);

/// The rings of nd-fmm's units `[n_phi, RINGS]` (design, Section 2.5).
const RINGS: usize = 9;

// Precisions, inputs and result lines.

/// A precision of the tests.
trait Precision: FftwReal {
    /// Round an `f64` to this precision.
    fn round_from(value: f64) -> Self;
    /// Append the little-endian bytes of the value.
    fn push_bytes(self, bytes: &mut Vec<u8>);
}

impl Precision for f32 {
    fn round_from(value: f64) -> Self {
        value as f32
    }
    fn push_bytes(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_bits().to_le_bytes());
    }
}

impl Precision for f64 {
    fn round_from(value: f64) -> Self {
        value
    }
    fn push_bytes(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_bits().to_le_bytes());
    }
}

/// An array of complex values at `FFTW_ALIGNED`, so at alignment class 0 in every process.
type Aligned<T, const N: usize> = AlignedDynArray<Complex<T>, N, { rlst::FFTW_ALIGNED }>;

/// The 64-bit FNV-1a hash of `bytes`.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The hash of the bits of `values`: real and imaginary parts as little-endian bytes, in the
/// order given, which is column-major order of the output view.
fn hash<T: Precision>(values: impl IntoIterator<Item = Complex<T>>) -> u64 {
    let mut bytes = Vec::new();
    for value in values {
        value.re.push_bytes(&mut bytes);
        value.im.push_bytes(&mut bytes);
    }
    fnv1a_64(&bytes)
}

/// `len` complex values with parts uniform in [−1, 1], from ChaCha8 seeded by the hash of
/// `label`, so that a problem's input is the same wherever and whenever it runs.
fn values<T: Precision>(label: &str, len: usize) -> Vec<Complex<T>> {
    let mut rng = ChaCha8Rng::seed_from_u64(fnv1a_64(label.as_bytes()));
    (0..len)
        .map(|_| {
            let re = rng.random_range(-1.0..=1.0);
            let im = rng.random_range(-1.0..=1.0);
            Complex::new(T::round_from(re), T::round_from(im))
        })
        .collect()
}

/// A result line: the problem, the plan's fingerprint and the hash of the output's bits.
fn line<T: Precision>(
    label: &str,
    fingerprint: u64,
    output: impl IntoIterator<Item = Complex<T>>,
) -> String {
    format!(
        "{label}: fingerprint {fingerprint:016x}, output {:016x}",
        hash(output)
    )
}

/// The fingerprint of a result line.
fn fingerprint_of(line: &str) -> Option<&str> {
    line.split("fingerprint ").nth(1)?.split(',').next()
}

/// The sentinel of the padding between units.
fn sentinel() -> c64 {
    c64::new(1234.5, -678.25)
}

// Problems.

/// The input policy of a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Preserving,
    Destroying,
    InPlace,
}

/// The layout of a problem: from `input` and `output` out of place, from `input` in place.
fn layout<T, A, const N: usize>(
    policy: Policy,
    input: &rlst::Array<A, N>,
    output: &rlst::Array<A, N>,
    axes: &[usize],
) -> Layout
where
    T: Precision,
    A: rlst::StridedRawAccess<Item = Complex<T>> + rlst::Shape<N> + rlst::Stride<N>,
{
    match policy {
        Policy::InPlace => Layout::over_axes_inplace(input, axes),
        _ => Layout::over_axes(input, output, axes),
    }
    .unwrap()
}

/// Plan with `planner` and `policy`, on scratch, and execute once: from `input` into
/// `output` out of place, on `input` in place. Returns the plan's fingerprint.
fn plan_and_execute<T, I, O>(
    planner: &Planner<T>,
    policy: Policy,
    layout: &Layout,
    direction: Direction,
    input: &mut I,
    output: &mut O,
) -> u64
where
    T: Precision,
    I: FftBufferMut<Complex<T>> + ?Sized,
    O: FftBufferMut<Complex<T>> + ?Sized,
{
    match policy {
        Policy::Preserving => {
            let plan = planner.c2c(layout, direction).unwrap();
            plan.execute(&*input, output).unwrap();
            plan.fingerprint()
        }
        Policy::Destroying => {
            let plan = planner
                .clone()
                .may_destroy_input()
                .c2c(layout, direction)
                .unwrap();
            plan.execute(input, output).unwrap();
            plan.fingerprint()
        }
        Policy::InPlace => {
            let plan = planner.c2c_inplace(layout, direction).unwrap();
            plan.execute(input).unwrap();
            plan.fingerprint()
        }
    }
}

/// A problem of the `estimate` case.
#[derive(Clone, Copy, Debug)]
enum Problem {
    /// nd-fmm's unit `[n_phi, RINGS]` along axis 0.
    Unit {
        n_phi: usize,
        double: bool,
        direction: Direction,
        policy: Policy,
    },
    /// A column-major `[8, 6, 4]` over axes `[0, 1]`, f64, forward.
    ColumnMajor,
    /// A subview `[12, 7]` at `[4, 1]` of an aligned `[16, 10]`, whose offset of 20 elements
    /// is a multiple of 64 bytes, along axis 1, f64, forward.
    Subview,
    /// A view `[10, 3]` with steps of 2, strides `[2, 20]`, over aligned storage, along axis
    /// 0, f64, forward.
    Strided,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Problem::Unit {
                n_phi,
                double,
                direction,
                policy,
            } => write!(
                f,
                "unit [{n_phi}, {RINGS}] along axis 0, {policy:?}, {direction:?}, {}",
                if double { "f64" } else { "f32" }
            ),
            Problem::ColumnMajor => {
                write!(
                    f,
                    "column-major [8, 6, 4] over [0, 1], Preserving, Forward, f64"
                )
            }
            Problem::Subview => write!(
                f,
                "subview [12, 7] at [4, 1] of [16, 10] along axis 1, Preserving, Forward, f64"
            ),
            Problem::Strided => write!(
                f,
                "view [10, 3] with strides [2, 20] along axis 0, Preserving, Forward, f64"
            ),
        }
    }
}

/// nd-fmm's unit `[n_phi, RINGS]`, forward, f64, preserving: the problem of the
/// `fixed-batch-units` and `estimate-after-measure` cases.
fn unit_f64_forward(n_phi: usize) -> Problem {
    Problem::Unit {
        n_phi,
        double: true,
        direction: Direction::Forward,
        policy: Policy::Preserving,
    }
}

impl Problem {
    /// The problems of the `estimate` case.
    fn estimate() -> Vec<Problem> {
        let mut problems = Vec::new();
        for n_phi in [140, 216, 588, 1080] {
            for direction in [Direction::Forward, Direction::Backward] {
                problems.push(Problem::Unit {
                    n_phi,
                    double: true,
                    direction,
                    policy: Policy::Preserving,
                });
            }
        }
        for direction in [Direction::Forward, Direction::Backward] {
            problems.push(Problem::Unit {
                n_phi: 140,
                double: false,
                direction,
                policy: Policy::Preserving,
            });
        }
        problems.extend([Problem::ColumnMajor, Problem::Subview, Problem::Strided]);
        for policy in [Policy::Destroying, Policy::InPlace] {
            problems.push(Problem::Unit {
                n_phi: 140,
                double: true,
                direction: Direction::Forward,
                policy,
            });
        }
        problems
    }

    /// Plan the problem with `rigor`, execute it once on its seeded input, and return its
    /// result line.
    fn run(self, rigor: Rigor) -> String {
        let label = self.to_string();
        match self {
            Problem::Unit {
                n_phi,
                double: true,
                direction,
                policy,
            } => unit::<f64>(&label, n_phi, direction, policy, rigor),
            Problem::Unit {
                n_phi,
                double: false,
                direction,
                policy,
            } => unit::<f32>(&label, n_phi, direction, policy, rigor),
            Problem::ColumnMajor => {
                let shape = [8, 6, 4];
                let mut input = Aligned::<f64, 3>::from_shape(shape);
                let mut output = Aligned::<f64, 3>::from_shape(shape);
                input.fill_from_iter(values::<f64>(&label, 8 * 6 * 4).into_iter());
                let layout = Layout::over_axes(&input, &output, &[0, 1]).unwrap();
                let planner = Planner::new().rigor(rigor);
                let fingerprint = plan_and_execute(
                    &planner,
                    Policy::Preserving,
                    &layout,
                    Direction::Forward,
                    &mut input,
                    &mut output,
                );
                line(&label, fingerprint, output.iter_value())
            }
            Problem::Subview => {
                let (origin, shape) = ([4, 1], [12, 7]);
                let mut input = Aligned::<f64, 2>::from_shape([16, 10]);
                let mut output = Aligned::<f64, 2>::from_shape([16, 10]);
                let mut input = input.r_mut().into_subview(origin, shape);
                let mut output = output.r_mut().into_subview(origin, shape);
                input.fill_from_iter(values::<f64>(&label, 12 * 7).into_iter());
                let layout = Layout::along_axis(&input, &output, 1).unwrap();
                let planner = Planner::new().rigor(rigor);
                let fingerprint = plan_and_execute(
                    &planner,
                    Policy::Preserving,
                    &layout,
                    Direction::Forward,
                    &mut input,
                    &mut output,
                );
                line(&label, fingerprint, output.iter_value())
            }
            Problem::Strided => {
                let (shape, stride) = ([10, 3], [2, 20]);
                let mut input = Aligned::<f64, 1>::from_shape([60]);
                let mut output = Aligned::<f64, 1>::from_shape([60]);
                let input = input.data_mut().unwrap();
                let output = output.data_mut().unwrap();
                let mut input = StridedSliceArrayMut::from_shape_and_stride(input, shape, stride);
                let mut output = StridedSliceArrayMut::from_shape_and_stride(output, shape, stride);
                input.fill_from_iter(values::<f64>(&label, 10 * 3).into_iter());
                let layout = Layout::along_axis(&input, &output, 0).unwrap();
                let planner = Planner::new().rigor(rigor);
                let fingerprint = plan_and_execute(
                    &planner,
                    Policy::Preserving,
                    &layout,
                    Direction::Forward,
                    &mut input,
                    &mut output,
                );
                line(&label, fingerprint, output.iter_value())
            }
        }
    }
}

/// Plan nd-fmm's unit `[n_phi, RINGS]` along axis 0 and execute it once.
fn unit<T: Precision>(
    label: &str,
    n_phi: usize,
    direction: Direction,
    policy: Policy,
    rigor: Rigor,
) -> String {
    let shape = [n_phi, RINGS];
    let mut input = Aligned::<T, 2>::from_shape(shape);
    let mut output = Aligned::<T, 2>::from_shape(shape);
    input.fill_from_iter(values::<T>(label, n_phi * RINGS).into_iter());
    let layout = layout(policy, &input, &output, &[0]);
    let planner = Planner::new().rigor(rigor);
    let fingerprint = plan_and_execute(
        &planner,
        policy,
        &layout,
        direction,
        &mut input,
        &mut output,
    );
    match policy {
        Policy::InPlace => line(label, fingerprint, input.iter_value()),
        _ => line(label, fingerprint, output.iter_value()),
    }
}

// Cases, each run as the only FFTW work of a fresh child process.

/// `estimate`: the problems of [`Problem::estimate`], planned with `Estimate`.
fn case_estimate() -> Vec<String> {
    Problem::estimate()
        .into_iter()
        .map(|problem| problem.run(Rigor::Estimate))
        .collect()
}

/// Plan and execute `Estimate` plans of `T` unrelated to the problems of `estimate`: other
/// sizes, and the same sizes with other batches and layouts.
fn other_plans<T: Precision>() {
    for (shape, axes, policy) in [
        ([100, 3], &[0][..], Policy::Preserving),
        ([252, 9], &[0], Policy::Destroying),
        ([140, 5], &[0], Policy::Preserving),
        ([588, 8], &[0], Policy::InPlace),
        ([9, 1080], &[1], Policy::Preserving),
        ([216, 216], &[0, 1], Policy::Preserving),
        ([6, 140], &[1, 0], Policy::Destroying),
    ] {
        for direction in [Direction::Forward, Direction::Backward] {
            let label = format!("other {shape:?} {axes:?} {policy:?} {direction:?}");
            let mut input = Aligned::<T, 2>::from_shape(shape);
            let mut output = Aligned::<T, 2>::from_shape(shape);
            input.fill_from_iter(values::<T>(&label, shape[0] * shape[1]).into_iter());
            let layout = layout(policy, &input, &output, axes);
            plan_and_execute(
                &Planner::new(),
                policy,
                &layout,
                direction,
                &mut input,
                &mut output,
            );
        }
    }
}

/// `estimate-after-other-plans`: `Estimate` plans of unrelated problems, through the new
/// interface and the 0.10 module, and then the problems of `estimate` in reverse order. The
/// lines are printed in the order of `estimate`.
fn case_estimate_after_other_plans() -> Vec<String> {
    other_plans::<f64>();
    other_plans::<f32>();
    for shape in [[30, 7], [588, 3]] {
        let mut array = DynArray::<c64, 2>::from_shape(shape);
        array.fill_from_seed_equally_distributed(1);
        let fft = array.into_fft();
        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();
        let mut array = DynArray::<c32, 2>::from_shape(shape);
        array.fill_from_seed_equally_distributed(2);
        let fft = array.into_fft();
        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Backward)
            .unwrap()
            .execute();
    }
    let mut lines: Vec<String> = Problem::estimate()
        .into_iter()
        .rev()
        .map(|problem| problem.run(Rigor::Estimate))
        .collect();
    lines.reverse();
    lines
}

/// `fixed-batch-units`: one plan of nd-fmm's unit `[140, 9]`, the first plan of the process,
/// executed on units at different positions of larger buffers: the boxes `slice(2, k)` of a
/// parent `[140, 9, 4]`, and four units of a flat buffer with 4 complex values (64 bytes) of
/// padding between them. Every unit holds the input of the unit in `estimate`.
fn case_fixed_batch_units() -> Vec<String> {
    const N_PHI: usize = 140;
    const UNITS: usize = 4;
    const LEN: usize = N_PHI * RINGS;
    const PADDING: usize = 4;
    let shape = [N_PHI, RINGS];
    let label = unit_f64_forward(N_PHI).to_string();
    let values = values::<f64>(&label, LEN);

    let mut parent_in = Aligned::<f64, 3>::from_shape([N_PHI, RINGS, UNITS]);
    let mut parent_out = Aligned::<f64, 3>::from_shape([N_PHI, RINGS, UNITS]);
    for k in 0..UNITS {
        parent_in
            .r_mut()
            .slice::<2>(2, k)
            .fill_from_iter(values.iter().copied());
    }
    let layout = Layout::along_axis(
        &parent_in.r().slice::<2>(2, 0),
        &parent_out.r().slice::<2>(2, 0),
        0,
    )
    .unwrap();
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();

    // The same plan on a standalone aligned copy.
    let mut copy = Aligned::<f64, 2>::from_shape(shape);
    let mut copy_out = Aligned::<f64, 2>::from_shape(shape);
    copy.fill_from_iter(values.iter().copied());
    plan.execute(&copy, &mut copy_out).unwrap();
    let expected: Vec<c64> = copy_out.iter_value().collect();
    let same = |result: &[c64], what: &str| {
        assert!(
            hash(result.iter().copied()) == hash(expected.iter().copied())
                && result.iter().zip(&expected).all(|(a, b)| {
                    a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                }),
            "fixed-batch units: {what} differs, bit for bit, from the plan on a standalone copy"
        );
    };

    // The boxes of the parent.
    let mut own_out = Aligned::<f64, 3>::from_shape([N_PHI, RINGS, UNITS]);
    for k in 0..UNITS {
        plan.execute(
            &parent_in.r().slice::<2>(2, k),
            &mut parent_out.r_mut().slice::<2>(2, k),
        )
        .unwrap();
        let result: Vec<c64> = parent_out.r().slice::<2>(2, k).iter_value().collect();
        same(&result, &format!("box {k} of the parent"));

        let own = Planner::<f64>::new()
            .c2c_on(
                &mut parent_in.r_mut().slice::<2>(2, k),
                &mut own_out.r_mut().slice::<2>(2, k),
                &layout,
                Direction::Forward,
            )
            .unwrap();
        own.execute(
            &parent_in.r().slice::<2>(2, k),
            &mut own_out.r_mut().slice::<2>(2, k),
        )
        .unwrap();
        let result: Vec<c64> = own_out.r().slice::<2>(2, k).iter_value().collect();
        same(&result, &format!("a plan made for box {k} of the parent"));
    }

    // The units of a flat buffer, at element offsets k (LEN + PADDING): 1264 complex doubles
    // are a multiple of 64 bytes, so every unit is at class 0.
    let total = UNITS * LEN + (UNITS - 1) * PADDING;
    let start = |k: usize| k * (LEN + PADDING);
    let mut flat_in = Aligned::<f64, 1>::from_shape([total]);
    let mut flat_out = Aligned::<f64, 1>::from_shape([total]);
    let mut own_flat = Aligned::<f64, 1>::from_shape([total]);
    for buffer in [&mut flat_in, &mut flat_out, &mut own_flat] {
        buffer.data_mut().unwrap().fill(sentinel());
    }
    for k in 0..UNITS {
        flat_in.data_mut().unwrap()[start(k)..start(k) + LEN].copy_from_slice(&values);
    }
    for k in 0..UNITS {
        let range = start(k)..start(k) + LEN;
        let input = SliceArray::from_shape(&flat_in.data().unwrap()[range.clone()], shape);
        let mut output =
            SliceArrayMut::from_shape(&mut flat_out.data_mut().unwrap()[range.clone()], shape);
        plan.execute(&input, &mut output).unwrap();
        let result: Vec<c64> = output.iter_value().collect();
        same(&result, &format!("unit {k} of the flat buffer"));

        let mut input_mut =
            SliceArrayMut::from_shape(&mut flat_in.data_mut().unwrap()[range.clone()], shape);
        let mut own_output =
            SliceArrayMut::from_shape(&mut own_flat.data_mut().unwrap()[range.clone()], shape);
        let own = Planner::<f64>::new()
            .c2c_on(&mut input_mut, &mut own_output, &layout, Direction::Forward)
            .unwrap();
        let input = SliceArray::from_shape(&flat_in.data().unwrap()[range.clone()], shape);
        let mut own_output =
            SliceArrayMut::from_shape(&mut own_flat.data_mut().unwrap()[range], shape);
        own.execute(&input, &mut own_output).unwrap();
        let result: Vec<c64> = own_output.iter_value().collect();
        same(
            &result,
            &format!("a plan made for unit {k} of the flat buffer"),
        );
    }
    for buffer in [&flat_in, &flat_out, &own_flat] {
        let data = buffer.data().unwrap();
        for k in 0..UNITS - 1 {
            let padding = &data[start(k) + LEN..start(k + 1)];
            assert!(
                padding
                    .iter()
                    .all(|value| value.re.to_bits() == sentinel().re.to_bits()
                        && value.im.to_bits() == sentinel().im.to_bits()),
                "fixed-batch units: the padding after unit {k} changed"
            );
        }
    }

    vec![line(&label, plan.fingerprint(), expected)]
}

/// `wisdom-only-without-wisdom`: as the first planning of the process, wisdom-only planning
/// fails with `PlanCreationFailed` in both precisions; an `Estimate` plan of the same problem
/// then succeeds.
fn case_wisdom_only_without_wisdom() -> Vec<String> {
    let mut lines = Vec::new();
    let shape = [140, RINGS];
    let input = Aligned::<f64, 2>::from_shape(shape);
    let output = Aligned::<f64, 2>::from_shape(shape);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let result = Planner::<f64>::new()
        .wisdom_only()
        .c2c(&layout, Direction::Forward);
    assert!(
        matches!(result, Err(FftwError::PlanCreationFailed { .. })),
        "wisdom-only without wisdom, f64: expected PlanCreationFailed, found {result:?}"
    );
    lines.push("wisdom-only, f64: PlanCreationFailed".to_string());
    let result = Planner::<f32>::new()
        .wisdom_only()
        .c2c(&layout, Direction::Forward);
    assert!(
        matches!(result, Err(FftwError::PlanCreationFailed { .. })),
        "wisdom-only without wisdom, f32: expected PlanCreationFailed, found {result:?}"
    );
    lines.push("wisdom-only, f32: PlanCreationFailed".to_string());
    lines.push(unit_f64_forward(140).run(Rigor::Estimate));
    lines
}

/// `estimate-after-measure`: an `Estimate` plan (A), a `Measure` plan (B) and another
/// `Estimate` plan (C) of nd-fmm's unit `[588, 9]`, forward, f64, each executed once.
fn case_estimate_after_measure() -> Vec<String> {
    let problem = unit_f64_forward(588);
    [
        ("A", Rigor::Estimate),
        ("B", Rigor::Measure),
        ("C", Rigor::Estimate),
    ]
    .into_iter()
    .map(|(name, rigor)| format!("{name} {}", problem.run(rigor)))
    .collect()
}

/// A case: its name and what a child runs, printing the lines it returns.
type Case = (&'static str, fn() -> Vec<String>);

/// The cases.
const CASES: &[Case] = &[
    ("estimate", case_estimate),
    (
        "estimate-after-other-plans",
        case_estimate_after_other_plans,
    ),
    ("fixed-batch-units", case_fixed_batch_units),
    (
        "wisdom-only-without-wisdom",
        case_wisdom_only_without_wisdom,
    ),
    ("estimate-after-measure", case_estimate_after_measure),
];

// Scenarios, run by the parent.

/// The outputs of the children, by case, in the order they ran.
struct Runs(BTreeMap<&'static str, Vec<Vec<String>>>);

impl Runs {
    /// The lines of run `index` of `case`; the parent runs every case a scenario names, as
    /// often as it names it, before its check.
    fn lines(&self, case: &str, index: usize) -> &[String] {
        &self.0[case][index]
    }
}

/// A scenario: the cases it needs and how many runs of each, and the check of their outputs,
/// which returns notes for the summary or a failure.
struct Scenario {
    name: &'static str,
    runs: &'static [(&'static str, usize)],
    check: fn(&Runs) -> Result<Vec<String>, String>,
}

/// Compare two outputs line by line.
fn compare(first: &[String], second: &[String], what: &str) -> Result<Vec<String>, String> {
    if first.len() != second.len() {
        return Err(format!(
            "{what}: {} lines against {}",
            first.len(),
            second.len()
        ));
    }
    let differing: Vec<String> = first
        .iter()
        .zip(second)
        .filter(|(a, b)| a != b)
        .map(|(a, b)| format!("  {a}\n  {b}"))
        .collect();
    if differing.is_empty() {
        Ok(vec![format!("{what}: {} lines equal", first.len())])
    } else {
        Err(format!(
            "{what}: {} of {} lines differ:\n{}",
            differing.len(),
            first.len(),
            differing.join("\n")
        ))
    }
}

/// The line of `estimate`'s first run for `problem`.
fn estimate_line(runs: &Runs, problem: Problem) -> Result<&String, String> {
    let label = format!("{problem}: ");
    runs.lines("estimate", 0)
        .iter()
        .find(|line| line.starts_with(&label))
        .ok_or_else(|| format!("`estimate` has no line for {problem}"))
}

fn check_fresh_processes(runs: &Runs) -> Result<Vec<String>, String> {
    compare(
        runs.lines("estimate", 0),
        runs.lines("estimate", 1),
        "two fresh processes",
    )
}

fn check_planning_history(runs: &Runs) -> Result<Vec<String>, String> {
    // DC22: a failure is raised as a design question, not relaxed.
    compare(
        runs.lines("estimate", 0),
        runs.lines("estimate-after-other-plans", 0),
        "a fresh process against one with earlier unrelated plans",
    )
}

fn check_fixed_batch_units(runs: &Runs) -> Result<Vec<String>, String> {
    let mut notes = compare(
        runs.lines("fixed-batch-units", 0),
        runs.lines("fixed-batch-units", 1),
        "two fresh processes",
    )?;
    let line = &runs.lines("fixed-batch-units", 0)[0];
    let estimate = estimate_line(runs, unit_f64_forward(140))?;
    if fingerprint_of(line) != fingerprint_of(estimate) {
        return Err(format!(
            "the plan of the units differs from the plan of the unit in `estimate`:\n  {line}\n  {estimate}"
        ));
    }
    notes.push("the fingerprint equals that of the unit in `estimate`".to_string());
    // The same plan on the same input at the same class (design, Section 5 rule 3).
    if line != estimate {
        return Err(format!(
            "the units' output differs from that of the unit in `estimate`:\n  {line}\n  {estimate}"
        ));
    }
    notes.push("the output equals that of the unit in `estimate`".to_string());
    Ok(notes)
}

fn check_wisdom_only(_: &Runs) -> Result<Vec<String>, String> {
    Ok(vec!["the child's assertions hold".to_string()])
}

fn check_estimate_after_measure(runs: &Runs) -> Result<Vec<String>, String> {
    let lines = runs.lines("estimate-after-measure", 0);
    let fingerprint = |name: &str| {
        lines
            .iter()
            .find(|line| line.starts_with(&format!("{name} ")))
            .and_then(|line| fingerprint_of(line))
            .ok_or_else(|| format!("no line for plan {name}"))
    };
    let (a, b, c) = (fingerprint("A")?, fingerprint("B")?, fingerprint("C")?);
    let estimate = estimate_line(runs, unit_f64_forward(588))?;
    if Some(a) != fingerprint_of(estimate) {
        return Err(format!(
            "plan A differs from the plan of the same problem in `estimate`:\n  A {a}\n  {estimate}"
        ));
    }
    // Section 3 item 4: reported, not asserted.
    Ok(vec![
        format!("A (estimate) {a}, equal to `estimate`"),
        format!("B (measure)  {b}"),
        format!("C (estimate) {c}"),
        format!("C equals A: {}", c == a),
        format!("C equals B: {}", c == b),
    ])
}

/// The scenarios.
const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "fresh processes agree",
        runs: &[("estimate", 2)],
        check: check_fresh_processes,
    },
    Scenario {
        name: "planning history",
        runs: &[("estimate", 1), ("estimate-after-other-plans", 1)],
        check: check_planning_history,
    },
    Scenario {
        name: "fixed-batch units",
        runs: &[("estimate", 1), ("fixed-batch-units", 2)],
        check: check_fixed_batch_units,
    },
    Scenario {
        name: "wisdom-only",
        runs: &[("wisdom-only-without-wisdom", 1)],
        check: check_wisdom_only,
    },
    Scenario {
        name: "estimate after measure",
        runs: &[("estimate", 1), ("estimate-after-measure", 1)],
        check: check_estimate_after_measure,
    },
];

// Children.

/// How a child ended.
struct Child {
    status: Option<ExitStatus>,
    stdout: String,
    stderr: String,
    elapsed: Duration,
}

impl Child {
    fn succeeded(&self) -> bool {
        self.status.is_some_and(|status| status.success())
    }

    fn describe_failure(&self) -> String {
        let ended = match self.status {
            Some(status) => format!("exited with {status}"),
            None => format!("was killed at its deadline of {} s", DEADLINE.as_secs()),
        };
        format!("{ended}; standard error:\n{}", self.stderr.trim_end())
    }
}

/// Run `case` in a fresh child process of this executable, with a deadline.
fn spawn(case: &str) -> Child {
    let start = Instant::now();
    let mut child = Command::new(env::current_exe().expect("the path of this executable"))
        .env(CASE_VARIABLE, case)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("a child process");
    // Read both pipes while the child runs, so that a full pipe cannot block it.
    let read = |mut pipe: Box<dyn Read + Send>| {
        thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            text
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let status = loop {
        if let Some(status) = child.try_wait().expect("the child's status") {
            break Some(status);
        }
        if start.elapsed() > DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    Child {
        status,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
        elapsed: start.elapsed(),
    }
}

/// Run the scenarios; print the children's outputs, every comparison and every failure.
fn parent() -> ExitCode {
    let start = Instant::now();
    let mut children: BTreeMap<&'static str, Vec<Child>> = BTreeMap::new();
    let mut failures = Vec::new();
    for scenario in SCENARIOS {
        for &(case, count) in scenario.runs {
            let runs = children.entry(case).or_default();
            while runs.len() < count {
                let child = spawn(case);
                println!(
                    "child `{case}` #{} ({:.2} s){}",
                    runs.len() + 1,
                    child.elapsed.as_secs_f64(),
                    if child.succeeded() { "" } else { ": FAILED" }
                );
                for line in child.stdout.lines() {
                    println!("    {line}");
                }
                runs.push(child);
            }
        }
        let failed: Vec<String> = scenario
            .runs
            .iter()
            .flat_map(|&(case, count)| {
                children[case][..count]
                    .iter()
                    .enumerate()
                    .filter(|(_, child)| !child.succeeded())
                    .map(move |(index, child)| {
                        format!("child `{case}` #{} {}", index + 1, child.describe_failure())
                    })
            })
            .collect();
        let result = if failed.is_empty() {
            let runs = Runs(
                children
                    .iter()
                    .map(|(&case, runs)| {
                        let lines = runs
                            .iter()
                            .map(|child| child.stdout.lines().map(str::to_string).collect())
                            .collect();
                        (case, lines)
                    })
                    .collect(),
            );
            (scenario.check)(&runs)
        } else {
            Err(failed.join("\n"))
        };
        match result {
            Ok(notes) => {
                println!("scenario `{}`: ok", scenario.name);
                for note in notes {
                    println!("    {note}");
                }
            }
            Err(failure) => {
                println!("scenario `{}`: FAILED", scenario.name);
                failures.push((scenario.name, failure));
            }
        }
    }
    println!(
        "fftw_determinism: {} scenarios, {} failed, {:.2} s",
        SCENARIOS.len(),
        failures.len(),
        start.elapsed().as_secs_f64()
    );
    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        for (name, failure) in failures {
            eprintln!("scenario `{name}` failed:\n{failure}\n");
        }
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    // `cargo test` passes filters and options such as `--nocapture`; they are ignored.
    let Ok(case) = env::var(CASE_VARIABLE) else {
        return parent();
    };
    let (_, run) = CASES
        .iter()
        .find(|(name, _)| *name == case)
        .unwrap_or_else(|| panic!("unknown case `{case}`"));
    for line in run() {
        println!("{line}");
    }
    ExitCode::SUCCESS
}
