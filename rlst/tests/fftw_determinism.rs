//! Determinism of FFTW plans across fresh processes, FFTW's wisdom, and threaded plans (tasks
//! F4, F3, F5a, F5b, F6a and F6b; design Sections 5, 7.5 and 7.6).
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
//! output's bits; a real output is hashed as complex values with imaginary part 0. Lines are compared as text, and only between processes of one machine and
//! build: fingerprints and bits depend on the FFTW build and the CPU's SIMD support. Every
//! buffer whose bits are compared is at alignment class 0, from an aligned allocation.
//!
//! Later tasks add cases to [`CASES`] and scenarios to [`SCENARIOS`]. The cases and scenarios
//! of threaded plans, in [`THREADED_CASES`] and [`THREADED_SCENARIOS`], run with
//! `fftw_threads` only.

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::io::Read;
#[cfg(feature = "fftw_threads")]
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use num::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rlst::dense::array::{AlignedDynArray, StridedSliceArrayMut};
use rlst::fftw::{
    Direction, FftBufferMut, FftwError, FftwReal, Layout, Planner, R2rKind, Rigor, as_reals,
    as_reals_mut, wisdom,
};
use rlst::{SliceArray, SliceArrayMut, c64};

/// The environment variable that makes the executable a child running one case.
const CASE_VARIABLE: &str = "RLST_FFTW_DETERMINISM_CASE";

/// The time a child may take before the parent kills it.
const DEADLINE: Duration = Duration::from_secs(120);

/// The rings of nd-fmm's units `[n_phi, RINGS]` (design, Section 2.5).
const RINGS: usize = 9;

/// The options a problem is planned with: a rigor and, with `fftw_threads`, a thread count.
#[derive(Clone, Copy, Debug)]
struct Options {
    rigor: Rigor,
    threads: usize,
}

impl From<Rigor> for Options {
    /// Single-threaded plans of `rigor`.
    fn from(rigor: Rigor) -> Self {
        Options { rigor, threads: 1 }
    }
}

impl Options {
    /// A planner with these options.
    fn planner<T: FftwReal>(self) -> Planner<T> {
        let planner = Planner::new().rigor(self.rigor);
        #[cfg(feature = "fftw_threads")]
        let planner = planner.threads(NonZeroUsize::new(self.threads).unwrap());
        assert!(
            cfg!(feature = "fftw_threads") || self.threads == 1,
            "threads need `fftw_threads`"
        );
        planner
    }

    /// The label of `problem` planned with these options: the problem, and the thread count
    /// if it is not 1.
    fn label(self, problem: impl fmt::Display) -> String {
        match self.threads {
            1 => problem.to_string(),
            threads => format!("{problem}, {threads} threads"),
        }
    }
}

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

/// An array of reals at `FFTW_ALIGNED`, for split storage.
type AlignedReal<T, const N: usize> = AlignedDynArray<T, N, { rlst::FFTW_ALIGNED }>;

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

// Split boxes (task F3): nd-fmm's box `[n_phi, RINGS]` as a real array with a part axis of
// extent 2, the planes `[n_phi, RINGS, 2]` (part axis 2) or the interleaved memory
// `[2, n_phi, RINGS]` (part axis 0).

/// The index of a split box with part axis `part` (2 or 0) for logical index `[i, j]` and
/// part `p`.
fn split_index(part: usize, [i, j]: [usize; 2], p: usize) -> [usize; 3] {
    if part == 0 { [p, i, j] } else { [i, j, p] }
}

/// The shape of a split box with part axis `part`.
fn split_shape(part: usize, n_phi: usize) -> [usize; 3] {
    split_index(part, [n_phi, RINGS], 2)
}

/// Write `values`, in column-major order of the logical shape `[n_phi, RINGS]`, into the two
/// parts of a split box.
fn fill_split<T, A>(view: &mut rlst::Array<A, 3>, part: usize, values: &[Complex<T>])
where
    T: Precision,
    A: rlst::RandomAccessMut<3> + rlst::BaseItem<Item = T>,
{
    let n_phi = values.len() / RINGS;
    for (k, value) in values.iter().enumerate() {
        let index = [k % n_phi, k / n_phi];
        *view.get_mut(split_index(part, index, 0)).unwrap() = value.re;
        *view.get_mut(split_index(part, index, 1)).unwrap() = value.im;
    }
}

/// The complex values of a split box, in column-major order of its logical shape.
fn split_values<T, A>(view: &rlst::Array<A, 3>, part: usize, n_phi: usize) -> Vec<Complex<T>>
where
    T: Precision,
    A: rlst::RandomAccessByRef<3> + rlst::BaseItem<Item = T>,
{
    (0..n_phi * RINGS)
        .map(|k| {
            let index = [k % n_phi, k / n_phi];
            Complex::new(
                *view.get(split_index(part, index, 0)).unwrap(),
                *view.get(split_index(part, index, 1)).unwrap(),
            )
        })
        .collect()
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

/// Plan a split c2c transform with `planner` and `policy`, on scratch, and execute once: from
/// `input` into `output` out of place, on `input` in place. Returns the plan's fingerprint.
fn plan_and_execute_split<T, I, O>(
    planner: &Planner<T>,
    policy: Policy,
    layout: &Layout,
    direction: Direction,
    input: &mut I,
    output: &mut O,
) -> u64
where
    T: Precision,
    I: FftBufferMut<T> + ?Sized,
    O: FftBufferMut<T> + ?Sized,
{
    match policy {
        Policy::Preserving => {
            let plan = planner.split_c2c(layout, direction).unwrap();
            plan.execute(&*input, output).unwrap();
            plan.fingerprint()
        }
        Policy::Destroying => {
            let plan = planner
                .clone()
                .may_destroy_input()
                .split_c2c(layout, direction)
                .unwrap();
            plan.execute(input, output).unwrap();
            plan.fingerprint()
        }
        Policy::InPlace => {
            let plan = planner.split_c2c_inplace(layout, direction).unwrap();
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
    /// nd-fmm's box `[n_phi, RINGS]` on split storage along logical axis 0, `Preserving`: the
    /// planes `[n_phi, RINGS, 2]` with part axis 2, or with `innermost` the interleaved
    /// memory `[2, n_phi, RINGS]` with part axis 0.
    Split {
        n_phi: usize,
        double: bool,
        direction: Direction,
        innermost: bool,
    },
    /// nd-fmm's unit `[140, RINGS]` as reals, along axis 0, f64, with the default policy: the
    /// real-to-complex transform into `[71, RINGS]` (`Preserving`), or with `r2c` false the
    /// complex-to-real transform back (`Destroying`).
    Real { r2c: bool },
    /// A real-to-real transform, f64, `Preserving`: `Redft10` along axis 0 of nd-fmm's unit
    /// `[140, RINGS]` as reals, or with `mixed` the kinds `Redft01` and `Rodft00` over both
    /// axes of `[17, RINGS]`.
    R2r { mixed: bool },
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
            Problem::Split {
                n_phi,
                double,
                direction,
                innermost,
            } => {
                let part = if innermost { 0 } else { 2 };
                write!(
                    f,
                    "split box {:?}, part axis {part}, along logical axis 0, Preserving, \
                     {direction:?}, {}",
                    split_shape(part, n_phi),
                    if double { "f64" } else { "f32" }
                )
            }
            Problem::Real { r2c: true } => write!(
                f,
                "real [140, {RINGS}] into complex [71, {RINGS}] along axis 0, r2c, Preserving, \
                 f64"
            ),
            Problem::Real { r2c: false } => write!(
                f,
                "complex [71, {RINGS}] into real [140, {RINGS}] along axis 0, c2r, Destroying, \
                 f64"
            ),
            Problem::R2r { mixed: false } => write!(
                f,
                "real [140, {RINGS}] along axis 0, r2r REDFT10, Preserving, f64"
            ),
            Problem::R2r { mixed: true } => write!(
                f,
                "real [17, {RINGS}] over [0, 1], r2r REDFT01 and RODFT00, Preserving, f64"
            ),
        }
    }
}

/// nd-fmm's box as split planes `[n_phi, RINGS, 2]`, forward, f64, preserving: the problem of
/// the `split-fixed-batch-units` case.
fn split_box_f64_forward(n_phi: usize) -> Problem {
    Problem::Split {
        n_phi,
        double: true,
        direction: Direction::Forward,
        innermost: false,
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
        // Split storage (task F3).
        for direction in [Direction::Forward, Direction::Backward] {
            problems.push(Problem::Split {
                n_phi: 140,
                double: true,
                direction,
                innermost: false,
            });
        }
        problems.push(split_box_f64_forward(588));
        problems.push(Problem::Split {
            n_phi: 140,
            double: false,
            direction: Direction::Forward,
            innermost: false,
        });
        problems.push(Problem::Split {
            n_phi: 140,
            double: true,
            direction: Direction::Forward,
            innermost: true,
        });
        // Real-to-complex and complex-to-real (task F5a).
        problems.extend([Problem::Real { r2c: true }, Problem::Real { r2c: false }]);
        // Real-to-real (task F5b).
        problems.extend([Problem::R2r { mixed: false }, Problem::R2r { mixed: true }]);
        problems
    }

    /// Plan the problem with `options` (a rigor, or a rigor and a thread count), execute it
    /// once on its seeded input, and return its result line.
    fn run(self, options: impl Into<Options>) -> String {
        let options = options.into();
        let label = options.label(self);
        match self {
            Problem::Unit {
                n_phi,
                double: true,
                direction,
                policy,
            } => unit::<f64>(&label, n_phi, direction, policy, &options.planner()),
            Problem::Unit {
                n_phi,
                double: false,
                direction,
                policy,
            } => unit::<f32>(&label, n_phi, direction, policy, &options.planner()),
            Problem::ColumnMajor => {
                let shape = [8, 6, 4];
                let mut input = Aligned::<f64, 3>::from_shape(shape);
                let mut output = Aligned::<f64, 3>::from_shape(shape);
                input.fill_from_iter(values::<f64>(&label, 8 * 6 * 4).into_iter());
                let layout = Layout::over_axes(&input, &output, &[0, 1]).unwrap();
                let planner = options.planner();
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
                let planner = options.planner();
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
                let planner = options.planner();
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
            Problem::Split {
                n_phi,
                double: true,
                direction,
                innermost,
            } => split::<f64>(&label, n_phi, direction, innermost, options),
            Problem::Split {
                n_phi,
                double: false,
                direction,
                innermost,
            } => split::<f32>(&label, n_phi, direction, innermost, options),
            Problem::Real { r2c } => {
                let (fingerprint, output) =
                    real_transform::<f64>(&label, [140, RINGS], &[0], r2c, false, options);
                line(&label, fingerprint, output)
            }
            Problem::R2r { mixed: false } => {
                let (fingerprint, output) = r2r_transform::<f64>(
                    &label,
                    [140, RINGS],
                    &[0],
                    &[R2rKind::Redft10],
                    false,
                    options,
                );
                line(&label, fingerprint, output)
            }
            Problem::R2r { mixed: true } => {
                let (fingerprint, output) = r2r_transform::<f64>(
                    &label,
                    [17, RINGS],
                    &[0, 1],
                    &[R2rKind::Redft01, R2rKind::Rodft00],
                    false,
                    options,
                );
                line(&label, fingerprint, output)
            }
        }
    }

    /// As [`Problem::run`], planning from wisdom only, for nd-fmm's units: the line equals
    /// that of `run` when the wisdom holds the plan `run` made with `rigor`.
    fn run_from_wisdom(self, rigor: Rigor) -> String {
        let label = self.to_string();
        let Problem::Unit {
            n_phi,
            double,
            direction,
            policy,
        } = self
        else {
            panic!("`run_from_wisdom` plans nd-fmm's units only, not {self}");
        };
        if double {
            let planner = Planner::<f64>::new().rigor(rigor).wisdom_only();
            unit(&label, n_phi, direction, policy, &planner)
        } else {
            let planner = Planner::<f32>::new().rigor(rigor).wisdom_only();
            unit(&label, n_phi, direction, policy, &planner)
        }
    }
}

/// Plan, with `options` and on scratch, a real-to-real transform over `axes` of `shape` with
/// `kinds`, preserving out of place or in place, and execute it once on its seeded input.
/// Returns the plan's fingerprint and the output as complex values with imaginary part 0.
fn r2r_transform<T: Precision>(
    label: &str,
    shape: [usize; 2],
    axes: &[usize],
    kinds: &[R2rKind],
    in_place: bool,
    options: Options,
) -> (u64, Vec<Complex<T>>) {
    let planner = options.planner::<T>();
    let zero = T::round_from(0.0);
    let mut input = AlignedReal::<T, 2>::from_shape(shape);
    input.fill_from_iter(
        values::<T>(label, shape[0] * shape[1])
            .into_iter()
            .map(|v| v.re),
    );
    let (fingerprint, output) = if in_place {
        let layout = Layout::over_axes_inplace(&input, axes).unwrap();
        let plan = planner.r2r_inplace(&layout, kinds).unwrap();
        plan.execute(&mut input).unwrap();
        (plan.fingerprint(), input)
    } else {
        let mut output = AlignedReal::<T, 2>::from_shape(shape);
        let layout = Layout::over_axes(&input, &output, axes).unwrap();
        let plan = planner.r2r(&layout, kinds).unwrap();
        plan.execute(&input, &mut output).unwrap();
        (plan.fingerprint(), output)
    };
    let output = output.iter_value().map(|x| Complex::new(x, zero)).collect();
    (fingerprint, output)
}

/// Plan, with `options` and on scratch, a real-to-complex (`r2c`) or complex-to-real transform
/// over `axes` of the real shape `shape`, with the default policy out of place or in place,
/// and execute it once on its seeded input. Returns the plan's fingerprint and the output,
/// reals as complex values with imaginary part 0. In place, the halved axis must be axis 0.
fn real_transform<T: Precision>(
    label: &str,
    shape: [usize; 2],
    axes: &[usize],
    r2c: bool,
    in_place: bool,
    options: Options,
) -> (u64, Vec<Complex<T>>) {
    let halved = *axes.last().unwrap();
    let mut complex_shape = shape;
    complex_shape[halved] = shape[halved] / 2 + 1;
    let planner = options.planner::<T>();
    let zero = T::round_from(0.0);
    let real_values = || {
        values::<T>(label, shape[0] * shape[1])
            .into_iter()
            .map(|v| v.re)
    };
    let complex_values = || values::<T>(label, complex_shape[0] * complex_shape[1]);
    let mut complex = Aligned::<T, 2>::from_shape(complex_shape);
    if in_place {
        assert_eq!(halved, 0, "{label}: in place, axis 0 is halved");
        if r2c {
            // Column j of the input is the reals 2 h j to 2 h j + n of the complex storage.
            let (n, h) = (shape[0], complex_shape[0]);
            let input: Vec<T> = real_values().collect();
            let reals = as_reals_mut(complex.data_mut().unwrap());
            for (column, values) in input.chunks_exact(n).enumerate() {
                reals[2 * h * column..2 * h * column + n].copy_from_slice(values);
            }
            let layout = Layout::r2c_inplace(&complex, axes, n).unwrap();
            let plan = planner.r2c_inplace(&layout).unwrap();
            plan.execute(&mut complex).unwrap();
            (plan.fingerprint(), complex.iter_value().collect())
        } else {
            complex.fill_from_iter(complex_values().into_iter());
            let layout = Layout::c2r_inplace(&complex, axes, shape[0]).unwrap();
            let plan = planner.c2r_inplace(&layout).unwrap();
            plan.execute(&mut complex).unwrap();
            let output = as_reals(complex.data().unwrap())
                .iter()
                .map(|&x| Complex::new(x, zero))
                .collect();
            (plan.fingerprint(), output)
        }
    } else {
        let mut real = AlignedReal::<T, 2>::from_shape(shape);
        if r2c {
            real.fill_from_iter(real_values());
            let layout = Layout::over_axes(&real, &complex, axes).unwrap();
            let plan = planner.r2c(&layout).unwrap();
            plan.execute(&real, &mut complex).unwrap();
            (plan.fingerprint(), complex.iter_value().collect())
        } else {
            complex.fill_from_iter(complex_values().into_iter());
            let layout = Layout::over_axes(&complex, &real, axes).unwrap();
            let plan = planner.c2r(&layout).unwrap();
            plan.execute(&mut complex, &mut real).unwrap();
            let output = real.iter_value().map(|x| Complex::new(x, zero)).collect();
            (plan.fingerprint(), output)
        }
    }
}

/// Plan nd-fmm's box `[n_phi, RINGS]` on split storage along logical axis 0, preserving, and
/// execute it once.
fn split<T: Precision>(
    label: &str,
    n_phi: usize,
    direction: Direction,
    innermost: bool,
    options: Options,
) -> String {
    let part = if innermost { 0 } else { 2 };
    let mut input = AlignedReal::<T, 3>::from_shape(split_shape(part, n_phi));
    let mut output = AlignedReal::<T, 3>::from_shape(split_shape(part, n_phi));
    fill_split(&mut input, part, &values::<T>(label, n_phi * RINGS));
    let layout = Layout::split_along_axis(&input, part, &output, part, 0).unwrap();
    let fingerprint = plan_and_execute_split(
        &options.planner(),
        Policy::Preserving,
        &layout,
        direction,
        &mut input,
        &mut output,
    );
    line(label, fingerprint, split_values(&output, part, n_phi))
}

/// Plan nd-fmm's unit `[n_phi, RINGS]` along axis 0 with `planner` and execute it once.
fn unit<T: Precision>(
    label: &str,
    n_phi: usize,
    direction: Direction,
    policy: Policy,
    planner: &Planner<T>,
) -> String {
    let shape = [n_phi, RINGS];
    let mut input = Aligned::<T, 2>::from_shape(shape);
    let mut output = Aligned::<T, 2>::from_shape(shape);
    input.fill_from_iter(values::<T>(label, n_phi * RINGS).into_iter());
    let layout = layout(policy, &input, &output, &[0]);
    let fingerprint =
        plan_and_execute(planner, policy, &layout, direction, &mut input, &mut output);
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
    // Split plans of other sizes and policies (task F3).
    for (n_phi, policy) in [
        (100, Policy::Preserving),
        (252, Policy::Destroying),
        (216, Policy::Preserving),
        (588, Policy::InPlace),
    ] {
        for direction in [Direction::Forward, Direction::Backward] {
            let label = format!("other split [{n_phi}, {RINGS}, 2] {policy:?} {direction:?}");
            let mut input = AlignedReal::<T, 3>::from_shape(split_shape(2, n_phi));
            let mut output = AlignedReal::<T, 3>::from_shape(split_shape(2, n_phi));
            fill_split(&mut input, 2, &values::<T>(&label, n_phi * RINGS));
            let layout = match policy {
                Policy::InPlace => Layout::split_along_axis_inplace(&input, 2, 0),
                _ => Layout::split_along_axis(&input, 2, &output, 2, 0),
            }
            .unwrap();
            plan_and_execute_split(
                &Planner::new(),
                policy,
                &layout,
                direction,
                &mut input,
                &mut output,
            );
        }
    }
    // Real-to-complex and complex-to-real plans of other sizes and placements (task F5a).
    for (shape, axes, in_place) in [
        ([100, 3], &[0][..], false),
        ([252, 9], &[0], true),
        ([140, 5], &[0], false),
        ([216, 216], &[0, 1], false),
        ([9, 1080], &[1], false),
    ] {
        for r2c in [true, false] {
            let label = format!("other real {shape:?} {axes:?} r2c {r2c} in place {in_place}");
            real_transform::<T>(&label, shape, axes, r2c, in_place, Rigor::Estimate.into());
        }
    }
    // Real-to-real plans of other sizes, kinds and placements (task F5b).
    for (shape, axes, kinds, in_place) in [
        ([100, 3], &[0][..], &[R2rKind::Redft10][..], false),
        ([140, 5], &[0], &[R2rKind::Redft01], true),
        (
            [17, 9],
            &[1, 0],
            &[R2rKind::Rodft00, R2rKind::Redft01],
            false,
        ),
        ([33, 9], &[0, 1], &[R2rKind::Redft00, R2rKind::Dht], true),
        ([140, 9], &[0], &[R2rKind::R2hc], false),
    ] {
        let label = format!("other r2r {shape:?} {axes:?} {kinds:?} in place {in_place}");
        r2r_transform::<T>(&label, shape, axes, kinds, in_place, Rigor::Estimate.into());
    }
}

/// `estimate-after-other-plans`: `Estimate` plans of unrelated problems, and then the
/// problems of `estimate` in reverse order. The lines are printed in the order of `estimate`.
fn case_estimate_after_other_plans() -> Vec<String> {
    other_plans::<f64>();
    other_plans::<f32>();
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

/// `split-fixed-batch-units`: one split plan of nd-fmm's box `[140, 9, 2]`, the first plan of
/// the process, executed on boxes at different positions of larger buffers: the boxes
/// `slice(3, k)` of a parent `[140, 9, 2, 4]`, and four boxes of a flat buffer with 8 reals
/// (64 bytes) of padding between them. Every box holds the input of the split box in
/// `estimate`.
fn case_split_fixed_batch_units() -> Vec<String> {
    const N_PHI: usize = 140;
    const UNITS: usize = 4;
    const LEN: usize = 2 * N_PHI * RINGS;
    const PADDING: usize = 8;
    let shape = split_shape(2, N_PHI);
    let label = split_box_f64_forward(N_PHI).to_string();
    let values = values::<f64>(&label, N_PHI * RINGS);

    let mut parent_in = AlignedReal::<f64, 4>::from_shape([N_PHI, RINGS, 2, UNITS]);
    let mut parent_out = AlignedReal::<f64, 4>::from_shape([N_PHI, RINGS, 2, UNITS]);
    for k in 0..UNITS {
        fill_split(&mut parent_in.r_mut().slice::<3>(3, k), 2, &values);
    }
    let layout = Layout::split_along_axis(
        &parent_in.r().slice::<3>(3, 0),
        2,
        &parent_out.r().slice::<3>(3, 0),
        2,
        0,
    )
    .unwrap();
    let plan = Planner::<f64>::new()
        .split_c2c(&layout, Direction::Forward)
        .unwrap();

    // The same plan on a standalone aligned copy.
    let mut copy = AlignedReal::<f64, 3>::from_shape(shape);
    let mut copy_out = AlignedReal::<f64, 3>::from_shape(shape);
    fill_split(&mut copy, 2, &values);
    plan.execute(&copy, &mut copy_out).unwrap();
    let expected: Vec<f64> = copy_out.iter_value().collect();
    let same = |result: &[f64], what: &str| {
        assert!(
            result.len() == expected.len()
                && result
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "split fixed-batch units: {what} differs, bit for bit, from the plan on a standalone \
             copy"
        );
    };

    // The boxes of the parent.
    let mut own_out = AlignedReal::<f64, 4>::from_shape([N_PHI, RINGS, 2, UNITS]);
    for k in 0..UNITS {
        plan.execute(
            &parent_in.r().slice::<3>(3, k),
            &mut parent_out.r_mut().slice::<3>(3, k),
        )
        .unwrap();
        let result: Vec<f64> = parent_out.r().slice::<3>(3, k).iter_value().collect();
        same(&result, &format!("box {k} of the parent"));

        let own = Planner::<f64>::new()
            .split_c2c_on(
                &mut parent_in.r_mut().slice::<3>(3, k),
                &mut own_out.r_mut().slice::<3>(3, k),
                &layout,
                Direction::Forward,
            )
            .unwrap();
        own.execute(
            &parent_in.r().slice::<3>(3, k),
            &mut own_out.r_mut().slice::<3>(3, k),
        )
        .unwrap();
        let result: Vec<f64> = own_out.r().slice::<3>(3, k).iter_value().collect();
        same(&result, &format!("a plan made for box {k} of the parent"));
    }

    // The boxes of a flat buffer, at real offsets k (LEN + PADDING): 2528 doubles are a
    // multiple of 64 bytes, so every box is at class 0.
    let total = UNITS * LEN + (UNITS - 1) * PADDING;
    let start = |k: usize| k * (LEN + PADDING);
    let mut flat_in = AlignedReal::<f64, 1>::from_shape([total]);
    let mut flat_out = AlignedReal::<f64, 1>::from_shape([total]);
    let mut own_flat = AlignedReal::<f64, 1>::from_shape([total]);
    for buffer in [&mut flat_in, &mut flat_out, &mut own_flat] {
        buffer.data_mut().unwrap().fill(sentinel().re);
    }
    for k in 0..UNITS {
        let range = start(k)..start(k) + LEN;
        fill_split(
            &mut SliceArrayMut::from_shape(&mut flat_in.data_mut().unwrap()[range], shape),
            2,
            &values,
        );
    }
    for k in 0..UNITS {
        let range = start(k)..start(k) + LEN;
        let input = SliceArray::from_shape(&flat_in.data().unwrap()[range.clone()], shape);
        let mut output =
            SliceArrayMut::from_shape(&mut flat_out.data_mut().unwrap()[range.clone()], shape);
        plan.execute(&input, &mut output).unwrap();
        let result: Vec<f64> = output.iter_value().collect();
        same(&result, &format!("box {k} of the flat buffer"));

        let mut input_mut =
            SliceArrayMut::from_shape(&mut flat_in.data_mut().unwrap()[range.clone()], shape);
        let mut own_output =
            SliceArrayMut::from_shape(&mut own_flat.data_mut().unwrap()[range.clone()], shape);
        let own = Planner::<f64>::new()
            .split_c2c_on(&mut input_mut, &mut own_output, &layout, Direction::Forward)
            .unwrap();
        let input = SliceArray::from_shape(&flat_in.data().unwrap()[range.clone()], shape);
        let mut own_output =
            SliceArrayMut::from_shape(&mut own_flat.data_mut().unwrap()[range], shape);
        own.execute(&input, &mut own_output).unwrap();
        let result: Vec<f64> = own_output.iter_value().collect();
        same(
            &result,
            &format!("a plan made for box {k} of the flat buffer"),
        );
    }
    for buffer in [&flat_in, &flat_out, &own_flat] {
        let data = buffer.data().unwrap();
        for k in 0..UNITS - 1 {
            let padding = &data[start(k) + LEN..start(k + 1)];
            assert!(
                padding
                    .iter()
                    .all(|value| value.to_bits() == sentinel().re.to_bits()),
                "split fixed-batch units: the padding after box {k} changed"
            );
        }
    }

    vec![line(
        &label,
        plan.fingerprint(),
        split_values(&copy_out, 2, N_PHI),
    )]
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

// Wisdom (task F6a).

/// The problems of the wisdom cases: nd-fmm's unit `[588, RINGS]`, forward, f64, and
/// `[140, RINGS]`, forward, f32, both `Preserving`.
fn wisdom_problems() -> [Problem; 2] {
    [
        unit_f64_forward(588),
        Problem::Unit {
            n_phi: 140,
            double: false,
            direction: Direction::Forward,
            policy: Policy::Preserving,
        },
    ]
}

/// The outcome of a wisdom-only `Measure` plan of `problem`, one of [`wisdom_problems`],
/// without executing it.
fn plan_from_wisdom(problem: Problem) -> Result<(), FftwError> {
    let Problem::Unit {
        n_phi,
        double,
        direction,
        ..
    } = problem
    else {
        unreachable!("the wisdom problems are units");
    };
    let shape = [n_phi, RINGS];
    if double {
        let input = Aligned::<f64, 2>::from_shape(shape);
        let output = Aligned::<f64, 2>::from_shape(shape);
        let layout = Layout::along_axis(&input, &output, 0).unwrap();
        let planner = Planner::<f64>::new().rigor(Rigor::Measure).wisdom_only();
        planner.c2c(&layout, direction).map(drop)
    } else {
        let input = Aligned::<f32, 2>::from_shape(shape);
        let output = Aligned::<f32, 2>::from_shape(shape);
        let layout = Layout::along_axis(&input, &output, 0).unwrap();
        let planner = Planner::<f32>::new().rigor(Rigor::Measure).wisdom_only();
        planner.c2c(&layout, direction).map(drop)
    }
}

/// `wisdom-round-trip` (design Section 7.6): `Measure` plans A of the wisdom problems; their
/// wisdom exported and forgotten, after which wisdom-only plans fail; the wisdom imported
/// again, after which wisdom-only plans B have the lines of A. Also: f64 wisdom is refused as
/// f32, and whether a failed import of text that is not wisdom left the wisdom unchanged is
/// reported.
fn case_wisdom_round_trip() -> Vec<String> {
    let problems = wisdom_problems();
    let measured: Vec<String> = problems
        .iter()
        .map(|problem| problem.run(Rigor::Measure))
        .collect();
    let (double, single) = (wisdom::export::<f64>(), wisdom::export::<f32>());
    wisdom::forget::<f64>();
    wisdom::forget::<f32>();
    for problem in problems {
        let result = plan_from_wisdom(problem);
        assert!(
            matches!(result, Err(FftwError::PlanCreationFailed { .. })),
            "{problem}: a wisdom-only plan after `forget`: expected PlanCreationFailed, found \
             {result:?}"
        );
    }

    let result = wisdom::import::<f32>(&double);
    assert_eq!(
        result,
        Err(FftwError::WisdomImportFailed { precision: "f32" }),
        "f64 wisdom imported as f32"
    );
    let mut lines = vec!["f64 wisdom imported as f32: WisdomImportFailed".to_string()];
    let before = wisdom::export::<f64>();
    let result = wisdom::import::<f64>("(not wisdom");
    assert_eq!(
        result,
        Err(FftwError::WisdomImportFailed { precision: "f64" }),
        "text that is not wisdom imported as f64"
    );
    // FFTW does not document what a failed import leaves; reported only.
    lines.push(format!(
        "a failed import left the wisdom unchanged: {}",
        wisdom::export::<f64>() == before
    ));

    wisdom::import::<f64>(&double).unwrap();
    wisdom::import::<f32>(&single).unwrap();
    for (problem, measured) in problems.into_iter().zip(measured) {
        let from_wisdom = problem.run_from_wisdom(Rigor::Measure);
        assert_eq!(
            from_wisdom, measured,
            "the wisdom-only plan after export, forget and import differs from the measured plan"
        );
        lines.push(format!("A {measured}"));
        lines.push(format!("B {from_wisdom}"));
    }
    lines
}

/// The file of the wisdom of `precision` that `wisdom-export` writes and `wisdom-import`
/// reads: named by the parent's process id, which the two children share, so that
/// concurrent runs of this binary do not collide.
fn wisdom_file(precision: &str) -> PathBuf {
    let parent = std::os::unix::process::parent_id();
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("fftw-determinism-{parent}-{precision}.wisdom"))
}

/// `wisdom-export` (DC40): `Measure` plans of the wisdom problems, whose wisdom is written to
/// [`wisdom_file`]s.
fn case_wisdom_export() -> Vec<String> {
    let lines = wisdom_problems()
        .into_iter()
        .map(|problem| problem.run(Rigor::Measure))
        .collect();
    wisdom::export_to_file::<f64>(wisdom_file("f64")).unwrap();
    wisdom::export_to_file::<f32>(wisdom_file("f32")).unwrap();
    lines
}

/// `wisdom-import` (DC40): as its first FFTW work, imports the [`wisdom_file`]s of
/// `wisdom-export`, removes them, and plans the wisdom problems from wisdom only.
fn case_wisdom_import() -> Vec<String> {
    let imported = [
        ("f64", wisdom::import_from_file::<f64>(wisdom_file("f64"))),
        ("f32", wisdom::import_from_file::<f32>(wisdom_file("f32"))),
    ];
    for precision in ["f64", "f32"] {
        let _ = std::fs::remove_file(wisdom_file(precision));
    }
    for (precision, result) in imported {
        assert_eq!(result, Ok(()), "importing the {precision} wisdom file");
    }
    wisdom_problems()
        .into_iter()
        .map(|problem| problem.run_from_wisdom(Rigor::Measure))
        .collect()
}

/// `wisdom-concurrent` (DC39, design Section 7.5): 4 threads plan and execute the problems of
/// `estimate` while 2 threads export, forget and import the wisdom of both precisions; then
/// the problems of `estimate` once more, whose lines are printed.
fn case_wisdom_concurrent() -> Vec<String> {
    const PLANNERS: usize = 4;
    const WISDOM: usize = 2;
    // At least this many rounds per wisdom thread, and more until every planner finished.
    const MIN_ROUNDS: usize = 4;

    let finished = AtomicUsize::new(0);
    let barrier = Barrier::new(PLANNERS + WISDOM);
    thread::scope(|scope| {
        for _ in 0..PLANNERS {
            let (barrier, finished) = (&barrier, &finished);
            scope.spawn(move || {
                barrier.wait();
                for problem in Problem::estimate() {
                    problem.run(Rigor::Estimate);
                }
                finished.fetch_add(1, Ordering::SeqCst);
            });
        }
        for _ in 0..WISDOM {
            let (barrier, finished) = (&barrier, &finished);
            scope.spawn(move || {
                barrier.wait();
                let mut round = 0;
                while round < MIN_ROUNDS || finished.load(Ordering::SeqCst) < PLANNERS {
                    let (double, single) = (wisdom::export::<f64>(), wisdom::export::<f32>());
                    wisdom::forget::<f64>();
                    wisdom::forget::<f32>();
                    wisdom::import::<f64>(&double).unwrap();
                    wisdom::import::<f32>(&single).unwrap();
                    round += 1;
                    // Paces the rounds: the FFTW lock is not fair, and importing a few
                    // kilobytes of wisdom holds it for milliseconds, so rounds back to back
                    // starve the planners (6 s instead of 0.2 s on macOS). Nothing is
                    // synchronised by it.
                    thread::sleep(Duration::from_millis(1));
                }
            });
        }
    });
    case_estimate()
}

/// `system-wisdom`: imports the system's wisdom of both precisions and reports whether each
/// was imported. Nothing else plans in this process.
fn case_system_wisdom() -> Vec<String> {
    let outcome = |result: Result<(), FftwError>, precision: &str| match result {
        Ok(()) => format!("system wisdom, {precision}: imported"),
        Err(FftwError::WisdomImportFailed { .. }) => {
            format!("system wisdom, {precision}: WisdomImportFailed")
        }
        Err(error) => panic!("system wisdom, {precision}: unexpected error {error:?}"),
    };
    vec![
        outcome(wisdom::import_system_wisdom::<f64>(), "f64"),
        outcome(wisdom::import_system_wisdom::<f32>(), "f32"),
    ]
}

// Threaded plans (feature `fftw_threads`, task F6b; DC42).

/// The thread counts of threaded plans: fixed and small, for debug builds and the macOS
/// runner's 3 cores.
#[cfg(feature = "fftw_threads")]
const THREAD_COUNTS: [usize; 2] = [2, 4];

/// nd-fmm's units among the problems of `estimate`: `Preserving`, f64 in both directions and
/// n_phi = 140 in f32.
#[cfg(feature = "fftw_threads")]
fn estimate_units() -> Vec<Problem> {
    Problem::estimate()
        .into_iter()
        .filter(|problem| {
            matches!(
                problem,
                Problem::Unit {
                    policy: Policy::Preserving,
                    ..
                }
            )
        })
        .collect()
}

/// `estimate-threads`: nd-fmm's units of `estimate`, planned with `Estimate` for each of
/// [`THREAD_COUNTS`].
#[cfg(feature = "fftw_threads")]
fn case_estimate_threads() -> Vec<String> {
    THREAD_COUNTS
        .into_iter()
        .flat_map(|threads| {
            estimate_units().into_iter().map(move |problem| {
                problem.run(Options {
                    rigor: Rigor::Estimate,
                    threads,
                })
            })
        })
        .collect()
}

/// `estimate-after-threads`: threaded `Estimate` and `Measure` plans of the problems of
/// `estimate`, for each of [`THREAD_COUNTS`], and then the problems of `estimate` without
/// threads, whose lines are printed.
#[cfg(feature = "fftw_threads")]
fn case_estimate_after_threads() -> Vec<String> {
    for threads in THREAD_COUNTS {
        for rigor in [Rigor::Estimate, Rigor::Measure] {
            for problem in Problem::estimate() {
                problem.run(Options { rigor, threads });
            }
        }
    }
    case_estimate()
}

/// `planner-thread-safe`: `make_planner_thread_safe`, twice; then 4 threads plan and execute
/// the problems of `estimate` at once; then the problems of `estimate` once more, whose lines
/// are printed.
#[cfg(feature = "fftw_threads")]
fn case_planner_thread_safe() -> Vec<String> {
    const PLANNERS: usize = 4;

    rlst::fftw::threads::make_planner_thread_safe();
    // The second call does nothing.
    rlst::fftw::threads::make_planner_thread_safe();
    let barrier = Barrier::new(PLANNERS);
    thread::scope(|scope| {
        for _ in 0..PLANNERS {
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                for problem in Problem::estimate() {
                    problem.run(Rigor::Estimate);
                }
            });
        }
    });
    case_estimate()
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
    ("split-fixed-batch-units", case_split_fixed_batch_units),
    (
        "wisdom-only-without-wisdom",
        case_wisdom_only_without_wisdom,
    ),
    ("estimate-after-measure", case_estimate_after_measure),
    ("wisdom-round-trip", case_wisdom_round_trip),
    ("wisdom-export", case_wisdom_export),
    ("wisdom-import", case_wisdom_import),
    ("wisdom-concurrent", case_wisdom_concurrent),
    ("system-wisdom", case_system_wisdom),
];

/// The cases of threaded plans, which need `fftw_threads`.
#[cfg(feature = "fftw_threads")]
const THREADED_CASES: &[Case] = &[
    ("estimate-threads", case_estimate_threads),
    ("estimate-after-threads", case_estimate_after_threads),
    ("planner-thread-safe", case_planner_thread_safe),
];

/// The cases of threaded plans, which need `fftw_threads`.
#[cfg(not(feature = "fftw_threads"))]
const THREADED_CASES: &[Case] = &[];

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

/// Two runs of the fixed-batch `case` agree, and its line equals that of `problem` in
/// `estimate`.
fn check_units(runs: &Runs, case: &str, problem: Problem) -> Result<Vec<String>, String> {
    let mut notes = compare(
        runs.lines(case, 0),
        runs.lines(case, 1),
        "two fresh processes",
    )?;
    let line = &runs.lines(case, 0)[0];
    let estimate = estimate_line(runs, problem)?;
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

fn check_fixed_batch_units(runs: &Runs) -> Result<Vec<String>, String> {
    check_units(runs, "fixed-batch-units", unit_f64_forward(140))
}

fn check_split_fixed_batch_units(runs: &Runs) -> Result<Vec<String>, String> {
    check_units(runs, "split-fixed-batch-units", split_box_f64_forward(140))
}

fn check_child_assertions(_: &Runs) -> Result<Vec<String>, String> {
    Ok(vec!["the child's assertions hold".to_string()])
}

fn check_shared_wisdom(runs: &Runs) -> Result<Vec<String>, String> {
    compare(
        runs.lines("wisdom-export", 0),
        runs.lines("wisdom-import", 0),
        "`wisdom-import` against `wisdom-export` (a failure is a design question, DC40)",
    )
}

fn check_wisdom_during_planning(runs: &Runs) -> Result<Vec<String>, String> {
    compare(
        runs.lines("estimate", 0),
        runs.lines("wisdom-concurrent", 0),
        "`wisdom-concurrent` against `estimate`",
    )
}

fn check_system_wisdom(runs: &Runs) -> Result<Vec<String>, String> {
    Ok(runs.lines("system-wisdom", 0).to_vec())
}

#[cfg(feature = "fftw_threads")]
fn check_threaded_fresh_processes(runs: &Runs) -> Result<Vec<String>, String> {
    // DC42: a failure is raised as a design question, not relaxed.
    compare(
        runs.lines("estimate-threads", 0),
        runs.lines("estimate-threads", 1),
        "two fresh processes with threaded plans (a failure is a design question, DC42)",
    )
}

#[cfg(feature = "fftw_threads")]
fn check_threads_leave_no_trace(runs: &Runs) -> Result<Vec<String>, String> {
    // DC42: a failure is raised as a design question, not relaxed.
    compare(
        runs.lines("estimate", 0),
        runs.lines("estimate-after-threads", 0),
        "`estimate-after-threads` against `estimate` (a failure is a design question, DC42)",
    )
}

#[cfg(feature = "fftw_threads")]
fn check_planner_thread_safe(runs: &Runs) -> Result<Vec<String>, String> {
    compare(
        runs.lines("estimate", 0),
        runs.lines("planner-thread-safe", 0),
        "`planner-thread-safe` against `estimate`",
    )
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
        name: "split fixed-batch units",
        runs: &[("estimate", 1), ("split-fixed-batch-units", 2)],
        check: check_split_fixed_batch_units,
    },
    Scenario {
        name: "wisdom-only",
        runs: &[("wisdom-only-without-wisdom", 1)],
        check: check_child_assertions,
    },
    Scenario {
        name: "estimate after measure",
        runs: &[("estimate", 1), ("estimate-after-measure", 1)],
        check: check_estimate_after_measure,
    },
    Scenario {
        name: "wisdom round trip",
        runs: &[("wisdom-round-trip", 1)],
        check: check_child_assertions,
    },
    Scenario {
        name: "shared wisdom",
        runs: &[("wisdom-export", 1), ("wisdom-import", 1)],
        check: check_shared_wisdom,
    },
    Scenario {
        name: "wisdom during planning",
        runs: &[("estimate", 1), ("wisdom-concurrent", 1)],
        check: check_wisdom_during_planning,
    },
    Scenario {
        name: "system wisdom",
        runs: &[("system-wisdom", 1)],
        check: check_system_wisdom,
    },
];

/// The scenarios of threaded plans, which need `fftw_threads` (DC42).
#[cfg(feature = "fftw_threads")]
const THREADED_SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "threaded fresh processes agree",
        runs: &[("estimate-threads", 2)],
        check: check_threaded_fresh_processes,
    },
    Scenario {
        name: "threads leave no trace",
        runs: &[("estimate", 1), ("estimate-after-threads", 1)],
        check: check_threads_leave_no_trace,
    },
    Scenario {
        name: "thread-safe planner",
        runs: &[("estimate", 1), ("planner-thread-safe", 1)],
        check: check_planner_thread_safe,
    },
];

/// The scenarios of threaded plans, which need `fftw_threads` (DC42).
#[cfg(not(feature = "fftw_threads"))]
const THREADED_SCENARIOS: &[Scenario] = &[];

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
    let scenarios = SCENARIOS.iter().chain(THREADED_SCENARIOS);
    for scenario in scenarios.clone() {
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
        scenarios.count(),
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
        .chain(THREADED_CASES)
        .find(|(name, _)| *name == case)
        .unwrap_or_else(|| panic!("unknown case `{case}`"));
    for line in run() {
        println!("{line}");
    }
    ExitCode::SUCCESS
}
