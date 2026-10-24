//! Plans, and the markers of their transform family and input policy.

use std::ffi::c_uint;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

use super::align::AlignmentClass;
use super::kind::{Direction, R2rKind};
use super::layout::{Layout, LayoutFamily, Placement};
use super::lock::with_fftw;
use super::options::Rigor;
use super::real::FftwReal;
use super::scale::Scale;

pub(crate) mod sealed {
    use std::ffi::c_uint;

    use crate::fftw::layout::{LayoutFamily, Placement};

    /// Seals [`Family`](super::Family) and names the layouts of the family.
    pub trait Family {
        /// The family of the layouts that plans of this family are made for.
        const KIND: LayoutFamily;
    }

    /// Seals [`Policy`](super::Policy) and carries what the policy means for FFTW.
    pub trait Policy {
        /// The input-policy flag passed to FFTW.
        const FLAGS: c_uint;
        /// The placement of the plans of this policy.
        const PLACEMENT: Placement;
    }
}

/// The transform family of a [`Plan`]: [`C2c`], [`SplitC2c`], [`R2c`], [`C2r`] or [`R2r`] (the
/// split real families follow).
///
/// A family needs layouts of its [`Storage`](super::Storage) on each side, and every planner
/// method rejects a layout of another family. The trait is sealed.
///
/// ```
/// use rlst::fftw::{C2c, C2r, Family, R2c, R2r, SplitC2c};
///
/// fn family<F: Family>() {}
/// family::<C2c>();
/// family::<SplitC2c>();
/// family::<R2c>();
/// family::<C2r>();
/// family::<R2r>();
/// ```
pub trait Family: sealed::Family {}

/// The input policy of a [`Plan`]: [`Preserving`], [`Destroying`] or [`InPlace`].
///
/// The policy decides how [`Plan::execute`] borrows its input: shared for a preserving plan,
/// mutably for a plan that may overwrite it. The trait is sealed.
///
/// ```
/// use rlst::fftw::{Destroying, InPlace, Policy, Preserving};
///
/// fn policy<P: Policy>() {}
/// policy::<Preserving>();
/// policy::<Destroying>();
/// policy::<InPlace>();
/// ```
pub trait Policy: sealed::Policy {}

/// Complex-to-complex transforms on interleaved complex data.
///
/// A type-level marker; it is never constructed.
///
/// ```
/// use rlst::fftw::{C2c, Plan, Preserving};
///
/// type ForwardPlan = Plan<f64, C2c, Preserving>;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum C2c {}

/// Complex-to-complex transforms on split complex storage: the real and imaginary parts in
/// one buffer of reals, a fixed gap apart.
///
/// Plans of this family are made for layouts with gaps ([`Layout::new_split`],
/// [`Layout::split_along_axis`] and their companions) by
/// [`Planner::split_c2c`](super::Planner::split_c2c) and its companions, and execute on
/// buffers of reals: slices of `f32` or `f64`, and real RLST arrays and views, whose part axis
/// of extent 2 holds the real part at index 0 and the imaginary part at index 1. A type-level
/// marker; it is never constructed.
///
/// FFTW's SIMD code needs interleaved complex storage, so a split plan can use fewer SIMD
/// stages than an interleaved plan of the same transform, copying into buffers for some
/// stages and using scalar code for others (design, Sections 3 item 7 and 4.5). Copying split
/// data into interleaved scratch and using an interleaved plan can then be faster; the
/// benchmarks of the FFTW rework (task F8b) compare the two. Results of the two agree to
/// round-off, not bit for bit.
///
/// ```
/// use rlst::rlst_dynamic_array;
/// use rlst::fftw::{Direction, Layout, Plan, Planner, Preserving, SplitC2c};
///
/// // Two transforms of length 8, real plane then imaginary plane.
/// let mut input = rlst_dynamic_array!(f64, [8, 2, 2] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [8, 2, 2] | rlst::FFTW_ALIGNED);
/// input[[0, 1, 0]] = 1.0;
/// let layout = Layout::split_along_axis(&input, 2, &output, 2, 0)?;
/// let plan: Plan<f64, SplitC2c, Preserving> =
///     Planner::new().split_c2c(&layout, Direction::Forward)?;
/// plan.execute(&input, &mut output)?;
/// // The transform of a unit impulse is constant, with no imaginary part.
/// assert!((0..8).all(|k| output[[k, 1, 0]] == 1.0 && output[[k, 1, 1]] == 0.0));
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// A split plan executes on real buffers, so complex arrays do not compile:
///
/// ```compile_fail,E0277
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Direction, Layout, Planner};
///
/// let input = rlst_dynamic_array!(f64, [8, 2] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [8, 2] | rlst::FFTW_ALIGNED);
/// let layout = Layout::split_along_axis(&input, 1, &output, 1, 0)?;
/// let plan = Planner::<f64>::new().split_c2c(&layout, Direction::Forward)?;
/// let complex_input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut complex_output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// plan.execute(&complex_input, &mut complex_output)?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// while the same code on the real arrays does:
///
/// ```
/// use rlst::rlst_dynamic_array;
/// use rlst::fftw::{Direction, Layout, Planner};
///
/// let input = rlst_dynamic_array!(f64, [8, 2] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [8, 2] | rlst::FFTW_ALIGNED);
/// let layout = Layout::split_along_axis(&input, 1, &output, 1, 0)?;
/// let plan = Planner::<f64>::new().split_c2c(&layout, Direction::Forward)?;
/// plan.execute(&input, &mut output)?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum SplitC2c {}

/// An out-of-place plan that leaves its input unchanged; it executes with `&input`.
///
/// FFTW is given `FFTW_PRESERVE_INPUT`. A type-level marker; it is never constructed.
///
/// ```
/// use rlst::fftw::{C2c, Plan, Preserving};
///
/// type PreservingPlan = Plan<f64, C2c, Preserving>;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum Preserving {}

/// An out-of-place plan that may overwrite its input; it executes with `&mut input`.
///
/// FFTW is given `FFTW_DESTROY_INPUT`, which can allow faster algorithms. A type-level
/// marker; it is never constructed.
///
/// ```
/// use rlst::fftw::{C2c, Destroying, Plan};
///
/// type DestroyingPlan = Plan<f64, C2c, Destroying>;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum Destroying {}

/// An in-place plan; it executes on one buffer, `&mut data`.
///
/// A type-level marker; it is never constructed.
///
/// ```
/// use rlst::fftw::{C2c, InPlace, Plan};
///
/// type InPlacePlan = Plan<f64, C2c, InPlace>;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum InPlace {}

/// Real-to-complex transforms: real input, interleaved complex output, always forward (FFTW's
/// sign −1).
///
/// The output holds the non-redundant half of the spectrum of the real input: along the
/// halved axis, the last transform dimension, `n / 2 + 1` complex values for the `n` reals,
/// from which Hermitian symmetry gives the rest. Plans of this family are made for layouts of
/// real input and complex output ([`Layout::over_axes`] of a real and a complex array,
/// [`Layout::r2c_inplace`], [`Layout::with_storage`]) by
/// [`Planner::r2c`](super::Planner::r2c) and its companions, and execute with a real input
/// and a complex output. A type-level marker; it is never constructed.
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Layout, Planner};
///
/// let mut input = rlst_dynamic_array!(f64, [8, 2] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [5, 2] | rlst::FFTW_ALIGNED);
/// input[[1, 0]] = 1.0;
/// let layout = Layout::along_axis(&input, &output, 0)?;
/// let plan = Planner::<f64>::new().r2c(&layout)?;
/// plan.execute(&input, &mut output)?;
/// // The spectrum of an impulse at 1 is exp(-2πi k / 8).
/// let expected = c64::new(0.0, -std::f64::consts::PI / 2.0).exp();
/// assert!((output[[2, 0]] - expected).norm() < 1e-12);
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// The input is real, so a complex input does not compile:
///
/// ```compile_fail,E0277
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Layout, Planner};
///
/// let input = rlst_dynamic_array!(f64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [5] | rlst::FFTW_ALIGNED);
/// let layout = Layout::along_axis(&input, &output, 0)?;
/// let plan = Planner::<f64>::new().r2c(&layout)?;
/// let complex_input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// plan.execute(&complex_input, &mut output)?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// while the same code with the real input does:
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Layout, Planner};
///
/// let input = rlst_dynamic_array!(f64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [5] | rlst::FFTW_ALIGNED);
/// let layout = Layout::along_axis(&input, &output, 0)?;
/// let plan = Planner::<f64>::new().r2c(&layout)?;
/// plan.execute(&input, &mut output)?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum R2c {}

/// Complex-to-real transforms: interleaved complex input, real output, always backward
/// (FFTW's sign +1).
///
/// The inverse of [`R2c`]: the input is the non-redundant half of a Hermitian spectrum, `n /
/// 2 + 1` complex values along the halved axis, the last transform dimension, for `n` real
/// outputs. FFTW reads only that half, and assumes the symmetry: imaginary parts that
/// Hermitian symmetry forces to zero are ignored. Unlike every other family, **a
/// complex-to-real plan destroys its input by default**, as in FFTW: the default plan is
/// [`Destroying`] and executes with `&mut input`, so the default is visible at the call site.
/// [`Planner::preserve_input`](super::Planner::preserve_input) makes a [`Preserving`] plan,
/// over one transform dimension only (design, Section 3 item 5). A type-level marker; it is
/// never constructed.
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{C2r, Destroying, Layout, Plan, Planner};
///
/// let mut input = rlst_dynamic_array!(c64, [5] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [8] | rlst::FFTW_ALIGNED);
/// input[[0]] = c64::new(8.0, 0.0);
/// let layout = Layout::along_axis(&input, &output, 0)?;
/// let plan: Plan<f64, C2r, Destroying> = Planner::new().c2r(&layout)?;
/// plan.execute(&mut input, &mut output)?;
/// // The spectrum of a constant 1 of length 8.
/// assert!((0..8).all(|i| (output[[i]] - 8.0).abs() < 1e-12));
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// A default complex-to-real plan may overwrite its input, so this does not compile:
///
/// ```compile_fail,E0308
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Layout, Planner};
///
/// let input = rlst_dynamic_array!(c64, [5] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [8] | rlst::FFTW_ALIGNED);
/// let layout = Layout::along_axis(&input, &output, 0)?;
/// let plan = Planner::<f64>::new().c2r(&layout)?;
/// plan.execute(&input, &mut output)?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// while a plan made with `preserve_input()` takes the input shared:
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Layout, Planner};
///
/// let input = rlst_dynamic_array!(c64, [5] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [8] | rlst::FFTW_ALIGNED);
/// let layout = Layout::along_axis(&input, &output, 0)?;
/// let plan = Planner::<f64>::new().preserve_input().c2r(&layout)?;
/// plan.execute(&input, &mut output)?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum C2r {}

/// Real-to-real transforms: real input, real output, one [`R2rKind`] per transform dimension.
///
/// `kinds[i]` belongs to `layout.dims()[i]`, so for a layout made from arrays to `axes[i]`,
/// whatever the memory order of the arrays. (The 0.10 interface applied the kinds of
/// column-major arrays to the axes in reverse order.) Plans of this family are made for
/// layouts of real input and real output storage ([`Layout::over_axes`] of two real arrays,
/// [`Layout::over_axes_inplace`] of one, [`Layout::with_storage`]) by
/// [`Planner::r2r`](super::Planner::r2r) and its companions, which record the kinds
/// ([`Plan::kinds`]). A type-level marker; it is never constructed.
///
/// ```
/// use rlst::rlst_dynamic_array;
/// use rlst::fftw::{Layout, Planner, R2rKind};
///
/// // A DCT-II along axis 0 and a DST-II along axis 1 of a [4, 3] array.
/// let mut input = rlst_dynamic_array!(f64, [4, 3] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(f64, [4, 3] | rlst::FFTW_ALIGNED);
/// input.fill_from_seed_equally_distributed(0);
/// let layout = Layout::over_axes(&input, &output, &[0, 1])?;
/// let kinds = [R2rKind::Redft10, R2rKind::Rodft10];
/// let plan = Planner::<f64>::new().r2r(&layout, &kinds)?;
/// plan.execute(&input, &mut output)?;
/// assert_eq!(plan.kinds(), &kinds);
/// // The logical size is 2 · 4 times 2 · 3.
/// assert_eq!(plan.logical_size(), 48);
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum R2r {}

impl sealed::Family for C2c {
    const KIND: LayoutFamily = LayoutFamily::C2c;
}
impl Family for C2c {}

impl sealed::Family for SplitC2c {
    const KIND: LayoutFamily = LayoutFamily::SplitC2c;
}
impl Family for SplitC2c {}

impl sealed::Family for R2c {
    const KIND: LayoutFamily = LayoutFamily::R2c;
}
impl Family for R2c {}

impl sealed::Family for C2r {
    const KIND: LayoutFamily = LayoutFamily::C2r;
}
impl Family for C2r {}

impl sealed::Family for R2r {
    const KIND: LayoutFamily = LayoutFamily::R2r;
}
impl Family for R2r {}

impl sealed::Policy for Preserving {
    const FLAGS: c_uint = fftw_sys::FFTW_PRESERVE_INPUT;
    const PLACEMENT: Placement = Placement::OutOfPlace;
}
impl Policy for Preserving {}

impl sealed::Policy for Destroying {
    const FLAGS: c_uint = fftw_sys::FFTW_DESTROY_INPUT;
    const PLACEMENT: Placement = Placement::OutOfPlace;
}
impl Policy for Destroying {}

impl sealed::Policy for InPlace {
    // FFTW ignores the input-policy flags in place.
    const FLAGS: c_uint = 0;
    const PLACEMENT: Placement = Placement::InPlace;
}
impl Policy for InPlace {}

/// An FFTW plan of precision `T`, family `F` and input policy `P`.
///
/// A plan describes a [`Layout`], not particular buffers. It executes on any buffers that fit
/// the layout and have the [`AlignmentClass`]es it was made for, from any thread: `Plan` is
/// `Send` and `Sync`, since FFTW's execution is thread-safe and a plan is not changed by it.
/// Plans are made by a [`Planner`](super::Planner), and destroyed under RLST's FFTW lock when
/// dropped.
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
///
/// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
/// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
///
/// let mut input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// input.data_mut().unwrap()[0] = c64::new(1.0, 0.0);
/// plan.execute(input.data().unwrap(), output.data_mut().unwrap())?;
/// // The transform of a unit impulse is constant.
/// assert!(output.data().unwrap().iter().all(|&y| y == c64::new(1.0, 0.0)));
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// A [`Destroying`] plan needs its input mutably, so this does not compile:
///
/// ```compile_fail,E0308
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
///
/// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
/// let plan = Planner::<f64>::new().may_destroy_input().c2c(&layout, Direction::Forward)?;
/// let input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// plan.execute(input.data().unwrap(), output.data_mut().unwrap())?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// while the same code with the input borrowed mutably does:
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
///
/// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
/// let plan = Planner::<f64>::new().may_destroy_input().c2c(&layout, Direction::Forward)?;
/// let mut input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// plan.execute(input.data_mut().unwrap(), output.data_mut().unwrap())?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// An [`InPlace`] plan executes on one buffer, so this does not compile:
///
/// ```compile_fail,E0061
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
///
/// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::InPlace)?;
/// let plan = Planner::<f64>::new().c2c_inplace(&layout, Direction::Forward)?;
/// let mut input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// plan.execute(input.data_mut().unwrap(), output.data_mut().unwrap())?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// while the same code with one buffer does:
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
///
/// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::InPlace)?;
/// let plan = Planner::<f64>::new().c2c_inplace(&layout, Direction::Forward)?;
/// let mut input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// plan.execute(input.data_mut().unwrap())?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
///
/// # Threads and determinism
///
/// - **Concurrent execution.** One plan may be executed from several threads at once on
///   different buffers. This is safe and takes no lock.
/// - **Same plan, same bits.** A plan gives bit-for-bit identical results on inputs of the
///   same alignment class. Two plans are the same plan when they have the same normalised
///   layout, placement, alignment classes, input policy, rigor and wisdom-only setting, were
///   made with the same FFTW build on a CPU with the same SIMD support, and found the same
///   wisdom for their problem in the process. FFTW uses any wisdom of equal or greater rigor,
///   so an [`Estimate`](Rigor::Estimate) plan made after a [`Measure`](Rigor::Measure) plan
///   of the same problem can be the `Measure` plan. [`Plan::fingerprint`] tells whether two
///   plans are the same.
/// - **Timed planning.** Plans made with [`Rigor::Measure`] and stronger rigor choose
///   algorithms by timing them, so they, and their bits, can differ between runs and between
///   processes.
/// - **Batch and threads.** A plan for another batch count or layout is another plan and can
///   round differently. For results that do not depend on the number of threads or on
///   scheduling, fix the unit of work, such as one box of a fixed batch, and execute one plan
///   on every unit. Never size a plan from a scheduler's chunking.
///
/// One plan executed from several threads on disjoint units of one aligned buffer, each unit
/// a multiple of 64 bytes long so that all of them are at alignment class 0:
///
/// ```
/// use rlst::{SliceArray, SliceArrayMut, c64, rlst_dynamic_array};
/// use rlst::fftw::{Direction, Layout, Planner};
///
/// let (shape, units) = ([16, 4], 8);
/// let len = shape[0] * shape[1];
/// let mut input = rlst_dynamic_array!(c64, [len * units] | rlst::FFTW_ALIGNED);
/// let mut output = rlst_dynamic_array!(c64, [len * units] | rlst::FFTW_ALIGNED);
/// input.fill_from_seed_equally_distributed(0);
///
/// let first = SliceArray::from_shape(&input.data().unwrap()[..len], shape);
/// let layout = Layout::along_axis(&first, &first, 0)?;
/// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
///
/// std::thread::scope(|scope| {
///     let inputs = input.data().unwrap().chunks_exact(len);
///     let outputs = output.data_mut().unwrap().chunks_exact_mut(len);
///     for (input, output) in inputs.zip(outputs) {
///         let plan = &plan;
///         scope.spawn(move || {
///             let input = SliceArray::from_shape(input, shape);
///             let mut output = SliceArrayMut::from_shape(output, shape);
///             plan.execute(&input, &mut output).unwrap();
///         });
///     }
/// });
///
/// // Every unit's result is that of the plan executed on the unit alone.
/// let unit = SliceArray::from_shape(&input.data().unwrap()[len..2 * len], shape);
/// let mut alone = rlst_dynamic_array!(c64, shape | rlst::FFTW_ALIGNED);
/// plan.execute(&unit, &mut alone)?;
/// assert_eq!(alone.data().unwrap(), &output.data().unwrap()[len..2 * len]);
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
pub struct Plan<T: FftwReal, F: Family, P: Policy> {
    raw: NonNull<T::RawPlan>,
    layout: Layout,
    direction: Direction,
    rigor: Rigor,
    flags: c_uint,
    alignment: Option<(AlignmentClass, AlignmentClass)>,
    /// The kind of each transform dimension of a real-to-real plan; empty otherwise.
    kinds: Vec<R2rKind>,
    _marker: PhantomData<(T, F, P)>,
}

// SAFETY: FFTW's execute functions are thread-safe, and the same plan may be executed by
// several threads at once on different arrays (design, Section 3 item 2); execution does not
// modify the plan. Every other FFTW call on a plan (`fftw_sprint_plan`, and destruction, which
// needs ownership) runs under the FFTW lock, so sharing or sending a plan between threads
// never makes two such calls race.
unsafe impl<T: FftwReal, F: Family, P: Policy> Send for Plan<T, F, P> {}
// SAFETY: as for `Send`.
unsafe impl<T: FftwReal, F: Family, P: Policy> Sync for Plan<T, F, P> {}

impl<T: FftwReal, F: Family, P: Policy> Plan<T, F, P> {
    /// Wrap a plan that FFTW made for `layout`.
    pub(crate) fn from_raw(
        raw: NonNull<T::RawPlan>,
        layout: Layout,
        direction: Direction,
        rigor: Rigor,
        flags: c_uint,
        alignment: Option<(AlignmentClass, AlignmentClass)>,
    ) -> Self {
        Self {
            raw,
            layout,
            direction,
            rigor,
            flags,
            alignment,
            kinds: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// The plan with the kinds of a real-to-real plan.
    pub(crate) fn with_kinds(mut self, kinds: &[R2rKind]) -> Self {
        self.kinds = kinds.to_vec();
        self
    }

    /// The raw FFTW plan.
    pub(crate) fn raw(&self) -> *mut T::RawPlan {
        self.raw.as_ptr()
    }

    /// The normalised layout the plan was made for.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// assert_eq!(plan.layout(), &layout);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The rigor the plan was made with.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner, Rigor};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// assert_eq!(plan.rigor(), Rigor::Estimate);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn rigor(&self) -> Rigor {
        self.rigor
    }

    /// The alignment classes of input and output the plan was made for, or `None` for a plan
    /// made with [`Planner::unaligned`](super::Planner::unaligned), which runs on buffers of
    /// any class.
    ///
    /// ```
    /// use rlst::fftw::{AlignmentClass, Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// let class = AlignmentClass::default();
    /// assert_eq!(plan.alignment(), Some((class, class)));
    ///
    /// let unaligned = Planner::<f64>::new().unaligned().c2c(&layout, Direction::Forward)?;
    /// assert_eq!(unaligned.alignment(), None);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn alignment(&self) -> Option<(AlignmentClass, AlignmentClass)> {
        self.alignment
    }

    /// The logical size N of one transform: the product over the transform dimensions of the
    /// extent n, which for real-to-complex and complex-to-real plans is the logical real
    /// extent, and for real-to-real plans of each kind's logical size for its dimension
    /// ([`R2rKind::logical_size`]: 2 (n − 1) for a DCT-I, 2 n for a DCT-II, …).
    ///
    /// A backward transform after a forward one multiplies the data by N, and
    /// [`scale_factor`](Plan::scale_factor) undoes it (design, Section 4.9).
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// // Real-to-complex over both axes of a real [6, 8] array; axis 1 is halved.
    /// let real = rlst_dynamic_array!(f64, [6, 8] | rlst::FFTW_ALIGNED);
    /// let complex = rlst_dynamic_array!(c64, [6, 5] | rlst::FFTW_ALIGNED);
    /// let plan = Planner::<f64>::new().r2c(&Layout::over_axes(&real, &complex, &[0, 1])?)?;
    /// assert_eq!(plan.logical_size(), 48);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn logical_size(&self) -> usize {
        match F::KIND {
            LayoutFamily::R2r => self
                .layout
                .dims()
                .iter()
                .zip(&self.kinds)
                .map(|(dim, kind)| kind.logical_size(dim.n as usize))
                .product(),
            _ => self.layout.logical_size(),
        }
    }

    /// The factor 1, 1/N or 1/√N of `scale`, with N the [`logical_size`](Plan::logical_size),
    /// in the plan's precision.
    ///
    /// FFTW's transforms are unnormalised; the caller multiplies the output by the factor
    /// with RLST's array arithmetic. A uniform factor does not make the discrete cosine and
    /// sine transforms orthonormal (see [`Scale`]).
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner, Scale};
    ///
    /// let layout = Layout::new(&[Dim { n: 16, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f32>::new().c2c(&layout, Direction::Backward)?;
    /// assert_eq!(plan.scale_factor(Scale::None), 1.0);
    /// assert_eq!(plan.scale_factor(Scale::Inverse), 0.0625);
    /// assert_eq!(plan.scale_factor(Scale::InverseSqrt), 0.25);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn scale_factor(&self, scale: Scale) -> T {
        T::from_f64(scale.factor(self.logical_size()))
            .expect("a scale factor in (0, 1] is a finite real of every FFTW precision")
    }
}

impl<T: FftwReal, P: Policy> Plan<T, C2c, P> {
    /// The direction of the transform.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f32>::new().c2c(&layout, Direction::Backward)?;
    /// assert_eq!(plan.direction(), Direction::Backward);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn direction(&self) -> Direction {
        self.direction
    }
}

impl<T: FftwReal, P: Policy> Plan<T, SplitC2c, P> {
    /// The direction of the transform.
    ///
    /// FFTW's split planner has no sign and always plans a forward transform; a backward
    /// plan is FFTW's forward plan with the real and imaginary parts swapped on both sides,
    /// at planning and at every execution (design, Section 3 item 8). Nothing else of the
    /// plan depends on the direction.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout =
    ///     Layout::new_split(&[Dim { n: 8, is: 1, os: 1 }], &[], (8, 8), Placement::OutOfPlace)?;
    /// let plan = Planner::<f32>::new().split_c2c(&layout, Direction::Backward)?;
    /// assert_eq!(plan.direction(), Direction::Backward);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// The offsets, in reals from the real part of the first element, of the two pointers
    /// FFTW receives on each side, as in [`split_offsets`].
    pub(crate) fn split_offsets(&self) -> [(usize, usize); 2] {
        let (input_gap, output_gap) = self
            .layout
            .gaps()
            .expect("split plans are made for layouts with gaps");
        [
            split_offsets(self.direction, input_gap),
            split_offsets(self.direction, output_gap),
        ]
    }
}

impl<T: FftwReal, P: Policy> Plan<T, R2r, P> {
    /// The kind of each transform dimension, in the order of
    /// [`layout().dims()`](Layout::dims): for a layout made from arrays, the kind of each
    /// entry of `axes`.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement, Planner, R2rKind, Storage};
    ///
    /// let layout = Layout::with_storage(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[],
    ///     (Storage::Real, Storage::Real),
    ///     Placement::OutOfPlace,
    /// )?;
    /// let plan = Planner::<f64>::new().r2r(&layout, &[R2rKind::Dht])?;
    /// assert_eq!(plan.kinds(), &[R2rKind::Dht]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn kinds(&self) -> &[R2rKind] {
        &self.kinds
    }
}

/// The offsets, in reals from the real part of the first element of a side with gap `gap`,
/// of the pointers FFTW receives as its real and imaginary parts for a split transform in
/// `direction`.
///
/// FFTW's split planner always plans a forward transform (design, Section 3 item 8). Forward,
/// it receives the real part and the real part plus the gap. Backward, the parts are swapped:
/// it receives the real part plus the gap as its real part and the real part as its imaginary
/// part, so that it sees the signed gap −gap, and its forward transform of the swapped parts
/// is the backward transform of the data. Planning and execution use this one function, so
/// they pass FFTW the same arrangement.
pub(crate) fn split_offsets(direction: Direction, gap: isize) -> (usize, usize) {
    // Layouts with gaps have positive gaps (`Layout::new_split`).
    let gap = gap.unsigned_abs();
    match direction {
        Direction::Forward => (0, gap),
        Direction::Backward => (gap, 0),
    }
}

impl<T: FftwReal, F: Family, P: Policy> fmt::Debug for Plan<T, F, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("precision", &std::any::type_name::<T>())
            .field("family", &std::any::type_name::<F>())
            .field("policy", &std::any::type_name::<P>())
            .field("layout", &self.layout)
            .field("direction", &self.direction)
            .field("rigor", &self.rigor)
            .field("flags", &format_args!("{:#x}", self.flags))
            .field("alignment", &self.alignment)
            .field("kinds", &self.kinds)
            .finish()
    }
}

impl<T: FftwReal, F: Family, P: Policy> Drop for Plan<T, F, P> {
    fn drop(&mut self) {
        // SAFETY: `raw` is a live plan owned by `self` and not used after this call; the FFTW
        // lock is held.
        with_fftw(|_| unsafe { T::destroy_plan(self.raw.as_ptr()) });
    }
}
