//! The planner: options, scratch and the planning call.

use std::marker::PhantomData;

use super::align::AlignmentClass;
use super::buffer::FftBufferMut;
use super::error::FftwError;
use super::kind::Direction;
use super::layout::Layout;
use super::options::sealed::PolicyFor;
use super::options::{DefaultRequest, DestroyRequest, PreserveRequest, Rigor};
use super::plan::{C2c, InPlace, Plan};
use super::real::FftwReal;
use num::Complex;

/// The default bound on scratch memory for planning: 1 GiB for input and output together.
///
/// ```no_run
/// assert_eq!(rlst::fftw::DEFAULT_MAX_SCRATCH, 1 << 30);
/// ```
pub const DEFAULT_MAX_SCRATCH: usize = 1 << 30;

/// A builder for [`Plan`]s of precision `T`.
///
/// The options apply to every plan the planner makes. The type parameter `S` records the
/// input-policy request: [`DefaultRequest`] from [`new`](Planner::new),
/// [`PreserveRequest`] after [`preserve_input`](Planner::preserve_input) and
/// [`DestroyRequest`] after [`may_destroy_input`](Planner::may_destroy_input). Together with
/// the family it decides the policy of the plan:
///
/// | Planner call | Plan |
/// | --- | --- |
/// | `Planner::new().c2c(…)`, `.preserve_input().c2c(…)` | `Plan<T, C2c, Preserving>` |
/// | `Planner::new().may_destroy_input().c2c(…)` | `Plan<T, C2c, Destroying>` |
/// | `Planner::new().c2c_inplace(…)` | `Plan<T, C2c, InPlace>` |
///
/// Planning methods without `_on` plan on scratch buffers that the planner allocates for the
/// layout's span, at the requested alignment classes, and frees after planning, so planning
/// never touches user data, whatever the rigor. The scratch is bounded by
/// [`max_scratch`](Planner::max_scratch). The `*_on` methods plan on the caller's buffers.
///
/// Every planning call runs under RLST's FFTW lock, which serialises all of RLST's FFTW calls
/// other than execution.
///
/// ```no_run
/// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner, Rigor};
///
/// let layout = Layout::new(&[Dim { n: 16, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
/// let planner = Planner::<f64>::new().rigor(Rigor::Estimate);
/// let forward = planner.c2c(&layout, Direction::Forward)?;
/// let backward = planner.c2c(&layout, Direction::Backward)?;
/// # let _ = (forward, backward);
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Planner<T: FftwReal, S = DefaultRequest> {
    // The options follow in F1c.
    _marker: PhantomData<(T, S)>,
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// A planner with the default options: [`Rigor::Estimate`], planning from wisdom and
    /// heuristics, aligned plans for class 0 on both sides, no time limit, and a scratch bound
    /// of [`DEFAULT_MAX_SCRATCH`].
    ///
    /// ```no_run
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f32>::new();
    /// # let _ = planner;
    /// ```
    pub fn new() -> Self {
        todo!()
    }

    /// Request plans that preserve their input.
    ///
    /// For c2c this is the default. For c2r over more than one dimension FFTW has no
    /// input-preserving algorithm, and planning returns
    /// [`FftwError::PreserveInputUnsupported`].
    ///
    /// ```no_run
    /// use rlst::fftw::{C2c, Dim, Direction, Layout, Placement, Plan, Planner, Preserving};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan: Plan<f64, C2c, Preserving> =
    ///     Planner::new().preserve_input().c2c(&layout, Direction::Forward)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn preserve_input(self) -> Planner<T, PreserveRequest> {
        todo!()
    }

    /// Request plans that may overwrite their input, which can allow faster algorithms.
    ///
    /// The plans are [`Destroying`](super::Destroying) and take their input mutably.
    ///
    /// ```no_run
    /// use rlst::fftw::{C2c, Destroying, Dim, Direction, Layout, Placement, Plan, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan: Plan<f64, C2c, Destroying> =
    ///     Planner::new().may_destroy_input().c2c(&layout, Direction::Forward)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn may_destroy_input(self) -> Planner<T, DestroyRequest> {
        todo!()
    }
}

impl<T: FftwReal> Default for Planner<T, DefaultRequest> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FftwReal, S> Planner<T, S> {
    /// Set the planning rigor; the default is [`Rigor::Estimate`].
    ///
    /// With any other rigor FFTW runs transforms while planning: on scratch buffers for the
    /// planning methods without `_on`, on the caller's buffers for the `*_on` methods.
    ///
    /// ```no_run
    /// use rlst::fftw::{Planner, Rigor};
    ///
    /// let planner = Planner::<f64>::new().rigor(Rigor::Estimate);
    /// # let _ = planner;
    /// ```
    pub fn rigor(self, rigor: Rigor) -> Self {
        let _ = rigor;
        todo!()
    }

    /// Plan only from wisdom (`FFTW_WISDOM_ONLY`).
    ///
    /// FFTW uses wisdom of the planner's rigor or greater; without such wisdom, planning
    /// returns [`FftwError::PlanCreationFailed`].
    ///
    /// ```no_run
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().wisdom_only();
    /// # let _ = planner;
    /// ```
    pub fn wisdom_only(self) -> Self {
        todo!()
    }

    /// Make plans that run on buffers of any alignment class (`FFTW_UNALIGNED`).
    ///
    /// Execution then skips the alignment check, at the cost of most of FFTW's SIMD code.
    ///
    /// ```no_run
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().unaligned();
    /// # let _ = planner;
    /// ```
    pub fn unaligned(self) -> Self {
        todo!()
    }

    /// Bound the time FFTW spends planning, in seconds (`fftw_set_timelimit`).
    ///
    /// A negative value means no limit, the default. The limit applies only to the planner's
    /// own calls, and is reset before RLST's FFTW lock is released.
    ///
    /// ```no_run
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().time_limit(0.5);
    /// # let _ = planner;
    /// ```
    pub fn time_limit(self, seconds: f64) -> Self {
        let _ = seconds;
        todo!()
    }

    /// Plan for buffers of the given alignment classes; the default is class 0 on both sides.
    ///
    /// The planning methods without `_on` allocate their scratch at these classes. The
    /// `*_on` methods take the classes of the caller's buffers instead.
    ///
    /// ```no_run
    /// use rlst::fftw::{AlignmentClass, Planner};
    ///
    /// let class = AlignmentClass::from_offset(16);
    /// let planner = Planner::<f64>::new().alignment(class, class);
    /// # let _ = planner;
    /// ```
    pub fn alignment(self, input: AlignmentClass, output: AlignmentClass) -> Self {
        let _ = (input, output);
        todo!()
    }

    /// Bound the scratch memory of planning, in bytes, for input and output together; the
    /// default is [`DEFAULT_MAX_SCRATCH`].
    ///
    /// A layout that needs more returns [`FftwError::ScratchTooLarge`]; the `*_on` methods
    /// plan such layouts on the caller's buffers.
    ///
    /// ```no_run
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().max_scratch(1 << 20);
    /// # let _ = planner;
    /// ```
    pub fn max_scratch(self, bytes: usize) -> Self {
        let _ = bytes;
        todo!()
    }
}

impl<T: FftwReal, S: PolicyFor<C2c>> Planner<T, S> {
    /// An out-of-place complex-to-complex plan for `layout`, planned on scratch.
    ///
    /// The plan is [`Preserving`](super::Preserving), or [`Destroying`](super::Destroying)
    /// after [`may_destroy_input`](Planner::may_destroy_input). Returns
    /// [`FftwError::PlacementMismatch`] for an in-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch).
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// // Three transforms of length 8, one after the other.
    /// let layout = Layout::new(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[Dim { n: 3, is: 8, os: 8 }],
    ///     Placement::OutOfPlace,
    /// )?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// # let _ = plan;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2c(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, C2c, S::Policy>, FftwError> {
        let _ = (layout, direction);
        todo!()
    }

    /// An out-of-place complex-to-complex plan for `layout`, planned on the caller's buffers.
    ///
    /// The layout's spans must fit the buffers ([`FftwError::LayoutMismatch`]), and the plan
    /// is made for the buffers' alignment classes. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```no_run
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let mut input = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let plan = Planner::<f64>::new().c2c_on(
    ///     input.data_mut().unwrap(),
    ///     output.data_mut().unwrap(),
    ///     &layout,
    ///     Direction::Forward,
    /// )?;
    /// plan.execute(input.data().unwrap(), output.data_mut().unwrap())?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2c_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, C2c, S::Policy>, FftwError>
    where
        I: FftBufferMut<Complex<T>> + ?Sized,
        O: FftBufferMut<Complex<T>> + ?Sized,
    {
        let _ = (input, output, layout, direction);
        todo!()
    }
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// An in-place complex-to-complex plan for `layout`, planned on scratch.
    ///
    /// Returns [`FftwError::PlacementMismatch`] for an out-of-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch). The input-policy requests do not apply in
    /// place, so this method is only available on a planner without one.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::InPlace)?;
    /// let plan = Planner::<f64>::new().c2c_inplace(&layout, Direction::Forward)?;
    /// # let _ = plan;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2c_inplace(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, C2c, InPlace>, FftwError> {
        let _ = (layout, direction);
        todo!()
    }

    /// An in-place complex-to-complex plan for `layout`, planned on the caller's buffer.
    ///
    /// The layout's span must fit the buffer ([`FftwError::LayoutMismatch`]), and the plan
    /// is made for the buffer's alignment class. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffer untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites it.**
    ///
    /// ```no_run
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::InPlace)?;
    /// let mut data = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let plan =
    ///     Planner::<f64>::new().c2c_inplace_on(data.data_mut().unwrap(), &layout, Direction::Forward)?;
    /// plan.execute(data.data_mut().unwrap())?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2c_inplace_on<D>(
        &self,
        data: &mut D,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, C2c, InPlace>, FftwError>
    where
        D: FftBufferMut<Complex<T>> + ?Sized,
    {
        let _ = (data, layout, direction);
        todo!()
    }
}
