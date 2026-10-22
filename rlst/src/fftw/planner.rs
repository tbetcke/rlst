//! The planner: options, scratch and the planning call.

use std::ffi::{c_int, c_uint};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;

use aligned_vec::{AVec, ConstAlign};
use num::Complex;

use super::align::{AlignmentClass, PLAN_ALIGNMENT};
use super::buffer::FftBufferMut;
use super::error::{FftwError, Side};
use super::execute::check_overlap;
use super::kind::Direction;
use super::layout::{Dim, Layout, Placement};
use super::lock::{SettingsGuard, with_fftw};
use super::options::sealed::PolicyFor;
use super::options::{DefaultRequest, DestroyRequest, PreserveRequest, Rigor};
use super::plan::{C2c, InPlace, Plan, Policy};
use super::real::FftwReal;

/// The default bound on scratch memory for planning: 1 GiB for input and output together.
///
/// ```
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
/// [`max_scratch`](Planner::max_scratch). For a layout made from a view, the span runs from
/// the view's first element to the last one it addresses, which for a column of a row-major
/// matrix is nearly the whole matrix. The `*_on` methods plan on the caller's buffers, slices
/// or RLST arrays and views.
///
/// Every planning call runs under RLST's FFTW lock, which serialises all of RLST's FFTW calls
/// other than execution.
///
/// ```
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
    rigor: Rigor,
    wisdom_only: bool,
    unaligned: bool,
    time_limit: Option<f64>,
    alignment: (AlignmentClass, AlignmentClass),
    max_scratch: usize,
    _marker: PhantomData<(T, S)>,
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// A planner with the default options: [`Rigor::Estimate`], planning from wisdom and
    /// heuristics, aligned plans for class 0 on both sides, no time limit, and a scratch bound
    /// of [`DEFAULT_MAX_SCRATCH`].
    ///
    /// ```
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f32>::new();
    /// # let _ = planner;
    /// ```
    pub fn new() -> Self {
        Self {
            rigor: Rigor::Estimate,
            wisdom_only: false,
            unaligned: false,
            time_limit: None,
            alignment: (AlignmentClass::default(), AlignmentClass::default()),
            max_scratch: DEFAULT_MAX_SCRATCH,
            _marker: PhantomData,
        }
    }

    /// Request plans that preserve their input.
    ///
    /// For c2c this is the default. For c2r over more than one dimension FFTW has no
    /// input-preserving algorithm, and planning returns
    /// [`FftwError::PreserveInputUnsupported`].
    ///
    /// ```
    /// use rlst::fftw::{C2c, Dim, Direction, Layout, Placement, Plan, Planner, Preserving};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan: Plan<f64, C2c, Preserving> =
    ///     Planner::new().preserve_input().c2c(&layout, Direction::Forward)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn preserve_input(self) -> Planner<T, PreserveRequest> {
        self.with_request()
    }

    /// Request plans that may overwrite their input, which can allow faster algorithms.
    ///
    /// The plans are [`Destroying`](super::Destroying) and take their input mutably.
    ///
    /// ```
    /// use rlst::fftw::{C2c, Destroying, Dim, Direction, Layout, Placement, Plan, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan: Plan<f64, C2c, Destroying> =
    ///     Planner::new().may_destroy_input().c2c(&layout, Direction::Forward)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn may_destroy_input(self) -> Planner<T, DestroyRequest> {
        self.with_request()
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
    /// ```
    /// use rlst::fftw::{Planner, Rigor};
    ///
    /// let planner = Planner::<f64>::new().rigor(Rigor::Estimate);
    /// # let _ = planner;
    /// ```
    pub fn rigor(self, rigor: Rigor) -> Self {
        Self { rigor, ..self }
    }

    /// Plan only from wisdom (`FFTW_WISDOM_ONLY`).
    ///
    /// FFTW uses wisdom of the planner's rigor or greater; without such wisdom, planning
    /// returns [`FftwError::PlanCreationFailed`].
    ///
    /// ```
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().wisdom_only();
    /// # let _ = planner;
    /// ```
    pub fn wisdom_only(self) -> Self {
        Self {
            wisdom_only: true,
            ..self
        }
    }

    /// Make plans that run on buffers of any alignment class (`FFTW_UNALIGNED`).
    ///
    /// Execution then skips the alignment check, at the cost of most of FFTW's SIMD code.
    ///
    /// ```
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().unaligned();
    /// # let _ = planner;
    /// ```
    pub fn unaligned(self) -> Self {
        Self {
            unaligned: true,
            ..self
        }
    }

    /// Bound the time FFTW spends planning, in seconds (`fftw_set_timelimit`).
    ///
    /// A negative value or NaN means no limit, the default. The limit applies only to the
    /// planner's own calls, and is reset before RLST's FFTW lock is released.
    ///
    /// ```
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().time_limit(0.5);
    /// # let _ = planner;
    /// ```
    pub fn time_limit(self, seconds: f64) -> Self {
        Self {
            time_limit: (seconds >= 0.0).then_some(seconds),
            ..self
        }
    }

    /// Plan for buffers of the given alignment classes; the default is class 0 on both sides.
    ///
    /// The planning methods without `_on` allocate their scratch at these classes. The
    /// `*_on` methods take the classes of the caller's buffers instead.
    ///
    /// ```
    /// use rlst::fftw::{AlignmentClass, Planner};
    ///
    /// let class = AlignmentClass::from_offset(16);
    /// let planner = Planner::<f64>::new().alignment(class, class);
    /// # let _ = planner;
    /// ```
    pub fn alignment(self, input: AlignmentClass, output: AlignmentClass) -> Self {
        Self {
            alignment: (input, output),
            ..self
        }
    }

    /// Bound the scratch memory of planning, in bytes, for input and output together; the
    /// default is [`DEFAULT_MAX_SCRATCH`].
    ///
    /// A layout that needs more returns [`FftwError::ScratchTooLarge`]; the `*_on` methods
    /// plan such layouts on the caller's buffers.
    ///
    /// ```
    /// use rlst::fftw::Planner;
    ///
    /// let planner = Planner::<f64>::new().max_scratch(1 << 20);
    /// # let _ = planner;
    /// ```
    pub fn max_scratch(self, bytes: usize) -> Self {
        Self {
            max_scratch: bytes,
            ..self
        }
    }

    /// The same options under another input-policy request.
    fn with_request<R>(self) -> Planner<T, R> {
        Planner {
            rigor: self.rigor,
            wisdom_only: self.wisdom_only,
            unaligned: self.unaligned,
            time_limit: self.time_limit,
            alignment: self.alignment,
            max_scratch: self.max_scratch,
            _marker: PhantomData,
        }
    }

    /// The FFTW flags of a plan of policy `P`.
    fn flags<P: Policy>(&self) -> c_uint {
        let mut flags = self.rigor.flag() | P::FLAGS;
        if self.wisdom_only {
            flags |= fftw_sys::FFTW_WISDOM_ONLY;
        }
        if self.unaligned {
            flags |= fftw_sys::FFTW_UNALIGNED;
        }
        flags
    }

    /// Plan a c2c transform of policy `P` on scratch.
    fn c2c_on_scratch<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, C2c, P>, FftwError> {
        check_placement::<P>(layout)?;
        let (input_class, output_class) = self.alignment;
        if layout.placement() == Placement::InPlace && input_class != output_class {
            return Err(FftwError::InvalidLayout {
                dimension: "alignment".to_string(),
                reason: format!(
                    "an in-place plan has one buffer, but the input class {input_class} differs \
                     from the output class {output_class}"
                ),
            });
        }
        for class in [input_class, output_class] {
            if class.offset() % align_of::<Complex<T>>() != 0 {
                return Err(FftwError::InvalidLayout {
                    dimension: "alignment".to_string(),
                    reason: format!(
                        "the class {class} is not a multiple of the {}-byte alignment of the \
                         complex type",
                        align_of::<Complex<T>>()
                    ),
                });
            }
        }

        // A size that overflows `usize` is above every bound, `usize::MAX` included.
        let elements = match layout.placement() {
            Placement::OutOfPlace => layout.input_span().checked_add(layout.output_span()),
            Placement::InPlace => Some(layout.input_span()),
        };
        let needed = elements.and_then(|elements| elements.checked_mul(size_of::<Complex<T>>()));
        match needed {
            Some(needed) if needed <= self.max_scratch => {}
            _ => {
                return Err(FftwError::ScratchTooLarge {
                    needed: needed.unwrap_or(usize::MAX),
                    bound: self.max_scratch,
                });
            }
        }

        // The scratch lives until the end of this function, after planning.
        let mut input = Scratch::<T>::new(layout.input_span(), input_class);
        let mut output = match layout.placement() {
            Placement::OutOfPlace => Some(Scratch::<T>::new(layout.output_span(), output_class)),
            Placement::InPlace => None,
        };
        let input_ptr = input.as_mut_ptr();
        let output_ptr = output.as_mut().map_or(input_ptr, Scratch::as_mut_ptr);
        // SAFETY: `input_ptr` and `output_ptr` point to scratch buffers of at least the input
        // and output spans, which outlive the call; in place they are the same buffer.
        unsafe {
            self.plan_c2c(
                layout,
                direction,
                input_ptr,
                output_ptr,
                (input_class, output_class),
            )
        }
    }

    /// Plan a c2c transform of policy `P` on the caller's buffers.
    fn c2c_on_buffers<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
        input: *mut Complex<T>,
        output: *mut Complex<T>,
    ) -> Result<Plan<T, C2c, P>, FftwError> {
        if layout.placement() == Placement::OutOfPlace {
            check_overlap(input, layout.input_span(), output, layout.output_span())?;
        }
        let classes = (
            AlignmentClass::of_ptr(input),
            AlignmentClass::of_ptr(output),
        );
        // SAFETY: `input` and `output` are the pointers that `check_side_mut` returned after
        // checking the caller's buffers against the input and output spans of `layout`, and
        // the caller borrows the buffers mutably for the call.
        unsafe { self.plan_c2c(layout, direction, input, output, classes) }
    }

    /// Call FFTW's guru64 planner under the FFTW lock.
    ///
    /// # Safety
    ///
    /// `input` and `output` point to buffers of at least the input and output spans of
    /// `layout`, valid for reads and writes for the duration of the call (FFTW writes them
    /// with rigors other than `Estimate`); for an in-place layout they are equal.
    unsafe fn plan_c2c<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
        input: *mut Complex<T>,
        output: *mut Complex<T>,
        classes: (AlignmentClass, AlignmentClass),
    ) -> Result<Plan<T, C2c, P>, FftwError> {
        let flags = self.flags::<P>();
        let dims = iodims(layout.dims());
        let batch = iodims(layout.batch());
        let too_many = |what: &str| FftwError::InvalidLayout {
            dimension: "layout".to_string(),
            reason: format!("FFTW takes at most {} {what} dimensions", c_int::MAX),
        };
        let rank = c_int::try_from(dims.len()).map_err(|_| too_many("transform"))?;
        let howmany_rank = c_int::try_from(batch.len()).map_err(|_| too_many("batch"))?;

        let raw = with_fftw(|state| {
            state.version_check()?;
            let _settings = SettingsGuard::<T>::new(state, self.time_limit);
            // SAFETY: the FFTW lock is held. `dims` and `batch` hold `rank` and
            // `howmany_rank` dimensions with non-negative strides, and by this function's
            // contract `input` and `output` cover the spans of these dimensions; they are
            // equal exactly for an in-place layout, whose strides are equal on both sides.
            let raw = unsafe {
                T::plan_guru64_dft(
                    rank,
                    dims.as_ptr(),
                    howmany_rank,
                    batch.as_ptr(),
                    input,
                    output,
                    direction.sign(),
                    flags,
                )
            };
            NonNull::new(raw).ok_or_else(|| FftwError::PlanCreationFailed {
                rigor: self.rigor,
                flags,
                layout: layout.describe(),
            })
        })?;

        let alignment = (!self.unaligned).then_some(classes);
        Ok(Plan::from_raw(
            raw,
            layout.clone(),
            direction,
            self.rigor,
            flags,
            alignment,
        ))
    }
}

/// Scratch for planning: zeroed reals at `PLAN_ALIGNMENT`, long enough for a span of complex
/// elements starting at an alignment class.
struct Scratch<T> {
    data: AVec<T, ConstAlign<PLAN_ALIGNMENT>>,
    offset: usize,
}

impl<T: FftwReal> Scratch<T> {
    /// Scratch for `span` complex elements at `class`, which is a multiple of the alignment of
    /// `Complex<T>` and therefore of the size of `T`. The caller has checked that `span`
    /// complex elements fit in `usize` bytes.
    fn new(span: usize, class: AlignmentClass) -> Self {
        let offset = class.offset() / size_of::<T>();
        let len = span
            .checked_mul(2)
            .and_then(|reals| reals.checked_add(offset))
            .expect("the scratch size was checked against `max_scratch`");
        let data = AVec::from_iter(PLAN_ALIGNMENT, std::iter::repeat_n(T::zero(), len));
        Self { data, offset }
    }

    /// A pointer to the first complex element, at the requested class.
    fn as_mut_ptr(&mut self) -> *mut Complex<T> {
        // `offset` reals and 2 * span reals after it lie inside the allocation, and
        // `Complex<T>` is two `T`s with the alignment of `T`.
        self.data
            .as_mut_ptr()
            .wrapping_add(self.offset)
            .cast::<Complex<T>>()
    }
}

/// The dimensions as FFTW's `fftw_iodim64`.
fn iodims(dims: &[Dim]) -> Vec<fftw_sys::fftw_iodim64> {
    dims.iter()
        .map(|dim| fftw_sys::fftw_iodim64 {
            n: dim.n,
            is: dim.is,
            os: dim.os,
        })
        .collect()
}

/// Check that the layout's placement is that of the policy.
fn check_placement<P: Policy>(layout: &Layout) -> Result<(), FftwError> {
    if layout.placement() != P::PLACEMENT {
        return Err(FftwError::PlacementMismatch {
            expected: P::PLACEMENT,
            found: layout.placement(),
        });
    }
    Ok(())
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
    /// ```
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
        self.c2c_on_scratch(layout, direction)
    }

    /// An out-of-place complex-to-complex plan for `layout`, planned on the caller's buffers.
    ///
    /// The buffers are slices or RLST arrays and views, checked against the layout as at
    /// execution ([`FftwError::LayoutMismatch`]), and the plan is made for the buffers'
    /// alignment classes. No scratch is allocated, so this plans layouts whose span exceeds
    /// [`max_scratch`](Planner::max_scratch), such as a column of a large row-major array.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
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
    ///
    /// // On a column of row-major arrays, whose span is nearly the whole array.
    /// let mut input = rlst::StridedDynArray::<c64, 2>::row_major([4, 1000]);
    /// let mut output = rlst::StridedDynArray::<c64, 2>::row_major([4, 1000]);
    /// let layout = Layout::along_axis(&input.r().col(7), &output.r().col(7), 0)?;
    /// let plan = Planner::<f64>::new().max_scratch(1024).c2c_on(
    ///     &mut input.r_mut().col(7),
    ///     &mut output.r_mut().col(7),
    ///     &layout,
    ///     Direction::Forward,
    /// )?;
    /// plan.execute(&input.r().col(7), &mut output.r_mut().col(7))?;
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
        check_placement::<S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.c2c_on_buffers(layout, direction, input, output)
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
    /// ```
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
        self.c2c_on_scratch(layout, direction)
    }

    /// An in-place complex-to-complex plan for `layout`, planned on the caller's buffer.
    ///
    /// The buffer is a slice or an RLST array or view, checked against the layout as at
    /// execution ([`FftwError::LayoutMismatch`]), and the plan is made for the buffer's
    /// alignment class. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffer untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites it.**
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::InPlace)?;
    /// let mut data = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
    /// let plan =
    ///     Planner::<f64>::new().c2c_inplace_on(data.data_mut().unwrap(), &layout, Direction::Forward)?;
    /// plan.execute(data.data_mut().unwrap())?;
    ///
    /// // On a subview of an aligned array; the plan is made for the subview's class.
    /// let mut array = rlst_dynamic_array!(c64, [8, 4] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis_inplace(&array.r().into_subview([0, 1], [8, 2]), 0)?;
    /// let plan = Planner::<f64>::new().c2c_inplace_on(
    ///     &mut array.r_mut().into_subview([0, 1], [8, 2]),
    ///     &layout,
    ///     Direction::Forward,
    /// )?;
    /// plan.execute(&mut array.r_mut().into_subview([0, 1], [8, 2]))?;
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
        check_placement::<InPlace>(layout)?;
        data.check_side(layout, Side::Output)?;
        // In place the input and output spans are equal, so one check covers both sides.
        let data = data.check_side_mut(layout, Side::Input)?;
        self.c2c_on_buffers(layout, direction, data, data)
    }
}
