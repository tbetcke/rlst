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
use super::kind::{Direction, R2rKind};
use super::layout::{Dim, Layout, LayoutFamily, Placement};
use super::lock::{SettingsGuard, with_fftw};
use super::options::sealed::PolicyFor;
use super::options::{DefaultRequest, DestroyRequest, PreserveRequest, Rigor};
use super::plan::{
    C2c, C2r, Family, InPlace, Plan, Policy, R2c, R2r, SplitC2c, SplitC2r, SplitR2c, split_offsets,
};
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
/// | `Planner::new().split_c2c(…)` | `Plan<T, SplitC2c, Preserving>` |
/// | `Planner::new().preserve_input().split_c2c(…)` | `Plan<T, SplitC2c, Preserving>` |
/// | `Planner::new().may_destroy_input().split_c2c(…)` | `Plan<T, SplitC2c, Destroying>` |
/// | `Planner::new().split_c2c_inplace(…)` | `Plan<T, SplitC2c, InPlace>` |
/// | `Planner::new().r2c(…)`, `.preserve_input().r2c(…)` | `Plan<T, R2c, Preserving>` |
/// | `Planner::new().may_destroy_input().r2c(…)` | `Plan<T, R2c, Destroying>` |
/// | `Planner::new().r2c_inplace(…)` | `Plan<T, R2c, InPlace>` |
/// | `Planner::new().c2r(…)`, `.may_destroy_input().c2r(…)` | `Plan<T, C2r, Destroying>` |
/// | `Planner::new().preserve_input().c2r(…)`, one transform dimension only | `Plan<T, C2r, Preserving>` |
/// | `Planner::new().c2r_inplace(…)` | `Plan<T, C2r, InPlace>` |
/// | `Planner::new().r2r(…)`, `.preserve_input().r2r(…)` | `Plan<T, R2r, Preserving>` |
/// | `Planner::new().may_destroy_input().r2r(…)` | `Plan<T, R2r, Destroying>` |
/// | `Planner::new().r2r_inplace(…)` | `Plan<T, R2r, InPlace>` |
/// | `Planner::new().split_r2c(…)`, `.preserve_input().split_r2c(…)` | `Plan<T, SplitR2c, Preserving>` |
/// | `Planner::new().may_destroy_input().split_r2c(…)` | `Plan<T, SplitR2c, Destroying>` |
/// | `Planner::new().split_c2r(…)`, `.may_destroy_input().split_c2r(…)` | `Plan<T, SplitC2r, Destroying>` |
/// | `Planner::new().preserve_input().split_c2r(…)`, one transform dimension only | `Plan<T, SplitC2r, Preserving>` |
///
/// Each family follows FFTW's default input policy (DC1): every plan preserves its input
/// unless [`may_destroy_input`](Planner::may_destroy_input) is given, except a
/// complex-to-real plan, split or not, which destroys its input unless
/// [`preserve_input`](Planner::preserve_input) is given. Every planner method rejects a
/// layout of another family with [`FftwError::InvalidLayout`], before FFTW is called.
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
    /// For c2c, split c2c and r2c this is the default. For c2r it makes a
    /// [`Preserving`](super::Preserving) plan, at some cost in speed; over more than one
    /// transform dimension FFTW has no input-preserving algorithm, and planning returns
    /// [`FftwError::PreserveInputUnsupported`] before FFTW is called.
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
    /// returns [`FftwError::PlanCreationFailed`]. The [`wisdom`](super::wisdom) module
    /// exports and imports wisdom, for example to make the same plans in several processes.
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

    /// The alignment classes to plan scratch for: the planner's, checked to be equal in
    /// place and, on each side, a multiple of `align` bytes, the alignment of that side's
    /// element type `what`.
    fn scratch_classes(
        &self,
        layout: &Layout,
        [(input_align, input_what), (output_align, output_what)]: [(usize, &str); 2],
    ) -> Result<(AlignmentClass, AlignmentClass), FftwError> {
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
        for (class, align, what) in [
            (input_class, input_align, input_what),
            (output_class, output_align, output_what),
        ] {
            if class.offset() % align != 0 {
                return Err(FftwError::InvalidLayout {
                    dimension: "alignment".to_string(),
                    reason: format!(
                        "the class {class} is not a multiple of the {align}-byte alignment of \
                         the {what}"
                    ),
                });
            }
        }
        Ok((input_class, output_class))
    }

    /// Check the scratch of `layout`, with input and output elements of `sizes` bytes, against
    /// the bound: the bytes of both spans out of place, and in place of the larger one, the
    /// one buffer that holds both sides.
    fn check_scratch_size(
        &self,
        layout: &Layout,
        (input_size, output_size): (usize, usize),
    ) -> Result<(), FftwError> {
        // A size that overflows `usize` is above every bound, `usize::MAX` included.
        let input = layout.input_span().checked_mul(input_size);
        let output = layout.output_span().checked_mul(output_size);
        let needed = match layout.placement() {
            Placement::OutOfPlace => input.zip(output).and_then(|(a, b)| a.checked_add(b)),
            Placement::InPlace => input.zip(output).map(|(a, b)| a.max(b)),
        };
        match needed {
            Some(needed) if needed <= self.max_scratch => Ok(()),
            _ => Err(FftwError::ScratchTooLarge {
                needed: needed.unwrap_or(usize::MAX),
                bound: self.max_scratch,
            }),
        }
    }

    /// Plan a c2c transform of policy `P` on scratch.
    fn c2c_on_scratch<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, C2c, P>, FftwError> {
        check_storage::<C2c>(layout)?;
        check_placement::<P>(layout)?;
        let complex = (align_of::<Complex<T>>(), "complex type");
        let (input_class, output_class) = self.scratch_classes(layout, [complex, complex])?;
        let size = size_of::<Complex<T>>();
        self.check_scratch_size(layout, (size, size))?;

        // The scratch lives until the end of this function, after planning.
        let mut input = Scratch::<T>::new(layout.input_span(), 2, input_class);
        let mut output = match layout.placement() {
            Placement::OutOfPlace => Some(Scratch::<T>::new(layout.output_span(), 2, output_class)),
            Placement::InPlace => None,
        };
        let input_ptr = input.as_mut_ptr().cast::<Complex<T>>();
        let output_ptr = output
            .as_mut()
            .map_or(input_ptr, |output| output.as_mut_ptr().cast::<Complex<T>>());
        // SAFETY: `input_ptr` and `output_ptr` point to scratch buffers of at least the input
        // and output spans, in complex elements, which outlive the call; `Complex<T>` is two
        // `T`s with the alignment of `T`, and in place they are the same buffer.
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
        let guru = GuruDims::new(layout)?;

        let raw = with_fftw(|state| {
            state.version_check()?;
            let _settings = SettingsGuard::<T>::new(state, self.time_limit);
            // SAFETY: the FFTW lock is held. `guru` holds `rank` and `howmany_rank`
            // dimensions with non-negative strides, and by this function's contract `input`
            // and `output` cover the spans of these dimensions; they are equal exactly for an
            // in-place layout, whose strides are equal on both sides.
            let raw = unsafe {
                T::plan_guru64_dft(
                    guru.rank,
                    guru.dims.as_ptr(),
                    guru.howmany_rank,
                    guru.batch.as_ptr(),
                    input,
                    output,
                    direction.sign(),
                    flags,
                )
            };
            self.plan_or_error(raw, flags, layout, None)
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

    /// Plan a split c2c transform of policy `P` on scratch.
    fn split_c2c_on_scratch<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, SplitC2c, P>, FftwError> {
        check_storage::<SplitC2c>(layout)?;
        check_placement::<P>(layout)?;
        let real = (align_of::<T>(), "real type");
        let (input_class, output_class) = self.scratch_classes(layout, [real, real])?;
        let size = size_of::<T>();
        self.check_scratch_size(layout, (size, size))?;

        // The scratch lives until the end of this function, after planning. The real part of
        // each side's first element is at the requested class.
        let mut input = Scratch::<T>::new(layout.input_span(), 1, input_class);
        let mut output = match layout.placement() {
            Placement::OutOfPlace => Some(Scratch::<T>::new(layout.output_span(), 1, output_class)),
            Placement::InPlace => None,
        };
        let input_ptr = input.as_mut_ptr();
        let output_ptr = output.as_mut().map_or(input_ptr, Scratch::as_mut_ptr);
        // SAFETY: `input_ptr` and `output_ptr` point to scratch buffers of at least the input
        // and output spans, in reals, which outlive the call; in place they are the same
        // buffer.
        unsafe {
            self.plan_split_c2c(
                layout,
                direction,
                input_ptr,
                output_ptr,
                (input_class, output_class),
            )
        }
    }

    /// Plan a split c2c transform of policy `P` on the caller's buffers.
    fn split_c2c_on_buffers<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
        input: *mut T,
        output: *mut T,
    ) -> Result<Plan<T, SplitC2c, P>, FftwError> {
        if layout.placement() == Placement::OutOfPlace {
            check_overlap(input, layout.input_span(), output, layout.output_span())?;
        }
        // The classes of the real parts; those of the imaginary parts follow from the gaps.
        let classes = (
            AlignmentClass::of_ptr(input),
            AlignmentClass::of_ptr(output),
        );
        // SAFETY: `input` and `output` are the pointers to the real parts that
        // `check_side_mut` returned after checking the caller's buffers against the input
        // and output spans of `layout`, gaps included, and the caller borrows the buffers
        // mutably for the call.
        unsafe { self.plan_split_c2c(layout, direction, input, output, classes) }
    }

    /// Call FFTW's guru64 split planner under the FFTW lock.
    ///
    /// # Safety
    ///
    /// `layout` has gaps; `input` and `output` point to the real parts of the first elements
    /// of buffers of at least the input and output spans of `layout`, in reals and gaps
    /// included, valid for reads and writes for the duration of the call (FFTW writes them
    /// with rigors other than `Estimate`); for an in-place layout they are equal.
    unsafe fn plan_split_c2c<P: Policy>(
        &self,
        layout: &Layout,
        direction: Direction,
        input: *mut T,
        output: *mut T,
        classes: (AlignmentClass, AlignmentClass),
    ) -> Result<Plan<T, SplitC2c, P>, FftwError> {
        let flags = self.flags::<P>();
        let guru = GuruDims::new(layout)?;
        let (input_gap, output_gap) = layout
            .gaps()
            .expect("`check_storage` accepted only split layouts, with gaps");
        // FFTW plans forward only; a backward plan swaps the parts on both sides.
        let (ri, ii) = split_offsets(direction, input_gap);
        let (ro, io) = split_offsets(direction, output_gap);

        let raw = with_fftw(|state| {
            state.version_check()?;
            let _settings = SettingsGuard::<T>::new(state, self.time_limit);
            // SAFETY: the FFTW lock is held. `guru` holds `rank` and `howmany_rank`
            // dimensions with non-negative strides, in reals. By this function's contract
            // `input` and `output` cover the spans of these dimensions, which include the gap
            // of each side, so the four pointers, at offsets 0 and the gap from the real
            // parts, lie inside the buffers, and every element FFTW addresses from them does.
            // In place `input == output` and the gaps are equal, so `ri == ro` and `ii == io`.
            let raw = unsafe {
                T::plan_guru64_split_dft(
                    guru.rank,
                    guru.dims.as_ptr(),
                    guru.howmany_rank,
                    guru.batch.as_ptr(),
                    input.wrapping_add(ri),
                    input.wrapping_add(ii),
                    output.wrapping_add(ro),
                    output.wrapping_add(io),
                    flags,
                )
            };
            self.plan_or_error(raw, flags, layout, None)
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

    /// Plan a real-to-complex (`F = R2c`) or complex-to-real (`F = C2r`) transform of policy
    /// `P` on scratch: on two buffers out of place, and in place on one buffer that holds
    /// both sides, the complex side and the real side inside it.
    fn real_on_scratch<F: Family, P: Policy>(
        &self,
        layout: &Layout,
    ) -> Result<Plan<T, F, P>, FftwError> {
        check_storage::<F>(layout)?;
        check_placement::<P>(layout)?;
        check_preserving_c2r::<F, P>(layout)?;
        let real = (align_of::<T>(), "real type");
        let complex = (align_of::<Complex<T>>(), "complex type");
        let r2c = F::KIND == LayoutFamily::R2c;
        let ((input_reals, input_kind), (output_reals, output_kind)) = if r2c {
            ((1, real), (2, complex))
        } else {
            ((2, complex), (1, real))
        };
        let (input_class, output_class) =
            self.scratch_classes(layout, [input_kind, output_kind])?;
        self.check_scratch_size(
            layout,
            (input_reals * size_of::<T>(), output_reals * size_of::<T>()),
        )?;

        // The scratch lives until the end of this function, after planning. In place, one
        // buffer of reals covers both sides: the complex side, whose span, in reals, is at
        // least the real side's (`Layout::with_storage`).
        let (input_span, output_span) = (
            layout.input_span() * input_reals,
            layout.output_span() * output_reals,
        );
        let (mut input, mut output) = match layout.placement() {
            Placement::OutOfPlace => (
                Scratch::<T>::new(input_span, 1, input_class),
                Some(Scratch::<T>::new(output_span, 1, output_class)),
            ),
            Placement::InPlace => (
                Scratch::<T>::new(input_span.max(output_span), 1, input_class),
                None,
            ),
        };
        let input_ptr = input.as_mut_ptr();
        let output_ptr = output.as_mut().map_or(input_ptr, Scratch::as_mut_ptr);
        // SAFETY: `input_ptr` and `output_ptr` point to scratch buffers of at least the input
        // and output spans, in elements of each side (a complex element is two reals with the
        // alignment of `T`), which outlive the call; in place they are the same buffer, which
        // holds the larger of the two spans.
        unsafe { self.plan_real(layout, input_ptr, output_ptr, (input_class, output_class)) }
    }

    /// Plan a real-to-complex or complex-to-real transform of policy `P` on the caller's
    /// buffers, given as pointers to their first reals: the pointers that the side checks
    /// returned, cast to reals for a complex side.
    fn real_on_buffers<F: Family, P: Policy>(
        &self,
        layout: &Layout,
        input: *mut T,
        output: *mut T,
    ) -> Result<Plan<T, F, P>, FftwError> {
        if layout.placement() == Placement::OutOfPlace {
            let (input_reals, output_reals) = match F::KIND {
                LayoutFamily::R2c => (1, 2),
                _ => (2, 1),
            };
            check_overlap(
                input,
                layout.input_span() * input_reals,
                output,
                layout.output_span() * output_reals,
            )?;
        }
        let classes = (
            AlignmentClass::of_ptr(input),
            AlignmentClass::of_ptr(output),
        );
        // SAFETY: `input` and `output` are the pointers that `check_side_mut` returned after
        // checking the caller's buffers against the input and output spans of `layout`, in
        // place the complex side, which holds the real side; the caller borrows the buffers
        // mutably for the call.
        unsafe { self.plan_real(layout, input, output, classes) }
    }

    /// Call FFTW's guru64 r2c (`F = R2c`) or c2r (`F = C2r`) planner under the FFTW lock.
    ///
    /// # Safety
    ///
    /// `layout` is a layout of family `F`; `input` and `output` point to the first reals of
    /// buffers of at least the input and output spans of `layout`, in elements of each side,
    /// valid for reads and writes for the duration of the call (FFTW writes them with rigors
    /// other than `Estimate`). For an in-place layout they are equal, and the buffer holds the
    /// complex side's span.
    unsafe fn plan_real<F: Family, P: Policy>(
        &self,
        layout: &Layout,
        input: *mut T,
        output: *mut T,
        classes: (AlignmentClass, AlignmentClass),
    ) -> Result<Plan<T, F, P>, FftwError> {
        let flags = self.flags::<P>();
        let guru = GuruDims::new(layout)?;
        let r2c = F::KIND == LayoutFamily::R2c;

        let raw = with_fftw(|state| {
            state.version_check()?;
            let _settings = SettingsGuard::<T>::new(state, self.time_limit);
            // SAFETY: the FFTW lock is held. `guru` holds `rank` and `howmany_rank`
            // dimensions with non-negative strides, in reals on the real side and in complex
            // elements on the complex side, with the logical real extent n of the last
            // transform dimension, as FFTW's guru64 r2c and c2r planners take them. By this
            // function's contract `input` and `output` cover the spans of these dimensions,
            // which use n / 2 + 1 on the complex side. In place they are equal and the
            // buffer holds the complex side, inside which the real side lies: along the last
            // transform dimension, contiguous on both sides, n reals fit in the 2 (n / 2 + 1)
            // reals of the complex elements, and on every other dimension the real stride is
            // twice the complex stride, the same distance (`Layout::with_storage`).
            let raw = unsafe {
                if r2c {
                    T::plan_guru64_dft_r2c(
                        guru.rank,
                        guru.dims.as_ptr(),
                        guru.howmany_rank,
                        guru.batch.as_ptr(),
                        input,
                        output.cast::<Complex<T>>(),
                        flags,
                    )
                } else {
                    T::plan_guru64_dft_c2r(
                        guru.rank,
                        guru.dims.as_ptr(),
                        guru.howmany_rank,
                        guru.batch.as_ptr(),
                        input.cast::<Complex<T>>(),
                        output,
                        flags,
                    )
                }
            };
            self.plan_or_error(raw, flags, layout, None)
        })?;

        // FFTW's r2c transforms are always forward and its c2r transforms always backward.
        let direction = if r2c {
            Direction::Forward
        } else {
            Direction::Backward
        };
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

    /// Plan a real-to-real transform of policy `P` with `kinds` on scratch.
    fn r2r_on_scratch<P: Policy>(
        &self,
        layout: &Layout,
        kinds: &[R2rKind],
    ) -> Result<Plan<T, R2r, P>, FftwError> {
        check_storage::<R2r>(layout)?;
        check_placement::<P>(layout)?;
        check_kinds(layout, kinds)?;
        let real = (align_of::<T>(), "real type");
        let (input_class, output_class) = self.scratch_classes(layout, [real, real])?;
        let size = size_of::<T>();
        self.check_scratch_size(layout, (size, size))?;

        // The scratch lives until the end of this function, after planning; in place the
        // spans of both sides are equal.
        let mut input = Scratch::<T>::new(layout.input_span(), 1, input_class);
        let mut output = match layout.placement() {
            Placement::OutOfPlace => Some(Scratch::<T>::new(layout.output_span(), 1, output_class)),
            Placement::InPlace => None,
        };
        let input_ptr = input.as_mut_ptr();
        let output_ptr = output.as_mut().map_or(input_ptr, Scratch::as_mut_ptr);
        // SAFETY: `input_ptr` and `output_ptr` point to scratch buffers of at least the input
        // and output spans, in reals, which outlive the call; in place they are the same
        // buffer.
        unsafe {
            self.plan_r2r(
                layout,
                kinds,
                input_ptr,
                output_ptr,
                (input_class, output_class),
            )
        }
    }

    /// Plan a real-to-real transform of policy `P` with `kinds` on the caller's buffers.
    fn r2r_on_buffers<P: Policy>(
        &self,
        layout: &Layout,
        kinds: &[R2rKind],
        input: *mut T,
        output: *mut T,
    ) -> Result<Plan<T, R2r, P>, FftwError> {
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
        unsafe { self.plan_r2r(layout, kinds, input, output, classes) }
    }

    /// Call FFTW's guru64 r2r planner under the FFTW lock.
    ///
    /// # Safety
    ///
    /// `layout` is a real-to-real layout and `kinds` has one kind per transform dimension
    /// (`check_kinds`); `input` and `output` point to buffers of at least the input and output
    /// spans of `layout`, in reals, valid for reads and writes for the duration of the call
    /// (FFTW writes them with rigors other than `Estimate`); for an in-place layout they are
    /// equal.
    unsafe fn plan_r2r<P: Policy>(
        &self,
        layout: &Layout,
        kinds: &[R2rKind],
        input: *mut T,
        output: *mut T,
        classes: (AlignmentClass, AlignmentClass),
    ) -> Result<Plan<T, R2r, P>, FftwError> {
        let flags = self.flags::<P>();
        let guru = GuruDims::new(layout)?;
        // FFTW's kinds in the order of the transform dimensions, which is the order FFTW
        // receives the dimensions in: `kinds[i]` belongs to `layout.dims()[i]`.
        let raw_kinds: Vec<fftw_sys::fftw_r2r_kind> = kinds.iter().map(|kind| kind.raw()).collect();
        // A preserving plan passes `FFTW_PRESERVE_INPUT`, as every preserving plan does. FFTW's
        // HC2R destroys its input by default, and FFTW may have no input-preserving algorithm
        // for an HC2R problem (design, Section 4.4; DC34).
        let preserving = P::FLAGS & fftw_sys::FFTW_PRESERVE_INPUT != 0;
        let hint = (preserving && kinds.contains(&R2rKind::Hc2r)).then_some(
            "FFTW may have no input-preserving algorithm for a transform with an HC2R kind; plan \
             it with `Planner::may_destroy_input()`, whose plans may overwrite their input",
        );

        let raw = with_fftw(|state| {
            state.version_check()?;
            let _settings = SettingsGuard::<T>::new(state, self.time_limit);
            // SAFETY: the FFTW lock is held. `guru` holds `rank` and `howmany_rank`
            // dimensions with non-negative strides, in reals, and `raw_kinds` holds `rank`
            // kinds, one per transform dimension in the order of `guru.dims`, which outlive
            // the call. By this function's contract `input` and `output` cover the spans of
            // these dimensions; they are equal exactly for an in-place layout, whose strides
            // are equal on both sides.
            let raw = unsafe {
                T::plan_guru64_r2r(
                    guru.rank,
                    guru.dims.as_ptr(),
                    guru.howmany_rank,
                    guru.batch.as_ptr(),
                    input,
                    output,
                    raw_kinds.as_ptr(),
                    flags,
                )
            };
            self.plan_or_error(raw, flags, layout, hint)
        })?;

        let alignment = (!self.unaligned).then_some(classes);
        // A real-to-real plan has no direction; its kinds say what it computes.
        Ok(Plan::from_raw(
            raw,
            layout.clone(),
            Direction::Forward,
            self.rigor,
            flags,
            alignment,
        )
        .with_kinds(kinds))
    }

    /// Plan a split real-to-complex (`F = SplitR2c`) or complex-to-real (`F = SplitC2r`)
    /// transform of policy `P` on scratch, which counts reals on both sides.
    fn split_real_on_scratch<F: Family, P: Policy>(
        &self,
        layout: &Layout,
    ) -> Result<Plan<T, F, P>, FftwError> {
        check_storage::<F>(layout)?;
        check_placement::<P>(layout)?;
        check_preserving_c2r::<F, P>(layout)?;
        let real = (align_of::<T>(), "real type");
        let (input_class, output_class) = self.scratch_classes(layout, [real, real])?;
        let size = size_of::<T>();
        self.check_scratch_size(layout, (size, size))?;

        // The scratch lives until the end of this function, after planning; split real
        // layouts are out of place (`Layout::with_storage`). The first real of each side, the
        // real part of the first element on the split side, is at the requested class.
        let mut input = Scratch::<T>::new(layout.input_span(), 1, input_class);
        let mut output = Scratch::<T>::new(layout.output_span(), 1, output_class);
        // SAFETY: the pointers point to separate scratch buffers of at least the input and
        // output spans, in reals and the gap of the split side included, which outlive the
        // call.
        unsafe {
            self.plan_split_real(
                layout,
                input.as_mut_ptr(),
                output.as_mut_ptr(),
                (input_class, output_class),
            )
        }
    }

    /// Plan a split real-to-complex or complex-to-real transform of policy `P` on the
    /// caller's buffers, given as the pointers to their first reals that the side checks
    /// returned.
    fn split_real_on_buffers<F: Family, P: Policy>(
        &self,
        layout: &Layout,
        input: *mut T,
        output: *mut T,
    ) -> Result<Plan<T, F, P>, FftwError> {
        check_overlap(input, layout.input_span(), output, layout.output_span())?;
        // The classes of the first reals; that of the split side's imaginary part follows
        // from its gap.
        let classes = (
            AlignmentClass::of_ptr(input),
            AlignmentClass::of_ptr(output),
        );
        // SAFETY: `input` and `output` are the pointers that `check_side_mut` returned after
        // checking the caller's buffers against the input and output spans of `layout`, the
        // gap of the split side included, and the caller borrows the buffers mutably for the
        // call.
        unsafe { self.plan_split_real(layout, input, output, classes) }
    }

    /// Call FFTW's guru64 split r2c (`F = SplitR2c`) or split c2r (`F = SplitC2r`) planner
    /// under the FFTW lock.
    ///
    /// # Safety
    ///
    /// `layout` is an out-of-place layout of family `F`; `input` and `output` point to the
    /// first reals of separate buffers of at least the input and output spans of `layout`, in
    /// reals and the gap of the split side included, valid for reads and writes for the
    /// duration of the call (FFTW writes them with rigors other than `Estimate`).
    unsafe fn plan_split_real<F: Family, P: Policy>(
        &self,
        layout: &Layout,
        input: *mut T,
        output: *mut T,
        classes: (AlignmentClass, AlignmentClass),
    ) -> Result<Plan<T, F, P>, FftwError> {
        let flags = self.flags::<P>();
        let guru = GuruDims::new(layout)?;
        let r2c = F::KIND == LayoutFamily::SplitR2c;
        let split_side = if r2c { Side::Output } else { Side::Input };
        // Gaps are positive (`Layout::with_storage`). FFTW's split r2c is always forward and
        // its split c2r always backward, so the parts are never swapped, unlike split c2c.
        let gap = layout
            .gap(split_side)
            .expect("`check_storage` accepted only layouts with a split side")
            .unsigned_abs();

        let raw = with_fftw(|state| {
            state.version_check()?;
            let _settings = SettingsGuard::<T>::new(state, self.time_limit);
            // SAFETY: the FFTW lock is held. `guru` holds `rank` and `howmany_rank`
            // dimensions with non-negative strides, in reals on both sides, with the logical
            // real extent n of the last transform dimension. By this function's contract the
            // buffers cover the spans of these dimensions, which use n / 2 + 1 on the split
            // side and include its gap, so the split side's real part at the buffer's first
            // real and its imaginary part `gap` reals further on lie inside it, as does every
            // element FFTW addresses from them.
            let raw = unsafe {
                if r2c {
                    T::plan_guru64_split_dft_r2c(
                        guru.rank,
                        guru.dims.as_ptr(),
                        guru.howmany_rank,
                        guru.batch.as_ptr(),
                        input,
                        output,
                        output.wrapping_add(gap),
                        flags,
                    )
                } else {
                    T::plan_guru64_split_dft_c2r(
                        guru.rank,
                        guru.dims.as_ptr(),
                        guru.howmany_rank,
                        guru.batch.as_ptr(),
                        input,
                        input.wrapping_add(gap),
                        output,
                        flags,
                    )
                }
            };
            self.plan_or_error(raw, flags, layout, None)
        })?;

        let direction = if r2c {
            Direction::Forward
        } else {
            Direction::Backward
        };
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

    /// The plan FFTW returned, or [`FftwError::PlanCreationFailed`] with `hint` for NULL.
    fn plan_or_error(
        &self,
        raw: *mut T::RawPlan,
        flags: c_uint,
        layout: &Layout,
        hint: Option<&'static str>,
    ) -> Result<NonNull<T::RawPlan>, FftwError> {
        NonNull::new(raw).ok_or_else(|| FftwError::PlanCreationFailed {
            rigor: self.rigor,
            flags,
            layout: layout.describe(),
            hint,
        })
    }
}

/// Check the kinds of a real-to-real plan before FFTW is called, each failure an
/// [`FftwError::InvalidKind`]: one kind per transform dimension, and `Redft00`, the DCT-I,
/// only on a dimension of extent 2 or more, as it is not defined for n = 1 (design, Section 3
/// item 11).
fn check_kinds(layout: &Layout, kinds: &[R2rKind]) -> Result<(), FftwError> {
    let rank = layout.dims().len();
    if kinds.len() != rank {
        return Err(FftwError::InvalidKind {
            reason: format!(
                "the layout has {rank} transform dimensions, and a real-to-real plan needs one \
                 kind for each, in their order, but {} kinds were given",
                kinds.len()
            ),
        });
    }
    for (index, (dim, kind)) in layout.dims().iter().zip(kinds).enumerate() {
        if *kind == R2rKind::Redft00 && dim.n < 2 {
            return Err(FftwError::InvalidKind {
                reason: format!(
                    "{kind} (DCT-I) is not defined for n = 1, the extent of transform dimension \
                     {index}"
                ),
            });
        }
    }
    Ok(())
}

/// A layout's dimensions as FFTW's guru64 interface takes them.
struct GuruDims {
    dims: Vec<fftw_sys::fftw_iodim64>,
    batch: Vec<fftw_sys::fftw_iodim64>,
    rank: c_int,
    howmany_rank: c_int,
}

impl GuruDims {
    fn new(layout: &Layout) -> Result<Self, FftwError> {
        let dims = iodims(layout.dims());
        let batch = iodims(layout.batch());
        let too_many = |what: &str| FftwError::InvalidLayout {
            dimension: "layout".to_string(),
            reason: format!("FFTW takes at most {} {what} dimensions", c_int::MAX),
        };
        let rank = c_int::try_from(dims.len()).map_err(|_| too_many("transform"))?;
        let howmany_rank = c_int::try_from(batch.len()).map_err(|_| too_many("batch"))?;
        Ok(Self {
            dims,
            batch,
            rank,
            howmany_rank,
        })
    }
}

/// Scratch for planning: zeroed reals at `PLAN_ALIGNMENT`, long enough for a span of elements
/// of one or more reals starting at an alignment class.
struct Scratch<T> {
    data: AVec<T, ConstAlign<PLAN_ALIGNMENT>>,
    offset: usize,
}

impl<T: FftwReal> Scratch<T> {
    /// Scratch for `span` elements of `reals` reals each (2 for complex elements, 1 for
    /// reals) at `class`, which is a multiple of the alignment of the element and therefore
    /// of the size of `T`. The caller has checked that the span fits in `usize` bytes.
    fn new(span: usize, reals: usize, class: AlignmentClass) -> Self {
        let offset = class.offset() / size_of::<T>();
        let len = span
            .checked_mul(reals)
            .and_then(|reals| reals.checked_add(offset))
            .expect("the scratch size was checked against `max_scratch`");
        let data = AVec::from_iter(PLAN_ALIGNMENT, std::iter::repeat_n(T::zero(), len));
        Self { data, offset }
    }

    /// A pointer to the first real, at the requested class.
    fn as_mut_ptr(&mut self) -> *mut T {
        // `offset` reals and the span's reals after it lie inside the allocation.
        self.data.as_mut_ptr().wrapping_add(self.offset)
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

/// Check that the layout is of the family `F` of the planner method, by the storages of its
/// sides (DC28, generalised by DC30): an [`FftwError::InvalidLayout`] naming the `part axis`
/// when a side is split where the family's is not, or the reverse, and naming the `layout`
/// otherwise; the reason names the family and the storages found.
fn check_storage<F: Family>(layout: &Layout) -> Result<(), FftwError> {
    let (expected, found) = (F::KIND, layout.family());
    if expected == found {
        return Ok(());
    }
    let dimension = if expected.split_sides() == found.split_sides() {
        "layout"
    } else {
        "part axis"
    };
    let (input, output) = expected.storage_names();
    let (found_input, found_output) = layout.storage();
    let remedy = match expected {
        LayoutFamily::C2c => "make such layouts from two complex arrays, or with `Layout::new`",
        LayoutFamily::SplitC2c => {
            "make split layouts with `Layout::new_split`, `Layout::split_along_axis` or their \
             companions"
        }
        LayoutFamily::R2c => {
            "make such layouts from a real input and a complex output array, with \
             `Layout::r2c_inplace`, or with `Layout::with_storage`"
        }
        LayoutFamily::C2r => {
            "make such layouts from a complex input and a real output array, with \
             `Layout::c2r_inplace`, or with `Layout::with_storage`"
        }
        LayoutFamily::R2r => {
            "make such layouts from two real arrays, from one in place, or with \
             `Layout::with_storage`"
        }
        LayoutFamily::SplitR2c => {
            "make such layouts with `Layout::split_r2c_along_axis`, \
             `Layout::split_r2c_over_axes` or `Layout::with_storage`"
        }
        LayoutFamily::SplitC2r => {
            "make such layouts with `Layout::split_c2r_along_axis`, \
             `Layout::split_c2r_over_axes` or `Layout::with_storage`"
        }
    };
    let methods = match found {
        LayoutFamily::C2c => "the `c2c` planner methods",
        LayoutFamily::SplitC2c => "the `split_c2c` planner methods",
        LayoutFamily::R2c => "the `r2c` planner methods",
        LayoutFamily::C2r => "the `c2r` planner methods",
        LayoutFamily::R2r => "the `r2r` planner methods",
        LayoutFamily::SplitR2c => "the `split_r2c` planner methods",
        LayoutFamily::SplitC2r => "the `split_c2r` planner methods",
    };
    Err(FftwError::InvalidLayout {
        dimension: dimension.to_string(),
        reason: format!(
            "a {} plan needs a layout of {input} input and {output} output storage, but was \
             given a {} layout, of {found_input} input and {found_output} output storage; \
             plan that layout with {methods}, or {remedy}",
            expected.name(),
            found.name()
        ),
    })
}

/// Check that a preserving complex-to-real plan, split or not, has one transform dimension:
/// FFTW has no input-preserving algorithm for more (design, Section 3 item 5), and returns
/// NULL.
fn check_preserving_c2r<F: Family, P: Policy>(layout: &Layout) -> Result<(), FftwError> {
    let preserving = P::FLAGS & fftw_sys::FFTW_PRESERVE_INPUT != 0;
    let rank = layout.dims().len();
    let c2r = matches!(F::KIND, LayoutFamily::C2r | LayoutFamily::SplitC2r);
    if c2r && preserving && rank > 1 {
        return Err(FftwError::PreserveInputUnsupported { rank });
    }
    Ok(())
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
    /// [`FftwError::InvalidLayout`] naming the `part axis` for a layout of split storage,
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
        check_storage::<C2c>(layout)?;
        check_placement::<S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.c2c_on_buffers(layout, direction, input, output)
    }
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// An in-place complex-to-complex plan for `layout`, planned on scratch.
    ///
    /// Returns [`FftwError::InvalidLayout`] naming the `part axis` for a layout of split
    /// storage, [`FftwError::PlacementMismatch`] for an out-of-place layout and
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
        check_storage::<C2c>(layout)?;
        check_placement::<InPlace>(layout)?;
        data.check_side(layout, Side::Output)?;
        // In place the input and output spans are equal, so one check covers both sides.
        let data = data.check_side_mut(layout, Side::Input)?;
        self.c2c_on_buffers(layout, direction, data, data)
    }
}

impl<T: FftwReal, S: PolicyFor<SplitC2c>> Planner<T, S> {
    /// An out-of-place complex-to-complex plan on split storage for `layout`, planned on
    /// scratch.
    ///
    /// `layout` is a layout of split storage, with gaps ([`Layout::new_split`],
    /// [`Layout::split_along_axis`] and their companions); its dimensions, gaps and spans
    /// count reals. The plan is [`Preserving`](super::Preserving), or
    /// [`Destroying`](super::Destroying) after [`may_destroy_input`](Planner::may_destroy_input),
    /// and executes on buffers of reals. Returns [`FftwError::InvalidLayout`] naming the
    /// `part axis` for a layout of interleaved storage, without gaps,
    /// [`FftwError::PlacementMismatch`] for an in-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch). The scratch holds each side's span in reals,
    /// with the real part of its first element at the requested
    /// [`alignment`](Planner::alignment) class, which must be a multiple of the alignment of
    /// `T`.
    ///
    /// FFTW's split planner has no sign and always plans a forward transform; for
    /// [`Direction::Backward`] the plan swaps the real and imaginary parts on both sides, at
    /// planning and at every execution (design, Section 3 item 8).
    ///
    /// Split plans can use fewer SIMD stages than interleaved plans of the same transform,
    /// since FFTW's SIMD code needs interleaved storage (design, Sections 3 item 7 and 4.5); an
    /// interleaved copy and an interleaved plan can be faster. The benchmarks of the FFTW
    /// rework (task F8b) compare the two.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// // Three transforms of length 8, real plane of 24 reals and then the imaginary plane.
    /// let layout = Layout::new_split(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[Dim { n: 3, is: 8, os: 8 }],
    ///     (24, 24),
    ///     Placement::OutOfPlace,
    /// )?;
    /// let plan = Planner::<f64>::new().split_c2c(&layout, Direction::Forward)?;
    ///
    /// let mut input = rlst::rlst_dynamic_array!(f64, [48] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst::rlst_dynamic_array!(f64, [48] | rlst::FFTW_ALIGNED);
    /// // The first transform's input is 1 + 2i at index 0.
    /// input.data_mut().unwrap()[0] = 1.0;
    /// input.data_mut().unwrap()[24] = 2.0;
    /// plan.execute(input.data().unwrap(), output.data_mut().unwrap())?;
    /// let output = output.data().unwrap();
    /// assert!((0..8).all(|k| output[k] == 1.0 && output[24 + k] == 2.0));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2c(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, SplitC2c, S::Policy>, FftwError> {
        self.split_c2c_on_scratch(layout, direction)
    }

    /// An out-of-place complex-to-complex plan on split storage for `layout`, planned on the
    /// caller's buffers.
    ///
    /// As [`split_c2c`](Planner::split_c2c), but the buffers, slices of reals or real RLST
    /// arrays and views, are checked against the layout as at execution
    /// ([`FftwError::LayoutMismatch`]), and the plan is made for the alignment classes of
    /// their real parts. No scratch is allocated, so this plans layouts whose span exceeds
    /// [`max_scratch`](Planner::max_scratch).
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Direction, Layout, Planner};
    ///
    /// // Box 1 of a parent of four split boxes [140, 9, 2]; a box is 20160 bytes, a multiple of
    /// // 64, so every box is at class 0.
    /// let mut input = rlst_dynamic_array!(f64, [140, 9, 2, 4] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [140, 9, 2, 4] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_along_axis(
    ///     &input.r().slice::<3>(3, 1),
    ///     2,
    ///     &output.r().slice::<3>(3, 1),
    ///     2,
    ///     0,
    /// )?;
    /// let plan = Planner::<f64>::new().split_c2c_on(
    ///     &mut input.r_mut().slice::<3>(3, 1),
    ///     &mut output.r_mut().slice::<3>(3, 1),
    ///     &layout,
    ///     Direction::Backward,
    /// )?;
    /// // The plan runs on every box.
    /// for k in 0..4 {
    ///     plan.execute(&input.r().slice::<3>(3, k), &mut output.r_mut().slice::<3>(3, k))?;
    /// }
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2c_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, SplitC2c, S::Policy>, FftwError>
    where
        I: FftBufferMut<T> + ?Sized,
        O: FftBufferMut<T> + ?Sized,
    {
        check_storage::<SplitC2c>(layout)?;
        check_placement::<S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.split_c2c_on_buffers(layout, direction, input, output)
    }
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// An in-place complex-to-complex plan on split storage for `layout`, planned on scratch.
    ///
    /// As [`split_c2c`](Planner::split_c2c), for an in-place layout with equal gaps:
    /// returns [`FftwError::InvalidLayout`] naming the `part axis` for a layout of
    /// interleaved storage, [`FftwError::PlacementMismatch`] for an out-of-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch). The input-policy requests do not apply in
    /// place, so this method is only available on a planner without one.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// // Interleaved storage read as split: the imaginary part follows the real part.
    /// let layout =
    ///     Layout::new_split(&[Dim { n: 8, is: 2, os: 2 }], &[], (1, 1), Placement::InPlace)?;
    /// let plan = Planner::<f64>::new().split_c2c_inplace(&layout, Direction::Backward)?;
    /// # let _ = plan;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2c_inplace(
        &self,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, SplitC2c, InPlace>, FftwError> {
        self.split_c2c_on_scratch(layout, direction)
    }

    /// An in-place complex-to-complex plan on split storage for `layout`, planned on the
    /// caller's buffer.
    ///
    /// The buffer is a slice of reals or a real RLST array or view, checked against the
    /// layout as at execution ([`FftwError::LayoutMismatch`]), and the plan is made for the
    /// alignment class of its real part. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffer untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites it.**
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Direction, Layout, Planner};
    ///
    /// let mut data = rlst_dynamic_array!(f32, [64, 2] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_along_axis_inplace(&data, 1, 0)?;
    /// let plan =
    ///     Planner::<f32>::new().split_c2c_inplace_on(&mut data, &layout, Direction::Forward)?;
    /// plan.execute(&mut data)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2c_inplace_on<D>(
        &self,
        data: &mut D,
        layout: &Layout,
        direction: Direction,
    ) -> Result<Plan<T, SplitC2c, InPlace>, FftwError>
    where
        D: FftBufferMut<T> + ?Sized,
    {
        check_storage::<SplitC2c>(layout)?;
        check_placement::<InPlace>(layout)?;
        data.check_side(layout, Side::Output)?;
        // In place the input and output spans are equal, so one check covers both sides.
        let data = data.check_side_mut(layout, Side::Input)?;
        self.split_c2c_on_buffers(layout, direction, data, data)
    }
}

impl<T: FftwReal, S: PolicyFor<R2c>> Planner<T, S> {
    /// An out-of-place real-to-complex plan for `layout`, planned on scratch.
    ///
    /// `layout` has real input and complex output storage: made from a real input and a
    /// complex output array by [`Layout::along_axis`] or [`Layout::over_axes`], or by
    /// [`Layout::with_storage`]. Its last transform dimension is halved: `n` reals in,
    /// `n / 2 + 1` complex values out. The plan is always forward, and is
    /// [`Preserving`](super::Preserving), or [`Destroying`](super::Destroying) after
    /// [`may_destroy_input`](Planner::may_destroy_input). Returns
    /// [`FftwError::InvalidLayout`] for a layout of another family (naming the `part axis` for
    /// a split one, the `layout` otherwise), [`FftwError::PlacementMismatch`] for an in-place
    /// layout and [`FftwError::ScratchTooLarge`] if the scratch, the input span in reals and
    /// the output span in complex values, would exceed [`max_scratch`](Planner::max_scratch).
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// // A real [6, 4] array into a complex [4, 4] one, over both axes, axis 0 halved.
    /// let mut input = rlst_dynamic_array!(f64, [6, 4] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(c64, [4, 4] | rlst::FFTW_ALIGNED);
    /// input.fill_from_seed_equally_distributed(0);
    /// let layout = Layout::over_axes(&input, &output, &[1, 0])?;
    /// let plan = Planner::<f64>::new().r2c(&layout)?;
    /// plan.execute(&input, &mut output)?;
    ///
    /// // Frequency 0 is the sum of the input.
    /// let sum: f64 = input.iter_value().sum();
    /// assert!((output[[0, 0]].re - sum).abs() < 1e-12);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2c(&self, layout: &Layout) -> Result<Plan<T, R2c, S::Policy>, FftwError> {
        self.real_on_scratch(layout)
    }

    /// An out-of-place real-to-complex plan for `layout`, planned on the caller's buffers.
    ///
    /// As [`r2c`](Planner::r2c), but the buffers, a real input and a complex output, slices
    /// or RLST arrays and views, are checked against the layout as at execution
    /// ([`FftwError::LayoutMismatch`]), and the plan is made for their alignment classes. No
    /// scratch is allocated, so this plans layouts whose span exceeds
    /// [`max_scratch`](Planner::max_scratch).
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut input = rlst_dynamic_array!(f32, [16, 3] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst::rlst_dynamic_array!(rlst::c32, [9, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// let plan = Planner::<f32>::new().r2c_on(&mut input, &mut output, &layout)?;
    /// plan.execute(&input, &mut output)?;
    /// # let _ = c64::new(0.0, 0.0);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2c_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
    ) -> Result<Plan<T, R2c, S::Policy>, FftwError>
    where
        I: FftBufferMut<T> + ?Sized,
        O: FftBufferMut<Complex<T>> + ?Sized,
    {
        check_storage::<R2c>(layout)?;
        check_placement::<S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.real_on_buffers(layout, input, output.cast::<T>())
    }
}

impl<T: FftwReal, S: PolicyFor<C2r>> Planner<T, S> {
    /// An out-of-place complex-to-real plan for `layout`, planned on scratch.
    ///
    /// **A complex-to-real plan destroys its input unless
    /// [`preserve_input`](Planner::preserve_input) is given**, as in FFTW, unlike the plans
    /// of every other family: the default plan is [`Destroying`](super::Destroying) and
    /// executes with `&mut input`. With `preserve_input()` the plan is
    /// [`Preserving`](super::Preserving), over one transform dimension only; over more,
    /// planning returns [`FftwError::PreserveInputUnsupported`] before FFTW is called, as
    /// FFTW has no input-preserving algorithm for them.
    ///
    /// `layout` has complex input and real output storage: made from a complex input and a
    /// real output array by [`Layout::along_axis`] or [`Layout::over_axes`], or by
    /// [`Layout::with_storage`]. Its last transform dimension is halved: `n / 2 + 1` complex
    /// values in, `n` reals out. The input is the non-redundant half of a Hermitian spectrum;
    /// FFTW assumes the symmetry and ignores the imaginary parts that it forces to zero. The
    /// plan is always backward. Returns [`FftwError::InvalidLayout`] for a layout of another
    /// family, [`FftwError::PlacementMismatch`] for an in-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch).
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{FftwError, Layout, Planner};
    ///
    /// let mut spectrum = rlst_dynamic_array!(c64, [5, 3] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
    /// spectrum[[0, 1]] = c64::new(8.0, 0.0);
    /// let layout = Layout::along_axis(&spectrum, &output, 0)?;
    ///
    /// // The default plan destroys its input.
    /// let plan = Planner::<f64>::new().c2r(&layout)?;
    /// plan.execute(&mut spectrum, &mut output)?;
    ///
    /// // A preserving plan exists over one transform dimension only.
    /// let preserving = Planner::<f64>::new().preserve_input().c2r(&layout)?;
    /// preserving.execute(&spectrum, &mut output)?;
    /// let spectrum = rlst_dynamic_array!(c64, [3, 5] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(f64, [3, 8] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::over_axes(&spectrum, &output, &[0, 1])?;
    /// assert_eq!(
    ///     Planner::<f64>::new().preserve_input().c2r(&layout).unwrap_err(),
    ///     FftwError::PreserveInputUnsupported { rank: 2 }
    /// );
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn c2r(&self, layout: &Layout) -> Result<Plan<T, C2r, S::Policy>, FftwError> {
        self.real_on_scratch(layout)
    }

    /// An out-of-place complex-to-real plan for `layout`, planned on the caller's buffers.
    ///
    /// As [`c2r`](Planner::c2r), but the buffers, a complex input and a real output, slices
    /// or RLST arrays and views, are checked against the layout as at execution
    /// ([`FftwError::LayoutMismatch`]), and the plan is made for their alignment classes. No
    /// scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut input = rlst_dynamic_array!(c64, [9] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [17] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// let plan = Planner::<f64>::new().c2r_on(&mut input, &mut output, &layout)?;
    /// plan.execute(&mut input, &mut output)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2r_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
    ) -> Result<Plan<T, C2r, S::Policy>, FftwError>
    where
        I: FftBufferMut<Complex<T>> + ?Sized,
        O: FftBufferMut<T> + ?Sized,
    {
        check_storage::<C2r>(layout)?;
        check_placement::<S::Policy>(layout)?;
        check_preserving_c2r::<C2r, S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.real_on_buffers(layout, input.cast::<T>(), output)
    }
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// An in-place real-to-complex plan for `layout`, planned on scratch.
    ///
    /// `layout` is an in-place real-to-complex layout, made by [`Layout::r2c_inplace`] or
    /// [`Layout::with_storage`]: the buffer is the complex output, whose reals hold the
    /// input, padded along the halved axis. Returns [`FftwError::InvalidLayout`] for a layout
    /// of another family, [`FftwError::PlacementMismatch`] for an out-of-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch, the complex span, would exceed
    /// [`max_scratch`](Planner::max_scratch). The input-policy requests do not apply in
    /// place, so this method is only available on a planner without one.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut data = rlst_dynamic_array!(c64, [9, 4] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::r2c_inplace(&data, &[0], 16)?;
    /// let plan = Planner::<f64>::new().r2c_inplace(&layout)?;
    /// plan.execute(&mut data)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2c_inplace(&self, layout: &Layout) -> Result<Plan<T, R2c, InPlace>, FftwError> {
        self.real_on_scratch(layout)
    }

    /// An in-place real-to-complex plan for `layout`, planned on the caller's buffer.
    ///
    /// The buffer is the complex side, a slice of complex values or a complex RLST array or
    /// view, checked against the layout's complex side as at execution
    /// ([`FftwError::LayoutMismatch`]); the real side lies inside it. The plan is made for
    /// the buffer's alignment class. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffer untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites it.**
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut data = rlst_dynamic_array!(c64, [9, 4] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::r2c_inplace(&data, &[0], 17)?;
    /// let plan = Planner::<f64>::new().r2c_inplace_on(&mut data, &layout)?;
    /// plan.execute(&mut data)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2c_inplace_on<D>(
        &self,
        data: &mut D,
        layout: &Layout,
    ) -> Result<Plan<T, R2c, InPlace>, FftwError>
    where
        D: FftBufferMut<Complex<T>> + ?Sized,
    {
        check_storage::<R2c>(layout)?;
        check_placement::<InPlace>(layout)?;
        // The buffer is the complex side, the output, whose span holds the real side.
        let data = data.check_side_mut(layout, Side::Output)?.cast::<T>();
        self.real_on_buffers(layout, data, data)
    }

    /// An in-place complex-to-real plan for `layout`, planned on scratch.
    ///
    /// `layout` is an in-place complex-to-real layout, made by [`Layout::c2r_inplace`] or
    /// [`Layout::with_storage`]: the buffer is the complex input, whose reals receive the
    /// output, padded along the halved axis. In place the input is overwritten anyway.
    /// Returns [`FftwError::InvalidLayout`] for a layout of another family,
    /// [`FftwError::PlacementMismatch`] for an out-of-place layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch). The input-policy requests do not apply in
    /// place, so this method is only available on a planner without one.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut data = rlst_dynamic_array!(c64, [4, 9] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::c2r_inplace(&data, &[1, 0], 7)?;
    /// let plan = Planner::<f64>::new().c2r_inplace(&layout);
    /// // The halved axis, axis 0, has extent 4 = 7 / 2 + 1 and is contiguous.
    /// assert!(plan.is_ok());
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2r_inplace(&self, layout: &Layout) -> Result<Plan<T, C2r, InPlace>, FftwError> {
        self.real_on_scratch(layout)
    }

    /// An in-place complex-to-real plan for `layout`, planned on the caller's buffer.
    ///
    /// As [`r2c_inplace_on`](Planner::r2c_inplace_on), for the complex input of an in-place
    /// complex-to-real layout.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffer untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites it.**
    ///
    /// ```
    /// use rlst::{c32, rlst_dynamic_array};
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut data = rlst_dynamic_array!(c32, [5] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::c2r_inplace(&data, &[0], 9)?;
    /// let plan = Planner::<f32>::new().c2r_inplace_on(&mut data, &layout)?;
    /// plan.execute(&mut data)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2r_inplace_on<D>(
        &self,
        data: &mut D,
        layout: &Layout,
    ) -> Result<Plan<T, C2r, InPlace>, FftwError>
    where
        D: FftBufferMut<Complex<T>> + ?Sized,
    {
        check_storage::<C2r>(layout)?;
        check_placement::<InPlace>(layout)?;
        // The buffer is the complex side, the input, whose span holds the real side.
        let data = data.check_side_mut(layout, Side::Input)?.cast::<T>();
        self.real_on_buffers(layout, data, data)
    }
}

impl<T: FftwReal, S: PolicyFor<R2r>> Planner<T, S> {
    /// An out-of-place real-to-real plan for `layout` with `kinds`, planned on scratch.
    ///
    /// `layout` has real input and real output storage: made from two real arrays by
    /// [`Layout::along_axis`] or [`Layout::over_axes`], or by [`Layout::with_storage`].
    /// `kinds[i]` is the [`R2rKind`] of `layout.dims()[i]`, so for a layout made from arrays
    /// of `axes[i]`, whatever the memory order of the arrays. The plan records the kinds
    /// ([`Plan::kinds`]) and is [`Preserving`](super::Preserving), or
    /// [`Destroying`](super::Destroying) after [`may_destroy_input`](Planner::may_destroy_input).
    ///
    /// A preserving plan passes FFTW `FFTW_PRESERVE_INPUT`, with an HC2R kind too, which FFTW
    /// destroys by default. If FFTW has no input-preserving algorithm for the problem, it
    /// returns no plan, and planning returns [`FftwError::PlanCreationFailed`] whose `hint`
    /// names the remedy, `may_destroy_input()`.
    ///
    /// Before FFTW is called: [`FftwError::InvalidLayout`] for a layout of another family,
    /// [`FftwError::PlacementMismatch`] for an in-place layout, [`FftwError::InvalidKind`]
    /// for a number of kinds other than the number of transform dimensions or for
    /// [`R2rKind::Redft00`] on a dimension of extent 1, and [`FftwError::ScratchTooLarge`] if
    /// the scratch would exceed [`max_scratch`](Planner::max_scratch).
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{FftwError, Layout, Planner, R2rKind};
    ///
    /// // A DCT-II along each column of a [8, 3] array, and the DCT-III back.
    /// let mut data = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let mut spectrum = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let mut result = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
    /// data.fill_from_seed_equally_distributed(0);
    /// let layout = Layout::along_axis(&data, &spectrum, 0)?;
    /// let forward = Planner::<f64>::new().r2r(&layout, &[R2rKind::Redft10])?;
    /// let backward = Planner::<f64>::new().r2r(&layout, &[R2rKind::Redft01])?;
    /// forward.execute(&data, &mut spectrum)?;
    /// backward.execute(&spectrum, &mut result)?;
    /// // The round trip multiplies by the logical size 2 n = 16.
    /// assert_eq!(forward.logical_size(), 16);
    /// assert!((result[[3, 2]] - 16.0 * data[[3, 2]]).abs() < 1e-12);
    ///
    /// // One kind per transform dimension.
    /// let error = Planner::<f64>::new().r2r(&layout, &[R2rKind::Redft10, R2rKind::Dht]);
    /// assert!(matches!(error, Err(FftwError::InvalidKind { .. })));
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn r2r(
        &self,
        layout: &Layout,
        kinds: &[R2rKind],
    ) -> Result<Plan<T, R2r, S::Policy>, FftwError> {
        self.r2r_on_scratch(layout, kinds)
    }

    /// An out-of-place real-to-real plan for `layout` with `kinds`, planned on the caller's
    /// buffers.
    ///
    /// As [`r2r`](Planner::r2r), but the buffers, slices of reals or real RLST arrays and
    /// views, are checked against the layout as at execution ([`FftwError::LayoutMismatch`]),
    /// and the plan is made for their alignment classes. No scratch is allocated, so this
    /// plans layouts whose span exceeds [`max_scratch`](Planner::max_scratch).
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner, R2rKind};
    ///
    /// let mut input = rlst_dynamic_array!(f32, [17] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f32, [17] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// let plan = Planner::<f32>::new().r2r_on(&mut input, &mut output, &layout, &[R2rKind::Dht])?;
    /// plan.execute(&input, &mut output)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2r_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
        kinds: &[R2rKind],
    ) -> Result<Plan<T, R2r, S::Policy>, FftwError>
    where
        I: FftBufferMut<T> + ?Sized,
        O: FftBufferMut<T> + ?Sized,
    {
        check_storage::<R2r>(layout)?;
        check_placement::<S::Policy>(layout)?;
        check_kinds(layout, kinds)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.r2r_on_buffers(layout, kinds, input, output)
    }
}

impl<T: FftwReal> Planner<T, DefaultRequest> {
    /// An in-place real-to-real plan for `layout` with `kinds`, planned on scratch.
    ///
    /// As [`r2r`](Planner::r2r), for an in-place real-to-real layout, made from one real
    /// array by [`Layout::along_axis_inplace`] or [`Layout::over_axes_inplace`], or by
    /// [`Layout::with_storage`]. Returns [`FftwError::InvalidLayout`] for a layout of another
    /// family, [`FftwError::PlacementMismatch`] for an out-of-place layout,
    /// [`FftwError::InvalidKind`] for kinds that do not fit the layout and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch). The input-policy requests do not apply in
    /// place, so this method is only available on a planner without one.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner, R2rKind};
    ///
    /// // A DCT-I over both axes of a [5, 4] array, in place, as RLST's Chebyshev module uses.
    /// let mut data = rlst_dynamic_array!(f64, [5, 4] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::over_axes_inplace(&data, &[0, 1])?;
    /// let plan = Planner::<f64>::new().r2r_inplace(&layout, &[R2rKind::Redft00; 2])?;
    /// plan.execute(&mut data)?;
    /// assert_eq!(plan.logical_size(), 8 * 6);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2r_inplace(
        &self,
        layout: &Layout,
        kinds: &[R2rKind],
    ) -> Result<Plan<T, R2r, InPlace>, FftwError> {
        self.r2r_on_scratch(layout, kinds)
    }

    /// An in-place real-to-real plan for `layout` with `kinds`, planned on the caller's
    /// buffer.
    ///
    /// The buffer is a slice of reals or a real RLST array or view, checked against the
    /// layout as at execution ([`FftwError::LayoutMismatch`]), and the plan is made for the
    /// buffer's alignment class. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffer untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites it.**
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner, R2rKind};
    ///
    /// let mut data = rlst_dynamic_array!(f64, [6, 2] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis_inplace(&data, 0)?;
    /// let plan =
    ///     Planner::<f64>::new().r2r_inplace_on(&mut data, &layout, &[R2rKind::Rodft11])?;
    /// plan.execute(&mut data)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2r_inplace_on<D>(
        &self,
        data: &mut D,
        layout: &Layout,
        kinds: &[R2rKind],
    ) -> Result<Plan<T, R2r, InPlace>, FftwError>
    where
        D: FftBufferMut<T> + ?Sized,
    {
        check_storage::<R2r>(layout)?;
        check_placement::<InPlace>(layout)?;
        check_kinds(layout, kinds)?;
        data.check_side(layout, Side::Output)?;
        // In place the input and output spans are equal, so one check covers both sides.
        let data = data.check_side_mut(layout, Side::Input)?;
        self.r2r_on_buffers(layout, kinds, data, data)
    }
}

impl<T: FftwReal, S: PolicyFor<SplitR2c>> Planner<T, S> {
    /// A real-to-complex plan into split complex storage for `layout`, planned on scratch.
    ///
    /// `layout` has real input and split output storage, made by
    /// [`Layout::split_r2c_along_axis`], [`Layout::split_r2c_over_axes`] or
    /// [`Layout::with_storage`]; its dimensions, gap and spans count reals, and its last
    /// transform dimension is halved: `n` reals in, `n / 2 + 1` complex values out. The plan
    /// is always forward, executes on buffers of reals on both sides, and is
    /// [`Preserving`](super::Preserving), or [`Destroying`](super::Destroying) after
    /// [`may_destroy_input`](Planner::may_destroy_input). There are no in-place split r2c
    /// plans (DC33). Returns [`FftwError::InvalidLayout`] for a layout of another family
    /// (naming the `part axis` when its sides are split differently), and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch).
    ///
    /// Split plans can use fewer of FFTW's SIMD stages than interleaved ones (design,
    /// Sections 3 item 7 and 4.5); results of the two agree to round-off.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// // nd-fmm's planes: [n, rings] reals into planes [n / 2 + 1, rings, 2].
    /// let mut input = rlst_dynamic_array!(f64, [140, 9] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [71, 9, 2] | rlst::FFTW_ALIGNED);
    /// input.fill_from_seed_equally_distributed(0);
    /// let layout = Layout::split_r2c_along_axis(&input, &output, 2, 0)?;
    /// let plan = Planner::<f64>::new().split_r2c(&layout)?;
    /// plan.execute(&input, &mut output)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_r2c(&self, layout: &Layout) -> Result<Plan<T, SplitR2c, S::Policy>, FftwError> {
        self.split_real_on_scratch(layout)
    }

    /// A real-to-complex plan into split complex storage for `layout`, planned on the
    /// caller's buffers.
    ///
    /// As [`split_r2c`](Planner::split_r2c), but the buffers, slices of reals or real RLST
    /// arrays and views, are checked against the layout as at execution
    /// ([`FftwError::LayoutMismatch`]), and the plan is made for the classes of their first
    /// reals. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut input = rlst_dynamic_array!(f32, [16] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f32, [2, 9] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_r2c_along_axis(&input, &output, 0, 0)?;
    /// let plan = Planner::<f32>::new().split_r2c_on(&mut input, &mut output, &layout)?;
    /// plan.execute(&input, &mut output)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_r2c_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
    ) -> Result<Plan<T, SplitR2c, S::Policy>, FftwError>
    where
        I: FftBufferMut<T> + ?Sized,
        O: FftBufferMut<T> + ?Sized,
    {
        check_storage::<SplitR2c>(layout)?;
        check_placement::<S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.split_real_on_buffers(layout, input, output)
    }
}

impl<T: FftwReal, S: PolicyFor<SplitC2r>> Planner<T, S> {
    /// A complex-to-real plan from split complex storage for `layout`, planned on scratch.
    ///
    /// **As for [`c2r`](Planner::c2r), a split complex-to-real plan destroys its input unless
    /// [`preserve_input`](Planner::preserve_input) is given**: the default plan is
    /// [`Destroying`](super::Destroying); with `preserve_input()` it is
    /// [`Preserving`](super::Preserving), over one transform dimension only, and over more
    /// planning returns [`FftwError::PreserveInputUnsupported`] before FFTW is called.
    ///
    /// `layout` has split input and real output storage, made by
    /// [`Layout::split_c2r_along_axis`], [`Layout::split_c2r_over_axes`] or
    /// [`Layout::with_storage`]: `n / 2 + 1` complex values in along the halved dimension,
    /// `n` reals out. The plan is always backward; there are no in-place split c2r plans
    /// (DC33). Returns [`FftwError::InvalidLayout`] for a layout of another family and
    /// [`FftwError::ScratchTooLarge`] if the scratch would exceed
    /// [`max_scratch`](Planner::max_scratch).
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{FftwError, Layout, Planner};
    ///
    /// let input = rlst_dynamic_array!(f64, [3, 5, 2] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(f64, [3, 8] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_c2r_along_axis(&input, 2, &output, 1)?;
    /// let plan = Planner::<f64>::new().split_c2r(&layout)?;
    /// let preserving = Planner::<f64>::new().preserve_input().split_c2r(&layout)?;
    ///
    /// let over_both = Layout::split_c2r_over_axes(&input, 2, &output, &[0, 1])?;
    /// assert_eq!(
    ///     Planner::<f64>::new().preserve_input().split_c2r(&over_both).unwrap_err(),
    ///     FftwError::PreserveInputUnsupported { rank: 2 }
    /// );
    /// # let _ = (plan, preserving);
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn split_c2r(&self, layout: &Layout) -> Result<Plan<T, SplitC2r, S::Policy>, FftwError> {
        self.split_real_on_scratch(layout)
    }

    /// A complex-to-real plan from split complex storage for `layout`, planned on the
    /// caller's buffers.
    ///
    /// As [`split_c2r`](Planner::split_c2r), but the buffers, slices of reals or real RLST
    /// arrays and views, are checked against the layout as at execution
    /// ([`FftwError::LayoutMismatch`]), and the plan is made for the classes of their first
    /// reals. No scratch is allocated.
    ///
    /// With [`Rigor::Estimate`] or wisdom-only planning FFTW leaves the buffers untouched.
    /// **With [`Rigor::Measure`] and stronger, FFTW overwrites both buffers.**
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// let mut input = rlst_dynamic_array!(f64, [9, 2] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [17] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_c2r_along_axis(&input, 1, &output, 0)?;
    /// let plan = Planner::<f64>::new().split_c2r_on(&mut input, &mut output, &layout)?;
    /// plan.execute(&mut input, &mut output)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2r_on<I, O>(
        &self,
        input: &mut I,
        output: &mut O,
        layout: &Layout,
    ) -> Result<Plan<T, SplitC2r, S::Policy>, FftwError>
    where
        I: FftBufferMut<T> + ?Sized,
        O: FftBufferMut<T> + ?Sized,
    {
        check_storage::<SplitC2r>(layout)?;
        check_placement::<S::Policy>(layout)?;
        check_preserving_c2r::<SplitC2r, S::Policy>(layout)?;
        let input = input.check_side_mut(layout, Side::Input)?;
        let output = output.check_side_mut(layout, Side::Output)?;
        self.split_real_on_buffers(layout, input, output)
    }
}
