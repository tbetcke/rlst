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
use super::plan::{C2c, InPlace, Plan, Policy, SplitC2c, split_offsets};
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

    /// The alignment classes to plan scratch for: the planner's, checked to be equal in
    /// place and multiples of `align` bytes, the alignment of the element type `what`.
    fn scratch_classes(
        &self,
        layout: &Layout,
        align: usize,
        what: &str,
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
        for class in [input_class, output_class] {
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

    /// Check the scratch of `layout`, with elements of `size` bytes, against the bound.
    fn check_scratch_size(&self, layout: &Layout, size: usize) -> Result<(), FftwError> {
        // A size that overflows `usize` is above every bound, `usize::MAX` included.
        let elements = match layout.placement() {
            Placement::OutOfPlace => layout.input_span().checked_add(layout.output_span()),
            Placement::InPlace => Some(layout.input_span()),
        };
        let needed = elements.and_then(|elements| elements.checked_mul(size));
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
        check_storage(layout, false)?;
        check_placement::<P>(layout)?;
        let (input_class, output_class) =
            self.scratch_classes(layout, align_of::<Complex<T>>(), "complex type")?;
        self.check_scratch_size(layout, size_of::<Complex<T>>())?;

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
            self.plan_or_error(raw, flags, layout)
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
        check_storage(layout, true)?;
        check_placement::<P>(layout)?;
        let (input_class, output_class) =
            self.scratch_classes(layout, align_of::<T>(), "real type")?;
        self.check_scratch_size(layout, size_of::<T>())?;

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
            .expect("`check_storage` accepted only layouts with gaps");
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
            self.plan_or_error(raw, flags, layout)
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

    /// The plan FFTW returned, or [`FftwError::PlanCreationFailed`] for NULL.
    fn plan_or_error(
        &self,
        raw: *mut T::RawPlan,
        flags: c_uint,
        layout: &Layout,
    ) -> Result<NonNull<T::RawPlan>, FftwError> {
        NonNull::new(raw).ok_or_else(|| FftwError::PlanCreationFailed {
            rigor: self.rigor,
            flags,
            layout: layout.describe(),
        })
    }
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

/// Check that the layout describes the storage of the planner method: split storage, with
/// gaps, for the split methods, interleaved storage otherwise (DC28).
fn check_storage(layout: &Layout, split: bool) -> Result<(), FftwError> {
    let reason = match (split, layout.gaps()) {
        (false, Some((input, output))) => format!(
            "a complex-to-complex plan on interleaved storage was given a layout of split \
             storage, with gaps (input {input}, output {output}); plan it with the `split_c2c` \
             planner methods"
        ),
        (true, None) => "a split complex-to-complex plan was given a layout of interleaved \
                         storage, without gaps; make split layouts with `Layout::new_split`, \
                         `Layout::split_along_axis` or their companions"
            .to_string(),
        _ => return Ok(()),
    };
    Err(FftwError::InvalidLayout {
        dimension: "part axis".to_string(),
        reason,
    })
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
        check_storage(layout, false)?;
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
        check_storage(layout, false)?;
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
        check_storage(layout, true)?;
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
        check_storage(layout, true)?;
        check_placement::<InPlace>(layout)?;
        data.check_side(layout, Side::Output)?;
        // In place the input and output spans are equal, so one check covers both sides.
        let data = data.check_side_mut(layout, Side::Input)?;
        self.split_c2c_on_buffers(layout, direction, data, data)
    }
}
