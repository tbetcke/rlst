//! Plans, and the markers of their transform family and input policy.

use std::ffi::c_uint;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

use super::align::AlignmentClass;
use super::kind::Direction;
use super::layout::{Layout, Placement};
use super::lock::with_fftw;
use super::options::Rigor;
use super::real::FftwReal;

pub(crate) mod sealed {
    use std::ffi::c_uint;

    use crate::fftw::layout::Placement;

    /// Seals [`Family`](super::Family).
    pub trait Family {}

    /// Seals [`Policy`](super::Policy) and carries what the policy means for FFTW.
    pub trait Policy {
        /// The input-policy flag passed to FFTW.
        const FLAGS: c_uint;
        /// The placement of the plans of this policy.
        const PLACEMENT: Placement;
    }
}

/// The transform family of a [`Plan`]: [`C2c`] or [`SplitC2c`] (r2c, c2r, r2r and the other
/// split families follow).
///
/// The trait is sealed.
///
/// ```
/// use rlst::fftw::{C2c, Family, SplitC2c};
///
/// fn family<F: Family>() {}
/// family::<C2c>();
/// family::<SplitC2c>();
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

impl sealed::Family for C2c {}
impl Family for C2c {}

impl sealed::Family for SplitC2c {}
impl Family for SplitC2c {}

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
            _marker: PhantomData,
        }
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
