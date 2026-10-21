//! Plans, and the markers of their transform family and input policy.

use std::marker::PhantomData;

use super::align::AlignmentClass;
use super::kind::Direction;
use super::layout::Layout;
use super::options::Rigor;
use super::real::FftwReal;

pub(crate) mod sealed {
    /// Seals [`Family`](super::Family).
    pub trait Family {}

    /// Seals [`Policy`](super::Policy).
    pub trait Policy {}
}

/// The transform family of a [`Plan`]: [`C2c`] (r2c, c2r, r2r and the split families follow).
///
/// The trait is sealed.
///
/// ```no_run
/// use rlst::fftw::{C2c, Family};
///
/// fn family<F: Family>() {}
/// family::<C2c>();
/// ```
pub trait Family: sealed::Family {}

/// The input policy of a [`Plan`]: [`Preserving`], [`Destroying`] or [`InPlace`].
///
/// The policy decides how [`Plan::execute`] borrows its input: shared for a preserving plan,
/// mutably for a plan that may overwrite it. The trait is sealed.
///
/// ```no_run
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
/// ```no_run
/// use rlst::fftw::{C2c, Plan, Preserving};
///
/// type ForwardPlan = Plan<f64, C2c, Preserving>;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum C2c {}

/// An out-of-place plan that leaves its input unchanged; it executes with `&input`.
///
/// FFTW is given `FFTW_PRESERVE_INPUT`. A type-level marker; it is never constructed.
///
/// ```no_run
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
/// ```no_run
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
/// ```no_run
/// use rlst::fftw::{C2c, InPlace, Plan};
///
/// type InPlacePlan = Plan<f64, C2c, InPlace>;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum InPlace {}

impl sealed::Family for C2c {}
impl Family for C2c {}

impl sealed::Policy for Preserving {}
impl Policy for Preserving {}
impl sealed::Policy for Destroying {}
impl Policy for Destroying {}
impl sealed::Policy for InPlace {}
impl Policy for InPlace {}

/// An FFTW plan of precision `T`, family `F` and input policy `P`.
///
/// A plan describes a [`Layout`], not particular buffers. It executes on any buffers that fit
/// the layout and have the [`AlignmentClass`]es it was made for, from any thread: `Plan` is
/// `Send` and `Sync`, since FFTW's execution is thread-safe and a plan is not changed by it.
/// Plans are made by a [`Planner`](super::Planner), and destroyed under RLST's FFTW lock when
/// dropped.
///
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
pub struct Plan<T: FftwReal, F: Family, P: Policy> {
    // The raw plan, the layout, the options and the classes follow in F1c.
    _marker: PhantomData<(T, F, P)>,
}

impl<T: FftwReal, F: Family, P: Policy> Plan<T, F, P> {
    /// The normalised layout the plan was made for.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// assert_eq!(plan.layout(), &layout);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn layout(&self) -> &Layout {
        todo!()
    }

    /// The rigor the plan was made with.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner, Rigor};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// assert_eq!(plan.rigor(), Rigor::Estimate);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn rigor(&self) -> Rigor {
        todo!()
    }

    /// The alignment classes of input and output the plan was made for, or `None` for a plan
    /// made with [`Planner::unaligned`](super::Planner::unaligned), which runs on buffers of
    /// any class.
    ///
    /// ```no_run
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
        todo!()
    }
}

impl<T: FftwReal, P: Policy> Plan<T, C2c, P> {
    /// The direction of the transform.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f32>::new().c2c(&layout, Direction::Backward)?;
    /// assert_eq!(plan.direction(), Direction::Backward);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn direction(&self) -> Direction {
        todo!()
    }
}
