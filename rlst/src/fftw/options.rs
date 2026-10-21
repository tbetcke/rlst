//! Planning rigor and the input-policy requests of the planner's type state.

use std::fmt;

/// How hard FFTW searches for a fast plan.
///
/// [`Estimate`](Rigor::Estimate), the default, chooses a plan by heuristics without running
/// any transform. The others time candidate plans: they take longer to plan, and their plans,
/// and so their results bit for bit, can differ between runs and between processes. FFTW also
/// uses any wisdom of equal or greater rigor present in the process, so an `Estimate` plan
/// made after a `Measure` plan of the same problem can be the `Measure` plan.
///
/// ```no_run
/// use rlst::fftw::Rigor;
///
/// assert_eq!(Rigor::default(), Rigor::Estimate);
/// assert_eq!(Rigor::Patient.to_string(), "patient");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rigor {
    /// Heuristics only (`FFTW_ESTIMATE`); planning never runs a transform.
    #[default]
    Estimate,
    /// Time a set of candidate plans (`FFTW_MEASURE`).
    Measure,
    /// Time a wider set of candidates (`FFTW_PATIENT`).
    Patient,
    /// Time the widest set of candidates (`FFTW_EXHAUSTIVE`).
    Exhaustive,
}

impl fmt::Display for Rigor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = f;
        todo!()
    }
}

/// The planner's default input policy: each family follows FFTW's default.
///
/// For c2c, the plan preserves its input ([`Preserving`](super::Preserving)). This is the
/// type parameter of [`Planner::new`](super::Planner::new); it is never constructed.
///
/// ```no_run
/// use rlst::fftw::{DefaultRequest, Planner};
///
/// let planner: Planner<f64, DefaultRequest> = Planner::new();
/// # let _ = planner;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum DefaultRequest {}

/// A request that the plan preserves its input, made with
/// [`Planner::preserve_input`](super::Planner::preserve_input).
///
/// For c2c this is FFTW's default as well. It is a type parameter of the planner and is never
/// constructed.
///
/// ```no_run
/// use rlst::fftw::{Planner, PreserveRequest};
///
/// let planner: Planner<f64, PreserveRequest> = Planner::new().preserve_input();
/// # let _ = planner;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum PreserveRequest {}

/// A request that the plan may overwrite its input, made with
/// [`Planner::may_destroy_input`](super::Planner::may_destroy_input).
///
/// Plans made with it are [`Destroying`](super::Destroying) and take their input mutably. It
/// is a type parameter of the planner and is never constructed.
///
/// ```no_run
/// use rlst::fftw::{DestroyRequest, Planner};
///
/// let planner: Planner<f64, DestroyRequest> = Planner::new().may_destroy_input();
/// # let _ = planner;
/// ```
#[derive(Clone, Copy, Debug)]
pub enum DestroyRequest {}

pub(crate) mod sealed {
    use crate::fftw::plan::{C2c, Destroying, Policy, Preserving};

    use super::{DefaultRequest, DestroyRequest, PreserveRequest};

    /// Maps an input-policy request and a transform family to the policy of the plan.
    pub trait PolicyFor<F> {
        /// The policy of a plan of family `F` made under this request.
        type Policy: Policy;
    }

    impl PolicyFor<C2c> for DefaultRequest {
        type Policy = Preserving;
    }

    impl PolicyFor<C2c> for PreserveRequest {
        type Policy = Preserving;
    }

    impl PolicyFor<C2c> for DestroyRequest {
        type Policy = Destroying;
    }
}
