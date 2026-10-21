//! Plan descriptions and fingerprints.

use super::plan::{Family, Plan, Policy};
use super::real::FftwReal;

impl<T: FftwReal, F: Family, P: Policy> Plan<T, F, P> {
    /// FFTW's description of the algorithm it chose for the plan (`fftw_sprint_plan`).
    ///
    /// Read under RLST's FFTW lock.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// assert!(!plan.description().is_empty());
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn description(&self) -> String {
        todo!()
    }

    /// A 64-bit FNV-1a hash of [`description`](Plan::description).
    ///
    /// The hash function is the same in every process and build, so equal fingerprints mean
    /// that FFTW chose the same algorithm, for example on every rank of a distributed
    /// computation.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let planner = Planner::<f64>::new();
    /// let first = planner.c2c(&layout, Direction::Forward)?;
    /// let second = planner.c2c(&layout, Direction::Forward)?;
    /// assert_eq!(first.fingerprint(), second.fingerprint());
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn fingerprint(&self) -> u64 {
        todo!()
    }
}
