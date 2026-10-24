//! FFTW's threads (feature `fftw_threads`), and how RLST's FFTW calls are serialised.
//!
//! # One lock for RLST's FFTW calls
//!
//! FFTW's planner, plan destruction, wisdom and global settings must be used from one thread
//! at a time; only plan execution is thread-safe. Every FFTW call RLST makes other than
//! execution runs under one lock in RLST, so RLST's planners, plans and wisdom functions can
//! be used from any thread. The lock has two limits:
//!
//! - it does not cover other libraries in the process that call FFTW themselves;
//! - it does not cover a second, semver-incompatible copy of RLST in the same build, which
//!   has a lock of its own.
//!
//! [`make_planner_thread_safe`] adds FFTW's own lock around plan creation and destruction, in
//! every caller, for a process where another library plans with FFTW. It does not cover
//! wisdom or FFTW's global settings.
//!
//! # Threaded plans
//!
//! With `fftw_threads`, RLST links FFTW's threads libraries and initialises them on its first
//! FFTW call, as FFTW asks, before any other FFTW call RLST makes. FFTW calls that other
//! libraries made earlier in the process are outside RLST's control. Plans for more than one
//! thread are made with [`Planner::threads`](super::Planner::threads); the thread count is
//! set for the planner's call only, and reset to 1 before the lock is released, so it never
//! applies to another caller's plan. [`Plan::threads`](super::Plan::threads) returns it.
//!
//! A threaded plan is another plan than a single-threaded one, and can round differently.
//! Executing a threaded plan from several threads at once is allowed, but runs more threads
//! than the machine has cores, notably under rayon.
//!
//! ```
//! use std::num::NonZeroUsize;
//!
//! use rlst::{c64, rlst_dynamic_array};
//! use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
//!
//! let layout = Layout::new(&[Dim { n: 1024, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
//! let two = NonZeroUsize::new(2).unwrap();
//! let threaded = Planner::<f64>::new().threads(two).c2c(&layout, Direction::Forward)?;
//! let serial = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
//! assert_eq!((threaded.threads(), serial.threads()), (2, 1));
//!
//! let mut input = rlst_dynamic_array!(c64, [1024] | rlst::FFTW_ALIGNED);
//! let mut output = rlst_dynamic_array!(c64, [1024] | rlst::FFTW_ALIGNED);
//! input.fill_from_seed_equally_distributed(0);
//! threaded.execute(input.data().unwrap(), output.data_mut().unwrap())?;
//! # Ok::<(), rlst::fftw::FftwError>(())
//! ```

use super::lock::with_fftw;

/// Install FFTW's own lock around plan creation and destruction
/// (`fftw_make_planner_thread_safe`), for both precisions.
///
/// RLST's own FFTW calls are serialised by RLST's lock without it (see [the module
/// documentation](self)). Call it only when another library in the process plans with FFTW,
/// before that library or RLST plans from more than one thread; calls after the first do
/// nothing. FFTW's manual advises against it unless there is no other choice. Its limits:
///
/// - FFTW's lock covers plan creation and destruction only, not wisdom, the thread count or
///   the time limit, which other libraries must still not change while RLST plans;
/// - it does not work with FFTW's OpenMP threads library (`fftw3_omp`), which RLST does not
///   link, but another library in the process might;
/// - it works from FFTW 3.3.6, which `fftw_threads` requires.
///
/// ```
/// rlst::fftw::threads::make_planner_thread_safe();
/// ```
pub fn make_planner_thread_safe() {
    with_fftw(|state| state.make_planner_thread_safe());
}

#[cfg(test)]
mod tests {
    // `make_planner_thread_safe` changes FFTW for the rest of the process; the case
    // `planner-thread-safe` of `tests/fftw_determinism.rs` calls it.

    use std::ffi::c_int;
    use std::num::NonZeroUsize;

    use crate::fftw::{Dim, Direction, FftwError, Layout, Placement, Planner};

    #[test]
    fn test_invalid_thread_count() {
        let layout =
            Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace).unwrap();
        let requested = c_int::MAX as usize + 1;
        let planner = Planner::<f64>::new().threads(NonZeroUsize::new(requested).unwrap());
        let error = planner.c2c(&layout, Direction::Forward).unwrap_err();
        assert_eq!(error, FftwError::InvalidThreadCount { requested });
        assert_eq!(
            error.to_string(),
            format!(
                "the thread count {requested} exceeds FFTW's largest thread count {}",
                c_int::MAX
            )
        );
    }

    #[test]
    fn test_threads_unavailable_message() {
        let text = FftwError::ThreadsUnavailable.to_string();
        assert!(text.contains("could not initialise its threads"), "{text}");
        assert!(text.contains("`Planner::threads`"), "{text}");
    }
}
