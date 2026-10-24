//! The FFTW lock, which serialises every FFTW call RLST makes other than plan execution.
//!
//! FFTW's planner, plan destruction, wisdom, `fftw_sprint_plan` and its global settings must
//! be used from one thread at a time (design, Sections 3 item 2 and 4.7). Every such call in
//! RLST runs inside [`with_fftw`]. The lock cannot protect FFTW
//! from other libraries in the process, nor from a second copy of RLST in the same build.

use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};

use super::backend::check_version;
use super::error::FftwError;
use super::real::FftwReal;

/// The state guarded by the FFTW lock.
pub(crate) struct FftwState {
    /// The result of the run-time version check, made on the first acquisition.
    version: Option<Result<(), FftwError>>,
    /// Whether FFTW's threads were initialised for both precisions, on the first acquisition.
    #[cfg(feature = "fftw_threads")]
    threads_initialised: bool,
    /// Whether `fftw_make_planner_thread_safe` was called.
    #[cfg(feature = "fftw_threads")]
    planner_thread_safe: bool,
}

impl FftwState {
    /// The state before the first acquisition.
    const fn new() -> Self {
        Self {
            version: None,
            #[cfg(feature = "fftw_threads")]
            threads_initialised: false,
            #[cfg(feature = "fftw_threads")]
            planner_thread_safe: false,
        }
    }

    /// The result of the run-time version check: [`FftwError::VersionTooOld`] if the linked
    /// FFTW is older than the minimum.
    pub(crate) fn version_check(&self) -> Result<(), FftwError> {
        self.version
            .clone()
            .expect("`with_fftw` checks the version on the first acquisition")
    }

    /// Install FFTW's own lock around plan creation and destruction, for both precisions,
    /// once per process.
    #[cfg(feature = "fftw_threads")]
    pub(crate) fn make_planner_thread_safe(&mut self) {
        if !self.planner_thread_safe {
            // SAFETY: `&mut self` proves that the caller is inside `with_fftw`, so it holds
            // the FFTW lock.
            unsafe {
                <f64 as super::real::sealed::Sealed>::make_planner_thread_safe();
                <f32 as super::real::sealed::Sealed>::make_planner_thread_safe();
            }
            self.planner_thread_safe = true;
        }
    }
}

static FFTW_LOCK: Mutex<FftwState> = Mutex::new(FftwState::new());

/// Run `f` under the FFTW lock.
///
/// The first acquisition checks the version of the linked FFTW and, with `fftw_threads`,
/// initialises FFTW's threads (design, Section 3 item 12): every FFTW call RLST makes runs
/// here, so this precedes RLST's first FFTW call.
///
/// A lock poisoned by a panic in an earlier critical section is recovered, and its poison
/// cleared, rather than propagated: FFTW's own state does not depend on how RLST's closure ended, and the settings
/// guard resets FFTW's global settings during unwinding.
pub(crate) fn with_fftw<R>(f: impl FnOnce(&mut FftwState) -> R) -> R {
    let mut guard = FFTW_LOCK.lock().unwrap_or_else(|poisoned| {
        FFTW_LOCK.clear_poison();
        PoisonError::into_inner(poisoned)
    });
    let state = &mut *guard;
    if state.version.is_none() {
        state.version = Some(check_version());
        #[cfg(feature = "fftw_threads")]
        {
            state.threads_initialised = init_threads();
        }
    }
    f(state)
}

/// Initialise FFTW's threads for both precisions; whether both succeeded.
///
/// Called once, by the first acquisition of [`with_fftw`], which holds the FFTW lock.
#[cfg(feature = "fftw_threads")]
fn init_threads() -> bool {
    use super::real::sealed::Sealed;

    // SAFETY: called by `with_fftw`, which holds the FFTW lock.
    let double = unsafe { <f64 as Sealed>::init_threads() } != 0;
    // SAFETY: as above.
    let single = unsafe { <f32 as Sealed>::init_threads() } != 0;
    double && single
}

/// Applies a plan's FFTW settings for precision `T` while it is alive, and resets them in
/// `Drop`, during unwinding as well: the time limit and, with `fftw_threads`, the thread count.
///
/// It borrows the [`FftwState`] mutably, so it exists only inside [`with_fftw`]. At rest, the
/// time limit is none and the thread count 1.
pub(crate) struct SettingsGuard<'a, T: FftwReal> {
    /// Whether the guard set the thread count, which it resets to 1.
    #[cfg(feature = "fftw_threads")]
    threads_set: bool,
    _state: PhantomData<&'a mut FftwState>,
    _real: PhantomData<T>,
}

impl<'a, T: FftwReal> SettingsGuard<'a, T> {
    /// Apply `time_limit`, in seconds, or no limit if it is `None`, and the thread count
    /// `threads`.
    ///
    /// Before FFTW is called, returns [`FftwError::InvalidThreadCount`] if `threads` exceeds
    /// FFTW's `int`, and [`FftwError::ThreadsUnavailable`] if it exceeds 1 and FFTW's threads
    /// could not be initialised. Without `fftw_threads`, `threads` is 1.
    pub(crate) fn new(
        state: &'a mut FftwState,
        time_limit: Option<f64>,
        threads: NonZeroUsize,
    ) -> Result<Self, FftwError> {
        #[cfg(feature = "fftw_threads")]
        let threads_set = {
            let count = std::ffi::c_int::try_from(threads.get()).map_err(|_| {
                FftwError::InvalidThreadCount {
                    requested: threads.get(),
                }
            })?;
            if count > 1 && !state.threads_initialised {
                return Err(FftwError::ThreadsUnavailable);
            }
            if state.threads_initialised {
                // SAFETY: `state` proves that the caller is inside `with_fftw`, so it holds
                // the FFTW lock; FFTW's threads of this precision were initialised, and
                // `count` is positive.
                unsafe { T::plan_with_nthreads(count) };
            }
            state.threads_initialised
        };
        #[cfg(not(feature = "fftw_threads"))]
        debug_assert_eq!(threads.get(), 1, "threads need `fftw_threads`");
        let _ = state;
        // SAFETY: `state` proves that the caller is inside `with_fftw`, so it holds the FFTW
        // lock.
        unsafe { T::set_timelimit(time_limit.unwrap_or(fftw_sys::FFTW_NO_TIMELIMIT)) };
        Ok(Self {
            #[cfg(feature = "fftw_threads")]
            threads_set,
            _state: PhantomData,
            _real: PhantomData,
        })
    }
}

impl<T: FftwReal> Drop for SettingsGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: the guard borrows the `FftwState` of the enclosing `with_fftw`, so the FFTW
        // lock is held until after this reset.
        unsafe { T::set_timelimit(fftw_sys::FFTW_NO_TIMELIMIT) };
        #[cfg(feature = "fftw_threads")]
        if self.threads_set {
            // SAFETY: as above; the guard set the thread count, so FFTW's threads of this
            // precision were initialised.
            unsafe { T::plan_with_nthreads(1) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::{FFTW_LOCK, SettingsGuard, with_fftw};
    use crate::fftw::{Dim, Direction, Layout, Placement, Planner};

    #[test]
    fn test_lock_recovers_after_panic() {
        // nd-fmm's unit, whose threaded plans differ from its single-threaded ones.
        let layout = Layout::new(
            &[Dim {
                n: 140,
                is: 1,
                os: 1,
            }],
            &[Dim {
                n: 8,
                is: 140,
                os: 140,
            }],
            Placement::OutOfPlace,
        )
        .unwrap();
        let plan = || {
            Planner::<f64>::new()
                .c2c(&layout, Direction::Forward)
                .unwrap()
        };
        let before = plan().fingerprint();
        // With `fftw_threads`, the panic leaves a thread count set, which the guard resets.
        let threads = NonZeroUsize::new(if cfg!(feature = "fftw_threads") { 4 } else { 1 });

        let result = catch_unwind(AssertUnwindSafe(|| {
            with_fftw(|state| {
                let _settings =
                    SettingsGuard::<f64>::new(state, Some(1.0), threads.unwrap()).unwrap();
                panic!("a panic inside the FFTW lock");
            })
        }));
        assert!(result.is_err());

        // The lock is usable again, whether or not another test has acquired it since.
        assert!(with_fftw(|state| state.version_check()).is_ok());
        assert!(!FFTW_LOCK.is_poisoned());

        // The settings were reset: a plan made now is the plan made before.
        let after = plan();
        assert_eq!(after.threads(), 1);
        assert_eq!(after.fingerprint(), before, "{}", after.description());
    }
}
