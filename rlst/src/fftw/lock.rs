//! The FFTW lock, which serialises every FFTW call RLST makes other than plan execution.
//!
//! FFTW's planner, plan destruction, wisdom, `fftw_sprint_plan` and its global settings must
//! be used from one thread at a time (design, Sections 3 item 2 and 4.7). Every such call in
//! RLST, in the 0.10 module as well, runs inside [`with_fftw`]. The lock cannot protect FFTW
//! from other libraries in the process, nor from a second copy of RLST in the same build.

use std::marker::PhantomData;
use std::sync::{Mutex, PoisonError};

use super::backend::check_version;
use super::error::FftwError;
use super::real::FftwReal;

/// The state guarded by the FFTW lock.
pub(crate) struct FftwState {
    /// The result of the run-time version check, made on the first acquisition.
    version: Option<Result<(), FftwError>>,
}

impl FftwState {
    /// The result of the run-time version check: [`FftwError::VersionTooOld`] if the linked
    /// FFTW is older than the minimum.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the planner from task F1c")
    )]
    pub(crate) fn version_check(&self) -> Result<(), FftwError> {
        self.version
            .clone()
            .expect("`with_fftw` checks the version on the first acquisition")
    }
}

static FFTW_LOCK: Mutex<FftwState> = Mutex::new(FftwState { version: None });

/// Run `f` under the FFTW lock.
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
    }
    f(state)
}

/// Applies a plan's FFTW settings for precision `T` while it is alive, and resets them in
/// `Drop`, during unwinding as well.
///
/// It borrows the [`FftwState`] mutably, so it exists only inside [`with_fftw`]. Today it
/// sets the time limit; the thread count follows with the `fftw_threads` feature (task F6).
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the planner from task F1c")
)]
pub(crate) struct SettingsGuard<'a, T: FftwReal> {
    _state: PhantomData<&'a mut FftwState>,
    _real: PhantomData<T>,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the planner from task F1c")
)]
impl<'a, T: FftwReal> SettingsGuard<'a, T> {
    /// Apply `time_limit`, in seconds, or no limit if it is `None`.
    pub(crate) fn new(state: &'a mut FftwState, time_limit: Option<f64>) -> Self {
        let _ = state;
        // SAFETY: `state` proves that the caller is inside `with_fftw`, so it holds the FFTW
        // lock.
        unsafe { T::set_timelimit(time_limit.unwrap_or(fftw_sys::FFTW_NO_TIMELIMIT)) };
        Self {
            _state: PhantomData,
            _real: PhantomData,
        }
    }
}

impl<T: FftwReal> Drop for SettingsGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: the guard borrows the `FftwState` of the enclosing `with_fftw`, so the FFTW
        // lock is held until after this reset.
        unsafe { T::set_timelimit(fftw_sys::FFTW_NO_TIMELIMIT) };
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::{FFTW_LOCK, SettingsGuard, with_fftw};
    use crate::fftw::{C2CDirection, C2CPlan, FftPlanInplace, FftwPlanFlags};
    use crate::{DynArray, c64};

    #[test]
    fn test_lock_recovers_after_panic() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            with_fftw(|state| {
                let _settings = SettingsGuard::<f64>::new(state, Some(1.0));
                panic!("a panic inside the FFTW lock");
            })
        }));
        assert!(result.is_err());

        // The lock is usable again, whether or not another test has acquired it since.
        assert!(with_fftw(|state| state.version_check()).is_ok());
        assert!(!FFTW_LOCK.is_poisoned());

        // Planning with the 0.10 module, which shares the lock, still works.
        let mut data = DynArray::<c64, 1>::from_shape([8]);
        data[[0]] = c64::new(1.0, 0.0);
        let fft = data.into_fft();
        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();
        let transformed = fft.borrow().unwrap();
        for k in 0..8 {
            approx::assert_relative_eq!(transformed[[k]], c64::new(1.0, 0.0), epsilon = 1e-14);
        }
    }
}
