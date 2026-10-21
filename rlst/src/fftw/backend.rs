//! The FFTW back end and its version.

use std::fmt;

/// The FFTW library RLST is linked against.
///
/// RLST 0.11 has one back end, the system's FFTW (feature `fftw_system`). The enum is
/// non-exhaustive so that a later back end is an addition.
///
/// ```no_run
/// use rlst::fftw::{Backend, backend};
///
/// assert_eq!(backend(), Backend::System);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The system's `libfftw3` and `libfftw3f`, found by pkg-config (feature `fftw_system`).
    System,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = f;
        todo!()
    }
}

/// The back end of this build.
///
/// ```no_run
/// use rlst::fftw::{Backend, backend};
///
/// assert_eq!(backend().to_string(), "system");
/// ```
pub fn backend() -> Backend {
    todo!()
}

/// FFTW's version string for double precision, for example `fftw-3.3.10-sse2-avx`.
///
/// The string names the version and, after it, the SIMD extensions FFTW was built with. It is
/// read from the linked library at run time.
///
/// ```no_run
/// let version = rlst::fftw::version();
/// assert!(version.starts_with("fftw-3."));
/// ```
pub fn version() -> &'static str {
    todo!()
}
