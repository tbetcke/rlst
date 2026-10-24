//! The FFTW back end and its version.

use std::ffi::{CStr, c_char};
use std::fmt;

use super::error::FftwError;
use super::version::{MINIMUM_TEXT, meets_minimum};

// fftw-sys does not bind FFTW's version strings.
unsafe extern "C" {
    static fftw_version: [c_char; 0];
    static fftwf_version: [c_char; 0];
}

/// The FFTW library RLST is linked against.
///
/// RLST 0.11 has one back end, the system's FFTW (feature `fftw_system`). The enum is
/// non-exhaustive so that a later back end is an addition.
///
/// ```
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
        match self {
            Backend::System => f.write_str("system"),
        }
    }
}

/// The back end of this build.
///
/// ```
/// use rlst::fftw::{Backend, backend};
///
/// assert_eq!(backend().to_string(), "system");
/// ```
pub fn backend() -> Backend {
    Backend::System
}

/// FFTW's version string for double precision, for example `fftw-3.3.10-sse2-avx`.
///
/// The string names the version and, after it, the SIMD extensions FFTW was built with. It is
/// read from the linked library at run time.
///
/// ```
/// let version = rlst::fftw::version();
/// assert!(version.starts_with("fftw-3."));
/// ```
pub fn version() -> &'static str {
    // SAFETY: fftw3.h declares `extern const char fftw_version[]`, which FFTW defines as a
    // NUL-terminated constant array of chars. It lives as long as the program and is never
    // written.
    let text = unsafe { CStr::from_ptr((&raw const fftw_version).cast::<c_char>()) };
    as_str(text)
}

/// FFTW's version string for single precision, from `libfftw3f`.
fn single_version() -> &'static str {
    // SAFETY: as in `version`, for `fftwf_version` of `libfftw3f`.
    let text = unsafe { CStr::from_ptr((&raw const fftwf_version).cast::<c_char>()) };
    as_str(text)
}

fn as_str(text: &'static CStr) -> &'static str {
    // FFTW's version strings are ASCII.
    text.to_str()
        .unwrap_or("fftw-(version string is not UTF-8)")
}

/// The version numbers (major, minor, patch) of an FFTW version string such as
/// `fftw-3.3.10-sse2-avx`.
pub(crate) fn parse_version(version: &str) -> Option<(u32, u32, u32)> {
    let numbers = version.strip_prefix("fftw-")?.split('-').next()?;
    super::version::parse_numbers(numbers)
}

/// Whether an FFTW version string names a SIMD extension: a suffix other than `-fma`
/// (design, Section 3 item 6). Tests that depend on SIMD skip on builds without it.
#[cfg(test)]
pub(crate) fn has_simd(version: &str) -> bool {
    version
        .strip_prefix("fftw-")
        .is_some_and(|rest| rest.split('-').skip(1).any(|suffix| suffix != "fma"))
}

/// Check that both linked FFTW libraries are at least the minimum version.
///
/// A version string that does not parse counts as too old.
pub(crate) fn check_version() -> Result<(), FftwError> {
    for found in [version(), single_version()] {
        if !parse_version(found).is_some_and(meets_minimum) {
            return Err(FftwError::VersionTooOld {
                found: found.to_string(),
                minimum: MINIMUM_TEXT,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_version, has_simd, parse_version, single_version, version};

    #[test]
    fn test_parse_version() {
        assert_eq!(parse_version("fftw-3.3.10-sse2-avx"), Some((3, 3, 10)));
        assert_eq!(parse_version("fftw-3.3.8"), Some((3, 3, 8)));
        assert_eq!(parse_version("fftw-3.3.9-unknown-suffix"), Some((3, 3, 9)));
        assert_eq!(parse_version("fftw-3.3.11"), Some((3, 3, 11)));
        assert_eq!(parse_version("3.3.10"), None);
        assert_eq!(parse_version("fftw-"), None);
    }

    #[test]
    fn test_has_simd() {
        assert!(has_simd("fftw-3.3.10-sse2-avx"));
        assert!(has_simd("fftw-3.3.10-neon"));
        assert!(has_simd("fftw-3.3.10-fma-sse2"));
        assert!(!has_simd("fftw-3.3.11"));
        assert!(!has_simd("fftw-3.3.10-fma"));
    }

    #[test]
    fn test_version_at_run_time() {
        assert!(version().starts_with("fftw-3."), "{}", version());
        assert!(
            single_version().starts_with("fftw-3."),
            "{}",
            single_version()
        );
        assert!(parse_version(version()).is_some());
        assert_eq!(check_version(), Ok(()));
    }
}
