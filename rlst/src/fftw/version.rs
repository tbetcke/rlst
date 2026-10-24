//! FFTW version numbers.
//!
//! `rlst/build.rs` includes this file to check the version pkg-config reports, and the crate
//! uses it to check the version of the library linked at run time, so it depends on nothing
//! but `core`. Both are compiled with the package's features, so they share the minimum.

/// The oldest supported FFTW: `fftw_sprint_plan`, used by plan fingerprints, needs 3.3.4.
#[cfg(not(feature = "fftw_threads"))]
pub(crate) const MINIMUM: (u32, u32, u32) = (3, 3, 4);

/// The oldest supported FFTW with `fftw_threads`: `fftw_make_planner_thread_safe` works from
/// 3.3.6 (design, Section 3 item 3).
#[cfg(feature = "fftw_threads")]
pub(crate) const MINIMUM: (u32, u32, u32) = (3, 3, 6);

/// [`MINIMUM`] as text, for messages.
#[cfg(not(feature = "fftw_threads"))]
pub(crate) const MINIMUM_TEXT: &str = "3.3.4";

/// [`MINIMUM`] as text, for messages.
#[cfg(feature = "fftw_threads")]
pub(crate) const MINIMUM_TEXT: &str = "3.3.6";

/// Parse a version `major.minor[.patch]` such as `3.3.10`, ignoring anything after the
/// digits of each component, as in `3.3.10-sse2` or `3.3.10rc1`. A missing patch is 0.
pub(crate) fn parse_numbers(text: &str) -> Option<(u32, u32, u32)> {
    fn leading_number(part: &str) -> Option<u32> {
        let end = part
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(part.len());
        part[..end].parse().ok()
    }

    let mut parts = text.split('.');
    let major = leading_number(parts.next()?)?;
    let minor = leading_number(parts.next()?)?;
    let patch = match parts.next() {
        Some(part) => leading_number(part)?,
        None => 0,
    };
    Some((major, minor, patch))
}

/// Whether `version` is at least [`MINIMUM`].
pub(crate) fn meets_minimum(version: (u32, u32, u32)) -> bool {
    version >= MINIMUM
}

#[cfg(test)]
mod tests {
    use super::{MINIMUM, MINIMUM_TEXT, meets_minimum, parse_numbers};

    #[test]
    fn test_parse_numbers() {
        assert_eq!(parse_numbers("3.3.10"), Some((3, 3, 10)));
        assert_eq!(parse_numbers("3.3.8"), Some((3, 3, 8)));
        assert_eq!(parse_numbers("3.3"), Some((3, 3, 0)));
        assert_eq!(parse_numbers("3.3.10-sse2-avx"), Some((3, 3, 10)));
        assert_eq!(parse_numbers("3.3.10rc1"), Some((3, 3, 10)));
        assert_eq!(parse_numbers("3"), None);
        assert_eq!(parse_numbers(""), None);
        assert_eq!(parse_numbers("x.3.4"), None);
    }

    #[test]
    fn test_meets_minimum() {
        let meets = |text| parse_numbers(text).is_some_and(meets_minimum);
        assert_eq!(parse_numbers(MINIMUM_TEXT), Some(MINIMUM));
        assert!(meets("3.3.6"));
        assert!(meets("3.3.10"));
        assert!(meets("3.3.11"));
        assert!(meets("3.4"));
        assert!(meets("4.0.0"));
        assert!(!meets("3.3.3"));
        assert!(!meets("3.2.20"));
        assert!(!meets("2.1.5"));
        assert!(!meets("unknown"));
        // `fftw_make_planner_thread_safe` is broken in 3.3.5.
        let threads = cfg!(feature = "fftw_threads");
        assert_eq!(meets("3.3.4"), !threads);
        assert_eq!(meets("3.3.5"), !threads);
    }
}
