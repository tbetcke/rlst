//! FFTW's wisdom: what FFTW's planner has learned about the problems planned in this process.
//!
//! Every plan adds to the wisdom of its precision, and FFTW plans from any wisdom of equal or
//! greater rigor, an [`Estimate`](super::Rigor::Estimate) plan included. The functions here
//! export the wisdom of one precision to a string or a file, import it, forget it, and import
//! the system's wisdom. Each runs under RLST's FFTW lock; the file functions read and write
//! their files outside it.
//!
//! # Wisdom and determinism
//!
//! - **Wisdom changes plans.** After wisdom of a greater rigor is imported, or a plan of
//!   greater rigor made, a later plan of the same problem, an `Estimate` plan included, can
//!   be another plan, which can round differently. Wisdom from `Estimate` planning does not
//!   change later `Estimate` plans.
//! - **Shared wisdom.** For the same plans in several processes, import the same wisdom in
//!   every process before any planning, plan with
//!   [`wisdom_only`](super::Planner::wisdom_only), and compare
//!   [`fingerprint`](super::Plan::fingerprint)s across the processes.
//! - **The same options.** A wisdom-only plan finds the wisdom only when it is made with the
//!   planner options the wisdom was made with, other than `wisdom_only`, and a rigor no
//!   greater: the same input policy, alignment classes and number of threads.
//! - **Per precision.** Wisdom is kept per precision: f64 wisdom imported as f32 is refused.
//! - **Per machine.** Wisdom made on a CPU with other SIMD support, or with another FFTW
//!   build, may not apply.
//!
//! Plans that exist are not affected by importing or forgetting wisdom.
//!
//! ```
//! use rlst::fftw::wisdom;
//!
//! let exported = wisdom::export::<f64>();
//! assert!(exported.starts_with("(fftw-3."));
//! ```

use std::ffi::{c_char, c_int, c_void};
use std::path::Path;

use super::error::FftwError;
use super::lock::with_fftw;
use super::real::FftwReal;

/// The wisdom of precision `T` as text, in FFTW's format.
///
/// The text starts with a header that names FFTW's version and the precision
/// (`fftw_wisdom` or `fftwf_wisdom`). Exporting reads the wisdom only.
///
/// ```
/// use rlst::fftw::wisdom;
///
/// assert!(wisdom::export::<f64>().contains("fftw_wisdom"));
/// assert!(wisdom::export::<f32>().contains("fftwf_wisdom"));
/// ```
pub fn export<T: FftwReal>() -> String {
    let mut bytes: Vec<u8> = Vec::new();
    with_fftw(|_| {
        // SAFETY: the FFTW lock is held. `write_char` matches `fftw_write_char_func` and
        // cannot unwind; FFTW passes `data`, which points to `bytes`, only to it, and only
        // during this call, while `bytes` is live and borrowed by nothing else.
        unsafe { T::export_wisdom(Some(write_char), (&raw mut bytes).cast()) }
    });
    // FFTW writes ASCII.
    String::from_utf8(bytes)
        .unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned())
}

/// FFTW's callback of [`export`]: append the character `c` to the `Vec<u8>` behind `data`.
///
/// # Safety
///
/// `data` points to a live `Vec<u8>` that nothing else accesses during the call.
unsafe extern "C" fn write_char(c: c_char, data: *mut c_void) {
    // SAFETY: by this function's contract.
    let bytes = unsafe { &mut *data.cast::<Vec<u8>>() };
    // A panic here aborts the process: it cannot unwind through FFTW. Only a capacity
    // overflow could panic.
    bytes.push(c as u8);
}

/// Import wisdom of precision `T` from text in FFTW's format, as [`export`] writes it.
///
/// Returns [`FftwError::WisdomImportFailed`] if FFTW does not accept it, which it does not
/// for wisdom of the other precision; FFTW gives no reason. The wisdom imported changes later
/// plans of its problems; see [the module documentation](self).
///
/// ```no_run
/// use rlst::fftw::wisdom;
///
/// let saved = wisdom::export::<f64>();
/// // …in this process or another:
/// wisdom::import::<f64>(&saved)?;
/// assert!(wisdom::import::<f32>(&saved).is_err());
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
pub fn import<T: FftwReal>(wisdom: &str) -> Result<(), FftwError> {
    import_bytes::<T>(wisdom.as_bytes())
}

/// Import wisdom of precision `T` from `bytes`.
fn import_bytes<T: FftwReal>(bytes: &[u8]) -> Result<(), FftwError> {
    let mut reader = Reader { bytes, position: 0 };
    let accepted = with_fftw(|_| {
        // SAFETY: the FFTW lock is held. `read_char` matches `fftw_read_char_func` and
        // cannot unwind; FFTW passes `data`, which points to `reader`, only to it, and only
        // during this call, while `reader` is live and borrowed by nothing else.
        unsafe { T::import_wisdom(Some(read_char), (&raw mut reader).cast()) }
    });
    if accepted == 0 {
        Err(FftwError::WisdomImportFailed {
            precision: T::PRECISION_NAME,
        })
    } else {
        Ok(())
    }
}

/// The bytes that FFTW's callback of [`import`] reads, and how many it has read.
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

/// FFTW's callback of [`import`]: the next byte of the [`Reader`] behind `data`, or `EOF`.
///
/// # Safety
///
/// `data` points to a live `Reader` that nothing else accesses during the call.
unsafe extern "C" fn read_char(data: *mut c_void) -> c_int {
    // SAFETY: by this function's contract.
    let reader = unsafe { &mut *data.cast::<Reader<'_>>() };
    match reader.bytes.get(reader.position) {
        Some(&byte) => {
            reader.position += 1;
            c_int::from(byte)
        }
        None => libc::EOF,
    }
}

/// Write the wisdom of precision `T` to the file at `path`, replacing it if it exists.
///
/// The wisdom is exported under the FFTW lock and written after it is released. A file that
/// cannot be written is [`FftwError::WisdomFile`].
///
/// ```no_run
/// use rlst::fftw::wisdom;
///
/// wisdom::export_to_file::<f64>("wisdom-f64.txt")?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
pub fn export_to_file<T: FftwReal>(path: impl AsRef<Path>) -> Result<(), FftwError> {
    let path = path.as_ref();
    let wisdom = export::<T>();
    std::fs::write(path, wisdom).map_err(|error| FftwError::wisdom_file(path, &error))
}

/// Import wisdom of precision `T` from the file at `path`, as [`export_to_file`] writes it.
///
/// The file is read before the FFTW lock is taken. A file that cannot be read is
/// [`FftwError::WisdomFile`]; wisdom that FFTW does not accept is
/// [`FftwError::WisdomImportFailed`], as for [`import`].
///
/// ```no_run
/// use rlst::fftw::wisdom;
///
/// wisdom::import_from_file::<f64>("wisdom-f64.txt")?;
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
pub fn import_from_file<T: FftwReal>(path: impl AsRef<Path>) -> Result<(), FftwError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|error| FftwError::wisdom_file(path, &error))?;
    import_bytes::<T>(&bytes)
}

/// Forget the wisdom of precision `T`.
///
/// Plans that exist are not affected; later plans no longer find the forgotten wisdom.
///
/// ```no_run
/// use rlst::fftw::wisdom;
///
/// wisdom::forget::<f64>();
/// ```
pub fn forget<T: FftwReal>() {
    with_fftw(|_| {
        // SAFETY: the FFTW lock is held.
        unsafe { T::forget_wisdom() }
    });
}

/// Import the system's wisdom of precision `T`: the file FFTW reads for it, under
/// `/etc/fftw/` on most systems.
///
/// RLST imports it only when this function is called. Without a system wisdom file, or if
/// FFTW does not accept it, this returns [`FftwError::WisdomImportFailed`]: FFTW does not tell
/// the two apart.
///
/// ```no_run
/// use rlst::fftw::wisdom;
///
/// if wisdom::import_system_wisdom::<f64>().is_err() {
///     println!("no system wisdom");
/// }
/// ```
pub fn import_system_wisdom<T: FftwReal>() -> Result<(), FftwError> {
    let accepted = with_fftw(|_| {
        // SAFETY: the FFTW lock is held.
        unsafe { T::import_system_wisdom() }
    });
    if accepted == 0 {
        Err(FftwError::WisdomImportFailed {
            precision: T::PRECISION_NAME,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{export, export_to_file};
    use crate::fftw::FftwError;

    // No test here imports or forgets wisdom: those change the wisdom every later test of
    // this process plans with (DC39). They are cases of `tests/fftw_determinism.rs`.

    #[test]
    fn test_export_names_precision() {
        let double = export::<f64>();
        let single = export::<f32>();
        assert!(double.starts_with("(fftw-3."), "{double}");
        assert!(double.contains(" fftw_wisdom "), "{double}");
        assert!(single.starts_with("(fftw-3."), "{single}");
        assert!(single.contains(" fftwf_wisdom "), "{single}");
    }

    #[test]
    fn test_export_to_missing_directory() {
        let path = std::env::temp_dir()
            .join(format!("rlst-no-such-directory-{}", std::process::id()))
            .join("wisdom");
        let error = export_to_file::<f64>(&path).unwrap_err();
        let FftwError::WisdomFile {
            path: found,
            kind,
            message,
        } = &error
        else {
            panic!("expected `WisdomFile`, found {error:?}");
        };
        assert_eq!(found, &path);
        assert_eq!(*kind, io::ErrorKind::NotFound);
        assert!(!message.is_empty());
        let text = error.to_string();
        assert!(text.contains(&path.display().to_string()), "{text}");
        assert!(text.contains("could not be read or written"), "{text}");
    }

    #[test]
    fn test_import_failed_message() {
        let error = FftwError::WisdomImportFailed { precision: "f32" };
        assert_eq!(
            error.to_string(),
            "FFTW did not accept the f32 wisdom; FFTW gives no reason"
        );
    }
}
