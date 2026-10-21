//! Plan descriptions and fingerprints.

use std::ffi::CStr;

use super::lock::with_fftw;
use super::plan::{Family, Plan, Policy};
use super::real::FftwReal;

/// The 64-bit FNV-1a hash of `bytes`.
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

impl<T: FftwReal, F: Family, P: Policy> Plan<T, F, P> {
    /// FFTW's description of the algorithm it chose for the plan (`fftw_sprint_plan`).
    ///
    /// Read under RLST's FFTW lock.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
    ///
    /// let layout = Layout::new(&[Dim { n: 8, is: 1, os: 1 }], &[], Placement::OutOfPlace)?;
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// assert!(!plan.description().is_empty());
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn description(&self) -> String {
        with_fftw(|_| {
            // SAFETY: the plan is live and the FFTW lock is held.
            let text = unsafe { T::sprint_plan(self.raw()) };
            if text.is_null() {
                // FFTW returns NULL only when it cannot allocate the string.
                std::alloc::handle_alloc_error(std::alloc::Layout::new::<u8>());
            }
            // SAFETY: `fftw_sprint_plan` returns a NUL-terminated string, which stays valid
            // until it is freed below.
            let description = unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: FFTW allocated the string with `malloc` and hands it to the caller, who
            // frees it with `free`; it is freed once and not used afterwards.
            unsafe { libc::free(text.cast()) };
            description
        })
    }

    /// A 64-bit FNV-1a hash of [`description`](Plan::description).
    ///
    /// The hash function is the same in every process and build, so equal fingerprints mean
    /// that FFTW chose the same algorithm, for example on every rank of a distributed
    /// computation.
    ///
    /// ```
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
        fnv1a_64(self.description().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::fnv1a_64;
    use crate::fftw::{Dim, Direction, Layout, Placement, Planner};

    #[test]
    fn test_fnv1a_64() {
        assert_eq!(fnv1a_64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn test_equal_layouts_have_equal_fingerprints() {
        let dims = [Dim {
            n: 588,
            is: 1,
            os: 1,
        }];
        let batch = [Dim {
            n: 9,
            is: 588,
            os: 588,
        }];
        let first = Layout::new(&dims, &batch, Placement::OutOfPlace).unwrap();
        let second = Layout::new(&dims, &batch, Placement::OutOfPlace).unwrap();

        let plan = Planner::<f64>::new()
            .c2c(&first, Direction::Forward)
            .unwrap();
        let again = Planner::<f64>::new()
            .c2c(&second, Direction::Forward)
            .unwrap();
        let description = plan.description();
        assert!(!description.is_empty());
        assert_eq!(description, again.description());
        assert_eq!(plan.fingerprint(), again.fingerprint());
        assert_eq!(plan.fingerprint(), fnv1a_64(description.as_bytes()));

        let plan = Planner::<f32>::new()
            .c2c_inplace(
                &Layout::new(&dims, &batch, Placement::InPlace).unwrap(),
                Direction::Backward,
            )
            .unwrap();
        assert!(!plan.description().is_empty());
    }
}
