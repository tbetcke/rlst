//! Precision dispatch: [`FftwReal`] for `f32` and `f64`.

use crate::RlstScalar;

pub(crate) mod sealed {
    /// Seals [`FftwReal`](super::FftwReal).
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A real type with an FFTW precision: `f32` (FFTW's `fftwf_` functions) or `f64` (`fftw_`).
///
/// [`Planner`](super::Planner) and [`Plan`](super::Plan) are generic over it, and dispatch
/// statically to the functions of that precision. Complex data uses RLST's
/// [`c32`](crate::c32) and [`c64`](crate::c64), which are FFTW's `fftwf_complex` and
/// `fftw_complex`. The trait is sealed.
///
/// ```no_run
/// use rlst::fftw::FftwReal;
///
/// fn precision<T: FftwReal>() -> usize {
///     8 * std::mem::size_of::<T>()
/// }
///
/// assert_eq!(precision::<f32>(), 32);
/// assert_eq!(precision::<f64>(), 64);
/// ```
pub trait FftwReal: RlstScalar<Real = Self> + sealed::Sealed {}

impl FftwReal for f32 {}
impl FftwReal for f64 {}
