//! The element types of FFTW buffers, and real views of complex slices.

use num::Complex;

use super::real::FftwReal;

pub(crate) mod sealed {
    /// Seals [`FftElement`](super::FftElement) and tells real elements from complex ones.
    pub trait Sealed {
        /// Whether the element is complex.
        const COMPLEX: bool;
    }
}

/// An element type of the arrays that FFTW layouts are made from: a real `f32` or `f64`, or a
/// complex [`c32`](crate::c32) or [`c64`](crate::c64), whose precision is
/// [`Real`](FftElement::Real). Every [`FftwReal`] is an `FftElement` of its own precision.
///
/// [`Layout::along_axis`](super::Layout::along_axis), [`Layout::over_axes`](super::Layout::over_axes)
/// and their in-place variants read the storage of each side from the element type of its
/// array: a real input and a complex output give a real-to-complex layout, a complex input and
/// a real output a complex-to-real layout, two complex arrays a complex-to-complex layout and
/// two real arrays a real-to-real layout. Both arrays have one precision. The trait is sealed.
///
/// ```
/// use rlst::c64;
/// use rlst::fftw::FftElement;
///
/// fn precision<E: FftElement>() -> usize {
///     8 * std::mem::size_of::<E::Real>()
/// }
///
/// assert_eq!(precision::<f32>(), 32);
/// assert_eq!(precision::<c64>(), 64);
/// ```
pub trait FftElement: sealed::Sealed + Copy {
    /// The real type of the element's precision.
    type Real: FftwReal;
}

impl sealed::Sealed for f32 {
    const COMPLEX: bool = false;
}
impl FftElement for f32 {
    type Real = f32;
}

impl sealed::Sealed for f64 {
    const COMPLEX: bool = false;
}
impl FftElement for f64 {
    type Real = f64;
}

// `c32` and `c64`, and generic code over a precision `T`.
impl<T: FftwReal> sealed::Sealed for Complex<T> {
    const COMPLEX: bool = true;
}
impl<T: FftwReal> FftElement for Complex<T> {
    type Real = T;
}

/// The reals of a complex slice: the real and imaginary part of each element in turn, twice
/// as many as the elements.
///
/// In-place real-to-complex and complex-to-real plans execute on a complex buffer, whose reals
/// hold the real side of the transform (see
/// [`Layout::r2c_inplace`](super::Layout::r2c_inplace)); this function and [`as_reals_mut`]
/// give access to them, for example through a [`SliceArray`](crate::SliceArray).
///
/// ```
/// use rlst::c64;
/// use rlst::fftw::as_reals;
///
/// let values = [c64::new(1.0, 2.0), c64::new(3.0, 4.0)];
/// assert_eq!(as_reals(&values), &[1.0, 2.0, 3.0, 4.0]);
/// ```
pub fn as_reals<T: FftwReal>(values: &[Complex<T>]) -> &[T] {
    // SAFETY: `Complex<T>` is `#[repr(C)]` with the two fields `re` and `im` of type `T`, so
    // it has the size of two `T`s and the alignment of `T`, and a slice of `len` complex
    // values is `2 · len` initialised `T`s, real part first. The returned slice borrows
    // `values` for its lifetime, so no other access aliases it mutably.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<T>(), 2 * values.len()) }
}

/// The reals of a mutable complex slice, as [`as_reals`] gives them, to be written.
///
/// ```
/// use rlst::c64;
/// use rlst::fftw::as_reals_mut;
///
/// let mut values = [c64::new(0.0, 0.0); 2];
/// as_reals_mut(&mut values)[3] = 5.0;
/// assert_eq!(values[1], c64::new(0.0, 5.0));
/// ```
pub fn as_reals_mut<T: FftwReal>(values: &mut [Complex<T>]) -> &mut [T] {
    let len = 2 * values.len();
    // SAFETY: as in `as_reals`; every bit pattern of the two `T`s is a valid `Complex<T>`, so
    // writes through the returned slice leave valid complex values, and the returned slice
    // holds the exclusive borrow of `values` for its lifetime.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<T>(), len) }
}

#[cfg(test)]
mod tests {
    use super::{as_reals, as_reals_mut};
    use crate::c32;

    #[test]
    fn test_reals_of_complex_slices() {
        let mut values = [
            c32::new(1.0, -1.0),
            c32::new(2.0, -2.0),
            c32::new(3.0, -3.0),
        ];
        assert_eq!(as_reals(&values), &[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
        assert_eq!(
            as_reals(&values[1..]).as_ptr(),
            as_reals(&values)[2..].as_ptr()
        );
        as_reals_mut(&mut values)[0] = 7.0;
        assert_eq!(values[0], c32::new(7.0, -1.0));
        assert!(as_reals::<f64>(&[]).is_empty());
    }
}
