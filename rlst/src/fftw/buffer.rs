//! Buffers that plans execute on: slices, and RLST arrays from task F2.

use crate::{Array, Shape, Stride, StridedRawAccess, StridedRawAccessMut};

use super::error::{FftwError, Side};
use super::layout::Layout;

pub(crate) mod sealed {
    use crate::fftw::error::{FftwError, Side};
    use crate::fftw::layout::Layout;

    /// Seals [`FftBuffer`](super::FftBuffer) and carries the check of a buffer against one
    /// side of a layout.
    pub trait Sealed<E> {
        /// Check that the buffer fits `side` of `layout`.
        fn check_side(&self, layout: &Layout, side: Side) -> Result<(), FftwError>;
    }
}

/// A buffer of elements `E` that a plan reads.
///
/// Implemented for slices `[E]`, whose layout offsets count from the start of the slice, and
/// for RLST arrays and views with [`StridedRawAccess`], subviews included, whose offsets count
/// from their first element (task F2). The trait is sealed.
///
/// ```
/// use rlst::c64;
/// use rlst::fftw::FftBuffer;
///
/// fn length<B: FftBuffer<c64> + ?Sized>(buffer: &B) -> usize {
///     FftBuffer::len(buffer)
/// }
///
/// let data = [c64::new(1.0, 0.0); 4];
/// assert_eq!(length(&data[..]), 4);
/// ```
pub trait FftBuffer<E>: sealed::Sealed<E> {
    /// A pointer to the first element.
    ///
    /// ```
    /// use rlst::fftw::FftBuffer;
    ///
    /// let data = [1.0_f64, 2.0];
    /// assert_eq!(FftBuffer::as_ptr(&data[..]), data.as_ptr());
    /// ```
    fn as_ptr(&self) -> *const E;

    /// The number of elements.
    ///
    /// ```
    /// use rlst::fftw::FftBuffer;
    ///
    /// let data = [1.0_f64, 2.0];
    /// assert_eq!(FftBuffer::len(&data[..]), 2);
    /// ```
    fn len(&self) -> usize;

    /// Whether the buffer has no elements.
    ///
    /// ```
    /// use rlst::fftw::FftBuffer;
    ///
    /// let data: [f64; 0] = [];
    /// assert!(FftBuffer::is_empty(&data[..]));
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A buffer of elements `E` that a plan writes.
///
/// Implemented for slices `[E]` and for mutable RLST arrays and views with
/// [`StridedRawAccessMut`] (task F2). The trait is sealed.
///
/// ```
/// use rlst::c64;
/// use rlst::fftw::FftBufferMut;
///
/// fn first<B: FftBufferMut<c64> + ?Sized>(buffer: &mut B) -> *mut c64 {
///     FftBufferMut::as_mut_ptr(buffer)
/// }
///
/// let mut data = [c64::new(0.0, 0.0); 4];
/// assert_eq!(first(&mut data[..]), data.as_mut_ptr());
/// ```
pub trait FftBufferMut<E>: FftBuffer<E> {
    /// A mutable pointer to the first element.
    ///
    /// ```
    /// use rlst::fftw::FftBufferMut;
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let ptr = FftBufferMut::as_mut_ptr(&mut data[..]);
    /// assert_eq!(ptr, data.as_mut_ptr());
    /// ```
    fn as_mut_ptr(&mut self) -> *mut E;
}

impl<E> sealed::Sealed<E> for [E] {
    fn check_side(&self, layout: &Layout, side: Side) -> Result<(), FftwError> {
        let span = layout.span(side);
        if self.len() < span {
            return Err(FftwError::LayoutMismatch {
                side,
                expected: format!("at least {span} elements, the layout's {side} span"),
                found: format!("a slice of {} elements", self.len()),
            });
        }
        Ok(())
    }
}

impl<E> FftBuffer<E> for [E] {
    fn as_ptr(&self) -> *const E {
        <[E]>::as_ptr(self)
    }

    fn len(&self) -> usize {
        <[E]>::len(self)
    }
}

impl<E> FftBufferMut<E> for [E] {
    fn as_mut_ptr(&mut self) -> *mut E {
        <[E]>::as_mut_ptr(self)
    }
}

// RLST arrays and views through their strided storage, implemented in task F2.

impl<A, const N: usize> sealed::Sealed<A::Item> for Array<A, N>
where
    A: StridedRawAccess + Shape<N> + Stride<N>,
{
    fn check_side(&self, layout: &Layout, side: Side) -> Result<(), FftwError> {
        let _ = (layout, side);
        todo!()
    }
}

impl<A, const N: usize> FftBuffer<A::Item> for Array<A, N>
where
    A: StridedRawAccess + Shape<N> + Stride<N>,
{
    fn as_ptr(&self) -> *const A::Item {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }
}

impl<A, const N: usize> FftBufferMut<A::Item> for Array<A, N>
where
    A: StridedRawAccessMut + Shape<N> + Stride<N>,
{
    fn as_mut_ptr(&mut self) -> *mut A::Item {
        todo!()
    }
}
