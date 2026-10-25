//! Buffers that plans execute on: slices, and RLST arrays and views.

use crate::{Array, Shape, Stride, StridedRawAccess, StridedRawAccessMut};

use super::error::{FftwError, Side};
use super::layout::Layout;

pub(crate) mod sealed {
    use crate::fftw::error::{FftwError, Side};
    use crate::fftw::layout::Layout;

    /// The shape and strides of an array buffer.
    pub type ArrayGeometry<'a> = (&'a [usize], &'a [usize]);

    /// Seals [`FftBuffer`](super::FftBuffer) and carries the check of a buffer against one
    /// side of a layout.
    pub trait Sealed<E> {
        /// Check that the buffer fits `side` of `layout`, and return the pointer to the first
        /// element that the check validated.
        ///
        /// Execution passes FFTW this pointer, never one from a later call: a safe
        /// implementation of `StridedRawAccess` may return another slice each time.
        fn check_side(&self, layout: &Layout, side: Side) -> Result<*const E, FftwError>;

        /// Call `f` with the shape and strides of an array buffer, for the pairing of input
        /// and output axes at execution (DC15), or with `None` for a slice. A callback rather
        /// than a return value, so that an array's shape and strides stay on the stack.
        fn with_array_geometry(&self, f: &mut dyn FnMut(Option<ArrayGeometry<'_>>)) {
            f(None)
        }
    }

    /// Seals [`FftBufferMut`](super::FftBufferMut) and carries the check of a buffer that a
    /// plan writes.
    pub trait SealedMut<E> {
        /// The check of [`Sealed::check_side`] through the exclusive borrow, returning the
        /// mutable pointer that the check validated, which derives from that borrow.
        fn check_side_mut(&mut self, layout: &Layout, side: Side) -> Result<*mut E, FftwError>;

        /// The check of [`check_side_mut`](SealedMut::check_side_mut) without the comparison
        /// of an array's shape and strides with the layout, for execution after
        /// [`Sealed::check_side`] compared them on the shared borrow: an array's storage is
        /// exposed and holds the layout's span from the first element on, the check that
        /// execution's memory safety rests on; a slice holds the span. Returns the mutable
        /// pointer that the check validated, which derives from the exclusive borrow.
        fn check_span_mut(&mut self, layout: &Layout, side: Side) -> Result<*mut E, FftwError>;
    }
}

/// A buffer of elements `E` that a plan reads.
///
/// Implemented for slices `[E]`, whose layout offsets count from the start of the slice, and
/// for RLST arrays and views with [`StridedRawAccess`], [`Shape`] and [`Stride`] (column-major,
/// row-major and strided arrays, subviews, slices and nested views), whose offsets count from
/// their first element. The trait is sealed.
///
/// An array is checked against its side of a plan's layout by shape and strides, and its
/// storage from the first element on must hold the layout's span. A slice is checked by its
/// length alone. Split plans execute on buffers of reals, `E` = `f32` or `f64`: a slice whose
/// offsets count reals from the real part of its first element, or a real array whose part
/// axis, an axis of extent 2 with the stride of the layout's gap, holds the real and
/// imaginary parts.
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::FftBuffer;
///
/// fn length<B: FftBuffer<c64> + ?Sized>(buffer: &B) -> usize {
///     FftBuffer::len(buffer)
/// }
///
/// let data = [c64::new(1.0, 0.0); 4];
/// assert_eq!(length(&data[..]), 4);
///
/// // A 2 x 2 subview starting at element 5 of a 4 x 3 array.
/// let array = rlst_dynamic_array!(c64, [4, 3]);
/// assert_eq!(length(&array.r().into_subview([1, 1], [2, 2])), 12 - 5);
/// ```
pub trait FftBuffer<E>: sealed::Sealed<E> {
    /// A pointer to the first element.
    ///
    /// For an array, the pointer to its first element in its storage, or a null pointer if
    /// the storage is not exposed as a slice.
    ///
    /// ```
    /// use rlst::{DynArray, fftw::FftBuffer};
    ///
    /// let data = [1.0_f64, 2.0];
    /// assert_eq!(FftBuffer::as_ptr(&data[..]), data.as_ptr());
    ///
    /// let array = DynArray::<f64, 2>::from_shape([4, 3]);
    /// let view = array.r().into_subview([1, 1], [2, 2]);
    /// assert_eq!(FftBuffer::as_ptr(&view), &array[[1, 1]] as *const f64);
    /// ```
    fn as_ptr(&self) -> *const E;

    /// The number of elements from the first element on.
    ///
    /// For a slice, its length. For an array, the number of storage elements from its first
    /// element to the end of its storage, `storage.len() - offset`, which a plan may address;
    /// 0 if the storage is not exposed as a slice. For a view this counts elements of other
    /// views too, and is not the number of the view's own elements.
    ///
    /// ```
    /// use rlst::{DynArray, fftw::FftBuffer};
    ///
    /// let data = [1.0_f64, 2.0];
    /// assert_eq!(FftBuffer::len(&data[..]), 2);
    ///
    /// // A column of a 4 x 3 column-major array starts at element 4 of 12.
    /// let array = DynArray::<f64, 2>::from_shape([4, 3]);
    /// assert_eq!(FftBuffer::len(&array.r().col(1)), 8);
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
/// [`StridedRawAccessMut`], [`Shape`] and [`Stride`], such as owned arrays and their `r_mut()`
/// views. A shared view such as `r()` is not a `FftBufferMut`. The trait is sealed.
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
pub trait FftBufferMut<E>: FftBuffer<E> + sealed::SealedMut<E> {
    /// A mutable pointer to the first element.
    ///
    /// For an array, it derives from the exclusive borrow of its storage
    /// ([`StridedRawAccessMut::storage_mut`]); a null pointer if the storage is not exposed as
    /// a slice.
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

/// Check that a slice of `len` elements holds `side` of `layout`.
fn check_slice(len: usize, layout: &Layout, side: Side) -> Result<(), FftwError> {
    let span = layout.span(side);
    if len < span {
        return Err(FftwError::LayoutMismatch {
            side,
            expected: format!("at least {span} elements, the layout's {side} span"),
            found: format!("a slice of {len} elements"),
        });
    }
    Ok(())
}

impl<E> sealed::Sealed<E> for [E] {
    fn check_side(&self, layout: &Layout, side: Side) -> Result<*const E, FftwError> {
        check_slice(self.len(), layout, side)?;
        Ok(self.as_ptr())
    }
}

impl<E> sealed::SealedMut<E> for [E] {
    fn check_side_mut(&mut self, layout: &Layout, side: Side) -> Result<*mut E, FftwError> {
        check_slice(self.len(), layout, side)?;
        Ok(self.as_mut_ptr())
    }

    fn check_span_mut(&mut self, layout: &Layout, side: Side) -> Result<*mut E, FftwError> {
        self.check_side_mut(layout, side)
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

// RLST arrays and views through their strided storage.

/// Check an array against `side` of `layout`, in the order of the F2 brief: its storage is
/// exposed, as `storage` (the length and the offset of the first element); its shape and
/// strides, `geometry`, fit that side, unless `geometry` is `None` because a check of the
/// shared borrow compared them already; and the storage holds the layout's span from the
/// first element on.
fn check_array(
    layout: &Layout,
    side: Side,
    storage: Option<(usize, usize)>,
    geometry: Option<sealed::ArrayGeometry<'_>>,
) -> Result<(), FftwError> {
    let mismatch = |expected: String, found: String| FftwError::LayoutMismatch {
        side,
        expected,
        found,
    };
    let Some((len, offset)) = storage else {
        return Err(mismatch(
            format!("an array whose storage is exposed as a slice, for the layout's {side} side"),
            "an array whose storage is not exposed".to_string(),
        ));
    };
    if let Some((shape, stride)) = geometry
        && !layout.fits_array_side(side, shape, stride)
    {
        return Err(mismatch(
            format!("{}, the layout's {side} side", layout.describe_side(side)),
            format!("an array of shape {shape:?} and strides {stride:?}"),
        ));
    }
    // The check that execution's memory safety rests on: `StridedRawAccess`, `Shape` and
    // `Stride` are safe to implement, so the comparison above trusts what the type claims.
    let span = layout.span(side);
    if offset.checked_add(span).is_none_or(|end| end > len) {
        return Err(mismatch(
            format!(
                "at least {span} storage elements from the first element, the layout's {side} span"
            ),
            format!("a storage of {len} elements with the first element at offset {offset}"),
        ));
    }
    Ok(())
}

impl<A, const N: usize> sealed::Sealed<A::Item> for Array<A, N>
where
    A: StridedRawAccess + Shape<N> + Stride<N>,
{
    fn check_side(&self, layout: &Layout, side: Side) -> Result<*const A::Item, FftwError> {
        let storage = self.storage();
        let parts = storage
            .as_ref()
            .map(|(slice, offset)| (slice.len(), *offset));
        check_array(layout, side, parts, Some((&self.shape(), &self.stride())))?;
        // `check_array` accepted this storage, so the offset lies inside it.
        Ok(storage.map_or(std::ptr::null(), |(slice, offset)| {
            slice.as_ptr().wrapping_add(offset)
        }))
    }

    fn with_array_geometry(&self, f: &mut dyn FnMut(Option<sealed::ArrayGeometry<'_>>)) {
        f(Some((&self.shape(), &self.stride())))
    }
}

impl<A, const N: usize> FftBuffer<A::Item> for Array<A, N>
where
    A: StridedRawAccess + Shape<N> + Stride<N>,
{
    fn as_ptr(&self) -> *const A::Item {
        // The offset of a view without elements may lie past the end of its storage (DC19),
        // so the pointer is formed with `wrapping_add`. Execution does not use this pointer
        // but the one `check_side` returns after accepting the array.
        self.storage()
            .map_or(std::ptr::null(), |(storage, offset)| {
                storage.as_ptr().wrapping_add(offset)
            })
    }

    fn len(&self) -> usize {
        self.storage()
            .map_or(0, |(storage, offset)| storage.len().saturating_sub(offset))
    }
}

impl<A, const N: usize> sealed::SealedMut<A::Item> for Array<A, N>
where
    A: StridedRawAccessMut + Shape<N> + Stride<N>,
{
    fn check_side_mut(&mut self, layout: &Layout, side: Side) -> Result<*mut A::Item, FftwError> {
        let (shape, stride) = (self.shape(), self.stride());
        self.check_storage_mut(layout, side, Some((&shape, &stride)))
    }

    fn check_span_mut(&mut self, layout: &Layout, side: Side) -> Result<*mut A::Item, FftwError> {
        self.check_storage_mut(layout, side, None)
    }
}

impl<A, const N: usize> Array<A, N>
where
    A: StridedRawAccessMut + Shape<N> + Stride<N>,
{
    /// The check of [`check_array`] through the exclusive borrow of the storage, comparing
    /// `geometry` with the layout if it is given.
    fn check_storage_mut(
        &mut self,
        layout: &Layout,
        side: Side,
        geometry: Option<sealed::ArrayGeometry<'_>>,
    ) -> Result<*mut A::Item, FftwError> {
        let storage = self.storage_mut();
        let parts = storage
            .as_ref()
            .map(|(slice, offset)| (slice.len(), *offset));
        check_array(layout, side, parts, geometry)?;
        // From the exclusive borrow of the storage that `check_array` accepted (design
        // Section 4.4), so that FFTW may write through it.
        Ok(storage.map_or(std::ptr::null_mut(), |(slice, offset)| {
            slice.as_mut_ptr().wrapping_add(offset)
        }))
    }
}

impl<A, const N: usize> FftBufferMut<A::Item> for Array<A, N>
where
    A: StridedRawAccessMut + Shape<N> + Stride<N>,
{
    fn as_mut_ptr(&mut self) -> *mut A::Item {
        // From the exclusive borrow of the storage, never from `storage()` (design Section
        // 4.4).
        self.storage_mut()
            .map_or(std::ptr::null_mut(), |(storage, offset)| {
                storage.as_mut_ptr().wrapping_add(offset)
            })
    }
}
