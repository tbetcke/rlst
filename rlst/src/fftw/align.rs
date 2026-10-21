//! Alignment classes of data pointers.

use std::fmt;

use super::buffer::FftBuffer;

/// The modulus of an [`AlignmentClass`], in bytes; equal to [`FFTW_ALIGNED`](crate::FFTW_ALIGNED).
///
/// Buffers that start at a multiple of `PLAN_ALIGNMENT` have class 0, the class plans are
/// made for by default.
///
/// ```
/// assert_eq!(rlst::fftw::PLAN_ALIGNMENT, rlst::FFTW_ALIGNED);
/// ```
pub const PLAN_ALIGNMENT: usize = crate::FFTW_ALIGNED;

// Classes are stored in a u8, and the rule of design Section 4.6 is stated for 64 bytes.
const _: () = assert!(PLAN_ALIGNMENT == 64);

/// A data pointer's address modulo [`PLAN_ALIGNMENT`], in bytes.
///
/// A plan runs on any buffer whose class equals the class of the buffer it was planned for.
/// Equal classes modulo 64 imply equal alignment in FFTW's sense on every FFTW build, so the
/// check needs no call into FFTW. Plans are made for class 0 unless
/// [`Planner::alignment`](super::Planner::alignment) asks for another.
///
/// ```
/// use rlst::{c64, rlst_dynamic_array};
/// use rlst::fftw::AlignmentClass;
///
/// let buffer = rlst_dynamic_array!(c64, [8] | rlst::FFTW_ALIGNED);
/// let data = buffer.data().unwrap();
/// assert_eq!(AlignmentClass::of(data), AlignmentClass::default());
/// // One complex double further on, the class is 16 bytes.
/// assert_eq!(AlignmentClass::of(&data[1..]).offset(), 16);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AlignmentClass(u8);

impl AlignmentClass {
    /// The class of a pointer.
    ///
    /// ```
    /// use rlst::fftw::AlignmentClass;
    ///
    /// let values = [0.0_f64; 4];
    /// let class = AlignmentClass::of_ptr(values.as_ptr());
    /// assert_eq!(class.offset() % std::mem::align_of::<f64>(), 0);
    /// ```
    pub fn of_ptr<E>(ptr: *const E) -> Self {
        Self::from_offset(ptr.addr())
    }

    /// The class of a buffer: of its first element.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::AlignmentClass;
    ///
    /// let buffer = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// assert_eq!(AlignmentClass::of(buffer.data().unwrap()).offset(), 0);
    /// ```
    pub fn of<E, B: FftBuffer<E> + ?Sized>(buffer: &B) -> Self {
        Self::of_ptr(buffer.as_ptr())
    }

    /// The class of a buffer that starts `bytes` bytes after a multiple of
    /// [`PLAN_ALIGNMENT`].
    ///
    /// Use it to plan for buffers that are not yet allocated, for example for the units of a
    /// larger allocation that start at a fixed offset.
    ///
    /// ```
    /// use rlst::fftw::AlignmentClass;
    ///
    /// assert_eq!(AlignmentClass::from_offset(80).offset(), 16);
    /// assert_eq!(AlignmentClass::from_offset(0), AlignmentClass::default());
    /// ```
    pub fn from_offset(bytes: usize) -> Self {
        // The remainder is below PLAN_ALIGNMENT, which fits in a u8.
        Self((bytes % PLAN_ALIGNMENT) as u8)
    }

    /// The address modulo [`PLAN_ALIGNMENT`], in bytes.
    ///
    /// ```
    /// use rlst::fftw::AlignmentClass;
    ///
    /// assert_eq!(AlignmentClass::from_offset(24).offset(), 24);
    /// ```
    pub fn offset(self) -> usize {
        usize::from(self.0)
    }
}

impl fmt::Display for AlignmentClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} mod {}", self.0, PLAN_ALIGNMENT)
    }
}

#[cfg(test)]
mod tests {
    use aligned_vec::{AVec, ConstAlign};

    use super::{AlignmentClass, PLAN_ALIGNMENT};

    #[test]
    fn test_alignment_class_of_offsets() {
        let buffer = AVec::<u8, ConstAlign<PLAN_ALIGNMENT>>::from_iter(PLAN_ALIGNMENT, [0; 256]);
        let base = buffer.as_ptr();
        assert_eq!(base.addr() % 64, 0);
        for (offset, class) in [(0, 0), (8, 8), (16, 16), (64, 0), (72, 8)] {
            let ptr = base.wrapping_add(offset);
            assert_eq!(
                AlignmentClass::of_ptr(ptr).offset(),
                class,
                "offset {offset}"
            );
            assert_eq!(AlignmentClass::of(&buffer[offset..]).offset(), class);
            assert_eq!(AlignmentClass::from_offset(offset).offset(), class);
        }
        assert_eq!(AlignmentClass::of_ptr(base), AlignmentClass::default());
        assert_eq!(AlignmentClass::from_offset(16).to_string(), "16 mod 64");
    }
}
