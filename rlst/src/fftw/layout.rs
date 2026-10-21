//! Transform layouts: dimensions, batch dimensions and placement.

use std::fmt;

use crate::{Array, Shape, Stride, StridedRawAccess};

use super::error::FftwError;

/// One dimension of a layout: its extent and its input and output strides.
///
/// Strides are counted in elements of the input and output buffers (complex elements for
/// complex-to-complex transforms), and are non-negative. The fields are `isize`, as FFTW's
/// `fftw_iodim64` uses `ptrdiff_t`.
///
/// ```no_run
/// use rlst::fftw::Dim;
///
/// // 8 contiguous elements on both sides.
/// let dim = Dim { n: 8, is: 1, os: 1 };
/// assert_eq!(dim.n, 8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dim {
    /// The extent.
    pub n: isize,
    /// The input stride, in elements.
    pub is: isize,
    /// The output stride, in elements.
    pub os: isize,
}

/// Whether a transform writes to a separate output buffer or overwrites its input.
///
/// ```no_run
/// use rlst::fftw::Placement;
///
/// assert_eq!(Placement::InPlace.to_string(), "in-place");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Input and output are separate buffers, which must not overlap.
    OutOfPlace,
    /// The output overwrites the input in one buffer.
    InPlace,
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = f;
        todo!()
    }
}

/// The layout of a batch of transforms, in normalised form.
///
/// A layout lists the transform dimensions, in the order given, and the batch dimensions over
/// which the transform is repeated, with the placement. Offsets are counted in elements from
/// the start of each buffer. A layout is validated when it is made, and normalised: batch
/// dimensions of extent 1 are dropped, the batch dimensions are sorted by input stride and
/// then output stride, largest first, and adjacent batch dimensions that are contiguous on
/// both sides are merged. Two layouts that describe the same transforms of the same elements
/// are therefore equal, whatever the order of their batch dimensions.
///
/// ```no_run
/// use rlst::fftw::{Dim, Layout, Placement};
///
/// // Three transforms of length 8, stored one after the other.
/// let layout = Layout::new(
///     &[Dim { n: 8, is: 1, os: 1 }],
///     &[Dim { n: 3, is: 8, os: 8 }],
///     Placement::OutOfPlace,
/// )?;
/// assert_eq!(layout.input_span(), 24);
/// assert_eq!(layout.logical_size(), 8);
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    // The fields follow in F1c.
}

impl Layout {
    /// Make a layout from transform dimensions, batch dimensions and a placement.
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] with the
    /// dimension and the reason:
    ///
    /// 1. there is at least one transform dimension;
    /// 2. every extent is at least 1;
    /// 3. every stride is non-negative;
    /// 4. in place, the input and output strides are equal in every dimension;
    /// 5. the output elements are distinct: with the dimensions of extent above 1 (transform
    ///    and batch together) sorted by output stride, every output stride is positive and at
    ///    least the extent times the stride of the next smaller one;
    /// 6. the spans fit in `isize`.
    ///
    /// The input may address an element more than once, for example with a batch stride of
    /// 0, unless the layout is in place.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, FftwError, Layout, Placement};
    ///
    /// // A 4 x 6 transform, row-major, out of place.
    /// let layout = Layout::new(
    ///     &[Dim { n: 4, is: 6, os: 6 }, Dim { n: 6, is: 1, os: 1 }],
    ///     &[],
    ///     Placement::OutOfPlace,
    /// )?;
    /// assert_eq!(layout.logical_size(), 24);
    ///
    /// // In place, the strides of both sides must agree.
    /// let error = Layout::new(&[Dim { n: 4, is: 1, os: 2 }], &[], Placement::InPlace);
    /// assert!(matches!(error, Err(FftwError::InvalidLayout { .. })));
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn new(dims: &[Dim], batch: &[Dim], placement: Placement) -> Result<Self, FftwError> {
        let _ = (dims, batch, placement);
        todo!()
    }

    /// The transform dimensions, in the order given.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let dim = Dim { n: 5, is: 1, os: 1 };
    /// let layout = Layout::new(&[dim], &[], Placement::OutOfPlace)?;
    /// assert_eq!(layout.dims(), &[dim]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn dims(&self) -> &[Dim] {
        todo!()
    }

    /// The batch dimensions, normalised.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// // A batch of 2 x 3 contiguous transforms of length 4 merges into one of 6.
    /// let layout = Layout::new(
    ///     &[Dim { n: 4, is: 1, os: 1 }],
    ///     &[Dim { n: 2, is: 12, os: 12 }, Dim { n: 3, is: 4, os: 4 }],
    ///     Placement::OutOfPlace,
    /// )?;
    /// assert_eq!(layout.batch(), &[Dim { n: 6, is: 4, os: 4 }]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn batch(&self) -> &[Dim] {
        todo!()
    }

    /// The placement.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 1, os: 1 }], &[], Placement::InPlace)?;
    /// assert_eq!(layout.placement(), Placement::InPlace);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn placement(&self) -> Placement {
        todo!()
    }

    /// The number of input elements the layout spans: 1 plus the sum of (n − 1) · is over
    /// all dimensions.
    ///
    /// An input buffer must have at least this many elements.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 3, os: 1 }], &[], Placement::OutOfPlace)?;
    /// assert_eq!(layout.input_span(), 10);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn input_span(&self) -> usize {
        todo!()
    }

    /// The number of output elements the layout spans: 1 plus the sum of (n − 1) · os over
    /// all dimensions.
    ///
    /// An output buffer must have at least this many elements.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 3, os: 1 }], &[], Placement::OutOfPlace)?;
    /// assert_eq!(layout.output_span(), 4);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn output_span(&self) -> usize {
        todo!()
    }

    /// The logical size of one transform: the product of the transform extents.
    ///
    /// A backward transform after a forward one multiplies the data by this number.
    ///
    /// ```no_run
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(
    ///     &[Dim { n: 3, is: 5, os: 5 }, Dim { n: 5, is: 1, os: 1 }],
    ///     &[Dim { n: 7, is: 15, os: 15 }],
    ///     Placement::OutOfPlace,
    /// )?;
    /// assert_eq!(layout.logical_size(), 15);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn logical_size(&self) -> usize {
        todo!()
    }

    /// One-dimensional transforms along `axis` of two arrays of rank `N`, batched over the
    /// other axes.
    ///
    /// The input and output may differ in element type and shape; the planner checks them
    /// against the transform family. Implemented in task F2, which also implements
    /// [`StridedRawAccess`] for RLST's arrays and views and makes this example run.
    ///
    /// ```ignore
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// // Three transforms of length 8, one per column.
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn along_axis<AIn, AOut, const N: usize>(
        input: &Array<AIn, N>,
        output: &Array<AOut, N>,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        AIn: StridedRawAccess + Shape<N> + Stride<N>,
        AOut: StridedRawAccess + Shape<N> + Stride<N>,
    {
        let _ = (input, output, axis);
        todo!()
    }

    /// Transforms over `axes`, in the order given, of two arrays of rank `N`, batched over the
    /// other axes.
    ///
    /// The input and output may differ in element type and shape; the planner checks them
    /// against the transform family. Implemented in task F2, which also implements
    /// [`StridedRawAccess`] for RLST's arrays and views and makes this example run.
    ///
    /// ```ignore
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(c64, [8, 6, 3] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [8, 6, 3] | rlst::FFTW_ALIGNED);
    /// // Three two-dimensional 8 x 6 transforms.
    /// let layout = Layout::over_axes(&input, &output, &[0, 1])?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn over_axes<AIn, AOut, const N: usize>(
        input: &Array<AIn, N>,
        output: &Array<AOut, N>,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        AIn: StridedRawAccess + Shape<N> + Stride<N>,
        AOut: StridedRawAccess + Shape<N> + Stride<N>,
    {
        let _ = (input, output, axes);
        todo!()
    }

    /// In-place one-dimensional transforms along `axis` of an array of rank `N`, batched over
    /// the other axes. Implemented in task F2, which also implements [`StridedRawAccess`] for
    /// RLST's arrays and views and makes this example run.
    ///
    /// ```ignore
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let data = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis_inplace(&data, 0)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn along_axis_inplace<A, const N: usize>(
        data: &Array<A, N>,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        A: StridedRawAccess + Shape<N> + Stride<N>,
    {
        let _ = (data, axis);
        todo!()
    }

    /// In-place transforms over `axes`, in the order given, of an array of rank `N`, batched
    /// over the other axes. Implemented in task F2, which also implements
    /// [`StridedRawAccess`] for RLST's arrays and views and makes this example run.
    ///
    /// ```ignore
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let data = rlst_dynamic_array!(c64, [8, 6, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::over_axes_inplace(&data, &[0, 1])?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn over_axes_inplace<A, const N: usize>(
        data: &Array<A, N>,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        A: StridedRawAccess + Shape<N> + Stride<N>,
    {
        let _ = (data, axes);
        todo!()
    }
}
