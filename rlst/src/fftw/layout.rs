//! Transform layouts: dimensions, batch dimensions and placement.

use std::fmt;

use crate::{Array, Shape, Stride, StridedRawAccess};

use super::error::{FftwError, Side};

/// One dimension of a layout: its extent and its input and output strides.
///
/// Strides are counted in elements of the input and output buffers (complex elements for
/// complex-to-complex transforms), and are non-negative. The fields are `isize`, as FFTW's
/// `fftw_iodim64` uses `ptrdiff_t`.
///
/// ```
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
/// ```
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
        f.write_str(match self {
            Placement::OutOfPlace => "out-of-place",
            Placement::InPlace => "in-place",
        })
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
/// ```
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
    dims: Vec<Dim>,
    batch: Vec<Dim>,
    placement: Placement,
    input_span: usize,
    output_span: usize,
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
    /// ```
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
        let invalid = |dimension: &str, reason: String| FftwError::InvalidLayout {
            dimension: dimension.to_string(),
            reason,
        };

        if dims.is_empty() {
            return Err(invalid(
                "layout",
                "a layout needs at least one transform dimension".to_string(),
            ));
        }

        let named: Vec<(String, Dim)> = dims
            .iter()
            .enumerate()
            .map(|(index, &dim)| (format!("transform dimension {index}"), dim))
            .chain(
                batch
                    .iter()
                    .enumerate()
                    .map(|(index, &dim)| (format!("batch dimension {index}"), dim)),
            )
            .collect();

        for (name, dim) in &named {
            if dim.n < 1 {
                return Err(invalid(name, format!("the extent {} is below 1", dim.n)));
            }
        }
        for (name, dim) in &named {
            if dim.is < 0 || dim.os < 0 {
                return Err(invalid(
                    name,
                    format!(
                        "the strides must be non-negative, found input stride {} and output \
                         stride {}",
                        dim.is, dim.os
                    ),
                ));
            }
        }
        if placement == Placement::InPlace {
            for (name, dim) in &named {
                if dim.is != dim.os {
                    return Err(invalid(
                        name,
                        format!(
                            "in place, the input stride {} and the output stride {} must be equal",
                            dim.is, dim.os
                        ),
                    ));
                }
            }
        }

        // Distinct outputs, by the sufficient rule of design Section 4.3.
        let mut by_output: Vec<&(String, Dim)> =
            named.iter().filter(|(_, dim)| dim.n > 1).collect();
        by_output.sort_by_key(|(_, dim)| dim.os);
        if let Some((name, dim)) = by_output.first()
            && dim.os == 0
        {
            return Err(invalid(
                name,
                format!(
                    "the output stride is 0 with extent {}, so the outputs overlap",
                    dim.n
                ),
            ));
        }
        for pair in by_output.windows(2) {
            let (inner_name, inner) = pair[0];
            let (name, dim) = pair[1];
            let Some(needed) = inner.n.checked_mul(inner.os) else {
                return Err(invalid(
                    inner_name,
                    "the extent times the output stride overflows isize".to_string(),
                ));
            };
            if dim.os < needed {
                return Err(invalid(
                    name,
                    format!(
                        "the output stride {} is below {needed}, the extent {} times the output \
                         stride {} of {inner_name}, so the outputs may overlap",
                        dim.os, inner.n, inner.os
                    ),
                ));
            }
        }

        let span = |stride: fn(&Dim) -> isize| {
            named
                .iter()
                .try_fold(1_isize, |span, (_, dim)| {
                    span.checked_add((dim.n - 1).checked_mul(stride(dim))?)
                })
                .and_then(|span| usize::try_from(span).ok())
        };
        let input_span = span(|dim| dim.is)
            .ok_or_else(|| invalid("layout", "the input span overflows isize".to_string()))?;
        let output_span = span(|dim| dim.os)
            .ok_or_else(|| invalid("layout", "the output span overflows isize".to_string()))?;

        Ok(Self {
            dims: dims.to_vec(),
            batch: normalise_batch(batch),
            placement,
            input_span,
            output_span,
        })
    }

    /// The transform dimensions, in the order given.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let dim = Dim { n: 5, is: 1, os: 1 };
    /// let layout = Layout::new(&[dim], &[], Placement::OutOfPlace)?;
    /// assert_eq!(layout.dims(), &[dim]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    /// The batch dimensions, normalised.
    ///
    /// ```
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
        &self.batch
    }

    /// The placement.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 1, os: 1 }], &[], Placement::InPlace)?;
    /// assert_eq!(layout.placement(), Placement::InPlace);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// The number of input elements the layout spans: 1 plus the sum of (n − 1) · is over
    /// all dimensions.
    ///
    /// An input buffer must have at least this many elements.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 3, os: 1 }], &[], Placement::OutOfPlace)?;
    /// assert_eq!(layout.input_span(), 10);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn input_span(&self) -> usize {
        self.input_span
    }

    /// The number of output elements the layout spans: 1 plus the sum of (n − 1) · os over
    /// all dimensions.
    ///
    /// An output buffer must have at least this many elements.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let layout = Layout::new(&[Dim { n: 4, is: 3, os: 1 }], &[], Placement::OutOfPlace)?;
    /// assert_eq!(layout.output_span(), 4);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn output_span(&self) -> usize {
        self.output_span
    }

    /// The logical size of one transform: the product of the transform extents.
    ///
    /// A backward transform after a forward one multiplies the data by this number.
    ///
    /// ```
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
        // The outputs are distinct, so the product is at most the output span.
        self.dims.iter().map(|dim| dim.n as usize).product()
    }

    /// The span of `side`, in elements.
    pub(crate) fn span(&self, side: Side) -> usize {
        match side {
            Side::Input => self.input_span,
            Side::Output => self.output_span,
        }
    }

    /// A description for error messages.
    pub(crate) fn describe(&self) -> String {
        let list = |dims: &[Dim]| {
            dims.iter()
                .map(|dim| format!("(n {}, is {}, os {})", dim.n, dim.is, dim.os))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "the {} layout with transform dimensions [{}] and batch dimensions [{}]",
            self.placement,
            list(&self.dims),
            list(&self.batch)
        )
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

/// Normalise batch dimensions (design Section 4.3): drop those of extent 1, sort by input
/// stride and then output stride, largest first, and merge each adjacent pair whose outer
/// strides are the inner extent times the inner strides, on both sides.
fn normalise_batch(batch: &[Dim]) -> Vec<Dim> {
    let mut sorted: Vec<Dim> = batch.iter().copied().filter(|dim| dim.n > 1).collect();
    sorted.sort_by_key(|dim| std::cmp::Reverse((dim.is, dim.os)));

    // Merge from the innermost dimension outwards.
    let mut merged: Vec<Dim> = Vec::with_capacity(sorted.len());
    for dim in sorted.into_iter().rev() {
        if let Some(inner) = merged.last_mut()
            && inner.n.checked_mul(inner.is) == Some(dim.is)
            && inner.n.checked_mul(inner.os) == Some(dim.os)
            && let Some(n) = inner.n.checked_mul(dim.n)
        {
            inner.n = n;
        } else {
            merged.push(dim);
        }
    }
    merged.reverse();
    merged
}

#[cfg(test)]
mod tests {
    use super::{Dim, Layout, Placement};
    use crate::fftw::FftwError;

    fn dim(n: isize, is: isize, os: isize) -> Dim {
        Dim { n, is, os }
    }

    fn invalid(dims: &[Dim], batch: &[Dim], placement: Placement) -> (String, String) {
        match Layout::new(dims, batch, placement) {
            Err(FftwError::InvalidLayout { dimension, reason }) => (dimension, reason),
            other => panic!("expected InvalidLayout, found {other:?}"),
        }
    }

    #[test]
    fn test_contiguous_batch_merges() {
        // A [4, 2, 3] block, transformed along the first axis: the batch over the other two
        // axes is contiguous, in either order.
        let dims = [dim(4, 1, 1)];
        let batch = [dim(2, 4, 4), dim(3, 8, 8)];
        let layout = Layout::new(&dims, &batch, Placement::OutOfPlace).unwrap();
        assert_eq!(layout.batch(), &[dim(6, 4, 4)]);
        assert_eq!(layout.dims(), &dims);

        let reversed = [batch[1], batch[0]];
        assert_eq!(
            Layout::new(&dims, &reversed, Placement::OutOfPlace).unwrap(),
            layout
        );
    }

    #[test]
    fn test_transposed_batch_does_not_merge() {
        // Contiguous on the input side, transposed on the output side.
        let dims = [dim(4, 1, 1)];
        let batch = [dim(2, 4, 12), dim(3, 8, 4)];
        let layout = Layout::new(&dims, &batch, Placement::OutOfPlace).unwrap();
        assert_eq!(layout.batch(), &[dim(3, 8, 4), dim(2, 4, 12)]);
    }

    #[test]
    fn test_extent_one_is_dropped() {
        let dims = [dim(4, 1, 1)];
        let with_ones = [dim(1, 100, 7), dim(3, 4, 4), dim(1, 0, 0)];
        let layout = Layout::new(&dims, &with_ones, Placement::OutOfPlace).unwrap();
        assert_eq!(layout.batch(), &[dim(3, 4, 4)]);
        assert_eq!(
            layout,
            Layout::new(&dims, &[dim(3, 4, 4)], Placement::OutOfPlace).unwrap()
        );
        // Transform dimensions of extent 1 are kept.
        let layout = Layout::new(&[dim(1, 1, 1), dim(4, 1, 1)], &[], Placement::OutOfPlace);
        assert_eq!(layout.unwrap().dims().len(), 2);
    }

    #[test]
    fn test_batch_order_does_not_matter() {
        let dims = [dim(5, 3, 3)];
        let batch = [dim(3, 1, 1), dim(2, 15, 15), dim(4, 30, 30)];
        let reference = Layout::new(&dims, &batch, Placement::InPlace).unwrap();
        for permutation in [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let permuted = permutation.map(|index| batch[index]);
            assert_eq!(
                Layout::new(&dims, &permuted, Placement::InPlace).unwrap(),
                reference
            );
        }
        // The strides 15 and 30 merge into one dimension; stride 1 does not merge.
        assert_eq!(reference.batch(), &[dim(8, 15, 15), dim(3, 1, 1)]);
    }

    #[test]
    fn test_spans() {
        let layout = Layout::new(
            &[dim(4, 3, 2), dim(5, 12, 8)],
            &[dim(2, 100, 50)],
            Placement::OutOfPlace,
        )
        .unwrap();
        assert_eq!(layout.input_span(), 1 + 3 * 3 + 4 * 12 + 100);
        assert_eq!(layout.output_span(), 1 + 3 * 2 + 4 * 8 + 50);
        assert_eq!(layout.logical_size(), 20);

        // A broadcast input: batch stride 0 on the input side.
        let layout = Layout::new(&[dim(4, 1, 1)], &[dim(3, 0, 4)], Placement::OutOfPlace).unwrap();
        assert_eq!(layout.input_span(), 4);
        assert_eq!(layout.output_span(), 12);
    }

    #[test]
    fn test_no_transform_dimension() {
        let (dimension, _) = invalid(&[], &[dim(3, 1, 1)], Placement::OutOfPlace);
        assert_eq!(dimension, "layout");
    }

    #[test]
    fn test_extent_zero() {
        let (dimension, reason) = invalid(&[dim(0, 1, 1)], &[], Placement::OutOfPlace);
        assert_eq!(dimension, "transform dimension 0");
        assert!(reason.contains("extent 0"), "{reason}");
        let (dimension, _) = invalid(&[dim(4, 1, 1)], &[dim(0, 4, 4)], Placement::OutOfPlace);
        assert_eq!(dimension, "batch dimension 0");
    }

    #[test]
    fn test_negative_stride() {
        let (dimension, _) = invalid(&[dim(4, -1, 1)], &[], Placement::OutOfPlace);
        assert_eq!(dimension, "transform dimension 0");
        let (dimension, _) = invalid(&[dim(4, 1, 1)], &[dim(2, 4, -4)], Placement::OutOfPlace);
        assert_eq!(dimension, "batch dimension 0");
    }

    #[test]
    fn test_unequal_strides_in_place() {
        let (dimension, _) = invalid(&[dim(4, 1, 2)], &[], Placement::InPlace);
        assert_eq!(dimension, "transform dimension 0");
        // Out of place the same dimension is valid.
        assert!(Layout::new(&[dim(4, 1, 2)], &[], Placement::OutOfPlace).is_ok());
    }

    #[test]
    fn test_output_stride_zero() {
        let (dimension, reason) = invalid(&[dim(2, 1, 0)], &[], Placement::OutOfPlace);
        assert_eq!(dimension, "transform dimension 0");
        assert!(reason.contains("stride is 0"), "{reason}");
        // An output stride 0 with extent 1 addresses one element and is valid.
        assert!(Layout::new(&[dim(1, 1, 0)], &[], Placement::OutOfPlace).is_ok());
    }

    #[test]
    fn test_overlapping_outputs() {
        // Two dimensions that address the same outputs.
        let (dimension, _) = invalid(&[dim(4, 1, 1)], &[dim(2, 4, 1)], Placement::OutOfPlace);
        assert_eq!(dimension, "batch dimension 0");
        // Interleaved outputs: stride 2 below 4 x 1.
        let (dimension, _) = invalid(&[dim(4, 1, 1)], &[dim(2, 4, 2)], Placement::OutOfPlace);
        assert_eq!(dimension, "batch dimension 0");
        // The input may repeat elements out of place.
        assert!(Layout::new(&[dim(4, 1, 1)], &[dim(2, 0, 4)], Placement::OutOfPlace).is_ok());
    }

    #[test]
    fn test_span_overflow() {
        let (dimension, reason) = invalid(&[dim(4, isize::MAX / 2, 1)], &[], Placement::OutOfPlace);
        assert_eq!(dimension, "layout");
        assert!(reason.contains("input span"), "{reason}");
    }
}
