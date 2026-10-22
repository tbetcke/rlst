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
        Self::with_names(
            dims,
            batch,
            placement,
            |index| format!("transform dimension {index}"),
            |index| format!("batch dimension {index}"),
        )
    }

    /// [`Layout::new`], with the names its errors give each transform and batch dimension.
    fn with_names(
        dims: &[Dim],
        batch: &[Dim],
        placement: Placement,
        dim_name: impl Fn(usize) -> String,
        batch_name: impl Fn(usize) -> String,
    ) -> Result<Self, FftwError> {
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
            .map(|(index, &dim)| (dim_name(index), dim))
            .chain(
                batch
                    .iter()
                    .enumerate()
                    .map(|(index, &dim)| (batch_name(index), dim)),
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
    /// The same as [`over_axes`](Layout::over_axes) with the one axis `axis`, whose checks,
    /// errors and notes apply. The arrays can be any RLST arrays or views with strided
    /// storage: column-major, row-major or strided, subviews, slices and nested views.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Direction, Layout, Planner};
    ///
    /// let mut input = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// input.fill_from_seed_equally_distributed(0);
    ///
    /// // Three transforms of length 8, one per column.
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// assert_eq!(layout.dims(), &[Dim { n: 8, is: 1, os: 1 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 3, is: 8, os: 8 }]);
    ///
    /// let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
    /// plan.execute(&input, &mut output)?;
    ///
    /// // The first entry of each column's transform is the sum of the column.
    /// for j in 0..3 {
    ///     let sum: c64 = (0..8).map(|i| input[[i, j]]).sum();
    ///     assert!((output[[0, j]] - sum).norm() < 1e-12);
    /// }
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// Arrays without a [`Stride`], such as the lazy view of `transpose()`, are not accepted,
    /// so this does not compile:
    ///
    /// ```compile_fail,E0277
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [3, 8] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&input.r().transpose(), &output, 0)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// while the same code on an array with strides does:
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [3, 8] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&input.r(), &output, 0);
    /// // The shapes differ, which the layout functions reject.
    /// assert!(layout.is_err());
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
        Self::over_axes(input, output, &[axis])
    }

    /// Transforms over `axes`, in the order given, of two arrays of rank `N`, batched over the
    /// other axes.
    ///
    /// The transform dimensions are the axes of `axes`, in that order, each with its extent
    /// and the strides of both arrays; the batch dimensions are the other axes. FFTW receives
    /// the dimensions in this order with their own strides, whatever the memory order of the
    /// arrays, so `&[0, 1]` and `&[1, 0]` are different layouts, and plans, of the same
    /// transform. The layout is then made by [`Layout::new`], whose validation and
    /// normalisation apply.
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] that names
    /// `axes` or the axis concerned (`axis 1`):
    ///
    /// 1. `axes` is not empty;
    /// 2. every entry of `axes` is below `N`;
    /// 3. no entry is repeated;
    /// 4. the input and output extents are equal on every axis, as a complex-to-complex
    ///    transform needs;
    /// 5. every extent is at least 1: an array with an extent 0 has nothing to transform;
    /// 6. every extent and stride fits in `isize`;
    ///
    /// and those of [`Layout::new`], with the dimensions named by their axes. In particular,
    /// an output whose strides fail the sufficient distinct-output rule of [`Layout::new`],
    /// such as shape `[2, 2]` with strides `[2, 3]`, is rejected, although its elements are
    /// distinct; it is accepted as an input.
    ///
    /// Building a layout reads only shapes and strides. A layout carries no offsets: the
    /// offsets of an array count from its first element, so a plan runs on any arrays and
    /// views of this memory layout, wherever they start.
    ///
    /// The scratch of [`Planner::c2c`](super::Planner::c2c) covers the span from the first
    /// element to the last one the layout addresses. For a view of a large array, such as a
    /// column of a row-major matrix, that can be nearly the whole array: raise
    /// [`max_scratch`](super::Planner::max_scratch) or plan on the arrays themselves with
    /// [`c2c_on`](super::Planner::c2c_on).
    ///
    /// A plan describes memory, not logical axes. Execution checks each array against its
    /// side of the layout and, when both are arrays of the same rank, pairs their axes by
    /// index, but no check can tell apart two arrays with the same memory layout: a plan
    /// along axis 0 of a square column-major array transforms along axis 1 of a square
    /// row-major one.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Layout};
    ///
    /// let input = rlst_dynamic_array!(c64, [4, 5, 6] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [4, 5, 6] | rlst::FFTW_ALIGNED);
    /// // Six two-dimensional 4 x 5 transforms, one per index of the last axis.
    /// let layout = Layout::over_axes(&input, &output, &[0, 1])?;
    /// assert_eq!(layout.dims(), &[Dim { n: 4, is: 1, os: 1 }, Dim { n: 5, is: 4, os: 4 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 6, is: 20, os: 20 }]);
    /// assert_eq!(layout.logical_size(), 20);
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
        Self::from_arrays(
            (input.shape(), input.stride()),
            (output.shape(), output.stride()),
            axes,
            Placement::OutOfPlace,
        )
    }

    /// In-place one-dimensional transforms along `axis` of an array of rank `N`, batched over
    /// the other axes.
    ///
    /// The same as [`over_axes_inplace`](Layout::over_axes_inplace) with the one axis `axis`.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let data = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
    /// // Three in-place transforms of length 8, one per column.
    /// let layout = Layout::along_axis_inplace(&data, 0)?;
    /// assert_eq!(layout.placement(), Placement::InPlace);
    /// assert_eq!(layout.dims(), &[Dim { n: 8, is: 1, os: 1 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 3, is: 8, os: 8 }]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn along_axis_inplace<A, const N: usize>(
        data: &Array<A, N>,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        A: StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::over_axes_inplace(data, &[axis])
    }

    /// In-place transforms over `axes`, in the order given, of an array of rank `N`, batched
    /// over the other axes.
    ///
    /// As [`over_axes`](Layout::over_axes) with `data` as both input and output, whose checks,
    /// errors and notes apply; the input and output strides of every dimension are the
    /// array's strides, and the placement is [`Placement::InPlace`].
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, Layout};
    ///
    /// let data = rlst_dynamic_array!(c64, [4, 5, 6] | rlst::FFTW_ALIGNED);
    /// // Four in-place two-dimensional 6 x 5 transforms over the last two axes.
    /// let layout = Layout::over_axes_inplace(&data, &[2, 1])?;
    /// assert_eq!(layout.dims(), &[Dim { n: 6, is: 20, os: 20 }, Dim { n: 5, is: 4, os: 4 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 4, is: 1, os: 1 }]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn over_axes_inplace<A, const N: usize>(
        data: &Array<A, N>,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        A: StridedRawAccess + Shape<N> + Stride<N>,
    {
        let geometry = (data.shape(), data.stride());
        Self::from_arrays(geometry, geometry, axes, Placement::InPlace)
    }

    /// The layout of transforms over `axes` of arrays with the given shapes and strides, with
    /// the checks of [`over_axes`](Layout::over_axes).
    fn from_arrays<const N: usize>(
        (input_shape, input_stride): ([usize; N], [usize; N]),
        (output_shape, output_stride): ([usize; N], [usize; N]),
        axes: &[usize],
        placement: Placement,
    ) -> Result<Self, FftwError> {
        let invalid =
            |dimension: String, reason: String| FftwError::InvalidLayout { dimension, reason };
        let name = |axis: usize| format!("axis {axis}");

        if axes.is_empty() {
            return Err(invalid(
                "axes".to_string(),
                "no axis is given, and a transform needs at least one".to_string(),
            ));
        }
        if let Some(&axis) = axes.iter().find(|&&axis| axis >= N) {
            return Err(invalid(
                "axes".to_string(),
                format!("the axis {axis} is not below the rank {N}"),
            ));
        }
        for (position, &axis) in axes.iter().enumerate() {
            if axes[..position].contains(&axis) {
                return Err(invalid(
                    name(axis),
                    format!("the axis is listed more than once in {axes:?}"),
                ));
            }
        }
        for axis in 0..N {
            if input_shape[axis] != output_shape[axis] {
                return Err(invalid(
                    name(axis),
                    format!(
                        "the input extent {} differs from the output extent {}, and a \
                         complex-to-complex transform needs arrays of equal shape",
                        input_shape[axis], output_shape[axis]
                    ),
                ));
            }
        }
        if let Some(axis) = (0..N).find(|&axis| input_shape[axis] == 0) {
            return Err(invalid(
                name(axis),
                "the extent is 0, so there is nothing to transform".to_string(),
            ));
        }
        let mut dim_of = Vec::with_capacity(N);
        for axis in 0..N {
            let convert = |what: &str, value: usize| {
                isize::try_from(value).map_err(|_| {
                    invalid(
                        name(axis),
                        format!("the {what} {value} does not fit in isize"),
                    )
                })
            };
            dim_of.push(Dim {
                n: convert("extent", input_shape[axis])?,
                is: convert("input stride", input_stride[axis])?,
                os: convert("output stride", output_stride[axis])?,
            });
        }

        let batch_axes: Vec<usize> = (0..N).filter(|axis| !axes.contains(axis)).collect();
        let dims: Vec<Dim> = axes.iter().map(|&axis| dim_of[axis]).collect();
        let batch: Vec<Dim> = batch_axes.iter().map(|&axis| dim_of[axis]).collect();
        Self::with_names(
            &dims,
            &batch,
            placement,
            |index| name(axes[index]),
            |index| name(batch_axes[index]),
        )
    }

    /// Whether an array of `shape` and `stride` fits `side` of the layout (design Section
    /// 4.8 item 2, DC15): each transform dimension is matched by an axis of the same extent
    /// and, above extent 1, the same stride, and the other axes, normalised on this side
    /// alone, equal the layout's batch on this side, normalised in the same way.
    pub(crate) fn fits_array_side(&self, side: Side, shape: &[usize], stride: &[usize]) -> bool {
        let on_side = |dim: &Dim| {
            let stride = match side {
                Side::Input => dim.is,
                Side::Output => dim.os,
            };
            Dim {
                n: dim.n,
                is: stride,
                os: stride,
            }
        };
        let Some(dims) = array_dims(shape, stride, stride) else {
            return false;
        };
        let Some(used) =
            match_transform_dims(&self.dims, &dims, |dim, axis| on_side(dim).is == axis.is)
        else {
            return false;
        };
        let rest: Vec<Dim> = unused(&dims, &used);
        let batch: Vec<Dim> = self.batch.iter().map(on_side).collect();
        normalise_batch(&rest) == normalise_batch(&batch)
    }

    /// Whether an input and an output array of the same rank fit the layout with their axes
    /// paired by index (DC15): equal shapes, each transform dimension matched by one axis
    /// with its extent and, above extent 1, both strides, and the other axes, normalised
    /// together, equal to the layout's batch.
    pub(crate) fn fits_array_pair(
        &self,
        (input_shape, input_stride): (&[usize], &[usize]),
        (output_shape, output_stride): (&[usize], &[usize]),
    ) -> bool {
        if input_shape != output_shape {
            return false;
        }
        let Some(dims) = array_dims(input_shape, input_stride, output_stride) else {
            return false;
        };
        let Some(used) = match_transform_dims(&self.dims, &dims, |dim, axis| {
            dim.is == axis.is && dim.os == axis.os
        }) else {
            return false;
        };
        normalise_batch(&unused(&dims, &used)) == self.batch
    }

    /// The dimensions on `side`, for error messages.
    pub(crate) fn describe_side(&self, side: Side) -> String {
        let list = |dims: &[Dim]| {
            dims.iter()
                .map(|dim| {
                    let stride = match side {
                        Side::Input => dim.is,
                        Side::Output => dim.os,
                    };
                    format!("(n {}, stride {stride})", dim.n)
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "transform dimensions [{}] and batch dimensions [{}]",
            list(&self.dims),
            list(&self.batch)
        )
    }
}

/// The axes of an array as dimensions with the given input and output strides, or `None` if
/// an extent or a stride does not fit in `isize`.
fn array_dims(
    shape: &[usize],
    input_stride: &[usize],
    output_stride: &[usize],
) -> Option<Vec<Dim>> {
    shape
        .iter()
        .zip(input_stride.iter().zip(output_stride))
        .map(|(&n, (&is, &os))| {
            Some(Dim {
                n: isize::try_from(n).ok()?,
                is: isize::try_from(is).ok()?,
                os: isize::try_from(os).ok()?,
            })
        })
        .collect()
}

/// Match each transform dimension with a distinct axis of the same extent for which
/// `strides_match` holds, or, for a dimension of extent 1, with any distinct axis of extent 1,
/// whose stride addresses no other element. Returns the axes used, or `None` if a dimension
/// has no match. Dimensions above extent 1 are matched first.
fn match_transform_dims(
    transform: &[Dim],
    axes: &[Dim],
    strides_match: impl Fn(&Dim, &Dim) -> bool,
) -> Option<Vec<bool>> {
    let mut used = vec![false; axes.len()];
    for extent_one in [false, true] {
        for dim in transform.iter().filter(|dim| (dim.n == 1) == extent_one) {
            let axis = (0..axes.len()).find(|&axis| {
                !used[axis]
                    && axes[axis].n == dim.n
                    && (extent_one || strides_match(dim, &axes[axis]))
            })?;
            used[axis] = true;
        }
    }
    Some(used)
}

/// The axes not used by the transform dimensions.
fn unused(axes: &[Dim], used: &[bool]) -> Vec<Dim> {
    axes.iter()
        .zip(used)
        .filter(|&(_, &used)| !used)
        .map(|(&dim, _)| dim)
        .collect()
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
    use crate::dense::array::StridedSliceArray;
    use crate::fftw::{FftwError, Side};
    use crate::{DynArray, StridedDynArray, c64};

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

    // Layouts from arrays (task F2).

    fn array_invalid(result: Result<Layout, FftwError>) -> (String, String) {
        match result {
            Err(FftwError::InvalidLayout { dimension, reason }) => (dimension, reason),
            other => panic!("expected InvalidLayout, found {other:?}"),
        }
    }

    fn expected(dims: &[Dim], batch: &[Dim], placement: Placement) -> Layout {
        Layout::new(dims, batch, placement).unwrap()
    }

    #[test]
    fn test_array_layouts_of_the_brief() {
        let out = Placement::OutOfPlace;
        let a = DynArray::<c64, 2>::from_shape([8, 3]);
        assert_eq!(
            Layout::along_axis(&a, &a, 0).unwrap(),
            expected(&[dim(8, 1, 1)], &[dim(3, 8, 8)], out)
        );
        assert_eq!(
            Layout::along_axis(&a, &a, 1).unwrap(),
            expected(&[dim(3, 8, 8)], &[dim(8, 1, 1)], out)
        );

        let a = DynArray::<c64, 3>::from_shape([4, 5, 6]);
        let along = |axis| Layout::along_axis(&a, &a, axis).unwrap();
        assert_eq!(along(0), expected(&[dim(4, 1, 1)], &[dim(30, 4, 4)], out));
        assert_eq!(along(0).batch(), &[dim(30, 4, 4)]);
        assert_eq!(
            along(1),
            expected(&[dim(5, 4, 4)], &[dim(6, 20, 20), dim(4, 1, 1)], out)
        );
        assert_eq!(along(1).batch().len(), 2);
        assert_eq!(along(2), expected(&[dim(6, 20, 20)], &[dim(20, 1, 1)], out));
        assert_eq!(along(2).batch(), &[dim(20, 1, 1)]);

        let row_major = StridedDynArray::<c64, 3>::row_major([4, 5, 6]);
        let layout = Layout::along_axis(&row_major, &row_major, 2).unwrap();
        assert_eq!(layout, expected(&[dim(6, 1, 1)], &[dim(20, 6, 6)], out));
        assert_eq!(layout.batch(), &[dim(20, 6, 6)]);

        let layout = Layout::over_axes(&a, &a, &[0, 1]).unwrap();
        assert_eq!(
            layout,
            expected(&[dim(4, 1, 1), dim(5, 4, 4)], &[dim(6, 20, 20)], out)
        );
        let swapped = Layout::over_axes(&a, &a, &[1, 0]).unwrap();
        assert_eq!(
            swapped,
            expected(&[dim(5, 4, 4), dim(4, 1, 1)], &[dim(6, 20, 20)], out)
        );
        // The same transform, but another layout and another plan.
        assert_ne!(layout, swapped);

        let parent = DynArray::<c64, 2>::from_shape([10, 10]);
        let view = parent.r().into_subview([2, 3], [4, 5]);
        assert_eq!(
            Layout::along_axis(&view, &view, 0).unwrap(),
            expected(&[dim(4, 1, 1)], &[dim(5, 10, 10)], out)
        );
    }

    #[test]
    fn test_along_axis_is_over_one_axis() {
        let a = DynArray::<c64, 3>::from_shape([4, 5, 6]);
        let b = StridedDynArray::<c64, 3>::row_major([4, 5, 6]);
        for axis in 0..3 {
            assert_eq!(
                Layout::along_axis(&a, &b, axis).unwrap(),
                Layout::over_axes(&a, &b, &[axis]).unwrap()
            );
            assert_eq!(
                Layout::along_axis_inplace(&b, axis).unwrap(),
                Layout::over_axes_inplace(&b, &[axis]).unwrap()
            );
        }
        // Input and output strides differ for arrays of different memory order.
        assert_eq!(
            Layout::along_axis(&a, &b, 2).unwrap().dims(),
            &[dim(6, 20, 1)]
        );
    }

    #[test]
    fn test_in_place_array_layouts() {
        let a = DynArray::<c64, 3>::from_shape([4, 5, 6]);
        let layout = Layout::along_axis_inplace(&a, 1).unwrap();
        assert_eq!(
            layout,
            expected(
                &[dim(5, 4, 4)],
                &[dim(6, 20, 20), dim(4, 1, 1)],
                Placement::InPlace
            )
        );
        let layout = Layout::over_axes_inplace(&a, &[2, 0]).unwrap();
        assert_eq!(
            layout,
            expected(
                &[dim(6, 20, 20), dim(4, 1, 1)],
                &[dim(5, 4, 4)],
                Placement::InPlace
            )
        );
        let row_major = StridedDynArray::<c64, 2>::row_major([3, 7]);
        assert_eq!(
            Layout::along_axis_inplace(&row_major, 1).unwrap(),
            expected(&[dim(7, 1, 1)], &[dim(3, 7, 7)], Placement::InPlace)
        );
    }

    #[test]
    fn test_extent_one_axis_is_dropped_from_the_batch() {
        let column = DynArray::<c64, 2>::from_shape([12, 1]);
        let vector = DynArray::<c64, 1>::from_shape([12]);
        let layout = Layout::along_axis(&column, &column, 0).unwrap();
        assert_eq!(layout, Layout::along_axis(&vector, &vector, 0).unwrap());
        assert!(layout.batch().is_empty());
        // Along the axis of extent 1, the transform dimension is kept.
        let layout = Layout::along_axis(&column, &column, 1).unwrap();
        assert_eq!(layout.dims(), &[dim(1, 12, 12)]);
    }

    #[test]
    fn test_array_layout_errors() {
        let a = DynArray::<c64, 2>::from_shape([8, 3]);
        let (dimension, _) = array_invalid(Layout::over_axes(&a, &a, &[]));
        assert_eq!(dimension, "axes");
        let (dimension, reason) = array_invalid(Layout::over_axes(&a, &a, &[0, 2]));
        assert_eq!(dimension, "axes");
        assert!(
            reason.contains("axis 2 is not below the rank 2"),
            "{reason}"
        );
        let (dimension, _) = array_invalid(Layout::over_axes(&a, &a, &[1, 1]));
        assert_eq!(dimension, "axis 1");
        let (dimension, _) = array_invalid(Layout::over_axes_inplace(&a, &[0, 0]));
        assert_eq!(dimension, "axis 0");

        let b = DynArray::<c64, 2>::from_shape([8, 4]);
        let (dimension, reason) = array_invalid(Layout::along_axis(&a, &b, 0));
        assert_eq!(dimension, "axis 1");
        assert!(reason.contains("input extent 3"), "{reason}");

        // An array with an extent 0, on a transform axis and on a batch axis.
        let empty = DynArray::<c64, 2>::from_shape([8, 0]);
        let (dimension, reason) = array_invalid(Layout::along_axis(&empty, &empty, 0));
        assert_eq!(dimension, "axis 1");
        assert!(reason.contains("extent is 0"), "{reason}");
        let (dimension, _) = array_invalid(Layout::along_axis_inplace(&empty, 1));
        assert_eq!(dimension, "axis 1");

        // A stride that does not fit in `isize`, on an axis of extent 1.
        let values = [c64::new(0.0, 0.0); 2];
        let huge = StridedSliceArray::from_shape_and_stride(&values, [2, 1], [1, usize::MAX]);
        let small = DynArray::<c64, 2>::from_shape([2, 1]);
        let (dimension, reason) = array_invalid(Layout::along_axis(&huge, &small, 0));
        assert_eq!(dimension, "axis 1");
        assert!(reason.contains("input stride"), "{reason}");

        // The checks of `Layout::new` name the axis.
        let (dimension, _) = array_invalid(Layout::along_axis_inplace(
            &StridedDynArray::<c64, 2>::from_shape_and_stride([2, 2], [2, 3]),
            0,
        ));
        assert!(dimension.starts_with("axis "), "{dimension}");
    }

    #[test]
    fn test_strides_failing_the_distinct_output_rule() {
        // Offsets 0, 2, 3 and 5: distinct, but 3 is below 2 x 2 (DC16).
        let odd = StridedDynArray::<c64, 2>::from_shape_and_stride([2, 2], [2, 3]);
        let plain = DynArray::<c64, 2>::from_shape([2, 2]);
        let (dimension, _) = array_invalid(Layout::along_axis(&plain, &odd, 0));
        assert_eq!(dimension, "axis 1");
        array_invalid(Layout::along_axis_inplace(&odd, 0));
        array_invalid(Layout::along_axis_inplace(&odd, 1));
        // As input it is accepted.
        let layout = Layout::along_axis(&odd, &plain, 0).unwrap();
        assert_eq!(layout.dims(), &[dim(2, 2, 1)]);
    }

    #[test]
    fn test_array_sides() {
        // A layout fits the arrays it was made from, side by side and paired.
        let input = DynArray::<c64, 3>::from_shape([4, 2, 3]);
        let output = StridedDynArray::<c64, 3>::from_shape_and_stride([4, 2, 3], [1, 12, 4]);
        let layout = Layout::along_axis(&input, &output, 0).unwrap();
        let geometry = |shape: [usize; 3], stride: [usize; 3]| (shape.to_vec(), stride.to_vec());
        let (in_shape, in_stride) = geometry(input.shape(), input.stride());
        let (out_shape, out_stride) = geometry(output.shape(), output.stride());
        assert!(layout.fits_array_side(Side::Input, &in_shape, &in_stride));
        assert!(layout.fits_array_side(Side::Output, &out_shape, &out_stride));
        assert!(layout.fits_array_pair((&in_shape, &in_stride), (&out_shape, &out_stride)));
        // On one side, the batch axes are normalised alone: the output's batch (2, 12) and
        // (3, 4) is the input's (2, 4) and (3, 8) after merging, a batch of 6 with stride 4.
        assert!(layout.fits_array_side(Side::Input, &out_shape, &out_stride));
        // A row-major array does not fit.
        assert!(!layout.fits_array_side(Side::Input, &[4, 2, 3], &[6, 3, 1]));

        // Column-major arrays of shape [4, 3, 2] fit each side, but their axes do not pair
        // as the layout's (DC15).
        let (other_shape, other_stride) = (vec![4, 3, 2], vec![1, 4, 12]);
        assert!(layout.fits_array_side(Side::Input, &other_shape, &other_stride));
        assert!(layout.fits_array_side(Side::Output, &other_shape, &other_stride));
        assert!(
            !layout.fits_array_pair((&other_shape, &other_stride), (&other_shape, &other_stride))
        );

        // A wrong extent or stride on the transform axis.
        assert!(!layout.fits_array_side(Side::Input, &[5, 2, 3], &[1, 5, 10]));
        assert!(!layout.fits_array_side(Side::Input, &[4, 2, 3], &[2, 8, 16]));
    }
}
