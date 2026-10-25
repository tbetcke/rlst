//! Transform layouts: dimensions, batch dimensions, placement and the storage of each side.

use std::fmt;

use num::Complex;

use crate::{Array, BaseItem, Shape, Stride, StridedRawAccess};

use super::element::{self, FftElement};
use super::error::{FftwError, Side};
use super::real::FftwReal;

/// One dimension of a layout: its extent and its input and output strides.
///
/// Strides are counted in elements of each side, and are non-negative: complex elements on a
/// side of interleaved complex storage, reals on a real side or a side of split storage (see
/// [`Storage`]). For real-to-complex and complex-to-real layouts, the extent of the last
/// transform dimension is the logical real extent `n`; the complex side has `n / 2 + 1`
/// elements along it. The fields are `isize`, as FFTW's `fftw_iodim64` uses `ptrdiff_t`.
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
    /// The input stride, in elements of the input side.
    pub is: isize,
    /// The output stride, in elements of the output side.
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

/// The storage of one side of a [`Layout`]: interleaved complex, real, or split complex.
///
/// The storages of the input and the output decide the family of a layout, and the units of
/// its strides and spans on each side:
///
/// | Input | Output | Family | Halved side |
/// | --- | --- | --- | --- |
/// | `Complex` | `Complex` | complex-to-complex (c2c) | none |
/// | `Split` | `Split` | split complex-to-complex | none |
/// | `Real` | `Complex` | real-to-complex (r2c) | output |
/// | `Complex` | `Real` | complex-to-real (c2r) | input |
/// | `Real` | `Real` | real-to-real (r2r) | none |
/// | `Real` | `Split` | split real-to-complex | output |
/// | `Split` | `Real` | split complex-to-real | input |
///
/// A split side does not pair with an interleaved complex one. On the halved side of an r2c
/// or c2r layout the last transform dimension has extent `n / 2 + 1`, with `n` the logical
/// real extent of the layout's dimension.
///
/// ```
/// use rlst::fftw::{Dim, Layout, Placement, Storage};
///
/// // A real-to-complex transform of length 8: 8 reals in, 5 complex values out.
/// let layout = Layout::with_storage(
///     &[Dim { n: 8, is: 1, os: 1 }],
///     &[],
///     (Storage::Real, Storage::Complex),
///     Placement::OutOfPlace,
/// )?;
/// assert_eq!((layout.input_span(), layout.output_span()), (8, 5));
/// assert_eq!(Storage::Split { gap: 15 }.to_string(), "split (gap 15)");
/// # Ok::<(), rlst::fftw::FftwError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// Interleaved complex elements, [`c32`](crate::c32) or [`c64`](crate::c64); strides
    /// count complex elements.
    Complex,
    /// Reals, `f32` or `f64`; strides count reals.
    Real,
    /// Split complex storage in a buffer of reals: the real part of an element at an offset
    /// and its imaginary part `gap` reals further on; strides count reals.
    Split {
        /// The stride from an element's real part to its imaginary part, in reals.
        gap: isize,
    },
}

impl fmt::Display for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Complex => f.write_str("complex"),
            Storage::Real => f.write_str("real"),
            Storage::Split { gap } => write!(f, "split (gap {gap})"),
        }
    }
}

/// The family of a layout, from the storages of its sides (DC30).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutFamily {
    C2c,
    SplitC2c,
    R2c,
    C2r,
    R2r,
    SplitR2c,
    SplitC2r,
}

impl LayoutFamily {
    /// The family of a layout with these storages, or `None` for a split side paired with an
    /// interleaved complex one.
    pub(crate) fn of(input: Storage, output: Storage) -> Option<Self> {
        Some(match (input, output) {
            (Storage::Complex, Storage::Complex) => LayoutFamily::C2c,
            (Storage::Split { .. }, Storage::Split { .. }) => LayoutFamily::SplitC2c,
            (Storage::Real, Storage::Complex) => LayoutFamily::R2c,
            (Storage::Complex, Storage::Real) => LayoutFamily::C2r,
            (Storage::Real, Storage::Real) => LayoutFamily::R2r,
            (Storage::Real, Storage::Split { .. }) => LayoutFamily::SplitR2c,
            (Storage::Split { .. }, Storage::Real) => LayoutFamily::SplitC2r,
            _ => return None,
        })
    }

    /// The name of the family, for messages.
    pub(crate) fn name(self) -> &'static str {
        match self {
            LayoutFamily::C2c => "complex-to-complex",
            LayoutFamily::SplitC2c => "split complex-to-complex",
            LayoutFamily::R2c => "real-to-complex",
            LayoutFamily::C2r => "complex-to-real",
            LayoutFamily::R2r => "real-to-real",
            LayoutFamily::SplitR2c => "split real-to-complex",
            LayoutFamily::SplitC2r => "split complex-to-real",
        }
    }

    /// The names of the storages of the family's input and output, for messages.
    pub(crate) fn storage_names(self) -> (&'static str, &'static str) {
        match self {
            LayoutFamily::C2c => ("complex", "complex"),
            LayoutFamily::SplitC2c => ("split", "split"),
            LayoutFamily::R2c => ("real", "complex"),
            LayoutFamily::C2r => ("complex", "real"),
            LayoutFamily::R2r => ("real", "real"),
            LayoutFamily::SplitR2c => ("real", "split"),
            LayoutFamily::SplitC2r => ("split", "real"),
        }
    }

    /// Whether the family's input and output are split.
    pub(crate) fn split_sides(self) -> (bool, bool) {
        match self {
            LayoutFamily::SplitC2c => (true, true),
            LayoutFamily::SplitR2c => (false, true),
            LayoutFamily::SplitC2r => (true, false),
            _ => (false, false),
        }
    }

    /// The side whose last transform dimension is halved: the complex or split side of r2c
    /// and c2r and their split forms.
    pub(crate) fn halved(self) -> Option<Side> {
        match self {
            LayoutFamily::R2c | LayoutFamily::SplitR2c => Some(Side::Output),
            LayoutFamily::C2r | LayoutFamily::SplitC2r => Some(Side::Input),
            _ => None,
        }
    }
}

/// The extent `n / 2 + 1` of the halved dimension on the complex side, for the real extent
/// `n`.
pub(crate) fn halved_extent(n: isize) -> isize {
    n / 2 + 1
}

/// The storage of an array given to a layout function: of its element type, or split with
/// the stride of its part axis.
#[derive(Clone, Copy, Debug)]
enum ArrayStorage {
    Complex,
    Real,
    Split { part_stride: usize },
}

impl ArrayStorage {
    /// The storage of arrays of `E`.
    fn of<E: FftElement>() -> Self {
        if <E as element::sealed::Sealed>::COMPLEX {
            ArrayStorage::Complex
        } else {
            ArrayStorage::Real
        }
    }

    /// Whether the side holds complex values, interleaved or split.
    fn is_complex(self) -> bool {
        !matches!(self, ArrayStorage::Real)
    }
}

/// The layout of a batch of transforms, in normalised form.
///
/// A layout lists the transform dimensions, in the order given, and the batch dimensions over
/// which the transform is repeated, with the placement and the [`Storage`] of each side, from
/// which its family follows: complex-to-complex, real-to-complex, complex-to-real or
/// real-to-real, on interleaved or split complex storage (see [`Layout::with_storage`]).
/// Offsets are counted in elements of each side from the start of its buffer. A layout is
/// validated when it is made, and normalised: batch dimensions of extent 1 are dropped, the
/// batch dimensions are sorted by input stride and then output stride, largest first, and
/// adjacent batch dimensions that are contiguous on both sides are merged. Two layouts that
/// describe the same transforms of the same elements are therefore equal, whatever the order
/// of their batch dimensions.
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
#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    dims: Vec<Dim>,
    batch: Vec<Dim>,
    placement: Placement,
    /// The storage of the input and of the output.
    storage: (Storage, Storage),
    input_span: usize,
    output_span: usize,
    /// The batch on the input and on the output side alone, normalised, which
    /// [`fits_array_side`](Layout::fits_array_side) compares with at every execution. It
    /// follows from `batch`, so it does not change equality.
    side_batches: (Vec<Dim>, Vec<Dim>),
}

// By hand, to leave the cached `side_batches` out of the output.
impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Layout")
            .field("dims", &self.dims)
            .field("batch", &self.batch)
            .field("placement", &self.placement)
            .field("storage", &self.storage)
            .field("input_span", &self.input_span)
            .field("output_span", &self.output_span)
            .finish()
    }
}

/// One dimension of one side of a layout, for the distinct-output rule and the spans: its
/// name in errors, its extent on that side and its stride on that side.
struct SideDim<'a> {
    name: &'a str,
    n: isize,
    stride: isize,
}

impl Layout {
    /// Make a complex-to-complex layout of interleaved complex storage from transform
    /// dimensions, batch dimensions and a placement.
    ///
    /// The same as [`Layout::with_storage`] with the storages `(Storage::Complex,
    /// Storage::Complex)`. Its checks, in this order, each returning
    /// [`FftwError::InvalidLayout`] with the dimension and the reason:
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
        Self::with_storage(dims, batch, (Storage::Complex, Storage::Complex), placement)
    }

    /// Make a layout of split complex storage from transform dimensions, batch dimensions, the
    /// gaps of input and output, and a placement.
    ///
    /// Split storage keeps the real and imaginary parts of the complex values in one buffer of
    /// reals: the real part of an element at an offset, its imaginary part a fixed *gap*
    /// further on. The strides of `dims` and `batch`, the gaps `(input_gap, output_gap)` and
    /// the spans count reals, and the offsets of a slice count reals from its start, where the
    /// real part of the first element is. Plans of this layout are made by
    /// [`Planner::split_c2c`](super::Planner::split_c2c) and its companions. The gaps are
    /// recorded beside the dimensions, never as a batch dimension, so normalisation never
    /// merges them into the batch.
    ///
    /// The same as [`Layout::with_storage`] with the storages `(Storage::Split { gap:
    /// input_gap }, Storage::Split { gap: output_gap })`. Its checks, in this order, each
    /// returning [`FftwError::InvalidLayout`] with the dimension (`part axis` for the gaps)
    /// and the reason:
    ///
    /// 1. checks 1 to 4 of [`Layout::new`] on the dimensions: at least one transform
    ///    dimension, extents at least 1, non-negative strides, and in place equal strides;
    /// 2. both gaps are positive, and in place they are equal;
    /// 3. the distinct-output rule of [`Layout::new`], with the gaps as a dimension of extent
    ///    2, the *part axis*, whose output stride is the output gap: no real or imaginary
    ///    part of one output element is a part of another;
    /// 4. the spans, which include the gap of their side, fit in `isize`.
    ///
    /// ```
    /// use rlst::fftw::{Dim, FftwError, Layout, Placement};
    ///
    /// // Nine transforms of length 140, the real plane of 1260 reals and then the imaginary
    /// // plane, as nd-fmm keeps the rings of a box.
    /// let layout = Layout::new_split(
    ///     &[Dim { n: 140, is: 1, os: 1 }],
    ///     &[Dim { n: 9, is: 140, os: 140 }],
    ///     (1260, 1260),
    ///     Placement::OutOfPlace,
    /// )?;
    /// assert_eq!(layout.gaps(), Some((1260, 1260)));
    /// assert_eq!(layout.input_span(), 2520);
    ///
    /// // Interleaved storage read as split: the imaginary part follows the real part.
    /// let layout = Layout::new_split(
    ///     &[Dim { n: 8, is: 2, os: 2 }],
    ///     &[],
    ///     (1, 1),
    ///     Placement::InPlace,
    /// )?;
    /// assert_eq!(layout.input_span(), 16);
    ///
    /// // A gap of 4 makes the parts of elements 0 to 3 overlap the real parts of 4 to 7.
    /// let error = Layout::new_split(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[],
    ///     (8, 4),
    ///     Placement::OutOfPlace,
    /// );
    /// assert!(matches!(error, Err(FftwError::InvalidLayout { .. })));
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn new_split(
        dims: &[Dim],
        batch: &[Dim],
        (input_gap, output_gap): (isize, isize),
        placement: Placement,
    ) -> Result<Self, FftwError> {
        Self::with_storage(
            dims,
            batch,
            (
                Storage::Split { gap: input_gap },
                Storage::Split { gap: output_gap },
            ),
            placement,
        )
    }

    /// Make a layout of any family from transform dimensions, batch dimensions, the storage of
    /// the input and of the output, and a placement.
    ///
    /// The storages decide the family, as the table of [`Storage`] lists: complex-to-complex,
    /// real-to-complex, complex-to-real and real-to-real, on interleaved or split complex
    /// storage. Strides, spans and the offsets of a slice count elements of each side:
    /// complex elements on a complex side, reals on a real or split side. For real-to-complex
    /// and complex-to-real layouts, and their split forms, the dimensions record the logical
    /// real extent `n`, as FFTW's guru interface takes it, and the last transform dimension
    /// is the *halved* one: on the complex side it has extent `n / 2 + 1`. [`Layout::new`]
    /// and [`Layout::new_split`] are this function for complex-to-complex layouts.
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] with the
    /// dimension (`layout`, a dimension, or `part axis` for the gaps) and the reason:
    ///
    /// 1. the storages are a pair of the table of [`Storage`];
    /// 2. there is at least one transform dimension, every extent is at least 1 and every
    ///    stride is non-negative;
    /// 3. in place, both sides are one buffer: for complex-to-complex, split
    ///    complex-to-complex and real-to-real layouts the input and output strides are equal
    ///    in every dimension; for real-to-complex and complex-to-real layouts the buffer is
    ///    the complex side, padded as FFTW requires, so the halved dimension has stride 1 on
    ///    both sides, the reals of each complex element holding two real elements, and on
    ///    every other dimension the real stride is twice the complex stride (dimensions of
    ///    extent 1 address one element and are not checked); split real-to-complex and
    ///    complex-to-real layouts cannot be in place;
    /// 4. the gap of a split side is positive, and in place the gaps are equal;
    /// 5. the output elements are distinct, by the rule of [`Layout::new`] on the output side,
    ///    with the extent `n / 2 + 1` of the halved dimension on a complex output and the part
    ///    axis, of extent 2 and the gap as stride, on a split output; in place, the input
    ///    side is written too, and the rule applies to it as well;
    /// 6. the spans, which use the extent `n / 2 + 1` on the complex side and include the gap
    ///    of a split side, fit in `isize`.
    ///
    /// ```
    /// use rlst::fftw::{Dim, FftwError, Layout, Placement, Storage};
    ///
    /// // Three real-to-complex transforms of length 8: columns of a real [8, 3] array into
    /// // the columns of a complex [5, 3] array.
    /// let r2c = Layout::with_storage(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[Dim { n: 3, is: 8, os: 5 }],
    ///     (Storage::Real, Storage::Complex),
    ///     Placement::OutOfPlace,
    /// )?;
    /// assert_eq!((r2c.input_span(), r2c.output_span()), (24, 15));
    ///
    /// // The same transforms in place: a complex [5, 3] array whose columns hold 10 reals,
    /// // the 8 reals of each transform padded by 2.
    /// let in_place = Layout::with_storage(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[Dim { n: 3, is: 10, os: 5 }],
    ///     (Storage::Real, Storage::Complex),
    ///     Placement::InPlace,
    /// )?;
    /// assert_eq!(in_place.output_span(), 15);
    ///
    /// // Without the padding the complex outputs of the columns overlap.
    /// let error = Layout::with_storage(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[Dim { n: 3, is: 8, os: 4 }],
    ///     (Storage::Real, Storage::Complex),
    ///     Placement::InPlace,
    /// );
    /// assert!(matches!(error, Err(FftwError::InvalidLayout { .. })));
    ///
    /// // A split side does not pair with an interleaved complex one.
    /// let error = Layout::with_storage(
    ///     &[Dim { n: 8, is: 1, os: 1 }],
    ///     &[],
    ///     (Storage::Split { gap: 8 }, Storage::Complex),
    ///     Placement::OutOfPlace,
    /// );
    /// assert!(matches!(error, Err(FftwError::InvalidLayout { .. })));
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn with_storage(
        dims: &[Dim],
        batch: &[Dim],
        storage: (Storage, Storage),
        placement: Placement,
    ) -> Result<Self, FftwError> {
        Self::with_names(
            dims,
            batch,
            storage,
            placement,
            |index| format!("transform dimension {index}"),
            |index| format!("batch dimension {index}"),
        )
    }

    /// [`Layout::with_storage`] with the names its errors give each transform and batch
    /// dimension.
    fn with_names(
        dims: &[Dim],
        batch: &[Dim],
        (input, output): (Storage, Storage),
        placement: Placement,
        dim_name: impl Fn(usize) -> String,
        batch_name: impl Fn(usize) -> String,
    ) -> Result<Self, FftwError> {
        let invalid = |dimension: &str, reason: String| FftwError::InvalidLayout {
            dimension: dimension.to_string(),
            reason,
        };

        let Some(family) = LayoutFamily::of(input, output) else {
            return Err(invalid(
                "layout",
                format!(
                    "an input of {input} storage and an output of {output} storage describe no \
                     transform; a split side pairs with a split or a real side"
                ),
            ));
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
            match family {
                LayoutFamily::SplitR2c | LayoutFamily::SplitC2r => {
                    return Err(invalid(
                        "layout",
                        format!(
                            "a {} layout cannot be in place; RLST has no in-place split \
                             real-to-complex or complex-to-real plans",
                            family.name()
                        ),
                    ));
                }
                LayoutFamily::R2c | LayoutFamily::C2r => {
                    check_padded_in_place(&named, dims.len() - 1, family == LayoutFamily::R2c)?;
                }
                _ => {
                    for (name, dim) in &named {
                        if dim.is != dim.os {
                            return Err(invalid(
                                name,
                                format!(
                                    "in place, the input stride {} and the output stride {} \
                                     must be equal",
                                    dim.is, dim.os
                                ),
                            ));
                        }
                    }
                }
            }
        }
        let gap = |storage: Storage| match storage {
            Storage::Split { gap } => Some(gap),
            _ => None,
        };
        for (side, gap) in [(Side::Input, gap(input)), (Side::Output, gap(output))] {
            if let Some(gap) = gap
                && gap < 1
            {
                return Err(invalid(
                    PART_AXIS,
                    format!("the gaps must be positive, found the {side} gap {gap}"),
                ));
            }
        }
        if placement == Placement::InPlace
            && let (Some(input_gap), Some(output_gap)) = (gap(input), gap(output))
            && input_gap != output_gap
        {
            return Err(invalid(
                PART_AXIS,
                format!(
                    "in place, the input gap {input_gap} and the output gap {output_gap} must \
                     be equal"
                ),
            ));
        }

        // The dimensions of each side: the transform dimensions with that side's extent, the
        // batch dimensions, and for split storage the part axis, which takes part in the
        // distinct-output rule and the spans but never in the batch.
        let last = dims.len() - 1;
        let side_dims = |side: Side| -> Vec<SideDim<'_>> {
            let storage = match side {
                Side::Input => input,
                Side::Output => output,
            };
            let mut list: Vec<SideDim<'_>> = named
                .iter()
                .enumerate()
                .map(|(index, (name, dim))| SideDim {
                    name,
                    n: if index == last && family.halved() == Some(side) {
                        halved_extent(dim.n)
                    } else {
                        dim.n
                    },
                    stride: match side {
                        Side::Input => dim.is,
                        Side::Output => dim.os,
                    },
                })
                .collect();
            if let Some(gap) = gap(storage) {
                list.push(SideDim {
                    name: PART_AXIS,
                    n: 2,
                    stride: gap,
                });
            }
            list
        };

        check_distinct(&side_dims(Side::Output), Side::Output)?;
        if placement == Placement::InPlace {
            check_distinct(&side_dims(Side::Input), Side::Input)?;
        }

        let span = |side: Side| {
            side_dims(side)
                .iter()
                .try_fold(1_isize, |span, dim| {
                    span.checked_add((dim.n - 1).checked_mul(dim.stride)?)
                })
                .and_then(|span| usize::try_from(span).ok())
        };
        let input_span = span(Side::Input)
            .ok_or_else(|| invalid("layout", "the input span overflows isize".to_string()))?;
        let output_span = span(Side::Output)
            .ok_or_else(|| invalid("layout", "the output span overflows isize".to_string()))?;

        // In place, the real side of a real-to-complex or complex-to-real layout lies inside
        // the complex side: along the halved dimension n reals fit in the 2 (n / 2 + 1) reals
        // of the complex elements, and every other stride is the same distance on both sides.
        // Execution checks the complex span only, and relies on this.
        debug_assert!(
            placement == Placement::OutOfPlace
                || match family {
                    LayoutFamily::R2c => input_span <= 2 * output_span,
                    LayoutFamily::C2r => output_span <= 2 * input_span,
                    _ => input_span == output_span,
                }
        );

        let batch = normalise_batch(batch);
        let side_batch = |side: Side| {
            let on_side: Vec<Dim> = batch
                .iter()
                .map(|dim| {
                    let stride = side_stride(dim, side);
                    Dim {
                        n: dim.n,
                        is: stride,
                        os: stride,
                    }
                })
                .collect();
            normalise_batch(&on_side)
        };
        let side_batches = (side_batch(Side::Input), side_batch(Side::Output));
        Ok(Self {
            dims: dims.to_vec(),
            batch,
            placement,
            storage: (input, output),
            input_span,
            output_span,
            side_batches,
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

    /// The storage `(input, output)` of the two sides, from which the family of the layout
    /// follows (see [`Storage`]).
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement, Storage};
    ///
    /// let dims = [Dim { n: 8, is: 1, os: 1 }];
    /// let c2c = Layout::new(&dims, &[], Placement::OutOfPlace)?;
    /// assert_eq!(c2c.storage(), (Storage::Complex, Storage::Complex));
    /// let split = Layout::new_split(&dims, &[], (8, 8), Placement::OutOfPlace)?;
    /// assert_eq!(split.storage(), (Storage::Split { gap: 8 }, Storage::Split { gap: 8 }));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn storage(&self) -> (Storage, Storage) {
        self.storage
    }

    /// The gaps `(input_gap, output_gap)` of a layout of split complex storage on both sides,
    /// in reals: the stride from an element's real part to its imaginary part on each side.
    /// `None` unless both sides are split; [`storage`](Layout::storage) gives the gap of a
    /// split side beside a real one.
    ///
    /// ```
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// let dims = [Dim { n: 8, is: 1, os: 1 }];
    /// let split = Layout::new_split(&dims, &[], (8, 16), Placement::OutOfPlace)?;
    /// assert_eq!(split.gaps(), Some((8, 16)));
    /// let interleaved = Layout::new(&dims, &[], Placement::OutOfPlace)?;
    /// assert_eq!(interleaved.gaps(), None);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn gaps(&self) -> Option<(isize, isize)> {
        match self.storage {
            (Storage::Split { gap: input }, Storage::Split { gap: output }) => {
                Some((input, output))
            }
            _ => None,
        }
    }

    /// The number of input elements the layout spans: 1 plus the sum of (n − 1) · is over
    /// all dimensions, with the extent n / 2 + 1 of the halved dimension on a complex input,
    /// plus the input gap for split storage.
    ///
    /// An input buffer must have at least this many elements of the input side.
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
    /// all dimensions, with the extent n / 2 + 1 of the halved dimension on a complex
    /// output, plus the output gap for split storage.
    ///
    /// An output buffer must have at least this many elements of the output side.
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

    /// The product of the extents n of the transform dimensions.
    ///
    /// This is the logical size of one complex-to-complex, real-to-complex or complex-to-real
    /// transform, whose n are the logical real extents: a backward transform after a forward
    /// one multiplies the data by it. The logical size of a real-to-real transform depends on
    /// its kinds and is [`Plan::logical_size`](super::Plan::logical_size).
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
        // The outputs are distinct, so the product is at most twice the output span.
        self.dims.iter().map(|dim| dim.n as usize).product()
    }

    /// The family of the layout, from its storages.
    pub(crate) fn family(&self) -> LayoutFamily {
        LayoutFamily::of(self.storage.0, self.storage.1)
            .expect("a layout's storages were validated when it was made")
    }

    /// The span of `side`, in elements of that side.
    pub(crate) fn span(&self, side: Side) -> usize {
        match side {
            Side::Input => self.input_span,
            Side::Output => self.output_span,
        }
    }

    /// The extent of transform dimension `index` on `side`: n, or n / 2 + 1 for the halved
    /// dimension on its side.
    pub(crate) fn extent(&self, index: usize, side: Side) -> isize {
        let n = self.dims[index].n;
        if index + 1 == self.dims.len() && self.family().halved() == Some(side) {
            halved_extent(n)
        } else {
            n
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
        let storage = match self.storage {
            (Storage::Complex, Storage::Complex) => String::new(),
            (input, output) => format!(" and {input} input and {output} output storage"),
        };
        format!(
            "the {} {} layout with transform dimensions [{}] and batch dimensions [{}]{storage}",
            self.placement,
            self.family().name(),
            list(&self.dims),
            list(&self.batch)
        )
    }

    /// The gap of `side`, if that side is split.
    pub(crate) fn gap(&self, side: Side) -> Option<isize> {
        let storage = match side {
            Side::Input => self.storage.0,
            Side::Output => self.storage.1,
        };
        match storage {
            Storage::Split { gap } => Some(gap),
            _ => None,
        }
    }

    /// One-dimensional transforms along `axis` of two arrays of rank `N`, batched over the
    /// other axes.
    ///
    /// The same as [`over_axes`](Layout::over_axes) with the one axis `axis`, whose checks,
    /// errors and notes apply: complex arrays give a complex-to-complex layout, a real input
    /// and a complex output a real-to-complex layout with `axis` halved, a complex input and
    /// a real output a complex-to-real one, two real arrays a real-to-real one. The arrays
    /// can be any RLST arrays or views with strided storage: column-major, row-major or
    /// strided, subviews, slices and nested views.
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
    ///
    /// Both arrays have one precision, so a real `f32` input with a `c64` output does not
    /// compile:
    ///
    /// ```compile_fail,E0271
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(f32, [8, 3] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [5, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::along_axis(&input, &output, 0)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// while an `f64` input does:
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [5, 3] | rlst::FFTW_ALIGNED);
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
        AIn::Item: FftElement,
        AOut::Item: FftElement<Real = <AIn::Item as FftElement>::Real>,
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
    /// transform. The layout is then made by [`Layout::with_storage`], whose validation and
    /// normalisation apply.
    ///
    /// The element types of the arrays, which implement [`FftElement`] with one precision,
    /// give the [`Storage`] of each side and so the family: two complex arrays give a
    /// complex-to-complex layout, two real arrays a real-to-real one, a real input and a
    /// complex output a real-to-complex one, and a complex input and a real output a
    /// complex-to-real one. Real-to-complex and complex-to-real transforms halve the axis
    /// given last in `axes`, whatever the memory order (DC2): along it the complex array has
    /// `n / 2 + 1` elements for the real array's `n`, and the layout's dimension records `n`.
    /// Which axis to halve is the caller's choice: RLST 0.11's benchmarks timed halving the
    /// contiguous axis against halving a strided one, and neither order was faster on every
    /// machine measured, so RLST recommends neither. Strides count elements of each array,
    /// reals or complex values.
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] that names
    /// `axes` or the axis concerned (`axis 1`):
    ///
    /// 1. `axes` is not empty;
    /// 2. every entry of `axes` is below `N`;
    /// 3. no entry is repeated;
    /// 4. the input and output extents are equal on every axis, except along the halved axis
    ///    of a real-to-complex or complex-to-real layout, where the complex extent is
    ///    `n / 2 + 1` for the real extent `n`; an error names the axis, the extent expected
    ///    and the extent found;
    /// 5. every extent is at least 1: an array with an extent 0 has nothing to transform;
    /// 6. every extent and stride fits in `isize`;
    ///
    /// and those of [`Layout::with_storage`], with the dimensions named by their axes. In
    /// particular, an output whose strides fail the sufficient distinct-output rule of
    /// [`Layout::new`], such as shape `[2, 2]` with strides `[2, 3]`, is rejected, although its
    /// elements are distinct; it is accepted as an input.
    ///
    /// Building a layout reads only shapes and strides. A layout carries no offsets: the
    /// offsets of an array count from its first element, so a plan runs on any arrays and
    /// views of this memory layout, wherever they start.
    ///
    /// The scratch of [`Planner::c2c`](super::Planner::c2c) and the other planner methods
    /// without `_on` covers the span from the first element to the last one the layout
    /// addresses. For a view of a large array, such as a column of a row-major matrix, that
    /// can be nearly the whole array: raise [`max_scratch`](super::Planner::max_scratch) or
    /// plan on the arrays themselves with [`c2c_on`](super::Planner::c2c_on) and the other
    /// `*_on` methods.
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
    ///
    /// A real-to-complex layout over both axes of a real `[4, 6]` array: axis 1, given last,
    /// is halved, so the complex array is `[4, 4]`.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::{Dim, FftwError, Layout, Storage};
    ///
    /// let real = rlst_dynamic_array!(f64, [4, 6] | rlst::FFTW_ALIGNED);
    /// let complex = rlst_dynamic_array!(c64, [4, 4] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::over_axes(&real, &complex, &[0, 1])?;
    /// assert_eq!(layout.storage(), (Storage::Real, Storage::Complex));
    /// assert_eq!(layout.dims(), &[Dim { n: 4, is: 1, os: 1 }, Dim { n: 6, is: 4, os: 4 }]);
    /// assert_eq!(layout.logical_size(), 24);
    ///
    /// // With axis 0 last, axis 0 is halved, and the complex array would be [3, 6].
    /// let error = Layout::over_axes(&real, &complex, &[1, 0]).unwrap_err();
    /// assert!(matches!(error, FftwError::InvalidLayout { dimension, .. } if dimension == "axis 0"));
    /// # Ok::<(), FftwError>(())
    /// ```
    pub fn over_axes<AIn, AOut, const N: usize>(
        input: &Array<AIn, N>,
        output: &Array<AOut, N>,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        AIn: StridedRawAccess + Shape<N> + Stride<N>,
        AOut: StridedRawAccess + Shape<N> + Stride<N>,
        AIn::Item: FftElement,
        AOut::Item: FftElement<Real = <AIn::Item as FftElement>::Real>,
    {
        Self::from_arrays(
            (
                &input.shape(),
                &input.stride(),
                ArrayStorage::of::<AIn::Item>(),
            ),
            (
                &output.shape(),
                &output.stride(),
                ArrayStorage::of::<AOut::Item>(),
            ),
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
        A::Item: FftElement,
    {
        Self::over_axes_inplace(data, &[axis])
    }

    /// In-place transforms over `axes`, in the order given, of an array of rank `N`, batched
    /// over the other axes.
    ///
    /// As [`over_axes`](Layout::over_axes) with `data` as both input and output, whose checks,
    /// errors and notes apply; the input and output strides of every dimension are the
    /// array's strides, and the placement is [`Placement::InPlace`]. A complex array gives a
    /// complex-to-complex layout and a real array a real-to-real one. In-place
    /// real-to-complex and complex-to-real layouts, whose two sides differ in shape, are made
    /// by [`r2c_inplace`](Layout::r2c_inplace) and [`c2r_inplace`](Layout::c2r_inplace).
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
        A::Item: FftElement,
    {
        let (shape, stride) = (data.shape(), data.stride());
        let storage = ArrayStorage::of::<A::Item>();
        Self::from_arrays(
            (&shape, &stride, storage),
            (&shape, &stride, storage),
            axes,
            Placement::InPlace,
        )
    }

    /// In-place real-to-complex transforms over `axes`, in the order given, of the complex
    /// array `complex` of rank `N`, batched over the other axes, with the logical real extent
    /// `n_last` along the halved axis, the last entry of `axes` (DC2, DC32).
    ///
    /// The input is the same memory as the output, read as reals, as FFTW's in-place layout
    /// requires: along the halved axis the `2 (n_last / 2 + 1)` reals of the complex elements
    /// hold the `n_last` real elements, padded at the end, and on every other axis the real
    /// stride is twice the complex stride. [`as_reals`](super::as_reals) and
    /// [`as_reals_mut`](super::as_reals_mut) view the complex storage as reals, to write the
    /// input. The real extent cannot be recovered from the complex shape, so `n_last` is
    /// given; along the other axes both sides have the complex array's extents. The layout's
    /// dimensions record the real extent `n_last` on the halved axis.
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] that names
    /// `axes` or the axis concerned: checks 1 to 3 of [`over_axes`](Layout::over_axes) on
    /// `axes`; `n_last / 2 + 1` equals the array's extent along the halved axis; the halved
    /// axis is contiguous, with stride 1, unless its extent is 1; the strides of the real side
    /// fit in `usize`; and the checks of [`Layout::with_storage`] for an in-place
    /// real-to-complex layout.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array, SliceArrayMut};
    /// use rlst::fftw::{Dim, Layout, Planner, Storage, as_reals, as_reals_mut};
    ///
    /// // Three transforms of length 8, one per column, in a complex [5, 3] array.
    /// let mut data = rlst_dynamic_array!(c64, [5, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::r2c_inplace(&data, &[0], 8)?;
    /// assert_eq!(layout.storage(), (Storage::Real, Storage::Complex));
    /// assert_eq!(layout.dims(), &[Dim { n: 8, is: 1, os: 1 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 3, is: 10, os: 5 }]);
    ///
    /// // The input of column j: the reals 10 j to 10 j + 7, here all ones.
    /// let mut reals = SliceArrayMut::from_shape(as_reals_mut(data.data_mut().unwrap()), [10, 3]);
    /// for j in 0..3 {
    ///     for i in 0..8 {
    ///         reals[[i, j]] = 1.0;
    ///     }
    /// }
    /// let plan = Planner::<f64>::new().r2c_inplace(&layout)?;
    /// plan.execute(&mut data)?;
    /// // The transform of a constant is its sum at frequency 0, and 0 elsewhere.
    /// assert_eq!(data[[0, 2]], c64::new(8.0, 0.0));
    /// assert!((1..5).all(|k| data[[k, 2]].norm() < 1e-12));
    /// # let _ = as_reals(data.data().unwrap());
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn r2c_inplace<T, A, const N: usize>(
        complex: &Array<A, N>,
        axes: &[usize],
        n_last: usize,
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        A: BaseItem<Item = Complex<T>> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::padded_in_place(&complex.shape(), &complex.stride(), axes, n_last, true)
    }

    /// In-place complex-to-real transforms over `axes`, in the order given, of the complex
    /// array `complex` of rank `N`, batched over the other axes, with the logical real extent
    /// `n_last` along the halved axis, the last entry of `axes` (DC2, DC32).
    ///
    /// The inverse of [`r2c_inplace`](Layout::r2c_inplace), whose memory layout, checks and
    /// notes apply: the input is the complex array, and the output its memory read as reals,
    /// the `n_last` real elements of each transform along the halved axis followed by padding.
    /// [`as_reals`](super::as_reals) reads the output.
    ///
    /// ```
    /// use rlst::{c64, rlst_dynamic_array, SliceArray};
    /// use rlst::fftw::{Dim, Layout, Planner, Storage, as_reals};
    ///
    /// // A transform of length 6 in a complex [4] array: the spectrum of a constant 1.
    /// let mut data = rlst_dynamic_array!(c64, [4] | rlst::FFTW_ALIGNED);
    /// data[[0]] = c64::new(6.0, 0.0);
    /// let layout = Layout::c2r_inplace(&data, &[0], 6)?;
    /// assert_eq!(layout.storage(), (Storage::Complex, Storage::Real));
    /// assert_eq!(layout.dims(), &[Dim { n: 6, is: 1, os: 1 }]);
    ///
    /// let plan = Planner::<f64>::new().c2r_inplace(&layout)?;
    /// plan.execute(&mut data)?;
    /// // The 6 reals are 6 times the constant; reals 6 and 7 are padding.
    /// let reals = SliceArray::from_shape(as_reals(data.data().unwrap()), [8]);
    /// assert!((0..6).all(|i| (reals[[i]] - 6.0).abs() < 1e-12));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn c2r_inplace<T, A, const N: usize>(
        complex: &Array<A, N>,
        axes: &[usize],
        n_last: usize,
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        A: BaseItem<Item = Complex<T>> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::padded_in_place(&complex.shape(), &complex.stride(), axes, n_last, false)
    }

    /// The in-place real-to-complex (`r2c`) or complex-to-real layout of a complex array of
    /// `shape` and `stride`, with the checks of [`r2c_inplace`](Layout::r2c_inplace).
    fn padded_in_place(
        shape: &[usize],
        stride: &[usize],
        axes: &[usize],
        n_last: usize,
        r2c: bool,
    ) -> Result<Self, FftwError> {
        let invalid =
            |dimension: String, reason: String| FftwError::InvalidLayout { dimension, reason };
        let rank = shape.len();
        let name = |axis: usize| format!("axis {axis}");
        check_axes(axes, rank, name, &format!("the rank {rank}"))?;
        let halved = *axes.last().expect("`check_axes` rejects empty axes");
        let expected = n_last / 2 + 1;
        if shape[halved] != expected {
            return Err(invalid(
                name(halved),
                format!(
                    "the real extent n_last = {n_last} gives the complex extent n_last / 2 + 1 = \
                     {expected} along the halved axis, the last entry of `axes`, but the \
                     array's extent along it is {}",
                    shape[halved]
                ),
            ));
        }
        // The real side: the same memory, with n_last reals along the halved axis, which has
        // the same stride, and twice the complex strides elsewhere.
        let mut real_shape = shape.to_vec();
        real_shape[halved] = n_last;
        let mut real_stride = Vec::with_capacity(rank);
        for (axis, &complex_stride) in stride.iter().enumerate() {
            if axis == halved {
                real_stride.push(complex_stride);
            } else {
                real_stride.push(complex_stride.checked_mul(2).ok_or_else(|| {
                    invalid(
                        name(axis),
                        format!(
                            "the real stride, twice the complex stride {complex_stride}, does \
                             not fit in usize"
                        ),
                    )
                })?);
            }
        }
        let complex = (shape, stride, ArrayStorage::Complex);
        let real = (&real_shape[..], &real_stride[..], ArrayStorage::Real);
        let (input, output) = if r2c {
            (real, complex)
        } else {
            (complex, real)
        };
        Self::from_arrays(input, output, axes, Placement::InPlace)
    }

    /// The layout of transforms over `axes` of arrays with the given shapes, strides and
    /// storages, with the checks of [`over_axes`](Layout::over_axes). For split storage, the
    /// shapes and strides are those of the logical axes, without the part axes, and the
    /// storage records the stride of the part axis, which becomes the gap.
    fn from_arrays(
        (input_shape, input_stride, input_storage): (&[usize], &[usize], ArrayStorage),
        (output_shape, output_stride, output_storage): (&[usize], &[usize], ArrayStorage),
        axes: &[usize],
        placement: Placement,
    ) -> Result<Self, FftwError> {
        let invalid =
            |dimension: String, reason: String| FftwError::InvalidLayout { dimension, reason };
        let rank = input_shape.len();
        // The split functions count logical axes (DC26), and their errors say so.
        let split = matches!(input_storage, ArrayStorage::Split { .. })
            || matches!(output_storage, ArrayStorage::Split { .. });
        let (name, out_of_range): (fn(usize) -> String, String) = if split {
            (
                |axis| format!("logical axis {axis}"),
                format!("{rank}, the rank of the arrays without their part axes"),
            )
        } else {
            (|axis| format!("axis {axis}"), format!("the rank {rank}"))
        };

        check_axes(axes, rank, name, &out_of_range)?;

        // The halved axis of real-to-complex and complex-to-real layouts is the last entry of
        // `axes` (DC2), on the complex side.
        let halved = match (input_storage.is_complex(), output_storage.is_complex()) {
            (false, true) => Some(Side::Output),
            (true, false) => Some(Side::Input),
            _ => None,
        };
        let halved_axis = halved.and(axes.last().copied());
        let family = match (input_storage, output_storage) {
            (ArrayStorage::Complex, ArrayStorage::Complex) => "complex-to-complex",
            (ArrayStorage::Real, ArrayStorage::Real) => "real-to-real",
            (ArrayStorage::Split { .. }, ArrayStorage::Split { .. }) => "split complex-to-complex",
            (ArrayStorage::Real, ArrayStorage::Split { .. }) => "split real-to-complex",
            (ArrayStorage::Split { .. }, ArrayStorage::Real) => "split complex-to-real",
            (ArrayStorage::Real, _) => "real-to-complex",
            _ => "complex-to-real",
        };
        for axis in 0..rank {
            let (input_extent, output_extent) = (input_shape[axis], output_shape[axis]);
            if let Some(side) = halved
                && halved_axis == Some(axis)
            {
                let (real, complex) = match side {
                    Side::Output => (input_extent, output_extent),
                    Side::Input => (output_extent, input_extent),
                };
                let expected = real / 2 + 1;
                if complex != expected {
                    return Err(invalid(
                        name(axis),
                        format!(
                            "the {side} extent along the halved axis, the last entry of `axes`, \
                             must be {expected}, n / 2 + 1 for the real extent n = {real} of \
                             the {} array, but is {complex}",
                            match side {
                                Side::Input => Side::Output,
                                Side::Output => Side::Input,
                            }
                        ),
                    ));
                }
            } else if input_extent != output_extent {
                let except = if halved.is_some() {
                    " except along the halved axis"
                } else {
                    ""
                };
                return Err(invalid(
                    name(axis),
                    format!(
                        "the input extent {input_extent} differs from the output extent \
                         {output_extent}, and a {family} transform needs arrays of equal \
                         shape{except}"
                    ),
                ));
            }
        }
        if let Some(axis) =
            (0..rank).find(|&axis| input_shape[axis] == 0 || output_shape[axis] == 0)
        {
            return Err(invalid(
                name(axis),
                "the extent is 0, so there is nothing to transform".to_string(),
            ));
        }
        let convert = |dimension: String, what: &str, value: usize| {
            isize::try_from(value).map_err(|_| {
                invalid(
                    dimension,
                    format!("the {what} {value} does not fit in isize"),
                )
            })
        };
        let mut dim_of = Vec::with_capacity(rank);
        for axis in 0..rank {
            // The real extent on the halved axis, the common extent elsewhere.
            let extent = match halved {
                Some(Side::Input) if halved_axis == Some(axis) => output_shape[axis],
                _ => input_shape[axis],
            };
            dim_of.push(Dim {
                n: convert(name(axis), "extent", extent)?,
                is: convert(name(axis), "input stride", input_stride[axis])?,
                os: convert(name(axis), "output stride", output_stride[axis])?,
            });
        }
        let storage = |storage: ArrayStorage, what: &str| -> Result<Storage, FftwError> {
            Ok(match storage {
                ArrayStorage::Complex => Storage::Complex,
                ArrayStorage::Real => Storage::Real,
                ArrayStorage::Split { part_stride } => Storage::Split {
                    gap: convert(PART_AXIS.to_string(), what, part_stride)?,
                },
            })
        };
        let storage = (
            storage(input_storage, "input gap")?,
            storage(output_storage, "output gap")?,
        );

        let batch_axes: Vec<usize> = (0..rank).filter(|axis| !axes.contains(axis)).collect();
        let dims: Vec<Dim> = axes.iter().map(|&axis| dim_of[axis]).collect();
        let batch: Vec<Dim> = batch_axes.iter().map(|&axis| dim_of[axis]).collect();
        Self::with_names(
            &dims,
            &batch,
            storage,
            placement,
            |index| name(axes[index]),
            |index| name(batch_axes[index]),
        )
    }

    /// One-dimensional transforms on split complex storage along the logical axis `axis` of
    /// two real arrays of rank `N`, batched over the other logical axes.
    ///
    /// The same as [`split_over_axes`](Layout::split_over_axes) with the one axis `axis`,
    /// whose checks, errors and notes apply. `part_in` and `part_out` are the *part axes*,
    /// of extent 2: index 0 along them is the real part, index 1 the imaginary part.
    ///
    /// nd-fmm keeps the rings of a box as split planes: a column-major box `[n_phi, rings,
    /// 2]`, the real plane and then the imaginary plane, with the part axis outermost and a
    /// gap of one plane.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Direction, Layout, Planner};
    ///
    /// let (n_phi, rings) = (16, 3);
    /// let mut input = rlst_dynamic_array!(f64, [n_phi, rings, 2] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [n_phi, rings, 2] | rlst::FFTW_ALIGNED);
    /// input.fill_from_seed_equally_distributed(0);
    ///
    /// // Transforms of length n_phi along each ring; the part axis is axis 2.
    /// let layout = Layout::split_along_axis(&input, 2, &output, 2, 0)?;
    /// assert_eq!(layout.dims(), &[Dim { n: 16, is: 1, os: 1 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 3, is: 16, os: 16 }]);
    /// assert_eq!(layout.gaps(), Some((48, 48)));
    ///
    /// let plan = Planner::<f64>::new().split_c2c(&layout, Direction::Forward)?;
    /// plan.execute(&input, &mut output)?;
    ///
    /// // The first entry of each ring's transform is the sum of the ring, in both planes.
    /// for ring in 0..rings {
    ///     for part in 0..2 {
    ///         let sum: f64 = (0..n_phi).map(|i| input[[i, ring, part]]).sum();
    ///         assert!((output[[0, ring, part]] - sum).abs() < 1e-12);
    ///     }
    /// }
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// The arrays must be real, so complex arrays do not compile:
    ///
    /// ```compile_fail,E0277
    /// use rlst::{c64, rlst_dynamic_array};
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(c64, [16, 2] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(c64, [16, 2] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_along_axis(&input, 1, &output, 1, 0)?;
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    ///
    /// while real arrays of the same shape do:
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::Layout;
    ///
    /// let input = rlst_dynamic_array!(f64, [16, 2] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(f64, [16, 2] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_along_axis(&input, 1, &output, 1, 0)?;
    /// assert_eq!(layout.gaps(), Some((16, 16)));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_along_axis<T, AIn, AOut, const N: usize>(
        input: &Array<AIn, N>,
        part_in: usize,
        output: &Array<AOut, N>,
        part_out: usize,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        AIn: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
        AOut: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::split_over_axes(input, part_in, output, part_out, &[axis])
    }

    /// Transforms on split complex storage over the logical axes `axes`, in the order given,
    /// of two real arrays of rank `N`, batched over the other logical axes.
    ///
    /// A split complex array is a real array with one extra axis of extent 2, its *part axis*
    /// (`part_in` for the input, `part_out` for the output): index 0 along it is the real
    /// part, index 1 the imaginary part, and its stride is the array's *gap*. The other
    /// axes are the array's *logical axes*, numbered from 0 in their order without the part
    /// axis, and `axes` counts them: for an input with part axis 0, logical axis 1 is its
    /// axis 2. One `axes` therefore means the same transform on an input and an output whose
    /// part axes are at different positions.
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] that names
    /// `part axis`, `axes` or the logical axis concerned (`logical axis 1`):
    ///
    /// 1. `part_in` and `part_out` are below `N`, and each array's extent along its part axis
    ///    is 2;
    /// 2. the checks of [`over_axes`](Layout::over_axes) on the logical axes: `axes` is not
    ///    empty, its entries are below `N − 1` and not repeated, the logical shapes of input
    ///    and output are equal, every extent is at least 1, and every extent and stride fits
    ///    in `isize`;
    ///
    /// and those of [`Layout::new_split`], whose dimensions are the logical axes of `axes`
    /// with the strides of both arrays, whose batch dimensions are the other logical axes,
    /// and whose gaps are the strides of the two part axes, in reals.
    ///
    /// The arrays must be real arrays, of `f32` or `f64`; the layout counts reals. Its notes
    /// on memory are those of [`over_axes`](Layout::over_axes): building a layout reads only
    /// shapes and strides, and a plan describes memory, not logical axes, so two arrays with
    /// the same memory layout, such as column-major `[2, 8, 3]` with part axis 0 and row-major
    /// `[3, 8, 2]` with part axis 2, give equal layouts.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Layout};
    ///
    /// // A real plane [4, 5] and an imaginary plane after it.
    /// let input = rlst_dynamic_array!(f32, [4, 5, 2] | rlst::FFTW_ALIGNED);
    /// // The same values interleaved: the part axis first.
    /// let output = rlst_dynamic_array!(f32, [2, 4, 5] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_over_axes(&input, 2, &output, 0, &[0, 1])?;
    /// assert_eq!(layout.dims(), &[Dim { n: 4, is: 1, os: 2 }, Dim { n: 5, is: 4, os: 8 }]);
    /// assert!(layout.batch().is_empty());
    /// assert_eq!(layout.gaps(), Some((20, 1)));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_over_axes<T, AIn, AOut, const N: usize>(
        input: &Array<AIn, N>,
        part_in: usize,
        output: &Array<AOut, N>,
        part_out: usize,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        AIn: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
        AOut: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::from_split_arrays(
            &[
                ("input", Some(part_in), &input.shape(), &input.stride()),
                ("output", Some(part_out), &output.shape(), &output.stride()),
            ],
            axes,
            Placement::OutOfPlace,
        )
    }

    /// In-place one-dimensional transforms on split complex storage along the logical axis
    /// `axis` of a real array of rank `N` with part axis `part`, batched over the other
    /// logical axes.
    ///
    /// The same as [`split_over_axes_inplace`](Layout::split_over_axes_inplace) with the one
    /// axis `axis`.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Layout, Placement};
    ///
    /// // Three transforms of length 8 on interleaved storage read as split: part axis 0.
    /// let data = rlst_dynamic_array!(f64, [2, 8, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_along_axis_inplace(&data, 0, 0)?;
    /// assert_eq!(layout.placement(), Placement::InPlace);
    /// assert_eq!(layout.dims(), &[Dim { n: 8, is: 2, os: 2 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 3, is: 16, os: 16 }]);
    /// assert_eq!(layout.gaps(), Some((1, 1)));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_along_axis_inplace<T, A, const N: usize>(
        data: &Array<A, N>,
        part: usize,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        A: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::split_over_axes_inplace(data, part, &[axis])
    }

    /// In-place transforms on split complex storage over the logical axes `axes`, in the
    /// order given, of a real array of rank `N` with part axis `part`, batched over the other
    /// logical axes.
    ///
    /// As [`split_over_axes`](Layout::split_over_axes) with `data` as both input and output,
    /// whose checks, errors and notes apply; the placement is [`Placement::InPlace`].
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Layout};
    ///
    /// // The part axis in the middle: [4, 2, 6], transforms over logical axes 1 and 0.
    /// let data = rlst_dynamic_array!(f64, [4, 2, 6] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_over_axes_inplace(&data, 1, &[1, 0])?;
    /// assert_eq!(layout.dims(), &[Dim { n: 6, is: 8, os: 8 }, Dim { n: 4, is: 1, os: 1 }]);
    /// assert_eq!(layout.gaps(), Some((4, 4)));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_over_axes_inplace<T, A, const N: usize>(
        data: &Array<A, N>,
        part: usize,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        A: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::from_split_arrays(
            &[("array", Some(part), &data.shape(), &data.stride())],
            axes,
            Placement::InPlace,
        )
    }

    /// One-dimensional real-to-complex transforms along the logical axis `axis`, batched over
    /// the other axes, from the real array `input` of rank `N` into the split complex array
    /// `output` of rank `M = N + 1`, with part axis `part_out`.
    ///
    /// The same as [`split_r2c_over_axes`](Layout::split_r2c_over_axes) with the one axis
    /// `axis`, whose checks, errors and notes apply; `axis` is halved.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Layout, Planner, Storage};
    ///
    /// // Three transforms of length 8 into a real and an imaginary plane [5, 3] each.
    /// let mut input = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [5, 3, 2] | rlst::FFTW_ALIGNED);
    /// input.fill_from_seed_equally_distributed(0);
    /// let layout = Layout::split_r2c_along_axis(&input, &output, 2, 0)?;
    /// assert_eq!(layout.storage(), (Storage::Real, Storage::Split { gap: 15 }));
    /// assert_eq!(layout.dims(), &[Dim { n: 8, is: 1, os: 1 }]);
    /// assert_eq!(layout.batch(), &[Dim { n: 3, is: 8, os: 5 }]);
    ///
    /// let plan = Planner::<f64>::new().split_r2c(&layout)?;
    /// plan.execute(&input, &mut output)?;
    /// // Frequency 0 of each column is its sum, with imaginary part 0.
    /// let sum: f64 = (0..8).map(|i| input[[i, 1]]).sum();
    /// assert!((output[[0, 1, 0]] - sum).abs() < 1e-12 && output[[0, 1, 1]].abs() < 1e-12);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_r2c_along_axis<T, AIn, AOut, const N: usize, const M: usize>(
        input: &Array<AIn, N>,
        output: &Array<AOut, M>,
        part_out: usize,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        AIn: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
        AOut: BaseItem<Item = T> + StridedRawAccess + Shape<M> + Stride<M>,
    {
        Self::split_r2c_over_axes(input, output, part_out, &[axis])
    }

    /// Real-to-complex transforms over the logical axes `axes`, in the order given, batched
    /// over the other axes, from the real array `input` of rank `N` into the split complex
    /// array `output` of rank `M = N + 1`, with part axis `part_out` (DC33).
    ///
    /// The output keeps the real parts of the half spectrum at index 0 of its part axis and
    /// the imaginary parts at index 1, the part axis's stride, its gap, apart. Its other axes
    /// are its logical axes, which `axes` counts (DC26), and which pair with the input's axes
    /// in order. The halved axis is the last entry of `axes` (DC2): along it the output has
    /// `n / 2 + 1` complex values for the input's `n` reals; as for
    /// [`over_axes`](Layout::over_axes), no order is recommended. Both arrays are real (DC29), of
    /// one precision. There is no in-place variant: RLST has no in-place split
    /// real-to-complex plans (DC33).
    ///
    /// The checks, in this order, each returning [`FftwError::InvalidLayout`] that names
    /// `part axis`, `axes` or the logical axis concerned: `part_out` is below `M`, the
    /// output's extent along it is 2, and `M = N + 1`; then the checks of
    /// [`over_axes`](Layout::over_axes) on the logical axes, the halved extent included; and
    /// those of [`Layout::with_storage`] for the storages `(Storage::Real, Storage::Split {
    /// gap })`.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Layout};
    ///
    /// // Over both axes of a real [6, 4]: logical axis 1 is halved, to 3. The output's part
    /// // axis is innermost, so its memory is that of interleaved complex values.
    /// let input = rlst_dynamic_array!(f32, [6, 4] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(f32, [2, 6, 3] | rlst::FFTW_ALIGNED);
    /// let layout = Layout::split_r2c_over_axes(&input, &output, 0, &[0, 1])?;
    /// assert_eq!(layout.dims(), &[Dim { n: 6, is: 1, os: 2 }, Dim { n: 4, is: 6, os: 12 }]);
    /// assert_eq!(layout.output_span(), 36);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_r2c_over_axes<T, AIn, AOut, const N: usize, const M: usize>(
        input: &Array<AIn, N>,
        output: &Array<AOut, M>,
        part_out: usize,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        AIn: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
        AOut: BaseItem<Item = T> + StridedRawAccess + Shape<M> + Stride<M>,
    {
        Self::from_split_arrays(
            &[
                ("input", None, &input.shape(), &input.stride()),
                ("output", Some(part_out), &output.shape(), &output.stride()),
            ],
            axes,
            Placement::OutOfPlace,
        )
    }

    /// One-dimensional complex-to-real transforms along the logical axis `axis`, batched over
    /// the other axes, from the split complex array `input` of rank `M = N + 1`, with part
    /// axis `part_in`, into the real array `output` of rank `N`.
    ///
    /// The same as [`split_c2r_over_axes`](Layout::split_c2r_over_axes) with the one axis
    /// `axis`, whose checks, errors and notes apply; `axis` is halved.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Layout, Planner};
    ///
    /// // The spectrum of a constant 1 of length 8, real plane then imaginary plane.
    /// let mut input = rlst_dynamic_array!(f64, [5, 2] | rlst::FFTW_ALIGNED);
    /// let mut output = rlst_dynamic_array!(f64, [8] | rlst::FFTW_ALIGNED);
    /// input[[0, 0]] = 8.0;
    /// let layout = Layout::split_c2r_along_axis(&input, 1, &output, 0)?;
    /// // Like c2r, split c2r destroys its input by default.
    /// let plan = Planner::<f64>::new().split_c2r(&layout)?;
    /// plan.execute(&mut input, &mut output)?;
    /// assert!((0..8).all(|i| (output[[i]] - 8.0).abs() < 1e-12));
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2r_along_axis<T, AIn, AOut, const N: usize, const M: usize>(
        input: &Array<AIn, M>,
        part_in: usize,
        output: &Array<AOut, N>,
        axis: usize,
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        AIn: BaseItem<Item = T> + StridedRawAccess + Shape<M> + Stride<M>,
        AOut: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::split_c2r_over_axes(input, part_in, output, &[axis])
    }

    /// Complex-to-real transforms over the logical axes `axes`, in the order given, batched
    /// over the other axes, from the split complex array `input` of rank `M = N + 1`, with
    /// part axis `part_in`, into the real array `output` of rank `N` (DC33).
    ///
    /// The inverse of [`split_r2c_over_axes`](Layout::split_r2c_over_axes), whose memory
    /// layout, checks and notes apply with the sides swapped: the input holds the half
    /// spectrum, `n / 2 + 1` complex values along the halved axis, the last entry of `axes`,
    /// for the output's `n` reals, and the storages are `(Storage::Split { gap },
    /// Storage::Real)`. There is no in-place variant.
    ///
    /// ```
    /// use rlst::rlst_dynamic_array;
    /// use rlst::fftw::{Dim, Layout, Storage};
    ///
    /// let input = rlst_dynamic_array!(f64, [3, 2, 7] | rlst::FFTW_ALIGNED);
    /// let output = rlst_dynamic_array!(f64, [3, 12] | rlst::FFTW_ALIGNED);
    /// // Logical axis 1 of the input is its axis 2: 7 = 12 / 2 + 1.
    /// let layout = Layout::split_c2r_over_axes(&input, 1, &output, &[1])?;
    /// assert_eq!(layout.storage(), (Storage::Split { gap: 3 }, Storage::Real));
    /// assert_eq!(layout.dims(), &[Dim { n: 12, is: 6, os: 3 }]);
    /// # Ok::<(), rlst::fftw::FftwError>(())
    /// ```
    pub fn split_c2r_over_axes<T, AIn, AOut, const N: usize, const M: usize>(
        input: &Array<AIn, M>,
        part_in: usize,
        output: &Array<AOut, N>,
        axes: &[usize],
    ) -> Result<Self, FftwError>
    where
        T: FftwReal,
        AIn: BaseItem<Item = T> + StridedRawAccess + Shape<M> + Stride<M>,
        AOut: BaseItem<Item = T> + StridedRawAccess + Shape<N> + Stride<N>,
    {
        Self::from_split_arrays(
            &[
                ("input", Some(part_in), &input.shape(), &input.stride()),
                ("output", None, &output.shape(), &output.stride()),
            ],
            axes,
            Placement::OutOfPlace,
        )
    }

    /// The split layout of transforms over the logical axes `axes` of arrays given as
    /// `(name, part axis, shape, strides)`: input and output, or one array in place, with the
    /// checks of [`split_over_axes`](Layout::split_over_axes). An array without a part axis
    /// (`None`) is a real side, of split real-to-complex and complex-to-real layouts, whose
    /// rank must be one less than the split array's.
    fn from_split_arrays(
        arrays: &[SplitArray<'_>],
        axes: &[usize],
        placement: Placement,
    ) -> Result<Self, FftwError> {
        let invalid = |reason: String| FftwError::InvalidLayout {
            dimension: PART_AXIS.to_string(),
            reason,
        };
        let mut logical = Vec::with_capacity(arrays.len());
        for &(name, part, shape, stride) in arrays {
            let rank = shape.len();
            let Some(part) = part else {
                logical.push((shape.to_vec(), stride.to_vec(), ArrayStorage::Real));
                continue;
            };
            if part >= rank {
                return Err(invalid(format!(
                    "the {name}'s part axis {part} is not below the rank {rank}"
                )));
            }
            if shape[part] != 2 {
                return Err(invalid(format!(
                    "the {name}'s extent along its part axis {part} is {}, and split storage \
                     needs 2, the real and the imaginary part",
                    shape[part]
                )));
            }
            let gap = stride[part];
            let (shape, stride) = remove_axis(shape, stride, part);
            logical.push((shape, stride, ArrayStorage::Split { part_stride: gap }));
        }
        let (input, output) = (&logical[0], &logical[logical.len() - 1]);
        if input.0.len() != output.0.len() {
            let (real, split) = if matches!(input.2, ArrayStorage::Real) {
                (arrays[0], arrays[1])
            } else {
                (arrays[1], arrays[0])
            };
            return Err(invalid(format!(
                "the {} has rank {}, and its logical axes, without the part axis, are {}, but \
                 the real {} has rank {}; the split array needs one axis more than the real \
                 array, its part axis",
                split.0,
                split.2.len(),
                split.2.len() - 1,
                real.0,
                real.2.len()
            )));
        }
        Self::from_arrays(
            (&input.0, &input.1, input.2),
            (&output.0, &output.1, output.2),
            axes,
            placement,
        )
    }

    /// Whether an array of `shape` and `stride` fits `side` of the layout (design Section
    /// 4.8 item 2, DC15): each transform dimension is matched by an axis of its extent on
    /// this side (`n / 2 + 1` for the halved dimension on the complex side) and, above extent
    /// 1, the same stride, and the other axes, normalised on this side alone, equal the
    /// layout's batch on this side, normalised in the same way.
    ///
    /// For split storage (DC27) the part axis is found first: an axis of extent 2 whose
    /// stride is this side's gap. The array fits if, for some such axis, the other axes fit
    /// as above. A part axis that a transform dimension needs leaves that dimension without
    /// a match, so trying every candidate covers the rule that no transform dimension needs
    /// it; several candidates can exist only on an input, whose elements need not be
    /// distinct.
    pub(crate) fn fits_array_side(&self, side: Side, shape: &[usize], stride: &[usize]) -> bool {
        part_choices(self.gap(side), shape, stride)
            .any(|part| self.fits_logical_side(side, (shape, stride, part)))
    }

    /// The check of [`fits_array_side`](Layout::fits_array_side) on the axes without the part
    /// axis `part`, if any. Ranks up to [`INLINE_RANK`] need no allocation.
    fn fits_logical_side(&self, side: Side, (shape, stride, part): LogicalArray<'_>) -> bool {
        let rank = shape.len() - usize::from(part.is_some());
        with_scratch(rank, NO_DIM, |axes| {
            let (shape, stride) = (without(shape, part), without(stride, part));
            if !array_dims(shape, stride.clone(), stride, axes) {
                return false;
            }
            with_scratch(rank, false, |used| {
                let extent_one = |index: usize| self.extent(index, side) == 1;
                let matched =
                    match_transform_dims(self.dims.len(), extent_one, used, |index, axis| {
                        axes[axis].n == self.extent(index, side)
                            && (extent_one(index)
                                || axes[axis].is == side_stride(&self.dims[index], side))
                    });
                if !matched {
                    return false;
                }
                let rest = keep_unused(axes, used);
                let len = normalise_in_place(&mut axes[..rest]);
                axes[..len] == *self.side_batch(side)
            })
        })
    }

    /// Whether an input and an output array of the same rank fit the layout with their axes
    /// paired by index (DC15): each transform dimension matched by one axis with its extent on
    /// each side (`n / 2 + 1` for the halved dimension on the complex side) and, above extent
    /// 1, both strides, the other axes of equal extent on both sides, and, normalised
    /// together, equal to the layout's batch.
    ///
    /// For split storage (DC27) each side's part axis, found as in
    /// [`fits_array_side`](Layout::fits_array_side), is removed first, and the remaining
    /// logical axes pair by index; the arrays fit if some choice of part axes does.
    pub(crate) fn fits_array_pair(
        &self,
        (input_shape, input_stride): (&[usize], &[usize]),
        (output_shape, output_stride): (&[usize], &[usize]),
    ) -> bool {
        part_choices(self.gap(Side::Input), input_shape, input_stride).any(|input_part| {
            part_choices(self.gap(Side::Output), output_shape, output_stride).any(|output_part| {
                self.fits_logical_pair(
                    (input_shape, input_stride, input_part),
                    (output_shape, output_stride, output_part),
                )
            })
        })
    }

    /// The check of [`fits_array_pair`](Layout::fits_array_pair) on the axes without the part
    /// axes, if any. Ranks up to [`INLINE_RANK`] need no allocation.
    fn fits_logical_pair(
        &self,
        (input_shape, input_stride, input_part): LogicalArray<'_>,
        (output_shape, output_stride, output_part): LogicalArray<'_>,
    ) -> bool {
        let rank = input_shape.len() - usize::from(input_part.is_some());
        if rank != output_shape.len() - usize::from(output_part.is_some()) {
            return false;
        }
        with_scratch(rank, NO_DIM, |axes| {
            if !array_dims(
                without(input_shape, input_part),
                without(input_stride, input_part),
                without(output_stride, output_part),
                axes,
            ) {
                return false;
            }
            with_scratch(rank, 0, |output_extents| {
                for (extent, n) in output_extents
                    .iter_mut()
                    .zip(without(output_shape, output_part))
                {
                    let Ok(n) = isize::try_from(n) else {
                        return false;
                    };
                    *extent = n;
                }
                with_scratch(rank, false, |used| {
                    let extent_one = |index: usize| {
                        self.extent(index, Side::Input) == 1
                            && self.extent(index, Side::Output) == 1
                    };
                    let matched =
                        match_transform_dims(self.dims.len(), extent_one, used, |index, axis| {
                            let dim = &self.dims[index];
                            axes[axis].n == self.extent(index, Side::Input)
                                && output_extents[axis] == self.extent(index, Side::Output)
                                && (extent_one(index)
                                    || (axes[axis].is == dim.is && axes[axis].os == dim.os))
                        });
                    if !matched
                        || !(0..rank).all(|axis| used[axis] || axes[axis].n == output_extents[axis])
                    {
                        return false;
                    }
                    let rest = keep_unused(axes, used);
                    let len = normalise_in_place(&mut axes[..rest]);
                    axes[..len] == *self.batch
                })
            })
        })
    }

    /// The batch on `side` alone, normalised.
    fn side_batch(&self, side: Side) -> &[Dim] {
        match side {
            Side::Input => &self.side_batches.0,
            Side::Output => &self.side_batches.1,
        }
    }

    /// The dimensions on `side`, for error messages.
    pub(crate) fn describe_side(&self, side: Side) -> String {
        let stride = |dim: &Dim| match side {
            Side::Input => dim.is,
            Side::Output => dim.os,
        };
        let dims = self
            .dims
            .iter()
            .enumerate()
            .map(|(index, dim)| format!("(n {}, stride {})", self.extent(index, side), stride(dim)))
            .collect::<Vec<_>>()
            .join(", ");
        let batch = self
            .batch
            .iter()
            .map(|dim| format!("(n {}, stride {})", dim.n, stride(dim)))
            .collect::<Vec<_>>()
            .join(", ");
        let part = match self.gap(side) {
            Some(gap) => format!(" and a part axis (n 2, stride {gap})"),
            None => String::new(),
        };
        format!("transform dimensions [{dims}] and batch dimensions [{batch}]{part}")
    }
}

/// The name of the part axis of split storage in errors.
const PART_AXIS: &str = "part axis";

/// An array given to the split layout functions: its name in errors, its part axis (`None`
/// for the real side of a split real-to-complex or complex-to-real layout), its shape and its
/// strides.
type SplitArray<'a> = (&'a str, Option<usize>, &'a [usize], &'a [usize]);

/// Checks 1 to 3 of [`Layout::over_axes`]: `axes` is not empty, its entries are below `rank`
/// (`out_of_range` describes the bound) and none is repeated. `name` names an axis.
fn check_axes(
    axes: &[usize],
    rank: usize,
    name: impl Fn(usize) -> String,
    out_of_range: &str,
) -> Result<(), FftwError> {
    let invalid =
        |dimension: String, reason: String| FftwError::InvalidLayout { dimension, reason };
    if axes.is_empty() {
        return Err(invalid(
            "axes".to_string(),
            "no axis is given, and a transform needs at least one".to_string(),
        ));
    }
    if let Some(&axis) = axes.iter().find(|&&axis| axis >= rank) {
        return Err(invalid(
            "axes".to_string(),
            format!("the {} is not below {out_of_range}", name(axis)),
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
    Ok(())
}

/// The in-place rule of real-to-complex (`r2c`) and complex-to-real layouts (DC32), on the
/// named dimensions, transform dimensions first, whose transform dimension `last` is halved:
/// the halved dimension has stride 1 on both sides, and every other dimension a real stride
/// twice its complex stride. Dimensions of extent 1 address one element and are not checked.
fn check_padded_in_place(named: &[(String, Dim)], last: usize, r2c: bool) -> Result<(), FftwError> {
    for (index, (name, dim)) in named.iter().enumerate() {
        if dim.n == 1 {
            continue;
        }
        let (real, complex) = if r2c {
            (dim.is, dim.os)
        } else {
            (dim.os, dim.is)
        };
        let reason = if index == last {
            (real != 1 || complex != 1).then(|| {
                format!(
                    "in place, the halved dimension, the last transform dimension, must be \
                     contiguous, as the reals of each complex element hold two real elements: \
                     both strides must be 1, but the real stride is {real} and the complex \
                     stride {complex}"
                )
            })
        } else {
            (complex.checked_mul(2) != Some(real)).then(|| {
                format!(
                    "in place, the real side is the complex buffer read as reals, so the real \
                     stride must be twice the complex stride {complex}, but it is {real}"
                )
            })
        };
        if let Some(reason) = reason {
            return Err(FftwError::InvalidLayout {
                dimension: name.clone(),
                reason,
            });
        }
    }
    Ok(())
}

/// The distinct-output rule of design Section 4.3 on the dimensions of one side: with the
/// dimensions of extent above 1 sorted by stride, every stride is positive and at least the
/// extent times the stride of the next smaller one.
fn check_distinct(dims: &[SideDim<'_>], side: Side) -> Result<(), FftwError> {
    let invalid = |dimension: &str, reason: String| FftwError::InvalidLayout {
        dimension: dimension.to_string(),
        reason,
    };
    let mut sorted: Vec<&SideDim<'_>> = dims.iter().filter(|dim| dim.n > 1).collect();
    sorted.sort_by_key(|dim| dim.stride);
    if let Some(dim) = sorted.first()
        && dim.stride == 0
    {
        return Err(invalid(
            dim.name,
            format!(
                "the {side} stride is 0 with extent {}, so the {side}s overlap",
                dim.n
            ),
        ));
    }
    for pair in sorted.windows(2) {
        let (inner, dim) = (pair[0], pair[1]);
        let Some(needed) = inner.n.checked_mul(inner.stride) else {
            return Err(invalid(
                inner.name,
                format!("the extent times the {side} stride overflows isize"),
            ));
        };
        if dim.stride < needed {
            return Err(invalid(
                dim.name,
                format!(
                    "the {side} stride {} is below {needed}, the extent {} times the {side} \
                     stride {} of {}, so the {side}s may overlap",
                    dim.stride, inner.n, inner.stride, inner.name
                ),
            ));
        }
    }
    Ok(())
}

/// An array given to the execution checks: its shape, its strides, and its part axis, if one
/// is removed.
type LogicalArray<'a> = (&'a [usize], &'a [usize], Option<usize>);

/// The rank up to which the execution checks keep their scratch on the stack.
const INLINE_RANK: usize = 16;

/// The fill of scratch dimensions.
const NO_DIM: Dim = Dim { n: 0, is: 0, os: 0 };

/// Run `f` on `len` copies of `fill`: on the stack up to [`INLINE_RANK`], so that executing a
/// plan does not allocate, and on the heap above it.
fn with_scratch<T: Copy, R>(len: usize, fill: T, f: impl FnOnce(&mut [T]) -> R) -> R {
    if len <= INLINE_RANK {
        let mut scratch = [fill; INLINE_RANK];
        f(&mut scratch[..len])
    } else {
        f(&mut vec![fill; len])
    }
}

/// The stride of `dim` on `side`.
fn side_stride(dim: &Dim, side: Side) -> isize {
    match side {
        Side::Input => dim.is,
        Side::Output => dim.os,
    }
}

/// The axes of extent 2 whose stride is `gap`: the candidates for a split array's part axis.
fn part_axes<'a>(
    shape: &'a [usize],
    stride: &'a [usize],
    gap: isize,
) -> impl Iterator<Item = usize> + 'a {
    (0..shape.len())
        .filter(move |&axis| shape[axis] == 2 && isize::try_from(stride[axis]) == Ok(gap))
}

/// The part axes to try on a side with `gap`: `None` alone if the side is not split, and
/// otherwise each candidate of [`part_axes`].
fn part_choices<'a>(
    gap: Option<isize>,
    shape: &'a [usize],
    stride: &'a [usize],
) -> impl Iterator<Item = Option<usize>> + 'a {
    let split = gap.map(|gap| part_axes(shape, stride, gap).map(Some));
    let interleaved = gap.is_none().then_some(None);
    split.into_iter().flatten().chain(interleaved)
}

/// The shape and strides without `axis`.
fn remove_axis(shape: &[usize], stride: &[usize], axis: usize) -> (Vec<usize>, Vec<usize>) {
    let without = |values: &[usize]| {
        values
            .iter()
            .enumerate()
            .filter(|&(index, _)| index != axis)
            .map(|(_, &value)| value)
            .collect()
    };
    (without(shape), without(stride))
}

/// The values without the one at `skip`, if any.
fn without(values: &[usize], skip: Option<usize>) -> impl Iterator<Item = usize> + Clone + '_ {
    values
        .iter()
        .enumerate()
        .filter(move |&(index, _)| Some(index) != skip)
        .map(|(_, &value)| value)
}

/// Write the axes of an array to `axes` as dimensions with the given extents and input and
/// output strides. Returns `false` if an extent or a stride does not fit in `isize`.
fn array_dims(
    shape: impl Iterator<Item = usize>,
    input_stride: impl Iterator<Item = usize>,
    output_stride: impl Iterator<Item = usize>,
    axes: &mut [Dim],
) -> bool {
    for (axis, (n, (is, os))) in axes
        .iter_mut()
        .zip(shape.zip(input_stride.zip(output_stride)))
    {
        let (Ok(n), Ok(is), Ok(os)) =
            (isize::try_from(n), isize::try_from(is), isize::try_from(os))
        else {
            return false;
        };
        *axis = Dim { n, is, os };
    }
    true
}

/// Match each of the `count` transform dimensions with a distinct axis for which
/// `fits(dimension, axis)` holds, marking the axes used in `used`, one entry per axis.
/// Dimensions for which `extent_one` holds address one element, and are matched after the
/// others, so that they take axes of extent 1 that no other dimension needs. Returns `false`
/// if a dimension has no match.
fn match_transform_dims(
    count: usize,
    extent_one: impl Fn(usize) -> bool,
    used: &mut [bool],
    fits: impl Fn(usize, usize) -> bool,
) -> bool {
    used.fill(false);
    for one in [false, true] {
        for index in (0..count).filter(|&index| extent_one(index) == one) {
            let Some(axis) = (0..used.len()).find(|&axis| !used[axis] && fits(index, axis)) else {
                return false;
            };
            used[axis] = true;
        }
    }
    true
}

/// Move the axes not used by the transform dimensions to the front of `axes`, in their order,
/// and return their number.
fn keep_unused(axes: &mut [Dim], used: &[bool]) -> usize {
    let mut kept = 0;
    for axis in 0..axes.len() {
        if !used[axis] {
            axes[kept] = axes[axis];
            kept += 1;
        }
    }
    kept
}

/// Normalise batch dimensions (design Section 4.3): drop those of extent 1, sort by input
/// stride and then output stride, largest first, and merge each adjacent pair whose outer
/// strides are the inner extent times the inner strides, on both sides.
fn normalise_batch(batch: &[Dim]) -> Vec<Dim> {
    let mut dims = batch.to_vec();
    let len = normalise_in_place(&mut dims);
    dims.truncate(len);
    dims
}

/// [`normalise_batch`] in place: the normalised dimensions are the first ones of `dims`, and
/// their number is returned. The sort is stable, as the merge depends on the order of
/// dimensions with equal strides, and for up to 20 dimensions it does not allocate.
fn normalise_in_place(dims: &mut [Dim]) -> usize {
    let mut len = 0;
    for index in 0..dims.len() {
        if dims[index].n > 1 {
            dims[len] = dims[index];
            len += 1;
        }
    }
    let dims = &mut dims[..len];
    dims.sort_by_key(|dim| std::cmp::Reverse((dim.is, dim.os)));

    // Merge from the innermost dimension outwards.
    dims.reverse();
    let mut merged = 0usize;
    for index in 0..dims.len() {
        let dim = dims[index];
        if let Some(inner) = merged.checked_sub(1).map(|last| &mut dims[last])
            && inner.n.checked_mul(inner.is) == Some(dim.is)
            && inner.n.checked_mul(inner.os) == Some(dim.os)
            && let Some(n) = inner.n.checked_mul(dim.n)
        {
            inner.n = n;
        } else {
            dims[merged] = dim;
            merged += 1;
        }
    }
    dims[..merged].reverse();
    merged
}

#[cfg(test)]
mod tests {
    use super::{Dim, Layout, Placement, Storage};
    use crate::dense::array::StridedSliceArray;
    use crate::fftw::{FftwError, Side};
    use crate::{DynArray, StridedDynArray, c32, c64};

    fn split(dims: &[Dim], batch: &[Dim], gaps: (isize, isize), placement: Placement) -> Layout {
        Layout::new_split(dims, batch, gaps, placement).unwrap()
    }

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

    // Split layouts (task F3).

    #[test]
    fn test_split_layouts_of_the_brief() {
        let out = Placement::OutOfPlace;

        // nd-fmm's box: column-major [140, 9, 2], part axis 2.
        let a = DynArray::<f64, 3>::from_shape([140, 9, 2]);
        assert_eq!(a.stride(), [1, 140, 1260]);
        assert_eq!(
            Layout::split_along_axis(&a, 2, &a, 2, 0).unwrap(),
            split(&[dim(140, 1, 1)], &[dim(9, 140, 140)], (1260, 1260), out)
        );

        // Column-major [2, 8, 3], part axis 0, and the same memory as row-major [3, 8, 2]
        // with part axis 2: equal layouts (DC15).
        let a = DynArray::<f64, 3>::from_shape([2, 8, 3]);
        let interleaved = split(&[dim(8, 2, 2)], &[dim(3, 16, 16)], (1, 1), out);
        assert_eq!(
            Layout::split_along_axis(&a, 0, &a, 0, 0).unwrap(),
            interleaved
        );
        let b = StridedDynArray::<f64, 3>::row_major([3, 8, 2]);
        assert_eq!(b.stride(), [16, 2, 1]);
        assert_eq!(
            Layout::split_along_axis(&b, 2, &b, 2, 1).unwrap(),
            interleaved
        );

        // Column-major [8, 2, 3], part axis 1 in the middle, along logical axis 1.
        let a = DynArray::<f64, 3>::from_shape([8, 2, 3]);
        assert_eq!(
            Layout::split_along_axis(&a, 1, &a, 1, 1).unwrap(),
            split(&[dim(3, 16, 16)], &[dim(8, 1, 1)], (8, 8), out)
        );

        // Column-major [4, 5, 2], part axis 2, over both logical axes.
        let a = DynArray::<f64, 3>::from_shape([4, 5, 2]);
        assert_eq!(
            Layout::split_over_axes(&a, 2, &a, 2, &[0, 1]).unwrap(),
            split(&[dim(4, 1, 1), dim(5, 4, 4)], &[], (20, 20), out)
        );

        // Mixed part axes: planes in, interleaved out.
        let input = DynArray::<f32, 3>::from_shape([8, 3, 2]);
        let output = DynArray::<f32, 3>::from_shape([2, 8, 3]);
        assert_eq!(
            Layout::split_along_axis(&input, 2, &output, 0, 0).unwrap(),
            split(&[dim(8, 1, 2)], &[dim(3, 8, 16)], (24, 1), out)
        );
    }

    #[test]
    fn test_split_along_axis_is_over_one_axis() {
        let a = DynArray::<f64, 4>::from_shape([6, 5, 4, 2]);
        let b = StridedDynArray::<f64, 4>::row_major([2, 6, 5, 4]);
        for axis in 0..3 {
            assert_eq!(
                Layout::split_along_axis(&a, 3, &b, 0, axis).unwrap(),
                Layout::split_over_axes(&a, 3, &b, 0, &[axis]).unwrap()
            );
            assert_eq!(
                Layout::split_along_axis_inplace(&b, 0, axis).unwrap(),
                Layout::split_over_axes_inplace(&b, 0, &[axis]).unwrap()
            );
        }
        let layout = Layout::split_over_axes_inplace(&a, 3, &[2, 0]).unwrap();
        assert_eq!(
            layout,
            split(
                &[dim(4, 30, 30), dim(6, 1, 1)],
                &[dim(5, 6, 6)],
                (120, 120),
                Placement::InPlace
            )
        );
        assert_eq!(layout.input_span(), 240);
    }

    #[test]
    fn test_split_part_axis_is_not_merged_into_the_batch() {
        // A part axis of stride 1 beside a batch axis of stride 2: merging them would describe
        // one batch of 2 x 3 with stride 1.
        let a = DynArray::<f64, 3>::from_shape([2, 3, 4]);
        let layout = Layout::split_along_axis(&a, 0, &a, 0, 1).unwrap();
        assert_eq!(layout.dims(), &[dim(4, 6, 6)]);
        assert_eq!(layout.batch(), &[dim(3, 2, 2)]);
        assert_eq!(layout.gaps(), Some((1, 1)));
        assert_eq!(layout.input_span(), 24);
        // A split layout and an interleaved one of the same dimensions differ.
        assert_ne!(
            layout,
            Layout::new(layout.dims(), layout.batch(), Placement::OutOfPlace).unwrap()
        );
    }

    #[test]
    fn test_new_split_rules() {
        let out = Placement::OutOfPlace;
        let split_invalid =
            |dims: &[Dim], gaps, placement| match Layout::new_split(dims, &[], gaps, placement) {
                Err(FftwError::InvalidLayout { dimension, reason }) => (dimension, reason),
                other => panic!("expected InvalidLayout, found {other:?}"),
            };

        // The rules of `Layout::new` on the dimensions.
        let (dimension, _) = split_invalid(&[], (1, 1), out);
        assert_eq!(dimension, "layout");
        let (dimension, _) = split_invalid(&[dim(0, 1, 1)], (1, 1), out);
        assert_eq!(dimension, "transform dimension 0");
        let (dimension, _) = split_invalid(&[dim(4, 2, 1)], (8, 8), Placement::InPlace);
        assert_eq!(dimension, "transform dimension 0");

        // Positive gaps, equal in place.
        for gaps in [(0, 4), (4, 0), (-4, 4)] {
            let (dimension, reason) = split_invalid(&[dim(4, 1, 1)], gaps, out);
            assert_eq!(dimension, "part axis");
            assert!(reason.contains("positive"), "{reason}");
        }
        let (dimension, reason) = split_invalid(&[dim(4, 1, 1)], (4, 8), Placement::InPlace);
        assert_eq!(dimension, "part axis");
        assert!(reason.contains("must be equal"), "{reason}");
        assert!(Layout::new_split(&[dim(4, 1, 1)], &[], (4, 8), out).is_ok());

        // Distinct outputs with the part axis: an output gap below the transform's extent
        // overlaps, a gap equal to it does not.
        let (dimension, _) = split_invalid(&[dim(4, 1, 1)], (4, 3), out);
        assert_eq!(dimension, "part axis");
        let (dimension, _) = split_invalid(&[dim(4, 2, 2)], (8, 3), out);
        assert_eq!(dimension, "part axis");
        assert!(Layout::new_split(&[dim(4, 2, 2)], &[], (8, 1), out).is_ok());
        // The input may overlap.
        assert!(Layout::new_split(&[dim(4, 1, 1)], &[], (1, 4), out).is_ok());

        // The spans include the gaps.
        let layout = split(&[dim(4, 3, 1)], &[dim(2, 12, 4)], (100, 8), out);
        assert_eq!(layout.input_span(), 1 + 9 + 12 + 100);
        assert_eq!(layout.output_span(), 1 + 3 + 4 + 8);
        assert_eq!(layout.logical_size(), 4);
        let (dimension, reason) = split_invalid(&[dim(2, 1, 1)], (isize::MAX, 2), out);
        assert_eq!(dimension, "layout");
        assert!(reason.contains("input span"), "{reason}");
    }

    #[test]
    fn test_split_array_layout_errors() {
        let a = DynArray::<f64, 3>::from_shape([8, 3, 2]);
        let (dimension, reason) = array_invalid(Layout::split_along_axis(&a, 3, &a, 2, 0));
        assert_eq!(dimension, "part axis");
        assert!(
            reason.contains("input's part axis 3 is not below the rank 3"),
            "{reason}"
        );
        let (dimension, reason) = array_invalid(Layout::split_along_axis(&a, 2, &a, 1, 0));
        assert_eq!(dimension, "part axis");
        assert!(
            reason.contains("output's extent along its part axis 1 is 3"),
            "{reason}"
        );
        let b = DynArray::<f64, 2>::from_shape([8, 3]);
        let (dimension, reason) = array_invalid(Layout::split_along_axis_inplace(&b, 1, 0));
        assert_eq!(dimension, "part axis");
        assert!(reason.contains("array's extent"), "{reason}");

        // Logical axes (DC26): rank 3 with a part axis has logical axes 0 and 1.
        let (dimension, reason) = array_invalid(Layout::split_along_axis(&a, 2, &a, 2, 2));
        assert_eq!(dimension, "axes");
        assert!(reason.contains("logical axis 2 is not below 2"), "{reason}");
        let (dimension, _) = array_invalid(Layout::split_over_axes(&a, 2, &a, 2, &[]));
        assert_eq!(dimension, "axes");
        let (dimension, _) = array_invalid(Layout::split_over_axes_inplace(&a, 2, &[1, 1]));
        assert_eq!(dimension, "logical axis 1");

        // Unequal logical shapes, named by the logical axis: [8, 3] against [8, 4].
        let c = DynArray::<f64, 3>::from_shape([2, 8, 4]);
        let (dimension, reason) = array_invalid(Layout::split_along_axis(&a, 2, &c, 0, 0));
        assert_eq!(dimension, "logical axis 1");
        assert!(reason.contains("input extent 3"), "{reason}");
        // The same logical shape with the part axes elsewhere is accepted.
        let d = DynArray::<f64, 3>::from_shape([8, 2, 3]);
        assert!(Layout::split_along_axis(&a, 2, &d, 1, 0).is_ok());

        let empty = DynArray::<f64, 3>::from_shape([8, 0, 2]);
        let (dimension, reason) = array_invalid(Layout::split_along_axis(&empty, 2, &empty, 2, 0));
        assert_eq!(dimension, "logical axis 1");
        assert!(reason.contains("extent is 0"), "{reason}");

        // The checks of `Layout::new_split` name the logical axes: strides [2, 3] fail the
        // distinct-output rule (DC16) on logical axis 1.
        let odd = StridedDynArray::<f64, 3>::from_shape_and_stride([2, 2, 2], [2, 3, 8]);
        let (dimension, _) = array_invalid(Layout::split_along_axis_inplace(&odd, 2, 0));
        assert_eq!(dimension, "logical axis 1");
    }

    #[test]
    fn test_split_array_sides() {
        let input = DynArray::<f64, 3>::from_shape([8, 3, 2]);
        let output = DynArray::<f64, 3>::from_shape([2, 8, 3]);
        let layout = Layout::split_along_axis(&input, 2, &output, 0, 0).unwrap();
        let geometry = |shape: [usize; 3], stride: [usize; 3]| (shape.to_vec(), stride.to_vec());
        let (in_shape, in_stride) = geometry(input.shape(), input.stride());
        let (out_shape, out_stride) = geometry(output.shape(), output.stride());
        assert!(layout.fits_array_side(Side::Input, &in_shape, &in_stride));
        assert!(layout.fits_array_side(Side::Output, &out_shape, &out_stride));
        assert!(layout.fits_array_pair((&in_shape, &in_stride), (&out_shape, &out_stride)));
        // Each side has its own gap.
        assert!(!layout.fits_array_side(Side::Output, &in_shape, &in_stride));
        assert!(!layout.fits_array_side(Side::Input, &out_shape, &out_stride));

        // A wrong gap: planes of a box with one ring more.
        assert!(!layout.fits_array_side(Side::Input, &[8, 3, 2], &[1, 8, 32]));
        // A part extent of 3.
        assert!(!layout.fits_array_side(Side::Input, &[8, 3, 3], &[1, 8, 24]));
        // The same memory without the part axis, as a rank-2 array, has no part axis.
        assert!(!layout.fits_array_side(Side::Input, &[8, 3], &[1, 8]));

        // A transform of extent 2 with the stride of the gap: either axis can be the part
        // axis of an input, and the check accepts the array if one choice fits.
        let layout =
            Layout::new_split(&[dim(2, 3, 1)], &[], (3, 2), Placement::OutOfPlace).unwrap();
        assert!(layout.fits_array_side(Side::Input, &[2, 2], &[3, 3]));
        assert!(layout.fits_array_side(Side::Output, &[2, 2], &[1, 2]));
        assert!(layout.fits_array_side(Side::Output, &[2, 2], &[2, 1]));
        assert!(!layout.fits_array_side(Side::Output, &[2, 2], &[2, 2]));

        // Pairing without the part axes (DC15, DC27): column-major [4, 3, 2] planes fit each
        // side of a plan from [4, 2, 3] planes along logical axis 0, but their logical axes do
        // not pair as the layout's when the output's batch is transposed.
        let input = DynArray::<f64, 4>::from_shape([4, 2, 3, 2]);
        let output = StridedDynArray::<f64, 4>::from_shape_and_stride([4, 2, 3, 2], [1, 12, 4, 24]);
        let layout = Layout::split_along_axis(&input, 3, &output, 3, 0).unwrap();
        let other = (vec![4, 3, 2, 2], vec![1, 4, 12, 24]);
        assert!(layout.fits_array_side(Side::Input, &other.0, &other.1));
        assert!(layout.fits_array_side(Side::Output, &other.0, &other.1));
        assert!(!layout.fits_array_pair((&other.0, &other.1), (&other.0, &other.1)));
        let (in_shape, in_stride) = (input.shape().to_vec(), input.stride().to_vec());
        let (out_shape, out_stride) = (output.shape().to_vec(), output.stride().to_vec());
        assert!(layout.fits_array_pair((&in_shape, &in_stride), (&out_shape, &out_stride)));
    }

    // The storage of each side, and real-to-complex and complex-to-real layouts (task F5a).

    fn with_storage(
        dims: &[Dim],
        batch: &[Dim],
        storage: (Storage, Storage),
        placement: Placement,
    ) -> Layout {
        Layout::with_storage(dims, batch, storage, placement).unwrap()
    }

    fn storage_invalid(
        dims: &[Dim],
        batch: &[Dim],
        storage: (Storage, Storage),
        placement: Placement,
    ) -> (String, String) {
        array_invalid(Layout::with_storage(dims, batch, storage, placement))
    }

    const R2C: (Storage, Storage) = (Storage::Real, Storage::Complex);
    const C2R: (Storage, Storage) = (Storage::Complex, Storage::Real);
    const R2R: (Storage, Storage) = (Storage::Real, Storage::Real);

    #[test]
    fn test_real_layouts_of_the_brief() {
        let out = Placement::OutOfPlace;

        // Real [8, 3] into complex [5, 3] over axis 0.
        let real = DynArray::<f64, 2>::from_shape([8, 3]);
        let complex = DynArray::<c64, 2>::from_shape([5, 3]);
        let layout = Layout::over_axes(&real, &complex, &[0]).unwrap();
        assert_eq!(
            layout,
            with_storage(&[dim(8, 1, 1)], &[dim(3, 8, 5)], R2C, out)
        );
        assert_eq!(layout.storage(), R2C);
        assert_eq!((layout.input_span(), layout.output_span()), (24, 15));

        // The inverse, complex [5, 3] into real [8, 3].
        assert_eq!(
            Layout::along_axis(&complex, &real, 0).unwrap(),
            with_storage(&[dim(8, 1, 1)], &[dim(3, 5, 8)], C2R, out)
        );

        // Along axis 1 of real [6, 9] into complex [6, 5]: axis 1 is halved.
        let real = DynArray::<f64, 2>::from_shape([6, 9]);
        let complex = DynArray::<c64, 2>::from_shape([6, 5]);
        assert_eq!(
            Layout::along_axis(&real, &complex, 1).unwrap(),
            with_storage(&[dim(9, 6, 6)], &[dim(6, 1, 1)], R2C, out)
        );

        // Over both axes of real [4, 6] into complex [4, 4]: axis 1, listed last, is halved.
        let real = DynArray::<f32, 2>::from_shape([4, 6]);
        let complex = DynArray::<c32, 2>::from_shape([4, 4]);
        let layout = Layout::over_axes(&real, &complex, &[0, 1]).unwrap();
        assert_eq!(
            layout,
            with_storage(&[dim(4, 1, 1), dim(6, 4, 4)], &[], R2C, out)
        );
        assert_eq!(layout.logical_size(), 24);

        // In place, a complex [5, 3] array with n_last 8: columns of 10 reals.
        let complex = DynArray::<c64, 2>::from_shape([5, 3]);
        let layout = Layout::r2c_inplace(&complex, &[0], 8).unwrap();
        assert_eq!(
            layout,
            with_storage(&[dim(8, 1, 1)], &[dim(3, 10, 5)], R2C, Placement::InPlace)
        );
        assert_eq!((layout.input_span(), layout.output_span()), (28, 15));
        assert_eq!(
            Layout::c2r_inplace(&complex, &[0], 8).unwrap(),
            with_storage(&[dim(8, 1, 1)], &[dim(3, 5, 10)], C2R, Placement::InPlace)
        );

        // Two real arrays: a real-to-real layout.
        let real = DynArray::<f64, 2>::from_shape([8, 3]);
        let layout = Layout::along_axis(&real, &real, 0).unwrap();
        assert_eq!(
            layout,
            with_storage(&[dim(8, 1, 1)], &[dim(3, 8, 8)], R2R, out)
        );
        assert_eq!(Layout::along_axis_inplace(&real, 0).unwrap().storage(), R2R);

        // The split real-to-complex layout of the brief, from raw dimensions (its array
        // function follows in task F5c): real [8, 3] into split [5, 3, 2] with strides
        // [1, 5, 15].
        let layout = with_storage(
            &[dim(8, 1, 1)],
            &[dim(3, 8, 5)],
            (Storage::Real, Storage::Split { gap: 15 }),
            out,
        );
        assert_eq!((layout.input_span(), layout.output_span()), (24, 30));
        assert_eq!(layout.gaps(), None);
    }

    #[test]
    fn test_storages_tell_the_families_apart() {
        let dims = [dim(8, 1, 1)];
        let out = Placement::OutOfPlace;
        let c2c = Layout::new(&dims, &[], out).unwrap();
        let r2c = with_storage(&dims, &[], R2C, out);
        let c2r = with_storage(&dims, &[], C2R, out);
        let r2r = with_storage(&dims, &[], R2R, out);
        assert_eq!(c2c.storage(), (Storage::Complex, Storage::Complex));
        for (a, b) in [(&c2c, &r2c), (&r2c, &c2r), (&c2r, &r2r), (&r2r, &c2c)] {
            assert_ne!(a, b);
        }
        // The halved extent is on the complex side only: 5 = 8 / 2 + 1, and 5 for n = 9.
        assert_eq!((r2c.input_span(), r2c.output_span()), (8, 5));
        assert_eq!((c2r.input_span(), c2r.output_span()), (5, 8));
        assert_eq!((r2r.input_span(), r2r.output_span()), (8, 8));
        let odd = with_storage(&[dim(9, 1, 1)], &[], R2C, out);
        assert_eq!(odd.output_span(), 5);
        assert_eq!(odd.logical_size(), 9);
        // n = 1 and n = 2: one complex element and two.
        assert_eq!(
            with_storage(&[dim(1, 1, 1)], &[], R2C, out).output_span(),
            1
        );
        assert_eq!(with_storage(&[dim(2, 1, 1)], &[], C2R, out).input_span(), 2);
        // Only the last transform dimension is halved.
        let layout = with_storage(&[dim(4, 6, 5), dim(6, 1, 1)], &[], R2C, out);
        assert_eq!(layout.output_span(), 1 + 3 * 5 + 3);
        assert_eq!(layout.input_span(), 1 + 3 * 6 + 5);

        // A split side does not pair with an interleaved complex one.
        for storage in [
            (Storage::Split { gap: 8 }, Storage::Complex),
            (Storage::Complex, Storage::Split { gap: 8 }),
        ] {
            let (dimension, reason) = storage_invalid(&dims, &[], storage, out);
            assert_eq!(dimension, "layout");
            assert!(reason.contains("describe no transform"), "{reason}");
        }
        // `gaps` only when both sides are split.
        let split_r2c = with_storage(&dims, &[], (Storage::Real, Storage::Split { gap: 5 }), out);
        assert_eq!(split_r2c.gaps(), None);
        assert_eq!(split_r2c.output_span(), 10);
        assert!(split_r2c.describe().contains("split real-to-complex"));
        assert!(
            r2c.describe()
                .contains("real input and complex output storage")
        );
        assert!(!c2c.describe().contains("storage"));
    }

    #[test]
    fn test_with_storage_rules_in_place() {
        let in_place = Placement::InPlace;
        // The padded layout: the halved dimension contiguous, real strides twice the complex
        // ones elsewhere, on transform and batch dimensions.
        let layout = with_storage(
            &[dim(3, 20, 10), dim(8, 1, 1)],
            &[dim(2, 60, 30)],
            R2C,
            in_place,
        );
        assert_eq!(layout.output_span(), 1 + 2 * 10 + 4 + 30);
        with_storage(&[dim(3, 10, 20), dim(8, 1, 1)], &[], C2R, in_place);

        // The halved dimension must be contiguous on both sides.
        let (dimension, reason) = storage_invalid(&[dim(8, 2, 1)], &[], R2C, in_place);
        assert_eq!(dimension, "transform dimension 0");
        assert!(reason.contains("contiguous"), "{reason}");
        let (dimension, _) = storage_invalid(&[dim(8, 2, 2)], &[], C2R, in_place);
        assert_eq!(dimension, "transform dimension 0");
        // Elsewhere the real stride is twice the complex stride.
        let (dimension, reason) = storage_invalid(&[dim(8, 1, 1)], &[dim(3, 10, 6)], R2C, in_place);
        assert_eq!(dimension, "batch dimension 0");
        assert!(reason.contains("twice the complex stride 6"), "{reason}");
        let (dimension, _) = storage_invalid(&[dim(3, 10, 10), dim(8, 1, 1)], &[], C2R, in_place);
        assert_eq!(dimension, "transform dimension 0");
        // Dimensions of extent 1 are not checked, the halved one with n = 1 included.
        with_storage(&[dim(1, 7, 3)], &[dim(4, 2, 1)], R2C, in_place);
        with_storage(&[dim(8, 1, 1)], &[dim(1, 3, 3)], R2C, in_place);

        // The complex outputs must be distinct: complex stride 4 is below n / 2 + 1 = 5.
        let (dimension, reason) = storage_invalid(&[dim(8, 1, 1)], &[dim(3, 8, 4)], R2C, in_place);
        assert_eq!(dimension, "batch dimension 0");
        assert!(reason.contains("output stride 4 is below 5"), "{reason}");
        // In place the input is written too: a complex-to-real layout whose real outputs are
        // distinct but whose complex inputs overlap.
        let (dimension, reason) = storage_invalid(&[dim(8, 1, 1)], &[dim(3, 4, 8)], C2R, in_place);
        assert_eq!(dimension, "batch dimension 0");
        assert!(reason.contains("input stride 4 is below 5"), "{reason}");
        // Out of place the same input is accepted, as inputs may repeat elements.
        with_storage(&[dim(8, 1, 1)], &[dim(3, 4, 8)], C2R, Placement::OutOfPlace);

        // Real-to-real in place needs equal strides, as complex-to-complex does.
        let (dimension, _) = storage_invalid(&[dim(8, 1, 2)], &[], R2R, in_place);
        assert_eq!(dimension, "transform dimension 0");
        with_storage(&[dim(8, 2, 2)], &[], R2R, in_place);

        // Split real-to-complex and complex-to-real layouts are never in place.
        for storage in [
            (Storage::Real, Storage::Split { gap: 5 }),
            (Storage::Split { gap: 5 }, Storage::Real),
        ] {
            let (dimension, reason) = storage_invalid(&[dim(8, 1, 1)], &[], storage, in_place);
            assert_eq!(dimension, "layout");
            assert!(reason.contains("cannot be in place"), "{reason}");
        }
    }

    #[test]
    fn test_with_storage_rules_out_of_place() {
        let out = Placement::OutOfPlace;
        // The halved extent in the distinct-output rule: a complex output stride of 5 is
        // enough for n = 8 and n = 9, not for n = 10.
        with_storage(&[dim(8, 1, 1)], &[dim(3, 8, 5)], R2C, out);
        with_storage(&[dim(9, 1, 1)], &[dim(3, 9, 5)], R2C, out);
        let (dimension, _) = storage_invalid(&[dim(10, 1, 1)], &[dim(3, 10, 5)], R2C, out);
        assert_eq!(dimension, "batch dimension 0");
        // On a real output the full extent counts.
        let (dimension, _) = storage_invalid(&[dim(8, 1, 1)], &[dim(3, 5, 5)], C2R, out);
        assert_eq!(dimension, "batch dimension 0");

        // A split output's part axis takes part in the rule: a gap of 4 is below the 5
        // complex elements.
        let (dimension, _) = storage_invalid(
            &[dim(8, 1, 1)],
            &[],
            (Storage::Real, Storage::Split { gap: 4 }),
            out,
        );
        assert_eq!(dimension, "part axis");
        // A positive gap.
        let (dimension, reason) = storage_invalid(
            &[dim(8, 1, 1)],
            &[],
            (Storage::Split { gap: 0 }, Storage::Real),
            out,
        );
        assert_eq!(dimension, "part axis");
        assert!(reason.contains("input gap 0"), "{reason}");

        // The rules of `Layout::new` apply to every family.
        let (dimension, _) = storage_invalid(&[], &[], R2C, out);
        assert_eq!(dimension, "layout");
        let (dimension, _) = storage_invalid(&[dim(0, 1, 1)], &[], C2R, out);
        assert_eq!(dimension, "transform dimension 0");
        let (dimension, _) = storage_invalid(&[dim(4, -1, 1)], &[], R2R, out);
        assert_eq!(dimension, "transform dimension 0");
        let (dimension, reason) = storage_invalid(&[dim(4, isize::MAX / 2, 1)], &[], R2C, out);
        assert_eq!(dimension, "layout");
        assert!(reason.contains("input span"), "{reason}");
    }

    #[test]
    fn test_real_array_layout_errors() {
        let real = DynArray::<f64, 2>::from_shape([8, 3]);

        // A wrong extent on the halved axis names it, the extent expected and the one found.
        let wrong = DynArray::<c64, 2>::from_shape([4, 3]);
        let (dimension, reason) = array_invalid(Layout::along_axis(&real, &wrong, 0));
        assert_eq!(dimension, "axis 0");
        assert!(reason.contains("must be 5"), "{reason}");
        assert!(reason.contains("but is 4"), "{reason}");
        let (dimension, reason) = array_invalid(Layout::along_axis(&wrong, &real, 0));
        assert_eq!(dimension, "axis 0");
        assert!(
            reason.contains("input extent along the halved axis"),
            "{reason}"
        );

        // The other axes need equal extents.
        let wrong = DynArray::<c64, 2>::from_shape([5, 4]);
        let (dimension, reason) = array_invalid(Layout::along_axis(&real, &wrong, 0));
        assert_eq!(dimension, "axis 1");
        assert!(reason.contains("input extent 3"), "{reason}");
        assert!(reason.contains("real-to-complex"), "{reason}");

        // The halved axis is the last entry of `axes`, whatever the memory order (DC2).
        let real = DynArray::<f64, 2>::from_shape([4, 6]);
        let complex = DynArray::<c64, 2>::from_shape([4, 4]);
        let (dimension, reason) = array_invalid(Layout::over_axes(&real, &complex, &[1, 0]));
        assert_eq!(dimension, "axis 0");
        assert!(reason.contains("must be 3"), "{reason}");
        let complex = DynArray::<c64, 2>::from_shape([3, 6]);
        assert!(Layout::over_axes(&real, &complex, &[1, 0]).is_ok());

        // An extent 0 on the real side.
        let empty = DynArray::<f64, 2>::from_shape([0, 3]);
        let one = DynArray::<c64, 2>::from_shape([1, 3]);
        let (dimension, reason) = array_invalid(Layout::along_axis(&empty, &one, 0));
        assert_eq!(dimension, "axis 0");
        assert!(reason.contains("extent is 0"), "{reason}");

        // The checks of `axes` come first.
        let (dimension, _) = array_invalid(Layout::over_axes(&real, &complex, &[]));
        assert_eq!(dimension, "axes");
    }

    #[test]
    fn test_padded_in_place_errors() {
        let complex = DynArray::<c64, 2>::from_shape([5, 3]);
        // n_last must give the complex extent.
        for n_last in [7, 10] {
            let (dimension, reason) = array_invalid(Layout::r2c_inplace(&complex, &[0], n_last));
            assert_eq!(dimension, "axis 0");
            assert!(reason.contains(&format!("n_last = {n_last}")), "{reason}");
        }
        assert!(Layout::r2c_inplace(&complex, &[0], 9).is_ok());
        let (dimension, _) = array_invalid(Layout::c2r_inplace(&complex, &[0], 4));
        assert_eq!(dimension, "axis 0");

        // The halved axis must be contiguous: along axis 0 of a row-major array.
        let row_major = StridedDynArray::<c64, 2>::row_major([5, 3]);
        let (dimension, reason) = array_invalid(Layout::r2c_inplace(&row_major, &[0], 8));
        assert_eq!(dimension, "axis 0");
        assert!(reason.contains("contiguous"), "{reason}");
        // Along axis 1 of the same array, which is contiguous: real stride 6 on axis 0.
        let layout = Layout::c2r_inplace(&row_major, &[1], 4).unwrap();
        assert_eq!(layout.dims(), &[dim(4, 1, 1)]);
        assert_eq!(layout.batch(), &[dim(5, 3, 6)]);

        // The checks of `axes` come first.
        let (dimension, _) = array_invalid(Layout::r2c_inplace(&complex, &[], 8));
        assert_eq!(dimension, "axes");
        let (dimension, _) = array_invalid(Layout::r2c_inplace(&complex, &[0, 0], 8));
        assert_eq!(dimension, "axis 0");
        let (dimension, _) = array_invalid(Layout::r2c_inplace(&complex, &[2], 8));
        assert_eq!(dimension, "axes");

        // A real stride twice a complex stride that does not fit in `usize`.
        let values = [c64::new(0.0, 0.0); 2];
        let huge = StridedSliceArray::from_shape_and_stride(&values, [2, 1], [1, usize::MAX]);
        let (dimension, reason) = array_invalid(Layout::r2c_inplace(&huge, &[0], 2));
        assert_eq!(dimension, "axis 1");
        assert!(reason.contains("does not fit"), "{reason}");
    }

    #[test]
    fn test_real_array_sides() {
        let input = DynArray::<f64, 3>::from_shape([8, 2, 3]);
        let output = StridedDynArray::<c64, 3>::from_shape_and_stride([5, 2, 3], [1, 15, 5]);
        let layout = Layout::along_axis(&input, &output, 0).unwrap();
        let geometry = |shape: [usize; 3], stride: [usize; 3]| (shape.to_vec(), stride.to_vec());
        let (in_shape, in_stride) = geometry(input.shape(), input.stride());
        let (out_shape, out_stride) = geometry(output.shape(), output.stride());
        assert!(layout.fits_array_side(Side::Input, &in_shape, &in_stride));
        assert!(layout.fits_array_side(Side::Output, &out_shape, &out_stride));
        assert!(layout.fits_array_pair((&in_shape, &in_stride), (&out_shape, &out_stride)));
        // The halved extent is on the output side only.
        assert!(!layout.fits_array_side(Side::Output, &[8, 2, 3], &[1, 15, 5]));
        assert!(!layout.fits_array_side(Side::Input, &[5, 2, 3], &[1, 5, 10]));
        // Each side fits arrays of shape [n, 3, 2], whose axes do not pair as the layout's.
        let (other_in, other_out) = (
            (vec![8, 3, 2], vec![1, 8, 24]),
            (vec![5, 3, 2], vec![1, 5, 15]),
        );
        assert!(layout.fits_array_side(Side::Input, &other_in.0, &other_in.1));
        assert!(layout.fits_array_side(Side::Output, &other_out.0, &other_out.1));
        assert!(!layout.fits_array_pair((&other_in.0, &other_in.1), (&other_out.0, &other_out.1)));
        // Pairing needs equal extents off the halved axis.
        assert!(!layout.fits_array_pair((&in_shape, &in_stride), (&[5, 3, 2], &[1, 5, 15])));

        // n = 1: one real and one complex element per transform, matched as extent 1.
        let input = DynArray::<f64, 2>::from_shape([1, 4]);
        let output = DynArray::<c64, 2>::from_shape([1, 4]);
        let layout = Layout::along_axis(&input, &output, 0).unwrap();
        assert!(layout.fits_array_pair((&[1, 4], &[1, 1]), (&[1, 4], &[1, 1])));

        // In place, the buffer is the complex side.
        let data = DynArray::<c64, 2>::from_shape([5, 3]);
        let layout = Layout::r2c_inplace(&data, &[0], 8).unwrap();
        assert!(layout.fits_array_side(Side::Output, &[5, 3], &[1, 5]));
        assert!(layout.fits_array_side(Side::Input, &[8, 3], &[1, 10]));
    }

    /// `normalise_batch` as it was before it worked in place (issue #44), the reference for
    /// [`test_normalise_in_place`].
    fn normalise_reference(batch: &[Dim]) -> Vec<Dim> {
        let mut sorted: Vec<Dim> = batch.iter().copied().filter(|dim| dim.n > 1).collect();
        sorted.sort_by_key(|dim| std::cmp::Reverse((dim.is, dim.os)));
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

    #[test]
    fn test_normalise_in_place() {
        // Every batch of up to three dimensions from these values, including extent 1,
        // stride 0 and equal strides, whose order decides some merges.
        let mut values = Vec::new();
        for n in [1, 2, 3] {
            for is in [0, 1, 2, 3, 6] {
                for os in [0, 1, 2, 6] {
                    values.push(dim(n, is, os));
                }
            }
        }
        let mut batches: Vec<Vec<Dim>> = vec![vec![]];
        for _ in 0..3 {
            let longer: Vec<Vec<Dim>> = batches
                .iter()
                .filter(|batch| batch.len() == batches.last().unwrap().len())
                .flat_map(|batch| {
                    values.iter().map(move |&value| {
                        let mut batch = batch.clone();
                        batch.push(value);
                        batch
                    })
                })
                .collect();
            batches.extend(longer);
        }
        assert_eq!(batches.len(), 1 + 60 + 60 * 60 + 60 * 60 * 60);
        for batch in &batches {
            assert_eq!(
                super::normalise_batch(batch),
                normalise_reference(batch),
                "{batch:?}"
            );
        }
        // Equal strides: (2, 1) and (3, 1) do not merge with each other, and which of them
        // merges with (2, 2) depends on their order.
        let batch = [dim(2, 1, 1), dim(3, 1, 1), dim(2, 2, 2)];
        assert_eq!(super::normalise_batch(&batch), normalise_reference(&batch));
    }

    #[test]
    fn test_fits_above_inline_rank() {
        // Rank 18, above the stack scratch of the execution checks, with 16 axes of extent 1.
        let mut shape = [1; 18];
        shape[0] = 4;
        shape[17] = 3;
        let input = DynArray::<c64, 18>::from_shape(shape);
        let output = DynArray::<c64, 18>::from_shape(shape);
        let layout = Layout::along_axis(&input, &output, 0).unwrap();
        assert_eq!(layout.dims(), &[dim(4, 1, 1)]);
        assert_eq!(layout.batch(), &[dim(3, 4, 4)]);
        let stride = input.stride();
        assert!(layout.fits_array_side(Side::Input, &shape, &stride));
        assert!(layout.fits_array_side(Side::Output, &shape, &stride));
        assert!(layout.fits_array_pair((&shape, &stride), (&shape, &stride)));

        // The batch axis moved to another position still fits each side, and pairs.
        let mut moved = [1; 18];
        moved[0] = 4;
        moved[9] = 3;
        let moved_stride = DynArray::<c64, 18>::from_shape(moved).stride();
        assert!(layout.fits_array_side(Side::Input, &moved, &moved_stride));
        assert!(layout.fits_array_pair((&moved, &moved_stride), (&moved, &moved_stride)));

        // A batch of 2 does not fit.
        let mut other = shape;
        other[17] = 2;
        let other_stride = DynArray::<c64, 18>::from_shape(other).stride();
        assert!(!layout.fits_array_side(Side::Input, &other, &other_stride));
        assert!(!layout.fits_array_pair((&other, &other_stride), (&other, &other_stride)));
    }
}
