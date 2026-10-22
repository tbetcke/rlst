//! The reference of design Section 7.1 and helpers for the FFTW tests.

use std::f64::consts::PI;

use aligned_vec::{AVec, ConstAlign};
use num::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::align::PLAN_ALIGNMENT;
use super::kind::Direction;
use super::layout::{Dim, Layout, Placement};
use super::real::FftwReal;
use crate::dense::array::AlignedDynArray;
use crate::dense::layout::{convert_1d_nd_from_shape, convert_nd_raw};
use crate::{
    Array, BaseItem, DynArray, FFTW_ALIGNED, Shape, Stride, StridedRawAccess,
    UnsafeRandomAccessByRef, UnsafeRandomAccessMut,
};

/// The constant c of the c2c tolerance c · ε · (1 + log₂ N), fixed once for the family and
/// not tuned per size (design Section 7.1).
pub(crate) const C2C_TOLERANCE_FACTOR: f64 = 4.0;

/// The fixed seed of the test inputs.
pub(crate) const SEED: u64 = 0x5e_ed0f_ff7e;

/// A precision under test.
pub(crate) trait TestReal: FftwReal + TestElement {
    /// The machine epsilon.
    const EPSILON: f64;
    /// The name of the precision, for messages.
    const NAME: &'static str;
    /// Round an `f64` to this precision.
    fn round_from(value: f64) -> Self;
    /// Widen to `f64`.
    fn to_f64(self) -> f64;
}

impl TestReal for f32 {
    const EPSILON: f64 = f32::EPSILON as f64;
    const NAME: &'static str = "f32";
    fn round_from(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl TestReal for f64 {
    const EPSILON: f64 = f64::EPSILON;
    const NAME: &'static str = "f64";
    fn round_from(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// The c2c tolerance in the relative L2 norm for a transform of logical size `n`.
pub(crate) fn c2c_tolerance<T: TestReal>(n: usize) -> f64 {
    C2C_TOLERANCE_FACTOR * T::EPSILON * (1.0 + (n as f64).log2())
}

/// `len` complex values with parts uniform in [−1, 1], from ChaCha8 seeded with `seed`.
pub(crate) fn random_values<T: TestReal>(seed: u64, len: usize) -> Vec<Complex<T>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| {
            let re = rng.random_range(-1.0..=1.0);
            let im = rng.random_range(-1.0..=1.0);
            Complex::new(T::round_from(re), T::round_from(im))
        })
        .collect()
}

/// A buffer at alignment class 0 holding `values`.
pub(crate) fn aligned<T: Copy>(values: &[T]) -> AVec<T, ConstAlign<PLAN_ALIGNMENT>> {
    AVec::from_slice(PLAN_ALIGNMENT, values)
}

/// A buffer at alignment class 0 of `len` copies of `value`.
pub(crate) fn filled<T: Copy>(value: T, len: usize) -> AVec<T, ConstAlign<PLAN_ALIGNMENT>> {
    AVec::from_iter(PLAN_ALIGNMENT, std::iter::repeat_n(value, len))
}

/// Widen complex values to `f64`.
pub(crate) fn widen<T: TestReal>(values: &[Complex<T>]) -> Vec<Complex<f64>> {
    values
        .iter()
        .map(|value| Complex::new(value.re.to_f64(), value.im.to_f64()))
        .collect()
}

/// The element offsets, on the input or output side, of every transform of a layout: one
/// list per batch index, each in row-major order of the transform dimensions.
pub(crate) fn transform_offsets(layout: &Layout, input: bool) -> Vec<Vec<usize>> {
    let stride = |dim: &Dim| if input { dim.is } else { dim.os } as usize;
    let offsets = |dims: &[Dim]| {
        let mut offsets = vec![0_usize];
        for dim in dims {
            offsets = offsets
                .iter()
                .flat_map(|&offset| (0..dim.n as usize).map(move |i| offset + i * stride(dim)))
                .collect();
        }
        offsets
    };
    let block = offsets(layout.dims());
    offsets(layout.batch())
        .into_iter()
        .map(|base| block.iter().map(|offset| base + offset).collect())
        .collect()
}

/// Compensated (Neumaier) summation.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) {
        let sum = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - sum) + value;
        } else {
            self.compensation += (value - sum) + self.sum;
        }
        self.sum = sum;
    }

    fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// An O(n²) DFT of `line` in place, with twiddles from the reduced index `(j · k) mod n`
/// and compensated summation of real and imaginary parts.
pub(crate) fn dft(line: &mut [Complex<f64>], direction: Direction) {
    let n = line.len();
    let sign = match direction {
        Direction::Forward => -1.0,
        Direction::Backward => 1.0,
    };
    let twiddles: Vec<(f64, f64)> = (0..n)
        .map(|r| {
            let (sin, cos) = (2.0 * PI * r as f64 / n as f64).sin_cos();
            (cos, sign * sin)
        })
        .collect();
    let result: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let mut re = Neumaier::default();
            let mut im = Neumaier::default();
            for (j, x) in line.iter().enumerate() {
                let (cos, sin) = twiddles[(j * k) % n];
                re.add(x.re * cos);
                re.add(-x.im * sin);
                im.add(x.re * sin);
                im.add(x.im * cos);
            }
            Complex::new(re.value(), im.value())
        })
        .collect();
    line.copy_from_slice(&result);
}

/// The reference c2c transform of `input` (in `f64`) for `layout`: one vector per batch
/// index, in the order of [`transform_offsets`], applying the DFT along each transform
/// dimension in turn.
pub(crate) fn reference_c2c(
    layout: &Layout,
    direction: Direction,
    input: &[Complex<f64>],
) -> Vec<Vec<Complex<f64>>> {
    let extents: Vec<usize> = layout.dims().iter().map(|dim| dim.n as usize).collect();
    transform_offsets(layout, true)
        .into_iter()
        .map(|offsets| {
            let mut block: Vec<Complex<f64>> =
                offsets.iter().map(|&offset| input[offset]).collect();
            for (axis, &n) in extents.iter().enumerate() {
                let inner: usize = extents[axis + 1..].iter().product();
                let outer: usize = extents[..axis].iter().product();
                let mut line = vec![Complex::new(0.0, 0.0); n];
                for o in 0..outer {
                    for i in 0..inner {
                        let base = o * n * inner + i;
                        for (j, value) in line.iter_mut().enumerate() {
                            *value = block[base + j * inner];
                        }
                        dft(&mut line, direction);
                        for (j, value) in line.iter().enumerate() {
                            block[base + j * inner] = *value;
                        }
                    }
                }
            }
            block
        })
        .collect()
}

/// The relative L2 error of `actual` against `expected`.
pub(crate) fn relative_error(actual: &[Complex<f64>], expected: &[Complex<f64>]) -> f64 {
    let difference: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).norm_sqr())
        .sum();
    let norm: f64 = expected.iter().map(Complex::norm_sqr).sum();
    if norm == 0.0 {
        difference.sqrt()
    } else {
        (difference / norm).sqrt()
    }
}

/// The layouts of the fixed cells (design Section 7.2, raw layouts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayoutKind {
    /// One contiguous transform.
    Contiguous,
    /// A contiguous batch of 3.
    Batch3,
    /// A batch of 3 with stride 1, and a transform stride equal to the batch count.
    BatchInner,
    /// One transform with stride 2.
    Strided,
    /// A rank-2 transform, `RANK2_EXTENT` × n, row-major.
    Rank2,
    /// A contiguous batch of 2 rank-2 transforms.
    BatchRank2,
}

/// The extent of the first dimension of the rank-2 layouts.
pub(crate) const RANK2_EXTENT: isize = 6;

impl LayoutKind {
    pub(crate) const ALL: [LayoutKind; 6] = [
        LayoutKind::Contiguous,
        LayoutKind::Batch3,
        LayoutKind::BatchInner,
        LayoutKind::Strided,
        LayoutKind::Rank2,
        LayoutKind::BatchRank2,
    ];

    /// The layout for transforms of size `n`.
    pub(crate) fn layout(self, n: usize, placement: Placement) -> Layout {
        let n = n as isize;
        let d = |n, s| Dim { n, is: s, os: s };
        let m = RANK2_EXTENT;
        let (dims, batch) = match self {
            LayoutKind::Contiguous => (vec![d(n, 1)], vec![]),
            LayoutKind::Batch3 => (vec![d(n, 1)], vec![d(3, n)]),
            LayoutKind::BatchInner => (vec![d(n, 3)], vec![d(3, 1)]),
            LayoutKind::Strided => (vec![d(n, 2)], vec![]),
            LayoutKind::Rank2 => (vec![d(m, n), d(n, 1)], vec![]),
            LayoutKind::BatchRank2 => (vec![d(m, n), d(n, 1)], vec![d(2, m * n)]),
        };
        Layout::new(&dims, &batch, placement).unwrap()
    }
}

/// The input policies of the c2c tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolicyKind {
    Preserving,
    Destroying,
    InPlace,
}

impl PolicyKind {
    pub(crate) const ALL: [PolicyKind; 3] = [
        PolicyKind::Preserving,
        PolicyKind::Destroying,
        PolicyKind::InPlace,
    ];

    pub(crate) fn placement(self) -> Placement {
        match self {
            PolicyKind::InPlace => Placement::InPlace,
            _ => Placement::OutOfPlace,
        }
    }
}

/// The sizes of the fixed cells: 1, 2, 3, the primes 17, 97 and 1009, the powers of two up to
/// 4096, and nd-fmm's grids 140, 216, 336, 588 and 1080.
pub(crate) fn fixed_sizes() -> Vec<usize> {
    let mut sizes = vec![1, 2, 3, 17, 97, 1009, 140, 216, 336, 588, 1080];
    sizes.extend((0..=12).map(|k| 1 << k));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The layouts of a size: all of them below 1009, the rank-1 layout with a batch of 1 from
/// 1009 on, which keeps the O(n²) reference fast in debug builds.
pub(crate) fn layouts_for(n: usize) -> &'static [LayoutKind] {
    if n >= 1009 {
        &[LayoutKind::Contiguous]
    } else {
        &LayoutKind::ALL
    }
}

// Helpers for the tests on RLST arrays and views (task F2).

/// The multi-indices of `shape`, in column-major order.
pub(crate) fn multi_indices<const N: usize>(shape: [usize; N]) -> impl Iterator<Item = [usize; N]> {
    (0..shape.iter().product()).map(move |index| convert_1d_nd_from_shape(index, shape))
}

/// The sentinel of the array tests: finite, compared by its bits, and far above every value a
/// transform in the tests produces, whose parts are bounded by the logical size times √2.
pub(crate) fn sentinel<T: TestReal>() -> Complex<T> {
    Complex::new(T::round_from(1234.5), T::round_from(-678.25))
}

/// A column-major parent of `shape` at `FFTW_ALIGNED`, filled with the sentinel.
pub(crate) fn sentinel_parent<T: TestReal, const N: usize>(
    shape: [usize; N],
) -> AlignedDynArray<Complex<T>, N, FFTW_ALIGNED> {
    let mut parent = AlignedDynArray::<Complex<T>, N, FFTW_ALIGNED>::from_shape(shape);
    parent.fill_with_value(sentinel::<T>());
    parent
}

/// Write random values from `seed` into the elements of `view` by multi-index, and return
/// them in column-major order of the view.
pub(crate) fn fill_view<T, A, const N: usize>(view: &mut Array<A, N>, seed: u64) -> Vec<Complex<T>>
where
    T: TestReal,
    A: BaseItem<Item = Complex<T>> + UnsafeRandomAccessMut<N> + Shape<N>,
{
    let values = random_values::<T>(seed, view.shape().iter().product());
    for (index, &value) in multi_indices(view.shape()).zip(&values) {
        *view.get_mut(index).unwrap() = value;
    }
    values
}

/// An element of the buffers of the tests: a complex value, or a real for split storage.
pub(crate) trait TestElement: Copy + std::fmt::Debug {
    /// The name of the precision, for messages.
    const PRECISION: &'static str;
    /// The sentinel of the gaps between and around views.
    fn sentinel() -> Self;
    /// The bits of the element: of the real and imaginary parts, of a real and 0.
    fn bits(self) -> (u64, u64);
}

impl<T: TestReal> TestElement for Complex<T> {
    const PRECISION: &'static str = T::NAME;
    fn sentinel() -> Self {
        sentinel::<T>()
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_f64().to_bits(), self.im.to_f64().to_bits())
    }
}

macro_rules! impl_test_element_for_real {
    ($real:ty) => {
        impl TestElement for $real {
            const PRECISION: &'static str = <$real as TestReal>::NAME;
            fn sentinel() -> Self {
                real_sentinel::<$real>()
            }
            fn bits(self) -> (u64, u64) {
                (self.to_f64().to_bits(), 0)
            }
        }
    };
}

impl_test_element_for_real!(f32);
impl_test_element_for_real!(f64);

/// The bits of every element of the storage of `view`, the elements of other views included.
pub(crate) fn storage_bits<E, A, const N: usize>(view: &Array<A, N>) -> Vec<(u64, u64)>
where
    E: TestElement,
    A: BaseItem<Item = E> + StridedRawAccess,
{
    let (storage, _) = view.storage().unwrap();
    storage.iter().map(|&value| value.bits()).collect()
}

/// Assert that every element of the storage of `view` outside the view is still the
/// sentinel, bit for bit. The view's elements are found from its storage offset and strides.
pub(crate) fn assert_gaps_unchanged<E, A, const N: usize>(
    view: &Array<A, N>,
    context: &dyn std::fmt::Display,
) where
    E: TestElement,
    A: BaseItem<Item = E> + StridedRawAccess + Shape<N> + Stride<N>,
{
    let (storage, offset) = view.storage().unwrap();
    let stride = view.stride();
    let mut own = vec![false; storage.len()];
    for index in multi_indices(view.shape()) {
        own[offset + convert_nd_raw(index, stride)] = true;
    }
    let sentinel = E::sentinel().bits();
    for (position, (&value, &own)) in storage.iter().zip(&own).enumerate() {
        assert!(
            own || value.bits() == sentinel,
            "{} {context}: storage element {position}, outside the view, changed to {value:?}",
            E::PRECISION
        );
    }
}

/// The logical reference (design Section 7.1, F2 brief work item 7): the values of a view of
/// `shape`, given in column-major order of the view, widened to f64 and transformed by the
/// DFT along each axis of `axes`. It reads values by multi-index and builds no `Layout`.
pub(crate) fn logical_reference<T: TestReal, const N: usize>(
    shape: [usize; N],
    values: &[Complex<T>],
    axes: &[usize],
    direction: Direction,
) -> DynArray<Complex<f64>, N> {
    let mut reference = DynArray::<Complex<f64>, N>::from_shape(shape);
    for (index, value) in multi_indices(shape).zip(widen(values)) {
        reference[index] = value;
    }
    for &axis in axes {
        let mut line = vec![Complex::new(0.0, 0.0); shape[axis]];
        for start in multi_indices(shape).filter(|index| index[axis] == 0) {
            let at = |j: usize| {
                let mut index = start;
                index[axis] = j;
                index
            };
            for (j, value) in line.iter_mut().enumerate() {
                *value = reference[at(j)];
            }
            dft(&mut line, direction);
            for (j, value) in line.iter().enumerate() {
                reference[at(j)] = *value;
            }
        }
    }
    reference
}

/// Assert that `view` equals `expected` by multi-index, to the c2c tolerance of transforms
/// over `axes`, in the relative L2 norm of each transform: the elements that share their
/// indices off `axes`.
pub(crate) fn assert_matches_reference<T, A, const N: usize>(
    view: &Array<A, N>,
    expected: &DynArray<Complex<f64>, N>,
    axes: &[usize],
    context: &dyn std::fmt::Display,
) where
    T: TestReal,
    A: BaseItem<Item = Complex<T>> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    let shape = view.shape();
    let logical_size: usize = axes.iter().map(|&axis| shape[axis]).product();
    let tolerance = c2c_tolerance::<T>(logical_size);
    let mut transforms = std::collections::BTreeMap::<Vec<usize>, (Vec<_>, Vec<_>)>::new();
    for index in multi_indices(shape) {
        let batch: Vec<usize> = (0..N)
            .map(|axis| if axes.contains(&axis) { 0 } else { index[axis] })
            .collect();
        let (actual, wanted) = transforms.entry(batch).or_default();
        actual.extend(widen(&[*view.get(index).unwrap()]));
        wanted.push(expected[index]);
    }
    for (batch, (actual, wanted)) in transforms {
        let error = relative_error(&actual, &wanted);
        assert!(
            error <= tolerance,
            "{} {context}, transform at {batch:?}: relative error {error:e} above {tolerance:e}",
            T::NAME
        );
    }
}

// Helpers for split storage (task F3): a split view is a real view with a part axis of
// extent 2, the real part at index 0 and the imaginary part at index 1.

/// The real sentinel of the split tests, compared by its bits.
pub(crate) fn real_sentinel<T: TestReal>() -> T {
    T::round_from(1234.5)
}

/// A column-major real parent of `shape` at `FFTW_ALIGNED`, filled with the real sentinel.
pub(crate) fn real_sentinel_parent<T: TestReal, const N: usize>(
    shape: [usize; N],
) -> AlignedDynArray<T, N, FFTW_ALIGNED> {
    let mut parent = AlignedDynArray::<T, N, FFTW_ALIGNED>::from_shape(shape);
    parent.fill_with_value(real_sentinel::<T>());
    parent
}

/// `values` with `value` inserted at `position`: a logical multi-index or shape with its part
/// axis.
pub(crate) fn insert_axis<const M: usize, const N: usize>(
    values: [usize; M],
    position: usize,
    value: usize,
) -> [usize; N] {
    assert_eq!(N, M + 1);
    let mut result = [value; N];
    result[..position].copy_from_slice(&values[..position]);
    result[position + 1..].copy_from_slice(&values[position..]);
    result
}

/// Write random complex values from `seed` into the two parts of the split view `view` with
/// part axis `part`, by logical multi-index, and return them in column-major order of the
/// logical shape `shape`.
pub(crate) fn fill_split_view<T, A, const M: usize, const N: usize>(
    view: &mut Array<A, N>,
    part: usize,
    shape: [usize; M],
    seed: u64,
) -> Vec<Complex<T>>
where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessMut<N> + Shape<N>,
{
    assert_eq!(view.shape(), insert_axis(shape, part, 2));
    let values = random_values::<T>(seed, shape.iter().product());
    for (index, value) in multi_indices(shape).zip(&values) {
        *view.get_mut(insert_axis(index, part, 0)).unwrap() = value.re;
        *view.get_mut(insert_axis(index, part, 1)).unwrap() = value.im;
    }
    values
}

/// The complex values of the split view `view` with part axis `part`, read by logical
/// multi-index from its two parts.
pub(crate) fn split_values<T, A, const M: usize, const N: usize>(
    view: &Array<A, N>,
    part: usize,
    shape: [usize; M],
) -> DynArray<Complex<T>, M>
where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    assert_eq!(view.shape(), insert_axis(shape, part, 2));
    let mut values = DynArray::<Complex<T>, M>::from_shape(shape);
    for index in multi_indices(shape) {
        values[index] = Complex::new(
            *view.get(insert_axis(index, part, 0)).unwrap(),
            *view.get(insert_axis(index, part, 1)).unwrap(),
        );
    }
    values
}
