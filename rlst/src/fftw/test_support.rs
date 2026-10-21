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

/// The constant c of the c2c tolerance c · ε · (1 + log₂ N), fixed once for the family and
/// not tuned per size (design Section 7.1).
pub(crate) const C2C_TOLERANCE_FACTOR: f64 = 4.0;

/// The fixed seed of the test inputs.
pub(crate) const SEED: u64 = 0x5eed_0f_ff7e;

/// A precision under test.
pub(crate) trait TestReal: FftwReal {
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
fn dft(line: &mut [Complex<f64>], direction: Direction) {
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
