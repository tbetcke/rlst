//! Split real-to-complex and complex-to-real transforms (task F5c; design Sections 7.1 to 7.4).
//!
//! Every plan here is made with `Rigor::Estimate`, depends on no wisdom and assumes no SIMD.
//! The split side has its part axis outermost (planes) or innermost (the memory of
//! interleaved complex values), and its results are compared with the references of
//! `test_support` and with the interleaved plan of the same values, to round-off (design
//! Section 7.3): split plans can choose other codelets than interleaved ones.

use std::fmt;

use num::Complex;

use crate::fftw::test_support::{
    PolicyKind, SEED, TestElement, TestReal, aligned, assert_transforms_close, filled,
    halved_shape, insert_axis, multi_indices, random_reals, real_sentinel, real_sentinel_parent,
    reference_c2r, reference_r2c, sentinel_parent, split_values, storage_bits, widen,
};
use crate::fftw::{
    AlignmentClass, C2r, Destroying, Dim, FftwError, Layout, Placement, Plan, Planner, Preserving,
    Side, SplitC2r, SplitR2c, Storage,
};
use crate::{Array, BaseItem, DynArray, Shape, StridedDynArray, UnsafeRandomAccessByRef};

/// The constant c of the tolerances c · ε · (1 + log₂ N) of these families, the r2c and c2r
/// constants of F5a.
const TOLERANCE_FACTOR: f64 = 4.0;

/// The tolerance for the logical size `n`.
fn tolerance<T: TestReal>(n: usize) -> f64 {
    TOLERANCE_FACTOR * T::EPSILON * (1.0 + (n as f64).log2())
}

/// The inputs and references of one real shape and its axes in one precision.
struct Case<T, const N: usize> {
    shape: [usize; N],
    axes: Vec<usize>,
    seed: u64,
    /// The input of r2c, in column-major order of `shape`.
    reals: Vec<T>,
    /// The r2c reference of `reals`.
    spectrum: DynArray<Complex<f64>, N>,
    /// The input of c2r: `spectrum` rounded to `T`, in column-major order of the complex
    /// shape.
    half: Vec<Complex<T>>,
    /// The c2r reference of `half`.
    inverse: DynArray<f64, N>,
}

impl<T: TestReal, const N: usize> Case<T, N> {
    fn new(shape: [usize; N], axes: &[usize]) -> Self {
        let seed = axes
            .iter()
            .chain(&shape)
            .fold(SEED ^ 0xf5c, |code, &value| {
                code.wrapping_mul(31).wrapping_add(value as u64)
            });
        let reals = random_reals::<T>(seed, shape.iter().product());
        let spectrum = reference_r2c(shape, &reals, axes);
        let complex_shape = halved_shape(shape, *axes.last().unwrap());
        let half: Vec<Complex<T>> = multi_indices(complex_shape)
            .map(|index| {
                let value = spectrum[index];
                Complex::new(T::round_from(value.re), T::round_from(value.im))
            })
            .collect();
        let mut wide = DynArray::<Complex<f64>, N>::from_shape(complex_shape);
        for (index, value) in multi_indices(complex_shape).zip(widen(&half)) {
            wide[index] = value;
        }
        let inverse = reference_c2r(shape, &wide, axes);
        Self {
            shape,
            axes: axes.to_vec(),
            seed,
            reals,
            spectrum,
            half,
            inverse,
        }
    }

    fn complex_shape(&self) -> [usize; N] {
        halved_shape(self.shape, *self.axes.last().unwrap())
    }

    fn logical_size(&self) -> usize {
        self.axes.iter().map(|&axis| self.shape[axis]).product()
    }
}

/// The context of a cell's messages.
struct Context<'a, T, const N: usize> {
    case: &'a Case<T, N>,
    family: &'static str,
    part: usize,
    policy: PolicyKind,
}

impl<T: TestReal, const N: usize> fmt::Display for Context<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}, real shape {:?}, axes {:?}, part axis {}, {:?} (seed {:#x})",
            T::NAME,
            self.family,
            self.case.shape,
            self.case.axes,
            self.part,
            self.policy,
            self.case.seed
        )
    }
}

/// The values of a complex array, widened.
fn complex_of<T: TestReal, const N: usize>(
    values: &DynArray<Complex<T>, N>,
) -> DynArray<Complex<f64>, N> {
    let mut wide = DynArray::<Complex<f64>, N>::from_shape(values.shape());
    for index in multi_indices(values.shape()) {
        wide[index] = widen(&[values[index]])[0];
    }
    wide
}

/// The values of a real view as complex values with imaginary part 0, widened.
fn reals_of<T, A, const N: usize>(view: &Array<A, N>) -> DynArray<Complex<f64>, N>
where
    T: TestReal,
    A: BaseItem<Item = T> + UnsafeRandomAccessByRef<N> + Shape<N>,
{
    let mut wide = DynArray::<Complex<f64>, N>::from_shape(view.shape());
    for index in multi_indices(view.shape()) {
        wide[index] = Complex::new(TestReal::to_f64(*view.get(index).unwrap()), 0.0);
    }
    wide
}

/// Reals as complex values with imaginary part 0.
fn as_complex<const N: usize>(reals: &DynArray<f64, N>) -> DynArray<Complex<f64>, N> {
    let mut values = DynArray::<Complex<f64>, N>::from_shape(reals.shape());
    for index in multi_indices(reals.shape()) {
        values[index] = Complex::new(reals[index], 0.0);
    }
    values
}

/// Run the split r2c and c2r cells of `case` with the split side's part axis at `part`, the
/// out-of-place policies, against the references and the interleaved plans.
macro_rules! run_split_real {
    ($t:ty, $n:literal, $m:literal, $case:expr, $part:expr) => {{
        let case: &Case<$t, $n> = $case;
        let part: usize = $part;
        let shape = case.shape;
        let complex_shape = case.complex_shape();
        let split_shape: [usize; $m] = insert_axis(complex_shape, part, 2);
        let n = case.logical_size();

        // The interleaved r2c and c2r results of the same values.
        let mut real = real_sentinel_parent::<$t, $n>(shape);
        let mut complex = sentinel_parent::<$t, $n>(complex_shape);
        for (index, &value) in multi_indices(shape).zip(&case.reals) {
            real[index] = value;
        }
        Planner::<$t>::new()
            .r2c(&Layout::over_axes(&real, &complex, &case.axes).unwrap())
            .unwrap()
            .execute(&real, &mut complex)
            .unwrap();
        let interleaved_spectrum = complex_of(&{
            let mut values = DynArray::<Complex<$t>, $n>::from_shape(complex_shape);
            for index in multi_indices(complex_shape) {
                values[index] = complex[index];
            }
            values
        });
        for (index, &value) in multi_indices(complex_shape).zip(&case.half) {
            complex[index] = value;
        }
        let plan: Plan<$t, C2r, Destroying> = Planner::<$t>::new()
            .c2r(&Layout::over_axes(&complex, &real, &case.axes).unwrap())
            .unwrap();
        plan.execute(&mut complex, &mut real).unwrap();
        let interleaved_inverse = reals_of(&real);

        for policy in [PolicyKind::Preserving, PolicyKind::Destroying] {
            // Split r2c.
            let context = Context {
                case,
                family: "split r2c",
                part,
                policy,
            };
            let mut input = real_sentinel_parent::<$t, $n>(shape);
            let mut output = real_sentinel_parent::<$t, $m>(split_shape);
            for (index, &value) in multi_indices(shape).zip(&case.reals) {
                input[index] = value;
            }
            let layout = Layout::split_r2c_over_axes(&input, &output, part, &case.axes).unwrap();
            let before = storage_bits(&input);
            match policy {
                PolicyKind::Preserving => {
                    let plan: Plan<$t, SplitR2c, Preserving> =
                        Planner::new().split_r2c(&layout).unwrap();
                    plan.execute(&input, &mut output).unwrap();
                    assert!(
                        storage_bits(&input) == before,
                        "{context}: the input changed"
                    );
                }
                _ => {
                    let plan: Plan<$t, SplitR2c, Destroying> = Planner::new()
                        .may_destroy_input()
                        .split_r2c(&layout)
                        .unwrap();
                    plan.execute(&mut input, &mut output).unwrap();
                }
            }
            let result = complex_of(&split_values(&output, part, complex_shape));
            assert_transforms_close(
                &result,
                &case.spectrum,
                &case.axes,
                tolerance::<$t>(n),
                &context,
            );
            let against = format!("{context}, against the interleaved plan");
            assert_transforms_close(
                &result,
                &interleaved_spectrum,
                &case.axes,
                2.0 * tolerance::<$t>(n),
                &against,
            );

            // Split c2r of the Hermitian half.
            let context = Context {
                case,
                family: "split c2r",
                part,
                policy,
            };
            let mut input = real_sentinel_parent::<$t, $m>(split_shape);
            let mut output = real_sentinel_parent::<$t, $n>(shape);
            for (index, value) in multi_indices(complex_shape).zip(&case.half) {
                input[insert_axis(index, part, 0)] = value.re;
                input[insert_axis(index, part, 1)] = value.im;
            }
            let layout = Layout::split_c2r_over_axes(&input, part, &output, &case.axes).unwrap();
            match policy {
                PolicyKind::Preserving => {
                    match Planner::<$t>::new().preserve_input().split_c2r(&layout) {
                        Ok(plan) => {
                            assert_eq!(case.axes.len(), 1, "{context}");
                            let before = storage_bits(&input);
                            plan.execute(&input, &mut output).unwrap();
                            assert!(
                                storage_bits(&input) == before,
                                "{context}: the input changed"
                            );
                        }
                        Err(error) => {
                            assert_eq!(
                                error,
                                FftwError::PreserveInputUnsupported {
                                    rank: case.axes.len()
                                },
                                "{context}"
                            );
                            continue;
                        }
                    }
                }
                _ => {
                    // The default split c2r plan destroys its input.
                    let plan: Plan<$t, SplitC2r, Destroying> =
                        Planner::new().split_c2r(&layout).unwrap();
                    plan.execute(&mut input, &mut output).unwrap();
                }
            }
            let result = reals_of(&output);
            assert_transforms_close(
                &result,
                &as_complex(&case.inverse),
                &case.axes,
                tolerance::<$t>(n),
                &context,
            );
            let against = format!("{context}, against the interleaved plan");
            assert_transforms_close(
                &result,
                &interleaved_inverse,
                &case.axes,
                2.0 * tolerance::<$t>(n),
                &against,
            );
        }
    }};
}

fn check_split_real<T: TestReal>() {
    for n in [1, 2, 3, 17, 140] {
        let case = Case::<T, 1>::new([n], &[0]);
        for part in [0, 1] {
            run_split_real!(T, 1, 2, &case, part);
        }
    }
    for n in [2, 17, 64] {
        for axis in [0, 1] {
            let case = Case::<T, 2>::new([n, 3], &[axis]);
            for part in [0, 2] {
                run_split_real!(T, 2, 3, &case, part);
            }
        }
    }
    for axes in [&[0, 1][..], &[2, 0]] {
        let case = Case::<T, 3>::new([6, 5, 4], axes);
        // Outermost, innermost, and in the middle.
        for part in [3, 0, 2] {
            run_split_real!(T, 3, 4, &case, part);
        }
    }
}

#[test]
fn test_split_real_against_references_and_interleaved_f64() {
    check_split_real::<f64>();
}

#[test]
fn test_split_real_against_references_and_interleaved_f32() {
    check_split_real::<f32>();
}

#[test]
fn test_split_real_layouts() {
    // The row of the brief's table: real [8, 3]; split [5, 3, 2] with strides [1, 5, 15], part
    // axis 2.
    let real = DynArray::<f64, 2>::from_shape([8, 3]);
    let split = DynArray::<f64, 3>::from_shape([5, 3, 2]);
    assert_eq!(split.stride(), [1, 5, 15]);
    let expected = Layout::with_storage(
        &[Dim { n: 8, is: 1, os: 1 }],
        &[Dim { n: 3, is: 8, os: 5 }],
        (Storage::Real, Storage::Split { gap: 15 }),
        Placement::OutOfPlace,
    )
    .unwrap();
    assert_eq!(
        Layout::split_r2c_along_axis(&real, &split, 2, 0).unwrap(),
        expected
    );
    assert_eq!(
        Layout::split_r2c_over_axes(&real, &split, 2, &[0]).unwrap(),
        expected
    );
    let inverse = Layout::split_c2r_along_axis(&split, 2, &real, 0).unwrap();
    assert_eq!(
        inverse.storage(),
        (Storage::Split { gap: 15 }, Storage::Real)
    );
    assert_eq!(inverse.dims(), &[Dim { n: 8, is: 1, os: 1 }]);
    assert_eq!(inverse.batch(), &[Dim { n: 3, is: 5, os: 8 }]);
    assert_eq!(inverse.gaps(), None);

    // The part axis innermost: the split array [2, 5, 3] is interleaved memory; logical axis
    // 1 is its axis 2.
    let interleaved = DynArray::<f64, 3>::from_shape([2, 5, 3]);
    let layout = Layout::split_r2c_along_axis(&real, &interleaved, 0, 0).unwrap();
    assert_eq!(layout.dims(), &[Dim { n: 8, is: 1, os: 2 }]);
    assert_eq!(
        layout.batch(),
        &[Dim {
            n: 3,
            is: 8,
            os: 10
        }]
    );
    assert_eq!(layout.storage().1, Storage::Split { gap: 1 });
    // Along logical axis 1, halved to 2 for the real extent 3.
    let real = DynArray::<f64, 2>::from_shape([8, 3]);
    let split = DynArray::<f64, 3>::from_shape([8, 2, 2]);
    let layout = Layout::split_r2c_along_axis(&real, &split, 2, 1).unwrap();
    assert_eq!(layout.dims(), &[Dim { n: 3, is: 8, os: 8 }]);
}

#[test]
fn test_split_real_layout_errors() {
    let invalid = |result: Result<Layout, FftwError>| match result {
        Err(FftwError::InvalidLayout { dimension, reason }) => (dimension, reason),
        other => panic!("expected InvalidLayout, found {other:?}"),
    };
    let real = DynArray::<f64, 2>::from_shape([8, 3]);
    let split = DynArray::<f64, 3>::from_shape([5, 3, 2]);

    // The part axis: in range, of extent 2, and one axis more than the real array.
    let (dimension, reason) = invalid(Layout::split_r2c_along_axis(&real, &split, 3, 0));
    assert_eq!(dimension, "part axis");
    assert!(reason.contains("not below the rank 3"), "{reason}");
    let (dimension, reason) = invalid(Layout::split_r2c_along_axis(&real, &split, 1, 0));
    assert_eq!(dimension, "part axis");
    assert!(
        reason.contains("extent along its part axis 1 is 3"),
        "{reason}"
    );
    let flat = DynArray::<f64, 2>::from_shape([5, 2]);
    let (dimension, reason) = invalid(Layout::split_r2c_along_axis(&real, &flat, 1, 0));
    assert_eq!(dimension, "part axis");
    assert!(reason.contains("one axis more"), "{reason}");
    let (dimension, _) = invalid(Layout::split_c2r_along_axis(
        &split,
        2,
        &DynArray::<f64, 3>::from_shape([8, 3, 1]),
        0,
    ));
    assert_eq!(dimension, "part axis");

    // The halved extent on the split side, named by the logical axis.
    let wrong = DynArray::<f64, 3>::from_shape([4, 3, 2]);
    let (dimension, reason) = invalid(Layout::split_r2c_along_axis(&real, &wrong, 2, 0));
    assert_eq!(dimension, "logical axis 0");
    assert!(reason.contains("must be 5"), "{reason}");
    let (dimension, reason) = invalid(Layout::split_c2r_along_axis(&wrong, 2, &real, 0));
    assert_eq!(dimension, "logical axis 0");
    assert!(reason.contains("but is 4"), "{reason}");
    // Equal extents elsewhere.
    let wrong = DynArray::<f64, 3>::from_shape([5, 4, 2]);
    let (dimension, reason) = invalid(Layout::split_r2c_along_axis(&real, &wrong, 2, 0));
    assert_eq!(dimension, "logical axis 1");
    assert!(reason.contains("split real-to-complex"), "{reason}");
    // The axes.
    let (dimension, _) = invalid(Layout::split_r2c_over_axes(&real, &split, 2, &[]));
    assert_eq!(dimension, "axes");
    let (dimension, reason) = invalid(Layout::split_c2r_over_axes(&split, 2, &real, &[2]));
    assert_eq!(dimension, "axes");
    assert!(reason.contains("logical axis 2"), "{reason}");
    let (dimension, _) = invalid(Layout::split_r2c_over_axes(&real, &split, 2, &[1, 1]));
    assert_eq!(dimension, "logical axis 1");
}

#[test]
fn test_split_real_defaults_and_policies() {
    let real = DynArray::<f64, 2>::from_shape([8, 3]);
    let split = DynArray::<f64, 3>::from_shape([5, 3, 2]);
    let r2c = Layout::split_r2c_along_axis(&real, &split, 2, 0).unwrap();
    let c2r = Layout::split_c2r_along_axis(&split, 2, &real, 0).unwrap();
    let _: Plan<f64, SplitR2c, Preserving> = Planner::new().split_r2c(&r2c).unwrap();
    let _: Plan<f64, SplitR2c, Preserving> =
        Planner::new().preserve_input().split_r2c(&r2c).unwrap();
    let _: Plan<f64, SplitR2c, Destroying> =
        Planner::new().may_destroy_input().split_r2c(&r2c).unwrap();
    let _: Plan<f64, SplitC2r, Destroying> = Planner::new().split_c2r(&c2r).unwrap();
    let _: Plan<f64, SplitC2r, Preserving> =
        Planner::new().preserve_input().split_c2r(&c2r).unwrap();
    let _: Plan<f64, SplitC2r, Destroying> =
        Planner::new().may_destroy_input().split_c2r(&c2r).unwrap();

    // Over two transform dimensions a preserving split c2r is unsupported, on scratch and on
    // the caller's buffers, before FFTW is called.
    let mut input = real_sentinel_parent::<f64, 3>([3, 5, 2]);
    let mut output = real_sentinel_parent::<f64, 2>([3, 8]);
    let layout = Layout::split_c2r_over_axes(&input, 2, &output, &[0, 1]).unwrap();
    let unsupported = FftwError::PreserveInputUnsupported { rank: 2 };
    assert_eq!(
        Planner::<f64>::new()
            .preserve_input()
            .split_c2r(&layout)
            .unwrap_err(),
        unsupported
    );
    let before = storage_bits(&output);
    assert_eq!(
        Planner::<f64>::new()
            .preserve_input()
            .split_c2r_on(&mut input, &mut output, &layout)
            .unwrap_err(),
        unsupported
    );
    assert!(storage_bits(&output) == before);

    // The plans' properties.
    let plan = Planner::<f64>::new().split_r2c(&r2c).unwrap();
    assert_eq!(plan.logical_size(), 8);
    assert!(format!("{plan:?}").contains("SplitR2c"));
    assert!(!plan.description().is_empty());
}

#[test]
fn test_layouts_of_other_families() {
    let dims = [Dim { n: 8, is: 1, os: 1 }];
    let make = |storage| Layout::with_storage(&dims, &[], storage, Placement::OutOfPlace).unwrap();
    let named = |error: FftwError| match error {
        FftwError::InvalidLayout { dimension, reason } => (dimension, reason),
        other => panic!("expected InvalidLayout, found {other}"),
    };
    let planner = Planner::<f64>::new();
    let split_r2c = (Storage::Real, Storage::Split { gap: 5 });
    let split_c2r = (Storage::Split { gap: 5 }, Storage::Real);
    for (storage, dimension) in [
        ((Storage::Real, Storage::Complex), "part axis"),
        ((Storage::Real, Storage::Real), "part axis"),
        ((Storage::Complex, Storage::Complex), "part axis"),
        (
            (Storage::Split { gap: 8 }, Storage::Split { gap: 8 }),
            "part axis",
        ),
        (split_c2r, "part axis"),
    ] {
        let (found, reason) = named(planner.split_r2c(&make(storage)).unwrap_err());
        assert_eq!(found, dimension, "{reason}");
        assert!(
            reason.contains("split real-to-complex plan needs"),
            "{reason}"
        );
    }
    let (found, reason) = named(planner.split_c2r(&make(split_r2c)).unwrap_err());
    assert_eq!(found, "part axis");
    assert!(reason.contains("`split_r2c` planner methods"), "{reason}");
    let (found, _) = named(
        planner
            .split_c2r(&make((Storage::Complex, Storage::Real)))
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");
    // The other planner methods reject the split real layouts.
    let (found, reason) = named(planner.r2c(&make(split_r2c)).unwrap_err());
    assert_eq!(found, "part axis");
    assert!(reason.contains("`split_r2c` planner methods"), "{reason}");
    let (found, _) = named(planner.c2r(&make(split_c2r)).unwrap_err());
    assert_eq!(found, "part axis");
    let (found, _) = named(
        planner
            .split_c2c(&make(split_r2c), crate::fftw::Direction::Forward)
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");
    let (found, _) = named(
        planner
            .r2r(&make(split_c2r), &[crate::fftw::R2rKind::Dht])
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");

    // On the caller's buffers, which keep their sentinel.
    let mut input = filled(real_sentinel::<f64>(), 16);
    let mut output = filled(real_sentinel::<f64>(), 16);
    let sentinel = real_sentinel::<f64>().bits();
    let (found, _) = named(
        planner
            .split_r2c_on(&mut input[..], &mut output[..], &make(split_c2r))
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");
    let (found, _) = named(
        planner
            .split_c2r_on(
                &mut input[..],
                &mut output[..],
                &make((Storage::Complex, Storage::Real)),
            )
            .unwrap_err(),
    );
    assert_eq!(found, "part axis");
    assert!(output.iter().all(|value| value.bits() == sentinel));
    // No in-place split real layouts exist, so neither do in-place plans.
    assert!(Layout::with_storage(&dims, &[], split_r2c, Placement::InPlace).is_err());
}

#[test]
fn test_split_real_execution_checks() {
    // A plan from real [8, 3] and split planes [5, 3, 2].
    let mut real = real_sentinel_parent::<f64, 2>([8, 3]);
    let mut split = real_sentinel_parent::<f64, 3>([5, 3, 2]);
    let layout = Layout::split_r2c_along_axis(&real, &split, 2, 0).unwrap();
    let plan = Planner::<f64>::new().split_r2c(&layout).unwrap();
    plan.execute(&real, &mut split).unwrap();

    // The halved extent and the gap of the output's part axis are checked.
    for shape in [[4, 3, 2], [6, 3, 2]] {
        let mut wrong = real_sentinel_parent::<f64, 3>(shape);
        let before = storage_bits(&wrong);
        let error = plan.execute(&real, &mut wrong).unwrap_err();
        assert!(
            matches!(
                error,
                FftwError::LayoutMismatch {
                    side: Side::Output,
                    ..
                }
            ),
            "{error}"
        );
        assert!(storage_bits(&wrong) == before);
    }
    // Axes that fit each side alone but do not pair without the part axis (DC15, DC27): a plan
    // from a split output whose batch is transposed in memory, given column-major arrays of
    // the shape with the batch axes swapped.
    let input = real_sentinel_parent::<f64, 3>([4, 2, 3]);
    let transposed = StridedDynArray::<f64, 4>::from_shape_and_stride([3, 2, 3, 2], [1, 9, 3, 18]);
    let plan_t = Planner::<f64>::new()
        .split_r2c(&Layout::split_r2c_along_axis(&input, &transposed, 3, 0).unwrap())
        .unwrap();
    let other_input = real_sentinel_parent::<f64, 3>([4, 3, 2]);
    let mut other_output = real_sentinel_parent::<f64, 4>([3, 3, 2, 2]);
    let before = storage_bits(&other_output);
    let error = plan_t.execute(&other_input, &mut other_output).unwrap_err();
    assert!(
        matches!(&error, FftwError::LayoutMismatch { side: Side::Output, expected, .. } if expected.contains("pair by index")),
        "{error}"
    );
    assert!(storage_bits(&other_output) == before);

    // Alignment: one real off class 0.
    let layout = Layout::with_storage(
        &[Dim {
            n: 16,
            is: 1,
            os: 1,
        }],
        &[],
        (Storage::Real, Storage::Split { gap: 9 }),
        Placement::OutOfPlace,
    )
    .unwrap();
    let plan = Planner::<f64>::new().split_r2c(&layout).unwrap();
    let input = aligned(&random_reals::<f64>(SEED, 17));
    let mut output = filled(real_sentinel::<f64>(), 19);
    let error = plan.execute(&input[1..], &mut output[..18]).unwrap_err();
    assert_eq!(
        error,
        FftwError::AlignmentMismatch {
            side: Side::Input,
            expected: AlignmentClass::default(),
            found: AlignmentClass::from_offset(8),
        }
    );
    let error = plan.execute(&input[..16], &mut output[1..]).unwrap_err();
    assert!(
        matches!(
            error,
            FftwError::AlignmentMismatch {
                side: Side::Output,
                ..
            }
        ),
        "{error}"
    );
    // A slice shorter than the split span, 18 reals with the gap.
    let error = plan.execute(&input[..16], &mut output[..17]).unwrap_err();
    assert!(
        matches!(
            error,
            FftwError::LayoutMismatch {
                side: Side::Output,
                ..
            }
        ),
        "{error}"
    );
    let sentinel = real_sentinel::<f64>().bits();
    assert!(output.iter().all(|value| value.bits() == sentinel));
    let _ = &mut real;
}

#[test]
fn test_split_real_scratch_cap() {
    // A column of a row-major real [8, 1000] into the column of row-major planes [5, 1000, 2]:
    // 8 reals 1000 apart span 7001 reals; 5 complex values 2000 apart and a gap of 1 span
    // 8002 reals.
    let mut real = StridedDynArray::<f64, 2>::row_major([8, 1000]);
    let mut split = StridedDynArray::<f64, 3>::row_major([5, 1000, 2]);
    let values = random_reals::<f64>(SEED, 8);
    for (i, &value) in values.iter().enumerate() {
        real[[i, 7]] = value;
    }
    let layout =
        Layout::split_r2c_along_axis(&real.r().col(7), &split.r().slice::<2>(1, 7), 1, 0).unwrap();
    let planner = Planner::<f64>::new().max_scratch(1024);
    assert_eq!(
        planner.split_r2c(&layout).unwrap_err(),
        FftwError::ScratchTooLarge {
            needed: (7001 + 8002) * 8,
            bound: 1024,
        }
    );
    let plan = planner
        .split_r2c_on(
            &mut real.r_mut().col(7),
            &mut split.r_mut().slice::<2>(1, 7),
            &layout,
        )
        .unwrap();
    plan.execute(&real.r().col(7), &mut split.r_mut().slice::<2>(1, 7))
        .unwrap();
    let result = complex_of(&split_values(&split.r().slice::<2>(1, 7), 1, [5]));
    assert_transforms_close(
        &result,
        &reference_r2c([8], &values, &[0]),
        &[0],
        tolerance::<f64>(8),
        &"scratch cap",
    );
    // Other columns are not written.
    assert_eq!(split[[0, 6, 0]], 0.0);
}
