use crate::dense::array::StridedSliceArray;
use crate::operator::{element::Element, interface::array_vector_space::ArrayVectorSpace};
use crate::{
    ArrayFrameImpl, DynArray, ElementFrame, FrameApply, FrameQr, FrameQrOptions, Gemm,
    GenericFrameImpl, InnerProductSpace, Lapack, LinearSpace, Norm, RandScalar, RlstScalar,
    SliceArray, c32, c64,
};

// A non-array implementation with a non-Euclidean inner product. Its operations
// deliberately depend on instance metadata, exercising the borrowed space.
struct WeightedSpace<Item> {
    weights: [Item; 4],
}

impl<Item: RlstScalar> LinearSpace for WeightedSpace<Item> {
    type F = Item;
    type Impl = [Item; 4];

    fn zero(&self) -> Element<'_, Self> {
        Element::new(self, [Item::zero(); 4])
    }
    fn add(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, std::array::from_fn(|i| x.imp()[i] + y.imp()[i]))
    }
    fn sub(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, std::array::from_fn(|i| x.imp()[i] - y.imp()[i]))
    }
    fn scalar_mul(&self, scalar: &Item, x: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, x.imp().map(|v| v * *scalar))
    }
    fn neg(&self, x: &Element<Self>) -> Element<'_, Self> {
        self.scalar_mul(&-Item::one(), x)
    }
    fn sum_inplace(&self, x: &mut Element<Self>, y: &Element<Self>) {
        for i in 0..4 {
            x.imp_mut()[i] += y.imp()[i];
        }
    }
    fn sub_inplace(&self, x: &mut Element<Self>, y: &Element<Self>) {
        for i in 0..4 {
            x.imp_mut()[i] -= y.imp()[i];
        }
    }
    fn scale_inplace(&self, scalar: &Item, x: &mut Element<Self>) {
        for value in x.imp_mut() {
            *value *= *scalar;
        }
    }
    fn copy_from(&self, x: &Element<Self>) -> Element<'_, Self> {
        Element::new(self, *x.imp())
    }
}

impl<Item: RlstScalar> InnerProductSpace for WeightedSpace<Item> {
    fn inner_product(&self, x: &Element<Self>, y: &Element<Self>) -> Item {
        (0..4)
            .map(|i| self.weights[i] * x.imp()[i] * y.imp()[i].conj())
            .sum()
    }
}

fn verify_qr<'a, F, Item>(frame: &F, options: FrameQrOptions<Item::Real>, rank: usize, tol: f64)
where
    F: FrameQr<'a> + FrameApply<'a>,
    F::Space: InnerProductSpace<F = Item>,
    Item: RlstScalar,
{
    let result = frame.qr_pivoted(options).unwrap();
    assert_eq!(result.rank(), rank);
    assert_eq!(result.q.len(), rank);
    assert_eq!(result.r.shape(), [rank, frame.len()]);
    let mut sorted = result.permutation.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..frame.len()).collect::<Vec<_>>());
    for (j, &original) in result.permutation.iter().enumerate() {
        let actual = result.q.apply(&result.r.r().col(j)).unwrap();
        let expected = frame.element(original).unwrap();
        let error = (&actual - &expected).norm();
        assert!(error <= Item::real(tol), "reconstruction error {error}");
        for i in j + 1..rank {
            assert_eq!(result.r[[i, j]], Item::zero());
        }
    }
    for i in 0..rank {
        for j in 0..rank {
            let value = frame
                .space()
                .inner_product(&result.q.element(i).unwrap(), &result.q.element(j).unwrap());
            let expected = if i == j { Item::one() } else { Item::zero() };
            assert!(
                (value - expected).abs() < Item::real(tol.clamp(1e-12, 1e-5)),
                "orthogonality error {}",
                (value - expected).abs()
            );
        }
    }
}

fn exercise_scalar<Item: Lapack + Gemm + RandScalar>(tol: f64)
where
    rand::distr::StandardUniform: rand::distr::Distribution<Item::Real>,
    rand_distr::StandardNormal: rand::distr::Distribution<Item::Real>,
{
    // Including an imaginary component makes reversed complex inner products fail.
    let phase = if Item::IS_COMPLEX {
        Item::one() + (-Item::one()).sqrt().mul_real(Item::real(2.0))
    } else {
        Item::one()
    };
    let space = WeightedSpace {
        weights: [Item::from_real(Item::real(2.0)); 4],
    };
    let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
    let columns = [
        [Item::one(), Item::zero(), Item::zero(), Item::zero()],
        [
            phase,
            Item::from_real(Item::real(3.0)),
            Item::zero(),
            Item::zero(),
        ],
        [
            Item::zero(),
            phase,
            Item::from_real(Item::real(8.0)),
            Item::zero(),
        ],
        [Item::zero(); 4],
    ];
    for column in columns {
        generic.push(&Element::new(&space, column)).unwrap();
    }
    let dependent = &generic.element(0).unwrap() * phase + generic.element(2).unwrap();
    generic.push(&dependent).unwrap();
    verify_qr(&generic, FrameQrOptions::default(), 3, tol);

    for (m, n) in [(7, 3), (3, 7), (4, 4)] {
        let space = ArrayVectorSpace::<Item>::new(m);
        let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
        for j in 0..n {
            let mut element = space.zero();
            element.imp_mut().fill_from_seed_equally_distributed(j + 10);
            dense.push(&element).unwrap();
        }
        verify_qr(&dense, FrameQrOptions::default(), m.min(n), tol);
        // Orthogonality and reconstruction agree across backends even when
        // LAPACK and CGS choose different signs or tied pivots.
        let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
        for j in 0..n {
            generic.push(&dense.element(j).unwrap()).unwrap();
        }
        verify_qr(&generic, FrameQrOptions::default(), m.min(n), tol);
    }
}

#[test]
fn qr_real_double() {
    exercise_scalar::<f64>(1e-11);
}
#[test]
fn qr_real_single() {
    exercise_scalar::<f32>(2e-4);
}
#[test]
fn qr_complex_double() {
    exercise_scalar::<c64>(1e-11);
}
#[test]
fn qr_complex_single() {
    exercise_scalar::<c32>(2e-4);
}

#[test]
fn dense_storage_growth_and_views() {
    let space = ArrayVectorSpace::<f64>::new(3);
    let mut frame = ElementFrame::new(ArrayFrameImpl::new(&space));
    frame.reserve(2);
    let mut element = space.zero();
    for j in 0..9 {
        element.imp_mut().fill_with_value(j as f64);
        frame.push(&element).unwrap();
    }
    assert!(frame.capacity() >= 9);
    assert_eq!(frame.matrix().shape(), [3, 9]);
    for j in 0..9 {
        assert_eq!(frame.column(j).unwrap()[[2]], j as f64);
    }
    frame.column_mut(3).unwrap()[[1]] = 42.0;
    assert_eq!(frame.matrix()[[1, 3]], 42.0);
    frame.matrix_mut()[[0, 5]] = 27.0;
    assert_eq!(frame.element(5).unwrap().imp()[[0]], 27.0);
    assert!(frame.element(9).is_none());
    assert!(frame.column_mut(9).is_none());
    assert!(frame.set_element(9, &element).is_err());
    element.imp_mut().fill_with_value(-1.0);
    frame.set_element(0, &element).unwrap();
    assert_eq!(frame.column(0).unwrap()[[1]], -1.0);
    let extracted = frame.element(3).unwrap();
    assert_eq!(frame.pop().unwrap().imp()[[0]], 8.0);
    let capacity = frame.capacity();
    frame.clear();
    assert!(frame.is_empty());
    assert!(frame.pop().is_none());
    assert_eq!(frame.capacity(), capacity);
    frame.push(&element).unwrap();
    assert_eq!(frame.matrix().shape(), [3, 1]);
    drop(frame);
    assert_eq!(extracted.imp()[[1]], 42.0);
}

#[test]
fn generic_storage_owns_values_and_preserves_space_lifetime() {
    let space = WeightedSpace { weights: [1.0; 4] };
    let mut element = space.zero();
    let extracted;
    {
        let mut frame = ElementFrame::new(GenericFrameImpl::new(&space));
        frame.reserve(4);
        element.imp_mut()[0] = 2.0;
        frame.push(&element).unwrap();
        element.imp_mut()[0] = 3.0;
        assert_eq!(frame.iter().next().unwrap().imp()[0], 2.0);
        frame.set_element(0, &element).unwrap();
        assert!(frame.set_element(1, &element).is_err());
        extracted = frame.pop().unwrap();
        assert!(frame.pop().is_none());
        frame.push(&element).unwrap();
        let capacity = frame.capacity();
        frame.clear();
        assert_eq!(frame.capacity(), capacity);
        assert!(frame.is_empty());
    }
    assert_eq!(extracted.imp()[0], 3.0);
}

fn exercise_application<'a, F: FrameApply<'a, Space = ArrayVectorSpace<c64>>>(frame: &mut F) {
    let space = frame.space();
    let mut element = space.zero();
    element.imp_mut()[[0]] = c64::new(1.0, 2.0);
    frame.push(&element).unwrap();
    element.imp_mut()[[0]] = c64::new(3.0, -1.0);
    frame.push(&element).unwrap();
    let data = [
        c64::new(2.0, -1.0),
        c64::new(999.0, 0.0),
        c64::new(-1.0, 3.0),
    ];
    let coefficients = StridedSliceArray::from_shape_and_stride(&data, [2], [2]);
    let expected = c64::new(1.0, 2.0) * data[0] + c64::new(3.0, -1.0) * data[2];
    assert_eq!(frame.apply(&coefficients).unwrap().imp()[[0]], expected);
    assert_eq!(
        frame
            .apply(&(coefficients.r() + coefficients.r()))
            .unwrap()
            .imp()[[0]],
        2.0 * expected
    );
    let alpha = c64::new(2.0, 1.0);
    let beta = c64::new(-1.0, 2.0);
    let previous = element.imp()[[0]];
    frame
        .apply_into(alpha, &coefficients, beta, &mut element)
        .unwrap();
    assert_eq!(element.imp()[[0]], alpha * expected + beta * previous);
    let bad = DynArray::<c64, 1>::from_shape([1]);
    let previous = element.imp()[[0]];
    assert!(frame.apply_into(alpha, &bad, beta, &mut element).is_err());
    assert_eq!(element.imp()[[0]], previous);
    element.imp_mut()[[0]] = c64::new(f64::NAN, f64::NAN);
    frame
        .apply_into(
            c64::new(1.0, 0.0),
            &coefficients,
            c64::new(0.0, 0.0),
            &mut element,
        )
        .unwrap();
    assert_eq!(element.imp()[[0]], expected);
    frame.clear();
    let empty = DynArray::<c64, 1>::from_shape([0]);
    assert_eq!(frame.apply(&empty).unwrap().norm(), 0.0);
}

#[test]
fn application_strides_expressions_and_alpha_beta() {
    let space = ArrayVectorSpace::<c64>::new(2);
    exercise_application(&mut ElementFrame::new(ArrayFrameImpl::new(&space)));
    exercise_application(&mut ElementFrame::new(GenericFrameImpl::new(&space)));
}

#[test]
fn adjoint_apply_conjugates_columns_and_matches_between_storages() {
    let space = ArrayVectorSpace::<c64>::new(2);
    let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
    let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
    let x = {
        let mut x = space.zero();
        x.imp_mut()[[0]] = c64::new(1.0, 2.0);
        x.imp_mut()[[1]] = c64::new(-3.0, 0.5);
        x
    };

    // An empty frame yields an empty coefficient vector.
    assert!(dense.adjoint_apply(&x).unwrap().is_empty());
    assert!(generic.adjoint_apply(&x).unwrap().is_empty());

    let columns = [
        [c64::new(0.0, 1.0), c64::new(2.0, 0.0)],
        [c64::new(1.0, -1.0), c64::new(0.0, 0.0)],
        [c64::new(0.5, 0.5), c64::new(-1.0, 2.0)],
    ];
    for values in columns {
        let mut element = space.zero();
        element.imp_mut()[[0]] = values[0];
        element.imp_mut()[[1]] = values[1];
        dense.push(&element).unwrap();
        generic.push(&element).unwrap();
    }

    let dense_result = dense.adjoint_apply(&x).unwrap();
    let generic_result = generic.adjoint_apply(&x).unwrap();
    assert_eq!(dense_result.len(), 3);
    assert_eq!(generic_result.len(), 3);
    for (index, values) in columns.iter().enumerate() {
        let expected = x.imp()[[0]] * values[0].conj() + x.imp()[[1]] * values[1].conj();
        assert!((dense_result[index] - expected).norm() < 1e-12);
        assert!((generic_result[index] - expected).norm() < 1e-12);
    }

    // Wrong dimensions are rejected by the dense frame.
    let other = ArrayVectorSpace::<c64>::new(3);
    assert!(dense.adjoint_apply(&other.zero()).is_err());
}

#[test]
fn dense_rejects_incompatible_elements_without_mutating() {
    let space = ArrayVectorSpace::<f64>::new(2);
    let other = ArrayVectorSpace::<f64>::new(3);
    let mut frame = ElementFrame::new(ArrayFrameImpl::new(&space));
    assert!(frame.push(&other.zero()).is_err());
    let malformed = Element::new(&space, DynArray::from_shape([3]));
    assert!(frame.push(&malformed).is_err());
    assert!(frame.is_empty());
    frame.push(&space.zero()).unwrap();
    assert!(frame.set_element(0, &malformed).is_err());
    let mut output = other.zero();
    assert!(
        frame
            .apply_into(1.0, &SliceArray::from_shape(&[1.0], [1]), 0.0, &mut output)
            .is_err()
    );
    assert_eq!(output.imp().shape(), [3]);
}

#[test]
fn qr_empty_zero_dimensional_and_zero_columns() {
    for m in [0, 3] {
        let space = ArrayVectorSpace::<f64>::new(m);
        let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
        let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
        for n in 0..4 {
            verify_qr(&dense, FrameQrOptions::default(), 0, 1e-12);
            verify_qr(&generic, FrameQrOptions::default(), 0, 1e-12);
            assert_eq!(dense.len(), n);
            dense.push(&space.zero()).unwrap();
            generic.push(&space.zero()).unwrap();
        }
    }
}

#[test]
fn qr_truncation_and_relative_scale() {
    for scale in [1e-8, 1.0, 1e8] {
        let space = ArrayVectorSpace::<f64>::new(3);
        let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
        let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
        for values in [
            [scale, 0.0, 0.0],
            [scale, scale * 1e-8, 0.0],
            [0.0, 0.0, scale * 2.0],
        ] {
            let element = Element::new(&space, DynArray::from_shape_and_vec([3], values.to_vec()));
            dense.push(&element).unwrap();
            generic.push(&element).unwrap();
        }
        let options = FrameQrOptions {
            atol: 0.0,
            rtol: 1e-7,
        };
        verify_qr(&dense, options, 2, scale * 1e-7);
        verify_qr(&generic, options, 2, scale * 1e-7);
        let absolute = FrameQrOptions {
            atol: scale * 1e-7,
            rtol: 0.0,
        };
        verify_qr(&dense, absolute, 2, scale * 1e-7);
        verify_qr(&generic, absolute, 2, scale * 1e-7);
        verify_qr(&dense, FrameQrOptions::default(), 3, scale * 1e-12);
        verify_qr(&generic, FrameQrOptions::default(), 3, scale * 1e-12);
        let discard_all = FrameQrOptions {
            atol: 3.0 * scale,
            rtol: 0.0,
        };
        verify_qr(&dense, discard_all, 0, 3.0 * scale);
        verify_qr(&generic, discard_all, 0, 3.0 * scale);
    }
}

#[test]
fn qr_reorthogonalizes_ill_conditioned_columns() {
    // Lauchli matrix: a single classical pass loses orthogonality for these
    // nearly parallel columns. CGS2 should retain all three directions.
    let space = WeightedSpace { weights: [1.0; 4] };
    let mut frame = ElementFrame::new(GenericFrameImpl::new(&space));
    for j in 1..4 {
        let mut values = [0.0; 4];
        values[0] = 1.0;
        values[j] = 1e-7;
        frame.push(&Element::new(&space, values)).unwrap();
    }
    verify_qr(&frame, FrameQrOptions::default(), 3, 1e-12);
}

#[test]
fn qr_pivots_and_preserves_input() {
    let space = ArrayVectorSpace::<f64>::new(3);
    let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
    let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
    for values in [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 10.0],
        [0.0, 5.0, 0.0],
    ] {
        let element = Element::new(&space, DynArray::from_shape_and_vec([3], values.to_vec()));
        dense.push(&element).unwrap();
        generic.push(&element).unwrap();
    }
    let before = DynArray::new_from(&dense.matrix());
    let result = dense.qr_pivoted(FrameQrOptions::default()).unwrap();
    assert_eq!(&result.permutation[..2], &[2, 3]);
    let result = generic.qr_pivoted(FrameQrOptions::default()).unwrap();
    assert_eq!(&result.permutation[..2], &[2, 3]);
    verify_qr(&dense, FrameQrOptions::default(), 3, 1e-12);
    verify_qr(&generic, FrameQrOptions::default(), 3, 1e-12);
    assert_eq!(dense.matrix().data(), before.data());
    for j in 0..dense.len() {
        assert_eq!(
            generic.element(j).unwrap().imp().data(),
            dense.column(j).unwrap().data()
        );
    }
}

#[test]
fn qr_rejects_invalid_tolerances_and_nonfinite_data() {
    let space = ArrayVectorSpace::<f64>::new(1);
    let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
    let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
    for options in [
        FrameQrOptions {
            atol: -1.0,
            rtol: 0.0,
        },
        FrameQrOptions {
            atol: 0.0,
            rtol: -1.0,
        },
        FrameQrOptions {
            atol: f64::NAN,
            rtol: 0.0,
        },
        FrameQrOptions {
            atol: 0.0,
            rtol: f64::INFINITY,
        },
    ] {
        assert!(dense.qr_pivoted(options).is_err());
        assert!(generic.qr_pivoted(options).is_err());
    }
    for value in [f64::NAN, f64::INFINITY] {
        dense.clear();
        generic.clear();
        let element = Element::new(&space, DynArray::from_shape_and_vec([1], vec![value]));
        dense.push(&element).unwrap();
        generic.push(&element).unwrap();
        assert!(dense.qr_pivoted(FrameQrOptions::default()).is_err());
        assert!(generic.qr_pivoted(FrameQrOptions::default()).is_err());
    }
}

#[test]
fn qr_results_borrow_space_not_input_frame() {
    let space = ArrayVectorSpace::<f64>::new(2);
    let mut element = space.zero();
    element.imp_mut()[[1]] = 2.0;
    let (dense_qr, generic_qr) = {
        let mut dense = ElementFrame::new(ArrayFrameImpl::new(&space));
        let mut generic = ElementFrame::new(GenericFrameImpl::new(&space));
        dense.push(&element).unwrap();
        generic.push(&element).unwrap();
        (
            dense.qr_pivoted(FrameQrOptions::default()).unwrap(),
            generic.qr_pivoted(FrameQrOptions::default()).unwrap(),
        )
    };
    let dense_result = dense_qr.q.apply(&dense_qr.r.r().col(0)).unwrap();
    let generic_result = generic_qr.q.apply(&generic_qr.r.r().col(0)).unwrap();
    assert!((&dense_result - &element).norm() < 1e-12);
    assert!((&generic_result - &element).norm() < 1e-12);
}
