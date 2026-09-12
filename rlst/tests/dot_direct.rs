//! External-crate regression tests for direct tensor contraction.

use rlst::{DynArray, StridedDynArray, c64};
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

// Visit coordinates recursively, independently of RLST's storage/index helpers.
fn coordinates<const N: usize>(shape: [usize; N]) -> Vec<[usize; N]> {
    fn visit<const N: usize>(
        shape: &[usize; N],
        index: &mut [usize; N],
        axis: usize,
        result: &mut Vec<[usize; N]>,
    ) {
        if axis == N {
            result.push(*index);
        } else {
            for value in 0..shape[axis] {
                index[axis] = value;
                visit(shape, index, axis + 1, result);
            }
        }
    }
    let mut result = Vec::new();
    visit(&shape, &mut [0; N], 0, &mut result);
    result
}

fn assert_product<const A: usize, const B: usize, const C: usize>(
    a_shape: [usize; A],
    b_shape: [usize; B],
    c_shape: [usize; C],
    a: impl Fn([usize; A]) -> f64,
    b: impl Fn([usize; B]) -> f64,
    c: impl Fn([usize; C]) -> f64,
) {
    let mut expected_shape = a_shape[..A - 1].to_vec();
    expected_shape.extend_from_slice(&b_shape[1..]);
    if expected_shape.is_empty() {
        expected_shape.push(1);
    }
    assert_eq!(c_shape.as_slice(), expected_shape);
    assert_eq!(a_shape[A - 1], b_shape[0]);
    for output_index in coordinates(c_shape) {
        let mut ai = [0; A];
        let mut bi = [0; B];
        ai[..A - 1].copy_from_slice(&output_index[..A - 1]);
        bi[1..].copy_from_slice(&output_index[A - 1..A + B - 2]);
        let mut expected = 0.0;
        for k in 0..a_shape[A - 1] {
            ai[A - 1] = k;
            bi[0] = k;
            expected += a(ai) * b(bi);
        }
        assert_eq!(c(output_index), expected, "at {output_index:?}");
    }
}

fn check_dynamic<const A: usize, const B: usize, const C: usize>(
    a_shape: [usize; A],
    b_shape: [usize; B],
    c_shape: [usize; C],
) {
    let mut a = DynArray::<f64, A>::from_shape(a_shape);
    let mut b = DynArray::<f64, B>::from_shape(b_shape);
    for (n, index) in coordinates(a_shape).into_iter().enumerate() {
        a[index] = (n % 13) as f64 - 4.0;
    }
    for (n, index) in coordinates(b_shape).into_iter().enumerate() {
        b[index] = (n % 7) as f64 - 2.0;
    }
    let allocated: DynArray<f64, C> = rlst::dot_direct!(a, b);
    assert_product(
        a_shape,
        b_shape,
        allocated.shape(),
        |i| a[i],
        |i| b[i],
        |i| allocated[i],
    );

    let mut output = DynArray::<f64, C>::from_shape(c_shape);
    output.fill_with_value(f64::NAN);
    let pointer = output.data().unwrap().as_ptr();
    let (): () = rlst::dot_direct_into!(output, a, b);
    assert_eq!(output.data().unwrap().as_ptr(), pointer);
    assert_eq!(output.shape(), c_shape);
    assert_product(
        a_shape,
        b_shape,
        output.shape(),
        |i| a[i],
        |i| b[i],
        |i| output[i],
    );
}

#[test]
fn every_vector_matrix_and_rank_three_pairing() {
    check_dynamic([4], [4], [1]);
    check_dynamic([4], [4, 3], [3]);
    check_dynamic([4], [4, 3, 2], [3, 2]);
    check_dynamic([2, 4], [4], [2]);
    check_dynamic([2, 4], [4, 3], [2, 3]);
    check_dynamic([2, 4], [4, 3, 5], [2, 3, 5]);
    check_dynamic([2, 3, 4], [4], [2, 3]);
    check_dynamic([2, 3, 4], [4, 5], [2, 3, 5]);
    check_dynamic([2, 3, 4], [4, 5, 2], [2, 3, 5, 2]);
}

#[test]
fn arbitrary_rank_and_asymmetric_axes() {
    check_dynamic([2, 3, 1, 4], [4, 5, 2], [2, 3, 1, 5, 2]);
}

#[test]
fn empty_outputs_and_zero_length_contractions() {
    check_dynamic([0], [0], [1]);
    check_dynamic([2, 0], [0, 3], [2, 3]);
    check_dynamic([2, 3, 0], [0, 4, 2], [2, 3, 4, 2]);
    check_dynamic([0, 4], [4, 3], [0, 3]);
    check_dynamic([2, 4], [4, 0], [2, 0]);
    check_dynamic([2, 0, 4], [4, 3, 0], [2, 0, 3, 0]);
}

#[test]
fn static_arrays_and_lazy_expressions() {
    let mut a = rlst::rlst_static_array!(f64, [2, 3]);
    let mut b = rlst::rlst_static_array!(f64, [3, 4]);
    a.fill_from_iter((0..6).map(f64::from));
    b.fill_from_iter((0..12).map(|i| f64::from(i) - 3.0));
    let mut output = rlst::rlst_static_array!(f64, [2, 4]);
    output.fill_with_value(91.0);
    rlst::dot_direct_into!(output, a, b);
    assert_product(
        a.shape(),
        b.shape(),
        output.shape(),
        |i| a[i],
        |i| b[i],
        |i| output[i],
    );
    let lazy: DynArray<f64, 2> = rlst::dot_direct!(2.0_f64 * a.r() + a.r(), b.r() + b.r());
    assert_product(
        a.shape(),
        b.shape(),
        lazy.shape(),
        |i| 3.0 * a[i],
        |i| 2.0 * b[i],
        |i| lazy[i],
    );
}

#[test]
fn strided_transposed_inputs_and_mutable_subview() {
    let mut backing = StridedDynArray::<f64, 2>::from_shape_and_stride([4, 2], [2, 11]);
    backing.fill_from_iter((0..8).map(|i| f64::from(i) - 2.0));
    let a = backing.r().transpose();
    let mut b = StridedDynArray::<f64, 2>::row_major([4, 3]);
    b.fill_from_iter((0..12).map(f64::from));
    let allocated: DynArray<f64, 2> = rlst::dot_direct!(a, b);
    assert_product(
        a.shape(),
        b.shape(),
        allocated.shape(),
        |i| a.get_value(i).unwrap(),
        |i| b[i],
        |i| allocated[i],
    );
    let mut parent = DynArray::<f64, 2>::from_shape([5, 6]);
    parent.fill_with_value(-99.0);
    rlst::dot_direct_into!(parent.r_mut().into_subview([1, 2], [2, 3]), a, b);
    for i in coordinates(parent.shape()) {
        let expected = if (1..3).contains(&i[0]) && (2..5).contains(&i[1]) {
            allocated[[i[0] - 1, i[1] - 2]]
        } else {
            -99.0
        };
        assert_eq!(parent[i], expected);
    }
}

#[test]
fn complex_product_does_not_conjugate() {
    let mut a = DynArray::<c64, 1>::from_shape([2]);
    let mut b = DynArray::<c64, 1>::from_shape([2]);
    a[[0]] = c64::new(1.0, 2.0);
    a[[1]] = c64::new(-3.0, 1.0);
    b[[0]] = c64::new(2.0, -1.0);
    b[[1]] = c64::new(1.0, 4.0);
    let expected = a[[0]] * b[[0]] + a[[1]] * b[[1]];
    let allocated: DynArray<c64, 1> = rlst::dot_direct!(a, b);
    assert_eq!(allocated.shape(), [1]);
    assert_eq!(allocated[[0]], expected);
    let mut output = rlst::rlst_static_array!(c64, [1]);
    output.fill_with_value(c64::new(99.0, 88.0));
    rlst::dot_direct_into!(output, a, b);
    assert_eq!(output[[0]], expected);
}

#[test]
fn invalid_metadata_panics_before_mutating_output() {
    let a = DynArray::<f64, 2>::from_shape([2, 3]);
    let b = DynArray::<f64, 2>::from_shape([3, 4]);
    let mut wrong_shape = DynArray::<f64, 2>::from_shape([4, 2]);
    wrong_shape.fill_with_value(17.0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| rlst::dot_direct_into!(
            wrong_shape,
            a,
            b
        )))
        .is_err()
    );
    assert_eq!(wrong_shape.shape(), [4, 2]);
    assert!(wrong_shape.iter_value().all(|v| v == 17.0));

    let mut wrong_rank = DynArray::<f64, 1>::from_shape([8]);
    wrong_rank.fill_with_value(23.0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| rlst::dot_direct_into!(
            wrong_rank, a, b
        )))
        .is_err()
    );
    assert_eq!(wrong_rank.shape(), [8]);
    assert!(wrong_rank.iter_value().all(|v| v == 23.0));

    let bad_b = DynArray::<f64, 2>::from_shape([5, 4]);
    let mut output = DynArray::<f64, 2>::from_shape([2, 4]);
    output.fill_with_value(31.0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| rlst::dot_direct_into!(
            output, a, bad_b
        )))
        .is_err()
    );
    assert!(output.iter_value().all(|v| v == 31.0));
    assert!(
        catch_unwind(|| {
            let _: DynArray<f64, 2> = rlst::dot_direct!(a, bad_b);
        })
        .is_err()
    );
    assert!(
        catch_unwind(|| {
            let _: DynArray<f64, 1> = rlst::dot_direct!(a, b);
        })
        .is_err()
    );
}

#[test]
fn argument_expressions_are_evaluated_once_and_arrays_remain_owned() {
    let a_calls = Cell::new(0);
    let b_calls = Cell::new(0);
    let c_calls = Cell::new(0);
    let mut a = DynArray::<f64, 1>::from_shape([3]);
    let mut b = DynArray::<f64, 1>::from_shape([3]);
    a.fill_with_value(2.0);
    b.fill_with_value(3.0);
    let mut c = DynArray::<f64, 1>::from_shape([1]);
    rlst::dot_direct_into!(
        {
            c_calls.set(c_calls.get() + 1);
            c.r_mut()
        },
        {
            a_calls.set(a_calls.get() + 1);
            a.r()
        },
        {
            b_calls.set(b_calls.get() + 1);
            b.r()
        }
    );
    assert_eq!((a_calls.get(), b_calls.get(), c_calls.get()), (1, 1, 1));
    assert_eq!(c[[0]], 18.0);
    let allocated: DynArray<f64, 1> = rlst::dot_direct!(
        {
            a_calls.set(a_calls.get() + 1);
            a.r()
        },
        {
            b_calls.set(b_calls.get() + 1);
            b.r()
        }
    );
    assert_eq!((a_calls.get(), b_calls.get()), (2, 2));
    assert_eq!(allocated[[0]], 18.0);
    rlst::dot_direct_into!(c, a, b);
    assert_eq!((a[[0]], b[[0]], c[[0]]), (2.0, 3.0, 18.0));
}
