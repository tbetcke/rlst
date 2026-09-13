//! Regression tests for scalar-only arrays and call-site rank dispatch.

use rlst::{DynArray, ZeroDimArray, rlst_dynamic_array};

#[test]
fn zero_rank_literal_and_expression() {
    let array: ZeroDimArray<f64> = rlst_dynamic_array!(f64, []);
    assert_eq!(array.shape(), []);
    assert_eq!(array.into_scalar(), 0.0);

    let mut calls = 0;
    let mut shape = || {
        calls += 1;
        []
    };
    let array: ZeroDimArray<i32> = rlst_dynamic_array!(i32, shape());
    assert_eq!(calls, 1);
    assert_eq!(array.into_scalar(), 0);

    let shape: [usize; 0] = [];
    let mut array = rlst_dynamic_array!(f64, shape);
    assert_eq!(array.len(), 1);
    assert!(!array.is_empty());
    *array.scalar_mut() = 4.5;
    assert_eq!(*array.scalar(), 4.5);
    assert_eq!(array.r().shape(), []);
    assert_eq!(array.r_mut().shape(), []);
    assert_eq!(array.into_scalar(), 4.5);
}

#[test]
fn generic_shape_contract() {
    fn allocate<Item: Copy + Default, const NDIM: usize>(
        shape: [usize; NDIM],
    ) -> DynArray<Item, NDIM> {
        rlst_dynamic_array!(Item, shape)
    }

    let empty: DynArray<f64, 1> = rlst_dynamic_array!(f64, [0]);
    assert_eq!(empty.shape(), [0]);
    assert!(empty.is_empty());

    let array = allocate::<i32, 2>([2, 3]);
    assert_eq!(array.shape(), [2, 3]);
    assert_eq!(array[[1, 2]], 0);

    let array: DynArray<i32, 257> = rlst_dynamic_array!(i32, [1; 257]);
    assert_eq!(array.number_of_elements(), 1);
    assert_eq!(allocate::<i32, 257>([1; 257]).number_of_elements(), 1);
}

#[test]
fn contextual_shape_inference() {
    fn shape<const NDIM: usize>() -> [usize; NDIM] {
        [1; NDIM]
    }

    let array: DynArray<f64, 2> = rlst_dynamic_array!(f64, shape());
    assert_eq!(array.shape(), [1, 1]);
    let array: DynArray<f64, 2> = rlst_dynamic_array!(f64, Default::default());
    assert_eq!(array.shape(), [0, 0]);
    let array: ZeroDimArray<f64> = rlst_dynamic_array!(f64, Default::default());
    assert_eq!(array.into_scalar(), 0.0);

    let array: rlst::dense::array::AlignedDynArray<f64, 2, 64> =
        rlst_dynamic_array!(f64, Default::default() | 64);
    assert_eq!(array.shape(), [0, 0]);

    let shape: [usize; 2] = Default::default();
    let array = rlst_dynamic_array!(f64, shape);
    let result_shape: [usize; 2] = array.shape();
    assert_eq!(result_shape, [0, 0]);
}

#[test]
#[should_panic(expected = "BaseArray requires a positive rank")]
fn unresolved_generic_zero_rank_is_rejected() {
    fn allocate<const NDIM: usize>(shape: [usize; NDIM]) -> DynArray<f64, NDIM> {
        rlst_dynamic_array!(f64, shape)
    }
    let _ = allocate([]);
}

#[test]
fn aligned_positive_rank_evaluates_shape_once() {
    let mut calls = 0;
    let mut shape = || {
        calls += 1;
        [2, 3]
    };
    let array = rlst_dynamic_array!(f64, shape() | 64);
    assert_eq!(calls, 1);
    assert_eq!(array.shape(), [2, 3]);
    assert_eq!(array[[1, 2]], 0.0);
    assert_eq!(array.data().unwrap().as_ptr() as usize % 64, 0);
}

#[test]
#[should_panic(expected = "Aligned zero-dimensional arrays are not supported")]
fn aligned_zero_rank_is_rejected() {
    let shape: [usize; 0] = [];
    let _ = rlst_dynamic_array!(f64, shape | 64);
}

#[test]
#[should_panic(expected = "BaseArray requires a positive rank")]
fn ordinary_zero_rank_storage_is_rejected() {
    let _ = DynArray::from_shape_and_vec([], vec![1.0]);
}

#[test]
#[should_panic(expected = "StridedBaseArray requires a positive rank")]
fn strided_zero_rank_storage_is_rejected() {
    let _ = rlst::StridedDynArray::<f64, 0>::from_shape_and_stride([], []);
}

#[test]
fn generic_item_initialization_uses_default() {
    #[derive(Clone, Copy)]
    struct Item(i32);

    impl Default for Item {
        fn default() -> Self {
            Self(42)
        }
    }

    fn scalar<Item: Copy + Default>() -> ZeroDimArray<Item> {
        rlst_dynamic_array!(Item, [])
    }

    assert_eq!(scalar::<Item>().scalar().0, 42);
}
