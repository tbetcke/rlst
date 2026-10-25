//! Regression tests for zero-dimensional arrays in concrete and rank-generic code.

use rlst::{
    DynArray, EvaluateArray, MemoryLayout, StridedDynArray, ZeroDimArray,
    dense::array::AlignedDynArray, dense::array::empty_axis::AxisPosition, empty_array,
    rlst_dynamic_array, rlst_static_array,
};

#[test]
fn zero_rank_literal_and_expression() {
    let array: ZeroDimArray<f64> = rlst_dynamic_array!(f64, []);
    assert_eq!(array.shape(), [0usize; 0]);
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
    assert_eq!(array.r().shape(), [0usize; 0]);
    assert_eq!(array.r_mut().shape(), [0usize; 0]);
    assert_eq!(array.into_scalar(), 4.5);
}

#[test]
fn macro_gives_the_same_type_in_concrete_and_generic_code() {
    fn allocate<const NDIM: usize>(shape: [usize; NDIM]) -> DynArray<f64, NDIM> {
        rlst_dynamic_array!(f64, shape)
    }

    let concrete: DynArray<f64, 0> = rlst_dynamic_array!(f64, []);
    let generic: ZeroDimArray<f64> = allocate([]);
    assert_eq!(concrete.data(), generic.data());
    assert_eq!(generic.data().unwrap().len(), 1);
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

    let array = allocate::<i32, 0>([]);
    assert_eq!(array.number_of_elements(), 1);
    assert_eq!(array[[]], 0);

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

    let array: AlignedDynArray<f64, 2, 64> = rlst_dynamic_array!(f64, Default::default() | 64);
    assert_eq!(array.shape(), [0, 0]);

    let shape: [usize; 2] = Default::default();
    let array = rlst_dynamic_array!(f64, shape);
    let result_shape: [usize; 2] = array.shape();
    assert_eq!(result_shape, [0, 0]);
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
fn every_storage_accepts_rank_zero() {
    let shape: [usize; 0] = [];
    let mut aligned = rlst_dynamic_array!(f64, shape | 64);
    aligned[[]] = 1.5;
    assert_eq!(aligned.data().unwrap(), &[1.5]);
    assert_eq!(aligned.data().unwrap().as_ptr() as usize % 64, 0);

    let from_vec = DynArray::from_shape_and_vec([], vec![2.5]);
    assert_eq!(from_vec.into_scalar(), 2.5);

    let mut strided = StridedDynArray::<f64, 0>::from_shape_and_stride([], []);
    assert_eq!(strided.stride(), [0usize; 0]);
    *strided.scalar_mut() = 3.5;
    assert_eq!(strided.data().unwrap(), &[3.5]);

    let mut stack = rlst_static_array!(f64, []);
    stack[[]] = 4.5;
    assert_eq!(stack.shape(), [0usize; 0]);
    assert_eq!(*stack.scalar(), 4.5);

    let empty = empty_array::<f64, 0>();
    assert_eq!(empty.data().unwrap(), &[0.0]);
}

#[test]
fn zero_dimensional_operations() {
    let mut array = ZeroDimArray::from_scalar(3.0_f64);
    assert_eq!(array[[]], 3.0);
    assert_eq!(array.get([]), Some(&3.0));
    assert_eq!(array.get_value([]), Some(3.0));
    assert_eq!(array.memory_layout(), MemoryLayout::ColumnMajor);
    assert!(array.is_contiguous());

    assert_eq!(array.iter_value().collect::<Vec<_>>(), vec![3.0]);
    for value in array.iter_mut() {
        *value += 1.0;
    }
    assert_eq!(array.iter_ref().copied().collect::<Vec<_>>(), vec![4.0]);
    assert_eq!(array.diag_iter_value().collect::<Vec<_>>(), vec![4.0]);
    assert_eq!(array.trace(), 4.0);

    let tripled = (2.0_f64 * array.r() + array.r()).eval();
    assert_eq!(tripled.into_scalar(), 12.0);

    let mut copy = ZeroDimArray::<f64>::from_scalar(0.0);
    copy.fill_from(&array.r().transpose());
    assert_eq!(copy[[]], 4.0);
    assert_eq!(DynArray::new_from(&array).into_scalar(), 4.0);
    assert_eq!(array.r().into_flat().shape(), [1]);

    array.set_identity();
    assert_eq!(array.into_scalar(), 1.0);
}

#[test]
fn slices_reach_and_axes_leave_rank_zero() {
    let mut vec = DynArray::<f64, 1>::from_shape([3]);
    vec.fill_from_iter([1.0, 2.0, 3.0].into_iter());

    let element = vec.r().slice::<0>(0, 1);
    assert_eq!(element.shape(), [0usize; 0]);
    assert_eq!(*element.scalar(), 2.0);

    vec.r_mut().slice::<0>(0, 2)[[]] = 7.0;
    assert_eq!(vec.iter_value().collect::<Vec<_>>(), vec![1.0, 2.0, 7.0]);

    let mut scalar = ZeroDimArray::from_scalar(5.0_f64);
    for position in [AxisPosition::Front, AxisPosition::Back] {
        let vec = scalar.r().insert_empty_axis::<1>(position);
        assert_eq!(vec.shape(), [1]);
        assert_eq!(vec[[0]], 5.0);
    }
    scalar.r_mut().insert_empty_axis::<1>(AxisPosition::Back)[[0]] = 6.0;
    assert_eq!(scalar.into_scalar(), 6.0);
}

#[test]
fn rank_generic_code_covers_rank_zero() {
    fn sum_of_squares<const NDIM: usize>(shape: [usize; NDIM]) -> f64 {
        let mut array = DynArray::<f64, NDIM>::from_shape(shape);
        array.fill_with_value(3.0);
        let squared = (array.r() * array.r()).eval();
        squared.iter_value().sum()
    }

    assert_eq!(sum_of_squares([2, 2]), 36.0);
    assert_eq!(sum_of_squares([4]), 36.0);
    assert_eq!(sum_of_squares([]), 9.0);
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
