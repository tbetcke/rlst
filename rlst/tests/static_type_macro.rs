use rlst::{rlst_static_array, rlst_static_type};

#[test]
fn static_type_matches_static_array_constructor() {
    let array: rlst_static_type!(f64, 2, 3) = rlst_static_array!(f64, [2, 3]);

    assert_eq!(array.shape(), [2, 3]);
}
