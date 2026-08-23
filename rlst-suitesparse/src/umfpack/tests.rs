use rlst::sparse::csr_mat::CsrMatrix;
use rlst::{DynArray, FromAij, c64};

use super::csc::CscMatrix;
use super::{UmfpackError, UmfpackFactorizationStatus, UmfpackFactorize};

fn test_matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_aij([2, 2], &[0, 0, 1, 1], &[0, 1, 0, 1], &[4.0, 1.0, 2.0, 3.0])
}

fn complex_test_matrix() -> CsrMatrix<c64> {
    CsrMatrix::from_aij(
        [2, 2],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[
            c64::new(4.0, 1.0),
            c64::new(1.0, -1.0),
            c64::new(2.0, 1.0),
            c64::new(3.0, -2.0),
        ],
    )
}

#[test]
fn converts_csr_to_csc() {
    let csc = CscMatrix::from_csr(&test_matrix()).unwrap();

    assert_eq!(csc.column_pointers, vec![0, 2, 4]);
    assert_eq!(csc.row_indices, vec![0, 1, 0, 1]);
    assert_eq!(csc.values, vec![4.0, 2.0, 1.0, 3.0]);
}

#[test]
fn combines_duplicate_csr_entries() {
    let matrix = CsrMatrix::new(
        [2, 2],
        DynArray::from_shape_and_vec([3], vec![0, 0, 1]),
        DynArray::from_shape_and_vec([3], vec![0, 2, 3]),
        DynArray::from_shape_and_vec([3], vec![1.0, 3.0, 2.0]),
    );
    let csc = CscMatrix::from_csr(&matrix).unwrap();

    assert_eq!(csc.column_pointers, vec![0, 1, 2]);
    assert_eq!(csc.row_indices, vec![0, 1]);
    assert_eq!(csc.values, vec![4.0, 2.0]);
}

#[test]
fn solves_a_real_system() {
    let numeric = test_matrix().factorize().unwrap();
    let solution = numeric.solve(&[1.0, 1.0]).unwrap();

    assert_eq!(numeric.status(), UmfpackFactorizationStatus::Success);
    assert!((solution[0] - 0.2).abs() < 1e-12);
    assert!((solution[1] - 0.2).abs() < 1e-12);
}

#[test]
fn rejects_incompatible_rhs() {
    let numeric = test_matrix().factorize().unwrap();

    assert_eq!(
        numeric.solve(&[1.0]),
        Err(UmfpackError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn solves_a_complex_system() {
    let numeric = complex_test_matrix().factorize().unwrap();
    let solution = numeric
        .solve(&[c64::new(4.0, 2.0), c64::new(5.0, -4.0)])
        .unwrap();

    assert_eq!(numeric.status(), UmfpackFactorizationStatus::Success);
    assert!((solution[0] - c64::new(1.0, 1.0)).norm() < 1e-12);
    assert!((solution[1] - c64::new(2.0, -1.0)).norm() < 1e-12);
}
