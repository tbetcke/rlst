use rlst::sparse::csr_mat::CsrMatrix;
use rlst::{DynArray, FromAij, StridedDynArray, c64, rlst_dynamic_array};

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
fn rejects_empty_matrix() {
    let matrix: CsrMatrix<f64> = CsrMatrix::new(
        [0, 0],
        DynArray::from_shape_and_vec([0], vec![]),
        DynArray::from_shape_and_vec([1], vec![0]),
        DynArray::from_shape_and_vec([0], vec![]),
    );

    assert!(matches!(matrix.factorize(), Err(UmfpackError::EmptyMatrix)));
}

#[test]
fn rejects_nonsquare_matrix() {
    let matrix = CsrMatrix::from_aij([1, 2], &[0], &[1], &[1.0]);

    assert!(matches!(
        matrix.factorize(),
        Err(UmfpackError::NonSquareMatrix {
            rows: 1,
            columns: 2,
        })
    ));
}

#[test]
fn reports_singular_factorization_warning() {
    let matrix = CsrMatrix::from_aij([2, 2], &[0, 1], &[0, 0], &[1.0, 2.0]);
    let numeric = matrix.factorize().unwrap();

    assert_eq!(numeric.status(), UmfpackFactorizationStatus::SingularMatrix);
}

#[test]
fn solves_a_real_system() {
    let numeric = test_matrix().factorize().unwrap();
    let mut rhs = rlst_dynamic_array!(f64, [2]);
    rhs[[0]] = 1.0;
    rhs[[1]] = 1.0;
    let solution = numeric.solve(&rhs).unwrap();

    assert_eq!(numeric.status(), UmfpackFactorizationStatus::Success);
    assert!((solution[[0]] - 0.2).abs() < 1e-12);
    assert!((solution[[1]] - 0.2).abs() < 1e-12);
}

#[test]
fn rejects_incompatible_rhs() {
    let numeric = test_matrix().factorize().unwrap();
    let rhs = rlst_dynamic_array!(f64, [1]);

    assert!(matches!(
        numeric.solve(&rhs),
        Err(UmfpackError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn solves_into_a_mutable_array() {
    let numeric = test_matrix().factorize().unwrap();
    let mut rhs = rlst_dynamic_array!(f64, [2]);
    let mut solution = rlst_dynamic_array!(f64, [2]);
    rhs[[0]] = 1.0;
    rhs[[1]] = 1.0;

    numeric.solve_into(&rhs, &mut solution).unwrap();

    assert!((solution[[0]] - 0.2).abs() < 1e-12);
    assert!((solution[[1]] - 0.2).abs() < 1e-12);
}

#[test]
fn rejects_non_contiguous_rhs() {
    let numeric = test_matrix().factorize().unwrap();
    let rhs = StridedDynArray::from_shape_and_stride([2], [2]);

    assert!(matches!(
        numeric.solve(&rhs),
        Err(UmfpackError::NonContiguousArray("right-hand side"))
    ));
}

#[test]
fn rejects_non_contiguous_solution() {
    let numeric = test_matrix().factorize().unwrap();
    let rhs = rlst_dynamic_array!(f64, [2]);
    let mut solution = StridedDynArray::from_shape_and_stride([2], [2]);

    assert!(matches!(
        numeric.solve_into(&rhs, &mut solution),
        Err(UmfpackError::NonContiguousArray("solution"))
    ));
}

#[test]
fn solves_a_complex_system() {
    let numeric = complex_test_matrix().factorize().unwrap();
    let mut rhs = rlst_dynamic_array!(c64, [2]);
    rhs[[0]] = c64::new(4.0, 2.0);
    rhs[[1]] = c64::new(5.0, -4.0);
    let solution = numeric.solve(&rhs).unwrap();

    assert_eq!(numeric.status(), UmfpackFactorizationStatus::Success);
    assert!((solution[[0]] - c64::new(1.0, 1.0)).norm() < 1e-12);
    assert!((solution[[1]] - c64::new(2.0, -1.0)).norm() < 1e-12);
}
