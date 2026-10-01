//! Regression tests for BLAS thread control against the linked backend.
//!
//! OpenBLAS's thread count is process-global, so these tests run in their own test binary and
//! change the setting from a single test function.

extern crate blas_src;

#[cfg(target_os = "linux")]
#[test]
fn openblas_thread_count_round_trips() {
    use rlst::DynArray;
    use rlst::threading::{
        BlasBackend, BlasThreadSetting, BlasThreads, get_blas_threads, set_blas_threads,
    };

    // Use BLAS so that the shared OpenBLAS library is kept by `--as-needed` linking.
    let a = DynArray::from_shape_and_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = DynArray::from_shape_and_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let product: DynArray<f64, 2> = rlst::dot_blas!(a, b);
    assert_eq!(product.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let openblas = |settings: Vec<BlasThreadSetting>| {
        settings
            .into_iter()
            .find(|setting| setting.backend == BlasBackend::OpenBlas)
            .map(|setting| setting.threads)
    };

    let Some(BlasThreads::Count(initial)) = openblas(get_blas_threads()) else {
        panic!("OpenBLAS was not detected: {:?}", get_blas_threads());
    };

    assert_eq!(
        openblas(set_blas_threads(1).unwrap()),
        Some(BlasThreads::Count(1))
    );
    assert_eq!(openblas(get_blas_threads()), Some(BlasThreads::Count(1)));

    // OpenBLAS caps the request at its compiled maximum, and sequential builds stay at one.
    let Some(BlasThreads::Count(two)) = openblas(set_blas_threads(2).unwrap()) else {
        panic!("OpenBLAS did not report a thread count");
    };
    assert!((1..=2).contains(&two));

    assert_eq!(
        openblas(set_blas_threads(initial).unwrap()),
        Some(BlasThreads::Count(initial))
    );
}
