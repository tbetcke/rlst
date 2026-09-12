//! BLAS threading control.
//!
//! Thread control is available only when RLST is compiled with the feature matching the linked
//! backend. Without such a feature, [`set_blas_threads`] is a no-op.

#[allow(unused_imports)]
use std::ffi::c_int;

/// Return the number of physical CPU cores.
///
/// Note that physical CPU core information is only
/// support on Linux, Mac OS or Windows. On other systems
/// the logical number of cores is returned, which may be higher.
pub fn get_num_cores() -> usize {
    num_cpus::get()
}

#[cfg(feature = "blis_threading")]
extern "C" {
    fn bli_thread_set_num_threads(nthreads: c_int);
}

#[cfg(feature = "mkl_threading")]
extern "C" {
    fn mkl_set_num_threads(nthreads: c_int);
}

#[cfg(feature = "mkl_threading")]
extern "C" {
    fn openblas_set_num_threads(nthreads: c_int);
}

/// Request `nthreads` threads from the configured BLAS backend.
///
/// This changes process-global backend state and can affect concurrent BLAS calls. With none of
/// `blis_threading`, `mkl_threading`, or `openblas_threading` enabled, this function is a no-op.
/// Callers should pass a positive count; the meaning of zero is backend-defined.
#[allow(unused_variables)]
pub fn set_blas_threads(nthreads: usize) {
    #[cfg(feature = "blis_threading")]
    bli_thread_set_num_threads(nthreads as c_int);

    #[cfg(feature = "mkl_threading")]
    mkl_set_num_threads(nthreads as c_int);

    #[cfg(feature = "openblas_threading")]
    openblas_set_num_threads(nthreads as c_int);
}
