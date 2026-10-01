//! BLAS threading control.
//!
//! RLST does not select the BLAS library of a binary, so [`set_blas_threads`] and
//! [`get_blas_threads`] find the linked backend themselves. The supported backends are listed in
//! [`BlasBackend`].
//!
//! On Linux and macOS, the backends' thread-control functions are looked up at runtime among the
//! libraries loaded into the process. This finds dynamically linked OpenBLAS, Intel MKL, BLIS and
//! FlexiBLAS, as well as Apple Accelerate on macOS 15 or later, without enabling any feature.
//! Statically linked libraries are not visible to this lookup. For them, enable the feature
//! matching the backend (`openblas_threading`, `mkl_threading` or `blis_threading`), which links
//! the thread-control functions directly. Each enabled feature requires its library to be linked
//! into the final binary. On other targets, only these features are available.
//!
//! The functions are resolved on first use and cached for the lifetime of the process, so a BLAS
//! library loaded later with `dlopen` is not detected.
//!
//! Backends can also be configured before startup through environment variables such as
//! `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` or
//! `OMP_NUM_THREADS`.

use std::ffi::{CStr, c_int, c_void};
use std::sync::OnceLock;

use thiserror::Error;

/// Return the number of logical CPUs available to the process.
///
/// This counts hardware threads rather than physical cores and, on Linux, respects the CPU
/// affinity mask and cgroup quotas of the process.
pub fn get_num_cores() -> usize {
    num_cpus::get()
}

/// A BLAS library whose threading RLST can control.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlasBackend {
    /// OpenBLAS, including ILP64 builds with the `64_` symbol suffix.
    OpenBlas,
    /// Intel oneAPI Math Kernel Library.
    Mkl,
    /// BLIS, including AMD AOCL-BLIS.
    Blis,
    /// FlexiBLAS, which forwards the setting to its selected backend.
    FlexiBlas,
    /// Apple Accelerate (macOS 15 or later).
    Accelerate,
}

/// Threading state reported by a BLAS backend.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlasThreads {
    /// The backend uses up to this many threads.
    Count(usize),
    /// The backend runs single-threaded (Accelerate).
    SingleThreaded,
    /// The backend chooses the number of threads itself (Accelerate).
    BackendManaged,
    /// The backend did not report a valid state, for example BLIS before a thread count has
    /// been set.
    Unknown,
}

/// The threading state of one BLAS backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlasThreadSetting {
    /// The backend.
    pub backend: BlasBackend,
    /// Its threading state.
    pub threads: BlasThreads,
}

/// Error returned by [`set_blas_threads`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BlasThreadingError {
    /// A thread count of zero was requested.
    #[error("the number of BLAS threads must be positive")]
    ZeroThreads,
    /// The requested thread count does not fit into a C `int`.
    #[error("{0} BLAS threads requested, more than a C int can represent")]
    TooManyThreads(usize),
    /// No supported BLAS backend was found.
    #[error(
        "no BLAS backend with thread control was found; statically linked backends require \
         the matching `*_threading` feature"
    )]
    NoBackendFound,
    /// A backend rejected the setting with the given status code.
    #[error("{backend:?} rejected the threading setting with status {code}")]
    Rejected {
        /// The backend that rejected the setting.
        backend: BlasBackend,
        /// The status code it returned.
        code: i32,
    },
}

/// Request `nthreads` threads from every detected BLAS backend.
///
/// On success, the threading state of each detected backend is returned, read back after
/// setting it. It can differ from the request: OpenBLAS caps the count at its compiled maximum,
/// and sequential builds of OpenBLAS or MKL report a single thread.
///
/// Accelerate only distinguishes single-threaded from multi-threaded execution: `nthreads == 1`
/// selects single-threaded execution and any larger count lets Accelerate choose the number of
/// threads ([`BlasThreads::BackendManaged`]). The Accelerate setting applies to the **calling
/// thread** only. For the other backends, the setting is process-global, so it should not be
/// changed while other threads run BLAS routines.
///
/// See the [module documentation](self) for how backends are detected.
///
/// # Errors
///
/// - [`BlasThreadingError::ZeroThreads`] or [`BlasThreadingError::TooManyThreads`] if
///   `nthreads` is invalid. No backend is changed.
/// - [`BlasThreadingError::NoBackendFound`] if no supported backend was detected.
/// - [`BlasThreadingError::Rejected`] if a backend rejected the setting. Backends processed
///   before it keep the new setting.
pub fn set_blas_threads(nthreads: usize) -> Result<Vec<BlasThreadSetting>, BlasThreadingError> {
    let count = validate_thread_count(nthreads)?;
    let backends = backends();
    if backends.is_empty() {
        return Err(BlasThreadingError::NoBackendFound);
    }
    for backend in backends {
        backend.set(count)?;
    }
    Ok(backends.iter().map(Backend::get).collect())
}

/// Return the threading state of every detected BLAS backend.
///
/// The result is empty if no supported backend was detected. For Accelerate, the state of the
/// calling thread is reported. See [`set_blas_threads`].
pub fn get_blas_threads() -> Vec<BlasThreadSetting> {
    backends().iter().map(Backend::get).collect()
}

/// Check that `nthreads` is positive and fits into a C `int`.
fn validate_thread_count(nthreads: usize) -> Result<c_int, BlasThreadingError> {
    if nthreads == 0 {
        return Err(BlasThreadingError::ZeroThreads);
    }
    c_int::try_from(nthreads).map_err(|_| BlasThreadingError::TooManyThreads(nthreads))
}

/// Convert a thread count reported by a backend.
fn reported_count(count: i64) -> BlasThreads {
    usize::try_from(count)
        .ok()
        .filter(|&count| count > 0)
        .map_or(BlasThreads::Unknown, BlasThreads::Count)
}

/// `BLAS_THREADING_MULTI_THREADED` from Accelerate's `vecLib/thread_api.h`.
const ACCELERATE_MULTI_THREADED: u32 = 0;
/// `BLAS_THREADING_SINGLE_THREADED` from Accelerate's `vecLib/thread_api.h`.
const ACCELERATE_SINGLE_THREADED: u32 = 1;

/// Integer width of BLIS's `dim_t`, which is configured when BLIS is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlisInt {
    I32,
    I64,
}

impl BlisInt {
    /// Interpret the value of `bli_info_get_int_type_size`, if available.
    ///
    /// BLIS reports 32 or 64 bits. Anything else, including a BLIS too old to report it,
    /// selects BLIS's default of 64 bits.
    fn from_type_size(bits: Option<c_int>) -> Self {
        match bits {
            Some(32) => BlisInt::I32,
            _ => BlisInt::I64,
        }
    }
}

/// Typed thread-control functions of a detected backend.
#[derive(Clone, Copy)]
enum Backend {
    /// `void set(int)` and `int get(void)`: OpenBLAS, MKL, FlexiBLAS, and BLIS with 32-bit
    /// `dim_t`.
    Int {
        backend: BlasBackend,
        set: unsafe extern "C" fn(c_int),
        get: unsafe extern "C" fn() -> c_int,
    },
    /// BLIS with 64-bit `dim_t`.
    Blis64 {
        set: unsafe extern "C" fn(i64),
        get: unsafe extern "C" fn() -> i64,
    },
    /// Accelerate's `int BLASSetThreading(enum BLAS_THREADING)` and
    /// `enum BLAS_THREADING BLASGetThreading(void)`, with an `unsigned int` enum.
    Accelerate {
        set: unsafe extern "C" fn(u32) -> c_int,
        get: unsafe extern "C" fn() -> u32,
    },
}

impl Backend {
    fn backend(&self) -> BlasBackend {
        match *self {
            Backend::Int { backend, .. } => backend,
            Backend::Blis64 { .. } => BlasBackend::Blis,
            Backend::Accelerate { .. } => BlasBackend::Accelerate,
        }
    }

    fn set(&self, count: c_int) -> Result<(), BlasThreadingError> {
        match *self {
            // SAFETY: `set` has the prototype checked in `resolve`, and `count` is positive.
            Backend::Int { set, .. } => unsafe { set(count) },
            // SAFETY: as above.
            Backend::Blis64 { set, .. } => unsafe { set(i64::from(count)) },
            Backend::Accelerate { set, .. } => {
                let mode = if count == 1 {
                    ACCELERATE_SINGLE_THREADED
                } else {
                    ACCELERATE_MULTI_THREADED
                };
                // SAFETY: `set` has the prototype checked in `resolve`, and `mode` is a valid
                // `BLAS_THREADING` value.
                let code = unsafe { set(mode) };
                if code != 0 {
                    return Err(BlasThreadingError::Rejected {
                        backend: BlasBackend::Accelerate,
                        code,
                    });
                }
            }
        }
        Ok(())
    }

    fn get(&self) -> BlasThreadSetting {
        let threads = match *self {
            // SAFETY: `get` has the prototype checked in `resolve` and takes no arguments.
            Backend::Int { get, .. } => reported_count(i64::from(unsafe { get() })),
            // SAFETY: as above.
            Backend::Blis64 { get, .. } => reported_count(unsafe { get() }),
            // SAFETY: as above.
            Backend::Accelerate { get, .. } => match unsafe { get() } {
                ACCELERATE_SINGLE_THREADED => BlasThreads::SingleThreaded,
                ACCELERATE_MULTI_THREADED => BlasThreads::BackendManaged,
                _ => BlasThreads::Unknown,
            },
        };
        BlasThreadSetting {
            backend: self.backend(),
            threads,
        }
    }
}

/// The detected backends, resolved on first use.
fn backends() -> &'static [Backend] {
    static BACKENDS: OnceLock<Vec<Backend>> = OnceLock::new();
    BACKENDS.get_or_init(resolve)
}

/// Addresses of a backend's setter and getter.
struct Symbols {
    set: *mut c_void,
    get: *mut c_void,
}

/// Reinterpret a non-null function address as the function pointer type `F`.
///
/// # Safety
///
/// `ptr` must be the address of a C function whose prototype matches `F`.
unsafe fn as_fn<F: Copy>(ptr: *mut c_void) -> F {
    const { assert!(size_of::<F>() == size_of::<*mut c_void>()) };
    debug_assert!(!ptr.is_null());
    // SAFETY: `F` is pointer-sized, and the caller guarantees that `ptr` is a function of type
    // `F`.
    unsafe { std::mem::transmute_copy(&ptr) }
}

fn resolve() -> Vec<Backend> {
    let mut backends = Vec::new();

    let int_backends = [
        (BlasBackend::FlexiBlas, flexiblas_symbols()),
        (BlasBackend::OpenBlas, openblas_symbols()),
        (BlasBackend::Mkl, mkl_symbols()),
    ];
    for (backend, symbols) in int_backends {
        if let Some(Symbols { set, get }) = symbols {
            // SAFETY: the symbols are `void set(int)` and `int get(void)` in the C APIs of
            // FlexiBLAS (`flexiblas_set_num_threads`), OpenBLAS (`openblas_set_num_threads`)
            // and MKL (`MKL_Set_Num_Threads`, `MKL_Get_Max_Threads`).
            backends.push(unsafe {
                Backend::Int {
                    backend,
                    set: as_fn(set),
                    get: as_fn(get),
                }
            });
        }
    }

    if let Some((Symbols { set, get }, int_type_size)) = blis_symbols() {
        let bits = (!int_type_size.is_null()).then(|| {
            // SAFETY: `bli_info_get_int_type_size` is `gint_t f(void)`. On the supported 64-bit
            // ABIs, the low 32 bits of a 64-bit integer result are a valid 32-bit result, and
            // the reported value is 0, 32 or 64.
            unsafe { as_fn::<unsafe extern "C" fn() -> c_int>(int_type_size)() }
        });
        // SAFETY: `bli_thread_set_num_threads` is `void f(dim_t)` and
        // `bli_thread_get_num_threads` is `dim_t f(void)`, with `dim_t` of the width BLIS
        // reports.
        backends.push(unsafe {
            match BlisInt::from_type_size(bits) {
                BlisInt::I32 => Backend::Int {
                    backend: BlasBackend::Blis,
                    set: as_fn(set),
                    get: as_fn(get),
                },
                BlisInt::I64 => Backend::Blis64 {
                    set: as_fn(set),
                    get: as_fn(get),
                },
            }
        });
    }

    if let Some(Symbols { set, get }) = lookup_pair(&[(c"BLASSetThreading", c"BLASGetThreading")]) {
        // SAFETY: Accelerate's `vecLib/thread_api.h` declares
        // `int BLASSetThreading(const enum BLAS_THREADING)` and
        // `enum BLAS_THREADING BLASGetThreading(void)` with `enum BLAS_THREADING : unsigned int`.
        backends.push(unsafe {
            Backend::Accelerate {
                set: as_fn(set),
                get: as_fn(get),
            }
        });
    }

    backends
}

/// Find a symbol among the libraries loaded into the process.
///
/// Returns null if the symbol is not found or runtime lookup is not supported on this target.
fn lookup(name: &CStr) -> *mut c_void {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        dl::lookup(name)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = name;
        std::ptr::null_mut()
    }
}

/// Return the first setter/getter pair whose symbols are both found.
fn lookup_pair(candidates: &[(&CStr, &CStr)]) -> Option<Symbols> {
    candidates.iter().find_map(|&(set, get)| {
        let (set, get) = (lookup(set), lookup(get));
        (!set.is_null() && !get.is_null()).then_some(Symbols { set, get })
    })
}

fn flexiblas_symbols() -> Option<Symbols> {
    lookup_pair(&[(c"flexiblas_set_num_threads", c"flexiblas_get_num_threads")])
}

fn openblas_symbols() -> Option<Symbols> {
    #[cfg(feature = "openblas_threading")]
    {
        Some(Symbols {
            set: linked::openblas_set_num_threads as *mut c_void,
            get: linked::openblas_get_num_threads as *mut c_void,
        })
    }
    #[cfg(not(feature = "openblas_threading"))]
    {
        lookup_pair(&[
            (c"openblas_set_num_threads", c"openblas_get_num_threads"),
            (
                c"openblas_set_num_threads64_",
                c"openblas_get_num_threads64_",
            ),
        ])
    }
}

fn mkl_symbols() -> Option<Symbols> {
    #[cfg(feature = "mkl_threading")]
    {
        Some(Symbols {
            set: linked::MKL_Set_Num_Threads as *mut c_void,
            get: linked::MKL_Get_Max_Threads as *mut c_void,
        })
    }
    #[cfg(not(feature = "mkl_threading"))]
    {
        lookup_pair(&[(c"MKL_Set_Num_Threads", c"MKL_Get_Max_Threads")])
    }
}

/// The BLIS setter and getter, and the address of `bli_info_get_int_type_size` (null if absent).
fn blis_symbols() -> Option<(Symbols, *mut c_void)> {
    #[cfg(feature = "blis_threading")]
    {
        Some((
            Symbols {
                set: linked::bli_thread_set_num_threads as *mut c_void,
                get: linked::bli_thread_get_num_threads as *mut c_void,
            },
            linked::bli_info_get_int_type_size as *mut c_void,
        ))
    }
    #[cfg(not(feature = "blis_threading"))]
    {
        lookup_pair(&[(c"bli_thread_set_num_threads", c"bli_thread_get_num_threads")])
            .map(|symbols| (symbols, lookup(c"bli_info_get_int_type_size")))
    }
}

/// Runtime symbol lookup with `dlsym`, provided by the C library on Linux and macOS.
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod dl {
    use std::ffi::{CStr, c_char, c_void};

    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// `RTLD_DEFAULT` searches the global symbol scope of the process.
    #[cfg(target_os = "linux")]
    const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();
    #[cfg(target_os = "macos")]
    const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

    pub(super) fn lookup(name: &CStr) -> *mut c_void {
        // SAFETY: `name` is a valid NUL-terminated string and `RTLD_DEFAULT` is a valid pseudo
        // handle. `dlsym` returns null if the symbol is absent.
        unsafe { dlsym(RTLD_DEFAULT, name.as_ptr()) }
    }
}

/// Thread-control functions linked directly by the `*_threading` features.
///
/// The signatures are only used to take the functions' addresses; `resolve` assigns the
/// prototypes. BLIS functions are declared with 64-bit `dim_t`.
#[cfg(any(
    feature = "openblas_threading",
    feature = "mkl_threading",
    feature = "blis_threading"
))]
mod linked {
    #[cfg(any(feature = "openblas_threading", feature = "mkl_threading"))]
    use std::ffi::c_int;

    #[cfg(feature = "openblas_threading")]
    unsafe extern "C" {
        pub(super) fn openblas_set_num_threads(nthreads: c_int);
        pub(super) fn openblas_get_num_threads() -> c_int;
    }

    #[cfg(feature = "mkl_threading")]
    unsafe extern "C" {
        pub(super) fn MKL_Set_Num_Threads(nthreads: c_int);
        pub(super) fn MKL_Get_Max_Threads() -> c_int;
    }

    #[cfg(feature = "blis_threading")]
    unsafe extern "C" {
        pub(super) fn bli_thread_set_num_threads(nthreads: i64);
        pub(super) fn bli_thread_get_num_threads() -> i64;
        pub(super) fn bli_info_get_int_type_size() -> i64;
    }
}

#[cfg(test)]
mod test {
    use super::{BlasThreadingError, BlasThreads, BlisInt, reported_count, set_blas_threads};

    #[test]
    fn test_invalid_thread_counts() {
        assert_eq!(set_blas_threads(0), Err(BlasThreadingError::ZeroThreads));

        #[cfg(target_pointer_width = "64")]
        {
            let too_many = std::ffi::c_int::MAX as usize + 1;
            assert_eq!(
                set_blas_threads(too_many),
                Err(BlasThreadingError::TooManyThreads(too_many))
            );
        }
    }

    #[test]
    fn test_reported_count() {
        assert_eq!(reported_count(4), BlasThreads::Count(4));
        assert_eq!(reported_count(0), BlasThreads::Unknown);
        assert_eq!(reported_count(-1), BlasThreads::Unknown);
    }

    #[test]
    fn test_blis_int_width() {
        assert_eq!(BlisInt::from_type_size(Some(32)), BlisInt::I32);
        assert_eq!(BlisInt::from_type_size(Some(64)), BlisInt::I64);
        assert_eq!(BlisInt::from_type_size(Some(0)), BlisInt::I64);
        assert_eq!(BlisInt::from_type_size(None), BlisInt::I64);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_accelerate_threading() {
        use super::{BlasBackend, get_blas_threads, lookup};

        let accelerate = |settings: Vec<super::BlasThreadSetting>| {
            settings
                .into_iter()
                .find(|setting| setting.backend == BlasBackend::Accelerate)
                .map(|setting| setting.threads)
        };

        // `BLASSetThreading` exists from macOS 15.
        let initial = accelerate(get_blas_threads());
        assert_eq!(initial.is_some(), !lookup(c"BLASSetThreading").is_null());
        if initial.is_none() {
            return;
        }

        // Accelerate's setting is thread-local, so this does not affect concurrent tests.
        assert_eq!(
            accelerate(set_blas_threads(1).unwrap()),
            Some(BlasThreads::SingleThreaded)
        );
        assert_eq!(
            accelerate(get_blas_threads()),
            Some(BlasThreads::SingleThreaded)
        );
        assert_eq!(
            accelerate(set_blas_threads(4).unwrap()),
            Some(BlasThreads::BackendManaged)
        );
    }
}
