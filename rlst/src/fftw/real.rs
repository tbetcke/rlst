//! Precision dispatch: [`FftwReal`] for `f32` and `f64`.

use super::element::FftElement;
use crate::{RlstScalar, c32, c64};

pub(crate) mod sealed {
    use std::ffi::{c_char, c_int, c_uint};

    use num::Complex;

    /// Seals [`FftwReal`](super::FftwReal) and carries FFTW's functions for one precision.
    ///
    /// Every function except the `execute_*` functions must be called inside
    /// [`with_fftw`](crate::fftw::lock::with_fftw).
    pub trait Sealed: Sized {
        /// FFTW's opaque plan type of this precision, `fftw_plan_s` or `fftwf_plan_s`.
        type RawPlan;

        /// `fftw_plan_guru64_dft` or `fftwf_plan_guru64_dft`.
        ///
        /// # Safety
        ///
        /// The arguments must satisfy FFTW's conditions for the guru64 planner: `dims` and
        /// `howmany_dims` point to `rank` and `howmany_rank` dimensions, and `input` and
        /// `output` to buffers that cover the spans of those dimensions. The caller holds the
        /// FFTW lock.
        #[expect(clippy::too_many_arguments, reason = "mirrors FFTW's guru64 planner")]
        unsafe fn plan_guru64_dft(
            rank: c_int,
            dims: *const fftw_sys::fftw_iodim64,
            howmany_rank: c_int,
            howmany_dims: *const fftw_sys::fftw_iodim64,
            input: *mut Complex<Self>,
            output: *mut Complex<Self>,
            sign: c_int,
            flags: c_uint,
        ) -> *mut Self::RawPlan;

        /// `fftw_execute_dft` or `fftwf_execute_dft`.
        ///
        /// # Safety
        ///
        /// `plan` is a live plan, and `input` and `output` satisfy FFTW's new-array conditions
        /// for it.
        unsafe fn execute_dft(
            plan: *mut Self::RawPlan,
            input: *mut Complex<Self>,
            output: *mut Complex<Self>,
        );

        /// `fftw_plan_guru64_split_dft` or `fftwf_plan_guru64_split_dft`, which always plans a
        /// forward transform.
        ///
        /// # Safety
        ///
        /// The arguments must satisfy FFTW's conditions for the guru64 split planner: `dims`
        /// and `howmany_dims` point to `rank` and `howmany_rank` dimensions, in reals, and
        /// `ri`, `ii`, `ro` and `io` point to the real and imaginary parts of the first input
        /// and output elements of buffers that cover the spans of those dimensions. The
        /// caller holds the FFTW lock.
        #[expect(clippy::too_many_arguments, reason = "mirrors FFTW's guru64 planner")]
        unsafe fn plan_guru64_split_dft(
            rank: c_int,
            dims: *const fftw_sys::fftw_iodim64,
            howmany_rank: c_int,
            howmany_dims: *const fftw_sys::fftw_iodim64,
            ri: *mut Self,
            ii: *mut Self,
            ro: *mut Self,
            io: *mut Self,
            flags: c_uint,
        ) -> *mut Self::RawPlan;

        /// `fftw_execute_split_dft` or `fftwf_execute_split_dft`.
        ///
        /// # Safety
        ///
        /// `plan` is a live split plan, and `ri`, `ii`, `ro` and `io` satisfy FFTW's
        /// new-array conditions for it, the gaps `ii − ri` and `io − ro` included.
        unsafe fn execute_split_dft(
            plan: *mut Self::RawPlan,
            ri: *mut Self,
            ii: *mut Self,
            ro: *mut Self,
            io: *mut Self,
        );

        /// `fftw_plan_guru64_dft_r2c` or `fftwf_plan_guru64_dft_r2c`, which always plans a
        /// forward transform.
        ///
        /// # Safety
        ///
        /// The arguments must satisfy FFTW's conditions for the guru64 r2c planner: `dims` and
        /// `howmany_dims` point to `rank` and `howmany_rank` dimensions, with input strides in
        /// reals and output strides in complex elements, and `input` and `output` to buffers
        /// that cover the spans of those dimensions, the last transform dimension with `n`
        /// reals in and `n / 2 + 1` complex elements out. The caller holds the FFTW lock.
        unsafe fn plan_guru64_dft_r2c(
            rank: c_int,
            dims: *const fftw_sys::fftw_iodim64,
            howmany_rank: c_int,
            howmany_dims: *const fftw_sys::fftw_iodim64,
            input: *mut Self,
            output: *mut Complex<Self>,
            flags: c_uint,
        ) -> *mut Self::RawPlan;

        /// `fftw_execute_dft_r2c` or `fftwf_execute_dft_r2c`.
        ///
        /// # Safety
        ///
        /// `plan` is a live r2c plan, and `input` and `output` satisfy FFTW's new-array
        /// conditions for it.
        unsafe fn execute_dft_r2c(
            plan: *mut Self::RawPlan,
            input: *mut Self,
            output: *mut Complex<Self>,
        );

        /// `fftw_plan_guru64_dft_c2r` or `fftwf_plan_guru64_dft_c2r`, which always plans a
        /// backward transform.
        ///
        /// # Safety
        ///
        /// As for `plan_guru64_dft_r2c`, with the sides swapped: input strides in complex
        /// elements and output strides in reals, the last transform dimension with `n / 2 + 1`
        /// complex elements in and `n` reals out. The caller holds the FFTW lock.
        unsafe fn plan_guru64_dft_c2r(
            rank: c_int,
            dims: *const fftw_sys::fftw_iodim64,
            howmany_rank: c_int,
            howmany_dims: *const fftw_sys::fftw_iodim64,
            input: *mut Complex<Self>,
            output: *mut Self,
            flags: c_uint,
        ) -> *mut Self::RawPlan;

        /// `fftw_execute_dft_c2r` or `fftwf_execute_dft_c2r`.
        ///
        /// # Safety
        ///
        /// `plan` is a live c2r plan, and `input` and `output` satisfy FFTW's new-array
        /// conditions for it.
        unsafe fn execute_dft_c2r(
            plan: *mut Self::RawPlan,
            input: *mut Complex<Self>,
            output: *mut Self,
        );

        /// `fftw_destroy_plan` or `fftwf_destroy_plan`.
        ///
        /// # Safety
        ///
        /// `plan` is a live plan that is not used afterwards. The caller holds the FFTW lock.
        unsafe fn destroy_plan(plan: *mut Self::RawPlan);

        /// `fftw_sprint_plan` or `fftwf_sprint_plan`: a C string to be freed with `free`, or
        /// NULL if FFTW could not allocate it.
        ///
        /// # Safety
        ///
        /// `plan` is a live plan. The caller holds the FFTW lock.
        unsafe fn sprint_plan(plan: *mut Self::RawPlan) -> *mut c_char;

        /// `fftw_set_timelimit` or `fftwf_set_timelimit`.
        ///
        /// # Safety
        ///
        /// The caller holds the FFTW lock.
        unsafe fn set_timelimit(seconds: f64);
    }

    macro_rules! impl_sealed {
        ($real:ty, $prefix:ident) => {
            paste::paste! {
                impl Sealed for $real {
                    type RawPlan = fftw_sys::[<$prefix plan_s>];

                    unsafe fn plan_guru64_dft(
                        rank: c_int,
                        dims: *const fftw_sys::fftw_iodim64,
                        howmany_rank: c_int,
                        howmany_dims: *const fftw_sys::fftw_iodim64,
                        input: *mut Complex<Self>,
                        output: *mut Complex<Self>,
                        sign: c_int,
                        flags: c_uint,
                    ) -> *mut Self::RawPlan {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe {
                            fftw_sys::[<$prefix plan_guru64_dft>](
                                rank,
                                dims,
                                howmany_rank,
                                howmany_dims,
                                input,
                                output,
                                sign,
                                flags,
                            )
                        }
                    }

                    unsafe fn execute_dft(
                        plan: *mut Self::RawPlan,
                        input: *mut Complex<Self>,
                        output: *mut Complex<Self>,
                    ) {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix execute_dft>](plan, input, output) }
                    }

                    unsafe fn plan_guru64_split_dft(
                        rank: c_int,
                        dims: *const fftw_sys::fftw_iodim64,
                        howmany_rank: c_int,
                        howmany_dims: *const fftw_sys::fftw_iodim64,
                        ri: *mut Self,
                        ii: *mut Self,
                        ro: *mut Self,
                        io: *mut Self,
                        flags: c_uint,
                    ) -> *mut Self::RawPlan {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe {
                            fftw_sys::[<$prefix plan_guru64_split_dft>](
                                rank,
                                dims,
                                howmany_rank,
                                howmany_dims,
                                ri,
                                ii,
                                ro,
                                io,
                                flags,
                            )
                        }
                    }

                    unsafe fn execute_split_dft(
                        plan: *mut Self::RawPlan,
                        ri: *mut Self,
                        ii: *mut Self,
                        ro: *mut Self,
                        io: *mut Self,
                    ) {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix execute_split_dft>](plan, ri, ii, ro, io) }
                    }

                    unsafe fn plan_guru64_dft_r2c(
                        rank: c_int,
                        dims: *const fftw_sys::fftw_iodim64,
                        howmany_rank: c_int,
                        howmany_dims: *const fftw_sys::fftw_iodim64,
                        input: *mut Self,
                        output: *mut Complex<Self>,
                        flags: c_uint,
                    ) -> *mut Self::RawPlan {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe {
                            fftw_sys::[<$prefix plan_guru64_dft_r2c>](
                                rank,
                                dims,
                                howmany_rank,
                                howmany_dims,
                                input,
                                output,
                                flags,
                            )
                        }
                    }

                    unsafe fn execute_dft_r2c(
                        plan: *mut Self::RawPlan,
                        input: *mut Self,
                        output: *mut Complex<Self>,
                    ) {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix execute_dft_r2c>](plan, input, output) }
                    }

                    unsafe fn plan_guru64_dft_c2r(
                        rank: c_int,
                        dims: *const fftw_sys::fftw_iodim64,
                        howmany_rank: c_int,
                        howmany_dims: *const fftw_sys::fftw_iodim64,
                        input: *mut Complex<Self>,
                        output: *mut Self,
                        flags: c_uint,
                    ) -> *mut Self::RawPlan {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe {
                            fftw_sys::[<$prefix plan_guru64_dft_c2r>](
                                rank,
                                dims,
                                howmany_rank,
                                howmany_dims,
                                input,
                                output,
                                flags,
                            )
                        }
                    }

                    unsafe fn execute_dft_c2r(
                        plan: *mut Self::RawPlan,
                        input: *mut Complex<Self>,
                        output: *mut Self,
                    ) {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix execute_dft_c2r>](plan, input, output) }
                    }

                    unsafe fn destroy_plan(plan: *mut Self::RawPlan) {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix destroy_plan>](plan) }
                    }

                    unsafe fn sprint_plan(plan: *mut Self::RawPlan) -> *mut c_char {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix sprint_plan>](plan) }
                    }

                    unsafe fn set_timelimit(seconds: f64) {
                        // SAFETY: forwarded from the caller, see the trait's contract.
                        unsafe { fftw_sys::[<$prefix set_timelimit>](seconds) }
                    }
                }
            }
        };
    }

    impl_sealed!(f64, fftw_);
    impl_sealed!(f32, fftwf_);
}

/// A real type with an FFTW precision: `f32` (FFTW's `fftwf_` functions) or `f64` (`fftw_`).
///
/// [`Planner`](super::Planner) and [`Plan`](super::Plan) are generic over it, and dispatch
/// statically to the functions of that precision. Complex data uses RLST's
/// [`c32`](crate::c32) and [`c64`](crate::c64), which are FFTW's `fftwf_complex` and
/// `fftw_complex`. The trait is sealed.
///
/// ```
/// use rlst::fftw::FftwReal;
///
/// fn precision<T: FftwReal>() -> usize {
///     8 * std::mem::size_of::<T>()
/// }
///
/// assert_eq!(precision::<f32>(), 32);
/// assert_eq!(precision::<f64>(), 64);
/// ```
pub trait FftwReal: RlstScalar<Real = Self> + FftElement<Real = Self> + sealed::Sealed {}

impl FftwReal for f32 {}
impl FftwReal for f64 {}

// RLST's complex types are FFTW's: these functions compile only while `c64` is
// `fftw_complex` and `c32` is `fftwf_complex`, so the planner can pass RLST's complex data to
// FFTW without conversion.
const _: fn(c64) -> fftw_sys::fftw_complex = |value| value;
const _: fn(c32) -> fftw_sys::fftwf_complex = |value| value;
