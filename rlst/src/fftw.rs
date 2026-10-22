//! Implementation of fft using fftw
//!
//! FFTW uses plans to execute an FFT computation. Within FFTW there are two steps
//! to get the FFT of a given array.
//!
//! 1. Generate a plan for the desired FFT
//! 2. Execute the FFT
//!
//! FFTW supports FFT computations on one-, two-, and three-dimensional input arrays.
//! Enable the `fftw_system` feature, the only backend, which implies `fftw`. The system
//! must provide both single- and double-precision libraries (`fftw3f` and `fftw3`), version
//! 3.3.4 or later, normally discovered through `pkg-config`.
//!
//! ## Complex-to-Complex transform
//!
//! The following gives an example of computing the Complex-to-Complex transform
//! of a given array `arr`.
//! ```
//! use rlst::{rlst_dynamic_array, fftw::{FftwPlanFlags, C2CDirection, C2CPlan, FftPlanInplace},
//!     EvaluateArray, c64,
//!     assert_array_relative_eq};
//! use num::FromPrimitive;
//!
//! let shape = [3, 5, 2];
//! // We create an array aligned to `FFTW_ALIGNED` (64) bytes,
//! // which suits FFTW's SIMD code on every build.
//! let mut arr = rlst_dynamic_array!(c64, shape | rlst::FFTW_ALIGNED);
//! arr.fill_from_seed_equally_distributed(0);
//!
//! // We use `eval` to copy the data into a new array before generating the plan
//! // so that `arr` is not overwritten. If we want to overwrite `arr` with the
//! // FFT computations we could use `arr.r_mut().into_fft()` instead.
//! let mut fft = arr.eval().into_fft();
//! let forward_plan = fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward);
//! let backward_plan = fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Backward);
//!
//! forward_plan.unwrap().execute();
//! backward_plan.unwrap().execute();
//!
//! // We can get a reference to the array in `fft` using the `borrow` method.
//! let actual_arr = fft.borrow().unwrap();
//!
//! assert_array_relative_eq!(c64::from_usize(30).unwrap() * arr.r(), actual_arr.r(), 1E-10);
//!
//! ```
//! The method [c2c_fft](C2CPlan::c2c_fft) generates the c2c plan.
//! One can then execute a forward or backward fft as shown above.
//! By default the FFT overwrites the original array. With the `eval` function
//! the result is written into a new array. Note that the array in `backward`
//! is scaled compared to the original array. This is the default behaviour of FFTW.
//!
//! The FFTW interface in RLST always generates a forward and a backward FFT plan and
//! stores both plans.
//!
//! The `plan` struct satisfies all the usual traits for mutable array access. Hence,
//! it can be used itself like any other array. The underlying data is that of the original array
//! for who a plan was computed.
//!
//! ## Real-to-Real transform
//!
//! The FFTW library supports a number of real-to-real transformations which are accessible
//! through RLST. The following example demonstrates the use of the DCT-II transform in three
//! dimensions.  
//!
//! ```
//! use rlst::{rlst_dynamic_array, fftw::{FftwPlanFlags, R2RPlanKind, R2RPlan, FftPlanInplace},
//!     EvaluateArray, c64,
//!     assert_array_relative_eq};
//! use num::FromPrimitive;
//!
//! let shape = [3, 5, 2];
//! // We create an array aligned to `FFTW_ALIGNED` (64) bytes,
//! // which suits FFTW's SIMD code on every build.
//! let mut arr = rlst_dynamic_array!(f64, shape | rlst::FFTW_ALIGNED);
//! arr.fill_from_seed_equally_distributed(0);
//!
//! // We use `eval` to copy the data into a new array before generating the plan
//! // so that `arr` is not overwritten. If we want to overwrite `arr` with the
//! // FFT computations we could use `arr.r_mut().into_fft()` instead.
//! let mut fft = arr.eval().into_fft();
//! let forward_plan = fft.r2r_fft(FftwPlanFlags::Estimate,
//!   &[
//!     R2RPlanKind::FFTW_REDFT10,
//!     R2RPlanKind::FFTW_REDFT10,
//!     R2RPlanKind::FFTW_REDFT10
//!   ]);
//! let backward_plan = fft.r2r_fft(FftwPlanFlags::Estimate,
//!   &[
//!     R2RPlanKind::FFTW_REDFT01,
//!     R2RPlanKind::FFTW_REDFT01,
//!     R2RPlanKind::FFTW_REDFT01
//!   ]);
//!
//! forward_plan.unwrap().execute();
//! backward_plan.unwrap().execute();
//!
//! // We can get a reference to the array in `fft` using the `borrow` method.
//! let actual_arr = fft.borrow().unwrap();
//!
//! assert_array_relative_eq!(f64::from_usize(240).unwrap() * arr.r(), actual_arr.r(), 1E-10);
//!
//! ```
//! The `FFTW_REDFT10` plan is the standard DCT and `FFTW_REDFT01` is its inverse. For an overview
//! of real-to-real transformations in FFTW
//! see also [here](https://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transform-Kinds.html).
//! In multi-dimensional transformation a different kind of real-to-real transform can be assigned
//! to each dimension. Hence, the `kind` is a vector with the same dimension as the input array.
//!
//! ## Thread Safety
//!
//! Plan generation is not thread-safe in FFTW. This is solved in `rlst` by hiding the plan
//! generation inside a mutex. However, if other libraries access FFTW at the same time as `rlst`
//! a race condition may occur. The execution of a plan to compute the forward or backward FFT
//! is thread-safe.
//!
//! ## A note on ownership
//!
//! In the example above we are creating two plans, `forward_plan` and `backward_plan`. Both require
//! mutable access to the array. This is solved by storing the array as an `Arc<RwLock<...>>` object
//! in the `fft` struct. Each plan generation creates another shared reference by cloning the `Arc`
//! and write access is then controlled at runtime through the `RwLock`, which prevents two plans
//! writing into the data at the same time. Since `Arc` and `RwLock` are thread-safe this works
//! also across multiple threads.
//! Moving an array into this shared reference is why `into_fft` is necessary in the first place.
//! Calling `borrow` or `borrow_mut` on `fft` creates a new read or write lock and allows the user
//! to access the array data directly without other threads being able to write at the same time.
//! Mutable access is intentionally limited to the stored elements: the array itself cannot be
//! replaced while a plan retains pointers into its allocation.
//!
//! ## New interface (0.11, in progress)
//!
//! RLST 0.11 replaces the interface above. Until it is complete, the new types live in this
//! module beside the 0.10 ones, which keep their names and share the new FFTW lock; the 0.10
//! types are removed when the new interface is complete. So far the new interface has
//! complex-to-complex plans on slices and on RLST arrays and views:
//!
//! - [`Layout`], made from [`Dim`]s and a [`Placement`]: transform and batch dimensions with
//!   strides, validated and normalised;
//! - layouts from RLST arrays and views of any memory order, subviews, slices and nested views
//!   included: [`Layout::along_axis`], [`Layout::over_axes`] and their in-place variants,
//!   which pass FFTW the dimensions in the order of the axes given, with their own strides;
//! - [`Planner`], a builder with [`Rigor`], wisdom-only, unaligned, time-limit, alignment-class
//!   and scratch options, whose input-policy requests ([`DefaultRequest`],
//!   [`PreserveRequest`], [`DestroyRequest`]) decide a plan's policy;
//! - [`Plan`], of precision [`FftwReal`], family [`C2c`] and policy [`Preserving`],
//!   [`Destroying`] or [`InPlace`], which runs on any buffers that fit its layout and
//!   [`AlignmentClass`]es, from any thread, with [`Plan::description`] and
//!   [`Plan::fingerprint`];
//! - [`FftBuffer`] and [`FftBufferMut`], implemented for slices and for RLST arrays and views
//!   through [`StridedRawAccess`](crate::StridedRawAccess), checked at execution by shape,
//!   strides and storage;
//! - [`FftwError`] and [`Side`]; [`Direction`]; [`PLAN_ALIGNMENT`]; [`Backend`],
//!   [`backend`] and [`version`].
//!
//! Plans are made through FFTW's 64-bit guru interface, on scratch buffers unless a `*_on`
//! method is used, so planning never touches user data. Execution checks the buffers against
//! the plan before calling FFTW and takes no lock. Split storage, the real-to-complex and
//! real-to-real families, wisdom and threads follow.
//!
//! ```
//! use rlst::{c64, rlst_dynamic_array};
//! use rlst::fftw::{Dim, Direction, Layout, Placement, Planner};
//!
//! // Three transforms of length 8, one after the other.
//! let layout = Layout::new(
//!     &[Dim { n: 8, is: 1, os: 1 }],
//!     &[Dim { n: 3, is: 8, os: 8 }],
//!     Placement::OutOfPlace,
//! )?;
//! let forward = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
//! let backward = Planner::<f64>::new().c2c(&layout, Direction::Backward)?;
//!
//! let mut data = rlst_dynamic_array!(c64, [24] | rlst::FFTW_ALIGNED);
//! let mut spectrum = rlst_dynamic_array!(c64, [24] | rlst::FFTW_ALIGNED);
//! let mut result = rlst_dynamic_array!(c64, [24] | rlst::FFTW_ALIGNED);
//! data.fill_from_seed_equally_distributed(0);
//!
//! forward.execute(data.data().unwrap(), spectrum.data_mut().unwrap())?;
//! backward.execute(spectrum.data().unwrap(), result.data_mut().unwrap())?;
//!
//! // The round trip multiplies by the transform length.
//! for (x, y) in data.data().unwrap().iter().zip(result.data().unwrap()) {
//!     assert!((8.0 * x - y).norm() < 1e-12);
//! }
//! # Ok::<(), rlst::fftw::FftwError>(())
//! ```

pub mod c2c_inplace;
pub mod r2r_inplace;

pub use c2c_inplace::{C2CDirection, C2CPlan};
pub use r2r_inplace::{R2RPlan, R2RPlanKind};

// The new interface of 0.11 (docs/fftw/rlst-fftw-design.md).
mod align;
mod backend;
mod buffer;
mod error;
mod execute;
mod fingerprint;
mod kind;
mod layout;
mod lock;
mod options;
mod plan;
mod planner;
mod real;
#[cfg(test)]
mod test_support;
mod version;

use lock::with_fftw;

pub use align::{AlignmentClass, PLAN_ALIGNMENT};
pub use backend::{Backend, backend, version};
pub use buffer::{FftBuffer, FftBufferMut};
pub use error::{FftwError, Side};
pub use kind::Direction;
pub use layout::{Dim, Layout, Placement};
pub use options::{DefaultRequest, DestroyRequest, PreserveRequest, Rigor};
pub use plan::{C2c, Destroying, Family, InPlace, Plan, Policy, Preserving};
pub use planner::{DEFAULT_MAX_SCRATCH, Planner};
pub use real::FftwReal;

use std::{
    ffi::c_void,
    marker::PhantomData,
    ops::Deref,
    sync::{LockResult, PoisonError, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{
    Array, BaseItem, BorrowAtRuntime, BorrowAtRuntimeMut, RawAccessMut, RlstScalar, Shape,
    StableRawAccessMut, Stride, c32, c64, dense::array::shared_array::ArrayAtomicSharedRef,
};
use fftw_sys;
use num::Complex;

/// FFT Precision
#[derive(Copy, Clone, Debug)]
pub enum FftPrecision {
    /// Single Precision FFT
    Single,
    /// Double Precision FFT
    Double,
}

/// Flags for FFT Plans
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum FftwPlanFlags {
    /// Use a simple heuristic to select the plan. Does not overwrite data.
    Estimate = fftw_sys::FFTW_ESTIMATE,
    /// Compute several FFTs and measure execution time. Overwrites data.
    Measure = fftw_sys::FFTW_MEASURE,
    /// Like [FftwPlanFlags::Measure] but selects from a wider set of algorithms.
    Patient = fftw_sys::FFTW_PATIENT,
    /// Even wider search than [`FftwPlanFlags::Patient`]. May overwrite input while planning.
    Exhaustive = fftw_sys::FFTW_EXHAUSTIVE,
    /// Allow out of place transform to overwrite the input.
    DestroyInput = fftw_sys::FFTW_DESTROY_INPUT,
    /// Only use saved wisdom to generate a plan; fail if no applicable wisdom exists.
    WisdomOnly = fftw_sys::FFTW_WISDOM_ONLY,
    /// Do not assume SIMD-compatible alignment of input and output arrays.
    Unaligned = fftw_sys::FFTW_UNALIGNED,
}

/// Description of a plan with different input and output arrays.
pub trait FftPlanInplace {
    /// Execute the plan in place, overwriting the plan's shared array data.
    fn execute(&mut self);
}

struct FftwPlanPtrType<T> {
    _marker: PhantomData<T>,
}

trait FftwPlanPtrTypeTrait {
    type PtrType;

    fn get_fftw_ptr(ptr: *mut c_void) -> Self::PtrType;
}

impl FftwPlanPtrTypeTrait for FftwPlanPtrType<f64> {
    type PtrType = fftw_sys::fftw_plan;

    fn get_fftw_ptr(ptr: *mut c_void) -> Self::PtrType {
        ptr as fftw_sys::fftw_plan
    }
}

impl FftwPlanPtrTypeTrait for FftwPlanPtrType<f32> {
    type PtrType = fftw_sys::fftwf_plan;

    fn get_fftw_ptr(ptr: *mut c_void) -> Self::PtrType {
        ptr as fftw_sys::fftwf_plan
    }
}

/// Storage for a pointer to an FFT Plan
struct FftwPlanPtr<T>
where
    T: 'static,
    FftwPlanPtrType<T>: FftwPlanPtrTypeTrait,
{
    ptr: *mut c_void,
    _marker: std::marker::PhantomData<T>,
}

trait PlanInterfaceTrait {
    type PlanPtr;

    type RealType;

    /// Destroy a given plan
    unsafe fn destroy_plan(&self, plan_ptr: Self::PlanPtr);

    /// Create a c2c plan
    unsafe fn create_c2c(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Complex<Self::RealType>,
        arr_out: *mut Complex<Self::RealType>,
        sign: i32,
        flags: u32,
    ) -> Self::PlanPtr;

    /// Create a r2r plan
    unsafe fn create_r2r(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Self::RealType,
        arr_out: *mut Self::RealType,
        kind: &[R2RPlanKind],
        flags: u32,
    ) -> Self::PlanPtr;
}

impl<T> Drop for FftwPlanPtr<T>
where
    T: 'static,
    FftwPlanPtrType<T>: FftwPlanPtrTypeTrait,
{
    fn drop(&mut self) {
        if coe::is_same::<T, f64>() {
            let ptr = FftwPlanPtrType::<f64>::get_fftw_ptr(self.ptr);
            with_fftw(|_| unsafe { FftwPlanInterface::default().fftw_double.destroy_plan(ptr) });
        } else if coe::is_same::<T, f32>() {
            let ptr = FftwPlanPtrType::<f32>::get_fftw_ptr(self.ptr);
            with_fftw(|_| unsafe { FftwPlanInterface::default().fftw_single.destroy_plan(ptr) });
        }
    }
}

/// An interface to plan generation routines, called under the FFTW lock (`with_fftw`)
struct ConcretePlanInterface<T> {
    _marker: PhantomData<T>,
}

impl<T> Default for ConcretePlanInterface<T> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl PlanInterfaceTrait for ConcretePlanInterface<f64> {
    type PlanPtr = fftw_sys::fftw_plan;

    type RealType = f64;

    unsafe fn destroy_plan(&self, plan_ptr: Self::PlanPtr) {
        unsafe {
            fftw_sys::fftw_destroy_plan(plan_ptr);
        }
    }

    unsafe fn create_c2c(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Complex<Self::RealType>,
        arr_out: *mut Complex<Self::RealType>,
        sign: i32,
        flags: u32,
    ) -> Self::PlanPtr {
        unsafe { fftw_sys::fftw_plan_dft(rank, shape, arr_in, arr_out, sign, flags) }
    }

    unsafe fn create_r2r(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Self::RealType,
        arr_out: *mut Self::RealType,
        kind: &[R2RPlanKind],
        flags: u32,
    ) -> Self::PlanPtr {
        unsafe { fftw_sys::fftw_plan_r2r(rank, shape, arr_in, arr_out, kind.as_ptr(), flags) }
    }
}

impl PlanInterfaceTrait for ConcretePlanInterface<f32> {
    type PlanPtr = fftw_sys::fftwf_plan;

    type RealType = f32;

    unsafe fn destroy_plan(&self, plan_ptr: Self::PlanPtr) {
        unsafe {
            fftw_sys::fftwf_destroy_plan(plan_ptr);
        }
    }

    unsafe fn create_c2c(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Complex<Self::RealType>,
        arr_out: *mut Complex<Self::RealType>,
        sign: i32,
        flags: u32,
    ) -> Self::PlanPtr {
        unsafe { fftw_sys::fftwf_plan_dft(rank, shape, arr_in, arr_out, sign, flags) }
    }

    unsafe fn create_r2r(
        &self,
        rank: i32,
        shape: *const i32,
        arr_in: *mut Self::RealType,
        arr_out: *mut Self::RealType,
        kind: &[R2RPlanKind],
        flags: u32,
    ) -> Self::PlanPtr {
        unsafe { fftw_sys::fftwf_plan_r2r(rank, shape, arr_in, arr_out, kind.as_ptr(), flags) }
    }
}

/// An interface to work with FFTW Plans; every call runs under the FFTW lock (`with_fftw`)
#[derive(Default)]
struct FftwPlanInterface {
    pub fftw_single: ConcretePlanInterface<f32>,
    pub fftw_double: ConcretePlanInterface<f64>,
}

/// An in-place FFT plan
#[allow(private_bounds)]
pub struct PlanInplace<Item, ArrayImpl, const NDIM: usize>
where
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
{
    shared_ref: ArrayAtomicSharedRef<ArrayImpl, NDIM>,
    plan: FftwPlanPtr<Item::Real>,
    _marker: std::marker::PhantomData<Item>,
}

#[allow(private_interfaces, private_bounds)]
impl<Item, ArrayImpl, const NDIM: usize> PlanInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
{
    /// Create a new plan interface
    pub fn new(
        shared_ref: ArrayAtomicSharedRef<ArrayImpl, NDIM>,
        plan: FftwPlanPtr<Item::Real>,
    ) -> Self {
        Self {
            shared_ref,
            plan,
            _marker: PhantomData,
        }
    }
}

macro_rules! impl_plan_inplace {

    ($dtype:ty, $execute:ident) => {

impl<ArrayImpl, const NDIM: usize> FftPlanInplace for PlanInplace<$dtype, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = $dtype>,
{

    fn execute(&mut self) {
        // Lock access to the array. This will be returned when `lock` goes out of scope.
        // We need `lock` to stay in scope until the end of the function. Using `_lock`
        // as name achieves it without triggering a rust-analyzer warning about unused variables.
        let mut _lock = self.shared_ref.borrow_mut().unwrap();

        unsafe {
            fftw_sys::$execute(
                <FftwPlanPtrType<<$dtype as RlstScalar>::Real> as FftwPlanPtrTypeTrait>::get_fftw_ptr(
                    self.plan.ptr,
                ),
            );
        }

    }


}


    };

}

impl_plan_inplace!(f32, fftwf_execute);
impl_plan_inplace!(f64, fftw_execute);
impl_plan_inplace!(c32, fftwf_execute);
impl_plan_inplace!(c64, fftw_execute);

/// The FFT struct is a factory to create FFT plans.
///
/// It holds a shared reference to the original array,
/// which allow cloning so that multiple plans can be created
/// for the same underlying array without violating borrowing
/// constraints.
pub struct Fft<ArrayImpl, const NDIM: usize> {
    shared_ref: ArrayAtomicSharedRef<ArrayImpl, NDIM>,
}

/// A mutable lock on the elements used by FFT plans.
///
/// Unlike [`RwLockWriteGuard`], this guard does not implement `DerefMut`. This prevents replacing
/// the array (and therefore its allocation) while an FFTW plan stores pointers into that allocation.
///
/// ```compile_fail
/// use rlst::{DynArray, fftw::FftMutGuard};
///
/// let fft = DynArray::<f64, 1>::from_shape([4]).into_fft();
/// let mut guard = fft.borrow_mut().unwrap();
/// // Replacing the planned array would invalidate FFTW's stored pointer.
/// *guard = DynArray::<f64, 1>::from_shape([4]);
/// ```
pub struct FftMutGuard<'a, ArrayImpl, const NDIM: usize> {
    guard: RwLockWriteGuard<'a, Array<ArrayImpl, NDIM>>,
}

impl<ArrayImpl, const NDIM: usize> Deref for FftMutGuard<'_, ArrayImpl, NDIM> {
    type Target = Array<ArrayImpl, NDIM>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<ArrayImpl, const NDIM: usize> FftMutGuard<'_, ArrayImpl, NDIM>
where
    ArrayImpl: RawAccessMut,
{
    /// Return the mutable backing slice, when the array implementation exposes one.
    ///
    /// Mutating elements does not change the allocation or layout captured by existing plans.
    pub fn data_mut(&mut self) -> Option<&mut [ArrayImpl::Item]> {
        self.guard.data_mut()
    }

    pub(crate) fn array_mut(&mut self) -> &mut Array<ArrayImpl, NDIM> {
        &mut self.guard
    }
}

#[allow(private_bounds)]
impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: StableRawAccessMut<Item = Item> + Shape<NDIM> + Stride<NDIM>,
{
    /// Initialize a FFT computation.
    pub fn into_fft(self) -> Fft<ArrayImpl, NDIM> {
        Fft::from_arr(self)
    }
}

#[allow(private_bounds)]
impl<Item, ArrayImpl, const NDIM: usize> Fft<ArrayImpl, NDIM>
where
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: StableRawAccessMut<Item = Item> + Shape<NDIM> + Stride<NDIM>,
{
    /// Create a new FFT struct from an existing array
    pub fn from_arr(arr: Array<ArrayImpl, NDIM>) -> Self {
        Self {
            shared_ref: arr.into_shared_atomic(),
        }
    }

    /// Borrow the array as read-only
    pub fn borrow(&self) -> LockResult<RwLockReadGuard<'_, Array<ArrayImpl, NDIM>>> {
        self.shared_ref.borrow()
    }

    /// Borrow the stored elements as read-write.
    ///
    /// The returned guard permits element mutation through [`FftMutGuard::data_mut`] but prevents
    /// replacing the array or changing its allocation and layout while plans exist.
    pub fn borrow_mut(&self) -> LockResult<FftMutGuard<'_, ArrayImpl, NDIM>> {
        match self.shared_ref.borrow_mut() {
            Ok(guard) => Ok(FftMutGuard { guard }),
            Err(error) => Err(PoisonError::new(FftMutGuard {
                guard: error.into_inner(),
            })),
        }
    }
}
