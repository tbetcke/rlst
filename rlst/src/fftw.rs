//! Implementation of fft using fftw
//!
//! FFTW uses plans to execute an FFT computation. Within FFTW there are two steps
//! to get the FFT of a given array.
//!
//! 1. Generate a plan for the desired FFT
//! 2. Execute the FFT
//!
//! FFTW supports FFT computations on one, two, and three dimensional input arrays.
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
//! // We create a 16 byte aligned array. This is
//! // the required alignment for FFTW to use SIMD features.
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
//! // We create a 16 byte aligned array. This is
//! // the required alignment for FFTW to use SIMD features.
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

pub mod c2c_inplace;
pub mod r2r_inplace;

pub use c2c_inplace::{C2CDirection, C2CPlan};
pub use r2r_inplace::{R2RPlan, R2RPlanKind};

use std::{
    ffi::c_void,
    marker::PhantomData,
    ops::Deref,
    sync::{LazyLock, LockResult, Mutex, PoisonError, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{
    Array, BaseItem, BorrowAtRuntime, BorrowAtRuntimeMut, RawAccessMut, RlstScalar, Shape, Stride,
    c32, c64, dense::array::shared_array::ArrayAtomicSharedRef,
};
use fftw_sys;
use num::Complex;

static FFTW_PLAN_INTERFACE: LazyLock<Mutex<FftwPlanInterface>> =
    LazyLock::new(|| Mutex::new(FftwPlanInterface::default()));

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
    /// Even wider search than [FftwPlanFlags::Patient]
    Exhaustive = fftw_sys::FFTW_EXHAUSTIVE,
    /// Allow out of place transform to overwrite the input.
    DestroyInput = fftw_sys::FFTW_DESTROY_INPUT,
    /// Only use wisdowm to generate a plan. Do not return a plan if there is no wisdom.
    WisdomOnly = fftw_sys::FFTW_WISDOM_ONLY,
    /// Out-of-place transforms must not overwrite input (not available for all plan types).
    Unaligned = fftw_sys::FFTW_UNALIGNED,
}

/// Description of a plan with different input and output arrays.
pub trait FftPlanInplace {
    /// Execute the plan
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
            unsafe {
                FFTW_PLAN_INTERFACE
                    .lock()
                    .unwrap()
                    .fftw_double
                    .destroy_plan(ptr);
            };
        } else if coe::is_same::<T, f32>() {
            let ptr = FftwPlanPtrType::<f32>::get_fftw_ptr(self.ptr);
            unsafe {
                FFTW_PLAN_INTERFACE
                    .lock()
                    .unwrap()
                    .fftw_single
                    .destroy_plan(ptr)
            };
        }
    }
}

/// An interface to plan generation routines that will be hidden behind a mutex
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

/// A thread-safe interface to work with FFTW Plans
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
    ArrayImpl: RawAccessMut<Item = Item> + Shape<NDIM> + Stride<NDIM>,
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
    ArrayImpl: RawAccessMut<Item = Item> + Shape<NDIM> + Stride<NDIM>,
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
