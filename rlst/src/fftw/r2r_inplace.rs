//! Definition of a real-to-real inplace FFT plan

use std::{
    ffi::{c_int, c_void},
    marker::PhantomData,
};

use crate::{
    BaseItem, RlstScalar, Shape, StableRawAccessMut, Stride,
    fftw::{FftwPlanFlags, FftwPlanPtrType, FftwPlanPtrTypeTrait, PlanInplace},
};

use itertools::{Itertools, izip};

use super::{Fft, FftwPlanInterface, FftwPlanPtr, PlanInterfaceTrait, with_fftw};

pub use fftw_sys::fftw_r2r_kind as R2RPlanKind;

/// Create an in-place real-to-real FFTW plan.
#[allow(private_bounds)]
pub trait R2RPlan<Item, ArrayImpl, const NDIM: usize>
where
    ArrayImpl: BaseItem<Item = Item>,
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
{
    /// Create a real-to-real plan with one transform kind per axis.
    ///
    /// Returns `None` for empty or noncontiguous arrays, ranks above three, or
    /// when FFTW cannot construct the requested plan. Planning flags other than
    /// [`FftwPlanFlags::Estimate`] may overwrite the array while benchmarking.
    ///
    /// # Panics
    ///
    /// Panics unless `kind.len() == NDIM`.
    /// For details see the [FFTW module description](crate::fftw).
    fn r2r_fft(
        &self,
        flags: FftwPlanFlags,
        kind: &[R2RPlanKind],
    ) -> Option<PlanInplace<Item, ArrayImpl, NDIM>>;
}

macro_rules! impl_r2r_inplace {
    ($ty: ty, $precision: ident) => {
        impl<ArrayImpl, const NDIM: usize> R2RPlan<$ty, ArrayImpl, NDIM> for Fft<ArrayImpl, NDIM>
        where
            ArrayImpl: Shape<NDIM> + Stride<NDIM> + StableRawAccessMut<Item = $ty>,
        {
            /// Create a new plan wrapped inside an array structure. Returns None if the Plan cannot be created.
            fn r2r_fft(
                &self,
                flags: FftwPlanFlags,
                kind: &[R2RPlanKind],
            ) -> Option<PlanInplace<$ty, ArrayImpl, NDIM>> {
                let mut guard = self.borrow_mut().unwrap();
                let arr = guard.array_mut();
                assert_eq!(
                    kind.len(),
                    NDIM,
                    "`kind` has {} dimensions but {NDIM} required",
                    kind.len()
                );

                // Return None if NDIM > 3 as FFTW only supports up to three dimensions
                if NDIM > 3 {
                    return None;
                }
                // Return None if the array is not contiguous or is empty
                if !arr.is_contiguous() || arr.is_empty() {
                    return None;
                }

                let mut shape = arr
                    .shape()
                    .iter()
                    .map(|&elem| elem as i32)
                    .collect_array()
                    .unwrap();

                // If the array is column-major reverse the order of the shape as
                // FFTW uses row-major order

                if matches!(arr.memory_layout(), crate::MemoryLayout::ColumnMajor) {
                    shape = {
                        let mut rev_shape = [0_i32; NDIM];

                        for (&orig_index, rev_index) in
                            izip!(shape.iter().rev(), rev_shape.iter_mut())
                        {
                            *rev_index = orig_index;
                        }
                        rev_shape
                    };
                }

                let data = arr.data_mut()?.as_mut_ptr();

                let plan_ptr = with_fftw(|_| unsafe {
                    FftwPlanInterface::default().$precision.create_r2r(
                        NDIM as c_int,
                        shape.as_ptr(),
                        data,
                        data,
                        kind,
                        flags as u32,
                    )
                });

                if plan_ptr.is_null() {
                    None
                } else {
                    Some(PlanInplace::new(
                        self.shared_ref.clone(),
                        FftwPlanPtr {
                            ptr: plan_ptr as *mut c_void,
                            _marker: PhantomData,
                        },
                    ))
                }
            }
        }
    };
}

impl_r2r_inplace!(f64, fftw_double);
impl_r2r_inplace!(f32, fftw_single);

#[cfg(test)]
mod test {

    use std::f64::consts::PI;

    use crate::{
        DynArray, EvaluateArray,
        fftw::{
            FftPlanInplace, FftwPlanFlags,
            r2r_inplace::{R2RPlan, R2RPlanKind},
        },
    };

    #[test]
    fn test_dct_ii_against_direct_reference() {
        let input = [1.0, -2.0, 0.5, 3.0, -0.25];
        let fft = DynArray::<f64, _>::from_shape([input.len()]).into_fft();
        let mut plan = fft
            .r2r_fft(FftwPlanFlags::Estimate, &[R2RPlanKind::FFTW_REDFT10])
            .unwrap();
        fft.borrow_mut()
            .unwrap()
            .data_mut()
            .unwrap()
            .copy_from_slice(&input);
        plan.execute();

        let actual = fft.borrow().unwrap().eval();
        for k in 0..input.len() {
            let expected = 2.0
                * input
                    .iter()
                    .enumerate()
                    .map(|(j, value)| {
                        value * (PI * (j as f64 + 0.5) * k as f64 / input.len() as f64).cos()
                    })
                    .sum::<f64>();
            approx::assert_relative_eq!(actual[[k]], expected, epsilon = 1E-10);
        }
    }

    #[test]
    fn test_r2r_1d_f64() {
        let n = 10;
        let mut original = DynArray::<f64, _>::from_shape([10]);
        original.fill_from_seed_equally_distributed(0);

        let fft = original.eval().into_fft();

        fft.r2r_fft(FftwPlanFlags::Estimate, &[R2RPlanKind::FFTW_REDFT10])
            .unwrap()
            .execute();

        fft.r2r_fft(FftwPlanFlags::Estimate, &[R2RPlanKind::FFTW_REDFT01])
            .unwrap()
            .execute();

        let actual = fft.borrow().unwrap().eval();

        // Check that the zeroth frequency is the sum of all input elements.
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!((2 * n) as f64 * original.r(), actual.r(), 1E-10);
    }

    #[test]
    fn test_r2r_1d_f32() {
        let n = 10;
        let mut original = DynArray::<f32, _>::from_shape([10]);
        original.fill_from_seed_equally_distributed(0);

        let fft = original.eval().into_fft();

        fft.r2r_fft(FftwPlanFlags::Estimate, &[R2RPlanKind::FFTW_REDFT10])
            .unwrap()
            .execute();

        fft.r2r_fft(FftwPlanFlags::Estimate, &[R2RPlanKind::FFTW_REDFT01])
            .unwrap()
            .execute();

        let actual = fft.borrow().unwrap().eval();

        // Check that the zeroth frequency is the sum of all input elements.
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!((2 * n) as f32 * original.r(), actual.r(), 1E-5);
    }

    #[test]
    fn test_r2r_3d_f64() {
        let mut original = DynArray::<f64, _>::from_shape([4, 6, 2]);
        original.fill_from_seed_equally_distributed(0);

        let fft = original.eval().into_fft();

        fft.r2r_fft(
            FftwPlanFlags::Estimate,
            &[
                R2RPlanKind::FFTW_REDFT10,
                R2RPlanKind::FFTW_REDFT10,
                R2RPlanKind::FFTW_REDFT10,
            ],
        )
        .unwrap()
        .execute();

        fft.r2r_fft(
            FftwPlanFlags::Estimate,
            &[
                R2RPlanKind::FFTW_REDFT01,
                R2RPlanKind::FFTW_REDFT01,
                R2RPlanKind::FFTW_REDFT01,
            ],
        )
        .unwrap()
        .execute();

        let actual = fft.borrow().unwrap().eval();

        // Check that the zeroth frequency is the sum of all input elements.
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(384.0_f64 * original.r(), actual.r(), 1E-10);
    }
}
