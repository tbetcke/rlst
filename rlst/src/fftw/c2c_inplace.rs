//! Definition of a complex 2 complex inplace FFT plan

use std::{
    ffi::{c_int, c_void},
    marker::PhantomData,
};

use crate::{
    BaseItem, RawAccessMut, RlstScalar, Shape, Stride, c32, c64,
    fftw::{FftwPlanFlags, FftwPlanPtrType, FftwPlanPtrTypeTrait, PlanInplace},
};

use itertools::{Itertools, izip};

use super::{FFTW_PLAN_INTERFACE, Fft, FftwPlanPtr, PlanInterfaceTrait};

/// The direction of the C2C FFT
#[derive(Copy, Clone)]
#[repr(i32)]
pub enum C2CDirection {
    /// Denotes the forward transform
    Forward = -1,
    /// Denotes the backward (inverse) transform
    Backward = 1,
}

/// Create a C2C FFT Plan
#[allow(private_bounds)]
pub trait C2CPlan<Item, ArrayImpl, const NDIM: usize>
where
    ArrayImpl: BaseItem<Item = Item>,
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
{
    /// Create a new c2c plan
    fn c2c_fft(
        &self,
        flags: FftwPlanFlags,
        direction: C2CDirection,
    ) -> Option<PlanInplace<Item, ArrayImpl, NDIM>>;
}

macro_rules! impl_c2c_inplace {
    ($ty: ty, $precision: ident) => {
        impl<ArrayImpl, const NDIM: usize> C2CPlan<$ty, ArrayImpl, NDIM> for Fft<ArrayImpl, NDIM>
        where
            ArrayImpl: Shape<NDIM> + Stride<NDIM> + RawAccessMut<Item = $ty>,
        {
            /// Create a new plan wrapped inside an array structure. Returns None if the Plan cannot be created.
            fn c2c_fft(
                &self,
                flags: FftwPlanFlags,
                direction: C2CDirection,
            ) -> Option<PlanInplace<$ty, ArrayImpl, NDIM>> {
                let mut arr = self.borrow_mut().unwrap();

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

                let plan_ptr = unsafe {
                    FFTW_PLAN_INTERFACE.lock().unwrap().$precision.create_c2c(
                        NDIM as c_int,
                        shape.as_ptr(),
                        data,
                        data,
                        direction as i32,
                        flags as u32,
                    )
                };

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

impl_c2c_inplace!(c32, fftw_single);
impl_c2c_inplace!(c64, fftw_double);

#[cfg(test)]
mod test {

    use num::FromPrimitive;

    use crate::fftw::c2c_inplace::C2CDirection;
    use crate::{EvaluateObject, c64};

    use crate::DynArray;
    use crate::fftw::{FftPlanInplace, FftwPlanFlags};

    use super::*;

    #[test]
    fn test_c2c_1d_f64() {
        let n = 10;
        let mut original = DynArray::<c64, _>::from_shape([10]);
        original.fill_from_seed_equally_distributed(0);

        let mut copied = original.eval();
        let sum = copied.iter_value().sum::<c64>();

        copied
            .r_mut()
            .into_fft()
            .c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();

        approx::assert_relative_eq!(sum, copied[[0]], epsilon = 1E-10);

        let fft = original.eval().into_fft();

        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();

        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Backward)
            .unwrap()
            .execute();

        let actual = fft.borrow().unwrap().eval();

        // Check that the zeroth frequency is the sum of all input elements.
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(
            c64::from_usize(n).unwrap() * original.r(),
            actual.r(),
            1E-10
        );
    }

    #[test]
    fn test_c2c_1d_f32() {
        let n = 10;
        let mut original = DynArray::<c32, _>::from_shape([10]);
        original.fill_from_seed_equally_distributed(0);

        let mut copied = original.eval();
        let sum = copied.iter_value().sum::<c32>();

        copied
            .r_mut()
            .into_fft()
            .c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();

        approx::assert_relative_eq!(sum, copied[[0]], epsilon = 1E-10);

        let fft = original.eval().into_fft();

        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Forward)
            .unwrap()
            .execute();

        fft.c2c_fft(FftwPlanFlags::Estimate, C2CDirection::Backward)
            .unwrap()
            .execute();

        let actual = fft.borrow().unwrap().eval();

        // Check that the zeroth frequency is the sum of all input elements.
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(
            c32::from_usize(n).unwrap() * original.r(),
            actual.r(),
            1E-5
        );
    }

    //     #[test]
    //     fn test_c2c_1d_f32() {
    //         let n = 10;
    //         let mut arr = DynArray::<c32, _>::from_shape([n]);
    //         arr.fill_from_seed_equally_distributed(0);
    //         let expected = arr.eval();

    //         let sum = arr
    //             .eval()
    //             .r_mut()
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .iter_value()
    //             .sum::<c32>();

    //         let actual = arr
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .ifft();

    //         // Check that the zeroth frequency is the sum of all input elements.
    //         approx::assert_relative_eq!(
    //             sum,
    //             c32::from_usize(n).unwrap() * expected[[0]],
    //             epsilon = 1E-5
    //         );
    //         // Compute the forward and inverse FFT and check that the result is still the same.
    //         crate::assert_array_relative_eq!(actual, c32::from_usize(n).unwrap() * expected.r(), 1E-5);
    //     }

    //     #[test]
    //     fn test_c2c_2d_f64() {
    //         let shape = [5, 2];
    //         let n = shape.iter().product();
    //         let mut arr = DynArray::<c64, _>::from_shape(shape);
    //         arr.fill_from_seed_equally_distributed(0);

    //         let expected = arr.eval();

    //         let sum = arr
    //             .eval()
    //             .r_mut()
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .iter_value()
    //             .sum::<c64>();

    //         let actual = arr
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .ifft();

    //         // Check that the zeroth frequency is the sum of all input elements.
    //         approx::assert_relative_eq!(
    //             sum,
    //             c64::from_usize(n).unwrap() * expected[[0, 0]],
    //             epsilon = 1E-10
    //         );
    //         // Compute the forward and inverse FFT and check that the result is still the same.
    //         crate::assert_array_relative_eq!(actual, c64::from_usize(n).unwrap() * expected.r(), 1E-10);
    //     }

    //     #[test]
    //     fn test_c2c_2d_f32() {
    //         let shape = [5, 2];
    //         let n = shape.iter().product();
    //         let mut arr = DynArray::<c32, _>::from_shape(shape);
    //         arr.fill_from_seed_equally_distributed(0);

    //         let expected = arr.eval();

    //         let sum = arr
    //             .eval()
    //             .r_mut()
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .iter_value()
    //             .sum::<c32>();

    //         let actual = arr
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .ifft();

    //         // Check that the zeroth frequency is the sum of all input elements.
    //         approx::assert_relative_eq!(
    //             sum,
    //             c32::from_usize(n).unwrap() * expected[[0, 0]],
    //             epsilon = 1E-5
    //         );
    //         // Compute the forward and inverse FFT and check that the result is still the same.
    //         crate::assert_array_relative_eq!(actual, c32::from_usize(n).unwrap() * expected.r(), 1E-5);
    //     }

    //     #[test]
    //     fn test_c2c_3d_f64() {
    //         let shape = [5, 2, 4];
    //         let n = shape.iter().product();
    //         let mut arr = DynArray::<c64, _>::from_shape(shape);
    //         arr.fill_from_seed_equally_distributed(0);

    //         let expected = arr.eval();

    //         let sum = arr
    //             .eval()
    //             .r_mut()
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .iter_value()
    //             .sum::<c64>();

    //         let actual = arr
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .ifft();

    //         // Check that the zeroth frequency is the sum of all input elements.
    //         approx::assert_relative_eq!(
    //             sum,
    //             c64::from_usize(n).unwrap() * expected[[0, 0, 0]],
    //             epsilon = 1E-10
    //         );
    //         // Compute the forward and inverse FFT and check that the result is still the same.
    //         crate::assert_array_relative_eq!(actual, c64::from_usize(n).unwrap() * expected.r(), 1E-10);
    //     }

    //     #[test]
    //     fn test_c2c_3d_f32() {
    //         let shape = [5, 2, 4];
    //         let n = shape.iter().product();
    //         let mut arr = DynArray::<c32, _>::from_shape(shape);
    //         arr.fill_from_seed_equally_distributed(0);

    //         let expected = arr.eval();

    //         let sum = arr
    //             .eval()
    //             .r_mut()
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .iter_value()
    //             .sum::<c32>();

    //         let actual = arr
    //             .into_c2c_fft(FftwPlanFlags::Estimate)
    //             .unwrap()
    //             .fft()
    //             .ifft();

    //         // Check that the zeroth frequency is the sum of all input elements.
    //         approx::assert_relative_eq!(
    //             sum,
    //             c32::from_usize(n).unwrap() * expected[[0, 0, 0]],
    //             epsilon = 1E-5
    //         );
    //         // Compute the forward and inverse FFT and check that the result is still the same.
    //         crate::assert_array_relative_eq!(actual, c32::from_usize(n).unwrap() * expected.r(), 1E-5);
    //     }
}
