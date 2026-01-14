//! Interface routines to [Burn Tensors](burn::Tensor).

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{DType, Element, Float, Int, TensorData, TensorKind},
};
use itertools::Itertools as _;

use crate::{
    Array, EvaluateRowMajorArray, RlstResult, RlstScalar, StridedDynArray, ValueArrayImpl,
    dense::{
        data_container::VectorContainer, layout::row_major_stride_from_shape,
        strided_base_array::StridedBaseArray,
    },
};

/// Convert a Burn tensor into a RLST array
pub trait TensorIntoRlstArray<const NDIM: usize> {
    /// Convert a Burn tensor into a RLST array
    fn into_array<Item: RlstScalar + RlstAsTensorDtype + Element>(
        self,
    ) -> RlstResult<StridedDynArray<Item, NDIM>>;
}

/// Convert a RLST array into a Burn tensor
pub trait RlstArrayIntoBurnTensor<Item, B: Backend, const NDIM: usize, K: TensorKind<B>> {
    /// Convert array into Burn tensor
    fn into_tensor(self) -> RlstResult<Tensor<B, NDIM, K>>;
}

/// Helper trait to return the Burn runtime type from a Rust type
pub trait RlstAsTensorDtype {
    /// Return the Burn type
    fn burn_type() -> DType;
}

macro_rules! impl_rlst_as_tensor_type {
    ($dtype:ty, $tensor:expr) => {
        impl RlstAsTensorDtype for $dtype {
            fn burn_type() -> DType {
                $tensor
            }
        }
    };
}

impl_rlst_as_tensor_type!(f64, DType::F64);
impl_rlst_as_tensor_type!(f32, DType::F32);
impl_rlst_as_tensor_type!(u8, DType::U8);
impl_rlst_as_tensor_type!(u16, DType::U16);
impl_rlst_as_tensor_type!(u32, DType::U32);
impl_rlst_as_tensor_type!(u64, DType::U64);
impl_rlst_as_tensor_type!(i8, DType::I8);
impl_rlst_as_tensor_type!(i16, DType::I16);
impl_rlst_as_tensor_type!(i32, DType::I32);
impl_rlst_as_tensor_type!(i64, DType::I64);

impl<const NDIM: usize, B: Backend> TensorIntoRlstArray<NDIM> for Tensor<B, NDIM> {
    fn into_array<Item: RlstScalar + RlstAsTensorDtype + Element>(
        self,
    ) -> RlstResult<StridedDynArray<Item, NDIM>> {
        if self.dtype() != Item::burn_type() {
            Err(crate::RlstError::GeneralError(format!(
                "Array type does not match tensor type {:#?}",
                Item::burn_type()
            )))
        } else {
            let tensor_data = self.clone().into_data();

            let shape = tensor_data
                .shape
                .iter()
                .copied()
                .collect_array::<NDIM>()
                .unwrap();
            let container = VectorContainer::from_vec(tensor_data.into_vec::<Item>().unwrap());

            Ok(Array::new(StridedBaseArray::new(
                container,
                shape,
                row_major_stride_from_shape(shape),
            )))
        }
    }
}

impl<Item, ArrayImpl, B, const NDIM: usize> RlstArrayIntoBurnTensor<Item, B, NDIM, Float>
    for Array<ArrayImpl, NDIM>
where
    Item: RlstAsTensorDtype + Element,
    ArrayImpl: ValueArrayImpl<Item, NDIM>,
    B: Backend,
{
    fn into_tensor(self) -> RlstResult<Tensor<B, NDIM>> {
        let arr_row_major = self.eval_row_major();

        let device = B::Device::default();
        if !coe::is_same::<B::FloatElem, Item>() {
            Err(crate::RlstError::GeneralError(
                "Type not supported by device".to_string(),
            ))
        } else {
            Ok(Tensor::<B, NDIM>::from_data(
                TensorData::new(
                    arr_row_major.data().unwrap().to_vec(),
                    self.shape().as_slice(),
                ),
                &device,
            ))
        }
    }
}

impl<Item, ArrayImpl, B, const NDIM: usize> RlstArrayIntoBurnTensor<Item, B, NDIM, Int>
    for Array<ArrayImpl, NDIM>
where
    Item: RlstAsTensorDtype + Element,
    ArrayImpl: ValueArrayImpl<Item, NDIM>,
    B: Backend,
{
    fn into_tensor(self) -> RlstResult<Tensor<B, NDIM, Int>> {
        let arr_row_major = self.eval_row_major();

        let device = B::Device::default();
        if !coe::is_same::<B::IntElem, Item>() {
            Err(crate::RlstError::GeneralError(
                "Type not supported by device".to_string(),
            ))
        } else {
            Ok(Tensor::<B, NDIM, Int>::from_data(
                TensorData::new(
                    arr_row_major.data().unwrap().to_vec(),
                    self.shape().as_slice(),
                ),
                &device,
            ))
        }
    }
}
