//! Conversions between RLST arrays and [Burn tensors](burn::Tensor).
//!
//! Enable the `burn` feature to use this module. Conversions copy data: Burn tensors
//! become row-major [`StridedDynArray`] values, while RLST arrays
//! are evaluated in row-major order before creating a tensor on a Burn [`Device`].
//!
//! Burn selects its execution backend at runtime through the device. Enable at least one
//! backend feature of the `burn` crate (for example `flex` or `wgpu`) in the application.

use burn::{
    Tensor,
    tensor::{DType, Device, Element, Float, Int, TensorData, kind::Basic},
};
use itertools::Itertools as _;

use crate::{
    Array, EvaluateRowMajorArray, RlstError, RlstResult, StridedDynArray, ValueArrayImpl,
    dense::{
        data_container::VectorContainer, layout::row_major_stride_from_shape,
        strided_base_array::StridedBaseArray,
    },
};

/// Convert a float or integer Burn tensor into an owned RLST array.
///
/// The requested `Item` must exactly match the tensor's runtime dtype. A mismatch,
/// or a failure to read the tensor from its device, is returned as
/// [`crate::RlstError::GeneralError`]. Shape and row-major logical order are preserved,
/// and the tensor data is copied.
pub trait TensorIntoRlstArray<const NDIM: usize> {
    /// Consume the tensor and copy it into a row-major RLST array.
    fn into_array<Item: RlstAsTensorDtype + Element>(
        self,
    ) -> RlstResult<StridedDynArray<Item, NDIM>>;
}

/// Convert an RLST array into an owned Burn tensor.
///
/// The implementation evaluates arbitrary RLST layouts in row-major logical order and
/// creates a tensor of kind `K` ([`Float`] or [`Int`]) with the dtype of `Item`. `Item`
/// must be a floating-point type for float tensors and an integer type for integer
/// tensors, and the device must support its dtype; otherwise conversion returns
/// [`crate::RlstError::GeneralError`].
pub trait RlstArrayIntoBurnTensor<Item, const NDIM: usize, K: Basic> {
    /// Consume and copy the array into a tensor on `device`, preserving its shape.
    fn into_tensor_on(self, device: &Device) -> RlstResult<Tensor<NDIM, K>>;

    /// Consume and copy the array into a tensor on Burn's default device.
    ///
    /// The default device is the first enabled Burn backend in Burn's priority order,
    /// unless the `BURN_DEVICE` environment variable names one.
    ///
    /// # Panics
    ///
    /// Panics if no Burn execution backend is enabled.
    fn into_tensor(self) -> RlstResult<Tensor<NDIM, K>>
    where
        Self: Sized,
    {
        self.into_tensor_on(&Device::default())
    }
}

/// Map a supported Rust scalar to its Burn runtime [`DType`].
///
/// Implementations are provided for `f32`, `f64`, and signed and unsigned integer
/// types from 8 through 64 bits. Complex RLST scalars are not supported.
pub trait RlstAsTensorDtype {
    /// Return the corresponding Burn dtype.
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

impl<const NDIM: usize, K: Basic> TensorIntoRlstArray<NDIM> for Tensor<NDIM, K> {
    fn into_array<Item: RlstAsTensorDtype + Element>(
        self,
    ) -> RlstResult<StridedDynArray<Item, NDIM>> {
        if self.dtype() != Item::burn_type() {
            return Err(RlstError::GeneralError(format!(
                "Array type {:?} does not match tensor type {:?}",
                Item::burn_type(),
                self.dtype()
            )));
        }

        let tensor_data = self
            .try_into_data()
            .map_err(|err| RlstError::GeneralError(format!("Cannot read tensor: {err}")))?;

        let shape = tensor_data
            .shape()
            .iter()
            .copied()
            .collect_array::<NDIM>()
            .ok_or_else(|| {
                RlstError::GeneralError(format!(
                    "Tensor data shape {:?} does not have {NDIM} dimensions",
                    tensor_data.shape()
                ))
            })?;
        let data = tensor_data
            .try_into_vec::<Item>()
            .map_err(|err| RlstError::GeneralError(format!("Cannot read tensor data: {err}")))?;

        Ok(Array::new(StridedBaseArray::new(
            VectorContainer::from_vec(data),
            shape,
            row_major_stride_from_shape(shape),
        )))
    }
}

/// Copy `arr` in row-major order into a tensor of kind `K` with the dtype of `Item`.
///
/// `kind_matches` decides whether the dtype of `Item` is valid for the tensor kind.
fn array_into_tensor<Item, ArrayImpl, K, const NDIM: usize>(
    arr: Array<ArrayImpl, NDIM>,
    device: &Device,
    kind: &str,
    kind_matches: fn(&DType) -> bool,
) -> RlstResult<Tensor<NDIM, K>>
where
    Item: RlstAsTensorDtype + Element,
    ArrayImpl: ValueArrayImpl<Item, NDIM>,
    K: Basic,
{
    let dtype = Item::burn_type();
    if !kind_matches(&dtype) {
        return Err(RlstError::GeneralError(format!(
            "Type {dtype:?} cannot be stored in {kind} tensor"
        )));
    }
    if !device.supports_dtype(dtype) {
        return Err(RlstError::GeneralError(format!(
            "Type {dtype:?} not supported by device {device:?}"
        )));
    }

    let arr_row_major = arr.eval_row_major();
    let data = TensorData::new(arr_row_major.data().unwrap().to_vec(), arr.shape());

    Ok(Tensor::<NDIM, K>::from_data(data, (device, dtype)))
}

impl<Item, ArrayImpl, const NDIM: usize> RlstArrayIntoBurnTensor<Item, NDIM, Float>
    for Array<ArrayImpl, NDIM>
where
    Item: RlstAsTensorDtype + Element,
    ArrayImpl: ValueArrayImpl<Item, NDIM>,
{
    fn into_tensor_on(self, device: &Device) -> RlstResult<Tensor<NDIM, Float>> {
        array_into_tensor(self, device, "a float", DType::is_float)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> RlstArrayIntoBurnTensor<Item, NDIM, Int>
    for Array<ArrayImpl, NDIM>
where
    Item: RlstAsTensorDtype + Element,
    ArrayImpl: ValueArrayImpl<Item, NDIM>,
{
    fn into_tensor_on(self, device: &Device) -> RlstResult<Tensor<NDIM, Int>> {
        array_into_tensor(self, device, "an integer", |dtype| {
            dtype.is_int() || dtype.is_uint()
        })
    }
}
