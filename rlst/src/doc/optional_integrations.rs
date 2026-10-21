//! Optional integrations
//!
//! RLST keeps native and ecosystem integrations behind Cargo features so applications only build
//! what they use. This guide summarizes setup and the contracts that differ from ordinary arrays.
//!
//! # FFTW transforms
//!
//! Enable `fftw_system`, the only backend, which implies `fftw`. It needs both single- and
//! double-precision FFTW libraries, version 3.3.4 or later. FFTW is GPL-licensed; review the
//! licensing implications before distributing linked binaries.
//!
//! FFT plans store pointers into the planned array. Since RLST 0.7, `borrow_mut()`
//! therefore returns an `FftMutGuard`: mutate elements through `data_mut()`, rather than replacing
//! the array or changing its layout while a plan exists.
//!
//! ```ignore
//! use rlst::{BorrowAtRuntimeMut, DynArray};
//!
//! let fft = DynArray::<f64, 1>::from_shape([8]).into_fft();
//! let mut elements = fft.borrow_mut().unwrap();
//! elements.data_mut().unwrap().fill(0.0);
//! ```
//!
//! Transform direction, normalization, planning flags, dimensionality, and contiguous-layout
//! requirements are covered by the `rlst::fftw` module examples.
//!
//! # Burn tensors
//!
//! Enable `burn` to expose `rlst::interface::burn`. RLST 0.10 uses Burn 0.22, where tensors have
//! no backend type parameter: the backend is chosen at runtime through a Burn `Device`. A
//! downstream application must enable at least one Burn 0.22 backend feature, such as `flex`
//! (pure Rust CPU) or `wgpu`. Conversions consume and copy their input while preserving shape:
//! tensors become row-major RLST arrays, and RLST arrays are evaluated in row-major order and
//! copied to the given device, or to Burn's default device.
//!
//! ```ignore
//! use burn::tensor::{Device, Int, Tensor};
//! use rlst::interface::burn::{RlstArrayIntoBurnTensor, TensorIntoRlstArray};
//!
//! let device = Device::flex();
//! let tensor: Tensor<2> = array.r().into_tensor_on(&device)?;
//! let round_trip = tensor.into_array::<f64>()?;
//!
//! // `into_tensor` uses `Device::default()`, which panics if no Burn backend is enabled.
//! let indices: Tensor<1, Int> = index_array.into_tensor()?;
//! ```
//!
//! Each tensor takes the dtype of the RLST array's scalar, so an `f64` array becomes an `F64`
//! tensor whatever the device's default float type; a device that does not support the dtype is
//! an error. Float tensors take `f32` or `f64`, and integer tensors take signed or unsigned
//! integers from 8 through 64 bits. When converting back, the requested Rust scalar must exactly
//! match the tensor dtype. Complex RLST arrays are not supported.
