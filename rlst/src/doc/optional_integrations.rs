//! Optional integrations
//!
//! RLST keeps native and ecosystem integrations behind Cargo features so applications only build
//! what they use. This guide summarizes setup and the contracts that differ from ordinary arrays.
//!
//! # FFTW transforms
//!
//! Enable `fftw` together with exactly one backend feature: `fftw_system`, `fftw_source`, or
//! `fftw_mkl`. The system backend needs both single- and double-precision FFTW libraries. FFTW is
//! GPL-licensed; review the licensing implications before distributing linked binaries.
//!
//! FFT plans store pointers into the planned array. In the 0.7 development API, `borrow_mut()`
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
//! Enable `burn` to expose `rlst::interface::burn`. RLST 0.7 development uses Burn 0.21; a
//! downstream application should select a compatible Burn 0.21 backend. Conversions consume and
//! copy their input while preserving shape: tensors become row-major RLST arrays, and RLST arrays
//! are evaluated in row-major order on the backend's default device.
//!
//! ```ignore
//! use rlst::interface::burn::{RlstArrayIntoBurnTensor, TensorIntoRlstArray};
//!
//! // Given a concrete Burn 0.21 backend `Backend`:
//! let tensor = array.into_tensor::<Backend>()?;
//! let round_trip = tensor.into_array::<f32>()?;
//! ```
//!
//! The requested Rust scalar must exactly match the Burn tensor/backend dtype. Real floating-point
//! and integer scalar families are supported; complex RLST arrays are not.
