//! The Rust linear solver toolbox (RLST).
//!
//! The purpose of this library is to provide a comprehensive set of tools
//! for dense and sparse linear algebra operations required in the solution
//! of partial differential equations and other problems.
//! RLST has the following feature set.
//! - n-dimensional array structures that can be allocated on the stack or heap.
//! - BLAS matrix multiplication and LAPACK decompositions, eigenproblems, and matrix equations.
//! - CSR sparse matrices on one node or distributed with MPI, plus optional UMFPACK solves through
//!   the companion `rlst-suitesparse` crate.
//! - Matrix Market import and export.
//! - Abstract linear operators and iterative solvers.
//! - Optional FFTW transforms and Burn tensor conversion.
//!
//! To learn about the features of RLST please have a look at the following documents.
//! - [Getting started with RLST](crate::doc::getting_started)
//! - [Changelog and migration notes](crate::doc::changelog)
//! - [An introduction to dense linear algebra with RLST](crate::doc::dense_linear_algebra)
//! - [Matrix decompositions](crate::doc::matrix_decompositions)
//! - [Sparse matrix operations](crate::doc::sparse_matrices)
//! - [Abstract linear algebra and iterative solvers](crate::doc::abstract_linear_algebra)
//! - [MPI distributed computations](crate::doc::distributed_computations)
//! - [Optional FFTW and Burn integrations](crate::doc::optional_integrations)
//! - [Interpolation utilities](crate::interpolation)

#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod base_types;
#[cfg(feature = "fftw")]
pub mod chebychev;
pub mod dense;
#[cfg(feature = "mpi")]
pub mod distributed_tools;
pub mod doc;
#[cfg(feature = "fftw")]
pub mod fftw;
pub mod interface;
pub mod interpolation;
pub mod io;
pub mod simd;
pub mod sparse;
pub mod tracing;
pub mod traits;

pub mod threading;

pub mod operator;

pub use operator::frame::{
    ArrayFrame, ElementFrame, FrameQr, FrameQrDecomposition, FrameQrOptions,
};

// Re-exports
pub use rlst_proc_macro::measure_duration;
pub use rlst_proc_macro::rlst_dynamic_array;
pub use rlst_proc_macro::rlst_static_array;
pub use rlst_proc_macro::rlst_static_type;

pub use crate::dense::array::Array;
pub use crate::dense::array::empty_array;

// Re-export the traits
pub use traits::*;

// Re-export the base types
pub use base_types::*;

pub use dense::array::DynArray;
pub use dense::array::SliceArray;
pub use dense::array::SliceArrayMut;
pub use dense::array::StridedDynArray;
pub use dense::array::ZeroDimArray;

// Important constants

/// Align memory along cache lines
pub const CACHE_ALIGNED: usize = aligned_vec::CACHELINE_ALIGN;

/// Align memory for FFTW. FFTW requires 16 bytes alignment
pub const FFTW_ALIGNED: usize = 16;

/// Align memory along page lines
///
// On x86_64 architectures a page size of 4kb is assumed.
// On other architectures we default to 64kb to take into account
// large page sizes on some ARM platforms.
pub const PAGE_ALIGNED: usize = {
    if cfg!(target_arch = "x86_64") {
        4096
    } else {
        65536
    }
};

#[cfg(test)]
mod tests {

    extern crate blas_src;
    extern crate lapack_src;

    #[cfg(target_os = "linux")]
    extern crate openblas_src;
}
