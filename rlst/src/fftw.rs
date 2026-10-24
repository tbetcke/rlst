//! Fast Fourier transforms with FFTW.
//!
//! This module wraps every transform family of FFTW's serial interface, through FFTW's 64-bit
//! guru interface: complex-to-complex, real-to-complex and complex-to-real plans, on
//! interleaved and on split complex storage, and real-to-real plans, on slices and on RLST
//! arrays and views. A plan is independent of the buffers it runs on: it is made once for a
//! [`Layout`] and runs on any buffers that fit that layout and its [`AlignmentClass`]es, from
//! any thread, with every buffer checked before FFTW is called.
//!
//! Enable the feature `fftw_system`, the only back end, which implies `fftw`. The system must
//! provide both single- and double-precision libraries (`fftw3f` and `fftw3`), version 3.3.4
//! or later, normally discovered through `pkg-config`. The feature `fftw_threads`, which
//! implies `fftw_system`, adds threaded plans; it also links FFTW's threads libraries
//! (`fftw3_threads` and `fftw3f_threads`) and needs version 3.3.6 or later. The guide in
//! [`doc::optional_integrations`](crate::doc::optional_integrations) describes the setup.
//!
//! # Parts
//!
//! - [`Layout`], made from [`Dim`]s, the [`Storage`] of each side (complex, real or split)
//!   and a [`Placement`]: transform and batch dimensions with strides, validated and
//!   normalised, whose storages give the family ([`Layout::with_storage`]);
//! - layouts from RLST arrays and views of any memory order, subviews, slices and nested views
//!   included: [`Layout::along_axis`], [`Layout::over_axes`] and their in-place variants,
//!   which pass FFTW the dimensions in the order of the axes given, with their own strides,
//!   and read each side's storage from the [`FftElement`] type of its array;
//! - complex-to-complex transforms, [`C2c`], in a [`Direction`], made by [`Planner::c2c`] and
//!   its companions;
//! - real-to-complex and complex-to-real transforms, [`R2c`] and [`C2r`]: a real array and a
//!   complex one whose extent along the halved axis, the last of the axes given, is `n / 2 +
//!   1` for the real extent `n`, made by [`Planner::r2c`], [`Planner::c2r`] and their
//!   companions. In place, the complex array holds the real side in its reals, padded
//!   ([`Layout::r2c_inplace`], [`Layout::c2r_inplace`]), which [`as_reals`] and
//!   [`as_reals_mut`] give access to. As in FFTW, **a complex-to-real plan destroys its input
//!   unless [`Planner::preserve_input`] is given**, which it supports over one transform
//!   dimension only;
//! - real-to-real transforms, [`R2r`], with all eleven [`R2rKind`]s of FFTW and one kind per
//!   transform dimension, `kinds[i]` for `axes[i]` whatever the memory order, made by
//!   [`Planner::r2r`] and its companions;
//! - split complex storage, the real and imaginary parts in one buffer of reals a fixed gap
//!   apart, such as a real plane and then an imaginary plane: a split complex array is a real
//!   array with a *part axis* of extent 2, and [`Layout::split_along_axis`],
//!   [`Layout::split_over_axes`], their in-place variants and [`Layout::new_split`] make
//!   layouts with gaps, which count reals, for [`Planner::split_c2c`] and its companions;
//! - split real-to-complex and complex-to-real transforms, [`SplitR2c`] and [`SplitC2r`],
//!   between a real array and a split complex one with a part axis
//!   ([`Layout::split_r2c_along_axis`], [`Layout::split_c2r_along_axis`] and their
//!   companions), made by [`Planner::split_r2c`] and [`Planner::split_c2r`] and their
//!   companions; they have no in-place plans, and split c2r destroys its input by default;
//! - [`Planner`], a builder with [`Rigor`], wisdom-only, unaligned, time-limit, alignment-class
//!   and scratch options, and with `fftw_threads` a thread count (`Planner::threads`), whose
//!   input-policy requests ([`DefaultRequest`], [`PreserveRequest`], [`DestroyRequest`])
//!   decide a plan's policy;
//! - [`Plan`], of precision [`FftwReal`], family [`C2c`], [`SplitC2c`], [`R2c`], [`C2r`],
//!   [`R2r`], [`SplitR2c`] or [`SplitC2r`] and policy [`Preserving`], [`Destroying`] or
//!   [`InPlace`], which runs on any buffers that fit its layout and [`AlignmentClass`]es,
//!   from any thread, with [`Plan::description`], [`Plan::fingerprint`], [`Plan::threads`],
//!   [`Plan::logical_size`] and [`Plan::scale_factor`] for a [`Scale`];
//! - [`FftBuffer`] and [`FftBufferMut`], implemented for slices and for RLST arrays and views
//!   through [`StridedRawAccess`](crate::StridedRawAccess), checked at execution by shape,
//!   strides and storage;
//! - [`FftwError`] and [`Side`]; [`PLAN_ALIGNMENT`]; [`Backend`], [`backend`] and [`version`];
//! - [`wisdom`]: FFTW's wisdom of each precision, exported to and imported from strings and
//!   files, forgotten, and the system's wisdom, with what wisdom means for determinism;
//! - `threads`, with the feature `fftw_threads`: RLST's FFTW lock, its limits, and
//!   `make_planner_thread_safe`, which adds FFTW's own lock for other libraries that plan
//!   with FFTW in the process.
//!
//! Plans are made on scratch buffers unless a `*_on` method is used, so planning never
//! touches user data. Execution checks the buffers against the plan before calling FFTW and
//! takes no lock. Every other FFTW call RLST makes, planning, plan destruction and wisdom
//! included, runs under one process-wide lock. Split plans can use fewer of FFTW's SIMD
//! stages than interleaved ones (see [`SplitC2c`]). Every planner method rejects a layout of
//! another family. FFTW's transforms are unnormalised: a forward and a backward transform
//! multiply by the [logical size](Plan::logical_size), and [`Plan::scale_factor`] gives the
//! factor that undoes it.
//!
//! The rules on threads and determinism, when two plans are the same plan and give the same
//! bits, are stated on [`Plan`](Plan#threads-and-determinism) and [`Plan::fingerprint`]; the
//! rules on shared wisdom in [`wisdom`].
//!
//! # Example
//!
//! A batch of transforms along the first axis of an RLST array, planned once and executed on
//! two arrays of the same layout and alignment class:
//!
//! ```
//! use rlst::{c64, rlst_dynamic_array};
//! use rlst::fftw::{Direction, Layout, Planner, Scale};
//!
//! // Three transforms of length 8, one per column, in buffers aligned to `FFTW_ALIGNED`.
//! let mut data = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
//! let mut spectrum = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
//! let mut result = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
//! data.fill_from_seed_equally_distributed(0);
//!
//! let layout = Layout::along_axis(&data, &spectrum, 0)?;
//! let forward = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
//! let backward = Planner::<f64>::new().c2c(&layout, Direction::Backward)?;
//!
//! forward.execute(&data, &mut spectrum)?;
//! backward.execute(&spectrum, &mut result)?;
//!
//! // The round trip multiplies by the transform length, which `Scale::Inverse` undoes.
//! let scale = backward.scale_factor(Scale::Inverse);
//! for (x, y) in data.iter_value().zip(result.iter_value()) {
//!     assert!((x - scale * y).norm() < 1e-12);
//! }
//!
//! // The same plans run on any other arrays of this layout and alignment class.
//! let mut other = rlst_dynamic_array!(c64, [8, 3] | rlst::FFTW_ALIGNED);
//! other.fill_from_seed_equally_distributed(1);
//! forward.execute(&other, &mut spectrum)?;
//! # Ok::<(), rlst::fftw::FftwError>(())
//! ```

mod align;
mod backend;
mod buffer;
#[cfg(test)]
mod concurrency_tests;
mod element;
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
mod scale;
#[cfg(test)]
mod test_support;
#[cfg(feature = "fftw_threads")]
#[cfg_attr(docsrs, doc(cfg(feature = "fftw_threads")))]
pub mod threads;
mod version;
pub mod wisdom;

pub use align::{AlignmentClass, PLAN_ALIGNMENT};
pub use backend::{Backend, backend, version};
pub use buffer::{FftBuffer, FftBufferMut};
pub use element::{FftElement, as_reals, as_reals_mut};
pub use error::{FftwError, Side};
pub use kind::{Direction, R2rKind};
pub use layout::{Dim, Layout, Placement, Storage};
pub use options::{DefaultRequest, DestroyRequest, PreserveRequest, Rigor};
pub use plan::{
    C2c, C2r, Destroying, Family, InPlace, Plan, Policy, Preserving, R2c, R2r, SplitC2c, SplitC2r,
    SplitR2c,
};
pub use planner::{DEFAULT_MAX_SCRATCH, Planner};
pub use real::FftwReal;
pub use scale::Scale;
