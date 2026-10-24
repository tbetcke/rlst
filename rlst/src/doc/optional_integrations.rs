//! Optional integrations
//!
//! RLST keeps native and ecosystem integrations behind Cargo features so applications only build
//! what they use. This guide summarizes setup and the contracts that differ from ordinary arrays.
//!
//! # FFTW transforms
//!
//! The module [`rlst::fftw`](crate::fftw) wraps FFTW's serial transforms: complex-to-complex,
//! real-to-complex and complex-to-real plans, on interleaved and on split complex storage, and
//! real-to-real plans. A plan is made once for a [`Layout`](crate::fftw::Layout), the
//! dimensions and strides of the transforms and of their batch, and executes on any buffers
//! that fit that layout and the plan's alignment classes, from any thread. This guide covers
//! the setup, a worked example, the main choices, and the rules on threads and determinism;
//! the rustdoc of each item states it in full and has an example.
//!
//! The examples of this section run when the feature `fftw` is enabled.
//!
//! ## Setup
//!
//! Enable the feature `fftw_system`, the only back end, which implies `fftw`; `fftw` alone is
//! a compile error. The feature `fftw_threads` implies `fftw_system` and adds threaded plans.
//!
//! ```toml
//! [dependencies]
//! rlst = { version = "0.11", features = ["fftw_system"] }
//! ```
//!
//! - **Versions.** RLST needs FFTW 3.3.4 or later, in both precisions (`fftw3` and
//!   `fftw3f`), and 3.3.6 or later with `fftw_threads`, which also links `fftw3_threads` and
//!   `fftw3f_threads`. The build checks the version pkg-config reports, and the first planner
//!   call checks the version of the library linked at run time. 3.3.10 or later is
//!   recommended: 3.3.9 fixed the alignment requirement of FFTW's AVX-512 code and its
//!   wisdom, and 3.3.10 fixed a crash from unaligned accesses of SSE2 code in double
//!   precision in rare strided real-to-complex cases. [`version`](crate::fftw::version)
//!   returns the version string of the linked library.
//! - **Discovery.** `build.rs` finds FFTW with pkg-config (`brew install fftw pkg-config`,
//!   `apt install libfftw3-dev pkg-config`, or `dnf install fftw-devel pkgconf-pkg-config`),
//!   with a fallback to Homebrew's paths on macOS. For a custom installation, set
//!   `PKG_CONFIG_PATH` to the directory that holds `fftw3.pc` and `fftw3f.pc`.
//! - **Apple silicon.** Homebrew's FFTW on Apple silicon is built without SIMD, which is
//!   correct but slower. Where its speed matters, build FFTW with `--enable-neon` (and with
//!   `--enable-single --enable-neon` for `fftw3f`), with `--enable-threads` for
//!   `fftw_threads`, and point `PKG_CONFIG_PATH` at that installation.
//! - **Licence tools.** fftw-sys declares its licence through a `license-file`, so cargo-deny
//!   and similar tools likely need a clarification entry for it in configurations that
//!   enable `fftw`.
//!
//! ## Licence
//!
//! > **Warning: FFTW is licensed under the GNU General Public License (version 2 or later).**
//! > RLST's optional `fftw` feature links FFTW and the GPL-licensed `fftw-sys` bindings. RLST
//! > itself is licensed MIT / Apache-2.0, but a program built with this feature links
//! > GPL-licensed code, and distributing such a
//! > program is subject to the terms of the GPL unless you hold a non-free FFTW licence from
//! > MIT's Technology Licensing Office. Cargo enables a feature for the whole build when any crate in the dependency graph requests it,
//! > so check your graph (for example with `cargo tree -e features`). The feature is off by
//! > default. This note is not legal advice; check the licences that apply to your use.
//!
//! ## A worked example
//!
//! Many units of the same shape, each a batch of transforms along one axis: here four boxes
//! of `[140, 9]` complex values, the slices of a column-major parent `[140, 9, 4]`, each
//! transformed along its first axis. The plan is made once, for the layout of one box, and
//! executed on every box.
//!
//! ```
//! # #[cfg(feature = "fftw")]
//! # {
//! use rlst::fftw::{AlignmentClass, Direction, Layout, Planner, Scale};
//! use rlst::{c64, rlst_dynamic_array};
//!
//! let (n, rings, boxes) = (140, 9, 4);
//! // `FFTW_ALIGNED` buffers start at alignment class 0, the class plans are made for.
//! let mut input = rlst_dynamic_array!(c64, [n, rings, boxes] | rlst::FFTW_ALIGNED);
//! let mut spectrum = rlst_dynamic_array!(c64, [n, rings, boxes] | rlst::FFTW_ALIGNED);
//! let mut output = rlst_dynamic_array!(c64, [n, rings, boxes] | rlst::FFTW_ALIGNED);
//! input.fill_from_seed_equally_distributed(0);
//!
//! // One layout, from the first box: 9 transforms of length 140, along axis 0.
//! let layout = Layout::along_axis(
//!     &input.r().slice::<2>(2, 0),
//!     &spectrum.r().slice::<2>(2, 0),
//!     0,
//! )?;
//! let forward = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
//! let backward = Planner::<f64>::new().c2c(&layout, Direction::Backward)?;
//!
//! // Each box is 140 * 9 complex doubles, a multiple of 64 bytes, so every box is at
//! // class 0 and fits the plans.
//! assert_eq!(AlignmentClass::of(&input.r().slice::<2>(2, 3)), AlignmentClass::default());
//! for k in 0..boxes {
//!     forward.execute(&input.r().slice::<2>(2, k), &mut spectrum.r_mut().slice::<2>(2, k))?;
//!     backward.execute(&spectrum.r().slice::<2>(2, k), &mut output.r_mut().slice::<2>(2, k))?;
//! }
//!
//! // FFTW does not normalise: the round trip multiplies by n.
//! let scale = backward.scale_factor(Scale::Inverse);
//! for (x, y) in input.iter_value().zip(output.iter_value()) {
//!     assert!((x - scale * y).norm() < 1e-12);
//! }
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The boxes could be executed from several threads at once with the same plan (rule 2
//! below), and give the same bits as executed one by one.
//!
//! ## Layouts from arrays and views
//!
//! [`Layout::along_axis`](crate::fftw::Layout::along_axis) and
//! [`Layout::over_axes`](crate::fftw::Layout::over_axes) take an input and an output array of
//! the same rank, and their in-place variants one array. They accept owned arrays and views of
//! any memory order: column-major, row-major and strided arrays, `r()` and `r_mut()`, slices,
//! multislices, subviews and nested views. FFTW receives the dimensions in the order of the
//! axes given, each with its own strides, and touches only the elements a layout addresses,
//! never the elements of other views between them. Execution checks each buffer's shape,
//! strides and storage against the plan before FFTW is called.
//!
//! ```
//! # #[cfg(feature = "fftw")]
//! # {
//! use rlst::fftw::{Direction, Layout, Planner};
//! use rlst::{c64, rlst_dynamic_array};
//!
//! // Transforms along the rows of the top-left 4 x 8 block of a 6 x 8 matrix.
//! let matrix = rlst_dynamic_array!(c64, [6, 8] | rlst::FFTW_ALIGNED);
//! let mut result = rlst_dynamic_array!(c64, [6, 8] | rlst::FFTW_ALIGNED);
//! let block = matrix.r().into_subview([0, 0], [4, 8]);
//! let layout = Layout::along_axis(&block, &result.r().into_subview([0, 0], [4, 8]), 1)?;
//! let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
//! plan.execute(&block, &mut result.r_mut().into_subview([0, 0], [4, 8]))?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A plan describes memory, not logical axes: it runs on any arrays of the same memory
//! layout and alignment class. Slices with [`Layout::new`](crate::fftw::Layout::new) and its
//! companions are the low-level form, for callers that manage their own buffers.
//!
//! ## Families and input policies
//!
//! The storage of each side of a layout, from the element types of its arrays, gives the
//! family: complex to complex ([`Planner::c2c`](crate::fftw::Planner::c2c)), real to complex
//! ([`r2c`](crate::fftw::Planner::r2c)), complex to real ([`c2r`](crate::fftw::Planner::c2r)),
//! real to real ([`r2r`](crate::fftw::Planner::r2r)), and their split forms for complex
//! values stored as a real plane and an imaginary plane
//! ([`split_c2c`](crate::fftw::Planner::split_c2c) and its companions). For r2c and c2r, the
//! last axis given is halved: `n` reals along it on one side, `n / 2 + 1` complex values on
//! the other. Whether to halve the contiguous axis or a strided one was measured for 0.11:
//! neither order was faster on every machine, so RLST recommends neither; time both at your
//! sizes if it matters. For r2r, `kinds[i]` applies to `axes[i]`, whatever the memory order.
//!
//! A plan's input policy is part of its type, so the borrow checker enforces it: a
//! `Preserving` plan executes with `&input`, a `Destroying` plan with `&mut input`, an
//! `InPlace` plan on one `&mut` buffer. Each family follows FFTW's default. **A
//! complex-to-real plan destroys its input unless
//! [`preserve_input`](crate::fftw::Planner::preserve_input) is given**, which FFTW supports
//! over one transform dimension only;
//! [`may_destroy_input`](crate::fftw::Planner::may_destroy_input) lets any other plan
//! overwrite its input, which can be faster.
//!
//! ```
//! # #[cfg(feature = "fftw")]
//! # {
//! use rlst::fftw::{Layout, Planner, R2rKind};
//! use rlst::{c64, rlst_dynamic_array};
//!
//! // Real to complex and back along the first axis: 8 reals, 5 complex values.
//! let mut data = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
//! let mut spectrum = rlst_dynamic_array!(c64, [5, 3] | rlst::FFTW_ALIGNED);
//! let mut back = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
//! data.fill_from_seed_equally_distributed(0);
//! let r2c = Planner::<f64>::new().r2c(&Layout::along_axis(&data, &spectrum, 0)?)?;
//! r2c.execute(&data, &mut spectrum)?;
//!
//! // The default c2r plan destroys its input, so it takes `&mut spectrum`.
//! let c2r = Planner::<f64>::new().c2r(&Layout::along_axis(&spectrum, &back, 0)?)?;
//! c2r.execute(&mut spectrum, &mut back)?;
//! assert!((back[[2, 1]] - 8.0 * data[[2, 1]]).abs() < 1e-12);
//!
//! // A DCT-II along axis 1 and a DST-II along axis 0, in that order of `axes`.
//! let mut out = rlst_dynamic_array!(f64, [8, 3] | rlst::FFTW_ALIGNED);
//! let layout = Layout::over_axes(&data, &out, &[1, 0])?;
//! let plan = Planner::<f64>::new().r2r(&layout, &[R2rKind::Redft10, R2rKind::Rodft10])?;
//! plan.execute(&data, &mut out)?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! RLST 0.10 applied the kinds of an r2r plan on a column-major array in reverse order of the
//! axes; code written for it with different kinds on different axes must be checked.
//!
//! ## Alignment classes
//!
//! FFTW's SIMD code requires each buffer to be aligned as the buffer the plan was made for. A
//! plan records the [`AlignmentClass`](crate::fftw::AlignmentClass) of its input and output,
//! the address of the first element modulo [`PLAN_ALIGNMENT`](crate::fftw::PLAN_ALIGNMENT)
//! (64 bytes), class 0 by default, and execution rejects a buffer of another class with
//! [`FftwError::AlignmentMismatch`](crate::fftw::FftwError::AlignmentMismatch). Allocate plan
//! targets with `rlst_dynamic_array!(…, shape | rlst::FFTW_ALIGNED)` or as an
//! `AlignedDynArray`, which start at class 0; a plain `DynArray` is aligned only to its
//! element. For a view at an offset, plan for its class with
//! [`Planner::alignment`](crate::fftw::Planner::alignment), or plan without alignment
//! requirements with [`unaligned`](crate::fftw::Planner::unaligned), at the cost of most SIMD
//! code.
//!
//! ```
//! # #[cfg(feature = "fftw")]
//! # {
//! use rlst::fftw::{AlignmentClass, Direction, FftwError, Layout, Planner};
//! use rlst::{c64, rlst_dynamic_array};
//!
//! let data = rlst_dynamic_array!(c64, [9, 3] | rlst::FFTW_ALIGNED);
//! let mut result = rlst_dynamic_array!(c64, [9, 3] | rlst::FFTW_ALIGNED);
//! // Rows 1 to 8 start one complex double, 16 bytes, after the aligned start.
//! let view = data.r().into_subview([1, 0], [8, 3]);
//! let layout = Layout::along_axis(&view, &result.r().into_subview([1, 0], [8, 3]), 0)?;
//!
//! let plan = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
//! let error = plan.execute(&view, &mut result.r_mut().into_subview([1, 0], [8, 3]));
//! assert!(matches!(error, Err(FftwError::AlignmentMismatch { .. })));
//!
//! let class = AlignmentClass::of(&view);
//! let plan = Planner::<f64>::new()
//!     .alignment(class, class)
//!     .c2c(&layout, Direction::Forward)?;
//! plan.execute(&view, &mut result.r_mut().into_subview([1, 0], [8, 3]))?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Scratch and the `*_on` methods
//!
//! The planner methods plan on scratch buffers that they allocate for the layout's span, so
//! planning never touches user data, whatever the rigor. The scratch of a strided view spans
//! its whole parent, and it is bounded by [`max_scratch`](crate::fftw::Planner::max_scratch),
//! 1 GiB by default; above it, planning returns
//! [`FftwError::ScratchTooLarge`](crate::fftw::FftwError::ScratchTooLarge). The `*_on`
//! methods, such as [`c2c_on`](crate::fftw::Planner::c2c_on), plan on the caller's buffers
//! instead and need no scratch. With [`Rigor::Estimate`](crate::fftw::Rigor::Estimate), the
//! default, FFTW leaves them untouched; **with `Measure` and stronger, it overwrites them**.
//!
//! ```
//! # #[cfg(feature = "fftw")]
//! # {
//! use rlst::fftw::{Direction, FftwError, Layout, Planner};
//! use rlst::{StridedDynArray, c64};
//!
//! // One column of a row-major 4 x 1000 array spans nearly the whole array.
//! let mut input = StridedDynArray::<c64, 2>::row_major([4, 1000]);
//! let mut output = StridedDynArray::<c64, 2>::row_major([4, 1000]);
//! let layout = Layout::along_axis(&input.r().col(7), &output.r().col(7), 0)?;
//! let planner = Planner::<f64>::new().max_scratch(1024);
//! let error = planner.c2c(&layout, Direction::Forward).unwrap_err();
//! assert!(matches!(error, FftwError::ScratchTooLarge { .. }));
//!
//! let plan = planner.c2c_on(
//!     &mut input.r_mut().col(7),
//!     &mut output.r_mut().col(7),
//!     &layout,
//!     Direction::Forward,
//! )?;
//! plan.execute(&input.r().col(7), &mut output.r_mut().col(7))?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Scaling
//!
//! FFTW's transforms are unnormalised: a forward transform followed by the backward one
//! multiplies by the plan's [logical size](crate::fftw::Plan::logical_size), the product of
//! the transform lengths, or for r2r of each kind's logical size (`2 (n - 1)` for a DCT-I).
//! [`Plan::scale_factor`](crate::fftw::Plan::scale_factor) returns the factor for a
//! [`Scale`](crate::fftw::Scale), `None`, `Inverse` (`1 / N`) or `InverseSqrt`
//! (`1 / √N`), which the caller applies with RLST's array arithmetic. A uniform factor does
//! not make the DCT and DST kinds orthonormal.
//!
//! ## Errors
//!
//! Layout functions, planners and execution return
//! [`FftwError`](crate::fftw::FftwError), whose variants name the values involved: an invalid
//! layout, a buffer that does not fit the plan, another alignment class, overlapping or
//! misplaced buffers, an invalid r2r kind (a DCT-I needs two points), an unsupported
//! `preserve_input`, scratch above the bound, an FFTW older than the minimum, a plan FFTW
//! refuses, and wisdom or thread errors. Every check runs before FFTW is called, so a failed
//! execution leaves the output untouched.
//!
//! ## Threads and determinism: seven rules
//!
//! 1. **One lock for FFTW's global state.** Every FFTW call RLST makes, other than execution,
//!    runs under one process-wide lock, so RLST's planners, plans and wisdom functions can be
//!    used from any thread. Other libraries in the process that call FFTW, and a second copy
//!    of RLST in the same build, are not covered; with `fftw_threads`,
//!    [`make_planner_thread_safe`](crate::fftw::threads::make_planner_thread_safe) extends
//!    protection to their plan creation and destruction only. See
//!    [`rlst::fftw::threads`](crate::fftw::threads) (feature `fftw_threads`).
//! 2. **Concurrent execution.** One plan may be executed from several threads at once on
//!    different buffers; this is safe and takes no lock. See [`Plan`](crate::fftw::Plan).
//! 3. **Same plan, same bits.** A plan gives bit-for-bit identical results on inputs of the
//!    same alignment class. Two plans are the same plan when their layout, alignment classes,
//!    input policy, rigor, wisdom-only setting and thread count are equal, they were made
//!    with the same FFTW build on a CPU with the same SIMD support, and they found the same
//!    wisdom: an `Estimate` plan made after a stronger plan of the same problem, or after a
//!    wisdom import, can be another plan. See
//!    [`Plan`, "Threads and determinism"](crate::fftw::Plan#threads-and-determinism).
//! 4. **Verify rather than assume.** [`Plan::fingerprint`](crate::fftw::Plan::fingerprint)
//!    identifies the algorithm FFTW chose. A distributed caller that needs the same plan on
//!    every rank compares fingerprints across ranks, for example in one all-reduce, and acts
//!    on the result on all ranks together.
//! 5. **Timed planning.** `Measure`, `Patient` and `Exhaustive` choose algorithms by timing
//!    them; their plans, and their bits, can differ between runs and processes. See
//!    [`Plan`](crate::fftw::Plan#threads-and-determinism).
//! 6. **Batch and threads.** A plan for another batch count or layout is another plan and can
//!    round differently. For results that do not depend on the number of threads, fix the
//!    unit of work, as the boxes of the worked example, and execute one plan on every unit;
//!    never size a plan from a scheduler's chunking. A threaded plan can round differently
//!    from a single-threaded one. See [`Plan`](crate::fftw::Plan#threads-and-determinism).
//! 7. **Shared wisdom.** For the same plans in several processes, import the same wisdom in
//!    every process before any planning, plan with
//!    [`wisdom_only`](crate::fftw::Planner::wisdom_only), and compare fingerprints (rule 4).
//!    Wisdom made on a CPU with other SIMD support may not apply. See
//!    [`rlst::fftw::wisdom`](crate::fftw::wisdom#wisdom-and-determinism).
//!
//! Two more notes on these rules:
//!
//! - **Compare fingerprints of one problem and one storage form.** A fingerprint names
//!   FFTW's algorithm, not the problem: plans in both directions, both precisions or both
//!   input policies can share one, and a split plan can have the fingerprint of an
//!   interleaved one on one build and another on the next.
//! - **Wisdom calls hold the lock that planning needs.** Exporting, importing or forgetting
//!   wisdom in a tight loop delays every planner in the process; call them once, around
//!   planning.
//!
//! Plans with `Measure` and wisdom files belong in a program's setup, not in a loop:
//!
//! ```no_run
//! # #[cfg(feature = "fftw")]
//! # {
//! use rlst::fftw::{Direction, Layout, Planner, Rigor, wisdom};
//! use rlst::{c64, rlst_dynamic_array};
//!
//! // Import the wisdom of an earlier run, if there is any.
//! let _ = wisdom::import_from_file::<f64>("fftw-wisdom-f64");
//!
//! let mut input = rlst_dynamic_array!(c64, [588, 9] | rlst::FFTW_ALIGNED);
//! let mut output = rlst_dynamic_array!(c64, [588, 9] | rlst::FFTW_ALIGNED);
//! let layout = Layout::along_axis(&input, &output, 0)?;
//! // `Measure` on scratch: the arrays are not touched.
//! let plan = Planner::<f64>::new()
//!     .rigor(Rigor::Measure)
//!     .c2c(&layout, Direction::Forward)?;
//! println!("plan {:016x}", plan.fingerprint());
//! plan.execute(&input, &mut output)?;
//! input.fill_from_seed_equally_distributed(1);
//!
//! wisdom::export_to_file::<f64>("fftw-wisdom-f64")?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Chebychev conversions
//!
//! The conversions of [`rlst::chebychev`](crate::chebychev) between data at Chebychev points
//! and series coefficients use these plans: one DCT-I per axis of extent above 1, planned
//! with `Estimate` on the result array. Since 0.11 they return `Result<_, FftwError>`, take
//! any number of dimensions, treat an axis of extent 1 as constant and an axis of extent 0 as
//! empty, where 0.10 panicked; their results agree with 0.10's to round-off.
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

// The FFTW section links into `rlst::fftw` and `rlst::chebychev`, which exist only with `fftw`,
// and into `rlst::fftw::threads`, which exists only with `fftw_threads`. Builds with
// `fftw_threads`, docs.rs among them, check every link.
#![cfg_attr(not(feature = "fftw_threads"), allow(rustdoc::broken_intra_doc_links))]
