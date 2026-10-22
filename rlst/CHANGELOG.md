# Changelog

## v0.11.0 (unreleased)

Changes since [v0.10.3](https://github.com/tbetcke/rlst/releases/tag/v0.10.3), on the
`fftw-0.11` integration branch. All workspace crates use version 0.11.0-dev until the release.

### FFTW: `fftw_system` is the only back end

The `fftw_source` and `fftw_mkl` features are removed. `fftw_system`, which links the system's
`libfftw3` and `libfftw3f`, is now the only back end, and it implies `fftw`: enabling `fftw`
without `fftw_system` is a compile error instead of a link error. `build.rs` rejects an FFTW
older than 3.3.4 reported by pkg-config, and the crate checks the version of the library linked
at run time; 3.3.10 or later is recommended. When the `DOCS_RS` environment variable is set,
`build.rs` skips FFTW discovery, so the documentation builds without FFTW installed.

**Migration:** replace `fftw_source` by `fftw_system` and install FFTW from the system (`brew
install fftw pkg-config`, `apt install libfftw3-dev pkg-config` or `dnf install fftw-devel
pkgconf-pkg-config`; set `PKG_CONFIG_PATH` for a custom installation). Intel MKL's FFTW3
wrappers (`fftw_mkl`) are out of scope for now; a later MKL back end would be an addition.
`features = ["fftw", "fftw_system"]` keeps working, and `features = ["fftw_system"]` is enough.

### `FFTW_ALIGNED` is 64

`rlst::FFTW_ALIGNED` changes from 16 to 64 bytes, the modulus of the alignment classes of the
new FFTW plans: buffers allocated with `rlst_dynamic_array!(…, shape | rlst::FFTW_ALIGNED)` or
as `AlignedDynArray<…, { rlst::FFTW_ALIGNED }>` start at class 0, the class plans are made for
by default. Equal classes modulo 64 imply equal alignment for FFTW on every build.

**Migration:** none for code that only passes the constant to RLST's allocation; code that
relied on its value being 16 must use 16 directly.

### New FFTW interface (in progress)

A new FFTW interface replaces the 0.10 one: plans independent of the arrays they run on,
checked and sound execution, input policies in the type of a plan, alignment classes and plan
fingerprints. This entry is completed, with migration notes, at the release. So far it
provides complex-to-complex plans on slices (`Layout`, `Planner`, `Plan` and their companions
in `rlst::fftw`); the 0.10 types stay beside them, unchanged, until the new interface is
complete.

Plans also run on RLST arrays and views. `Layout::along_axis`, `Layout::over_axes` and their
in-place variants build layouts from arrays of any memory order, column-major, row-major and
strided, and from subviews, slices, multislices and nested views; they pass FFTW the
dimensions in the order of the axes given, each with its own strides, so no shape is reversed
for column-major arrays, and they reject unequal shapes, repeated or out-of-range axes and
extents of 0 with `InvalidLayout` naming the axis. Arrays and views are `FftBuffer`s and, when
mutable, `FftBufferMut`s: execution checks an array's shape and strides against its side of
the plan, pairs the axes of an input and an output array of the same rank, and checks that
the array's storage holds the layout's span from its first element on, before FFTW is
called. FFTW touches only the elements a layout addresses, never the elements of other views
between them. A plan describes memory, not logical axes: it runs on any arrays of the same
memory layout and alignment class.

As tested, one plan executes from many threads at once, and `Estimate` plans give the same
fingerprint and the same bits in every fresh process; the rustdoc of `Plan` and
`Plan::fingerprint` states the rules on threads and determinism.

Complex-to-complex plans also run on split storage: the real and imaginary parts in one
buffer of reals, a fixed gap apart, such as a real plane followed by an imaginary plane. A
split complex array is a real array with a *part axis* of extent 2 (index 0 the real part,
index 1 the imaginary part) whose stride is the gap. `Layout::split_along_axis`,
`Layout::split_over_axes` and their in-place variants build split layouts from real arrays and
views of any memory order, with the part axis anywhere and possibly at different positions on
input and output; their axes count the logical axes, the axes without the part axis.
`Layout::new_split` builds split layouts from raw dimensions for slices, and `Layout::gaps`
returns the gaps. `Planner::split_c2c`, `split_c2c_on`, `split_c2c_inplace` and
`split_c2c_inplace_on` make `Plan<T, SplitC2c, P>` plans with the three input policies, which
execute on slices of reals and on real arrays and views, checked as interleaved plans are,
the part axis and the gap included. FFTW's split planner always plans forward; a backward
plan swaps the real and imaginary parts on both sides. The interleaved planner methods reject
a split layout, and the split ones an interleaved layout, with `InvalidLayout` naming the part
axis. Split plans can use fewer of FFTW's SIMD stages than interleaved ones, so copying into
interleaved storage can be faster; they agree with interleaved plans to round-off, not bit for
bit.

### New traits `StridedRawAccess` and `StridedRawAccessMut`

`rlst::StridedRawAccess::storage` and `StridedRawAccessMut::storage_mut` return the backing
storage of an array or view and the offset of its first element; with the view's strides,
element `i` is `storage[offset + Σ i[k] · stride[k]]`. Unlike `RawAccess::data`, which holds
exactly the array's elements and is `None` for most views, they cover subviews and every
slice, at the price of a slice that may hold elements of other views. `RawAccess` does not
change.

They are implemented for the owned and borrowed arrays (`DynArray`, `AlignedDynArray`,
`StaticArray`, `StridedDynArray`, `SliceArray`, `StridedSliceArray` and their mutable forms,
at offset 0), for `r()` and `r_mut()`, for `slice`, `row`, `col`, `multislice` and
`into_subview`, nested and mixed, and for the wrappers of `with_container_type`, `coerce_dim`
and `insert_empty_axis`; `StridedRawAccessMut` wherever the array or view is mutable, so not
for `r()`. Lazy expressions, `transpose()`, `permute_axes()`, `reverse_axis()`, `into_flat()`,
the shared references of `into_shared` and `into_shared_atomic`, and `ZeroDimArray` do not
implement them. For a view without elements the offset may lie past the end of the slice and
must not be used to index it; a view whose offset does not fit in `usize`, which only a view
without elements can have, returns `None`.

New inherent methods `Array::storage()` and `Array::storage_mut()`, beside `data()` and
`data_mut()`, give the same access without naming the implementation type.

### FFTW: one lock for all FFTW calls

The 0.10 FFTW module now plans and destroys its plans under the same process-wide lock as the
new interface, which serialises every FFTW call RLST makes other than plan execution. Its API
and results do not change.

## v0.10.3

Changes since [v0.10.2](https://github.com/tbetcke/rlst/releases/tag/v0.10.2).
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.10.3.
No code changes: every public signature and result is identical to 0.10.2's.

### Repository moved to GitHub

The repository moved from Codeberg to <https://github.com/tbetcke/rlst>, with its full branch
and tag history. The `repository` and `homepage` fields of all three crates point there, and
continuous integration runs on GitHub Actions instead of Forgejo Actions. The Codeberg
repository is archived; its issues and pull requests remain readable there.

**Migration:** update git dependencies, `[patch]` sections and git remotes from
`https://codeberg.org/rlst/rlst` to `https://github.com/tbetcke/rlst`. Dependencies on the
crates.io releases need no change.

## v0.10.2

Changes since [v0.10.1](https://github.com/tbetcke/rlst/releases/tag/v0.10.1).
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.10.2.
No public signature or result of 0.10.1 changes.

### Sparse exchanges: rotated posting order

The non-blocking consensus round behind `sparse_all_to_allv`, `SparseRoutes::new` and
`GhostCommunicatorBuilder::build`, and `SparseRoutes`' `exchange` and `exchange_back`, posted
their messages in ascending rank order, so every rank sent to rank 0 first and all senders
converged on one receiver at a time. Rank `r` now posts to `r + 1, ..., P - 1, 0, ..., r - 1`,
as MPI's all-to-all algorithms do, and posts its receives in the same rotated order. Only the
order of posting changes: every buffer keeps its layout, and every result, the received data
and its order included, is identical to 0.10.1's.

On 4 nodes of an Omni-Path cluster at 160 ranks (Open MPI 4.1.6 over PSM2), the first
`SparseRoutes` exchange after `SparseRoutes::new` takes half as long (median 4.1 ms to 2.0
ms) and the others 5-10 % less (nd-fmm, Phase 5S T8q).

## v0.10.1

Changes since [v0.10.0](https://github.com/tbetcke/rlst/releases/tag/v0.10.0).
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.10.1.
Every change is additive or internal: no public signature or result of 0.10.0 changes.

### Sparse exchanges

`all_to_allv` exchanges the counts of every pair of ranks in an `MPI_Alltoall` and the data
in an `MPI_Alltoallv`; both take time proportional to the number of ranks `P` even when each
rank sends to a few others. The new routines communicate by a sparse dynamic exchange
(non-blocking consensus, NBX; Hoefler, Siebert and Lumsdaine 2010): one synchronous send per
message, a matched probe for whatever arrives, and a non-blocking barrier once a rank's own
messages have been received. Their cost is that of the messages plus `O(log P)`.

- **`sparse_all_to_allv`** (`distributed_tools::array_tools`, re-exported from
  `distributed_tools`) takes the arguments of `all_to_allv` and, for valid input, returns
  its result element for element: the counts received from every rank and the data grouped
  by source rank. It sends one message per nonzero count and ends with one all-reduce, which
  agrees the validity of the input on every rank: a rank with invalid counts sends nothing,
  and every rank panics after the exchange with no message pending. Totals and displacements
  no longer need to fit `i32`; each count still does.
- **`SparseRoutes`** (new module `distributed_tools::sparse_routes`) is a fixed set of routes
  for patterns built once and used often. `SparseRoutes::new` takes each rank's item counts
  per destination, as passed to `all_to_allv`, duplicates the communicator, learns every
  rank's sources and counts by NBX, and agrees the validity and `max_per_item` in one
  all-reduce. Then `exchange`/`exchange_into` send any number of values per item along the
  routes and `exchange_back`/`exchange_back_into` return values along the reversed routes,
  with point-to-point messages between the ranks that share a route only: no collective and
  no array with one entry per rank per exchange. The received values are laid out as with
  `all_to_allv`, so the bytes and their order are the same. A buffer of the wrong length
  takes part with default values and then panics on its rank only, as the ghost
  communicator's slice exchanges do.
- **`GhostCommunicatorBuilder::build`** now sends each rank's ghost indices to their owners
  by NBX, instead of an `MPI_Alltoall` of `P` counts followed by a neighbourhood all-to-all of
  the indices, and agrees its validity in two all-reduces instead of three. Every public
  result is identical to 0.10.0's, element for element on every rank: `out_ranks`,
  `in_ranks`, `send_indices`, `receive_indices`, the counts, chunk sizes, offsets and buffer
  lengths. `DataPermutation` and `Global2LocalDataMapper` build through it.
- **`SPARSE_EXCHANGE_TAG`** is the tag of the messages of `sparse_all_to_allv` and of the
  ghost build on the caller's communicator. During those calls the caller must not use the
  tag on that communicator nor keep a receive with `MPI_ANY_TAG` pending on it. A pending
  message of the caller with another tag is neither received nor disturbed.

Where it matters (measured in the nd-fmm project, Phase 5S T8, not in rlst): each FMM build
ran 21–32 `MPI_Alltoall`s of `P` counts, 29–30 of them in rlst (12–23 in
`GhostCommunicatorBuilder::build`, 7 in `all_to_allv`), 3–7 ms of a 121–373 ms build at 160
ranks; that project's model puts them at 32–113 ms per build at 10⁴ ranks (6–27 %). The
gain with this release is to be measured there.

### Fix: no empty buffers reach MPI

- `all_to_allv` no longer passes an empty send or receive buffer to MPI. With Open MPI an
  empty `Vec` of a one-byte type has the address `0x1`, which is `MPI_IN_PLACE`, so
  `MPI_Alltoallv` failed with `MPI_ERR_ARG`, for example for `u8` data on a rank that sends
  or receives nothing. The new routines never pass an empty buffer either.

### Tests and documentation

- `tests/mpi_empty_ranks.rs` has three new cases, run by `scripts/run_mpi_tests.py`:
  `sparse-all-to-allv` (against `all_to_allv` on eight count patterns, among them all-zero,
  one peer, ring, random sparse, dense, empty ranks and one sender, with `u8`, `u64` and a
  derived `Equivalence` payload; invalid counts rejected on every rank; a caller's pending
  message on the same communicator left alone through twenty back-to-back exchanges),
  `ghost-build-oracle` (every public result of the build against 0.10.0's algorithm with
  dense all-to-alls, uniform and per-index chunk sizes, self-owned and repeated ghosts and
  empty ranks; the forward values against a dense exchange; invalid owners rejected on every
  rank) and `sparse-routes` (both directions against `all_to_allv`, `per_item` 0, 1 and 3,
  three exchanges per route set, `max_per_item`, buffer mismatches and invalid counts).
- The distributed computations guide has a section on the sparse exchanges.
- `rlst-proc-macro` and `rlst-suitesparse` have no changes beyond the version number.

## v0.10.0

Changes since [v0.9.0](https://github.com/tbetcke/rlst/releases/tag/v0.9.0).
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.10.0.

### Parallel sort — exactly balanced output

- **`parsort` now finds its splitters by histogram refinement** (histogram sort
  with sampling) instead of four weighted local quantiles per rank. Rank `r` of
  `P` receives exactly the records at global positions `⌊r N / P⌋` up to
  `⌊(r + 1) N / P⌋`, whatever the input distribution. Before, inputs whose local
  distributions looked alike (every rank holding a random share of one data
  set) sent a quarter of all records to each of four ranks (the ranks near
  `P/4`, `P/2`, `3P/4` and `P`) and almost none to the others.
- Each refinement round gathers about `OVERSAMPLING` random candidates per
  unresolved splitter to every rank and all-reduces their global positions;
  brackets of at most `4 * OVERSAMPLING` records are resolved exactly. The number
  of rounds grows like `log(N / P) / log(OVERSAMPLING)`.
- `OVERSAMPLING` changes meaning and value: it is now the expected number of
  candidates per unresolved splitter per round (16), not the number of local
  representatives per rank (4). The `rng` argument, previously unused, draws the
  candidates; it changes the communication, never the result.
- The signature of `parsort`, its stable tie order (input rank, input position)
  and its support for empty ranks and globally empty input are unchanged.

### Burn 0.22 — migration required

Burn 0.22 removes the backend type parameter from `Tensor` (now
`Tensor<D, K>`) and selects the backend at runtime through a `Device`. The
`burn` feature now uses Burn 0.22, and `rlst::interface::burn` follows it.

- **`RlstArrayIntoBurnTensor<Item, B, NDIM, K>` is now
  `RlstArrayIntoBurnTensor<Item, NDIM, K>`.** Replace
  `array.into_tensor::<Backend>()` by a type annotation on the result, for
  example `let t: Tensor<2> = array.into_tensor()?;` or
  `let t: Tensor<1, Int> = array.into_tensor()?;`.
- The new `into_tensor_on(&device)` creates the tensor on a given Burn
  `Device`. `into_tensor()` uses `Device::default()`, the first enabled Burn
  backend unless `BURN_DEVICE` names one, and panics if the application
  enables no Burn backend feature (for example `flex` or `wgpu`).
- The tensor takes the dtype of the array's scalar instead of requiring it to
  equal the backend's default float or integer element type: an `f64` array
  becomes an `F64` tensor on any device that supports `F64`. An integer scalar
  for a float tensor (or the reverse), or a dtype the device does not support,
  is an error.
- `TensorIntoRlstArray` is now implemented for integer tensors as well as
  float tensors, and `into_array` no longer requires `Item: RlstScalar`, so
  integer tensors convert to integer arrays. A failure to read the tensor from
  its device is returned as an error instead of panicking.

### Dependencies, documentation, and build improvements

- Checked all workspace dependencies against their latest releases. Burn moves
  from 0.21 to 0.22 (see above); every other requirement already targets the
  newest release series.
- `rlst-proc-macro` and `rlst-suitesparse` have no changes beyond the version
  number and package metadata.
- The optional integrations guide describes the Burn 0.22 conversions.
- The repository and homepage links of all crates and the documentation now
  point to `https://codeberg.org/rlst`, replacing the old `linalg-rs`
  organization.
- The distributed computations guide describes the exact output balance of
  `parsort`, and the bounded MPI test runner gained a `sort-balance` case that
  checks it on 1, 2, and 4 ranks.

## v0.9.0

Changes since [v0.8.0](https://github.com/tbetcke/rlst/releases/tag/v0.8.0).
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.9.0.

### BLAS thread control — migration required

- **`set_blas_threads` now returns
  `Result<Vec<BlasThreadSetting>, BlasThreadingError>`** instead of `()`. On
  success, it reports the state of every detected backend, read back after
  the change: a `BlasBackend` with `BlasThreads::Count(n)`, or for Accelerate
  `SingleThreaded`/`BackendManaged`. A zero or out-of-range count, a missing
  backend, and a backend rejecting the setting are errors rather than silent
  no-ops. Callers that ignored the result can write
  `let _ = set_blas_threads(n);`, or use `?` or `expect` to require that a
  backend was configured.
- The setter applies to every detected backend, and the new
  `get_blas_threads` reports their current states. Both live in
  `rlst::threading`.
- Backends are found at runtime on Linux and macOS without any feature:
  dynamically linked OpenBLAS (including ILP64 `64_` builds), Intel MKL, BLIS,
  FlexiBLAS, and Apple Accelerate on macOS 15 or later. Accelerate only
  distinguishes single-threaded from multi-threaded execution, and its
  setting applies to the calling thread only.
- The `openblas_threading`, `mkl_threading`, and `blis_threading` features are
  now only needed for statically linked libraries, or on targets other than
  Linux and macOS. They link the backend's functions directly and can be
  combined.
- Fixed the previous implementation. No threading feature compiled under
  edition 2024, `openblas_threading` called a function declared only under
  `mkl_threading`, BLIS received a 32-bit count where its default `dim_t` is
  64-bit, and MKL used the Fortran-style `mkl_set_num_threads` entry point
  instead of the C interface `MKL_Set_Num_Threads`.
- The documentation of `get_num_cores` now states that it counts logical
  CPUs; its behavior is unchanged.

### Dependencies, documentation, and build improvements

- Checked all workspace dependencies against their latest releases. Every
  requirement already targets the newest release series, so there are no
  dependency changes.
- `rlst-proc-macro` and `rlst-suitesparse` have no changes beyond the version
  number.
- The getting started guide gained a section on controlling BLAS threads.
- CI also runs Clippy with the `openblas_threading`, `mkl_threading`, and
  `blis_threading` features, which need no native BLAS library to check.

## v0.8.0

Changes since [v0.7.0](https://github.com/tbetcke/rlst/releases/tag/v0.7.0).
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.8.0.

### Frames of elements — migration required

- The previously unimplemented `Frame` trait has been replaced by `FrameImpl<'a>`.
  `ElementFrame<'a, Implementation>` is the public wrapper and forwards storage,
  application, and QR operations to its implementation. `GenericFrameImpl` stores
  owned elements; `ArrayFrameImpl` uses a column-major matrix. Both support growth,
  linear combinations, and rank-adaptive pivoted QR. The former reference-based
  `get`, `get_mut`, and iterator requirements have been replaced by owned
  `element` extraction and `set_element`; array-backed wrappers provide zero-copy
  matrix and column views. `push` borrows and copies an element and returns a result.
- The initial development types `ElementFrame<'a, Space>` and `ArrayFrame<'a, Item>`
  are now `GenericFrameImpl<'a, Space>` and `ArrayFrameImpl<'a, Item>`. Explicit
  construction uses `ElementFrame::new(GenericFrameImpl::new(&space))` or
  `ElementFrame::new(ArrayFrameImpl::new(&space))`. A frame is built entirely
  outside the space this way; spaces carry no frame-related trait or method, and a
  custom space needs no code of its own to be used with one. `imp`, `imp_mut`, and
  `into_imp` provide implementation access.
- Implementations opt into `FrameApply` and `FrameQr` for application and QR.
  `FrameApply::adjoint_apply` returns the inner products of an element with every
  column, `V^H x`; for orthonormal columns such as a Q factor or a Krylov basis,
  `adjoint_apply` followed by `apply_into` projects an element onto their span.
  `ArrayFrameImpl` computes it with one dense matrix-vector product. The wrapper
  provides inherent methods and implements the corresponding traits by
  delegation; its QR result contains a wrapped Q with the same implementation
  type. See the abstract linear algebra guide for construction and QR examples.
- Added `FrameRefAccess`, a frame capability that borrows stored columns instead
  of copying them: `element_ref`, `element_mut`, `iter`, and `iter_mut`, on both
  the backend and the `ElementFrame` wrapper. `GenericFrameImpl` implements it.
  `ArrayFrameImpl` stores its columns packed in one matrix and has no stored
  element to borrow, so it does not implement the trait and continues to offer
  `column`, `column_mut`, `matrix`, and `matrix_mut`. The mutating accessors
  expose the stored element directly, so callers must preserve the space's
  invariants, as for `Element::new` and `Element::imp_mut`.
- Added `LinearSpace::axpy_inplace`, computing `y += alpha * x`. It has a default
  implementation in terms of `scalar_mul` and `sum_inplace`, so existing spaces
  need no changes; implementations backed by arrays may override it with a fused,
  allocation-free update, as `ArrayVectorSpace` and
  `DistributedArrayVectorSpace` now do.
- Generic frame application no longer copies columns or forms temporaries.
  `GenericFrameImpl::apply`, `adjoint_apply`, and `apply_into` borrow the stored
  columns, and the `FrameApply::apply_into` default folds each column into the
  output through `axpy_inplace` without an intermediate element, skipping the
  scaling pass for a unit `beta`. The method's scalar bound is now
  `Zero + One + PartialEq + Copy` instead of `Zero`; every RLST scalar satisfies
  it. Results, error conditions, and the zero-`beta` contract are unchanged;
  array-backed frames already used BLAS and are unaffected.

### GMRES for abstract operators

- Added `GmresIteration` in `operator::algorithms::gmres`, a generalized minimal
  residual solver for operator equations whose operator need not be self-adjoint.
  It follows the same builder shape as `CgIteration`, and `run` reports the
  residual by the same rules: the relative residual `||b-Ax||/||b||`, or the
  absolute residual for a zero right-hand side. The residual is recomputed from
  the operator at the end of every restart cycle, so the returned value is the
  true residual of `x` rather than the estimate from the Givens recursion.
- The Krylov basis is stored in an `ElementFrame` and orthogonalized with two
  passes of classical Gram-Schmidt through the frame's `adjoint_apply` and
  `apply_into`, so no basis vector is copied; the same routine backs the generic
  pivoted QR. By default the frame is a `GenericFrameImpl` holding one element
  per basis vector, so GMRES requires nothing of a space beyond
  `InnerProductSpace` and no LAPACK linkage. `set_storage` replaces it with any
  other frame borrowing the right-hand side's space and panics for a frame over a
  different space instance; passing
  `ElementFrame::new(ArrayFrameImpl::new(rhs.space()))` keeps the basis as one
  dense matrix and runs the orthogonalization and the solution update as BLAS
  matrix-vector products. A supplied frame is scratch space and is cleared at the
  start of every restart cycle. `set_restart` bounds the size of that basis; by
  default GMRES does not restart.
- Breakdown is detected relative to the norm of the new Hessenberg column with a
  threshold of `100 * epsilon`, not by an exact comparison with zero, so a
  numerically exhausted Krylov space ends the cycle instead of normalizing
  rounding noise into basis vectors. A cycle that does not reduce the true
  residual, or produces a non-finite one, is undone and ends the iteration, so
  `x` is never worse than the initial guess and the returned residual always
  belongs to `x`.
- Two callbacks are available: `set_callable` receives the iteration count and
  the current residual from the Givens recursion at no extra cost, while
  `set_element_callable` reconstructs the approximation and the residual element
  for every call at the cost of one extra linear combination and one extra
  operator application per iteration.

### Iterative solvers — migration required

- `CgIteration` and `GmresIteration` now take the initial guess as
  `x: &'a mut Element<'b, Space>` and carry a second lifetime parameter. The
  mutable borrow of `x` previously lasted as long as the element's borrow of its
  space, so the solution could only be observed through a callback; it now ends
  when `run` returns and `x` can be read directly. Calls such as
  `CgIteration::new(&op, &rhs, &mut x)` are unchanged. Code that names the type
  explicitly must add the lifetime, for example `CgIteration<'a, 'b, f64, S, Op>`
  in place of `CgIteration<'a, f64, S, Op>`.
- Conjugate gradients updates the approximation through
  `LinearSpace::axpy_inplace` and no longer allocates a scaled search direction
  per step.

### Ghost communicator — migration required

- `GhostCommunicator::new(&ghost_indices, &owning_ranks, comm)` has been replaced
  by `GhostCommunicatorBuilder`. Ghosts are added one at a time with `add_ghost`
  or as `(index, owner)` pairs with `add_ghosts`, and `build(comm)` creates the
  communicator collectively. The builder assigns no positions: ghost values
  arrive in the order of `receive_indices()`, grouped by ascending owning rank
  and in insertion order within one owner, as before. Replace
  `GhostCommunicator::new(&indices, &owners, comm)` with
  `GhostCommunicatorBuilder::new().add_ghosts(indices.iter().copied().zip(owners.iter().copied())).build(comm)`.
- Chunk sizes are now fixed when the communicator is built, with
  `.chunk_sizes(ChunkSizes::Uniform(n))` (the default is `Uniform(1)`) or
  `.chunk_sizes(ChunkSizes::PerIndex(Box::new(|index| ...)))`, with which each
  owner gives the number of values for each of its indices. The size of a chunk
  may vary per index and may be zero. `forward_send_values_by_chunks` and
  `backward_send_values_by_chunks` have been removed; build a communicator with
  `ChunkSizes::Uniform(chunk_size)` instead. `send_offsets()` and
  `receive_offsets()` locate the chunk of each index, and `send_chunk_sizes()`
  and `receive_chunk_sizes()` return the chunk sizes.
- Added `GhostSendBuffer` and `GhostReceiveBuffer`, value buffers created by
  the communicator with `send_buffer`, `send_buffer_from_fn`, or
  `receive_buffer`. `GhostSendBuffer::forward` returns the received ghost values
  and `GhostReceiveBuffer::backward` returns the values sent back to the owners.
  Both give access to each chunk with `chunk`, `chunks`, and their mutable
  variants. The buffers always have the communicator's layout, so these
  exchanges cannot fail; they are the recommended way to exchange values.
- Value exchanges no longer communicate across the whole communicator. The
  slice-based `forward_send_values` and `backward_send_values` remain for
  writing into existing storage, now require `T: Copy + Default`, and expect
  slices of `send_buffer_len()` and `receive_buffer_len()` values. They check
  the lengths locally; on a mismatch the rank still takes part in the exchange
  with default values, so its neighbours do not block but receive default
  values from it, and then panics. Invalid owners, a zero uniform chunk size,
  uniform chunk sizes that disagree between an owner and a receiver, and value
  counts beyond MPI's `i32` limit are rejected collectively by `build`.
- All fields of `GhostCommunicator` are now private. Use the accessors, for
  example `receive_indices()` in place of `receive_indices` and
  `total_receive_count()` in place of `total_receive_count`. The send and
  receive displacements are no longer exposed.
- `Global2LocalDataMapper::new` and `DataPermutation::new` take the chunk size
  as a new last argument, and `map_data`, `forward_permute`, and
  `backward_permute` no longer take one. The constructors reject a zero chunk
  size or chunk sizes that differ between ranks collectively. The exchanges no
  longer communicate globally; on wrong buffer lengths they take part in the
  exchange with default values and then panic on the offending rank.
  Create one mapper or permutation per chunk size, for example
  `DataPermutation::new(layout, &indices, 2)` followed by
  `permutation.forward_permute(&data, &mut permuted)`.

### Dependencies, documentation, and build improvements

- Checked all workspace dependencies against their latest releases. Every
  requirement already targets the newest release series except the `cc` build
  dependency, which now requires 1.5. There are no other version changes; in
  particular `rand` stays at 0.10, `syn` at 3, and Burn integration at 0.21.
- The frame types `ElementFrame`, `GenericFrameImpl`, `ArrayFrameImpl`,
  `FrameQr`, `FrameQrDecomposition`, and `FrameQrOptions` are re-exported from
  the crate root. `distributed_tools` re-exports `GhostCommunicatorBuilder`,
  `ChunkSizes`, `GhostSendBuffer`, and `GhostReceiveBuffer`.
- The abstract linear algebra guide gained sections on frames, borrowing frame
  columns, rank-adaptive pivoted QR, and GMRES; the distributed computations
  guide and the `ghost_communicator` example now use the builder and exchange
  buffers.
- The separate documentation changelog has been merged into this file, which
  is now rendered as the crate's changelog documentation page.
- The bounded MPI regression runner covers variable ghost chunk sizes, buffer
  length mismatches, and edge-case ghost topologies.

## v0.7.0

Changes since [v0.6.1](https://github.com/tbetcke/rlst/releases/tag/v0.6.1),
ordered from major features and compatibility changes to smaller fixes.
All workspace crates (`rlst`, `rlst-proc-macro`, and `rlst-suitesparse`) use version 0.7.0.

### Expanded dense linear algebra

- Added generalized eigenvalue decompositions for real and complex matrices,
  including symmetric/Hermitian-definite problems and selection of left, right,
  both, or no eigenvectors for general problems.
- Added generalized Schur (QZ) decompositions, with Schur vectors and homogeneous
  eigenvalue pairs that retain information about infinite eigenvalues.
- Added pivoted symmetric-indefinite LDL factorization and linear-system solves,
  including complex symmetric matrices (using transpose, not Hermitian, symmetry).
- Added LQ factorization, application of its orthogonal/unitary factor, and
  minimum-norm solves for full-row-rank underdetermined systems.
- Added triangular/quasi-triangular Sylvester equation solves, exposing LAPACK's
  scaling and coefficient-perturbation status.
- These new LAPACK-backed facilities support `f32`, `f64`, `c32`, and `c64`.

### SuiteSparse sparse direct solvers

- Added the `rlst-suitesparse` companion crate with UMFPACK factorization of
  `CsrMatrix<f64>` and `CsrMatrix<c64>`, including CSR-to-CSC conversion, symbolic
  analysis, reusable numeric factors, and real/complex solves.
- Right-hand sides and solution buffers use RLST arrays; factorization and solve
  APIs report errors and singular-factorization status explicitly.
- Added native SuiteSparse discovery for Homebrew and Linux installations and a
  `SUITESPARSE_PATH` override. UMFPACK is GPL-2.0-or-later; see the companion
  crate's README for licensing implications.

### Dense multiplication and scalar arrays — migration required

- **Use `dot_blas!` in place of the old dense `dot!` for BLAS-backed matrix
  products**, including chains of matrices. It remains the recommended choice
  for general dense matrix multiplication.
- `dot!` now performs direct dense tensor contractions of the left operand's
  last axis with the right operand's first axis. It supports arbitrary positive
  ranks, strided views, lazy expressions, and scalar types without BLAS support.
  Dense output rank must be inferable from context, for example a
  `DynArray<_, R>` annotation. Complex products do not conjugate inputs.
- Added `dot_into!` to overwrite existing output storage without allocation.
  Contraction metadata is validated before writes, and empty contracted axes
  produce zeros.
- Added `ZeroDimArray` for scalar results with shape `[]`, including `scalar`,
  `scalar_mut`, and `into_scalar` accessors. Vector-vector direct contractions
  return this type; a length-one vector is not a scalar output buffer.
- `rlst_dynamic_array!(T, [])` creates scalar storage when rank zero is known at
  the call site. Ordinary and strided array storage require positive rank;
  aligned scalar construction and unresolved generic rank-zero construction
  are rejected.
- Sparse matrices and abstract operators retain their `dot!` dispatch and
  right-to-left product chains. Distributed CSR matrices now support collective
  `dot`/`dot!` products with distributed vectors and multiple right-hand sides.

### Safety and API contracts — migration required

- **MPI is now opt-in:** enable the `mpi` feature to use distributed APIs.
  The default feature set is empty.
- Mutable strided layouts now reject overlapping elements and insufficient
  backing storage. Axis permutations and multislices validate axes and indices
  before constructing views, including empty-array cases.
- Added explicit disjoint-access contracts (`UnsafeDisjointRandom1DAccessMut`
  and `DisjointRefDataContainerMut`) for mutable iteration and operations that
  keep multiple element references alive. Custom array/container implementations
  must satisfy the stronger bounds where required.
- FFTW plans now require stable storage (`StableRawAccessMut` and the related
  container contract). `Fft::borrow_mut` returns `FftMutGuard`, permitting element
  updates through `data_mut` while preventing replacement of the planned array.
- Raw CSR construction now validates and canonicalizes its input: columns are
  sorted, duplicates summed, and explicit zeros removed. `CsrMatrix::new` has
  corresponding scalar bounds, and empty matrix shapes are supported.
- Sparse binary operations require equal shapes. Ordinary sparse numerical
  traits for non-zero-preserving functions (such as `exp`, `ln`, `cos`, and
  `recip`) and sparse matrix/matrix division have been removed. Use `map_stored`
  explicitly when transforming stored entries only is intended; it skips
  implicit zeros and removes zero-valued results.
- Custom `MmIdentifier` implementations must now provide `MM_NFIELDS` and
  `from_mm_fields` in addition to `MMTYPE`.
- Random-number APIs now use `rand` 0.10. Downstream RNGs must use compatible
  versions; import `rand::RngExt` for `random`/`random_range` and use `rand::rng()`
  in place of `thread_rng()`.

### Correctness and robustness fixes

- Hardened distributed layouts, sorting, mapping, ghost communication, gathers,
  scatters, and sparse operations for empty local ranks and globally empty
  inputs. Added collective metadata checks and preserved rank participation
  when rejecting incompatible inputs.
- Matrix Market readers now validate headers, field types, record counts,
  dimensions, and coordinate bounds; malformed, truncated, and excess data
  produce errors. Fixed real/complex parsing and rectangular column-major array
  input handling.
- Conjugate gradients now returns immediately for an exact initial solution,
  uses an absolute residual for a zero right-hand side, and avoids invalid
  updates on zero or non-finite search-direction denominators.
- Fixed pseudo-inverse handling for zero retained rank and all-zero matrices,
  with regression coverage for rank deficiency and truncation thresholds.
- Sparse application with multiple right-hand sides validates all dimensions
  before modifying the output. Stored-entry filtering no longer recurses over
  long runs of removed entries.
- Repaired `rlst_static_type!` expansion and added external-crate coverage.
- Added LAPACK storage/shape validation and numerical edge-case coverage,
  including generalized eigenvector ordering, padded views, empty inputs,
  Schur structure, and Sylvester scaling.

### Dependencies, documentation, and build improvements

- Updated workspace dependencies to their latest stable release series,
  including `rand`/`rand_chacha` 0.10, `rand_distr` 0.6, `itertools` 0.15,
  `syn` 3, `darling` 0.24, `bindgen` 0.73, `git2` 0.21, and `cc` 1.4.
  Updated `bytemuck`, `mpi`, and `mpi-sys` requirements and refreshed compatible
  dependency resolution. Burn integration targets 0.21.
- Fixed system FFTW discovery through `pkg-config` and native macOS Homebrew
  fallbacks, requiring both single- and double-precision libraries and avoiding
  host Homebrew discovery during cross-compilation.
- Refreshed user guides and API documentation for dense/sparse operations,
  decompositions, distributed computation, interpolation, and optional
  integrations. Added this changelog to the crate documentation.
- Updated random-number examples and fixed zero-dimensional shape assertions
  when optional Burn dependencies introduce additional equality implementations.
- Expanded CI with SuiteSparse tests and bounded MPI regressions on 1, 2, and
  4 ranks, including timeout/process-group cleanup tests; refreshed CI runners
  and lint/format compatibility.

## v0.6.1

- Update burn dependency to 0.21

## v0.6.0

- Added an interface to convert Burn tensors into rlst arrays and back

## v0.5.0

- Added the `multislice` and `multislice_iter` method to slice over multiple axes
  and iterate over multiple axes at the same time.
- Added an `interpolation` module for barycentric interpolation in 1, 2, and
  3 dimensions.
- Added a `chebychev` module for the evaluation, differentiation and conversion
  to interpolation data of Chebychev series coefficients in 1, 2, and 3 dimensions.
- Improved the `eval` routines. They now work in most situations where generic
  arrays are provided.
- Small bug fixes, some CI improvements, etc.

