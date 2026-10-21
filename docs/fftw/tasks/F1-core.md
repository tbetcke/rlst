# F1: Core of the new FFTW interface

| | |
| --- | --- |
| Task | F1 (design, Section 9) |
| Status | done |
| Depends on | F0 |
| Pull requests | three, into `fftw-0.11`: F1a (API skeleton), F1b (build, features, lock, base types), F1c (layouts, c2c plans, execution, fingerprints) |
| Read first | design: Section 3 (all items), Sections 4.1–4.8, 4.10–4.12 and 4.15, Section 5, Sections 7.1–7.4, Section 9 (F1 row, Transition), Section 10, Appendix B; `docs/fftw/F0-findings.md`; `CLAUDE.md` |

## Goal

The core of the new interface: the feature rules, the FFTW lock, the base types, raw
layouts, complex-to-complex plans through the 64-bit guru interface with the three input
policies, checked execution on slices, and fingerprints. Everything is tested against an
independent reference. Later tasks build on it: layouts from RLST arrays (F2), split
storage (F3), the concurrency tests and the separate-process determinism binary (F4), the
other families (F5), wisdom and threads (F6).

## Scope

In scope:

- `rlst/Cargo.toml` (features; the `libc` dependency, in F1c), `rlst/build.rs`;
- `rlst/src/lib.rs` (`FFTW_ALIGNED`, the feature check);
- the new modules under `rlst/src/fftw/` and their declarations in `rlst/src/fftw.rs`;
- in the 0.10 module (`rlst/src/fftw.rs`, `rlst/src/fftw/c2c_inplace.rs`,
  `rlst/src/fftw/r2r_inplace.rs`): only the move onto the new lock, and the sentences that
  name removed features;
- `README.md`, `rlst/src/doc/optional_integrations.rs`: only the sentences that name removed
  features;
- `rlst/CHANGELOG.md`, `CLAUDE.md`, the status table of `docs/fftw/README.md`.

Out of scope, for later tasks:

- layouts built from RLST arrays and views, and execution on them (F2), apart from their
  signatures in F1a;
- split storage (F3), r2c, c2r and r2r (F5), `Scale` (F5), wisdom and threads (F6);
- concurrency tests and every test that depends on the wisdom present in the process (F4);
- removing the 0.10 module, the Chebyshev port, the licence warning, the module guide
  (F7).

## Module layout

New files, following design Section 4.1 (`lock` and `buffer` are split out of `planner` and
`execute` for clarity):

| File | Contents |
| --- | --- |
| `rlst/src/fftw/real.rs` | `FftwReal`, sealed; precision dispatch to the `fftw_` and `fftwf_` functions |
| `rlst/src/fftw/kind.rs` | `Direction` (r2r kinds follow in F5) |
| `rlst/src/fftw/options.rs` | `Rigor`, the input-policy request types, planner options |
| `rlst/src/fftw/error.rs` | `FftwError`, `Side` |
| `rlst/src/fftw/align.rs` | `PLAN_ALIGNMENT`, `AlignmentClass` |
| `rlst/src/fftw/lock.rs` | the FFTW lock and the settings guard (crate-private) |
| `rlst/src/fftw/backend.rs` | `Backend`, `version()` |
| `rlst/src/fftw/layout.rs` | `Dim`, `Placement`, `Layout` |
| `rlst/src/fftw/buffer.rs` | `FftBuffer`, `FftBufferMut`, sealed, implemented for slices |
| `rlst/src/fftw/plan.rs` | `Plan<T, F, P>`, family and policy markers |
| `rlst/src/fftw/planner.rs` | `Planner<T, S>`, scratch, the FFI planning call |
| `rlst/src/fftw/execute.rs` | checked execution |
| `rlst/src/fftw/fingerprint.rs` | `Plan::description`, `Plan::fingerprint` |
| `rlst/src/fftw/test_support.rs` | `#[cfg(test)]`: the reference of design Section 7.1 and test helpers |

The modules are private; `rlst/src/fftw.rs` re-exports their public items by name (Clippy
forbids wildcard imports), beside the 0.10 items, which keep their names and paths until F7.
Unit tests go in each module under `#[cfg(test)]`, as `CLAUDE.md` asks.

## F1a: API skeleton (draft pull request, reviewed before F1b and F1c)

Write the public signatures and their rustdoc, with bodies `todo!()`, for every public item
of F1, and the signatures of F2's array-facing items so that their generic bounds are
reviewed now:

- F1: everything in the module table above;
- F2: `Layout::along_axis`, `Layout::over_axes`, `Layout::along_axis_inplace`,
  `Layout::over_axes_inplace` (design Section 4.3), generic over RLST arrays and views
  (`Array<A, N>` with `A: Shape<N> + Stride<N>` and raw access, `N` shared by both arrays),
  and the `FftBuffer` and `FftBufferMut` implementations for arrays.

The review checks that the bounds fit RLST's traits (`rlst/src/traits/`), that the input
policies read naturally at a call site, and that the compile-fail cases of this brief fail
for the intended reason. The pull request is merged into `fftw-0.11` after review; F1b, F1c
and F2 replace the `todo!()` bodies. No test may call a `todo!()` body.

## F1b: build, features, lock and base types

### 1. Features and build

- `rlst/Cargo.toml`:

  ```toml
  fftw = ["dep:fftw-sys"]
  fftw_system = ["fftw", "fftw-sys/system", "dep:pkg-config"]
  ```

  Remove `fftw_source` and `fftw_mkl` (design Section 9, Transition). Keep `fftw-sys` at
  `0.8` with `default-features = false`.
- A compile error when `fftw` is enabled without `fftw_system`, in `rlst/src/lib.rs`:

  ```rust
  #[cfg(all(feature = "fftw", not(feature = "fftw_system")))]
  compile_error!("the `fftw` feature needs a back end: enable `fftw_system`");
  ```

- `rlst/build.rs`, for `fftw_system` (design Section 4.10):
  - when the environment variable `DOCS_RS` is set, return before any discovery;
  - read the version of `fftw3` and `fftw3f` from the `pkg_config::Library` that the probe
    returns, and fail below 3.3.4 with a message that names the version found, the minimum
    and `PKG_CONFIG_PATH`;
  - keep the macOS Homebrew fallback unchanged (it reports no version; the run-time check
    of work item 4 covers it);
  - emit `cargo:rerun-if-env-changed` for `DOCS_RS` and `PKG_CONFIG_PATH`;
  - do not look for the threads libraries yet (F6).
- Remove the names of the dropped features from the sentence in `README.md` that notes
  FFTW's GPL licence, keeping the notice itself (F7 replaces it); from the back-end sentence
  of `rlst/src/doc/optional_integrations.rs`; and from the module documentation of
  `rlst/src/fftw.rs`.
- Update the `fftw` entry of "Features and native dependencies" and the licensing sentence
  in `CLAUDE.md`.

### 2. Precision dispatch (`real.rs`)

- `pub trait FftwReal: RlstScalar<Real = Self> + sealed::Sealed`, implemented for `f32` and
  `f64`.
- A crate-private trait carries the FFI functions per precision: the plan pointer type,
  `plan_guru64_dft`, `execute_dft`, `destroy_plan`, `sprint_plan`, `set_timelimit`.
  Implement it for both precisions with one macro over the `fftw_` and `fftwf_` names
  (`paste` is already a dependency).
- RLST's `c64` and `c32` are `num::complex::Complex64` and `Complex32`, and fftw-sys defines
  `fftw_complex` and `fftwf_complex` as the same types. Assert this at compile time with a
  function that returns its `c64` argument as `fftw_sys::fftw_complex` (and likewise for
  `c32`).

### 3. Errors (`error.rs`)

`#[non_exhaustive] pub enum FftwError`, with `thiserror` (already a dependency), and the
variants of design Section 4.15: `InvalidLayout`, `LayoutMismatch`, `AlignmentMismatch`,
`Overlap`, `PlacementMismatch`, `InvalidKind`, `PreserveInputUnsupported`,
`ScratchTooLarge`, `VersionTooOld`, `PlanCreationFailed`, `WisdomImportFailed`. Each variant
carries the values involved, and each message names the expected and found values and the
remedy where there is one:

- `AlignmentMismatch` points to `rlst_dynamic_array!(…, shape | rlst::FFTW_ALIGNED)` and
  `AlignedDynArray`;
- `ScratchTooLarge` names the bytes needed and the bound, and points to `max_scratch` and
  the `*_on` planner methods;
- `PlanCreationFailed` carries the rigor, the FFTW flags and a description of the layout,
  and says that FFTW gives no reason.

`pub enum Side { Input, Output }` identifies the side in `LayoutMismatch` and
`AlignmentMismatch`.

### 4. The FFTW lock (`lock.rs`) and the 0.10 module

- One process-wide `static` `Mutex` guarding all FFTW state. A crate-private
  `with_fftw(|state| …)` runs a closure under it and recovers a poisoned lock with
  `PoisonError::into_inner` (design Section 4.7).
- The first acquisition checks the run-time version (work item 5) and records the result;
  planning returns `VersionTooOld` if it is below 3.3.4.
- A settings guard applies a plan's time limit with `fftw_set_timelimit` or
  `fftwf_set_timelimit` before planning, and resets it to `FFTW_NO_TIMELIMIT` (-1.0) in
  `Drop`, so that it is reset during unwinding as well. F6 adds the thread count to it.
- Move the 0.10 module onto this lock: `create_c2c` and `create_r2r` in `c2c_inplace.rs` and
  `r2r_inplace.rs`, and plan destruction in `Drop for FftwPlanPtr` in `fftw.rs`, run inside
  `with_fftw` instead of locking `FFTW_PLAN_INTERFACE`. Remove the old mutex. Nothing else of
  the 0.10 module changes, and its tests and doctests must pass unchanged.

### 5. Version and back end (`backend.rs`)

- Declare FFTW's version strings yourself, since fftw-sys does not bind them. `rlst` uses
  the 2024 edition, which requires `unsafe extern` blocks:

  ```rust
  unsafe extern "C" {
      static fftw_version: [c_char; 0];
      static fftwf_version: [c_char; 0];
  }
  ```

  Read them with `CStr::from_ptr`; the `SAFETY:` comment states that FFTW defines them as
  NUL-terminated constant arrays.
- `pub fn version() -> &'static str` returns the double-precision string, for example
  `fftw-3.3.10-sse2-avx`. A crate-private parser extracts (major, minor, patch) and whether
  the build has SIMD: a suffix other than `-fma` (design Section 3 item 6). Later tests
  that depend on SIMD skip on builds without it (design Section 7.7).
- `#[non_exhaustive] pub enum Backend { System }` and `pub fn backend() -> Backend`.

### 6. Base types

- `kind.rs`: `pub enum Direction { Forward, Backward }`, mapped to FFTW's signs. fftw-sys
  gives `FFTW_FORWARD` as `i32` (-1) and `FFTW_BACKWARD` as `u32` (1); cast explicitly.
- `options.rs`: `pub enum Rigor { Estimate, Measure, Patient, Exhaustive }`, default
  `Estimate`, mapped to `FFTW_ESTIMATE`, `FFTW_MEASURE`, `FFTW_PATIENT`, `FFTW_EXHAUSTIVE`.
  The policy request types of the planner's type state (work item 9): a default request, a
  preserve request and a destroy request.
- `align.rs`: `pub const PLAN_ALIGNMENT: usize = crate::FFTW_ALIGNED;` and
  `pub struct AlignmentClass(u8)`, the address modulo `PLAN_ALIGNMENT`, with constructors
  from a pointer and from a slice, `Default` (class 0), `Display` and an accessor
  (design Section 4.6). Use `pointer::addr()`.
- `rlst/src/lib.rs`: `FFTW_ALIGNED` becomes 64, with a doc comment that states the
  alignment class rule of design Section 4.6 instead of "FFTW requires 16 bytes
  alignment". Update the comments in the 0.10 doctests that call it 16-byte alignment.

### 7. Tests of F1b

- the lock recovers after a panic inside `with_fftw` (`std::panic::catch_unwind`), and
  planning with the 0.10 module still works afterwards;
- version parsing of `fftw-3.3.10-sse2-avx`, `fftw-3.3.8` and a string with an unknown
  suffix; SIMD detection for `fftw-3.3.10-sse2-avx` and `fftw-3.3.10-neon` (SIMD) and for
  `fftw-3.3.11` and `fftw-3.3.10-fma` (none); `version()` at run time starts with `fftw-3.`;
- `AlignmentClass` of pointers at offsets 0, 8, 16 and 64 bytes from a 64-byte-aligned
  allocation;
- `PLAN_ALIGNMENT == 64`, as a compile-time assertion;
- all existing FFTW and Chebyshev tests and doctests pass.

## F1c: layouts, c2c plans, execution and fingerprints

### 8. Raw layouts (`layout.rs`)

- `pub struct Dim { pub n: isize, pub is: isize, pub os: isize }`, in elements of the input
  and output (complex elements for c2c). `pub enum Placement { OutOfPlace, InPlace }`.
- `Layout::new(dims: &[Dim], batch: &[Dim], placement: Placement) -> Result<Layout, FftwError>`
  validates, in this order, returning `InvalidLayout` with the dimension and the reason:
  1. at least one transform dimension;
  2. every extent at least 1;
  3. every stride non-negative (design Section 4.3, raw layouts);
  4. in place, input and output strides equal in every dimension;
  5. distinct output elements, by the rule of design Section 4.3: with the dimensions of
     extent above 1 (transform and batch together) sorted by output stride, every stride is
     positive and at least the extent times the stride of the next smaller one;
  6. spans computed with checked arithmetic; an overflow is `InvalidLayout`.
- Normalisation (design Section 4.3): drop batch dimensions of extent 1; sort the batch
  dimensions by input stride, then output stride, largest first; merge each adjacent pair
  whose outer strides equal the inner extent times the inner strides, on both sides.
  Transform dimensions keep the order given. The normalised form is what plans store and
  what execution compares; it derives `PartialEq` and `Debug`.
- Accessors: the transform and batch dimensions, the placement, `input_span()` and
  `output_span()` (1 plus the sum of (n − 1) · stride over all dimensions), and
  `logical_size()` (the product of the transform extents).
- Unit tests: merging (a contiguous batch over two dimensions becomes one), no merge for a
  transposed batch, dropping extent 1, independence of the order of the batch list, each
  validation error, spans.

### 9. Plans and the planner (`plan.rs`, `planner.rs`)

- `pub struct Plan<T: FftwReal, F: Family, P: Policy>`: the raw plan as a `NonNull`, the
  normalised layout, the direction, the rigor and flags, the alignment classes of input and
  output (or unaligned), and `PhantomData`. Family marker `C2c`; policy markers
  `Preserving`, `Destroying`, `InPlace`; `Family` and `Policy` are sealed. `unsafe impl Send`
  and `Sync`, with a `SAFETY:` comment: FFTW's execute functions are thread-safe, a plan is
  not modified by execution, and destruction needs ownership and runs under the lock.
  `Drop` destroys the plan inside `with_fftw`.
- `Planner<T: FftwReal, S = DefaultRequest>`, a builder: `new()`, `rigor()`, `wisdom_only()`,
  `unaligned()`, `time_limit(seconds)`, `alignment(input, output)`, `max_scratch(bytes)`
  (default 1 GiB), `preserve_input()` and `may_destroy_input()`, which change `S`.
  A crate-private trait maps (request, family) to the plan's policy: for c2c the default and
  preserve requests give `Preserving`, the destroy request gives `Destroying`. F5 adds
  c2r, whose default maps to `Destroying` (DC1).
- Planner methods:

  | Method | Plan | Planning arrays |
  | --- | --- | --- |
  | `c2c(&layout, direction)` | `Plan<T, C2c, Preserving>` or `Destroying` | scratch |
  | `c2c_inplace(&layout, direction)` | `Plan<T, C2c, InPlace>` | scratch |
  | `c2c_on(&mut input, &mut output, &layout, direction)` | as `c2c` | the caller's buffers |
  | `c2c_inplace_on(&mut data, &layout, direction)` | as `c2c_inplace` | the caller's buffer |

  A layout of the wrong placement is `PlacementMismatch`.
- FFTW flags: the rigor; `FFTW_WISDOM_ONLY` if requested; `FFTW_UNALIGNED` if requested;
  `FFTW_PRESERVE_INPUT` for `Preserving` and `FFTW_DESTROY_INPUT` for `Destroying`, passed
  explicitly.
- **Scratch** (DC3, design Section 4.4): the bytes needed are the input span plus the output
  span, times the element size (one span in place). Above `max_scratch`, return
  `ScratchTooLarge` before allocating. Otherwise allocate zeroed reals with aligned-vec at
  `PLAN_ALIGNMENT`, as `rlst/src/dense/data_container.rs` does, long enough for the span
  plus the class offset, and plan on the pointer offset by the requested class in bytes. A
  class that is not a multiple of the complex type's alignment is `InvalidLayout`. Free the
  scratch after planning.
- **`*_on` planning**: the spans must fit the buffers (`LayoutMismatch` otherwise); the
  plan's alignment classes are those of the buffers given. With `Measure` and stronger
  rigor FFTW overwrites the buffers; the rustdoc says so.
- **The FFI call**, inside `with_fftw`: the version check's result; the settings guard;
  dimensions converted to `fftw_sys::fftw_iodim64` (fields `n`, `is`, `os`, all `isize`);
  `fftw_plan_guru64_dft` or `fftwf_plan_guru64_dft`. A NULL plan is `PlanCreationFailed`.

### 10. Buffers and execution (`buffer.rs`, `execute.rs`)

- Sealed `pub trait FftBuffer<E>` (`as_ptr`, `len`, and a crate-private check of a buffer
  against one side of a layout) and `pub trait FftBufferMut<E>: FftBuffer<E>`
  (`as_mut_ptr`), implemented for `[E]` here and for RLST arrays in F2. For a slice the
  check is that the layout's span fits.
- Execution (design Section 4.4 and 4.8):

  | Plan | Signature |
  | --- | --- |
  | `Plan<T, C2c, Preserving>` | `execute(&self, input: &I, output: &mut O)`, `I: FftBuffer + ?Sized`, `O: FftBufferMut + ?Sized` |
  | `Plan<T, C2c, Destroying>` | `execute(&self, input: &mut I, output: &mut O)`, both `FftBufferMut` |
  | `Plan<T, C2c, InPlace>` | `execute(&self, data: &mut D)`, `D: FftBufferMut` |

  All return `Result<(), FftwError>`.
- Checks, in this order, before any FFI call: the spans fit (`LayoutMismatch`, with the side,
  the span and the length); the alignment class of each pointer equals the plan's, unless
  the plan is unaligned (`AlignmentMismatch`); for out-of-place plans, the byte ranges of
  input and output are disjoint (`Overlap`).
- Then `fftw_execute_dft` or `fftwf_execute_dft`, without the lock. The `SAFETY:` comment
  goes through FFTW's four new-array conditions (design Section 3 item 1): sizes and strides
  are the plan's and the spans fit; placement matches the plan's type; split storage does not
  apply; the classes are equal modulo 64, hence FFTW's alignment is equal. For a `Preserving`
  plan it also states that FFTW was given `FFTW_PRESERVE_INPUT`, so the pointer derived from
  `&input` is never written through.

### 11. Fingerprints (`fingerprint.rs`)

- `Plan::description(&self) -> String`: `fftw_sprint_plan` under the lock, copied into a
  `String`, the C string freed with `libc::free`. A NULL return means FFTW could not
  allocate; treat it as an allocation failure.
- `Plan::fingerprint(&self) -> u64`: FNV-1a 64 of the description (offset basis
  `0xcbf29ce484222325`, prime `0x100000001b3`), written inline (design Section 4.12).
- Add `libc = { version = "0.2", optional = true }` to `rlst/Cargo.toml` and
  `"dep:libc"` to the `fftw` feature in this pull request, where it is first used (the
  `strict` feature denies unused crate dependencies).

### 12. Tests of F1c (design Sections 7.1–7.4, c2c only)

**Rules for every FFTW test here:** plan with `Rigor::Estimate` only, and do not depend on
the wisdom present in the process. FFTW's wisdom is global to the process and the tests run
as threads of one process (design Section 3 item 4); tests that need other rigor or a fresh
process belong to F4.

- **The reference** (`test_support.rs`, design Section 7.1): an O(n²) DFT in f64 with
  compensated (Neumaier) summation of real and imaginary parts, twiddles from the reduced
  index `(j · k) mod n` in integer arithmetic through `f64::sin_cos(2π r / n)`, applied along
  each transform dimension of a raw layout and batched over the others. f32 transforms are
  compared with the reference evaluated on their f32-rounded inputs.
- **Tolerance**: relative L2 error at most c · ε · (1 + log₂ N), with N the logical size and ε
  the epsilon of the precision tested. Fix c once for c2c (start at 4), state it in the
  test with a comment, and do not tune it per size.
- **Inputs** from a seeded `rand_chacha` generator with a fixed seed, uniform in [−1, 1];
  every assertion message names the seed and the cell.
- **The fixed cells**:
  - sizes 1, 2, 3, 17, 97, 1009, the powers of two up to 4096, and nd-fmm's grids 140, 216,
    336, 588 and 1080;
  - both directions and both precisions;
  - the three policies;
  - layouts: one contiguous transform; a contiguous batch of 3; a batch with stride 1 and
    transform stride equal to the batch count; a transform with stride 2; a rank-2
    transform; a batch of rank-2 transforms.

  To keep debug-mode CI fast, cells with n ≥ 1009 use a batch of 1 and the rank-1 layout.
  A further 32 cells are sampled from the product of sizes up to 1080 and the layouts, with
  the same fixed seed.
- **Identities**: the backward transform after the forward one is N times the input, to the
  tolerance. A `Preserving` plan leaves its input bitwise unchanged.
- **One plan, many buffers** (in process; F4 repeats it in fresh processes): one `Estimate`
  plan executed on four distinct buffers of the same class gives, bit for bit, the results
  of a plan made for each buffer.
- **Negative tests**, each with an output buffer filled with a sentinel value that must be
  unchanged afterwards:
  - a slice shorter than the span (`LayoutMismatch`);
  - a buffer one element off class 0 (`AlignmentMismatch`);
  - an in-place layout passed to `c2c` (`PlacementMismatch`);
  - overlapping input and output, built in the test from one allocation with
    `slice::from_raw_parts`, with a comment (`Overlap`);
  - a strided layout whose span exceeds a small `max_scratch` (`ScratchTooLarge`), and the
    same layout planned with `c2c_on`;
  - invalid layouts: extent 0, a negative stride, unequal strides in place, an output
    stride 0 with extent 2, two dimensions that address the same outputs (`InvalidLayout`).
- **`compile_fail` doctests** on `Plan`: executing a `Destroying` plan with `&input`;
  executing an `InPlace` plan with two buffers. Each has a passing twin that differs only in
  the misuse, so that the `compile_fail` cannot pass for an unrelated reason.
- Unit tests of FNV-1a: `""` → `0xcbf29ce484222325`, `"a"` → `0xaf63dc4c8601ec8c`, `"foobar"`
  → `0x85944171f73967e8`. `description()` of an `Estimate` plan is non-empty, and equal
  layouts planned twice in the same test have equal fingerprints.

### 13. Documentation, CHANGELOG and status

- Rustdoc with an example for every new public item (design goal 8). Examples use slices,
  `Rigor::Estimate` and sizes that run quickly.
- A short section "New interface (0.11, in progress)" in the module documentation of
  `rlst/src/fftw.rs` that lists the new types; the full guide is F7.
- `rlst/CHANGELOG.md`: a section `## v0.11.0 (unreleased)` with the removal of
  `fftw_source` and `fftw_mkl` (migration: use `fftw_system`; MKL is out of scope for now),
  `FFTW_ALIGNED` now 64, and a placeholder entry for the new FFTW interface, completed at the
  release (F8b).
- The status table of `docs/fftw/README.md`; `CLAUDE.md` in step with the features.

## Acceptance

F1a:

- [ ] The skeleton's signatures and rustdoc are reviewed, including the F2 array signatures
      and the generic bounds over RLST's traits; the review's outcome is recorded in the
      pull request.

F1b:

- [ ] `fftw_source` and `fftw_mkl` are gone; `cargo check --features fftw` fails with the
      message of work item 1; `cargo check --features fftw_system` succeeds.
- [ ] On Linux (where `build.rs` has no Homebrew fallback), with `DOCS_RS=1` and
      `PKG_CONFIG_LIBDIR` pointing at an empty directory,
      `cargo doc --no-deps --features fftw_system` succeeds; without `DOCS_RS` the same
      build fails in discovery with the message of `build.rs`.
- [ ] `build.rs` rejects FFTW below 3.3.4 with a message naming the version found (checked
      by reading the code and by a unit test of the version comparison).
- [ ] The 0.10 module uses the new lock; no second mutex around FFTW remains; its tests and
      doctests pass unchanged.
- [ ] `FFTW_ALIGNED` is 64 and `PLAN_ALIGNMENT` equals it.
- [ ] The tests of work item 7 pass.

F1c:

- [ ] Every public item of F1 is implemented (no `todo!()` left outside F2's array items)
      and has rustdoc with an example.
- [ ] The tests of work item 12 pass, in CI on Linux and macOS.
- [ ] Every `unsafe` block has a `SAFETY:` comment; the comment on execution covers the four
      new-array conditions.
- [ ] Every FFTW call other than execution runs inside `with_fftw`.
- [ ] No dependency was added except `libc`.

All three:

- [ ] The six Clippy commands of `CLAUDE.md` pass (or are reported unavailable), as do
      `cargo fmt -- --check` and `cargo test --features "fftw,fftw_system,strict"`.
- [ ] `rlst/CHANGELOG.md`, `CLAUDE.md` and `docs/fftw/README.md` are updated.

## Pitfalls

- **Wisdom in tests.** A single `Measure` plan in a unit test changes later `Estimate`
  plans of the same problem in the same process (design Section 3 item 4). Keep to
  `Estimate`.
- **Debug builds.** CI runs tests without `--release`; the reference is O(n²), so keep large
  sizes to the few cells listed.
- **The 2024 edition.** `extern` blocks are `unsafe extern`; operations inside an
  `unsafe fn` need their own `unsafe` blocks.
- **Constant types.** fftw-sys's flag constants are `u32`, except `FFTW_FORWARD` (`i32`);
  the planner takes `c_uint` flags and a `c_int` sign.
- **No `fftw_alignment_of`.** Classes are computed in Rust (design Section 4.6); do not call
  FFTW to check alignment.
- **No SIMD on the macOS runner.** Homebrew's FFTW on Apple silicon has no SIMD, so
  `fftw_alignment_of` is always 0 there and plans use scalar codelets. No F1 test may assume
  SIMD; one that must skips through the SIMD helper (design Section 7.7).
- **Raw pointers from `&`.** Only `Preserving` plans may receive a pointer derived from a
  shared borrow, and only because `FFTW_PRESERVE_INPUT` was passed.
- **`compile_fail` tests** pass for any compile error; the passing twin is what makes them
  meaningful.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design
questions in code.

Raised by the F1 implementation and **resolved in the F1a review** (design draft 0.8,
Appendix A):

1. **Raw access of strided views (for F2).** `RawAccess::data` is `None` for most views
   (`ArraySlice` and `Multislice` only when the view is one contiguous block), and subviews
   do not implement `RawAccess`, so bounds on `RawAccess` could not reach the strided views
   of design goal 3. *Resolved (DC11):* new traits `StridedRawAccess` and
   `StridedRawAccessMut` in `rlst/src/traits/accessors.rs` return the backing storage and
   the offset of the view's first element. F1a declares them and puts them in the F2
   signatures; F2 implements them for every array and view type (subviews and nested views
   included) and tests subviews with sentinels between their elements. `RawAccess` keeps its
   stricter meaning.
2. **Input-policy requests in place.** *Resolved (DC12):* the in-place planner methods exist
   only on `Planner<T, DefaultRequest>`.
3. **An unparsable run-time version string.** *Resolved (DC13):* it counts as below the
   minimum, and planning returns `VersionTooOld` carrying the string.

Additions to the design's sketch, accepted in the same review: `AlignmentClass::of` is
generic over `FftBuffer` (slices now, arrays in F2) in place of a slice-only constructor, and
`AlignmentClass::from_offset` plans for buffers not yet allocated; `DEFAULT_MAX_SCRATCH`;
`Display` for `Rigor`, `Placement`, `Side`, `Backend` and `AlignmentClass`; `Debug` for
`Plan`; the accessors `Plan::layout`, `rigor`, `alignment` and, for c2c, `direction`.
`InvalidLayout` names the dimension as text (`transform dimension 0`, `batch dimension 1`,
`layout`, `alignment`), and `LayoutMismatch` carries the expected and found values as text,
so that F2 can use both for arrays.
