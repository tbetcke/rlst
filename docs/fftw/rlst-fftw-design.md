# FFTW interfaces for RLST: design proposal

Draft 0.17, 2026-10-24. A proposal for the RLST maintainers, not a plan until it is signed
off. It records the decisions taken from 2026-10-21 to 2026-10-24. It replaces RLST's FFTW interface
completely: every transform family of FFTW 3.3's serial interface, plans that are
independent of the arrays they run on, checked and sound execution, wisdom, threads,
alignment classes, determinism with stated preconditions, a test suite sized for CI, and
documentation that states FFTW's licence. The 0.10 interface is removed without a
compatibility layer (DF1), and RLST's own Chebyshev module is ported to the new plans.

The first user is nd-fmm. Its Helmholtz proposal (draft 0.7, `helmholtz-fmm-proposal.md`,
Sections 6.7 and 10) runs this work as Phase HR, an **independent track**: no Helmholtz
phase waits for it, and only CH4.6, the optional FFTW back end of the φ step, uses its
release, through CHR.1 to CHR.3; its schedule is Helmholtz decision D14. nd-fmm's needs
(Section 2.5) inform the order of the work (Section 9). This document makes no decisions
for nd-fmm: Section 11 lists open points and options for nd-fmm's own review.

In the RLST repository this document is `docs/fftw/rlst-fftw-design.md`; the task briefs
are in `docs/fftw/tasks/` and the workflow in `docs/fftw/README.md`.

What drafts 0.6 to 0.17 change, for the start of the implementation:

- **Work plan** (Section 9): the removal of `fftw_source` and `fftw_mkl` moves from F7 to
  F1; the 0.10 module coexists with the new types until F7 and shares the new FFTW lock;
  F0 brings forward CI version reports and a macOS job; each task starts from a brief.
- **Raw layouts and buffers** (Section 4.3): plans execute on slices from F1 and on RLST
  arrays from F2; raw layouts need non-negative strides, equal strides in place, and
  outputs that address distinct elements.
- **Versions and baseline** (draft 0.7, from task F0): Homebrew ships FFTW 3.3.11, which CI
  tests on macOS beside Ubuntu's 3.3.10 on Linux (Sections 2.4, 3 item 9, 4.10, 7.7, DF8);
  the baseline commit is corrected, and the integration branch's version is 0.11.0-dev.
- **Strided storage and two planner rules** (draft 0.8, from the F1a review): arrays and
  views reach FFTW through new traits `StridedRawAccess` and `StridedRawAccessMut`, which
  cover subviews (Sections 4.3, 4.8 and 7.2, DC11); the in-place planner methods take no
  input-policy request (Section 4.4, DC12); an unparsable FFTW version string counts as too
  old (Section 4.10, DC13).
- **Arrays and views at execution** (draft 0.9, from the F2 brief): equal c2c shapes are
  checked when a layout is built from arrays (DC14); execution pairs the axes of input and
  output arrays and documents that a plan describes memory, not logical axes (DC15); arrays
  that the distinct-output rule rejects (DC16); `FftBuffer::len` of an array (DC17); inherent
  `Array::storage()` and `storage_mut()` (DC18); the offset of a view without elements
  (DC19).
- **A view offset that overflows** (draft 0.10, from task F2a): a view whose offset does not
  fit in `usize`, which only a view without elements can have, has no storage (DC20).
- **The pointer that execution checks** (draft 0.11, from task F2b): the span check covers
  the storage behind the very pointer FFTW is given (Section 4.8, DC21).
- **Concurrency and determinism** (draft 0.12, from the F4 brief): `Estimate` plans are
  tested to be independent of earlier `Estimate` planning of other problems (Section 7.6,
  DC22); F4 documents rules 2 to 6 of Section 5 on `Plan` (DC23); the split of Sections 7.5
  and 7.6 between F4 and F6 (Section 9, DC24).
- **Split layouts** (draft 0.13, from the F3 brief): an optional gap per side in `Layout`,
  and raw split layouts for slices (Section 4.5, DC25); the split layout functions count
  logical axes (DC26); execution finds the part axis by its extent and stride (Section 4.8,
  DC27); a layout of the other storage kind is an `InvalidLayout` (Section 4.15, DC28); the
  split layout functions take real arrays (DC29).
- **Real-to-complex and real-to-real layouts** (draft 0.14, from the F5 brief): `Layout`
  records the storage of each side, complex, real or split, from which its family follows
  (Sections 4.2 and 4.3, DC30); the array layout functions read each side's storage from a
  sealed element trait (DC31); in-place r2c and c2r take the complex array, with a contiguous
  halved axis, and safe views of complex buffers as reals (DC32); split r2c and c2r have
  layout functions of their own and no in-place plans (Sections 4.4 and 4.5, DC33);
  `PlanCreationFailed` carries a hint (Section 4.15, DC34); `Plan::logical_size` and
  `Plan::scale_factor` on every plan (Section 4.9, DC35).
- **In-place layouts and preserving c2r** (draft 0.15, from task F5a): in place, both sides
  must address distinct elements, since the transform overwrites its input (Section 4.3,
  DC36); a preserving c2r plan counts every transform dimension, those of extent 1
  included (Section 4.4, DC37).
- **Wisdom and threads** (draft 0.16, from the F6 brief): the wisdom functions' types, with
  a `WisdomFile` error for the file functions (Sections 4.13 and 4.15, DC38); tests that
  import or forget wisdom run in fresh processes only (Section 7.5, DC39); wisdom shared
  between processes is tested to give the same plans (Section 7.6, DC40); `Planner::threads`
  takes a `NonZeroUsize` and exists only with `fftw_threads`, `Plan::threads` on every plan,
  with two new errors (Sections 4.14 and 4.15, DC41); threaded plans in the determinism tests
  (Section 7.6, DC42); F6 adds `fftw_threads` to CI (Sections 7.7 and 9, DC43).
- **The Chebyshev port, docs.rs and the build** (draft 0.17, from the F7 brief): the
  Chebyshev conversions take any number of dimensions and return an empty result for an
  extent 0 (Section 8, DC44); the comparison with 0.10.3 runs while the 0.10 module exists
  (Sections 7.3 and 8, DC45); docs.rs builds with `fftw_threads` (Section 6, DC46); F7
  removes `rlst`'s unused build dependencies (Sections 4.10 and 9, DC47).

Every decision for the RLST maintainers and every design choice is settled (Section 10).
This document makes no decisions for nd-fmm (Section 11). Earlier drafts' changes are
listed in Appendix A.

> **Baseline.** github.com/tbetcke/rlst, tag v0.10.3 (commit 1f5ab87, 2026-10-21; crates.io
> 0.10.3, whose `repository` field points there). Its FFTW code is identical to 0.10.2's
> (CHANGELOG 0.10.3: no code changes): `rlst/src/fftw.rs`, `rlst/src/fftw/c2c_inplace.rs`,
> `rlst/src/fftw/r2r_inplace.rs`, `rlst/src/chebychev.rs`, `rlst/src/doc/optional_integrations.rs`,
> `rlst/build.rs`, `rlst/src/lib.rs`. The Codeberg repository (codeberg.org/rlst/rlst) and the
> older GitHub repository (github.com/linalg-rs/rlst, 0.4.0-dev, last push 2025-12-15) are
> both archived and are not the baseline. CI is GitHub Actions on ubuntu-24.04 with a single
> workflow, `.github/workflows/run-tests.yml`, whose commands `CLAUDE.md` must mirror.
> `main` then received CI clean-up, this design and the workspace version 0.11.0-dev, none
> of which changes FFTW code; the integration branch `fftw-0.11` starts from that `main`
> (commit dc572e5, `docs/fftw/F0-findings.md`), so its crates carry version 0.11.0-dev.

## 1. Goals and non-goals

### 1.1 Goals

1. **Coverage of FFTW 3.3's serial transforms**: complex-to-complex, real-to-complex,
   complex-to-real and real-to-real (all eleven kinds); in place and out of place;
   interleaved and split complex storage; f32 and f64. Every plan is made through the 64-bit
   guru interface. The basic and advanced ("many") planners build the same internal
   problem, so they are special layouts here, not separate entry points.
2. **Plans independent of arrays**: a plan describes a layout, and runs on any arrays that
   satisfy FFTW's four new-array conditions (Section 3 item 1), from any thread.
3. **Layouts from RLST arrays and views**: transforms over chosen axes, batched over the
   others, for column-major, row-major and strided views.
4. **Sound execution**: every safe call either satisfies FFTW's conditions or returns an
   error before calling FFTW; plans that may overwrite their input take it mutably.
5. **Wisdom, threads and alignment** wrapped safely; **determinism** documented with its
   preconditions, testable through plan fingerprints, and tested in isolated processes.
6. **Errors instead of `Option`**: RLST validates layouts, kinds, sizes and back-end
   support before planning and says what failed; a NULL plan from FFTW, which carries no
   reason, is reported with the flags and layout that produced it.
7. **A test suite** against an independent reference, sized for CI (Section 7).
8. **Documentation** with an example for every public item, and the licence warning of
   Section 6.

### 1.2 Non-goals and exclusions

- FFTW's MPI interface; long-double and quad precision; GPU transforms; Windows (no CI
  covers it).
- Building FFTW from source (DF6). The `fftw_source` feature of 0.10 is removed; FFTW comes
  from the system (apt, dnf, Homebrew, HPC modules), as the README already describes.
- Compatibility with the 0.10 API (DF1).
- Intel MKL's FFTW3 wrappers (DF5). The `fftw_mkl` feature of 0.10 is removed in 0.11; an
  MKL back end can be added later without changing the API.
- A back-end-neutral FFT trait (DF4); the design keeps it possible.
- FFTW functions deliberately not wrapped: `fftw_cleanup` and `fftw_cleanup_threads`
  (they make every existing plan undefined); `fftw_execute` on the planning arrays (plans
  are independent of arrays); the 32-bit basic, advanced and guru planners (guru64 covers
  them); `fftw_malloc` and `fftw_free` (DF3); FILE-based wisdom and plan printing (Rust I/O
  is used instead); `fftw_threads_set_callback` and `fftw_planner_nthreads` (FFTW 3.3.9
  and later, not in fftw-sys 0.8; RLST records thread counts itself).

## 2. The existing interface and why it is replaced

### 2.1 What RLST 0.10.3 provides

- Features: `fftw` with one back end, `fftw_system`, `fftw_source` or `fftw_mkl`, through
  `fftw-sys` 0.8 with its default features off. `rlst/build.rs` discovers `fftw3` and
  `fftw3f` with pkg-config, with a fallback to native Homebrew paths on macOS.
- `Array::into_fft()` moves an array into `Fft<ArrayImpl, NDIM>`, which holds it as an
  `Arc<RwLock<…>>`; `borrow_mut()` returns an `FftMutGuard` that allows element mutation
  only. `c2c_fft(flags, direction)` and `r2r_fft(flags, kinds)` create in-place plans
  (`PlanInplace`) over the whole array, which must be contiguous and of at most three
  dimensions. `execute()` takes the array's write lock and runs the plan.
- Planning is serialised by a global mutex (`FFTW_PLAN_INTERFACE`); plans are destroyed
  under it.
- `FftwPlanFlags` is an enum of single flags; `R2RPlanKind` is a re-export of
  `fftw_sys::fftw_r2r_kind`; `FftPrecision` is public and unused.
- `FFTW_ALIGNED = 16` is a public constant in `lib.rs`, not behind the `fftw` feature.
- Precision is chosen at run time by `coe::is_same` checks in `Drop`.
- **In-crate user**: `chebychev.rs` (lines 557–562 and 650–655) computes multidimensional
  DCT-I transforms (`REDFT00` on every axis) through `r_mut().into_fft().r2r_fft(…)` and
  unwraps the result.
- Tests: seven for complex transforms, four for real-to-real, all with one kind on every
  axis.
- The README and `doc/optional_integrations.rs` already note that FFTW is GPL-licensed;
  `CLAUDE.md` asks contributors to keep the README notice.

### 2.2 Limitations

1. No real-to-complex or complex-to-real transforms, and no out-of-place plans.
2. No batched or strided transforms: a [batch, n] array gets a two-dimensional transform,
   and a strided view cannot be transformed at all.
3. A plan is tied to one array, so it cannot run on other buffers or be shared by threads
   working on different data.
4. Flags cannot be combined.
5. A failed plan is `None`, without a reason.
6. The 16-byte alignment constant matches FFTW's SIMD requirement on current x86 builds,
   but a plan's alignment class is not checked (Section 4.6).
7. Planning uses the caller's array, so `Measure` and stronger flags overwrite user data.
8. The global mutex protects RLST's planning only; another library, or another copy of
   RLST in the same build, can still race.
9. **Wrong axes for kinds.** `r2r_inplace.rs` reverses the shape of column-major arrays
   (FFTW's dimension order is row-major) but passes `kind` unreversed. On RLST's default
   column-major arrays, `kind[0]` is applied to the last axis. No test notices, because
   every test uses one kind on all axes, as does `chebychev.rs`.
10. With `fftw` and no back end, or two back ends, the build fails only at link time.

### 2.3 Why a rewrite rather than an extension

Items 3, 7 and 9 follow from the shape of the 0.10 API: a plan that owns a shared reference
to its array, planned on that array, with kinds ordered by FFTW's dimensions. Fixing them
changes every signature, so the new interface is designed from FFTW's guru64 and new-array
execute functions outward, and the old one is removed (DF1).

### 2.4 What the back ends actually are

| Back end | What it links | Notes |
| --- | --- | --- |
| `fftw_system` | the system's `libfftw3` and `libfftw3f`, any version pkg-config finds | Ubuntu 24.04 ships 3.3.10 (`fftw-3.3.10-sse2-avx`) and Homebrew 3.3.11, built with SSE2, AVX and AVX2 on Intel and without SIMD on Apple silicon (`fftw-3.3.11`). The threads libraries (`libfftw3_threads`, `libfftw3f_threads`) have no pkg-config files and are not linked today. |
| `fftw_source` | FFTW 3.3.8, built by fftw-src 0.8.0, statically | `configure` runs with `--with-pic --enable-static --disable-doc` (and `--enable-single`) only: **no SIMD and no threads library**. Without SIMD, `fftw_alignment_of` returns 0 for every pointer. 3.3.8 predates the fixes of 3.3.9 and 3.3.10 (Section 3 item 9). Removed in 0.11 (DF6). |
| `fftw_mkl` | Intel MKL's FFTW3 wrappers, through intel-mkl-src 0.8.1 | A documented subset of FFTW (Section 3 item 10). Removed in 0.11 (DF5). |

Licences of the crates in the graph: fftw-src declares GPL-2.0-or-later (it leaves the graph
in 0.11, DF6). fftw-sys declares a `license-file` whose text distributes the bindings under
GPL-2.0-or-later with the FFTW back ends, and under MIT with MKL. Licence checkers such as cargo-deny are unlikely to
classify that file and may need a clarification entry for it.

### 2.5 What nd-fmm needs (the first user)

From Helmholtz draft 0.7, Section 6.7:

- **Per-box plans with a fixed batch**: one plan per (level, N_φ) of 2P − 1 rings of length
  N_φ along the φ axis (FFTW's howmany = 2P − 1), executed box by box on each box's own
  aligned buffer by whichever rayon worker holds the box. Plans over all boxes of a level,
  or over chunks of varying size, are rejected, since they would make a box's bits depend
  on batch shape (Helmholtz constraint 2).
- c2c in f64 on the host; N_φ a multiple of 4 whose other prime factors are at most 7,
  from about 140 (p = 33) to 1080 (p = 261), for example 216, 588 and 1080; powers of two up
  to 2048 on device levels (not through FFTW).
- `ESTIMATE` plans, single-threaded, made once per operator build; never FFTW's threads.
- Split planes (Helmholtz Section 3.3) or interleaved scratch; spike S9 decides
  (Section 11).
- Determinism bit for bit across threads and ranks.
- Later, two-dimensional transforms over (θ, φ) for the Cecka–Darve upgrade.

r2c, c2r, r2r, wisdom and threads are not needed by nd-fmm.

## 3. FFTW facts the design rests on

Each item names its source: the FFTW 3.3 manual (`doc/*.texi` in the 3.3.8 sources), the
3.3.8 sources, FFTW's NEWS, or Intel's documentation of the FFTW3 wrappers. Items 4, 6, 7
and 11 were also checked with small C programs against Ubuntu 24.04's FFTW, whose version
string is `fftw-3.3.10-sse2-avx`.

1. **New-array execute** (manual, "New-array Execute Functions"). A plan may run on other
   arrays when all four hold: the sizes and strides are those of the plan; the arrays are in
   place or out of place as the plan was; for split arrays, the gaps `ii − ri` and `io − ro`
   are those of the planning arrays; and each array's alignment equals that of the
   corresponding planning array, unless the plan was made with `FFTW_UNALIGNED`.
2. **Thread safety** (manual, "Thread safety"). Only the execute functions are thread-safe,
   and the same plan may be executed by several threads at once on different arrays. Every
   other function, the planner, plan destruction and wisdom included, must be called from
   one thread at a time.
3. **`fftw_make_planner_thread_safe`** (3.3.6 and later; broken in 3.3.5) installs a lock
   around plan creation and destruction only (`api/apiplan.c`). Wisdom import, export and
   forgetting, `fftw_plan_with_nthreads` and `fftw_set_timelimit` stay unprotected. The
   manual says it does not work with the OpenMP threading library (3.3.5, 3.3.6) and advises
   against it unless there is no other choice.
4. **Wisdom is process-global and always used** (manual, planner flags). Every plan adds to
   the wisdom of its precision. When wisdom for a problem exists from equal or greater
   planning rigor it is used, so in `ESTIMATE` mode any available wisdom is used. An
   `ESTIMATE` plan therefore depends on what was planned or imported earlier in the process.
   Checked: for 289 transforms of length 588, an `ESTIMATE` plan made after a `PATIENT`
   plan of the same problem is the `PATIENT` plan, not the earlier `ESTIMATE` plan.
5. **Input destruction** (manual, planner flags). Out-of-place transforms preserve their
   input by default, except c2r and HC2R, which destroy it by default. `FFTW_DESTROY_INPUT`
   allows any out-of-place plan to overwrite its input. Multidimensional c2r has no
   input-preserving algorithm: with `FFTW_PRESERVE_INPUT` the planner returns NULL.
6. **Alignment** (`kernel/align.c`, `simd-support/simd-common.h`, NEWS). On a SIMD build,
   `fftw_alignment_of(p)` is `p mod 16`; without SIMD it is 0. In 3.3.8 and older, AVX-512
   and KCVI codelets test 64-byte alignment at plan time while `fftw_alignment_of` still
   reports modulo 16, so a plan made on 64-byte-aligned arrays can pass that check on a
   16-byte-aligned array and then fault. 3.3.9 sets the AVX-512 requirement to 16 bytes.
   `fftw_malloc` aligns to 16, 32 or 64 bytes depending on the build. Checked:
   `fftw_alignment_of` gives 0, 8 and 0 for offsets of 0, 1 and 2 doubles. Homebrew's 3.3.11
   on Apple silicon has no SIMD, and both precisions give 0 for offsets of 0 to 3 elements.
   The version string names the SIMD extensions of the build: `api/version.c` appends
   `-fma` and then one suffix per extension (`-sse2`, `-avx`, `-neon`, `-sve`, …), so a
   build has SIMD when its string carries a suffix other than `-fma`.
7. **SIMD codelets and layout** (`dft/simd/common/genus.c`). FFTW's SIMD codelets need
   interleaved complex storage (`ii == ri + 1`) and a batch count that is a multiple of the
   SIMD width; the one split-storage family needs a batch stride of 1. For split storage
   FFTW either copies into buffers for a SIMD stage or uses scalar codelets. A plan for a
   different batch count or layout can therefore choose different codelets and round
   differently. Checked: for 289 transforms of length 588, the interleaved `ESTIMATE` plan
   uses AVX codelets in every stage, the split plan mixes one buffered AVX stage with
   scalar ones, and the plan for 288 transforms differs from the plan for 289.
8. **Split DFTs have no sign** (manual, "Guru Complex DFTs"). The split guru planner always
   plans a forward transform; a backward transform is the forward one with real and
   imaginary parts swapped. Split strides are in units of reals, equal for both parts.
9. **Versions** (NEWS). `fftw_alignment_of` and `fftw_sprint_plan`: 3.3.4. Guru64: 3.2.
   `fftw_make_planner_thread_safe`: 3.3.5, working from 3.3.6. `fftw_planner_nthreads` and
   `fftw_threads_set_callback`: 3.3.9. AVX-512 alignment and wisdom fixed: 3.3.9. A
   segfault from unaligned accesses of 2-way SIMD (SSE2 in double precision) in rare
   strided r2c cases fixed: 3.3.10. `fftw_copy_plan`, SVE and LoongArch SIMD, and cycle
   counters for Apple ARM and RISC-V: 3.3.11, which lists no correctness fixes.
10. **MKL's FFTW3 wrappers** (Intel oneMKL developer reference, "Using FFTW3 Wrappers"),
    kept as background for a later MKL back end; this design does not use them (DF5).
    Not supported: c2r and r2c with split complex storage; the r2r kinds R2HC, HC2R and DHT
    (only the sine and cosine kinds); multidimensional r2r; a `howmany_rank` other than 0
    or 1 in the guru and guru64 planners; long double. The wrappers use 32-bit `int`
    internally even with ILP64, and report problems by returning a NULL plan.
11. **Kinds and sizes** (manual, "Real-to-Real Transform Kinds"). Logical sizes N: R2HC,
    HC2R and DHT: n; REDFT00: 2(n − 1); RODFT00: 2(n + 1); the other DCT and DST kinds: 2n.
    REDFT00 is not defined for n = 1. Checked: the planner returns NULL for it.
12. **Threads initialisation** (manual, "Usage of Multi-threaded FFTW"). `fftw_init_threads`
    must be called before any other FFTW routine. The threads library is linked before
    `fftw3` (`-lfftw3_threads -lfftw3`), which matters for static linking.

## 4. Design

### 4.1 Module structure

```
rlst::fftw
├── real         FftwReal: f32 and f64, sealed, mapping to the fftwf_ and fftw_ functions
├── kind         Direction, R2rKind (RLST's own enum), logical sizes
├── options      Rigor, wisdom-only, input policy, unaligned, time limit, threads
├── element      FftElement (sealed): f32, f64, c32, c64; as_reals, as_reals_mut
├── layout       Dim, Storage, Layout: from RLST arrays and views, normalised
├── align        PLAN_ALIGNMENT, AlignmentClass
├── plan         Plan<T, F, P>: families C2c, R2c, C2r, R2r, SplitC2c, SplitR2c, SplitC2r;
│                policies Preserving, Destroying, InPlace
├── planner      Planner<T, P>: builder, scratch, the global FFTW lock
├── execute      checked new-array execution
├── scale        Scale: None, Inverse, InverseSqrt
├── fingerprint  Plan::fingerprint, Plan::description
├── wisdom       export and import (strings, files), forget, system wisdom
├── threads      (feature fftw_threads) initialisation, make_planner_thread_safe
├── backend      Backend (System), version()
└── error        FftwError
```

### 4.2 Types and their contracts (a sketch, not the final API)

```rust
/// f32 or f64; static dispatch to the fftwf_ / fftw_ functions.
pub trait FftwReal: RlstScalar<Real = Self> + sealed::Sealed {}

pub enum Direction { Forward, Backward }

/// FFTW's eleven real-to-real kinds, as RLST's own type (not a re-export of fftw-sys).
pub enum R2rKind { R2hc, Hc2r, Dht, Redft00, Redft01, Redft10, Redft11,
                   Rodft00, Rodft01, Rodft10, Rodft11 }

/// Planning rigor. Combined with `wisdom_only`, FFTW plans only from wisdom of that rigor
/// or more.
pub enum Rigor { Estimate, Measure, Patient, Exhaustive }

/// One dimension of a guru64 layout: length and strides, in elements of the input and
/// output arrays (complex elements for interleaved complex, reals otherwise). `isize`
/// because FFTW's `fftw_iodim64` uses `ptrdiff_t`.
pub struct Dim { pub n: isize, pub is: isize, pub os: isize }

/// The storage of one side of a layout (DC30).
pub enum Storage { Complex, Real, Split { gap: isize } }

/// Transform and batch dimensions in the order FFTW receives them, the placement, and the
/// storage of each side, from which the family follows (c2c, r2c, c2r, r2r and their split
/// forms). For r2c and c2r the dimensions record the logical real extent n; the last
/// transform dimension has extent n/2 + 1 on the complex side.
pub struct Layout { /* … */ }

/// A plan of family F (C2c, R2c, …) and input policy P (Preserving, Destroying, InPlace).
/// Send + Sync: execution is thread-safe in FFTW; destruction goes through the FFTW lock.
pub struct Plan<T: FftwReal, F: Family, P: Policy> { /* raw plan, layout, classes, … */ }
```

Usage, for nd-fmm's per-box φ step:

```rust
// The rings of one box: [n_phi, rings], column-major, in an FFTW_ALIGNED buffer.
let layout = Layout::along_axis(&box_in, &box_out, 0)?;   // n = n_phi, batch = rings
let plan = Planner::<f64>::new()                          // Rigor::Estimate by default
    .c2c(&layout, Direction::Backward)?;                  // Plan<f64, C2c, Preserving>

// Any worker, any box with the same layout and alignment class:
plan.execute(&box_in, &mut box_out)?;
```

### 4.3 Layouts

- `Layout::along_axis(&input, &output, axis)`: one-dimensional transforms along `axis`,
  batched over the other axes. `Layout::over_axes(&input, &output, &axes)`: transforms
  over `axes` in the order given, batched over the rest. `*_inplace` variants take one
  array. The functions are generic over RLST arrays and views with
  `StridedRawAccess + Shape<N> + Stride<N>`, with `N` the rank of both arrays, whose items
  implement a sealed trait `FftElement<Real = T>` (`f32`, `f64`, `c32`, `c64`, one precision
  `T` for both arrays): the function reads each side's storage, real or complex, from its
  element type (DC31), so the input and output may differ in element type and shape (real
  and complex for r2c and c2r, two real arrays for r2r). The split layout functions take
  real arrays only (Section 4.5, DC29).
- **Strided storage (DC11).** `StridedRawAccess::storage` returns a view's backing storage
  and the offset of its first element; with the view's strides, element `i` is
  `storage[offset + Σ i[k] · stride[k]]`, and the slice may hold elements of other views
  before, between and after the view's own. `StridedRawAccessMut::storage_mut` is the
  mutable counterpart. They live beside `RawAccess` in `rlst/src/traits/accessors.rs`, are
  not gated by `fftw`, and are implemented explicitly for each array and view type: owned
  arrays return their storage at offset 0, `r()` and `r_mut()` forward, and `ArraySlice`,
  `Multislice` and `ArraySubView` return their parent's storage with their own offset added,
  so nested and mixed views compose. `RawAccess` keeps its stricter promise, on which
  existing callers rely (its slice holds the array's elements and starts with the first);
  a type that implements both keeps them consistent: when `data()` is `Some(d)`, `d` is
  `&storage[offset..offset + d.len()]`. Lazy expressions have no storage; `transpose()`,
  `permute_axes()` and `reverse_axis()` do not implement `Stride` and are not covered. A
  mutable view hands out its parent's whole storage, which is memory-safe, since the view
  holds the parent's exclusive borrow, but no longer a strict boundary; the documentation
  says so. `Array::storage()` and `Array::storage_mut()`, inherent methods beside `data()`
  and `data_mut()`, give users the same access without naming the implementation type
  (DC18). For a view without elements the offset may lie past the end of the slice, for
  example a subview of shape [0, 0] at the far corner of its parent, and must not be used to
  index it; the traits' contract says so (DC19). A view whose offset does not fit in
  `usize`, which only a view without elements can have (an array with an extent 0 accepts
  any strides), returns `None` from `storage()` and `storage_mut()`: views compute their
  offset with checked arithmetic (DC20).
- **Validation by family.** c2c needs complex arrays of equal shape. The layout functions
  check it when they build the layout, and return `InvalidLayout` naming the axis (DC14): a
  `Layout` records one extent per dimension, not the shapes of both sides, so the planner
  cannot. F5 relaxes the check on the halved axis for r2c and c2r, and `Layout` records the
  storage of each side, which tells the families apart and gives the halved extent on the
  complex side (DC30). r2c needs a real input and a complex output whose extent along the **last entry of
  `axes`** (for `along_axis`, the axis itself) is n/2 + 1; that axis is the halved one,
  whatever the memory order. c2r is the inverse; r2r needs real arrays of equal shape and
  one `R2rKind` per entry of `axes`, in the same order. Kinds therefore belong to RLST axes by construction, which removes the 0.10
  bug of Section 2.2 item 9.
- **The halved axis (DC2)** is the last entry of `axes`, the rule numpy's and scipy's
  `rfftn` use for explicit axes, and the dimension FFTW halves, since RLST passes the
  dimensions in the order of `axes`. A shape that does not fit returns `InvalidLayout`
  naming the axis expected to have extent n/2 + 1 and the extent found. Whether listing the
  contiguous axis last is faster is measured by the F8b benchmarks before the documentation
  recommends it.
- **In-place r2c and c2r** use a complex array; the real side is the same buffer read with
  the real extent n along the halved axis and padded to 2(n/2 + 1) reals, as FFTW requires.
  `Layout::r2c_inplace(&complex, &axes, n_last)` and `Layout::c2r_inplace(&complex, &axes,
  n_last)` take the logical real extent explicitly, since it cannot be recovered from the
  complex shape, and n_last/2 + 1 must equal the complex extent along the halved axis
  (DC32). The halved axis must be contiguous (complex stride 1), and on every other axis the
  real stride is twice the complex stride; other strides return `InvalidLayout`, and a
  stride that FFTW also accepts is added only by a later design change. Execution takes the
  complex buffer. The safe functions `fftw::as_reals(&[Complex<T>]) -> &[T]` and
  `as_reals_mut` view a complex slice as its reals (`Complex<T>` is `#[repr(C)]` with two
  `T`s, Section 4.11), so that callers write the r2c input and read the c2r output through
  slices and `SliceArray`s.
- **Normalisation**, before planning and before every comparison at execution: batch
  dimensions of extent 1 are dropped, and adjacent batch dimensions that are contiguous in
  both input and output are merged. FFTW compresses its batch dimensions in the same way
  internally. When the batch axes are adjacent in memory, as for a transform along the
  first or the last axis of a contiguous 3-D array, one batch dimension remains.
- **Raw layouts.** `Layout` can also be built from `Dim` lists directly (F1), with offsets
  counted in elements from the start of each buffer. `Layout::with_storage(dims, batch,
  (input, output), placement)` builds raw layouts of every family from the storage of each
  side (DC30); `Layout::new` and `Layout::new_split` are its shorthands for c2c and split
  c2c, and the halved dimension of r2c and c2r is the last transform dimension. Strides must be non-negative in 0.11
  (RLST's views have no negative strides); an in-place layout needs equal input and output
  strides for c2c, split c2c and r2r, as FFTW requires, and the padded layout of DC32 for
  r2c and c2r; and the output must address distinct elements, as must the input of an
  in-place layout, which the transform overwrites (DC36). For c2c, split c2c and r2r in
  place the two sides have equal strides, and for r2c the rule on the complex output implies
  it for the real input, but an in-place c2r layout can have distinct real outputs and
  overlapping complex inputs, such as `(8, 1, 1)` with the batch `(3, 4, 8)`, whose last
  complex element of each transform is the first of the next. That last condition is
  checked by a sufficient rule: with the dimensions of extent above 1 sorted by their stride
  on the side checked, every stride is positive and at least the extent times the stride of
  the next smaller one. Layouts built from RLST arrays satisfy the first two by
  construction, and usually the third. `StridedBaseArray::new` accepts any strides that map
  indices to distinct offsets, such as shape [2, 2] with strides [2, 3] (offsets 0, 2, 3 and
  5), which the sufficient rule rejects, since 3 is below 2 · 2. Such arrays are rejected
  with `InvalidLayout` as outputs and in place, and accepted as inputs, whose elements need
  not be distinct; the documentation says so (DC16).
- **Buffers.** Plans execute on slices, with the layout's offsets counted from the start of
  the slice (F1), and on RLST arrays and views through their strided storage, with the
  offsets counted from the view's first element (F2). Slices are the low-level form, for
  callers that manage their own buffers.
- Layouts carry no FFTW types (DF4).

### 4.4 Plans and input policy

The input policy is part of a plan's type, so the borrow checker enforces Section 3 item 5:
a plan that may overwrite its input can only be executed with the input borrowed mutably.

| Policy | Execution | FFTW flags |
| --- | --- | --- |
| `Preserving` | `execute(&input, &mut output)` | `FFTW_PRESERVE_INPUT`, given explicitly for c2r and HC2R |
| `Destroying` | `execute(&mut input, &mut output)` | `FFTW_DESTROY_INPUT` |
| `InPlace` | `execute(&mut data)` | |

**Defaults (DC1).** Each family follows FFTW's own default:

| Planner call | Policy of the plan |
| --- | --- |
| `Planner::new().c2c(…)`, `r2c`, `r2r`, `split_c2c`, `split_r2c` | `Preserving` |
| `Planner::new().c2r(…)`, `split_c2r` | `Destroying`, as in FFTW |
| `Planner::new().preserve_input().c2r(…)` | `Preserving`; one transform axis only |
| `Planner::new().may_destroy_input().c2c(…)`, and for every other family | `Destroying` |
| `Planner::new().c2c_inplace(…)`, and for every other family except split r2c and c2r, which have no in-place plans in 0.11 (DC33) | `InPlace` |

The in-place methods exist only on a planner without an input-policy request
(`Planner<T, DefaultRequest>`), since the requests mean nothing in place (DC12); allowing
them later would not be a breaking change.

- Because a destroying plan's `execute` takes `&mut input`, the default of c2r is visible
  at the call site; nothing is overwritten that the caller did not lend mutably. The
  documentation states the exception once: c2r destroys its input unless asked not to.
- `preserve_input()` on a c2r plan over more than one transform dimension, those of extent
  1 included (DC37), is rejected before FFTW is called, with
  `FftwError::PreserveInputUnsupported`: FFTW has no input-preserving algorithm for it
  (Section 3 item 5). On one axis FFTW preserves the input at some cost in
  speed.
- r2r stays preserving by default although HC2R destroys its input in FFTW: the kinds are
  known only at run time, so the plan's type cannot depend on them. RLST passes
  `FFTW_PRESERVE_INPUT` when an HC2R kind is present; if FFTW cannot make such a plan, the
  result is `PlanCreationFailed` with a hint to `may_destroy_input()`.

**Planning arrays (DC3).**

- `Planner::c2c(&layout, …)` plans on scratch arrays that the planner allocates for the
  layout's span, at the requested alignment class (and, for split storage, with the
  layout's gap), and frees after planning. Planning never touches user data, whatever the
  rigor. The scratch covers the full span because FFTW forms pointers across it; a smaller
  buffer would rely on undefined behaviour in C.
- **Scratch cap.** A strided view of a large array spans the whole array: one column of a
  10 GB matrix needs 10 GB of scratch. `Planner::max_scratch(bytes)` bounds the scratch, by
  default at 1 GiB for input and output together. Above the bound, planning returns
  `FftwError::ScratchTooLarge` with the size needed and a pointer to the `*_on` variants.
- `Planner::c2c_on(&mut input, &mut output, …)` plans on the caller's arrays and needs no
  scratch. With `Estimate` or wisdom-only planning FFTW leaves the arrays untouched (manual,
  planner flags); with `Measure` and stronger it overwrites them, as documented on the
  method.
- The same pattern holds for every family (`r2c`, `c2r`, `r2r`, `split_c2c`, …).

### 4.5 Split complex storage

A split complex array is a **real** array with one extra axis of extent 2, the *part axis*:
index 0 is the real part, index 1 the imaginary part. The stride of that axis is the gap
`ii − ri` of Section 3 item 1. `Layout::split_along_axis(&input, part_in, &output,
part_out, axis)` and `split_over_axes` build split layouts. Because the gap is a stride of
the layout, the comparison of strides at execution checks it.

- **Representation (DC25).** A split layout's dimensions, gaps and spans count reals.
  `Layout` records an optional gap per side, never as a batch dimension, so normalisation
  never merges the part axis into a batch axis; `Layout::gaps()` returns
  `Some((input_gap, output_gap))` for a split c2c layout. `Layout::new_split(dims, batch,
  (input_gap, output_gap), placement)` builds raw split layouts for slices, with the rules
  of `Layout::new`, positive gaps, equal gaps in place, and the distinct-output rule with
  the part axis as a dimension of extent 2. F5's split r2c and c2r, which have a gap on one
  side only, fit the per-side record without a breaking change: DC30 generalises it to the
  storage of each side, `Storage::Split { gap }` on a split side, and `gaps()` keeps its
  meaning, `Some` only when both sides are split.
- **Split r2c and c2r (DC33).** `Layout::split_r2c_along_axis(&input, &output, part_out,
  axis)`, `split_r2c_over_axes(&input, &output, part_out, &axes)`,
  `split_c2r_along_axis(&input, part_in, &output, axis)` and `split_c2r_over_axes(&input,
  part_in, &output, &axes)` take two real arrays (DC29), the split side with a part axis of
  extent 2, and count its logical axes (DC26); the halved axis is the last entry of `axes`.
  They have no in-place variants, and there are no in-place split r2c or c2r plans in 0.11:
  FFTW's in-place forms would pass the real buffer as one plane of the split side, which no
  user needs and which the checks of DC27 would have to cover.
- **Axes (DC26).** `axis` and `axes` count **logical axes**: the array's axes with its part
  axis removed, numbered in order. One `axes` then means the same transform on an input
  and an output whose part axes are at different positions; errors name the logical axis.
- **Real arrays (DC29).** The split layout functions require real items on both arrays
  (`BaseItem<Item = T>` with `T: FftwReal`), a bound only, so that a layout built from
  complex arrays, which would count complex elements, does not compile.

- nd-fmm's split planes, the real plane then the imaginary plane of each box at a size
  fixed per level, are a part axis outermost with a stride equal to the plane size, the same
  for every box. Four independent arrays, as in draft 0.1, cannot be expressed and are not
  needed.
- Backward split plans swap the parts internally (Section 3 item 8); the plan stores the
  resulting signed gap and execution applies the same swap.
- Split plans can use fewer SIMD stages than interleaved ones (Section 3 item 7). The
  documentation says so, and benchmarks compare them with interleaved plans.

### 4.6 Alignment classes

- `PLAN_ALIGNMENT = 64`. `AlignmentClass` is a data pointer's address modulo 64, computed
  in Rust, for each of the input and output pointers (and both parts of split storage).
  Equal classes modulo 64 imply equal `fftw_alignment_of` (modulo 16) and also cover the
  64-byte requirement of AVX-512 builds of FFTW 3.3.8 and older (Section 3 item 6), so the
  rule is sufficient on every build, and execution makes no FFI call to check it.
- The planner plans for class 0 by default; `.alignment(in_class, out_class)` plans for
  others, and `AlignmentClass::of(&array)` reads a class from an array. Scratch arrays are
  allocated at the requested class.
- `.unaligned()` plans with `FFTW_UNALIGNED` and skips the class check, at the cost of most
  SIMD codelets.
- `FFTW_ALIGNED` becomes 64 (= `PLAN_ALIGNMENT`). It stays a public constant outside the
  `fftw` feature; the change is recorded in the CHANGELOG.
- The default `DynArray` is a plain `Vec`, aligned only to its element (8 bytes for
  f64 and c64, 4 for f32 and c32), so its class depends on the global allocator, and views
  at an offset change class. The documentation steers plan targets to
  `rlst_dynamic_array!(…, shape | rlst::FFTW_ALIGNED)` and `AlignedDynArray`, and the
  alignment error names the expected and found class and that remedy.
- Allocation stays with RLST's existing aligned containers (aligned-vec); `fftw_malloc` is
  not used (DF3).

### 4.7 The planner, the FFTW lock and global state

One process-wide lock serialises **every FFTW call except execution**: planning, plan
destruction, wisdom export, import and forgetting, system wisdom, `fftw_sprint_plan`,
`fftw_init_threads`, `fftw_plan_with_nthreads` and `fftw_set_timelimit`. A planner call
sets the thread count and time limit inside the critical section and resets them (to 1 and
to no limit) before releasing it, so they never leak into another caller's plan. Execution
takes no lock.

The lock protects RLST's own calls. It cannot protect FFTW from another library in the
process, nor from a second, semver-incompatible copy of RLST in the same build, which has a
lock of its own (DF7). With `fftw_threads`, `make_planner_thread_safe()` adds FFTW's own
lock around plan creation and destruction in every caller, within the limits of Section 3
item 3. The documentation states both limits.

The thread count and time limit are reset by a guard that also runs during unwinding, so a
panic in RLST code inside the critical section leaves FFTW's settings as they were, and a
poisoned lock is recovered rather than propagated.

### 4.8 Execution checks

Before every FFTW call, in O(rank) time:

1. element types and family, by the types of `Plan` and of the arrays (compile time);
2. for an array, its shape and strides match its side of the plan's layout: each transform
   dimension is matched by an axis of the same extent and, above extent 1, the same stride,
   and the other axes, normalised on that side alone, equal the layout's batch on that side,
   including the part-axis stride of split storage. When both buffers are arrays of the same
   rank, their axes are also paired by index: equal shapes, each transform dimension matched
   by one axis with its extent and both strides, and the other axes, normalised together,
   equal to the layout's batch (DC15). The offset of the array's first element plus the
   layout's span fits its strided storage: the storage behind the pointer that FFTW is
   given, through the exclusive borrow for a buffer the plan writes. The check returns that
   pointer and execution uses no other, so an implementation of the safe traits that returns
   another slice on another call, or a shorter one from `storage_mut()`, cannot get around it
   (DC21). For a slice, the layout's span fits within it. For split storage the layout
   records gaps, not the position of the part axis (DC27): on each side, the part axis is
   an axis of extent 2 whose stride is that side's gap and that no transform dimension
   needs, and the array is accepted if any such choice fits, since several can qualify on
   an input; when both buffers are arrays of the same rank, each side's part axis is
   removed and the remaining axes pair by index;
3. the alignment class of every data pointer equals the plan's, unless the plan is
   unaligned;
4. placement: an in-place plan gets one array; for an out-of-place plan the address spans
   of input and output are disjoint. The check is conservative: two strided views that
   interleave without sharing elements are rejected. Safe borrows already rule out overlap
   except through arrays built from raw parts.

A failed check returns its error, names the expected and found values, and calls nothing.

The span of a strided view can include elements that belong to other views, such as the
other rows of a matrix around a subview. FFTW reads and writes only the elements a layout
addresses, never the gaps between them, as its advanced interface relies on for
transforming parts of larger arrays; execution's safety argument states this, and F2's
tests check it with sentinel values in the gaps.

A plan describes memory, not logical axes (DC15). No check on strides can tell logical axes
apart: a plan made along axis 0 of a square column-major array transforms along axis 1 of a
square row-major array, whose memory layout is the same. Without the pairing of item 2, a
plan from a column-major input of shape [4, 2, 3] and an output of the same shape with
strides [1, 12, 4] would also accept column-major arrays of shape [4, 3, 2], and send the
transform of input column [:, 1, 0] to output column [:, 0, 1]. The pairing rejects that
when both buffers are arrays; a call that mixes a slice and an array is checked per side
only. The documentation of the layout functions and of execution states both points.

### 4.9 Scaling

FFTW's transforms are unnormalised. `Plan::logical_size() -> usize`, on every plan (DC35),
returns the product over the transform dimensions of each one's logical size: n for c2c and
split c2c; the logical real length for r2c and c2r and their split forms; for r2r, the
logical size of each kind (Section 3 item 11). `Layout::logical_size` stays the product of
the extents, which is not the logical size of r2r, and its documentation says so.
`Plan::scale_factor(s) -> T`, on every plan, returns 1, 1/N or 1/√N for `Scale::None`, `Scale::Inverse` and `Scale::InverseSqrt`, which
the caller applies with RLST's array arithmetic. A uniform factor does not make the DCT and
DST kinds orthonormal, which needs different weights at the end points; orthonormal r2r
variants are not provided, and the documentation says so. `Scale` avoids the names
Forward and Backward, which belong to `Direction`.

### 4.10 Back ends, versions and features

**Features.**

- `fftw_system` is the only back end and implies `fftw`, which enables the module (today
  `fftw_system` alone enables fftw-sys and links FFTW without the module). `fftw_source`
  (DF6) and `fftw_mkl` (DF5) are removed. Keeping the back end a feature of its own leaves
  room for a later one without renaming features.
- A `compile_error!` fires when `fftw` is enabled without `fftw_system`.
- `fftw_threads` implies `fftw_system`.

**Versions.**

- `build.rs` reads the version of `fftw3` and `fftw3f` from pkg-config and fails with a
  clear message below **3.3.4** (`fftw_sprint_plan`, used by fingerprints), or below
  **3.3.6** when `fftw_threads` is on (a working `make_planner_thread_safe`).
- 3.3.10 or later is recommended, since 3.3.10 has the last correctness fixes. CI tests
  3.3.10 on Linux (Ubuntu 24.04) and 3.3.11 on macOS (Homebrew); the documentation lists
  the fixes of 3.3.9 and 3.3.10 (Section 3 item 9). It also states that Homebrew's FFTW on
  Apple silicon is built without SIMD, and that users who need its speed there build FFTW
  with `--enable-neon` (and `--enable-single --enable-neon` for `fftw3f`) and point
  `PKG_CONFIG_PATH` at that installation.
- `fftw::version()` returns FFTW's own version string, through RLST's declaration of the
  `fftw_version` symbol (fftw-sys does not bind it). It is checked at the first planner call,
  which covers the Homebrew fallback, where pkg-config reports no version. A version string
  that does not parse counts as below the minimum: planning returns `VersionTooOld`, which
  carries the string (DC13).
- When `DOCS_RS` is set, `build.rs` skips discovery: rustdoc does not link, so docs.rs can
  build with `fftw_system` without FFTW installed. The build still compiles RLST's build
  dependencies. `build.rs` uses only `pkg-config`, but `rlst` also declared `bindgen`, `cc`,
  `cmake` and `git2`, used by no build script since 0.10.3 or earlier, and `bindgen` needs
  `libclang` (found while checking F1 in a container without it). F7 removes those four
  (DC47); the docs.rs build image provides `libclang` and FFTW in any case (brief F7).

**`fftw_system`** links `fftw3` and `fftw3f` as today. With `fftw_threads` it also links
`fftw3_threads` and `fftw3f_threads`, never the `_omp` libraries. Those libraries have no
pkg-config files, so `build.rs` looks for them in the directories pkg-config reported for
`fftw3`, emits them before `fftw3`, and fails with a clear message if they are missing.

### 4.11 Bindings and the licence of the dependency

RLST keeps fftw-sys 0.8 for declarations and linking (DF2), and declares only the version
symbols itself. Generating RLST's own bindings is not proposed: fftw-sys distributes its
generated bindings under GPL-2.0-or-later with the FFTW back ends, and generating
equivalents from `fftw3.h` inside the MIT/Apache `rlst` crate would raise the same question
there. It would also need libclang at build time. Functions newer than FFTW 3.3.8 are not
used. `R2rKind` maps to fftw-sys's constants internally, so a later change of bindings
breaks no public type. `libc`, already in the graph through fftw-sys, becomes an optional
dependency of `fftw`, for `free` on the strings of `fftw_sprint_plan`.

The documentation tells downstream users that fftw-sys declares its licence through a
`license-file`, so cargo-deny and similar tools will likely need a clarification entry
for it in configurations that enable `fftw`.

### 4.12 Fingerprints

`Plan::description()` returns FFTW's description of the plan (`fftw_sprint_plan`, under the
lock). `Plan::fingerprint()` returns a 64-bit FNV-1a hash of it, a hash function that is the
same in every process and build. Equal fingerprints mean FFTW chose the same algorithm,
which is what Section 5's determinism rules need and what a distributed caller can compare
across ranks in one all-reduce.

### 4.13 Wisdom

`wisdom::export::<T>()` returns a `String` through FFTW's callback export, so no C
allocation has to be freed. `import::<T>(&str)`, `export_to_file` and `import_from_file`
(through Rust I/O), `forget::<T>()` and `import_system_wisdom::<T>()` (FFTW's
`/etc/fftw/wisdom`, only on explicit request) complete it. All run under the lock, per
precision; file I/O runs outside it. Section 5 states their role in determinism.

The types (DC38):

```rust
pub fn export<T: FftwReal>() -> String;
pub fn import<T: FftwReal>(wisdom: &str) -> Result<(), FftwError>;
pub fn export_to_file<T: FftwReal>(path: impl AsRef<Path>) -> Result<(), FftwError>;
pub fn import_from_file<T: FftwReal>(path: impl AsRef<Path>) -> Result<(), FftwError>;
pub fn forget<T: FftwReal>();
pub fn import_system_wisdom<T: FftwReal>() -> Result<(), FftwError>;
```

FFTW's refusal of wisdom is `WisdomImportFailed`; a file that cannot be read or written is
`WisdomFile`, which carries the path, the `std::io::ErrorKind` and the message, since
`FftwError` is `Clone` and `Eq` and cannot hold a `std::io::Error`. Every fallible function
of `rlst::fftw` thus returns `FftwError`. Import goes through FFTW's callback import as well,
so no `CString` is built.

### 4.14 Threads (feature `fftw_threads`)

- `fftw_init_threads` and `fftwf_init_threads` run once, inside the lock, before RLST's
  first FFTW call (Section 3 item 12). FFTW calls made earlier by other libraries are
  outside RLST's control; the documentation says so.
- `Planner::threads(n: NonZeroUsize)`, present only with `fftw_threads`, makes a plan for n
  threads; `Plan::threads() -> usize`, on every plan, returns the count it was made for, 1
  without the feature (DC41). Planning returns `InvalidThreadCount` when n exceeds FFTW's
  `int`, and `ThreadsUnavailable` when n > 1 and `fftw_init_threads` failed.
- `threads::make_planner_thread_safe()` is an explicit opt-in, documented with the limits
  of Section 3 item 3.
- A threaded plan is a different plan from a single-threaded one and can round
  differently. Executing threaded plans from several threads at once is allowed (execution
  is thread-safe) but oversubscribes the machine, notably under rayon; the documentation
  says so.

### 4.15 Errors

`FftwError` variants, each carrying the values involved: `InvalidLayout`, `LayoutMismatch`,
`AlignmentMismatch`, `Overlap`, `PlacementMismatch`, `InvalidKind` (for example REDFT00
with n = 1), `PreserveInputUnsupported`, `ScratchTooLarge`, `VersionTooOld`, `PlanCreationFailed` (FFTW returned NULL; the flags and layout are
attached, since FFTW gives no reason, with a `hint: Option<&'static str>` shown in the
message when RLST knows a remedy, such as `may_destroy_input()` for a preserving r2r plan
with an HC2R kind, DC34), `WisdomImportFailed`, `WisdomFile` (a wisdom file that cannot
be read or written, with its path, `std::io::ErrorKind` and message, DC38), and `InvalidThreadCount` and
`ThreadsUnavailable`, which only planning with `fftw_threads` returns (DC41). A layout of the other storage kind, an
interleaved layout given to a split planner method or a split layout to an interleaved one,
is an `InvalidLayout` naming the part axis, with a reason naming the family and the storage
the layout describes (DC28). Every planner method, those of c2c included, rejects a layout
whose storages are not those of its family in the same way: naming the part axis when one
storage is split and the family's is not, or the reverse, and naming the layout otherwise
(DC30).

## 5. Thread safety and determinism: the rules users get

1. **One lock for FFTW's global state.** Every FFTW call RLST makes, other than execution,
   goes through RLST's lock (Section 4.7). Other libraries in the process, and a second
   copy of RLST in the same build, are not covered; with `fftw_threads`,
   `make_planner_thread_safe()` extends protection to their plan creation and destruction
   only.
2. **Concurrent execution.** Executing one plan from several threads at once on different
   arrays is safe and takes no lock.
3. **Same plan, same bits.** A plan gives bit-for-bit identical results on inputs of the
   same alignment class. Two plans are the same plan when all of the following hold:
   - the same normalised layout, placement, split gap, alignment classes, input policy,
     rigor, wisdom-only setting and thread count;
   - the same FFTW build (version and configure options) and the same CPU SIMD support,
     since FFTW selects SIMD codelets at run time;
   - the same wisdom for that problem in the process at planning time. FFTW uses any
     available wisdom of equal or greater rigor (Section 3 item 4), so an `ESTIMATE` plan
     can change after a `MEASURE`, `PATIENT` or `EXHAUSTIVE` plan of the same problem, or
     after a wisdom import, anywhere earlier in the process.
4. **Verify rather than assume.** `Plan::fingerprint()` identifies the algorithm FFTW chose.
   A distributed caller that needs identical plans on every rank compares fingerprints
   across ranks (one all-reduce) and acts on the result collectively, so that ranks never
   diverge.
5. **Timed planning.** `MEASURE`, `PATIENT` and `EXHAUSTIVE` choose algorithms by timing;
   their plans can differ between runs and between processes, and so can the bits.
6. **Batch and threads.** A plan for a different batch count or layout is a different plan
   and can round differently (Section 3 item 7). For results independent of the thread count
   and of scheduling, fix the unit of work (for example one plan per box with a fixed batch,
   as nd-fmm does) and execute the same plan on each unit. Never size a plan from a
   scheduler's chunking. A threaded plan can round differently from a single-threaded one.
7. **Shared wisdom.** For plans from shared wisdom, import the same wisdom in every process
   before any planning, plan with wisdom-only, and compare fingerprints (rule 4). Wisdom
   made on a CPU with different SIMD support may not apply.

## 6. Licence notice in RLST's documentation

FFTW is licensed under the GNU General Public License, version 2 or later; MIT's Technology
Licensing Office sells non-free licences. fftw-sys is GPL-2.0-or-later with the system back
end (Section 2.4). RLST is MIT / Apache-2.0. The existing notes in the README
and in `doc::optional_integrations` are replaced by the following warning:

> **Warning: FFTW is licensed under the GNU General Public License (version 2 or later).**
> RLST's optional `fftw` feature links FFTW and the GPL-licensed `fftw-sys` bindings. RLST
> itself is licensed MIT / Apache-2.0, but a program built with this feature links
> GPL-licensed code, and distributing such a
> program is subject to the terms of the GPL unless you hold a non-free FFTW licence from
> MIT's Technology Licensing Office. Cargo enables a feature for the whole build when any crate in the dependency graph requests it,
> so check your graph (for example with `cargo tree -e features`). The feature is off by
> default. This note is not legal advice; check the licences that apply to your use.

Placements:

1. `doc::optional_integrations`, which is not feature-gated, so docs.rs always shows it;
2. the `fftw` module documentation, made visible on docs.rs by
   `[package.metadata.docs.rs]` with `features = ["fftw", "fftw_system", "fftw_threads"]`,
   so that the items of `fftw_threads` appear too (DC46), and `rustdoc-args = ["--cfg",
   "docsrs"]`, with feature-gated items marked
   `#[cfg_attr(docsrs, doc(cfg(feature = "fftw")))]` (`build.rs` skips discovery on
   docs.rs, Section 4.10);
3. the README, replacing its current note (which `CLAUDE.md` asks contributors to keep);
4. the comment on the `fftw` feature in `rlst/Cargo.toml`, for readers of the manifest
   (crates.io does not show it).

The feature stays off by default, as today.

## 7. Tests

### 7.1 The reference

- An independent O(n²) DFT, and O(n²) definitions of the eleven real-to-real kinds,
  computed in f64 with compensated summation.
- Twiddles are computed from the reduced index (j · k mod N), so the reference's own error
  does not grow with n².
- f32 transforms are compared with the f64 reference evaluated on the f32-rounded inputs.
- Tolerance: c · ε · (1 + log₂ N) in the relative L2 norm, with c fixed per family and ε of
  the precision under test. The term 1 + log₂ N keeps the tolerance positive at N = 1.
- The reference runs for logical sizes up to 2¹². That covers every grid nd-fmm uses
  (Section 2.5) and keeps a reference evaluation well under a second in CI's debug builds.
  Larger sizes, up to 2¹⁶, are checked through the identities of Section 7.3.

### 7.2 The correctness matrix, sampled

| Axis | Values |
| --- | --- |
| Family | c2c forward and backward; r2c; c2r; r2r (all eleven kinds, and different kinds on different axes); split c2c, r2c, c2r |
| Precision | f32, f64 |
| Rank and batch | 1D; 2D and 3D over all axes; 1D along each axis of 2D and 3D arrays; 2D over two axes of a 3D array; fixed-batch per-unit layouts as nd-fmm's |
| Layout | column-major, row-major, strided views (sub-arrays, steps above 1), subviews of larger arrays and nested subviews, with sentinels in the elements between them, split with the part axis outermost and innermost |
| Placement and policy | out of place preserving, out of place destroying, in place |
| Sizes | 1, 2, 3, primes (17, 97, 1009), powers of two, products of 2, 3, 5 and 7 including nd-fmm's grids (140, 216, 336, 588, 1080) |

Every family, precision, placement and layout class has a fixed set of cells. The rest of
the product is sampled with a seeded ChaCha generator (`rand_chacha`, already a
dependency), the seed printed on failure. The full product runs as an ignored test in
release mode. REDFT00 at n = 1 is an expected `InvalidKind`. There is no "interface" axis:
every plan goes through guru64.

### 7.3 Identities

Round trip (the inverse after the forward transform is N times the input, with N the
logical size); linearity; Parseval; the shift theorem; r2c against c2c on real data; split
against interleaved; a batched plan against a loop of single plans; a multidimensional r2r
against successive one-dimensional ones. Each holds to round-off, and bit for bit where the
identity is the same computation. Regressions: kinds per axis on column-major and row-major
arrays (Section 2.2 item 9); the Chebyshev port against 0.10.3's results to round-off,
while the 0.10 module exists (DC45), and at extents 1 and 0 (DC44).

### 7.4 Plans and arrays

- One `ESTIMATE` plan run on many arrays of the same layout and class gives the bits of a
  plan made for each array, in a fresh process (Section 7.6).
- Negative tests: wrong shape, wrong strides, wrong split gap, another alignment class,
  in place against out of place, overlapping spans. Each returns
  its error, and an output filled with a sentinel is left unchanged.
- `compile_fail` doctests: executing a `Destroying` plan with `&input`, including a default
  c2r plan; an in-place plan with two arrays; a split plan with a complex array.
- Defaults: a default c2r plan is `Destroying`; `preserve_input()` gives a `Preserving`
  one-axis c2r and `PreserveInputUnsupported` over two axes; an r2r plan with an HC2R kind
  is `Preserving` and leaves its input unchanged.
- The scratch cap: a strided view whose span exceeds `max_scratch` returns
  `ScratchTooLarge`; the same layout plans with `*_on`.
- An r2c layout with the wrong extent on the halved axis returns `InvalidLayout` naming
  that axis.

### 7.5 Concurrency

With `std::thread::scope` (no new dependency):

- one plan executed from many threads on distinct buffers gives bit for bit the
  sequential results;
- plans are made and destroyed from many threads while others execute;
- wisdom export and import run concurrently with planning. In the test process, wisdom is
  only exported, which reads it; import and forgetting concurrent with planning run in a
  fresh process of Section 7.6, whose later `Estimate` plans equal those of a fresh process.
  No test of the test process imports or forgets wisdom, which would change the plans of
  every later test (DC39);
- with `fftw_threads`, threaded plans against single-threaded ones to round-off, and
  threaded plans executed from several threads at once, bit for bit against sequential
  execution of the same plan.

### 7.6 Determinism and wisdom, in separate processes

FFTW's wisdom is global to the process, and `cargo test` runs tests as threads of one
process, so these tests live in a `harness = false` test binary (as `mpi_empty_ranks`
already does), which runs each case in a fresh child process:

- `ESTIMATE` plans give the same fingerprint and the same bits across two processes;
- `ESTIMATE` plans made after `ESTIMATE` planning of other, unrelated problems in the same
  process give the fingerprints and bits of a fresh process. nd-fmm's operator build plans
  many sizes in one process and relies on this; the test asserts it, and a failure is a
  design question (DC22);
- the same plan executed on fixed-batch units at different positions of a larger buffer
  gives the same bits;
- export, forget, import, then wisdom-only: the same fingerprint and the same bits;
- wisdom shared between processes (Section 5 rule 7): a process that imports the wisdom
  another exported, and plans wisdom-only with the same options, gets the fingerprints and
  bits of the exporting process; a failure is a design question (DC40);
- with `fftw_threads`, threaded `Estimate` plans give the same fingerprints and bits in two
  fresh processes; single-threaded `Estimate` plans made after threaded `Estimate` and
  `Measure` plans of the same problems give those of a fresh process, since the thread count
  never leaks and wisdom is kept per thread count; and planning after
  `make_planner_thread_safe()` gives the plans of a fresh process. A failure of either of the
  first two is a design question (DC42);
- wisdom-only with no wisdom: `PlanCreationFailed`;
- an `ESTIMATE` plan made after a `MEASURE` plan of the same problem: the fingerprints are
  reported. This documents Section 3 item 4; it asserts that fingerprints are available
  and compared, not that they differ.

### 7.7 CI

**Builds without SIMD.** The macOS runner's FFTW has no SIMD (Section 2.4), and users may
link such a build anywhere. Every test passes on it: a test that depends on SIMD, such as
one that expects `fftw_alignment_of` to tell offsets apart (Section 3 item 6) or SIMD
codelets in a plan's description (Section 3 item 7), first asks a crate-private helper
whether `fftw::version()` names a SIMD extension (Section 3 item 6), and otherwise prints
that it is skipped and returns. Such tests are not moved to a Linux-only job. RLST's own
checks, the alignment class of Section 4.6 included, are computed in Rust and are tested
on every build.

GitHub Actions, in `run-tests.yml`, with `CLAUDE.md`'s validation list updated to match:

- **F6**: both jobs gain Clippy and `cargo test` with `fftw_threads`, beside their commands
  without it, so that the threads code runs in CI from the task that adds it (DC43).
- **F8a**: ubuntu-24.04 with `fftw_system` (`libfftw3-dev`, FFTW 3.3.10), with and without
  `fftw_threads` (added in F6; F8a reviews the matrix as a whole); macOS with Homebrew's
  FFTW (3.3.11, without SIMD on the arm64 runner); the
  default build, with a check that
  `cargo tree -e normal` lists no `fftw-sys`; doctests; `cargo doc` with `--cfg docsrs`.
- **F8b**: benchmarks (`examples/fftw_bench.rs`, with `required-features`, no new
  dependency) of a batched plan against a loop, split against
  interleaved, aligned against unaligned, and r2c halving the contiguous axis against
  halving a strided one (DC2). Benchmarks are not gates.

## 8. The Chebyshev module

`chebychev.rs` is ported to the new plans in F7, when the 0.10 module is removed:

- the multidimensional DCT-I becomes one `REDFT00` plan per axis, along that axis and
  batched over the others, applied in turn, which keeps the special case of extent 1 a
  matter of one axis;
- an extent of 1 along an axis, where REDFT00 is undefined and 0.10.3 panics through
  `unwrap`, is handled explicitly: one Chebyshev point gives a constant interpolant, so
  data and coefficient are equal along that axis;
- the conversion functions return `Result<DynArray<…>, FftwError>` instead of
  unwrapping;
- they take any number of dimensions, since the per-axis steps need nothing per dimension
  (0.10.3 has `unimplemented!()` above three; with none they return a copy), and an extent
  of 0 along any axis gives an empty result of the same shape without planning, where
  0.10.3 panics (DC44);
- results are compared with 0.10.3's to round-off (Section 7.3) by a test that exists
  while the 0.10 module does, and is removed with it; the analytic and extent-1 tests stay
  as the lasting regressions (DC45). The last bits may differ, since one-dimensional plans
  in turn are not the same plans as one multidimensional plan; the CHANGELOG says so.

## 9. Work plan

Task numbers are those of draft 0.1, which Helmholtz CHR.1 cites; F8 is split into F8a and
F8b. The last column records where Helmholtz draft 0.7 cites each task; it describes that
document and decides nothing for it.

| ID | Task | Acceptance | Cited by Helmholtz 0.7 |
| --- | --- | --- | --- |
| F0 | Baseline v0.10.3 and the integration branch; CI reports the FFTW version of each runner; a macOS job brought forward from F8a | CI green on Linux and macOS; findings in `docs/fftw/F0-findings.md` | CHR.1 |
| F1 | Core: the feature rules of Section 4.10, with `fftw_source` and `fftw_mkl` removed; `FftwReal`, options, `FftwError`, `Layout` from raw dimensions with normalisation, alignment classes, the FFTW lock (shared with the 0.10 module), versions in `build.rs`, c2c plans through guru64 with the three policies, scratch and `_on` planning, checked execution on slices, fingerprints | 7.2 c2c cells on raw layouts; 7.4; the `compile_fail` doctests | CHR.3, CH4.6 |
| F2 | Layouts from RLST arrays and views (`along_axis`, `over_axes`, in place), column-major, row-major and strided | 7.2 rank, batch and layout cells for c2c | CHR.3, CH4.6; later Cecka–Darve |
| F3 | Split c2c with the part axis | 7.2 split c2c cells; 7.3 split against interleaved | CHR.3, CH4.6 ("split planes, or interleaved scratch until then") |
| F4 | Concurrency and determinism for c2c; rules 2 to 6 of Section 5 in the rustdoc of `Plan` and `Plan::fingerprint` (DC23) | 7.5 and 7.6 without the wisdom API and threads, in the harness-free binary, wisdom-only planning without wisdom and the `ESTIMATE` plan after a `MEASURE` plan included (DC24) | CHR.3, CH4.6, CH8.1 |
| F5 | r2c and c2r (in place with padding), r2r with all kinds and kinds per axis, split r2c and c2r, `Scale` | 7.2 and 7.3 for these families | "not used" |
| F6 | Wisdom; the `fftw_threads` feature, with its CI commands (DC43) | 4.13, 4.14; 7.5 and 7.6 for them: wisdom export and import during planning, export, forget and import, shared wisdom, threaded plans (DC24, DC39, DC40, DC42) | "not used" |
| F7 | Removal of the 0.10 module; the Chebyshev port (Section 8, after F5); documentation with an example for every public item; the licence warning in its four places and the docs.rs metadata; the unused build dependencies removed (DC47); CHANGELOG with the migration notes of Appendix B | doctests; the warning in the four places; the Chebyshev regressions of 7.3 | CHR.1 (the warning) |
| F8a | CI on Linux (system FFTW, with and without `fftw_threads`) and macOS; the default-build check; docs | 7.7 F8a | CHR.1 (tests on Linux and macOS) |
| F8b | Benchmarks; **release 0.11.0** | 7.7 F8b; every task above done | CHR.1 and CHR.2 (the release) |

**Transition.** From F1 to F6 the 0.10 module stays in `rlst::fftw` beside the new types,
so that `chebychev.rs` and its doctests keep working. Its planning and plan destruction go
through the new FFTW lock, so that old and new code never plan at the same time; nothing
else of it changes. F7 removes it. The `fftw_source` and `fftw_mkl` features go in F1,
since only `fftw_system` is tested and the feature rules are then final from the start.

**Briefs.** Each task starts from a brief in `docs/fftw/tasks/`, written before the task
begins, which lists its scope, the files it touches and its acceptance checks.

**Release (DC10).** 0.11.0 is released once every task, F0 to F8b, is done; there is no
partial release, and no 0.11.x release adds a missing part of this design. Helmholtz CHR.1
lists F0 to F4, F7 and F8 and leaves F5 and F6 to the maintainers; under DC10 the release it
waits for is the complete one. When that release comes is Helmholtz decision D14.

## 10. Decisions for the RLST maintainers

| ID | Decision | Status |
| --- | --- | --- |
| DF1 | The 0.10 API | **Settled**: removed in 0.11.0 without a compatibility layer; `chebychev.rs` ported; migration notes in the CHANGELOG (Appendix B) |
| DF2 | Bindings | **Settled**: fftw-sys 0.8, with RLST declaring only the version symbols (Section 4.11); no bindings of RLST's own |
| DF3 | Allocation and alignment | **Settled**: RLST's aligned containers at `PLAN_ALIGNMENT` = 64; `FFTW_ALIGNED` = 64; classes modulo 64 computed in Rust; no `fftw_malloc` |
| DF4 | A back-end-neutral FFT trait | **Settled**: not in RLST now; `Layout`, `AlignmentClass`, `Scale` and the errors stay free of FFTW types |
| DF5 | MKL | **Settled**: out of scope for now; `fftw_mkl` removed in 0.11; a later MKL back end would be an addition |
| DF6 | The source back end | **Settled**: `fftw_source` dropped; FFTW comes from the system only (Section 10.2) |
| DF7 | The lock across copies of RLST in one build | **Settled**: the limit is documented (Section 4.7) |
| DF8 | Minimum system FFTW | **Settled**: 3.3.4 to build, 3.3.6 with `fftw_threads`, 3.3.10 or later recommended; 3.3.10 (Linux) and 3.3.11 (macOS) tested |

### 10.1 Design choices

| ID | Choice | Status |
| --- | --- | --- |
| DC1 | Defaults follow FFTW: c2r destroys its input unless `preserve_input()` is given (one axis only); every other family preserves it, r2r included (Section 4.4) | accepted, changed from draft 0.2's uniform preserving default |
| DC2 | The halved axis of r2c and c2r is the last entry of `axes`, whatever the memory order; errors name the axis; the speed of halving the contiguous axis measured before it is recommended (Section 4.3) | accepted |
| DC3 | Planning on scratch by default, bounded by `max_scratch` (1 GiB by default) with `ScratchTooLarge` above it; `*_on` variants plan on the caller's arrays (Section 4.4) | accepted, with the scratch cap added |
| DC4 | One planner path, guru64, with no separate basic or advanced entry points | accepted |
| DC5 | Uniform scale factors only; no orthonormal DCT or DST variants (Section 4.9) | accepted |
| DC6 | On MKL, plans with several batch dimensions loop over the outer ones | accepted; moot under DF5 |
| DC7 | The Chebyshev port: one REDFT00 plan per axis, `Result` returns, extent 1 as the identity (Section 8) | accepted |
| DC8 | Windows not supported | accepted |
| DC9 | No new test dependencies: seeded `rand_chacha`, `std::thread::scope`, benchmarks as an example | accepted |
| DC10 | 0.11.0 only with the complete implementation (Section 9) | accepted, as changed by the maintainers |
| DC11 | Arrays and views reach FFTW through new traits `StridedRawAccess` and `StridedRawAccessMut` (storage and offset of the first element), implemented explicitly for every array and view type, subviews included; `RawAccess` keeps its stricter meaning (Section 4.3) | accepted (F1a review) |
| DC12 | The in-place planner methods exist only on a planner without an input-policy request (Section 4.4) | accepted (F1a review) |
| DC13 | An FFTW version string that does not parse counts as below the minimum (Section 4.10) | accepted (F1a review) |
| DC14 | c2c's equal shapes are checked when a layout is built from arrays; `Layout` records no shapes; F5 relaxes the halved axis (Section 4.3) | accepted (F2 brief) |
| DC15 | Execution on arrays checks each side, and pairs the axes of input and output by index when both are arrays of the same rank; a plan describes memory, not logical axes, and the documentation says so (Section 4.8) | accepted (F2 brief) |
| DC16 | Arrays whose strides fail the sufficient distinct-output rule are rejected as outputs and in place, and accepted as inputs (Section 4.3) | accepted (F2 brief) |
| DC17 | `FftBuffer::len` of an array is the number of storage elements from its first element on, 0 without storage | accepted (F2 brief) |
| DC18 | Inherent `Array::storage()` and `Array::storage_mut()` beside `data()` and `data_mut()`, added in F2 (Section 4.3) | accepted (F2 brief), changed from the brief's proposal of none |
| DC19 | For a view without elements, the storage offset may lie past the end of the slice and must not be used to index it; the traits' contract says so (Section 4.3) | accepted (F2 brief) |
| DC20 | A view whose storage offset does not fit in `usize`, which only a view without elements can have, returns `None` from `storage()` and `storage_mut()`; offsets are computed with checked arithmetic (Section 4.3) | accepted (F2a) |
| DC21 | Execution checks an array's span against the storage behind the pointer it passes FFTW, obtained through the exclusive borrow for a written buffer; the sealed side checks return that pointer (Section 4.8) | accepted (F2b) |
| DC22 | The determinism tests assert that an `ESTIMATE` plan made after `ESTIMATE` planning of unrelated problems in the same process has the fingerprint and bits of a fresh process; a failure is a design question (Section 7.6) | accepted (F4 brief) |
| DC23 | F4 states rules 2 to 6 of Section 5 in the rustdoc of `Plan` and `Plan::fingerprint`; F7's guide states all seven and links them (Section 9) | accepted (F4 brief) |
| DC24 | F4 tests the items of Sections 7.5 and 7.6 that need only F1's planner, wisdom-only planning without wisdom and the `ESTIMATE` plan after a `MEASURE` plan included; F6 tests wisdom export, import and forgetting and `fftw_threads` (Section 9) | accepted (F4 brief) |
| DC25 | `Layout` records an optional gap per side, exposed by `Layout::gaps()`; `Layout::new_split` builds raw split layouts for slices (Section 4.5) | accepted (F3 brief) |
| DC26 | The split layout functions count logical axes, the array's axes without its part axis (Section 4.5) | accepted (F3 brief) |
| DC27 | Execution finds each side's part axis as an axis of extent 2 with that side's gap as stride, accepting any choice that fits; arrays of the same rank pair by index without their part axes (Section 4.8) | accepted (F3 brief) |
| DC28 | A layout of the other storage kind given to a planner method is an `InvalidLayout` naming the part axis (Section 4.15) | accepted (F3 brief) |
| DC29 | The split layout functions require real items on both arrays (Sections 4.3 and 4.5) | accepted (F3 brief) |
| DC30 | `Layout` records the storage of each side (`Storage::Complex`, `Real`, `Split { gap }`), set by `Layout::with_storage`, from which the family follows; every planner method checks it, extending DC28 (Sections 4.2, 4.3 and 4.15) | accepted (F5 brief) |
| DC31 | The array layout functions require items with a sealed `FftElement<Real = T>`, from which they read each side's storage (Section 4.3) | accepted (F5 brief) |
| DC32 | In-place r2c and c2r take the complex array, with a contiguous halved axis and real strides twice the complex ones elsewhere; `as_reals` and `as_reals_mut` view complex slices as reals (Section 4.3) | accepted (F5 brief) |
| DC33 | Split r2c and c2r have their own layout functions, with one part axis, and no in-place plans in 0.11 (Sections 4.4 and 4.5) | accepted (F5 brief) |
| DC34 | `PlanCreationFailed` carries `hint: Option<&'static str>` (Section 4.15) | accepted (F5 brief) |
| DC35 | `Plan::logical_size() -> usize` and `Plan::scale_factor(Scale) -> T` on every plan; `Layout::logical_size` stays the product of the extents (Section 4.9) | accepted (F5 brief) |
| DC36 | In place, the distinct-element rule applies to the input side as well as the output side: the transform overwrites its input, and an in-place c2r layout can have distinct outputs and overlapping inputs (Section 4.3) | accepted (F5a) |
| DC37 | `PreserveInputUnsupported` counts every transform dimension of a c2r plan, those of extent 1 included; a layout with one transform dimension of extent above 1 can be planned along it alone instead (Section 4.4) | accepted (F5a) |
| DC38 | The wisdom functions have the types of Section 4.13, `export` and `forget` infallible; file errors are a new `FftwError::WisdomFile` with the path, `std::io::ErrorKind` and message, so that every fallible function of `rlst::fftw` returns `FftwError` (Sections 4.13 and 4.15) | accepted (F6 brief) |
| DC39 | No test in the test process imports or forgets wisdom; import and forgetting concurrent with planning are tested in a fresh process (Section 7.5) | accepted (F6 brief) |
| DC40 | The determinism tests assert that wisdom shared between processes gives the importing process the fingerprints and bits of the exporting one; a failure is a design question (Section 7.6) | accepted (F6 brief) |
| DC41 | `Planner::threads(n: NonZeroUsize)` exists only with `fftw_threads`; `Plan::threads() -> usize` on every plan, 1 without the feature; `InvalidThreadCount` above FFTW's `int`, `ThreadsUnavailable` when n > 1 and thread initialisation failed; `make_planner_thread_safe()` returns `()` (Sections 4.14 and 4.15) | accepted (F6 brief) |
| DC42 | With `fftw_threads`, the determinism tests assert that threaded `Estimate` plans agree across fresh processes and that threaded planning leaves no trace on single-threaded plans; failures are design questions (Section 7.6) | accepted (F6 brief) |
| DC43 | F6 adds Clippy and tests with `fftw_threads` to both CI jobs, beside those without it; F8a reviews the matrix (Sections 7.7 and 9) | accepted (F6 brief) |
| DC44 | The Chebyshev conversions take any number of dimensions (none: a copy), and an extent 0 along any axis gives an empty result without planning (Section 8) | accepted (F7 brief) |
| DC45 | The comparison of the Chebyshev port with 0.10.3 is a test of F7a, run while the 0.10 module exists and removed with it; the analytic and extent-1 tests are the lasting regressions (Sections 7.3 and 8) | accepted (F7 brief) |
| DC46 | docs.rs builds with `features = ["fftw", "fftw_system", "fftw_threads"]` (Section 6) | accepted (F7 brief) |
| DC47 | F7 removes the build dependencies `bindgen`, `cc`, `cmake` and `git2`, which no build script of `rlst` uses (Sections 4.10 and 9) | accepted (F7 brief) |

### 10.2 DF6: why `fftw_source` is dropped

`fftw_source` used fftw-src 0.8.0 (2023), which vendors FFTW 3.3.8 and runs
`configure --with-pic --enable-static --disable-doc` (plus `--enable-single`): no SIMD, no
threads library, static linking, and none of FFTW's fixes of 3.3.9 and 3.3.10. Measured on
one core of an unspecified Xeon virtual machine with AVX2, for one box of nd-fmm's φ step at
p = 144 (289 transforms of length 588, interleaved, out of place, `ESTIMATE`, 200
executions):

| FFTW build | Time per box |
| --- | --- |
| 3.3.8, configured as fftw-src does (no SIMD) | about 1.23 ms |
| 3.3.8 with `--enable-sse2 --enable-avx --enable-avx2` | about 0.68 ms |
| Ubuntu 24.04's 3.3.10 (`fftw-3.3.10-sse2-avx`) | about 0.59 ms |

The numbers are indicative, not measurements of record. Three options were weighed: keep
fftw-src as it is (no maintenance, but a back end about twice as slow, without threads,
that looks equivalent to the system one); maintain a source crate of RLST's own; or drop
the back end. Dropping it is the simplest and leaves no misleading configuration; the cost
is that machines where no system FFTW can be installed have no route to the `fftw` feature.

If a source build is wanted later, it can be added as a new back end. The groundwork, for
reference:

- It would be a separate crate, licensed GPL-2.0-or-later because it vendors FFTW, built
  from the 3.3.10 release tarball (FFTW's git repository lacks the generated codelets, so a
  build from git needs OCaml), offline, with SIMD for the target architecture enabled
  unconditionally (FFTW detects support at run time) and `--enable-threads`. Building
  double precision alone took 45 s without SIMD and 226 s with SIMD and threads on the
  machine above.
- kifmm-fftw-src 0.2.1 (github.com/bempp/kifmm) builds FFTW 3.3.9 with SIMD. It downloads
  the release over FTP at build time without a checksum, unpacks it into its own source
  directory, enables AVX and AVX2 only when the compile-time target features include them,
  builds no threads library and sets `links = "fftw3"`. A shared crate could replace it.

## 11. Open points for nd-fmm

This document makes no decisions for nd-fmm. The points below are observations from the
cross-check against Helmholtz draft 0.7, with the options they raise, for nd-fmm's own
review.

1. **What S9 measures.** CH0.9 uses fftw-sys directly in a spike crate, and fftw-sys's
   default feature is `source`: FFTW 3.3.8 without SIMD (Section 10.2). As written, the
   spike's timings would understate FFTW, and its bit-invariance checks would pass
   trivially, since a build without SIMD has no SIMD-across-batch codelets. fftw-sys with
   `default-features = false, features = ["system"]` measures the system FFTW; recording
   the FFTW version and its SIMD options per machine would make the results comparable.
2. **Split planes or interleaved scratch.** The φ input is zero-padded from 2P − 1 modes to
   N_φ anyway, and FFTW's SIMD codelets need interleaved storage (Section 3 item 7). On the
   machine of Section 10.2 with Ubuntu's FFTW 3.3.10, the box took about 0.5–0.6 ms
   interleaved and 0.9–1.0 ms split, and an unoptimised copy of split planes into
   interleaved scratch about 0.2 ms. Helmholtz leaves the choice to S9.
3. **Wisdom and constraint 2.** An application that embeds nd-fmm and plans transforms of
   size N_φ with `MEASURE`, or imports wisdom, on some ranks changes nd-fmm's `ESTIMATE`
   plans on those ranks only (Section 3 item 4). R15 does not mention this. One option is
   to compare plan fingerprints across ranks at operator build and act on the result on
   every rank; another is to document the precondition only.
4. **Wisdom is mentioned in two ways.** Helmholtz's F6 row says wisdom is not used, while
   Section 6.7 and R15 keep "identical wisdom with `WisdomOnly`" as an option.
5. **CHR.2's lock-file condition** ("its `Cargo.lock` without `fftw-sys`") holds only until
   a workspace member gains a feature that enables `rlst/fftw`, since Cargo resolves the
   lock file as if every feature of every workspace member were enabled. A check of the
   default build with `cargo tree -e normal` does not have this limit.
6. **The RLST source.** green-kernels' main branch depends on
   `rlst = { git = "https://github.com/linalg-rs/rlst.git" }`, the archived repository, so
   CHR.2's `[patch]` presumably redirects that source. Helmholtz §6.7 says that F0 starts
   from Codeberg's main, and CHR.2's fallback says "stay on RLST 0.10.2"; the repository is
   now github.com/tbetcke/rlst, and 0.10.3 has the same code as 0.10.2.
7. **Feature forwarding.** Helmholtz §6.7 forwards nd-fmm's `fftw` feature to "`fftw_system`
   or `fftw_source`". RLST 0.11 has only `fftw_system` (DF6, Section 4.10).
8. **D13.** RLST 0.11 has no `fftw_mkl` back end (DF5), so Helmholtz D13 refers to a back
   end that 0.11 does not offer.
9. **The release.** Under DC10, the release that CHR.1 and CHR.2 refer to contains the
   complete implementation, F5, F6 and F8b included (Section 9).
10. **DF2 and nd-fmm.** Helmholtz §6.7 says DF2 has no effect on nd-fmm unless F6's threads
    are used. DF2 also fixes which FFTW versions are supported (DF8), and DF6 removes
    `fftw_source`.
11. **cargo deny.** In a configuration that enables `fftw`, fftw-sys will likely need a
    licence clarification entry, since it declares its licence through a `license-file`.
12. **Alignment.** Per-box buffers from RLST's aligned allocation (Helmholtz's DF3 note)
    satisfy Section 4.6. Boxes that share one allocation keep class 0 only at offsets that
    are multiples of 64 bytes.

## 12. References

- FFTW 3.3 manual (`doc/*.texi` of the 3.3.8 sources; https://fftw.org/doc/): planner
  flags, the guru and 64-bit guru interfaces, new-array execute functions, wisdom,
  multi-threaded FFTW and thread safety, real-to-real kinds.
- FFTW 3.3.8 sources: `kernel/align.c`, `kernel/kalloc.c`, `simd-support/simd-common.h`,
  `dft/simd/common/genus.c`, `api/apiplan.c`; FFTW's NEWS for 3.3.4 to 3.3.11 and the
  current `simd-support/simd-common.h` (github.com/FFTW/fftw3).
- FFTW's licence: https://fftw.org/doc/License-and-Copyright.html; non-free licences:
  https://tlo.mit.edu/technologies/fftw-fastest-fourier-transform-west
- Intel oneMKL developer reference, "Using FFTW3 Wrappers" (C and Fortran).
- rlst v0.10.3 (github.com/tbetcke/rlst): `rlst/src/fftw.rs`, `rlst/src/fftw/c2c_inplace.rs`,
  `rlst/src/fftw/r2r_inplace.rs`, `rlst/src/chebychev.rs`, `rlst/src/doc/optional_integrations.rs`,
  `rlst/src/lib.rs`, `rlst/src/dense/data_container.rs`, `rlst/build.rs`, `rlst/Cargo.toml`,
  `rlst/CHANGELOG.md`, `CLAUDE.md`, `.github/workflows/run-tests.yml`.
- fftw-sys 0.8.0 (crates.io): `src/lib.rs`, `src/fftw.rs`, `LICENSE.md`; fftw-src 0.8.0:
  `build.rs`.
- The Cargo Book, "Dependency Resolution", section on features and the lock file.
- nd-fmm, `helmholtz-fmm-proposal.md`, draft 0.7: Sections 2, 3.3, 6.7, 10 (CH0.9, CH4.6,
  CHR.1–CHR.3), R15, R16, R19, D7, D13, D14; green-kernels main, `Cargo.toml`.

## Appendix A. Revision history

- **Draft 0.1** (2026-10-20): first written, against rlst 0.10.2 and Helmholtz draft 0.6.
- **Draft 0.2** (2026-10-21), after a review of draft 0.1. Major changes:

  - **Baseline** moved to github.com/tbetcke/rlst, tag v0.10.3. Codeberg is archived.
  - **No compatibility layer.** The 0.10 API is removed; `chebychev.rs`, its one in-crate
    user, is ported (Section 8). A migration table is Appendix B.
  - **The FFTW facts the design rests on** are collected and sourced in Section 3.
  - **Split complex arrays** are one buffer with a fixed gap between the real and imaginary
    parts, as FFTW's new-array execute requires (Section 3 item 1, Section 4.5).
  - **Input destruction is in the type of a plan**: plans that may overwrite their input take
    it by `&mut` (Section 4.4). Draft 0.1 took it by `&`, which is unsound.
  - **Determinism** is stated with its preconditions, including FFTW's use of any available
    wisdom in `ESTIMATE` mode, and checked through plan fingerprints (Section 5).
  - **Alignment classes** are computed in Rust modulo 64 bytes, which is sufficient on every
    FFTW build (Section 4.6).
  - **Layouts, kinds and scaling** cover r2c, c2r and r2r, with the kind of each axis stated
    explicitly. The 0.10 code applies kinds to the wrong axes of column-major arrays
    (Section 2.2 item 9).
  - **Back ends** described as they are: `fftw_source` builds FFTW 3.3.8 without SIMD or
    threads; MKL's documented limits are listed now (Sections 2.4 and 4.10).
  - **Global planner state**: every FFTW call other than execution is serialised, including
    wisdom, thread count and time limit (Section 4.7).
  - **Tests** trimmed to a CI budget; determinism and wisdom tests run in separate processes
    (Section 7).
  - **Work plan** aligned with Helmholtz draft 0.7: task numbers kept, F8 split into the
    release nd-fmm pins (F8a) and the rest (F8b) (Section 9).
  - **Notes for nd-fmm** from the cross-check against Helmholtz draft 0.7 (Section 11).

  Smaller corrections:

  - Section 1: one planner path (guru64) instead of separate basic, advanced and guru
    entry points; exclusions listed.
  - Section 2: the in-crate Chebyshev user, the public `FFTW_ALIGNED` constant and the
    feature-implication gap recorded; the back ends and the licences of fftw-sys and
    fftw-src described.
  - Section 4: `Dim` uses `isize`; kinds are RLST's own type; scaling defined per family
    and renamed `Scale`; conservative overlap check stated; scratch allocation optional for
    large layouts; wisdom export through FFTW's callback; `fftw_sprint_plan` strings freed
    with `libc::free`; feature hygiene by `compile_error!`; docs.rs build without FFTW.
  - Section 6: the warning made back-end specific, with Cargo's feature unification and the
    licence of fftw-sys; docs.rs metadata.
  - Section 7: no new test dependencies (seeded `rand_chacha`, `std::thread::scope`,
    benchmarks as an example); sentinels for "calls nothing"; f32 inputs rounded before
    the reference.
  - Section 3: facts 4, 6, 7 and 11 checked with small C programs against Ubuntu 24.04's
    FFTW 3.3.10; Section 11 item 2 adds an indicative timing of split against interleaved.
  - Sections 9 and 11: cross-references to Helmholtz draft 0.7 (CH4.6, CHR.1–CHR.3, D7, D13,
    D14); its draft 0.6 framing (a prerequisite phase ahead of H0, CH4.0) removed.

- **Draft 0.3** (2026-10-21), after the maintainers' decisions: DF2, DF3, DF4, DF7 and DF8
  accepted; DF5 settled as MKL out of scope, with `fftw_mkl` removed in 0.11; DF6 opened for
  discussion, with measurements and prior art (Section 10.2); design choices recorded as
  DC1–DC10 (Section 10.1), DC10 changed so that 0.11.0 is released only with the complete
  implementation; Section 11 reworded as open points, with no decisions for nd-fmm.

- **Draft 0.4** (2026-10-21): DF6 settled; `fftw_source` dropped, fftw-src removed from the
  dependency graph, and `fftw_system` the only back end; Section 10.2 keeps the
  measurements behind the decision and the groundwork for a later source build.

- **Draft 0.5** (2026-10-21): DC1 changed (defaults follow FFTW; c2r destroys its input
  unless `preserve_input()` is given) and DC1–DC3 accepted; DC2 with axis-naming errors and a
  benchmark of axis order; DC3 with the scratch cap `max_scratch` and `ScratchTooLarge`.

- **Draft 0.6** (2026-10-21): the removal of `fftw_source` and `fftw_mkl` moved from F7 to
  F1; the transition rule for the 0.10 module (the shared FFTW lock); slices as the
  low-level buffers from F1; the rules for raw layouts; F0 extended by CI version reports
  and a macOS job; task briefs in `docs/fftw/tasks/`.

- **Draft 0.7** (2026-10-21), from the findings of task F0 (`docs/fftw/F0-findings.md`):
  Homebrew ships FFTW 3.3.11, so Section 2.4, Section 3 item 9 (the additions of 3.3.11),
  Section 4.10, Section 7.7 and DF8 name 3.3.10 or later as recommended and 3.3.10 (Linux)
  and 3.3.11 (macOS) as tested; the baseline note gives v0.10.3's commit as 1f5ab87, not
  02eb2ec, and records that `fftw-0.11` starts from `main` at dc572e5 with version
  0.11.0-dev. Homebrew's FFTW on Apple silicon has no SIMD: Sections 2.4 and 3 item 6
  record it with the rule for reading SIMD from the version string, Section 7.7 makes
  SIMD-dependent tests skip on such builds rather than run on Linux only, and Section 4.10
  has the documentation tell Apple silicon users how to build FFTW with NEON.

- **Draft 0.8** (2026-10-21), from the review of task F1a (`docs/fftw/tasks/F1-core.md`,
  "Open questions"): `RawAccess::data` is `None` for most views and subviews do not
  implement `RawAccess`, so the layouts and buffers of F2 could not reach the strided views
  of goal 3. Arrays and views now reach FFTW through new traits `StridedRawAccess` and
  `StridedRawAccessMut`, which return the backing storage and the offset of the first
  element and cover subviews and nested views (Section 4.3, DC11); Section 4.8 states that
  FFTW touches only the elements a layout addresses, and Section 7.2 adds subviews to the
  layout cells. The in-place planner methods take no input-policy request (Section 4.4,
  DC12). An unparsable run-time version string counts as too old (Section 4.10, DC13).

- **Draft 0.9** (2026-10-21), from the open questions of the F2 brief
  (`docs/fftw/tasks/F2-layouts.md`), settled by the maintainers before the task started:
  equal c2c shapes are checked when a layout is built from arrays, since `Layout` records no
  shapes (Section 4.3, DC14); execution on arrays checks each side and pairs the axes of
  input and output arrays, and a plan describes memory, not logical axes (Section 4.8, DC15);
  the sentence that layouts from arrays satisfy the raw-layout rules by construction is
  corrected, and arrays failing the distinct-output rule are rejected as outputs and in
  place (Section 4.3, DC16); `FftBuffer::len` of an array (DC17); inherent
  `Array::storage()` and `storage_mut()` in F2 (Section 4.3, DC18); the offset of a view
  without elements (Section 4.3, DC19). Also corrected: Section 4.1 no longer lists a
  `Source` back end (DF6); Section 4.10 notes that docs.rs must provide `libclang` for the
  `bindgen` build dependency.

- **Draft 0.10** (2026-10-22), from task F2a (`docs/fftw/tasks/F2-layouts.md`, "Open
  questions" item 7), confirmed by the maintainer: an array with an extent 0 accepts any
  strides, so the offset of a view of it can overflow `usize`; such a view returns `None`
  from `storage()` and `storage_mut()` rather than panicking in debug builds or wrapping in
  release builds (Section 4.3, DC20).

- **Draft 0.11** (2026-10-22), from task F2b (`docs/fftw/tasks/F2-layouts.md`, "Open
  questions" item 8), confirmed by the maintainer: `StridedRawAccess` and
  `StridedRawAccessMut` are safe to implement, so a pointer taken from a later call than the
  one whose storage was checked need not lie in checked memory. Section 4.8 item 2 now says
  that the span is checked against the storage behind the pointer FFTW is given, which the
  check returns (DC21).

- **Draft 0.12** (2026-10-22), from the open questions of the F4 brief
  (`docs/fftw/tasks/F4-determinism.md`), settled by the maintainer before the task
  started: the determinism tests assert that `ESTIMATE` plans do not depend on earlier
  `ESTIMATE` planning of other problems, which nd-fmm relies on (Section 7.6, DC22); F4
  documents rules 2 to 6 of Section 5 on `Plan` and `Plan::fingerprint`, and F7's guide all
  seven (Section 9, DC23); the tests of Sections 7.5 and 7.6 are split between F4 and F6 by
  whether they need the wisdom API or threads (Section 9, DC24).

- **Draft 0.13** (2026-10-22), from the open questions of the F3 brief
  (`docs/fftw/tasks/F3-split.md`), settled by the maintainer before the task started, each
  with the proposed option: `Layout` records an optional gap per side, and
  `Layout::new_split` builds raw split layouts (Section 4.5, DC25); the split layout
  functions count logical axes (Section 4.5, DC26); execution finds the part axis by its
  extent and stride, and pairs arrays without their part axes (Section 4.8, DC27); a layout
  of the other storage kind is an `InvalidLayout` (Section 4.15, DC28); the split layout
  functions require real arrays (Sections 4.3 and 4.5, DC29).

- **Draft 0.14** (2026-10-22), from the open questions of the F5 brief
  (`docs/fftw/tasks/F5-real.md`), settled by the maintainer before the task started, each
  with the proposed option: `Layout` records the storage of each side, complex, real or split,
  set by `Layout::with_storage`, and every planner method checks it (Sections 4.2, 4.3 and
  4.15, DC30); the array layout functions require a sealed `FftElement` on their items, from
  which they read each side's storage (Section 4.3, DC31); in-place r2c and c2r take the
  complex array, with a contiguous halved axis, and `as_reals` and `as_reals_mut` view
  complex slices as reals (Section 4.3, DC32); split r2c and c2r have layout functions of
  their own and no in-place plans, which corrects the table of Section 4.4 (Sections 4.4 and
  4.5, DC33); `PlanCreationFailed` carries a hint (Section 4.15, DC34); `Plan::logical_size`
  and `Plan::scale_factor` exist on every plan, with their types (Section 4.9, DC35).

- **Draft 0.15** (2026-10-23), from task F5a (`docs/fftw/tasks/F5-real.md`, "Open questions"
  notes 8 and 10), confirmed by the maintainer as implemented: in place, the
  distinct-element rule applies to both sides of a layout, which an in-place c2r layout
  needs (Section 4.3, DC36), and Section 4.3 no longer says that every in-place layout has
  equal strides on both sides, which DC32 changed for r2c and c2r; a preserving c2r plan
  counts every transform dimension, those of extent 1 included (Section 4.4, DC37).

- **Draft 0.16** (2026-10-24), from the open questions of the F6 brief
  (`docs/fftw/tasks/F6-wisdom-threads.md`), settled by the maintainer before the task
  started, each with the proposed option: the wisdom functions' types, and a `WisdomFile`
  error for the file functions (Sections 4.13 and 4.15, DC38); tests that import or forget
  wisdom run only in fresh processes (Section 7.5, DC39); wisdom shared between processes is
  tested to give the same plans (Section 7.6, DC40); `Planner::threads` takes a
  `NonZeroUsize` and exists only with `fftw_threads`, `Plan::threads` exists on every plan,
  and planning gains `InvalidThreadCount` and `ThreadsUnavailable` (Sections 4.14 and 4.15,
  DC41); threaded plans in the determinism tests (Section 7.6, DC42); F6 adds `fftw_threads`
  to both CI jobs (Sections 7.7 and 9, DC43).

- **Draft 0.17** (2026-10-24), from the open questions of the F7 brief
  (`docs/fftw/tasks/F7-removal-docs.md`), settled by the maintainer before the task
  started: the Chebyshev conversions take any number of dimensions and give an empty result
  for an extent 0, where 0.10.3 panics (Section 8, DC44); the comparison with 0.10.3 is a
  test that exists while the 0.10 module does (Sections 7.3 and 8, DC45); docs.rs builds
  with `fftw_threads` as well, which corrects Section 6 (DC46); F7 removes the build
  dependencies `bindgen`, `cc`, `cmake` and `git2`, which no build script uses, and Section
  4.10 no longer says that docs.rs must provide `libclang` for them (Sections 4.10 and 9,
  DC47). The first three take the proposed options; the last takes the brief's option (b)
  instead of the proposed separate pull request into `main`.

## Appendix B. Migration from 0.10

| 0.10 | 0.11 |
| --- | --- |
| `arr.into_fft()`, `Fft`, `borrow`, `borrow_mut`, `FftMutGuard` | removed: plans hold no array; arrays stay ordinary RLST arrays |
| `fft.c2c_fft(flags, direction)` then `plan.execute()` | `Layout::over_axes_inplace(&arr, &axes)?`, `Planner::<T>::new().c2c_inplace(&layout, direction)?`, `plan.execute(&mut arr)?` |
| `fft.r2r_fft(flags, &kinds)` | `Planner::<T>::new().r2r_inplace(&layout, &kinds)?`, with `kinds[i]` for `axes[i]`. Code written for 0.10 with different kinds per axis on column-major arrays had the kinds applied in reverse order and must be checked |
| `FftwPlanFlags` | `Rigor`, `.wisdom_only()`, `.may_destroy_input()`, `.unaligned()`, `.time_limit()` |
| `C2CDirection` | `Direction` |
| `R2RPlanKind` (fftw-sys's enum) | `R2rKind` (RLST's enum) |
| `PlanInplace`, `FftPlanInplace`, `C2CPlan`, `R2RPlan`, `FftPrecision` | removed; `Plan<T, F, P>` |
| `Option` on failure | `Result<_, FftwError>` |
| planning with `Measure` overwrote the array | planning uses scratch (at most `max_scratch`), unless `*_on` is chosen |
| `FFTW_ALIGNED` = 16 | `FFTW_ALIGNED` = 64 |
| `chebychev` conversions return arrays and panic on failure | they return `Result`; extents 1 and 0 and any number of dimensions are handled |
| `fftw_system` without `fftw` linked FFTW | `fftw_system` implies `fftw`; `fftw` without `fftw_system` is a compile error |
| `fftw_mkl` | removed (DF5) |
| `fftw_source` | removed (DF6); install FFTW from the system, as the README describes |
