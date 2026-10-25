# CLAUDE.md

## Project overview

RLST (Rust Linear Solver Toolbox) is a dual-licensed (MIT OR Apache-2.0) Rust library for
high-performance dense and sparse linear algebra. It provides n-dimensional arrays,
BLAS/LAPACK-backed decompositions, Matrix Market I/O, abstract operators and iterative solvers,
plus optional MPI and FFTW support.

The repository uses the Rust 2024 edition for `rlst` and `rlst-suitesparse`, and Rust 2021 for
`rlst-proc-macro`. No Rust toolchain is pinned; use the current stable toolchain used by CI.

## Repository layout

- `Cargo.toml` — explicitly lists `rlst` and `rlst-suitesparse`; Cargo also includes the in-tree
  path dependency `rlst-proc-macro` as a workspace member. `default-members = ["rlst"]`, so
  unqualified root Cargo build/test/Clippy commands target only `rlst`. Use `-p` for another
  crate; `--workspace` also pulls in native SuiteSparse requirements.
- `rlst/` — main `rlst` library crate.
  - `src/lib.rs` — crate root, public module declarations, re-exports, feature gates, and
    crate-wide lint configuration.
  - `src/dense/` — n-dimensional array implementations, layouts, operators, GEMM, and
    LAPACK-based dense linear algebra. `src/dense/linalg/lapack/` contains factorization, solve,
    eigenvalue, generalized Schur (QZ), Sylvester, and QR/LQ implementations; shared contracts
    live in `src/traits/linalg/`.
  - `src/base_types/`, `src/simd.rs`, `src/threading.rs` — numerical base types, SIMD support,
    and backend threading controls.
  - `src/io/` — Matrix Market I/O; `src/interface/burn.rs` — optional Burn integration.
  - `src/interpolation.rs` — interpolation utilities.
  - `src/sparse/` — CSR matrices and sparse operations; distributed variants are gated by `mpi`.
  - `src/operator/` — abstract operator/vector-space abstractions and algorithms such as
    conjugate gradients.
  - `src/traits/` — shared generic array, numerical, linear-algebra, sparse, I/O, and distributed
    traits. Many are re-exported from the crate root.
  - `src/distributed_tools/` — MPI-only distributed arrays, layouts, communication, and sorting
    tools.
  - `src/fftw/`, `src/chebychev.rs` — FFTW-gated transforms and Chebyshev tools.
  - `src/doc/` — crate-level user documentation.
  - `examples/` — executable usage examples; MPI examples declare `required-features = ["mpi"]`.
  - `tests/` — integration tests for static-type macros and two feature-gated, harness-free
    executables: `mpi_empty_ranks` (`mpi`) and `fftw_determinism` (`fftw`), which runs each
    FFTW determinism case in a fresh child process of itself.
  - `build.rs` — system FFTW library discovery.
- `proc-macro/` — `rlst-proc-macro`, which supplies the array-construction, static-type, and
  tracing macros re-exported by `rlst`. Use `-p rlst-proc-macro` or
  `--manifest-path proc-macro/Cargo.toml` to target it directly.
- `rlst-suitesparse/` — UMFPACK factorization and solves for real (`f64`) and complex (`c64`) CSR
  matrices. `src/umfpack/` contains the FFI, CSR-to-CSC conversion, factorization types, errors,
  and tests; `build.rs` discovers native SuiteSparse.
- `scripts/run_mpi_tests.py` — bounded MPI regression runner; `scripts/test_run_mpi_tests.py`
  tests runner process cleanup without requiring MPI.
- `docs/fftw/` — the record of the FFTW rework of 0.11: the design as implemented
  (`rlst-fftw-design.md`), one brief per task (`tasks/`), the tasks' findings
  (`F0-findings.md`, `F8b-benchmarks.md`), and the task status (`README.md`). See "FFTW
  interface (0.11)" below.
- `.github/workflows/` — GitHub Actions CI on hosted runners. `run-tests.yml` is the
  authoritative validation matrix and the only workflow. It runs on pushes and pull requests to
  `main`, with three jobs:
  - `run-tests` (`ubuntu-24.04`, FFTW 3.3.10 with SIMD): formatting; the check that the default
    build of `rlst` has no `fftw-sys` (`cargo tree -p rlst -e normal`); Clippy on the library,
    the examples and, with FFTW, the tests; `cargo test` without features (the default build
    and its doctests), with `mpi,fftw,fftw_system,strict`, for `rlst-suitesparse`, and with
    `fftw,fftw_system,fftw_threads,strict`; the bounded MPI runner; and `cargo doc` with
    `-D warnings` with and without `fftw_threads`.
  - `run-tests-macos` (arm64 `macos-26`, Homebrew FFTW 3.3.11 without SIMD): Clippy (library
    and tests) and `cargo test` with `fftw,fftw_system,strict` and with
    `fftw,fftw_system,fftw_threads,strict`.
  - `docs` (`ubuntu-24.04`, current nightly, no native packages): the documentation as docs.rs
    builds it (`DOCS_RS=1`, `--cfg docsrs`, `-D warnings`). A failure there alone usually
    means the nightly changed.

  The two test jobs first report the FFTW that pkg-config finds: its version, the threads
  libraries beside it, and `fftw_version`.

The guides in `rlst/src/doc/` are the best existing description of intended usage and double as
doctests. Read the relevant one before changing a user-facing API, and update it when behavior
changes:

| Area | Guide |
| --- | --- |
| Arrays, macros, BLAS/LAPACK selection | `doc/getting_started.rs` |
| Layouts, views, slicing, iteration, multiplication | `doc/dense_linear_algebra.rs` |
| LU/QR/LQ/SVD/eig/QZ/LDL/Sylvester | `doc/matrix_decompositions.rs` |
| CSR matrices | `doc/sparse_matrices.rs` |
| Spaces, operators, elements, frames, solvers | `doc/abstract_linear_algebra.rs` |
| MPI | `doc/distributed_computations.rs` |
| FFTW, Burn | `doc/optional_integrations.rs` |
| FFTW design and rework record (0.11) | `docs/fftw/rlst-fftw-design.md`, `docs/fftw/tasks/` |

## Features and native dependencies

`rlst` defaults to no optional features.

- `strict` — denies warnings and unused crate dependencies. Code must pass with this feature.
- `mpi` — enables `mpi`/`mpi-sys` and distributed APIs; requires a working MPI installation.
- `fftw` — exposes FFTW and Chebyshev APIs. `fftw_system` (system FFTW) is the only back end
  and implies `fftw`; `fftw` without `fftw_system` is a `compile_error!` in `lib.rs`. System
  discovery in `build.rs` requires both `fftw3` and `fftw3f`, using `pkg-config` with a native
  macOS Homebrew fallback, and rejects a pkg-config version below 3.3.4 (the Homebrew fallback
  has no version; the crate checks `fftw_version` at the first planner call). With `DOCS_RS`
  set, `build.rs` skips discovery. Set `PKG_CONFIG_PATH` for custom installations;
  cross-compilation must use target libraries, not host Homebrew paths. The interface was
  rewritten for 0.11 (see "FFTW interface (0.11)" below); `fftw_source` and `fftw_mkl` were
  removed. docs.rs builds with `fftw`, `fftw_system` and
  `fftw_threads` and `--cfg docsrs` (`[package.metadata.docs.rs]` in `rlst/Cargo.toml`), and
  `pub mod fftw` and `pub mod chebychev` carry `doc(cfg(feature = "fftw"))` there. `rlst`'s
  only build dependency is `pkg-config`, with `fftw_system`; task F7c removed the unused
  `bindgen`, `cc`, `cmake` and `git2`, so `libclang` is needed only by `mpi-sys`, with `mpi`.
- `fftw_threads` — implies `fftw_system`; adds threaded plans (`Planner::threads`) and
  `rlst::fftw::threads`. `build.rs` also links `fftw3_threads` and `fftw3f_threads`, found in
  the directories of `fftw3` and `fftw3f` (pkg-config's `libdir` or the Homebrew fallback) and
  linked before them, and fails if they are missing; the minimum version becomes 3.3.6, at
  build time and at run time. Without it, nothing references a symbol of the threads
  libraries.
- `burn` — enables the optional Burn integration.
- `enable_tracing` — enables tracing macro support and `log`.
- `blis_threading`, `openblas_threading`, `mkl_threading` — link that backend's thread-control
  functions directly for `rlst::threading`. They are only needed for statically linked BLAS or
  targets other than Linux and macOS, where `dlsym` finds dynamically linked OpenBLAS, MKL, BLIS,
  FlexiBLAS, and Accelerate at runtime. Library checks with them need no native BLAS, but linked
  binaries require each enabled library.
- `nightly` — specialized configuration flag; follow existing code patterns before using it.

On Linux, the CI feature matrix installs `libclang-dev` (for `mpi-sys`), OpenBLAS, FFTW,
OpenMPI, and SuiteSparse. On macOS, CI installs FFTW and pkg-config with Homebrew (`brew install fftw
pkg-config`) and runs Clippy and the tests with `fftw,fftw_system,strict` and with
`fftw,fftw_system,fftw_threads,strict`; it installs no MPI,
SuiteSparse, or OpenBLAS. The normal test configuration links BLAS/LAPACK through the platform dev dependencies
(`Accelerate` on macOS; system OpenBLAS on Linux). Do not assume MPI, FFTW, SuiteSparse, or their
native libraries are available locally.

SuiteSparse discovery checks Homebrew prefixes, `/usr/local`, and `/usr` (including multiarch
library directories). Set `SUITESPARSE_PATH` to an installation prefix containing `include/` and
`lib/` to override discovery; see `rlst-suitesparse/README.md`.

FFTW linkage through `fftw_system` can impose GPL obligations. The licence warning of the
FFTW design (Section 6), "Warning: FFTW is licensed under the GNU General Public License
(version 2 or later)", stands word for word in four places: the README's "LICENSE" section,
the guide `doc/optional_integrations.rs`, the module documentation of `rlst::fftw`, and (in
shorter form) the comment on the `fftw` feature in `rlst/Cargo.toml`. Preserve all four when
changing that integration; a change to the wording goes into the design first. UMFPACK is GPL-2.0-or-later; also preserve
the SuiteSparse README and build-script licensing warnings.

## Coding conventions

- Preserve the public API unless the task explicitly calls for a breaking change. Public APIs
  should have Rustdoc, because `rlst` and key modules use `#![warn(missing_docs)]`.
- `strict` denies all warnings. Do not silence warnings with broad `allow` attributes.
- Clippy **forbids** wildcard imports (`[lints.clippy] wildcard_imports = "forbid"`). `forbid`
  cannot be locally overridden — import explicitly.
- Follow the existing trait-oriented design. Before adding an array, operator, sparse, or
  distributed implementation, identify the relevant traits in `rlst/src/traits/` and follow a
  nearby implementation.
- Keep generic numerical APIs compatible with the existing RLST number traits rather than
  specializing prematurely.
- Preserve array shape, stride, layout, ownership, and aliasing invariants. Use existing checked
  access and shape-validation paths unless an established performance-critical pattern requires
  otherwise.
- Minimize new `unsafe`. Where it is necessary, keep the unsafe block small, document its safety
  invariants, and maintain the checked public boundary. Existing unchecked array access is a
  useful pattern to follow.
- Retain feature gates (`#[cfg(feature = "...")]`) around optional native integrations and
  distributed code.
- Do not add dependencies unless necessary. Do not modify `Cargo.lock`: this is a library
  repository and the lockfile is ignored.
- Keep changes focused; avoid unrelated formatting or refactors.
- Record user-facing changes in `rlst/CHANGELOG.md`, including migration notes for breaking
  changes (see the `dot!` / `dot_blas!` entry for the expected level of detail).

## Dense array idioms

Arrays are **column-major** and use **expression-template arithmetic**. `3.0 * a + b` builds a
lazy expression type, not an array; nothing is computed until `.eval()` (which allocates) or a
fill/assignment consumes it.

```rust
let mut a = rlst::rlst_dynamic_array!(f64, [2, 3]);   // rank inferred from the shape expr
a.fill_from_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter());
assert_eq!(a.stride(), [1, 2]);                        // column-major

let doubled = (2.0 * a.r()).eval();                    // a.r() borrows; bare `a` moves
```

- `r()` / `r_mut()` take a borrowing view. Operators consume their operands, so an array used
  more than once in an expression needs `r()`. This is the single most common compile error when
  writing new RLST code.
- `DynArray::<f64, N>::from_shape(shape)` when the rank must be explicit (the macro infers it
  from the shape expression and cannot do so inside a const-generic function).
- Alignment: `rlst_dynamic_array!(f64, shape | rlst::CACHE_ALIGNED)`; also `PAGE_ALIGNED`,
  `FFTW_ALIGNED`, or a literal byte count.
- **`dot!` and `dot_blas!` are different operations.** `dot_blas!` is the BLAS-backed dense matrix
  product (and chains: `dot_blas!(a.r(), b.r(), c.r())`) and is the right default for general
  dense matrices. `dot!` performs a direct tensor contraction. `dot_into!` writes into an existing
  buffer. Picking the wrong one is a silent performance or semantics bug.
- `ZeroDimArray` (shape `[]`, one scalar) is the reduction result type and is distinct from shape
  `[0]` (an empty rank-1 array). It has no indexing — use `scalar()`, `scalar_mut()`,
  `into_scalar()`, or the `From`/`Into` conversions.
- Element type only needs `Copy + Default`, so arrays of e.g. `[usize; 3]` are valid.

Decompositions are reached through the traits in `rlst/src/traits/linalg/`, implemented in
`rlst/src/dense/linalg/lapack/`. Follow a neighbouring factorization when adding one — they share
a consistent shape (builder/decomposition type, explicit error and status reporting, support for
`f32`/`f64`/`c32`/`c64`).

There is no `rlst::prelude`: `rlst/src/prelude.rs` exists but is **not** declared in `lib.rs`. The
crate root re-exports `traits::*` and `base_types::*` directly. Don't route new exports through
that file.

## Operators, spaces, and solvers

`rlst/src/operator/` is the abstraction layer used by iterative solvers.

- A `LinearSpace` yields `Element`s; an `Element` **borrows its space**. Construct elements via
  `zero_element(&space)` or `space.copy_from(&elem)` — `Element::new` does not check the dimension
  invariant.
- `InnerProductSpace` blanket-implements `NormedSpace` and `DualSpace`.
- Unlike `Array`, element arithmetic is **eager**: `5.0 * &elem1 + &elem2` allocates.
- Solvers are builder-style with in-place iteration — see `CgIteration` in
  `operator/algorithms/conjugate_gradients.rs` for the pattern to follow (`new(op, rhs, &mut x)`,
  chained `set_tol` / `set_max_iter` / callback setters, `run` mutating `x` and returning a
  residual measure rather than a convergence flag). The solver types carry two lifetimes and take
  `x: &'a mut Element<'b, Space>`, so the mutable borrow ends with `run` and callers read the
  solution from `x` afterwards; keep that shape for new solvers. New iterative solvers belong in
  `operator/algorithms/` alongside it. `GmresIteration` in `operator/algorithms/gmres.rs` follows
  the same pattern for general (non-self-adjoint) operators and adds `set_restart`.
- `ElementFrame` is a growable collection of columns in one space, backed by a `FrameImpl`
  (`GenericFrameImpl` for arbitrary spaces, `ArrayFrameImpl` for array-backed ones), with pivoted
  QR in `operator/frame/qr.rs`. `GmresIteration` stores its Krylov basis in one, constructed as
  `ElementFrame::new(GenericFrameImpl::new(rhs.space()))` by default; `set_storage` swaps in
  another frame over the right-hand side's space, such as an `ArrayFrameImpl`-backed one. The
  two-pass Gram-Schmidt step shared by GMRES and the generic QR is `orthogonalize_twice` in
  `operator/frame/mod.rs`.

## FFTW interface (0.11)

RLST's FFTW interface was rewritten for 0.11.0, in tasks F0 to F8b on the integration branch
`fftw-0.11`, which was merged into `main` for the release and deleted. `docs/fftw/` keeps the
record: `docs/fftw/rlst-fftw-design.md` describes the interface as released, and the briefs in
`docs/fftw/tasks/` and the findings beside them say how each part was built and checked. FFTW
changes now go to `main` as ordinary pull requests.

- The decisions and design choices of the design's Section 10 are settled. A change that
  departs from one is a design change: describe it in the design document, with an entry in
  its Appendix A, rather than deviating silently in code.
- No compatibility with the 0.10 FFTW API exists. `rlst::fftw` holds only the new interface,
  and `rlst::chebychev` uses it.
- Rules for all new FFTW code:
  - every FFTW call other than plan execution goes through the FFTW lock (design,
    Section 4.7);
  - every `unsafe` FFI block has a `SAFETY:` comment that names the conditions it relies on,
    in particular FFTW's new-array conditions (design, Section 3 item 1);
  - never pass FFTW a pointer derived from a shared reference for memory that the plan may
    write (design, Section 4.4);
  - ordinary unit tests plan with `Rigor::Estimate` only: FFTW's wisdom is global to the
    process, so a `Measure` plan in one test can change another test's `Estimate` plan
    (design, Section 3 item 4). Tests that need other rigor, or that depend on the wisdom
    present, belong in the harness-free binary `rlst/tests/fftw_determinism.rs`, as a case
    run in a fresh child process (a row of its `CASES`) and a scenario comparing the
    children's outputs (a row of its `SCENARIOS`). Tests that import or forget wisdom belong
    there too: in the test process they would change every later test's plans (DC39). Only
    exporting wisdom, which reads it, is allowed in unit tests. Concurrency tests, which need
    no fresh process, are in `rlst/src/fftw/concurrency_tests.rs`;
  - no new dependencies beyond those the design names (`libc`, from F1).
- Section 11 of the design lists open points for nd-fmm, the first user of the new
  interface. They are not RLST tasks and are not decided in this repository.

## Tests and documentation

- Put unit tests in the relevant source module under `#[cfg(test)]`. Existing numerical tests
  commonly use `approx` for floating-point comparisons.
- Add regression tests for behavioral fixes and tests for new public functionality. Update the
  relevant `rlst/src/doc/` guide or an example when a user-facing feature changes.
- The crate root declares BLAS/LAPACK linkage for tests, in the `#[cfg(test)]` block at the bottom
  of `rlst/src/lib.rs`. Preserve this arrangement when adding tests that call native
  linear-algebra routines.
- MPI tests and examples require a working MPI runtime. `cargo test --features mpi` includes the
  harness-free `mpi_empty_ranks` executable but does not replace multi-rank validation.
  `python3 scripts/run_mpi_tests.py` builds with `mpi,strict`, runs each case on 1, 2, and 4 ranks
  via Open MPI's `mpiexec --oversubscribe`, and enforces per-case timeouts and process-group
  cleanup. Use this bounded runner for distributed changes rather than an unbounded launcher.
- For distributed operations, cover empty local ranks and globally empty inputs, and preserve
  collective participation and agreement when rejecting invalid metadata.

## Validation commands

Run commands from the repository root. After every code edit, and again after any corrective
edit, run all nine Clippy commands below independently so a failure does not prevent later
checks. `.github/workflows/run-tests.yml` is authoritative; keep this list synchronized with it.

```bash
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --features "openblas_threading,mkl_threading,blis_threading,strict" -- -D warnings
cargo clippy -p rlst-suitesparse -- -D warnings
cargo clippy --features "fftw,fftw_system,fftw_threads,strict" -- -D warnings
cargo clippy --tests --features "fftw,fftw_system,strict" -- -D warnings
cargo clippy --tests --features "fftw,fftw_system,fftw_threads,strict" -- -D warnings
```

The feature and SuiteSparse checks require their native dependencies. If those dependencies are
unavailable, report the check as unavailable or failed due to the environment; never represent it
as passing. Do not use `--all-features` as a substitute: optional FFTW backends and native
integrations have distinct environment and linkage requirements.

```bash
# Fast default validation
cargo fmt -- --check
cargo test

# The default build must not pull in fftw-sys; this must print nothing
cargo tree -p rlst -e normal --prefix none | grep '^fftw-sys '

# Check the full CI feature configuration (requires libclang, OpenBLAS, FFTW, OpenMPI,
# and SuiteSparse on Linux)
RUST_MIN_STACK=8388608 cargo test --features "mpi,fftw,fftw_system,strict"
cargo test -p rlst-suitesparse
cargo test --features "fftw,fftw_system,fftw_threads,strict"
python3 scripts/run_mpi_tests.py

# The documentation, with warnings denied. The guide's links into `rlst::fftw` are
# checked only in builds with `fftw_threads`
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features "mpi,fftw,fftw_system,strict"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features "fftw,fftw_system,fftw_threads,strict"

# The documentation as docs.rs builds it (CI job `docs`; current nightly, no FFTW needed)
DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc --no-deps --features "fftw,fftw_system,fftw_threads"

# The macOS CI job (Homebrew FFTW and pkg-config; BLAS/LAPACK through Accelerate)
cargo clippy --features "fftw,fftw_system,strict" -- -D warnings
cargo clippy --features "fftw,fftw_system,fftw_threads,strict" -- -D warnings
cargo test --features "fftw,fftw_system,strict"
cargo test --features "fftw,fftw_system,fftw_threads,strict"

# The FFTW determinism binary alone (part of every `cargo test` with `fftw`); its threaded
# cases and scenarios run only with `fftw_threads`
cargo test --features "fftw,fftw_system" --test fftw_determinism
cargo test --features "fftw,fftw_system,fftw_threads" --test fftw_determinism

# Work on the proc-macro crate directly
cargo test -p rlst-proc-macro
cargo clippy -p rlst-proc-macro -- -D warnings

# Test the MPI runner itself (no native MPI installation needed)
python3 scripts/test_run_mpi_tests.py
```

Use `cargo fmt --all` to apply formatting after Rust changes.

## Before submitting

1. Format changed Rust code and run the narrowest relevant tests first.
2. Run the default validation commands; run the full CI feature configuration when the environment
   supports its native dependencies, especially for feature-gated changes.
3. Confirm public API changes include Rustdoc, tests, and appropriate user documentation/examples.
4. Inspect `git diff --check` and `git status`; ensure generated artifacts and `Cargo.lock` are not
   included.
5. Use a small, descriptive commit message (for example, `fix(sparse): validate CSR row offsets`).
