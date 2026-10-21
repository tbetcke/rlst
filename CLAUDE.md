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
  - `tests/` — integration tests for static-type macros and a feature-gated, harness-free
    `mpi_empty_ranks` executable.
  - `build.rs` — system FFTW library discovery.
- `proc-macro/` — `rlst-proc-macro`, which supplies the array-construction, static-type, and
  tracing macros re-exported by `rlst`. Use `-p rlst-proc-macro` or
  `--manifest-path proc-macro/Cargo.toml` to target it directly.
- `rlst-suitesparse/` — UMFPACK factorization and solves for real (`f64`) and complex (`c64`) CSR
  matrices. `src/umfpack/` contains the FFI, CSR-to-CSC conversion, factorization types, errors,
  and tests; `build.rs` discovers native SuiteSparse.
- `scripts/run_mpi_tests.py` — bounded MPI regression runner; `scripts/test_run_mpi_tests.py`
  tests runner process cleanup without requiring MPI.
- `.github/workflows/` — GitHub Actions CI on hosted `ubuntu-24.04` runners. `run-tests.yml` is
  the authoritative validation matrix and the only workflow.

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

## Features and native dependencies

`rlst` defaults to no optional features.

- `strict` — denies warnings and unused crate dependencies. Code must pass with this feature.
- `mpi` — enables `mpi`/`mpi-sys` and distributed APIs; requires a working MPI installation.
- `fftw` — exposes FFTW and Chebyshev APIs. Enable it together with one backend: `fftw_system`
  (system FFTW), `fftw_source` (build FFTW), or `fftw_mkl` (Intel MKL). System discovery requires
  both `fftw3` and `fftw3f`, using `pkg-config` with a native macOS Homebrew fallback. Set
  `PKG_CONFIG_PATH` for custom installations; cross-compilation must use target libraries, not
  host Homebrew paths.
- `burn` — enables the optional Burn integration.
- `enable_tracing` — enables tracing macro support and `log`.
- `blis_threading`, `openblas_threading`, `mkl_threading` — link that backend's thread-control
  functions directly for `rlst::threading`. They are only needed for statically linked BLAS or
  targets other than Linux and macOS, where `dlsym` finds dynamically linked OpenBLAS, MKL, BLIS,
  FlexiBLAS, and Accelerate at runtime. Library checks with them need no native BLAS, but linked
  binaries require each enabled library.
- `nightly` — specialized configuration flag; follow existing code patterns before using it.

On Linux, the CI feature matrix installs `libclang-dev`, CMake, OpenBLAS, FFTW, OpenMPI, and
SuiteSparse. The normal test configuration links BLAS/LAPACK through the platform dev dependencies
(`Accelerate` on macOS; system OpenBLAS on Linux). Do not assume MPI, FFTW, SuiteSparse, or their
native libraries are available locally.

SuiteSparse discovery checks Homebrew prefixes, `/usr/local`, and `/usr` (including multiarch
library directories). Set `SUITESPARSE_PATH` to an installation prefix containing `include/` and
`lib/` to override discovery; see `rlst-suitesparse/README.md`.

FFTW linkage can impose GPL obligations (`fftw_system` and `fftw_source`); preserve the licensing
notice in the README when changing that integration. UMFPACK is GPL-2.0-or-later; also preserve
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
edit, run all six Clippy commands below independently so a failure does not prevent later
checks. `.github/workflows/run-tests.yml` is authoritative; keep this list synchronized with it.

```bash
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --features "openblas_threading,mkl_threading,blis_threading,strict" -- -D warnings
cargo clippy -p rlst-suitesparse -- -D warnings
```

The feature and SuiteSparse checks require their native dependencies. If those dependencies are
unavailable, report the check as unavailable or failed due to the environment; never represent it
as passing. Do not use `--all-features` as a substitute: optional FFTW backends and native
integrations have distinct environment and linkage requirements.

```bash
# Fast default validation
cargo fmt -- --check
cargo test

# Check the full CI feature configuration (requires libclang, CMake, OpenBLAS,
# FFTW, OpenMPI, and SuiteSparse on Linux)
RUST_MIN_STACK=8388608 cargo test --features "mpi,fftw,fftw_system,strict"
cargo test -p rlst-suitesparse
python3 scripts/run_mpi_tests.py
cargo doc --no-deps --features "mpi,fftw,fftw_system,strict"

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
