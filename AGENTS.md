# AGENTS.md

## Project overview

RLST (Rust Linear Solver Toolbox) is a dual-licensed (MIT OR Apache-2.0) Rust library for high-performance dense and sparse linear algebra. It provides n-dimensional arrays, BLAS/LAPACK-backed decompositions, Matrix Market I/O, abstract operators and iterative solvers, plus optional MPI and FFTW support.

The repository uses the Rust 2024 edition for `rlst` and `rlst-suitesparse`, and Rust 2021 for `rlst-proc-macro`. No Rust toolchain is pinned; use the current stable toolchain used by CI.

## Repository layout

- `Cargo.toml` — explicitly lists `rlst` and `rlst-suitesparse`; Cargo also includes the in-tree path dependency `rlst-proc-macro` as a workspace member. `default-members = ["rlst"]`, so unqualified root Cargo build/test/Clippy commands target only `rlst`. Use `-p` for another crate; `--workspace` also pulls in native SuiteSparse requirements.
- `rlst/` — main `rlst` library crate.
  - `src/lib.rs` — crate root, public module declarations, re-exports, feature gates, and crate-wide lint configuration.
  - `src/dense/` — n-dimensional array implementations, layouts, operators, GEMM, and LAPACK-based dense linear algebra. `src/dense/linalg/lapack/` contains factorization, solve, eigenvalue, generalized Schur (QZ), Sylvester, and QR/LQ implementations; shared contracts live in `src/traits/linalg/`.
  - `src/base_types/`, `src/simd.rs`, `src/threading.rs` — numerical base types, SIMD support, and backend threading controls.
  - `src/io/` — Matrix Market I/O; `src/interface/burn.rs` — optional Burn integration.
  - `src/interpolation.rs` — interpolation utilities.
  - `src/sparse/` — CSR matrices and sparse operations; distributed variants are gated by `mpi`.
  - `src/operator/` — abstract operator/vector-space abstractions and algorithms such as conjugate gradients.
  - `src/traits/` — shared generic array, numerical, linear-algebra, sparse, I/O, and distributed traits. Many are re-exported from the crate root.
  - `src/distributed_tools/` — MPI-only distributed arrays, layouts, communication, and sorting tools.
  - `src/fftw/`, `src/chebychev.rs` — FFTW-gated transforms and Chebyshev tools.
  - `src/doc/` — crate-level user documentation.
  - `examples/` — executable usage examples; MPI examples declare `required-features = ["mpi"]`.
  - `tests/` — integration tests for static-type macros and a feature-gated, harness-free `mpi_empty_ranks` executable.
  - `build.rs` — system FFTW library discovery.
- `proc-macro/` — `rlst-proc-macro`, which supplies the array-construction, static-type, and tracing macros re-exported by `rlst`. Use `-p rlst-proc-macro` or `--manifest-path proc-macro/Cargo.toml` to target it directly.
- `rlst-suitesparse/` — UMFPACK factorization and solves for real (`f64`) and complex (`c64`) CSR matrices. `src/umfpack/` contains the FFI, CSR-to-CSC conversion, factorization types, errors, and tests; `build.rs` discovers native SuiteSparse.
- `scripts/run_mpi_tests.py` — bounded MPI regression runner; `scripts/test_run_mpi_tests.py` tests runner process cleanup without requiring MPI.
- `.forgejo/workflows/` — CI configuration. `run-tests.yml` is the authoritative validation matrix. Separate scheduled/manual workflows run `cargo templated-examples` and `cargo upgrades`.

## Features and native dependencies

`rlst` defaults to no optional features.

- `strict` — denies warnings and unused crate dependencies. Code must pass with this feature.
- `mpi` — enables `mpi`/`mpi-sys` and distributed APIs; requires a working MPI installation.
- `fftw` — exposes FFTW and Chebyshev APIs. Enable it together with one backend: `fftw_system` (system FFTW), `fftw_source` (build FFTW), or `fftw_mkl` (Intel MKL). System discovery requires both `fftw3` and `fftw3f`, using `pkg-config` with a native macOS Homebrew fallback. Set `PKG_CONFIG_PATH` for custom installations; cross-compilation must use target libraries, not host Homebrew paths.
- `burn` — enables the optional Burn integration.
- `enable_tracing` — enables tracing macro support and `log`.
- `blis_threading`, `openblas_threading`, `mkl_threading`, and `nightly` — specialized configuration flags; follow existing code patterns before using them.

On Linux, the CI feature matrix installs `libclang-dev`, CMake, OpenBLAS, FFTW, OpenMPI, and SuiteSparse. The normal test configuration links BLAS/LAPACK through the platform dev dependencies (`Accelerate` on macOS; system OpenBLAS on Linux). Do not assume MPI, FFTW, SuiteSparse, or their native libraries are available locally.

SuiteSparse discovery checks Homebrew prefixes, `/usr/local`, and `/usr` (including multiarch library directories). Set `SUITESPARSE_PATH` to an installation prefix containing `include/` and `lib/` to override discovery; see `rlst-suitesparse/README.md`.

FFTW linkage can impose GPL obligations (`fftw_system` and `fftw_source`); preserve the licensing notice in the README when changing that integration. UMFPACK is GPL-2.0-or-later; also preserve the SuiteSparse README and build-script licensing warnings.

## Coding conventions

- Preserve the public API unless the task explicitly calls for a breaking change. Public APIs should have Rustdoc, because `rlst` and key modules use `#![warn(missing_docs)]`.
- `strict` denies all warnings. Do not silence warnings with broad `allow` attributes.
- Clippy forbids wildcard imports. Use explicit imports.
- Follow the existing trait-oriented design. Before adding an array, operator, sparse, or distributed implementation, identify the relevant traits in `rlst/src/traits/` and follow a nearby implementation.
- Keep generic numerical APIs compatible with the existing RLST number traits rather than specializing prematurely.
- Preserve array shape, stride, layout, ownership, and aliasing invariants. Use existing checked access and shape-validation paths unless an established performance-critical pattern requires otherwise.
- Minimize new `unsafe`. Where it is necessary, keep the unsafe block small, document its safety invariants, and maintain the checked public boundary. Existing unchecked array access is a useful pattern to follow.
- Retain feature gates (`#[cfg(feature = "...")]`) around optional native integrations and distributed code.
- Do not add dependencies unless necessary. Do not modify `Cargo.lock`: this is a library repository and the lockfile is ignored.
- Keep changes focused; avoid unrelated formatting or refactors.

## Tests and documentation

- Put unit tests in the relevant source module under `#[cfg(test)]`. Existing numerical tests commonly use `approx` for floating-point comparisons.
- Add regression tests for behavioral fixes and tests for new public functionality. Update the relevant `rlst/src/doc/` guide or an example when a user-facing feature changes.
- The crate root declares BLAS/LAPACK linkage for tests. Preserve this arrangement when adding tests that call native linear-algebra routines.
- MPI tests and examples require a working MPI runtime. `cargo test --features mpi` includes the harness-free `mpi_empty_ranks` executable but does not replace multi-rank validation. `python3 scripts/run_mpi_tests.py` builds with `mpi,strict`, runs each case on 1, 2, and 4 ranks via Open MPI's `mpiexec --oversubscribe`, and enforces per-case timeouts and process-group cleanup. Use this bounded runner for distributed changes rather than an unbounded launcher.
- For distributed operations, cover empty local ranks and globally empty inputs, and preserve collective participation and agreement when rejecting invalid metadata.

## Validation commands

Run commands from the repository root. After every code edit, and again after any corrective edit, run all five Clippy commands below independently so a failure does not prevent later checks. `.forgejo/workflows/run-tests.yml` is authoritative; keep this list synchronized with it.

```bash
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy -p rlst-suitesparse -- -D warnings
```

The feature and SuiteSparse checks require their native dependencies. If those dependencies are unavailable, report the check as unavailable or failed due to the environment; never represent it as passing. Do not use `--all-features` as a substitute: optional FFTW backends and native integrations have distinct environment and linkage requirements.

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
2. Run the default validation commands; run the full CI feature configuration when the environment supports its native dependencies, especially for feature-gated changes.
3. Confirm public API changes include Rustdoc, tests, and appropriate user documentation/examples.
4. Inspect `git diff --check` and `git status`; ensure generated artifacts and `Cargo.lock` are not included.
5. Use a small, descriptive commit message (for example, `fix(sparse): validate CSR row offsets`).
