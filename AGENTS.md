# AGENTS.md

## Project overview

RLST (Rust Linear Solver Toolbox) is a dual-licensed (MIT OR Apache-2.0) Rust library for high-performance dense and sparse linear algebra. It provides n-dimensional arrays, BLAS/LAPACK-backed decompositions, Matrix Market I/O, abstract operators and iterative solvers, plus optional MPI and FFTW support.

The repository uses the Rust 2024 edition for the main crate and Rust 2021 for its procedural-macro crate. No Rust toolchain is pinned; use the current stable toolchain used by CI.

## Repository layout

- `Cargo.toml` — workspace manifest for `rlst` and `rlst-suitesparse`.
- `rlst/` — main `rlst` library crate.
  - `src/lib.rs` — crate root, public module declarations, re-exports, feature gates, and crate-wide lint configuration.
  - `src/dense/` — n-dimensional array implementations, layouts, GEMM, and LAPACK-based dense linear algebra.
  - `src/sparse/` — CSR matrices and sparse operations; distributed variants are gated by `mpi`.
  - `src/operator/` — abstract operator/vector-space abstractions and algorithms such as conjugate gradients.
  - `src/traits/` — shared generic array, numerical, linear-algebra, sparse, I/O, and distributed traits. Many are re-exported from the crate root.
  - `src/distributed_tools/` — MPI-only distributed arrays, layouts, communication, and sorting tools.
  - `src/fftw/`, `src/chebychev.rs` — FFTW-gated transforms and Chebyshev tools.
  - `src/doc/` — crate-level user documentation.
  - `examples/` — executable usage examples; MPI examples declare `required-features = ["mpi"]`.
- `proc-macro/` — `rlst-proc-macro`, a path dependency that supplies the array-construction and tracing macros re-exported by `rlst`. It is not a workspace member, so invoke Cargo for it with `--manifest-path proc-macro/Cargo.toml` when working on it directly.
- `rlst-suitesparse/` — bindings between RLST sparse matrices and native SuiteSparse libraries. Its initial UMFPACK support requires an installed SuiteSparse distribution; set `SUITESPARSE_PATH` to an installation prefix to override standard lookup paths.
- `.forgejo/workflows/` — CI configuration. `run-tests.yml` is the authoritative validation matrix.

## Features and native dependencies

`rlst` defaults to no optional features.

- `strict` — denies warnings and unused crate dependencies. Code must pass with this feature.
- `mpi` — enables `mpi`/`mpi-sys` and distributed APIs; requires a working MPI installation.
- `fftw` — enables FFTW support. Select exactly one relevant backend feature where needed: `fftw_system` (system FFTW), `fftw_source` (build FFTW), or `fftw_mkl` (Intel MKL).
- `burn` — enables the optional Burn integration.
- `enable_tracing` — enables tracing macro support and `log`.
- `blis_threading`, `openblas_threading`, `mkl_threading`, and `nightly` — specialized configuration flags; follow existing code patterns before using them.

On Linux, the CI feature matrix installs `libclang-dev`, CMake, OpenBLAS, FFTW, OpenMPI, and SuiteSparse. The normal test configuration links BLAS/LAPACK through the platform dev dependencies (`Accelerate` on macOS; system OpenBLAS on Linux). Do not assume MPI, FFTW, SuiteSparse, or their native libraries are available locally.

FFTW linkage can impose GPL obligations (`fftw_system` and `fftw_source`); preserve the licensing notice in the README when changing that integration.

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
- MPI tests and examples may require `mpirun` and a multi-process-capable environment; do not run them as an implicit part of a default test command.

## Validation commands

Run commands from the repository root.

```bash
# Fast default validation
cargo fmt -- --check
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
cargo test

# Check the full CI feature configuration (requires libclang, CMake, OpenBLAS,
# FFTW, OpenMPI, and SuiteSparse on Linux)
cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings
RUST_MIN_STACK=8388608 cargo test --features "mpi,fftw,fftw_system,strict"
cargo doc --no-deps --features "mpi,fftw,fftw_system,strict"

# Work on the proc-macro crate directly
cargo test --manifest-path proc-macro/Cargo.toml
cargo clippy --manifest-path proc-macro/Cargo.toml -- -D warnings
```

Use `cargo fmt --all` to apply formatting after Rust changes. Do not use `--all-features` as a substitute for the CI command: optional FFTW backends and native integrations have distinct environment and linkage requirements.

## Before submitting

1. Format changed Rust code and run the narrowest relevant tests first.
2. Run the default validation commands; run the full CI feature configuration when the environment supports its native dependencies, especially for feature-gated changes.
3. Confirm public API changes include Rustdoc, tests, and appropriate user documentation/examples.
4. Inspect `git diff --check` and `git status`; ensure generated artifacts and `Cargo.lock` are not included.
5. Use a small, descriptive commit message (for example, `fix(sparse): validate CSR row offsets`).
