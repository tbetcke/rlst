# RLST (Rust Linear Solver Toolbox) - Project Mandates & Guidelines

## Project Context
RLST is a comprehensive Rust library for dense and sparse linear algebra, designed for high-performance scientific computing and solving partial differential equations.

### Core Architecture
- **Dense Linear Algebra**: Multi-dimensional arrays (`Array`, `DynArray`, `SliceArray`) with BLAS/LAPACK integration.
- **Sparse Matrices**: CSR format with single-node and MPI-distributed support.
- **Operators**: Abstract linear operator interfaces for iterative solvers (e.g., Conjugate Gradients).
- **Distributed Computing**: MPI-based tools for distributed arrays and communicators.

## Engineering Standards

### Rust Conventions
- **Edition**: Rust 2024.
- **Idiomatic Code**: Prioritize type safety, explicit traits, and clear ownership.
- **No Wildcard Imports**: Forbidden by clippy configuration (`[lints.clippy] wildcard_imports = "forbid"`).
- **Documentation**: All public items must be documented (`#![warn(missing_docs)]`).
- **Strict Mode**: The `strict` feature enables `deny(warnings)` and `deny(unused_crate_dependencies)`.

### Linear Algebra Specifics
- **BLAS/LAPACK**: Tests must include `extern crate blas_src;` and `extern crate lapack_src;` to ensure proper linkage.
- **Memory Alignment**: Use `CACHE_ALIGNED`, `FFTW_ALIGNED`, or `PAGE_ALIGNED` as appropriate for performance.
- **Traits**: Heavy reliance on traits in `rlst::traits` for generic linear algebra operations.

### Licensing & Dependencies
- **Dual-Licensed**: MIT OR Apache-2.0.
- **FFTW (GPL)**: Be mindful that linking against FFTW (via `fftw` feature) imposes GPL requirements unless using `fftw_mkl`.

## Workflows for Gemini CLI

### Research & Strategy
- Use `codebase_investigator` for understanding complex trait hierarchies or distributed logic.
- Verify existing patterns in `rlst/src/dense` or `rlst/src/sparse` before implementing new routines.

### Execution & Validation
- **Testing**: 
  - Standard: `cargo test`
  - All Features: `cargo test --all-features`
  - MPI Tests: Require a working MPI environment. Use `mpirun -n <N> cargo test --features mpi` if applicable.
- **Linting**: Always run `cargo clippy --all-targets --all-features` and `cargo fmt --all --check` before finalizing changes.
- **Example Validation**: Check `rlst/examples` for usage patterns and ensure new features are reflected in examples if appropriate.

### Sub-Agent Delegation
- **Generalist**: Use for batch updates (e.g., fixing doc comments across multiple modules) or running heavy test suites.
- **Codebase Investigator**: Use for mapping dependencies between `proc-macro` and the main `rlst` crate.

## Foundation Mandates
- **NEVER** suppress warnings or use `unsafe` unless it's strictly necessary for performance and follows existing patterns in the codebase.
- **ALWAYS** check for and update related tests and documentation after making changes.
- **PRIORITIZE** consistency with the existing trait-based architecture.
