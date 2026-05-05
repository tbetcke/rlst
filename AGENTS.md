# AGENTS.md

## Overview

This is a Rust project built using Cargo. The goal of this repository is to provide a reliable, maintainable codebase with clear structure and predictable workflows for both humans and automated agents.

Agents interacting with this repo should follow the guidelines below to ensure consistency and safety.

---

## Project Structure

The root folder is a Rust workspace. The projects in this workspace are:

* `rlst` - The Rust Linear Solver Toolbox library (RLST)
* `proc-macro` - A library of procedural macros for RLST

The source folders are:

* `rlst/src` - The source code for RLST
* `proc-macro/src` - The proc-macro source code
* `rlst/src/lib.rs` - The RLST library root
* `Cargo.toml` - The workspace manifest
* `rlst/Cargo.toml` - The RLST library manifest
* `proc-macro/Cargo.toml` - The proc-macro manifest
* `Cargo.lock` - Locked dependencies
* `rlst/examples` - Example files for RLST
 
---

## Common Commands

### Build

```bash
cargo build
```

### Run

```bash
cargo run
```

### Test

```bash
cargo test
```

### Lint (Clippy)

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Format

```bash
cargo fmt --all
```

### Check without building

```bash
cargo check
```

---

## Coding Guidelines

* Follow Rust idioms and ownership principles.
* Prefer explicit error handling using `Result` and `thiserror` or `anyhow`.
* Avoid `unwrap()` in production code.
* Use meaningful variable and function names.
* Keep functions small and focused.

---

## Dependencies

* Minimize external dependencies.
* Prefer well-maintained crates.
* Avoid adding dependencies without clear justification.

---

## Testing

* Unit tests go inside modules using `#[cfg(test)]`.
* All new features must include tests.
* Ensure all tests pass before committing.

---

## Formatting & Linting

* Code must be formatted with `cargo fmt`.
* Clippy warnings are treated as errors.
* CI will fail if formatting or linting checks fail.

---

## CI Expectations

Before submitting changes, ensure:

* `cargo clippy -- -D warnings` passes
* `cargo clippy --examples -- -D warnings` passes
* `cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings` passes
* `cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings` passes
* `cargo test --features "mpi,fftw,fftw_system,strict` passes
* `cargo doc --no-deps --features "mpi,fftw,fftw_system,strict"` passes

---

## Safe Modification Rules for Agents

* Do not modify `Cargo.lock` unless dependencies change.
* Do not introduce breaking API changes without updating documentation.
* Avoid large refactors unless explicitly requested.
* Preserve public API stability unless instructed otherwise.

---

## Documentation

* Public APIs must include Rustdoc comments (`///`).
* Include examples in documentation when possible.
* Keep README.md in sync with major changes.

---

## Commit Guidelines

* Use clear, descriptive commit messages.
* Keep commits small and focused.
* Reference issues where applicable.

Example:

```
feat(parser): add support for JSON input
fix(api): handle missing headers gracefully
```

---

## When in Doubt

* Prefer clarity over cleverness.
* Ask for clarification rather than making assumptions.
* Make minimal, reversible changes.

---
