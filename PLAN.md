# Implementation Plan: Address `REVIEW.md`

## Context

The review identifies safe-API soundness gaps, inconsistent sparse semantics, solver and Matrix Market failures, a broken exported macro, workspace friction, and focused test gaps. The implementation should close those findings while preserving existing APIs and feature gates where practical.

Per project direction, **checked arithmetic for integer-overflow prevention is out of scope for now**. Therefore the reviewed `ArraySubView` overflow item (P0-003), its release-overflow regression, and defensive checked index reconstruction are explicitly deferred. Ordinary bounds, shape, and structural validation remain in scope.

## Approach

1. Establish the invariants assumed by unsafe dense-array consumers: safe strided arrays must map every logical element to valid, distinct storage, multislices must validate their fixed dimensions, FFT plans must not expose replaceable storage, and mutable iteration must require an audited disjoint-access capability.
2. Make CSR canonical at construction (row-major sorted, duplicate-combined, explicit zeros removed), then rely on that invariant consistently in conversion and sparse merges. Validate all matrix and elementwise shapes before mutation/iteration.
3. Correct numerical and public API behavior in CG, Matrix Market I/O, and the static-type macro.
4. Make a deliberate breaking sparse API correction: retain ordinary numerical traits only for zero-preserving operations, replace non-zero-preserving unary operations with an explicit `map_stored` API, and remove ordinary sparse matrix/matrix division whose implicit-zero semantics cannot be represented sparsely.
5. Add targeted regressions alongside each fix, then complete broader numerical/native coverage and workspace configuration.

## Files to modify

Critical paths (exact test placement may remain in each module's existing `#[cfg(test)]` section):

- `rlst/src/dense/strided_base_array.rs` — stride-span and non-overlap validation.
- `rlst/src/dense/array/multislice.rs` — fixed-axis/index validation and mutable write-through tests.
- `rlst/src/traits/accessors.rs`, `rlst/src/dense/array/iterators.rs`, and audited mutable array/view implementations under `rlst/src/dense/` — unsafe disjoint mutable-iteration capability.
- `rlst/src/fftw.rs` and plan-family tests in `rlst/src/fftw/{c2c_inplace,r2r_inplace}.rs` — non-replaceable plan storage access and reference-transform tests.
- `rlst/src/sparse/csr_mat.rs`, `rlst/src/sparse/tools.rs`, `rlst/src/sparse/binary_operator.rs`, and `rlst/src/sparse/mat_operations.rs` — canonical CSR, eager shape checks, multi-RHS checks, empty dimensions, the new `map_stored` API, removal of non-zero-preserving unary traits and matrix/matrix division, migration docs, and tests.
- `rlst/src/operator/algorithms/conjugate_gradients.rs` — exact-solution, zero-RHS, and breakdown handling.
- `rlst/src/io/matrix_market.rs` and `rlst/src/traits/io.rs` — fallible parsing and real/complex record handling, with fixtures/tests.
- `proc-macro/src/dense.rs` plus a main-crate doctest or integration test — current `rlst_static_type!` expansion.
- `rlst/src/dense/array.rs` and `rlst/src/dense/linalg/lapack/pseudo_inverse.rs` (or the owning decomposition test module) — targeted accumulation and pseudoinverse regressions.
- `rlst-suitesparse/src/umfpack/tests.rs` — empty/nonsquare/singular behavior.
- `Cargo.toml` and, if required, `.forgejo/workflows/run-tests.yml` — default workspace member while retaining explicit native checks.
- Relevant user-facing Rustdoc under the sparse/I/O APIs (and `rlst/src/doc/` if the contract change needs guide-level treatment).

## Reuse

- Reuse `normalize_aij` in `rlst/src/sparse/tools.rs` as the single sorted/unique/zero-eliding representation path; adapt raw CSR rows to AIJ for canonicalization rather than creating a second merge algorithm.
- Reuse the dimension checks already used by CSR vector application in `rlst/src/sparse/csr_mat.rs` for matrix RHS validation.
- Follow the adjacent checked single-axis slice constructor in `rlst/src/dense/array/slice.rs` for multislice assertions.
- Keep the existing `MmIdentifier::MMTYPE` mapping in `rlst/src/traits/io.rs` as the declared scalar-format contract, extending parsing through type-aware trait behavior as needed for real and complex records.
- Reuse existing `RlstError::IoError`/`RlstResult` conventions and convert file, line, token, dimension, count, and index failures instead of unwrapping.
- Use existing array constructors and dense application routines as independent references for sparse and solver tests.

## Steps

- [ ] **Validate safe strided construction.** Compute the required storage span implied by shape/stride, reject layouts that cannot address every element, and reject overlapping logical locations for mutable-capable storage. Keep ordinary contiguous layouts and padded non-overlapping layouts valid; ensure dynamic constructors allocate the stride-implied span rather than only `product(shape)`. Do not add checked-overflow arithmetic as part of this work.
- [ ] **Validate multislice inputs before mask construction.** Assert every axis is `< ADIM`, axes are unique, and each fixed index is below its source extent (including zero-length axes). Add immutable and mutable regressions for invalid axes/indices and parent write-through.
- [ ] **Introduce an unsafe-to-implement disjoint mutable 1D access capability.** Document stability and pairwise-disjointness requirements, make `ArrayDefaultIteratorMut` depend on it, and implement it only for audited owning arrays/layouts/views whose mappings preserve uniqueness. Keep ordinary one-at-a-time mutable access separate from the iterator capability. Add compile/API coverage and Miri tests for representative owners and views.
- [ ] **Pin FFTW plan storage identity.** Replace `Fft::borrow_mut`'s whole-array write guard with a restricted mutable element/slice/view guard that cannot assign a new array or alter layout/allocation while plans exist. Preserve locking and read access. Add compile-fail/API coverage for whole-array replacement and repeated mutate/execute tests; compare nonsymmetric transforms with direct DFT/DCT references so direction conventions are independently tested.
- [ ] **Canonicalize raw CSR construction.** Validate raw CSR structure, convert rows to coordinates, normalize via `normalize_aij`, and store sorted unique entries with duplicates summed and zeros removed. Document this invariant. Ensure `from_aij` accepts empty `[0,n]`, `[n,0]`, and `[0,0]` shapes by checking coordinate maxima only when entries exist. Test raw/from-AIJ/iterator construction, duplicate conversion/application equivalence, and deliberately unsorted input.
- [ ] **Validate sparse operations eagerly.** Before CSR matrix-RHS iteration, require input rows, output rows, and RHS column counts to match, and perform no output mutation on failure. Require equal shapes before add/subtract/multiply/divide construction in both operand orders. Test both RHS column mismatch directions, zero-column row mismatch, mismatched empty/nonempty sparse matrices, and valid multi-RHS `alpha*A*x + beta*y` against a dense reference with nonzero initial output.
- [ ] **Replace misleading sparse unary/division APIs with explicit stored-entry mapping.** Add a public, documented `map_stored` API on the sparse operation surface that applies a closure only to stored entries, preserves the matrix shape, and omits mapped values equal to the scalar default/zero. Remove ordinary numerical trait implementations for non-zero-preserving operations such as `Exp`, `Cos`, and `Recip`; callers must migrate from `sparse.exp()`-style calls to `sparse.map_stored(|value| value.exp())` when stored-entry semantics are intended. Remove sparse matrix/matrix `Div`, because absent/absent and present/absent division generally produces `NaN`/infinity or dense results; do not silently redefine `/` as stored-coordinate division. Retain ordinary traits only where the operation preserves implicit zero. Add compile-fail/API tests for removed traits where supported, plus runtime tests showing that `map_stored` treats implicit and explicitly stored zeros consistently with the canonical CSR zero-elision contract. Document these as intentional breaking changes and provide migration examples.
- [ ] **Fix conjugate gradients termination and breakdown behavior.** Check exact zero residual before normalization; define zero-RHS convergence using absolute residual; reject/terminate cleanly on a zero or invalid search-direction denominator before updating the solution. Test zero RHS with zero and nonzero initial guesses, finite output, callback/iteration behavior, and independently computed true residuals for identity and non-diagonal SPD systems.
- [ ] **Make Matrix Market reads fully fallible and format-aware.** Replace every file/line/token/dimension unwrap with contextual `RlstError`; enforce header format/data type compatibility with `MmIdentifier`; parse complex records as separate real/imaginary fields; validate dimensions, exact record counts, 1-based coordinate indices, and rectangular array column-major ordering. Add hand-written real/complex array and coordinate fixtures plus missing, malformed, truncated, excess, zero-index, and out-of-range cases.
- [ ] **Repair `rlst_static_type!`.** Emit the current two-parameter `Array<BaseArray<ArrayContainer<T, N>, NDIM>, NDIM>` shape, preserve useful malformed-input diagnostics, and add a main-crate compile-pass doctest/integration test that initializes the generated type from the matching static constructor; add focused compile-fail macro cases where supported.
- [ ] **Fill remaining targeted numerical coverage.** Change dense `apply` tests to use distinct nonzero initial outputs and verify beta accumulation; add rank-deficient, cutoff-boundary, rank-zero, and all-zero pseudoinverse cases; add SuiteSparse empty/nonsquare rejection and singular-warning tests when the native library is available.
- [ ] **Reduce native workspace friction.** Set `default-members = ["rlst"]` in root `Cargo.toml`; keep SuiteSparse explicitly selected by native-enabled CI so workspace-default commands no longer require it.
- [ ] **Update Rustdoc/user documentation** for canonical CSR input; the breaking migration from non-zero-preserving unary traits to `map_stored`; removal of sparse matrix/matrix division; zero-RHS CG convergence; Matrix Market support/errors; and FFT mutable access changes.

## Verification

Run narrow tests first, then from the repository root:

```bash
cargo fmt --all
cargo fmt -- --check
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
cargo test
cargo test --manifest-path proc-macro/Cargo.toml
cargo clippy --manifest-path proc-macro/Cargo.toml -- -D warnings
cargo doc --no-deps
```

Where native dependencies are available:

```bash
cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings
cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings
RUST_MIN_STACK=8388608 cargo test --features "mpi,fftw,fftw_system,strict"
python3 scripts/run_mpi_tests.py
cargo test -p rlst-suitesparse
cargo doc --no-deps --features "mpi,fftw,fftw_system,strict"
```

Additionally:

- Run the dense mutable-iterator/strided/view suite under Miri.
- Run FFTW ownership/execution regressions under ASan or an equivalent native memory checker.
- Confirm unqualified root commands select only `rlst`, while explicit workspace/SuiteSparse CI still runs.
- Run `git diff --check` and inspect `git status`; do not include `Cargo.lock` or generated artifacts.
- No release-mode integer-overflow/subview regression is required in this scope because P0-003 is intentionally deferred.
