# LAPACK QZ, Sylvester, and LQ extensions

## Context
Extend RLST with low-level LAPACK wrappers and trait-based array convenience APIs for generalized Schur/QZ decomposition, Sylvester equations, and LQ decomposition. Support f32, f64, c32, and c64, following existing dense LAPACK conventions. Planning only; implementation awaits approval.

## Approach
- QZ: wrap GGES, initially without eigenvalue sorting; expose A = Q S Z^H and B = Q T Z^H, retaining homogeneous eigenvalues (alpha, beta) rather than dividing by possibly zero beta.
- Sylvester: wrap TRSYL and provide a general dense solver for A X ± X B = C using existing Schur decompositions and matrix multiplication. Return the scaled solution, scale, and near-singularity status explicitly; never silently divide by scale.
- LQ: wrap GELQF, ORGLQ/UNGLQ, and ORMLQ/UNMLQ. Provide an LqDecomposition holding packed reflectors and tau, with full/compact Q and extraction of L, mirroring QR.
- No new dependencies or unrelated API refactors.

## Files to modify
- New wrappers under `rlst/src/dense/linalg/lapack/interface/`: `gges.rs`, `trsyl.rs`, `gelqf.rs`, `orglq.rs`, and `mlq.rs`.
- `rlst/src/dense/linalg/lapack/interface.rs`: register wrappers.
- `rlst/src/traits/linalg/lapack.rs`: integrate scalar interface traits following the existing aggregate-trait pattern.
- New high-level modules under `rlst/src/dense/linalg/lapack/`: `generalized_schur.rs`, `sylvester.rs`, and `lq.rs`; register in `lapack.rs`.
- `rlst/src/traits/linalg/decompositions.rs`: array-facing traits. Existing wildcard re-exports through `traits/linalg.rs` and `lib.rs` already expose these traits; keep result types in their decomposition modules, matching QR.
- `rlst/src/doc/matrix_decompositions.rs`: examples and numerical contracts.

## Reuse
- `rlst/src/dense/linalg/lapack/qr.rs`: packed factorization object, `QMode`, factor extraction, and generic array copying via `DynArray::new_from`.
- `rlst/src/dense/linalg/lapack/eigenvalue_decomposition.rs`: existing `schur()` returns `(T, Z)` for A = Z T Z^H; reuse for the dense Sylvester solver.
- `rlst/src/dense/linalg/lapack/interface.rs`: `lapack_return`; TRSYL's INFO=1 requires explicit handling rather than blindly treating it as ordinary success.
- `interface/gees.rs`: `JobVs`, unsorted calls with `SORT='N'`/no callback, workspace queries, and optional Schur-vector buffers.
- `interface/ggev.rs`: homogeneous `alpha` and `beta` both represented as `Self::Complex`; use the same convention for QZ, but Q/Z remain `Self` even for real inputs.
- `interface/mqr.rs`: left/right application and real transpose versus complex adjoint mapping. Reuse its public `MqrSide` and `MqrTransMode` enums without renaming existing APIs; document their use for LQ too. Adapt storage validation: LQ reflectors occupy rows, not QR's columns.
- `lapack/lu.rs::LuDecomposition::solve`: reusable factor solves and 1D/2D RHS handling; explicitly reject NDIM other than 1 or 2 in the new solver.
- Existing `solve_triangular`/TRSM, `dot!`, conjugate-transpose views, and `DynArray` allocation for LQ solves and Sylvester basis transforms. Reuse `RlstError::GeneralError` for high-level validation errors.

## Steps
- [x] Confirm QZ sorting scope, Sylvester equation variants/result semantics, and LQ convenience scope.
- [x] Inspect wrapper validation, error types, transposition enums, exports, and existing numerical-test patterns.
- [x] Before coding each wrapper, check exact signatures and workspace requirements in the installed `lapack` 0.20 bindings and LAPACK documentation; no dependency upgrade.
  - Evidence: inspected `lapack-0.20.0/src/lapack-sys.rs` signatures for S/D/C/Z GGES, TRSYL, GELQF, ORGLQ/UNGLQ, and ORMLQ/UNMLQ, and the corresponding Netlib LAPACK routine sources. GGES, GELQF, ORGLQ/UNGLQ, and ORMLQ/UNMLQ use `LWORK=-1` queries; complex GGES additionally requires `RWORK(8*N)`; TRSYL has no workspace and returns `SCALE` with `INFO=1` denoting perturbation.
- [x] Implement and test low-level wrappers for all four scalar types.
  - Evidence: registered `gges`, `trsyl`, `gelqf`, `orglq`, and `mlq` interfaces, implemented S/D/C/Z bindings and aggregate `Lapack` bounds, and verified targeted factorization, orthogonality/application, every GGES job option, empty input, perturbation, malformed storage/leading dimensions, checked integer/size overflow, and padded leading dimensions with sentinels.
- [x] Implement and test QZ convenience API, preserving real quasi-triangular structure and homogeneous eigenvalues.
  - Evidence: added `GeneralizedSchur::qz` and `QzDecomposition` with copied arbitrary-layout inputs, immutable S/T/Q/Z/alpha/beta accessors, unsorted homogeneous eigenvalues, shape errors, and empty handling. Numerical tests cover S/D/C/Z reconstruction and orthogonality/unitarity, real and complex Schur structure, real conjugate pairs, a singular-B infinite eigenvalue, a known finite homogeneous ratio, 1x1/empty inputs, and transposed views.
- [x] Implement and test Sylvester convenience API, including scaling and near-singularity behavior.
  - Evidence: added `SylvesterSolve::solve_sylvester` and `SylvesterSolution`, reusing real/complex Schur decomposition, matrix multiplication, and TRSYL while retaining scale and perturbation status without unscaling. Verified manufactured TRSYL solutions for both signs and all four operation combinations across all four scalar types, including complex adjoints; high-level tests cover rectangular solutions, complex data, real 2x2 Schur blocks, transposed A/C views, empty dimensions and zero columns, shape errors, an unperturbed forced `scale < 1` case with normalized scaled residual, and separate coincident-spectrum `Perturbed` reporting.
- [x] Implement and test LQ factor object, full/compact factors, and implicit Q application.
  - Evidence: added `Lq::lq`, `LqDecomposition` full/compact Q and L extraction, non-destructive implicit left/right Q/Q^H application, and reusable full-row-rank minimum-norm vector/matrix solves without changing generic solve dispatch. Tests cover S/D/C/Z wide, square, and tall reconstruction, compact/full orthogonality, all implicit application combinations against independent rectangular right-hand sides, repeated calls, matrix residuals and nullspace orthogonality, vector/matrix solve agreement, copied transposed views, invalid/tall/rank-deficient/NDIM contracts, empty factors, zero-row solves, and zero RHS columns.
- [x] Add Rustdoc and user-guide examples; run validation.
  - Evidence: expanded public QZ, Sylvester, and LQ contracts, including implicit Q application, and added compiling guide examples for factor reconstruction, homogeneous eigenvalues, scaled Sylvester results, and minimum-norm solves. Targeted module tests and guide doctests pass, as do formatting, default Clippy/examples/tests, strict Clippy/examples, rustdoc, and final diff checks.

## Detailed API and numerical contracts

### Generalized Schur / QZ
- Low-level `Gges::gges`: destructive column-major A/B buffers, separate `JobVs` controls and optional buffers for Q/Z, homogeneous alpha/beta output, internal workspace query/allocation. Implement all four GGES variants; no sorting callback or sorting API.
- High-level `GeneralizedSchur::qz(&self, b)` returns `RlstResult<QzDecomposition<Item>>`. Own S, T, Q, Z and alpha/beta arrays; expose documented immutable accessors. Require equal square shapes and copy inputs, including noncontiguous views.
- Contract: A = Q S Z^H and B = Q T Z^H. For real input S is quasi-upper triangular and T upper triangular; for complex input both are upper triangular. Q/Z are orthogonal/unitary. No promise of eigenvalue ordering.
- Retain alpha/beta without division: beta=0 can represent infinite eigenvalues and alpha=beta=0 an indeterminate pair. Do not reject singular B merely because it is singular. Propagate GGES failure codes.

### Sylvester equations
- Low-level `Trsyl::trsyl`: solve op(A) X + sign X op(B) = scale C in place on C, with real quasi-triangular or complex triangular coefficients. Expose sign as a two-variant enum and no-transpose/adjoint modes (transpose for real). Shape checks cannot establish Schur structure; document that mathematical precondition.
- Return a small `TrsylOutcome<Real>` with scale and a named status (`Success` or `Perturbed`). INFO=1 means LAPACK perturbed close/common eigenvalues: retain the computed solution and report this status. Negative INFO and other unexpected codes remain errors. TRSYL has no workspace query.
- High-level `SylvesterSolve::solve_sylvester(&self, b, c, sign)` returns `SylvesterSolution<Item>` containing X, scale, and status. A and B must be square, of possibly different orders m/n; C must be m-by-n. Inputs are not modified.
- Use A=U R U^H and B=V S V^H via existing `schur()`, form F=U^H C V, solve R Y ± Y S=scale F, then X=U Y V^H. Support real 2x2 blocks directly through real TRSYL; do not unnecessarily convert real problems to complex.
- Public result satisfies A X ± X B = scale C when unperturbed (up to numerical error). For perturbed results, explain the nearby-problem interpretation. Do not automatically unscale: X/scale may overflow. Document how callers may unscale when safe. No generalized Sylvester, Lyapunov-specific API, or reusable Schur solver object in this scope.

### LQ decomposition and reusable solver
- Low-level `Gelqf`, `Orglq` (ORGLQ/UNGLQ), and `Mlq` (ORMLQ/UNMLQ) mirror QR wrappers with correct row-reflector dimensions and workspace queries.
- `Lq::lq(&self)` returns `LqDecomposition<Item>` containing a copied packed factorization and tau; no pivoting option.
- For A of shape m-by-n and k=min(m,n), `q_mat(QMode::Compact)` returns k-by-n; `q_mat(QMode::Full)` returns n-by-n. `l_mat()` returns m-by-k, lower trapezoidal. Compact reconstruction is A=L Q; with full Q use Q's first k rows or pad L to m-by-n. Document these shapes explicitly.
- `apply_q(side, trans, rhs)` copies a 2D input and applies full Q or Q^H implicitly, from either side, without constructing Q. The affected dimension must be n. Reuse existing QR option enums and preserve packed factors across calls.
- `solve(rhs)` accepts 1D/2D RHS and m<=n with full row rank. Solve L11 Y=B, pad Y with n-m zero rows, and apply Q^H from the left; return n or n-by-nrhs shape. This gives the minimum Euclidean/Frobenius-norm solution without forming Q.
- Reject tall factors and incompatible RHS shapes. Check for an exactly zero L11 diagonal before using existing triangular solves (TRSM does not report singularity). Explicitly document that this is not rank revealing: near-rank-deficient input has no reliability guarantee; do not invent an arbitrary rank tolerance or change the existing generic `solve()` dispatch.

### Shared safety and compatibility
- New low-level wrappers follow existing assertions for malformed dimensions/storage, but use checked size arithmetic and checked usize-to-LAPACK-integer conversion before FFI. Validate optional output buffers and use valid dummy storage for disabled outputs. No broad lint suppressions.
- High-level shape failures return existing `RlstError` values. Handle zero-sized problems before unsafe calls: empty QZ, empty Sylvester with scale=1/status=Success, and empty LQ with identity full Q; minimum-norm solve with m=0 returns zeros. Treat no-op Q application as a copy. Test zero RHS columns too.
- Extend the existing aggregate `Lapack` trait in the established style; existing four supported scalars keep their functionality. No unrelated edits to older wrappers, no changes to existing methods' semantics, and no new native dependencies.

## Verification
- Put unit tests in new modules, using existing scalar-generating macros and approximate array comparisons. Separate low-level buffer/option tests from convenience-level numerical tests.
- QZ: reconstruct both A and B; check Q/Z orthogonality/unitarity and Schur structure. Include real conjugate pairs, complex matrices, singular B/infinite eigenvalues, empty/1x1 inputs, and every low-level Q/Z job combination. Test homogeneous eigenvalues against known diagonal pencils without unstable division; do not compare raw Schur factors across backends.
- Sylvester: both signs and all supported low-level operation combinations; rectangular X; manufactured solutions with separated spectra; real 2x2 blocks; complex adjoints; overflow-prone deterministic examples forcing scale<1; and coincident spectra reporting Perturbed. Check normalized scaled residuals, not an unscaled equation, and status semantics for perturbed cases.
- LQ: square, wide, and tall reconstruction; compact row orthogonality/full unitarity; implicit left/right Q/Q^H versus explicit multiplication; repeated calls preserving factors. For solves test vector/multiple RHS and complex data, residuals, and minimum norm via a well-conditioned SVD pseudoinverse reference or orthogonality to the nullspace. Reject tall/rank-zero-diagonal cases.
- Exercise copied strided/transposed input views, invalid dimensions/storage and integer overflow rejection, zero-sized cases, and zero RHS columns. Low-level buffer tests include padded leading dimensions.
- During implementation: targeted tests, LSP diagnostics before builds, `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, and `cargo test`; strict-feature checks and documentation checks where available.
- Final edited-file diagnostics and `git diff --check`; do not change Cargo.lock.

## Confirmed scope
- User confirmed unsorted QZ only; no reordering or invariant-subspace selection.
- Ordinary Sylvester equations only, with explicit scaling and near-singularity status as recommended.
- Include a reusable full-row-rank minimum-norm LQ solver.
