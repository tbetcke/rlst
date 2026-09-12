# Code Review: `code_review` at `4797a16`

## Scope and methodology

This is a stand-alone review of the current repository state on branch `code_review`, HEAD `4797a16548210ace662c9f53e567f2bd20851ef3` (`Merge pull request 'Made MPI routines stable with respect to no data on a given rank' (#25)`). It is a current-source review rather than an attribution of every finding to the merge commit.

The review covered the merge's MPI/distributed changes and tests, then followed public safety and correctness boundaries through dense arrays and views, FFTW plan ownership, sparse construction and operations, iterative solvers, Matrix Market I/O, procedural macros, SuiteSparse integration, workspace configuration, and their tests. Candidate findings from the supplied specialist reviews were checked against HEAD. Duplicate claims were consolidated; coverage observations were not presented as functional defects unless current code supplied a concrete failing path.

Commands actually run are listed under **Validation limits**. No project source file was modified.

## Executive summary

**Verdict: BLOCK.** Five independent P0 findings expose undefined behavior through safe public APIs: unchecked custom strides, unchecked multislice indices, overflowable subview bounds, replaceable FFTW plan buffers, and an insufficient mutable-iterator capability contract. These need correction before release or merge regardless of whether they predate the MPI branch.

Seven P1 findings produce silent wrong results, invalid numerical states, panics where recoverable errors are promised, or a broken exported macro. The most consequential non-memory-safety problems are inconsistent raw CSR semantics, silent matrix-RHS truncation, and conjugate gradients corrupting an already solved zero system.

The MPI empty-rank work itself is notably well defended: a dedicated 12-case executable is run at 1, 2, and 4 ranks by a bounded process-group-aware runner, and CI invokes it separately. The runner's Python unit tests passed locally. The actual MPI matrix was not run locally, so no runtime claim is made for MPI behavior.

## Prioritized findings

### P0 — release blockers

#### P0-001 — Safe custom-stride arrays can access out of bounds or create mutable aliases

**Evidence:** `StridedBaseArray::new` checks only `data.number_of_elements() == product(shape)` and accepts arbitrary strides (`rlst/src/dense/strided_base_array.rs:34-49`). Access uses `convert_nd_raw(multi_index, stride)` directly with unchecked container access (`:64-86`, `:117-126`). Public dynamic and slice constructors expose this safely (`rlst/src/dense/array.rs:310-319`, `:661-694`). `VectorContainer` ultimately uses slice `get_unchecked` (`rlst/src/dense/data_container.rs:118-123`).

**Impact:** `StridedDynArray::<f64, 1>::from_shape_and_stride([2], [2])` allocates two elements, yet logical index `[1]` maps to physical index 2. A zero or overlapping stride (for example `[2]`/`[0]` or `[2,2]`/`[1,1]`) lets mutable iteration expose aliases. Both are undefined behavior reachable from safe code.

**Recommendation:** Establish constructor invariants with checked arithmetic: every reachable offset must be within storage and mutable layouts must be injective. If general padded/overlapping views are not required, initially restrict safe mutable construction to validated nonoverlapping layouts. Dynamic padded layouts must allocate the storage span implied by strides rather than `product(shape)`.

**Specific tests:** Reject insufficient spans and overlapping mutable layouts; accept validated row-major and padded nonoverlapping layouts. Run access and mutable iteration regressions under Miri.

#### P0-002 — `Multislice::new` does not validate fixed axes or indices

**Evidence:** `Multislice::new` copies `indices[pos]` into the source offset without comparing it with `arr.shape()` (`rlst/src/dense/array/multislice.rs:32-60`). Access reconstructs a source index and calls the underlying unchecked accessor (`:114-159`). The safe public entry point forwards inputs unchanged (`rlst/src/dense/array.rs:1454-1463`).

**Impact:** For a `[2,2]` array, `.multislice::<1,1>([1], [2])` advertises a valid one-dimensional shape, but accessing its logical index `[0]` maps to source index `[0,2]`, causing an out-of-bounds read or write. Duplicate or out-of-range axes also violate the mask-count assumption and can panic during construction.

**Recommendation:** Validate that every axis is in range, axes are distinct, and each fixed index is below the corresponding source extent before constructing the mask. Follow the adjacent checked single-axis slice convention.

**Specific tests:** Reject an index equal to the axis extent, a fixed index on a zero-length axis, duplicate axes, and axes `>= ADIM`; cover immutable and mutable access.

#### P0-003 — Subview bounds checking overflows in release builds

**Evidence:** `ArraySubView::new` validates `offset[index] + shape[index] <= arr_shape[index]` with unchecked `usize` addition (`rlst/src/dense/array/subview.rs:21-34`). Access later adds the logical index to the offset and delegates to unchecked source access (`:81-92`, `:121-145`). The constructor is publicly reachable via `Array::into_subview` (`rlst/src/dense/array.rs:1482-1493`).

**Impact:** On a length-2 array, offset `[usize::MAX]` and shape `[2]` wrap to 1 when overflow checks are disabled, pass validation, and allow source access at `usize::MAX`. This is safe-code undefined behavior in release builds.

**Recommendation:** Use `checked_add`, or validate `offset <= extent` followed by `shape <= extent - offset`. Apply checked arithmetic in index reconstruction as a defensive invariant.

**Specific tests:** Assert rejection of `[usize::MAX]`/`[2]` in both debug and release profiles, while preserving a valid empty view at the end boundary.

#### P0-004 — FFTW plans retain a pointer to an allocation that safe code can replace

**Evidence:** in-place plan creation stores the current `arr.data_mut()?.as_mut_ptr()` in FFTW (`rlst/src/fftw/c2c_inplace.rs:91-113`; equivalent plan families use the same architecture). Execution later invokes the stored native plan (`rlst/src/fftw.rs:403-421`). `Fft::borrow_mut` returns `RwLockWriteGuard<Array<...>>`, permitting replacement of the entire array value (`rlst/src/fftw.rs:475-483`).

**Impact:** Safe code can create a plan, assign a different same-typed array through `*fft.borrow_mut().unwrap()`, drop the original allocation, and then execute the plan. The lock prevents concurrent mutation but does not preserve allocation identity, so FFTW dereferences stale pointers.

**Recommendation:** While any plan exists, expose only non-replaceable element access (for example, a restricted mutable slice/view), not `&mut Array`. Alternatively make the plan own the fixed buffer and ensure all public access preserves its allocation and layout.

**Specific tests:** Add a compile-time API test demonstrating that whole-array replacement is impossible. Exercise repeated mutation/execution under ASan or another native memory checker with FFTW enabled.

#### P0-005 — Mutable iteration assumes disjointness that its public trait does not require

**Evidence:** `UnsafeRandom1DAccessMut` is safe to implement and documents only the caller's in-bounds requirement (`rlst/src/traits/accessors.rs:76-83`). `ArrayDefaultIteratorMut::next` calls it repeatedly and transmutes each result to the full iterator lifetime (`rlst/src/dense/array/iterators.rs:147-170`).

**Impact:** A conforming implementation can return the same stable element for every in-range index. Each individual call meets the documented contract, but the safe iterator then exposes multiple live `&mut` references to one location. Custom strides already provide one concrete route; the generic extension boundary remains unsound even after fixing that constructor.

**Recommendation:** Introduce an unsafe-to-implement mutable-iteration capability whose safety contract requires stable, pairwise-disjoint locations for distinct logical indices, and implement it only for audited storage/views. Prefer slice iteration or explicit raw-pointer splitting where possible.

**Specific tests:** Add compile/API coverage for the capability boundary and run all audited mutable iterators, including views and strided layouts, under Miri.

### P1 — high-priority correctness/API defects

#### P1-001 — CSR matrix application silently truncates mismatched RHS columns

**Evidence:** vector application validates both dimensions (`rlst/src/sparse/csr_mat.rs:273-281`). Matrix application simply zips `x.col_iter()` and `y.col_iter_mut()` (`:301-323`) and performs no matrix-level shape checks.

**Impact:** If input and output have different column counts, only the shared prefix is processed. Extra output columns are not even scaled by `beta`; extra input columns are discarded. With zero columns, row mismatches bypass validation entirely.

**Recommendation:** Before iteration, require input rows equal matrix columns, output rows equal matrix rows, and equal input/output column counts.

**Specific tests:** Test both column-mismatch directions and zero-column row mismatches, requiring rejection before output mutation. For a valid multi-RHS case, compare against an independent dense reference using nonzero `alpha`, nonzero `beta`, and nonzero initial output.

#### P1-002 — Duplicate raw CSR entries change meaning during dense conversion

**Evidence:** `CsrMatrix::new` validates pointers and bounds but permits duplicate columns (`rlst/src/sparse/csr_mat.rs:40-79`). Application sums every stored contribution (`:282-297`). `todense` delegates to `DynArray::from_iter_aij` (`:226-230`), whose loop assigns rather than accumulates (`rlst/src/dense/array.rs:232-245`). By contrast, `normalize_aij` explicitly sums duplicates (`rlst/src/sparse/tools.rs:7-13`, `:65-83`).

**Impact:** A `[1,1]` CSR with two entries at `(0,0)` valued 2 and 3 applies as 5, while its dense conversion contains 3. Conversion does not preserve the represented linear map.

**Recommendation:** Either canonicalize raw CSR to sorted unique coordinates in `CsrMatrix::new`, or make `todense` accumulate duplicates. Document whether generic dense AIJ construction is assignment or assembly.

**Specific test:** Assert sparse and converted-dense application equivalence for duplicate raw CSR input.

#### P1-003 — Sparse binary merge assumes ordering that public producers do not guarantee

**Evidence:** `BinaryAijOperator` performs a sorted row-major merge (`rlst/src/sparse/binary_operator.rs:58-76`, `:107-130`) but its public constructor does not state or validate ordering (`:26-43`). `CsrMatrix::new` accepts unsorted columns (`rlst/src/sparse/csr_mat.rs:40-79`), and `iter_aij_value` preserves storage order (`:174-190`).

**Impact:** Elementwise operations can silently miss matching coordinates. For example, a raw one-row CSR ordered as columns `[1,0]` multiplied componentwise by a matrix containing only column 0 consumes entries in the wrong merge order and yields zero instead of the product at column 0.

**Recommendation:** Prefer a canonical sorted/unique invariant in raw CSR construction. Independently document and enforce sorted input for public iterator-based operators, or make them normalize.

**Specific test:** Apply each binary operation to deliberately unsorted raw CSR and compare with dense elementwise results.

#### P1-004 — Sparse elementwise operations do not validate shape equality

**Evidence:** sparse addition/subtraction retain only the left shape (`rlst/src/sparse/mat_operations.rs:153-197`); multiplication/division do the same (`:244-288`). No comparison with `other.shape()` occurs.

**Impact:** Empty matrices of shapes `[1,1]` and `[2,2]` can be combined successfully, with result shape depending on operand order. Nonempty mismatches may fail later only if a produced coordinate falls outside the retained left shape, making validation sparsity-dependent.

**Recommendation:** Require identical shapes at binary-operation construction, matching dense elementwise operations.

**Specific tests:** Reject mismatched empty and nonempty matrices in both operand orders for add, subtract, multiply, and divide.

#### P1-005 — Conjugate gradients corrupts an already solved zero system

**Evidence:** `CgIteration::run` computes `sqrt(res_inner.abs()) / rhs_norm` before checking convergence (`rlst/src/operator/algorithms/conjugate_gradients.rs:117-135`). It then computes `alpha = res_inner / p_conj_inner` (`:137-143`). The only test uses a nonzero random RHS and checks only the solver-reported residual (`:184-215`).

**Impact:** For SPD identity, `b=0`, and `x=0`, the initial relative residual is `0/0 = NaN`; convergence fails and the first step computes another `0/0`, replacing the correct finite solution with NaNs. A zero RHS with a nonzero initial guess also lacks a defined normalized stopping rule.

**Recommendation:** Check exact zero residual before normalization. When `rhs_norm == 0`, use a documented absolute-residual criterion (or another explicit zero-RHS convention), and guard breakdown denominators.

**Specific tests:** Identity with zero RHS and zero/nonzero initial guesses; assert finite output, true residual, callback count, and zero-iteration behavior. Add a non-diagonal SPD case and independently compute `b-Ax`.

#### P1-006 — Matrix Market readers panic on ordinary I/O errors and reject standard complex records

**Evidence:** public readers return `RlstResult` but unwrap file opening and header parsing (`rlst/src/io/matrix_market.rs:133-135`, `:191-195`). Additional line and dimension parses are also unwrapped (`:158-174`, `:332-350`, `:382-423`). The header accepts `complex`, but array parsing requires one token and coordinate parsing exactly three (`:325-369`, `:373-443`), whereas Matrix Market complex values use separate real and imaginary fields.

**Impact:** Missing/malformed files can panic instead of returning `RlstError`. Standards-compliant complex array and coordinate files are rejected despite the accepted header and generic numeric API.

**Recommendation:** Propagate all I/O/parse errors with context. Parse records according to declared Matrix Market data type, validate record counts and 1-based indices, and either support complex data fully or reject it explicitly at the header boundary.

**Specific tests:** Hand-authored real and complex array/coordinate fixtures; missing files; invalid headers/dimensions; truncated and excess records; zero/out-of-range coordinate indices; rectangular column-major ordering. Assert returned errors and values, not only round trips.

#### P1-007 — Exported `rlst_static_type!` expands to obsolete generic signatures

**Evidence:** the macro emits `Array` and `BaseArray` with three generic arguments (`proc-macro/src/dense.rs:94-114`), while current declarations take two (`rlst/src/dense/array.rs:115-116`; `rlst/src/dense/base_array.rs:22-24`). Repository search found no invocation. A temporary external crate invoking `rlst_static_type!(f64, 2, 3)` failed with `E0107`, directly confirming the public API break.

**Impact:** Every consumer invocation of this exported macro fails to compile, while the proc-macro crate's zero-test suite remains green.

**Recommendation:** Generate `Array<BaseArray<ArrayContainer<T, N>, NDIM>, NDIM>` using current signatures and add a main-crate compile-pass doctest or integration test.

**Specific test:** Declare a value with `rlst_static_type!(f64, 2, 3)` and initialize it from the matching static-array constructor; add malformed-input compile-fail cases separately.

### P2 — medium-priority behavior, integration, and coverage issues

#### P2-001 — Empty-shape support differs between CSR constructors

**Evidence:** raw `CsrMatrix::new` can represent `[0,n]` with `indptr=[0]` (`rlst/src/sparse/csr_mat.rs:40-79`). `from_aij` substitutes zero when empty rows/columns have no maxima and then requires that zero be below both dimensions (`:98-125`).

**Impact:** `from_aij([0,3], &[], &[], &[])`, `[3,0]`, and `[0,0]` panic although equivalent empty matrices are representable through the raw constructor.

**Recommendation/tests:** Only validate maxima when entries exist; test all three zero-dimension shapes through raw, AIJ, and iterator conversions.

#### P2-002 — Root workspace defaults require SuiteSparse for routine main-crate commands

**Evidence:** root `Cargo.toml:1-3` includes `rlst` and `rlst-suitesparse` with no `default-members`. `rlst-suitesparse/build.rs:7-16` unconditionally locates and links SuiteSparse.

**Impact:** unqualified root `cargo test`/`cargo clippy` requires an optional native integration even though the main crate defaults to no optional features. This conflicts with the intended separation and increases local-development friction.

**Recommendation/tests:** Set `default-members = ["rlst"]`; retain explicit SuiteSparse validation in native-enabled CI. Verify unqualified root commands on a machine without SuiteSparse and explicit workspace commands in CI.

#### P2-003 — Sparse unary/division APIs have storage-dependent rather than logical elementwise semantics

**Evidence:** `Exp`, `Cos`, `Recip`, and similar ordinary numeric traits are implemented for sparse operators (`rlst/src/sparse/mat_operations.rs:104-151`), but `UnaryAijOperator` visits only stored coordinates (`rlst/src/sparse/unary_aij_operator.rs:28-53`). Binary division visits only the union of stored coordinates (`rlst/src/sparse/mat_operations.rs:267-288`; `rlst/src/sparse/binary_operator.rs:84-142`).

**Impact:** `exp` of an implicit zero remains zero, while an explicitly stored zero becomes one; absent/absent floating division remains zero instead of `NaN`. Results depend on representation, not only matrix values. This may reflect a sparse-preserving optimization, but ordinary numerical trait names do not communicate stored-entry semantics.

**Recommendation/tests:** Restrict sparse-preserving implementations to zero-preserving functions, or rename/document stored-entry transformations explicitly. Add tests contrasting implicit and explicit zero so the selected contract is deliberate.

No additional P2 functional finding is asserted solely from missing tests.

## Dedicated unit-test coverage assessment

### Strong coverage

- The locally executed default main-crate suite passed **170 unit tests and 49 doctests**. Dense decomposition tests span real/complex and single/double precision across LU, QR, Cholesky, eigensolvers, SVD, inverse, and solves.
- The branch adds `rlst/tests/mpi_empty_ranks.rs`, with 12 named cases covering zero-count collectives, empty scans/boundaries/extrema, sparse sorting, mapper/permutation behavior, collective rejection, empty CSR contracts, remapping/layout agreement, and distributed consumers (`scripts/run_mpi_tests.py:18-33`; `rlst/tests/mpi_empty_ranks.rs:85-994`).
- The MPI runner builds once, executes every case at 1/2/4 ranks, assigns a 30-second case timeout, and terminates the owned process group with TERM then KILL (`scripts/run_mpi_tests.py:17-38`, `:41-84`, `:140-191`). CI invokes this matrix separately (`.forgejo/workflows/run-tests.yml:56-60`). Its Python runner tests passed locally.
- Existing multislice tests cover shape, stride, access, row-/column-major raw access, and noncontiguous `data()` rejection (`rlst/src/dense/array/multislice.rs:334-499`). SuiteSparse tests cover CSC conversion, duplicate aggregation, real/complex solves, and selected RHS/layout errors (`rlst-suitesparse/src/umfpack/tests.rs:25-124`).

### Material gaps

- None of the P0 safety boundaries has a negative regression: invalid/overlapping stride, invalid multislice index, subview overflow, FFT allocation replacement, or mutable-iterator disjointness.
- Serial CSR has one vector application test (`rlst/src/sparse/csr_mat.rs:331-348`) and no matrix-RHS, raw duplicate/order, shape-mismatch, or empty-dimension tests.
- CG has one diagonal positive case; callback residuals are collected but never asserted (`rlst/src/operator/algorithms/conjugate_gradients.rs:184-215`).
- Matrix Market has no module tests or fixture coverage.
- `rlst-proc-macro` ran zero unit tests and zero doctests; main-crate doctests do not invoke `rlst_static_type!`.
- Dense `apply` tests set nonzero `beta` but initialize output to zero (`rlst/src/dense/array.rs:2151-2193`), so they cannot detect deletion of accumulation. Initialize output to distinct nonzero values.
- Mutable multislice tests read through mutable views without writing (`rlst/src/dense/array/multislice.rs:442-499`; `rlst/src/dense/array.rs:2246-2262`). Write distinct markers and verify parent write-through and untouched entries.
- Pseudoinverse tests use random full-rank matrices with `None, None` (`rlst/src/dense/linalg/lapack/pseudo_inverse.rs:103-168`); truncation/count logic at `rlst/src/traits/linalg/decompositions.rs:245-282` lacks rank-deficient, cutoff-boundary, rank-zero, and all-zero cases.
- FFT tests emphasize DC sums and forward/backward round trips (`rlst/src/fftw/c2c_inplace.rs:136-172`; `rlst/src/fftw/r2r_inplace.rs:132-211`), which cannot detect swapping both direction conventions. Compare nonsymmetric data against a direct DFT/DCT.
- SuiteSparse does not test empty/nonsquare rejection or singular-warning behavior (`rlst-suitesparse/src/umfpack/csc.rs:25-32`; `rlst-suitesparse/src/umfpack/mod.rs:46-51`).

## Architectural assessment

The broad architecture is sound in several respects. Dense/sparse storage, abstract operator adapters, iterative algorithms, distributed tools, and native integrations are separated cleanly. Dense and CSR operator adapters consistently derive domain size from columns and range size from rows. MPI and FFTW remain feature gated, and SuiteSparse is isolated in its own crate. The new MPI harness recognizes that ordinary Rust test execution is insufficient for collective protocols and supplies bounded multi-rank process isolation.

The dominant architectural weakness is that unsafe consumers rely on invariants not established at safe construction or trait boundaries. Logical bounds checks are effective only if shape-to-storage mappings are valid. `StridedBaseArray`, `Multislice`, and `ArraySubView` violate that premise; mutable iteration relies on undocumented disjointness; FFTW locking models concurrency but not allocation identity. These are one systemic issue—capability and ownership contracts are too weak—even though each has a distinct exploit path and fix.

Sparse architecture has a similar, non-memory-safety invariant gap. Some producers normalize to sorted unique coordinates, raw CSR construction does not, while conversion and merge consumers assume different duplicate/order semantics. A single documented canonical CSR invariant would eliminate several correctness defects.

## Inconsistent design choices

1. **Checked logical indices vs unchecked layout invariants:** accessors validate shape, but safe custom views/layouts can map valid logical indices outside storage.
2. **Single slice vs multislice:** single-axis slicing validates axis/index bounds; multislicing does not.
3. **Dense vs sparse matrix application:** dense and CSR-vector paths validate dimensions; CSR multi-RHS relies on truncating `zip`.
4. **Normalized AIJ vs raw CSR:** `from_aij` sorts and combines duplicates; `new` permits both, although downstream sparse merge and dense conversion assume incompatible semantics.
5. **Dense vs sparse elementwise arithmetic:** dense operations require equal shapes and act on every logical element; sparse operations retain the left shape and act on stored coordinates.
6. **`RlstResult` API vs parser behavior:** Matrix Market readers advertise recoverable failure but unwrap common I/O and parse errors.
7. **Feature isolation vs workspace defaults:** SuiteSparse is architecturally separate but selected by default root workspace commands.

## Strengths and intentional tradeoffs

- The MPI change invests substantially in empty-rank and collective-failure behavior rather than only happy-path distributed tests.
- The bounded runner's separate MPI sessions and process-group cleanup are appropriate safeguards against deadlocked collective tests.
- Native integrations are feature/crate separated, and SuiteSparse conversion uses checked error types and duplicate aggregation before native calls.
- Operator adapters keep algorithms independent of concrete dense/sparse storage.
- Broad macro-generated numerical tests across scalar types provide useful regression depth.
- Sparse-preserving iteration is a legitimate performance goal. The issue in P2-003 is the semantic surface exposed through ordinary elementwise trait names, not the goal of avoiding densification.
- Runtime locks around FFT arrays are useful for concurrency. P0-004 is specifically that locking alone does not pin buffer identity.

## Validation limits

Commands run from the repository root:

- `git rev-parse HEAD`, `git log -1 --oneline`, and diff-name/stat inspection: confirmed the requested HEAD and reviewed merge scope.
- `cargo test -p rlst`: passed 170 unit tests and 49 doctests.
- `python3 scripts/test_run_mpi_tests.py`: passed with no output.
- `cargo test --manifest-path proc-macro/Cargo.toml`: passed, but ran 0 unit tests and 0 doctests.
- A temporary external crate ran `cargo check` while invoking `rlst_static_type!(f64, 2, 3)`: failed as expected with `E0107`, confirming P1-007. The temporary crate was removed.
- `git diff --check`, `git status --porcelain=v1`, and `git diff --cached --name-only`: passed/empty; no project changes or staged files were present.

Not run:

- The actual `python3 scripts/run_mpi_tests.py` multi-rank matrix; MPI runtime availability and behavior were not established.
- FFTW-enabled tests, SuiteSparse tests, full strict-feature CI, Clippy, rustfmt check, Miri, sanitizers, or release-mode regressions.
- The unqualified root test command, because it would select SuiteSparse; the main-crate test was intentionally run with `-p rlst`.

Passing default tests do not refute the findings: the concrete edge cases are absent, and memory-safety failures need targeted release/Miri/native instrumentation.

## Prioritized action plan

1. **Close all safe-code UB paths:** validate stride span/nonoverlap, validate multislice inputs, make subview arithmetic checked, pin FFTW allocation identity, and strengthen the mutable-iteration capability contract.
2. **Add safety regressions before refactoring further:** negative constructor/view tests, release overflow tests, Miri mutable iteration, and FFTW native memory-instrumented ownership tests.
3. **Define and enforce a canonical CSR contract:** decide where raw input is sorted/combined, then make application, dense conversion, and binary merge preserve one mathematical interpretation.
4. **Make all matrix operations validate shapes eagerly:** start with CSR multi-RHS and sparse binary operators; tests must assert no partial output mutation.
5. **Fix CG's zero-RHS/exact-solution path and denominator breakdown handling**, with true-residual tests independent of the reported residual.
6. **Repair the public macro and Matrix Market error/complex handling**, adding compile-pass and fixture-based regression suites.
7. **Fill targeted numerical coverage gaps:** nonzero-beta accumulation, write-through mutable views, rank-deficient/truncated pseudoinverse, direct FFT convention references, and SuiteSparse rejected/singular matrices.
8. **Reduce optional-native friction:** add `default-members = ["rlst"]` and keep explicit native integration jobs in CI.
9. **Run the full validation matrix after fixes:** formatting, Clippy with warnings denied, default tests, strict MPI/FFTW tests, bounded MPI ranks, SuiteSparse tests, docs, Miri, and relevant sanitizers.
