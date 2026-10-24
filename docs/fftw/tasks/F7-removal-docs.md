# F7: Removal of the 0.10 module; the Chebyshev port; documentation and licence warning

| | |
| --- | --- |
| Task | F7 (design, Section 9) |
| Status | not started |
| Depends on | F1 to F6 |
| Pull requests | three, into `fftw-0.11`, in this order: F7a (the Chebyshev port to the new plans, while the 0.10 module still exists to compare with), F7b (the removal of the 0.10 module, its tests and its documentation; the module documentation of `rlst::fftw` for the new interface), F7c (the user guide with the seven rules of Section 5, the licence warning in its four places, the docs.rs metadata, the removal of the unused build dependencies, and the CHANGELOG with the migration notes of Appendix B) |
| Read first | design (draft 0.17): Sections 2.1 and 2.2 (what 0.10.3 provides, and its limitations), Section 4.10 (versions, docs.rs and the build dependencies), Section 5 (all seven rules), Section 6 (the licence notice), Section 7.3 (the Chebyshev regression), Section 8 (the Chebyshev module), Section 9 (F7 row and Transition), Section 10, in particular DC10, DC22, DC23, DC39 and DC44 to DC47, Appendix B (migration); the "Open questions" of `docs/fftw/tasks/F3-split.md` (note 7), `F4-determinism.md` (notes on the fingerprint) and `F6-wisdom-threads.md` (notes 7, 11 and 14); `CLAUDE.md` |

## Goal

The interface of design Section 4 is complete since F6; F7 makes it the only one and documents
it. **The Chebyshev port** moves `rlst::chebychev`, the in-crate user of the 0.10 module, to
the new plans (design, Section 8): one `REDFT00` plan per axis, extent 1 handled explicitly,
and `Result` instead of a panic. **The removal** deletes the 0.10 module (`Fft`,
`PlanInplace`, `FftPlanInplace`, `C2CPlan`, `R2RPlan`, `FftwPlanFlags`, `C2CDirection`,
`R2RPlanKind`, `FftPrecision`, `FftMutGuard`, `into_fft`), the tests that used it, and its
documentation, and rewrites the module documentation of `rlst::fftw` for the new interface.
**The documentation** gives users the guide in `doc::optional_integrations` with all seven
rules of design Section 5 (DC23), the GPL warning of design Section 6 in its four places, the
docs.rs metadata that makes the FFTW documentation visible on docs.rs, an example for every
public item, and the CHANGELOG entry with the migration notes of Appendix B. **The build**
loses the four build dependencies that no build script uses (DC47). After F7 only
F8a (CI review) and F8b (benchmarks, release) remain.

## Checked while writing this brief

On macOS (Homebrew FFTW 3.3.11, no SIMD) unless stated:

1. **0.10.3's Chebyshev conversions panic at extent 1 and 0.** With today's code,
   `chebychev_coeffs_from_data_second_kind` panics for shapes `[1]` and `[0]`, and
   `chebychev_data_from_coeffs_second_kind` for `[1, 5]`: the 0.10 planner returns `None`
   for REDFT00 with n = 1 (design, Section 3 item 11) and for n = 0, and the conversion
   unwraps it. For n = 1, the scale `1 / ∏(n − 1)` would also be infinite.
2. **Per-axis plans give 0.10.3's bits here.** REDFT00 planned with the new interface
   (`Rigor::Estimate`, in place, along each axis in turn on an aligned copy) and scaled as
   0.10.3 scales, gives coefficients bit-identical to 0.10.3's multidimensional plan for
   `[9]`, `[17]`, `[5, 7]`, `[9, 9]`, `[4, 6, 3]` and `[9, 9, 9]`, f64, seeded data. On a SIMD
   build the plans may differ (design, Section 8), so tests compare to round-off.
3. **Examples.** rustdoc's lint `missing_doc_code_examples` (nightly, with
   `fftw,fftw_system,fftw_threads`) reports 34 items in `rlst/src/fftw*` and
   `rlst/src/chebychev.rs`: 17 of the 0.10 module, which F7b removes, and 17 of
   `chebychev.rs` (its traits and their methods). Every item of the new interface has an
   example.
4. **Who uses the 0.10 module.** `chebychev.rs` (its two conversions); the tests
   `lock::tests::test_lock_recovers_after_panic` (its last part),
   `concurrency_tests::plan_with_the_old_module` (in "planning while executing") and the
   determinism case `estimate-after-other-plans` (two 0.10 plans of each precision); the
   module documentation of `rlst/src/fftw.rs` and the `ignore`d example of
   `doc/optional_integrations.rs`. Removing it leaves no dependency unused: `itertools` and
   `num` are used elsewhere, and `ArrayAtomicSharedRef`, `BorrowAtRuntime` and
   `BorrowAtRuntimeMut` belong to RLST's shared arrays, which stay.
5. **docs.rs.** The build image of docs.rs (`rust-lang/crates-build-env`, `linux/packages.txt`,
   Ubuntu "resolute") lists `clang`, `libclang-dev`, `llvm-dev`, `pkg-config` and
   `libfftw3-dev`, so the concern of design Section 4.10 about `libclang` is settled. A local
   build as docs.rs runs it, `DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly
   doc --no-deps --features "fftw,fftw_system,fftw_threads"`, passes and shows the
   `fftw_threads` badges added by F6b (brief F6, note 11). `rlst` has no
   `[package.metadata.docs.rs]` today, so docs.rs builds without `fftw`.
6. **What needs `libclang`.** `rlst/Cargo.toml` declares the build-dependencies `bindgen`,
   `cc`, `cmake` and `git2`, which `rlst/build.rs` does not use (it uses `pkg-config` only),
   nor did 0.10.3's; `bindgen` is the one that needs `libclang` (open question 4). `cargo
   tree -e build -i <crate>` shows `rlst` as the only crate that pulls in `bindgen` 0.73,
   `cmake` and `git2` (which builds the C libraries `libgit2` and `libz` through
   `libgit2-sys` and `libz-sys`), with and without the FFTW features. With `mpi`, `mpi-sys`
   brings its own `bindgen` 0.72, so `mpi` builds still need `libclang`; `cc` then stays in
   the graph through `mpi-sys` and `libffi-sys`, and without `mpi` it leaves it. `strict`
   denies `unused_crate_dependencies` in `lib.rs`, which does not reach the build script,
   so the lint never saw them.
7. **The licence notes today.** The README's "LICENSE" section says that linking FFTW with
   `fftw_system` means adhering to the GPL; `doc/optional_integrations.rs` has one sentence;
   the comment on the `fftw` feature in `rlst/Cargo.toml` says "GPL-licensed; see the README";
   the module documentation of `rlst::fftw` has none.
8. **Done earlier.** Appendix B's `FFTW_ALIGNED` row (16 to 64) and the feature rows
   (`fftw_source`, `fftw_mkl`, `fftw` without `fftw_system`) are in the CHANGELOG since F1.

## Scope

In scope:

- the Chebyshev port of design Section 8, with an example for each of its public items
  (work item 1; F7a);
- the removal of the 0.10 module, of its uses in tests, and of its documentation; the
  module documentation of `rlst::fftw` for the new interface (work item 2; F7b);
- the user guide, with all seven rules of Section 5 and the notes earlier tasks left for it
  (work item 3; F7c);
- the licence warning of Section 6 in its four places, and the docs.rs metadata (work
  item 4; F7c);
- the removal of the unused build dependencies `bindgen`, `cc`, `cmake` and `git2` (DC47;
  work item 4; F7c);
- the CHANGELOG entry with the migration notes of Appendix B (work item 5; F7c);
- `CLAUDE.md`, the README, the files table of `docs/fftw/README.md`, and this brief's
  "Open questions". The design changes that the open questions brought (draft 0.17) come
  with this brief.

Out of scope, for later tasks or not at all:

- the review of the CI matrix and the default-build check (F8a); benchmarks and the release
  (F8b);
- caching plans in the Chebyshev conversions: they plan on every call, as 0.10.3 did; a
  cache is an addition for later, if nd-fmm needs it;
- renaming `chebychev` (the module, its traits and functions keep their spelling);
- the evaluation and derivative functions of `rlst::chebychev`, which do not use FFTW,
  beyond their examples;
- docs.rs features other than FFTW's (`mpi`, `burn`, SuiteSparse);
- the native packages CI installs: the Linux job installs `libclang-dev` and CMake, of
  which `libclang` stays needed by `mpi-sys` (fact 6) and CMake, as far as `cargo tree`
  shows, by nothing once `cmake` is gone; F8a reviews the install line, and `CLAUDE.md`
  follows it then;
- any change to the API of the new interface. A reason to change it is an open question,
  not an edit.

Files touched:

| File | Change | PR |
| --- | --- | --- |
| `rlst/src/chebychev.rs` | the port; `Result` return types; examples for every public item; tests | F7a |
| `rlst/src/fftw.rs` | the 0.10 items removed; the module documentation rewritten; the licence warning | F7b, F7c |
| `rlst/src/fftw/c2c_inplace.rs`, `r2r_inplace.rs` | deleted | F7b |
| `rlst/src/fftw/lock.rs`, `concurrency_tests.rs` | the uses of the 0.10 module removed; documentation that names it | F7b |
| `rlst/tests/fftw_determinism.rs` | the 0.10 plans of `estimate-after-other-plans` removed | F7b |
| `rlst/src/doc/optional_integrations.rs` | the FFTW guide; the licence warning | F7c |
| `rlst/src/lib.rs` | `doc(cfg(feature = "fftw"))` on `fftw` and `chebychev` | F7c |
| `rlst/Cargo.toml` | `[package.metadata.docs.rs]`; the licence warning on the `fftw` feature; the build dependencies `bindgen`, `cc`, `cmake` and `git2` removed | F7c |
| `README.md` | the licence warning replacing the current note | F7c |
| `rlst/CHANGELOG.md` | the Chebyshev change (F7a); the removal (F7b); the final entry with the migration notes (F7c) | F7a, F7b, F7c |
| `CLAUDE.md` | the 0.10 module gone; the licence warning to keep; the docs.rs build | F7b, F7c |
| `docs/fftw/rlst-fftw-design.md` | draft 0.17 with this brief (DC44 to DC47); during the task, only for a question raised then | — |
| `docs/fftw/README.md` | the files table (with this brief); the status table after F7c merges | — |

## 1. The Chebyshev port (`chebychev.rs`; F7a)

- **Signatures** (design, Section 8; Appendix B): the two methods of
  `ChebychevCoefficientsToData` return `Result<DynArray<Item, NDIM>, FftwError>`. The complex
  implementations (`c32`, `c64`) propagate the errors of the real ones with `?`.
- **One REDFT00 plan per axis** along that axis, batched over the others, in place on the
  result array, applied axis by axis; `Rigor::Estimate`, as 0.10.3. The result array is a
  `DynArray`, as today; the plans are made for the alignment class of its buffer (or on it,
  with the `_on` methods, which do not write the buffer with `Estimate`), so that execution
  never returns `AlignmentMismatch`.
- **Scaling per axis.** The scale `1 / ∏(n − 1)` and the halving of the boundary values
  become per-axis steps: along an axis of extent n > 1, the transform, the factor
  `1 / (n − 1)` and the halving of the first and last index (and the inverse steps in
  `chebychev_data_from_coeffs_second_kind`); along an axis of extent 1, nothing, since one
  Chebyshev point gives a constant interpolant, so data and coefficient are equal along that
  axis (design, Section 8). This keeps the special case a matter of one axis.
- **Dimensions and empty arrays** (DC44): any `NDIM` (the per-axis steps need nothing per
  dimension; today `NDIM > 3` is `unimplemented!()`), `NDIM = 0` returning a copy, and an
  axis of extent 0 gives an empty result of the same shape without planning.
- **Errors.** A plan that FFTW refuses is returned as its `FftwError`, never unwrapped. With
  extent 1 handled and extent 0 not planned, none is expected; the rustdoc says when one can
  occur (an FFTW older than the minimum, `VersionTooOld`).
- **Examples**: every public trait and method of `rlst::chebychev` gets one (fact 3), the
  evaluation and derivative traits included.
- **Tests** (design, Section 7.3): the existing analytic tests, unchanged except for `?` or
  `unwrap()`; extent 1 along each axis in 1, 2 and 3 dimensions, in both directions, against
  the constant interpolant; extent 0; `NDIM = 0` and `NDIM = 4` (DC44); and the comparison
  with 0.10.3 (DC45): a test that compares the port with the 0.10 conversion while both
  exist, to round-off, for 1 to 3 dimensions, both precisions and the complex types, and
  reports where the bits are equal. Its results go into the pull request; F7b removes it.

## 2. The removal (`fftw.rs`, `c2c_inplace.rs`, `r2r_inplace.rs`, tests; F7b)

- **Deleted**: `c2c_inplace.rs` and `r2r_inplace.rs`; in `fftw.rs`, `FftPrecision`,
  `FftwPlanFlags`, `FftPlanInplace`, `PlanInplace`, `Fft`, `FftMutGuard`, `into_fft` and the
  `impl` blocks and tests that belong to them. Nothing of the new interface changes.
- **Tests**: the last part of `test_lock_recovers_after_panic` (planning with the 0.10
  module) and `plan_with_the_old_module` in "planning while executing" go; the 0.10 plans of
  `estimate-after-other-plans` go, and the case keeps its unrelated plans of the new
  interface, so the scenario "planning history" (DC22) still tests what it tested. The
  binary's summary should be unchanged apart from the case's time.
- **Documentation**: the module documentation of `rlst/src/fftw.rs` is rewritten for the new
  interface: what it is, the list of its parts as today's "New interface" section gives
  them, the example, and links to the guide (work item 3) and to the rules on `Plan`. The
  `lock` module documentation and comments that name the 0.10 module are updated. The
  `ignore`d 0.10 example in `doc/optional_integrations.rs` goes (the guide replaces it in
  F7c). The comparison test of F7a (DC45) goes with the module.
- **CLAUDE.md**: its "FFTW rework" section no longer says that the 0.10 module stays until
  F7, and its layout section no longer lists it.

## 3. The user guide (`doc/optional_integrations.rs`; F7c)

The "FFTW transforms" section becomes the guide. It is not feature-gated, so docs.rs always
shows it (design, Section 6). Its examples run as doctests when `fftw` is on and compile to
nothing without it, for example with their body inside a hidden
`# #[cfg(feature = "fftw")] { … }`, since doctests are compiled with the crate's features;
the Burn examples of the same guide are `ignore`d instead, which F7 does not copy. It covers:

- **Setup**: the features (`fftw_system`, which implies `fftw`; `fftw_threads`), the versions
  (3.3.4, or 3.3.6 with `fftw_threads`; 3.3.10 recommended, with the fixes of 3.3.9 and
  3.3.10 of design Section 3 item 9), pkg-config and `PKG_CONFIG_PATH`, and that Homebrew's
  FFTW on Apple silicon has no SIMD, with `--enable-neon` for those who need its speed
  (design, Section 4.10);
- **the licence warning** (work item 4);
- **a worked example** in nd-fmm's shape: a batch of transforms along one axis of an RLST
  array, planned once and executed on several arrays; layouts from arrays and views; the
  families and the input policies, including that a c2r plan destroys its input unless
  `preserve_input` is given; alignment classes and `FFTW_ALIGNED`; scratch and the `*_on`
  methods; scaling with `Scale`; errors;
- **the seven rules of design Section 5** (DC23), each stated briefly and linked to where the
  rustdoc states it in full: rule 1 to `rlst::fftw::threads` (with the feature) and the lock;
  rules 2 to 6 to `Plan` and `Plan::fingerprint`; rule 7 to `rlst::fftw::wisdom`;
- **the notes left for the guide**: compare fingerprints only between plans of one problem
  and one storage form (brief F3, note 7); wisdom calls hold the lock that planning needs, so
  wisdom functions in a tight loop delay planners (brief F6, note 7); the Chebyshev
  conversions and what changed in them.

Examples that plan with `Measure`, or import or forget wisdom, are `no_run` (brief F6, work
item 8); the others run.

## 4. The licence warning, docs.rs and the build dependencies (F7c)

- **The warning** of design Section 6, word for word, in its four places: the guide; the
  module documentation of `rlst::fftw`; the README, replacing the note in its "LICENSE"
  section; and the comment on the `fftw` feature in `rlst/Cargo.toml` (as a comment, in
  shorter form if the full text does not suit a manifest comment, with a pointer to the
  README). `CLAUDE.md`'s rule to keep the README notice names the new warning.
- **docs.rs**: `[package.metadata.docs.rs]` in `rlst/Cargo.toml` with `features =
  ["fftw", "fftw_system", "fftw_threads"]` (DC46) and `rustdoc-args = ["--cfg",
  "docsrs"]`. `pub mod fftw` and `pub mod chebychev` carry
  `#[cfg_attr(docsrs, doc(cfg(feature = "fftw")))]`; the crate-level
  `#![cfg_attr(docsrs, feature(doc_cfg))]` of F6b stays. The pull request reports the local
  build of fact 5 with the final metadata.
- **The build dependencies** (DC47): `bindgen`, `cc`, `cmake` and `git2` go from
  `[build-dependencies]` in `rlst/Cargo.toml`; `pkg-config`, optional, stays. Nothing in
  `rlst/build.rs` changes. The pull request reports `cargo tree -p rlst -e build --depth 1`
  before and after, with and without the FFTW features, and that `rlst`'s build without
  `mpi` no longer pulls in `bindgen`, `cmake`, `git2` or `cc` (fact 6). Cargo updates the
  local `Cargo.lock`, which is ignored; it is not committed.

## 5. The CHANGELOG (F7a, F7b, F7c)

- F7a adds the Chebyshev change: `Result` return types, extent 1 handled, the per-axis
  plans, and that the last bits may differ from 0.10.3 on SIMD builds (design, Section 8).
- F7b adds the removal of the 0.10 module, and folds the entry "FFTW: one lock for all FFTW
  calls", which concerns the 0.10 module, into the FFTW entry.
- F7c turns "New FFTW interface (in progress)" into the final entry: an overview, then the
  migration notes of Appendix B as a table or list at the level of detail of the `dot!` /
  `dot_blas!` entry, including the warning about kinds per axis on column-major arrays
  (design, Section 2.2 item 9); and, outside the FFTW entry, the removal of the unused build
  dependencies, so that a build without `mpi` no longer needs `libclang` or compiles
  `libgit2`.

## Pull requests and CI

- Three pull requests, each branched from `fftw-0.11` after the previous one merged, titled
  for example `feat(fftw): F7a Chebyshev port`, `feat(fftw): F7b remove the 0.10 module` and
  `docs(fftw): F7c guide, licence warning, docs.rs, migration notes`, each with the
  acceptance items it closes ticked.
- The open questions below are settled (DC44 to DC47); a question that arises during the
  task is recorded below and raised in the pull request.
- CI runs for pull requests into `fftw-0.11`. Report the determinism binary's summary from
  both jobs in F7b, which changes a case.

## Acceptance

- [ ] The Chebyshev conversions use one REDFT00 plan per axis of the new interface, return
      `Result<DynArray<Item, NDIM>, FftwError>`, handle extent 1 along any axis, and take
      any number of dimensions and extent 0 (DC44); their tests, the comparison with 0.10.3
      of DC45 included, pass in CI on Linux and macOS.
- [ ] The 0.10 module and its tests are gone; no code, test or documentation of RLST uses
      it; the determinism binary's scenarios pass in CI on Linux and macOS, with and without
      `fftw_threads`.
- [ ] The module documentation of `rlst::fftw` describes the new interface; the guide in
      `doc::optional_integrations` covers work item 3 and states all seven rules of design
      Section 5, linking each to the rustdoc that states it in full.
- [ ] The licence warning of design Section 6 is in its four places, and `CLAUDE.md` names it.
- [ ] `rlst/Cargo.toml` has the docs.rs metadata of DC46, and the docs.rs-style build of
      fact 5 passes with `-D warnings`.
- [ ] `rlst/Cargo.toml` no longer declares `bindgen`, `cc`, `cmake` or `git2` (DC47);
      `cargo tree -p rlst -e build --depth 1` lists only `pkg-config` with the FFTW
      features, and nothing without them; CI passes on Linux and macOS.
- [ ] rustdoc's `missing_doc_code_examples` lint reports no item in `rlst/src/fftw*`,
      `rlst/src/chebychev.rs` or `rlst/src/doc/optional_integrations.rs`.
- [ ] `rlst/CHANGELOG.md` has the final FFTW entry with the migration notes of Appendix B and
      the Chebyshev change.
- [ ] The seven Clippy commands of `CLAUDE.md` pass (or are reported unavailable), as do
      `cargo fmt -- --check`, `cargo test`, `cargo test --features "fftw,fftw_system,strict"`,
      `cargo test --features "fftw,fftw_system,fftw_threads,strict"`, `cargo clippy --tests
      --features "fftw,fftw_system,fftw_threads,strict" -- -D warnings` and `RUSTDOCFLAGS="-D
      warnings" cargo doc --no-deps` with `fftw,fftw_system,strict` and with
      `fftw,fftw_system,fftw_threads,strict`.
- [ ] `CLAUDE.md`, the README and `docs/fftw/README.md` are updated; no dependency was
      added, and `Cargo.lock` is not committed.

## Pitfalls

- **Order.** F7a needs the 0.10 module to compare with; F7b removes it. Do not remove it
  before the Chebyshev port has merged.
- **Alignment of the Chebyshev result.** A `DynArray` is not necessarily at alignment class
  0; a plan made on scratch of class 0 then fails with `AlignmentMismatch`. Plan for the
  class of the result's buffer, or on it with `Estimate`.
- **The scaling at extent 1.** `1 / (n − 1)` is infinite at n = 1, and indices 0 and n − 1
  coincide, so halving "both ends" halves the one value twice. Treat such an axis as the
  identity.
- **Bits.** Per-axis plans gave 0.10.3's bits on macOS (fact 2); on a SIMD build they may
  not. Tests compare to round-off and report bits.
- **Guide doctests and wisdom.** The guide's doctests run in the doctest processes; they plan
  with `Estimate` only, and examples that import or forget wisdom or plan with `Measure` are
  `no_run` (brief F6, work item 8).
- **The guide without `fftw`.** `doc::optional_integrations` is always compiled; its FFTW
  examples must not fail the doctests of a build without `fftw` (as `cargo test` runs them).
- **The warning's wording.** Word for word as design Section 6 gives it; it is a design
  text, and a change to it goes into the design with an Appendix A entry.
- **The build dependencies and `libclang`.** Their removal frees builds without `mpi` from
  `libclang`, not every build: `mpi-sys` needs it (fact 6). The guide, the README and the
  CHANGELOG must not say that RLST needs no `libclang`; CI keeps installing it.
- **`strict`, wildcard imports, debug builds**: as in the earlier briefs.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design questions
in code.

Raised while writing this brief:

1. **Chebyshev dimensions and empty arrays.** Design Section 8 handles extent 1 and says
   nothing of extent 0 or of `NDIM > 3`, which 0.10.3 rejects with `unimplemented!()` and
   panics on (fact 1). Options: (a) *proposed*: any `NDIM`, the per-axis steps needing
   nothing per dimension (`NDIM = 0` returns a copy), and an axis of extent 0 gives an empty
   result of the same shape, without planning; (b) 1 to 3 dimensions as today, with
   `NDIM > 3` and extent 0 returned as `FftwError::InvalidLayout` instead of a panic; (c) as
   (b), with the panics kept for those cases. *Decided: (a), DC44.*
2. **The comparison with 0.10.3** (design, Section 7.3). Once the 0.10 module is gone, no
   test can run 0.10.3's conversions. Options: (a) *proposed*: F7a adds a test that compares
   the port with the 0.10 conversion while both exist, to round-off, reporting where the bits
   are equal, and its results are in the pull request; F7b removes that test with the 0.10
   module, and the lasting regressions are the analytic tests and the extent-1 tests;
   (b) as (a), and F7a also stores 0.10.3's outputs for a few inputs as constants that later
   tests compare with to round-off; (c) no comparison with 0.10.3 beyond the analytic tests
   (design Section 7.3 corrected). *Decided: (a), DC45.*
3. **The docs.rs features.** Design Section 6 lists `features = ["fftw", "fftw_system"]`;
   the items F6b gated by `fftw_threads` then do not appear on docs.rs (brief F6, work item
   8). Options: (a) *proposed*: `["fftw", "fftw_system", "fftw_threads"]`, design Section 6
   corrected with an Appendix A entry; (b) as the design says, without `fftw_threads`.
   *Decided: (a), DC46.*
4. **The unused build-dependencies.** `bindgen`, `cc`, `cmake` and `git2` are declared but
   not used by `rlst/build.rs` (fact 6); every build of `rlst` compiles them, and `bindgen`
   needs `libclang`, which design Section 4.10 records as a docs.rs requirement. docs.rs has
   `libclang` (fact 5), so F7 does not need their removal. Options: (a) *proposed*: not F7;
   recorded here, for a separate pull request into `main` outside the FFTW rework, since they
   concern every build and not FFTW; (b) F7c removes them; (c) they stay. *Decided: (b),
   DC47: F7c removes them (work item 4), and design Section 4.10 no longer says that docs.rs
   must provide `libclang` for them. They reach `main` with the 0.11.0 release.*
