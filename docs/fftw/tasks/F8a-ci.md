# F8a: CI on Linux and macOS, the default-build check, and the documentation builds

| | |
| --- | --- |
| Task | F8a (design, Section 9) |
| Status | done |
| Depends on | F0 to F7 |
| Pull requests | one, into `fftw-0.11`: `ci(fftw): F8a CI matrix, default-build check, documentation builds` |
| Read first | design (draft 0.19): Section 4.10 (versions, docs.rs), Section 6 (the docs.rs metadata), Section 7.7 (CI, the F8a item and the builds without SIMD), Section 9 (F8a row), Section 10 (DF8, DC43, DC46, DC47), Section 11 item 5 (the default-build check for nd-fmm); `docs/fftw/F0-findings.md` (the runners); the "Open questions" of `docs/fftw/tasks/F6-wisdom-threads.md` (note 11, `doc(cfg)`) and `F7-removal-docs.md` (facts 5 and 6); `.github/workflows/run-tests.yml`; `CLAUDE.md` ("Validation commands") |

## Goal

Every part of the new interface exists since F7; F8a makes CI check it as the release will
ship it. Design Section 7.7 lists what F8a adds to the jobs that F0 and F6 built: the
**default build**, with a check that it does not pull in fftw-sys, **doctests**, and
**`cargo doc` with `--cfg docsrs`**, and it asks F8a to review the matrix as a whole
(DC43). Since F7 the default build has more to prove than before: the guide in
`doc::optional_integrations` is always compiled, and its FFTW examples must compile to
nothing without `fftw`; and `rlst` no longer has build dependencies that need `libclang` or
CMake outside `mpi` (DC47), which the install line can reflect. After F8a only F8b
(benchmarks and the release) remains.

## Checked while writing this brief

From the CI run of the merge of #34 into `fftw-0.11` (run 40126922306, `rustc` stable
1.99.1) and a local macOS checkout of `fftw-0.11` at `39780fb`, unless stated:

1. **The matrix today.** `run-tests.yml` has two jobs, triggered by pushes and pull requests
   to `main` and `fftw-0.11`:

   | Job | Runner | Installs | Commands |
   | --- | --- | --- | --- |
   | `run-tests` | `ubuntu-24.04`, FFTW `fftw-3.3.10-sse2-avx` | `libclang-dev cmake libfftw3-dev libopenblas-dev openmpi-bin libopenmpi-dev libsuitesparse-dev` | the FFTW report; `cargo fmt -- --check`; the seven Clippy commands of `CLAUDE.md`; `cargo test` with `mpi,fftw,fftw_system,strict` (with `RUST_MIN_STACK`), `-p rlst-suitesparse`, and `fftw,fftw_system,fftw_threads,strict`; `scripts/run_mpi_tests.py`; `cargo doc --no-deps --features "mpi,fftw,fftw_system,strict"` |
   | `run-tests-macos` | `macos-26` arm64, Homebrew FFTW `fftw-3.3.11` without SIMD | `fftw pkg-config` | the FFTW report; Clippy and `cargo test` with `fftw,fftw_system,strict` and with `fftw,fftw_system,fftw_threads,strict` |

   Times per step, with a warm cache: Linux 27 s installing packages, 14 s Clippy, 51 s
   tests, 24 s MPI regressions, 1 s docs (about 2.3 minutes for the job); macOS 8 s Clippy
   and 50 s tests (about 1.3 minutes).
2. **What CI does not check.**
   - The default configuration: no job runs `cargo test` without features. The default
     build, its doctests, and the guide's FFTW examples compiled without `fftw` (F7c) are
     checked only locally.
   - That the default build has no fftw-sys (design Section 7.7; Section 11 item 5).
     Locally, `cargo tree -p rlst -e normal --prefix none` lists no `fftw-sys`; `libc` is
     there, through other dependencies.
   - Warnings in the documentation: `cargo doc` runs without `-D warnings`, and only with
     `mpi,fftw,fftw_system,strict`. The guide's links into `rlst::fftw` are checked only in
     builds with `fftw_threads` (F7c, a `cfg_attr` allow of `broken_intra_doc_links`
     elsewhere), and no CI build has that.
   - The docs.rs build: `DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly
     doc --no-deps --features "fftw,fftw_system,fftw_threads"`, which F7c ran locally only,
     with a nightly of 2026-04-12. `CLAUDE.md` lists it as "not in CI until task F8a".
   - `cargo clippy --tests`, which F7's acceptance ran locally with and without
     `fftw_threads`; the Clippy commands of CI lint the library and the examples only.
3. **What the install line needs.** `cargo tree --target x86_64-unknown-linux-gnu -e
   build,normal,dev -i <crate>`: `bindgen` 0.72 and `clang-sys` are in the graph of `rlst`
   with `mpi` (through `mpi-sys`), and in no other configuration CI builds, so `libclang-dev`
   stays. The `cmake` crate is in no configuration since F7c removed it from `rlst`'s
   build dependencies, `rlst-suitesparse` included. Whether a native build calls the
   `cmake` program without the crate cannot be read from `cargo tree`; a CI run without the
   package tells.
4. **Doctests already run.** Every `cargo test` runs the doctests of its configuration:
   with `fftw` the guide's FFTW examples and every example of `rlst::fftw` and
   `rlst::chebychev`. Without features they compile to nothing (F7c), which nothing in CI
   runs (fact 2).
5. **Required checks.** `main` is protected and requires the status check `run-tests`
   only; `run-tests-macos` is not required, and force pushes are not allowed. `fftw-0.11`
   is not protected. Required checks are a repository setting, which only the maintainer
   changes.
6. **The runners** are those of F0: Ubuntu 24.04 with FFTW 3.3.10 and SIMD, macOS 26 on
   arm64 with Homebrew's FFTW 3.3.11 without SIMD, so CI covers a build with SIMD and one
   without (design Section 7.7). Neither tests the minimum versions 3.3.4 and 3.3.6, which
   DF8 settles: 3.3.10 and 3.3.11 are tested.
7. **docs.rs.** The metadata of F7c (`features = ["fftw", "fftw_system", "fftw_threads"]`,
   `rustdoc-args = ["--cfg", "docsrs"]`) and `build.rs`, which skips FFTW discovery when
   `DOCS_RS` is set, mean that the docs.rs build needs neither FFTW nor pkg-config: rustdoc
   does not link. A CI job that sets `DOCS_RS=1` and installs no FFTW reproduces that.

## Scope

In scope:

- the review of the CI matrix of `run-tests.yml` as a whole (DC43), and the additions of
  design Section 7.7 for F8a: the default build and its check, doctests in the default
  configuration, `cargo doc` with `-D warnings` and with `--cfg docsrs` (work items 1 to 3);
- the install line of the Linux job (work item 4);
- `CLAUDE.md`'s CI description and validation list, which follow `run-tests.yml`
  (work item 5);
- this brief's "Open questions".

Out of scope, for F8b or not at all:

- benchmarks and the release (F8b);
- testing FFTW's minimum versions 3.3.4 and 3.3.6 in CI: DF8 settles that 3.3.10 and 3.3.11
  are tested;
- other operating systems, architectures or Rust toolchains than those of today's jobs,
  apart from the nightly that the docs.rs build needs;
- the `burn` feature, which no CI job enables today;
- changing required checks of `main`, a repository setting; the pull request recommends
  what the maintainer could set (DC52);
- any change to the code of `rlst`. A failure that CI finds is reported in the pull
  request and fixed there only if it is a fault of the documentation or the build
  configuration; anything else is raised first.

Files touched:

| File | Change |
| --- | --- |
| `.github/workflows/run-tests.yml` | the steps of work items 1 to 4, and the `docs` job (DC48) |
| `CLAUDE.md` | the CI description in "Repository layout" and the validation list, in step with `run-tests.yml` |
| `docs/fftw/tasks/F8a-ci.md` | "Open questions" |
| `docs/fftw/README.md` | the status table, after the pull request merges |

## 1. The default build (Linux job)

- `cargo test` without features, which runs the default unit tests, integration tests and
  doctests, the guide's FFTW examples compiled to nothing included.
- The default-build check of design Section 7.7: a step that fails if
  `cargo tree -p rlst -e normal --prefix none` lists `fftw-sys`, with a message that names
  the check. It checks the resolved graph of the default build, not the lock file (design
  Section 11 item 5).
- The macOS job does not run the default configuration (DC50).

## 2. Lints and tests in the FFTW configurations

- `cargo clippy --tests -- -D warnings` with `fftw,fftw_system,strict` and with
  `fftw,fftw_system,fftw_threads,strict`, in both jobs, as F7's acceptance ran them.
- The test commands stay as they are: both jobs test with and without `fftw_threads` (DC43),
  the Linux job with `mpi` as well. The determinism binary runs in each `cargo test` with
  `fftw`, and its summary stays in the log.

## 3. The documentation builds

- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` with `mpi,fftw,fftw_system,strict`, as
  today but with `-D warnings`, and with `fftw,fftw_system,fftw_threads,strict`, which
  checks the guide's links into `rlst::fftw`.
- The docs.rs build: `DOCS_RS=1 RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo +nightly doc
  --no-deps --features "fftw,fftw_system,fftw_threads"`, without FFTW installed (fact 7).
  It runs in a third job, `docs`, on `ubuntu-24.04` with the current nightly and no native
  packages (DC48, DC49).

## 4. The install line (Linux job)

- `libclang-dev` stays: `mpi-sys` needs it (fact 3).
- `cmake` goes if the job passes without it (DC51). The pull request reports
  the run.

## 5. `CLAUDE.md`

- The CI description in "Repository layout" names the jobs and what each runs.
- The validation list follows `run-tests.yml` command for command; the docs.rs build moves
  from "not in CI until task F8a" to the list of CI commands.

## Pull requests and CI

- One pull request into `fftw-0.11`, titled `ci(fftw): F8a CI matrix, default-build check,
  documentation builds`, with the acceptance list ticked.
- The pull request reports the job times before (fact 1) and after, and the result of the
  run without `cmake`.

## Acceptance

- [ ] CI runs `cargo test` in the default configuration, and fails if the default build of
      `rlst` lists `fftw-sys` (`cargo tree -p rlst -e normal`).
- [ ] CI runs `cargo clippy --tests -- -D warnings` with `fftw,fftw_system,strict` and with
      `fftw,fftw_system,fftw_threads,strict`, on Linux and macOS.
- [ ] CI builds the documentation with `-D warnings` with `mpi,fftw,fftw_system,strict` and
      with `fftw,fftw_system,fftw_threads,strict`, and builds it as docs.rs does
      (`DOCS_RS=1`, nightly, `--cfg docsrs`, the features of DC46) in the `docs` job,
      without FFTW installed (DC48).
- [ ] The Linux job's install line has only what the build needs: `libclang-dev` for
      `mpi-sys`, and `cmake` only if a run without it fails.
- [ ] Both jobs and any new job pass on the pull request; the determinism binary's
      scenarios pass on Linux and macOS, with and without `fftw_threads`.
- [ ] `CLAUDE.md`'s CI description and validation list match `run-tests.yml`, and the pull
      request recommends the required checks of DC52.
- [ ] No dependency was added, no code of `rlst` changed, and `Cargo.lock` is not
      committed.

## Pitfalls

- **The lock file is not the build.** Cargo resolves `Cargo.lock` as if every feature of
  every workspace member were on, so `fftw-sys` is in the lock file; the check reads the
  graph of the default build with `cargo tree -e normal` (design Section 11 item 5).
- **`cargo tree` output.** Use `--prefix none` and match the package name at the start of
  a line, so that a crate whose name contains `fftw-sys` or a feature list does not match.
- **Nightly.** A nightly can break the docs.rs build for reasons outside RLST, as
  `doc(cfg)` did before F6b added the crate-level gate (brief F6, note 11). The job's
  failure message should make clear that only the nightly build failed.
- **`DOCS_RS` and the cache.** A job with `DOCS_RS=1` builds `build.rs` in another mode;
  sharing the rust-cache of the test jobs is harmless but gains little.
- **Time.** The Linux job is the longer one; adding the default `cargo test` and the extra
  Clippy and doc builds to it adds compilations of other feature sets. The pull request
  reports the times, and a split into further jobs is an option if it grows much.
- **`strict`, wildcard imports**: as in the earlier briefs; F8a changes no Rust code.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design questions
in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.19), each with the proposed option:

1. **Where the docs.rs build runs.** Options: (a) *proposed*: a third job, `docs`, on
   `ubuntu-24.04` with the nightly toolchain, which sets `DOCS_RS=1` and installs no native
   packages, so that it builds the documentation as docs.rs does, without FFTW, and fails
   on its own; (b) a step of the Linux job, with a second toolchain and FFTW installed;
   (c) not in CI, run by hand before the release.
   *Decided: (a), DC48.*
2. **Which nightly.** Options: (a) *proposed*: the current nightly, which is what docs.rs
   uses, and a failure caused by the nightly is fixed or raised when it occurs; (b) a
   dated nightly, updated by hand.
   *Decided: (a), DC49.*
3. **The default configuration on macOS.** Options: (a) *proposed*: the Linux job only runs
   `cargo test` without features, since the default build has no native dependency that
   differs between the runners beyond BLAS and LAPACK; (b) both jobs run it, which also
   tests the default build against Accelerate on CI.
   *Decided: (a), DC50.*
4. **CMake.** Options: (a) *proposed*: remove `cmake` from the install line if the job
   passes without it, and record the run in the pull request; (b) keep it until the
   release.
   *Decided: (a), DC51.*
5. **Required checks of `main`.** `main` requires `run-tests` only (fact 5). Options:
   (a) *proposed*: the pull request recommends that the maintainer also require
   `run-tests-macos` (and `docs`, under 1 (a)) on `main` before the release merge of F8b;
   (b) leave the required checks as they are.
   *Decided: (a), DC52.*
