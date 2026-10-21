# F0 findings: baseline and the FFTW on each CI runner

Written by task F0 (`tasks/F0-baseline.md`). Later briefs read this file.

## Base of `fftw-0.11`

`main` has moved since tag v0.10.3, so `fftw-0.11` starts from the current `main`:

| | Commit | |
| --- | --- | --- |
| tag v0.10.3 | `1f5ab87` | Merge pull request #1 from tbetcke/chore/move-to-github |
| `02eb2ec` | | ci: remove the run-examples and run-dependency-checks workflows |
| `b415f19` | | Added fftw design documents |
| **base of `fftw-0.11`** | **`dc572e5`** | chore: set workspace version to 0.11.0-dev |

The commits after v0.10.3 change CI, add `docs/fftw/` and set the workspace version to
0.11.0-dev. None of them touches FFTW code, so the FFTW code of the base is that of v0.10.3.

## Local baseline

Machine: macOS 27.0.1 on arm64 (Apple silicon). Toolchain: `rustc 1.99.0 (b940084d7
2026-09-28)`, `cargo 1.99.0 (5f94df478 2026-08-27)`. FFTW from Homebrew: pkg-config reports
3.3.11 for `fftw3` and `fftw3f`, and `fftw_version` and `fftwf_version` are both
`fftw-3.3.11`. MPI (Homebrew Open MPI) and SuiteSparse were also installed, so every Clippy
command of `CLAUDE.md` could run.

The commands were run at `dc572e5`, from the repository root.

| Command | Result |
| --- | --- |
| `cargo fmt -- --check` | pass |
| `cargo test` | pass: 293 unit tests, 27 integration tests (`blas_threads` 0, `dot` 17, `static_type_macro` 1, `zero_dim_array` 9), 74 doctests (2 ignored) and 3 more doctests |
| `cargo test --features "fftw,fftw_system,strict"` | pass: 321 unit tests (28 more than the default, the FFTW and Chebyshev tests), the same 27 integration tests, 76 doctests (2 ignored) and 4 more doctests |
| `cargo clippy --features "fftw,fftw_system,strict" -- -D warnings` | pass |
| `cargo clippy -- -D warnings` | pass |
| `cargo clippy --examples -- -D warnings` | pass |
| `cargo clippy --features "mpi,fftw,fftw_system,strict" -- -D warnings` | pass |
| `cargo clippy --examples --features "mpi,fftw,fftw_system,strict" -- -D warnings` | pass |
| `cargo clippy --features "openblas_threading,mkl_threading,blis_threading,strict" -- -D warnings` | pass |
| `cargo clippy -p rlst-suitesparse -- -D warnings` | pass |

No failures, so nothing unrelated to FFTW needs to be raised, and the macOS job runs the
whole test suite rather than only the FFTW and Chebyshev tests.

## FFTW on the CI runners

From the "Report FFTW" step of each job in `run-tests.yml`, on the first CI run of the
F0 pull request.

| | ubuntu-24.04 (`run-tests`) | macos-26, arm64 (`run-tests-macos`) |
| --- | --- | --- |
| pkg-config `fftw3` | _to be filled from the CI log_ | _to be filled from the CI log_ |
| pkg-config `fftw3f` | _to be filled from the CI log_ | _to be filled from the CI log_ |
| `fftw_version` | _to be filled from the CI log_ (expected `fftw-3.3.10-sse2-avx`) | _to be filled from the CI log_ |
| `_threads` in pkg-config's `libdir` | _to be filled from the CI log_ | _to be filled from the CI log_ |
| `_omp` in pkg-config's `libdir` | _to be filled from the CI log_ | _to be filled from the CI log_ |

For comparison, the local Homebrew installation (arm64, FFTW 3.3.11) has
`libfftw3_threads`, `libfftw3f_threads`, `libfftw3_omp` and `libfftw3f_omp`, as `.dylib` and
`.a`, in pkg-config's `libdir` (`/opt/homebrew/Cellar/fftw/3.3.11/lib`). The macOS runner
installs the same Homebrew formula and is expected to match.

## What this means for later tasks

- **Minimum versions (DF8).** Locally FFTW 3.3.11 meets 3.3.4 and, for `fftw_threads`, 3.3.6.
  _For each runner: to be confirmed from the CI log._
- **Threads libraries (F6).** Locally the `_threads` libraries are in pkg-config's `libdir`,
  so F6 can link them from there on macOS. _For Ubuntu: to be confirmed from the CI log._
- **Section 7.7's FFTW 3.3.10 on ubuntu-24.04.** _To be confirmed from the CI log._

## Unexpected

1. **Homebrew ships FFTW 3.3.11, not 3.3.10.** Section 2.4 says Homebrew ships 3.3.10, and
   Section 4.10 and DF8 call 3.3.10 "the version CI tests on Linux and macOS". Locally
   Homebrew installs 3.3.11, and the macOS runner will install whatever the formula is when
   the job runs. The design says nothing about 3.3.11's changes. This is recorded as an open
   question in `tasks/F0-baseline.md`.
2. **Homebrew's FFTW on arm64 has no SIMD.** Homebrew's formula passes the SIMD options
   (`--enable-sse2 --enable-avx --enable-avx2`) only on Intel. On arm64 the version string
   has no SIMD suffix (`fftw-3.3.11`), and `fftw_alignment_of` and `fftwf_alignment_of`
   return 0 for offsets of 0, 1, 2 and 3 elements from an `fftw_malloc` pointer (Ubuntu's
   gives 0, 8, 0 for offsets of 0, 1, 2 doubles, Section 3 item 6). Consequences:
   - Section 4.6's rule (equal classes modulo 64) is still sufficient. It is only stricter
     than this build needs. Nothing in the design has to change.
   - A test that expects `fftw_alignment_of` to tell offsets apart, or that expects SIMD
     codelets in a plan's `fftw_sprint_plan` output (Section 3 item 7), fails on the macOS
     runner. Such tests must be gated on the version string, or run on Linux only.
   - On macOS, the alignment check of F1 (Section 4.8) guards against a condition that this
     FFTW build cannot violate. The tests of that check are still valid, because the check
     is RLST's own and makes no FFI call.
3. **The design's baseline note names the wrong commit.** The note at the top of the design
   gives v0.10.3 as commit `02eb2ec`. The tag v0.10.3 points to `1f5ab87`. `02eb2ec` is the
   next commit on `main`, which removed two CI workflows.
4. **CI did not run on pull requests into `fftw-0.11`.** `run-tests.yml` ran only for pushes
   and pull requests to `main`. F0 adds `fftw-0.11` to both triggers, so that pull requests
   into the integration branch, and pushes to it, run CI. The Linux job itself is unchanged
   apart from the FFTW report.
