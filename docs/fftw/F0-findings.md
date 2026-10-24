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

From the "Report FFTW" step of each job in `run-tests.yml`, in the first CI run of the F0
pull request ([#2](https://github.com/tbetcke/rlst/pull/2), run 39675711877). Both jobs
passed, with `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.

| | ubuntu-24.04 (`run-tests`) | macos-26, arm64 (`run-tests-macos`) |
| --- | --- | --- |
| Runner image | `ubuntu-24.04`, version 20261018.339.1 | `macos-26-arm64`, version 20261013.0424.1 |
| pkg-config `fftw3` | 3.3.10 | 3.3.11 |
| pkg-config `fftw3f` | 3.3.10 | 3.3.11 |
| `fftw_version` | `fftw-3.3.10-sse2-avx` | `fftw-3.3.11` (no SIMD) |
| `_threads` in pkg-config's `libdir` | `libfftw3_threads` and `libfftw3f_threads` (`.so`, `.so.3`, `.so.3.6.10`, `.a`) | `libfftw3_threads` and `libfftw3f_threads` (`.dylib`, `.3.dylib`, `.a`) |
| `_omp` in pkg-config's `libdir` | `libfftw3_omp` and `libfftw3f_omp` (`.so`, `.so.3`, `.so.3.6.10`, `.a`) | `libfftw3_omp` and `libfftw3f_omp` (`.dylib`, `.3.dylib`, `.a`) |
| Tests | full CI configuration (`mpi,fftw,fftw_system,strict`), all pass | `fftw,fftw_system,strict`, the whole suite: 321 unit tests, 27 integration tests, 76 doctests (2 ignored) and 4 more, all pass |

The macOS runner matches the local Homebrew installation, apart from the toolchain.

## What this means for later tasks

- **Minimum versions (DF8).** Both runners meet 3.3.4, and 3.3.6 for `fftw_threads`:
  Ubuntu has 3.3.10, macOS 3.3.11.
- **Threads libraries (F6).** On both runners, `libfftw3_threads` and `libfftw3f_threads`
  are in pkg-config's `libdir` for `fftw3`, as shared and static libraries. F6 can link them
  from there, as Section 4.10 plans. The `_omp` libraries are there too, and F6 does not
  link them.
- **Section 7.7's FFTW 3.3.10 on ubuntu-24.04** holds: `fftw-3.3.10-sse2-avx`, as Section 3
  expects. macOS tests 3.3.11 without SIMD (design draft 0.7).

## Unexpected

1. **Homebrew ships FFTW 3.3.11, not 3.3.10.** Section 2.4 says Homebrew ships 3.3.10, and
   Section 4.10 and DF8 call 3.3.10 "the version CI tests on Linux and macOS". Locally
   Homebrew installs 3.3.11, and the macOS runner will install whatever the formula is when
   the job runs. FFTW's NEWS lists no correctness fixes for 3.3.11; it adds `fftw_copy_plan`,
   SVE and LoongArch SIMD, and cycle counters for Apple ARM and RISC-V. Resolved in design
   draft 0.7: 3.3.10 or later is recommended, and CI tests 3.3.10 on Linux and 3.3.11 on
   macOS.
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

   Resolved in design draft 0.7. The macOS job keeps testing Homebrew's FFTW as users get it.
   A test that depends on SIMD skips on a build without it, read from the version string
   (Sections 3 item 6 and 7.7). The documentation tells Apple silicon users how to build
   FFTW with NEON (Section 4.10).
3. **The design's baseline note names the wrong commit.** The note at the top of the design
   gives v0.10.3 as commit `02eb2ec`. The tag v0.10.3 points to `1f5ab87`. `02eb2ec` is the
   next commit on `main`, which removed two CI workflows. It was followed by the design
   documents and by `dc572e5`, which sets the workspace version to 0.11.0-dev. Corrected in
   design draft 0.7.
4. **CI did not run on pull requests into `fftw-0.11`.** `run-tests.yml` ran only for pushes
   and pull requests to `main`. F0 adds `fftw-0.11` to both triggers, so that pull requests
   into the integration branch, and pushes to it, run CI. The Linux job itself is unchanged
   apart from the FFTW report.
