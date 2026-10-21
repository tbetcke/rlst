# F0: Baseline, integration branch and CI reports

| | |
| --- | --- |
| Task | F0 (design, Section 9) |
| Status | ready |
| Depends on | nothing |
| Pull requests | one, into `fftw-0.11` |
| Read first | design: the baseline note at the top, Section 2.4, Section 7.7, Section 9 (F0 row, Transition, Briefs); `docs/fftw/README.md`; `CLAUDE.md` |

## Goal

Start the rework from a known state: the integration branch exists, the baseline passes,
CI shows which FFTW each runner uses, a macOS job runs the FFTW tests from the first
commit, and the findings are written down for the later tasks.

## Scope

In scope:

- the `fftw-0.11` branch;
- `.github/workflows/run-tests.yml`: a report of the FFTW version on each runner, and a
  macOS job (brought forward from F8a);
- `CLAUDE.md`, kept in step with the workflow;
- `docs/fftw/F0-findings.md` (new) and the status table of `docs/fftw/README.md`.

Out of scope: any change under `rlst/`, `proc-macro/` or `rlst-suitesparse/`. F0 changes no
code. If the baseline fails for a reason unrelated to FFTW, record it in the findings and
raise it; do not fix it in this task.

## Work items

### 1. The integration branch

Create `fftw-0.11` from `main`. Use tag v0.10.3 if `main` has not moved since; otherwise use
the current `main` and record its commit in the findings. Push it. All later FFTW pull
requests target this branch.

### 2. The baseline, locally

With a system FFTW installed (`docs/fftw/README.md`, "Local setup"), run from the repository
root and record the results:

```bash
rustc --version
cargo fmt -- --check
cargo test
cargo test --features "fftw,fftw_system,strict"
cargo clippy --features "fftw,fftw_system,strict" -- -D warnings
```

Run the remaining Clippy commands of `CLAUDE.md` where their native dependencies are
available, and report the others as unavailable, never as passing.

### 3. Report the FFTW version in CI

In `run-tests.yml`, after the step that installs the native dependencies, add a step that
prints the versions pkg-config reports, the threads libraries present beside FFTW (task F6
links them), and FFTW's own version string, which includes the SIMD extensions of the build:

```yaml
      - name: Report FFTW
        run: |
          pkg-config --modversion fftw3 fftw3f
          ls "$(pkg-config --variable=libdir fftw3)" | grep -E '^libfftw3f?_(threads|omp)' || true
          printf '#include <fftw3.h>\n#include <stdio.h>\nint main(void){puts(fftw_version);return 0;}\n' > "$RUNNER_TEMP/fv.c"
          cc "$RUNNER_TEMP/fv.c" $(pkg-config --cflags --libs fftw3) -o "$RUNNER_TEMP/fv"
          "$RUNNER_TEMP/fv"
```

On Ubuntu 24.04 the last line is expected to print `fftw-3.3.10-sse2-avx` (design, Section 3).

### 4. A macOS job

Add a second job to `run-tests.yml` on a current arm64 macOS runner, pinned to an explicit
image label as the Linux job is (for example `macos-15`; use the newest label GitHub
supports at the time). Steps:

1. install Rust stable with `clippy` and `rustfmt`, check out, and cache as the Linux job does;
2. `brew install fftw pkg-config`;
3. the "Report FFTW" step of work item 3;
4. `cargo clippy --features "fftw,fftw_system,strict" -- -D warnings`;
5. `cargo test --features "fftw,fftw_system,strict"`.

No MPI, SuiteSparse or OpenBLAS: on macOS the tests link BLAS and LAPACK through Accelerate
(`rlst/Cargo.toml`, target-specific dev-dependencies). If tests unrelated to FFTW fail on
macOS, record them in the findings and limit the job's test step to the FFTW and Chebyshev
tests (`cargo test --features "fftw,fftw_system,strict" fftw` and the same with `chebychev`)
until they are fixed elsewhere.

### 5. CLAUDE.md

Keep `CLAUDE.md` in step with `run-tests.yml`, which it calls authoritative:

- in "Repository layout", the `.github/workflows/` entry mentions the macOS job;
- in "Features and native dependencies", add that CI on macOS installs FFTW and pkg-config
  with Homebrew and runs the FFTW tests;
- in "Validation commands", list the macOS job's commands.

### 6. The findings

Write `docs/fftw/F0-findings.md` with:

- the base commit of `fftw-0.11` and the `rustc` version;
- the results of work item 2, command by command;
- for each CI runner (Ubuntu and macOS): the pkg-config versions of `fftw3` and `fftw3f`, the
  FFTW version string, and which `_threads` and `_omp` libraries exist in pkg-config's
  `libdir`;
- what these mean for later tasks: whether every runner meets the minimum of DF8 (3.3.4,
  and 3.3.6 for `fftw_threads`), whether F6 can link the threads libraries from pkg-config's
  `libdir` on both runners, and whether Section 7.7's assumption of FFTW 3.3.10 on
  ubuntu-24.04 holds;
- anything unexpected.

### 7. Status

Set F0 to "done" in the status table of `docs/fftw/README.md`, with the pull request.

## Acceptance

- [ ] `fftw-0.11` exists, based on v0.10.3 or on a later `main` commit named in the findings.
- [ ] CI on the branch runs the Linux job (unchanged apart from the FFTW report) and the
      macOS job, and both pass.
- [ ] The CI logs of both jobs show the pkg-config versions, the threads libraries and the
      FFTW version string.
- [ ] `docs/fftw/F0-findings.md` contains every item of work item 6.
- [ ] `CLAUDE.md` matches `run-tests.yml`.
- [ ] No file under `rlst/`, `proc-macro/` or `rlst-suitesparse/` changed.
- [ ] The status table of `docs/fftw/README.md` is updated.

Suggested commit message: `ci(fftw): report FFTW versions and add a macOS job`.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design
questions in code or CI.

1. **FFTW 3.3.11.** Section 2.4 says Homebrew ships 3.3.10, and Section 4.10 and DF8 call
   3.3.10 the version that CI tests on Linux and macOS. Homebrew now ships 3.3.11 (see
   `../F0-findings.md`), so the macOS job tests 3.3.11. Does the design need to change: the
   version Section 4.10 recommends, Section 3 item 9's list of fixes, and Appendix A?
   **Resolved** in design draft 0.7: 3.3.10 or later is recommended; CI tests 3.3.10 on Linux
   and 3.3.11 on macOS.
2. **Homebrew's FFTW on arm64 has no SIMD**, so `fftw_alignment_of` is always 0 there. Section
   4.6 is still sufficient. Later briefs must keep tests that depend on SIMD codelets or on
   `fftw_alignment_of` values off the macOS runner, or gate them. See `../F0-findings.md`.
   **Resolved** in design draft 0.7: SIMD-dependent tests skip on builds without SIMD, read
   from the version string (Sections 3 item 6 and 7.7). The documentation tells Apple
   silicon users how to build FFTW with NEON (Section 4.10).
3. **Baseline commit.** The design's baseline note gives v0.10.3 as `02eb2ec`, but the tag
   points to `1f5ab87`. The note should be corrected, with an entry in Appendix A.
   **Resolved** in design draft 0.7: the baseline is v0.10.3 at `1f5ab87`. `02eb2ec`
   followed it on `main`, and so did the design and the version 0.11.0-dev. `fftw-0.11`
   starts from that `main`.
