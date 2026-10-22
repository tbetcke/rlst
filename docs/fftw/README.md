# FFTW rework for RLST 0.11

This directory holds everything needed to implement the new FFTW interface of RLST 0.11.

| File | What it is |
| --- | --- |
| `rlst-fftw-design.md` | The design, draft 0.10. Authoritative for the new interface; its Section 10 records the settled decisions and design choices. |
| `tasks/F0-baseline.md` | Brief for task F0: integration branch, baseline, CI version reports, a macOS job. |
| `tasks/F1-core.md` | Brief for task F1: the core of the new interface, with c2c plans on raw layouts and slices. |
| `tasks/F2-layouts.md` | Brief for task F2: strided storage of RLST arrays and views, layouts from them, and execution on them. |
| `F0-findings.md` | Written by task F0: the baseline and the FFTW found on each CI runner. |

Briefs for F2 to F8b are written when their task is next, from Section 9 of the design and
from what the earlier tasks found.

## Workflow

1. **Branch.** All work goes into the `fftw-0.11` branch, created by F0. `main` keeps
   receiving 0.10.x patch releases. `fftw-0.11` is merged into `main` only for the 0.11.0
   release, once every task below is done (design, Section 9, DC10).
2. **One brief per task.** A task starts from its brief in `tasks/`. The brief lists the
   design sections to read, the scope, the files to touch, the work items and the
   acceptance checks.
3. **Pull requests.** One pull request per task, or the few a brief lists, into
   `fftw-0.11`. The title names the task, for example `feat(fftw): F1b core types and FFTW
   lock`. The description contains the brief's acceptance list, ticked.
4. **Questions.** If the design turns out wrong, unclear or incomplete, the implementer stops,
   records the question under "Open questions" in the brief, and raises it. Design changes
   go into `rlst-fftw-design.md` with an entry in its Appendix A; they are not made silently
   in code.
5. **Status.** When a pull request merges, its row below is updated, and `CLAUDE.md` is kept
   in step with anything the task changed in features, CI or validation commands.

## Order

F0, F1, F2, F4, F3, F5, F6, F7, F8a, F8b. F4 (concurrency and determinism) comes before F3
so that every later family is written against the separate-process determinism tests.

## Status

| Task | Content (design, Section 9) | Brief | Status | Pull requests |
| --- | --- | --- | --- | --- |
| F0 | Baseline, integration branch, CI version reports, macOS job | `tasks/F0-baseline.md` | done | [#2](https://github.com/tbetcke/rlst/pull/2) |
| F1 | Core: features, types, lock, raw layouts, c2c plans, checked execution on slices, fingerprints | `tasks/F1-core.md` | done | [#3](https://github.com/tbetcke/rlst/pull/3) (F1a), [#4](https://github.com/tbetcke/rlst/pull/4) (F1b), [#5](https://github.com/tbetcke/rlst/pull/5) (F1c) |
| F2 | Layouts from RLST arrays and views | `tasks/F2-layouts.md` | in progress (F2b) | [#8](https://github.com/tbetcke/rlst/pull/8) (F2a) |
| F4 | Concurrency and determinism, harness-free test binary | to be written | not started | |
| F3 | Split c2c with the part axis | to be written | not started | |
| F5 | r2c, c2r, r2r, split r2c and c2r, `Scale` | to be written | not started | |
| F6 | Wisdom; `fftw_threads` | to be written | not started | |
| F7 | Removal of the 0.10 module; Chebyshev port; documentation; licence warning; CHANGELOG | to be written | not started | |
| F8a | CI on Linux and macOS, default-build check, docs | to be written | not started | |
| F8b | Benchmarks; release 0.11.0 | to be written | not started | |

## Local setup

The FFTW tests need a system FFTW with both precisions and pkg-config:

- macOS (Homebrew): `brew install fftw pkg-config`
- Debian and Ubuntu: `sudo apt install libfftw3-dev pkg-config`
- Fedora: `sudo dnf install fftw-devel pkgconf-pkg-config`

Then, from the repository root:

```bash
cargo test --features "fftw,fftw_system,strict"
cargo clippy --features "fftw,fftw_system,strict" -- -D warnings
```

`CLAUDE.md` lists the full validation commands that CI runs.
