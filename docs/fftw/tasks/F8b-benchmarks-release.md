# F8b: Benchmarks and the release of 0.11.0

| | |
| --- | --- |
| Task | F8b (design, Section 9) |
| Status | done |
| Depends on | F0 to F8a |
| Pull requests | three, in this order: F8b-1, the benchmarks and their findings, into `fftw-0.11`; F8b-2, the release preparation (versions, CHANGELOG, documentation), into `fftw-0.11`; F8b-3, `fftw-0.11` into `main`. The tag, the publication on crates.io and the GitHub release follow the merge of F8b-3 (work item 4) |
| Read first | design (draft 0.19): Section 1.1 (goals), Section 2.5 (what nd-fmm needs), Section 4.3 (the halved axis, DC2), Section 4.5 (split storage and SIMD), Section 4.6 (alignment classes, `.unaligned()`), Section 7.7 (the F8b item), Section 9 (F8b row, "Release (DC10)"), Section 10 (DF8, DC2, DC9, DC10), Section 10.2 (the earlier timings), Section 11 items 2 and 9; `docs/fftw/F0-findings.md`; the "Open questions" of `docs/fftw/tasks/F3-split.md` and `F8a-ci.md`; `rlst/CHANGELOG.md` (the v0.10.2 and v0.10.3 entries, for the form of a release); `CLAUDE.md` |

## Goal

F8b measures what the design left to measurement, and releases the result. **The
benchmarks** (design Section 7.7, DC9) compare a batched plan with a loop of single plans,
split storage with interleaved storage, aligned plans with unaligned ones, and real-to-complex
plans that halve the contiguous axis with plans that halve a strided one; they are an
example, not a test, and gate nothing. The last of them decides whether the documentation
recommends listing the contiguous axis last (DC2). **The release** makes the workspace
0.11.0, merges `fftw-0.11` into `main`, and publishes the three crates, with the CHANGELOG
that F7c completed. Under DC10 there is no release before this one and no 0.11.x that adds a
missing part of the design; after F8b the FFTW rework is done.

## Checked while writing this brief

On macOS (Homebrew FFTW 3.3.11, no SIMD, cargo 1.99.0), at `fftw-0.11` `39780fb`, unless
stated:

1. **No benchmarks exist.** `rlst` has no `benches/` directory and no benchmark dependency;
   DC9 settles that benchmarks are an example and add no dependency, so they time with
   `std::time::Instant`. `rlst/examples/` holds eight examples, of which the MPI ones
   declare `required-features = ["mpi"]`. The workflow that ran examples was removed
   (`02eb2ec`), so examples are only compiled, by `cargo clippy --examples` with
   `mpi,fftw,fftw_system,strict` on Linux: an example with `required-features = ["fftw"]` is
   linted there and never run in CI.
2. **The earlier timings.** Design Section 10.2 timed one box of nd-fmm's φ step (289
   transforms of length 588, interleaved, out of place, `Estimate`) at about 0.59 ms with
   Ubuntu's FFTW 3.3.10 on an AVX2 Xeon virtual machine, and Section 11 item 2 the box
   `[140, 9]` at about 0.5–0.6 ms interleaved, 0.9–1.0 ms split, and about 0.2 ms for an
   unoptimised copy of split planes into interleaved scratch. Both are marked indicative.
   nd-fmm's sizes are N_φ of 140, 216, 336, 588 and 1080, with 9 rings per box in the
   determinism binary's units.
3. **The state of the release.** crates.io has 0.10.3 of `rlst`, `rlst-proc-macro` and
   `rlst-suitesparse`; the workspace is at 0.11.0-dev, with `rlst`'s requirement on
   `rlst-proc-macro` and `rlst-suitesparse`'s on `rlst` at the same version. `main` is at
   `dc572e5`, the base of `fftw-0.11`, which is 77 commits ahead: `main` has had no 0.10.x
   release since 0.10.3, so the merge has no conflict today. `main` is protected (the check
   `run-tests` required, no force pushes); tags `v0.10.0` to `v0.10.3` exist.
4. **How releases were made.** 0.10.2 and 0.10.3 were each one commit that set the three
   crates and their requirements on each other to the new version and added the CHANGELOG
   entry ("Changes since [v0.10.1](…). All workspace crates … use version 0.10.2."),
   merged into `main`, then tagged. Between releases `main` kept the released version; the
   only `-dev` version is 0.11.0-dev, set for the FFTW rework (`dc572e5`).
5. **Publishing works.** `cargo publish --workspace --dry-run` (cargo 1.99) packages and
   verifies the three crates in dependency order against a temporary local registry:
   `rlst-proc-macro` (8 files), `rlst` (211 files, 2.5 MiB, 515 KiB compressed) and
   `rlst-suitesparse` (14 files, built against Homebrew's SuiteSparse), and stops before
   uploading. Publishing needs the maintainer's crates.io token, which this repository's
   automation does not have.
6. **The CHANGELOG** begins with "## v0.11.0 (unreleased)" and a paragraph that names the
   integration branch and the version 0.11.0-dev; its FFTW entry is final since F7c.
7. **What names the rework as in progress.** `CLAUDE.md` ("The interface is being rewritten
   for 0.11 on the `fftw-0.11` branch", the section "FFTW rework (0.11)", the guide table's
   row "FFTW rework for 0.11 (in progress)"), `docs/fftw/README.md` (its workflow and
   status table), the design's first line ("Draft 0.18 … not a plan until it is signed
   off"), and `run-tests.yml`, which triggers on `fftw-0.11`.

## Scope

In scope:

- the benchmark example and its findings (work item 1; F8b-1);
- the documentation that follows from the findings: DC2's recommendation, and the notes on
  split storage and alignment where the measurements change them (work item 2; F8b-1);
- the release preparation: versions, CHANGELOG, and the documentation that names the rework
  as in progress (work item 3; F8b-2);
- the merge of `fftw-0.11` into `main`, the tag, the publication and the GitHub release
  (work item 4; F8b-3 and after);
- `docs/fftw/README.md`'s status table and this brief's "Open questions".

Out of scope, for later or not at all:

- benchmarks as CI gates, and benchmarks of anything but FFTW plans;
- tuning: a benchmark that shows a slow path is reported, and a change it suggests is
  raised, not made in F8b;
- the open points for nd-fmm of design Section 11, which are nd-fmm's;
- `main`'s version after the release: it stays at 0.11.0 (DC59);
- any change to the API of the new interface.

Files touched:

| File | Change | PR |
| --- | --- | --- |
| `rlst/examples/fftw_bench.rs` | new: the benchmarks | F8b-1 |
| `rlst/Cargo.toml` | `[[example]] name = "fftw_bench"`, `required-features = ["fftw"]` (F8b-1); version 0.11.0 and the requirement on `rlst-proc-macro` (F8b-2) | F8b-1, F8b-2 |
| `docs/fftw/F8b-benchmarks.md` | new: the machines, the FFTW builds, the results and what follows from them | F8b-1 |
| `rlst/src/doc/optional_integrations.rs`, `rlst/src/fftw/layout.rs`, `rlst/src/fftw/plan.rs` | DC2's recommendation, or the statement that there is none; the notes on split storage and alignment, if the results change them | F8b-1 |
| `proc-macro/Cargo.toml`, `rlst-suitesparse/Cargo.toml` | version 0.11.0, and `rlst-suitesparse`'s requirement on `rlst` | F8b-2 |
| `rlst/CHANGELOG.md` | "## v0.11.0", "Changes since v0.10.3", the benchmark findings if they change the documentation | F8b-1, F8b-2 |
| `CLAUDE.md`, `README.md` | the rework as done; the branch workflow and CI triggers after the release | F8b-2 |
| `.github/workflows/run-tests.yml` | the `fftw-0.11` triggers removed (DC58) | F8b-2 |
| `docs/fftw/README.md`, `docs/fftw/rlst-fftw-design.md` | the status table; the design's status as implemented in 0.11.0 (DC60) | F8b-2 |

## 1. The benchmarks (`examples/fftw_bench.rs`; F8b-1)

Run with `cargo run --release --example fftw_bench --features fftw_system`. The example
prints the machine's FFTW version string (`rlst::fftw::version()`), each plan's fingerprint,
and for each case and size the time per execution. It plans with `Rigor::Estimate`, as nd-fmm
does, and only with it (DC54). Four comparisons, f64 unless stated, at
nd-fmm's N_φ of 140, 216, 336, 588 and 1080 (fact 2):

- **Batched against a loop.** One plan over the box `[N_φ, 9]` along axis 0, against a plan
  of one transform executed nine times on the box's columns.
- **Split against interleaved.** The split box `[N_φ, 9, 2]` with the part axis outermost
  (nd-fmm's planes), against the interleaved box `[N_φ, 9]`, and against the copy of the
  split planes into interleaved scratch followed by the interleaved plan (design Section
  4.5, Section 11 item 2).
- **Aligned against unaligned.** A plan at class 0 on `FFTW_ALIGNED` buffers, against a
  plan made with `.unaligned()` on the same buffers, and on buffers at class 16.
- **Halving the contiguous axis against a strided one** (DC2). Real-to-complex over both
  axes of a column-major `[n, m]` array with `axes = [1, 0]` (axis 0, the contiguous one,
  halved) against `axes = [0, 1]` (axis 1 halved), at sizes of nd-fmm's order, and the
  same for complex-to-real.

Each case warms up, then times 7 rounds of as many executions as take at least 0.2 s, and
reports the median time per execution and the range of the rounds (DC54).

## 2. What follows from them (F8b-1)

- `docs/fftw/F8b-benchmarks.md` records, per machine, the CPU, the operating system, the
  FFTW version string, the commands, and the results as tables, with the spread of the
  repetitions. The machines are macOS on Apple silicon with Homebrew's FFTW (no SIMD), run
  locally, and a Linux CI runner with Ubuntu's FFTW (SIMD), through a temporary step on a
  throwaway branch; both are indicative (DC53).
- **DC2.** The guide's section on families and the rustdoc of `Layout::over_axes` (and the
  r2c and c2r layout functions) recommend listing the contiguous axis last only if the
  measurements support it: faster by more than the spread of the rounds and by at least
  10 % at most sizes on every machine measured (DC55); otherwise they say that the order
  was measured and recommend nothing.
- **Split storage and alignment.** The documentation says today that split plans can use
  fewer SIMD stages than interleaved ones and that `.unaligned()` costs most SIMD code. If
  the measurements contradict that on a machine, the text says what was measured; otherwise
  it does not change.

## 3. The release preparation (F8b-2)

- **Versions.** `rlst`, `rlst-proc-macro` and `rlst-suitesparse` at 0.11.0, with `rlst`'s
  requirement on `rlst-proc-macro` and `rlst-suitesparse`'s on `rlst` at 0.11.0 (fact 4).
- **CHANGELOG.** "## v0.11.0 (unreleased)" becomes "## v0.11.0", and its first paragraph
  says "Changes since [v0.10.3](…). All workspace crates (`rlst`, `rlst-proc-macro`, and
  `rlst-suitesparse`) use version 0.11.0." The entries are those of F1 to F7c.
- **The rework as done** (fact 7): `CLAUDE.md`'s feature entry for `fftw` and its section
  "FFTW rework (0.11)" describe the finished interface and the record in `docs/fftw/`,
  and the guide table drops "(in progress)"; `docs/fftw/README.md`'s workflow section says
  that the rework is complete; the design's first line says that it is implemented in
  0.11.0, with an Appendix A entry (DC60); `run-tests.yml` no longer triggers on
  `fftw-0.11` (DC58).
- **Checks.** Every command of `CLAUDE.md`'s validation list, and `cargo publish
  --workspace --dry-run` (fact 5), whose output the pull request reports.

## 4. The merge, the tag and the publication (F8b-3 and after)

- **The merge.** A pull request from `fftw-0.11` into `main`, after F8b-2 has merged,
  titled `release: 0.11.0`, whose description lists the tasks F0 to F8b with their pull
  requests and links the CHANGELOG, merged with a merge commit once its CI passes (DC56).
- **After the merge**, in this order, each step confirmed with the maintainer:
  1. the tag `v0.11.0` on `main`'s merge commit, pushed by the agent (DC56, DC57);
  2. the publication of the three crates, in dependency order (`rlst-proc-macro`, `rlst`,
     `rlst-suitesparse`, or `cargo publish --workspace`), by the maintainer (fact 5, DC57);
  3. the GitHub release `v0.11.0`, with the CHANGELOG's v0.11.0 entry as its notes,
     created by the agent with `gh` (DC57);
  4. a check that docs.rs built `rlst` 0.11.0 with the FFTW documentation and its badges,
     and that the README's warning shows on crates.io.
- **The branch.** `fftw-0.11` is deleted once `main` has the merge (DC58).
- **Status.** `docs/fftw/README.md`'s status table marks F8a done in F8b-2, and F8b done in
  a last small pull request into `main` after the publication, as the earlier tasks did.

## Pull requests and CI

- Three pull requests (see the table at the top), each with the acceptance items it closes
  ticked: `feat(fftw): F8b-1 benchmarks`, `chore(release): F8b-2 prepare the workspace for
  0.11.0`, and `release: 0.11.0`.
- CI runs for pull requests into `fftw-0.11` and `main`; `main` requires `run-tests`, and
  `run-tests-macos` and `docs` if the maintainer follows F8a's recommendation (DC52).
- The benchmarks do not run in CI; their Linux run is a temporary step on a throwaway
  branch, closed afterwards, as F7a did for its comparison (DC53).

## Acceptance

- [ ] `examples/fftw_bench.rs` measures the four comparisons of work item 1 at nd-fmm's
      sizes, with `required-features = ["fftw"]` and no new dependency, and passes Clippy
      with `--examples` and the FFTW features.
- [ ] `docs/fftw/F8b-benchmarks.md` records the machines, the FFTW builds and the results;
      the documentation states DC2's recommendation, or that there is none, according to
      the rule of DC55.
- [ ] The three crates are at 0.11.0 with their requirements on each other; the CHANGELOG's
      entry is "## v0.11.0", changes since v0.10.3; `cargo publish --workspace --dry-run`
      passes.
- [ ] `CLAUDE.md`, the README, `docs/fftw/README.md` and the design describe the rework as
      done; no text calls the interface in progress.
- [ ] `fftw-0.11` is merged into `main`, CI passes on `main`, and the tag `v0.11.0` points
      at the released commit.
- [ ] `rlst`, `rlst-proc-macro` and `rlst-suitesparse` 0.11.0 are on crates.io, the GitHub
      release exists, and docs.rs shows `rlst::fftw` with the `fftw` and `fftw_threads`
      badges.
- [ ] Every command of `CLAUDE.md`'s validation list passes (or is reported unavailable);
      no dependency was added, and `Cargo.lock` is not committed.

## Pitfalls

- **Benchmarks in debug builds.** Timings only mean something with `--release`; the example
  says so when built without optimisations.
- **Noise.** Shared CI runners and laptops vary from run to run. Report the spread of the
  repetitions, and do not base DC2's recommendation on a difference smaller than the
  spread.
- **Wisdom in the benchmark.** A `Measure` plan, or imported wisdom, would change later
  `Estimate` plans of the same problem in the process (design, Section 3 item 4). The
  example plans with `Estimate` only (DC54) and imports no wisdom, and prints each plan's
  fingerprint, so that the findings name the plans they timed.
- **Alignment of the copies.** Buffers for the class-16 case are views of `FFTW_ALIGNED`
  buffers at one complex value's offset; a plain `DynArray` has an unknown class.
- **The order of the release.** The tag and the publication follow the merge into `main`
  and its CI; publishing is irreversible (a version can be yanked, not replaced), so each
  step is confirmed first.
- **The CHANGELOG as a doctest.** `doc::changelog` includes the CHANGELOG, whose code blocks
  run as doctests; the release text uses inline code, as F7a found.
- **`strict`, wildcard imports, debug builds**: as in the earlier briefs.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design questions
in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.19), each with the proposed option:

1. **The machines.** Options: (a) *proposed*: macOS on Apple silicon with Homebrew's FFTW
   (no SIMD), run locally, and Linux on a CI runner with Ubuntu's FFTW (SIMD), through a
   temporary step on a throwaway branch; both recorded as indicative; (b) as (a), and the
   maintainer adds a run on a dedicated Linux machine, such as that of design Section
   10.2, whose numbers the documentation cites; (c) macOS only.
   *Decided: (a), DC53.*
2. **The method and the rigor.** Options: (a) *proposed*: per case, a warm-up, then 7
   rounds of as many executions as take at least 0.2 s, reporting the median time per
   execution and the range of the rounds; `Estimate` plans only, nd-fmm's rigor;
   (b) as (a), and a second table with `Measure` plans, made after every `Estimate` case.
   *Decided: (a), DC54.*
3. **DC2's rule.** Options: (a) *proposed*: the documentation recommends listing the
   contiguous axis last if that is faster by more than the spread of the rounds and by at
   least 10 % at most sizes on every machine measured; otherwise it says that the order was
   measured and makes no recommendation; (b) the maintainer decides from the findings.
   *Decided: (a), DC55.*
4. **The merge into `main`.** `main` is an ancestor of `fftw-0.11` (fact 3). Options:
   (a) *proposed*: a pull request from `fftw-0.11` into `main`, merged with a merge commit
   once its CI passes, so that `main` records the release as one merge, as the task pull
   requests did; the tag goes on that merge commit; (b) a fast-forward of `main` to
   `fftw-0.11`, which leaves no merge commit.
   *Decided: (a), DC56.*
5. **Who publishes and tags.** Publishing needs the maintainer's crates.io token (fact 5).
   Options: (a) *proposed*: the maintainer runs the publication; the agent prepares the
   release notes and, after the maintainer's confirmation, pushes the tag and creates the
   GitHub release with `gh`; (b) the maintainer does all three.
   *Decided: (a), DC57.*
6. **`fftw-0.11` after the release.** Options: (a) *proposed*: delete the branch after the
   merge, remove it from `run-tests.yml`'s triggers in F8b-2, and say in `CLAUDE.md` that
   FFTW work goes to `main`; (b) keep the branch, frozen, and its triggers.
   *Decided: (a), DC58.*
7. **`main`'s version after the release.** Options: (a) *proposed*: `main` stays at 0.11.0
   until the next release is prepared, as it stayed at 0.10.1 and 0.10.2 (fact 4);
   (b) `main` moves to 0.11.1-dev or 0.12.0-dev at once.
   *Decided: (a), DC59.*
8. **The design after the release.** Options: (a) *proposed*: the design stays in
   `docs/fftw/` as the record, its first line says that it is implemented in 0.11.0 (with
   an Appendix A entry), and Section 11 stays as nd-fmm's open points; the briefs stay as
   they are; (b) as (a), and `docs/fftw/` moves to an archive directory; (c) the design is
   removed from the repository after the release.
   *Decided: (a), DC60.*

Raised during F8b-1. None needs a design change; they are recorded for review. The numbers
are in `docs/fftw/F8b-benchmarks.md`.

9. **Checking arrays at execution costs about 0.6–0.85 µs per call.** Executing the batched
   plan of the box `[N, 9]` on the arrays' data as slices is faster than executing it on
   the arrays by a constant 0.85 µs on macOS (Apple M3 Max) and 0.6 µs on Linux (AMD EPYC
   9V74), at every N. That is a quarter of the time at N = 140 and 2–3 % at N = 1080. The
   difference is RLST's check of the arrays' shapes, strides and storage against the
   layout. Per the brief, F8b reports this and does not tune it. A later change could make
   the check cheaper; it would not change the API. Options: (a) leave it, since nd-fmm's
   boxes at N ≥ 216 lose 2–15 %; (b) an issue to profile the array check after the
   release. Proposed: (b). *Decided: (b), by the maintainer; issue
   [#44](https://github.com/tbetcke/rlst/issues/44).* *Later: #44 removed the check's heap
   allocations, which cut the cost to about 0.15 µs on macOS; see `F8b-benchmarks.md`,
   comparison 1.*
10. **c2r halving the contiguous axis on Linux.** On the Linux runner, c2r over both axes
    with the contiguous axis halved is 21–50 % faster at four of the five sizes. r2c is
    faster at two of the five, and on macOS without SIMD neither is faster below
    N = 1080. Under DC55 the documentation therefore recommends no order (work item 2).
    Recording a narrower statement, for example "on SIMD builds of FFTW, c2r was faster with
    the contiguous axis halved", would be a design change to DC55. Options: (a) keep DC55's
    outcome as it stands; (b) a design change that lets the guide report the c2r result on
    SIMD builds, without a recommendation. Proposed: (a); the findings are in
    `F8b-benchmarks.md` for anyone who needs them. *Decided: (a), by the maintainer.*
11. **Split storage and alignment, measured.** On Linux with SIMD:
    - split plans take 1.5–2.8× the interleaved time;
    - copying the planes into interleaved scratch costs only 1–8 % above the interleaved
      plan;
    - `.unaligned()` plans take 1.6–3.7× the time;
    - a plan for a view's alignment class costs 3–6 % more.

    Without SIMD (macOS) none of these differ. The documentation's notes are confirmed, so by
    work item 2 they do not change. The guide could quote the magnitudes, which nd-fmm's
    choice of design Section 11 item 2 depends on; it is left as it is unless the
    maintainer asks. *Decided: the guide is left as it is, by the maintainer; the
    magnitudes stay in `F8b-benchmarks.md`.*

Raised during F8b-2.

12. **When the `fftw-0.11` triggers leave `run-tests.yml`.** Work item 3 removes them in
    F8b-2 (DC58). F8b-2 is itself a pull request into `fftw-0.11`, and GitHub takes the
    workflow from the pull request's merge commit, so F8b-2 would run no CI, and neither
    would its push to `fftw-0.11`. Options: (a) *proposed*: keep the triggers through the
    release, and remove them, with `CLAUDE.md`'s description of the triggers, in the last
    small pull request into `main` that marks F8b done, after `fftw-0.11` is deleted; DC58
    holds, in a later pull request; (b) remove them in F8b-2 as briefed, which leaves F8b-2
    without CI until the release pull request into `main`. *Decided: (a), by the
    maintainer.*
