# F4: Concurrency and determinism

| | |
| --- | --- |
| Task | F4 (design, Section 9) |
| Status | not started |
| Depends on | F1, F2 |
| Pull requests | one, into `fftw-0.11`: F4 (concurrency tests, the harness-free determinism binary, rustdoc) |
| Read first | design: Section 2.5, Section 3 items 1–4 and 6–7, Sections 4.4, 4.6, 4.7 and 4.12, Section 5, Sections 7.4–7.7, Section 9 (F4 and F6 rows), Section 10, in particular DC22–DC24, Section 11 items 3 and 12, Appendix A (draft 0.12); `docs/fftw/tasks/F1-core.md` and `docs/fftw/tasks/F2-layouts.md`, including their "Open questions"; `docs/fftw/F0-findings.md`; `CLAUDE.md` |

## Goal

Show, by tests that run in CI on Linux and macOS, that the rules of design Section 5 hold
for complex-to-complex plans: one plan executes from many threads at once and gives the
bits of sequential execution; plans are made, described and destroyed from many threads
while others execute; execution takes no lock; and an `Estimate` plan gives the same
fingerprint and the same bits in every fresh process, including on nd-fmm's fixed-batch
units at different positions of a larger buffer. The tests that need a fresh process,
another rigor or a known wisdom state live in a new `harness = false` test binary, which
later tasks extend: F3 with split storage, F5 with the other families, F6 with wisdom and
threads. F4 adds no public API.

## Scope

In scope:

- the concurrency tests of design Section 7.5 for c2c, in process, in a new test module
  `rlst/src/fftw/concurrency_tests.rs`;
- the determinism tests of design Section 7.6 for c2c that need only F1's planner, in a new
  harness-free test binary `rlst/tests/fftw_determinism.rs`: fingerprints and bits across
  fresh processes, fixed-batch units at different positions of a larger buffer (which F1
  and F2 tested in process and left to F4 in fresh processes), wisdom-only planning without
  wisdom, and an `Estimate` plan made after a `Measure` plan;
- rustdoc on threads and determinism for `Plan` and `Plan::fingerprint` (work item 3,
  DC23);
- `rlst/Cargo.toml` (the `[[test]]` entry), `CLAUDE.md`, `rlst/CHANGELOG.md`, the status
  table of `docs/fftw/README.md`, and this brief's "Open questions".

Out of scope, for later tasks (design, Section 9, F6 row; DC24):

- wisdom export, import, forgetting and system wisdom, and their tests: wisdom export and
  import concurrent with planning (Section 7.5, third item) and export, forget, import,
  then wisdom-only (Section 7.6, third item) (F6);
- `fftw_threads`: threaded plans against single-threaded ones and threaded plans executed
  from several threads (Section 7.5, last item) (F6);
- split storage (F3) and the r2c, c2r and r2r families (F5); the binary is written so that
  they add cases (work item 2);
- the user guide in `rlst/src/doc/optional_integrations.rs`, which states all seven rules of
  Section 5 (F7);
- comparing fingerprints across MPI ranks. RLST provides `Plan::fingerprint()`; how nd-fmm
  uses it is nd-fmm's decision (design, Section 11 item 3);
- any change to the public API or to library behaviour. If a test shows that a rule of
  Section 5 does not hold, that is an open question, not a change of the test.

Files touched:

| File | Change |
| --- | --- |
| `rlst/src/fftw/concurrency_tests.rs` (new) | the tests of work item 1 |
| `rlst/src/fftw.rs` | `#[cfg(test)] mod concurrency_tests;` |
| `rlst/tests/fftw_determinism.rs` (new) | the harness-free binary of work item 2 |
| `rlst/Cargo.toml` | `[[test]] name = "fftw_determinism"`, `harness = false`, `required-features = ["fftw"]` |
| `rlst/src/fftw/plan.rs`, `rlst/src/fftw/fingerprint.rs` | rustdoc of work item 3 |
| `CLAUDE.md` | the new binary in the repository layout, the FFTW rules and the validation commands |
| `rlst/CHANGELOG.md`, `docs/fftw/README.md` | yes |

## 1. Concurrency tests (`concurrency_tests.rs`, design Section 7.5, c2c)

Ordinary unit tests: they run in the test harness, beside the other tests, as threads of
one process. They plan with `Rigor::Estimate` only (`CLAUDE.md`), use `std::thread::scope`
and no new dependency (DC9), use fixed thread counts rather than
`available_parallelism` (the macOS runner has 3 cores, the Linux runner 4; oversubscription
is intended), and synchronise with `std::sync::Barrier` so that threads overlap, never with
sleeps. Inputs come from F1's seeded generator; messages name the seed, the thread, the
iteration, the policy and the precision.

- **Send and Sync.** A compile-time check that `Plan<T, C2c, P>` is `Send + Sync` for both
  precisions and the three policies.
- **One plan, many threads** (Section 5 rule 2; Section 7.5 first item): one plan per policy
  and precision for nd-fmm's unit, `along_axis(…, 0)` of boxes `[140, 8]`; 8 threads, each
  executing it 8 times on its own box. The boxes are disjoint views of one parent at
  `FFTW_ALIGNED`, made from `data_mut().unwrap().chunks_exact_mut(1120)` and
  `SliceArrayMut::from_shape`; a box is 1120 complex elements, a multiple of 64 bytes in
  both precisions, so every box has class 0 (with `[140, 5]`, single-precision boxes would
  alternate between classes 0 and 32). Each result equals, bit for bit,
  the result of the same plan executed sequentially on the same input before the threads
  start. A `Destroying` thread restores its input before each execution; an `InPlace`
  thread restores its data.
- **Planning while executing** (Section 7.5 second item): in one scope, 4 threads execute a
  shared plan in a loop (as above), while 4 other threads each make, describe
  (`description()` and `fingerprint()`), execute once and drop plans for a fixed list of
  layouts from F2's memory orders (column-major, row-major, a subview, a slice; both
  precisions; three policies), and one thread plans and executes with the 0.10 module
  (`into_fft`, `c2c_fft` with `FftwPlanFlags::Estimate`), which shares the FFTW lock. Every
  result is within F1's tolerance of the logical reference of F2's `test_support`; the
  executing threads' results equal their sequential results bit for bit; and plans of the
  same layout made in different threads have equal fingerprints. The last holds because every
  unit test plans with `Estimate` only, so no wisdom of a greater rigor exists in the test
  process (design, Section 3 item 4).
- **Descriptions from many threads**: 8 threads call `description()` and `fingerprint()`
  on one shared plan at once; all results are equal.
- **Execution takes no lock** (Section 5 rule 2; Section 4.7): one thread holds the FFTW
  lock (`with_fftw`, crate-private) until another thread signals over a channel that it
  has executed a plan made earlier; the holder waits with `recv_timeout` (10 s) and fails
  if no signal arrives, then releases the lock, so a regression fails the test instead of
  hanging it.
- **Plans move between threads**: a plan made in one thread is moved to another, executed
  and dropped there.

Budget: these tests together take at most about 0.5 s in a debug build; measure and report.

## 2. The determinism binary (`rlst/tests/fftw_determinism.rs`, design Section 7.6)

A `harness = false` test binary, like `rlst/tests/mpi_empty_ranks.rs`, with
`required-features = ["fftw"]`, so that both CI jobs run it as part of
`cargo test --features "…fftw,fftw_system,strict"` without a change to the workflow. It uses
only the public API of `rlst` and its normal dependencies (`rand_chacha` for inputs), and
needs no BLAS (as `rlst/tests/zero_dim_array.rs`).

**Structure.** The same executable is the parent and the child:

- With the environment variable `RLST_FFTW_DETERMINISM_CASE` set, it is a child: it runs
  that case, prints its result lines to standard output, and exits with status 0, or
  panics.
- Without it, it is the parent: it runs the scenarios below, each by spawning
  `std::env::current_exe()` with the variable set, standard input closed and standard
  output and error piped and read on their own threads, with a deadline of 120 s per child
  after which it kills the child. It compares the children's outputs, prints a summary
  (every fingerprint and comparison, so that CI logs show them), lists every failure with
  the child's standard error, and exits with status 1 if anything failed. It ignores its
  command-line arguments, since `cargo test` passes filters and options such as
  `--nocapture` to harness-free binaries.
- Cases are entries of one table (name and function), and scenarios state which cases they
  run and how many times and what they compare, so that F3, F5 and F6 add rows rather than
  structure.

**Result lines.** One line per problem: a label naming the problem (layout, shape, axes,
policy, direction, precision), the plan's `fingerprint()` in hexadecimal, and a 64-bit
FNV-1a hash of the output's bits (the real and imaginary parts as little-endian `u64` or
`u32`, in column-major order of the output view), computed by a small function in the
test file. Parents compare lines as text.

**Buffers.** Every buffer whose bits are compared across processes is at alignment class
0: `rlst_dynamic_array!(…, shape | rlst::FFTW_ALIGNED)` or `AlignedDynArray`, or views of
them at offsets that are multiples of 64 bytes. A `Vec`-backed array has a class that
depends on the allocator and may differ between processes (design, Section 4.6), and a
plan for another class is another plan (Section 5 rule 3).

**Cases** (c2c, all planned on scratch unless stated):

1. `estimate`: the problems of a fixed list, each planned with `Rigor::Estimate` and
   executed once on seeded input:
   - nd-fmm's units `[n_phi, rings]` along axis 0, with n_phi in 140, 216, 588 and 1080 and
     rings = 9, f64, both directions, `Preserving`; and n_phi = 140 in f32;
   - from F2's memory orders, f64 forward: `[8, 6, 4]` over `[0, 1]`; a subview `[12, 7]`
     of an aligned `[16, 10]` at an origin whose offset is a multiple of 64 bytes, along
     axis 1; a strided view with steps of 2 over aligned storage, along axis 0;
   - the policies: one problem of the list also as `Destroying` and as `InPlace`.
2. `estimate-after-other-plans`: the child first plans and executes `Estimate` plans of
   unrelated problems (other sizes and layouts, both precisions, through the new interface
   and through the 0.10 module), and then the problems of case 1 in reverse order (DC22).
3. `fixed-batch-units` (Section 7.6 second item, Section 7.4 first item): nd-fmm's unit
   n_phi = 140, rings = 9, f64, forward, in two arrangements: boxes `slice(2, k)` of an
   aligned parent `[140, 9, 4]`, and units of `[140, 9]` at four positions of one aligned
   flat buffer, separated by a padding of 4 complex doubles (64 bytes) so that each starts
   at class 0, with the padding filled with a sentinel. Every unit holds the same seeded
   values. One plan, made from the first unit's layout as the first plan of the process,
   executes on every unit; the child asserts that all results are equal bit for bit, equal
   to those of a plan made with `c2c_on` for each unit and of the same plan on a standalone
   aligned copy, and that the padding still holds the sentinel. It prints one result line.
4. `wisdom-only-without-wisdom` (Section 7.6 fourth item): as the first planning of the
   process, `Planner::new().wisdom_only().c2c(…)` returns
   `FftwError::PlanCreationFailed` for f64 and f32; then an ordinary `Estimate` plan of
   the same problem succeeds and executes. Nothing else in the child plans before.
5. `estimate-after-measure` (Section 7.6 fifth item; Section 3 item 4): for the problem
   n_phi = 588, rings = 9, forward, f64, in this order: an `Estimate` plan (A), a
   `Rigor::Measure` plan (B), and another `Estimate` plan (C), each executed once; the
   child prints the three fingerprints. Keep the problem small enough that the `Measure`
   plan takes well under a second.

**Scenarios** (what the parent runs and compares):

| Scenario | Children | Assertion |
| --- | --- | --- |
| Fresh processes agree | `estimate` twice | the two outputs are equal line for line |
| Planning history | `estimate-after-other-plans` once | its lines equal those of `estimate` (DC22); a failure is raised as a design question, not relaxed |
| Fixed-batch units | `fixed-batch-units` twice | the child's own assertions; the two outputs are equal; the fingerprint equals that of the unit `[140, 9]`, forward, f64, in `estimate` |
| Wisdom-only | `wisdom-only-without-wisdom` once | exit status 0 |
| Estimate after measure | `estimate-after-measure` once | A equals the fingerprint of the same problem in `estimate`; A, B and C are printed, with whether C equals A and whether C equals B. Nothing is asserted about B or C: this documents Section 3 item 4 (design, Section 7.6) |

Budget: the binary takes at most about 10 s in a debug build on the CI runners, child
processes included; measure locally and report.

## 3. Documentation (DC23)

- `Plan`: a section "Threads and determinism" in its rustdoc, stating Section 5 rules 2,
  3, 5 and 6 in short: concurrent execution of one plan on different buffers is safe and
  takes no lock; when a plan is the same plan (rule 3's list, alignment classes included)
  and the input is at the same class, the bits are the same; plans made with `Measure` and
  stronger rigor can differ between runs; for results independent of the thread count, fix
  the unit of work and execute one plan per unit. Add an example that executes one plan
  from several threads with `std::thread::scope` on disjoint buffers.
- `Plan::fingerprint`: rule 4, what equal fingerprints mean (FFTW chose the same
  algorithm) and what they do not (equal bits also need the same input class and FFTW
  build), and that an `Estimate` plan depends on the wisdom present in the process
  (Section 3 item 4), with a pointer to `Rigor`.
- Rules 1 (the lock and its limits) and 7 (shared wisdom) stay for F6 and F7.
- `CLAUDE.md`: the repository layout lists `tests/fftw_determinism.rs` beside
  `mpi_empty_ranks`; the FFTW rule on tests names the binary; the validation section adds
  `cargo test --features "fftw,fftw_system" --test fftw_determinism` as the narrow command.
  CI needs no change; check that both jobs list the binary in their test output.
- `rlst/CHANGELOG.md`: the entry "New FFTW interface (in progress)" gains one sentence:
  one plan executes from many threads at once, and `Estimate` plans give the same
  fingerprint and bits in every process, as tested.
- `docs/fftw/README.md`: the files table lists this brief, and the status table is updated
  when the pull request merges.

## Pull request and CI

- One pull request, titled for example `test(fftw): F4 concurrency and determinism`,
  branched from `fftw-0.11`, with this brief's acceptance list ticked.
- CI (`.github/workflows/run-tests.yml`) runs for pull requests into `fftw-0.11`. Both jobs
  run `cargo test` with `fftw` and so run the new binary. Report in the pull request the
  binary's summary from both jobs: the fingerprints of case 1 and of the
  `estimate-after-measure` scenario differ between Linux (SIMD) and macOS (no SIMD), which
  is expected, and are worth recording.

## Acceptance

- [ ] The tests of work item 1 pass in the default FFTW test run, in CI on Linux and macOS;
      they plan with `Estimate` only; their added time in a debug build is reported and is
      at most about 0.5 s.
- [ ] `rlst/tests/fftw_determinism.rs` exists as a `harness = false` binary with
      `required-features = ["fftw"]`, runs every case in a fresh child process with a
      deadline, and passes in CI on Linux and macOS; its summary from both jobs is in the
      pull request; its time in a debug build is reported and is at most about 10 s.
- [ ] The five cases and the scenarios of work item 2 are implemented; the
      `estimate-after-measure` scenario reports its fingerprints and asserts only what work
      item 2 says.
- [ ] The rustdoc of work item 3 has an example that runs.
- [ ] No public API changed, no library behaviour changed, and no dependency was added.
- [ ] The six Clippy commands of `CLAUDE.md` pass (or are reported unavailable), as do
      `cargo fmt -- --check`, `cargo test`, `cargo test --features "fftw,fftw_system,strict"`
      and `cargo clippy --tests --features "fftw,fftw_system,strict" -- -D warnings`.
- [ ] `CLAUDE.md`, `rlst/CHANGELOG.md` and `docs/fftw/README.md` are updated.

## Pitfalls

- **Wisdom is global to the process** (Section 3 item 4). Unit tests plan with `Estimate`
  only; `Measure` and wisdom-only planning belong to the binary's children. A unit test
  that compares fingerprints or bits of two distinct plan objects relies on that rule.
- **Fresh means fresh.** In a child nothing may plan before the case starts: no lazy
  statics that plan, no shared setup. `wisdom-only-without-wisdom` must be the first FFTW
  planning of its process.
- **Alignment classes.** Compare bits across processes only for buffers at class 0 from
  aligned allocations. A unit at an offset that is not a multiple of 64 bytes has another
  class and needs a plan of its own.
- **SIMD.** The Linux runner's FFTW has SIMD and the macOS runner's has none
  (`docs/fftw/F0-findings.md`). Bits and fingerprints are compared only between processes
  of one machine and build; never compare with hard-coded hashes. Nothing here depends on
  SIMD, so no test skips.
- **Child processes.** Read standard output and error on separate threads while the child
  runs (a full pipe blocks the child), kill a child at its deadline and report it, report
  every failing scenario rather than stopping at the first, and exit with a non-zero
  status on failure. Do not inherit the case variable into grandchildren; set it
  explicitly for each child.
- **Arguments.** `cargo test somefilter` and `cargo test -- --nocapture` pass their
  arguments to harness-free binaries; the parent ignores them.
- **Threads.** Use `std::thread::scope` and `Barrier`; a panic in a scoped thread
  propagates when the scope ends, so assertions in threads fail the test. Keep thread
  counts fixed and work per thread small for debug builds.
- **The lock test.** Never hold the FFTW lock without a bound: wait with `recv_timeout`,
  then release, then assert, so that a regression fails rather than hangs the test suite.
- **The 0.10 module.** It shares the FFTW lock; use `FftwPlanFlags::Estimate` with it in
  unit tests.
- **`strict`, wildcard imports, debug builds**: as in the F1 and F2 briefs.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design
questions in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.12, DC22–DC24, Appendix A): all three with the option marked *proposed*.
The work items follow these answers.

1. **Estimate plans after unrelated planning.** Design Section 7.6 asks that `Estimate`
   plans give the same fingerprint and bits across two fresh processes. nd-fmm also relies
   on a stronger property: that an `Estimate` plan does not depend on the `Estimate`
   planning of other problems earlier in the process, since an operator build plans many
   sizes (Section 2.5). Section 5 rule 3 conditions sameness on "the same wisdom for that
   problem", and Section 3 item 4 on wisdom of equal or greater rigor; whether `Estimate`
   planning of other problems leaves wisdom that changes a later `Estimate` plan is not
   stated. Options: (a) *proposed*: the `estimate-after-other-plans` scenario asserts that
   fingerprints and bits equal those of fresh processes, and a failure is raised as a
   design question; (b) it only reports whether they are equal, as the
   `estimate-after-measure` scenario does. *Decided: (a), DC22.*
2. **Where the rules of Section 5 are documented in F4.** Section 9 gives the
   documentation to F7, but F4 is the task that tests the rules. Options: (a) *proposed*:
   F4 states rules 2, 3, 5 and 6 in the rustdoc of `Plan` and rule 4 in that of
   `Plan::fingerprint` (work item 3), and F7's guide states all seven and links them;
   (b) F4 adds no documentation, and F7 writes all of it. *Decided: (a), DC23.*
3. **The split between F4 and F6.** Section 9 gives F6 "7.5 and 7.6 for them", wisdom and
   threads. This brief reads that as: F6 tests wisdom export, import and forgetting (7.5
   third item, 7.6 third item) and `fftw_threads` (7.5 last item), while F4 tests
   wisdom-only planning without wisdom (7.6 fourth item) and the `Estimate` plan after a
   `Measure` plan (7.6 fifth item), which need only F1's planner options. Options:
   (a) *proposed*: as read; (b) F6 takes the fourth and fifth items of 7.6 too.
   *Decided: (a), DC24; the F4 and F6 rows of design Section 9 say so.*

Raised during F4. None needs a design change; they are recorded for review and for F5,
F6 and F7.

4. **`as_chunks_mut` for the units.** Work item 1 names `chunks_exact_mut(1120)`. With a
   constant chunk size, Clippy 1.99 denies that under `strict`
   (`chunks_exact_to_as_chunks`), so the tests split the parents with
   `as_chunks_mut::<1120>()`, which gives the same disjoint chunks, and build each box with
   `SliceArrayMut::from_shape` as the brief says. *Decided: accepted.*
5. **Row-major arrays in "planning while executing".** F2's row-major order is
   `StridedDynArray::row_major`, which is `Vec`-backed, so its alignment class depends on
   the allocator and can differ between threads; plans for different classes are different
   plans (Section 5 rule 3), and the equal-fingerprint assertion would then depend on the
   allocator. The test uses row-major strides over aligned storage
   (`StridedSliceArrayMut::from_shape_and_stride`), the same memory order at class 0.
   *Decided: row major stays, as work item 1 lists it.*
6. **The fixed-batch scenario also compares bits with `estimate`.** Work item 2 asserts
   that the fingerprint of `fixed-batch-units` equals that of the unit `[140, 9]` in
   `estimate`. Both cases fill the unit with the same seeded input, so Section 5 rule 3
   also gives equal bits; the scenario asserts the whole result line, fingerprint and
   output hash. *Decided: the stronger assertion is kept.*
7. **What fingerprints do not tell apart.** `fftw_sprint_plan` describes the algorithm,
   not the problem: on macOS (FFTW 3.3.11 without SIMD) the forward and backward plans of a
   unit, its f64 and f32 plans, and its `Preserving` and `Destroying` plans all have the
   same fingerprint. This matches Section 4.12 (equal fingerprints mean the same
   algorithm), but users comparing fingerprints across ranks should compare plans of one
   problem; F7's guide could say so.
8. **`estimate-after-measure` on macOS.** Locally (Homebrew FFTW 3.3.11 without SIMD,
   arm64) the `Estimate` plan C made after the `Measure` plan B is plan B (C equals B, C
   differs from A), which confirms Section 3 item 4 on that build.
