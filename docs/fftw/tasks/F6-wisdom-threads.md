# F6: Wisdom; the `fftw_threads` feature

| | |
| --- | --- |
| Task | F6 (design, Section 9) |
| Status | not started |
| Depends on | F1, F4, F5 |
| Pull requests | two, into `fftw-0.11`, in this order: F6a (wisdom: export, import, files, forgetting, system wisdom, and their tests), F6b (the `fftw_threads` feature: linking, thread initialisation, `Planner::threads`, `make_planner_thread_safe`, and their tests and CI) |
| Read first | design: Section 1.2 (the functions not wrapped), Section 3 items 2–4, 9 and 12, Sections 4.7, 4.10 and 4.12–4.15, Section 5 (rules 1, 3, 6 and 7), Sections 7.5–7.7, Section 9 (F6 row), Section 10, in particular DF7, DF8, DC13, DC22–DC24 and DC38–DC43, Section 11 items 3 and 4, Appendix A (drafts 0.12 and 0.16); `docs/fftw/tasks/F1-core.md` (the lock and the settings guard) and `F4-determinism.md` (the binary), including their "Open questions"; `docs/fftw/F0-findings.md` (the threads libraries on the runners); `CLAUDE.md` |

## Goal

The last two parts of design Section 1.1 goal 5. **Wisdom**: export and import per precision,
through strings and files, forgetting, and the system wisdom on explicit request, every call
under the FFTW lock (design, Section 4.13); tests in fresh processes show that wisdom
exported, forgotten and imported again gives the same plan, and that wisdom shared between
processes gives the same plans in each (Section 5 rule 7). **Threads**: the feature
`fftw_threads`, which links FFTW's threads libraries, initialises them once under the lock,
and adds `Planner::threads(n)` and `threads::make_planner_thread_safe()` (design, Sections
4.10 and 4.14); the thread count joins the time limit in the settings guard, so it never
leaks into another caller's plan (Section 4.7). F6 also documents rules 1 and 7 of Section 5,
which F4 left to F6 and F7 (DC23). nd-fmm uses neither wisdom nor threads (design, Section
2.5), but Section 11 items 3 and 4 note options of nd-fmm's that use wisdom.

## Checked while writing this brief

With Homebrew's FFTW 3.3.11 on arm64 (no SIMD), in small C programs:

1. **Estimate planning adds wisdom.** Before any planning, the f64 export is the header
   alone, `(fftw-3.3.11 fftw_wisdom #x… #x… #x… #x…)`; after one `FFTW_ESTIMATE` plan of
   nd-fmm's unit (588, 9 rings) it lists five entries. The f32 header reads `fftwf_wisdom`.
2. **A failed import changes nothing.** Importing `(garbage` or a truncated header returns
   0, and the export afterwards equals the export before. FFTW does not document this.
3. **Wisdom of the other precision is refused.** f64 wisdom given to
   `fftwf_import_wisdom_from_string` returns 0, and the f32 wisdom stays empty.
4. **Round trip.** After `fftw_forget_wisdom`, a wisdom-only `FFTW_MEASURE` plan of the
   unit returns NULL; after importing the wisdom exported before, it succeeds with the
   description of the earlier `MEASURE` plan, and a wisdom-only `FFTW_ESTIMATE` plan
   succeeds too (wisdom of greater rigor, design Section 3 item 4).
5. **No system wisdom** on macOS: `fftw_import_system_wisdom` returns 0.
6. **Where the symbols are.** `fftw_init_threads`, `fftw_plan_with_nthreads`,
   `fftw_planner_nthreads` and `fftw_make_planner_thread_safe` are defined in
   `libfftw3_threads`, not in `libfftw3`; the wisdom functions are in `libfftw3`. fftw-sys
   declares all of them but links only `fftw3` and `fftw3f`.
7. **Threaded plans differ.** With 4 threads, the `Estimate` plans of `[1080, 64]` and of
   `[140, 8]` along axis 0 are threaded (`dft-thr-vrank>=1-x4/1 …`), so their fingerprints
   differ from those of the single-threaded plans, even for small problems.
8. **Wisdom is kept per thread count.** A single-threaded `Estimate` plan of `[1080, 64]`
   made after a threaded `Estimate` and a threaded `Measure` plan of the same problem
   equals the one made before them. A threaded `Estimate` plan made after the threaded
   `Measure` plan is the `Measure` plan (design, Section 3 item 4, for that thread count).

The determinism binary takes 0.80 s locally (debug build, children included) before F6.

## Scope

In scope:

- the wisdom functions of design Section 4.13 for both precisions (work item 1; F6a);
- the feature `fftw_threads`, the threads libraries in `build.rs`, the minimum 3.3.6
  (work item 2; F6b);
- thread initialisation, the thread count in the settings guard, and
  `make_planner_thread_safe` (work item 3; F6b);
- `Planner::threads`, `Plan::threads`, and the thread count in the rules on `Plan`
  (work item 4; F6b);
- the tests of design Sections 7.5 and 7.6 that DC24 gives to F6, in the concurrency tests
  and the determinism binary (work items 5 and 6);
- the CI commands of DC43 (work item 7; F6b);
- rustdoc, the module documentation of `rlst/src/fftw.rs`, `rlst/CHANGELOG.md`, the README's
  "System FFTW" section, `CLAUDE.md`, the files table of `docs/fftw/README.md`, and this
  brief's "Open questions" (work item 8).

Out of scope, for later tasks or not at all:

- FFTW functions that design Section 1.2 excludes: `fftw_cleanup` and
  `fftw_cleanup_threads`, FILE-based wisdom, `fftw_planner_nthreads` and
  `fftw_threads_set_callback`; the `_omp` libraries;
- the user guide in `rlst/src/doc/optional_integrations.rs` with all seven rules of Section
  5, the docs.rs metadata, and the migration notes (F7);
- the review of the CI matrix and the default-build check (F8a); benchmarks of threaded
  plans (F8b, if wanted);
- comparing fingerprints across MPI ranks, and whether nd-fmm uses wisdom (design, Section
  11 items 3 and 4; nd-fmm's decisions);
- wisdom shared between machines with different SIMD support: documented (Section 5
  rule 7), not tested;
- any change to the API of F1 to F5 other than the additions of this brief. A reason to
  change more is an open question, not an edit.

Files touched:

| File | Change | PR |
| --- | --- | --- |
| `rlst/src/fftw/wisdom.rs` (new) | the functions of work item 1 | F6a |
| `rlst/src/fftw/real.rs` | export, import, forgetting and system wisdom in the sealed `FftwReal`; with `fftw_threads`, thread initialisation, the thread count and `make_planner_thread_safe` | F6a, F6b |
| `rlst/src/fftw/error.rs` | `WisdomFile` (DC38); `InvalidThreadCount` and `ThreadsUnavailable` (DC41) | F6a, F6b |
| `rlst/src/fftw/lock.rs` | thread initialisation on the first acquisition; the thread count in `SettingsGuard`; its test | F6b |
| `rlst/src/fftw/threads.rs` (new) | `make_planner_thread_safe`, the module documentation of rule 1 | F6b |
| `rlst/src/fftw/planner.rs` | `Planner::threads`; the count passed to the guard and recorded in every plan | F6b |
| `rlst/src/fftw/plan.rs` | `Plan::threads`; the thread count in "Threads and determinism" | F6b |
| `rlst/src/fftw/version.rs`, `backend.rs` | the minimum 3.3.6 with `fftw_threads` | F6b |
| `rlst/build.rs` | the threads libraries | F6b |
| `rlst/Cargo.toml` | `fftw_threads = ["fftw_system"]` | F6b |
| `rlst/src/fftw.rs` | `pub mod wisdom;`, `#[cfg(feature = "fftw_threads")] pub mod threads;`, module documentation | F6a, F6b |
| `rlst/src/fftw/concurrency_tests.rs` | the tests of work item 5 | F6a, F6b |
| `rlst/tests/fftw_determinism.rs` | the cases and scenarios of work item 6 | F6a, F6b |
| `.github/workflows/run-tests.yml` | the commands of work item 7 (DC43) | F6b |
| `README.md` | `fftw_threads` in "System FFTW" | F6b |
| `rlst/CHANGELOG.md`, `docs/fftw/README.md`, `CLAUDE.md` | yes | F6a, F6b |

## 1. Wisdom (`wisdom.rs`, `real.rs`; F6a)

The functions, generic over the precision `T: FftwReal`, as DC38 settles (open question 1):

```rust
pub mod wisdom {
    pub fn export<T: FftwReal>() -> String;
    pub fn import<T: FftwReal>(wisdom: &str) -> Result<(), FftwError>;
    pub fn export_to_file<T: FftwReal>(path: impl AsRef<Path>) -> Result<(), FftwError>;
    pub fn import_from_file<T: FftwReal>(path: impl AsRef<Path>) -> Result<(), FftwError>;
    pub fn forget<T: FftwReal>();
    pub fn import_system_wisdom<T: FftwReal>() -> Result<(), FftwError>;
}
```

- **Under the lock.** Every FFTW call runs inside `with_fftw` (design, Sections 3 item 2 and
  4.7). File I/O does not: `export_to_file` exports under the lock and writes the file after
  releasing it; `import_from_file` reads the file first and imports under the lock.
- **Export** through `fftw_export_wisdom(write_char, data)`, as Section 4.13 says: an
  `extern "C"` callback appends each character to a `Vec<u8>` passed as `data`, so no C
  allocation has to be freed. FFTW writes ASCII; the conversion to `String` must not panic.
- **Import** through `fftw_import_wisdom(read_char, data)`: the callback returns the bytes of
  the `&str` one by one and `libc::EOF` at its end, so no `CString` is built; an interior NUL
  reaches FFTW's parser like any other byte and fails it. FFTW's result 0 is
  `WisdomImportFailed { precision }` (Section 4.15). The rustdoc says that FFTW gives no
  reason, and that FFTW 3.3.11 leaves the wisdom unchanged after a failed import (fact 2),
  which RLST does not rely on.
- **Forget** through `fftw_forget_wisdom`. Existing plans are not affected (FFTW manual);
  later plans no longer find the forgotten wisdom.
- **System wisdom** through `fftw_import_system_wisdom`, only when called: FFTW's own file
  under `/etc/fftw/`, one per precision. Without such a file FFTW returns 0, which is the
  same `WisdomImportFailed`; the rustdoc says so (fact 5).
- **Callbacks** are `extern "C" fn`s that cannot unwind: they do nothing that can panic, since
  a panic in them aborts the process.
- **`SAFETY:`** comments name the conditions: the FFTW lock is held; `data` points to a live
  buffer or cursor for the duration of the call, which FFTW passes back to the callback only
  during it; the callback's signature matches fftw-sys's `fftw_write_char_func` or
  `fftw_read_char_func`.
- **The module documentation** states Section 5 rule 7 (import the same wisdom in every
  process before any planning, plan wisdom-only, compare fingerprints), rule 3's wisdom
  condition (importing wisdom of greater rigor changes later `Estimate` plans of those
  problems, fact 4), that wisdom is kept per precision and per thread count (facts 3 and 8),
  and that wisdom from a CPU with other SIMD support may not apply. It says that a
  wisdom-only plan finds the wisdom only when made with the planner options the wisdom was
  made with, other than `wisdom_only`, and a rigor no greater.

## 2. The `fftw_threads` feature (`Cargo.toml`, `build.rs`, `version.rs`; F6b)

- **Feature**: `fftw_threads = ["fftw_system"]`, with a comment in `rlst/Cargo.toml` (design,
  Section 4.10). It enables the `threads` module and `Planner::threads`.
- **Linking** (design, Section 4.10; fact 6): for each of `fftw3` and `fftw3f`, `build.rs`
  looks for `lib{name}_threads` (shared or static) in the directories pkg-config reports for
  `{name}`, or in the Homebrew prefix of the fallback, emits `rustc-link-lib={name}_threads`
  **before** `{name}` (design, Section 3 item 12), and fails with a message naming the
  directories searched if it is missing. Never the `_omp` libraries. With `DOCS_RS` set,
  discovery is skipped, as today.
- **Minimum 3.3.6** with `fftw_threads` (DF8), for a working `make_planner_thread_safe`
  (design, Section 3 item 3): `version.rs`, shared by `build.rs` and the crate, gives the
  minimum by feature, so that both the build-time check and the run-time check of DC13, which
  covers the Homebrew fallback, use it. `VersionTooOld` names that minimum.
- Without `fftw_threads`, nothing in RLST references a symbol of the threads libraries, or
  builds with `fftw_system` alone fail to link.

## 3. Initialisation and the settings guard (`lock.rs`, `threads.rs`, `real.rs`; F6b)

- **Initialisation** (design, Sections 3 item 12 and 4.14): with `fftw_threads`, the first
  acquisition of `with_fftw` calls `fftw_init_threads` and `fftwf_init_threads`, after the
  version check, and records in `FftwState` whether both succeeded. Every FFTW call RLST
  makes, those of the 0.10 module included, runs inside `with_fftw`, and execution needs a
  plan, so this precedes RLST's first FFTW call. Calls that other libraries made earlier are
  outside RLST's control; the `threads` module documentation says so.
- **The settings guard** sets the plan's thread count with `fftw_plan_with_nthreads` beside
  the time limit, and resets it to 1 in `Drop`, during unwinding as well (design, Section
  4.7). It calls `fftw_plan_with_nthreads` only if initialisation succeeded, and only with
  `fftw_threads`. At rest the count is 1, so the 0.10 module keeps planning single-threaded.
- **`threads::make_planner_thread_safe()`** calls `fftw_make_planner_thread_safe` under the
  lock, at most once per process (a flag in `FftwState`). Its rustdoc states the limits of
  Section 3 item 3: FFTW's lock covers plan creation and destruction only, not wisdom, the
  thread count or the time limit; it does not work with FFTW's OpenMP library; FFTW's manual
  advises against it unless there is no other choice. It is needed only when another library
  in the process plans with FFTW (Section 5 rule 1, DF7).
- The `threads` module documentation states Section 5 rule 1: one lock for RLST's calls, its
  limits, and what `make_planner_thread_safe` adds.

## 4. Planner and plans (`planner.rs`, `plan.rs`; F6b)

- **`Planner::threads(n: NonZeroUsize)`**, present only with `fftw_threads`, as DC41 settles
  (open question 4): the planner records `n`, and every
  planner method, those of every family, on scratch, `*_on` and in place, passes it to the
  settings guard. Its rustdoc says that a threaded plan is another plan than a
  single-threaded one, with another fingerprint even for small problems (fact 7), and can
  round differently (Section 5 rule 6).
- **`Plan::threads() -> usize`**, on every plan, returns the thread count the plan was made
  for, 1 without the feature; the plan's `Debug` output shows it.
- **Errors** (DC41): planning returns `InvalidThreadCount { requested }` when `n` exceeds
  `c_int::MAX`, and `ThreadsUnavailable` when `n > 1` and thread initialisation failed, before
  FFTW is called.
- **"Threads and determinism"** on `Plan` (DC23): the list of rule 3 gains the thread count;
  the bullet on batch and threads gains rule 6's last sentence, that a threaded plan can round
  differently from a single-threaded one; and a sentence states Section 4.14: executing a
  threaded plan from several threads at once is allowed but oversubscribes the machine,
  notably under rayon.

## 5. Concurrency tests (`concurrency_tests.rs`; design Section 7.5)

Ordinary unit tests, in the test process, with `Rigor::Estimate` only, `std::thread::scope`,
`Barrier`, fixed thread counts and F1's seeded inputs, as in F4. **No unit test imports or
forgets wisdom** (DC39, open question 2): those change the wisdom of every later test in the
process.

F6a:

- **Export during planning** (Section 7.5 third item, export half): 4 threads export f64 and
  f32 wisdom in a loop while 4 threads make, execute and drop plans as in F4's "planning while
  executing". Every export starts with `(fftw-3.` and names `fftw_wisdom` or `fftwf_wisdom`
  by precision; the planning threads' results are as F4 asserts.
- **Errors without FFTW**: `export_to_file` into a directory that does not exist returns
  `WisdomFile` (DC38) and leaves the process's wisdom as it was (export reads it only);
  the messages of the new and changed variants.

F6b, with `fftw_threads`:

- **Threaded against single-threaded** (Section 7.5 last item): c2c along axis 0 of `[1080,
  64]` and of `[140, 8]`, f64 and f32, the three policies, and one r2c row, with 2 and 4
  threads; each result within F1's tolerance of the single-threaded plan's; `Plan::threads()`
  is the count.
- **Threaded plans from several threads at once** (Section 7.5 last item): 3 threads each
  execute one 2-thread plan 4 times on their own buffers; each result equals, bit for bit,
  that of the same plan executed sequentially (Section 5 rule 3, which counts the thread count
  among the plan's settings). A failure is an open question, not a relaxed assertion.
- **The count does not leak** (Section 4.7): after a threaded plan, and after a panic inside
  the critical section with a thread count set (the existing lock test, extended), a plan
  without `threads` has `threads() == 1` and the fingerprint of the same problem planned
  before.
- **`threads(1)`** gives the fingerprint of a plan without `threads`.

Budget: these tests add at most about 1 s in a debug build over both pull requests; measure
and report each.

## 6. The determinism binary (`rlst/tests/fftw_determinism.rs`; design Section 7.6)

New rows of `CASES` and `SCENARIOS`, in F4's structure. Every buffer compared across
processes is at alignment class 0. Problems are named as in `estimate`.

F6a:

- **`wisdom-round-trip`** (Section 7.6 third item; fact 4): for nd-fmm's unit n_phi = 588,
  9 rings, forward, f64, and n_phi = 140 in f32: a `Rigor::Measure` plan A, executed; its
  wisdom exported to a string; `forget`; a wisdom-only `Measure` plan of the problem returns
  `PlanCreationFailed`; the string imported; a wisdom-only `Measure` plan B, executed. The
  child asserts that A and B have equal fingerprints and equal bits, and that f64 wisdom
  given to `import::<f32>` returns `WisdomImportFailed` (fact 3); it prints the lines of A
  and B, and whether a failed import of a malformed string left the export unchanged
  (fact 2, reported only).
- **`wisdom-export`** and **`wisdom-import`** (DC40, open question 3; Section 5 rule 7): the first
  plans the problems of `wisdom-round-trip` with `Measure`, prints their lines, and writes
  both precisions' wisdom with `export_to_file` to files in `CARGO_TARGET_TMPDIR` named by
  the parent's process id (`std::os::unix::process::parent_id`), which both children share.
  The second, as its first FFTW work, imports the files with `import_from_file`, removes them,
  plans the same problems wisdom-only with otherwise the same planner options, and prints
  their lines.
- **`wisdom-concurrent`** (DC39, open question 2; Section 7.5 third item): 4 threads plan and
  execute the problems of `estimate` in a loop, while 2 threads export, forget and import
  their own exports in a loop, for both precisions; afterwards the child plans the problems of
  `estimate` once more and prints their lines.
- **`system-wisdom`**: imports the system wisdom of both precisions and prints whether each
  succeeded. Nothing else plans in it.

| Scenario | Children | Assertion |
| --- | --- | --- |
| Wisdom round trip | `wisdom-round-trip` once | the child's own assertions |
| Shared wisdom | `wisdom-export` once, then `wisdom-import` once | the two outputs are equal line for line; a failure is raised as a design question |
| Wisdom during planning | `estimate` once, `wisdom-concurrent` once | its lines equal those of `estimate` (DC22: wisdom from `Estimate` planning does not change `Estimate` plans) |
| System wisdom | `system-wisdom` once | exit status 0; the outcomes are printed |

F6b, with `fftw_threads`, as DC42 settles (open question 5):

- **`estimate-threads`**: nd-fmm's units of `estimate` (f64, both directions, and n_phi = 140
  in f32) with `threads(2)` and `threads(4)`, `Estimate`.
- **`estimate-after-threads`**: threaded `Estimate` and `Measure` plans of the problems of
  `estimate`, then the problems of `estimate` without threads (fact 8).
- **`planner-thread-safe`**: `make_planner_thread_safe()` first; then 4 threads plan and
  execute the problems of `estimate` concurrently; then one thread plans them again and prints
  their lines.

| Scenario | Children | Assertion |
| --- | --- | --- |
| Threaded fresh processes agree | `estimate-threads` twice | the two outputs are equal line for line |
| Threads leave no trace | `estimate` once, `estimate-after-threads` once | its lines equal those of `estimate`; a failure is raised as a design question |
| Thread-safe planner | `estimate` once, `planner-thread-safe` once | its lines equal those of `estimate` |

The parent's summary lists the threaded fingerprints, which differ between the Linux and macOS
runners, as F4's do. Budget: the binary stays within about 10 s in a debug build, children
included; measure and report.

## 7. CI (`run-tests.yml`; F6b)

As DC43 settles (open question 6), each job gains, beside its commands without the feature:

```bash
cargo clippy --features "fftw,fftw_system,fftw_threads,strict" -- -D warnings
cargo test --features "fftw,fftw_system,fftw_threads,strict"
```

and `CLAUDE.md`'s validation commands list them, its Clippy commands becoming seven. F8a
reviews the matrix as a whole. The
pull request reports the determinism binary's summary from both jobs, with and without the
feature.

## 8. Documentation

- Rustdoc with an example for every new public item. Examples that import or forget wisdom,
  or plan with `Measure`, are `no_run` unless each doctest is shown to run in a process of its
  own: they would change the wisdom of the doctests around them. The binary runs them.
- The module documentation of `rlst/src/fftw.rs` lists `wisdom` and `threads`, and
  `Planner`'s options gain `threads`.
- Items gated by `fftw_threads` carry `#[cfg_attr(docsrs, doc(cfg(feature = "fftw_threads")))]`,
  as design Section 6 asks of `fftw`; F7, which adds the docs.rs metadata, includes
  `fftw_threads` in its features so that they appear on docs.rs (recorded here for F7).
- `rlst/CHANGELOG.md`: the entry "New FFTW interface (in progress)" gains a paragraph per pull
  request: wisdom (F6a); `fftw_threads`, its minimum 3.3.6 and the threads libraries it links
  (F6b).
- `README.md`, "System FFTW": `fftw_threads` and the threads libraries it needs (F6b).
- `CLAUDE.md`: the features list gains `fftw_threads`; the FFTW rules say that tests which
  import or forget wisdom belong in the binary; the validation commands follow DC43.
- `docs/fftw/README.md`: the files table lists this brief; the status table is updated when
  the last pull request merges.

## Pull requests and CI

- Two pull requests, each branched from `fftw-0.11` after the previous one merged, titled for
  example `feat(fftw): F6a wisdom` and `feat(fftw): F6b fftw_threads`, each with the acceptance
  items it closes ticked.
- The open questions below are settled before the task starts; a question that arises during
  the task is recorded below and raised in the pull request.
- CI runs for pull requests into `fftw-0.11`. Report the determinism binary's summary from
  both jobs in each pull request.

## Acceptance

- [ ] The wisdom functions of work item 1 exist for both precisions, as DC38 settles; every FFTW call runs inside `with_fftw`, file I/O outside it.
- [ ] `fftw_threads` implies `fftw_system`, links `fftw3_threads` and `fftw3f_threads` before
      `fftw3` and `fftw3f`, fails the build with a clear message when they are missing, and
      raises the minimum to 3.3.6 at build time and at run time; builds without it reference
      no symbol of the threads libraries.
- [ ] Threads are initialised once, inside the lock, before RLST's first FFTW call; the
      settings guard sets and resets the thread count; `make_planner_thread_safe` exists with
      the limits of design Section 3 item 3 in its rustdoc.
- [ ] `Planner::threads` and `Plan::threads` exist as DC41 settles, with its errors, for every
      family and planner method.
- [ ] The tests of work item 5 pass in the default FFTW test run and with `fftw_threads`, in CI
      on Linux and macOS; no unit test imports or forgets wisdom; their added time in a debug
      build is reported and is at most about 1 s.
- [ ] The cases and scenarios of work item 6 pass in CI on Linux and macOS, with and without
      `fftw_threads`; the binary's summaries are in the pull requests; its time in a debug
      build is at most about 10 s.
- [ ] CI runs the commands of work item 7 (DC43), and `CLAUDE.md` lists them.
- [ ] The rustdoc states Section 5 rules 1 and 7 (in `threads` and `wisdom`) and the thread
      count in rules 3 and 6 (on `Plan`); every new public item has an example.
- [ ] Every `unsafe` block has a `SAFETY:` comment naming the conditions it relies on; no
      dependency was added.
- [ ] The seven Clippy commands of `CLAUDE.md` (six before F6b) pass (or are
      reported unavailable), as do `cargo fmt -- --check`, `cargo test`,
      `cargo test --features "fftw,fftw_system,strict"`,
      `cargo test --features "fftw,fftw_system,fftw_threads,strict"` and
      `cargo clippy --tests --features "fftw,fftw_system,fftw_threads,strict" -- -D warnings`.
- [ ] `rlst/CHANGELOG.md`, `README.md`, `CLAUDE.md` and `docs/fftw/README.md` are updated.

## Pitfalls

- **Wisdom is global to the process** (design, Section 3 item 4), and estimate planning adds
  to it (fact 1). Unit tests may export; importing, forgetting and `Measure` planning belong
  to the binary's children (DC39). In a child of a scenario that compares with
  `estimate`, nothing may plan before the case starts.
- **Symbols of the threads libraries** (fact 6). Every reference to `fftw_init_threads`,
  `fftw_plan_with_nthreads` and `fftw_make_planner_thread_safe`, in `FftwReal` and in the
  settings guard as well, is behind `fftw_threads`; otherwise `cargo test --features
  "fftw,fftw_system"` fails to link.
- **Link order.** `pkg_config::Config::probe` prints its `rustc-link-lib` lines as it runs.
  Probe with `cargo_metadata(false)`, emit the threads library, then emit what pkg-config
  found, so that `-lfftw3_threads` precedes `-lfftw3`. Check with `cargo build -vv` on Linux,
  where the order matters for static libraries and `--as-needed`.
- **Features in `build.rs`.** The build script is compiled with the package's features, so
  `#[cfg(feature = "fftw_threads")]` works there and in `version.rs`; keep `version.rs`
  dependent on `core` only.
- **Callbacks across FFI** must not unwind; keep them to a push or a read.
- **The lock and I/O.** Never hold the FFTW lock during file I/O: a slow file system would
  block every planner in the process.
- **Reset on every path.** The thread count is reset to 1 on error paths and during
  unwinding, by the guard, as the time limit is; a thread count that leaks changes another
  caller's plans silently.
- **Wisdom-only needs the same options** (fact 8): the same thread count, input policy and
  alignment class as when the wisdom was made, and a rigor no greater. The `wisdom-import`
  child uses the options of `wisdom-export`.
- **The shared wisdom files**: in `CARGO_TARGET_TMPDIR`, named by the parent's process id so
  that concurrent runs of the binary do not collide, removed by the importing child, and
  never left behind on success. `std::os::unix` is fine: Windows is not supported (DC8).
- **Cases behind a feature.** `#[cfg]` on elements of an array expression is not stable;
  build the threaded rows of `CASES` and `SCENARIOS` another way, for example as separate
  tables that the parent chains with `fftw_threads`.
- **Thread counts.** Fixed (2, 4) and small, for debug builds and the macOS runner's 3 cores;
  oversubscription is intended. Threaded plans of tiny problems are still threaded (fact 7).
- **SIMD.** Fingerprints and bits are compared only within one machine and build, never with
  hard-coded values.
- **`strict`, wildcard imports, debug builds**: as in the earlier briefs.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design questions
in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.16, DC38–DC43, Appendix A): all six with the option marked *proposed*. The
work items follow these answers.

1. **The wisdom functions' types and file errors.** Design Section 4.13 names the functions
   without types. `FftwError` is `Clone + PartialEq + Eq`, so it cannot hold a
   `std::io::Error`, and Section 4.15 has no variant for file errors. Options:
   (a) *proposed*: the signatures of work item 1, with `export` and `forget` infallible, and a
   new variant `FftwError::WisdomFile { path: PathBuf, kind: std::io::ErrorKind, message:
   String }` for the file functions, so that every fallible function of `rlst::fftw` returns
   `FftwError`; (b) the file functions return `std::io::Result<()>`, an FFTW rejection mapped
   to `io::ErrorKind::InvalidData` with `WisdomImportFailed` as its source; (c) no file
   functions, users write `export()` and read `import()` with `std::fs` (Section 4.13
   corrected).
   *Decided: (a), DC38.*
2. **Where the tests that change the wisdom run.** Design Section 7.5 puts "wisdom export and
   import run concurrently with planning" among the concurrency tests, which run in the test
   process, but importing or forgetting wisdom there changes the wisdom every later unit test
   plans with (`CLAUDE.md`'s rule for tests). Options: (a) *proposed*: unit tests only export,
   concurrently with planning; import and forgetting concurrent with planning run in a child
   of the binary (`wisdom-concurrent`), whose later plans are compared with `estimate`;
   (b) the unit tests import only wisdom exported from the same process and forget nothing,
   relying on DC22 inside the test process.
   *Decided: (a), DC39.*
3. **Wisdom shared between processes.** Section 7.6 tests export, forget and import in one
   process; Section 5 rule 7, and the option of Section 11 item 4, rely on wisdom shared
   between processes giving the same plans. Options: (a) *proposed*: the `shared wisdom`
   scenario of work item 6 asserts that a process that imports another's wisdom and plans
   wisdom-only gets the fingerprints and bits of the exporting process, and a failure is a
   design question, as DC22; (b) the scenario reports whether they are equal; (c) no such
   scenario.
   *Decided: (a), DC40.*
4. **The threads API and its errors.** Section 4.14 names `Planner::threads(n)` and says that
   the plan records `n`, without types; FFTW takes an `int`, and `fftw_init_threads` can fail
   (Section 3 item 12), for which Section 4.15 has no variant. Options: (a) *proposed*:
   `Planner::threads(n: NonZeroUsize) -> Self`, present only with `fftw_threads`;
   `Plan::threads() -> usize` on every plan, 1 without the feature; planning returns a new
   `FftwError::InvalidThreadCount { requested }` when `n` exceeds `c_int::MAX`, and a new
   `FftwError::ThreadsUnavailable` when `n > 1` and thread initialisation failed;
   `threads::make_planner_thread_safe()` returns `()`; (b) `Planner::threads(n: usize)` and
   `Plan::threads` always present, with `n = 0` an error and `n > 1` an error without the
   feature; (c) as (a), with `n` clamped to `c_int::MAX` instead of an error.
   *Decided: (a), DC41.*
5. **Threaded plans in the determinism binary.** Section 7.6 lists no threaded case, but
   Section 5 rule 3 counts the thread count among a plan's settings, which claims that
   threaded `Estimate` plans are the same in every fresh process, and Section 4.7 says the
   count never leaks into another plan. Options: (a) *proposed*: the three threaded scenarios
   of work item 6, each asserting; a failure of "threads leave no trace" or of the threaded
   fresh processes is a design question; (b) the threaded cases report only, as `estimate
   after measure` does; (c) no threaded cases, the unit tests of work item 5 only.
   *Decided: (a), DC42.*
6. **CI for `fftw_threads`.** Section 7.7 gives "with and without `fftw_threads`" to F8a, so
   F6b's code would run in no CI job until then. Options: (a) *proposed*: F6b adds the Clippy
   and test commands of work item 7 to both jobs, beside the existing ones, and `CLAUDE.md`
   lists them; F8a reviews the matrix as a whole; (b) F8a adds them, and F6b reports local
   runs on macOS only.
   *Decided: (a), DC43.*
