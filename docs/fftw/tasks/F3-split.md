# F3: Split complex-to-complex transforms with the part axis

| | |
| --- | --- |
| Task | F3 (design, Section 9) |
| Status | not started |
| Depends on | F1, F2, F4 |
| Pull requests | one, into `fftw-0.11`: F3 (split layouts, the `SplitC2c` family, execution, tests, the split rows of the determinism binary) |
| Read first | design: Section 2.5, Section 3 items 1, 5–8, Sections 4.2–4.8, Section 5, Sections 7.1–7.6, Section 9 (F3 row), Section 10, in particular DC1, DC3, DC11–DC29, Section 11 item 2, Appendix A (drafts 0.8 to 0.13); `docs/fftw/tasks/F1-core.md`, `F2-layouts.md` and `F4-determinism.md`, including their "Open questions"; `docs/fftw/F0-findings.md`; `CLAUDE.md` |

## Goal

Complex-to-complex transforms on split storage: the real and imaginary parts in one real
buffer, separated by a fixed gap, as nd-fmm's split planes keep them (design, Sections 2.5
and 4.5). A split complex array is a **real** array with one extra axis of extent 2, the
*part axis*, whose stride is the gap. `Layout::split_along_axis`, `Layout::split_over_axes`
and their in-place variants build split layouts from RLST arrays and views of any memory
order; `Planner::split_c2c` and its companions make `Plan<T, SplitC2c, P>` plans with the
three input policies; execution checks real arrays, views and slices as F2 does for complex
ones, the gap included. Everything is tested against F2's logical reference and against
interleaved plans, and the determinism binary of F4 gains split rows. After F3, nd-fmm has
both storage forms its spike S9 chooses between (design, Section 11 item 2); RLST decides
nothing for it.

## Scope

In scope:

- the split representation in `Layout`, and split layouts from raw dimensions and from
  arrays (work item 1);
- the `SplitC2c` family, FFTW's guru64 split planner and executor in `FftwReal`, and the
  planner methods with scratch and `*_on` planning (work item 2);
- execution on real slices, arrays and views, with the checks of design Section 4.8 for
  split storage, and the swap of parts for backward transforms (work item 3);
- the tests of design Sections 7.2, 7.3 and 7.4 for split c2c (work item 4);
- split rows in `rlst/tests/fftw_determinism.rs` and in the concurrency tests (work item 5);
- rustdoc, the module documentation of `rlst/src/fftw.rs`, `rlst/CHANGELOG.md`, the status
  table of `docs/fftw/README.md`, `CLAUDE.md` (checked), and this brief's "Open questions"
  (work item 6).

Out of scope, for later tasks:

- split r2c and c2r (F5), which use a split complex array on one side only; work item 1
  keeps the representation open for them (DC25);
- r2c, c2r, r2r and `Scale` (F5); wisdom and threads (F6); the user guide in
  `rlst/src/doc/optional_integrations.rs` and the removal of the 0.10 module (F7);
- benchmarks of split against interleaved plans (F8b, design Section 7.7);
- whether nd-fmm uses split planes or interleaved scratch: that is spike S9's decision
  (design, Section 11 item 2);
- negative strides, so a buffer with the imaginary plane before the real plane (design,
  Section 4.3: strides are non-negative in 0.11);
- any change to the interleaved c2c API of F1 and F2, other than the rejection of a split
  layout by the interleaved planner methods (DC28). A reason to change more is
  an open question, not an edit.

Files touched:

| File | Change |
| --- | --- |
| `rlst/src/fftw/layout.rs` | the split representation, the split constructors, the per-side and pairing checks with the part axis, tests |
| `rlst/src/fftw/real.rs` | `plan_guru64_split_dft` and `execute_split_dft` in the sealed trait |
| `rlst/src/fftw/plan.rs` | the `SplitC2c` family, `direction()` for it, rustdoc with `compile_fail` examples |
| `rlst/src/fftw/options.rs` | `PolicyFor<SplitC2c>` for the three requests |
| `rlst/src/fftw/planner.rs` | `split_c2c`, `split_c2c_on`, `split_c2c_inplace`, `split_c2c_inplace_on`; scratch in reals; the rejection of mismatched layouts |
| `rlst/src/fftw/buffer.rs` | the split side checks for slices and arrays |
| `rlst/src/fftw/execute.rs` | `execute` for the three policies of `SplitC2c`, the `SAFETY:` comment (item 3 of the new-array conditions now applies) |
| `rlst/src/fftw/execute/split_tests.rs` (new) | the tests of work item 4 |
| `rlst/src/fftw/test_support.rs` | helpers for split views: values from and into the two parts |
| `rlst/src/fftw/concurrency_tests.rs` | a split row in "one plan, many threads" |
| `rlst/tests/fftw_determinism.rs` | split problems in `estimate`, a `split-fixed-batch-units` case and its scenario |
| `rlst/src/fftw.rs` | re-exports, module documentation |
| `rlst/CHANGELOG.md`, `docs/fftw/README.md`, `CLAUDE.md` | yes |

## 1. Split layouts (`layout.rs`)

**Representation.** A split layout is a `Layout` whose transform and batch dimensions count
reals, with a gap on each side: the stride, in reals, from an element's real part to its
imaginary part. The gaps are recorded in `Layout` beside the dimensions, never as a batch
dimension, so batch normalisation never merges the part axis into another axis. How the
gaps are recorded and exposed, and how a raw split layout is built for slices, is DC25:
`Layout::gaps() -> Option<(isize, isize)>` and
`Layout::new_split(dims, batch, (input_gap, output_gap), placement)`.

**Rules of a split layout**, checked in `Layout::new_split` and by the array functions
through it, each failure an `InvalidLayout` naming `part axis` or the dimension:

1. the rules of `Layout::new` for the dimensions (extents at least 1, non-negative strides,
   equal strides in place);
2. both gaps are positive; in place they are equal;
3. the distinct-output rule of design Section 4.3 holds with the part axis included as a
   dimension of extent 2 and output stride equal to the output gap, so that no real or
   imaginary part of one output element is another element's part;
4. the spans include the part axis: the span of a side is that of the dimensions plus its
   gap, in reals.

**Layouts from arrays.** `Layout::split_along_axis(&input, part_in, &output, part_out, axis)`
is `split_over_axes(&input, part_in, &output, part_out, &[axis])`, the signature of design
Section 4.5; `split_along_axis_inplace(&data, part, axis)` and
`split_over_axes_inplace(&data, part, &axes)` take one array. Both arrays are real arrays
(`BaseItem<Item = T>` with `T: FftwReal`; DC29) of the same rank `N`, which
includes the part axis. The checks, in this order, each returning `InvalidLayout` naming
the axis:

1. `part_in` and `part_out` are below `N`; the extent of each array along its part axis is
   2;
2. the checks of F2's `over_axes`, on the **logical axes**: each array with its part axis
   removed, axes numbered from 0 in their order (DC26): `axes` not empty, below
   `N − 1`, not repeated; equal logical shapes (DC14); extents at least 1; extents and
   strides fit in `isize`;
3. the transform dimensions are the logical axes of `axes`, in that order, with the strides
   of both arrays; the batch dimensions are the other logical axes; the gaps are the
   strides of the two part axes. The result is made by `Layout::new_split`, whose rules
   apply.

Expected layouts, which the tests assert with `Layout::new_split` (strides and gaps in
reals; `(n, is, os)`):

| Arrays (shape, strides, part axis) | Call | Transform dims | Batch dims | Gaps |
| --- | --- | --- | --- | --- |
| nd-fmm's box, column-major `[140, 9, 2]` (1, 140, 1260), part 2 | `split_along_axis(…, 0)` | `(140, 1, 1)` | `(9, 140, 140)` | `(1260, 1260)` |
| column-major `[2, 8, 3]` (1, 2, 16), part 0 | `split_along_axis(…, 0)` | `(8, 2, 2)` | `(3, 16, 16)` | `(1, 1)` |
| row-major `[3, 8, 2]` (16, 2, 1), part 2 | `split_along_axis(…, 1)` | `(8, 2, 2)` | `(3, 16, 16)` | `(1, 1)` |
| column-major `[8, 2, 3]` (1, 8, 16), part 1 | `split_along_axis(…, 1)` | `(3, 16, 16)` | `(8, 1, 1)` | `(8, 8)` |
| column-major `[4, 5, 2]` (1, 4, 20), part 2 | `split_over_axes(…, &[0, 1])` | `(4, 1, 1)`, `(5, 4, 4)` | none | `(20, 20)` |
| input `[8, 3, 2]` (1, 8, 24), part 2; output `[2, 8, 3]` (1, 2, 16), part 0 | `split_along_axis(…, 0)` | `(8, 1, 2)` | `(3, 8, 16)` | `(24, 1)` |

The second and third rows are the same memory and give equal layouts: a plan describes
memory, not logical axes (DC15). Building a layout reads only shapes and strides.

The interleaved layout functions of F2 stay as they are, and `Layout::new` makes layouts
without gaps.

## 2. The family and the planner (`plan.rs`, `real.rs`, `options.rs`, `planner.rs`)

- **`SplitC2c`**, a family marker like `C2c` (design, Section 4.1), with
  `Plan::<T, SplitC2c, P>::direction()`. `PolicyFor<SplitC2c>`: `Preserving` by default and
  after `preserve_input()`, `Destroying` after `may_destroy_input()` (DC1).
- **FFTW calls.** The sealed `FftwReal` gains `plan_guru64_split_dft` and
  `execute_split_dft` (`fftw_` and `fftwf_`, in fftw-sys 0.8 and in FFTW since 3.2), with
  the contract of the existing functions. Planning runs inside `with_fftw` with the
  settings guard, as for c2c.
- **Planner methods**, as for c2c: `split_c2c(&layout, direction)` and
  `split_c2c_on(&mut input, &mut output, &layout, direction)` on `Planner<T, S>` with
  `S: PolicyFor<SplitC2c>`; `split_c2c_inplace` and `split_c2c_inplace_on` on
  `Planner<T, DefaultRequest>` only (DC12). Buffers are `FftBufferMut<T>`: real slices,
  arrays and views.
- **The direction (design, Section 3 item 8, Section 4.5).** FFTW's split planner has no
  sign; it always plans a forward transform. A backward transform of `x` is the forward
  transform with the real and imaginary parts swapped on both sides. A plan records the
  pointers it passes FFTW as offsets from the real part of each side: forward, the real part
  and the real part plus the gap; backward, the real part plus the gap and the real part, so
  that FFTW sees the signed gap `−gap`. Planning and execution pass FFTW the same
  arrangement. Nothing in the layout or the public API changes with the direction.
- **Alignment classes (design, Section 4.6).** The class of a side is the class of its real
  part, which is the first element of a split array (the part index 0 on the part axis) and
  the start of a slice. `Planner::alignment(input, output)` and `Plan::alignment()` give these
  classes; a class must be a multiple of `align_of::<T>()`. The class of the imaginary part
  follows from the class of the real part and the gap, which execution checks equal to the
  plan's, so checking the real part's class covers both; the `SAFETY:` comment says so.
- **Scratch (DC3).** Planning on scratch allocates, for each side, the layout's span in reals
  at the requested class, `max_scratch` counting bytes as for c2c; the scratch helper of F1
  is generalised from complex to real elements.
- **Layouts of the other storage.** `split_c2c` and its companions reject a layout without
  gaps, and `c2c` and its companions reject a layout with gaps, before FFTW is called
  with `InvalidLayout` naming `part axis` (DC28).
- **Fingerprints, descriptions, `Debug`, `Send + Sync` and destruction** work for every
  family already and need only tests.

## 3. Execution (`buffer.rs`, `execute.rs`)

- **Slices.** The layout's offsets count reals from the start of the slice; the real part of
  each side starts at the slice's first element, the imaginary part at the gap. The span
  check covers the part axis.
- **Arrays and views, per side.** As F2's per-side rule (DC15), with the part axis found
  first: an axis of extent 2 whose stride equals that side's gap, not needed by a transform
  dimension; the remaining axes, normalised on that side, equal the layout's batch on that
  side. Several axes can qualify only on an input (outputs address distinct elements), and
  the check accepts the array if any choice fits (DC27). The span check (DC21)
  covers the part axis.
- **Pairing (DC15).** When both buffers are arrays of the same rank, each side's part axis is
  removed as found above, and the remaining logical axes pair by index as in F2: equal
  shapes, each transform dimension matched by one axis with its extent and both strides, the
  other axes normalised together equal to the layout's batch (DC27).
- **Pointers.** The checks return the pointer to each side's real part, derived as in F2
  (`check_side_mut` for buffers the plan writes, DC21); execution forms the imaginary part's
  pointer from it and the gap with `wrapping_add`, inside the span the check accepted, and
  passes FFTW the arrangement of the plan's direction (work item 2).
- **Order of checks**, as for c2c: buffers against the layout, alignment classes, overlap
  (conservative, on the spans in bytes). A failed check returns its error and writes
  nothing.
- **`SAFETY:`.** Item 3 of FFTW's new-array conditions (design, Section 3 item 1) now
  applies: the gaps `ii − ri` and `io − ro` FFTW sees equal those of planning, since the
  layout check compared each side's gap with the plan's and the plan's direction fixes the
  arrangement. The other items are those of F2's comment.

## 4. Tests (`execute/split_tests.rs`, design Sections 7.2 to 7.4)

Every plan here uses `Rigor::Estimate` and assumes no SIMD. Inputs come from F1's seeded
generator; messages name the seed, the arrangement, the shape, the part axes, the axes, the
policy, the direction and the precision.

- **The reference.** A split view's complex values are read by logical multi-index from its
  two parts into a `DynArray` of `Complex<T>` and compared with F2's logical reference by
  logical multi-index, to F1's c2c tolerance. No reference is evaluated on the layout under
  test.
- **Fixed cells**, both precisions, both directions, the three policies:
  - part axis outermost, nd-fmm's planes: `[n, b, 2]`, `[n, 2]` with n in 1, 2, 17, 97,
    140; and `[6, 5, 4, 2]` over two and three axes;
  - part axis innermost: `[2, n, b]`, the memory of interleaved storage read as split;
  - part axis in the middle: `[n, 2, b]`;
  - the views of F2's matrix that apply to real arrays: a subview of a larger parent with
    sentinels around it, a strided view with steps of 2 over aligned storage, a slice of a
    parent of higher rank, row-major strides over aligned storage;
  - mixed part axes: input outermost, output innermost.
- **Sampled cells**: the rest of this product with sizes from F1's list, seeded, the seed
  printed on failure, with the budget below.
- **Split against interleaved (Section 7.3).** For nd-fmm's units `[n_phi, 9]`, n_phi in
  140 and 588, f64, both directions: the split plan on planes against the interleaved plan
  on the same values, to the c2c tolerance. Bits are not compared: the plans differ
  (design, Section 3 item 7).
- **Round trip**: forward then backward is N times the input, for one cell per arrangement.
- **One plan, many units in process**: one split plan of nd-fmm's box executed on every box
  `slice(3, k)` of a parent `[140, 9, 2, 4]` gives the bits of a plan made with
  `split_c2c_on` for each box and of the same plan on a standalone copy (as F2's
  fixed-batch test; fresh processes are work item 5).
- **Sentinels**: every view's parent outside the view, gaps between the planes included,
  holds its sentinel bit for bit after execution, for every policy.
- **Negative tests (Section 7.4)**, each returning its error with an output whose parent
  holds the sentinel afterwards: a wrong gap (planes of another size); a wrong part extent
  (3), a part axis not below the rank, and a logical axis not below `N − 1` at layout
  construction; a wrong logical shape; another
  alignment class of the real part; an in-place plan on two buffers and an out-of-place plan
  on overlapping planes; a split layout given to `c2c` and an interleaved layout given to
  `split_c2c`; the scratch cap with `split_c2c_on` as the remedy.
- **`compile_fail` doctests (Section 7.4)**: a split plan executed on complex arrays; a
  `Destroying` split plan with `&input`; an in-place split plan with two arrays. Each beside
  the example that compiles, as F1 and F2 do.

Budget: these tests add at most about 1 s in a debug build; measure and report.

## 5. Concurrency and determinism (`concurrency_tests.rs`, `tests/fftw_determinism.rs`)

- **Concurrency**: "one plan, many threads" gains a split row, nd-fmm's box `[140, 8, 2]`
  (2240 reals, a multiple of 64 bytes in both precisions), f64 and f32, the three policies,
  bit for bit against sequential execution.
- **`estimate`** gains split problems: nd-fmm's box `[140, 9, 2]`, part axis 2, both
  directions, f64, `Preserving`; `[588, 9, 2]` forward; `[140, 9, 2]` in f32; the part axis
  innermost `[2, 140, 9]`. Their lines follow the existing ones, so `fresh processes
  agree` and `planning history` cover them unchanged; `other_plans` gains split plans of
  other sizes.
- **`split-fixed-batch-units`**, a new case and scenario as `fixed-batch-units`: one split
  plan of nd-fmm's box, the first plan of its process, executed on the boxes of an aligned
  parent `[140, 9, 2, 4]` and on four boxes of a flat aligned buffer separated by 8 reals
  (64 bytes) of sentinel padding; the child asserts equal bits with a plan made with
  `split_c2c_on` for each box and with the same plan on a standalone copy, and the padding;
  the scenario runs it twice and compares the two outputs and its line with the split box
  of `estimate`.
- The binary's time stays within about 10 s in a debug build; measure and report.

## 6. Documentation

- Rustdoc with an example for every new public item. `Layout::split_along_axis` shows
  nd-fmm's planes: a box `[n_phi, rings, 2]`, the real plane then the imaginary plane.
- The rustdoc of `SplitC2c` and of `Planner::split_c2c` states that split plans can use
  fewer SIMD stages than interleaved ones (design, Sections 3 item 7 and 4.5), so that an
  interleaved copy can be faster, and points to the F8b benchmarks.
- The module documentation of `rlst/src/fftw.rs` lists split plans.
- `rlst/CHANGELOG.md`: the entry "New FFTW interface (in progress)" gains a paragraph on
  split storage.
- `docs/fftw/README.md`: the files table lists this brief; the status table is updated when
  the pull request merges. `CLAUDE.md` is checked and changes only if a command or layout
  entry does.

## Pull request and CI

- One pull request, titled for example `feat(fftw): F3 split c2c with the part axis`,
  branched from `fftw-0.11`, with this brief's acceptance list ticked.
- The open questions below were settled before the task started (design draft 0.13,
  DC25–DC29). A question that arises during the task is recorded below and raised in the
  pull request.
- CI runs for pull requests into `fftw-0.11`. Report the determinism binary's summary from
  both jobs in the pull request, as F4 did.

## Acceptance

- [ ] Split layouts are built from raw dimensions and from arrays as work item 1 says, and
      the six layouts of its table are asserted.
- [ ] `Plan<T, SplitC2c, P>` plans are made on scratch and on the caller's buffers, for the
      three policies and both precisions, and execute on real slices, arrays and views with
      the checks of work item 3.
- [ ] The tests of work item 4 pass in the default FFTW test run, in CI on Linux and macOS;
      their added time in a debug build is reported and is at most about 1 s.
- [ ] The split rows of work item 5 pass; the determinism binary's summary from both jobs is
      in the pull request, and its time in a debug build is at most about 10 s.
- [ ] Every `unsafe` block has a `SAFETY:` comment; the comment on split execution names the
      gaps of new-array condition 3 and the swap of parts.
- [ ] Every FFTW call other than execution runs inside `with_fftw`; no dependency was added;
      no interleaved c2c behaviour changed other than the rejection of split layouts.
- [ ] Every new public item has rustdoc with an example; the `compile_fail` doctests of work
      item 4 exist beside examples that compile.
- [ ] The six Clippy commands of `CLAUDE.md` pass (or are reported unavailable), as do
      `cargo fmt -- --check`, `cargo test`, `cargo test --features "fftw,fftw_system,strict"`
      and `cargo clippy --tests --features "fftw,fftw_system,strict" -- -D warnings`.
- [ ] `rlst/CHANGELOG.md` and `docs/fftw/README.md` are updated; `CLAUDE.md` is checked.

## Pitfalls

- **Units.** Split strides, gaps and spans count reals; interleaved ones count complex
  elements. The scratch, the span checks and the overlap check use the element size of the
  buffer, `size_of::<T>()` here.
- **The part axis is not a batch dimension.** Do not pass it to batch normalisation: a part
  axis of stride 1 next to a batch axis of stride 2 would merge into one batch of twice the
  extent, and the layout would describe interleaved storage with the wrong count.
- **The direction.** FFTW's split planner always plans forward. A backward plan swaps the
  real and imaginary pointers on both sides, at planning and at every execution; swapping on
  one side only, or only at execution, gives the conjugate of the wrong transform.
- **Pointers.** Form the imaginary part's pointer from the checked real-part pointer, with
  `wrapping_add`, and only after the span check; for a buffer the plan writes, both pointers
  derive from the exclusive borrow.
- **Alignment.** With gap g, the imaginary part's class is the real part's plus g ·
  `size_of::<T>()`, modulo 64. Boxes of nd-fmm's planes keep class 0 only when the box size
  in bytes is a multiple of 64 (design, Section 11 item 12): `[140, 9, 2]` in f64 is, and so
  is every box of an aligned parent of such boxes.
- **SIMD.** Split plans can mix buffered SIMD stages with scalar ones (design, Section 3 item
  7). Compare split with interleaved only to round-off, and depend on SIMD nowhere.
- **Axis numbering.** `axes` count logical axes (DC26), the
  array's axes without its part axis, so `axis 1` of an input with part axis 0 is its RLST
  axis 2. Errors name the logical axis and say so.
- **Wisdom, determinism, `strict`, wildcard imports, debug builds**: as in the F1, F2 and F4
  briefs. Unit tests plan with `Estimate` only.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design
questions in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.13, DC25–DC29, Appendix A): all five with the option marked *proposed*.
The work items follow these answers.

1. **How a split layout is represented and built from raw dimensions.** Design Section 4.2
   says a `Layout` holds "for split storage, the gap of each side", and Section 4.5 names
   only the array functions; F1's raw layouts (`Layout::new` from `Dim` lists, for slices)
   have no split counterpart. Options: (a) *proposed*: `Layout` records an optional gap per
   side, private, with `Layout::gaps() -> Option<(isize, isize)>` for split c2c and
   `Layout::new_split(dims, batch, (input_gap, output_gap), placement)` for slices; F5 adds
   what its split r2c and c2r need (a gap on one side only), which the per-side storage
   allows without a breaking change; (b) no raw split layouts: slices are wrapped in
   `SliceArray` views, and only the array functions build split layouts; (c) a separate
   `SplitLayout` type, which would also make the storage kind part of the planner's
   signature and make open question 4 a compile-time check, at the cost of a second layout
   type in every split family of F5. *Decided: (a), DC25.*
2. **What `axis` and `axes` count in the split layout functions.** The arrays include the
   part axis, and `part_in` and `part_out` may differ (design, Sections 4.5 and 7.2: the
   part axis outermost and innermost). Options: (a) *proposed*: logical axes, the array's
   axes without its part axis, numbered in order, so that one `axes` means the same
   transform on an input and an output whose part axes are at different positions; errors
   name the logical axis; (b) RLST axes of the arrays, which needs `part_in == part_out`,
   and then one `part` argument instead of the two of Section 4.5. *Decided: (a), DC26.*
3. **Finding the part axis at execution, and pairing (DC15).** A layout records gaps, not
   the position of the part axis, as it records no axes (DC15: a plan describes memory).
   Options: (a) *proposed*: per side, the part axis is an axis of extent 2 whose stride is
   that side's gap and that no transform dimension needs; if several qualify, which only an
   input can have, the check accepts the array if any choice fits; when both buffers are
   arrays of the same rank, each side's part axis is removed and the rest pair by index as
   in F2; (b) the layout records the part axes' positions, and execution requires arrays of
   the layout's rank with the part axis there, which ties a plan to logical axes. *Decided: (a), DC27.*
4. **The error for a layout of the other storage kind.** `split_c2c` given an interleaved
   layout, or `c2c` a split one. Design Section 4.15's list has no variant for it.
   Options: (a) *proposed*: `InvalidLayout` with dimension `part axis` and a reason naming
   the family and the storage the layout describes; (b) a new variant, for example
   `StorageMismatch { expected, found }`, like `PlacementMismatch` (`FftwError` is
   `#[non_exhaustive]`, so adding one is not breaking); (c) option 1 (c), which makes it a
   compile-time error. *Decided: (a), DC28.*
5. **The element type of the split layout functions.** Design Section 4.3 makes the layout
   functions generic over any element type, which lets the r2c functions of F5 take a real
   input and a complex output. A split layout counts reals, and one built from complex
   arrays would count complex elements and fail every execution check. Options:
   (a) *proposed*: the split functions require real items (`BaseItem<Item = T>` with
   `T: FftwReal`) on both arrays, a bound only; (b) any element type, as the interleaved
   functions, with the rustdoc saying that the arrays must be real. *Decided: (a), DC29.*

Raised during F3. None needs a design change; they are recorded for review and for F5,
F7 and nd-fmm.

6. **How the plan records the arrangement of the parts.** Work item 2 says a plan records
   the pointers it passes FFTW as offsets from the real part of each side. The plan stores
   no extra field: one crate-private function, `split_offsets(direction, gap)`, gives the
   offsets of FFTW's real and imaginary pointers from the plan's direction and the layout's
   gap, and planning and execution both call it, so they cannot pass FFTW different
   arrangements. The `SAFETY:` comment of split execution names it.
7. **Fingerprints of split and interleaved plans.** On macOS (Homebrew FFTW 3.3.11 without
   SIMD) the split plan of nd-fmm's box `[140, 9, 2]` and the interleaved plan of the unit
   `[140, 9]` have the same fingerprint, as do forward and backward and f32 and f64 plans
   (F4, note 7): `fftw_sprint_plan` names the algorithm, not the storage. Bits are compared
   only to round-off (design, Section 3 item 7), and a caller comparing fingerprints across
   ranks should compare plans of one problem and one storage form; F7's guide could say so.
8. **The sizes of the cells.** The fixed cells of the part axis outermost use nd-fmm's
   sizes 1, 2, 17, 97 and 140; the other arrangements and views use 17 and 2, with
   `[6, 5, 4]` over two and three logical axes in every arrangement. The sampled cells draw
   from F1's sizes up to 588, and cap two-dimensional transforms at 64 per axis, so that the
   O(n²) reference keeps the added time near 0.5 s in a debug build; F1's sizes above 588
   are covered on interleaved storage, and split against interleaved runs at 588.
9. **The test helpers.** `assert_gaps_unchanged` and `storage_bits` of `test_support` now
   take any `TestElement`, complex values or reals, so that the split tests reuse them; the
   F2 tests are unchanged.
