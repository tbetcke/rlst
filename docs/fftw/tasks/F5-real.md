# F5: Real-to-complex, complex-to-real and real-to-real transforms; `Scale`

| | |
| --- | --- |
| Task | F5 (design, Section 9) |
| Status | in progress: F5a done, F5b |
| Depends on | F1, F2, F3, F4 |
| Pull requests | three, into `fftw-0.11`, in this order: F5a (the storage of each side in `Layout`, the element types of the layout functions, r2c and c2r out of place and in place, `Plan::logical_size`, `Scale`), F5b (r2r with `R2rKind` and kinds per axis), F5c (split r2c and c2r) |
| Read first | design: Section 2.2 item 9, Section 3 items 1, 5, 7, 8 and 11, Sections 4.1–4.9 and 4.15, Section 5, Sections 7.1–7.6, Section 9 (F5 row), Section 10, in particular DC1, DC2, DC3, DC5, DC12, DC14–DC16, DC21, DC25–DC29, Appendix A (drafts 0.8 to 0.13), Appendix B; `docs/fftw/tasks/F1-core.md`, `F2-layouts.md`, `F3-split.md` and `F4-determinism.md`, including their "Open questions"; `docs/fftw/F0-findings.md`; `CLAUDE.md` |

## Goal

The remaining families of FFTW's serial interface (design, Section 1.1 goal 1): real-to-complex
(r2c) and complex-to-real (c2r) transforms out of place and in place, with the halved axis of
DC2; real-to-real (r2r) transforms with all eleven kinds and one kind per transform axis,
which removes the 0.10 bug of design Section 2.2 item 9; split r2c and c2r, which use split
storage on their complex side only (DC25); and the scale factors of design Section 4.9. Every
family gets layouts from raw dimensions and from RLST arrays and views, the three input
policies with FFTW's defaults (DC1), scratch and `*_on` planning (DC3), checked execution on
slices, arrays and views, and the tests of design Sections 7.2 to 7.6. nd-fmm uses none of
these families (design, Section 2.5); RLST's Chebyshev module will, after its port in F7.

## Scope

In scope:

- the storage of each side in `Layout` (complex, real, split), raw layouts of every family,
  and the halved dimension (work item 1);
- the element types of the array layout functions, and layouts of r2c and c2r from arrays,
  out of place and in place (work item 2);
- the families `R2c`, `C2r`, `R2r`, `SplitR2c` and `SplitC2r`, their FFTW calls in
  `FftwReal`, their planner methods with scratch and `*_on` planning, and the policy defaults
  of DC1, `PreserveInputUnsupported` included (work item 3);
- `R2rKind`, kinds per axis, their validation and logical sizes (work item 4);
- execution with the checks of design Section 4.8 for each family (work item 5);
- `Plan::logical_size`, `Scale` and `Plan::scale_factor` (work item 6);
- the tests of design Sections 7.1 to 7.4 for these families (work item 7), and rows in the
  concurrency tests and the determinism binary (work item 8);
- rustdoc, the module documentation of `rlst/src/fftw.rs`, `rlst/CHANGELOG.md`, the files
  table of `docs/fftw/README.md`, `CLAUDE.md` (checked), and this brief's "Open questions"
  (work item 9).

Out of scope, for later tasks:

- wisdom and `fftw_threads` (F6);
- the port of `chebychev.rs` to r2r plans, the removal of the 0.10 module (whose
  `R2RPlanKind` and `r2r_inplace.rs` stay unchanged), the user guide in
  `rlst/src/doc/optional_integrations.rs` and the migration notes (F7);
- benchmarks, in particular halving the contiguous axis against halving a strided one (DC2)
  and split against interleaved r2c (F8b, design Section 7.7);
- orthonormal DCT and DST variants (DC5);
- negative strides (design, Section 4.3);
- any change to the c2c and split c2c API of F1 to F3 other than those of open questions 1
  and 2. A reason to change more is an open question, not an edit.

Files touched:

| File | Change | PR |
| --- | --- | --- |
| `rlst/src/fftw/layout.rs` | the storage of each side, `Layout::with_storage`, the halved dimension in spans, distinct outputs and the side and pairing checks, the r2c and c2r array functions and their in-place variants, the split r2c and c2r array functions, tests | F5a, F5c |
| `rlst/src/fftw/element.rs` (new) | the sealed `FftElement` trait for `f32`, `f64`, `c32` and `c64` (open question 2), and `as_reals`, `as_reals_mut` (open question 3) | F5a |
| `rlst/src/fftw/real.rs` | `plan_guru64_dft_r2c`, `plan_guru64_dft_c2r`, `plan_guru64_r2r`, `plan_guru64_split_dft_r2c`, `plan_guru64_split_dft_c2r` and their `execute_*` functions in the sealed trait | F5a–F5c |
| `rlst/src/fftw/kind.rs` | `R2rKind`, its mapping to fftw-sys's constants, logical sizes | F5b |
| `rlst/src/fftw/scale.rs` (new) | `Scale` | F5a |
| `rlst/src/fftw/plan.rs` | the families, `Plan::logical_size`, `Plan::scale_factor`, `Plan::<_, R2r, _>::kinds`, rustdoc with `compile_fail` examples | F5a–F5c |
| `rlst/src/fftw/options.rs` | `PolicyFor` of the new families (DC1) | F5a–F5c |
| `rlst/src/fftw/planner.rs` | `r2c`, `c2r`, `r2r`, `split_r2c`, `split_c2r`, their `*_on` and in-place companions; the storage check of every planner method; `PreserveInputUnsupported`; scratch of two element types | F5a–F5c |
| `rlst/src/fftw/error.rs` | the hint of `PlanCreationFailed` (open question 5) | F5b |
| `rlst/src/fftw/buffer.rs`, `execute.rs` | execution of the new families; the overlap check of two element types; the `SAFETY:` comments | F5a–F5c |
| `rlst/src/fftw/execute/real_tests.rs`, `r2r_tests.rs` (new) | the tests of work item 7 | F5a–F5c |
| `rlst/src/fftw/test_support.rs` | the r2c, c2r and r2r references; real views | F5a, F5b |
| `rlst/src/fftw/concurrency_tests.rs`, `rlst/tests/fftw_determinism.rs` | the rows of work item 8 | F5a, F5b |
| `rlst/src/fftw.rs` | re-exports, module documentation | F5a–F5c |
| `rlst/CHANGELOG.md`, `docs/fftw/README.md`, `CLAUDE.md` | yes | F5a–F5c |

## 1. The storage of each side (`layout.rs`; F5a, F5c)

F3 records an optional gap per side (DC25). F5 generalises that record to the storage of
each side, as DC25 foresaw: complex interleaved, real, or split with a gap, as DC30 settles
(open question 1):

```rust
pub enum Storage { Complex, Real, Split { gap: isize } }
Layout::with_storage(dims, batch, (input, output): (Storage, Storage), placement)
Layout::storage(&self) -> (Storage, Storage)
```

`Layout::new` is `with_storage` with `(Complex, Complex)`, `Layout::new_split` with
`(Split, Split)`; `Layout::gaps()` keeps its meaning (`Some` only when both sides are split).

**The family of a layout** follows from the pair of storages:

| Input | Output | Family | Halved side |
| --- | --- | --- | --- |
| `Complex` | `Complex` | c2c | none |
| `Split` | `Split` | split c2c | none |
| `Real` | `Complex` | r2c | output |
| `Complex` | `Real` | c2r | input |
| `Real` | `Real` | r2r | none |
| `Real` | `Split` | split r2c | output |
| `Split` | `Real` | split c2r | input |

Any other pair (`Complex` with `Split`) is rejected by `with_storage` with `InvalidLayout`
naming `layout`.

**The halved dimension (DC2).** For r2c and c2r, and their split forms, the dimensions record
the logical real extent `n`, as FFTW's guru64 interface takes it. On the complex side the
*last transform dimension* has extent `n/2 + 1`, which is what DC2's "last entry of `axes`"
becomes for raw layouts. Strides count elements of each side: reals on a real or split side,
complex elements on a complex side.

**Rules**, checked in `with_storage`, each failure an `InvalidLayout`:

1. the rules of `Layout::new` for the dimensions, and of `Layout::new_split` for the gaps of
   split sides (positive; equal in place);
2. in place, both sides are the same buffer: for c2c, split c2c and r2r the strides of both
   sides are equal (as today); for r2c and c2r in place, the rule of open question 3; split
   r2c and c2r have no in-place layouts (open question 4);
3. the distinct-output rule of `Layout::new` on the output side, with the halved extent on a
   complex output and the part axis on a split output;
4. the spans, which use the halved extent on the complex side and include the gap on a split
   side, fit in `isize`.

`Layout::logical_size` stays the product of the dimensions' `n`, which is the logical size of
c2c, r2c and c2r; its rustdoc says that the logical size of an r2r transform depends on the
kinds and is [`Plan::logical_size`] (work item 6).

## 2. Layouts from arrays (`layout.rs`, `element.rs`; F5a, F5c)

**Element types (open question 2).** `along_axis`, `over_axes` and their in-place variants
take arrays whose items implement a new sealed trait `FftElement<Real = T>` with
`T: FftwReal` (implemented for `f32`, `f64`, `c32`, `c64`), so that the layout function knows
each side's storage from its element type, and the two sides have one precision. Real input
and complex output give an r2c layout, complex input and real output a c2r layout, two real
arrays an r2r layout, two complex arrays a c2c layout as today. The split functions of F3 keep
their real-array bound (DC29).

**Checks** of `over_axes` for r2c and c2r, in the order of F2, each returning `InvalidLayout`
naming `axes` or the axis:

1. to 3. as for c2c (`axes` not empty, below `N`, not repeated);
4. equal extents on every axis except the halved axis, the last entry of `axes` (DC2); on the
   halved axis the complex extent is `n/2 + 1` with `n` the real extent, and an error names
   the axis, the extent expected and the extent found;
5. and 6. as for c2c (extents at least 1, values fit in `isize`);

and those of `with_storage`. For r2r the checks are those of c2c (equal shapes).

**In place (open question 3).** `Layout::r2c_inplace(&complex, &axes, n_last)` and
`Layout::c2r_inplace(&complex, &axes, n_last)` take the complex array and the logical real
extent of the halved axis, which cannot be recovered from the complex shape (design, Section
4.3): `n_last / 2 + 1` must equal the complex extent along the last entry of `axes`. The
real side is the same memory: along the halved axis the reals of a complex element pair, and
on every other axis the real stride is twice the complex stride, as FFTW's padded in-place
layout requires. The halved axis must be contiguous (complex stride 1); otherwise
`InvalidLayout` names it.

**Split r2c and c2r (open question 4; F5c).** `Layout::split_r2c_along_axis(&input, &output,
part_out, axis)`, `split_r2c_over_axes(&input, &output, part_out, &axes)`,
`split_c2r_along_axis(&input, part_in, &output, axis)` and `split_c2r_over_axes(&input,
part_in, &output, &axes)`. Both arrays are real (DC29); the split side has a part axis of
extent 2, and `axes` count its logical axes (DC26); the halved axis is the last entry of
`axes`, with extent `n/2 + 1` on the split side.

Expected layouts, which the tests assert with `with_storage` (strides in elements of each
side; `(n, is, os)`; F5a and F5c):

| Arrays (shape, strides) | Call | Transform dims | Batch dims | Storage |
| --- | --- | --- | --- | --- |
| real `[8, 3]` (1, 8); complex `[5, 3]` (1, 5) | `over_axes(…, &[0])` | `(8, 1, 1)` | `(3, 8, 5)` | real, complex |
| complex `[5, 3]` (1, 5); real `[8, 3]` (1, 8) | `along_axis(…, 0)` | `(8, 1, 1)` | `(3, 5, 8)` | complex, real |
| real `[6, 9]` (1, 6); complex `[6, 5]` (1, 6) | `along_axis(…, 1)` | `(9, 6, 6)` | `(6, 1, 1)` | real, complex |
| real `[4, 6]` (1, 4); complex `[4, 4]` (1, 4) | `over_axes(…, &[0, 1])` | `(4, 1, 1)`, `(6, 4, 4)` | none | real, complex |
| complex `[5, 3]` (1, 5), `n_last` 8 | `r2c_inplace(…, &[0], 8)` | `(8, 1, 1)` | `(3, 10, 5)` | real, complex |
| real `[8, 3]`; split `[5, 3, 2]` (1, 5, 15), part 2 | `split_r2c_along_axis(…, 2, 0)` | `(8, 1, 1)` | `(3, 8, 5)` | real, split 15 |
| real `[8, 3]` and real `[8, 3]` | `along_axis(…, 0)` | `(8, 1, 1)` | `(3, 8, 8)` | real, real |

**Execution's side and pairing checks** (DC15, DC27) compare an array's extent along a
transform dimension with `n`, or with `n/2 + 1` for the last dimension on a complex or split
side of r2c and c2r; when both buffers are arrays of the same rank, their shapes must be equal
except along the axis matched by the halved dimension.

## 3. Families and planner (`plan.rs`, `real.rs`, `options.rs`, `planner.rs`; F5a–F5c)

- **Families**: `R2c`, `C2r`, `R2r`, `SplitR2c`, `SplitC2r`, markers like `C2c` (design,
  Section 4.1). r2c and split r2c are always forward and c2r and split c2r always backward
  (FFTW's fixed signs), so their plans have no `direction()`.
- **Policies (DC1)**, by `PolicyFor`:

  | Request | `R2c`, `R2r`, `SplitR2c` | `C2r`, `SplitC2r` |
  | --- | --- | --- |
  | `DefaultRequest` | `Preserving` | `Destroying` |
  | `PreserveRequest` | `Preserving` | `Preserving`, one transform dimension only |
  | `DestroyRequest` | `Destroying` | `Destroying` |

  `preserve_input()` with c2r or split c2r over more than one transform dimension returns
  `PreserveInputUnsupported { rank }` before FFTW is called (design, Section 3 item 5).
- **Planner methods**, as for c2c (DC3, DC12): `r2c`, `r2c_on`, `c2r`, `c2r_on`, `r2r`,
  `r2r_on`, `split_r2c`, `split_r2c_on`, `split_c2r`, `split_c2r_on` on `Planner<T, S>` with
  `S: PolicyFor<F>`; `r2c_inplace`, `r2c_inplace_on`, `c2r_inplace`, `c2r_inplace_on`,
  `r2r_inplace`, `r2r_inplace_on` on `Planner<T, DefaultRequest>` only. r2r methods take
  `&[R2rKind]` after the layout (Appendix B). Buffers are `FftBufferMut<T>` on a real or split
  side and `FftBufferMut<Complex<T>>` on a complex side; in place, the complex buffer (open
  question 3).
- **The storage check (DC28, generalised).** Every planner method, those of F1 to F3
  included, rejects a layout whose storages are not those of its family, before FFTW is
  called, with `InvalidLayout`: naming `part axis` when one storage is split and the other is
  not split where the family needs it (as DC28), and naming `layout` otherwise; the reason
  names the family and the storages found.
- **Scratch (DC3)**: per side, the span in elements of that side, at the requested class,
  `max_scratch` counting bytes of both element types; a class must be a multiple of the
  alignment of that side's element.
- **FFTW calls.** The sealed `FftwReal` gains the guru64 r2c, c2r, r2r, split r2c and split
  c2r planners and their execute functions, with the contracts of the existing ones; planning
  runs inside `with_fftw` with the settings guard.
- **HC2R** (design, Section 4.4): a `Preserving` r2r plan with an HC2R kind passes
  `FFTW_PRESERVE_INPUT`, as every preserving plan does; if FFTW returns NULL, the
  `PlanCreationFailed` carries a hint to `may_destroy_input()` (open question 5).

## 4. Kinds (`kind.rs`; F5b)

- `R2rKind`: `R2hc`, `Hc2r`, `Dht`, `Redft00`, `Redft01`, `Redft10`, `Redft11`, `Rodft00`,
  `Rodft01`, `Rodft10`, `Rodft11` (design, Section 4.2), RLST's own enum mapped to fftw-sys's
  constants internally (DF4), with `Display` and `R2rKind::logical_size(n)` of design Section
  3 item 11.
- **Kinds per axis.** `kinds[i]` belongs to `layout.dims()[i]`, so for a layout from arrays to
  `axes[i]` (design, Section 4.3). The planner checks, before FFTW is called, each returning
  `InvalidKind` with the reason: `kinds.len()` equals the number of transform dimensions;
  `Redft00` has `n` of at least 2. The plan records its kinds (`Plan::<_, R2r, _>::kinds()`).
- **The regression of design Section 2.2 item 9**: different kinds on different axes of
  column-major and row-major arrays apply each kind to its own axis.

## 5. Execution (`buffer.rs`, `execute.rs`; F5a–F5c)

- `execute` for every family and policy, with the signatures of c2c: `execute(&input, &mut
  output)` for `Preserving`, `execute(&mut input, &mut output)` for `Destroying`,
  `execute(&mut data)` for `InPlace`; each side typed by its storage.
- **Checks**, in the order of design Section 4.8: buffers against the layout, per side and
  paired (work item 2); alignment classes of the pointers FFTW receives (both parts of a split
  side follow from the real part and the gap, as in F3); overlap, conservative, on the byte
  spans of two element types. A failed check returns its error and writes nothing.
- **`SAFETY:`** comments name FFTW's four new-array conditions for each family: sizes and
  strides with the halved extent on the complex side, placement, the gap on a split side
  (condition 3), and alignment. In place, the comment states why the real side lies inside the
  checked complex span.

## 6. Logical sizes and scaling (`plan.rs`, `scale.rs`; F5a, F5b)

- `Plan::logical_size()` on every plan (open question 6): the product over the transform
  dimensions of `n` for c2c, split c2c, r2c, c2r and their split forms; for r2r, of each
  kind's logical size (design, Section 3 item 11).
- `Scale { None, Inverse, InverseSqrt }` and `Plan::scale_factor(Scale) -> T`: 1, 1/N and
  1/√N with N the logical size (design, Section 4.9). The rustdoc states that a uniform factor
  does not make the DCT and DST kinds orthonormal (DC5).

## 7. Tests (`execute/real_tests.rs`, `r2r_tests.rs`; design Sections 7.1 to 7.4)

Every plan here uses `Rigor::Estimate` and assumes no SIMD. Inputs come from F1's seeded
generator; messages name the seed, the family, the layout or arrays, the axes, the kinds, the
policy and the precision.

- **References (design Section 7.1)**, in f64 with compensated summation and reduced twiddle
  indices, evaluated by logical multi-index, never on the layout under test:
  - r2c: F2's logical c2c reference of the real input, cut to `n/2 + 1` along the halved axis;
  - c2r: inputs are made Hermitian by construction, as the r2c reference of seeded reals
    rounded to the precision, and the reference is the real part of the logical c2c backward
    reference of their Hermitian extension;
  - r2r: O(n²) definitions of the eleven kinds, applied along each axis with its kind.
  Tolerances are c · ε · (1 + log₂ N) in the relative L2 norm, c fixed per family.
- **Fixed cells**, both precisions, the three policies where the family has them:
  - r2c and c2r: sizes 1, 2, 3, 17, 97, 140, 588 and a power of two along the halved axis;
    `[n]`, `[n, 3]` along each axis, `[6, 5, 4]` over two and three axes in two orders;
    column-major, row-major, a subview and a strided view of F2's matrix with sentinels;
    in place with `r2c_inplace` and `c2r_inplace`;
  - r2r: every kind on `[n]` with n in 1 (an expected `InvalidKind` for `Redft00`), 2, 3, 17
    and 97; mixed kinds on `[6, 5, 4]` over all axes, column-major and row-major; in place;
  - split r2c and c2r: the split side with its part axis outermost and innermost, against the
    interleaved plan of the same values to round-off (design Section 7.3).
- **Sampled cells**: the rest of the product, seeded, the seed printed on failure.
- **Identities (design Section 7.3)**: the round trip (c2r after r2c, and inverse kind pairs
  such as `Redft10` and `Redft01`) is N times the input, also at 2¹⁶ for r2c and c2r, beyond
  the reference's range; r2c against c2c on real data; linearity and Parseval for r2c; a
  multidimensional r2r against successive one-dimensional ones; a batched plan against a loop
  of single plans, bit for bit where they are the same computation.
- **Defaults and policies (design Section 7.4)**: a default c2r plan is `Destroying`;
  `preserve_input()` gives a `Preserving` one-axis c2r that leaves its input bit for bit, and
  `PreserveInputUnsupported` over two axes; an r2r plan with an HC2R kind is `Preserving` and
  leaves its input unchanged, or returns `PlanCreationFailed` with the hint.
- **Negative tests**, each returning its error with an output whose parent holds the
  sentinel: a wrong extent on the halved axis (`InvalidLayout` naming that axis), a wrong
  `n_last`, an in-place halved axis that is not contiguous, a layout of another family given
  to each planner method, a wrong number of kinds, another alignment class, overlap, the
  scratch cap with the `*_on` remedy.
- **`compile_fail` doctests (design Section 7.4)**: a default c2r plan executed with
  `&input`; an r2c plan executed with complex input; an in-place r2c plan with two arrays.
  Each beside the example that compiles.

Budget: these tests add at most about 2 s in a debug build over the three pull requests;
measure and report each.

## 8. Concurrency and determinism (`concurrency_tests.rs`, `tests/fftw_determinism.rs`)

- **Concurrency**: "one plan, many threads" gains an r2c row (`[140, 8]` real, `[71, 8]`
  complex, f64 and f32, the three policies) in F5a and an r2r row (`Redft10` along axis 0 of
  `[140, 8]`) in F5b, bit for bit against sequential execution.
- **`estimate`** gains r2c `[140, 9]` along axis 0, c2r of the same shape (`Destroying`),
  an r2r `Redft10` along axis 0 of `[140, 9]`, and an r2r with mixed kinds over both axes of
  `[17, 9]`. Their lines follow the existing ones, so `fresh processes agree` and `planning
  history` cover them unchanged; `other_plans` gains plans of these families of other sizes.
- The binary's time stays within about 10 s in a debug build; measure and report.

## 9. Documentation

- Rustdoc with an example for every new public item. `Layout::over_axes` shows an r2c layout
  and states the halved axis (DC2); `Planner::c2r` states once that c2r destroys its input
  unless `preserve_input()` is given (design, Section 4.4); `R2rKind` lists the logical sizes;
  `Scale` states DC5.
- The module documentation of `rlst/src/fftw.rs` lists the new families.
- `rlst/CHANGELOG.md`: the entry "New FFTW interface (in progress)" gains a paragraph per pull
  request.
- `docs/fftw/README.md`: the files table lists this brief; the status table is updated when
  the last pull request merges. `CLAUDE.md` is checked and changes only if a command or
  layout entry does.

## Pull requests and CI

- Three pull requests, each branched from `fftw-0.11` after the previous one merged, titled
  for example `feat(fftw): F5a r2c and c2r`, `feat(fftw): F5b r2r with kinds per axis` and
  `feat(fftw): F5c split r2c and c2r`, each with the acceptance items it closes ticked.
- The open questions below are settled before the task starts; a question that arises during
  the task is recorded below and raised in the pull request.
- CI runs for pull requests into `fftw-0.11`. Report the determinism binary's summary from
  both jobs in the pull requests that change it, as F3 and F4 did.

## Acceptance

- [ ] `Layout` records the storage of each side as open question 1 settles; raw layouts of
      every family are built and validated as work item 1 says, and the layouts of work item
      2's table are asserted.
- [ ] Layouts of r2c, c2r and r2r are built from arrays and views, out of place and in place,
      with the halved axis of DC2 and the errors of work item 2; split r2c and c2r layouts
      from real arrays with a part axis on one side.
- [ ] Plans of `R2c`, `C2r`, `R2r`, `SplitR2c` and `SplitC2r` are made on scratch and on the
      caller's buffers with the policies of DC1, in both precisions, and execute on slices,
      arrays and views with the checks of work item 5; `PreserveInputUnsupported` is
      returned as design Section 4.4 says.
- [ ] Every planner method, those of F1 to F3 included, rejects a layout of another family
      before FFTW is called.
- [ ] `R2rKind` covers the eleven kinds, kinds apply per axis, and the regression of design
      Section 2.2 item 9 is tested on column-major and row-major arrays.
- [ ] `Plan::logical_size`, `Scale` and `Plan::scale_factor` exist with the values of design
      Section 4.9.
- [ ] The tests of work item 7 pass in the default FFTW test run, in CI on Linux and macOS;
      their added time in a debug build is reported for each pull request and is at most
      about 2 s in total.
- [ ] The rows of work item 8 pass; the determinism binary's summary from both jobs is in the
      pull requests that change it, and its time in a debug build is at most about 10 s.
- [ ] Every `unsafe` block has a `SAFETY:` comment naming the new-array conditions it relies
      on; every FFTW call other than execution runs inside `with_fftw`; no dependency was
      added.
- [ ] Every new public item has rustdoc with an example; the `compile_fail` doctests of work
      item 7 exist beside examples that compile.
- [ ] The six Clippy commands of `CLAUDE.md` pass (or are reported unavailable), as do
      `cargo fmt -- --check`, `cargo test`, `cargo test --features "fftw,fftw_system,strict"`
      and `cargo clippy --tests --features "fftw,fftw_system,strict" -- -D warnings`.
- [ ] `rlst/CHANGELOG.md` and `docs/fftw/README.md` are updated; `CLAUDE.md` is checked.

## Pitfalls

- **Units.** Strides, spans and scratch count elements of each side: reals on a real or split
  side, complex elements on a complex side. The overlap check compares bytes of two element
  types.
- **The halved extent is on the complex side only.** The layout's `n` of the last transform
  dimension is the real extent; every computation on the complex side (span, distinct
  outputs, side check, pairing) uses `n/2 + 1`. An off-by-one here is memory-unsafe, since the
  span check rests on it.
- **The halved axis is the last entry of `axes`** (DC2), whatever the memory order, and for raw
  layouts the last transform dimension, as FFTW's guru interface halves its last dimension.
- **c2r input.** FFTW's c2r reads only the Hermitian half and ignores the imaginary parts that
  Hermitian symmetry forces to zero; tests build inputs that are Hermitian by construction, and
  compare only to round-off.
- **Destroying defaults.** c2r and split c2r are `Destroying` by default; their `execute` takes
  `&mut input`, which makes the default visible at the call site (design, Section 4.4).
- **Kinds belong to transform dimensions**, in the order of `axes`, never to FFTW's reversed
  order; the 0.10 code reversed the shape but not the kinds (design, Section 2.2 item 9).
- **`Redft00` with n = 1** is undefined; the planner returns `InvalidKind` before FFTW, which
  would return NULL (design, Section 3 item 11).
- **SIMD.** Plans of these families can choose different codelets for different batch counts
  and layouts; compare different plans only to round-off, and depend on SIMD nowhere.
- **Wisdom, determinism, `strict`, wildcard imports, debug builds**: as in the earlier briefs.
  Unit tests plan with `Estimate` only.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design questions
in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.14, DC30–DC35, Appendix A): all six with the option marked *proposed*. The
work items follow these answers.

1. **How `Layout` records the storage of each side.** DC14 says F5 "adds to `Layout` what its
   planner then needs", and DC25 records a gap per side so that split r2c and c2r fit "without
   a breaking change". A layout must now tell c2c, r2c, c2r and r2r apart, since the halved
   extent changes its spans and checks. Options: (a) *proposed*: a public enum
   `Storage { Complex, Real, Split { gap } }`, one per side, with
   `Layout::with_storage(dims, batch, (input, output), placement)` and `Layout::storage()`;
   `new` and `new_split` become its shorthands and `gaps()` keeps its meaning; the family
   follows from the pair (work item 1's table), and every planner method checks it, extending
   DC28 (`part axis` for split mismatches, `layout` for the others); (b) a private family tag
   and a halved side beside F3's gaps, with raw constructors per family (`new_r2c`, `new_c2r`,
   `new_r2r`, `new_split_r2c`, `new_split_c2r`); (c) a layout type per family, which makes
   mismatches compile-time errors at the cost of a type per family in every signature.
   *Decided: (a), DC30.*
2. **The element types of the array layout functions.** Design Section 4.3 makes
   `along_axis` and `over_axes` generic over any element type and has them build r2c and c2r
   layouts from a real and a complex array, but a function generic over any element type cannot
   tell a real side from a complex one. Options: (a) *proposed*: a sealed trait
   `FftElement<Real = T>` for `f32`, `f64`, `c32` and `c64`, required of both arrays' items,
   from which the function reads each side's storage, and which makes arrays of two precisions
   a compile-time error; the bound is added to F2's functions, whose c2c behaviour does not
   change; (b) separate functions per family (`r2c_over_axes`, `c2r_over_axes`,
   `r2r_over_axes`), `over_axes` staying complex-only (and design Section 4.3 corrected).
   *Decided: (a), DC31.*
3. **In-place r2c and c2r: the buffer and the rule.** Design Section 4.3 has in-place r2c and
   c2r use a complex array with `Layout::r2c_inplace(&complex, &axes, n_last)`, the real side
   padded to `2(n/2 + 1)` reals. RLST has no safe way to view a complex buffer as reals, which
   the caller needs to write the r2c input and read the c2r output, and the design does not
   state which strides FFTW accepts in place. Options: (a) *proposed*: the complex buffer, as
   the design says, with `Layout::r2c_inplace` and `Layout::c2r_inplace`; the halved axis must
   be contiguous (complex stride 1) and the real strides of the other axes twice the complex
   strides, and other strides are rejected with `InvalidLayout`; and two safe functions
   `fftw::as_reals(&[Complex<T>]) -> &[T]` and `as_reals_mut`, so that callers view the real
   side through slices and `SliceArray`s (`Complex<T>` is `#[repr(C)]` with two `T`s, a
   layout RLST already relies on, Section 4.11); the task checks with FFTW that layouts the
   rule accepts plan, and records any stride FFTW also accepts as an open question rather than
   allowing it; (b) as (a) without the helpers; (c) the padded real array as the buffer, with
   `Layout::r2c_inplace(&real_padded, &axes, n_last)`, which departs from the design.
   *Decided: (a), DC32.*
4. **Split r2c and c2r: layout functions and placement.** Design Section 4.5 names only the
   split c2c functions, and the DC1 table gives every family in-place plans. Options:
   (a) *proposed*: `split_r2c_along_axis`, `split_r2c_over_axes`, `split_c2r_along_axis` and
   `split_c2r_over_axes`, with one part-axis argument for the split side; no in-place split
   r2c or c2r in 0.11 (FFTW's in-place split forms would pass the real buffer as one plane of
   the split side, which no user needs and which the checks of DC27 would have to cover), with
   Section 4.4's table and Appendix A corrected; (b) `split_over_axes` taking an
   `Option<usize>` part axis per side; (c) as (a), with in-place variants.
   *Decided: (a), DC33; design Sections 4.4 and 4.5 are corrected.*
5. **The hint of `PlanCreationFailed` for HC2R.** Design Section 4.4 says the error carries a
   hint to `may_destroy_input()`; `PlanCreationFailed { rigor, flags, layout }` has no place
   for it. Options: (a) *proposed*: a field `hint: Option<&'static str>`, shown in the message
   when present (the variant's fields change before 0.11.0, which no release has); (b) a new
   variant for preserving plans that FFTW cannot make; (c) no hint, the rustdoc of `r2r`
   stating the remedy. *Decided: (a), DC34.*
6. **`logical_size` and `scale_factor`.** Design Section 4.9 names `Plan::logical_size()` and
   `Plan::scale_factor(s)` without types or the families they cover. Options: (a) *proposed*:
   both on every plan, `logical_size() -> usize` and `scale_factor(Scale) -> T`, with
   `Layout::logical_size` kept as the product of the extents and documented as not being the
   r2r logical size; (b) `scale_factor` returning `f64`; (c) on r2c, c2r and r2r plans only.
   *Decided: (a), DC35.*

Raised during F5a. None needs a design change; they are recorded for review and for F5b, F5c
and F7.

7. **In-place strides that FFTW also accepts (open question 3).** FFTW 3.3.11 (Homebrew,
   without SIMD) plans with `FFTW_ESTIMATE`, in place, every r2c and c2r guru64 layout
   tried outside the rule of DC32: a batch whose real stride is not twice its complex stride
   (12 against 5), the halved dimension with real and complex strides 2 and 2, 2 and 1, 1
   and 2, or 6 and 3, and a batch of stride 1 under a halved dimension of stride 3. Several
   of these put the real side outside the complex span (real stride 2 with complex stride 1
   along a halved dimension of 8 spans 15 reals of a complex span of 10), so FFTW's planner
   checks no memory, and its acceptance is no evidence that a layout is valid. The rule of
   DC32 stays; a wider rule would be a design change, and would have to keep the real side
   inside the complex span and both sides distinct.
8. **In place, the distinct-element rule applies to both sides.** Rule 3 of work item 1
   checks the output side. In place the input is overwritten too: for c2c, split c2c and
   r2r the two sides have equal strides, but a c2r layout can have distinct real outputs
   and overlapping complex inputs (`(8, 1, 1)` with the batch `(3, 4, 8)`: the last complex
   element of each transform is the first of the next). `Layout::with_storage` applies the
   rule to the input side as well in place; for r2c the rule on the complex output implies
   it for the real input. *Accepted by the maintainer as implemented: design draft 0.15,
   DC36.*
9. **`FftElement` and generic code.** DC31 names `f32`, `f64`, `c32` and `c64`. The trait is
   implemented for `f32`, `f64` and `Complex<T>` for every `T: FftwReal`, which are `c32`
   and `c64`, and `FftElement<Real = Self>` is a supertrait of the sealed `FftwReal`, so code
   generic over the precision, such as F2's tests, builds layouts from arrays of `T` and of
   `Complex<T>`. The bound is on the arrays' items (`AIn::Item: FftElement`), so the layout
   functions gained no type parameter.
10. **`PreserveInputUnsupported` counts every transform dimension**, those of extent 1
    included, as work item 3 says: a preserving c2r over both axes of an `[n, 1]` array is
    rejected, although FFTW might plan it as one-dimensional; it can be planned along axis 0.
    *Accepted by the maintainer as implemented: design draft 0.15, DC37.*
11. **`Layout`'s `Debug` output** shows `storage` where it showed `gaps`; one F3 test that
    asserted the debug text was updated.

Raised during F5b. None needs a design change.

12. **Preserving HC2R plans exist on Homebrew's FFTW 3.3.11.** FFTW planned every preserving
    real-to-real problem with an HC2R kind that the tests try, over one axis of `[8, 6]`,
    over both axes, and with a DCT-II on the other axis, and left the input unchanged bit for
    bit. The `hint` of `PlanCreationFailed` is set for every preserving plan with an HC2R
    kind that FFTW refuses, which this FFTW never did; its message is tested directly, and
    the HC2R test accepts either outcome, as work item 7 says. CI on Linux shows whether
    Ubuntu's FFTW 3.3.10 refuses any of them.
13. **The direction of a real-to-real plan.** `Plan` records a direction for every family;
    a real-to-real plan has none, and records `Forward`, which only its `Debug` output shows,
    beside its kinds. Its kinds say what it computes; `Plan::<_, R2r, _>` has no
    `direction()`.
