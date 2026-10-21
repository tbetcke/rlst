# F2: Layouts from RLST arrays and views

| | |
| --- | --- |
| Task | F2 (design, Section 9) |
| Status | not started |
| Depends on | F1 |
| Pull requests | two, into `fftw-0.11`: F2a (strided storage of arrays and views, `Array::storage()` and `storage_mut()`), F2b (layouts from arrays, execution on arrays, tests) |
| Read first | design: Section 2.5, Section 3 (all items), Sections 4.1–4.8, Section 5, Sections 7.1–7.4 and 7.7, Section 9 (F2 row, Transition), Section 10, in particular DC11–DC19, Appendix A (drafts 0.8 and 0.9); `docs/fftw/tasks/F1-core.md`, including its "Open questions"; `docs/fftw/F0-findings.md`; `CLAUDE.md` |

## Goal

Plans run on RLST arrays and views, not only on slices: `Layout::along_axis`,
`Layout::over_axes` and their in-place variants build layouts from arrays of any memory order,
and `FftBuffer` and `FftBufferMut` let a plan execute on them, with the checks of design
Section 4.8. Arrays and views reach FFTW through `StridedRawAccess` and `StridedRawAccessMut`
(DC11), which F1a declared and F2 implements for every array and view type with strides,
subviews and nested views included. Everything is tested against an independent reference
that works on logical indices, with sentinel values in the elements between a view's own.
This is design goal 3 for c2c; split storage (F3), the other families (F5) and the
separate-process tests (F4) build on it.

## Scope

In scope:

- F2a: the implementations of `StridedRawAccess` and `StridedRawAccessMut` in
  `rlst/src/dense/base_array.rs`, `rlst/src/dense/strided_base_array.rs`,
  `rlst/src/dense/array/reference.rs`, `slice.rs`, `multislice.rs`, `subview.rs`,
  `empty_axis.rs`, `operators/with_type_hint.rs` and `operators/coerce.rs`, with their tests;
  the rustdoc of the two traits in `rlst/src/traits/accessors.rs`; the inherent
  `Array::storage()` and `Array::storage_mut()` in `rlst/src/dense/array.rs` (DC18), and a
  paragraph on them in `rlst/src/doc/dense_linear_algebra.rs`;
- F2b: the `todo!()` bodies of F1 in `rlst/src/fftw/layout.rs` and `rlst/src/fftw/buffer.rs`;
  the execution checks and their `SAFETY:` comment in `rlst/src/fftw/execute.rs`; test
  helpers in `rlst/src/fftw/test_support.rs`; rustdoc in `rlst/src/fftw/align.rs`,
  `planner.rs` and the module documentation of `rlst/src/fftw.rs`;
- both: `rlst/CHANGELOG.md`, the status table of `docs/fftw/README.md`, and this brief's
  "Open questions".

Out of scope, for later tasks:

- split layouts and the part axis (F3); r2c and c2r layouts, the halved axis and
  `Layout::r2c_inplace` (F5); r2r kinds (F5);
- tests in fresh processes, concurrency, and anything that depends on the wisdom present
  (F4);
- the module guide in `rlst/src/doc/optional_integrations.rs` and the removal of the 0.10
  module (F7). The 0.10 module stays as it is (design, Section 9, Transition);
- any change to `RawAccess` or `RawAccessMut`, or to the meaning of `Array::data()`: existing
  callers (for example `rlst/src/interpolation.rs`, `rlst/src/operator/algorithms/gmres.rs`,
  `rlst/src/operator/frame/array.rs`) rely on their slice holding exactly the array's
  elements;
- any change to the F1 signatures of the layout functions and the buffer impls, which the
  F1a review approved. A reason to change them is an open question, not an edit.

Files touched:

| File | F2a | F2b |
| --- | --- | --- |
| `rlst/src/traits/accessors.rs` | rustdoc and an example for the two traits | |
| `rlst/src/dense/base_array.rs`, `strided_base_array.rs` | owned storage at offset 0 | |
| `rlst/src/dense/array/reference.rs` | `r()`, `r_mut()` forward | |
| `rlst/src/dense/array/slice.rs`, `multislice.rs`, `subview.rs` | parent's storage, own offset added | |
| `rlst/src/dense/array/empty_axis.rs`, `operators/with_type_hint.rs`, `operators/coerce.rs` | forward | |
| `rlst/src/dense/array.rs` | inherent `storage()`, `storage_mut()` | |
| `rlst/src/doc/dense_linear_algebra.rs` | a paragraph beside `data()` | |
| `rlst/src/fftw/layout.rs` | | `along_axis`, `over_axes`, in-place variants, tests |
| `rlst/src/fftw/buffer.rs` | | `FftBuffer`, `FftBufferMut` for arrays, tests |
| `rlst/src/fftw/execute.rs` | | the array checks, the `SAFETY:` comment, the correctness tests |
| `rlst/src/fftw/test_support.rs` | | the logical reference, sentinel parents |
| `rlst/src/fftw/align.rs`, `planner.rs`, `rlst/src/fftw.rs` | | rustdoc |
| `rlst/CHANGELOG.md`, `docs/fftw/README.md` | yes | yes |

## F2a: strided storage of arrays and views

### 1. Implementations

`StridedRawAccess::storage` returns the backing slice and the offset of the view's first
element; with the strides of the type's `Stride` impl, element `i` is
`storage[offset + Σ i[k] · stride[k]]` (design Section 4.3, DC11). Implement both traits
explicitly for each type below; a blanket impl over `RawAccess` cannot coexist with these
impls and would give subviews nothing. Every type that has `Stride` is covered:

| Type | `storage()` | Why | `StridedRawAccessMut` |
| --- | --- | --- | --- |
| `BaseArray<Data, N>` (`DynArray`, `AlignedDynArray`, `StaticArray`, `SliceArray`, `SliceArrayMut`) | `(data.data()?, 0)`, for `Data: RawAccessDataContainer` | the container holds exactly the elements, column-major from 0 | for `Data: MutableRawAccessDataContainer`, from `data_mut()` |
| `StridedBaseArray<Data, N>` (`StridedDynArray`, row-major arrays, `StridedSliceArray`, `StridedSliceArrayMut`) | `(data.data()?, 0)` | the container may hold gaps and be longer than the span; strides are non-negative, so the first element is at 0 | as above |
| `ArrayRef` (`r()`) | the parent's, unchanged | a borrow of the whole array | none: it holds a shared borrow |
| `ArrayRefMut` (`r_mut()`) | the parent's, unchanged | as above | the parent's `storage_mut()` |
| `ArraySlice<A, ADIM, N>` (`slice`, `row`, `col`) | the parent's `(s, o)` as `(s, o + index · stride[axis])` | the fixed index moves the first element; the other strides are the parent's | from the parent's `storage_mut()`, same offset |
| `Multislice<A, ADIM, SDIM, N>` | `(s, o + Σ offset[k] · stride[k])` over the parent's axes (`offset` is zero on axes not fixed) | as above, for several fixed axes | as above |
| `ArraySubView<A, N>` (`into_subview`) | `(s, o + Σ offset[k] · stride[k])` | the view's origin; strides are the parent's | as above |
| `WithEvalType` (`with_container_type`), `CoerceArray` (`coerce_dim`), `ArrayAppendAxis` (`insert_empty_axis`) | the wrapped array's, unchanged | same elements and strides; the inserted axis has extent 1, so its stride addresses no other element | the wrapped array's |

Bounds follow the neighbouring `RawAccess` and `Stride` impls of each type, with
`StridedRawAccess` in place of `RawAccess`: a view needs its parent's storage and, for the
offset, its parent's strides (`ArraySlice`, `Multislice`, `ArraySubView` need
`ArrayImpl: StridedRawAccess + Stride<ADIM>`), and no `RawAccess`. Views reach their parent
through the `Array` they hold (`self.arr.imp().storage()`, `self.arr.stride()`), so nested and
mixed views compose: a slice of a subview of a row-major array adds the offsets of each level.

Excluded, as they have no `Stride` or no storage: lazy expressions (the operators in
`rlst/src/dense/array/operators/` other than `with_type_hint` and `coerce`, including `cast`);
`transpose()`, `permute_axes()` and `reverse_axis()` (design Section 4.3); `ArrayFlatView`
(`into_flat`), which has `RawAccess` but no `Stride`; `ArraySharedRef` and
`ArrayAtomicSharedRef`, whose storage sits behind a `RefCell` or `RwLock`; `ZeroDimArray`.

`Array<A, N>` itself implements none of the accessor traits (only `BaseItem`); it has inherent
`shape()`, `stride()` and `data()` instead. The F1 signatures bound the implementation type
`A` and take `&Array<A, N>`, so F2 needs no forwarding impls on `Array`. F2a adds inherent
`Array::storage()` and `Array::storage_mut()` beside `data()` and `data_mut()` (DC18), so that
users reach the storage without `arr.imp()`. They follow the pattern of `data()`: an `impl`
block on `Array<ArrayImpl, NDIM>` with `ArrayImpl: StridedRawAccess` (respectively
`StridedRawAccessMut`), forwarding to the implementation, with rustdoc and an example each
(a subview's storage and offset, and a write through `storage_mut()` seen through the view).

### 2. Rustdoc of the traits

- Replace "Implementations for RLST's arrays and views follow with the FFTW rework (task F2)"
  by the list of implementing and excluded types, in short.
- An example on each trait: the storage and offset of a subview of a column-major array,
  and an element read through them; for the mutable trait, a write through `storage_mut()`
  seen through the view.
- The sentence on views without elements (DC19): their offset may lie past the end of the
  slice and must not be used to index it.

### 3. Tests of F2a

In the module of each implementation, with the nested and mixed views in `subview.rs`. They
need no FFTW and run in the default `cargo test`.

- **Element identity.** For every multi-index `i` of a view,
  `std::ptr::eq(&s[offset + Σ i[k] · stride[k]], view.get(i).unwrap())`, for: `DynArray`,
  `AlignedDynArray` at `FFTW_ALIGNED`, `StaticArray`, `SliceArray`, `SliceArrayMut`,
  `StridedDynArray::row_major` and one with custom strides, a `StridedSliceArray` over a longer
  slice; `r()` and `r_mut()`; `slice` on each axis of a 3-D array, `row` and `col`; a
  multislice fixing two of four axes; `into_subview` at a non-zero origin; a subview of a
  subview, a slice of a subview, a subview of a slice and a subview of a row-major array;
  `with_container_type`, `coerce_dim`, and `insert_empty_axis` at the front and at the back.
- **Mutable storage.** `storage_mut()` gives the pointer and offset of `storage()`, and a
  write at element `i`'s offset is seen at `view[i]`, for `r_mut()` views of each kind.
- **The inherent methods** (DC18): `Array::storage()` and `storage_mut()` return what the
  implementation's `storage()` and `storage_mut()` return, for an owned array, a subview and
  an `r_mut()` view.
- **The consistency rule** (DC11): wherever `RawAccess::data()` returns `Some(d)`,
  `storage()` returns `Some((s, offset))` with `s[offset..].as_ptr() == d.as_ptr()` and
  `offset + d.len() <= s.len()`. Cover owned arrays (offset 0), the contiguous slices and
  multislices of column-major and row-major parents (the cases in which they return `data()`),
  the forwarding wrappers, and an empty owned array (`[0, 3]`: `(&[], 0)`).
- All existing tests and doctests pass unchanged; in particular no test of `data()` changes.

## F2b: layouts from arrays, and execution on them

### 4. Layout functions (`layout.rs`)

`along_axis(input, output, axis)` is `over_axes(input, output, &[axis])` and gives equal
layouts. `over_axes` checks, in this order, returning `InvalidLayout` with the dimension as
text (`axes`, or `axis k` for an RLST axis) and the reason:

1. `axes` is not empty;
2. every entry is below the rank `N`;
3. no entry is repeated;
4. the input and output extents are equal on every axis (complex-to-complex; DC14);
5. every extent is at least 1: an array with an extent 0 has nothing to transform, and
   `Layout::new` would reject it naming a dimension index rather than the axis;
6. every extent and stride converts to `isize` (`isize::try_from`, never `as`).

Then the transform dimensions, in the order of `axes`, are
`Dim { n: shape[k], is: input_stride[k], os: output_stride[k] }`; the batch dimensions are
the other axes, in any order, as the same `Dim`s; and the result is
`Layout::new(&dims, &batch, Placement::OutOfPlace)`, whose validation and normalisation
(F1, design Section 4.3) apply unchanged, the distinct-output rule included (DC16). FFTW receives the dimensions in the order of `axes`, with their own strides, so
no shape is reversed for column-major arrays (the 0.10 bug of design Section 2.2 item 9).

The in-place variants do the same with one array, `is == os ==` its stride, and
`Placement::InPlace`. Offsets are counted from each view's first element: a layout carries no
offset, and execution passes FFTW the pointer to that element. Building a layout reads only
shapes and strides, never the storage.

Family rules (design Section 4.3): F2 implements c2c, which needs equal shapes. The r2c and
c2r rule for the last entry of `axes` (extent n/2 + 1, DC2), the real side of in-place r2c,
and r2r's kinds per axis are F5's; F5 relaxes check 4 for them.

Expected normalised layouts, which the tests of work item 8 assert with `Layout::new`:

| Array (shape, strides) | Call | Transform dims | Batch dims |
| --- | --- | --- | --- |
| column-major `[8, 3]` (1, 8) | `along_axis(…, 0)` | `(8, 1, 1)` | `(3, 8, 8)` |
| column-major `[8, 3]` | `along_axis(…, 1)` | `(3, 8, 8)` | `(8, 1, 1)` |
| column-major `[4, 5, 6]` (1, 4, 20) | `along_axis(…, 0)` | `(4, 1, 1)` | `(30, 4, 4)`, merged |
| column-major `[4, 5, 6]` | `along_axis(…, 1)` | `(5, 4, 4)` | `(6, 20, 20)`, `(4, 1, 1)` |
| column-major `[4, 5, 6]` | `along_axis(…, 2)` | `(6, 20, 20)` | `(20, 1, 1)`, merged |
| row-major `[4, 5, 6]` (30, 6, 1) | `along_axis(…, 2)` | `(6, 1, 1)` | `(20, 6, 6)`, merged |
| column-major `[4, 5, 6]` | `over_axes(…, &[0, 1])` | `(4, 1, 1)`, `(5, 4, 4)` | `(6, 20, 20)` |
| column-major `[4, 5, 6]` | `over_axes(…, &[1, 0])` | `(5, 4, 4)`, `(4, 1, 1)` | `(6, 20, 20)` |
| subview `[4, 5]` at `[2, 3]` of column-major `[10, 10]` | `along_axis(…, 0)` | `(4, 1, 1)` | `(5, 10, 10)` |

The two `over_axes` rows are different layouts, and different plans, of the same transform:
transform dimensions keep the order given.

### 5. Buffers for arrays (`buffer.rs`)

For `Array<A, N>` with the F1 bounds:

- `as_ptr`: the first element, `s.as_ptr().wrapping_add(offset)` from `storage()`, or a null
  pointer if `storage()` is `None`. `wrapping_add`, because the offset of an empty view can
  lie past the end of its storage (DC19); the pointer is only dereferenced after
  `check_side` has accepted the buffer.
- `as_mut_ptr`: the same from `storage_mut()`. It must derive from the exclusive borrow,
  never from `storage()` cast to mutable (design Section 4.4; `CLAUDE.md`).
- `len`: the storage elements from the first element on, `s.len() - offset`, and 0 without
  storage (DC17), stated in the rustdoc of `FftBuffer::len`.
- `check_side(layout, side)`, in this order, each failure a `LayoutMismatch` with the side:
  1. `storage()` is `Some`;
  2. the array's shape and strides match that side of the layout (design Section 4.8
     item 2), by the per-side rule of DC15;
  3. `offset + span <= s.len()`, with checked addition, for the layout's span of that side.

  Memory safety rests on step 3 alone: `StridedRawAccess` is a safe trait, so an
  implementation may be wrong, and step 2 compares only what the type claims.
- Messages for arrays name the expected and found values as text, for example expected
  "transform dimensions [(n 8, stride 1)] and batch dimensions [(n 3, stride 8)], the
  layout's input side" and found "an array of shape [8, 4] and strides [1, 8]"; for step 3,
  expected "at least 24 storage elements from the first element, the layout's input span"
  and found "a storage of 30 elements with the first element at offset 10".
- `AlignmentClass::of(&array)` needs nothing new: it reads `as_ptr()`, the first element, so
  a view at an offset has the class of that element (design Section 4.6).

`Planner::c2c_on` and `c2c_inplace_on` take arrays through the same traits; their checks
call `check_side` and need no change.

### 6. Execution (`execute.rs`)

- `check_buffers` keeps F1's order: buffers against the layout (`check_side` for each side,
  and the two-sided pairing of DC15 when both are arrays of the same rank), alignment classes,
  overlap. The overlap check uses `as_ptr()` and the spans, so for arrays it compares the
  byte ranges from each first element; it stays conservative (design Section 4.8 item 4).
- The `SAFETY:` comment of `execute_unchecked` is extended for arrays: the span fits the
  storage from the first element; the span of a view can include elements of other views,
  and FFTW reads and writes only the elements the layout addresses, never the gaps (design
  Section 4.8), which the sentinel tests of work item 8 check; the output pointer derives
  from an exclusive borrow of the whole storage (`StridedRawAccessMut`), so writing through
  it is permitted; for a `Preserving` plan the input pointer derives from a shared borrow and
  FFTW was given `FFTW_PRESERVE_INPUT`.

### 7. Test helpers (`test_support.rs`)

- **A logical reference.** The reference of F1 is applied along a raw layout. A layout built
  wrongly by the code under test (axes swapped, strides of the wrong side) would pass a
  comparison with the reference evaluated on that same layout, so F2's correctness tests
  compare with a reference on **logical indices**: the input view's values, read by
  multi-index and widened to f64 into a `DynArray`, transformed by the O(n²) DFT of F1
  (`dft`, made crate-visible) along each entry of `axes`, and compared by multi-index with the
  output view. It never builds a `Layout`.
- **Sentinel parents.** A helper makes a parent array at `FFTW_ALIGNED`, fills it with a
  sentinel (finite, compared with `to_bits`), takes a view, writes random values into the
  view's elements by multi-index, and afterwards checks that every element of the parent
  outside the view is still the sentinel, bit for bit. The set of the view's elements is
  computed from the view's storage offset and strides.
- Inputs from the seeded generator and seed of F1; every assertion names the seed, the
  memory order, the shape, the axes, the policy and the precision.

### 8. Tests of F2b (design Sections 7.2, 7.3 and 7.4, c2c)

**Rules for every FFTW test here**, as in F1: plan with `Rigor::Estimate` only; do not
depend on the wisdom present in the process; do not assume SIMD (the macOS runner has
none, `docs/fftw/F0-findings.md`). Keep F1's c2c tolerance, c · ε · (1 + log₂ N) with c = 4
and N the logical size; do not tune it. Every cell plans for the alignment classes of its
own arrays (`Planner::alignment(AlignmentClass::of(&input), AlignmentClass::of(&output))`),
since row-major `StridedDynArray`s are `Vec`-backed and views at an offset change class;
the class check itself is a negative test below.

- **Layout construction** (`layout.rs`): the rows of the table of work item 4, compared with
  `Layout::new`; `along_axis` equal to `over_axes` with one axis; the in-place variants; an
  array of shape `[n, 1]` along axis 0 equal to the layout of `[n]`; the errors of checks
  1–6, each naming its axis; an array with an extent 0; and the array of DC16 (shape
  `[2, 2]`, strides `[2, 3]`), rejected with `InvalidLayout` as output and in place, and
  accepted as input.
- **The correctness matrix**, fixed cells:
  - memory orders: column-major at `FFTW_ALIGNED`; row-major (`StridedDynArray::row_major`);
    strided with steps above 1 (a `StridedSliceArray` or `StridedSliceArrayMut` over aligned
    storage with strides `[2, 2 · n0, …]`); a subview of a larger parent; a nested subview,
    and a subview of a slice; a `slice` and a `multislice` of a parent of higher rank. Every
    view's parent is two or three elements larger on each axis, with the view at a non-zero
    origin, and is a sentinel parent (work item 7);
  - axes: 1-D arrays; along each axis of 2-D and 3-D arrays; over all axes of 2-D and 3-D
    arrays; over two axes of a 3-D array (`[0, 1]`, `[1, 2]`, `[0, 2]` and `[2, 0]`);
  - shapes: `[97]`; `[8, 6]` and `[17, 4]`; `[6, 5, 4]`;
  - the three policies; both directions and both precisions for the column-major and
    subview cells, f64 forward for the others;
  - each cell is compared with the logical reference; the input's parent is unchanged
    outside the view for every policy (FFTW does not touch the gaps, even with
    `FFTW_DESTROY_INPUT`), and bit for bit everywhere for `Preserving`; the output's parent
    is unchanged outside the view.

  A further 24 cells are sampled from the product of memory orders, shapes with extents up
  to 12, axes, policies, directions and precisions, with the fixed seed.
- **Identities** (design Section 7.3): the round trip on the forward cells; a batched plan
  along axis 0 of `[n, b]` against a loop of single plans on its columns (`col(j)`, with n a
  multiple of 8 so that every column has class 0 in both precisions), and a batched 2-D plan
  over `[0, 1]` of
  `[n0, n1, b]` against single plans on `slice(2, k)`, both to the tolerance, as FFTW may
  choose other codelets for another batch (design Section 3 item 7); and, bit for bit, where
  it is the same computation: an array of shape `[n, 1]` and one of shape `[n]` give equal
  layouts, one plan, and equal bits.
- **nd-fmm's fixed-batch unit** (design Sections 2.5 and 5 rule 6): boxes of shape
  `[n_phi, rings]` as `slice(2, k)` of a column-major parent `[n_phi, rings, boxes]` of c64 at
  `FFTW_ALIGNED`, with n_phi = 140 (a multiple of 4, so every box has class 0), rings = 5 and
  boxes = 3. One plan from `Layout::along_axis(&box_in, &box_out, 0)` executes on every box;
  each result is, bit for bit, that of the same plan on a standalone aligned copy of the box,
  and of a plan made with `c2c_on` for that box (in process; F4 repeats it in fresh
  processes), and is within the tolerance of the reference.
- **Negative tests** (design Section 7.4), each with an output whose parent is a sentinel
  parent left unchanged: a wrong shape; wrong strides (a row-major `[8, 3]` array against a
  plan from a column-major one); another alignment class (a subview one element from class 0:
  `AlignmentMismatch` with class 16 for c64 and 8 for c32); in place against out of place
  (an in-place layout from `along_axis_inplace` to `c2c`, an out-of-place one to
  `c2c_inplace`: `PlacementMismatch`); overlapping spans, from two `StridedSliceArray` views
  built in the test from one allocation with `slice::from_raw_parts`, with a comment, as in
  F1 (`Overlap`); the scratch cap with a column of a row-major `[4, 1000]` array and a small
  `max_scratch` (`ScratchTooLarge`), and the same layout planned with `c2c_on`; and a
  test-only implementation type with `Shape`, `Stride` and a `StridedRawAccess` whose
  storage is shorter than its shape and strides claim, which execution rejects with
  `LayoutMismatch` before calling FFTW.
- **The pairing of DC15**: with the arrays of the example in "Open questions" item 2, the
  execution is rejected with `LayoutMismatch` when both buffers are arrays; the same memory
  passed as slices is accepted, as documented.
- **`compile_fail` doctests**, each with a passing twin that differs only in the misuse: a
  shared view `b.r()` as the output of `execute`; `Layout::along_axis` on a `transpose()`
  view.
- **Planning on arrays**: `c2c_on` and `c2c_inplace_on` on views give plans that run on them.
- **Budget.** The FFTW unit tests of F1 run in about 1.4 s in a debug build. F2's additions
  together take at most about 1 s more; measure locally and report it in the pull request.
  Keep the shapes of the matrix small; only the nd-fmm unit uses a size above 100.

### 9. Documentation, CHANGELOG and status

- The four `ignore` examples of the layout functions run, and say what the layout is; the
  `along_axis` example also plans and executes, in the form of design Section 4.2. Remove
  "Implemented in task F2" from their rustdoc and "(task F2)" from that of `FftBuffer` and
  `FftBufferMut`.
- Rustdoc of the layout functions: the checks and errors of work item 4; offsets counted from
  the first element; that the scratch of `Planner::c2c` covers the span from the first
  element to the last one addressed, so a view of a large array may need `max_scratch` or
  the `*_on` methods; that a plan describes memory, not logical axes (DC15).
- An array example for `AlignmentClass::of` (an aligned array and a subview at an offset),
  and for one `execute` method.
- The section "New interface (0.11, in progress)" of `rlst/src/fftw.rs`: layouts from arrays
  and views, and buffers for arrays, in the list; remove "Layouts from RLST arrays" from the
  sentence on what follows.
- `rlst/src/doc/dense_linear_algebra.rs`, in F2a: a short paragraph beside `data()` on
  `storage()` and `storage_mut()` (DC18): what they return, that the slice may hold elements
  of other views, and when to use them rather than `data()`. No other guide changes; the
  FFTW guide is written in F7 (design, Section 9).
- `rlst/CHANGELOG.md`, under `## v0.11.0 (unreleased)`: F2a completes the entry "New traits
  `StridedRawAccess` and `StridedRawAccessMut`" with the types that implement them and those
  that do not, and with the inherent `Array::storage()` and `storage_mut()`; F2b extends "New FFTW interface (in progress)" by layouts from arrays and views
  and execution on them.
- The status table of `docs/fftw/README.md` when each pull request merges. F2 changes no
  feature, CI job or validation command, so `CLAUDE.md` is expected not to change; check it.

## Pull requests and CI

- **F2a** first, titled for example `feat(dense): F2a strided storage of arrays and views`.
  The open questions of this brief were settled before the task started (DC14–DC19); a
  question that arises during the task is recorded under "Open questions" and raised in the
  pull request. F2b starts after F2a has merged.
- **F2b**, titled for example `feat(fftw): F2b layouts and execution on arrays and views`,
  branched from `fftw-0.11` after F2a has merged.
- CI (`.github/workflows/run-tests.yml`) runs only for pull requests into `main` or
  `fftw-0.11`. A pull request stacked on another branch gets no CI until it is retargeted, and
  retargeting alone does not start a run: close and reopen the pull request, or push to it.

## Acceptance

F2a:

- [ ] `StridedRawAccess` is implemented for every type of the table of work item 1, and
      `StridedRawAccessMut` for its mutable ones; no blanket impl; none for the excluded types.
- [ ] `RawAccess` and `RawAccessMut` are unchanged, and no existing test or doctest changed.
- [ ] `Array::storage()` and `Array::storage_mut()` exist, with rustdoc, an example each and
      the paragraph in `rlst/src/doc/dense_linear_algebra.rs` (DC18).
- [ ] The tests of work item 3 pass in the default `cargo test`.

F2b:

- [ ] No `todo!()` is left in `layout.rs` or `buffer.rs`; the four layout examples run.
- [ ] The tests of work item 8 pass, in CI on Linux and macOS; the added test time in a
      debug build is reported in the pull request and is at most about 1 s.
- [ ] Every `unsafe` block has a `SAFETY:` comment; the comment on execution covers arrays,
      the gaps of views and the provenance of the output pointer.
- [ ] Every FFTW call other than execution runs inside `with_fftw`; no dependency was added.
- [ ] Every new public behaviour has rustdoc with an example.

Both:

- [ ] The six Clippy commands of `CLAUDE.md` pass (or are reported unavailable), as do
      `cargo fmt -- --check`, `cargo test` and
      `cargo test --features "fftw,fftw_system,strict"`.
- [ ] `rlst/CHANGELOG.md` and `docs/fftw/README.md` are updated; `CLAUDE.md` is checked.

## Pitfalls

- **`usize` and `isize`.** RLST's shapes and strides are `usize`; `Dim` uses `isize`, as
  FFTW's `fftw_iodim64` does. Convert with `isize::try_from` and return `InvalidLayout` on
  overflow; never cast with `as`.
- **Dimension order.** RLST is column-major, and FFTW's documentation speaks of row-major
  arrays, but the guru interface takes each dimension with its own strides. Pass the
  dimensions in the order of `axes` and reverse nothing; reversing shapes is what put 0.10's
  kinds on the wrong axes (design Section 2.2 item 9).
- **Batch order.** Normalisation sorts and merges batch dimensions; transform dimensions keep
  the order of `axes`. Along the middle axis of a contiguous 3-D array two batch dimensions
  remain; along the first or the last axis, one.
- **Input and output from one array.** `Layout::along_axis(&a, &a, 0)` compiles and gives an
  out-of-place layout, but `a` cannot then be borrowed as input and output of one execution;
  in-place transforms use the `*_inplace` layouts and plans.
- **Pointer provenance.** `as_mut_ptr` comes from `storage_mut()`. A pointer from
  `storage()` must never reach FFTW for memory a plan writes.
- **Safe traits.** `StridedRawAccess`, `Shape` and `Stride` are safe to implement. The
  execution check of the span against the storage length is what keeps execution sound; do
  not skip it when the shape and strides already match.
- **Extent 0.** Empty arrays and views are rejected when a layout is made. The storage
  offset of an empty view can lie past the end of its storage (DC19): do not
  slice or index with it, and use `wrapping_add` for pointers.
- **Extent 1.** The stride of an axis of extent 1 is arbitrary in RLST (`insert_empty_axis`,
  subviews, slices); batch dimensions of extent 1 are dropped by normalisation, transform
  dimensions of extent 1 are kept with their stride.
- **Alignment of row-major arrays.** `StridedDynArray` is backed by a `Vec`, so its class
  depends on the allocator; there is no aligned row-major owned type. Tests plan for the
  classes of their arrays, or use strided views over aligned storage; none may assume
  class 0 for them.
- **Scratch for views.** The scratch spans from a view's first element to its last, which
  for a column of a row-major matrix is nearly the whole matrix (DC3).
- **The reference.** Do not check an array layout against the F1 reference applied to the
  same layout; use the logical reference of work item 7.
- **Sentinels.** Compare them with `to_bits`, and make them finite and distinct from every
  value a transform can produce in the test.
- **Wisdom, SIMD, debug builds, `strict`, wildcard imports**: as in the F1 brief.

## Open questions

Record here anything the design leaves unclear, and raise it; do not resolve design
questions in code.

Raised while writing this brief, and **settled by the maintainer before the task started**
(design draft 0.9, DC14–DC19, Appendix A): questions 1–4 and 6 with the option marked
*proposed*, question 5 with option (b). The work items follow these answers.

1. **Where c2c's equal shapes are checked.** Design Section 4.3 says validation by family
   happens at planning, and the sketch of Section 4.2 has a `Layout` holding "the shapes of
   both sides". F1's `Layout` holds one extent per dimension and no shapes or element types,
   and the planner receives only the `Layout`, so `Planner::c2c` cannot see unequal shapes.
   Options: (a) *proposed*: the array layout functions reject unequal extents on any axis
   (check 4 of work item 4), and F5 relaxes the last entry of `axes` for r2c and c2r and adds
   to `Layout` what its planner then needs; (b) F2 adds the extents of both sides to `Layout`
   now, and `Planner::c2c` checks their equality, with raw layouts recording equal extents.
   (a) leaves F1's `Layout` as it is; (b) fixes now the representation F5 will use.
   *Decided: (a), DC14.*
2. **What execution compares for an array.** Design Section 4.8 item 2 says the array's
   normalised shape and strides equal the plan's layout. F1's `check_side` sees one buffer
   and one side, and the layout is two-sided and records no axes. A per-side comparison
   (each transform dimension matched by an array axis of the same extent and, above extent 1,
   the same stride; the remaining axes normalised on that side alone and compared with the
   layout's batch on that side) catches wrong shapes and strides, and the span check keeps it
   memory-safe, but it cannot see how input and output batch axes pair. Example: a plan from
   a column-major input `[4, 2, 3]` and an output of the same shape with strides `[1, 12, 4]`
   has the batch `(3, 8, 4)`, `(2, 4, 12)`; column-major arrays of shape `[4, 3, 2]` pass
   both per-side checks, and the transform of input column `[:, 1, 0]` lands in output column
   `[:, 0, 1]`. No stride-based check can tell logical axes apart either: a plan along axis 0
   of a square column-major array transforms along axis 1 of a square row-major one, whose
   memory layout is the same. Options: (a) per-side checks only, with both points
   documented; (b) *proposed*: per-side checks, and, when both buffers are arrays of the same
   rank, a two-sided check that pairs their axes by index (equal shapes; each transform
   dimension matched by one axis with its extent and both strides; the other axes normalised
   by F1's batch normalisation and equal to the layout's batch), through a crate-private
   addition to the sealed trait; calls that mix a slice and an array keep the per-side
   check; (c) as (b), and layouts made from arrays also record their transform axes, which
   execution then requires. Under (a) and (b) the documentation states that a plan describes
   memory, not logical axes; (c) ties plans to axes, which design goal 2 does not ask for.
   *Decided: (b), DC15.*
3. **Arrays that the distinct-output rule rejects.** Design Section 4.3 says layouts built
   from RLST arrays satisfy the raw-layout rules by construction. `StridedBaseArray::new`
   accepts any strides that map indices to distinct offsets, such as shape `[2, 2]` with
   strides `[2, 3]` (offsets 0, 2, 3 and 5), which the sufficient rule of `Layout::new`
   rejects: 3 is below 2 · 2. Options: (a) *proposed*: such arrays are rejected with
   `InvalidLayout` as outputs and in place (they are accepted as inputs), the rustdoc says
   so, and the sentence of Section 4.3 is corrected (Appendix A); (b) an exact distinctness
   check for layouts made from arrays, in O(elements) at layout construction, as
   `StridedBaseArray::new` does. *Decided: (a), DC16; design Section 4.3 is corrected.*
4. **`FftBuffer::len` for arrays.** Its rustdoc says "the number of elements". For a slice
   that is also the memory behind `as_ptr`; for a strided view the two differ. Options:
   (a) *proposed*: the storage elements from the first element on (`s.len() - offset`), the
   memory a plan may address, with the rustdoc saying so for arrays; (b) the number of the
   array's elements, as `Array::len`. F1's execution does not use `len`; the choice concerns
   callers only. *Decided: (a), DC17.*
5. **Inherent `storage()` on `Array`.** `Array` implements none of the accessor traits, and
   F2 needs no forwarding impl, but users can then reach the storage only through
   `arr.imp().storage()`. Options: (a) *proposed*: no inherent methods, as the design names
   none; (b) inherent `Array::storage()` and `storage_mut()` beside `data()` and
   `data_mut()`, with a paragraph in `doc/dense_linear_algebra.rs`. (b) can be added later
   without breaking anything. *Decided: (b), in F2a, DC18.*
6. **The offset of an empty view.** A subview with an extent 0 can start past the end of the
   storage (origin `[4, 3]` and shape `[0, 0]` in a column-major `[4, 3]` array give offset
   16 in a storage of 12), and the contract of `StridedRawAccess` says nothing about views
   without elements. Options: (a) *proposed*: the rustdoc states that for a view without
   elements the offset may lie past the end of the slice and must not be used to index it;
   (b) clamp the offset to the slice length for empty views; (c) return `None` for them.
   *Decided: (a), DC19.*
