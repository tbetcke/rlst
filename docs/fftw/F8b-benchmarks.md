# F8b: benchmark findings

Written by task F8b-1 (`tasks/F8b-benchmarks-release.md`, work items 1 and 2). The benchmarks
are the example `rlst/examples/fftw_bench.rs`. Following DC53, the results are indicative: one
run per machine, on a laptop and on a shared CI runner.

## Method

```bash
cargo run --release --example fftw_bench --features fftw_system
```

- Double precision, at nd-fmm's sizes N_φ = 140, 216, 336, 588 and 1080, with 9 rings per box.
- Every plan uses `Rigor::Estimate`, and no wisdom is imported (DC54). The tables give each
  plan's fingerprint, so the findings name the plans they timed.
- Each case warms up while the example finds how many executions take at least 0.2 s. It then
  times 7 rounds of that many executions. The tables give the median time per execution and
  the range of the 7 rounds (DC54).
- Buffers are `FFTW_ALIGNED` arrays, filled once with seeded values. Except where a table says
  otherwise, execution takes the arrays, so each time includes RLST's checks at execution.

## Machines

| | macOS | Linux |
| --- | --- | --- |
| CPU | Apple M3 Max | AMD EPYC 9V74 80-Core Processor, 4 vCPUs (GitHub's `ubuntu-24.04` runner) |
| Operating system | macOS 27.0.1, arm64 | Ubuntu 24.04, x86_64 |
| FFTW | Homebrew `fftw-3.3.11`, no SIMD | Ubuntu `fftw-3.3.10-sse2-avx`; the CPU has AVX2 and AVX-512 |
| Rust | 1.99.0, release profile | stable 1.99, release profile |
| Run | locally, at `fftw-0.11` `46d8230` plus the example | temporary step of draft #39 (CI run 40133737984), closed afterwards |

## Results

### 1. Batched against a loop

c2c forward along axis 0 of `[N, 9]`:
- **One plan, batch of 9:** a single plan over the whole box.
- **On slices:** the same plan, executed on the arrays' data as slices, which execution checks
  by length only.
- **9 executions:** a plan of one transform, executed on each column view.

macOS:

| N | Batch of 9 (µs) | Batch of 9, on slices | 9 executions of one transform |
| ---: | ---: | ---: | ---: |
| 140 | 3.28 (3.27–3.34) | 2.45 (0.75×) | 6.05 (1.85×) |
| 216 | 5.26 (5.25–5.37) | 4.45 (0.85×) | 8.01 (1.52×) |
| 336 | 8.63 (8.60–9.13) | 7.80 (0.90×) | 11.50 (1.33×) |
| 588 | 14.59 (14.55–14.67) | 13.74 (0.94×) | 17.28 (1.18×) |
| 1080 | 28.94 (28.89–29.16) | 28.01 (0.97×) | 31.90 (1.10×) |

Linux:

| N | Batch of 9 (µs) | Batch of 9, on slices | 9 executions of one transform |
| ---: | ---: | ---: | ---: |
| 140 | 2.41 (2.40–2.50) | 1.79 (0.74×) | 5.42 (2.25×) |
| 216 | 5.66 (5.64–5.77) | 5.01 (0.89×) | 8.56 (1.51×) |
| 336 | 7.38 (7.38–7.39) | 6.79 (0.92×) | 10.59 (1.43×) |
| 588 | 14.37 (14.37–14.38) | 13.73 (0.96×) | 17.73 (1.23×) |
| 1080 | 33.93 (33.90–33.95) | 33.32 (0.98×) | 37.37 (1.10×) |

- **The batched plan is faster at every size on both machines:** from 1.10× at N = 1080 up
  to 1.85× (macOS) and 2.25× (Linux) at N = 140.
- **Checking arrays at execution costs about 0.85 µs per call on macOS and 0.6 µs on Linux.**
  That is the cost of checking the box `[N, 9]`'s shape, strides and storage, measured as the
  difference between arrays and slices; it does not depend on N. It is a quarter of the time
  at N = 140 and 2–3 % at N = 1080. The cost is reported here and raised in the brief's
  "Open questions"; F8b does not tune it.

**Later (2026-10-25, issue [#44](https://github.com/tbetcke/rlst/issues/44)).** The array
check allocated 40–48 times per execution: the shapes, strides and normalised batches it
compared were built as `Vec`s, and a buffer that the plan writes was compared twice. It now
works on the stack for arrays of rank up to 16, the layout normalises its batch on each side
once, when it is made, and a written buffer's second check covers its storage and span only.
The checks and their errors are unchanged. On macOS (the same machine and FFTW), the batch of
9 on arrays, before and after, against the same plan on slices:

| N | Arrays, before (µs) | Arrays, after | On slices |
| ---: | ---: | ---: | ---: |
| 140 | 3.22 | 2.58 | 2.43 |
| 216 | 5.25 | 4.60 | 4.45 |
| 336 | 8.62 | 7.95 | 7.83 |
| 588 | 14.59 | 13.73 | 13.61 |
| 1080 | 29.03 | 28.10 | 27.95 |

On Linux, before and after were built and run in one job on the `ubuntu-24.04` runner,
alternating, twice each (the means of the two runs' medians below). That runner had an AMD
EPYC 9V45 and FFTW 3.3.10 with SIMD, a different CPU from the table above, so its times are
compared only with each other:

| N | Arrays, before (µs) | On slices | Arrays, after | On slices |
| ---: | ---: | ---: | ---: | ---: |
| 140 | 1.65 | 1.25 | 1.35 | 1.23 |
| 216 | 3.96 | 3.50 | 3.64 | 3.55 |
| 336 | 5.21 | 4.70 | 4.95 | 4.79 |
| 588 | 10.20 | 9.77 | 9.84 | 9.62 |
| 1080 | 23.78 | 22.91 | 23.06 | 22.75 |

The cost of checking arrays falls from about 0.8 µs to about 0.15 µs per call on macOS, and
from about 0.45 µs to about 0.1 µs on Linux at N ≤ 336; at larger N the shared runner's
round-to-round variation, up to 0.3 µs, is as large as the remaining cost. At N = 140 it is now
6 % of the time on macOS and 8 % on Linux, from a quarter.

### 2. Split against interleaved

c2c forward along axis 0 of the box:
- **Interleaved:** `[N, 9]` complex.
- **Split:** `[N, 9, 2]` real, with the part axis outermost (nd-fmm's planes).
- **Copy, then interleaved:** a plain loop copies the planes into interleaved scratch, then the
  interleaved plan runs. The loop's output is interleaved, and its plan executes on the
  scratch as a slice.

| N | macOS: interleaved (µs) | split | copy, then interleaved | Linux: interleaved (µs) | split | copy, then interleaved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 140 | 3.22 | 3.36 (1.04×) | 3.04 (0.94×) | 2.47 | 7.03 (2.84×) | 2.51 (1.01×) |
| 216 | 5.29 | 5.43 (1.03×) | 5.25 (0.99×) | 5.39 | 8.21 (1.52×) | 5.67 (1.05×) |
| 336 | 8.63 | 8.94 (1.04×) | 9.27 (1.07×) | 7.48 | 13.21 (1.77×) | 8.02 (1.07×) |
| 588 | 14.66 | 14.71 (1.00×) | 16.12 (1.10×) | 14.28 | 30.82 (2.16×) | 15.44 (1.08×) |
| 1080 | 28.98 | 29.14 (1.01×) | 32.05 (1.11×) | 34.09 | 54.15 (1.59×) | 36.42 (1.07×) |

- **macOS (no SIMD):** split and interleaved take the same time, within 4 %, and their plans
  have the same fingerprints. Without SIMD, FFTW runs the same scalar codelets on both.
- **Linux (SIMD):** split plans take 1.5–2.8× the interleaved time, and their fingerprints
  differ. Copying the planes into interleaved scratch and running the interleaved plan costs
  1.01–1.08× the interleaved time, so on this build it is much faster than the split plan.
  This agrees with design Section 11 item 2, from another machine.
- **The documentation's notes stand.** Split plans can use fewer SIMD stages than interleaved
  ones, and an interleaved copy can be faster. On macOS the copy is up to 11 % slower, since
  there is no SIMD to recover. Nothing contradicts the notes, so they do not change (brief,
  work item 2).
- **The 0.94× at N = 140 on macOS** reflects the slice check of the copied path (comparison 1),
  not a faster transform.

### 3. Aligned against unaligned

c2c forward along axis 0 of `[N, 9]`, executed on slices of `FFTW_ALIGNED` buffers:
- class 0 at offset 0;
- class 16 at an offset of one complex value.

| N | macOS: class 0 plan, class 0 (µs) | `.unaligned()`, class 0 | `.unaligned()`, class 16 | class 16 plan, class 16 | Linux: class 0 plan, class 0 (µs) | `.unaligned()`, class 0 | `.unaligned()`, class 16 | class 16 plan, class 16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 140 | 2.44 | 0.99× | 0.99× | 0.99× | 1.73 | 3.65× | 3.65× | 1.06× |
| 216 | 4.48 | 0.99× | 0.99× | 0.99× | 4.85 | 2.10× | 2.11× | 1.03× |
| 336 | 7.80 | 1.00× | 1.00× | 1.00× | 6.79 | 3.22× | 3.22× | 1.03× |
| 588 | 13.62 | 1.00× | 1.00× | 1.00× | 13.86 | 2.15× | 2.14× | 1.05× |
| 1080 | 28.01 | 1.00× | 1.00× | 1.01× | 33.28 | 1.58× | 1.58× | 1.05× |

- **macOS (no SIMD):** alignment makes no difference, and all four plans of a size have one
  fingerprint.
- **Linux (SIMD):**
  - `.unaligned()` plans take 1.6–3.7× the time, on aligned and on offset buffers alike. At
    N = 140, 216 and 336 their fingerprints equal the macOS plans without SIMD
    (`358e29e51123774d`, `376e9cb2f8f3cc50`, `5aedf4800f4245c4`), so FFTW falls back to
    scalar codelets.
  - A plan made for class 16 with `Planner::alignment` keeps the class 0 plan's fingerprint
    and costs 3–6 % more on the offset buffers.
- **The documentation's notes stand.** `.unaligned()` costs most SIMD code, and planning for
  a view's class is the better remedy. Nothing contradicts the notes, so they do not change.

### 4. Halving the contiguous axis against a strided one (DC2)

r2c and c2r over both axes of a column-major `[N, N]`:
- `axes = [1, 0]` halves axis 0, the contiguous one, so the complex array is `[N / 2 + 1, N]`;
- `axes = [0, 1]` halves axis 1, a strided one, so the complex array is `[N, N / 2 + 1]`.

A c2r plan destroys its input, and `preserve_input` is not available over two dimensions. So
each c2r execution first restores the spectrum from a saved copy. The copy alone takes 2–6 %
of the c2r time; the ratios below include it, and leaving it out changes them by at most 3
percentage points.

| N | macOS r2c: contiguous halved (µs) | strided halved | Linux r2c: contiguous halved (µs) | strided halved |
| ---: | ---: | ---: | ---: | ---: |
| 140 | 41.76 (41.70–42.07) | 1.02× | 44.12 (44.09–44.19) | **1.14×** |
| 216 | 139.52 (139.19–140.91) | 0.98× | 163.95 (163.81–164.18) | 0.97× |
| 336 | 362.26 (361.01–366.14) | 0.98× | 398.90 (398.26–399.58) | 1.04× |
| 588 | 1042.80 (1040.45–1058.61) | 1.03× | 1311.22 (1302.36–1332.09) | **1.14×** |
| 1080 | 4029.31 (4010.83–4085.56) | **1.10×** | 7054.70 (6764.84–8003.55) | 1.06× |

| N | macOS c2r: contiguous halved (µs) | strided halved | Linux c2r: contiguous halved (µs) | strided halved |
| ---: | ---: | ---: | ---: | ---: |
| 140 | 45.51 (45.30–45.68) | 1.00× | 49.40 (49.19–54.44) | **1.50×** |
| 216 | 153.45 (153.38–154.98) | 0.97× | 174.30 (173.85–177.09) | 1.03× |
| 336 | 369.58 (368.43–371.60) | 1.00× | 413.63 (412.67–416.39) | **1.32×** |
| 588 | 1093.36 (1091.06–1106.43) | 1.03× | 1385.58 (1380.80–1388.61) | **1.48×** |
| 1080 | 4230.93 (4203.43–4262.44) | **1.14×** | 6507.60 (6393.19–6985.92) | **1.21×** |

In bold: the sizes where halving the contiguous axis passes DC55's test. It is faster by at
least 10 %, and the difference of the medians exceeds the wider of the two ranges.

**DC55's rule.** The rule recommends listing the contiguous axis last only if that passes
the test at most sizes on every machine measured.

| | macOS | Linux |
| --- | --- | --- |
| r2c | 1 of 5 sizes | 2 of 5 sizes |
| c2r | 1 of 5 sizes | 4 of 5 sizes |

So the rule holds for neither family: on macOS the order makes no difference below N = 1080.
The documentation therefore says that the order was measured and recommends nothing (DC55):
- the guide's "Families and input policies";
- the rustdoc of `Layout::over_axes`, which also covers r2c and c2r layouts from arrays;
- `Layout::split_r2c_over_axes`, whose notes `split_c2r_over_axes` inherits.

The in-place r2c and c2r layouts have no choice, since their halved axis must be contiguous.

Two notes:
- On Linux, c2r with the contiguous axis halved is 21–50 % faster at four of the five sizes.
  This is recorded in the brief's "Open questions", in case the maintainers want a narrower
  statement than DC55 allows.
- At N = 1080 the Linux ranges are wide (up to 18 % of the median) on the shared runner, and
  the fingerprints of the two orders differ at every size on both machines except N = 140 on
  macOS. The two orders are different plans, as `Layout::over_axes` says.

## Raw output

The example's output, unedited except that its headings are moved down two levels.

### macOS

- FFTW: `fftw-3.3.11`
- CPU: Apple M3 Max
- Target: macos aarch64
- Method: `Estimate` plans; per case a warm-up, then 7 rounds of at least 200 ms; median and range of the time per execution

#### 1. Batched against a loop: c2c forward along axis 0 of [N, 9]

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | one plan, batch of 9 | 3.28 | 3.27–3.34 | 1.00 | `358e29e51123774d` |
| 140 | one plan, batch of 9, on slices | 2.45 | 2.43–2.46 | 0.75 | `358e29e51123774d` |
| 140 | 9 executions of one transform | 6.05 | 6.03–6.10 | 1.85 | `8ac74426fe5b662f` |
| 216 | one plan, batch of 9 | 5.26 | 5.25–5.37 | 1.00 | `376e9cb2f8f3cc50` |
| 216 | one plan, batch of 9, on slices | 4.45 | 4.44–4.51 | 0.85 | `376e9cb2f8f3cc50` |
| 216 | 9 executions of one transform | 8.01 | 7.96–8.05 | 1.52 | `8fdddc9136c9601c` |
| 336 | one plan, batch of 9 | 8.63 | 8.60–9.13 | 1.00 | `5aedf4800f4245c4` |
| 336 | one plan, batch of 9, on slices | 7.80 | 7.77–7.98 | 0.90 | `5aedf4800f4245c4` |
| 336 | 9 executions of one transform | 11.50 | 11.46–11.60 | 1.33 | `e6cca57f02d82c58` |
| 588 | one plan, batch of 9 | 14.59 | 14.55–14.67 | 1.00 | `f45b5047f247041c` |
| 588 | one plan, batch of 9, on slices | 13.74 | 13.61–14.12 | 0.94 | `f45b5047f247041c` |
| 588 | 9 executions of one transform | 17.28 | 17.21–17.36 | 1.18 | `ad508aa1977c5d20` |
| 1080 | one plan, batch of 9 | 28.94 | 28.89–29.16 | 1.00 | `6467e8553ee10853` |
| 1080 | one plan, batch of 9, on slices | 28.01 | 27.95–28.33 | 0.97 | `6467e8553ee10853` |
| 1080 | 9 executions of one transform | 31.90 | 31.76–32.20 | 1.10 | `b249283e6fab62af` |

#### 2. Split against interleaved: c2c forward along axis 0 of the box [N, 9]

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | interleaved [N, 9] | 3.22 | 3.21–3.24 | 1.00 | `358e29e51123774d` |
| 140 | split [N, 9, 2] | 3.36 | 3.36–3.41 | 1.04 | `358e29e51123774d` |
| 140 | copy of the planes into interleaved scratch, then interleaved | 3.04 | 3.04–3.07 | 0.94 | `358e29e51123774d` |
| 216 | interleaved [N, 9] | 5.29 | 5.26–5.34 | 1.00 | `376e9cb2f8f3cc50` |
| 216 | split [N, 9, 2] | 5.43 | 5.42–5.47 | 1.03 | `376e9cb2f8f3cc50` |
| 216 | copy of the planes into interleaved scratch, then interleaved | 5.25 | 5.22–5.27 | 0.99 | `376e9cb2f8f3cc50` |
| 336 | interleaved [N, 9] | 8.63 | 8.61–8.71 | 1.00 | `5aedf4800f4245c4` |
| 336 | split [N, 9, 2] | 8.94 | 8.79–9.27 | 1.04 | `5aedf4800f4245c4` |
| 336 | copy of the planes into interleaved scratch, then interleaved | 9.27 | 9.21–9.32 | 1.07 | `5aedf4800f4245c4` |
| 588 | interleaved [N, 9] | 14.66 | 14.57–14.85 | 1.00 | `f45b5047f247041c` |
| 588 | split [N, 9, 2] | 14.71 | 14.68–14.81 | 1.00 | `f45b5047f247041c` |
| 588 | copy of the planes into interleaved scratch, then interleaved | 16.12 | 16.09–16.33 | 1.10 | `f45b5047f247041c` |
| 1080 | interleaved [N, 9] | 28.98 | 28.93–29.40 | 1.00 | `6467e8553ee10853` |
| 1080 | split [N, 9, 2] | 29.14 | 29.04–29.63 | 1.01 | `6467e8553ee10853` |
| 1080 | copy of the planes into interleaved scratch, then interleaved | 32.05 | 31.64–32.29 | 1.11 | `6467e8553ee10853` |

#### 3. Aligned against unaligned: c2c forward along axis 0 of [N, 9]

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | class 0 plan, class 0 buffers | 2.44 | 2.42–2.54 | 1.00 | `358e29e51123774d` |
| 140 | `.unaligned()` plan, class 0 buffers | 2.43 | 2.42–2.46 | 0.99 | `358e29e51123774d` |
| 140 | `.unaligned()` plan, class 16 buffers | 2.42 | 2.42–2.45 | 0.99 | `358e29e51123774d` |
| 140 | class 16 plan, class 16 buffers | 2.42 | 2.42–2.46 | 0.99 | `358e29e51123774d` |
| 216 | class 0 plan, class 0 buffers | 4.48 | 4.47–4.49 | 1.00 | `376e9cb2f8f3cc50` |
| 216 | `.unaligned()` plan, class 0 buffers | 4.45 | 4.44–4.49 | 0.99 | `376e9cb2f8f3cc50` |
| 216 | `.unaligned()` plan, class 16 buffers | 4.45 | 4.44–4.46 | 0.99 | `376e9cb2f8f3cc50` |
| 216 | class 16 plan, class 16 buffers | 4.45 | 4.44–4.49 | 0.99 | `376e9cb2f8f3cc50` |
| 336 | class 0 plan, class 0 buffers | 7.80 | 7.79–7.88 | 1.00 | `5aedf4800f4245c4` |
| 336 | `.unaligned()` plan, class 0 buffers | 7.83 | 7.79–8.08 | 1.00 | `5aedf4800f4245c4` |
| 336 | `.unaligned()` plan, class 16 buffers | 7.81 | 7.79–7.86 | 1.00 | `5aedf4800f4245c4` |
| 336 | class 16 plan, class 16 buffers | 7.80 | 7.80–7.90 | 1.00 | `5aedf4800f4245c4` |
| 588 | class 0 plan, class 0 buffers | 13.62 | 13.60–13.78 | 1.00 | `f45b5047f247041c` |
| 588 | `.unaligned()` plan, class 0 buffers | 13.61 | 13.60–13.66 | 1.00 | `f45b5047f247041c` |
| 588 | `.unaligned()` plan, class 16 buffers | 13.64 | 13.60–14.28 | 1.00 | `f45b5047f247041c` |
| 588 | class 16 plan, class 16 buffers | 13.64 | 13.62–13.88 | 1.00 | `f45b5047f247041c` |
| 1080 | class 0 plan, class 0 buffers | 28.01 | 27.99–29.03 | 1.00 | `6467e8553ee10853` |
| 1080 | `.unaligned()` plan, class 0 buffers | 28.04 | 27.96–29.62 | 1.00 | `6467e8553ee10853` |
| 1080 | `.unaligned()` plan, class 16 buffers | 27.98 | 27.95–28.37 | 1.00 | `6467e8553ee10853` |
| 1080 | class 16 plan, class 16 buffers | 28.25 | 28.02–28.96 | 1.01 | `6467e8553ee10853` |

#### 4a. r2c over both axes of a column-major [N, N]: the halved axis

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | axes [1, 0]: axis 0 (contiguous) halved | 41.76 | 41.70–42.07 | 1.00 | `11333262de3cbb7d` |
| 140 | axes [0, 1]: axis 1 (strided) halved | 42.43 | 42.31–42.98 | 1.02 | `11333262de3cbb7d` |
| 216 | axes [1, 0]: axis 0 (contiguous) halved | 139.52 | 139.19–140.91 | 1.00 | `5049c45a86272dcd` |
| 216 | axes [0, 1]: axis 1 (strided) halved | 136.93 | 136.61–138.51 | 0.98 | `9c99fd95014783b4` |
| 336 | axes [1, 0]: axis 0 (contiguous) halved | 362.26 | 361.01–366.14 | 1.00 | `a8bfa14c43166bfa` |
| 336 | axes [0, 1]: axis 1 (strided) halved | 356.25 | 355.38–364.33 | 0.98 | `2805ff13dc3a81c8` |
| 588 | axes [1, 0]: axis 0 (contiguous) halved | 1042.80 | 1040.45–1058.61 | 1.00 | `5cb93f9fd54a2b32` |
| 588 | axes [0, 1]: axis 1 (strided) halved | 1070.98 | 1069.48–1087.50 | 1.03 | `554711c991a1c922` |
| 1080 | axes [1, 0]: axis 0 (contiguous) halved | 4029.31 | 4010.83–4085.56 | 1.00 | `17efed82c43d5c5f` |
| 1080 | axes [0, 1]: axis 1 (strided) halved | 4437.96 | 4407.63–4521.00 | 1.10 | `5cb0e9e0c683eef3` |

#### 4b. c2r over both axes of a column-major [N, N]: the halved axis

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 45.51 | 45.30–45.68 | 1.00 | `1a100f43a7462fea` |
| 140 | axes [0, 1]: axis 1 (strided) halved, with the copy | 45.65 | 45.38–45.87 | 1.00 | `1a100f43a7462fea` |
| 140 | the copy alone | 1.67 | 1.67–1.70 | 0.04 | — |
| 216 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 153.45 | 153.38–154.98 | 1.00 | `f567d16aa7c9b22a` |
| 216 | axes [0, 1]: axis 1 (strided) halved, with the copy | 148.83 | 148.74–150.51 | 0.97 | `11db6ee6139bf7ff` |
| 216 | the copy alone | 4.45 | 4.43–4.50 | 0.03 | — |
| 336 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 369.58 | 368.43–371.60 | 1.00 | `7ccd33b9846925eb` |
| 336 | axes [0, 1]: axis 1 (strided) halved, with the copy | 369.06 | 366.96–370.67 | 1.00 | `cce060eaa3d93e35` |
| 336 | the copy alone | 10.74 | 10.69–10.88 | 0.03 | — |
| 588 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 1093.36 | 1091.06–1106.43 | 1.00 | `7f72b230e30f6890` |
| 588 | axes [0, 1]: axis 1 (strided) halved, with the copy | 1125.39 | 1122.27–1146.00 | 1.03 | `e09680cff6cfad10` |
| 588 | the copy alone | 32.79 | 32.73–35.08 | 0.03 | — |
| 1080 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 4230.93 | 4203.43–4262.44 | 1.00 | `fbf7fbabe4139329` |
| 1080 | axes [0, 1]: axis 1 (strided) halved, with the copy | 4828.03 | 4803.07–4892.67 | 1.14 | `d9706d17b14dcd19` |
| 1080 | the copy alone | 112.33 | 110.71–116.27 | 0.03 | — |

### Linux

- FFTW: `fftw-3.3.10-sse2-avx`
- CPU: AMD EPYC 9V74 80-Core Processor
- Target: linux x86_64
- Method: `Estimate` plans; per case a warm-up, then 7 rounds of at least 200 ms; median and range of the time per execution

#### 1. Batched against a loop: c2c forward along axis 0 of [N, 9]

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | one plan, batch of 9 | 2.41 | 2.40–2.50 | 1.00 | `6727171e78ef83c0` |
| 140 | one plan, batch of 9, on slices | 1.79 | 1.79–1.81 | 0.74 | `6727171e78ef83c0` |
| 140 | 9 executions of one transform | 5.42 | 5.41–5.43 | 2.25 | `f3dd4ab76d588fce` |
| 216 | one plan, batch of 9 | 5.66 | 5.64–5.77 | 1.00 | `a20b554f2f8a179a` |
| 216 | one plan, batch of 9, on slices | 5.01 | 5.00–5.01 | 0.89 | `a20b554f2f8a179a` |
| 216 | 9 executions of one transform | 8.56 | 8.55–8.64 | 1.51 | `d4d56766b9cffab0` |
| 336 | one plan, batch of 9 | 7.38 | 7.38–7.39 | 1.00 | `c456a5ff0c130312` |
| 336 | one plan, batch of 9, on slices | 6.79 | 6.78–6.82 | 0.92 | `c456a5ff0c130312` |
| 336 | 9 executions of one transform | 10.59 | 10.58–11.05 | 1.43 | `66b2c7e1f5e2488c` |
| 588 | one plan, batch of 9 | 14.37 | 14.37–14.38 | 1.00 | `459b214d17a88ced` |
| 588 | one plan, batch of 9, on slices | 13.73 | 13.71–13.92 | 0.96 | `459b214d17a88ced` |
| 588 | 9 executions of one transform | 17.73 | 17.72–17.76 | 1.23 | `8ceb37728c04337d` |
| 1080 | one plan, batch of 9 | 33.93 | 33.90–33.95 | 1.00 | `3b427c76cf876f60` |
| 1080 | one plan, batch of 9, on slices | 33.32 | 33.31–41.22 | 0.98 | `3b427c76cf876f60` |
| 1080 | 9 executions of one transform | 37.37 | 37.33–41.13 | 1.10 | `b59bd31ddf45712a` |

#### 2. Split against interleaved: c2c forward along axis 0 of the box [N, 9]

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | interleaved [N, 9] | 2.47 | 2.47–2.58 | 1.00 | `6727171e78ef83c0` |
| 140 | split [N, 9, 2] | 7.03 | 7.02–7.05 | 2.84 | `82d9e1065908da09` |
| 140 | copy of the planes into interleaved scratch, then interleaved | 2.51 | 2.51–2.53 | 1.01 | `6727171e78ef83c0` |
| 216 | interleaved [N, 9] | 5.39 | 5.38–5.40 | 1.00 | `a20b554f2f8a179a` |
| 216 | split [N, 9, 2] | 8.21 | 8.20–8.23 | 1.52 | `17b3c39a8dfbfdfb` |
| 216 | copy of the planes into interleaved scratch, then interleaved | 5.67 | 5.66–5.67 | 1.05 | `a20b554f2f8a179a` |
| 336 | interleaved [N, 9] | 7.48 | 7.48–7.49 | 1.00 | `c456a5ff0c130312` |
| 336 | split [N, 9, 2] | 13.21 | 13.20–13.26 | 1.77 | `6da0395ce984ecfd` |
| 336 | copy of the planes into interleaved scratch, then interleaved | 8.02 | 8.02–8.03 | 1.07 | `c456a5ff0c130312` |
| 588 | interleaved [N, 9] | 14.28 | 14.27–14.29 | 1.00 | `459b214d17a88ced` |
| 588 | split [N, 9, 2] | 30.82 | 30.79–30.87 | 2.16 | `695bdfc656f05e81` |
| 588 | copy of the planes into interleaved scratch, then interleaved | 15.44 | 15.43–16.37 | 1.08 | `459b214d17a88ced` |
| 1080 | interleaved [N, 9] | 34.09 | 34.05–34.14 | 1.00 | `3b427c76cf876f60` |
| 1080 | split [N, 9, 2] | 54.15 | 54.10–54.25 | 1.59 | `3472769e5ab4a905` |
| 1080 | copy of the planes into interleaved scratch, then interleaved | 36.42 | 36.35–36.74 | 1.07 | `3b427c76cf876f60` |

#### 3. Aligned against unaligned: c2c forward along axis 0 of [N, 9]

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | class 0 plan, class 0 buffers | 1.73 | 1.73–1.74 | 1.00 | `6727171e78ef83c0` |
| 140 | `.unaligned()` plan, class 0 buffers | 6.31 | 6.31–6.80 | 3.65 | `358e29e51123774d` |
| 140 | `.unaligned()` plan, class 16 buffers | 6.30 | 6.30–6.31 | 3.65 | `358e29e51123774d` |
| 140 | class 16 plan, class 16 buffers | 1.83 | 1.83–1.84 | 1.06 | `6727171e78ef83c0` |
| 216 | class 0 plan, class 0 buffers | 4.85 | 4.84–4.86 | 1.00 | `a20b554f2f8a179a` |
| 216 | `.unaligned()` plan, class 0 buffers | 10.18 | 10.17–10.19 | 2.10 | `376e9cb2f8f3cc50` |
| 216 | `.unaligned()` plan, class 16 buffers | 10.22 | 10.19–11.00 | 2.11 | `376e9cb2f8f3cc50` |
| 216 | class 16 plan, class 16 buffers | 5.02 | 5.00–5.04 | 1.03 | `a20b554f2f8a179a` |
| 336 | class 0 plan, class 0 buffers | 6.79 | 6.77–6.93 | 1.00 | `c456a5ff0c130312` |
| 336 | `.unaligned()` plan, class 0 buffers | 21.87 | 21.86–24.57 | 3.22 | `5aedf4800f4245c4` |
| 336 | `.unaligned()` plan, class 16 buffers | 21.87 | 21.84–22.75 | 3.22 | `5aedf4800f4245c4` |
| 336 | class 16 plan, class 16 buffers | 7.00 | 7.00–7.01 | 1.03 | `c456a5ff0c130312` |
| 588 | class 0 plan, class 0 buffers | 13.86 | 13.84–14.03 | 1.00 | `459b214d17a88ced` |
| 588 | `.unaligned()` plan, class 0 buffers | 29.73 | 29.67–31.96 | 2.15 | `695bdfc656f05e81` |
| 588 | `.unaligned()` plan, class 16 buffers | 29.72 | 29.69–29.85 | 2.14 | `695bdfc656f05e81` |
| 588 | class 16 plan, class 16 buffers | 14.58 | 14.57–14.61 | 1.05 | `459b214d17a88ced` |
| 1080 | class 0 plan, class 0 buffers | 33.28 | 33.25–33.35 | 1.00 | `3b427c76cf876f60` |
| 1080 | `.unaligned()` plan, class 0 buffers | 52.70 | 52.55–55.94 | 1.58 | `4a209f17710f7907` |
| 1080 | `.unaligned()` plan, class 16 buffers | 52.57 | 52.54–52.61 | 1.58 | `4a209f17710f7907` |
| 1080 | class 16 plan, class 16 buffers | 34.92 | 34.90–35.29 | 1.05 | `3b427c76cf876f60` |

#### 4a. r2c over both axes of a column-major [N, N]: the halved axis

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | axes [1, 0]: axis 0 (contiguous) halved | 44.12 | 44.09–44.19 | 1.00 | `09a580db9fa04821` |
| 140 | axes [0, 1]: axis 1 (strided) halved | 50.34 | 50.30–50.40 | 1.14 | `28d638f9b1c07bb4` |
| 216 | axes [1, 0]: axis 0 (contiguous) halved | 163.95 | 163.81–164.18 | 1.00 | `527a62eddc08bf32` |
| 216 | axes [0, 1]: axis 1 (strided) halved | 158.45 | 158.42–158.49 | 0.97 | `1e6c51655afd3cf8` |
| 336 | axes [1, 0]: axis 0 (contiguous) halved | 398.90 | 398.26–399.58 | 1.00 | `32f3cf52dd781b95` |
| 336 | axes [0, 1]: axis 1 (strided) halved | 413.03 | 412.54–413.89 | 1.04 | `f3a14097b56a6b57` |
| 588 | axes [1, 0]: axis 0 (contiguous) halved | 1311.22 | 1302.36–1332.09 | 1.00 | `25374b1de7c1ab13` |
| 588 | axes [0, 1]: axis 1 (strided) halved | 1500.10 | 1497.29–1503.29 | 1.14 | `8c3a585636d1e306` |
| 1080 | axes [1, 0]: axis 0 (contiguous) halved | 7054.70 | 6764.84–8003.55 | 1.00 | `7be8d55cf24cebc6` |
| 1080 | axes [0, 1]: axis 1 (strided) halved | 7509.93 | 6934.02–8108.54 | 1.06 | `ae2fb2c4eb55fc67` |

#### 4b. c2r over both axes of a column-major [N, N]: the halved axis

| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |
| ---: | --- | ---: | ---: | ---: | --- |
| 140 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 49.40 | 49.19–54.44 | 1.00 | `82381cd8d8d9a72a` |
| 140 | axes [0, 1]: axis 1 (strided) halved, with the copy | 74.03 | 73.95–79.72 | 1.50 | `46dd5248cfb67839` |
| 140 | the copy alone | 2.79 | 2.78–2.79 | 0.06 | — |
| 216 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 174.30 | 173.85–177.09 | 1.00 | `485d81395715f80b` |
| 216 | axes [0, 1]: axis 1 (strided) halved, with the copy | 179.85 | 179.73–180.78 | 1.03 | `b7bcd63753fc13d7` |
| 216 | the copy alone | 6.91 | 6.91–6.92 | 0.04 | — |
| 336 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 413.63 | 412.67–416.39 | 1.00 | `fbf819c8e738b694` |
| 336 | axes [0, 1]: axis 1 (strided) halved, with the copy | 546.66 | 544.98–596.28 | 1.32 | `13d8edca0311c2a3` |
| 336 | the copy alone | 18.79 | 18.78–18.81 | 0.05 | — |
| 588 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 1385.58 | 1380.80–1388.61 | 1.00 | `d200f931b79f31cd` |
| 588 | axes [0, 1]: axis 1 (strided) halved, with the copy | 2048.97 | 2046.40–2090.20 | 1.48 | `6402294383ea94a7` |
| 588 | the copy alone | 61.70 | 61.51–64.68 | 0.04 | — |
| 1080 | axes [1, 0]: axis 0 (contiguous) halved, with the copy | 6507.60 | 6393.19–6985.92 | 1.00 | `e567167e2db32e13` |
| 1080 | axes [0, 1]: axis 1 (strided) halved, with the copy | 7865.37 | 7372.72–8696.89 | 1.21 | `468c370785f6fdc7` |
| 1080 | the copy alone | 219.28 | 215.82–220.63 | 0.03 | — |
