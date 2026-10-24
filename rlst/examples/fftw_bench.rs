//! Benchmarks of RLST's FFTW plans (task F8b of the FFTW rework, design Section 7.7).
//!
//! Run with
//!
//! ```text
//! cargo run --release --example fftw_bench --features fftw_system
//! ```
//!
//! Four comparisons, in double precision, at nd-fmm's sizes N_φ = 140, 216, 336, 588 and 1080:
//!
//! 1. one plan over the box `[N_φ, 9]` along axis 0, against a plan of one transform executed
//!    on each of the nine columns; the batched plan also runs on the arrays' data as slices,
//!    which shows the cost of checking arrays at execution;
//! 2. the split box `[N_φ, 9, 2]` with the part axis outermost, against the interleaved box
//!    `[N_φ, 9]`, and against copying the split planes into interleaved scratch followed by
//!    the interleaved plan;
//! 3. a plan at alignment class 0 on `FFTW_ALIGNED` buffers, against an `.unaligned()` plan on
//!    the same buffers and on buffers at class 16, and a plan for class 16 on those;
//! 4. real-to-complex over both axes of a column-major `[N_φ, N_φ]` array halving axis 0, the
//!    contiguous one (`axes = [1, 0]`), against halving axis 1 (`axes = [0, 1]`), and the
//!    same for complex-to-real.
//!
//! Every plan uses `Rigor::Estimate`, nd-fmm's rigor, and no wisdom is imported, so the plans
//! are those any process with this FFTW build makes. Each case warms up, then times 7 rounds
//! of as many executions as take at least 0.2 s; the tables give the median time per
//! execution and the range of the rounds, and each plan's fingerprint. Timings are
//! indicative: they vary with the machine, its load and the FFTW build. Results print as
//! Markdown tables.

extern crate blas_src;
extern crate lapack_src;

use std::time::{Duration, Instant};

use rlst::fftw::{AlignmentClass, Direction, Layout, Planner};
use rlst::{c64, rlst_dynamic_array};

type Error = Box<dyn std::error::Error>;

/// nd-fmm's sizes N_φ.
const SIZES: [usize; 5] = [140, 216, 336, 588, 1080];
/// Rings per box, as in the determinism binary's units.
const RINGS: usize = 9;
/// Timed rounds per case.
const ROUNDS: usize = 7;
/// The least duration of a round.
const ROUND_TIME: Duration = Duration::from_millis(200);

/// Seconds per execution: the median and the range of the rounds.
#[derive(Clone, Copy)]
struct Timing {
    median: f64,
    min: f64,
    max: f64,
}

/// Times `run`: doubles the executions per round until a round takes at least
/// [`ROUND_TIME`], which warms up, then times [`ROUNDS`] rounds of that many executions.
fn time(mut run: impl FnMut()) -> Timing {
    let mut reps = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            run();
        }
        if start.elapsed() >= ROUND_TIME {
            break;
        }
        reps *= 2;
    }
    let mut rounds: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..reps {
                run();
            }
            start.elapsed().as_secs_f64() / reps as f64
        })
        .collect();
    rounds.sort_by(f64::total_cmp);
    Timing {
        median: rounds[ROUNDS / 2],
        min: rounds[0],
        max: rounds[ROUNDS - 1],
    }
}

/// Prints the header of a table whose rows [`row`] prints.
fn header(title: &str) {
    println!("\n### {title}\n");
    println!("| N | Case | Median (µs) | Range (µs) | Relative to first case | Fingerprint |");
    println!("| ---: | --- | ---: | ---: | ---: | --- |");
}

/// Prints one case of a table, relative to the first case of its size, `baseline`.
fn row(n: usize, case: &str, timing: Timing, baseline: Timing, fingerprint: Option<u64>) {
    let fingerprint = fingerprint.map_or_else(|| "—".to_string(), |f| format!("`{f:016x}`"));
    println!(
        "| {n} | {case} | {:.2} | {:.2}–{:.2} | {:.2} | {fingerprint} |",
        timing.median * 1e6,
        timing.min * 1e6,
        timing.max * 1e6,
        timing.median / baseline.median,
    );
}

/// The CPU's name, if the operating system reports it.
fn cpu_name() -> String {
    let name = if cfg!(target_os = "macos") {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(str::to_string)
            })
    };
    name.map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Comparison 1: one batched plan against a loop of single transforms.
fn batched_against_loop() -> Result<(), Error> {
    header("1. Batched against a loop: c2c forward along axis 0 of [N, 9]");
    for n in SIZES {
        let mut input = rlst_dynamic_array!(c64, [n, RINGS] | rlst::FFTW_ALIGNED);
        let mut output = rlst_dynamic_array!(c64, [n, RINGS] | rlst::FFTW_ALIGNED);
        input.fill_from_seed_equally_distributed(0);

        let batched = Planner::<f64>::new()
            .c2c(&Layout::along_axis(&input, &output, 0)?, Direction::Forward)?;
        // The columns are n complex doubles apart, a multiple of 64 bytes for every size, so
        // each is at class 0, the class of the plan.
        let single = Planner::<f64>::new().c2c(
            &Layout::along_axis(&input.r().slice::<1>(1, 0), &output.r().slice::<1>(1, 0), 0)?,
            Direction::Forward,
        )?;

        let base = time(|| batched.execute(&input, &mut output).expect("batched"));
        // The same plan on the arrays' data as slices, which execution checks by length only:
        // the difference is the cost of checking the arrays' shapes and strides.
        let on_slices = time(|| {
            batched
                .execute(
                    input.data().expect("contiguous"),
                    output.data_mut().expect("contiguous"),
                )
                .expect("batched on slices")
        });
        let looped = time(|| {
            for k in 0..RINGS {
                single
                    .execute(
                        &input.r().slice::<1>(1, k),
                        &mut output.r_mut().slice::<1>(1, k),
                    )
                    .expect("single");
            }
        });
        row(
            n,
            "one plan, batch of 9",
            base,
            base,
            Some(batched.fingerprint()),
        );
        row(
            n,
            "one plan, batch of 9, on slices",
            on_slices,
            base,
            Some(batched.fingerprint()),
        );
        row(
            n,
            "9 executions of one transform",
            looped,
            base,
            Some(single.fingerprint()),
        );
    }
    Ok(())
}

/// Comparison 2: split storage against interleaved storage and an interleaved copy.
fn split_against_interleaved() -> Result<(), Error> {
    header("2. Split against interleaved: c2c forward along axis 0 of the box [N, 9]");
    for n in SIZES {
        let mut input = rlst_dynamic_array!(c64, [n, RINGS] | rlst::FFTW_ALIGNED);
        let mut output = rlst_dynamic_array!(c64, [n, RINGS] | rlst::FFTW_ALIGNED);
        let mut scratch = rlst_dynamic_array!(c64, [n, RINGS] | rlst::FFTW_ALIGNED);
        input.fill_from_seed_equally_distributed(0);
        // nd-fmm's planes: the real plane, then the imaginary plane, the part axis outermost.
        let mut split_input = rlst_dynamic_array!(f64, [n, RINGS, 2] | rlst::FFTW_ALIGNED);
        let mut split_output = rlst_dynamic_array!(f64, [n, RINGS, 2] | rlst::FFTW_ALIGNED);
        split_input.fill_from_seed_equally_distributed(0);

        let interleaved = Planner::<f64>::new()
            .c2c(&Layout::along_axis(&input, &output, 0)?, Direction::Forward)?;
        let split = Planner::<f64>::new().split_c2c(
            &Layout::split_along_axis(&split_input, 2, &split_output, 2, 0)?,
            Direction::Forward,
        )?;

        let base = time(|| {
            interleaved
                .execute(&input, &mut output)
                .expect("interleaved")
        });
        let split_time = time(|| {
            split
                .execute(&split_input, &mut split_output)
                .expect("split")
        });
        let planes = split_input.data().expect("contiguous");
        let (re, im) = planes.split_at(n * RINGS);
        let copy_time = time(|| {
            let scratch = scratch.data_mut().expect("contiguous");
            for ((z, &x), &y) in scratch.iter_mut().zip(re).zip(im) {
                *z = c64::new(x, y);
            }
            interleaved
                .execute(&*scratch, &mut output)
                .expect("interleaved after copy");
        });
        row(
            n,
            "interleaved [N, 9]",
            base,
            base,
            Some(interleaved.fingerprint()),
        );
        row(
            n,
            "split [N, 9, 2]",
            split_time,
            base,
            Some(split.fingerprint()),
        );
        row(
            n,
            "copy of the planes into interleaved scratch, then interleaved",
            copy_time,
            base,
            Some(interleaved.fingerprint()),
        );
    }
    Ok(())
}

/// Comparison 3: aligned plans against unaligned ones.
fn aligned_against_unaligned() -> Result<(), Error> {
    header("3. Aligned against unaligned: c2c forward along axis 0 of [N, 9]");
    for n in SIZES {
        let len = n * RINGS;
        // One complex value more than the box, so that the box also fits at an offset of 16
        // bytes; the buffers are slices of these, at offset 0 (class 0) or 1 (class 16).
        let mut input = rlst_dynamic_array!(c64, [len + 1] | rlst::FFTW_ALIGNED);
        let mut output = rlst_dynamic_array!(c64, [len + 1] | rlst::FFTW_ALIGNED);
        input.fill_from_seed_equally_distributed(0);
        let shape_in = rlst_dynamic_array!(c64, [n, RINGS]);
        let layout = Layout::along_axis(&shape_in, &shape_in, 0)?;

        let class_16 = AlignmentClass::from_offset(16);
        assert_eq!(
            AlignmentClass::of(&input.data().expect("contiguous")[1..]),
            class_16
        );
        let aligned = Planner::<f64>::new().c2c(&layout, Direction::Forward)?;
        let unaligned = Planner::<f64>::new()
            .unaligned()
            .c2c(&layout, Direction::Forward)?;
        let at_16 = Planner::<f64>::new()
            .alignment(class_16, class_16)
            .c2c(&layout, Direction::Forward)?;

        let mut run = |plan: &rlst::fftw::Plan<f64, rlst::fftw::C2c, rlst::fftw::Preserving>,
                       offset: usize| {
            let input = &input.data().expect("contiguous")[offset..offset + len];
            let output = &mut output.data_mut().expect("contiguous")[offset..offset + len];
            time(|| plan.execute(input, output).expect("execute"))
        };
        let base = run(&aligned, 0);
        let unaligned_0 = run(&unaligned, 0);
        let unaligned_16 = run(&unaligned, 1);
        let aligned_16 = run(&at_16, 1);
        row(
            n,
            "class 0 plan, class 0 buffers",
            base,
            base,
            Some(aligned.fingerprint()),
        );
        row(
            n,
            "`.unaligned()` plan, class 0 buffers",
            unaligned_0,
            base,
            Some(unaligned.fingerprint()),
        );
        row(
            n,
            "`.unaligned()` plan, class 16 buffers",
            unaligned_16,
            base,
            Some(unaligned.fingerprint()),
        );
        row(
            n,
            "class 16 plan, class 16 buffers",
            aligned_16,
            base,
            Some(at_16.fingerprint()),
        );
    }
    Ok(())
}

/// Comparison 4: halving the contiguous axis against halving a strided one (DC2).
fn halving_contiguous_against_strided() -> Result<(), Error> {
    header("4a. r2c over both axes of a column-major [N, N]: the halved axis");
    for n in SIZES {
        let h = n / 2 + 1;
        let mut real = rlst_dynamic_array!(f64, [n, n] | rlst::FFTW_ALIGNED);
        real.fill_from_seed_equally_distributed(0);
        // Halving axis 0, the contiguous one: the complex array is [n / 2 + 1, n].
        let mut complex_0 = rlst_dynamic_array!(c64, [h, n] | rlst::FFTW_ALIGNED);
        // Halving axis 1, a strided one: the complex array is [n, n / 2 + 1].
        let mut complex_1 = rlst_dynamic_array!(c64, [n, h] | rlst::FFTW_ALIGNED);

        let contiguous =
            Planner::<f64>::new().r2c(&Layout::over_axes(&real, &complex_0, &[1, 0])?)?;
        let strided = Planner::<f64>::new().r2c(&Layout::over_axes(&real, &complex_1, &[0, 1])?)?;

        let base = time(|| contiguous.execute(&real, &mut complex_0).expect("r2c"));
        let other = time(|| strided.execute(&real, &mut complex_1).expect("r2c"));
        row(
            n,
            "axes [1, 0]: axis 0 (contiguous) halved",
            base,
            base,
            Some(contiguous.fingerprint()),
        );
        row(
            n,
            "axes [0, 1]: axis 1 (strided) halved",
            other,
            base,
            Some(strided.fingerprint()),
        );
    }

    // A c2r plan destroys its input, and `preserve_input` is not available over two
    // dimensions, so every execution first restores the spectrum from a copy; the copy alone
    // is timed too, and is the same for both orders.
    header("4b. c2r over both axes of a column-major [N, N]: the halved axis");
    for n in SIZES {
        let h = n / 2 + 1;
        let mut real = rlst_dynamic_array!(f64, [n, n] | rlst::FFTW_ALIGNED);
        let mut spectrum_0 = rlst_dynamic_array!(c64, [h, n] | rlst::FFTW_ALIGNED);
        let mut spectrum_1 = rlst_dynamic_array!(c64, [n, h] | rlst::FFTW_ALIGNED);
        let mut saved = rlst_dynamic_array!(c64, [h * n] | rlst::FFTW_ALIGNED);
        saved.fill_from_seed_equally_distributed(0);
        let saved = saved.data().expect("contiguous");

        let contiguous =
            Planner::<f64>::new().c2r(&Layout::over_axes(&spectrum_0, &real, &[1, 0])?)?;
        let strided =
            Planner::<f64>::new().c2r(&Layout::over_axes(&spectrum_1, &real, &[0, 1])?)?;

        let base = time(|| {
            spectrum_0
                .data_mut()
                .expect("contiguous")
                .copy_from_slice(saved);
            contiguous.execute(&mut spectrum_0, &mut real).expect("c2r");
        });
        let other = time(|| {
            spectrum_1
                .data_mut()
                .expect("contiguous")
                .copy_from_slice(saved);
            strided.execute(&mut spectrum_1, &mut real).expect("c2r");
        });
        let copy = time(|| {
            spectrum_0
                .data_mut()
                .expect("contiguous")
                .copy_from_slice(saved);
        });
        row(
            n,
            "axes [1, 0]: axis 0 (contiguous) halved, with the copy",
            base,
            base,
            Some(contiguous.fingerprint()),
        );
        row(
            n,
            "axes [0, 1]: axis 1 (strided) halved, with the copy",
            other,
            base,
            Some(strided.fingerprint()),
        );
        row(n, "the copy alone", copy, base, None);
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: built without optimisations; the timings mean little. Run with \
             `cargo run --release --example fftw_bench --features fftw_system`."
        );
    }
    println!("## FFTW benchmarks\n");
    println!("- FFTW: `{}`", rlst::fftw::version());
    println!("- CPU: {}", cpu_name());
    println!(
        "- Target: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "- Method: `Estimate` plans; per case a warm-up, then {ROUNDS} rounds of at least \
         {} ms; median and range of the time per execution",
        ROUND_TIME.as_millis()
    );

    batched_against_loop()?;
    split_against_interleaved()?;
    aligned_against_unaligned()?;
    halving_contiguous_against_strided()?;
    Ok(())
}
