//! The checks of plan execution allocate nothing on the heap (issue #44, design Section 4.8).
//!
//! Executing a plan checks each buffer against the plan's layout before FFTW is called. These
//! checks run on every execution, so for small transforms an allocation in them is a large
//! part of the time. This executable counts the allocations of the global allocator during
//! successful executions on arrays, views and slices of every family of buffers, and fails if
//! any execution allocates. A failed check may allocate for its error message.
//!
//! It has `harness = false`, because the allocator is global to the process: the test harness
//! allocates on its own threads while tests run. Every plan is `Rigor::Estimate`, the default.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use rlst::fftw::{Direction, Layout, Planner};
use rlst::{c64, rlst_dynamic_array};

/// The system allocator, counting its allocations.
struct Counting;

/// The number of allocations and reallocations so far.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System` with the caller's arguments, so `Counting` keeps
// the contract of `System`.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's guarantees for `alloc` hold for `System.alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's guarantees for `alloc_zeroed` hold for `System.alloc_zeroed`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's guarantees for `realloc` hold for `System.realloc`, which
        // allocated `ptr`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Executions counted per case, after one execution to warm up.
const EXECUTIONS: usize = 10;

/// The allocations per execution of `run`, which executes a plan once.
fn allocations(mut run: impl FnMut()) -> f64 {
    run();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..EXECUTIONS {
        run();
    }
    (ALLOCATIONS.load(Ordering::Relaxed) - before) as f64 / EXECUTIONS as f64
}

/// The cases: a name and the allocations per execution.
fn cases() -> Vec<(&'static str, f64)> {
    let mut results = Vec::new();

    // Complex-to-complex along axis 0 of the box [16, 9], as nd-fmm executes it.
    let mut input = rlst_dynamic_array!(c64, [16, 9] | rlst::FFTW_ALIGNED);
    let mut output = rlst_dynamic_array!(c64, [16, 9] | rlst::FFTW_ALIGNED);
    input.fill_from_seed_equally_distributed(0);
    let layout = Layout::along_axis(&input, &output, 0).unwrap();
    let plan = Planner::<f64>::new()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    results.push((
        "c2c, preserving, arrays",
        allocations(|| plan.execute(&input, &mut output).unwrap()),
    ));
    results.push((
        "c2c, preserving, slices",
        allocations(|| {
            plan.execute(input.data().unwrap(), output.data_mut().unwrap())
                .unwrap()
        }),
    ));
    results.push((
        "c2c, preserving, views",
        allocations(|| plan.execute(&input.r(), &mut output.r_mut()).unwrap()),
    ));

    let destroying = Planner::<f64>::new()
        .may_destroy_input()
        .c2c(&layout, Direction::Forward)
        .unwrap();
    results.push((
        "c2c, destroying, arrays",
        allocations(|| destroying.execute(&mut input, &mut output).unwrap()),
    ));

    let inplace_layout = Layout::along_axis_inplace(&input, 0).unwrap();
    let inplace = Planner::<f64>::new()
        .c2c_inplace(&inplace_layout, Direction::Forward)
        .unwrap();
    results.push((
        "c2c, in place, array",
        allocations(|| inplace.execute(&mut input).unwrap()),
    ));

    // A plan over two columns, executed on subviews of two columns.
    let two_layout = Layout::along_axis(
        &input.r().into_subview([0, 0], [16, 2]),
        &output.r().into_subview([0, 0], [16, 2]),
        0,
    )
    .unwrap();
    let two = Planner::<f64>::new()
        .c2c(&two_layout, Direction::Forward)
        .unwrap();
    results.push((
        "c2c, preserving, subviews",
        allocations(|| {
            two.execute(
                &input.r().into_subview([0, 4], [16, 2]),
                &mut output.r_mut().into_subview([0, 4], [16, 2]),
            )
            .unwrap()
        }),
    ));

    // Real-to-complex and complex-to-real along axis 0, which is halved.
    let mut real = rlst_dynamic_array!(f64, [16, 9] | rlst::FFTW_ALIGNED);
    let mut complex = rlst_dynamic_array!(c64, [9, 9] | rlst::FFTW_ALIGNED);
    real.fill_from_seed_equally_distributed(1);
    let r2c = Planner::<f64>::new()
        .r2c(&Layout::along_axis(&real, &complex, 0).unwrap())
        .unwrap();
    results.push((
        "r2c, arrays",
        allocations(|| r2c.execute(&real, &mut complex).unwrap()),
    ));
    let c2r = Planner::<f64>::new()
        .may_destroy_input()
        .c2r(&Layout::along_axis(&complex, &real, 0).unwrap())
        .unwrap();
    results.push((
        "c2r, destroying, arrays",
        allocations(|| c2r.execute(&mut complex, &mut real).unwrap()),
    ));

    // Split storage: the planes [16, 9] of real and imaginary parts, part axis 2.
    let mut split_in = rlst_dynamic_array!(f64, [16, 9, 2] | rlst::FFTW_ALIGNED);
    let mut split_out = rlst_dynamic_array!(f64, [16, 9, 2] | rlst::FFTW_ALIGNED);
    split_in.fill_from_seed_equally_distributed(2);
    let split = Planner::<f64>::new()
        .split_c2c(
            &Layout::split_along_axis(&split_in, 2, &split_out, 2, 0).unwrap(),
            Direction::Forward,
        )
        .unwrap();
    results.push((
        "split c2c, preserving, arrays",
        allocations(|| split.execute(&split_in, &mut split_out).unwrap()),
    ));

    // Split real-to-complex from the real [16, 9] into the planes [9, 9], part axis 2.
    let mut split_complex = rlst_dynamic_array!(f64, [9, 9, 2] | rlst::FFTW_ALIGNED);
    let split_r2c = Planner::<f64>::new()
        .split_r2c(&Layout::split_r2c_along_axis(&real, &split_complex, 2, 0).unwrap())
        .unwrap();
    results.push((
        "split r2c, arrays",
        allocations(|| split_r2c.execute(&real, &mut split_complex).unwrap()),
    ));

    results
}

fn main() -> ExitCode {
    let mut failed = false;
    for (name, count) in cases() {
        let verdict = if count == 0.0 {
            "ok"
        } else {
            failed = true;
            "FAILED"
        };
        println!("{name}: {count} allocations per execution ... {verdict}");
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
