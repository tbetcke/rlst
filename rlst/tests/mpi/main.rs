//! Multi-rank MPI tests.
//!
//! This executable has `harness = false`. Run without `RLST_MPI_CASE`, as `cargo test` runs
//! it, it is the parent: it checks its own launcher ([`launcher::self_test`]), then runs every
//! case on 1, 2, 3 and 4 ranks, each as `mpiexec -n <ranks> <this executable>` with the case
//! named in `RLST_MPI_CASE`, in a bounded process group of its own. A panic or deadlock in one
//! case therefore cannot prevent the others from being reported. `RLST_MPIEXEC` overrides the
//! launcher.
//!
//! Arguments that do not start with `-` select the cases whose names contain one of them, for
//! example `cargo test --features mpi --test mpi -- csr ghost`; `--list` lists the cases.
//!
//! Each case is a function of the world communicator that every rank calls. A case checks its
//! results with [`common::collective_assert`], so that every rank fails together, and computes
//! expected values independently of the collectives it tests wherever possible.

mod arrays;
mod common;
mod empty_ranks;
mod launcher;
mod layouts;
mod operators;
mod sparse;

use std::{
    env,
    process::ExitCode,
    time::{Duration, Instant},
};

use common::Case;
use launcher::{CASE_VARIABLE, Launcher};

/// The rank counts every case runs on. Three ranks give uneven splits of even sizes.
const RANKS: [usize; 4] = [1, 2, 3, 4];

/// The time a single run of a case may take before its process group is stopped.
const DEADLINE: Duration = Duration::from_secs(30);

fn cases() -> impl Iterator<Item = &'static Case> {
    [
        empty_ranks::CASES,
        layouts::CASES,
        arrays::CASES,
        sparse::CASES,
        operators::CASES,
    ]
    .into_iter()
    .flatten()
}

/// Run `case` on this rank.
fn child(case: &str) -> ExitCode {
    let (_, run) = cases()
        .find(|(name, _)| *name == case)
        .unwrap_or_else(|| panic!("unknown {CASE_VARIABLE}={case}"));
    let universe = mpi::initialize().expect("MPI must initialize once per executable");
    run(&universe.world());
    ExitCode::SUCCESS
}

/// Run the selected cases on every rank count and report each run.
fn parent() -> ExitCode {
    let start = Instant::now();
    let arguments: Vec<String> = env::args().skip(1).collect();
    let filters: Vec<&String> = arguments.iter().filter(|a| !a.starts_with('-')).collect();
    let selected: Vec<&Case> = cases()
        .filter(|(name, _)| filters.is_empty() || filters.iter().any(|f| name.contains(f.as_str())))
        .collect();

    if arguments.iter().any(|argument| argument == "--list") {
        for (name, _) in &selected {
            println!("{name}: test");
        }
        return ExitCode::SUCCESS;
    }
    // `cargo test <filter>` passes its filter to every test executable.
    if selected.is_empty() {
        println!("mpi: no case matches {filters:?}");
        return ExitCode::SUCCESS;
    }

    if let Err(failure) = launcher::self_test() {
        eprintln!("mpi: the launcher self-test failed: {failure}");
        return ExitCode::FAILURE;
    }
    println!("launcher self-test: ok");
    let launcher = match Launcher::from_env() {
        Ok(launcher) => launcher,
        Err(failure) => {
            eprintln!("mpi: {failure}");
            return ExitCode::FAILURE;
        }
    };
    let executable = env::current_exe().expect("the path of this executable");

    let mut failures = Vec::new();
    for (name, _) in &selected {
        for ranks in RANKS {
            let run_start = Instant::now();
            let outcome = launcher.run(&executable, ranks, name, DEADLINE);
            let elapsed = run_start.elapsed().as_secs_f64();
            if outcome.passed() {
                println!("[passed] ranks={ranks} case={name} ({elapsed:.2} s)");
            } else {
                println!(
                    "[FAILED] ranks={ranks} case={name} ({elapsed:.2} s): {}",
                    outcome.describe()
                );
                for line in outcome.output.lines() {
                    println!("    {line}");
                }
                failures.push(format!("ranks={ranks} case={name}"));
            }
        }
    }
    println!(
        "mpi: {} cases on {RANKS:?} ranks, {} runs failed, {:.2} s",
        selected.len(),
        failures.len(),
        start.elapsed().as_secs_f64()
    );
    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        eprintln!("MPI failures: {}", failures.join(", "));
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    match env::var(CASE_VARIABLE) {
        Ok(case) => child(&case),
        Err(_) => parent(),
    }
}
