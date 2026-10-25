//! The bounded MPI launcher of the parent process.
//!
//! Every case runs as `mpiexec -n <ranks> <this executable>` in a process group of its own.
//! When a run ends, by failure, by its deadline or by success, the launcher stops the whole
//! group: TERM, a bounded wait, then KILL. A launcher that exits early, a rank that ignores
//! TERM or a descendant that holds the output pipes therefore cannot outlive its run or block
//! the next one. Signals go to the group through the `kill` utility, so the harness needs no
//! dependency beyond the standard library.

use std::{
    env,
    ffi::OsString,
    fs,
    io::Read,
    os::unix::{fs::PermissionsExt, process::CommandExt},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

/// The environment variable that selects the case in a child.
pub const CASE_VARIABLE: &str = "RLST_MPI_CASE";

/// The environment variable that overrides the MPI launcher (default `mpiexec`).
pub const LAUNCHER_VARIABLE: &str = "RLST_MPIEXEC";

/// How long the group of a case may take to exit after TERM, and again after KILL.
const GRACE: Duration = Duration::from_secs(5);

/// How long to wait for the output pipes once the group is gone.
const COLLECTION: Duration = Duration::from_secs(5);

/// How a run ended.
pub enum Ended {
    /// The launcher exited with this status.
    Exited(ExitStatus),
    /// The run reached its deadline.
    TimedOut(Duration),
    /// The launcher could not be started.
    NotStarted(String),
}

/// The result of one bounded run.
pub struct Outcome {
    pub ended: Ended,
    /// Standard output followed by standard error.
    pub output: String,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        matches!(self.ended, Ended::Exited(status) if status.success())
    }

    pub fn describe(&self) -> String {
        match &self.ended {
            Ended::Exited(status) => format!("exited with {status}"),
            Ended::TimedOut(deadline) => format!("timeout after {} s", deadline.as_secs_f64()),
            Ended::NotStarted(error) => format!("could not start: {error}"),
        }
    }
}

/// The MPI launcher used for the cases.
pub struct Launcher {
    program: OsString,
    oversubscribe: bool,
}

impl Launcher {
    /// Use `$RLST_MPIEXEC`, or `mpiexec`, and ask it which MPI it belongs to.
    ///
    /// Open MPI refuses more ranks than cores unless given `--oversubscribe`, which other
    /// implementations such as MPICH reject.
    pub fn from_env() -> Result<Self, String> {
        let program = env::var_os(LAUNCHER_VARIABLE).unwrap_or_else(|| "mpiexec".into());
        let version = Command::new(&program)
            .arg("--version")
            .stdin(Stdio::null())
            .output()
            .map_err(|error| {
                format!(
                    "cannot run the MPI launcher `{}`: {error}; install MPI or set \
                     {LAUNCHER_VARIABLE} to its mpiexec",
                    program.to_string_lossy()
                )
            })?;
        let text =
            String::from_utf8_lossy(&version.stdout) + String::from_utf8_lossy(&version.stderr);
        let oversubscribe = text.contains("Open MPI") || text.contains("OpenRTE");
        Ok(Self {
            program,
            oversubscribe,
        })
    }

    /// Run `case` of `executable` on `ranks` ranks.
    pub fn run(&self, executable: &Path, ranks: usize, case: &str, deadline: Duration) -> Outcome {
        let mut command = Command::new(&self.program);
        if self.oversubscribe {
            command.arg("--oversubscribe");
        }
        command
            .arg("-n")
            .arg(ranks.to_string())
            .arg(executable)
            .env(CASE_VARIABLE, case)
            // Containerized runners often execute as root. Open MPI then requires both
            // variables as an explicit acknowledgement before launching ranks.
            .env("OMPI_ALLOW_RUN_AS_ROOT", "1")
            .env("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1");
        run_bounded(command, deadline, GRACE)
    }
}

/// Run `command` in a new process group, and stop the whole group when it ends.
///
/// The group gets `grace` to exit after TERM before it is killed.
pub fn run_bounded(mut command: Command, deadline: Duration, grace: Duration) -> Outcome {
    let start = Instant::now();
    let mut child = match command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return Outcome {
                ended: Ended::NotStarted(error.to_string()),
                output: String::new(),
            };
        }
    };
    // Read both pipes on threads, so that a full pipe cannot block the child. The threads
    // report through channels, so that a pipe held open by an escaped descendant cannot
    // block the parent either.
    let read = |mut pipe: Box<dyn Read + Send>| {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            let _ = sender.send(String::from_utf8_lossy(&bytes).into_owned());
        });
        receiver
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));

    let ended = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ended::Exited(status),
            Ok(None) if start.elapsed() > deadline => break Ended::TimedOut(deadline),
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(error) => break Ended::NotStarted(error.to_string()),
        }
    };
    // A launcher can report success before a detached descendant exits. The group belongs to
    // this run, so stop whatever is left of it in every case.
    stop_group(&mut child, grace);

    let mut output = String::new();
    for receiver in [stdout, stderr] {
        output += &receiver.recv_timeout(COLLECTION).unwrap_or_default();
    }
    Outcome { ended, output }
}

/// Send `signal` to the process group `group`; return whether any process received it.
fn signal_group(group: u32, signal: &str) -> bool {
    Command::new("kill")
        .args(["-s", signal, "--", &format!("-{group}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Wait at most `timeout` for the group of `child` to disappear, reaping `child` meanwhile.
fn wait_for_group(child: &mut Child, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        // An unreaped leader is a zombie that still belongs to the group.
        let _ = child.try_wait();
        if !signal_group(child.id(), "0") {
            return true;
        }
        if start.elapsed() > timeout {
            return false;
        }
        thread::sleep(Duration::from_millis(20));
    }
}

/// Stop the group of `child`: TERM, a bounded wait, then KILL.
fn stop_group(child: &mut Child, grace: Duration) {
    let _ = child.try_wait();
    if !signal_group(child.id(), "0") {
        return;
    }
    signal_group(child.id(), "TERM");
    if !wait_for_group(child, grace) {
        signal_group(child.id(), "KILL");
        wait_for_group(child, grace);
    }
    let _ = child.try_wait();
}

/// Whether process `pid` exists and is not a zombie.
///
/// A container without an init process may never reap a zombie, so `/proc` is consulted
/// where it exists. Elsewhere, such as on macOS, `launchd` reaps orphans promptly.
fn process_alive(pid: u32) -> bool {
    let zombie = fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        // The state follows the command name, which is in parentheses.
        stat.rsplit_once(')')
            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
    });
    !zombie
        && Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
}

/// Wait at most two seconds for `path` to hold a process id.
fn read_pid(path: &Path) -> Result<u32, String> {
    let start = Instant::now();
    loop {
        if let Some(pid) = fs::read_to_string(path)
            .ok()
            .and_then(|text| text.trim().parse().ok())
        {
            return Ok(pid);
        }
        if start.elapsed() > Duration::from_secs(2) {
            return Err(format!("the fixture did not write {}", path.display()));
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Check that process `pid` disappears within two seconds.
fn assert_stopped(pid: u32, scenario: &str) -> Result<(), String> {
    let start = Instant::now();
    while process_alive(pid) {
        if start.elapsed() > Duration::from_secs(2) {
            return Err(format!("{scenario}: process {pid} survived the cleanup"));
        }
        thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}

/// A fake launcher: a shell script that starts a TERM-ignoring child, writes its pid to
/// `marker`, and then runs `tail`.
fn write_fixture(directory: &Path, name: &str, tail: &str) -> (PathBuf, PathBuf) {
    let script = directory.join(name);
    let marker = directory.join(format!("{name}.pid"));
    fs::write(
        &script,
        format!(
            "#!/bin/sh\n\
             sh -c 'trap \"\" TERM; exec sleep 30' </dev/null >/dev/null 2>&1 &\n\
             echo $! > '{marker}.tmp' && mv '{marker}.tmp' '{marker}'\n\
             {tail}\n",
            marker = marker.display()
        ),
    )
    .expect("the fixture script");
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).expect("an executable fixture");
    (script, marker)
}

/// Check the launcher's cleanup with fake launchers that misbehave.
///
/// Ported from the runner test `scripts/test_run_mpi_tests.py`: a launcher that fails, one
/// that hangs, and one that succeeds must each leave no process of their group behind, and
/// the cleanup must not touch processes outside the group.
pub fn self_test() -> Result<(), String> {
    let directory = env::temp_dir().join(format!("rlst-mpi-launcher-{}", std::process::id()));
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let mut unrelated = Command::new("sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .map_err(|error| error.to_string())?;

    let result = (|| {
        let deadline = Duration::from_millis(500);
        let grace = Duration::from_millis(200);

        // The launcher fails and leaves a TERM-ignoring child with redirected output. This
        // proves that the cleanup follows the group rather than the leader or the pipes.
        let (script, marker) = write_fixture(&directory, "failure", "exit 7");
        let outcome = run_bounded(Command::new(&script), deadline, grace);
        if outcome.passed() || !matches!(outcome.ended, Ended::Exited(_)) {
            return Err(format!(
                "a failing launcher was reported as: {}",
                outcome.describe()
            ));
        }
        assert_stopped(read_pid(&marker)?, "failing launcher")?;

        // The launcher hangs: the run times out and the group is killed.
        let (script, marker) = write_fixture(&directory, "hang", "exec sleep 30");
        let outcome = run_bounded(Command::new(&script), deadline, grace);
        if !matches!(outcome.ended, Ended::TimedOut(_)) {
            return Err(format!(
                "a hanging launcher was reported as: {}",
                outcome.describe()
            ));
        }
        assert_stopped(read_pid(&marker)?, "hanging launcher")?;

        // The launcher succeeds but leaves a child behind, which is stopped as well.
        let (script, marker) = write_fixture(&directory, "success", "exit 0");
        let outcome = run_bounded(Command::new(&script), deadline, grace);
        if !outcome.passed() {
            return Err(format!(
                "a succeeding launcher was reported as: {}",
                outcome.describe()
            ));
        }
        assert_stopped(read_pid(&marker)?, "succeeding launcher")?;

        // A launcher that does not exist is a failure, not a pass.
        let outcome = run_bounded(Command::new(directory.join("missing")), deadline, grace);
        if !matches!(outcome.ended, Ended::NotStarted(_)) {
            return Err(format!(
                "a missing launcher was reported as: {}",
                outcome.describe()
            ));
        }

        if !process_alive(unrelated.id()) {
            return Err("the cleanup stopped an unrelated process".into());
        }
        Ok(())
    })();

    let _ = unrelated.kill();
    let _ = unrelated.wait();
    let _ = fs::remove_dir_all(&directory);
    result
}
