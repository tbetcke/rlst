#!/usr/bin/env python3
"""Controlled subprocess regressions for MPI runner process-group cleanup."""

import importlib.util
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

RUNNER_PATH = Path(__file__).with_name("run_mpi_tests.py")
spec = importlib.util.spec_from_file_location("run_mpi_tests", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load MPI runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
vars(runner).update(
    TIMEOUT_SECONDS=0.5,
    TERM_GRACE_SECONDS=0.1,
    KILL_GRACE_SECONDS=0.1,
    FINAL_COLLECTION_SECONDS=0.1,
)


def write_python_fixture(
    path: Path, marker: Path, exit_code: int, hang: bool, output: str = ""
) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, subprocess, sys, time\n"
        f"sys.stdout.write({output!r}); sys.stdout.flush()\n"
        "child = subprocess.Popen(\n"
        "    [\"sh\", \"-c\", \"trap '' TERM; exec sleep 30\"],\n"
        "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"open({str(marker)!r}, \"w\").write(str(child.pid))\n"
        + ("time.sleep(30)\n" if hang else "")
        + f"os._exit({exit_code})\n"
    )
    path.chmod(0o755)


def wait_for_path(path: Path) -> int:
    deadline = time.monotonic() + 2
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"child did not write {path}")
        time.sleep(0.01)
    return int(path.read_text())


def process_is_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists() and stat_path.read_text().split()[2] == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def assert_stopped(pid: int, description: str) -> None:
    deadline = time.monotonic() + 2
    while process_is_alive(pid):
        if time.monotonic() >= deadline:
            raise AssertionError(f"{description}: child {pid} survived cleanup")
        time.sleep(0.01)


def term_ignoring_child(marker: Path) -> str:
    return (
        "sh -c 'trap \"\" TERM; exec sleep 30' "
        f"</dev/null >/dev/null 2>&1 & child=$!; echo $child > '{marker}'\n"
        f"while [ ! -f '{marker}' ]; do sleep 0.01; done\n"
    )


with tempfile.TemporaryDirectory() as temporary_directory:
    temporary = Path(temporary_directory)
    original_path = os.environ.get("PATH", "")
    unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        # The launcher exits and its child ignores TERM with redirected output.
        # This specifically proves KILL is based on group liveness, not pipe EOF.
        marker = temporary / "leader-exit.pid"
        process = subprocess.Popen(
            ["sh", "-c", term_ignoring_child(marker) + "exit 1"],
            start_new_session=True,
        )
        child_pid = wait_for_path(marker)
        if process.wait(timeout=2) != 1:
            raise AssertionError("leader did not exit with its expected failure")
        runner.stop_process_group(process)
        assert_stopped(child_pid, "leader-exit cleanup")

        # An ordinary nonzero MPI launcher result must also clean its group.
        marker = temporary / "mpi-failure.pid"
        write_python_fixture(temporary / "mpiexec", marker, exit_code=7, hang=False)
        os.environ["PATH"] = f"{temporary}:{original_path}"
        selected_mpiexec = subprocess.check_output(
            ["which", "mpiexec"], text=True, env=os.environ
        ).strip()
        if selected_mpiexec != str(temporary / "mpiexec"):
            raise AssertionError(f"fixture mpiexec was not selected: {selected_mpiexec}")
        success, output = runner.run_bounded("unused-executable", 1, "fixture")
        if not marker.exists():
            raise AssertionError(f"ordinary MPI fixture did not start: {output!r}")
        if success or "timeout after" in output:
            raise AssertionError("ordinary MPI failure was not reported as a failure")
        assert_stopped(wait_for_path(marker), "ordinary MPI failure cleanup")

        # A hung launcher must kill its TERM-ignoring redirected-output child.
        marker = temporary / "timeout.pid"
        write_python_fixture(temporary / "mpiexec", marker, exit_code=0, hang=True)
        success, output = runner.run_bounded("unused-executable", 1, "fixture")
        if success or "timeout after" not in output:
            raise AssertionError("timeout was not reported as a timeout")
        assert_stopped(wait_for_path(marker), "timeout cleanup")

        # An expected-abort case passes only for a non-zero exit that prints its
        # marker, and never for a timeout, success, or an unrelated failure.
        marker = temporary / "abort.pid"
        write_python_fixture(
            temporary / "mpiexec", marker, exit_code=1, hang=False, output="abort marker"
        )
        success, output = runner.run_expected_abort(
            "unused-executable", 1, "fixture", "abort marker"
        )
        if not success:
            raise AssertionError(f"expected abort was not accepted: {output!r}")
        assert_stopped(wait_for_path(marker), "expected abort cleanup")
        success, output = runner.run_expected_abort(
            "unused-executable", 1, "fixture", "other marker"
        )
        if success:
            raise AssertionError("abort without its marker was accepted")
        marker = temporary / "abort-success.pid"
        write_python_fixture(
            temporary / "mpiexec", marker, exit_code=0, hang=False, output="abort marker"
        )
        success, output = runner.run_expected_abort(
            "unused-executable", 1, "fixture", "abort marker"
        )
        if success:
            raise AssertionError("successful exit was accepted as an abort")
        marker = temporary / "abort-timeout.pid"
        write_python_fixture(
            temporary / "mpiexec", marker, exit_code=1, hang=True, output="abort marker"
        )
        success, output = runner.run_expected_abort(
            "unused-executable", 1, "fixture", "abort marker"
        )
        if success or "timeout after" not in output:
            raise AssertionError("timeout was accepted as an abort")
        assert_stopped(wait_for_path(marker), "expected abort timeout cleanup")

        # A failed Cargo build owns the same kind of group and requires cleanup.
        marker = temporary / "build-failure.pid"
        write_python_fixture(temporary / "cargo", marker, exit_code=7, hang=False)
        try:
            runner.build_executable()
        except RuntimeError as error:
            if "build failed (7)" not in str(error):
                raise
        else:
            raise AssertionError("build failure was reported as success")
        assert_stopped(wait_for_path(marker), "build failure cleanup")

        if not process_is_alive(unrelated.pid):
            raise AssertionError("cleanup killed an unrelated process")
    finally:
        os.environ["PATH"] = original_path
        try:
            os.killpg(unrelated.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
