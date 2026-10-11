#!/usr/bin/env python3
"""Build and run bounded MPI empty-rank regression cases.

Each invocation runs one case in a separate MPI process session. Failures clean up
that complete session with TERM followed by KILL, so an exited launcher or closed
pipe cannot leave its ranks running.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "mpi_empty_ranks"
RANKS = (1, 2, 4)
CASES = (
    "zero-count-collectives",
    "empty-cumsum",
    "empty-boundaries",
    "empty-extrema",
    "sparse-sort",
    "sort-balance",
    "mapper-and-permutation",
    "rooted-and-aij-rejection",
    "collective-metadata-rejection",
    "expanded-empty-contracts",
    "csr-empty-matrix-contracts",
    "distributed-dot",
    "distributed-dot-rejection",
    "remap-layout-agreement",
    "layouts-and-consumers",
    "ghost-variable-chunks",
    "exchange-buffer-mismatch-panics",
    "ghost-edge-topologies",
)
TIMEOUT_SECONDS = 30
BUILD_TIMEOUT_SECONDS = 900
TERM_GRACE_SECONDS = 5
KILL_GRACE_SECONDS = 5
FINAL_COLLECTION_SECONDS = 5


def process_group_alive(process: subprocess.Popen[str]) -> bool:
    """Return whether any process remains in the runner-owned session group."""
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group(process: subprocess.Popen[str], timeout: float) -> bool:
    """Wait only a bounded interval for the owned process group to disappear."""
    deadline = time.monotonic() + timeout
    while process_group_alive(process):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def collect_output(process: subprocess.Popen[str]) -> str:
    """Collect launcher output without allowing a descendant-held pipe to block."""
    try:
        output, _ = process.communicate(timeout=FINAL_COLLECTION_SECONDS)
        return output
    except subprocess.TimeoutExpired as error:
        return error.output or ""


def stop_process_group(process: subprocess.Popen[str]) -> str:
    """Stop the owned group, escalating independently of launcher or pipe state."""
    if process_group_alive(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if not wait_for_process_group(process, TERM_GRACE_SECONDS):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        wait_for_process_group(process, KILL_GRACE_SECONDS)
    return collect_output(process)


def build_executable() -> str:
    """Compile the harness and return Cargo's exact executable path."""
    command = [
        "cargo",
        "test",
        "-p",
        "rlst",
        "--features",
        "mpi,strict",
        "--test",
        TARGET,
        "--no-run",
        "--message-format=json",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=BUILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        output = stop_process_group(process)
        raise RuntimeError(
            f"build timed out after {BUILD_TIMEOUT_SECONDS}s: {' '.join(command)}\n{output}"
        ) from None
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if completed.returncode:
        stop_process_group(process)
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"build failed ({completed.returncode}): {' '.join(command)}")

    executable = None
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("target", {}).get("name") == TARGET
            and message.get("executable")
        ):
            executable = message["executable"]
    if executable is None:
        raise RuntimeError("Cargo did not report an executable for mpi_empty_ranks")
    return executable


def run_bounded(executable: str, ranks: int, case: str) -> tuple[bool, str]:
    """Run one MPI case, terminating its entire process group on failure."""
    command = ["mpiexec", "--oversubscribe", "-n", str(ranks), executable]
    environment = os.environ | {
        "RLST_MPI_EMPTY_CASE": case,
        # The containerized CI runner executes as root. Open MPI requires both
        # variables as an explicit acknowledgement before launching ranks.
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        launcher_returncode = process.poll()
        output = stop_process_group(process)
        if launcher_returncode is not None:
            return False, output
        return False, f"timeout after {TIMEOUT_SECONDS}s\n{output}"
    if process.returncode:
        output += stop_process_group(process)
        return False, output
    # A launcher can report success before a detached-pipe descendant exits.
    # The runner owns the session, so do not leave such a descendant behind.
    stop_process_group(process)
    return True, output


def main() -> int:
    executable = build_executable()
    failures = []
    for ranks in RANKS:
        for case in CASES:
            success, output = run_bounded(executable, ranks, case)
            status = "passed" if success else "failed"
            print(f"[{status}] ranks={ranks} case={case}")
            if output:
                print(output, end="" if output.endswith("\n") else "\n")
            if not success:
                failures.append(f"ranks={ranks} case={case}")
    if failures:
        print("MPI regression failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
