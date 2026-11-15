"""Helpers for running a PERT panel from a hosted notebook such as Google Colab.

A notebook cell is a poor place for a job that lasts hours: the cell blocks the
notebook, and its output is lost when the browser tab reloads.  These helpers
run the panel workflow as a detached process that logs to a file, and let any
cell look at it:

* ``derive_config``   copy a panel config with a few values changed (output on a
                      mounted drive, a shorter run for a smoke test, ...);
* ``run_stream``      run a short command and echo its output into the notebook;
* ``launch``          start ``scripts/run_panel_workflow.sh`` in the background
                      (refuses to start a second copy of a panel that is running);
* ``follow``          print the log as it grows, with progress and an estimate
                      of the time left, until the workflow ends;
* ``stop``            stop a running workflow (it can be resumed by ``launch``);
* ``panel_progress`` / ``platforms_used``  read the state of the simulations;
* ``verdict_rows``       the null-control verdict of every scenario, as table rows;
* ``zip_light_results``  pack tables, figures and reports (no trajectories).

Only the standard library is used, so the module works (and is tested) anywhere.
A stopped or killed workflow loses nothing: launching the same config again
resumes every simulation from its last checkpoint.
"""

from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_panel_workflow.sh")

# Popen handles of workflows started by this interpreter, so they can be reaped.
_STARTED: Dict[str, subprocess.Popen] = {}


# --- configs -------------------------------------------------------------------------------------


def _panel_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The options of a panel config (the ``panel`` section when there is one)."""
    panel = payload.get("panel")
    return panel if isinstance(panel, dict) else payload


def derive_config(base_path: str, out_path: str, **overrides: Any) -> str:
    """Write a copy of ``base_path`` with ``overrides`` applied to the panel options.

    An override of ``None`` removes the option.  Relative input paths in the
    config stay valid wherever the copy is written, because the workflow always
    runs from the repository root.  Returns ``out_path``.
    """
    with open(base_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    panel = _panel_section(payload)
    for key, value in overrides.items():
        if value is None:
            panel.pop(key, None)
        else:
            panel[key] = value
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path


def expected_simulations(config_path: str) -> Dict[str, int]:
    """Number of simulations a panel config asks for, and the steps of each.

    Every scenario/replica simulates the WT and the mutant.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        panel = _panel_section(json.load(handle))
    names = list(panel.get("mutation_sets") or [])
    selected = panel.get("scenarios")
    if selected:
        wanted = selected.split(",") if isinstance(selected, str) else list(selected)
        wanted = [name.strip() for name in wanted if name.strip()]
        names = [name for name in names if name in wanted] or wanted
    replicas = int(panel.get("replicas", 2))
    return {
        "simulations": 2 * len(names) * replicas,
        "steps_each": int(panel.get("prod_steps", 0)),
    }


# --- running commands ----------------------------------------------------------------------------


def run_stream(command: Sequence[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None) -> int:
    """Run ``command``, echoing its output line by line; returns the exit code.

    ``env`` is added to the current environment.  (A plain ``subprocess.run``
    inside a notebook sends the output to the server log, not to the cell.)
    """
    full_env = dict(os.environ)
    full_env["PYTHONUNBUFFERED"] = "1"
    full_env.update(env or {})
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        errors="replace",
        env=full_env,
        cwd=cwd,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return process.wait()


# --- detached workflow ---------------------------------------------------------------------------


def _boot_id() -> str:
    """Identifies this machine since its last boot (a pid means nothing on another one)."""
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.uname().nodename


def _state_path(log_path: str) -> str:
    return f"{log_path}.pid"


def _exit_path(log_path: str) -> str:
    return f"{log_path}.exit"


def _read_state(log_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_state_path(log_path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """True while ``pid`` is a live process (a zombie waiting to be reaped is not)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            # "pid (name) state ..."; the name may contain spaces or parentheses.
            state = handle.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def is_running(log_path: str) -> bool:
    """True while the workflow that logs to ``log_path`` is running on this machine."""
    process = _STARTED.get(log_path)
    if process is not None and process.poll() is not None:
        return False
    state = _read_state(log_path)
    if state is None or state.get("boot_id") != _boot_id():
        return False
    return _pid_alive(int(state["pid"]))


def exit_code(log_path: str) -> Optional[int]:
    """Exit code of the last workflow that finished by itself, or ``None``."""
    try:
        with open(_exit_path(log_path), "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def launch(
    config_path: str,
    log_path: str,
    control_scenario: str = "",
    python: str = sys.executable,
    extra_env: Optional[Dict[str, str]] = None,
) -> bool:
    """Start the panel workflow for ``config_path`` in the background.

    Output is appended to ``log_path``.  Returns ``True`` if a workflow was
    started and ``False`` if one was already running (two processes writing
    the same panel would corrupt it).  Launching a panel that was interrupted
    resumes it.
    """
    if is_running(log_path):
        print(f"Already running (pid {_read_state(log_path)['pid']}); use follow() to watch it.")
        return False
    config_path = os.path.abspath(config_path)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    if os.path.exists(_exit_path(log_path)):
        os.remove(_exit_path(log_path))

    env = dict(os.environ)
    env.update({"PERT_PYTHON": python, "PYTHONUNBUFFERED": "1"})
    if control_scenario:
        env["CONTROL_SCENARIO"] = control_scenario
    env.update(extra_env or {})

    # The wrapper records the exit code, so that a notebook that reconnects
    # later can tell "finished" from "failed".
    wrapper = 'bash "$1" "$2"; code=$?; echo "${code}" > "$3"; exit "${code}"'
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')}  config: {config_path} =====\n")
        log.flush()
        process = subprocess.Popen(
            ["bash", "-c", wrapper, "pert-panel", WORKFLOW_SCRIPT, config_path, _exit_path(log_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    _STARTED[log_path] = process
    with open(_state_path(log_path), "w", encoding="utf-8") as handle:
        json.dump({"pid": process.pid, "boot_id": _boot_id(), "config": config_path, "started": time.time()}, handle)
    print(f"Started (pid {process.pid}). Log: {log_path}")
    return True


def stop(log_path: str, grace_seconds: float = 20.0) -> bool:
    """Stop a running workflow and everything it started; returns ``True`` if one was stopped.

    Nothing is lost beyond the steps since the last checkpoint: ``launch`` resumes.
    """
    if not is_running(log_path):
        print("Not running.")
        return False
    pid = int(_read_state(log_path)["pid"])
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)  # the workflow leads its own process group
        except ProcessLookupError:
            break
        deadline = time.time() + grace_seconds
        while time.time() < deadline and is_running(log_path):
            time.sleep(0.5)
        if not is_running(log_path):
            break
    print("Stopped.")
    return True


# --- progress ------------------------------------------------------------------------------------


def panel_progress(output_dir: str) -> List[Dict[str, Any]]:
    """State of every simulation that has started, from its restart bookkeeping."""
    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(output_dir, "*", "rep*", "**", "*_progress.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue  # being rewritten right now
        name = os.path.relpath(path, output_dir)[: -len("_progress.json")]
        rows.append(
            {
                "simulation": name,
                "stage": str(data.get("stage", "")),
                "steps_done": int(data.get("steps_done", 0)),
                "steps_total": int(data.get("production_steps_requested", 0)),
                "resumes": int(data.get("resumes", 0)),
            }
        )
    return rows


def platforms_used(output_dir: str) -> Dict[str, str]:
    """OpenMM platform each finished simulation ran on."""
    used: Dict[str, str] = {}
    pattern = os.path.join(output_dir, "*", "rep*", "**", "*_md.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        used[os.path.relpath(path, output_dir)[: -len("_md.json")]] = str(data.get("platform_selected", "?"))
    return used


def _format_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60.0))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def progress_line(output_dir: str, expected: Optional[Dict[str, int]] = None) -> str:
    """One-line summary of the MD part of a panel."""
    rows = panel_progress(output_dir)
    done = sum(1 for row in rows if row["stage"] == "completed" and row["steps_done"] >= row["steps_total"])
    total = int(expected["simulations"]) if expected else len(rows)
    line = f"simulations finished: {done}/{total}"
    current = [row for row in rows if row["steps_total"] and row["steps_done"] < row["steps_total"]]
    for row in current:
        line += f"; {row['simulation']} at {100.0 * row['steps_done'] / row['steps_total']:.0f}%"
    return line


def _steps_done(output_dir: str) -> int:
    return sum(row["steps_done"] for row in panel_progress(output_dir))


def follow(
    log_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    poll_seconds: float = 15.0,
    progress_every_seconds: float = 600.0,
    tail_lines: int = 60,
) -> Optional[int]:
    """Print the workflow log as it grows until the workflow ends; returns its exit code.

    Starts from the last ``tail_lines`` lines.  With ``output_dir`` (and
    ``config_path`` to know how much was asked for) a progress line with an
    estimate of the MD time left is printed every ``progress_every_seconds``.
    Interrupting this function does not stop the workflow.
    """
    expected = expected_simulations(config_path) if config_path else None
    position = 0
    if os.path.exists(log_path):
        with open(log_path, "rb") as handle:
            lines = handle.read().splitlines(keepends=True)
        position = sum(len(line) for line in lines[: max(0, len(lines) - tail_lines)])

    started = time.time()
    steps_at_start = _steps_done(output_dir) if output_dir else 0
    last_progress = started
    while True:
        running = is_running(log_path)
        if os.path.exists(log_path):
            with open(log_path, "rb") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()
            if chunk:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        if not running:
            break
        now = time.time()
        if output_dir and now - last_progress >= progress_every_seconds:
            last_progress = now
            line = progress_line(output_dir, expected)
            if expected and expected["steps_each"]:
                total_steps = expected["simulations"] * expected["steps_each"]
                steps = _steps_done(output_dir)
                gained = steps - steps_at_start
                if gained > 0 and steps < total_steps:
                    left = (total_steps - steps) * (now - started) / gained
                    line += f"; MD time left about {_format_duration(left)} (analysis comes after)"
            print(f"[{time.strftime('%H:%M')}] {line}", flush=True)
        time.sleep(poll_seconds)

    code = exit_code(log_path)
    if code == 0:
        print("\nWorkflow finished.")
    elif code is None:
        print("\nWorkflow is not running and did not finish (stopped, or the machine restarted). launch() resumes it.")
    else:
        print(f"\nWorkflow FAILED with exit code {code}; the reason is in the last lines above.")
    return code


# --- results -------------------------------------------------------------------------------------

def verdict_rows(panel_dir: str) -> List[Dict[str, Any]]:
    """Null-control verdict of every scenario of a panel, one row each (controls last)."""
    with open(os.path.join(panel_dir, "null_control", "null_control_summary.json"), "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = []
    for name, entry in summary["scenarios"].items():
        # Results written before the rules for "counts" existed only list what exceeds the null.
        counted = entry.get("metrics_supporting", entry.get("metrics_beyond_null", []))
        rows.append(
            {
                "scenario": name + (" (control)" if entry.get("is_control") else ""),
                "replicas": entry["replicas"],
                "evidence": entry["evidence"],
                "metrics that count": ", ".join(counted) or "-",
                "note": entry["evidence_note"],
            }
        )
    rows.sort(key=lambda row: row["scenario"].endswith("(control)"))
    return rows


# Trajectories and restart data: large, and not needed to read the results.
_HEAVY_SUFFIXES = (".npz", ".dcd", ".chk", "_start.xml", "_system.xml")


def zip_light_results(output_dir: str, zip_path: str) -> str:
    """Zip the tables, figures, reports and logs of a panel, leaving trajectories out."""
    output_dir = os.path.abspath(output_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, dirs, files in os.walk(output_dir):
            dirs[:] = sorted(d for d in dirs if not d.endswith("_restart"))
            for name in sorted(files):
                if name.endswith(_HEAVY_SUFFIXES):
                    continue
                path = os.path.join(root, name)
                archive.write(path, os.path.relpath(path, os.path.dirname(output_dir)))
    return zip_path
