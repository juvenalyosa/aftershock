"""
Command-line interface: ``python -m pert <command> [options]``.

Each command lives in its own module with a ``main()`` function, so adding a
command means adding one module and one line to ``COMMANDS``.  Modules are
imported lazily: asking for ``--help`` of a light command never loads OpenMM
or MODELLER.
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List, Optional, Tuple

# command -> (module, one-line description); grouped by workflow stage.
COMMANDS: Dict[str, Tuple[str, str]] = {
    # --- full MD workflow ---------------------------------------------------
    "pipeline": ("pert.cli.pipeline", "Run one WT-vs-MUT comparison: build, MD, analysis."),
    "pipeline-config": ("pert.cli.pipeline_from_config", "Same as 'pipeline', configured from a JSON file."),
    "md-panel": ("pert.cli.md_panel", "Run several mutation scenarios x replicas through the pipeline."),
    "md-panel-config": ("pert.cli.md_panel_from_config", "Same as 'md-panel', configured from a JSON file."),
    # --- post-processing of finished runs ------------------------------------
    "analyze-run": ("pert.cli.analyze_run", "Plots, binding-site table, residue-flow network, VMD scenes."),
    "diagnostics": ("pert.cli.diagnostics", "Extended diagnostic plots and interpretation summary."),
    "rank": ("pert.cli.rank", "MD-based allosteric residue/region ranking of one run."),
    "null-control": ("pert.cli.null_control", "Calibrate a panel against the WT-vs-WT null (no mutation)."),
    "consensus": ("pert.cli.consensus", "Aggregate rankings across replicas of a panel."),
    "evidence": ("pert.cli.evidence", "Grade the consensus into an evidence report."),
    "report": ("pert.cli.report", "Plain-language Markdown report of one run."),
    # --- fast screening (no MD) -----------------------------------------------
    "fast-scan": ("pert.cli.fast_scan", "Score one WT/mutant structure pair in seconds."),
    "fast-panel": ("pert.cli.fast_panel", "Build and fast-score a panel of mutation sets."),
    # --- benchmarking -----------------------------------------------------------
    "normalize-benchmark": ("pert.cli.normalize_benchmark", "Convert an external table to the benchmark format."),
    "benchmark": ("pert.cli.benchmark", "Benchmark the fast scanner against labels/effects."),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: python -m pert <command> [options]", "", "commands:"]
    for name, (_, description) in COMMANDS.items():
        lines.append(f"  {name.ljust(width)}  {description}")
    lines += ["", "Run 'python -m pert <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """Dispatch ``<command> [options]`` to the ``main()`` of the command module."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_usage())
        return
    command = args[0]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command '{command}'.\n\n{_usage()}")
    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    # The command modules parse sys.argv themselves (they are also runnable directly).
    sys.argv = [f"pert {command}"] + args[1:]
    module.main()
