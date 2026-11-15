"""Hosted-notebook helpers (scripts/colab_tools.py): configs, detached workflow, progress."""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import colab_tools as ct  # noqa: E402


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_derive_config_overrides_only_the_panel_options(tmp_path):
    base = tmp_path / "base.json"
    _write(str(base), {"panel": {"replicas": 2, "prod_steps": 100, "platform": "AUTO"}, "notes": {"x": 1}})
    out = ct.derive_config(str(base), str(tmp_path / "sub" / "derived.json"), replicas=1, platform=None, output_dir="/o")
    payload = json.load(open(out))
    assert payload["panel"] == {"replicas": 1, "prod_steps": 100, "output_dir": "/o"}
    assert payload["notes"] == {"x": 1}
    assert json.load(open(base))["panel"]["replicas"] == 2  # the base file is untouched


def test_expected_simulations_counts_wt_and_mutant_of_the_selected_scenarios(tmp_path):
    cfg = tmp_path / "cfg.json"
    panel = {"mutation_sets": {"a": ["K2R"], "b": ["K2A"], "c": ["K2G"]}, "replicas": 2, "prod_steps": 500}
    _write(str(cfg), {"panel": panel})
    assert ct.expected_simulations(str(cfg)) == {"simulations": 12, "steps_each": 500}
    _write(str(cfg), {"panel": dict(panel, scenarios="a, c", replicas=1)})
    assert ct.expected_simulations(str(cfg)) == {"simulations": 4, "steps_each": 500}


def test_progress_and_light_zip(tmp_path):
    panel = tmp_path / "panel"
    _write(str(panel / "snp" / "rep1" / "md" / "wt_progress.json"),
           {"stage": "completed", "steps_done": 100, "production_steps_requested": 100})
    _write(str(panel / "snp" / "rep1" / "md" / "mut_progress.json"),
           {"stage": "production", "steps_done": 25, "production_steps_requested": 100, "resumes": 1})
    _write(str(panel / "snp" / "rep1" / "md" / "wt_md.json"), {"platform_selected": "CUDA"})
    (panel / "snp" / "rep1" / "md" / "wt_traj.npz").write_bytes(b"big")
    os.makedirs(panel / "snp" / "rep1" / "md" / "wt_restart")
    (panel / "snp" / "rep1" / "md" / "wt_restart" / "part.json").write_text("{}")

    rows = ct.panel_progress(str(panel))
    assert [(row["simulation"], row["steps_done"]) for row in rows] == [
        (os.path.join("snp", "rep1", "md", "mut"), 25),
        (os.path.join("snp", "rep1", "md", "wt"), 100),
    ]
    line = ct.progress_line(str(panel), {"simulations": 4, "steps_each": 100})
    assert "finished: 1/4" in line and "at 25%" in line
    assert ct.platforms_used(str(panel)) == {os.path.join("snp", "rep1", "md", "wt"): "CUDA"}

    names = zipfile.ZipFile(ct.zip_light_results(str(panel), str(tmp_path / "light.zip"))).namelist()
    assert "panel/snp/rep1/md/wt_md.json" in names
    assert not any(name.endswith(".npz") or "_restart" in name for name in names)


@pytest.mark.skipif(os.name != "posix", reason="detached workflow uses POSIX process groups")
def test_launch_follow_stop_with_a_stand_in_workflow(tmp_path, monkeypatch, capsys):
    script = tmp_path / "workflow.sh"
    script.write_text('echo "config $1 python ${PERT_PYTHON:+set} control ${CONTROL_SCENARIO}"\nsleep "${NAP}"\nexit "${CODE}"\n')
    monkeypatch.setattr(ct, "WORKFLOW_SCRIPT", str(script))
    cfg = tmp_path / "cfg.json"
    _write(str(cfg), {"panel": {}})
    log = str(tmp_path / "logs" / "run.log")

    # A workflow that fails: the exit code and the output reach the notebook.
    assert ct.launch(str(cfg), log, control_scenario="wt_model", extra_env={"NAP": "0", "CODE": "3"}) is True
    assert ct.follow(log, poll_seconds=0.1) == 3
    assert "python set control wt_model" in capsys.readouterr().out
    assert ct.is_running(log) is False

    # A long one: a second launch is refused, stop() ends it, and it can be launched again.
    assert ct.launch(str(cfg), log, extra_env={"NAP": "60", "CODE": "0"}) is True
    assert ct.is_running(log) is True
    assert ct.launch(str(cfg), log, extra_env={"NAP": "0", "CODE": "0"}) is False
    started = time.time()
    assert ct.stop(log, grace_seconds=5) is True
    assert time.time() - started < 15
    assert ct.is_running(log) is False and ct.exit_code(log) is None
    assert ct.launch(str(cfg), log, extra_env={"NAP": "0", "CODE": "0"}) is True
    assert ct.follow(log, poll_seconds=0.1) == 0


def test_verdict_rows_put_the_control_last_and_read_older_summaries(tmp_path):
    panel = tmp_path / "panel"
    summary = {
        "scenarios": {
            "wt_model": {"replicas": 2, "evidence": "none", "is_control": True, "metrics_beyond_null": [],
                         "metrics_supporting": [], "evidence_note": "Control"},
            "snp": {"replicas": 2, "evidence": "suggestive", "is_control": False, "metrics_beyond_null": ["a", "b"],
                    "metrics_supporting": ["a"], "evidence_note": "x"},
            # written before 'metrics_supporting' existed
            "old": {"replicas": 1, "evidence": "suggestive", "metrics_beyond_null": ["b"], "evidence_note": "y"},
        }
    }
    _write(str(panel / "null_control" / "null_control_summary.json"), summary)
    rows = ct.verdict_rows(str(panel))
    assert [row["scenario"] for row in rows] == ["snp", "old", "wt_model (control)"]
    assert [row["metrics that count"] for row in rows] == ["a", "b", "-"]


def test_storage_estimate_uses_the_measured_system_size(tmp_path):
    cfg = tmp_path / "cfg.json"
    _write(str(cfg), {"panel": {"mutation_sets": {"a": ["K2R"]}, "replicas": 4, "prod_steps": 12500000, "report_interval": 10000}})
    assert ct.estimate_storage_gb(str(cfg), None) is None
    assert ct.estimate_storage_gb(str(cfg), str(tmp_path / "nothing")) is None
    _write(str(tmp_path / "smoke" / "a" / "rep1" / "md" / "wt_md.json"), {"num_atoms": 20000})
    _write(str(tmp_path / "smoke" / "a" / "rep1" / "md" / "mut_md.json"), {"num_atoms": 20589})
    # 8 simulations x 1251 frames x 20589 atoms x 60 bytes, plus 10%
    assert ct.estimate_storage_gb(str(cfg), str(tmp_path / "smoke")) == pytest.approx(8 * 1251 * 20589 * 60 * 1.1 / 1e9)
