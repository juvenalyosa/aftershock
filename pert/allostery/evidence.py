"""Evidence grading of the consensus (interpretation layer).

Turns consensus tables into a conservative Markdown/JSON report: each scenario
receives a tier (``no-evidence`` / ``exploratory`` / ``weak`` / ``moderate`` /
``strong``) from the number of replicas, the replicate support and the
conservative score of its top residue.  With too few replicas a result is
labelled exploratory no matter how large the score is.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from pert.utils.io import load_json, md_table, read_csv, safe_float, safe_int


def _fmt(value: Any, nd: int = 3) -> str:
    """Fixed-decimal formatting with ``NA`` for missing values."""
    val = safe_float(value)
    if val != val:
        return "NA"
    return f"{val:.{nd}f}"


def _scenario_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Consensus rows grouped by scenario and ordered by consensus rank."""
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scenario", ""))].append(dict(row))
    for scenario in grouped:
        grouped[scenario].sort(key=lambda row: safe_int(row.get("consensus_rank"), 10**9))
    return dict(grouped)


def _tier_for_scenario(
    rows: Sequence[Dict[str, Any]],
    min_replicates_strong: int,
    min_replicates_moderate: int,
    min_support_strong: float,
    min_support_moderate: float,
    min_conservative_strong: float,
    min_conservative_moderate: float,
) -> Tuple[str, str]:
    """Evidence tier of a scenario and the reason for it (see module docstring)."""
    if not rows:
        return "no-evidence", "No consensus residue rows were available."

    top = rows[0]
    reps = safe_int(top.get("replicates_total"))
    support = safe_float(top.get("support_top_n_fraction"), 0.0)
    conservative = safe_float(top.get("conservative_score"), 0.0)
    mean_score = safe_float(top.get("total_score_mean"), 0.0)

    if reps < min_replicates_moderate:
        return (
            "exploratory",
            f"Only {reps} replicate(s); residue ranking is a hypothesis, not a stable claim.",
        )
    if (
        reps >= min_replicates_strong
        and support >= min_support_strong
        and conservative >= min_conservative_strong
    ):
        return (
            "strong",
            "Top residue has high replicate support and a conservative score above threshold.",
        )
    if (
        reps >= min_replicates_moderate
        and support >= min_support_moderate
        and conservative >= min_conservative_moderate
    ):
        return (
            "moderate",
            "Top residue is recurrent, but confidence is not yet high enough for a strong call.",
        )
    if mean_score > 0.0:
        return (
            "weak",
            "A signal exists, but replicate support or conservative score is below threshold.",
        )
    return "no-evidence", "No positive consensus score was detected."


def _load_panel_metrics(panel_dir: str) -> Dict[str, Dict[str, Any]]:
    """Region-level panel metrics (``mutation_panel_summary.json``) if present."""
    path = os.path.join(panel_dir, "mutation_panel_summary.json")
    if os.path.exists(path):
        try:
            data = load_json(path)
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            return {}
    return {}


def _specificity_notes(grouped: Dict[str, List[Dict[str, Any]]], top_n: int) -> Dict[str, List[str]]:
    """For each scenario, whether its top residues are unique to it or shared with other scenarios."""
    residue_to_scenarios: DefaultDict[str, List[str]] = defaultdict(list)
    for scenario, rows in grouped.items():
        for row in rows[: max(1, int(top_n))]:
            residue_to_scenarios[str(row.get("node_label", ""))].append(scenario)

    notes: Dict[str, List[str]] = {}
    for scenario, rows in grouped.items():
        scenario_notes: List[str] = []
        for row in rows[: max(1, int(top_n))]:
            residue = str(row.get("node_label", ""))
            scenarios = residue_to_scenarios.get(residue, [])
            if len(scenarios) == 1:
                scenario_notes.append(f"{residue}: scenario-specific within top {top_n}")
            else:
                scenario_notes.append(f"{residue}: shared with {', '.join(sorted(scenarios))}")
        notes[scenario] = scenario_notes
    return notes


def write_evidence_report(
    panel_dir: str,
    output_md: Optional[str],
    output_json: Optional[str],
    top_residues: int,
    top_regions: int,
    specificity_top_n: int,
    control_scenario: Optional[str],
    baseline_specific_top: int,
    min_replicates_strong: int,
    min_replicates_moderate: int,
    min_support_strong: float,
    min_support_moderate: float,
    min_conservative_strong: float,
    min_conservative_moderate: float,
) -> Dict[str, Any]:
    """Write ``allostery_evidence_report.md`` and ``allostery_evidence_summary.json`` for a panel."""
    panel_dir = os.path.abspath(panel_dir)
    residue_csv = os.path.join(panel_dir, "allostery_consensus_residues.csv")
    region_csv = os.path.join(panel_dir, "allostery_consensus_regions.csv")
    baseline_csv = os.path.join(panel_dir, "allostery_baseline_specific_residues.csv")
    consensus_json_path = os.path.join(panel_dir, "allostery_consensus.json")
    if not os.path.exists(residue_csv):
        raise FileNotFoundError(f"Missing consensus residue CSV: {residue_csv}")
    if not os.path.exists(region_csv):
        raise FileNotFoundError(f"Missing consensus region CSV: {region_csv}")

    residue_rows = read_csv(residue_csv)
    region_rows = read_csv(region_csv)
    baseline_rows = read_csv(baseline_csv) if os.path.exists(baseline_csv) else []
    if not control_scenario and os.path.exists(consensus_json_path):
        try:
            consensus_payload = load_json(consensus_json_path)
            control_scenario = (
                consensus_payload.get("control_baseline", {}).get("control_scenario")
                if isinstance(consensus_payload.get("control_baseline"), dict)
                else None
            )
        except Exception:
            control_scenario = None
    grouped_residues = _scenario_groups(residue_rows)
    grouped_regions = _scenario_groups(region_rows)
    grouped_baseline = _scenario_groups(baseline_rows) if baseline_rows else {}
    panel_metrics = _load_panel_metrics(panel_dir)
    specificity = _specificity_notes(grouped_residues, top_n=specificity_top_n)

    scenario_evidence: Dict[str, Dict[str, Any]] = {}
    for scenario, rows in grouped_residues.items():
        tier, reason = _tier_for_scenario(
            rows=rows,
            min_replicates_strong=min_replicates_strong,
            min_replicates_moderate=min_replicates_moderate,
            min_support_strong=min_support_strong,
            min_support_moderate=min_support_moderate,
            min_conservative_strong=min_conservative_strong,
            min_conservative_moderate=min_conservative_moderate,
        )
        scenario_evidence[scenario] = {
            "tier": tier,
            "reason": reason,
            "replicates_total": safe_int(rows[0].get("replicates_total")) if rows else 0,
            "top_residues": rows[: max(1, int(top_residues))],
            "top_regions": grouped_regions.get(scenario, [])[: max(1, int(top_regions))],
            "top_baseline_specific_residues": grouped_baseline.get(scenario, [])[: max(1, int(baseline_specific_top))],
            "panel_metrics": panel_metrics.get(scenario, {}),
            "specificity_notes": specificity.get(scenario, []),
        }

    tier_order = {"strong": 0, "moderate": 1, "weak": 2, "exploratory": 3, "no-evidence": 4}
    scenario_names = sorted(
        scenario_evidence,
        key=lambda name: (
            tier_order.get(str(scenario_evidence[name]["tier"]), 99),
            -safe_float(scenario_evidence[name]["top_residues"][0].get("consensus_score"), 0.0)
            if scenario_evidence[name]["top_residues"]
            else 0.0,
            name,
        ),
    )

    lines: List[str] = []
    lines.append("# Allostery Evidence Report")
    lines.append("")
    lines.append(f"Panel directory: `{panel_dir}`")
    lines.append("")
    lines.append("## Interpretation Scope")
    lines.append("")
    lines.append("This report ranks mutation-associated allostery hypotheses from the consensus residue and region tables.")
    lines.append("It does not prove causation by itself. Strong claims require enough independent replicas, stable residue support, and agreement with structural/dynamical checks.")
    lines.append("")
    lines.append("Evidence tiers:")
    lines.append(f"- strong: at least {min_replicates_strong} replicas, top-residue support >= {min_support_strong:.2f}, conservative score >= {min_conservative_strong:.2f}")
    lines.append(f"- moderate: at least {min_replicates_moderate} replicas, top-residue support >= {min_support_moderate:.2f}, conservative score >= {min_conservative_moderate:.2f}")
    lines.append("- exploratory: fewer replicas than required for a stable claim")
    lines.append("- weak: signal exists but does not pass support/score thresholds")
    if control_scenario:
        lines.append("")
        lines.append(f"Control baseline scenario: `{control_scenario}`")
        lines.append("Control-adjusted tables rank residues by score above the control scenario for the same residue.")
    lines.append("")

    summary_rows = []
    for scenario in scenario_names:
        evidence = scenario_evidence[scenario]
        top = evidence["top_residues"][0] if evidence["top_residues"] else {}
        baseline_top = (
            evidence["top_baseline_specific_residues"][0]
            if evidence.get("top_baseline_specific_residues")
            else {}
        )
        region = evidence["top_regions"][0] if evidence["top_regions"] else {}
        metrics = evidence.get("panel_metrics", {})
        summary_rows.append(
            [
                scenario,
                str(evidence["tier"]),
                str(evidence["replicates_total"]),
                str(top.get("node_label", "NA")),
                _fmt(top.get("consensus_score")),
                _fmt(top.get("support_top_n_fraction")),
                str(baseline_top.get("node_label", "NA")),
                _fmt(baseline_top.get("baseline_delta_consensus_score")),
                str(region.get("region", "NA")),
                _fmt(metrics.get("te_delta_mut_minus_wt_mean")),
            ]
        )
    lines.append("## Scenario Summary")
    lines.append("")
    lines.append(
        md_table(
            [
                "scenario",
                "tier",
                "replicas",
                "top residue",
                "consensus",
                "support",
                "above-control residue",
                "above-control delta",
                "top region",
                "mean DeltaTE",
            ],
            summary_rows,
        )
    )
    lines.append("")

    for scenario in scenario_names:
        evidence = scenario_evidence[scenario]
        lines.append(f"## {scenario}")
        lines.append("")

        baseline_specific_rows = evidence.get("top_baseline_specific_residues", [])
        if control_scenario and baseline_specific_rows:
            baseline_table = []
            for row in baseline_specific_rows[: max(1, int(baseline_specific_top))]:
                baseline_table.append(
                    [
                        str(row.get("baseline_specific_rank", "")),
                        str(row.get("node_label", "")),
                        _fmt(row.get("baseline_delta_consensus_score")),
                        _fmt(row.get("consensus_score")),
                        _fmt(row.get("control_consensus_score")),
                        str(row.get("mutation_specific_vs_control", "")),
                        str(row.get("baseline_exceeds_control_ci", "")),
                    ]
                )
            lines.append(f"Control-adjusted residues versus `{control_scenario}`:")
            lines.append(
                md_table(
                    [
                        "rank",
                        "residue",
                        "delta consensus",
                        "scenario score",
                        "control score",
                        "specific?",
                        "CI exceeds control?",
                    ],
                    baseline_table,
                )
            )
            lines.append("")
        elif control_scenario and scenario != control_scenario:
            lines.append(f"No control-adjusted residue table was available for `{scenario}` versus `{control_scenario}`.")
            lines.append("")
        lines.append(f"Evidence tier: `{evidence['tier']}`")
        lines.append("")
        lines.append(f"Reason: {evidence['reason']}")
        lines.append("")

        metrics = evidence.get("panel_metrics", {})
        if metrics:
            lines.append("Global WT/MUT effect metrics:")
            lines.append(
                md_table(
                    ["metric", "mean", "std"],
                    [
                        [
                            "DeltaTE MUT-WT",
                            _fmt(metrics.get("te_delta_mut_minus_wt_mean")),
                            _fmt(metrics.get("te_delta_mut_minus_wt_std")),
                        ],
                        [
                            "JS divergence",
                            _fmt(metrics.get("js_divergence_mean")),
                            _fmt(metrics.get("js_divergence_std")),
                        ],
                        [
                            "susceptibility Delta Frobenius",
                            _fmt(metrics.get("susceptibility_delta_fro_norm_mean")),
                            _fmt(metrics.get("susceptibility_delta_fro_norm_std")),
                        ],
                    ],
                )
            )
            lines.append("")

        residue_table = []
        for row in evidence["top_residues"][: max(1, int(top_residues))]:
            residue_table.append(
                [
                    str(row.get("consensus_rank", "")),
                    str(row.get("node_label", "")),
                    _fmt(row.get("consensus_score")),
                    _fmt(row.get("total_score_mean")),
                    _fmt(row.get("total_score_ci95_low")),
                    _fmt(row.get("support_top_n_fraction")),
                    _fmt(row.get("is_source_hotspot_fraction")),
                    _fmt(row.get("is_target_site_fraction")),
                ]
            )
        lines.append("Top consensus residues:")
        lines.append(
            md_table(
                ["rank", "residue", "consensus", "mean", "CI low", "support", "source frac", "target frac"],
                residue_table,
            )
        )
        lines.append("")

        region_table = []
        for row in evidence["top_regions"][: max(1, int(top_regions))]:
            region_table.append(
                [
                    str(row.get("consensus_rank", "")),
                    str(row.get("region", "")),
                    _fmt(row.get("consensus_score")),
                    _fmt(row.get("mean_total_score_mean")),
                    _fmt(row.get("support_top_n_fraction")),
                    str(row.get("top_residue_modes", "")),
                ]
            )
        lines.append("Top consensus regions:")
        lines.append(
            md_table(
                ["rank", "region", "consensus", "mean score", "support", "top residue modes"],
                region_table,
            )
        )
        lines.append("")

        notes = evidence.get("specificity_notes", [])[: max(1, int(specificity_top_n))]
        if notes:
            lines.append("Specificity notes:")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append(f"- Consensus residues: `allostery_consensus_residues.csv`")
    lines.append(f"- Consensus regions: `allostery_consensus_regions.csv`")
    if os.path.exists(baseline_csv):
        lines.append(f"- Control-adjusted residues: `allostery_baseline_specific_residues.csv`")
    if os.path.exists(consensus_json_path):
        lines.append(f"- Consensus JSON: `allostery_consensus.json`")
    lines.append("")
    lines.append("## Recommended Next Validation")
    lines.append("")
    lines.append("- Run at least 3 independent replicas per mutation before treating any residue ranking as stable.")
    lines.append("- Add a WT-vs-WT or neutral-mutation control and require top residues to exceed that baseline.")
    lines.append("- Check VMD views for structural plausibility; reject routes that are statistical artifacts without spatial continuity.")
    lines.append("- Prioritize residues that are recurrent across replicas and supported by multiple components: dynamics, contacts, TE, and target coupling.")
    lines.append("")

    output_md = output_md or os.path.join(panel_dir, "allostery_evidence_report.md")
    output_json = output_json or os.path.join(panel_dir, "allostery_evidence_summary.json")
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    payload = {
        "panel_dir": panel_dir,
        "evidence_thresholds": {
            "min_replicates_strong": min_replicates_strong,
            "min_replicates_moderate": min_replicates_moderate,
            "min_support_strong": min_support_strong,
            "min_support_moderate": min_support_moderate,
            "min_conservative_strong": min_conservative_strong,
            "min_conservative_moderate": min_conservative_moderate,
        },
        "scenario_order": scenario_names,
        "scenario_evidence": scenario_evidence,
        "control_scenario": control_scenario,
        "outputs": {
            "report_md": output_md,
            "summary_json": output_json,
            "residue_consensus_csv": residue_csv,
            "region_consensus_csv": region_csv,
            "baseline_specific_residue_csv": baseline_csv if os.path.exists(baseline_csv) else None,
        },
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
