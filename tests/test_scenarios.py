"""Scenario definitions of a panel: parsing, files, validation, seeds, config launcher."""

from __future__ import annotations

import json

import pytest

from pert.cli.md_panel_from_config import _build_command
from pert.pipelines.scenarios import (
    MUTATION_SCENARIOS,
    collect_scenarios,
    load_mutation_sets,
    parse_mutation_set,
    scenario_seed_offsets,
    validate_scenarios,
)

ZNF = "YKCGLCERSFVEKSALSRHQRVHKN"


def test_default_is_the_builtin_toy_panel():
    assert collect_scenarios() == [(name, list(muts)) for name, muts in MUTATION_SCENARIOS]


def test_custom_sets_replace_the_builtins_unless_asked():
    custom = collect_scenarios(mutation_sets=["double:K2R,Q20dup", "silent:K2K"])
    assert custom == [("double", ["K2R", "Q20dup"]), ("silent", ["K2K"])]
    both = collect_scenarios(mutation_sets=["double:K2R,Q20dup"], include_builtins=True)
    assert [name for name, _ in both] == [name for name, _ in MUTATION_SCENARIOS] + ["double"]


def test_selection_filters_and_rejects_unknown_names():
    assert collect_scenarios(selected=["snp_k2r,dup_q20"]) == [("snp_k2r", ["K2R"]), ("dup_q20", ["Q20dup"])]
    with pytest.raises(ValueError, match="Unknown scenario"):
        collect_scenarios(selected=["snp_k2r", "typo"])


@pytest.mark.parametrize("bad", ["K2R", ":K2R", "name:", "a/b:K2R", "has space:K2R", "..:K2R"])
def test_malformed_or_unsafe_scenarios_are_rejected(bad):
    with pytest.raises(ValueError):
        collect_scenarios(mutation_sets=[bad])


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        collect_scenarios(mutation_sets=["a:K2R", "a:Q20dup"])


def test_scenarios_from_files(tmp_path):
    as_mapping = tmp_path / "sets.json"
    as_mapping.write_text(json.dumps({"mutation_sets": {"one": ["K2R"], "two": "K2R;Q20dup"}}))
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([{"name": "three", "mutations": ["10_11delinsG"]}]))
    as_csv = tmp_path / "sets.csv"
    as_csv.write_text('name,mutations\nfour,"R8_S9insA,Q20dup"\n')

    assert load_mutation_sets(str(as_mapping)) == [("one", ["K2R"]), ("two", ["K2R", "Q20dup"])]
    assert load_mutation_sets(str(as_list)) == [("three", ["10_11delinsG"])]
    assert load_mutation_sets(str(as_csv)) == [("four", ["R8_S9insA", "Q20dup"])]
    combined = collect_scenarios(mutation_files=[str(as_mapping), str(as_csv)], mutation_sets=["five:K2K"])
    assert [name for name, _ in combined] == ["one", "two", "four", "five"]
    assert parse_mutation_set("x: K2R ; Q20dup ") == ("x", ["K2R", "Q20dup"])


def test_validation_applies_every_scenario_to_the_reference():
    mutated = validate_scenarios([("k2r", ["K2R"]), ("silent", ["K2K"]), ("dup", ["Q20dup"])], ZNF, 1)
    assert mutated["k2r"] == "YRCGLCERSFVEKSALSRHQRVHKN"
    assert mutated["silent"] == ZNF  # modelled-WT control
    assert len(mutated["dup"]) == len(ZNF) + 1


@pytest.mark.parametrize(
    "mutations, reason",
    [(["A2R"], "mismatch"), (["K99R"], "missing position"), (["K2R", "2_3del"], "overlaps"), (["nonsense"], "parse")],
)
def test_validation_names_the_offending_scenario(mutations, reason):
    with pytest.raises(ValueError, match="Scenario 'bad'"):
        validate_scenarios([("fine", ["K2R"]), ("bad", mutations)], ZNF, 1)


def test_seed_offsets_do_not_depend_on_the_rest_of_the_panel():
    full = scenario_seed_offsets(["a", "b", "c", "d"])
    assert len(set(full.values())) == 4
    # removing, reordering or adding scenarios leaves the others untouched
    assert scenario_seed_offsets(["d"])["d"] == full["d"]
    assert scenario_seed_offsets(["c", "a"]) == {"c": full["c"], "a": full["a"]}
    assert scenario_seed_offsets(["a", "b", "c", "d", "e"])["b"] == full["b"]


def test_config_launcher_passes_scenarios_through():
    command = _build_command(
        {
            "reference_pdb": "protein.pdb",
            "mutation_sets": {"k2r": ["K2R"], "double": ["K2R", "Q20dup"]},
            "mutation_file": "more.json",
            "include_builtins": True,
            "no_resume": False,
        }
    )
    text = " ".join(command)
    assert "-m pert md-panel" in text
    assert "--mutation-set k2r:K2R" in text and "--mutation-set double:K2R,Q20dup" in text
    assert "--mutation-file more.json" in text and "--include-builtins" in text
    assert "--no-resume" not in text
    as_list = _build_command({"mutation_sets": [{"name": "x", "mutations": "K2R"}]})
    assert as_list[-2:] == ["--mutation-set", "x:K2R"]


# --- up-front checks of the MD panel ----------------------------------------------------------------


def _panel_args(repo_root, **overrides):
    import argparse
    import os

    values = dict(
        reference_pdb=os.path.join(repo_root, "data", "toy", "1ZNF.pdb"),
        template_pdb=os.path.join(repo_root, "data", "toy", "1ZNF.pdb"),
        ligand_xml=os.path.join(repo_root, "data", "toy", "empty_ligand.xml"),
        ligand_resname="ZN",
        template="1ZNFA",
        chain_id="A",
        replicas=1,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_panel_preflight_accepts_the_toy_system(repo_root):
    pytest.importorskip("openmm")
    from pert.pipelines.md_panel import validate_panel_inputs

    mutated = validate_panel_inputs(_panel_args(repo_root), [("k2r", ["K2R"]), ("wt_model", ["K2K"])])
    assert mutated == {"k2r": "YRCGLCERSFVEKSALSRHQRVHKN", "wt_model": ZNF}


def test_panel_preflight_stops_on_bad_inputs(repo_root, tmp_path):
    pytest.importorskip("openmm")
    from pert.pipelines.md_panel import validate_panel_inputs

    scenarios = [("k2r", ["K2R"])]
    with pytest.raises(FileNotFoundError, match="reference PDB"):
        validate_panel_inputs(_panel_args(repo_root, reference_pdb=str(tmp_path / "missing.pdb")), scenarios)
    with pytest.raises(ValueError, match="Ligand residue 'XYZ'"):
        validate_panel_inputs(_panel_args(repo_root, ligand_resname="XYZ"), scenarios)
    with pytest.raises(ValueError, match="Scenario 'bad'"):
        validate_panel_inputs(_panel_args(repo_root), [("bad", ["A2R"])])

    # a reference with a numbering gap would silently shift every later position
    gapped = tmp_path / "gapped.pdb"
    lines = []
    for line in open(_panel_args(repo_root).reference_pdb):
        if line.startswith(("ATOM", "HETATM")) and line[17:20] not in ("ACE", "NH2", " ZN") and int(line[22:26]) > 10:
            line = f"{line[:22]}{int(line[22:26]) + 5:4d}{line[26:]}"
        if line.startswith("ENDMDL"):
            lines.append(line)
            break
        lines.append(line)
    gapped.write_text("".join(lines) + "END\n")
    with pytest.raises(ValueError, match="not numbered consecutively"):
        validate_panel_inputs(_panel_args(repo_root, reference_pdb=str(gapped)), scenarios)
