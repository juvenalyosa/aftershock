# Benchmark Dataset Choices

The fast scanner predicts a structural/contact allostery proxy. Benchmark it in
tiers rather than treating every mutation table as the same target.

## Tier 1: Function And Indel Coverage

ProteinGym is useful for broad mutation-effect testing because it includes DMS
substitution and indel benchmarks. Its substitution rows may use colon-separated
multi-mutants such as `A1P:D2N`; use:

```bash
python -m pert normalize-benchmark \
  --input proteingym_assay.csv \
  --output proteingym_normalized.csv \
  --name-column mutant \
  --mutations-column mutant \
  --mutation-separators ',;:' \
  --effect-source-column DMS_score
```

For DMS assays where higher score means higher fitness, damaging/effect labels
usually need inverted logic. For example, use `--effect-mode neg` or derive a
label with `--label-direction le`, depending on the assay scale.

ProteinGym indel files can provide full mutant sequences instead of compact
mutation specs. Use sequence-derived normalization:

```bash
python -m pert normalize-benchmark \
  --input proteingym_indel_assay.csv \
  --output proteingym_indel_normalized.csv \
  --name-column mutant \
  --mutated-sequence-column mutant \
  --reference-sequence '<REFERENCE_SEQUENCE>' \
  --effect-source-column DMS_score
```

## Tier 2: Structure-Resolved Binding Effects

SKEMPI 2.0 is useful for structure-resolved protein-protein binding mutation
effects. It is a fairer place to compare against FoldX-style energy methods than
pure allostery labels, because it has PDB-backed complexes and experimental
binding-energy changes.

## Tier 3: Protein Stability

ProThermDB is useful for stability calibration. It should be treated as a
stability benchmark, not direct allostery evidence.

## Tier 4: True Allostery

For the final claim, use curated protein-specific allostery datasets where
mutation effects are measured at a distal active/binding/regulatory site. The
benchmark CSV should include:

- direct mutation specs
- WT/MUT structures or enough information to build them
- distal site definition through `ligand_resname` or structure annotations
- binary allosteric label or continuous activity/binding shift

The smoke files in this repository are only pipeline tests. They are not
scientific evidence that the model is better than FoldX/DDMut.
