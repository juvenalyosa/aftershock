# Fast Allostery Benchmarking

Use `python -m pert benchmark` to score many variants without production MD and
compare the fast allostery score to labels, measured effects, or external tools
such as FoldX/DDMut.

Use `python -m pert normalize-benchmark` first when the source table has different
column names, TSV format, external score columns, or a continuous effect that
needs to be converted into a binary label.

See `BENCHMARK_DATASETS.md` for dataset-specific guidance, including
ProteinGym, SKEMPI 2.0, and ProThermDB.

## Input CSV

Required:

- `name`
- `mutations` or `mutation_set`

Structure input, choose one:

- `wt_pdb` and `mut_pdb`
- `run_dir`
- leave structure columns empty and pass `--reference-pdb --template-pdb` so the
  script builds mutant structures first

Optional evaluation:

- `label`: binary `0/1`, where `1` means affected/allosteric/damaging
- `effect_score`: continuous measured effect; higher should mean stronger effect
- external comparator columns, for example `foldx_ddg`, `ddmut_score`

Mutation positions are interpreted in reference coordinates. Supported variant
types are SNP/SNPs, INS, DEL, DELINS/INSDEL, DUP, and combinations.

## Commands

Smoke test with existing toy structures:

```bash
python -m pert benchmark \
  --input-csv data/benchmarks/benchmark_smoke_examples.csv \
  --output-dir runs/fast_benchmark_smoke
```

Normalize a raw source table:

```bash
python -m pert normalize-benchmark \
  --input raw_dataset.csv \
  --output benchmark_normalized.csv \
  --name-column variant_id \
  --mutations-column mutation_spec \
  --wt-pdb-column wildtype_structure \
  --mut-pdb-column mutant_structure \
  --effect-source-column effect_value \
  --derive-label-from-effect \
  --label-threshold 1.0 \
  --label-direction abs-ge \
  --rename-column foldx_raw:foldx_ddg \
  --rename-column ddmut_raw:ddmut_score
```

Normalize ProteinGym-style colon-separated substitutions:

```bash
python -m pert normalize-benchmark \
  --input proteingym_assay.csv \
  --output proteingym_normalized.csv \
  --name-column mutant \
  --mutations-column mutant \
  --mutation-separators ',;:' \
  --effect-source-column DMS_score
```

Derive mutation specs from full reference/mutant sequences:

```bash
python -m pert normalize-benchmark \
  --input indel_assay.csv \
  --output indel_normalized.csv \
  --name-column mutant \
  --mutated-sequence-column mutant \
  --reference-sequence '<REFERENCE_SEQUENCE>'
```

Benchmark with external comparator columns:

```bash
python -m pert benchmark \
  --input-csv your_benchmark.csv \
  --output-dir runs/benchmark_results \
  --external-score-column foldx_ddg \
  --external-score-column ddmut_score
```

Benchmark while building structures:

```bash
python -m pert benchmark \
  --input-csv your_benchmark.csv \
  --output-dir runs/benchmark_results \
  --reference-pdb path/to/reference.pdb \
  --template-pdb path/to/template.pdb \
  --ligand-resname ZN \
  --num-models 1
```

## Outputs

- `benchmark_predictions.csv`: per-variant fast score, mutation sources, top affected residues
- `benchmark_summary.json`: ROC AUC/AP for binary labels, Pearson/Spearman for continuous effects, optional external-tool metrics
- `benchmark_summary.png`: score distribution and label/effect scatter
- `scans/<name>/`: full per-residue CSV/JSON, plot, and VMD script for each variant

Smoke examples:

- `data/benchmarks/benchmark_smoke_examples.csv`: unlabeled scoring smoke test
- `data/benchmarks/benchmark_raw_smoke_source.csv`: raw-table normalization smoke test with illustrative labels
- `data/benchmarks/benchmark_raw_smoke_normalized.csv`: generated canonical benchmark CSV
- `data/benchmarks/benchmark_sequence_derivation_smoke.csv`: colon separator and full-sequence derivation smoke test

Do not claim superiority over FoldX/DDMut from the smoke data. Use independent
experimental labels or measured effects, then compare on held-out proteins.
