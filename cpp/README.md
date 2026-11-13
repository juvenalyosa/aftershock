# PERT C++ Port

This directory starts an incremental C++ port of the Python pipeline.

Current scope:

- `pert_mutator`: C++17 implementation of the sequence mutation and affine-gap alignment logic from `pert/sequence/mutations.py` and `pert/sequence/alignment.py`.
- `pert_te`: dependency-free C++17 numerical kernels for lagged VAR matrices, ridge regression, linear-Gaussian TE, and pairwise residue-flow TE proxies.
- `pert_mutator_cli`: small command-line entry point for applying mutations.
- `test_mutator` / `test_te`: parity and directionality checks.

Build:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

Example:

```bash
cpp/build/pert_mutator_cli AKST 1 K2R
```
