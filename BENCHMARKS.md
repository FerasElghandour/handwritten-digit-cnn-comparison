# Reproduction and benchmark scope

The real 1,797-image digits comparison now trains both architectures for 30 epochs across three training seeds. On the fixed 360-image test split, baseline mean accuracy is **97.50%** and deep CNN mean accuracy is **98.89%**. Checkpoints are selected using validation loss, then tested once.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Source parsing only; not model training or behavior verification | None |
| `train` | New reference training protocol on supplied fixed splits; validation-selected checkpoint and held-out metrics | data, config |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

This new checkpoint-selection protocol differs from the retained final-epoch notebook results. Three seeds describe this small dataset and budget, not universal model superiority.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

The real sklearn digits dataset is bundled with scikit-learn; no external image download is needed. The preparer fixes stratified train/validation/test indices.

```sh
python prepare_data.py --output data/digits.npz
python train.py --data data/digits.npz --config baseline-config.json --epochs 30 --batch-size 32 --seed 42 --output runs/baseline-42
python train.py --data data/digits.npz --config deep-config.json --epochs 30 --batch-size 32 --seed 42 --output runs/deep-42
```

Repeat for seeds 43 and 44 without changing prepared splits. `results/multiseed-reference.json` retains all six runs. The architecture config files are supplied. Exported `inference.pt` removes optimizer state; `best.pt` supports validation-selected resume.

## Recorded command definitions

### checks

```text
{python} scripts/validate_source.py
{python} test_training_pipeline.py
```

### train

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
