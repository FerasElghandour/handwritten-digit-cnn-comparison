# Reproduction and benchmark scope

The real 1,797-image digits comparison now trains both architectures for 30 epochs across three training seeds. On the fixed 360-image test split, baseline mean accuracy is **97.50%** and deep CNN mean accuracy is **98.89%**. Checkpoints are selected using validation loss, then tested once.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark | None |
| `train` | Configurable training on supplied fixed splits; default budget differs from named published reference protocols | data, config |
| `reference-training` | Exact published fresh-training budgets on identically prepared splits; CPU reference; checkpoint selection uses validation only | data |

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


## Device selection

`train.py --device cpu` is the default reference path. `--device mps` explicitly selects an Apple Silicon GPU; `--device cuda` requires a CUDA GPU. Unsupported or unavailable devices fail explicitly. No automatic CPU fallback is enabled. Metrics record the device, runtime and elapsed time. CPU and GPU runs are separate numerical protocols; equal seeds do not imply equal weights or results across devices. Resume requires the checkpoint's original device type and saves its accelerator RNG state.

For a generated-data device integration check, set `TRAIN_TEST_DEVICE=mps` when running `python test_training_pipeline.py`. This checks training and checkpoint reload, not real-data model quality.

## Recorded command definitions

### checks

Offline source and generated-data training, checkpoint selection and inference reload checks; not a real-data benchmark

```text
{python} scripts/validate_source.py
{python} test_training_pipeline.py
```

### train

Configurable training on supplied fixed splits; default budget differs from named published reference protocols

```text
{python} train.py --data {input:data} --config {input:config} --output {output}/training
```

### reference-training

Exact published fresh-training budgets on identically prepared splits; CPU reference; checkpoint selection uses validation only

```text
{python} train.py --data {input:data} --config baseline-config.json --epochs 30 --batch-size 32 --seed 42 --device cpu --output {output}/baseline-42
{python} train.py --data {input:data} --config baseline-config.json --epochs 30 --batch-size 32 --seed 43 --device cpu --output {output}/baseline-43
{python} train.py --data {input:data} --config baseline-config.json --epochs 30 --batch-size 32 --seed 44 --device cpu --output {output}/baseline-44
{python} train.py --data {input:data} --config deep-config.json --epochs 30 --batch-size 32 --seed 42 --device cpu --output {output}/deep-42
{python} train.py --data {input:data} --config deep-config.json --epochs 30 --batch-size 32 --seed 43 --device cpu --output {output}/deep-43
{python} train.py --data {input:data} --config deep-config.json --epochs 30 --batch-size 32 --seed 44 --device cpu --output {output}/deep-44
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.
