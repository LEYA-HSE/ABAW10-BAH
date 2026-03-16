# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**Authors:** Elena Ryumina, Alexandr Axyonov, Dmitry Sysoev, Timur Abdulkadirov, Kirill Almetov, Yulia Morozova, Dmitry Ryumin

This repository contains the code for Team LEYA's submission to the 10th ABAW Competition for the ambivalence / hesitation recognition task.

The project implements a multimodal classification pipeline that combines four modalities:

- face
- audio
- text
- scene

The current training and inference pipeline is built around pre-extracted unimodal features and a multimodal fusion model with an optional prototype-based auxiliary head.

Related paper:
- [arXiv:2603.12848](https://arxiv.org/abs/2603.12848)

## What Is In The Repository

- Feature exporters for face, audio, text, and scene modalities
- A multimodal dataset loader over serialized feature artifacts
- Fusion training with:
  - single-run training
  - grid search
  - exhaustive search
  - Optuna-based hyperparameter search
- Challenge feature preparation and challenge/evaluation inference scripts
- Support for single-checkpoint inference and ensemble inference

## Repository Layout

```text
.
+-- assets/                     # checkpoints and auxiliary model files
+-- data/                       # local CSVs and challenge metadata
+-- features/                   # extracted modality artifacts
+-- results/                    # training and inference outputs
+-- scripts/
|   +-- prepare_challenge_csv.py
|   +-- run_challenge_feature_prepare.py
|   '-- run_challenge_inference.py
+-- src/
|   +-- data_loading/
|   +-- exporters/
|   +-- models/
|   +-- utils/
|   '-- train.py
+-- config.toml
+-- config.challenge.toml
+-- search_params.toml
'-- main.py
```

## Environment

Python `3.12` was used in the current setup.

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is pinned to CUDA 12.4 PyTorch wheels.
- The repository assumes local checkpoints and local datasets are already available.
- Some modality checkpoints and feature assets are expected to be taken from the related modality-specific branches of the same project. In practice, face, audio, text, and scene assets should be searched for in their corresponding branches and prepared locally before running the full pipeline.

## Configuration

There are two main configs:

- [config.toml](./config.toml): training / local evaluation pipeline
- [config.challenge.toml](./config.challenge.toml): challenge feature preparation and challenge inference

The main sections are:

- `datasets.*`: dataset roots, CSV templates, split-specific paths
- `dataloader`: batch size, workers, shuffling, prepare-only mode
- `search`: training mode (`none`, `greedy`, `exhaustive`, `optuna`)
- `model`: multimodal fusion architecture
- `training`: optimizer, scheduler, early stopping, prototype-loss weights
- `multimodal`: active modalities and artifact root
- `face_export`, `audio_export`, `text_export`, `scene_export`: modality-specific export settings

## Data And Features

The pipeline operates on serialized feature artifacts stored under `features/`.

Expected artifact layout:

```text
features/
+-- face/<artifact_tag>/<split>.pkl
+-- audio/<artifact_tag>/<split>.pkl
+-- text/<artifact_tag>/<split>.pkl
'-- scene/<artifact_tag>/<split>.pkl
```

Each artifact is a single pickle file per split. The multimodal loader reads these artifacts and joins modalities by `sample_id`.

If a required artifact is missing, `main.py` will attempt to run the corresponding exporter automatically.

## Training

Run the main training pipeline:

```bash
python main.py
```

Behavior depends on `search.type` in [config.toml](./config.toml):

- `none`: one training run
- `greedy`: greedy hyperparameter search
- `exhaustive`: full grid search
- `optuna`: Optuna-based search

Outputs are written to:

```text
results/results_multimodal_pipeline_<timestamp>/
```

Typical contents:

- `session_log.txt`
- `config_copy.toml`
- `overrides.txt`
- `fusion_metrics.json`
- `checkpoints/` if `training.save_checkpoints = true`

## Challenge Feature Preparation

Challenge preparation is handled by:

- [scripts/prepare_challenge_csv.py](./scripts/prepare_challenge_csv.py)
- [scripts/run_challenge_feature_prepare.py](./scripts/run_challenge_feature_prepare.py)

Run:

```bash
python scripts/run_challenge_feature_prepare.py
```

What it does:

1. Builds the challenge CSV from the official split text file
2. Validates or mirrors precomputed audio features
3. Runs face, text, and scene exporters for the challenge split

Important:

- This script is configured through module-level constants inside the file, not command-line arguments.
- Before running it, check:
  - `CONFIG_PATH`
  - `SPLIT`
  - `RUN_FACE`
  - `RUN_TEXT`
  - `RUN_SCENE`
  - `AUDIO_PRECOMPUTED_SOURCE`

## Inference

Inference is handled by [scripts/run_challenge_inference.py](./scripts/run_challenge_inference.py).

Run:

```bash
python scripts/run_challenge_inference.py
```

This script supports two modes:

- `challenge_submit`
- `eval_metrics`

The mode is selected by editing `RUN_MODE` inside the script.

### Challenge Submission Mode

Set:

```python
RUN_MODE = "challenge_submit"
```

The script writes submission files under:

```text
results/challenge_submissions/<tag>_<suffix>_<timestamp>/
```

Generated files:

- `no_probabilities/trial-0.txt`
- `no_probabilities/trial-0.csv`
- `with_probabilities/trial-0.txt`
- `with_probabilities/trial-0.csv`
- `with_probabilities_hard/trial-0.txt`
- `with_probabilities_hard/trial-0.csv`
- `submission_meta.json`

### Evaluation Mode

Set:

```python
RUN_MODE = "eval_metrics"
```

The script evaluates one checkpoint or an ensemble on `dev` / `test` and writes outputs to:

```text
results/eval_inference/<tag>_<suffix>_<timestamp>/
```

Generated files:

- `dev_predictions.csv`
- `test_predictions.csv`
- `eval_metrics.json`

### Single Checkpoint vs Ensemble

The inference script supports:

- one checkpoint via `CHECKPOINT_PATH`
- multiple checkpoints via `CHECKPOINT_PATHS`

If `CHECKPOINT_PATHS` is non-empty, the script performs probability averaging across models.

Important:

- Like the challenge feature preparation script, this script is configured through module-level constants.
- Before running it, check:
  - `CONFIG_PATH`
  - `EVAL_CONFIG_PATH`
  - `CHECKPOINT_PATH`
  - `CHECKPOINT_PATHS`
  - `RUN_MODE`
  - `RUN_TAG`
  - `EVAL_TAG`

## Model Overview

The current multimodal pipeline works in two stages:

1. Train or load unimodal feature extractors for face, audio, text, and scene
2. Train a multimodal fusion model over the extracted modality embeddings

Supported fusion backbones currently include:

- `exchange_transformer`
- `videoformer`
- `concat_mlp`
- `attn`
- `class_weighted`

The main working configuration in this repository is the transformer-based multimodal fusion model defined in [src/models/fusion_model.py](./src/models/fusion_model.py).

Prototype support is optional and controlled by:

```toml
[model]
use_prototypes = true
```

When enabled, the prototype branch contributes an auxiliary training loss. Final predictions still come from the main classifier logits.

## Search And Model Selection

Hyperparameter search is configured in [search_params.toml](./search_params.toml).

The repository supports:

- standard grid search
- exhaustive search
- Optuna

Optuna can be configured to:

- persist trials to SQLite
- continue an existing study
- evaluate multiple random seeds for the same hyperparameter configuration
- disable checkpoint saving during search

## Practical Notes

- Paths in the provided configs are local and should be adapted to your machine.
- `config.toml` and `config.challenge.toml` are the source of truth for dataset roots and artifact locations.
- Some local folders such as `features/` and `results/` are intentionally not versioned.
- The repository is currently optimized for local experimentation rather than packaging as a reusable library.

## Minimal Workflow

### Train / Local Evaluation

```bash
python main.py
```

### Prepare Challenge Features

```bash
python scripts/run_challenge_feature_prepare.py
```

### Run Challenge Submission Inference

1. Open [scripts/run_challenge_inference.py](./scripts/run_challenge_inference.py)
2. Set `RUN_MODE = "challenge_submit"`
3. Set checkpoint path(s)
4. Run:

```bash
python scripts/run_challenge_inference.py
```

### Run `dev` / `test` Evaluation Inference

1. Open [scripts/run_challenge_inference.py](./scripts/run_challenge_inference.py)
2. Set `RUN_MODE = "eval_metrics"`
3. Set checkpoint path(s)
4. Run:

```bash
python scripts/run_challenge_inference.py
```
