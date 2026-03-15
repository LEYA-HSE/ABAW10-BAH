# Project Structure

This repository contains the code, training pipeline, and inference scripts for the project.

## Main Files

- `run_training.py` — contains the main model and training configuration parameters.
- `inference.py` — contains the inference pipeline for running the model from a saved checkpoint.
- `best_1.pt` — checkpoint of the best-performing audio Mamba model.

## Project Folders

- `datasets/` — code for loading and processing dataset embeddings.
- `losses/` — implementations of loss functions used during training.
- `metrics/` — code for computing evaluation metrics.
- `models/` — different trainable model architectures.
- `training/` — hyperparameter tuning utilities and training-related code.
