# Experiments

These notebooks are an **iteration archive**, not a packaged library. Every notebook is the same
pipeline with one component changed and re-run as a separate experiment:

```
raw HDF5  ->  windowed shards (acc, gyr, emg, joints)  ->  multi-branch teacher  ->  joint angles
              (DataSharder)        (ImuJointPairDataset)   (Encoder_1 / Encoder_2 + GatingModule)
```

Evaluation is leave-one-subject-out over 13 subjects. The authoritative, aggregated results are in
[`../results/`](../results/) (pulled from Weights & Biases); the numbers printed inside individual
notebooks are single-run leftovers and are not authoritative.

**Start here:**
- [`Teacher_Regression_Benchmarks/regression_benchmark.ipynb`](./Teacher_Regression_Benchmarks/regression_benchmark.ipynb)
  — the cleanest cross-subject benchmark.
- [`Vanilla_SD_Experiments/Vanilla_SD.ipynb`](./Vanilla_SD_Experiments/Vanilla_SD.ipynb)
  — the full teacher to 1-4 IMU student sensor-distillation pipeline.

---

## Teacher_Regression_Benchmarks (19 notebooks)

The core cross-subject benchmark and most of the subject-invariance battery.

- `regression_benchmark.ipynb` — baseline multi-branch teacher (start here).
- `regression_benchmark_normalizebysubject.ipynb`, `_normalizebysubject_withstudents.ipynb` — per-subject normalization.
- `regression_benchmark_ssl.ipynb`, `_ssl_100reconstructepochs.ipynb` — self-supervised reconstruction pretraining.
- `regression_benchmark_prototypical.ipynb` — prototypical-network metric learning.
- `regression_benchmark_fml*.ipynb` (6) — feature-modulation variants (radians, finetune, frozen, averaged joints).
- `regression_benchmark_onlyimus.ipynb` — sEMG-removed ablation.
- `teacher_forecast*.ipynb` (6) — forecasting variant (predict future joint angles), including gradient-reversal and MMD versions.

## Adversarial_Training_Clustering (5 notebooks)

Adversarial subject-invariance (DANN-style subject classifier) and subject-structure analysis.

- `adversarial_training.ipynb`, `_increase_lambda.ipynb`, `_increase_lambda_normalize.ipynb` — adversary-weight sweep.
- `adversarial_training_nohandimu.ipynb` — drop the hand IMU (sensor ablation).
- `clustering_subjects_exploration.ipynb` — DBSCAN/HDBSCAN clustering of subjects in sensor space.

## OpenSim_Experiments (11 notebooks)

The same model evaluated on an alternate OpenSim joint-label set; clearest illustration of the
within- versus cross-subject gap.

- `new_opensimmodel_first_run_all_subjects*.ipynb` (4) — within-subject (subjects pooled).
- `new_opensimmodel_first_run_no_subject*.ipynb` (4) — held-out subject 5 (cross-subject), including dropout, no-overlap, and OHME/GHME label variants.
- `*_crossattention_convolution.ipynb` — cross-attention + convolution backbone.
- `plot_all_models.ipynb`, `FIX_GRAPH_*.ipynb` — figure aggregation utilities.

## Diffusion_Models (7 notebooks)

Alternative backbones.

- `regression_benchmark_diffusion.ipynb`, `regression_benchmark_dissusion_policy*.ipynb` (5) — diffusion-policy-style regression (angle loss, step count, temporal variants).
- `regression_benchmark_mamba.ipynb` — Mamba state-space backbone.
- `timesereis_diffusion.ipynb` — a diffusion proof-of-concept on synthetic data.

## Vanilla_SD_Experiments (17 notebooks)

Baseline teacher plus the sensor-distillation thread (teacher to 1/2/3/4-IMU students with combined
knowledge-and-sensor distillation), and loss/preprocessing/grid-search variants.

- `Vanilla_SD.ipynb`, `Vanilla_SD (1).ipynb` — full teacher-and-students distillation pipeline.
- `Vanilla_SD_LNN_Teacher*.ipynb`, `Vanilla_SD_LNN_TeacherandStudents.ipynb` — liquid-net teacher variants.
- `Vanilla_SD_csv_RMSE.ipynb`, `_NRMSE.ipynb`, `_customLoss*.ipynb`, `_smoothedjoints.ipynb`, `_singlejoint.ipynb` — loss and target variants.
- `gridsearch*.ipynb`, `Vanilla_SD_csv_RMSE_gridsearchtraining*.ipynb` — hyperparameter sweeps.
- `Vanilla_SD_csv_RMSE_different_test_subject.ipynb` — held-out-subject probe.

## LNN_SD_Models (3 notebooks)

Liquid neural network (LTC / CfC via `ncps`) teacher and distillation.

- `Vanilla_LNNipynb.ipynb` — standalone liquid-net model.
- `LNN+Dense keras.ipynb` — Keras LTC + dense variant.
- `LNNTEACHER_SD_csv_customLoss.ipynb` — liquid-net teacher in the distillation pipeline.
