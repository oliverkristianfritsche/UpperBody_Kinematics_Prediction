# Cross-subject generalization — method comparison

All numbers below are aggregated directly from the **Weights & Biases** project
`oliverkristianfritsche-university-of-central-florida/ultra_mocap` (478 finished runs).
Each method is a **leave-one-subject-out (LOSO)** sweep over the 13 subjects; we report the mean
over held-out subjects. Targets are three right-arm joint angles (`elbow_flex_r`, `arm_flex_r`,
`arm_add_r`), in **degrees**.

- **Within-subject** = `val` split (held-out trials of the *training* subjects).
- **Cross-subject** = `test` split (the held-out **subject**).

> Numbers are pulled from committed W&B run summaries — not re-estimated. The full 33-group
> battery (every ablation, including the ones omitted here) is in [`comparison.csv`](./comparison.csv).

## The gap (baseline model)

The dual-encoder teacher predicts seen subjects almost perfectly but degrades sharply on an unseen subject:

| Joint | Within-subject RMSE° / PCC | Cross-subject RMSE° / PCC |
|---|---|---|
| elbow_flex_r | 4.68 / 0.99 | 15.30 / 0.71 |
| arm_flex_r | 7.45 / 0.99 | 20.26 / 0.87 |
| arm_add_r | 3.98 / 0.98 | 9.37 / 0.69 |
| **Overall** | **5.06 / 0.99** | **15.40 / 0.72** |

## Method comparison (LOSO, overall over 3 joints)

| Method (W&B group) | Folds | Within-subject RMSE° / PCC | Cross-subject RMSE° / PCC |
|---|---|---|---|
| Baseline (dual-encoder teacher, RMSE) | 13 | 5.06 / 0.99 | **15.40 / 0.72** |
| Smoothed ground-truth joints | 13 | 5.32 / 0.99 | 14.62 / 0.76 |
| Gradient reversal (DANN) | 13 | 5.69 / 0.98 | 14.80 / 0.77 |
| Gradient reversal (DANN), higher penalty | 13 | 6.17 / 0.98 | 14.81 / 0.77 |
| **DANN architecture, reversal turned OFF (control)** | 13 | 5.43 / 0.99 | **14.49 / 0.77** |
| IMU-only (drop EMG) | 13 | 5.46 / 0.99 | 14.76 / 0.76 |
| Learnable per-output bias | 13 | 5.36 / 0.99 | 14.76 / 0.76 |
| Baseline, longer window (wl200) | 7 | 5.63 / 0.99 | 18.74 / 0.84 |
| Temporal-conv, single-timestep | 8 | 2.66 / 1.00 | 16.53 / 0.57 |
| L1 weight regularization (λ=1e-3) | 13 | 23.75 / 0.58 | 23.55 / 0.40 |
| iTransformer, single-timestep | 2 | 37.55 / 0.34 | 31.45 / 0.05 |

## Takeaways

1. **Within-subject is effectively solved; cross-subject is not.** ~5° / PCC 0.99 within-subject vs
   ~15° / PCC 0.72 on a held-out subject — a 3× RMSE gap that every method here inherits.
2. **Domain adaptation does not meaningfully close the gap.** The best subject-invariance variants land at
   ~14.5–14.8° — under 1° better than the 15.4° baseline. Crucially, the **gradient-reversal "OFF" control
   matches the "ON" version**, so the DANN *mechanism* isn't what helps; the small gain comes from the
   incidental architecture/training change.
3. **EMG adds little for cross-subject transfer.** IMU-only (14.76°) is within noise of the full IMU+EMG rig —
   relevant to the "deploy with a minimal wearable" goal.
4. **Heavier regularization and a transformer backbone hurt.** L1 degrades monotonically with λ, and the
   iTransformer fails to fit (one LOSO fold diverged).
5. **Per-joint:** `arm_flex_r` is the hardest to transfer (~20° RMSE despite PCC 0.87 — a correlated but
   offset error), `arm_add_r` the easiest (~9°).

## Notes & caveats

- Some groups in `comparison.csv` are intentionally omitted above: an **expanded-joint-set** run
  (`ekf_morejoints`, 13.4° — not comparable, different targets), several EKF/normalization/target-format
  ablations, and the full L1-λ sweep.
- Six groups (128 runs) are a **separate sensor inpainting / gap-fill task** (`*_inpainting*`,
  `DiffusionGapFill`), not joint-angle LOSO; they are tagged as such in `comparison.csv` and excluded here.
- Fold counts vary (most 13; a few 7–8) where a sweep was not run for every subject.
- A handful of group-name tokens (e.g. `feyman`) are described literally and flagged for confirmation.
