# FraudLens V3 Step 4A full-data Random Forest

Mean AP on the completed folds changed from **0.38030** to **0.42012**, a change of +0.03983 (+10.47% relative).

Step 3A’s full-data Logistic Regression stays the project baseline. This experiment compares two Random Forest training populations. The source has 497 V2 predictors. The rejected Step 2 features remain in their archived experiment. Both Random Forest models score identical chronological validation rows. Development contains all data used by this comparison.

Both arms use 140 bootstrap trees, Gini impurity, depth 22, leaf minimum 3, square-root feature sampling, seed 42 and no class weights. The full-data run retained all classifier settings, including 6 tree workers, with one native library thread per worker to avoid nested parallelism.

For each fold, the full training prefix refits category frequencies, median missing-value fills, ANOVA rankings and scaling. The transformation selects 180 numeric and missingness inputs and delivers the original dense float32 representation. Column and row batches bound preprocessing temporaries. Validation uses the fitted transform. The original 16,000-row sample was recreated solely for preprocessing parity checks. The original fitted classifier and scores supply the comparator.

## Paired validation metrics

Decisions use cutoff `0.5`. Review budgets use the highest-scored 5% or 10% of each fold, rounded up with source order breaking ties. Means give equal weight to each completed fold.

| Fold | Arm | Actual training rows | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | sampled | 16,000 | 0.40164 | 0.82339 | 0.80854 | 0.20737 | 0.33008 | 0.96420 | 0.41639 | 0.52820 |
| 1 | full | 212,594 | 0.46014 | 0.84818 | 0.85651 | 0.25747 | 0.39592 | 0.96658 | 0.46948 | 0.57896 |
| 2 | sampled | 16,000 | 0.31829 | 0.72818 | 0.82635 | 0.16707 | 0.27795 | 0.96965 | 0.36602 | 0.45843 |
| 2 | full | 283,459 | 0.35127 | 0.73906 | 0.83387 | 0.20864 | 0.33376 | 0.97087 | 0.38499 | 0.47942 |
| 3 | sampled | 16,000 | 0.42096 | 0.79942 | 0.83960 | 0.25668 | 0.39316 | 0.96694 | 0.44133 | 0.52080 |
| 3 | full | 354,324 | 0.44896 | 0.78539 | 0.85635 | 0.31248 | 0.45788 | 0.96912 | 0.46906 | 0.56070 |
| Mean | sampled | 16,000 | 0.38030 | 0.78366 | 0.82483 | 0.21037 | 0.33373 | 0.96693 | 0.40791 | 0.50248 |
| Mean | full | 283,459 | 0.42012 | 0.79088 | 0.84891 | 0.25953 | 0.39585 | 0.96886 | 0.44117 | 0.53970 |
| Mean change | full − sampled | — | 0.03983 | 0.00721 | 0.02408 | 0.04916 | 0.06213 | 0.00193 | 0.03326 | 0.03722 |

| Fold | Change in AP | Relative AP change |
| --- | ---: | ---: |
| 1 | +0.05850 | +14.56% |
| 2 | +0.03298 | +10.36% |
| 3 | +0.02800 | +6.65% |

## Runtime, memory and model size

Training includes preprocessing and the tree fits. Model sizes are actual compressed Joblib files including the fitted preprocessing. Working set is the peak for each process. Archived V2 process peaks can include earlier trials in that process.

| Fold | Arm | Training (s) | Validation scoring (s) | Peak GiB | Checkpoint MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | sampled (archived) | 5.40 | 4.24 | 1.30 | 1.52 |
| 1 | full | 49.01 | 2.00 | 0.94 | 7.11 |
| 2 | sampled (archived) | 5.70 | 4.76 | 1.30 | 1.28 |
| 2 | full | 66.05 | 2.23 | 1.13 | 7.48 |
| 3 | sampled (archived) | 6.07 | 4.79 | 1.30 | 1.23 |
| 3 | full | 85.56 | 2.08 | 1.33 | 8.82 |
| Mean | sampled | 5.73 | 4.60 | 1.30 | 1.34 |
| Mean | full | 66.87 | 2.10 | 1.13 | 7.80 |

Full training took 200.61 s total, including 136.01 s in tree fitting. The largest working set was 1.33 GiB. The supervised run took 226.30 s including archived-score checks, inference, and checkpointing.

The sampled classifier replay matched all stored scores within `2.22e-16`. Its preprocessing probe matched within `0`. Reloading the full checkpoints reproduced the complete validation predictions within `2.22e-16`.

## Reproduce or resume

The runner fits one fold at a time with a 30-minute wall-clock budget and a 7.5 GB working-set guard. It verifies and reuses completed folds at the same output path. Use a fresh output path for a new independent run.

```powershell
Set-Location E:\fraudlens-v3
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$development = 'E:\fraudlens-v2\artifacts\v2'
$out = 'artifacts\v3\step4a-forest'
& $python -B -m src.v3.forest_experiment --development $development --output $out
& $python -B -m src.v3.forest_report --development $development --output $out --markdown 'docs\V3_STEP4A.md'
```
