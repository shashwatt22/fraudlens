# FraudLens V3 Step 2 behavioral feature experiment

**Keep the V2 control.** Mean AP changed by -0.00005, with gains in 2 of three folds. Mean ROC-AUC changed by +0.00061. The prespecified promotion rule requires a mean AP gain of at least 0.01, AP gains in at least two folds, and an AUC decline of at most 0.01.

The read-only V2 control was verified by reloading every checkpoint and matching every saved development score exactly. The sole treatment adds 13 numeric predictors to the original 497 columns and uses the same 320-round, depth-8 XGBoost fit, seed 42, 6 CPU threads, training-only category handling and three chronological folds. The training prefixes contain 212,594, 283,459 and 354,324 rows. Validation contains 70,865, 70,865 and 70,864 rows. The experiment ends after fold 3's validation block. The later decision-policy block and retrospective holdout were outside the experiment.

The entity remains `card1`, `addr1` and `round(day_minus_D1)`. The amount window includes `t − 604800` and excludes all of timestamp `t`. D-offset summaries use all earlier timestamps for that entity. Counts include rows with missing amounts. Means and population standard deviations use finite values only, with standard deviation zero for a single value. No history yields a zero seven-day count and missing moments. An incomplete key yields all missing values. A deviation needs both a current value and a prior mean.

## Rankings and decisions

Decisions use the original cutoff `0.5`. Review budgets use the top-scored 5% and 10% of each fold, rounded up with original order breaking ties. The mean equally weights the three folds.

| Fold | Arm | AP | ROC-AUC | Precision | Recall | F1 | Recall at 5% | Recall at 10% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | control | 0.65989 | 0.92728 | 0.87368 | 0.43829 | 0.58374 | 0.63603 | 0.77074 |
| 1 | treatment | 0.65816 | 0.92853 | 0.86211 | 0.43563 | 0.57880 | 0.63006 | 0.76742 |
| 2 | control | 0.56917 | 0.90346 | 0.90412 | 0.32728 | 0.48059 | 0.59161 | 0.70662 |
| 2 | treatment | 0.57010 | 0.90414 | 0.89869 | 0.33293 | 0.48587 | 0.58797 | 0.70904 |
| 3 | control | 0.65723 | 0.93877 | 0.86586 | 0.43659 | 0.58049 | 0.64559 | 0.77240 |
| 3 | treatment | 0.65787 | 0.93867 | 0.86867 | 0.44065 | 0.58470 | 0.64660 | 0.77579 |
| Mean | control | 0.62876 | 0.92317 | 0.88122 | 0.40072 | 0.54827 | 0.62441 | 0.74992 |
| Mean | treatment | 0.62871 | 0.92378 | 0.87649 | 0.40307 | 0.54979 | 0.62155 | 0.75075 |

## Paired entity-block uncertainty

The 1,000 valid bootstrap replicates used the same resampled block weights for both scorers and all folds. There were 83,314 complete entity blocks and 26,347 incomplete-key singleton blocks. Intervals describe uncertainty from these development observations with fixed model fits. They do not measure variation across new time periods or refits.

| Fold | Treatment minus control AP | 95% percentile interval |
| --- | ---: | --- |
| 1 | -0.00173 | [-0.00820, +0.00485] |
| 2 | +0.00093 | [-0.00537, +0.00684] |
| 3 | +0.00064 | [-0.00447, +0.00631] |
| Mean | -0.00005 | [-0.00369, +0.00319] |

## Entity and history strata

Seen and unseen use complete keys and membership in that fold's training prefix. Prior history uses the unchanged V2 strictly prior count, so unseen keys can gain history during validation. Groups across the two definitions overlap. The table shows row counts, prevalence and AP. The JSON includes AUC and all cutoff metrics by group.

| Fold | Group | Rows | Fraud rate | Control AP | Treatment AP |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | seen in fit | 27,675 | 2.11% | 0.74688 | 0.76418 |
| 1 | unseen in fit | 34,564 | 3.75% | 0.49105 | 0.48125 |
| 1 | missing key | 8,626 | 13.12% | 0.81781 | 0.81776 |
| 1 | prior history | 40,863 | 3.17% | 0.54945 | 0.54787 |
| 1 | complete key without history | 21,376 | 2.74% | 0.58961 | 0.58558 |
| 2 | seen in fit | 27,435 | 1.71% | 0.62945 | 0.62747 |
| 2 | unseen in fit | 35,848 | 3.30% | 0.35813 | 0.35855 |
| 2 | missing key | 7,582 | 10.87% | 0.81049 | 0.81582 |
| 2 | prior history | 41,982 | 2.82% | 0.44881 | 0.44843 |
| 2 | complete key without history | 21,301 | 2.22% | 0.40290 | 0.40834 |
| 3 | seen in fit | 29,647 | 1.63% | 0.69771 | 0.71314 |
| 3 | unseen in fit | 31,078 | 3.64% | 0.44146 | 0.43988 |
| 3 | missing key | 10,139 | 13.24% | 0.81187 | 0.80973 |
| 3 | prior history | 41,273 | 2.70% | 0.53576 | 0.54514 |
| 3 | complete key without history | 19,452 | 2.57% | 0.47678 | 0.47489 |

## New feature importance

New columns account for 2.26% of total training split gain on average. Split importance describes how the fitted models used the columns. Validation metrics measure whether adding them improved ranking.

| New feature | Splits across folds | Total split gain |
| --- | ---: | ---: |
| `card_address_anchor_prior_day_minus_D4_mean` | 605 | 6,669.9 |
| `card_address_anchor_prior_day_minus_D10_mean` | 566 | 6,203.3 |
| `card_address_anchor_prior_day_minus_D15_mean` | 505 | 5,902.1 |
| `card_address_anchor_prior_day_minus_D10_std` | 213 | 2,791.3 |
| `card_address_anchor_prior_day_minus_D15_std` | 254 | 2,478.4 |
| `card_address_anchor_prior_day_minus_D4_std` | 229 | 2,131.3 |
| `card_address_anchor_prior_amount_mean_7d` | 261 | 1,968.9 |
| `card_address_anchor_day_minus_D10_deviation` | 242 | 1,835.0 |
| `card_address_anchor_day_minus_D15_deviation` | 211 | 1,742.5 |
| `card_address_anchor_amount_deviation_7d` | 267 | 1,507.3 |
| `card_address_anchor_prior_count_7d` | 46 | 1,391.2 |
| `card_address_anchor_day_minus_D4_deviation` | 144 | 1,185.0 |
| `card_address_anchor_prior_amount_std_7d` | 162 | 1,096.9 |

## Runtime and memory

The 425,188-row float32 feature block took 0.88 s to build. Existing V2 columns and the added block are combined only in the current batch. Windows peak working set includes mapped pages and process overhead. A separate six-thread worker handles each fold.

| Fold | Treatment fit (s) | Treatment inference (s) | Fit and score peak (GiB) | Worker peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 43.57 | 0.58 | 2.25 | 2.25 |
| 2 | 60.31 | 0.61 | 2.75 | 2.75 |
| 3 | 274.05 | 0.68 | 3.29 | 3.29 |
| Total time / maximum memory | 377.93 | 1.87 | 3.29 | 3.29 |

The three fold workers took 390.48 s in total, including checkpointing and control verification. The paired bootstrap took 30.08 s.

The archived V2 control run recorded 540.12 s of fitting, 7.98 s of scoring and a 3.50 GiB process high-water mark. These measurements came from the original V2 run and its process scope.

## Reproduce or resume

This repository keeps the cache, model checkpoints, validation predictions and reports under `artifacts/v3/step2-behavior`. Rerunning an intact completed fold verifies and reuses its saved files. A fresh output directory starts a new treatment run.

```powershell
Set-Location E:\fraudlens-v3
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '6'
$env:OPENBLAS_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$dev = 'E:\fraudlens-v2\artifacts\v2'
$out = 'artifacts\v3\step2-behavior'
foreach ($fold in 1..3) {
    & $python -B -m src.v3.behavior_experiment --development $dev --output $out --fold $fold
    if ($LASTEXITCODE -ne 0) { throw ('Failed fold ' + $fold) }
}
& $python -B -m src.v3.behavior_report --development $dev --output $out --markdown 'docs\V3_STEP2.md'
```
