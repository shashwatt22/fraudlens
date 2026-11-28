# FraudLens V3 Step 7C retrospective evaluation

This **reused retrospective holdout** contains 118,108 transactions and 4,064 frauds (3.441%) from transaction seconds 12,192,900 to 15,811,131. V1 and V2 previously evaluated these observations. The results describe historical performance on this fixed population.

Depth-10 XGBoost is the frozen V3 primary classifier. The Step 7A checkpoints and Step 7B thresholds were reused, including XGBoost identity calibration. No classifier, scaling, encoding, calibration or threshold fitting took place. All four V3 models scored identical rows before the label-access receipt was created.

## Original-score rankings and probability losses

AP, ROC-AUC and review budgets use original ranking scores. Recall at each budget reviews the highest scores, rounds up to whole rows and breaks ties in source order. Brier and log loss use the original probability outputs or score, which also equals the policy value under identity calibration. V2's weighted ensemble score remains uncalibrated.

| Model | AP | ROC-AUC | Brier | Log loss | Recall at 1% | Recall at 5% | Recall at 10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 Logistic Regression | 0.18280 | 0.81346 | 0.03669 | 0.33618 | 0.06127 | 0.38361 | 0.49680 |
| V3 Random Forest | 0.36335 | 0.75799 | 0.02607 | 0.11690 | 0.23720 | 0.40231 | 0.50172 |
| V3 depth-10 XGBoost | 0.56989 | 0.91798 | 0.02088 | 0.08653 | 0.25910 | 0.61171 | 0.73942 |
| V3 PyTorch MLP | 0.29646 | 0.78810 | 0.03048 | 0.12658 | 0.16806 | 0.38533 | 0.51304 |
| V1 XGBoost | 0.18778 | 0.79714 | 0.14666 | 0.46511 | 0.10310 | 0.34154 | 0.45743 |
| V2 weighted ensemble | 0.51147 | 0.90325 | 0.02240 | 0.09297 | 0.25418 | 0.55955 | 0.68873 |

## Default and frozen operating points

Confusion matrices give `[TN, FP]` and `[FN, TP]` rows. The V1 and V2 frozen results use their respective archived cutoffs. Their transaction IDs, timestamps, and frozen decisions were checked against the archived predictions.

| Model | Point | Threshold | Precision | Recall | F1 | Accuracy | Confusion matrix | Reviews | Reviews / 1,000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| V3 Logistic Regression | Default | 0.5 | 0.36988 | 0.23573 | 0.28795 | 0.95988 | `[[112412, 1632], [3106, 958]]` | 2,590 | 21.9 |
| V3 Logistic Regression | Frozen | 0.2944795427 | 0.32366 | 0.31275 | 0.31811 | 0.95386 | `[[111388, 2656], [2793, 1271]]` | 3,927 | 33.2 |
| V3 Random Forest | Default | 0.5 | 0.79920 | 0.24582 | 0.37599 | 0.97192 | `[[113793, 251], [3065, 999]]` | 1,250 | 10.6 |
| V3 Random Forest | Frozen | 0.1963565625 | 0.44726 | 0.35162 | 0.39372 | 0.96274 | `[[112278, 1766], [2635, 1429]]` | 3,195 | 27.1 |
| V3 depth-10 XGBoost | Default | 0.5 | 0.83055 | 0.36663 | 0.50871 | 0.97563 | `[[113740, 304], [2574, 1490]]` | 1,794 | 15.2 |
| V3 depth-10 XGBoost | Frozen | 0.1938110143 | 0.60864 | 0.50591 | 0.55254 | 0.97181 | `[[112722, 1322], [2008, 2056]]` | 3,378 | 28.6 |
| V3 PyTorch MLP | Default | 0.5 | 0.47473 | 0.23573 | 0.31503 | 0.96473 | `[[112984, 1060], [3106, 958]]` | 2,018 | 17.1 |
| V3 PyTorch MLP | Frozen | 0.2504911423 | 0.34533 | 0.32579 | 0.33527 | 0.95555 | `[[111534, 2510], [2740, 1324]]` | 3,834 | 32.5 |
| V1 XGBoost | Default | 0.5 | 0.11032 | 0.60974 | 0.18684 | 0.81738 | `[[94061, 19983], [1586, 2478]]` | 22,461 | 190.2 |
| V1 XGBoost | Frozen | 0.7483894229 | 0.22920 | 0.34916 | 0.27674 | 0.93720 | `[[109272, 4772], [2645, 1419]]` | 6,191 | 52.4 |
| V2 weighted ensemble | Default | 0.5 | 0.77842 | 0.33366 | 0.46710 | 0.97380 | `[[113658, 386], [2708, 1356]]` | 1,742 | 14.7 |
| V2 weighted ensemble | Frozen | 0.2752931571 | 0.62005 | 0.43529 | 0.51149 | 0.97139 | `[[112960, 1084], [2295, 1769]]` | 2,853 | 24.2 |

## Primary model compared with earlier versions

| Archive | V3 AP change | V3 ROC-AUC change | V3 frozen F1 change |
| --- | ---: | ---: | ---: |
| V1 XGBoost | +0.38211 | +0.12083 | +0.27580 |
| V2 weighted ensemble | +0.05842 | +0.01472 | +0.04105 |

## Entity-key performance

Seen card–address–D1 proxy keys occurred in the relevant V3 model's fit prefix. Unseen keys are complete keys absent from that prefix. PyTorch uses its gradient-fit cutoff. Missing keys form a separate group. The JSON includes metrics for each group at both operating points.

| Model | Entity group | Rows | Fraud prevalence | AP | ROC-AUC | Frozen precision | Frozen recall | Frozen F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 Logistic Regression | seen in fit | 42,291 | 1.54% | 0.04597 | 0.76363 | 0.13433 | 0.01380 | 0.02503 |
| V3 Logistic Regression | unseen in fit | 63,583 | 2.81% | 0.07928 | 0.74155 | 0.13297 | 0.13782 | 0.13535 |
| V3 Logistic Regression | missing key | 12,234 | 13.30% | 0.53886 | 0.84975 | 0.50547 | 0.62446 | 0.55870 |
| V3 Random Forest | seen in fit | 42,291 | 1.54% | 0.04516 | 0.64042 | 0.18750 | 0.01380 | 0.02571 |
| V3 Random Forest | unseen in fit | 63,583 | 2.81% | 0.18313 | 0.65050 | 0.30332 | 0.15854 | 0.20824 |
| V3 Random Forest | missing key | 12,234 | 13.30% | 0.66319 | 0.86351 | 0.51355 | 0.69883 | 0.59203 |
| V3 depth-10 XGBoost | seen in fit | 42,291 | 1.54% | 0.79070 | 0.96592 | 0.95538 | 0.55828 | 0.70474 |
| V3 depth-10 XGBoost | unseen in fit | 63,583 | 2.81% | 0.33082 | 0.83803 | 0.40381 | 0.27283 | 0.32564 |
| V3 depth-10 XGBoost | missing key | 12,234 | 13.30% | 0.75420 | 0.92658 | 0.67281 | 0.74063 | 0.70509 |
| V3 PyTorch MLP | seen in fit | 38,003 | 1.38% | 0.02932 | 0.59756 | 0.00000 | 0.00000 | 0.00000 |
| V3 PyTorch MLP | unseen in fit | 67,871 | 2.82% | 0.16092 | 0.71855 | 0.16587 | 0.18087 | 0.17304 |
| V3 PyTorch MLP | missing key | 12,234 | 13.30% | 0.61650 | 0.84352 | 0.56794 | 0.60111 | 0.58405 |

## Curves

[ROC](v3_step7c_roc.png)
[Precision–recall](v3_step7c_precision_recall.png)
[Calibration](v3_step7c_calibration.png)

## Inference and integrity

| Model | Load and prepare (s) | Inference (s) | Sampled process working set peak (GiB) |
| --- | ---: | ---: | ---: |
| V3 Logistic Regression | 0.01 | 2.69 | 2.49 |
| V3 Random Forest | 0.18 | 2.98 | 2.54 |
| V3 depth-10 XGBoost | 0.39 | 1.76 | 2.65 |
| V3 PyTorch MLP | 0.01 | 1.42 | 2.62 |

Feature build and audit took 37.18 s. The scoring-stage process peak, including feature preparation, was 4.04 GiB. Per-model measurements are sampled process working sets, including memory-mapped pages.

The seven inherited causal history columns were independently reconstructed. Appending holdout covariates left the 472,432 development rows unchanged in both feature matrices. The V2 builder calculates histories using strictly earlier timestamp groups. Forecasts and the feature matrix were hashed before opening labels. The four checkpoint, policy and manifest hashes matched before and after scoring and evaluation.

## Reproduce or render

Use a fresh directory for a separate retrospective evaluation. `evaluate` requires precomputed scores and creates an exclusive label-access receipt. Regenerate plots or the report from the saved metrics and curves without reopening labels. The local per-row predictions and mapped features remain under the ignored artifact directory. Use `dashboard.json` for the aggregate metrics and display curves. `results.json` records full scalar precision and evaluation lineage.

```powershell
Set-Location E:\fraudlens-v3
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '6'
$env:OPENBLAS_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$out = 'artifacts\v3\step7c-retrospective-reproduction'
$args7c = @('--raw','E:\fraudlens\data\raw','--v1','E:\fraudlens\artifacts\full',
    '--development','E:\fraudlens-v2\artifacts\v2','--refit','artifacts\v3\step7a-refit',
    '--policy','artifacts\v3\step7b-policy','--output',$out)
& $python -B -m src.v3.retrospective score @args7c
& $python -B -m src.v3.retrospective evaluate @args7c
& $python -B -m src.v3.retrospective_report --output $out --markdown (Join-Path $out 'V3_STEP7C.md')
& $python -B -m pytest -q tests/test_v3_retrospective.py
```
