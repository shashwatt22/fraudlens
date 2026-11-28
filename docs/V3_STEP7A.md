# FraudLens V3 Step 7A final refit

The selected Step 3A Logistic Regression, Step 4A Random Forest, Step 5A depth-10 XGBoost, and Step 6A 224-input MLP were refit. Step 2 and Step 6B inputs are excluded. All four checkpoints were saved and verified before policy scoring. The reserved block had been inspected in earlier work, so these are development diagnostics.

## Chronological populations

The times below are inclusive endpoints in `TransactionDT` seconds. Each boundary keeps equal timestamps together.

| Population | Rows | Fraud rows | First second | Last second |
| --- | ---: | ---: | ---: | ---: |
| pre policy | 425,188 | 14,945 | 86,400 | 10,766,686 |
| torch gradient | 382,669 | 13,098 | 86,400 | 9,582,396 |
| torch internal | 42,519 | 1,847 | 9,582,398 | 10,766,686 |
| policy | 47,244 | 1,654 | 10,766,698 | 12,192,842 |
| earlier policy | 23,622 | 824 | 10,766,698 | 11,449,871 |
| later policy | 23,622 | 830 | 11,449,894 | 12,192,842 |

The earlier policy portion combines the two earlier V2 policy quarters. The later portion is the original verification half. No policy observations were used for fitting, model selection, threshold selection or calibration in this step.

## Frozen model configurations

| Model | Gradient or estimator rows | Preprocessing and estimator |
| --- | ---: | --- |
| Logistic Regression | 425,188 | V2 log-frequency, median fill, missing flags and scaling. ANOVA top 192 plus bounded one-hot encoding. Unweighted L2 (`C=0.08`), L-BFGS, tolerance `1e-4`, cap 1,000 iterations. Converged in 272 iterations: True. |
| Random Forest | 425,188 | V2 log-frequency, median fill, missing flags and scaling. ANOVA top 180. Unweighted, 140 bootstrap trees, depth 22, leaf minimum 3, Gini, square-root feature sampling, seed 42. |
| XGBoost refit | 425,188 | Original 497 V2 columns, missing values and native categoricals. Depth 10, 320 rounds, learning rate 0.055, L2 4, max bin 128, row sampling 0.85, column sampling 0.8, no class weighting, seed 42. |
| PyTorch MLP | 382,669 | Gradient-only V2 transform with ANOVA top 224. 224 → 192 → 96 → 48 → 1, BatchNorm, GELU, dropout 0.18, unweighted BCE, AdamW (0.0007, decay 0.002), cosine schedule, seed 42. Selected epoch 3 of 7 by internal AP. |

## Reserved policy metrics

Decision metrics use the fixed 0.5 threshold. Budgets rank within each window and keep source order for tied scores.

| Window | Model | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 1% | Recall at 5% | Recall at 10% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| policy | Logistic Regression | 0.35770 | 0.83540 | 0.68097 | 0.22068 | 0.33333 | 0.96910 | 0.20617 | 0.42987 | 0.56106 |
| policy | Random Forest | 0.35292 | 0.74490 | 0.79140 | 0.22249 | 0.34733 | 0.97073 | 0.22612 | 0.40145 | 0.49940 |
| policy | XGBoost depth 10 refit | 0.68854 | 0.94638 | 0.88228 | 0.41233 | 0.56201 | 0.97750 | 0.27388 | 0.71342 | 0.83434 |
| policy | PyTorch MLP (224 inputs) | 0.36995 | 0.79897 | 0.81114 | 0.20254 | 0.32414 | 0.97043 | 0.22491 | 0.42805 | 0.54051 |
| policy | XGBoost depth 10 fold 3 | 0.58893 | 0.92403 | 0.83750 | 0.36457 | 0.50800 | 0.97528 | 0.25635 | 0.62152 | 0.73942 |
| earlier policy | Logistic Regression | 0.38521 | 0.84605 | 0.66272 | 0.27184 | 0.38554 | 0.96977 | 0.22573 | 0.44782 | 0.58495 |
| earlier policy | Random Forest | 0.39271 | 0.75208 | 0.79412 | 0.26214 | 0.39416 | 0.97189 | 0.23786 | 0.45146 | 0.53519 |
| earlier policy | XGBoost depth 10 refit | 0.72166 | 0.95043 | 0.87583 | 0.47937 | 0.61961 | 0.97947 | 0.28155 | 0.73665 | 0.84951 |
| earlier policy | PyTorch MLP (224 inputs) | 0.40872 | 0.81587 | 0.79528 | 0.24515 | 0.37477 | 0.97147 | 0.23665 | 0.47573 | 0.57646 |
| earlier policy | XGBoost depth 10 fold 3 | 0.61077 | 0.92480 | 0.84653 | 0.41505 | 0.55700 | 0.97697 | 0.26578 | 0.64684 | 0.75728 |
| later policy | Logistic Regression | 0.33220 | 0.82443 | 0.71212 | 0.16988 | 0.27432 | 0.96842 | 0.18554 | 0.41566 | 0.53494 |
| later policy | Random Forest | 0.31277 | 0.73597 | 0.78756 | 0.18313 | 0.29717 | 0.96956 | 0.20602 | 0.36024 | 0.46747 |
| later policy | XGBoost depth 10 refit | 0.65521 | 0.94256 | 0.89130 | 0.34578 | 0.49826 | 0.97553 | 0.26506 | 0.69759 | 0.81928 |
| later policy | PyTorch MLP (224 inputs) | 0.33304 | 0.78210 | 0.83648 | 0.16024 | 0.26896 | 0.96939 | 0.20602 | 0.38313 | 0.49398 |
| later policy | XGBoost depth 10 fold 3 | 0.56799 | 0.92344 | 0.82595 | 0.31446 | 0.45550 | 0.97358 | 0.25060 | 0.60120 | 0.72048 |

## Depth-10 recency comparison

The existing fold-3 checkpoint fit 354,324 rows. The refit adds 70,864 more recent pre-policy rows. The original external fold-3 scores matched exactly before using this checkpoint on the same policy rows as the new model.

| Window | AP change | ROC-AUC change |
| --- | ---: | ---: |
| policy | +0.09962 | +0.02235 |
| earlier policy | +0.11089 | +0.02562 |
| later policy | +0.08722 | +0.01912 |

## Entity proxies

A complete card–address–D1 key is seen if it appeared among the model's fitting rows. Missing keys form a separate group. PyTorch and fold-3 XGBoost have smaller fit prefixes, so their group memberships differ. The JSON includes all metrics for each stratum.

| Window | Model | Seen rows | Seen AP | Unseen rows | Unseen AP | Missing-key AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| policy | Logistic Regression | 22,499 | 0.09988 | 19,255 | 0.22715 | 0.63125 |
| policy | Random Forest | 22,499 | 0.09694 | 19,255 | 0.18343 | 0.67278 |
| policy | XGBoost depth 10 refit | 22,499 | 0.79498 | 19,255 | 0.46821 | 0.82369 |
| policy | PyTorch MLP (224 inputs) | 19,525 | 0.07439 | 22,229 | 0.23297 | 0.65022 |
| policy | XGBoost depth 10 fold 3 | 18,087 | 0.66610 | 23,667 | 0.39752 | 0.78191 |
| earlier policy | Logistic Regression | 11,859 | 0.10981 | 9,054 | 0.25458 | 0.64964 |
| earlier policy | Random Forest | 11,859 | 0.11280 | 9,054 | 0.21138 | 0.70560 |
| earlier policy | XGBoost depth 10 refit | 11,859 | 0.78398 | 9,054 | 0.49448 | 0.85810 |
| earlier policy | PyTorch MLP (224 inputs) | 10,128 | 0.10489 | 10,785 | 0.26354 | 0.67267 |
| earlier policy | XGBoost depth 10 fold 3 | 9,376 | 0.66769 | 11,537 | 0.39968 | 0.80742 |
| later policy | Logistic Regression | 10,640 | 0.09193 | 10,201 | 0.20339 | 0.61958 |
| later policy | Random Forest | 10,640 | 0.08142 | 10,201 | 0.15934 | 0.63456 |
| later policy | XGBoost depth 10 refit | 10,640 | 0.81821 | 10,201 | 0.45150 | 0.77434 |
| later policy | PyTorch MLP (224 inputs) | 9,397 | 0.05682 | 11,444 | 0.20558 | 0.63094 |
| later policy | XGBoost depth 10 fold 3 | 8,711 | 0.66694 | 12,130 | 0.39836 | 0.75021 |

## Runtime and checkpoint verification

Fit time includes preprocessing. Peak fit memory is per worker. Checkpoint size includes fitted preprocessing where applicable. All policy predictions were scored again from reloaded files and compared to the stored arrays.

| Model | Fit seconds | Policy inference seconds | Fit peak GiB | Checkpoint MiB | Maximum reload error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 180.10 | 2.88 | 2.27 | 0.076 | 0 |
| Random Forest | 225.55 | 2.74 | 1.56 | 10.499 | 2.22e-16 |
| XGBoost depth 10 refit | 262.63 | 1.65 | 4.02 | 49.758 | 0 |
| PyTorch MLP (224 inputs) | 149.92 | 1.04 | 1.58 | 0.291 | 0 |
| XGBoost depth 10 fold 3 | 93.03 | 1.75 | 3.44 | 39.086 | 0 |

Refit command wall time was 833.83 s. Policy scoring process peak was 0.81 GiB. Fold-3 fit resources are from the original run.

## Reproduce or resume

The refit runner verifies and reuses saved checkpoints, and the scoring command requires all four checkpoints. The prediction arrays align by source order to the `policy.start` and `policy.end` offsets in the plan. A fresh output path gives an independent refit.

```powershell
Set-Location E:\fraudlens-v3
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$dev = 'E:\fraudlens-v2\artifacts\v2'
$prior = 'artifacts\v3'
$out = 'artifacts\v3\step7a-refit'
& $python -B -m src.v3.final_refit --development $dev --prior $prior --output $out
& $python -B -m src.v3.final_refit_report --development $dev --prior $prior --output $out --markdown 'docs\V3_STEP7A.md'
& $python -B -m pytest -q tests/test_v3_final_refit.py
```
