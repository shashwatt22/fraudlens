# FraudLens V3 Step 5A XGBoost capacity experiment

**Promote depth 10.**

Mean AP changed by +0.01021 and ROC-AUC by +0.00246. AP improved in 3 of the 3 completed folds. Promotion requires an AP gain of at least 0.01, gains in at least two folds and an AUC drop no larger than 0.01.

Both arms used the original 497 V2 predictors, the same three chronological train and validation populations, 320 boosting rounds, learning rate 0.055, L2 regularization 4, seed 42, and six CPU threads. The challenger changes only maximum tree depth from 8 to 10. The saved depth-8 control predictions were verified by checkpoint reload. All metrics use development data. The later policy block and the retrospective holdout are outside the comparison.

## Validation performance

Precision, recall, F1 and accuracy use a cutoff of 0.5. Review budgets are ranked within each fold and rounded up to whole rows. Means weight folds equally.

| Fold | Fit rows | Model | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 212,594 | Depth 8 | 0.65989 | 0.92728 | 0.87368 | 0.43829 | 0.58374 | 0.97341 | 0.63603 | 0.77074 |
| 1 | 212,594 | Depth 10 | 0.66500 | 0.92807 | 0.89209 | 0.44161 | 0.59077 | 0.97398 | 0.64234 | 0.76111 |
| 2 | 283,459 | Depth 8 | 0.56917 | 0.90346 | 0.90412 | 0.32728 | 0.48059 | 0.97526 | 0.59161 | 0.70662 |
| 2 | 283,459 | Depth 10 | 0.58278 | 0.90726 | 0.91480 | 0.32930 | 0.48427 | 0.97547 | 0.60412 | 0.72881 |
| 3 | 354,324 | Depth 8 | 0.65723 | 0.93877 | 0.86586 | 0.43659 | 0.58049 | 0.97367 | 0.64559 | 0.77240 |
| 3 | 354,324 | Depth 10 | 0.66913 | 0.94157 | 0.87338 | 0.43389 | 0.57976 | 0.97375 | 0.65742 | 0.78323 |
| Mean | — | Depth 8 | 0.62876 | 0.92317 | 0.88122 | 0.40072 | 0.54827 | 0.97411 | 0.62441 | 0.74992 |
| Mean | — | Depth 10 | 0.63897 | 0.92563 | 0.89342 | 0.40160 | 0.55160 | 0.97440 | 0.63463 | 0.75772 |

## Training and validation gaps

Training metrics are in-sample and measure fit, including overfitting. Each gap subtracts the later validation score from the training score.

| Fold | Model | Train AP | Validation AP | AP gap | Train ROC-AUC | Validation ROC-AUC | ROC-AUC gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Depth 8 | 0.94968 | 0.65989 | 0.28979 | 0.99468 | 0.92728 | 0.06739 |
| 1 | Depth 10 | 0.98552 | 0.66500 | 0.32052 | 0.99893 | 0.92807 | 0.07086 |
| 2 | Depth 8 | 0.94215 | 0.56917 | 0.37299 | 0.99307 | 0.90346 | 0.08961 |
| 2 | Depth 10 | 0.98195 | 0.58278 | 0.39917 | 0.99848 | 0.90726 | 0.09122 |
| 3 | Depth 8 | 0.92249 | 0.65723 | 0.26526 | 0.98982 | 0.93877 | 0.05106 |
| 3 | Depth 10 | 0.97415 | 0.66913 | 0.30502 | 0.99752 | 0.94157 | 0.05595 |
| Mean | Depth 8 | 0.93811 | 0.62876 | 0.30935 | 0.99252 | 0.92317 | 0.06936 |
| Mean | Depth 10 | 0.98054 | 0.63897 | 0.34157 | 0.99831 | 0.92563 | 0.07268 |

## Entity results

A complete card–address–D1 key is seen when it occurs in that fold's training prefix. Missing keys form a separate stratum.

| Fold | Entity stratum | Rows | Fraud rate | Depth-8 AP | Depth-10 AP |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | seen in fit | 27,675 | 2.11% | 0.74688 | 0.78066 |
| 1 | unseen in fit | 34,564 | 3.75% | 0.49105 | 0.48510 |
| 1 | missing key | 8,626 | 13.12% | 0.81781 | 0.82517 |
| 2 | seen in fit | 27,435 | 1.71% | 0.62945 | 0.70106 |
| 2 | unseen in fit | 35,848 | 3.30% | 0.35813 | 0.36023 |
| 2 | missing key | 7,582 | 10.87% | 0.81049 | 0.81580 |
| 3 | seen in fit | 29,647 | 1.63% | 0.69771 | 0.78587 |
| 3 | unseen in fit | 31,078 | 3.64% | 0.44146 | 0.44482 |
| 3 | missing key | 10,139 | 13.24% | 0.81187 | 0.80944 |
| Mean | seen in fit | — | — | 0.69135 | 0.75587 |
| Mean | unseen in fit | — | — | 0.43022 | 0.43005 |
| Mean | missing key | — | — | 0.81339 | 0.81680 |

## Paired uncertainty

The 1,000 valid resamples use matching entity-block weights for both scorers and every fold. There are 83,314 complete entity blocks and 26,347 missing-key singleton blocks. Percentile intervals are conditional on the fitted checkpoints and observed development periods. They do not account for alternate seeds or future time periods.

| Fold | AP change | 95% interval |
| --- | ---: | --- |
| 1 | +0.00511 | [-0.00181, +0.01270] |
| 2 | +0.01362 | [+0.00695, +0.02006] |
| 3 | +0.01190 | [+0.00444, +0.01955] |
| Mean | +0.01021 | [+0.00586, +0.01456] |

## Runtime and checkpoints

Depth-8 timings and process peaks come from its archived run. Depth-10 resources come from separate fold workers. Inference is validation scoring. The reported stored sizes are compressed Joblib for depth 8 and native UBJSON for depth 10.

| Fold | Model | Training (s) | Inference (s) | Peak GiB | Checkpoint MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Depth 8 | 164.54 | 2.60 | 3.07 | 6.39 |
| 1 | Depth 10 | 57.21 | 1.12 | 2.47 | 19.41 |
| 2 | Depth 8 | 161.03 | 2.59 | 3.07 | 9.61 |
| 2 | Depth 10 | 79.33 | 1.14 | 2.94 | 29.95 |
| 3 | Depth 8 | 214.56 | 2.80 | 3.50 | 12.32 |
| 3 | Depth 10 | 93.03 | 1.20 | 3.44 | 39.09 |
| Mean | Depth 8 | 180.04 | 2.66 | 3.21 | 9.44 |
| Mean | Depth 10 | 76.52 | 1.15 | 2.95 | 29.48 |

The challenger fit time was 229.57 s total. The training command took 281.46 s wall time, including verification and checkpointing.

## Reproduce or resume

The runner verifies existing fold hashes before skipping completed work. Choose a fresh output directory for an independent run.

```powershell
Set-Location E:\fraudlens-v3
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$dev = 'E:\fraudlens-v2\artifacts\v2'
$out = 'artifacts\v3\step5a-depth'
& $python -B -m src.v3.depth_experiment --development $dev --output $out
& $python -B -m src.v3.depth_report --development $dev --output $out --markdown 'docs\V3_STEP5A.md'
```
