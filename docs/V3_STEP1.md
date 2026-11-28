# FraudLens V3 Step 1 foundation

V3 branches from the verified local V2 checkpoint `4b708981d33446958cbc8a3be2119dc205e05af9`, retained as tag `v3-v2-baseline`. The `v3/foundation` branch uses a separate Git checkout with no configured remote. The V3 diagnostic reads the original artifacts in place, with both NumPy matrices memory-mapped read-only. This step loaded saved scores and ran inference with one unchanged XGBoost checkpoint. It did not fit models, change features or open the V2 retrospective holdout.

## Local inputs

- `E:\fraudlens\data\raw` contains `train_transaction.csv` and `train_identity.csv`. This step checked their presence without opening the CSVs.
- `E:\fraudlens\artifacts` contains the V1 `quick` and `full` manifests. This step used the separate V2 development cache instead of those partitions.
- `E:\fraudlens-v2\artifacts\v2` contains the development cache, fold reports, out-of-fold scores, model checkpoints and decision-policy manifest.
- `E:\fraudlens\.venv\Scripts\python.exe` is Python 3.11.8. The checked runtime includes NumPy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, XGBoost 3.2.0 and PyTorch 2.14.1+cpu. V2 additional dependencies are available via `E:\fraudlens-v2\.venv\Lib\site-packages`.

## Training populations and validation

The three expanding folds use only the 472,432-row V2 development period. In each fold, the next chronological block is scored completely. Timestamp ties stay in the same block. The percentages that define the three training prefixes and their end-of-validation cuts are 45%, 60%, 75% and 90%, adjusted for timestamp ties.

| Learner | Fold | Available training prefix | Fitted rows | Fraud rows in fit | Internal early-stop rows | Best epoch | Outer validation rows | Outer AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (`logistic_l2`) | 1 | 212,594 | 16,000 | 489 | 0 | — | 70,865 | 0.36684 |
| Logistic Regression (`logistic_l2`) | 2 | 283,459 | 16,000 | 537 | 0 | — | 70,865 | 0.30646 |
| Logistic Regression (`logistic_l2`) | 3 | 354,324 | 16,000 | 541 | 0 | — | 70,864 | 0.41521 |
| Random Forest (`random_forest_depth`) | 1 | 212,594 | 16,000 | 489 | 0 | — | 70,865 | 0.40164 |
| Random Forest (`random_forest_depth`) | 2 | 283,459 | 16,000 | 537 | 0 | — | 70,865 | 0.31829 |
| Random Forest (`random_forest_depth`) | 3 | 354,324 | 16,000 | 541 | 0 | — | 70,864 | 0.42096 |
| XGBoost (depth 8) (`xgb_advanced_depth8`) | 1 | 212,594 | 212,594 | 6,496 | 0 | — | 70,865 | 0.65989 |
| XGBoost (depth 8) (`xgb_advanced_depth8`) | 2 | 283,459 | 283,459 | 9,510 | 0 | — | 70,865 | 0.56917 |
| XGBoost (depth 8) (`xgb_advanced_depth8`) | 3 | 354,324 | 354,324 | 11,988 | 0 | — | 70,864 | 0.65723 |
| PyTorch MLP (`torch_mlp_regularized`) | 1 | 212,594 | 16,000 | 467 | 21,260 | 7 | 70,865 | 0.41496 |
| PyTorch MLP (`torch_mlp_regularized`) | 2 | 283,459 | 16,000 | 523 | 28,346 | 12 | 70,865 | 0.30230 |
| PyTorch MLP (`torch_mlp_regularized`) | 3 | 354,324 | 16,000 | 544 | 35,433 | 5 | 70,864 | 0.41097 |

Logistic Regression and Random Forest each fit a deterministic, label-stratified sample of 16,000 rows from their training prefix. The MLP samples 16,000 rows before the last tenth of that prefix and uses the remaining chronological tail for early stopping. XGBoost uses the entire prefix, without positive-class weighting or early stopping, at the saved 320-round configuration. The four metrics use a common 0.5 cutoff when decisions are needed. AP and ROC-AUC use continuous scores.

## Features and entity proxies

V2 combines 431 source predictors, one derived join indicator, seven inherited behavioral features, and 58 advanced features, for 497 columns. Entity proxies are approximate grouping keys. Missing components leave a key incomplete. Same-timestamp transactions never count toward each other’s historical features.

| Proxy | Components |
| --- | --- |
| `v1_entity` | `card1`, `card2`, `addr1`, `addr2` |
| `card_address` | `card1`, `addr1` |
| `card_address_anchor` | `card1`, `addr1`, `round(day_minus_D1)` |
| `card_address_email` | `card1`, `addr1`, `P_emaildomain` |

The anchor uses `round(floor(TransactionDT / 86400) - D1)`. The purchaser domain is the original cached category. Each V2 proxy has six prior-history fields (`prior_count`, `prior_mean`, `prior_std`, `prior_gap`, `prior_count_1h`, `prior_count_24h`) and two amount comparison fields (`amount_deviation`, `amount_ratio`).

- **Join indicator** `identity_missing`.
- **Inherited behavior (7)** `prior_txn_count`, `prior_mean_amount`, `amount_to_prior_mean`, `seconds_since_prior_txn`, `prior_count_1h`, `prior_count_24h`, `is_new_product`.
- **Amount and periodic time (6)** `amount_log`, `amount_fraction`, `day_sin`, `day_cos`, `week_sin`, `week_cos`.
- **Day offsets, differences, and observed counts (13)** `day_minus_D1`, `day_minus_D4`, `day_minus_D10`, `day_minus_D11`, `day_minus_D15`, `D15_minus_D1`, `D10_minus_D1`, `D15_minus_D10`, `C1_minus_C2`, `observed_C`, `observed_D`, `observed_V`, `observed_id`.
- **card_address (8)** `card_address_prior_count`, `card_address_prior_mean`, `card_address_prior_std`, `card_address_prior_gap`, `card_address_prior_count_1h`, `card_address_prior_count_24h`, `card_address_amount_deviation`, `card_address_amount_ratio`.
- **card_address_anchor (8)** `card_address_anchor_prior_count`, `card_address_anchor_prior_mean`, `card_address_anchor_prior_std`, `card_address_anchor_prior_gap`, `card_address_anchor_prior_count_1h`, `card_address_anchor_prior_count_24h`, `card_address_anchor_amount_deviation`, `card_address_anchor_amount_ratio`.
- **card_address_email (8)** `card_address_email_prior_count`, `card_address_email_prior_mean`, `card_address_email_prior_std`, `card_address_email_prior_gap`, `card_address_email_prior_count_1h`, `card_address_email_prior_count_24h`, `card_address_email_amount_deviation`, `card_address_email_amount_ratio`.
- **Earlier category frequencies (4)** `ProductCD_past_frequency`, `P_emaildomain_past_frequency`, `R_emaildomain_past_frequency`, `DeviceInfo_past_frequency`.
- **Email and device indicators (11)** `email_equal`, `p_email_google`, `p_email_yahoo`, `p_email_microsoft`, `r_email_google`, `r_email_yahoo`, `r_email_microsoft`, `device_apple`, `device_windows`, `device_samsung`, `device_huawei`.

## Seen and unseen entities

A complete key is seen when it occurs in the rows that actually fitted that model. Unseen keys are complete and absent from those rows. Missing keys form a separate group. For sampled learners, an unseen key can still occur in unsampled earlier rows or in earlier validation rows. Historical features may therefore be available even for keys the model did not fit. This is a comparison of training exposure, not a customer-identity or cold-start claim. AP depends on the fraud rate, so the table includes that rate.

The table shows the card, address and D1 anchor proxy across all three folds. The generated JSON also includes every model and fold for all four proxies. It stores aggregate metrics only.

| Learner | Fold | Segment | Rows | Fraud rate | AP | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 1 | seen in fit | 8,442 | 3.07% | 0.13896 | 0.79126 | 0.00000 | 0.00000 | 0.00000 |
| Logistic Regression | 1 | unseen in fit | 53,797 | 3.02% | 0.24847 | 0.75684 | 0.81513 | 0.11953 | 0.20849 |
| Logistic Regression | 1 | missing key | 8,626 | 13.12% | 0.58438 | 0.78262 | 0.67712 | 0.45760 | 0.54613 |
| Logistic Regression | 2 | seen in fit | 7,211 | 2.18% | 0.06737 | 0.74827 | — | 0.00000 | 0.00000 |
| Logistic Regression | 2 | unseen in fit | 56,072 | 2.67% | 0.13890 | 0.74723 | 0.59146 | 0.06480 | 0.11680 |
| Logistic Regression | 2 | missing key | 7,582 | 10.87% | 0.64264 | 0.84331 | 0.75588 | 0.50728 | 0.60712 |
| Logistic Regression | 3 | seen in fit | 7,440 | 1.92% | 0.09171 | 0.77786 | 0.50000 | 0.01399 | 0.02721 |
| Logistic Regression | 3 | unseen in fit | 53,285 | 2.76% | 0.20760 | 0.79162 | 0.78049 | 0.08696 | 0.15648 |
| Logistic Regression | 3 | missing key | 10,139 | 13.24% | 0.65154 | 0.85833 | 0.70611 | 0.54247 | 0.61357 |
| Random Forest | 1 | seen in fit | 8,442 | 3.07% | 0.13883 | 0.78932 | — | 0.00000 | 0.00000 |
| Random Forest | 1 | unseen in fit | 53,797 | 3.02% | 0.30163 | 0.77785 | 0.96026 | 0.08934 | 0.16347 |
| Random Forest | 1 | missing key | 8,626 | 13.12% | 0.60547 | 0.82986 | 0.77170 | 0.42403 | 0.54732 |
| Random Forest | 2 | seen in fit | 7,211 | 2.18% | 0.06385 | 0.59562 | — | 0.00000 | 0.00000 |
| Random Forest | 2 | unseen in fit | 56,072 | 2.67% | 0.15485 | 0.64976 | 0.75000 | 0.05210 | 0.09744 |
| Random Forest | 2 | missing key | 7,582 | 10.87% | 0.65335 | 0.86106 | 0.84635 | 0.40777 | 0.55037 |
| Random Forest | 3 | seen in fit | 7,440 | 1.92% | 0.13190 | 0.56652 | — | 0.00000 | 0.00000 |
| Random Forest | 3 | unseen in fit | 53,285 | 2.76% | 0.19752 | 0.72266 | 0.94017 | 0.07473 | 0.13845 |
| Random Forest | 3 | missing key | 10,139 | 13.24% | 0.67805 | 0.86897 | 0.82465 | 0.48361 | 0.60968 |
| XGBoost (depth 8) | 1 | seen in fit | 27,675 | 2.11% | 0.74688 | 0.95958 | 0.99111 | 0.38120 | 0.55062 |
| XGBoost (depth 8) | 1 | unseen in fit | 34,564 | 3.75% | 0.49105 | 0.86108 | 0.85294 | 0.29067 | 0.43358 |
| XGBoost (depth 8) | 1 | missing key | 8,626 | 13.12% | 0.81781 | 0.94689 | 0.85325 | 0.63693 | 0.72939 |
| XGBoost (depth 8) | 2 | seen in fit | 27,435 | 1.71% | 0.62945 | 0.93633 | 0.93431 | 0.27234 | 0.42175 |
| XGBoost (depth 8) | 2 | unseen in fit | 35,848 | 3.30% | 0.35813 | 0.83595 | 0.87558 | 0.16047 | 0.27123 |
| XGBoost (depth 8) | 2 | missing key | 7,582 | 10.87% | 0.81049 | 0.94101 | 0.90792 | 0.59830 | 0.72129 |
| XGBoost (depth 8) | 3 | seen in fit | 29,647 | 1.63% | 0.69771 | 0.96696 | 0.98246 | 0.34783 | 0.51376 |
| XGBoost (depth 8) | 3 | unseen in fit | 31,078 | 3.64% | 0.44146 | 0.87573 | 0.84564 | 0.22261 | 0.35245 |
| XGBoost (depth 8) | 3 | missing key | 10,139 | 13.24% | 0.81187 | 0.94325 | 0.85225 | 0.64903 | 0.73689 |
| PyTorch MLP | 1 | seen in fit | 7,833 | 2.92% | 0.07030 | 0.71246 | — | 0.00000 | 0.00000 |
| PyTorch MLP | 1 | unseen in fit | 54,406 | 3.04% | 0.29546 | 0.78401 | 0.86900 | 0.12039 | 0.21148 |
| PyTorch MLP | 1 | missing key | 8,626 | 13.12% | 0.63862 | 0.84098 | 0.80905 | 0.42668 | 0.55870 |
| PyTorch MLP | 2 | seen in fit | 6,680 | 1.84% | 0.11644 | 0.69880 | 0.50000 | 0.03252 | 0.06107 |
| PyTorch MLP | 2 | unseen in fit | 56,603 | 2.70% | 0.14841 | 0.75330 | 0.63522 | 0.06597 | 0.11953 |
| PyTorch MLP | 2 | missing key | 7,582 | 10.87% | 0.62298 | 0.82887 | 0.72968 | 0.46845 | 0.57058 |
| PyTorch MLP | 3 | seen in fit | 6,638 | 2.30% | 0.05280 | 0.65730 | — | 0.00000 | 0.00000 |
| PyTorch MLP | 3 | unseen in fit | 54,087 | 2.70% | 0.19294 | 0.74841 | 0.78313 | 0.08892 | 0.15971 |
| PyTorch MLP | 3 | missing key | 10,139 | 13.24% | 0.66471 | 0.85676 | 0.83251 | 0.50373 | 0.62767 |

## XGBoost decision-policy comparison

The same fold-3 checkpoint scored both periods. Reloaded fold-3 validation predictions matched the saved values with maximum absolute difference `0`. The policy kept the identity calibration choice, so AP uses the raw scores in both periods. The last half of the policy block was the verification window. The full policy block also includes the earlier rows used for calibration comparison and threshold selection.

| Window | Rows | Fraud cases | Fraud rate | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fold-3 validation | 70,864 | 2,957 | 4.17% | 0.65723 | 0.93877 |
| whole block | 47,244 | 1,654 | 3.50% | 0.58048 | 0.91949 |
| calibrator fit | 11,811 | 378 | 3.20% | 0.66318 | 0.93492 |
| threshold selection | 11,811 | 446 | 3.78% | 0.55418 | 0.90573 |
| policy evaluation | 23,622 | 830 | 3.51% | 0.55538 | 0.91932 |

## Reproduce the diagnostic

The interrupted V2.1 diagnostic passed a scalar threshold to `evaluate_scores`, which expects one binary decision per transaction. The V3 diagnostic converts the score array with `scores >= cutoff` and retains the returned metrics separately from the curves. The V2 source and the interrupted V2.1 files stay untouched.

```powershell
Set-Location E:\fraudlens-v3
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
& $python -B -m pytest -q tests/test_v3_diagnostics.py
& $python -B -m src.v3.diagnostics --development 'E:\fraudlens-v2\artifacts\v2' --output 'artifacts\v3\step1-diagnostics.json' --threads 6
```
