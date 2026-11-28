# FraudLens project summary

This record describes the archived V1 project. See the [final project reference](FINAL_PROJECT_REFERENCE.md) for the V3 submission.

## V2 run

The frozen V2 champion, `ensemble_weighted`, achieved **0.51147 AP**, **0.90325 ROC-AUC**, and **0.51149 F1** across the same 118,108 later transactions. Its frozen policy measured 62.00% precision and 43.53% recall. The later transactions were already evaluated in V1 and are a reused retrospective holdout. V2 selected models, calibration and the cutoff on earlier data.

[Full temporal validation and retrospective results](V2_EXPERIMENTS.md)

## V1 objective and scope

FraudLens measures whether statistical behavior derived from an approximate entity's prior transactions improves fraud discrimination and review prioritization. It implements offline data engineering and evaluation, eight matched supplementary classifiers, two NumPy networks, a two-hidden-layer PyTorch benchmark, anomaly detection, exact TreeSHAP, and exploratory clustering and rules. The four-page dashboard reads local artifacts and computes validation threshold scenarios and local explanations on demand.

## Data and run lineage

IEEE-CIS supplies anonymized labeled transaction rows and a partial identity table. Required inputs are the two Kaggle training CSVs. The one-to-one left join keeps all transactions. The artifact root contains separate `quick/` and `full/` mode directories with immutable runs and latest-manifest pointers. Code, dataset and protocol hashes were recorded before training. Every training and validation stage completed before a shared holdout release. The test comparisons use the same frozen configurations and thresholds. [Experiment protocol](EXPERIMENT_PROTOCOL.md) · [IEEE-CIS data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

Raw CSV dimensions are transactions 590,540 × 394 (15 loaded columns), identities 144,233 × 41 (3 loaded columns). The joined Phase 1 output is 590,540 × 27. The behavioral encoder has 331 columns.

| Partition | Rows | Fraud rows | Fraud rate | Elapsed seconds |
| --- | --- | --- | --- | --- |
| train | 354,324 | 11,988 | 3.3833% | 86,400–8,745,772 |
| validation | 118,108 | 4,611 | 3.9041% | 8,745,798–12,192,842 |
| test | 118,108 | 4,064 | 3.4409% | 12,192,900–15,811,131 |

The join matched 144,233 rows and left 446,307 without identity. There were 74,115 incomplete proxy keys and 111,446 rows with zero history count. Phase 1 took 10.75 seconds.

| Processed column | Missing rows | Missing share |
| --- | --- | --- |
| `card2` | 8,933 | 1.5127% |
| `card3` | 1,565 | 0.2650% |
| `card4` | 1,577 | 0.2670% |
| `card5` | 4,259 | 0.7212% |
| `card6` | 1,571 | 0.2660% |
| `addr1` | 65,706 | 11.1264% |
| `addr2` | 65,706 | 11.1264% |
| `P_emaildomain` | 94,456 | 15.9949% |
| `R_emaildomain` | 453,249 | 76.7516% |
| `DeviceType` | 449,730 | 76.1557% |
| `DeviceInfo` | 471,874 | 79.9055% |
| `entity_key` | 74,115 | 12.5504% |
| `prior_mean_amount` | 111,446 | 18.8719% |
| `amount_to_prior_mean` | 111,446 | 18.8719% |
| `seconds_since_prior_txn` | 111,446 | 18.8719% |
| `is_new_product` | 74,115 | 12.5504% |

Every other processed column has zero missing values. The private EDA output includes original selected CSV-column missing counts, product totals and amount statistics.

## Historical features and preprocessing

The four-part card/address proxy is `card1 | card2 | addr1 | addr2`. A row with a missing component never contributes to a common unknown history. Group by proxy and elapsed second before cumulative sums, counts and lagging. Equal-time rows have identical eligible histories. Carry observed, label-free history forward through the two split boundaries.

| Input | Definition |
| --- | --- |
| `prior_txn_count` | Number of rows with the same key at elapsed seconds strictly before the current second |
| `prior_mean_amount` | Sum of observed prior amounts divided by the number of observed prior amounts |
| `amount_to_prior_mean` | Current amount divided by a strictly positive prior mean |
| `seconds_since_prior_txn` | Current elapsed second minus the last strictly earlier elapsed second |
| `prior_count_1h` | Prior count with timestamps in `[t−3,600, t)` |
| `prior_count_24h` | Prior count with timestamps in `[t−86,400, t)` |
| `is_new_product` | 1 if this known product's first observation for this key occurs at `t`, otherwise 0. Undefined for missing key or category |

Counts default to zero. Undefined means, ratios, times and novelty remain missing. Basic inputs are amount, log1p amount, the identity missing indicator and 13 categorical fields (`ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`). The basic and behavioral models receive 16 and 23 raw predictor columns respectively. IDs, target, elapsed seconds and the entity proxy stay out of the predictor matrix.

Train fits median imputation, `StandardScaler`, one missingness flag per numerical column and one-hot encoding. Each category gets up to 30 training levels and reserved `missing` and `other` bins. Training frequencies rank levels, with lexical tie-breaking. Explicit float32 CSR numeric zeros distinguish zero from an absent sparse value in XGBoost. MLP loaders densify only the current batch. The supplementary transform is independently fitted on the matched training subset. DBSCAN and Isolation Forest have separate numerical preprocessing.

## Primary classifiers

AP sums precision weighted by increases in recall along the score ranking. ROC-AUC measures the probability that a random positive ranks above a random negative, with half credit for ties. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, specificity is `TN/(TN+FP)`, F1 is `2TP/(2TP+FP+FN)` and accuracy is `(TP+TN)/N`. Precision is undefined when a model makes zero positive predictions. Confusion matrices use `[[TN, FP], [FN, TP]]`. The F1-maximizing validation threshold is held fixed during test. Dummy uses the majority label directly.

The majority dummy uses the training majority label. Logistic Regression uses balanced LBFGS, `C=1` and 500 maximum iterations. Random Forest fits 120 balanced trees with depth at most 12 and leaf size at least four. Both XGBoost variants use CPU histogram trees, 240 rounds, depth 5, learning rate 0.075, minimum child weight 4, row subsampling 0.85, column subsampling 0.8, 128 bins and the training class ratio as `scale_pos_weight`. Choose the classical champion by validation AP. Designate that XGBoost as the deployed model and compare the MLP separately.

### Validation

| Model | AP | ROC-AUC | Precision | Recall | Specificity | F1 | Accuracy | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority dummy | 0.03904 | 0.50000 | — | 0.00000 | 1.00000 | 0.00000 | 0.96096 | 113497/0/4611/0 |
| Logistic Regression | 0.21998 | 0.78712 | 0.24301 | 0.35827 | 0.95466 | 0.28960 | 0.93138 | 108351/5146/2959/1652 |
| Random Forest | 0.21233 | 0.81484 | 0.25039 | 0.38278 | 0.95344 | 0.30274 | 0.93116 | 108213/5284/2846/1765 |
| XGBoost with history | 0.24830 | 0.82098 | 0.25515 | 0.40859 | 0.95154 | 0.31413 | 0.93034 | 107997/5500/2727/1884 |
| PyTorch MLP | 0.26732 | 0.79655 | 0.33504 | 0.34071 | 0.97253 | 0.33785 | 0.94786 | 110379/3118/3040/1571 |

### Test

| Model | AP | ROC-AUC | Precision | Recall | Specificity | F1 | Accuracy | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority dummy | 0.03441 | 0.50000 | — | 0.00000 | 1.00000 | 0.00000 | 0.96559 | 114044/0/4064/0 |
| Logistic Regression | 0.11409 | 0.75291 | 0.18104 | 0.29503 | 0.95244 | 0.22438 | 0.92982 | 108620/5424/2865/1199 |
| Random Forest | 0.18521 | 0.78737 | 0.24736 | 0.33465 | 0.96372 | 0.28446 | 0.94207 | 109906/4138/2704/1360 |
| XGBoost with history | 0.18778 | 0.79714 | 0.22920 | 0.34916 | 0.95816 | 0.27674 | 0.93720 | 109272/4772/2645/1419 |
| PyTorch MLP | 0.15633 | 0.77692 | 0.25155 | 0.25984 | 0.97245 | 0.25563 | 0.94793 | 110902/3142/3008/1056 |

### Frozen thresholds and CPU time

| Model | Threshold | Fit seconds | Validation score seconds | Final test score seconds |
| --- | --- | --- | --- | --- |
| Majority dummy | — | 0.017 | 0.001 | 0.001 |
| Logistic Regression | 0.79311 | 16.830 | 0.041 | 0.044 |
| Random Forest | 0.69765 | 74.684 | 0.408 | 0.863 |
| XGBoost with history | 0.74839 | 3.798 | 0.136 | 0.420 |
| XGBoost basic | 0.75747 | 7.012 | 0.357 | 0.614 |
| PyTorch MLP | 0.82687 | 225.092 | 3.021 | 2.943 |

## Historical-behavior ablation

Both XGBoost variants use the same rows, tree settings and training weighting. Behavioral inputs are the seven extra prior-history columns. Each variant has its own validation-frozen threshold.

| Partition | Input | AP | ROC-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Validation | Basic | 0.24931 | 0.80990 | 0.26663 | 0.39471 | 0.31827 |
| Validation | Behavioral | 0.24830 | 0.82098 | 0.25515 | 0.40859 | 0.31413 |
| Validation | Delta Behavioral Minus Basic | -0.00100 | 0.01108 | -0.01148 | 0.01388 | -0.00413 |
| Test | Basic | 0.19348 | 0.79313 | 0.24953 | 0.32825 | 0.28353 |
| Test | Behavioral | 0.18778 | 0.79714 | 0.22920 | 0.34916 | 0.27674 |
| Test | Delta Behavioral Minus Basic | -0.00570 | 0.00401 | -0.02033 | 0.02092 | -0.00679 |

## PyTorch neural benchmark

The CPU MLP maps `d → 128 → 64 → 1`. Each hidden layer uses Linear, BatchNorm, ReLU and dropout 0.25. One logit `z` becomes a score `σ(z) = 1/(1+exp(−z))` during evaluation. Training uses numerically stable weighted BCE from logits, with positive class weight `n−/n+`, Adam at `0.001`, weight decay `0.0001`, batches of 512 and early stopping on validation AP with patience five and a maximum of 30 epochs. Evaluation uses `eval()` and no gradients. The restored checkpoint is the earliest epoch with the highest validation AP.

The network has 51,201 trainable parameters and ran 13 epochs. Validation AP selected epoch 8 at 0.26732. The positive weight was 28.556557. Maximum dense input per batch was 0.646 MiB.

| Epoch | Train loss | Validation loss | Validation AP |
| --- | --- | --- | --- |
| 1 | 1.03933 | 1.10841 | 0.24103 |
| 2 | 0.94795 | 1.13784 | 0.24162 |
| 3 | 0.90426 | 1.15521 | 0.25143 |
| 4 | 0.87530 | 1.17295 | 0.25017 |
| 5 | 0.85438 | 1.17459 | 0.25861 |
| 6 | 0.82889 | 1.20497 | 0.26628 |
| 7 | 0.80954 | 1.22477 | 0.26601 |
| 8 | 0.79276 | 1.24560 | 0.26732 |
| 9 | 0.78115 | 1.21757 | 0.26687 |
| 10 | 0.77626 | 1.24106 | 0.26429 |
| 11 | 0.75699 | 1.31542 | 0.25651 |
| 12 | 0.74923 | 1.24999 | 0.25747 |
| 13 | 0.73909 | 1.34344 | 0.24814 |

## Anomaly detection, risk and TreeSHAP

Isolation Forest uses 100 isolation trees, up to 256 rows per tree and a training-only fit sample. A higher `−score_samples(x)` is more unusual. The anomaly percentile is `(count below + 0.5 × count equal)/N` against fixed training scores. The fixed composite score is `R = 100 × (0.8P + 0.2A)`, where `P` is the classifier score and `A` the percentile. Low, Medium and High use `<40`, `[40,75)` and `≥75`. These are illustrative review bands. The classifier decision has its own threshold. Review-budget counts round up, and ties use ascending transaction ID. TreeSHAP decomposes the XGBoost raw margin as expected value plus signed contributions. The sigmoid of the margin recovers the saved score. Contributions describe the classifier and have no causal interpretation.

### Validation

| Rank | AP | ROC-AUC | Top 1% P/R | Top 5% P/R | Top 10% P/R |
| --- | --- | --- | --- | --- | --- |
| Classifier only | 0.24830 | 0.82098 | 44.75% / 11.47% | 27.75% / 35.55% | 19.96% / 51.12% |
| Anomaly only | 0.06146 | 0.67237 | 3.55% / 0.91% | 5.23% / 6.70% | 5.84% / 14.96% |
| Fixed 80:20 blend | 0.23182 | 0.81961 | 42.39% / 10.87% | 26.38% / 33.79% | 19.71% / 50.49% |

### Test

| Rank | AP | ROC-AUC | Top 1% P/R | Top 5% P/R | Top 10% P/R |
| --- | --- | --- | --- | --- | --- |
| Classifier only | 0.18778 | 0.79714 | 35.45% / 10.31% | 23.50% / 34.15% | 15.74% / 45.74% |
| Anomaly only | 0.04997 | 0.63888 | 0.68% / 0.20% | 4.49% / 6.52% | 4.92% / 14.30% |
| Fixed 80:20 blend | 0.17993 | 0.78948 | 35.36% / 10.29% | 22.37% / 32.50% | 16.09% / 46.75% |

Isolation Forest fitted on 60,000 training rows and scored 354,324 for the reference. Exact global TreeSHAP used 2,000 validation rows. Maximum raw-margin error was 1.94e-06 globally and 3.68e-07 across 6 verified final-test cases. Their maximum saved classifier score error was 0.

## DBSCAN and PCA

Before DBSCAN, log-transform skewed counts, amount ratio and gap, preserve a missing-history indicator, impute training-sample medians and standardize. DBSCAN uses radius neighborhoods in this standardized space. Core points have enough neighbors, border points are reachable from core points and the remainder are noise. The run caps computed neighbor links and silhouettes on sampled non-noise points. Two-dimensional PCA displays a projection only. Cluster fraud rates are labeled descriptions after fitting. New transactions have no DBSCAN assignment in this workflow.

The 12,000-row sample produced 3 clusters and 900 noise rows (7.50%). Selected radius 0.65 and minimum samples 20 yielded silhouette 0.38743 over 1,500 inliers. PCA components explain 42.91% and 28.78%.

| Cluster | Rows | Fraud | Fraud rate | Median amount | Median prior count | Median amount ratio | Median gap (s) | Median prior 24h | Cold starts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Noise | 900 | 51 | 5.67% | 335.00 | 29 | 2.480 | 2541 | 2 | 0.22% |
| 0 | 2545 | 174 | 6.84% | 50.03 | 0 | — | — | 0 | 100.00% |
| 1 | 5116 | 129 | 2.52% | 65.00 | 90 | 0.629 | 8251.5 | 3 | 0.00% |
| 2 | 3439 | 52 | 1.51% | 77.00 | 8 | 0.789 | 339112 | 0 | 0.00% |

## Apriori

Training-fitted Boolean conditions create a basket. Apriori mines frequent itemsets and retains supported rules with fraud as the consequent. `support = joint/N`, `confidence = joint/matches` and `lift = confidence/prevalence`. Evaluate the frozen rules on the full validation split. Labels within each partition supply the consequent. Discovered rules do not feed either model or risk score.

Discovery used 30,000 training rows. Verification used 118,108 validation rows. The fixed cutoffs retained 2 fraud rules.

| Antecedent | Partition | Matches | Fraud matches | Support | Confidence | Lift |
| --- | --- | --- | --- | --- | --- | --- |
| No prior history | Train | 6,439 | 441 | 1.4700% | 6.849% | 2.008 |
| No prior history | Validation | 19,349 | 2,119 | 1.7941% | 10.951% | 2.805 |
| High amount + High 24-hour activity | Train | 659 | 34 | 0.1133% | 5.159% | 1.513 |
| High amount + High 24-hour activity | Validation | 2,223 | 137 | 0.1160% | 6.163% | 1.579 |

## Matched supplementary benchmark

Stratify independently within each original chronological partition with seed 42 and restore source order. Train the core references and same-sample GaussianNB with empirical priors and `var_smoothing=1e-9`, brute-force Euclidean KNN with 15 distance-weighted neighbors, balanced LinearSVC with `C=1` and a raw-margin threshold, and 80 AdaBoost depth-one stumps with learning rate 0.5 and train ratio weighting. KNN scores at most 512 query rows per batch inside a 64 MiB scikit-learn working-memory setting. Compare these models only within the matched experiment.

| Sample | Rows | Fraud rows | Fraud rate |
| --- | --- | --- | --- |
| Train | 20,000 | 677 | 3.3850% |
| Validation | 4,000 | 156 | 3.9000% |
| Test | 4,000 | 138 | 3.4500% |

| Classifier | Validation AP | Final subset AP | Final subset ROC-AUC | Final subset F1 | Threshold | Fit s | Validation scoring s | Final scoring s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority dummy | 0.03900 | 0.03450 | 0.50000 | 0.00000 | — | 0.001 | 0.001 | 0.001 |
| Logistic Regression | 0.19619 | 0.11490 | 0.73193 | 0.20072 | 0.89117 | 0.820 | 0.003 | 0.002 |
| Random Forest | 0.24333 | 0.20852 | 0.77268 | 0.27027 | 0.77408 | 2.202 | 0.063 | 0.080 |
| XGBoost with history | 0.24016 | 0.22259 | 0.76088 | 0.25397 | 0.83253 | 0.841 | 0.016 | 0.015 |
| Gaussian Naive Bayes | 0.04030 | 0.03597 | 0.52102 | 0.06946 | 1.00000 | 0.062 | 0.020 | 0.016 |
| KNN | 0.14183 | 0.12866 | 0.70476 | 0.18727 | 0.20675 | 0.010 | 1.691 | 1.849 |
| Linear SVM | 0.18485 | 0.11075 | 0.72677 | 0.22436 | 0.69031 | 1.919 | 0.002 | 0.001 |
| AdaBoost | 0.18079 | 0.17191 | 0.75702 | 0.24633 | 0.66987 | 3.758 | 0.074 | 0.077 |

## NumPy linear and shallow networks

For the shallow model, `H = max(0, XW₁+b₁)` and `z = HW₂+b₂`. The linear variant emits `XW₂+b₂` directly. For one batch use `L = mean(w·(logaddexp(0,z)−y·z)) + λ/2·Σ||W||²`, with `w=n−/n+` for positives and 1 for negatives. The output derivative is `w·(sigmoid(z)−y)/batch_size`. Backpropagate through `W₂` and mask by the positive ReLU preactivation, add `λW` to weight gradients and update parameters with mini-batch SGD. Biases are unregularized. The matched sklearn linear reference optimizes the same weighted objective with L-BFGS. The PyTorch shallow reference uses the same initialized parameters and SGD. Finite differences and PyTorch comparisons cover specified small fixtures and one-step calculations.

| Network | Val AP | Final subset AP | Val epoch | Fit s |
| --- | --- | --- | --- | --- |
| NumPy linear | 0.20522 | 0.11647 | 5 | 0.790 |
| scikit-learn linear reference | 0.19505 | 0.11523 | — | 1.605 |
| NumPy ReLU-32 | 0.21765 | 0.12202 | 4 | 1.368 |
| PyTorch ReLU-32 reference | 0.21765 | 0.12202 | 4 | 1.109 |

| NumPy fixture | Max finite-difference absolute error | Max PyTorch absolute error |
| --- | --- | --- |
| NumPy linear | 5.73e-12 | 1.11e-16 |
| NumPy ReLU-32 | 2.02e-11 | 2.22e-16 |

| NumPy model against reference | Final mean score gap | Final max score gap |
| --- | --- | --- |
| NumPy linear | 0.1513 | 0.8042 |
| NumPy ReLU-32 | 1.746e-08 | 1.788e-07 |

## Application and verification

`src/dashboard/artifacts.py` checks phase and mode, paths, schemas and parent-manifest digests. The Streamlit overview and queue use saved aggregates and risk results. Investigator explains a selected row with the saved model and transformed feature order. Model Lab compares saved curves and recalculates threshold metrics only on validation. Behavioral Intelligence displays exploratory patterns and attributions. The viewer does not train.

The offline verifier checked 6 phase manifests and 48 frozen files. It verified 93 unchanged quick files and all prediction IDs and row order, plus reloaded scores on 512 primary and 500 matched test rows. `pip check` passed. 84 tests passed with three SHAP dependency deprecation warnings.

### Viewport and startup

The full dashboard opened in about 3 seconds at 1366 × 768 with Full selected by default. The measured document and viewport widths matched. The Streamlit Python process peaked at 614 MiB during verification. No browser console errors were reported during the checked flows.

### Full-mode interactions

Verified all four pages, the queue-to-Investigator handoff, curated case loading, hidden and revealed labels, on-demand TreeSHAP, the frozen test metrics, and a change to the validation-only threshold. Inspected DBSCAN/PCA, Apriori, global SHAP, PyTorch curves, and the two supplementary views.

### Quick-mode regression

Switched to the original 100,000-row dataset and checked all four pages and the explicit Quick test label. Automated tests cover the original quick metrics and interaction expectations.

### Runtime and resources

The verified run used Windows, six CPU threads, 1.23 GiB peak working set and 433.16 seconds end to end on an Intel Core i5-12450H with 15.7 GiB RAM. The test suite reported 84 passing tests. The tested environment was the existing .venv.

The README shows the three reviewed aggregate views.

## Software map and limitations

`data_loader.py`, `features.py` and `splits.py` define the observation grain. `preprocessing.py`, `train.py`, `neural.py` and `train_mlp.py` fit and evaluate the primary models. `behavior.py`, `risk_engine.py`, `explain.py` and `phase3.py` produce the decision-support outputs. `clustering.py`, `association.py` and `phase5.py` produce offline patterns. `scratch_neural.py`, `scratch_checks.py`, `lab_benchmark.py` and `phase6.py` cover the matched experiments. `pipeline.py` enforces the shared release and `verify_full.py` verifies the saved run. `app.py` routes the four views. `tests/` covers numerical behavior, leakage, data contracts, model reload and UI flows.

This is one historical data source with a proxy entity, no merchant ID or verified local clock, and potential temporal drift. Class weighting and the blend yield uncalibrated scores. Review cases are curated and cannot estimate population performance. The reduced-data lab and exploratory samples bound compute and change the estimand. DBSCAN profiles and association rules remain descriptive. There is no production validation or clean-machine reproduction claim.
