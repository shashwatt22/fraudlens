# FraudLens

**Financial Fraud & Behavioral Risk Intelligence**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-F7931E) ![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-1769AA) ![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B) ![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-4D77CF) ![93 verified passing tests](https://img.shields.io/badge/Verified%20tests-93%20passed-238636)

FraudLens ranks suspicious financial transactions for human review. V2 compares boosted trees, classical classifiers and neural networks on expanding chronological folds, then freezes an ensemble and review cutoff. The four-page Streamlit dashboard compares V1 and V2, ranks saved scores and computes TreeSHAP for the ensemble’s XGBoost component.

[Dashboard](#dashboard) · [Architecture](#architecture) · [V2 results](#v2-results) · [V1 results](#v1-holdout-reference) · [Reproduce](#reproduce-on-windows)

## V2 results

The V2 study ran **36 supervised configurations** on three expanding folds within the **472,432-row** V1 development period. The final comparison uses the same **118,108 later transactions** as V1, including 4,064 fraud cases. This is a reused retrospective holdout. V2 fitting, selection, calibration and threshold selection used earlier observations.

| Model | Average precision | ROC-AUC | Precision | Recall | F1 | Accuracy | Specificity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 XGBoost with history | 0.18778 | 0.79714 | 0.22920 | 0.34916 | 0.27674 | 0.93720 | 0.95816 |
| V2 depth-8 XGBoost | 0.50316 | 0.89645 | 0.55970 | 0.46137 | 0.50580 | 0.96898 | 0.98707 |
| **V2 weighted ensemble** | **0.51147** | **0.90325** | **0.62005** | **0.43529** | **0.51149** | **0.97139** | **0.99049** |

The weighted ensemble reached **0.63425 mean development AP** and **0.66017 AP on the third fold**. It assigns 77.21875% of the score to depth-8 XGBoost and 7.59375% each to LightGBM, CatBoost and a regularized PyTorch MLP. At the frozen cutoff of `0.275293`, it identified 1,769 frauds, missed 2,295 and flagged 1,084 legitimate transactions. Its score is uncalibrated. The V2 artifact-backed test suite passed **93 tests** on Windows with Python 3.11.8 and CPU training.

[All model results, ablations and operating points](docs/V2_EXPERIMENTS.md) · [V2 protocol](docs/V2_PROTOCOL.md) · [Verification record](docs/V2_COMPLETION.md)

![Full-mode dashboard comparing the frozen V1 and V2 results](docs/screenshots/dashboard-v2-comparison.jpg)

## Why this problem is hard

Fraud accounts for about 3.5% of the processed transactions. A majority classifier can appear accurate while catching no fraud. Increasing sensitivity can catch more fraud and send more legitimate transactions for review. Chronological evaluation tests the models on later activity, where spending patterns and fraud behavior can shift. Historical features give a reviewer behavioral context, and TreeSHAP explains how the classifier used its inputs. FraudLens is an educational investigation-support prototype for a human reviewer.

## Dashboard

![Full-mode Streamlit portfolio overview showing aggregate activity, prevalence, and review priorities](docs/screenshots/01-overview.jpg)

*Portfolio overview. Aggregate transaction activity and review priorities in full mode.*

| [Final-test model comparison](docs/screenshots/05-model-comparison.jpg) | [PyTorch learning curves](docs/screenshots/06-neural-learning-curves.jpg) |
| :---: | :---: |
| ![Aggregate final-test metrics and average precision by model](docs/screenshots/05-model-comparison.jpg) | ![Saved weighted training loss and validation average precision by epoch](docs/screenshots/06-neural-learning-curves.jpg) |
| Compare the classical and neural benchmarks on the same holdout. | Inspect weighted loss and see why validation AP selected epoch 8. |

| Page | What you can do |
| :--- | :--- |
| **Overview & Review Queue** | Compare transaction activity, open a case, and inspect the V2 review ranking in Full / Final test. V1 classifier, anomaly and composite rankings remain available. |
| **Transaction Investigator** | Inspect strictly prior context and the frozen decision. V2 explains the XGBoost component’s raw margin with TreeSHAP. Reveal the observed label after reviewing the case. |
| **Model Lab** | Compare V1 and V2 metrics, the accepted V2 model progression, precision–recall and calibration. Explore V1 validation thresholds, ablations, learning curves and matched benchmarks. |
| **Behavioral Intelligence & Patterns** | Compare V2 anomaly ranking, DBSCAN stability and Apriori experiments. Inspect V1 training-reference profiles and grouped TreeSHAP. [V2 dashboard capture](docs/screenshots/dashboard-v2-patterns.jpg) |

## Architecture

```mermaid
flowchart TD
    TX[IEEE-CIS transaction and identity CSVs] --> JOIN[Validated identity left join]
    JOIN --> SPLIT[Three chronological partitions]
    SPLIT --> V1[V1 models and behavioral risk baseline]
    V1 --> SAVE1[V1 versioned local artifacts]
    SPLIT --> DEV[472432 V1 train and validation rows]
    DEV --> FEAT[Expanded fields and strictly prior behavioral features]
    FEAT --> FOLDS[Three expanding validation folds]
    FOLDS --> FIT[XGBoost, LightGBM, CatBoost, classical models and neural networks]
    FOLDS --> PAT[Isolation Forest, DBSCAN and Apriori experiments]
    FIT --> FREEZE[Freeze ensemble, calibration choice and cutoff]
    DEV --> POLICY[Separate chronological decision block]
    POLICY --> FREEZE
    SPLIT --> HOLD[118108 reused retrospective holdout rows]
    HOLD --> EVAL[One frozen retrospective comparison]
    FREEZE --> EVAL
    EVAL --> SAVE2[V2 versioned local artifacts]
    PAT --> SAVE2
    FIT --> SHAP[TreeSHAP for the XGBoost component]
    SAVE1 --> APP[Four-page Streamlit dashboard]
    SAVE2 --> APP
    SHAP --> APP
```

Offline stages build the feature cache, fit and compare models, and save the frozen policy and score artifacts. The dashboard checks the V1/V2 artifact lineage and reads compatible saved results. It reloads the fitted XGBoost component locally when requested for a V2 case explanation.

## Data and evaluation protocol

The source is the IEEE-CIS Fraud Detection dataset hosted by Kaggle. The raw transaction and identity files have **590,540 × 394** and **144,233 × 41** dimensions respectively. The V1 loader selects 15 and 3 columns and preserves every transaction with a validated identity left join. The joined feature table has 27 columns. [IEEE-CIS dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) · [Competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)

| Partition | Transactions | Fraud cases | Fraud rate |
| :--- | ---: | ---: | ---: |
| Train | 354,324 | 11,988 | 3.3833% |
| Validation | 118,108 | 4,611 | 3.9041% |
| Final test | 118,108 | 4,064 | 3.4409% |

The original V1 entity proxy combines `card1`, `card2`, `addr1` and `addr2`. For a transaction at time *t*, every history calculation uses transactions strictly before *t*. Equal-timestamp transactions share the same eligible history. Split boundaries preserve whole timestamp groups, and incomplete entity keys receive a cold start.

| Historical feature | Calculation |
| :--- | :--- |
| `prior_txn_count` | Number of strictly earlier transactions for the proxy |
| `prior_mean_amount` | Mean observed amount across those transactions |
| `amount_to_prior_mean` | Current amount divided by a positive prior mean |
| `seconds_since_prior_txn` | Elapsed seconds since the latest strictly earlier transaction |
| `prior_count_1h` | Prior transactions in `[t − 3,600, t)` |
| `prior_count_24h` | Prior transactions in `[t − 86,400, t)` |
| `is_new_product` | 1 when this known product category has no earlier observation for the proxy |

V1 training fits median imputation, missing-value indicators, numeric standardization and capped one-hot encoding with missing and unseen-category handling. Supported classifiers use balanced weights or the training negative-to-positive ratio. The MLP uses that ratio as the positive-class weight in its logit loss. IDs and entity keys stay out of model inputs.

The full pipeline freezes fits and validation choices before it opens the final holdout for evaluation. Validation selects checkpoints and maximizes F1 for decision thresholds. Quick mode uses 100,000 early transactions, with **zero overlapping IDs** and an earlier end timestamp than the final holdout. [Frozen V1 experiment protocol](docs/EXPERIMENT_PROTOCOL.md)

V2 combines 431 source predictors, an identity-match indicator and seven V1 historical predictors. It adds 58 row-local or strictly historical features for 497 features in the advanced matrix. The final 47,244 development rows are reserved for calibration and the decision policy. Categorical vocabulary and fitted transformations are learned from the applicable training rows. The real-world availability of anonymized source fields is unverified. [V2 protocol](docs/V2_PROTOCOL.md)

## V1 holdout reference

AP is average precision. ROC-AUC, precision, recall and F1 below come from the full final holdout. Precision is undefined for the dummy model because it predicted no positive labels.

| Model | Validation AP | Final AP | ROC-AUC | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority dummy | 0.03904 | 0.03441 | 0.50000 | — | 0.00000 | 0.00000 |
| Logistic Regression | 0.21998 | 0.11409 | 0.75291 | 0.18104 | 0.29503 | 0.22438 |
| Random Forest | 0.21233 | 0.18521 | 0.78737 | 0.24736 | 0.33465 | 0.28446 |
| **XGBoost with history** | 0.24830 | **0.18778** | **0.79714** | 0.22920 | **0.34916** | 0.27674 |
| PyTorch MLP | **0.26732** | 0.15633 | 0.77692 | **0.25155** | 0.25984 | 0.25563 |

- **Classical versus neural.** The PyTorch MLP led validation AP. It restored epoch **8 of 13** from a `331 → 128 → 64 → 1` network with batch normalization, ReLU, dropout and Adam. Behavioral XGBoost was selected for the dashboard by validation AP among the learned classical candidates. It also had the highest final AP in the primary comparison.
- **Behavioral ablation.** Adding history reduced final AP from **0.19348** to **0.18778**, while raising ROC-AUC from **0.79313** to **0.79714** and recall from **0.32825** to **0.34916**. The result depends on the metric and the proxy's usefulness.
- **Pattern discovery.** DBSCAN found **3 clusters** and **7.50% noise** in 12,000 training observations, with silhouette **0.38743**. PCA is a visualization of the fitted sample. Two Apriori rules carried to validation, with lift **2.805** for no prior history and **1.579** for high amount with high 24-hour activity.
- **Matched supplementary models.** On the separate 20,000 training, 4,000 validation and 4,000 final subsets, Random Forest led validation AP (**0.24333**) and XGBoost led final AP (**0.22259**). These scores describe the matched subset experiment.

The technical summary includes confusion matrices, frozen thresholds, fit and scoring times, review budgets and matched comparisons. [Complete results](docs/PROJECT_SUMMARY.md)

## Neural networks from scratch

`ScratchNetwork` implements a linear logit model and a one-hidden-layer **ReLU-32** model using NumPy. It explicitly initializes weights and biases, computes forward activations, uses a stable sigmoid, and differentiates weighted binary cross-entropy with L2 on weights. Mini-batch gradient descent applies the hand-derived gradients. [NumPy implementation](src/scratch_neural.py) · [Independent numerical and autograd checks](src/scratch_checks.py)

For the shallow model, let $H = \operatorname{ReLU}(XW_1 + b_1)$ and $z = HW_2 + b_2$. The linear model computes $z$ directly from $X$. For a batch of $m$ labeled examples, $w_i$ is the training-derived positive weight for fraud rows and 1 otherwise.

$$
L = \frac{1}{m}\sum_{i=1}^m w_i\bigl[\operatorname{softplus}(z_i)-y_i z_i\bigr] + \frac{\lambda}{2}\sum_j \lVert W_j\rVert_F^2
$$

$$
\frac{\partial L}{\partial z_i} = \frac{w_i}{m}\bigl[\sigma(z_i)-y_i\bigr],\qquad W_j \gets W_j-\eta\nabla_{W_j}L
$$

The ReLU mask propagates the derivative into the hidden layer. The finite-difference checks had maximum absolute gradient error **5.73 × 10⁻¹²** for the linear fixture and **2.02 × 10⁻¹¹** for the shallow fixture. The maximum checked difference against PyTorch autograd, logits, loss and update was **2.22 × 10⁻¹⁶**. On the matched final subset, NumPy ReLU-32 and the equivalent PyTorch reference both scored **0.12202 AP**, with a maximum score difference of **1.79 × 10⁻⁷**.

## Explainability and risk intelligence

Exact TreeSHAP decomposes the frozen XGBoost raw margin into a base value and signed contributions. Applying sigmoid to the reconstructed margin yields the classifier score. The audit on 2,000 validation rows measured a maximum raw-margin error of **1.94 × 10⁻⁶**. Attributions explain the fitted model's output.

Isolation Forest learns unusual patterns from training data. The anomaly percentile compares a score with a fixed training reference, and the dashboard combines it with the classifier score using an uncalibrated review heuristic.

$$
\mathrm{Risk}=100\times\bigl(0.8\times\mathrm{classifier\ score}+0.2\times\mathrm{anomaly\ percentile}\bigr)
$$

| Final ranking | AP | Recall at 10% review budget |
| :--- | ---: | ---: |
| Classifier only | **0.18778** | 45.74% |
| Anomaly only | 0.04997 | 14.30% |
| Fixed 80:20 blend | 0.17993 | **46.75%** |

The blend lowered overall AP and caught a greater share of fraud at the 10% budget. The risk score and classifier score are both uncalibrated. The application displays anomaly, classifier decision and review priority separately.

## Reproduce on Windows

### 1. Clone and install

Use **Python 3.11**. The recorded run used 3.11.8 and the pinned CPU build of PyTorch. The requirements file records the tested package versions. [PyTorch CPU wheels](https://pytorch.org/get-started/previous-versions/)

```powershell
git clone https://github.com/shashwatt22/fraudlens.git
Set-Location fraudlens
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-v2.txt
.\.venv\Scripts\python.exe -m pip check
```

### 2. Obtain the authorized data and build the V1 reference

Obtain `train_transaction.csv` and `train_identity.csv` from Kaggle if you have authorized access under the competition rules. Place both in `data/raw/`. The evaluation holdout comes from the labeled transaction file. Data and runtime artifacts are obtained or generated locally. [Data download and access requirements](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

```powershell
.\.venv\Scripts\python.exe -m src.pipeline --mode quick --threads 6
.\.venv\Scripts\python.exe -m src.pipeline --mode full --threads 6
.\.venv\Scripts\python.exe -m src.verify_full --threads 6
```

Quick mode must precede full mode in the same artifact root. To reproduce with a fresh directory, add `--artifact-root artifacts/reproduction` to all three commands and set `$env:FRAUDLENS_ARTIFACTS = 'artifacts/reproduction'` for the app. Keep the original quick and full folders together for lineage verification. The full runner guards against reopening an already released directory.

### 3. Run the V2 stages

From the clone root, use the V1 `full` artifact directory created above. Use the same V2 output path across stages. Training stages resume at saved fold checkpoints. `finalize` freezes the policy, and `evaluate` runs once for that output directory.

```powershell
$v2 = @{
    Python = (Resolve-Path '.venv\Scripts\python.exe').Path
    RawData = (Resolve-Path 'data\raw').Path
    V1Artifacts = (Resolve-Path 'artifacts\full').Path
    OutputRoot = 'artifacts/v2'
    Threads = 6
}
foreach ($stage in 'baseline', 'boosting', 'matched', 'patterns', 'finalize', 'evaluate') {
    & .\scripts\run_v2.ps1 @v2 -Stage $stage
}
```

To reuse existing authorized local V1 artifacts, point `V1Artifacts` at their `full` directory and set `FRAUDLENS_ARTIFACTS` to its parent. For a separate V2 reproduction, choose a new `OutputRoot` and set `FRAUDLENS_V2_ARTIFACTS` to that directory before launching the dashboard. [V2 reproduction details](docs/V2_EXPERIMENTS.md#reproduction-and-checkpoints)

### 4. Test and launch

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://127.0.0.1:8501`. The sidebar selects Quick or Full and defaults to Full when the required artifacts agree. The artifact-backed dashboard requires a locally generated run. Select **Full** and **Final test** for the V2 panels. The V2 suite passed all 93 tests with the original V1 artifacts available. Seven integration tests need real V1 artifacts. They use `artifacts/` by default or the directory in `FRAUDLENS_ARTIFACTS`.

## Engineering and verification

```text
fraudlens/
├── app.py                     Streamlit entry point
├── .streamlit/config.toml      Local dashboard configuration
├── src/
│   ├── data_loader.py         Selected-column ingest and identity join
│   ├── features.py            Strictly prior transaction history
│   ├── splits.py              Chronological partitions
│   ├── preprocessing.py       Training-fitted feature transformation
│   ├── train.py               Classical models and ablation
│   ├── train_mlp.py            PyTorch benchmark and checkpoint selection
│   ├── scratch_neural.py       NumPy networks and gradients
│   ├── phase3.py              Anomaly, explanation and risk outputs
│   ├── phase5.py              DBSCAN, PCA and Apriori
│   ├── phase6.py              Matched supplementary experiments
│   ├── pipeline.py            Reproduction and holdout release gate
│   ├── verify_full.py         Artifact and model reload audit
│   ├── v2/                    Rich features, temporal experiments and freeze gate
│   └── dashboard/             Four interactive views with V1 and V2 results
├── scripts/run_v2.ps1         Resumable V2 stages
├── tests/                     Synthetic and artifact-backed tests
├── docs/                      Protocol, results and aggregate screenshots
├── requirements.txt          Pinned core dependencies
└── requirements-v2.txt       V2 boosting dependencies
```

| Area | Technology |
| :--- | :--- |
| Language and data processing | Python 3.11, pandas, NumPy, SciPy, PyArrow |
| Classical and unsupervised ML | scikit-learn, XGBoost, LightGBM, CatBoost, MLxtend |
| Deep learning and explainability | PyTorch CPU, NumPy, SHAP |
| Visualization | Streamlit, Plotly |
| Testing and artifacts | pytest, Streamlit AppTest, Parquet, JSON manifests, joblib, PyTorch state dictionaries and NumPy checkpoints |

The suite covers historical leakage, equal-timestamp boundaries, training-only preprocessing, threshold selection, the final release gate, serialization, transaction alignment, TreeSHAP additivity, numerical gradients and dashboard interactions. The saved-run verifier also checks cross-phase manifests, frozen digests and reloaded model scores. Reload model files only from trusted local runs.

## Scope and limitations

The benchmark uses one historical data source with missing identity information and an approximate entity proxy. The V2 comparison reuses the V1 holdout. Generalization to an independent dataset and the serving-time availability of anonymized fields are unverified. The frozen decision leaves significant review errors, and the ensemble score is uncalibrated. Clusters and association rules are exploratory, and the supplementary benchmark uses bounded matched samples. Verification used Windows and the recorded Python environment. This project is an educational prototype.

## Dataset and licensing

FraudLens uses the IEEE-CIS Fraud Detection dataset, provided by IEEE-CIS and Vesta through Kaggle, for non-commercial academic research and education. Obtain the dataset directly from Kaggle under its terms and competition rules. The dataset and trained models are excluded from this repository. [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) · [Competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)

The MIT License covers the original FraudLens source code. The dataset, trained model artifacts and third-party components are outside its scope. Third-party components retain their own licenses. [MIT License](LICENSE)

FraudLens is developed by `shashwatt22`. AI-assisted tooling supported development and documentation.
