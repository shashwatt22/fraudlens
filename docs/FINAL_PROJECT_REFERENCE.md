# FraudLens: An Explainable Machine Learning System for Financial Fraud Detection and Behavioral Risk Intelligence

## What the system does

FraudLens brings together two supervised fraud-screening tasks. The **research pipeline** tests whether transaction and strictly prior behavioral features rank labeled fraud in the IEEE-CIS benchmark. The **local checker** scores six practical form inputs with a separate fitted model and explains the recommendation. The aggregate-only public demo presents the research and illustrates the workflow. The local and public products each have **Dashboard** and **Check Transaction & Alerts**.

| Layer | Implementation |
| --- | --- |
| Ingestion | Pandas selected columns, schema checks, compact types and an identity left join keyed by unique transaction ID. |
| Research | NumPy/Pandas feature builders, expanding chronological development folds, CPU model fitting, a reserved policy block and frozen retrospective scoring. |
| Local product | Streamlit and Plotly, local manifest/checkpoint loading, exact XGBoost TreeSHAP, bounded session history and an opt-in SMTP worker. |
| Public package | Standalone Streamlit entrypoint, shared rendering and an allowlisted aggregate export. It requires no datasets, checkpoint, scoring backend or SMTP configuration. |
| Archive | Earlier interfaces, exact experiment protocols, replay, transaction investigation, unsupervised analysis, association rules and scratch neural-network work. |

## Data and features

The IEEE-CIS and Vesta release on Kaggle has 590,540 transaction rows and 394 columns, including 20,663 positive labels, and 144,233 identity rows with 41 columns. The training tables join one-to-one on unique IDs. Missing identity remains observable through an indicator. `TransactionDT` is an elapsed time offset. The V1 split was 354,324 train, 118,108 validation and 118,108 retrospective rows. Phase 1 also has a small chronological quick mode that preserves timestamp groups. [Dataset description](https://www.kaggle.com/competitions/ieee-fraud-detection/data) · [Original protocol](EXPERIMENT_PROTOCOL.md)

| Feature family | Definition |
| --- | --- |
| Research candidate schema | 391 transaction inputs after removing ID, raw time and label, 40 identity inputs, one identity-missing flag, seven basic historical columns and 58 advanced columns. The 497 candidates include 460 numerical and 37 categorical values. |
| Basic causal history | Prior transaction count, prior amount mean, amount divided by a positive prior mean, seconds since the prior timestamp, counts in the last hour and day, and product novelty at its first observed timestamp. Proxy components are card1, card2, addr1 and addr2. |
| Advanced causal history | Proxies are card1 with addr1, that pair with round(day minus D1), and that pair with purchaser email category encoding. Each builds prior count, amount mean, population standard deviation, time gap, one-hour/day counts, current amount deviation and amount ratio. Product, purchaser/recipient email and device information additionally have past-frequency counts. |
| Advanced row-level features | Log amount and fractional amount, cyclical time-of-day and week coordinates, day-minus-D offsets, differences of D and C columns, family completeness counts, purchaser/recipient email agreement, common provider flags and device family flags. |
| Manual XGBoost | Amount in USD with cent precision from 0.01 to 32,000.00, card network (card4), card type (card6), grouped purchaser provider (P_emaildomain), optional desktop/laptop or mobile device category (DeviceType), and optional grouped recipient provider (R_emaildomain). Optional unknowns stay unknown. Provider groups are Google, Yahoo, Microsoft, AOL, Apple, Other and Unknown. |

For cumulative histories, each entity and timestamp is aggregated before shifting and computing earlier statistics. A rolling window uses `[t − w, t)`. Cold-start counts and missing moments are treated according to the feature schema. Incomplete keys never pool into a shared entity. Means and variances use observed amounts. Past transaction covariates are usable history without their fraud labels. Masked product codes lack a verified manual mapping and there is no verified identity-linked ledger for a form submission. [Basic builder](../src/features.py) · [Advanced builder](../src/v2/advanced.py) · [Manual feature experiment](V3_MANUAL_CHALLENGER.md)

## Model and policy protocol

The 472,432-row V2/V3 development prefix contains expanding training populations of 212,594, 283,459 and 354,324 followed by validation blocks of 70,865, 70,865 and 70,864. Final research refits use the first 425,188 rows, through second 10,766,686. Every model and preprocessor excludes the 47,244-row policy block. Earlier policy development spans seconds 10,766,698–11,449,871 (23,622 rows, 824 frauds). Later verification spans 11,449,894–12,192,842 (23,622 rows, 830 frauds). All cuts preserve complete timestamp groups. The last 118,108 rows are the reused retrospective holdout, seconds 12,192,900–15,811,131.

| Final supervised model | Fit and preprocessing |
| --- | --- |
| Logistic Regression | 425,188 rows, train-only median fill, log-frequency encoding, missing indicators, scaling, ANOVA top 192 and bounded one-hot representation. L2, C=0.08, L-BFGS. |
| Random Forest | 425,188 rows, training-only transform with ANOVA top 180. 140 bootstrap trees, depth 22, minimum leaf 3, Gini and square-root feature sampling. |
| Research XGBoost | 425,188 rows, original 497 candidates with native categories and missing-value routes. Depth 10, 320 rounds, learning rate 0.055, L2 4, max bin 128, row/column sampling 0.85/0.8. |
| PyTorch MLP | Chronological 382,669 gradient rows and 42,519 internal early-stopping rows. Training-side transform selects 224 inputs. Layers 224 → 192 → 96 → 48 → 1 with BatchNorm, GELU and dropout 0.18. 67,105 trainable parameters. AdamW, unweighted BCE and cosine scheduling. Best internal AP at epoch 3 of 7. |
| Manual XGBoost | 354,324 development rows, 11,988 frauds. Training-only one-hot vocabulary, depth 4, 160 rounds, learning rate 0.06, minimum child weight 15, L2 4 and unweighted logistic loss. |

Research selection promoted the depth-10 model after mean fold AP improved from 0.62876 to 0.63897 and all three folds improved. The 13 experimental Step 2 features and the richer Step 6B neural representation remain archived. V2 also evaluated LightGBM, CatBoost, and a weighted score ensemble. Their archived results have their own populations and settings. [V2 experiments](V2_EXPERIMENTS.md) · [Final refit](V3_STEP7A.md) · [Archive map](archive/README.md)

The earlier policy window selected each model’s inclusive F1 threshold. Within that window, a chronological split compared identity, sigmoid and isotonic calibrators for the primary research model. Neither alternative improved both Brier and log loss with the required paired day-block bootstrap evidence, so identity was retained. The earlier manual window selected the separate manual cutoff. Later outcomes were scored with the respective policies unchanged. The research retrospective features and forecasts were hash-checked before fraud labels were opened for Step 7C. [Policy protocol](V3_STEP7B.md) · [Retrospective audit](V3_STEP7C.md)

## Mathematical methods

| Method | How it works |
| --- | --- |
| Logistic and neural models | A logit z maps to p = 1/(1+exp(−z)). Binary cross-entropy is −[y log(p) + (1−y) log(1−p)]. Logistic regression fits a linear logit with a squared-weight penalty. The MLP learns nonlinear hidden representations with gradient updates, dropout regularization and chronological early stopping. |
| Forest and boosting | Random Forest averages tree probabilities from bootstrap samples with randomly sampled split candidates and Gini impurity 1 − Σ pₖ². XGBoost adds trees to a raw margin and chooses regularized splits using first- and second-order loss terms. |
| Ranking and decisions | AP is the sum of precision times the increase in recall at each score step. ROC-AUC summarizes ranking across false-positive rates. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R), accuracy = (TP+TN)/N. Review-budget recall uses the top scores and rounds up the workload. Brier is the mean of (p−y)². Log loss is mean cross-entropy. |
| Model explanation | TreeSHAP assigns signed feature contributions in log-odds units. The base value plus contributions equals the raw margin; its sigmoid recovers the score. The manual view sums one-hot contributions within each form field. Global mean absolute SHAP ranks model reliance for the named sample. |
| Behavioral workbench | Isolation Forest uses short isolation paths to rank unusual cases. Historical experiments map scores to percentiles; the V1 composite uses 0.8 × classifier score + 0.2 × anomaly percentile. DBSCAN links dense neighborhoods in standardized space and identifies noise; PCA projects onto variance-maximizing axes for display. Apriori discovers supported Boolean itemsets, with support = joint count/N, confidence = joint count/antecedent count and lift = confidence/fraud prevalence. The archive also includes a NumPy backpropagation comparison and V2 anomaly, clustering and rule diagnostics. |

Supplementary outputs are retained as separately labeled samples and descriptive findings. Their exact settings and verified populations are in the [V1 experiment record](PROJECT_SUMMARY.md) and [V2 experiment record](V2_EXPERIMENTS.md).

## Final results

**Research**, reused retrospective holdout of 118,108 rows, including 4,064 frauds. Ranking uses original model scores. Operating metrics use each frozen cutoff.

| Model | Cutoff | AP | ROC-AUC | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.2944795427 | 0.18280 | 0.81346 | 0.32366 | 0.31275 | 0.31811 | 0.95386 |
| Random Forest | 0.1963565625 | 0.36335 | 0.75799 | 0.44726 | 0.35162 | 0.39372 | 0.96274 |
| V3 XGBoost | 0.1938110143 | 0.56989 | 0.91798 | 0.60864 | 0.50591 | 0.55254 | 0.97181 |
| PyTorch MLP | 0.2504911423 | 0.29646 | 0.78810 | 0.34533 | 0.32579 | 0.33527 | 0.95555 |
| V1 XGBoost | 0.7483894229 | 0.18778 | 0.79714 | 0.22920 | 0.34916 | 0.27674 | 0.93720 |
| V2 weighted ensemble | 0.2752931571 | 0.51147 | 0.90325 | 0.62005 | 0.43529 | 0.51149 | 0.97139 |

The V3 XGBoost confusion matrix is [[112722, 1322], [2008, 2056]] in [[TN, FP], [FN, TP]] order. It reviews 28.6 per 1,000, with recall 25.91%, 61.17% and 73.94% at 1%, 5% and 10% review budgets. Brier is 0.02088 and log loss 0.08653. AP is 0.79070 for complete entity proxies seen in fitting, 0.33082 for unseen and 0.75420 for missing keys. The groups have different fraud prevalences. [All metrics and visualizations](V3_STEP7C.md)

**Manual**, later 23,622 development rows with 830 frauds. Threshold **0.1169009730**, AP **0.15605**, ROC-AUC **0.75881**, precision **22.60%**, recall **23.01%**, F1 **0.22806**, accuracy **94.53%**. It flags **845 (3.58%)** with matrix [[22138, 654], [639, 191]]. The prior four-input model had AP 0.09275 on the same rows. At the earlier-policy-matched cutoff its F1 was 0.15846 and workload was 7.49%. The final manual cutoff favors a smaller, more precise review queue. [Earlier and later window comparison](V3_MANUAL_CHALLENGER.md)

## Reproduction map

Use the one Python 3.11 environment in the [README](../README.md). Existing source-only checkouts can start with the standalone public demo. The local environment variables point to trusted, locally supplied artifacts.

| Local input or output | Where to get it or reproduce it |
| --- | --- |
| Raw train transaction and identity CSVs | Obtain from Kaggle. In the verified workspace, they are in `E:\fraudlens\data\raw`. |
| V1 artifacts and reports | `E:\fraudlens\artifacts` contains the V1 full results and processing metadata. [V1 protocol](EXPERIMENT_PROTOCOL.md) |
| V2 supervised features, folds, predictions and patterns | `E:\fraudlens-v2\artifacts\v2`. [V2 protocol](V2_PROTOCOL.md) |
| Four V3 frozen research models and policies | `artifacts/v3/step7a-refit` and `artifacts/v3/step7b-policy`. [Refit recipe](V3_STEP7A.md) · [Policy recipe](V3_STEP7B.md) |
| Frozen research report | `artifacts/v3/step7c-retrospective/dashboard.json` plus its score manifest. [Retrospective recipe](V3_STEP7C.md) |
| Final local checker | `artifacts/v3/manual-challenger/manifest.json` and `model.ubj`. The prior checkpoint remains in `artifacts/v3/manual-assessment`. [Manual reproduction](V3_MANUAL_CHALLENGER.md#launch-and-reproduction) |
| Public results | `public_demo/results.json`, verified aggregate export. [Public export recipe](V3_STEP11.md#results-and-reproducibility) |

For old commands using an external virtual environment, set the script's Python variable to the new V3 `.venv/Scripts/python.exe` and leave `PYTHONPATH` empty. Preserve data inputs, application roots and output arguments. Use fresh output directories for any offline reproduction. Holdout rescoring is a separate archived protocol; dashboard launch and release verification reuse frozen reports and checkpoints.

## Demonstration workflow

1. On **Dashboard**, identify the reused holdout label, note AP and ROC-AUC, and explain the XGBoost confusion matrix and V1–V3 comparison. Open optional analysis only for research questions.
2. On **Check Transaction & Alerts**, pick **Debit · $49.95** and assess with both optional fields unknown. Expect about 2.05% and no flag. Next use **Credit · $899.99**. Expect about 18.45%, a flag and a manual-review recommendation.
3. Show the second assessment’s synthetic reference, UTC time, signed contributions and dry-run notification. Explain that the flag is based on the manual cutoff and that demonstration inputs are synthetic. The public demo illustrates these steps without making a prediction or sending an alert.

### Authorized local email

Restart the local server from the same shell that holds the provider settings. Use a permitted sender and exactly one authorized recipient. Enter the password locally at the secure prompt. The environment is private to the process and inherited by its child. Close the shell after stopping the server to clear it.

```powershell
Set-Location 'E:\fraudlens-v3'
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$env:FRAUDLENS_PUBLIC_DEMO = '0'
$env:FRAUDLENS_SMTP_HOST = Read-Host 'Provider SMTP host'
$env:FRAUDLENS_SMTP_PORT = Read-Host 'Provider SMTP port'
$env:FRAUDLENS_SMTP_TLS = Read-Host 'Provider TLS setting, ssl or starttls'
$env:FRAUDLENS_SMTP_FROM = Read-Host 'Authorized sender address'
$env:FRAUDLENS_SMTP_TO = Read-Host 'One authorized recipient address'
$env:FRAUDLENS_SMTP_USERNAME = Read-Host 'SMTP account name'
$smtpSecret = Read-Host 'SMTP password or app password' -AsSecureString
$env:FRAUDLENS_SMTP_PASSWORD = [System.Net.NetworkCredential]::new('', $smtpSecret).Password
Remove-Variable smtpSecret
$env:FRAUDLENS_SMTP_ENABLED = '1'
& $python -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8526
```

Set the local artifact roots as shown in the README if you use different paths. Open **Email alerts** on the Check page, confirm authorization and select **Activate real delivery**. Select **Send Test Email** once. Check **SMTP accepted** in history, then check the authorized inbox. After the 30-second spacing, a flagged assessment automatically queues a notification in an activated session. Dry run is the default at each fresh session. The same five-attempt session quota applies to tests and alerts. SMTP uses certificate-verified SSL or STARTTLS; failure details are redacted. The payload includes the assessment reference, risk score, cutoff, version and review status. It excludes input fields and dataset identifiers. Delivery history is session-local.

## Limitations and release restrictions

The IEEE-CIS benchmark is historical, single-source and imbalanced. Entity keys approximate observed relationships and cohort prevalences differ. The reused holdout is retrospective, the earlier and later policy blocks are development data, and the manual evaluations have no prospective validation. Optional manual fields are often unknown. Model scores and SHAP summaries describe fitted behavior, and the manual operating point misses many positive labels. Historical anomaly, clustering and association analyses are descriptive. The application supports review in an academic setting.

Approval to publish and any applicable Kaggle rights confirmation are outstanding. Verify whether B.8.B applies and whether aggregate results may be shared; if B.8.B applies, an authorized account holder should share the approved code or link in the competition’s Kaggle discussion or notebooks. Keep raw data, row-level predictions, fitted artifacts, secrets and private history local. The MIT license covers original FraudLens code. Streamlit Community Cloud requires an eligible account holder aged at least 18. Hosting performance and actual deployment remain unverified. [Kaggle rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) · [Streamlit terms](https://streamlit.io/terms-of-use) · [Deployment instructions](V3_STEP11.md)
