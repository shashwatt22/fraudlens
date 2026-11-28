# FraudLens V3 manual assessment and local workbench

This is the archived four-input evaluation. The final release uses six inputs. See the [final challenger report](V3_MANUAL_CHALLENGER.md) and [current installation and launch instructions](../README.md#install-and-launch-locally).

The local app has two pages. **Dashboard** shows frozen historical research results alongside recent session activity and separately labeled manual-model validation. **Check Transaction & Alerts** runs genuine XGBoost inference on four manually entered fields and displays a review decision, exact TreeSHAP contributions, a synthetic assessment reference and a UTC timestamp.

## Launch on Windows

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:FRAUDLENS_SMTP_ENABLED = '0'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m streamlit run app.py --server.port 8523
```

Open [the local application](http://127.0.0.1:8523). Select **Check Transaction & Alerts**, then **Use a prefilled example**. **Debit · $49.95** returns **3.49%** and is not flagged. **Credit · $450** returns **7.11%** and is flagged. These are synthetic inputs evaluated by the fitted model. In a dry-run session the second check creates a notification record, and **Send Test Email** adds a preview record. The full assessment and alert histories are session-local, with bounded retention of 25 and 80 entries respectively.

For the archived six-page research workbench, launch [app_research.py](../archive/apps/app_research.py) directly with the same environment and a separate port. It preserves the original historical replay, transaction investigation, frozen research explanations, behavioral studies and configuration. The packaged public demo has the same two navigation choices as the local product. Its Check page explains the workflow without collecting inputs or sending email.

## Form inputs and preprocessing

| Form input | IEEE-CIS column | Treatment |
| --- | --- | --- |
| Amount (USD) | `TransactionAmt` | Same cent precision in training and the form, accepted range 0.01–32,000.00 |
| Card network | `card4` | Visa, Mastercard, American Express, Discover, or unknown |
| Card type | `card6` | Debit, credit, charge card, or unknown |
| Purchaser email provider | `P_emaildomain` | Google, Yahoo, Microsoft, AOL, Apple, other, or unavailable. The form uses a category and asks for no address. |

The inspected development prefix had 833 missing network values, 828 missing card types and 73,586 missing purchaser domains. Missing values and masked providers map to explicit unknowns, ambiguous card types map to unknown, and unlisted observed providers map to other. One-hot vocabularies were fitted on the training partition only. The model never uses identifiers, transaction timestamps, purchase history, device identity, product code or anonymized benchmark features. See the [dataset description](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and [data field write-up](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/).

## Manual model evaluation

One deterministic CPU XGBoost fit used 160 rounds, depth 4, learning rate 0.06, minimum child weight 15, L2 regularization 4 and unweighted logistic loss. No hyperparameter search took place. The fit used 354,324 chronological rows with 11,988 fraud labels. The next 70,864 rows selected an inclusive **0.0573460199** cutoff by maximum F1. The final 47,244 development rows measured the already selected model and cutoff. The boundary seconds are `86400–8745772`, `8745798–10766686` and `10766698–12192842` respectively.

| Development window | Rows | Fraud | AP | ROC-AUC | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Threshold selection | 70,864 | 2,957 | 0.11250 | 0.72137 | 12.41% | 38.69% | 0.18793 | 86.05% |
| Later verification | 47,244 | 1,654 | 0.09541 | 0.71608 | 11.24% | 38.75% | 0.17423 | 87.14% |

The later window had 5,704 flags, including 641 true positives and 5,063 false positives. The confusion matrix was `[[40527, 5063], [1013, 641]]` in `[[TN, FP], [FN, TP]]` order. In this window, default cutoff 0.5 generated zero flags. Brier score was 0.03279 and log loss 0.14052. These estimates describe development data with already-inspected chronology. The score is not separately calibrated, and the observed precision and recall support a human review workflow. The frozen 497-feature research model has a different population, input set and policy. Its original holdout was not loaded for the manual model evaluation.

Model fitting took **2.77 seconds**, and total setup took **7.22 seconds** for the recorded run. The checkpoint is **256,256 bytes**. Reloaded predictions matched throughout the final development window. Native TreeSHAP contributions are grouped back to the four form inputs. The sum with the base value reconstructs the raw model margin and its sigmoid reconstructs the returned score.

## Reproduce and configure

Use a fresh output directory to reproduce the model. Only local artifacts are written.

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:OMP_NUM_THREADS = '6'
$env:OPENBLAS_NUM_THREADS = '6'
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
& $python -B -m scripts.train_manual_assessment --raw 'E:/fraudlens/data/raw/train_transaction.csv' --v1 'E:/fraudlens/artifacts' --v3 artifacts/v3 --output artifacts/v3/manual-assessment-reproduction
& $python -B -m pytest -q tests/test_v3_manual_assessment.py
```

Use `artifacts/v3/manual-assessment` as the output when setting up the app for the first time. The local manifest supplies the exact cutoff, preprocessing vocabulary, metric values and checkpoint digest. Release files must exclude the trained model and runtime assessment records.

The [SMTP configuration and authorized test guide](V3_STEP10.md#configure-an-authorized-account-and-test-delivery) has exact environment-variable commands. For this app, keep the SMTP variables set from that guide and launch `app.py` with `--server.port 8523`. Omit the SMTP opt-in override from the dry-run launch command. Open the **Email alerts** panel on **Check Transaction & Alerts** and activate delivery for that session. Dry-run is the default, and the five-attempt session limit and 30-second spacing still apply. Use an authorized account and recipient and explicitly activate delivery before testing. This verification used no real SMTP delivery.

## Screenshots

[Dashboard](screenshots/product/01-dashboard.png) · [Check Transaction & Alerts](screenshots/product/02-check-transaction.png). Visible assessment results use only prefilled synthetic form inputs.

## Verification record

- The complete suite passed **166 tests**, including form validation, real-model inference, grouped TreeSHAP, session history, deduplication and email integration with a mock transport.
- Chrome 158 exercised the two pages, both decisions, manual and research metrics, charts, form validation, automatic dry-run notification, test email preview and disabled alerts. The 430 px navigation and public privacy checks passed. The standalone two-page package also passed browser checks in Linux with the Windows data drives hidden.
- The audit matched **19 frozen files**, **11 recorded evaluation outputs** and the frozen feature matrix. Step 7C results and screenshots are preserved. The review instance is local and has SMTP disabled.
