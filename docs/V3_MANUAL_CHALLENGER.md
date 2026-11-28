# FraudLens manual assessment | Final challenger

The optional-context model is promoted. It adds device type and recipient email provider to the four original inputs. Both controls are collapsed and default to **Unknown**. The four-input checkpoint and all frozen V3 research models remain available.

## Candidate inputs

| Input family | Decision |
| --- | --- |
| Device type | Added. A visitor can supply desktop/laptop, mobile, or unknown. The identity table is left-joined by ID and missing observations stay unknown. |
| Email categories | Retained the original purchaser provider and added the recipient provider when explicitly known. Both share the same grouping of common providers, Other and Unknown. The form accepts provider categories only. |
| Product category | Excluded. IEEE-CIS supplies masked product codes, with no verified mapping to categories a person can enter. |
| Transaction history | Excluded. The form has no verified link to the dataset’s entity keys or authoritative prior ledger. |

On the later window, device type was unknown for 19,386 of 23,622 rows and recipient provider for 19,917. Those counts include masked recipient providers mapped to Unknown. Availability may differ in real use, so this evaluation does not establish performance on current manual traffic. [IEEE-CIS data description](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

## Experiment and results

Only the challenger was trained. Like the control, it fit the first **354,324** development observations, with the same 160-round, depth-4 CPU configuration and training-only vocabulary. A verified ordered metadata file confirmed the development prefix, and exactly 120,464 matching identity records were read. The retrospective holdout was not loaded. The existing control checkpoint reproduced its previously saved full policy-block metrics.

The earlier policy window contains **23,622** rows and **824** frauds, seconds **10,766,698–11,449,871**. The later window contains **23,622** rows and **830** frauds, seconds **11,449,894–12,192,842**. Both earlier-selected thresholds maximize F1 and were applied unchanged to the later window. The existing live control cutoff is included separately for operational comparison.

| Window | Model and cutoff | AP | ROC-AUC | Precision | Recall | F1 | Review workload |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier | Live four-input control, 0.05734602 | 0.09987 | 0.73120 | 11.53% | 38.47% | 0.17739 | 2,750 · 11.64% |
| Earlier | Matched four-input control, 0.07943562 | 0.09987 | 0.73120 | 14.81% | 30.58% | 0.19952 | 1,702 · 7.21% |
| Earlier | Challenger, 0.11690097 | 0.18524 | 0.80800 | 28.46% | 30.22% | 0.29311 | 875 · 3.70% |
| Later | Live four-input control, 0.05734602 | 0.09275 | 0.70124 | 10.97% | 39.04% | 0.17125 | 2,954 · 12.51% |
| Later | Matched four-input control, 0.07943562 | 0.09275 | 0.70124 | 11.64% | 24.82% | 0.15846 | 1,770 · 7.49% |
| Later | **Challenger, 0.11690097** | **0.15605** | **0.75881** | **22.60%** | **23.01%** | **0.22806** | **845 · 3.58%** |

The promotion rule required at least +0.01 AP, ROC-AUC loss no greater than 0.005, no F1 decline at either control cutoff and a workload increase no greater than 2 percentage points. All checks passed. Later AP rose by **0.06331** and ROC-AUC by **0.05757**. Against the live control, F1 gained **0.05681** and workload fell **8.93 percentage points**, with recall falling **16.02 percentage points**. The challenger flagged **191 of 830** fraud-labeled rows, compared with **324** for the live control and **206** for the matched control. The promoted cutoff prioritizes a smaller, more precise queue.

Training took **2.77 seconds** and the complete experiment took **5.82 seconds**. The 256,361-byte challenger checkpoint reloaded with identical scores. The local manifest and checkpoint are under `artifacts/v3/manual-challenger/`. Exact TreeSHAP contributions are grouped back to the six form inputs and checked against the returned prediction.

## Launch and reproduction

For the local review app, use:

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:FRAUDLENS_SMTP_ENABLED = '0'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m streamlit run app.py --server.port 8525
```

The **Debit · $49.95** example scores **2.05%** with optional fields omitted. **Credit · $899.99** scores **18.45%** and flags at the **11.6901%** cutoff. An assessment alert includes the model version, threshold and reference. Leave the review instance in dry run for demonstration. For real delivery with the updated model, restart FraudLens in the PowerShell window where your SMTP variables are set and explicitly activate real delivery in the new browser session. See [SMTP setup](V3_APPLICATION.md#reproduce-and-configure).

For a separate reproduction, use an empty output path. The application uses `manual-challenger` for promoted artifacts and falls back to the original four-input checkpoint in workspaces without that directory.

```powershell
$env:OMP_NUM_THREADS = '6'
$env:OPENBLAS_NUM_THREADS = '6'
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
& $python -B -m scripts.compare_manual_challenger --raw 'E:/fraudlens/data/raw' --development-root 'E:/fraudlens-v2/artifacts/v2' --v3 artifacts/v3 --output artifacts/v3/manual-challenger-reproduction
& $python -B -m pytest -q tests/test_v3_manual_challenger.py tests/test_v3_manual_assessment.py tests/test_v3_alerts.py
```

## Verification

- **26 focused tests passed**, covering input parity, real checkpoint predictions, model explanation, promotion checks, alert deduplication and authorized delivery integration.
- **Chrome 158** verified the two navigation pages, six-field inference, both decisions, grouped SHAP, notification history, and mobile/public privacy. The screenshots use prefilled synthetic inputs. [Dashboard](screenshots/product/01-dashboard.png) · [Transaction check](screenshots/product/02-check-transaction.png)
- A flagged assessment produced exactly one message through certificate-verified SMTP to an isolated local receiver. The email contained the promoted model name, cutoff, and assessment reference. The app displayed SMTP accepted. The existing SMTP-configured instance was left running and no external email was sent during this experiment.
- The audit matched all 19 frozen V3 model and decision-policy digests plus the original manual manifest.
