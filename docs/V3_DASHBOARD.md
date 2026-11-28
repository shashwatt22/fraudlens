# FraudLens V3 dashboard

This guide documents the archived research workbench. For the current two-page app, see [FraudLens V3 manual assessment](V3_APPLICATION.md).

This local Streamlit application reads the Step 7C result bundle and displays the frozen V3 policy. The primary model is the refitted depth-10 XGBoost at threshold **0.1938110143** with identity calibration. The 118,108-row holdout was also evaluated in V1 and V2. The threshold sandbox uses the separate 23,622-row development verification block.

## Launch on Windows

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:FRAUDLENS_V3_ARTIFACTS = 'E:\fraudlens-v3\artifacts\v3'
$env:FRAUDLENS_ARTIFACTS = 'E:\fraudlens\artifacts'
$env:FRAUDLENS_V2_ARTIFACTS = 'E:\fraudlens-v2\artifacts\v2'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m streamlit run archive/apps/app_research.py
```

Open [the local dashboard](http://127.0.0.1:8501). The server binds to localhost. The app reads trusted local offline artifacts, verifies frozen hashes, and loads the booster on the first SHAP request or historical replay inference. It never fits a model on launch. To revisit the historical workbench with the same environment, run `streamlit run archive/apps/app_legacy.py --theme.base dark` through the Python executable above.

| Local input | Dashboard usage |
| --- | --- |
| V3 Step 7C results bundle | Aggregate holdout metrics and display curves |
| V3 Step 7C scores and memory-mapped feature matrix | Review queue, investigator and behavioral features |
| V3 Step 7A checkpoint and Step 7B manifests | Verified exact TreeSHAP; frozen cutoff and model lineage |
| V3 development experiment summaries and V2 development metadata | Model selection evidence and exploratory threshold workload |
| V1 full artifacts and V2 pattern summary | Transaction metadata and observed label; historical anomaly, cluster and association panels |

A missing optional artifact affects its section. The sidebar **Hide local case details** switch hides the queue, case-level view, individual PCA points and Live Monitor feed for screen sharing. Open [the aggregate-only view](http://127.0.0.1:8501/?aggregate=1) to start with case details hidden. The saved screenshots use a fresh session in this mode. Keep local data, per-row scores, mapped features and model files outside version control.

## Screenshots

- [Command Center](screenshots/v3/01-command-center.png)
- [Transaction Investigator](screenshots/v3/02-transaction-investigator.png)
- [Model Intelligence](screenshots/v3/03-model-intelligence.png)
- [Behavioral Analytics](screenshots/v3/04-behavioral-analytics.png)
- [Live Monitor](screenshots/v3/05-live-monitor.png) · [Classroom demonstration and replay verification](V3_STEP9.md)
- [Alert Center](screenshots/v3/06-alert-center.png) · [Email setup and authorized test procedure](V3_STEP10.md)

## Verification commands

Run the tests from the repository root using the launch environment above.

```powershell
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m pytest -q -rs
```

For browser verification, install Playwright in a temporary directory and point it at an installed Chrome executable. Launch the app with `--server.port 8518` in a separate PowerShell window.

```powershell
$qa = Join-Path ([IO.Path]::GetTempPath()) 'fraudlens-v3-browser-qa'
$packages = Join-Path $qa 'packages'
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
& $python -m pip install --target $packages 'playwright==1.65.2'
$env:PYTHONPATH = $packages
& $python -B scripts/check_v3_dashboard_browser.py --chrome 'C:\Program Files\Google\Chrome\Application\chrome.exe' --screenshots docs/screenshots/v3
```

Global V3 SHAP summarizes 24 evenly spaced holdout rows and can be sensitive to that sample. ROC and precision–recall plots use the reduced coordinates in the verified bundle, while the reported metrics use the full population. The earlier research panels retain their original V1 or V2 provenance.

## Verified result

- **Automated tests** — 135 passed, including the original quick and full artifact integrations and the V3 tests. Pytest reported three existing SHAP dependency deprecation warnings.
- **Chrome 158** — All four pages rendered. Checked chronology, decision/product/risk filters, sorting, pagination, invalid and exact search, transaction handoff, observed-label reveal, and exact local TreeSHAP. Checked ROC, precision–recall, calibration, confusion matrices at both operating points, archived comparisons, neural learning curves, and the separate development threshold sandbox. Checked Isolation Forest cohorts, DBSCAN/PCA filtering, Apriori search, and archived/global SHAP. The 430 px touch viewport passed the initial layout and sidebar navigation checks.
- **Integrity** — All 19 Step 7A/7B frozen digests, all 11 recorded Step 7C outputs and the mapped feature matrix matched their saved hashes after dashboard verification.
- **Screenshot scope** — The four tracked screenshots show aggregate data. For a local record of actual investigation and TreeSHAP, append `--local-screenshots` to the browser command. Those captures are saved under the ignored `artifacts/v3/step8-dashboard` directory.
