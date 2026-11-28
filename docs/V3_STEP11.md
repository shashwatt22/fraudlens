# FraudLens V3 public deployment preparation

The current public app presents the verified aggregate evaluation and a guided workflow on two pages. The earlier six-page verification is recorded below. It uses a separate entrypoint, a 74 kB aggregate export, and the existing shared visual components. Both current pages work without the private dataset, archived row scores, checkpoints, local virtual environments or SMTP configuration. Dashboard shows the aggregate research. Check Transaction & Alerts explains the local workflow. No page scores or accepts a transaction or sends email.

## Hosting and data rules

- **Streamlit Community Cloud** is the recommended free host for this static research workload. Choose Python **3.11** in Advanced settings. Streamlit’s documented approximate envelope is 0.078–2 CPU cores, 690 MB–2.7 GB memory and up to 50 GB storage. Its documentation dates those figures to February 2024 and says limits can change. Apps currently hibernate after 12 hours without traffic. [Hosting resources and hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) · [Python and dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- **Hosting terms.** Community Cloud account holders must be at least 18 and able to enter the agreement. The terms restrict processing sensitive information, including financial information subject to heightened protections. The Cloud package contains aggregate research statistics and accepts no records or uploads. [Streamlit terms of use](https://streamlit.io/terms-of-use)
- **Dataset rights.** Kaggle permits non-commercial academic use of the IEEE-CIS competition data in B.7.A and restricts its redistribution and access in B.7.B. The public export contains only study-level metrics, cohorts and reduced curve coordinates. It omits transaction fields, timestamps, prediction arrays, per-score curve thresholds, original data and fitted artifacts. Nothing here grants rights to publish the dataset. [IEEE-CIS competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) · [Dataset attribution](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Code sharing.** The project owner describes FraudLens as independent academic work and reports no competition submission. The competition’s separate B.8.B provision calls for code sharing in the competition’s Kaggle discussion or notebooks when the material falls under its scope. Before public release, confirm applicability and any rights needed for the aggregate outputs. If B.8.B applies, an authorized account holder should share the approved code or link on the competition’s Kaggle forum or notebooks. This step makes no Kaggle post and does not treat the rules as affirmative permission to share derivatives. [IEEE-CIS competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules)

## Export and run locally

Use any Python 3.11 installation. Build the package in a new directory. This generates a local copy; it does not change the source repository or publish anything.

```powershell
Set-Location 'E:\fraudlens-v3'
py -3.11 scripts/package_public_demo.py artifacts/v3/public-demo-release
py -3.11 -m venv artifacts/v3/public-demo-venv
$publicPython = (Resolve-Path 'artifacts/v3/public-demo-venv/Scripts/python.exe').Path
Set-Location artifacts/v3/public-demo-release
& $publicPython -m pip install -r requirements.txt
& $publicPython -m pip check
& $publicPython -m streamlit run app.py --server.address 127.0.0.1
```

The export contains exactly `app.py`, `results.json`, `requirements.txt`, `.streamlit/config.toml` and `src/dashboard_v3/ui.py`. The export pins Streamlit and Plotly, uses the packaged configuration, and needs no environment variables or secrets. The app is always aggregate-only, including when query parameters or environment settings request another mode. Create any test environment outside the five-file publication inventory.

## Future deployment by an eligible account holder

1. After publication approval and final review, add the exported files to the **root** of the reviewed deployment branch in the existing approved GitHub repository. Preserve existing public history. Use the clean export’s configuration and dependency pins. Review the entire branch tree because Community Cloud copies the repository. Keep the V3 development history and all private files local.
2. The eligible account holder signs in to [Streamlit Community Cloud](https://share.streamlit.io/) and connects their own GitHub account with access to the approved repository. In the workspace select **Create app**, then **Yup, I have an app**. Use repository `shashwatt22/fraudlens`, the approved branch, and file path `app.py` from the exported package. Choose an available subdomain. In **Advanced settings** select **Python 3.11** and leave **Secrets** empty. [Account documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/create-your-account) · [Deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
3. Deploy after the release is approved. Check both pages in a signed-out browser, confirm AP `0.56989` and the public workflow notices, and review the deployment logs. Public repositories normally give public app visibility. Confirm the app’s sharing setting before distributing its URL. [Sharing settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

## Results and reproducibility

The export preserves scalar metrics from Step 7C, including the four final V3 models and the archived V1/V2 champions on the same **reused retrospective holdout**. It also includes development-fold experiment means, the refitted MLP’s internal learning curves, and aggregate V2 pattern diagnostics with their populations labeled. Chart decimation and rounding affect display coordinates only. The local app and frozen artifacts remain available through [the local dashboard guide](V3_DASHBOARD.md).

Regenerate the export from offline reports using the command below and compare it to the checked-in export. The frozen bundle and the supporting fit and partition data are hash-checked before export.

```powershell
py -3.11 scripts/build_public_results.py --v3 artifacts/v3 --v1 'E:/fraudlens/artifacts/full' --v2 'E:/fraudlens-v2/artifacts/v2' --output artifacts/v3/public-results-check.json
Get-FileHash -Algorithm SHA256 public_demo/results.json,artifacts/v3/public-results-check.json
```

## Archived Step 11 screenshots

[Command Center](screenshots/public/01-command-center.png) · [Live Monitor](screenshots/public/02-live-monitor.png) · [Alert Center](screenshots/public/03-alert-center.png) · [Transaction Investigator](screenshots/public/04-transaction-investigator.png) · [Model Intelligence](screenshots/public/05-model-intelligence.png) · [Behavioral Analytics](screenshots/public/06-behavioral-analytics.png)

## Verified locally

- **Clean Linux runtime.** WSL Ubuntu with Python 3.11.18 installed the two direct dependencies and their transitive dependencies in a fresh environment. Dependency checking passed. The serving process had an empty environment except for the test home, executable path and locale. Its mount namespace hid the Windows drives. The runtime contained the allowlisted source package and no dataset, fitted models, scoring exports or local-app modules.
- **Browser.** Chrome 158 verified all six pages on the isolated Linux server, including the workflow controls, both operating points, ROC, precision–recall, calibration, confusion matrix, development curves, cohort view and mobile navigation at 430 px. The check observed no external page requests. The isolated Linux process peaked at approximately **206 MiB RSS** in that smoke test, with **464 MiB** on disk for its test environment. Cloud concurrency and cold-start performance remain unmeasured.
- **Tests.** 153 repository tests passed, including three public deployment tests. Three pre-existing SHAP dependency deprecation warnings remain.
- **Artifact and publication audit.** All 19 frozen model/policy digests and 11 recorded evaluation output digests matched, including the separately checked feature matrix. The tracked-file audit checked restricted extensions and secret indicators. The image audit reviewed aggregate visualizations and the six new public screenshots. Local Step 8–10 case captures remain ignored. Actual cloud deployment and release-rights confirmation are pending.

To repeat the browser checks, start the exported app on port 8521 and use the temporary Playwright installation from the dashboard guide.

```powershell
Set-Location 'E:\fraudlens-v3'
$qa = Join-Path ([IO.Path]::GetTempPath()) 'fraudlens-v3-browser-qa'
$env:PYTHONPATH = Join-Path $qa 'packages'
py -3.11 scripts/check_v3_public_browser.py --chrome 'C:\Program Files\Google\Chrome\Application\chrome.exe' --screenshots artifacts/v3/manual-assessment/public-screenshots
```
