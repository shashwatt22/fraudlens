# FraudLens V3 submission verification

Verified on **28 November 2026** from the `c1df0c8` release baseline. The release updates documentation, dependency setup, screenshot evidence and the location of older interfaces. The public workflow text describes all six form inputs. The live application code and all fitted or frozen research and manual artifacts retain their existing behavior and contents.

## Automated verification

| Check | Result |
| --- | --- |
| Clean Windows environment | Python 3.11.8, CPU PyTorch 2.14.1+cpu, XGBoost 3.2.0 and Streamlit 1.67.0. All pinned direct dependencies installed into the V3 `.venv`. `pip check` passed. |
| Complete suite | **170 passed** in **42.52 seconds**. Three existing SHAP/Matplotlib pending deprecation warnings. |
| Inference and TreeSHAP | The promoted local checkpoint scored prefilled examples, reproduced expected decisions and returned grouped signed explanations. Tests verify that explained margins reconstruct the prediction, that validation rejects invalid inputs and that sessions retain assessment history. |
| Notifications | Browser exercised automatic dry run, Send Test Email preview, disable and restore. The promoted form test exercised the authorized-send path through a mocked transport and verified one accepted alert with the correct cutoff and assessment reference. SMTP tests covered both TLS modes, failures, duplicate prevention and rate controls. The earlier authorized inbox confirmation and certificate-verified loopback test remain on record. This release run used dry run and a mock relay. |
| Frozen integrity | SHA-256 checks matched all 19 frozen fit and policy inputs, 11 evaluation outputs, the mapped feature matrix, feature manifest, evaluation code, V1 manifest and both manual checkpoint digests. The verified public aggregate export rebuilt byte-for-byte beside the original. |
| Metric parity | Across six historical model entries, AP, ROC-AUC and probability losses match the frozen backend. Both operating points for each model and both manual policy windows reconcile to their confusion counts. Manual statistics are explicitly labeled as development results. |

## Browser and screenshots

**Chrome 158.0.8127.37** exercised both modes. Test scripts are in `scripts/check_v3_product_browser.py` and `scripts/check_v3_public_browser.py`. The new captures contain aggregate charts and synthetic manual examples only.

| Mode | What passed | Screenshots |
| --- | --- | --- |
| Local | Exactly two primary pages. Verified headline historical and manual metrics, plots, form validation, all six inputs, flagged and unflagged model decisions, exact explanation display, recent assessment and alert histories, test preview and disabled/dry-run switching. At 430 px, navigation and the aggregate-only privacy view passed. | [Dashboard](screenshots/release/local/01-dashboard.png) · [Check Transaction & Alerts](screenshots/release/local/02-check-transaction.png) |
| Standalone public | Exported exactly five files and installed them into a fresh Linux Python 3.11.18 runtime with an empty user home, sanitized environment and Windows data drives hidden by a mount namespace. Dependency check passed. Chrome checked the exact two pages, confusion matrix, frozen/default cutoffs, ROC, precision–recall, calibration, optional aggregate research and absence of transaction input and email controls. | [Dashboard](screenshots/release/public/01-dashboard.png) · [Check Transaction & Alerts](screenshots/release/public/02-check-transaction.png) |

To reproduce from the [README environment](../README.md#install-and-launch-locally), run the servers on ports 8526 and 8527, then use the installed Chrome executable.

```powershell
$chrome = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
& $python -B scripts/check_v3_product_browser.py --url http://127.0.0.1:8526 --chrome $chrome
& $python -B scripts/check_v3_public_browser.py --url http://127.0.0.1:8527 --chrome $chrome
```

## Inventory and publication status

The release source tree was checked for restricted extensions, common credential patterns, personal email literals, unexplained large files and broken relative links. Email literals were reserved test fixtures. The largest tracked candidate was the 302,117-byte aggregate learning-curve chart. Git ignores datasets, checkpoints under `artifacts/`, virtual environments, environment files and Streamlit secrets. A historical object scan found zero restricted filename matches and no common secret patterns in the inspected text objects. All four new screenshots were visually reviewed and have no embedded metadata. The review leaves the private development history private.

The public runtime consists of `app.py`, `results.json`, `requirements.txt`, `.streamlit/config.toml` and `src/dashboard_v3/ui.py`. The checked-in aggregate payload has SHA-256 `47db18c288e526e366f500ff39cf5b54943192b6cc76e29ccece6ad909157553`. The package contains no transaction-level records, forecasts, fitted models or email configuration.

**Local submission checkpoint only.** No production model training, frozen result edits, GitHub push or deployment occurred. Publication still needs explicit approval and applicable Kaggle aggregate and code-sharing rights confirmation. Cloud deployment also needs an eligible account holder and live hosting verification. [Publication constraints](FINAL_PROJECT_REFERENCE.md#limitations-and-release-restrictions) · [Deployment guide](V3_STEP11.md)
