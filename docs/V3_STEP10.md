# FraudLens V3 Alert Center

This guide documents the archived research workbench. For the current two-page app, see [FraudLens V3 manual assessment](V3_APPLICATION.md).

Live Monitor creates a local notification record when the frozen depth-10 XGBoost model flags a historical replay transaction at **0.1938110143** or above. Alert Center shows the mode, recent history, UTC timestamps, outcomes and real-send quota. Each email contains an independent case reference, model score, cutoff, model version and review status. A test email has a test reference and no model score.

## Dry-run walkthrough

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:FRAUDLENS_SMTP_ENABLED = '0'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m streamlit run archive/apps/app_research.py
```

Open [FraudLens on localhost](http://127.0.0.1:8501). In **Live Monitor**, select **Step one** four times. The first two flagged records appear in **Alert Center** with status **Dry run**. **Send Test Email** adds a preview record in this mode. **Disable alerts** records replay events as suppressed and disables the test control.

The [public demo view](http://127.0.0.1:8501/?demo=1) is aggregate-only and dry-run-only. To force public demo for all sessions, set `FRAUDLENS_PUBLIC_DEMO=1` in the server environment before launch. Public sessions show a synthetic email example and hide configuration, sending controls and local notification history. The app is served on localhost.

## Configure an authorized account and test delivery

1. Use a mail account you control or have explicit approval to use. Get the SMTP host, port, encryption mode, permitted sender address and account name from its provider. Use an SMTP-specific or app password if the provider requires one. The relay must support password-based SMTP over TLS.
2. Stop a running local app. In a fresh PowerShell window, enter the provider settings when prompted and launch the server from the same window. Enter exactly one authorized recipient address. Use a mailbox you are permitted to test.
3. Open **Alert Center**. Check the authorization box, then select **Activate real delivery**. Select **Send Test Email** once and wait for the history status. **SMTP accepted** means the relay accepted the message. Check the authorized inbox separately to confirm receipt.
4. Select **Dry run** to disarm real delivery before replaying if your intent was a test only. To remove credentials from the local process environment, stop the app and close the PowerShell window.

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:FRAUDLENS_PUBLIC_DEMO = '0'
$env:FRAUDLENS_SMTP_HOST = Read-Host 'Provider SMTP host'
$env:FRAUDLENS_SMTP_PORT = Read-Host 'Provider SMTP port'
$env:FRAUDLENS_SMTP_TLS = Read-Host 'Enter ssl for implicit TLS or starttls for explicit TLS'
$env:FRAUDLENS_SMTP_FROM = Read-Host 'Authorized sender address'
$env:FRAUDLENS_SMTP_TO = Read-Host 'One authorized recipient address'
$env:FRAUDLENS_SMTP_USERNAME = Read-Host 'SMTP account name'
$smtpSecret = Read-Host 'SMTP password or app password' -AsSecureString
$env:FRAUDLENS_SMTP_PASSWORD = [System.Net.NetworkCredential]::new('', $smtpSecret).Password
Remove-Variable smtpSecret
$env:FRAUDLENS_SMTP_ENABLED = '1'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m streamlit run archive/apps/app_research.py
```

The server flag exposes the activation control. Every browser session still starts in dry run. Real delivery requires both the confirmation and the activation click. If the provider rejects the test, inspect Alert Center’s redacted failure status and verify its host, TLS mode, credentials and sender permissions.

## Delivery and privacy behavior

- SMTP uses certificate-verified implicit TLS or STARTTLS. Two bounded background worker slots keep network operations out of replay inference.
- Each browser session allows **five real send attempts**, at least **30 seconds apart**. Tests and replay alerts use the same allowance, and failed attempts count. Rate-limited or busy events are recorded with their status. The app does not retry failures or ambiguous timeouts.
- Replay reset disarms real delivery and preserves history, the rate limiter and the remaining quota. New sends need explicit reactivation. An SMTP operation already in progress may still finish. The quota and latest 80 history entries are kept in browser-session memory.
- The email payload excludes historical transaction IDs, replay offsets, timestamps from the source dataset, raw transaction fields and observed labels. Actual historical case data and the local screenshot stay in ignored local artifacts.

## Reproduce verification

```powershell
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m pytest -q tests/test_v3_alerts.py tests/test_v3_replay.py
```

For Chrome verification, launch the app with `--server.port 8520` and leave SMTP opt-in off. Use the temporary Playwright installation from the [dashboard guide](V3_DASHBOARD.md).

```powershell
$qa = Join-Path ([IO.Path]::GetTempPath()) 'fraudlens-v3-browser-qa'
$env:PYTHONPATH = Join-Path $qa 'packages'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B scripts/check_v3_alerts_browser.py --chrome 'C:\Program Files\Google\Chrome\Application\chrome.exe' --screenshots docs/screenshots/v3 --local-screenshots
```

[Alert Center screenshot](screenshots/v3/06-alert-center.png) · The local dry-run history capture is saved under the ignored `artifacts/v3/step10-alerts/` directory.

## Verified result

- **150 automated tests passed**, including nine Step 10 checks. Pytest reported three existing SHAP dependency deprecation warnings.
- **Chrome 158** verified notifications from actual frozen-model replay, dry-run tests, disabled mode, reset, deduplication, confirmation controls and public-demo isolation. The browser test used reserved dummy SMTP settings and left real delivery inactive.
- **Mocked relays** verified both secure connection modes, accepted delivery, authentication and recipient failures, rate limiting and the shared session limit. No real messages were sent. Receipt in an authorized inbox remains an operator verification step.
- **Artifact integrity** — All 19 frozen Step 7A/7B digests, all 11 recorded Step 7C output digests, and the mapped feature matrix matched.
