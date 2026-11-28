# FraudLens V3 Live Monitor

This guide documents the archived research workbench. For the current two-page app, see [FraudLens V3 manual assessment](V3_APPLICATION.md).

Live Monitor replays the original 118,108-row reused retrospective holdout in source order. Equal timestamps retain their saved order. Each replay batch runs CPU inference with the refitted depth-10 XGBoost checkpoint, identity calibration and the inclusive frozen threshold **0.1938110143**.

## Launch on Windows

```powershell
Set-Location 'E:\fraudlens-v3'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m streamlit run archive/apps/app_research.py
```

Open [FraudLens on localhost](http://127.0.0.1:8501) and select **Live Monitor**. The existing local artifact paths are used by default. See the [dashboard setup](V3_DASHBOARD.md) for path overrides.

## Classroom demonstration

1. In **Live Monitor**, keep the sidebar privacy switch off. Select **Reset** if a replay already exists.
2. Select **Step one** four times. The processed count advances from 1 to 4. The flagged counts are **0, 0, 1, 2**. The review queue contains replay rows 3 and 4.
3. Select a flagged case and choose **Investigate flagged transaction**. Calculate TreeSHAP and point out the score reconstruction check. Use **Reveal observed label** only after discussing the model decision.
4. Return to **Live Monitor**. The replay is paused at row 4. Select a speed and choose **Resume**. Use **Pause** to freeze the count, then **Reset** to return to the first transaction.
5. To show the interface without individual records, open [the aggregate-only view](http://127.0.0.1:8501/?aggregate=1). Its replay controls are disabled.

## Inference and replay design

- The replay source reads only transaction IDs, dataset seconds and amounts. It checks the order against the frozen feature timeline and chronological boundaries. Ground-truth labels are available through the investigator’s explicit reveal control.
- The feature matrix comes from Step 7C and is memory-mapped and hash-verified. History excludes the entire current timestamp group. The existing V3 model adapter loads the frozen booster and reconstructs the same category boundaries from the original fit prefix. The simulator computes each risk score with native XGBoost inference. The investigator cross-checks that score against the frozen case result.
- A Streamlit fragment schedules a bounded batch about once per second while replay is running. The five speed settings are 1, 5, 20, 50 and 100 transactions per tick. Slow clients skip catch-up work. Leaving the page, hiding local records or opening an investigation pauses playback. The first inference may take a moment as the app verifies and loads local artifacts.
- Each session retains the latest 30 scored transactions and the latest 30 flagged events. Total processed and flagged counts cover the entire replay. Reloading the browser starts a new session.
- A successful step or tick returns a tuple of new `AlertEvent` objects. The event ID combines the replay instance ID with its one-based row sequence. The cursor commits after batch validation, so early reruns emit no duplicates. Reset creates a new replay instance. Events carry a historical-replay source tag and no observed label. Event retention is in memory.

## Verification and captures

Run the repository tests from the launch environment above.

```powershell
& 'E:\fraudlens\.venv\Scripts\python.exe' -B -m pytest -q -rs
```

For the Chrome check, launch the app with `--server.port 8519` and use the temporary Playwright environment described in the [dashboard guide](V3_DASHBOARD.md).

```powershell
$qa = Join-Path ([IO.Path]::GetTempPath()) 'fraudlens-v3-browser-qa'
$env:PYTHONPATH = Join-Path $qa 'packages'
& 'E:\fraudlens\.venv\Scripts\python.exe' -B scripts/check_v3_replay_browser.py --chrome 'C:\Program Files\Google\Chrome\Application\chrome.exe' --screenshots docs/screenshots/v3 --local-screenshots
```

The [tracked Live Monitor screenshot](screenshots/v3/05-live-monitor.png) shows the aggregate-only view. Chrome captures showing the real replay and its explanation are stored in the ignored `artifacts/v3/step9-replay/` directory.

- **Full inference parity** — A label-free audit scored all 118,108 rows in batches of at most 100. The maximum absolute difference from the saved Step 7C scores was 0. All frozen decisions matched, including 3,378 flags. Batch scoring took 11.27 seconds after artifact loading, with no UI pacing.
- **Automated tests** — 141 passed, including six Step 9 tests. Pytest reported three existing SHAP dependency deprecation warnings.
- **Chrome 158** — Verified the four-step classroom sequence, timer controls, speed adjustment, pause on navigation, duplicate-free review queue, investigator handoff, TreeSHAP, explicit label reveal and aggregate-only mode.
- **Integrity** — All 19 Step 7A/7B artifact digests and 11 Step 7C output digests matched. The memory-mapped feature matrix also matched.
