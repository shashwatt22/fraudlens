# Archived FraudLens interfaces and experiments

The [final project reference](../FINAL_PROJECT_REFERENCE.md) describes the submission. The two-page product launches from the root `app.py`. Earlier Streamlit interfaces are isolated under `archive/apps` and can still be launched from the repository root with the same Python environment and local artifact roots.

```powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$env:FRAUDLENS_SMTP_ENABLED = '0'
& $python -B -m streamlit run archive/apps/app_research.py --server.port 8530
# Run the V1 workbench from a separate terminal.
& $python -B -m streamlit run archive/apps/app_legacy.py --theme.base dark --server.port 8531
```

| Archive | Records |
| --- | --- |
| V1 preprocessing, models and supplementary analyses | [V1 summary](../PROJECT_SUMMARY.md), [experiment protocol](../EXPERIMENT_PROTOCOL.md), [phase notes](../../notes). |
| V2 benchmark and ensembles | [Protocol](../V2_PROTOCOL.md), [experiments](../V2_EXPERIMENTS.md), [completion](../V2_COMPLETION.md). |
| V3 development experiments | [Setup and diagnostics](../V3_STEP1.md), [additional behavioral features](../V3_STEP2.md), [Logistic Regression](../V3_STEP3A.md), [Random Forest](../V3_STEP4A.md), [XGBoost depth](../V3_STEP5A.md), [baseline MLP](../V3_STEP6A.md), [richer-input MLP](../V3_STEP6B.md). |
| Frozen V3 research | [Refit](../V3_STEP7A.md), [policy](../V3_STEP7B.md), [evaluation](../V3_STEP7C.md). |
| Earlier interfaces and studies | [V3 workbench](../V3_DASHBOARD.md), [historical replay](../V3_STEP9.md), [replay alerts](../V3_STEP10.md), [public demo history](../V3_STEP11.md). |
| Manual assessment | [Archived four-input release](../V3_APPLICATION.md), [active six-input challenger](../V3_MANUAL_CHALLENGER.md). |

Experiment records retain their established paths, which the report generators and manifests reference. Model code, ignored experiment outputs, local checkpoints and recorded metrics are preserved. Use the records’ original inputs and output arguments, with the submission’s single environment in place of older Python and PYTHONPATH examples.
