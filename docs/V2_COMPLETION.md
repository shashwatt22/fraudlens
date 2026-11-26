# FraudLens V2 completion record

The [experiment record](V2_EXPERIMENTS.md) includes all 36 supervised trial configurations across three expanding development folds, all frozen retrospective scorers and baseline comparisons, the unsupervised experiments, calibration, thresholds and review budgets. The [protocol](V2_PROTOCOL.md) describes the fitting and retrospective access rules.

The V2 checkpoint passed all 93 tests, including seven integration tests that use the existing V1 artifacts. The dependency check found no broken requirements. Streamlit verified the [V1/V2 comparison](screenshots/dashboard-v2-comparison.jpg), the V2 queue hand-off, the component-specific TreeSHAP explanation, the [patterns panel](screenshots/dashboard-v2-patterns.jpg) and the 100,000-row V1 Quick view.

V1 integrity matched 279 archived file hashes, including 200 experiment files, both raw CSVs and the original tracked source snapshot. All 60 migrated `DenseFeatures` checkpoints reloaded under the stable module name. All 19 frozen finalist checkpoints matched their model hashes and reproduced the saved third-fold development scores with maximum absolute difference `2.22e-16`.

The one-time retrospective evaluation remains the experiment from 25 November 2026. Verification did not reopen its labels or rerun any training. Use these SHA-256 values to identify that result.

| Existing artifact | SHA-256 |
| --- | --- |
| `retrospective_access.json` | `c291e20e23cb48925456f3981247e9806a9072312aaa668570505d861ddee57d` |
| `retrospective_results.json` | `5685c900bc9d58f5eda3d40f1db69c9db357631f87cd08b2c442b97e887edbd2` |

Original data, score rows, models and caches remain in local ignored artifact directories. V2.1 uses a separate experiment directory.
