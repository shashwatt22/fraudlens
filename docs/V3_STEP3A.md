# FraudLens V3 Step 3A full-data Logistic Regression

Mean AP on the completed folds changed from **0.36284** to **0.40418**, a change of +0.04134 (+11.39% relative).

The input is the verified 497-column V2 feature matrix. This step excludes the 13 Step 2 features, which remain in the separate completed experiment. The three validation windows contain 70,865, 70,865 and 70,864 rows. The sampled and full models score identical validation rows and share the original evaluation cutoff of 0.5. Review budgets rank the same rows, using source order to break ties and rounding up to the nearest row. Summary metrics equally weight the folds.

## Preprocessing and solver

The archived `logistic_l2` and the full-data fit both use unweighted L2 (`C=0.08`), L-BFGS, an intercept, tolerance `1e-4` and a maximum of 1,000 iterations. The experiment limits native libraries to 6 CPU threads where supported. The same `C` preserves the configured regularization. With more rows, the penalty is smaller relative to the summed data loss.

Each full training prefix refits categorical frequencies, median imputation, ANOVA ranking of 192 numeric or missingness inputs, scaling and the original one-hot encoder (`min_frequency=7`, `max_categories=96`). Preprocessing values can change because they are learned again on the larger training set. The implementation fits imputation and ranking in column batches, accumulates scaling moments by row batch, and builds one float64 CSR matrix for L-BFGS. Validation applies the fitted transform without altering it. No raw, policy-block, or retrospective scores enter fitting or evaluation.

## Paired validation results

| Fold | Model | Fit rows | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Recall at 5% | Recall at 10% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | sampled | 16,000 | 0.36684 | 0.80158 | 0.70916 | 0.23623 | 0.35441 | 0.96340 | 0.38155 | 0.46317 |
| 1 | full | 212,594 | 0.43517 | 0.84224 | 0.73769 | 0.27339 | 0.39893 | 0.96496 | 0.44691 | 0.54346 |
| 2 | sampled | 16,000 | 0.30646 | 0.79044 | 0.71827 | 0.20783 | 0.32238 | 0.96945 | 0.36441 | 0.46045 |
| 2 | full | 283,459 | 0.32951 | 0.80093 | 0.73669 | 0.21227 | 0.32957 | 0.96980 | 0.39588 | 0.49475 |
| 3 | sampled | 16,000 | 0.41521 | 0.84026 | 0.71560 | 0.29016 | 0.41290 | 0.96557 | 0.44133 | 0.53669 |
| 3 | full | 354,324 | 0.44786 | 0.85417 | 0.80356 | 0.30572 | 0.44292 | 0.96791 | 0.45519 | 0.56983 |
| Mean | sampled | — | 0.36284 | 0.81076 | 0.71434 | 0.24474 | 0.36323 | 0.96614 | 0.39576 | 0.48677 |
| Mean | full | — | 0.40418 | 0.83245 | 0.75931 | 0.26379 | 0.39048 | 0.96756 | 0.43266 | 0.53602 |
| Mean change | full − sampled | — | 0.04134 | 0.02169 | 0.04497 | 0.01905 | 0.02725 | 0.00142 | 0.03690 | 0.04925 |

## Convergence and resources

Training time includes preprocessing, sparse matrix construction and the solver. Peak memory is the fold worker's Windows process working set. The supervisor runs one worker at a time under a 30-minute wall-clock budget, with a 7.5 GB memory guard.

| Fold | Fit fraud rows | CSR shape | Training (s) | Solver (s) | Validation (s) | Peak GiB | L-BFGS iterations | Converged |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6,496 | 212,594 × 1,122 | 122.94 | 67.55 | 6.11 | 1.31 | 185 | Yes |
| 2 | 9,510 | 283,459 × 1,144 | 152.32 | 102.39 | 2.71 | 1.63 | 229 | Yes |
| 3 | 11,988 | 354,324 × 1,160 | 192.33 | 101.48 | 1.73 | 1.94 | 260 | Yes |

Total measured training time was 467.59 s. The fold workers took 504.09 s including control verification and checkpointing. Each sampled control score array matched its saved model's predictions exactly. Fitting the chunked preprocessing on each original 16,000-row sample reproduced the saved feature choices and transformation with maximum probe difference `0`. Reloading each full model and preprocessor reproduced its validation probe scores exactly.

## Reproduce or resume

Each completed fold has an independent fitted preprocessor, model, score array, and metric manifest. Intact folds are reused when the same command resumes. The artifact directory stays local and is ignored by Git. A new output path gives a clean reproduction.

```powershell
Set-Location E:\fraudlens-v3
$python = 'E:\fraudlens\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'E:\fraudlens-v2\.venv\Lib\site-packages'
$env:PYTHONIOENCODING = 'utf-8'
$out = 'artifacts\v3\step3a-logistic'
& $python -B -m src.v3.logistic_experiment --development 'E:\fraudlens-v2\artifacts\v2' --output $out
& $python -B -m src.v3.logistic_report --output $out --markdown 'docs\V3_STEP3A.md'
```
