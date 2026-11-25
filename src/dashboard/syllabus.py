"""Read-only supplementary benchmarks inside Model Lab's Deep learning tab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .artifacts import ArtifactError, generation_commands
from .logic import MODEL_NAMES
from .ui import AMBER, BLUE, TEAL, chart_layout, plot

LABELS = {**MODEL_NAMES, "gaussian_nb": "Gaussian Naive Bayes", "knn": "K-nearest neighbors", "linear_svm": "Linear SVM",
          "adaboost": "AdaBoost", "numpy_linear": "NumPy linear", "numpy_relu32": "NumPy ReLU-32",
          "sklearn_objective_reference": "scikit-learn logistic reference", "torch_relu32_reference": "PyTorch ReLU-32 reference"}
METRICS = {"average_precision": "AP", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall", "f1": "F1",
           "accuracy": "Accuracy", "specificity": "Specificity"}


def _table(keys, models, metrics, partition, checkpoints=False):
    rows = []
    for key in keys:
        info = models[key]
        row = {"Model": LABELS[key], **{label: metrics[key][metric] for metric, label in METRICS.items()},
               "Fit (s)": info["fit_seconds"], "Inference (s)": info[f"{partition}_inference_seconds"]}
        if checkpoints:
            row.update({"Parameters": info.get("parameter_count"), "Best epoch": info.get("best_epoch"),
                        "Validation threshold": info["threshold"]})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={column: st.column_config.NumberColumn(format="%.5f")
                 for column in (*METRICS.values(), "Fit (s)", "Inference (s)", "Validation threshold")})


def _classifiers(manifest, metrics, partition):
    st.subheader("Matched classifiers")
    models = manifest["models"]
    _table(manifest["benchmark_models"], models, metrics, partition)
    st.caption("Linear SVM uses raw decision margins. Its validation cutoff is in the same margin scale. "
               "All other classifier scores come from positive-class predict_proba.")
    with st.expander("Model settings and class imbalance"):
        for key in manifest["benchmark_models"]:
            info = models[key]
            threshold = info["threshold"]
            rule = "majority rule" if threshold is None else f"frozen threshold {threshold:.6g}"
            st.markdown(f"**{LABELS[key]}** · {info['imbalance']} · {rule}")


def _scratch(run, manifest, metrics, partition):
    history = run.report(6, "history")["models"]
    checks = run.report(6, "verification")
    comparisons = run.report(6, "reference_comparison")["pairs"]
    models = manifest["models"]
    numpy_names = ("numpy_linear", "numpy_relu32")
    rankings = (*numpy_names, "sklearn_objective_reference", "torch_relu32_reference")
    dimension = manifest["features"]["transformed_dimension"]
    st.subheader("NumPy networks from scratch")
    linear, hidden = st.columns(2, gap="medium")
    linear.metric(f"{dimension} inputs → one logit", f"{models['numpy_linear']['parameter_count']:,} parameters", border=True)
    hidden.metric(f"{dimension} inputs → 32 ReLU → one logit", f"{models['numpy_relu32']['parameter_count']:,} parameters", border=True)
    st.write("Forward propagation forms logits. Weighted binary cross-entropy measures classification error. "
             "Backpropagation uses the chain rule to compute weight and bias gradients, and mini-batch gradient descent applies the updates.")
    st.caption("Both variants use stable sigmoid scores, training-derived positive weights and L2 on weights. "
               "Validation AP selects the checkpoint. Validation F1 selects the threshold.")
    _table(rankings, models, metrics, partition, checkpoints=True)

    first, second = st.columns(2, gap="medium")
    with first, st.container(border=True):
        st.markdown("**Training objective by epoch**")
        fig = go.Figure()
        for color, name in zip((BLUE, TEAL), numpy_names, strict=True):
            rows = pd.DataFrame(history[name]["epochs"])
            fig.add_scatter(x=rows["epoch"], y=rows["train_loss"], mode="lines", name=LABELS[name], line=dict(color=color))
            fig.add_scatter(x=rows["epoch"], y=rows["validation_loss"], mode="lines", name=f"{LABELS[name]} · val",
                            line=dict(color=color, dash="dot"))
        plot(chart_layout(fig, height=290, x="Epoch", y="Weighted BCE + L2"), "syllabus_numpy_loss")
    with second, st.container(border=True):
        st.markdown("**Validation average precision**")
        fig = go.Figure()
        for color, name in zip((BLUE, TEAL, AMBER), (*numpy_names, "torch_relu32_reference"), strict=True):
            rows = pd.DataFrame(history[name]["epochs"])
            fig.add_scatter(x=rows["epoch"], y=rows["validation_ap"], mode="lines", name=LABELS[name], line=dict(color=color))
            best = history[name]["best_epoch"]
            fig.add_scatter(x=[best], y=[history[name]["best_validation_ap"]], mode="markers", showlegend=False,
                            marker=dict(color=color, symbol="diamond", size=9))
        plot(chart_layout(fig, height=290, x="Epoch", y="Average precision"), "syllabus_numpy_ap")

    st.subheader("Gradient verification and reference comparison")
    numeric = []
    for name in numpy_names:
        check = checks["models"][name]
        numeric.append({"Architecture": LABELS[name], "Fixture hidden units": check["fixture_hidden_width"],
                        "Max finite-difference error": check["max_gradient_absolute"],
                        "Max autograd, loss, logit or update error": check["max_pytorch_absolute"]})
    st.dataframe(pd.DataFrame(numeric), hide_index=True, column_config={key: st.column_config.NumberColumn(format="%.3e")
                 for key in ("Max finite-difference error", "Max autograd, loss, logit or update error")})
    st.caption("Float64 synthetic checks use identical initialization, weighting, regularization and one gradient-descent step.")
    comparison = [{"NumPy": LABELS[item["numpy"]], "Reference": LABELS[item["reference"]],
                   "Mean score difference": item["partitions"][partition]["mean_absolute_score_difference"],
                   "Max score difference": item["partitions"][partition]["maximum_absolute_score_difference"]} for item in comparisons]
    st.dataframe(pd.DataFrame(comparison), hide_index=True, column_config={key: st.column_config.NumberColumn(format="%.5g")
                 for key in ("Mean score difference", "Max score difference")})
    st.caption("Real-data references use the same rows and objective. The logistic reference uses L-BFGS, and the shallow reference uses PyTorch SGD.")


def render_syllabus(run, partition):
    st.divider()
    st.subheader("Syllabus Benchmark")
    try:
        run.require_optional(6)
        manifest = run.manifests[6]
        samples = manifest["samples"]
        artifact = run.report(6, f"{partition}_metrics")
        if artifact["partition"] != partition or artifact["rows"] != samples[partition]["rows"]:
            raise ArtifactError(6, "Supplementary metric rows do not match the saved evaluation sample.")
        st.caption(f"Training {samples['train']['rows']:,} · Validation {samples['validation']['rows']:,} · "
                   f"{run.partition_name('test')} {samples['test']['rows']:,} · {manifest['features']['transformed_dimension']} encoded features")
        st.write(f"Showing the **{run.partition_name(partition).lower()} subset**. All models in this benchmark use the same sampled rows. "
                 "The primary comparison above uses larger partitions.")
        view = st.radio("Supplementary view", ("Matched classifiers", "From-scratch networks"), horizontal=True, key="syllabus_view")
        if view == "Matched classifiers":
            _classifiers(manifest, artifact["models"], partition)
        else:
            _scratch(run, manifest, artifact["models"], partition)
    except (ArtifactError, KeyError, TypeError, ValueError) as error:
        st.warning(f"Phase 6 results are unavailable for this mode. {error}")
        st.code(generation_commands(run.mode, 6), language="powershell")
