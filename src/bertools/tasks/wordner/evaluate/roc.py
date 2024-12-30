import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from bertools.tasks.wordner.typing import Output


def compute_roc_values(
    gold_outputs: list[Output], pred_outputs: list[Output]
) -> dict[str, tuple[list[int | float], ...]]:
    """
    Compute False Positive Rate (FPR) vs. Recall curve values of the retrieval of the
    different categories.
    """
    # reshape to 'id' -> {'toxic': bool, 'priority': bool, 'spans': list, **extra_cols}
    golds = {m["id"]: m for m in gold_outputs}
    preds = {m["id"]: m for m in pred_outputs}

    msg_ids = list(golds.keys())
    label_list = list({sp["label"] for v in golds.values() for sp in v["spans"]})
    roc_values: dict[str, tuple[list[int | float], ...]] = {}

    for label in label_list:
        y_true = [any(sp["label"] == label for sp in golds[k]["spans"]) for k in msg_ids]
        y_pred = [
            max([0] + [sp["confidence"] * int(sp["label"] == label) for sp in preds[k]["spans"]]) for k in msg_ids
        ]
        fpr, rec, ths = roc_curve(y_true, y_pred)
        fpr = [round(100 * r, 2) for r in fpr]
        rec = [round(100 * r, 2) for r in rec]
        ths = [round(t, 2) for t in ths]
        roc_values[label] = (fpr, rec, ths)
    return roc_values


def compute_roc_curves(roc_values: dict[str, tuple[list[int | float], ...]]) -> go.Figure:
    """
    Compute False Positive Rate (FPR) vs. Recall curve of the retrieval of the
    different categories.
    """
    labels = list(roc_values.keys())
    colors = px.colors.sequential.Plasma[: len(labels)]
    figure = go.Figure()
    for label, color in zip(labels, colors):
        (fpr, rec, ths) = roc_values[label]
        hovertext = [
            f"Category: {label}<br>Recall: {r}<br>FPR: {f}<br>Threshold: {t}" for f, r, t in zip(fpr, rec, ths)
        ]
        figure.add_trace(
            go.Scatter(
                x=fpr,
                y=rec,
                hovertext=hovertext,
                hoverinfo="text",
                name=label,
                marker={"color": color},
                mode="lines",
            )
        )

    figure.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="Recall",
        xaxis_range=[0, 10],
        width=1000,
        height=700,
    )
    return figure


def compute_safety_thresholds(
    roc_values: dict[str, tuple[list[int | float], ...]], target_fpr: float
) -> dict[str, float]:
    """
    Compute the least threshold guaranteeing a target FPR for each individual toxic categories.
    """
    return {
        label: min([1.0] + [t for f, t in zip(fpr, ths) if f <= target_fpr])
        for label, (fpr, rec, ths) in roc_values.items()
        if max(fpr[:-1]) > target_fpr
    }
