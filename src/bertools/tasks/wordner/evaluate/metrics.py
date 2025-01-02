from typing import Any

import pandas as pd

from bertools.tasks.wordner.typing import Output


def compute_metrics(gold_outputs: list[Output], pred_outputs: list[Output]) -> dict[str, Any]:
    """
    Computes precision, recall and f1-scores of toxic span retrieval,
    considering exact matches, overlaps and cooccuring spans.
    """
    labels = list({sp["label"] for r in gold_outputs for sp in r["spans"]})

    # computes 'id' -> [('start', 'end', 'label')] view
    golds = {r['id']: [(sp["start"], sp["end"], sp["label"]) for sp in r["spans"]] for r in gold_outputs}
    preds = {r['id']: [(sp["start"], sp["end"], sp["label"]) for sp in r["spans"]] for r in pred_outputs}

    # computes [(label, cooccur, overlaps, matches)] view
    golds_found = [
        (v[-1], cooccurs(v, preds[k]), overlaps(v, preds[k]), matches(v, preds[k]))
        for k, vs in golds.items()
        for v in vs
    ]
    preds_found = [
        (v[-1], cooccurs(v, golds[k]), overlaps(v, golds[k]), matches(v, golds[k]))
        for k, vs in preds.items()
        for v in vs
    ]
    # cast to dataframes, for subsequent groupy operation
    columns = ["label", "cooccur", "overlap", "exact match"]
    df_golds_found = pd.DataFrame(golds_found, columns = columns)
    df_preds_found = pd.DataFrame(preds_found, columns = columns)

    # compute metrics
    rec_metrics = compute_stats(df_golds_found, "recall", labels)
    prc_metrics = compute_stats(df_preds_found, "precision", labels)
    merged_metrics = merge_metrics(rec_metrics, prc_metrics)
    all_metrics = add_f1_score(merged_metrics)
    all_metrics = {k: recast_value(k, v) for k, v in all_metrics.items()}
    return all_metrics


def cooccurs(span: tuple[int, int, str], span_list: list[tuple]) -> bool:
    """
    Check if a span label also occurs in a list of other spans.
    """
    return span[-1] in {sp[-1] for sp in span_list}


def overlaps(span: tuple[int, int, str], span_list: list[tuple]) -> bool:
    """
    Check if a span overlaps an item in a list of other spans.
    """
    start, end, label = span
    return any((label == lab) and not (e <= start or end <= s) for s, e, lab in span_list)


def matches(span: tuple[int, int, str], span_list: list[tuple]) -> bool:
    """
    Check if the span exactly occurs in a list of other spans.
    """
    return span in span_list


def compute_stats(df: pd.DataFrame, keyword: str, labels: list[str]) -> dict[str, Any]:
    def compute_stat(df: pd.DataFrame, field: str, label: str | None = None) -> float:
        g = df[df["label"] == label] if label else df
        return 0.0 if len(g) == 0 else g[field].sum() / len(g)

    fields = ["cooccur", "overlap", "exact match"]
    metrics = {label: {field: {keyword: compute_stat(df, field, label)} for field in fields} for label in labels}
    metrics |= {
        "macro avg": {f: {keyword: sum([v[f][keyword] for v in metrics.values()]) / len(metrics)} for f in fields},
    }
    metrics |= {"weighted avg": {f: {keyword: compute_stat(df, f)} for f in fields}}
    return metrics


def merge_metrics(m1: dict[str, Any], m2: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two nested dicts of metrics.
    """
    fields = ["cooccur", "overlap", "exact match"]
    default_dict: dict[str, Any] = {k: {f: {} for f in fields} for k in set(list(m1) + list(m2))}
    m1 = default_dict | m1
    m2 = default_dict | m2
    return {k: {kw: m1[k][kw] | m2[k][kw] for kw in fields} for k in default_dict}


def compute_f1(recall: float, precision: float) -> float:
    """
    Compute f1-score from recall and precision scores.
    """
    return 0.0 if (recall + precision) == 0.0 else (2 * recall * precision) / (recall + precision)


def add_f1_score(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Compute f1-score out of precision and recall scores found in a nested dict of metrics.
    """
    default_dict = {"recall": 0.0, "precision": 0.0}
    return {
        k: {
            kw: metrics[k][kw] | {"f1-score": compute_f1(**(default_dict | metrics[k][kw]))}
            for kw in ["cooccur", "overlap", "exact match"]
        }
        for k in metrics
    }


def recast_value(k: str, v: Any) -> Any:
    """
    Convert a value to integer or floating percentage, in a recursive way.
    """
    if isinstance(v, dict):
        return {key: recast_value(key, val) for key, val in v.items()}
    elif isinstance(v, int | float):
        if k in ["support", "total"]:
            return int(v)
        else:
            return round(float(v) * 100, 2)
    else:
        raise TypeError(f"'{v}' must be of type 'dict' or 'int' or 'float'")
