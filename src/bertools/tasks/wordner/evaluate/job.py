import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import plotly.graph_objects as go
import yaml  # type: ignore[import-untyped]
from tqdm import tqdm

from bertools.tasks.wordner.evaluate.metrics import compute_metrics
from bertools.tasks.wordner.evaluate.roc import compute_roc_curves, compute_roc_values
from bertools.tasks.wordner.load import load_annotations
from bertools.tasks.wordner.model import WordLevelCausalNER
from bertools.tasks.wordner.transforms.preprocess import concat_lists
from bertools.tasks.wordner.typing import Input, Output


def evaluate(
    config_path: str | Path,
    base_model_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Perform end-to-end evaluation of a word-level NER model against an evaluation .parquet file.
    """
    # raise error if output dir already exists
    output_dir = Path(base_model_dir) / output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # load model & data
    model_args = {"batch_size": bs} if (bs := config.get("batch_size", None)) else {}
    model = WordLevelCausalNER(base_model_dir, **model_args)
    texts = load_annotations(config["data_files"])

    # prepare gold and predicted outputs
    pred_inputs = [[Input(id=r["id"], content=r["content"]) for r in c] for c in texts.values()]
    pred_outputs, elapsed = run_model(model, pred_inputs)
    gold_outputs = [Output(id=r["id"], spans=r["spans"]) for c in texts.values() for r in c]

    # compute evaluation metrics
    curves = compute_roc_curves(compute_roc_values(gold_outputs, pred_outputs))
    scores = compute_metrics(gold_outputs, pred_outputs)
    scores = {
        "metadata": {"data_files": config["data_files"], "time elapsed (s)": elapsed},
        "scores": scores,
    }
    # save results
    save_evaluation(output_dir, scores, curves)
    return


def run_model(model: Callable, texts: Iterable[list[Input]]) -> tuple[list[Output], int]:
    """
    Run model on given dataframe of messages.
    """
    start = time.perf_counter()
    preds = concat_lists([model(text) for text in tqdm(texts)])
    end = time.perf_counter()
    return preds, int(end - start)


def save_evaluation(save_dir: str | Path, scores: dict[str, Any], curves: go.Figure) -> None:
    """
    Saves the computed datasets and metrics into specified folder.
    """
    with open(os.path.join(save_dir, "scores.yaml"), "wt") as f:
        yaml.safe_dump(scores, f, encoding="utf-8")
    curves.write_html(os.path.join(save_dir, "roc_curves.html"))
    return
