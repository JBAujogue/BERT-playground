import os
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from bertools.datasets.chia.load import load_lines_from_zipfile, load_spans_from_zipfile
from bertools.datasets.chia.transform import (
    append_lines_to_spans,
    drop_overlapped_spans,
    flatten_spans,
)


def build_ner_dataset(
    zip_file: str | Path,
    output_dir: str | Path,
    flatten: bool = False,
    drop_overlapped: bool = False,
) -> None:
    """
    Load chia dataset and prepare its annotated spans.
    """
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok=False)

    # load raw data
    df_lines = load_lines_from_zipfile(zip_file)
    df_spans = load_spans_from_zipfile(zip_file)
    logger.info(f"Loaded {len(df_lines)} lines and {len(df_spans)} spans.")

    # reconstruct line id and relative span offsets
    df_spans = append_lines_to_spans(df_lines, df_spans)
    logger.info(f"Reconstructed line id for {len(df_spans)} spans.")

    if flatten:
        # expand spans to consecutive words
        df_spans = flatten_spans(df_spans)
        logger.info(f"Converted to {len(df_spans)} flattened spans.")

    if drop_overlapped:
        # keep a maximal non-overlapping family of spans
        df_spans = drop_overlapped_spans(df_spans)
        logger.info(f"Converted to {len(df_spans)} non-overlapping spans.")

    # compute train / eval / test splits
    df_train, df_eval, df_test = split_spans(
        df_spans,
        train_ratio=0.8,
        eval_test_ratio=0.5,
    )
    logger.info(
        (
            f"Splitted spans into:"
            f"\n- {len(df_train)} train spans"
            f"\n- {len(df_eval)} eval spans"
            f"\n- {len(df_test)} test spans"
        )
    )
    # save splits to parquet
    df_train.to_parquet(out_dir / "train.parquet", index=False)
    df_eval.to_parquet(out_dir / "eval.parquet", index=False)
    df_test.to_parquet(out_dir / "test.parquet", index=False)
    logger.info(f"Splits saved to {output_dir}.")
    return


def split_spans(
    df_spans: pd.DataFrame,
    train_ratio: float,
    eval_test_ratio: float,
    random_state: int = 0,
) -> tuple[pd.DataFrame, ...]:
    """
    Split a DataFrame of spans into train/eval/test splits.
    """
    # split by text id
    ids = sorted(df_spans["text_id"].apply(lambda t_id: t_id.split("_")[0]).unique().tolist())
    ids_train, ids_dev = train_test_split(
        ids,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
    )
    ids_eval, ids_test = train_test_split(
        ids_dev,
        train_size=eval_test_ratio,
        random_state=random_state,
        shuffle=True,
    )
    ids_serie = df_spans["text_id"].apply(lambda t_id: t_id.split("_")[0])
    df_train = df_spans[ids_serie.isin(ids_train)].reset_index(drop=True)
    df_eval = df_spans[ids_serie.isin(ids_eval)].reset_index(drop=True)
    df_test = df_spans[ids_serie.isin(ids_test)].reset_index(drop=True)
    return (df_train, df_eval, df_test)
