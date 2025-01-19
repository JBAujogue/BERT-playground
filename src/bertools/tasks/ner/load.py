from pathlib import Path
from typing import Iterable

import pandas as pd

from bertools.tasks.ner.typing import LINE_FIELDS, SPAN_FIELDS, SPAN_GATHERING_FIELDS, Record


def fill_docstring(*args, **kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec


def load_annotations(file_list: Iterable[str | Path]) -> dict[str, list[Record]]:
    """
    Load records from provided list of files, and organize them into dict of channels,
    each channel containing a list of records sorted by timestamp.
    """
    df_expanded = pd.concat((load_annotation(f) for f in file_list), ignore_index=True)
    df_gathered = gather_annotation(df_expanded)
    return {
        t_id: [Record(**r) for r in t.sort_values(by="line_id").to_dict("records")]  # type: ignore
        for t_id, t in df_gathered.groupby("text_id")
    }


@fill_docstring(line_cols=LINE_FIELDS, span_cols=SPAN_FIELDS)
def load_annotation(file_path: str | Path) -> pd.DataFrame:
    """
    Load annotations of text lines, with one line per toxic span or fully safe line.
    Outputed DataFrame has the following columns:
        - line-level metadata {line_cols}
        - span-level metadata {span_cols}
    """
    # load file
    df = pd.read_parquet(file_path)

    # add metadatas to conform datamodels
    df["id"] = df.apply(func=lambda r: f"{r['text_id']}-{r['line_id']}", axis=1)
    df["confidence"] = 1.0

    # filter columns of interest
    return df[LINE_FIELDS + SPAN_FIELDS].drop_duplicates(ignore_index=True)


@fill_docstring(line_cols=LINE_FIELDS, span_cols=SPAN_GATHERING_FIELDS)
def gather_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one line per unique text line, with annotated spans gathered together.
    Outputed DataFrame has the following columns:
        - Line-level metadata {line_cols}
        - Span-level metadata {span_cols}
    """
    return (
        df.groupby(LINE_FIELDS)
        .apply(
            lambda g: pd.Series({"spans": [sp for sp in g.to_dict("records") if sp["label"]]}),
            include_groups=False,
        )
        .reset_index()
    )
