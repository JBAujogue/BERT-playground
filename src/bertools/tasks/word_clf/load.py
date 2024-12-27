from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from loguru import logger

from ..taxonomy import TAXONOMY_DEFAULT_CATEGORY, load_taxonomy
from ..transforms import gather_spans
from ..types import MESSAGE_FIELDS, SPAN_FIELDS, SPAN_GATHERING_FIELDS, Record


def fill_docstring(*args, **kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec


def load_annotated_channels(file_list: Iterable[str | Path], taxonomy_path: str) -> dict[str, list[Record]]:
    """
    Load records from provided list of files, and organize them into dict of channels,
    each channel containing a list of records sorted by timestamp.
    """
    df_expanded = pd.concat((load_annotation(f, taxonomy_path) for f in file_list), ignore_index=True)
    df_gathered = gather_annotation(df_expanded)
    return {
        c_id: [Record(**r) for r in c.sort_values(by="timestamp").to_dict("records")]  # type: ignore
        for c_id, c in df_gathered.groupby("channel_id")
    }


@fill_docstring(msg_cols=MESSAGE_FIELDS, span_cols=SPAN_FIELDS)
def load_annotation(file_path: str | Path, taxonomy_path: str | Path) -> pd.DataFrame:
    """
    Load annotations of chat messages, with one line per toxic span or fully safe message.
    Outputed DataFrame has the following columns:
        - Message-level metadata {msg_cols}
        - Span-level metadata {span_cols}
    """
    # load file
    if str(file_path).endswith(".csv"):
        df = pd.read_csv(file_path, sep=";")
    elif str(file_path).endswith(".tsv"):
        df = pd.read_csv(file_path, sep="\t")
    elif str(file_path).endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise NotImplementedError(f"'{file_path}' extension must be '.csv' or '.parquet'")

    taxonomy = load_taxonomy(taxonomy_path)

    # map column names to our names internally used
    df = df.rename(
        columns={
            "matchid": "channel_id",
            "profileid": "profile_id",
            "line_index": "message_id",
            "relative_timestamp": "timestamp",
            "full_text": "content",
        }
    )
    # drop messages with nan content
    nans = df.content.isna().sum()
    if nans > 0:
        logger.warning(f"{nans} messages with NaN content found in file '{file_path}'")
        df = df.loc[df.content.notna()].reset_index(drop=True)

    # add message-level metadatas
    df["id"] = df.apply(func=lambda r: f"{r['channel_id']}-{r['message_id']}", axis=1)
    df["version"] = "human"

    # add span-level metadatas
    # sanity check: ensure that start/end offsets don't fall ouside the message content
    df["label_id"] = df["category_id"].apply(lambda i: taxonomy[i]["label_id"])
    df["label"] = df["category_id"].apply(lambda i: taxonomy[i]["label"])
    df["priority"] = df["category_id"].apply(lambda i: taxonomy[i]["priority"])
    df["start"] = df["start_string_index"].fillna("0").apply(lambda i: max(int(i), 0))
    df["end"] = df["end_string_index"].fillna("0").apply(int)
    df["end"] = df.apply(func=lambda r: min(r.end, len(r.content)), axis=1)
    df["text"] = df.apply(func=lambda r: r.content[r.start : r.end], axis=1)
    df["confidence"] = 1.0

    # filter columns of interest
    return df[MESSAGE_FIELDS + SPAN_FIELDS].drop_duplicates(ignore_index=True)


@fill_docstring(msg_cols=MESSAGE_FIELDS, span_cols=SPAN_GATHERING_FIELDS)
def gather_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one line per unique message, with annotated spans gathered together.
    Outputed DataFrame has the following columns:
        - Message-level metadata {msg_cols}
        - Span-level metadata {span_cols}
    """
    return (
        df.groupby(MESSAGE_FIELDS)
        .apply(
            lambda g: pd.Series(gather_spans(g.to_dict("records"), TAXONOMY_DEFAULT_CATEGORY)),
            include_groups=False,
        )
        .reset_index()
    )
