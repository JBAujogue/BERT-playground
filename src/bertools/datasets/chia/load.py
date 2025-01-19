import zipfile
from pathlib import Path

import pandas as pd


def load_lines_from_zipfile(zip_file: str | Path) -> pd.DataFrame:
    """
    Load texts from zipfile.
    """
    archive = zipfile.ZipFile(zip_file, "r")
    files = [file for file in archive.namelist() if file.endswith(".txt")]
    texts = []
    for file in files:
        t_id = Path(file).stem
        with archive.open(file, "r") as f:
            lines = [line.decode("utf-8").replace("⁄", "/") for line in f.readlines()]
            texts += [{"text_id": t_id, "line_id": i, "content": line} for i, line in enumerate(lines)]

    return pd.DataFrame.from_records(texts)


def load_spans_from_zipfile(zip_file: str | Path) -> pd.DataFrame:
    """
    Load annotated spans from zipfile.
    """

    def parse_span_metadata(metadata: str) -> list[list[str | int]]:
        """
        Parse span metadata from raw annotation.
        """
        label = metadata.split(" ")[0]
        pairs = [p.split(" ") for p in " ".join(metadata.split(" ")[1:]).split(";")]
        return [[label, int(start), int(end)] for start, end in pairs]

    archive = zipfile.ZipFile(zip_file, "r")
    file_list = [f for f in archive.namelist() if f.endswith(".ann")]
    span_list = []
    for file in file_list:
        t_id = Path(file).stem
        with archive.open(file, "r") as f:
            lines = [line.decode("utf-8").replace("⁄", "/").strip() for line in f.readlines()]
            datas = [line.split("\t") for line in lines if line.startswith("T")]
            spans = [
                {
                    "text_id": t_id,
                    "span_id": sp_id,
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": sp_text,
                }
                for sp_id, sp_data, sp_text in datas
                for label, start, end in parse_span_metadata(sp_data)
            ]
            span_list += spans

    return pd.DataFrame.from_records(span_list)
