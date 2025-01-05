from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def load_rerank_dataset(data_files: dict[str, str | Path]) -> dict[str, "RerankDataset"]:
    """
    Load the train/eval/test splits of a dataset.
    """
    return {k: RerankDataset.from_path(v) for k, v in data_files.items()}


@dataclass
class RerankDataset:
    """
    Embedding Reranking Dataset.
    """

    pairs: list[list[str]]
    queries: dict[str, str]
    corpus: dict[str, str]
    relevant_docs: dict[str, set[str]]
    mode: str = "text"

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        input_col: str = "query",
        target_col: str = "document",
    ) -> "RerankDataset":
        """
        Load query-doc pairs from a pandas dataframe.
        """
        pairs = df.values.tolist()

        query2idx = df.groupby(input_col).indices
        query2idx = {str(k): set(v.flat) for k, v in query2idx.items()}
        doc2idx = df.groupby(target_col).indices
        doc2idx = {str(k): set(v.flat) for k, v in doc2idx.items()}

        id2query = {str(i): k for i, k in enumerate(query2idx.keys())}
        id2doc = {str(i): k for i, k in enumerate(doc2idx.keys())}

        relevant_docs = {i: {j for j, d in id2doc.items() if (doc2idx[d] & query2idx[q])} for i, q in id2query.items()}
        return cls(pairs, id2query, id2doc, relevant_docs)

    @classmethod
    def from_path(
        cls,
        file_path: str | Path,
        input_col: str = "query",
        target_col: str = "document",
    ) -> "RerankDataset":
        """
        Load query-doc pairs from path to a .tsv file.
        """
        return cls.from_df(pd.read_csv(Path(file_path), sep="\t"), input_col, target_col)
