import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd


def load_rerank_dataset(data_files: Dict[str, Any]):
    '''
    Load the train/eval/test splits of a dataset.
    '''
    return {k: RerankDataset.from_path(v) for k, v in data_files.items()}


@dataclass
class RerankDataset:
    '''
    Embedding Reranking Dataset.
    '''
    pairs: List[List[str]]
    queries: Dict[str | int, str]
    corpus: Dict[str | int, str]
    relevant_docs: Dict[str | int, List[str | int]]
    mode: str = "text"
    
    @classmethod
    def from_df(cls, df: pd.DataFrame, input_col: str = 'query', target_col: str = 'document'):
        '''
        Load query-doc pairs from a pandas dataframe.
        '''
        pairs = df.values.tolist()
        
        query2idx = df.groupby(input_col).indices
        query2idx = {k: set(v.flat) for k, v in query2idx.items()}
        doc2idx = df.groupby(target_col).indices
        doc2idx = {k: set(v.flat) for k, v in doc2idx.items()}
        
        id2query = {i: q for i, q in enumerate(query2idx.keys())}
        id2doc = {i: d for i, d in enumerate(doc2idx.keys())}
        
        relevant_docs = {
            i: [j for j, d in id2doc.items() if (doc2idx[d] & query2idx[q])]
            for i, q in id2query.items()
        }
        return cls(pairs, id2query, id2doc, relevant_docs)

    @classmethod
    def from_path(cls, file_path: str | Path, input_col: str = 'query', target_col: str = 'document'):
        '''
        Load query-doc pairs from path to a .tsv file.
        '''
        return cls.from_df(pd.read_csv(Path(file_path), sep = '\t'), input_col, target_col)
