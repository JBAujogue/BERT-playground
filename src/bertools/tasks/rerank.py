"""
Sentence Transformer Trainer.
Adapted from
    https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py
    https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/integrations/integration_utils.py#L579
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

import pandas as pd

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator


logger = logging.getLogger(__name__)


@dataclass
class RerankDataset:
    '''
    Embedding Reranking Dataset.

    Attributes:
        pairs List[List[str]]:
            list of aligned query/doc pairs.
        queries (Dict[str, str]): 
            Dict id -> query.
        corpus (Dict[str, str]): 
            Dict id -> doc.
        relevant_docs (Dict[str, List[str]]): 
            Dict query id -> list of doc ids.
    '''
    pairs: List[List[str]]
    queries: Dict[str | int, str]
    corpus: Dict[str | int, str]
    relevant_docs: Dict[str | int, List[str | int]]
    mode: str = "text"
    
    @classmethod
    def from_df(
        cls, df: pd.DataFrame, input_col: str = 'query', target_col: str = 'document',
        ):
        '''
        Args:
            df (pd.DataFrame):
                Pandas dataframe of query-doc pairs.
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
    def from_path(
        cls, file_path: str, input_col: str = 'query', target_col: str = 'document',
        ):
        '''
        Args:
            file_path (str):
                path to a .tsv file
        '''
        file_path = os.path.abspath(file_path)
        return cls.from_df(pd.read_csv(file_path, sep = '\t'), input_col, target_col)


class RerankTrainer:
    '''
    Simple Trainer on a rerank task for sentence-transformers models.
    '''
    def __init__(
        self,
        model_args: Dict[str, Any],
        logging_dir: str,
        train_dataset: Any,
        valid_dataset: Optional[Any] = None,
        training_args: Dict[str, Any] = dict(),
        ):
        '''
        Init params.
        training_args are listed in 
            https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
        '''
        self.logging_dir = os.path.abspath(logging_dir)
        self.training_args = training_args
        self.exportable_training_args = {
            k: v for k, v in self.training_args.items() 
            if v is None or isinstance(v, (int, float, list, tuple, dict, str, bool))
        }
        if 'batch_size' in self.training_args:
            batch_size = self.training_args['batch_size']
            self.training_args.pop('batch_size')
        else:
            batch_size = 8

        self.model = SentenceTransformer(**model_args)
        self.evaluator = (
            InformationRetrievalEvaluator(
                valid_dataset.queries, 
                valid_dataset.corpus, 
                valid_dataset.relevant_docs,
            )
            if valid_dataset is not None
            else None
        )
        self._SummaryWriter = SummaryWriter
        
        dataloader = DataLoader(
            dataset = [InputExample(texts = pair) for pair in train_dataset.pairs], 
            batch_size = batch_size,
        )
        loss = MultipleNegativesRankingLoss(self.model)
        
        self.training_args |= dict(
            train_objectives = [(dataloader, loss)],
            evaluator = self.evaluator,
        )

    def train(self):
        '''
        Train model according to training args specified in intanciation.
        Log results into a tensorboard summary.
        '''
        logger.info(
            f'''
            Using the following finetuning config:
            {self.exportable_training_args}
            '''
        )
        self.save_config()
        self.model.fit(**self.training_args)
        if self.evaluator is not None:
            self.send_logs_to_tensorboard(metric_key_prefix = 'eval')
        return

    def evaluate(self, dataset: Any, metric_key_prefix: str = 'test'):
        '''
        Evaluate model on an input dataset.
        Log results into a tensorboard summary.
        '''
        evaluator = InformationRetrievalEvaluator(
            dataset.queries, dataset.corpus, dataset.relevant_docs,
        )
        out_dir = os.path.join(self.logging_dir, metric_key_prefix)
        os.makedirs(out_dir, exist_ok = True)
        score = evaluator(self.model, output_path = out_dir)
        self.send_logs_to_tensorboard(metric_key_prefix)
        return score

    def save_config(self):
        '''
        Save training config into a yaml file.
        '''
        os.makedirs(self.logging_dir, exist_ok = True)
        out_path = os.path.join(self.logging_dir, 'config.yaml')
        OmegaConf.save(self.exportable_training_args, out_path)
        return

    def save_model(self, out_dir: str):
        '''
        Save model into the specified directory.
        '''
        self.model.save(os.path.abspath(out_dir))
        return

    def send_logs_to_tensorboard(self, metric_key_prefix: str):
        '''
        Maps the evaluation report .csv file generated by
        sentence-transformers into a tensorboard summary.
        '''
        tb_writer = self._SummaryWriter(log_dir = self.logging_dir)
        
        # set path to logs 
        out_path = os.path.join(
            self.logging_dir, 
            metric_key_prefix, 
            'Information-Retrieval_evaluation_results.csv',
        )
        if os.path.isfile(out_path):
            # parse sentence-transformers scores
            tb_scores = pd.read_csv(out_path).to_dict('records')
    
            # write scores to tensorboard log file
            max_epoch = max(scores['epoch'] for scores in tb_scores)
            max_steps = max(scores['steps'] for scores in tb_scores) + 1
            for scores in tb_scores:
                epoch, step = scores['epoch'], scores['steps']
                if max_epoch == -1:
                    epoch = 0
                if step == -1:
                    step = max_steps
    
                for k, v in scores.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f'{metric_key_prefix}/{k}', v, epoch * max_steps + step)
                    else:
                        logger.warning(
                            "Trainer is attempting to log a value of "
                            f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                            "This invocation of Tensorboard's writer.add_scalar() "
                            "is incorrect so we dropped this attribute."
                        )
                tb_writer.flush()
        tb_writer.close()
        return
