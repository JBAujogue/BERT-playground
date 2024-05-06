import os
from typing import Optional
from omegaconf import OmegaConf
from fire import Fire
import logging

from bertools.tasks.rerank import RerankDataset, RerankTrainer

logger = logging.getLogger(__name__)


def run_rerank(
    config_path: str,
    logging_dir: str,
    output_dir: Optional[str] = None,
    ):
    '''
    Args:
        config_path (str):
            path to a .yaml file providing arguments to the job.
            This config file must carry mandatory sections:
                - data_args
                - model_args
                - training_args
            Model args and training args are listed in the sentence-transformers 
            source code, see https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
            Other sections are discarded.
        logging_dir (str):
            path to the folder where finetuning metrics are logged.
        output_dir (Optional[str], default to None):
            path to the folder where finetuned model is persisted.
            If not specified, model is persisted into '<logging-dir>/model/'.
    '''
    # load job config
    config_path = os.path.abspath(config_path)
    logging_dir = os.path.abspath(logging_dir)
    output_dir = (
        os.path.abspath(output_dir) 
        if output_dir 
        else os.path.join(logging_dir, 'model')
    )
    
    job_config = OmegaConf.load(config_path)
    logger.warning(
        f'''
        #-------------------------------------#
        # Running Reranking training pipeline #
        #-------------------------------------#
            
        - Using job config at '{config_path}'.
        - Using logging dir '{logging_dir}'.
        - Using output dir '{output_dir}'.
        '''
    )
    
    # load train/valid/test datasets
    train_path = job_config.data_args.train_dataset_path
    valid_path = job_config.data_args.valid_dataset_path
    test_path = job_config.data_args.test_dataset_path
    
    train_dataset = RerankDataset.from_path(train_path)
    valid_dataset = (
        RerankDataset.from_path(valid_path) if valid_path else None
    )
    test_dataset = (
        RerankDataset.from_path(test_path) if test_path else None
    )
    
    # train, evaluate & save model
    trainer = RerankTrainer(
        model_args = OmegaConf.to_object(job_config.model_args),
        logging_dir = logging_dir,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        training_args = OmegaConf.to_object(job_config.training_args),
    )
    trainer.train()
    if test_dataset is not None:
        trainer.evaluate(test_dataset)
    trainer.save_model(output_dir)
    logger.info('Reranking training pipeline complete')


if __name__ == '__main__':
    Fire(run_rerank)