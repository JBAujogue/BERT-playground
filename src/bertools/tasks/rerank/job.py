import os
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger
from transformers import set_seed
import torch

from bertools.tasks.rerank.dataset import load_rerank_dataset
from bertools.tasks.rerank.trainer import RerankTrainer


def run_rerank(config_path: str, output_dir: str, save_model: bool = True):
    '''
    Run reranking training pipeline.
    '''
    logger.info(
        f'''
        #-------------------------------------#
        # Running Reranking training pipeline #
        #-------------------------------------#'''
    )
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok = False)
    logger.info(f'Saving experiment artifacts at {output_dir}')

    # load & save job config
    _job_config = OmegaConf.load(config_path)
    job_config = OmegaConf.to_object(_job_config)
    OmegaConf.save(_job_config, output_dir / 'job_config.yaml')
    logger.info(f'Using job config at {config_path}')
    
    # ensure reproducibility
    set_seed(job_config['global_seed'])
    
    # load train/valid/test datasets
    dataset = load_rerank_dataset(**job_config['dataset_args'])
    
    # train model
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = RerankTrainer(
        model_args = job_config.get('model_args') | dict(device = device),
        output_dir = output_dir,
        train_dataset = dataset.get('train'),
        eval_dataset = dataset.get('eval', None),
        training_args = job_config.get('training_args'),
    )
    trainer.train()
    
    # evaluate model
    if 'test' in dataset:
        test_result = trainer.evaluate(dataset['test'], metric_key_prefix = 'test')
        logger.info(f"Test result: {test_result}")
    
    # save model
    if save_model:
        trainer.save_model(output_dir / 'model')
    logger.info('Reranking training pipeline complete')
