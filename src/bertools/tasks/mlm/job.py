import os
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer,
    set_seed,
)
from bertools.tasks.mlm.collators import DataCollatorForMLM
from bertools.tasks.mlm.transforms import form_constant_length_blocks


def run_mlm(config_path: str, output_dir: str, save_model: bool = True):
    '''
    Args:
        config_path (str):
            path to a .yaml file providing arguments to the job.
            This config file must carry mandatory sections:
                - dataset_args
                - tokenizer_args
                - model_args
                - collator_args
                - training_args
        output_dir (str):
            path to the folder where finetuning artifact are serialized.
        save_model (Optional[bool], default to True):
            whether saving trained model or not.
    '''
    logger.info(
        f'''
        #----------------------------------------------------#
        # Running Masked Language Modeling training pipeline #
        #----------------------------------------------------#'''
    )
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
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
    dataset = load_dataset(**job_config['dataset_args'])
    
    # load tokenizer, model & collator
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(**job_config['tokenizer_args'])
    model = AutoModelForMaskedLM.from_pretrained(**job_config['model_args']).to(device).train()
    collator = DataCollatorForMLM(tokenizer, **job_config['collator_args'])
    logger.info(f'Loaded model has {model.num_parameters()} parameters')

    # prepare dataset
    if not job_config.get('is_tokenized', False):
        return_special_token_mask = (
            not job_config['training_args'].get('remove_unused_columns', False)
        )
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"], 
                return_special_tokens_mask = return_special_token_mask,
                padding = False,
                truncation = not job_config['concat_inputs'],
            ), 
            batched = True,
            keep_in_memory = True,
            remove_columns = ["text"],
        )
    if job_config.get('concat_inputs', False):
        max_len = (tokenizer.model_max_length if tokenizer.model_max_length < 1e12 else 512)
        dataset = dataset.map(
            lambda examples: form_constant_length_blocks(examples, block_size = max_len), 
            batched = True,
        )

    # train model
    training_args = TrainingArguments(
        output_dir = output_dir, logging_dir = output_dir / 'logs', **job_config['training_args'],
    )
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        data_collator = collator,
        train_dataset = dataset.get('train'),
        eval_dataset = dataset.get('eval', None),
    )
    trainer.train()

    # evaluate model
    if 'test' in dataset:
        test_result = trainer.evaluate(eval_dataset = dataset['test'], metric_key_prefix = 'test')
        OmegaConf.save(test_result, output_dir / 'test_result.yaml')
        logger.info('Test results:' + ''.join(
            (f'\n\t{k} ' + '-'*(30 - len(k)) + ' {:2f}'.format(100*v)) for k, v in test_result.items()
        ))
    
    # save model
    if save_model:
        tokenizer.save_pretrained(output_dir / 'tokenizer')
        model.save_pretrained(output_dir / 'model')
    logger.info('Masked Language Modeling training pipeline complete')
