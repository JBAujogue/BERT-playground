import os
from typing import Optional
from omegaconf import OmegaConf
from fire import Fire
import logging
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer,
)
from bertools.tasks.mlm import CustomDataCollatorForLanguageModeling, form_constant_length_blocks

logger = logging.getLogger(__name__)


def run_mlm(
    config_path: str,
    logging_dir: str,
    output_dir: Optional[str] = None,
    save_model: bool = True,
    ):
    '''
    Args:
        config_path (str):
            path to a .yaml file providing arguments to the job.
            This config file must carry mandatory sections:
                - data_args
                - model_args
                - training_args
            
        logging_dir (str):
            path to the folder where finetuning metrics are logged.
        output_dir (Optional[str], default to None):
            path to the folder where finetuned model is persisted.
            If not specified, model is persisted into '<logging-dir>/model/'.
        save_model (Optional[bool], default to True):
            whether saving trained model or not.
    '''
    # load job config
    config_path = os.path.abspath(config_path)
    logging_dir = os.path.abspath(logging_dir)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    _job_config = OmegaConf.load(config_path)
    job_config = OmegaConf.to_object(_job_config)
    logger.info(
        f'''
        #----------------------------------------------------#
        # Running Masked Language Modeling training pipeline #
        #----------------------------------------------------#'''
    )
    logger.info(f'Using job config at {config_path}')
    
    # load train/valid/test datasets
    dataset = load_dataset(**job_config['dataset_args'])
    
    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(**job_config['tokenizer_args'])
    model = AutoModelForMaskedLM.from_pretrained(**job_config['model_args']).to(device).train()
    logger.info(f'Loaded model has {model.num_parameters()} parameters')

    # prepare dataset
    if not job_config['is_tokenized']:
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"], 
                return_special_tokens_mask = True,
                padding = (False if job_config['concat_inputs'] else True),
                truncation = (False if job_config['concat_inputs'] else True),
            ), 
            batched = True,
            remove_columns = ["text"],
        )
    if job_config['concat_inputs']:
        max_len = (tokenizer.model_max_length if tokenizer.model_max_length < 1e12 else 512)
        dataset = dataset.map(
            lambda examples: form_constant_length_blocks(examples, block_size = max_len), 
            batched = True,
        )
    
    # train, evaluate & save model
    training_args = TrainingArguments(output_dir = logging_dir, **job_config['training_args'])
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer = tokenizer, **job_config['collator_args']
    )
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset['train'],
        eval_dataset = (dataset['valid'] if 'valid' in dataset else None),
    )
    logger.info(f'Logging experiment artifacts at {logging_dir}')
    trainer.train()
    OmegaConf.save(_job_config, os.path.join(trainer.args.logging_dir, 'config.yaml'))
    if 'test' in dataset:
        trainer.evaluate(dataset['test'], metric_key_prefix = 'test')
    if save_model:
        output_dir = (
            os.path.abspath(output_dir) 
            if output_dir else
            os.path.join(trainer.args.logging_dir, 'model')
        )
        trainer.save_model(output_dir)
        logger.info(f'Model saved to {output_dir}')
    logger.info('Masked Language Modeling training pipeline complete')


if __name__ == '__main__':
    Fire(run_mlm)