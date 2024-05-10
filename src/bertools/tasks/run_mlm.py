import os
from typing import Optional
from omegaconf import OmegaConf
from fire import Fire
import logging
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer,
)
from bertools.tasks.mlm import CustomDataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


def run_mlm(
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
        if output_dir else 
        os.path.join(logging_dir, 'model')
    )
    
    job_config = OmegaConf.load(config_path)
    logger.info(
        f'''
        #----------------------------------------------------#
        # Running Masked Language Modeling training pipeline #
        #----------------------------------------------------#
            
        - Using job config at '{config_path}'.
        - Using logging dir '{logging_dir}'.
        - Using output dir '{output_dir}'.
        '''
    )
    
    # load train/valid/test datasets
    train_path = job_config.data_args.train_dataset_path
    valid_path = job_config.data_args.valid_dataset_path
    test_path = job_config.data_args.test_dataset_path
    
    train_dataset = load_from_disk(train_path)
    valid_dataset = load_from_disk(valid_path) if valid_path else None
    test_dataset = load_from_disk(test_path) if test_path else None
    
    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(job_config.model_args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(job_config.model_args.model_name_or_path).train()
    logger.info(f'Loaded model has {model.num_parameters()} parameters')
    
    # train, evaluate & save model
    training_args = TrainingArguments(OmegaConf.to_object(job_config.training_args))
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        task_proportions = tuple(job_config.collator_args.task_proportions),
    )
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
    )
    trainer.train()
    if test_dataset is not None:
        trainer.evaluate(test_dataset, metric_key_prefix = 'test')
    trainer.save_model(output_dir)
    logger.info('Masked Language Modeling training pipeline complete')


if __name__ == '__main__':
    Fire(run_mlm)