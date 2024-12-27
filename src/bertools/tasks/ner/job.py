import os
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, 
    Trainer,
    set_seed,
)
import evaluate

from bertools.tasks.mlm.collators import DataCollatorForMLM
from bertools.tasks.mlm.transforms import form_constant_length_blocks


def run_ner(config_path: str, output_dir: str, save_model: bool = True):
    '''
    Run named entity recognition training pipeline.
    '''
    logger.info(
        f'''
        #----------------------------------------------------#
        # Running Named Entity Recognition training pipeline #
        #----------------------------------------------------#'''
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
    
    # load train/valid/test datasets & labels
    dataset = load_dataset(**job_config['dataset_args'])
    # TODO:
    # - groupby dataset by Id
    # - compute class labels
    job_config['model_args'] |= dict(
        label2id = class_labels._str2int,
        id2label = {i: l for l, i in class_labels._str2int.items()},
    )
    
    # load tokenizer, model & collator
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(**job_config['tokenizer_args'])
    model = AutoModelForTokenClassification.from_pretrained(**job_config['model_args']).to(device).train()
    collator = DataCollatorForTokenClassification(tokenizer, **job_config['collator_args'])
    logger.info(f'Loaded model has {model.num_parameters()} parameters')

    # prepare dataset
    # TODO
    kwargs = {
        'truncation': True, 
        'max_length': run_args['model_max_length'], 
        'is_split_into_words': True,
    }
    dataset = tokenize_chia_dataset(raw_datasets, tokenizer, class_labels, **kwargs)
    
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

    # train model
    metric = evaluate.load("seqeval")
    training_args = TrainingArguments(
        output_dir = output_dir, logging_dir = output_dir / 'logs', **job_config['training_args'],
    )
    # TODO
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        data_collator = collator,
        train_dataset = dataset.get('train'),
        eval_dataset = dataset.get('eval', None),
        compute_metrics = lambda p: compute_metrics_finegrained(p, metric, class_labels.names),
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
    logger.info('Named Entity Recognition training pipeline complete')
