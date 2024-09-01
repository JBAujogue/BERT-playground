import logging
import os
import sys
import json
import argparse



# model & training
import torch
import transformers
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate




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
    dataset = load_dataset(**job_config['dataset_args'])
    
    # load tokenizer, model & collator
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(**job_config['tokenizer_args'])
    model = AutoModelForMaskedLM.from_pretrained(**job_config['model_args']).to(device).train()
    collator = DataCollatorForMLM(tokenizer = tokenizer, **job_config['collator_args'])
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
        for k, v in test_result.items():
            logger.info(str(k) + ' ' + '-'*(30 - len(k)) + ' {:2f}'.format(100*v))
    
    # save model
    if save_model:
        tokenizer.save_pretrained(output_dir / 'tokenizer')
        model.save_pretrained(output_dir / 'model')
    logger.info('Masked Language Modeling training pipeline complete')





def run_ner():
    '''
    Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
    '''
    # parse run folder and args
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type = str, help = "path to a configuration json file")
    
    arg_path = parser.parse_args().path
    if os.path.exists(arg_path):
        with open(arg_path) as f:
            run_args = json.load(f)
    else:
        raise ValueError("The specified path must be a json file")
    
    default_args = {
        "fp16": False,
        "bf16": False,
        "tf32": False,
        "torch_compile": False,
        "optim": "adamw_hf",
    }
    run_args = dict(list(default_args.items()) + list(run_args.items())) 

    out_path = os.path.join(
        (path_to_logs if run_args['benchmark_mode'] else path_to_save), 
        run_args['final_model_task'].upper(), 
        run_args['final_model_name'].lower(), 
        run_args['run_name'],
    )
        
    # setup logging
    log_level = datasets.logging.INFO
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)],
    )
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(log_level)
    logger.warning(f"device: {run_args['device']}, 16-bits training: {run_args['fp16']}")
    logger.info(f"Training/evaluation parameters {run_args}")

    # set seed for full reproducibility
    set_seed(run_args['seed'])
    
    # load raw dataset
    raw_datasets, class_labels = load_chia_dataset(run_args['dataset_path'])

    # load model
    base_model_path = os.path.join(path_to_save, run_args['base_model_task'].upper(), run_args['base_model_name'].lower(), 'model')
    label2id = class_labels._str2int
    id2label = {i: l for l, i in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(base_model_path, label2id = label2id, id2label = id2label)

    tokenizer_path = os.path.join(path_to_save, run_args['base_model_task'].upper(), run_args['base_model_name'].lower(), 'tokenizer')
    tokenizer_kwgs = ({'add_prefix_space': True} if (('model_type' in run_args) and (run_args['model_type'] in {"bloom", "gpt2", "roberta"})) else {})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwgs)


    # preprocess dataset
    kwargs = {
        'truncation': True, 
        'max_length': run_args['model_max_length'], 
        'is_split_into_words': True,
    }
    tokenized_datasets = tokenize_chia_dataset(raw_datasets, tokenizer, class_labels, **kwargs)
    
    # run training
    model = model.to(run_args['device'])
    model = model.train()

    metric = evaluate.load("seqeval")
    
    # option 1: evaluate finetuning quality through train/dev/test split and periodic evaluations
    if run_args['benchmark_mode']:      
        args = TrainingArguments(
            # training args
            learning_rate = run_args['learning_rate'],
            weight_decay = run_args['weight_decay'],
            num_train_epochs = run_args['num_train_epochs'],
            max_steps = run_args['max_steps'],
            per_device_train_batch_size = run_args['batch_size'],
            per_device_eval_batch_size = run_args['batch_size'],
            bf16 = run_args['bf16'],
            fp16 = run_args['fp16'],
            torch_compile = run_args['torch_compile'],
            optim = run_args['optim'],
            # logging args
            output_dir = os.path.join(path_to_save, run_args['final_model_task'], '_checkpoints'),
            logging_dir = out_path,
            evaluation_strategy = run_args['evaluation_strategy'],
            save_strategy = run_args['save_strategy'],
            logging_steps = run_args['logging_steps'],
            report_to = ['tensorboard'],
        )
        train_dataset = tokenized_datasets['trn']
        eval_dataset  = tokenized_datasets['dev']
    
    # option 2: train on all data, no evaluation
    else:
        args = TrainingArguments(
            # training args
            learning_rate = run_args['learning_rate'],
            weight_decay = run_args['weight_decay'],
            num_train_epochs = run_args['num_train_epochs'],
            max_steps = run_args['max_steps'],
            per_device_train_batch_size = run_args['batch_size'],
            per_device_eval_batch_size = run_args['batch_size'],
            bf16 = run_args['bf16'],
            fp16 = run_args['fp16'],
            torch_compile = run_args['torch_compile'],
            optim = run_args['optim'],
            # logging args
            output_dir = os.path.join(path_to_save, run_args['final_model_task'], '_checkpoints'),
            evaluation_strategy = 'no',
            save_strategy = 'no',
        ) 
        train_dataset = tokenized_datasets['all']
        eval_dataset  = None
    
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of = (8 if run_args['fp16'] else None)),
        compute_metrics = lambda p: compute_metrics_finegrained(p, metric, class_labels.names),
    )
    train_results = trainer.train()
    trainer.save_metrics('train', train_results.metrics)
    
    # export final logs or model
    if run_args['benchmark_mode']:
        test_results = trainer.evaluate(eval_dataset = tokenized_datasets['tst'], metric_key_prefix = 'test')
        trainer.save_metrics('test', test_results)
        for k, v in test_results.items():
            logger.info(str(k) + ' ' + '-'*(30 - len(k)) + ' {:2f}'.format(100*v))
    else:
        tokenizer.save_pretrained(os.path.join(out_path, 'tokenizer'))
        model.save_pretrained(os.path.join(out_path, 'model'))

    if arg_path != os.path.join(out_path, "run_args.json"):
        with open(os.path.join(out_path, "run_args.json"), "w") as f:
            json.dump(run_args, f)
