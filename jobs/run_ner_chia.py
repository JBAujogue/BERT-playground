"""
Fine-tuning the library models for token classification.
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
"""

import logging
import os
import sys
import json
import argparse


# data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import (
    Dataset, 
    DatasetDict,
    ClassLabel, 
    Features, 
    Sequence, 
    Value,
    load_dataset,
)

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


# custom paths
path_to_repo = os.path.dirname(os.getcwd())
path_to_data = os.path.join(path_to_repo, 'datasets', 'chia', 'chia-ner')
path_to_logs = os.path.join(path_to_repo, 'logs')
path_to_save = os.path.join(path_to_repo, 'saves')
path_to_src  = os.path.join(path_to_repo, 'src')


# custom imports
sys.path.insert(0, path_to_src)

from nlptools.ner.preprocessing import tokenize_and_align_categories, create_labels
from nlptools.ner.metrics import compute_metrics, compute_metrics_finegrained



logger = logging.getLogger(__name__)






# ------------------------ functions -----------------------------
def load_chia_dataset(path_to_data):
    def get_item_list(df, grp_col, item_col):
        return df.groupby(grp_col).apply(lambda g: g[item_col].tolist()).tolist()

    def convert_dataframe_to_dataset(df):
        data = {
            'ids': df.Sequence_id.unique().tolist(),
            'mentions': get_item_list(df, 'Sequence_id', 'Mention'),
            'categories': get_item_list(df, 'Sequence_id', 'Category'),
        }
        return data
    
    df_bio = pd.read_csv(os.path.join(path_to_data, 'chia_bio.tsv'), sep = "\t")
    
    class_labels = sorted(list(set(df_bio.Category.unique())))
    class_labels = ClassLabel(names = class_labels)
    
    # dataset separation: 800 trials for training, 100 trials for validation and 100 trials for testing
    ids_bio = sorted(list(set(df_bio.Id.apply(lambda i: i.split('_')[0]))))
    ids_trn, ids_dev = train_test_split(ids_bio, train_size = 0.8, random_state = 13, shuffle = True)
    ids_dev, ids_tst = train_test_split(ids_dev, train_size = 0.5, random_state = 13, shuffle = True)
    
    df_trn = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_trn)]
    df_dev = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_dev)]
    df_tst = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_tst)]
    
    dict_bio = convert_dataframe_to_dataset(df_bio)
    dict_trn = convert_dataframe_to_dataset(df_trn)
    dict_dev = convert_dataframe_to_dataset(df_dev)
    dict_tst = convert_dataframe_to_dataset(df_tst)

    features = Features({
        'ids': Value(dtype = 'string'), 
        'mentions': Sequence(Value(dtype = 'string')), 
        'categories': Sequence(Value(dtype = 'string')),
    })

    raw_datasets = DatasetDict({
        'trn': Dataset.from_dict(dict_trn, features = features),
        'dev': Dataset.from_dict(dict_dev, features = features),
        'tst': Dataset.from_dict(dict_tst, features = features),
        'all': Dataset.from_dict(dict_bio, features = features),
    })
    return (raw_datasets, class_labels)



def tokenize_chia_dataset(raw_datasets, tokenizer, class_labels, **kwargs):
    B_I_mapping = {l: 'I'+l[1:] for l in class_labels.names if l.startswith('B-')}

    tokenized_datasets = raw_datasets.map(
        function = lambda examples: tokenize_and_align_categories(tokenizer, examples, B_I_mapping, **kwargs), 
        batched  = True,
    )
    tokenized_datasets = tokenized_datasets.map(
        function = lambda examples: create_labels(examples, class_labels), 
        batched  = True,
    )
    return tokenized_datasets





# -------------------------- main --------------------------------
# either loads a config file from ./logs/task/final_model_name/run_name/run_args.json
# and store log results in same run_name folder,
# or loads a config file from ./saves/task/final_model_name/run_name/run_args.json
# and store trained model in same run_name folder
def main():
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
    raw_datasets, class_labels = load_chia_dataset(path_to_data)

    # load model
    base_model_path = os.path.join(path_to_save, run_args['base_model_task'].upper(), run_args['base_model_name'].lower(), 'model')
    label2id = class_labels._str2int
    id2label = {i: l for l, i in label2id.items()}
    try:
        model = AutoModelForTokenClassification.from_pretrained(base_model_path, label2id = label2id, id2label = id2label)
        logger.warning('Model loaded from local checkpoint.')
    except:
        model = AutoModelForMaskedLM.from_pretrained(run_args['hub_model_name'])
        model.save_pretrained(base_model_path)
        model = AutoModelForTokenClassification.from_pretrained(base_model_path, label2id = label2id, id2label = id2label)
        logger.warning('Model downloaded from Huggingface model hub.')
    
    # load tokenizer
    tokenizer_path = os.path.join(path_to_save, run_args['base_model_task'].upper(), run_args['base_model_name'].lower(), 'tokenizer')
    tokenizer_kwgs = ({'add_prefix_space': True} if (('model_type' in run_args) and (run_args['model_type'] in {"bloom", "gpt2", "roberta"})) else {})
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwgs)
        logger.warning('Tokenizer loaded from local checkpoint.')
    except:
        tokenizer = AutoTokenizer.from_pretrained(run_args['hub_model_name'], use_fast = True, **tokenizer_kwgs)
        tokenizer.save_pretrained(tokenizer_path)
        logger.warning('Tokenizer downloaded from Huggingface model hub.')

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )
    
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




if __name__ == "__main__":
    main()


