"""
Fine-tuning the library models for token classification.
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
"""

import logging
import os
import sys


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

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# custom paths
path_to_repo = os.path.dirname(os.getcwd())
path_to_data = os.path.join(path_to_repo, 'datasets', 'chia', 'chia-ner')
path_to_logs = os.path.join(path_to_repo, 'logs', 'NER')
path_to_save = os.path.join(path_to_repo, 'saves')
path_to_src  = os.path.join(path_to_repo, 'src')


# custom imports
sys.path.insert(0, path_to_src)

from nlptools.ner.preprocessing import tokenize_and_align_categories, create_labels
from nlptools.ner.metrics import compute_metrics, compute_metrics_finegrained
from nlptools.utils import ModelArguments




check_min_version("4.22.2")
require_version("datasets>=2.5.2", "To fix: pip install -r requirements.txt")

device = ('cuda' if torch.cuda.is_available() else 'cpu')
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

    datasets = DatasetDict({
        'trn': Dataset.from_dict(dict_trn, features = features),
        'dev': Dataset.from_dict(dict_dev, features = features),
        'tst': Dataset.from_dict(dict_tst, features = features),
        'all': Dataset.from_dict(dict_bio, features = features),
    })
    return (datasets, class_labels)



def tokenize_chia_dataset(datasets, tokenizer, class_labels):
    B_I_mapping = {l: 'I'+l[1:] for l in class_labels.names if l.startswith('B-')}
    
    tokenized_datasets = datasets.map(
        function = lambda examples: tokenize_and_align_categories(tokenizer, examples, B_I_mapping), 
        batched  = True,
    )
    tokenized_datasets = tokenized_datasets.map(
        function = lambda examples: create_labels(examples, class_labels), 
        batched  = True,
    )
    return tokenized_datasets





# -------------------------- main --------------------------------
def main():
    # parse args
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file = os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()


    # setup logging
    log_level = 'info'
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
    logger.warning(f"device: {training_args.device}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")


    # set seed for full reproducibility
    set_seed(training_args.seed)
    

    # load raw dataset
    raw_datasets, class_labels = load_chia_dataset(path_to_data)


    # load model
    base_model_path = os.path.join(path_to_save, model_args.base_model_name, 'model')
    label2id = class_labels._str2int
    id2label = {i: l for l, i in label2id.items()}
    try:
        model = AutoModelForTokenClassification.from_pretrained(base_model_path, label2id = label2id, id2label = id2label)
        logger.warning('Model loaded from local checkpoint.')
    except:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.hub_model_name,
            cache_dir = model_args.cache_dir,
            revision = model_args.hub_model_revision,
            use_auth_token = (True if model_args.use_auth_token else None),
        )
        model.save_pretrained(base_model_path)
        model = AutoModelForTokenClassification.from_pretrained(base_model_path, label2id = label2id, id2label = id2label)
        logger.warning('Model downloaded from Huggingface model hub.')
    
    
    # load tokenizer
    tokenizer_path = os.path.join(path_to_save, model_args.base_model_name, 'tokenizer')
    tokenizer_kwgs = ({'add_prefix_space': True} if model.model_type in {"bloom", "gpt2", "roberta"} else {})
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwgs)
        logger.warning('Tokenizer loaded from local checkpoint.')
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.hub_model_name,
            cache_dir = model_args.cache_dir,
            use_fast = True,
            revision = model_args.hub_model_revision,
            use_auth_token = (True if model_args.use_auth_token else None),
            **tokenizer_kwgs,
        )
        tokenizer.save_pretrained(tokenizer_path)
        logger.warning('Tokenizer downloaded from Huggingface model hub.')

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )
    
    
    # preprocess dataset
    datasets = tokenize_chia_dataset(raw_datasets, tokenizer, class_labels)
    
    
    # run training
    model = model.to(device)
    model = model.train()

    metric = evaluate.load("seqeval")
    
    # option 1: evaluate finetuning quality through train/dev/test split and periodic evaluations
    if training_args.benchmark_mode:      
        args = TrainingArguments(
            # training args
            learning_rate = training_args.learning_rate,
            weight_decay = training_args.weight_decay,
            num_train_epochs = training_args.num_train_epochs,
            max_steps = training_args.max_steps,
            per_device_train_batch_size = training_args.batch_size,
            per_device_eval_batch_size = training_args.batch_size,
            fp16 = training_args.fp16,

            # logging args
            output_dir = os.path.join(path_to_save, model_args.final_model_task, '_checkpoints'),
            logging_dir = os.path.join(path_to_logs, model_args.final_model_name, training_args.run_name),
            evaluation_strategy = training_args.evaluation_strategy,
            save_strategy = training_args.save_strategy,
            logging_steps = training_args.logging_steps,
            report_to = ['tensorboard'],
            log_level = log_level,
        )
        train_dataset = datasets['trn']
        eval_dataset  = datasets['dev']
    # option 2: train on all data, no evaluation
    else:
        args = TrainingArguments(
            # training args
            learning_rate = training_args.learning_rate,
            weight_decay = training_args.weight_decay,
            num_train_epochs = training_args.num_train_epochs,
            max_steps = training_args.max_steps,
            per_device_train_batch_size = training_args.batch_size,
            per_device_eval_batch_size = training_args.batch_size,

            # logging args
            output_dir = os.path.join(path_to_save, model_args.final_model_task, '_checkpoints'),
            evaluation_strategy = 'no',
            save_strategy = 'no',
        ) 
        train_dataset = datasets['all']
        eval_dataset  = None
    
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of = (8 if training_args.fp16 else None)),
        compute_metrics = lambda p: compute_metrics_finegrained(p, metric, class_labels.names),
    )
    trainer.train()
    
    
    # export final logs or model
    if training_args.benchmark_mode:
        test_results = trainer.evaluate(eval_dataset = datasets['tst'], metric_key_prefix = 'test')
        for k, v in test_results.items():
            logger.info(str(k) + ' ' + '-'*(30 - len(k)) + ' {:2f}'.format(100*v))
    else:
        tokenizer.save_pretrained(os.path.join(path_to_save, model_args.final_model_name, training_args.run_name, 'tokenizer'))
        model.save_pretrained(os.path.join(path_to_save, model_args.final_model_name, training_args.run_name, 'model'))





if __name__ == "__main__":
    main()


