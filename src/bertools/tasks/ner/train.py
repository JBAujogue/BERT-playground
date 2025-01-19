import os
from pathlib import Path
import yaml  # type: ignore[import-untyped]
from loguru import logger

from datasets import Dataset, DatasetDict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, 
    Trainer,
    set_seed,
)

from bertools.tasks.ner.metrics import EvaluationMetric
from bertools.tasks.ner.transforms import preprocess_records, concat_lists, tokenize_dataset
from bertools.tasks.ner.load import load_annotations
from bertools.tasks.ner.typing import Record


def train(config_path: str | Path, output_dir: str | Path, save_model: bool = True) -> None:
    '''
    End-to-end training of a Named Entity Recognition Model.
    See https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
    '''
    logger.info(
        """
        ========================================================
        == Running Named Entity Recognition training pipeline ==
        ========================================================"""
    )
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    logging_dir = output_dir / "logs"
    os.makedirs(logging_dir, exist_ok=False)
    logger.info(f"Saving experiment artifacts at {output_dir}")

    # load job config
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    # ensure reproducibility
    set_seed(train_config.get("global_seed", 42))

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device {device}")

    # load & prepare train/eval/test datasets
    dataset, bio_labels = load_and_preprocess_dataset(**train_config["dataset_args"])
    id2label = dict(enumerate(bio_labels))
    label2id = {label: i for i, label in id2label.items()}
    logger.info("Loaded datasets:" + "".join(f"\n\t- '{k}' with {len(v)} lines" for k, v in dataset.items()))
    logger.info("Loaded labels:" + "".join(f"\n\t- {k}: {v}" for k, v in id2label.items()))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(**train_config["tokenizer_args"])
    
    # load model
    model_args = {"id2label": id2label, "label2id": label2id} | train_config["model_args"]
    model = AutoModelForTokenClassification.from_pretrained(**model_args).to(device).train()
    logger.info(f"Loaded model has {model.num_parameters()} parameters")

    # load collator
    collator = DataCollatorForTokenClassification(tokenizer, **train_config["collator_args"])

    # tokenize and build token-level target labels
    dataset = tokenize_dataset(
        dataset = dataset,
        tokenizer = tokenizer,
        label2id = label2id, 
        **train_config["collator_args"],
    )
    # train model
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        **train_config["training_args"],
    )
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        data_collator = collator,
        train_dataset = dataset['train'],
        eval_dataset = dataset.get('eval', None),
        compute_metrics = EvaluationMetric(id2label),
    )
    trainer.train()
    logger.success("Training phase complete")

    # evaluate model
    if "test" in dataset:
        test_result = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        logger.info(
            "Test results:"
            + "".join((f"\n\t{k} " + "-" * (30 - len(k)) + " {:2f}".format(100 * v)) for k, v in test_result.items())
        )
        # save eval
        os.makedirs(logging_dir / "test")
        with open(logging_dir / "test" / "scores.yaml", "wt") as f:
            yaml.safe_dump(test_result, f, encoding="utf-8")
        logger.success("Testing phase complete")

    # save train config
    with open(logging_dir / "train_config.yaml", "wt") as f:
        yaml.safe_dump(train_config, f, encoding="utf-8")

    # save model
    if save_model:
        tokenizer.save_pretrained(output_dir / "tokenizer")
        model.save_pretrained(output_dir / "model")

    logger.success("Job complete")
    return


def load_and_preprocess_dataset(data_files: dict[str, list[str]]) -> tuple[dict[str, list[Record]], list[str]]:
    """
    Load and prepare train/eval/test splits as a dict of Datasets.
    """
    dataset_as_records = {k: load_and_preprocess_split(v) for k, v in data_files.items()}
    dataset = DatasetDict({k: Dataset.from_list(v) for k, v in dataset_as_records.items()})

    # load list of labels and turn them to BIO format
    raw_labels = sorted({sp["label"] for rs in dataset_as_records.values() for r in rs for sp in r["spans"]})
    bio_labels = ["O"] + [f"B-{l}" for l in raw_labels] + [f"I-{l}" for l in raw_labels]
    
    return dataset, bio_labels


def load_and_preprocess_split(file_list: list[str]) -> list[Record]:
    """
    Prepare a single train/eval/test split as a list of preprocessed records.
    """
    texts = load_annotations(file_list)
    return concat_lists([preprocess_records(c) for c in texts.values()])
