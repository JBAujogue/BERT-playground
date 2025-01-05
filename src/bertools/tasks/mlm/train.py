import os
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bertools.tasks.mlm.collators import DataCollatorForMLM
from bertools.tasks.mlm.transforms import form_constant_length_blocks


def train(config_path: str, output_dir: str, save_model: bool = True):
    """
    End-to-end training of a Masked Language Model.
    """
    logger.info(
        """
        ========================================================
        == Running Masked Language Modeling training pipeline ==
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

    # load train/valid/test datasets
    dataset = load_dataset(**train_config["dataset_args"])

    # load tokenizer, model & collator
    tokenizer = AutoTokenizer.from_pretrained(**train_config["tokenizer_args"])
    model = AutoModelForMaskedLM.from_pretrained(**train_config["model_args"]).to(device).train()
    collator = DataCollatorForMLM(tokenizer, **train_config["collator_args"])
    logger.info(f"Loaded model has {model.num_parameters()} parameters")

    # prepare dataset
    if not train_config.get("is_tokenized", False):
        return_special_token_mask = not train_config["training_args"].get("remove_unused_columns", False)
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                return_special_tokens_mask=return_special_token_mask,
                padding=False,
                truncation=not train_config["concat_inputs"],
            ),
            batched=True,
            keep_in_memory=True,
            remove_columns=["text"],
        )
    if train_config.get("concat_inputs", False):
        max_len = tokenizer.model_max_length if tokenizer.model_max_length < 1e12 else 512
        dataset = dataset.map(
            lambda examples: form_constant_length_blocks(examples, block_size=max_len),
            batched=True,
        )

    # train model
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        **train_config["training_args"],
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval", None),
    )
    trainer.train()
    logger.success("Training phase complete")

    # evaluate
    if "test" in dataset:
        test_result = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        logger.info(
            "Test results:"
            + "".join((f"\n\t{k} " + "-" * (30 - len(k)) + " {:2f}".format(100 * v)) for k, v in test_result.items())
        )
        logger.success("Testing phase complete")

    # save artifacts
    with open(logging_dir / "train_config.yaml", "wt") as f:
        yaml.safe_dump(train_config, f, encoding="utf-8")

    os.makedirs(logging_dir / "test")
    with open(logging_dir / "test" / "scores.yaml", "wt") as f:
        yaml.safe_dump(test_result, f, encoding="utf-8")

    # save model
    if save_model:
        tokenizer.save_pretrained(output_dir / "tokenizer")
        model.save_pretrained(output_dir / "model")

    logger.success("Job complete")
    return
