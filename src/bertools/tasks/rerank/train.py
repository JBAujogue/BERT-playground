import os
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from loguru import logger
from transformers import set_seed

from bertools.tasks.rerank.load import load_rerank_dataset
from bertools.tasks.rerank.trainer import RerankTrainer


def train(config_path: str | Path, output_dir: str | Path, save_model: bool = True) -> None:
    """
    End-to-end training of a Reranking model.
    """
    logger.info(
        """
        =========================================
        == Running Reranking training pipeline ==
        ========================================="""
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
    dataset = load_rerank_dataset(**train_config["dataset_args"])

    # train model
    trainer = RerankTrainer(
        model_args=train_config.get("model_args") | {"device": device},
        output_dir=output_dir,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval", None),
        training_args=train_config.get("training_args"),
    )
    trainer.train()
    logger.success("Training phase complete")

    # evaluate model
    if "test" in dataset:
        test_result = trainer.evaluate(dataset["test"], metric_key_prefix="test")
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
        trainer.save_model(output_dir / "model")

    logger.success("Job complete")
    return
