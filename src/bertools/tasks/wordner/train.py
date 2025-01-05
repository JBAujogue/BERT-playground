import os
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]
from datasets import Dataset
from loguru import logger
from torch.nn.functional import one_hot
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bertools.tasks.wordner.evaluate import (
    compute_metrics,
    compute_roc_curves,
    compute_roc_values,
    compute_safety_thresholds,
    save_evaluation,
)
from bertools.tasks.wordner.transforms import (
    Collator,
    concat_lists,
    postprocess_predictions,
    preprocess_records,
    token_tensor_to_word_tensor,
    word_logits_to_word_predictions,
)
from bertools.tasks.wordner.utils.load import load_annotations
from bertools.tasks.wordner.utils.typing import Record


def train(config_path: str | Path, output_dir: str | Path) -> None:
    """
    End-to-end training of word-level NER model.
    """
    logger.info(
        """
        ==============================================
        == Running Word-level NER training pipeline ==
        =============================================="""
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
    dataset_as_records = prepare_dataset_for_training(**train_config["dataset_args"])
    dataset = {k: Dataset.from_list(v) for k, v in dataset_as_records.items()}
    logger.info("Loaded datasets:" + "".join(f"\n\t- '{k}' with {len(v)} lines" for k, v in dataset.items()))

    # load list of labels, with the empty label appearing first
    labels = [""] + sorted({sp["label"] for rs in dataset_as_records.values() for r in rs for sp in r["spans"]})
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in id2label.items()}
    logger.info("Loaded labels:" + "".join(f"\n\t- {k}: {v}" for k, v in id2label.items()))

    # load tokenizer
    tokenizer_args = {"truncation_side": "left", "add_prefix_space": True} | train_config["tokenizer_args"]
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

    # load model
    model_args = {"id2label": id2label, "label2id": label2id} | train_config["model_args"]
    model = AutoModelForTokenClassification.from_pretrained(**model_args).to(device).train()
    logger.info(f"Loaded model with {model.num_parameters()} parameters")

    # load collator
    collator = Collator(tokenizer, label2id, **train_config["collator_args"])

    # train model
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        **train_config["training_args"],
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator.collate_for_training,
        compute_metrics=lambda p: compute_metrics_for_training(
            p,
            id2label,
            train_config["target_fpr"],
        ),
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval", None),
    )
    trainer.train()
    logger.success("Training phase complete")

    # evaluate
    if "eval" in dataset:
        trainer.evaluate(eval_dataset=dataset["eval"], metric_key_prefix="eval")
        logger.success("Evaluation phase complete")

    if "test" in dataset:
        trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        logger.success("Testing phase complete")

    # save artifacts
    with open(logging_dir / "train_config.yaml", "wt") as f:
        yaml.safe_dump(train_config, f, encoding="utf-8")

    eval_result = get_trainer_last_log(trainer, prefix="eval")
    if eval_scores := eval_result.get("eval_scores", None):
        os.makedirs(logging_dir / "eval")
        save_evaluation(
            save_dir=logging_dir / "eval",
            scores=eval_scores,
            curves=eval_result["eval_curves"],
        )
    test_result = get_trainer_last_log(trainer, prefix="test")
    if test_scores := test_result.get("test_scores", None):
        os.makedirs(logging_dir / "test")
        save_evaluation(
            save_dir=logging_dir / "test",
            scores=test_scores,
            curves=test_result["test_curves"],
        )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.save_pretrained(output_dir / "model")

    with open(output_dir / "model_config.yaml", "wt") as f:
        inference_config = {
            "max_tokens": collator.max_length,
            "preprocess_args": train_config["dataset_args"]["preprocess_args"],
            "safety_thresholds": eval_result.get("eval_safety_thresholds", {}),
        }
        yaml.safe_dump(inference_config, f, encoding="utf-8")

    logger.success("Job complete")
    return


def prepare_dataset_for_training(
    data_files: dict[str, list[str]],
    preprocess_args: dict[str, Any],
    upsampling_coefs: dict[str, int] | None = None,
) -> dict[str, list[Record]]:
    """
    Load and prepare train/eval/test splits as a dict of Datasets.
    """
    return {k: prepare_split_for_training(k, v, preprocess_args, upsampling_coefs) for k, v in data_files.items()}


def prepare_split_for_training(
    split_name: str,
    file_list: list[str],
    preprocess_args: dict[str, Any],
    upsampling_coefs: dict[str, int] | None = None,
) -> list[Record]:
    """
    Prepare a single train/eval/test split, and aggregate preprocessed results into a Dataset.
    """
    texts = load_annotations(file_list)
    records = concat_lists([preprocess_records(c, **preprocess_args) for c in texts.values()])
    if split_name == "train" and upsampling_coefs and len(upsampling_coefs) > 0:
        records = upsample_records(records, upsampling_coefs)
    return records


def upsample_records(records: list[Record], upsampling_coefs: dict[str, int]) -> list[Record]:
    """
    Duplicate each record of certain categories, as many times as specified for this category.
    """
    upsampling_records = {
        k: [r for r in records if k in [sp["label"] for sp in r["spans"]]] * v for k, v in upsampling_coefs.items()
    }
    logger.warning(
        (
            "Adding duplicates to pool of records:"
            + "".join(f"\n\t- {k}: {len(v)} lines" for k, v in upsampling_records.items())
        )
    )
    for dupl in upsampling_records.values():
        if dupl:
            period = int(len(records) / len(dupl))
            records = concat_lists(
                [
                    records[i : i + period] + ([dupl[round(i / period)]] if round(i / period) < len(dupl) else [])
                    for i in range(0, len(records), period)
                ]
            )
    return records


def compute_metrics_for_training(p: EvalPrediction, id2label: dict[int, str], target_fpr) -> dict[str, Any]:
    """
    Compute our kpis.
    """
    gold_token_label_ids = torch.LongTensor(p.label_ids)
    masks = gold_token_label_ids != -100

    # recreate fake contents & words offsets, as it is necessary for transforming
    # token/word-level classification into final spans
    lengths = masks.sum(dim=1)
    ids = [str(i) for i in range(len(lengths))]
    contents = ["w " * length for length in lengths]
    offsets = [[(2 * i, 2 * i + 1) for i in range(length)] for length in lengths]

    # compute pred & gold logits as tensors of shape (num_msgs, max_num_tokens, num_classes)
    pred_token_logits = torch.Tensor(p.predictions)
    gold_token_logits = one_hot(gold_token_label_ids.clamp(min=0), pred_token_logits.shape[-1]).float()

    # convert token-level logits into word-level indices
    pred_indices_confs = [
        word_logits_to_word_predictions(logit) for logit in token_tensor_to_word_tensor(pred_token_logits, masks)
    ]
    gold_indices_confs = [
        word_logits_to_word_predictions(logit) for logit in token_tensor_to_word_tensor(gold_token_logits, masks)
    ]
    pred_records = (
        Record(
            id=_id,
            content=content,
            offsets=offs,
            indices=inds,
            confidences=confs,
        )
        for _id, content, offs, (inds, confs) in zip(ids, contents, offsets, pred_indices_confs)
    )
    gold_records = (
        Record(
            id=_id,
            content=content,
            offsets=offs,
            indices=inds,
            confidences=confs,
        )
        for _id, content, offs, (inds, confs) in zip(ids, contents, offsets, gold_indices_confs)
    )
    # convert into final predictions
    pred_outputs = [postprocess_predictions(r, id2label, {}) for r in pred_records]
    gold_outputs = [postprocess_predictions(r, id2label, {}) for r in gold_records]

    # compute roc values
    roc_values = compute_roc_values(gold_outputs, pred_outputs)

    return {
        "scores": compute_metrics(gold_outputs, pred_outputs),
        "curves": compute_roc_curves(roc_values),
        "safety_thresholds": compute_safety_thresholds(roc_values, target_fpr),
    }


def get_trainer_last_log(trainer: Trainer, prefix: str) -> dict[str, Any]:
    """
    Retrieve last eval result for the training log history.
    """
    logs = [log for log in trainer.state.log_history if any(key.startswith(prefix) for key in log)]
    return logs[-1] if logs else {}
