from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from bertools.tasks.wordner.transforms import (
    Collator,
    concat_lists,
    postprocess_predictions,
    preprocess_records,
    token_tensor_to_word_tensor,
    word_logits_to_word_predictions,
)
from bertools.tasks.wordner.utils.typing import Input, Output, Record


class WordLevelCausalNER:
    """
    Class for end-to-end inference of a word-level NER model.
    """

    def __init__(
        self,
        base_model_dir: str | Path,
        onnx_model_dir: str | Path | None = None,
        batch_size: int = 8,
    ):
        """
        Initialize model.
        """
        base_model_dir = Path(base_model_dir)
        with open(base_model_dir / "model_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Note: 'add_prefix_space' is set to True because we tokenize words, hence we
        # need to remind the tokenizer that those words don't mark the beginning of a
        # sentence.
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=base_model_dir / "tokenizer",
            add_prefix_space=True,
            truncation_side="left",
            **config.get("tokenizer_args", {}),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # if "onnx_model_dir" is provided, load onnx model from here
        if onnx_model_dir:
            from optimum.onnxruntime import ORTModelForTokenClassification

            self.model = ORTModelForTokenClassification.from_pretrained(
                model_id=base_model_dir / onnx_model_dir,
                file_name="model.onnx",
                provider=device.upper() + "ExecutionProvider",
                use_io_binding=True,
                local_files_only=True,
            )
            logger.warning(f"Model loaded with providers {self.model.providers}")
        # else, load safetensors model from "base_model_dir"
        else:
            from transformers import AutoModelForTokenClassification

            self.model = (
                AutoModelForTokenClassification.from_pretrained(
                    base_model_dir / "model",
                    **config.get("model_args", {}),
                )
                .to(device)
                .eval()
            )
            logger.warning(f"Model loaded on device {self.model.device}")

        self.id2label = self.model.config.id2label
        self.collator = Collator(
            tokenizer=self.tokenizer,
            label2id=self.model.config.label2id,
            max_tokens=config["max_tokens"],
        )
        self.batch_size = batch_size
        self.label2threshold = config.get("safety_thresholds", {})
        self.preprocess_args = config.get("preprocess_args", {})

    @torch.inference_mode()
    def compute_batched_predictions(
        self,
        records: list[Record],
        inputs: BatchEncoding,
        masks: Tensor,
    ) -> list[Record]:
        """
        Processes a batch of messages and returns for each message a tensor of word
        logits.
        """
        token_logits = self.model(**inputs.to(self.model.device)).logits.double()
        word_logits = token_tensor_to_word_tensor(token_logits, masks)
        word_preds = [word_logits_to_word_predictions(logits) for logits in word_logits]
        return [Record(indices=inds, confidences=confs, **r) for r, (inds, confs) in zip(records, word_preds)]

    def __call__(self, inputs: list[Input]) -> list[Output]:
        """
        End-to-end inference over list of input records, using Dataloader as minibatcher.
        """
        # preprocess records
        records = preprocess_records(inputs, **self.preprocess_args)
        batcher = DataLoader(
            dataset=records,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collator,
        )
        # compute word-level predictions over records
        records = concat_lists([self.compute_batched_predictions(**b) for b in batcher])
        return [postprocess_predictions(r, self.id2label, self.label2threshold) for r in records]
