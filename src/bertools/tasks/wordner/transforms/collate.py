from typing import Iterable

import numpy as np
from loguru import logger
from torch import LongTensor, Tensor, stack
from transformers import BatchEncoding, PreTrainedTokenizerFast

from bertools.tasks.wordner.utils.typing import Record


class Collator:
    """
    Collator class.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        label2id: dict[str, int],
        max_tokens: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_tokens or tokenizer.model_max_length

    @staticmethod
    def get_consistent_records(records: list[Record], masks: Tensor) -> Iterable[int]:
        """
        Return line indices where the number of unmasked tokens is identical to the
        number of word offsets.
        """
        valids = [(len(records[i]["offsets"]) == masks[i].sum().item()) for i in range(len(records))]
        if not all(valids):
            logger.warning(
                "\n\n".join(f"Collator issue with record {records[i]}" for i, b in enumerate(valids) if not b)
            )
        return np.where(valids)[0]

    @staticmethod
    def truncate_records(records: list[Record], masks: Tensor) -> list[Record]:
        """
        Truncate words and offsets to match the amount of unmasked tokens.
        """
        num_word_in_encoded = masks.sum(dim=1)
        return [
            r | {"words": r["words"][-num_word_in_encoded[i]:], "offsets": r["offsets"][-num_word_in_encoded[i]:]} for i, r in enumerate(records)
        ]

    @staticmethod
    def build_mask_tensor(inputs: BatchEncoding, records: list[Record]) -> Tensor:
        """
        Computes a mask of shape (batch_size, num_tokens) with 0-1 entries,
        where 1 marks the position of the first token of each word.
        The remaining positions are filled with 0, eg:
            - all context tokens
            - all special tokens
            - all tokens not at the begining of a word
        """
        special_mask = inputs.pop("special_tokens_mask")
        nobegin_mask = (inputs.pop("offset_mapping")[:, :, 0] == 0).long()

        # mask only showing the positions of the first token of each word
        # after left truncation by the tokenizer
        general_mask = (1 - special_mask) * nobegin_mask

        # compute number of context words to mask
        num_word_in_encoded = general_mask.sum(dim=1)
        num_word_to_discard = [max(0, num_word_in_encoded[i] - len(r["words"])) for i, r in enumerate(records)]

        # compute mask hiding context words
        cum_mask = general_mask.cumsum(dim=1)
        context_mask = stack([(cum_mask[i] > num_word_to_discard[i]).long() for i in range(len(records))], dim=0)
        return context_mask * general_mask

    @staticmethod
    def build_label_tensor(masks: Tensor, labels: list[Tensor]) -> Tensor:
        """
        For training purpose only.
        Computes a tensor of shape (batch_size, num_tokens), with label ids at token
        positions marking the beginning of a word with corresponding label, and -100
        everywhere else.
        """
        indexed_pos = masks.cumsum(dim=1) * masks
        labeled_pos = [labels[i][indexed_pos[i]] for i in range(len(labels))]
        return stack(labeled_pos, dim=0)

    def encode_labels(self, records: list[Record]) -> list[Tensor]:
        """
        For training purpose only.
        Computes for each record a LongTensor of word-level label indices, with the
        -100 index appended at position 0.
        """
        return [LongTensor([-100] + spans_to_indices(r, self.label2id)) for r in records]

    def tokenize(self, records: list[Record]) -> BatchEncoding:
        """
        Tokenize messages.
        """
        return self.tokenizer(
            text=[e["context"] + e["words"] for e in records],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    def collate_for_training(self, records: list[Record]) -> BatchEncoding:
        """
        For training purpose only.
        Computes torch tensors of token ids and labels.
        """
        inputs = self.tokenize(records)
        masks = self.build_mask_tensor(inputs, records)
        records = self.truncate_records(records, masks)
        labels = self.encode_labels(records)
        inputs["labels"] = self.build_label_tensor(masks, labels)
        return inputs

    def __call__(self, records: list[Record]) -> dict[str, BatchEncoding | Tensor]:
        """
        For inference purpose only.
        Tokenize messages and build boolean mask marking the positions of the first
        token of each word.
        """
        inputs = self.tokenize(records)
        masks = self.build_mask_tensor(inputs, records)
        records = self.truncate_records(records, masks)
        return {"records": records, "inputs": inputs, "masks": masks.bool()}


def spans_to_indices(record: Record, label2id: dict[str, int]) -> list[int]:
    """
    Convert labeled spans into word-level label indices in IO format.
    Words that are not overlapped by a span are endowed the default 0 index.
    """
    return [
        next(
            iter([label2id[sp["label"]] for sp in record["spans"] if (e > sp["start"] and sp["end"] > s)]),
            0,
        )
        for s, e in record["offsets"]
    ]
