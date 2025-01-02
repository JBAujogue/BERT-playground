from typing import Any

from loguru import logger
from torch import Tensor
from torch.nn.functional import softmax

from bertools.tasks.wordner.typing import Output, Record, Span


def postprocess_predictions(
    record: Record,
    id2label: dict[int, str],
    label2threshold: dict[str, float],
) -> Output:
    """
    Transform a record with word-level predictions into an output.
    """
    spans = word_indices_to_spans(record, id2label)
    spans = [sp for sp in spans if sp["label"] != id2label[0]]
    spans = [sp for sp in spans if is_above_threshold(label2threshold, **sp)]
    return Output(id = record["id"], spans = spans)


def word_indices_to_spans(record: Record, id2label: dict[int, str]) -> list[Span]:
    """
    Convert word-level label indices in IO format into spans.
    """
    content = record["content"]
    offsets = record["offsets"]
    indices = record["indices"]
    confs = record["confidences"]

    if len(offsets) != len(indices):
        logger.warning(f"Length mismatch between {len(offsets)} offsets and {len(indices)} labels")
    if len(indices) == 0:
        return []

    # discard items corresponding to words without thickness
    thicks = [i for i in range(len(offsets)) if offsets[i][1] != offsets[i][0]]
    offsets = [off for i, off in enumerate(offsets) if i in thicks]
    indices = [idx for i, idx in enumerate(indices) if i in thicks]
    confs = [conf for i, conf in enumerate(confs) if i in thicks]

    # spot word indices where there is a change of label, marking span boundaries
    changes = [i for i in range(1, len(indices)) if indices[i - 1] != indices[i]]
    return [
        Span(
            start = offsets[i][0],
            end = offsets[j - 1][1],
            text = content[offsets[i][0] : offsets[j - 1][1]],
            label = id2label[indices[i]],
            confidence = max(confs[i:j]),
        )
        for i, j in zip([0] + changes, changes + [len(indices)])
    ]


def is_above_threshold(
    label2threshold: dict[str, float],
    label: str,
    confidence: float,
    *args,
    **kwags,
) -> bool:
    """
    Assess whether input confidence if above a label-specific threshold.
    Returns True if no threshold is set for the input label.
    """
    return confidence >= label2threshold.get(label, 0.0)


def word_logits_to_word_predictions(logits: Tensor) -> tuple[list[int], list[float]]:
    """
    Convert word-level logits into word-level indices and confidences.
    """
    confs, indices = softmax(logits, dim=-1).max(dim=-1)
    word_indices: list[int] = tensor_to_list(indices)
    word_confs: list[float] = tensor_to_list(confs)
    return (word_indices, word_confs)


def token_tensor_to_word_tensor(tensor: Tensor, masks: Tensor) -> list[Tensor]:
    """
    Convert token-level torch tensor to word-level tensor using a boolean mask marking
    the positions of the first token of each word.
    Output is a list of tensors of variable length.
    """
    return [t[m] for t, m in zip(tensor, masks)]


def tensor_to_list(tensor: Tensor) -> list[Any]:
    """
    Detaches the tensor, move to cpu and convert to list.
    """
    return tensor.detach().cpu().tolist()
