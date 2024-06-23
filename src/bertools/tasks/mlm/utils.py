from itertools import chain
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorForMLM(DataCollatorMixin):
    '''
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>
    '''
    tokenizer: PreTrainedTokenizerBase
    task_proportions: Iterable = (12, 1.5, 1.5, 85)    # mask | random noise | keep to learn | ignore
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        # Ensures that entries in task_proportions sums to 1
        self.task_proportions = tuple(abs(v) for v in self.task_proportions)
        if sum(self.task_proportions) > 0:
            tot = sum(self.task_proportions)
            self.task_proportions = torch.tensor(tuple(v/tot for v in self.task_proportions))
        else:
            raise ValueError('"task_proportions" sum of entites should be positive"')

    def tf_call(self, *args, **kwargs):
        raise NotImplementedError("This data collator is Pytorch-only")

    def numpy_call(self, *args, **kwargs):
        raise NotImplementedError("This data collator is Pytorch-only")
    
    def torch_call(
        self, examples: List[Dict[str, torch.Tensor]], *args, **kwargs
        ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(
            examples, return_tensors = "pt", pad_to_multiple_of = self.pad_to_multiple_of
        )
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_edit_tokens(
            inputs = batch["input_ids"],
            special_tokens_mask = special_tokens_mask,
        )
        return batch

    def torch_edit_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor
        ) -> Tuple[torch.Tensor]:
        """
        Prepare noisy tokens inputs/labels for denoising language modeling: 100% random.
        """
        # init target labels
        labels = inputs.clone()
        
        # get special tokens mask
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens = True
                ) 
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype = torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
            
        # We split tokens into [mask | random noise | keep to learn | keep to ignore] 
        # subgroups through random sampling according to proportions given in self.task_proportions
        distribution_matrix = torch.multinomial(
            self.task_proportions, 
            labels.nelement(),
            replacement = True,
        ).reshape(labels.shape)
        distribution_matrix.masked_fill_(special_tokens_mask, value = 3)
        
        mask_matrix = (distribution_matrix == 0).bool()
        rand_matrix = (distribution_matrix == 1).bool()
        keep_matrix = (distribution_matrix == 2).bool()
        ignr_matrix = (distribution_matrix == 3).bool()

        # we mask part of tokens
        inputs[mask_matrix] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # we replace part of tokens by random ones
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[rand_matrix] = random_words[rand_matrix]
        
        # we keep part of tokens as-is
        # inputs[keep_matrix] = inputs[keep_matrix]
        
        # We only compute loss on masked / randomized / learned tokens
        labels[ignr_matrix] = -100 
        return (inputs, labels)


def form_constant_length_blocks(examples, block_size):
    # Concatenate all texts.
    keys = [k for k in examples.keys() if k != 'text']
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[keys[0]])
    
    # We could add padding instead of this drop
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split by chunks of max_len
    return {
        k: [t[i : i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }