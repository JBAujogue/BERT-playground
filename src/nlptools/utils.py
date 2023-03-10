
"""
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
"""
import os
from dataclasses import dataclass, field
from typing import Optional




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    hub_model_name: str = field(
        metadata={"help": "Name model identifier from huggingface.co/models"}
    )
    base_model_name: str = field(
        metadata={"help": "Name base model checkpoint prior to finetuning"}
    )
    base_model_task: str = field(
        metadata={"help": "Task name of base model checkpoint"}
    )
    final_model_name: str = field(
        metadata={"help": "Name of finetuned model"}
    )
    final_model_task: str = field(
        metadata={"help": "Task name of finetuned model"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as hub_model_name of base_model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as hub_model_name of base_model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    hub_model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    def __post_init__(self):
        if self.hub_model_name is None and self.base_model_name is None:
            raise ValueError("Need either a hub model name or a base model name")
        if self.base_model_name is None:
            raise ValueError("Need a base model name")
        if self.base_model_task is None:
            raise ValueError("Need a base model task")
        if self.final_model_name is None:
            raise ValueError("Need a final model name")
        if self.final_model_task is None:
            raise ValueError("Need a final model task")
        
        self.base_model_task = self.base_model_task.upper()
        self.final_model_task = self.final_model_task.upper()
        self.base_model_name = os.path.join(self.base_model_task, self.base_model_name.lower())
        self.final_model_name = os.path.join(self.final_model_task, self.final_model_name.lower())




