from typing import List, Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Pipeline
from transformers.modeling_outputs import BaseModelOutput
from optimum.onnxruntime import ORTModelForFeatureExtraction


class OnnxPipelineForFeatureExtraction(Pipeline):
    '''
    Transformers pipeline leveraging ORT model for feature extraction.
    '''
    def __init__(
        self, 
        model_name: str, 
        batch_size: int = 64,
        max_seq_length: int = 512,
        normalize_embeddings: bool = False,
        *args, **kwargs,
        ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length = max_seq_length,
        )
        if torch.cuda.is_available():
            provider = 'CUDAExecutionProvider'
            device = 0
        else:
            provider = 'CPUExecutionProvider'
            device = -1
        model = CustomORTModelForFeatureExtraction.from_pretrained(
            model_id = model_name, file_name = 'model.onnx', provider = provider, use_io_binding = True,
        )
        super().__init__(
            task = "feature-extraction", 
            tokenizer = tokenizer, 
            model = model, 
            batch_size = batch_size, 
            device = device,
            normalize_embeddings = normalize_embeddings,
            *args, **kwargs,
        )

    def _sanitize_parameters(self, **kwargs):
        normalize_embeddings = kwargs.pop('normalize_embeddings', None)
        postprocess_kwargs = (
            dict(normalize_embeddings = normalize_embeddings) if normalize_embeddings else {}
        )
        return {}, {}, postprocess_kwargs
      
    def preprocess(self, inputs: str | List[str], *args, **kwargs) -> Dict[str, torch.Tensor]:
        # optimum-cli skips 'token_type_ids' from the model's "_ordered_input_names"
        # attribute, it is thus necessary to force the tokenizer NOT to output it.
        return self.tokenizer(
            inputs,  padding = True, truncation = True, return_token_type_ids = False, return_tensors = 'pt', 
        )

    def _forward(self, model_inputs: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.model(**model_inputs)
        return {"token_embeddings": outputs[0], "attention_mask": model_inputs["attention_mask"]}

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masks = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (
            torch.sum(token_embeddings * masks, 1)/torch.clamp(masks.sum(1), min = 1e-9)
        )

    def postprocess(self, model_outputs: Dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        pooling = self._mean_pooling(
            model_outputs["token_embeddings"], model_outputs['attention_mask']
        )
        if kwargs.pop('normalize_embeddings', False):
            pooling = F.normalize(pooling, p = 2, dim = -1)
        return pooling.squeeze().tolist()


# Introduces a fix over ORTModelForFeatureExtraction
# Fix: select "token_embeddings" as content of "last_hidden_state".
# The key "sentence_embedding" is disregarded because it includes padding
# tokens in the pooling, which is detrimental in case of batched inputs. 
# Actual pooling occurs in a separate process done afterward.
class CustomORTModelForFeatureExtraction(ORTModelForFeatureExtraction):
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
        ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names = self._ordered_input_names,
            )
            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            key = ("last_hidden_state" if "last_hidden_state" in output_buffers else "token_embeddings")
            last_hidden_state = output_buffers[key].view(output_shapes[key])

        else:
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            onnx_inputs = self._prepare_onnx_inputs(use_torch, **model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, *onnx_outputs)

            key = ("last_hidden_state" if "last_hidden_state" in model_outputs else "token_embeddings")
            last_hidden_state = model_outputs[key]

        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(last_hidden_state = last_hidden_state)
