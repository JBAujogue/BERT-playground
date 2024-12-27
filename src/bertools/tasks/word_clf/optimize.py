import os
import subprocess
from itertools import chain
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from optimum.onnxruntime import (
    OptimizationConfig,
    ORTModelForTokenClassification,
    ORTOptimizer,
)


def optimize(config_path: str, base_model_dir: str, onnx_model_dir: str) -> None:
    """
    Attempt to wrap optimum CLI tool into python code.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    optim_flags = list(chain.from_iterable([[f"--{k}", v] for k, v in config.items()]))
    base_model = str(Path(base_model_dir) / "model")
    onnx_model = str(Path(base_model_dir) / onnx_model_dir)
    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "-m",
            base_model,
            "--framework",
            "pt",
            "--library-name",
            "transformers",
            "--task",
            "token-classification",
        ]
        + optim_flags
        + [onnx_model]
    )
    return


def optimize_ort(config_path: str, base_model_dir: str, onnx_model_dir: str) -> None:
    """
    Run basic, cross-platform optimization of onnx model.

    Documentation:
        - https://huggingface.co/docs/transformers/serialization#exporting-a--transformers-model-to-onnx-with-optimumonnxruntime
        - https://huggingface.co/docs/optimum/en/onnxruntime/usage_guides/optimization
        - https://huggingface.co/docs/optimum/v1.2.1/en/onnxruntime/configuration
    Examples:
        - https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/optimization/token-classification/run_ner.py
        - https://github.com/huggingface/blog/blob/main/optimum-inference.md
        - https://towardsdatascience.com/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c
        - https://github.com/urchade/GLiNER/blob/main/examples/convert_to_onnx.ipynb
    """
    with open(config_path, "r") as f:
        optim_config = yaml.safe_load(f)

    load_dir = Path(base_model_dir) / "model"
    save_dir = Path(base_model_dir) / onnx_model_dir

    os.makedirs(save_dir, exist_ok=False)
    model = ORTModelForTokenClassification.from_pretrained(load_dir, export=True, local_files_only=True)
    optimization_config = OptimizationConfig(**optim_config)
    optimizer = ORTOptimizer.from_pretrained(model)
    optimizer.optimize(optimization_config, save_dir, file_suffix=None)
    return
