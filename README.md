# :hugs: Transformers for NLP

**Disclaimer**: This work is an attempt to explore the landscape provided by the :hugs: Transformers library, by putting the accent on completeness and explainability.
It does **not** cover the use of "large" models, eg > 110M parameters


# Setup
This project uses `miniconda` as environment manager, `python 3.11` as core interpreter, and `poetry 1.8.3` as dependency manager.

Create a new conda environment
```
conda env create -f environment.yaml
```
Activate the environment
```
conda activate bert-playground
```
Install the project dependencies
```
poetry install
```
(Optional) Install pre-commit hooks
```
pre-commit install
```
(Optional) Remove the environment once you are done using this project
```
conda remove -n bert-playground --all
```


# Tasks
All experiments can be inspected by lauching a tensorboard session in a separate terminal
```shell
tensorboard --logdir=models
```


## • Masked Language Modeling
Features
- Standard Masked Language Modeling model
- Learning carries 3 subtasks: (1) token unmasking, (2) token denoising, (3) token autoencoding

Train
```shell
python -m bertools.tasks.mlm train --config-path configs/mlm/train.yaml --output-dir models/mlm/ctti-mlm-baseline
```
Run inference
```python
from transformers import pipeline

model_dir = 'models/mlm/ctti-mlm-baseline'
model = pipeline(
    task = 'fill-mask',
    tokenizer = f'{model_dir}/tokenizer',
    model = f'{model_dir}/model',
)

line = 'Systemic corticosteroids (oral or [MASK]) within 7 days of first dose of 852A (topical or inhaled steroids are allowed)'

model(line)
```


## • Named Entity Recognition
Features
- Standard Named Entity recognition model
- Operates at the token level, by classifying tokens in BIO format

Train
```shell
python -m bertools.tasks.ner train --config-path configs/ner/train.yaml --output-dir models/ner/chia-ner-baseline
```
Run inference
```python
from transformers import pipeline

model_dir = 'models/ner/chia-ner-baseline'
model = pipeline(
    task = 'ner', 
    tokenizer = f'{model_dir}/tokenizer', 
    model = f'{model_dir}/model',
    aggregation_strategy = 'simple',
)

lines = [
    'Multiple pregnancy (more than 3 fetuses)',
    'Had/have the following prior/concurrent therapy:\n',
    'Systemic corticosteroids (oral or injectable) within 7 days of first dose of 852A (topical or inhaled steroids are allowed)',
]

model(lines)
```


## • Named Entity Recognition (Word-level + Causal)
Features
- Custom Named Entity recognition model
- Operates at the word level, by classifying the first token of each word in IO format
- Causal, by taking previous lines as context
- Learning designed to faithfully optimize the behavior at inference


Train
```shell
python -m bertools.tasks.wordner train --config-path configs/wordner/train.yaml --output-dir models/wordner/chia-ner-baseline
```
Evaluate
```shell
python -m bertools.tasks.wordner evaluate --config-path configs/wordner/evaluate.yaml --base-model-dir models/wordner/chia-ner-baseline --output-dir eval
```
Run inference
```python
from bertools.tasks.wordner import WordLevelCausalNER

model = WordLevelCausalNER('models/wordner/chia-ner-baseline')

lines = [
    {'id': '0', 'content': 'Multiple pregnancy (more than 3 fetuses)'}, 
    {'id': '1', 'content': 'Had/have the following prior/concurrent therapy:\n'}, 
    {'id': '2', 'content': 'Systemic corticosteroids (oral or injectable) within 7 days of first dose of 852A (topical or inhaled steroids are allowed)'}
]
model(lines)
```


## • Reranking
Train
```shell
python -m bertools.tasks.rerank train --config-path configs/rerank/train.yaml --output-dir models/rerank/dummy-rerank-baseline
```
Run inference
```python
from sentence_transformers import SentenceTransformer
from bertools.tasks.rerank.inference import run_semantic_search

model = SentenceTransformer('models/rerank/dummy-rerank-baseline/model')

corpus = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
queries = ["how nice is it outside ?"]

run_semantic_search(model = model, corpus = corpus, queries = queries, top_k = 3)
```


# Datasets
## Chia
Prepare Chia dataset for Named Entity Recognition
```shell
python -m bertools.datasets.chia build_ner_dataset --flatten --drop-overlapped --zip_file data/chia/chia.zip --output-dir data/chia/ner-baseline
```
Options:
- `--flatten` ensures multi-expression spans are completed into spans of consecutive words.
- `--drop-overlapped` ensures no two spans overlap.


# Notebooks
- :black_square_button: = TODO
- :white_check_mark: = Functional
- :sparkles: = Documented

| Task | Notebook | Status | Description |
|-----|-----|-----|-----|
| Misc | Datasets | :black_square_button: |Practical description of Datasets & Dataloaders for memory efficiency |
| Tokenization | Tokenization - Benchmark - Pretrained tokenizers | :black_square_button: | Presentation of different tokenization approaches, along with example tokenizers provided by well-renouned pretrained models |
| | Tokenization - Unigram tokenizer - Clinical Trials ICTRP | :white_check_mark: | Fully documented construction and fitting of a Unigram tokenizer |
| Token Embedding | Token Embedding - Benchmark - SGD based methods | :white_check_mark: | Presentation of context-free, SGD-based token embedding methods |
| | Token Embedding - Benchmark - Matrix Factorization methods | :black_square_button: | Presentation of context-free, Matrix factorization token embedding methods |
| | Token Embedding - Clinical Trials ICTRP | :white_check_mark: | Fitting of W2V embedding table on a corpus of I/E criteria |
| Token Classification | [Token Classification - MLM - Albert Small - Clinical Trials ICTRP](https://github.com/JBAujogue/Transformers-for-NLP/blob/main/notebooks/Token%20Classification%20-%20MLM%20-%20Albert%20Small%20-%20Clinical%20Trials%20ICTRP.ipynb) | :white_check_mark: | Full training of Albert small model on Masked Language Model objective on I/E criteria |
| | Token Classification - NER - CHIA - Albert | :sparkles: | Finetuning of Albert model for Named Entity Recognition |


## Reference
- Hugginface full list of [tutorial notebooks](https://github.com/huggingface/transformers/tree/main/notebooks) (see also [here](https://huggingface.co/docs/transformers/main/notebooks#pytorch-examples))
- Huggingface full list of [training scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch)
- Huggingface & Pytorch 2.0 [post](https://www.philschmid.de/getting-started-pytorch-2-0-transformers)
