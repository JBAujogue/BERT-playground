# :hugs: Transformers for NLP

**Disclaimer**: This work is an attempt to explore the landscape provided by the :hugs: Transformers library, by putting the accent on completeness and explainability.
It does **not** cover the use of "large" models, eg > 110M parameters


# Getting started
This project uses python `3.11` as core interpreter, and poetry `1.6.1` as dependency manager.

Create a new conda environment
```
conda env create -f environment.yml
```
Activate the environment
```
conda activate bert-playground
```
Install the project dependencies
```
poetry install
```
Remove the environment
```
conda remove -n bert-playground --all
```


# Summary of the tasks
## • Masked Language Modeling
```shell
python -m bertools.tasks.mlm --config-path configs/mlm.yaml --output-dir models/mlm/dummy
```


## • Reranking
```shell
python -m bertools.tasks.rerank --config-path configs/rerank.yaml --output-dir models/rerank/dummy
```


## • Named Entity Recognition
```shell
python -m bertools.tasks.ner --config-path configs/ner.yaml --output-dir models/ner/dummy
```


## • Word-level Causal Named Entity Recognition
### Prepare training dataset
Prepare Chia dataset for Named Entity Recognition
```shell
python -m bertools.datasets.chia build_ner_dataset --flatten --drop-overlapped --zip_file data/chia/chia.zip --output-dir data/chia/spans-flat-nooverlap
```
Options:
- `--flatten` ensures multi-expression spans are completed into spans of consecutive words.
- `--drop-overlapped` ensures no two spans overlap.

Train a model
```shell
python -m bertools.tasks.wordner train --config-path configs/wordner/train.yaml --output-dir models/wordner/dummy
python -m bertools.tasks.wordner evaluate --config-path configs/wordner/evaluate.yaml --base-model-dir models/wordner/dummy --output-dir eval
```
Run inference
```python
from bertools.tasks.wordner import WordLevelCausalNER

model = WordLevelCausalNER('models/wordner/dummy')

lines = [
    {'id': 'NCT00236340_exc--0', 'content': 'Multiple pregnancy (more than 3 fetuses) \n'}, 
    {'id': 'NCT00236340_exc--1', 'content': 'Maternal history of placental abruptio \n'}, 
    {'id': 'NCT00236340_exc--2', 'content': 'Fetus with IUGR \n'}, 
    {'id': 'NCT00236340_exc--3', 'content': 'Pregnancy complicated with pre-eclampsia \n'}, 
    {'id': 'NCT00236340_exc--4', 'content': 'Unability to give informed consent       \n'}, 
    {'id': 'NCT00236340_inc--0', 'content': 'Pregnant women with abdomen discumfort and ultrasound diagnosis of polyhydramnios (AFI>25cm) \n'}, 
    {'id': 'NCT00236340_inc--1', 'content': 'Single or twin pregnancies \n'}, 
    {'id': 'NCT00319748_exc--0', 'content': 'Had/have the following prior/concurrent therapy: \n'}, 
    {'id': 'NCT00319748_exc--1', 'content': 'Systemic corticosteroids (oral or injectable) within 7 days of first dose of 852A (topical or inhaled steroids are allowed) \n'}
]

model(lines)
```


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
