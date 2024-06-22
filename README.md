# :hugs: Transformers for NLP

## Disclaimer

**This is an ongoing work**

This work is an attempt to explore the landscape provided by the :hugs: Transformers library, by putting the accent on completeness and explainability.
It does **not** cover the following aspects:
  - Experiment tracking, eg using mlflow or WandB
  - Usage of Google Colab
  - Usage of "large" models, eg > 110M parameters



## CLI tools

#### Run Masked Language Modeling
```powershell
python -m bertools.tasks.run_mlm --config-path .\configs\mlm.yaml --logging-dir .\mlruns\mlm\albert-base-v2-ctti-dummy --save-model false
```

#### Run Reranking
```powershell
python -m bertools.tasks.run_rerank --config-path .\configs\rerank.yaml --logging-dir .\mlruns\rerank\all-mpnet-base-v2-rerank-dummy --output-dir .\models\rerank\all-mpnet-base-v2-rerank-dummy
```
  
  

## Notebooks
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



## Published practical features of Transformers

### 1. Architecture design


### 2. Training protocol

- **Whole Word Masking**: Variant of Masked Language Model that jointy mask all tokens of a word, used in particular in [PubMedBert](https://arxiv.org/pdf/2007.15779.pdf)

- **Incremental token masking ratio**: The idea of increasing the ration of token masking during pretraining, first proposed in [Adversarial training](https://arxiv.org/pdf/2004.08994.pdf) as starting from 5% and augmenting by 5% every 20% of training process, ending with 25% masking.


