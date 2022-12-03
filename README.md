# Transformers for NLP


## Table of Content

| Notebook | Description |
|-----|-----|
| Dataset |
| Datasets - Benchmark | Practical description of Datasets & Dataloaders for memory efficiency |
| Tokenization | |
| Tokenization - Benchmark - Pretrained tokenizers | Presentation of different tokenization approaches, along with example tokenizers provided by well-renouned pretrained models |
| Tokenization - Unigram tokenizer - Clinical Trials ICTRP | Fully documented construction and fitting of a Unigram tokenizer |
| Token Embedding | |
| Token Embedding - Benchmark - Word2Vec, FastText, Doc2Vec | Presentation of context-free, SGD-based token embedding methods |
| Token Embedding - Benchmark - Matrix Factorization methods | Presentation of context-free, Matrix factorization token embedding methods |



| Field  | Description |  Optional | Default |
       | ------ | ----------- | --------- | ------- |
       | manual_entry_indicator | no: is not is allow manual entry <br /> yes: is manual entry enabled| yes | no |
       | amounts | json object containing all transaction amounts <br /> <br /> <table> <tr> <td> Subfield </td> <td> Description </td> <td> Optional </td> <td> Default </td> </tr> <tr> <td> tip </td>  <td> transaction tip amount </td> <td> yes </td> <td> NA </td> </tr> <tr> <td> total </td> <td> equal to Base  Amount + Base amount for  Reduced State Tax + City Tax + State Tax + Reduced State Tax + Tip or Cash back </td> <td> no </td> <td> NA </td> </tr> <tr> <td> cashback </td> <td> cash back amount </td> <td> yes </td> <td> NA </td> </tr> <tr> <td> state_tax </td> <td> State tax amount </td> <td> yes </td> <td> NA </td> </tr> <tr> <td> city_tax </td> <td> City tax amount </td> <td> yes </td> <td> NA </td> </tr> <tr> <td> reduced_tax </td> <td> Reduced state tax amount </td> <td> yes </td> <td> NA </td> </tr> <tr> <td> base_reduced_tax </td> <td> Reduced state tax base amount </td> <td> yes </td> <td> NA </td> </tr> </table> | no | NA |
