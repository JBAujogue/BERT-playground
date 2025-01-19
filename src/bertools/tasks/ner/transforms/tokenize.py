from datasets import Dataset, DatasetDict


def tokenize_dataset(dataset: Dataset | DatasetDict, tokenizer, label2id, **kwargs) -> Dataset | DatasetDict:
    """
    Add tokens and token labels to a dataset.
    """
    dataset = dataset.map(build_word_labels)
    dataset = dataset.select_columns(['words', 'word_labels'])
    dataset = dataset.map(
        function = lambda examples: tokenize_and_build_token_labels(examples, tokenizer, label2id, **kwargs),
        batched = True,
    )
    return dataset


def build_word_labels(example):
    """
    Convert span-level labels into word-level labels.
    """
    example["word_labels"] = [
        next(iter([sp["label"] for sp in example["spans"] if (e > sp["start"] and sp["end"] > s)]), "O")
        for s, e in example["offsets"]
    ]
    return example


def tokenize_and_build_token_labels(examples, tokenizer, label2id, **kwargs):
    """
    Add tokens and token labels to a dataset.
    """
    inputs = tokenizer(
        text=examples["words"],
        padding=True,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
        **kwargs,
    )
    inputs["labels"] = build_token_labels(inputs, examples, label2id)
    return inputs


def build_token_labels(inputs, examples, label2id):
    """
    Add tokens and token labels to a dataset.
    """
    labels = []
    for i, word_labels in enumerate(examples["word_labels"]):
        word_ids = inputs.word_ids(batch_index = i)
        
        token_labels = []
        previous_id = None
        for current_id, current_label in zip(word_ids, word_labels):
            # label special tokens to -100 so they are ignored in the loss function
            if current_id is None:
                token_labels.append(-100)

            elif current_label == "O":
                token_labels.append(label2id["O"])
                
            # label the first token of each word with "B-" prefix, else with "I-" prefix
            else:
                prefix = ("B-" if current_id != previous_id else "I-")
                token_labels.append(label2id[f"{prefix}{current_label}"])
            previous_id = current_id
        labels.append(token_labels)
    return labels
