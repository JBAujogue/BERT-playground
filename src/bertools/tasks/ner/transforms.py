def tokenize_dataset(raw_datasets, tokenizer, class_labels, **kwargs):
    B_I_mapping = {l: 'I'+l[1:] for l in class_labels.names if l.startswith('B-')}

    tokenized_datasets = raw_datasets.map(
        function = lambda examples: tokenize_and_align_categories(tokenizer, examples, B_I_mapping, **kwargs), 
        batched  = True,
    )
    tokenized_datasets = tokenized_datasets.map(
        function = lambda examples: create_labels(examples, class_labels), 
        batched  = True,
    )
    return tokenized_datasets


def switch_B_to_I(idx, mapping_dict):
    return mapping_dict[idx] if idx in mapping_dict else idx


def tokenize_and_align_categories(tokenizer, examples, B_I_mapping, **kwargs):
    # tokenize into a BatchEncoding instance
    # TODO: set 'padding' and 'max_length' in accordance with tokenizer and model
    tokenized_inputs = tokenizer(examples["mentions"], **kwargs)

    # align categories
    all_toks = []
    all_cats = []
    for i, cat in enumerate(examples['categories']):
        word_ids = tokenized_inputs.word_ids(batch_index = i)
        previous_word_idx = None
        cats = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                cats.append(None)
                
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                cats.append(cat[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                cats.append(switch_B_to_I(cat[word_idx], B_I_mapping))
            previous_word_idx = word_idx

        all_toks.append(tokenized_inputs.tokens(batch_index = i))
        all_cats.append(cats)

    tokenized_inputs["tokens"] = all_toks
    tokenized_inputs["token_categories"] = all_cats
    return tokenized_inputs


def create_labels(examples, class_labels):
    examples['labels'] = [
        [(class_labels.str2int(c) if c in class_labels.names else -100) for c in cats] 
        for cats in examples['token_categories']
    ]
    return examples
