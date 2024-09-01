from itertools import chain


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