import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import (
    Dataset, 
    DatasetDict,
    ClassLabel, 
    Features, 
    Sequence, 
    Value,
)


def load_chia_dataset(path_to_data):
    def get_item_list(df, grp_col, item_col):
        return df.groupby(grp_col).apply(lambda g: g[item_col].tolist()).tolist()

    def convert_dataframe_to_dataset(df):
        data = {
            'ids': df.Sequence_id.unique().tolist(),
            'mentions': get_item_list(df, 'Sequence_id', 'Mention'),
            'categories': get_item_list(df, 'Sequence_id', 'Category'),
        }
        return data
    
    df_bio = pd.read_csv(path_to_data, sep = "\t")
    
    class_labels = sorted(list(set(df_bio.Category.unique())))
    class_labels = ClassLabel(names = class_labels)
    
    # dataset separation: 800 trials for training, 100 trials for validation and 100 trials for testing
    ids_bio = sorted(list(set(df_bio.Id.apply(lambda i: i.split('_')[0]))))
    ids_trn, ids_dev = train_test_split(ids_bio, train_size = 0.8, random_state = 13, shuffle = True)
    ids_dev, ids_tst = train_test_split(ids_dev, train_size = 0.5, random_state = 13, shuffle = True)
    
    df_trn = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_trn)]
    df_dev = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_dev)]
    df_tst = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_tst)]
    
    dict_bio = convert_dataframe_to_dataset(df_bio)
    dict_trn = convert_dataframe_to_dataset(df_trn)
    dict_dev = convert_dataframe_to_dataset(df_dev)
    dict_tst = convert_dataframe_to_dataset(df_tst)

    features = Features({
        'ids': Value(dtype = 'string'), 
        'mentions': Sequence(Value(dtype = 'string')), 
        'categories': Sequence(Value(dtype = 'string')),
    })

    raw_datasets = DatasetDict({
        'train': Dataset.from_dict(dict_trn, features = features),
        'eval': Dataset.from_dict(dict_dev, features = features),
        'test': Dataset.from_dict(dict_tst, features = features),
    })
    return (raw_datasets, class_labels)
