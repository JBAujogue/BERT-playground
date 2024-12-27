import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_chia_dataset(dataset_path: str | Path, output_dir: str | Path):
    '''
    Splits chia dataset into train/eval/test splits.
    '''
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok = True)
    df_bio = pd.read_csv(dataset_path, sep = "\t")
    
    # dataset separation: 800 trials for training, 100 trials for validation and 100 trials for testing
    ids_bio = sorted(list(set(df_bio.Id.apply(lambda i: i.split('_')[0]))))
    ids_trn, ids_dev = train_test_split(ids_bio, train_size = 0.8, random_state = 13, shuffle = True)
    ids_dev, ids_tst = train_test_split(ids_dev, train_size = 0.5, random_state = 13, shuffle = True)
    
    df_trn = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_trn)]
    df_dev = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_dev)]
    df_tst = df_bio[df_bio.Id.apply(lambda i: i.split('_')[0]).isin(ids_tst)]
    
    df_trn.to_parquet(output_dir / 'train.parquet', index = False)
    df_dev.to_parquet(output_dir / 'eval.parquet', index = False)
    df_tst.to_parquet(output_dir / 'test.parquet', index = False)