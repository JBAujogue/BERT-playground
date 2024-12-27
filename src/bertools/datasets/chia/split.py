import pandas as pd
from sklearn.model_selection import train_test_split


def split_spans(
    df_spans: pd.DataFrame, 
    train_ratio: float, 
    eval_test_ratio: float,
    random_state: int = 0,
) -> tuple[pd.DataFrame,...]:
    """
    Split a DataFrame of spans into train/eval/test splits.
    """
    # split by text id
    ids = sorted(df_spans['id'].unique().tolist())
    ids_train, ids_dev = train_test_split(
        ids, 
        train_size = train_ratio, 
        random_state = random_state, 
        shuffle = True,
    )
    ids_eval, ids_test = train_test_split(
        ids_dev, 
        train_size = eval_test_ratio, 
        random_state = random_state, 
        shuffle = True,
    )
    df_train = df_spans[df_spans['id'].isin(ids_train)].reset_index(drop = True)
    df_eval = df_spans[df_spans['id'].isin(ids_eval)].reset_index(drop = True)
    df_test = df_spans[df_spans['id'].isin(ids_test)].reset_index(drop = True)
    return (df_train, df_eval, df_test)
