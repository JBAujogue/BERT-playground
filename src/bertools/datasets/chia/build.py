import os
from pathlib import Path
from loguru import logger


from bertools.datasets.chia.load import load_texts_from_zipfile, load_spans_from_zipfile
from bertools.datasets.chia.transform import flatten_spans, get_maximal_spans
from bertools.datasets.chia.split import split_spans


def build_ner_dataset(zip_file: str | Path, output_dir: str | Path) -> None:
    """ 
    Load chia dataset and prepare its annotated spans.
    """
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok = False)
    
    # load raw data
    df_texts = load_texts_from_zipfile(zip_file)
    df_spans = load_spans_from_zipfile(zip_file)
    logger.info(f"Loaded {len(df_texts)} texts and {len(df_spans)} spans.")
    
    # apply transforms over spans
    df_spans = flatten_spans(df_spans)
    df_spans = get_maximal_spans(df_spans)
    logger.info(f"Created {len(df_spans)} flat, non-overlapping spans.")
    
    # compute train / eval / test splits
    df_train, df_eval, df_test = split_spans(
        df_spans, 
        train_ratio = 0.8, 
        eval_test_ratio = 0.5,
    )
    logger.info((
        f"Splitted spans into:" 
        f"\n- {len(df_train)} train spans"
        f"\n- {len(df_eval)} eval spans"
        f"\n- {len(df_test)} test spans"
    ))
    # save splits to parquet
    df_train.to_parquet(out_dir/'train.parquet', index = False)
    df_eval.to_parquet(out_dir/'eval.parquet', index = False)
    df_test.to_parquet(out_dir/'test.parquet', index = False)
    logger.info(f"Splits saved to {output_dir}.")
    return
