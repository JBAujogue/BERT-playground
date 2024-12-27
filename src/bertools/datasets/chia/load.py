from pathlib import Path
import json
import zipfile
import gzip
import pandas as pd


def load_texts_from_zipfile(zip_file: str | Path) -> pd.DataFrame:
    """
    Load texts from zipfile.
    """
    archive = zipfile.ZipFile(zip_file, 'r')
    files = [f for f in archive.namelist() if f.endswith('.txt')]
    texts = []
    for file in files:
        t_id = Path(file).stem
        with archive.open(file, 'r') as f:
            lines = [l.decode('utf-8').strip().replace('⁄', '/') for l in f.readlines()]
            texts.append({'id': t_id, 'content': ' \n'.join(lines)})
    
    return pd.DataFrame.from_records(texts)


def load_spans_from_zipfile(zip_file: str | Path) -> pd.DataFrame:
    """
    Load annotated spans from zipfile.
    """
    archive = zipfile.ZipFile(zip_file, 'r')
    file_list = [f for f in archive.namelist() if f.endswith('.ann')]
    span_list = []
    for file in file_list:
        t_id = Path(file).stem
        with archive.open(file, 'r') as f:
            lines = [l.decode('utf-8').strip().replace('⁄', '/') for l in f.readlines()]
            spans = [l.split('\t')[:2] for l in lines if l.startswith('T')]
            spans = [
                {
                    'id': t_id, 
                    'span_id': sp_id, 
                    'label': label, 
                    'start': start, 
                    'end': end,
                } 
                for sp_id, sp_data in spans
                for label, start, end in parse_span_metadata(sp_data)
            ]
            span_list += spans
    
    return pd.DataFrame.from_records(span_list)


def parse_span_metadata(metadata: str) -> list[list[str | int]]:
    """
    Parse span metadata from raw annotation.
    """
    label = metadata.split(' ')[0]
    pairs = ' '.join(metadata.split(' ')[1:]).split(';')
    pairs = [p.split(' ') for p in pairs]
    return [[label, int(start), int(end)] for start, end in pairs]
    

def load_bio_baseline(path_to_txt, id_offset = 0):
    # load raw lines
    with open(path_to_txt, 'r') as f:
        texts = f.read()
        texts = texts.split('\n\n')
        texts = [[line.split() for line in text.split('\n')] for text in texts]
        texts = [text for text in texts if sum([len(line) != 0 for line in text]) == len(text)]
        
    # get ids, tokens and labels
    ids = [i + id_offset for i in range(len(texts))]
    tokens = [[l[0] for l in text] for i, text in enumerate(texts)]
    labels = [[l[-1] for l in text] for i, text in enumerate(texts)]
    
    # get dataset object
    data = {'ids': ids, 'tokens': tokens, 'ner_tags': labels}
    return data


def dicts_to_jsonl(data_list: list, filename: str, compress: bool = True) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. either endw with .jsonl or with .jsonl.gz
    :param compress: (bool) should file be compressed into a gzip archive
    """
    if compress:
        assert filename.endswith('.jsonl.gz'), 'When "compress" is set to True, parameter "filename" must end with ".jsonl.gz"'
        with gzip.open(filename, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        assert filename.endswith('.jsonl'), 'When "compress" is set to False, parameter "filename" must end with ".jsonl"'
        with open(filename, 'w') as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)
    return
