import json
import zipfile
import re
import gzip

# data
import pandas as pd





#**********************************************************
#*                         Texts                          *
#**********************************************************

def load_texts_from_zipfile(zip_file):
    archive = zipfile.ZipFile(zip_file, 'r')
    text_files = [f for f in archive.namelist() if f.endswith('.txt')]
    texts = []
    for text_file in text_files:
        with archive.open(text_file, 'r') as f:
            _id = text_file.split('/')[-1][:-4]
            text = [l.decode('utf-8') for l in f.readlines()]
            if _id in ['NCT02348918_exc', 'NCT02348918_inc', 'NCT01735955_exc']:
                text = '\n'.join([i.strip() for i in text])
            else:
                text = ' \n'.join([i.strip() for i in text])
            text = text.replace('⁄', '/')
            texts.append([_id, text])
            
    df_texts = pd.DataFrame(texts, columns = ['Id', 'Text'])
    return df_texts





#**********************************************************
#*                          NER                           *
#**********************************************************

def load_entities_from_zipfile(zip_file):
    archive = zipfile.ZipFile(zip_file, 'r')
    ann_files = [f for f in archive.namelist() if f.endswith('.ann')]
    ent_list = []
    for ann_file in ann_files:
        with archive.open(ann_file, 'r') as f:
            lines = [l.decode('utf-8').replace('⁄', '/').strip() for l in f.readlines()]
            ents = [l.split('\t') for l in lines if l.startswith('T')]
            ents = [
                [ann_file.split('/')[-1][:-4], ent[0], ent[2], ent[1]]
                for ent in ents
            ]
            ents = [
                ent[:-1] + [
                    ent[-1].replace(';', ' ').split(' ')[0], 
                    tuple([int(v) for v in ent[-1].replace(';', ' ').split(' ')[1:]]),
                ]
                for ent in ents
            ]
            ents = [
                ent[:-1] + [min(ent[-1]), max(ent[-1]), ent[-1]]
                for ent in ents
            ]
            ent_list += ents
            
    df_ents = pd.DataFrame(ent_list, columns = [
        'Id', 'Entity_id', 'Mention', 'Category', 'Start_char', 'End_char', 'Char_spans',
    ])
    return df_ents



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
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    sgz = '.gz'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl

    # Save data
    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w') as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)
    return
