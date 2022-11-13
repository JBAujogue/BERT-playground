import re

# for manipulating data 
import pandas as pd



def generate_bio_spans(df_sents, df_ents):
    columns = ['Id', 'Sentence_id', 'Entity_id', 'Mention', 'Start_char', 'End_char', 'Category']
    df_ents = df_ents[columns].drop_duplicates(ignore_index = True)
    all_ents = []
    for _id, sent_id, text in df_sents.values.tolist():
        df_tmp = df_ents[(df_ents.Id == _id) & (df_ents.Sentence_id == sent_id)]
        starts = [0] + df_tmp.End_char.tolist()
        ends = df_tmp.Start_char.tolist() + [len(text)]
        for i, (start, end) in enumerate(zip(starts, ends)):

            # add negative mentions, split by linebreaks
            mention = text[start: end]
            neg_ent = [_id, sent_id, -1, mention, start, end, 'O']
            all_ents.append(neg_ent)

            # add next positive entity
            if i<len(df_tmp):
                ent = df_tmp.iloc[i].tolist()
                all_ents.append(ent)

    df_bio_spans = pd.DataFrame(all_ents, columns = columns)
    return df_bio_spans