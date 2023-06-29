import copy
import re

# data
import pandas as pd






#**********************************************************
#*                          NER                           *
#**********************************************************

def flatten_entities(df_texts, df_ents):
    df_ents = copy.deepcopy(df_ents)
    id2text = {k: v for k, v in df_texts.values.tolist()}
    
    # compute flat mention spans
    df_ents['Mention'] = df_ents.apply(
        func = lambda row: id2text[row.Id][row.Start_char: row.End_char],
        axis = 1,
    )
    df_ents = df_ents[[
        'Id', 'Entity_id', 'Mention', 'Category', 'Start_char', 'End_char',
    ]]
    df_ents = df_ents.drop_duplicates(ignore_index = True)

    # group by unique mention
    df_ents = df_ents.groupby(['Id', 'Mention', 'Start_char', 'End_char'])\
       .apply(lambda g: (g.Entity_id.tolist(), g.Category.tolist()[0]))\
       .reset_index()\
       .rename(columns = {0: 'Tmp'})
    
    # merge entity ids
    df_ents['Entity_id'] = df_ents.Tmp.apply(lambda t: tuple(t[0]))
    
    # remove duplicate Categories
    df_ents['Category'] = df_ents.Tmp.apply(lambda t: t[1])
    
    # clean result
    df_ents = df_ents.drop(columns = ['Tmp'])
    df_ents = df_ents.sort_values(by = ['Id', 'Start_char'], ignore_index = True)
    return df_ents



def get_overlaping_entities(row, df):
    # get entities on same Id
    df = df[df.Id == row.Id]
    df = df[df.Entity_id != row.Entity_id]
    
    lengths = df.End_char - df.Start_char
    length = row.End_char - row.Start_char
    
    # get longer overlaping entities
    df_long = df[lengths > length]
    long = df_long.apply(
        func = lambda r: len(set(range(r.Start_char, r.End_char)) & set(range(row.Start_char, row.End_char)))>0, 
        axis = 1,
    )
    idx_long = df_long[long].index.tolist()
    
    # get shorter overlaping entities
    df_short = df[lengths < length]
    short = df_short.apply(
        func = lambda r: len(set(range(r.Start_char, r.End_char)) & set(range(row.Start_char, row.End_char)))>0, 
        axis = 1,
    )
    idx_short = df_short[short].index.tolist()
    return (idx_short, idx_long)



def map_entity_id_to_parent(df_ents, short_overlaps):
    df_ents = copy.deepcopy(df_ents)
    df_ents['Child'] = short_overlaps
    
    df_ents.Entity_id = df_ents.apply(
        func = lambda row: row.Entity_id + tuple(_id for ids in df_ents.loc[row.Child].Entity_id for _id in ids),
        axis = 1,
    )
    df_ents = df_ents.drop(columns = 'Child')
    return df_ents



def solve_overlaping_entities(df_ents):
    overlaps = df_ents.apply(
        func = lambda row: get_overlaping_entities(row, df_ents),
        axis = 1,
    ).tolist()
    short_overlaps = [o[0] for o in overlaps]
    long_overlaps  = [o[1] for o in overlaps]
    
    df_ents = map_entity_id_to_parent(df_ents, short_overlaps)
    df_ents = df_ents[[len(c) == 0 for c in long_overlaps]]
    return df_ents



def get_ner_entities(df_texts, df_ents, categories):
    df_ents = df_ents[df_ents.Category.isin(categories)].reset_index(drop = True)
    df_ents = flatten_entities(df_texts, df_ents)
    df_ents = solve_overlaping_entities(df_ents)
    df_ents = df_ents.reset_index(drop = True)
    return df_ents



def convert_to_bio(df_texts, df_ents, tokenizer = None):
    id2text  = {k: v for k, v in df_texts.values.tolist()}
    all_ents = []
    for _id, text in id2text.items():
        sent_id = 0
        df_tmp = df_ents[df_ents.Id == _id]
        starts = [0] + df_tmp.End_char.tolist()
        ends = df_tmp.Start_char.tolist() + [len(text)]
        for i, (start, end) in enumerate(zip(starts, ends)):

            # add negative mentions, with spliting into sentences by linebreaks
            mention = text[start: end]
            for j, m in enumerate(mention.split('\n')):
                sent_id += j
                if m:
                    if tokenizer:
                        neg_ents = [(_id, str(_id)+'_'+str(sent_id), t, 'O') for t in tokenizer(m)]
                        all_ents += neg_ents
                    else:
                        neg_ent = (_id, str(_id)+'_'+str(sent_id), m, 'O')
                        all_ents.append(neg_ent)

            # add next positive entity
            if i<len(df_tmp):
                ent = df_tmp.iloc[i]
                ent_mention, ent_cat = ent.Mention, ent.Category
                if tokenizer:
                    tokens = tokenizer(ent_mention)
                    labels = ['B-' + ent_cat] + ['I-' + ent_cat] * (len(tokens)-1)
                    ents = [(_id, str(_id)+'_'+str(sent_id), t, l) for t, l in zip(tokens, labels)]
                    all_ents += ents
                else:
                    ent = (_id, str(_id)+'_'+str(sent_id), ent_mention, ent_cat)
                    all_ents.append(ent)

    df_bio = pd.DataFrame(
        all_ents, 
        columns = ['Id', 'Sequence_id', 'Mention', 'Category'],
    )
    # remove blank spans
    df_bio = df_bio[df_bio.Mention.apply(lambda s: re.sub('\n+', '', s) != '')]
    return df_bio



def convert_to_prompts(df_texts, df_ents, bos_term = '', sep_term = '\n\n###\n\n', end_term = '\n\nEND'):
    df_spans = convert_to_bio(df_texts, df_ents)
    return df_spans.groupby('Sequence_id').apply(
        lambda g: {
            'id': g.Sequence_id.iat[0],
            'prompt': bos_term + ''.join(g.Mention.tolist()) + sep_term,
            'completion': '\n\n'.join(['\n'.join((text, label)) for text, label in g[g.Category != 'O'][['Mention', 'Category']].values.tolist()]) + end_term,
        }
    ).tolist()



def switch_B_to_I(idx, mapping_dict):
    return mapping_dict[idx] if idx in mapping_dict else idx



def tokenize_and_align_categories(tokenizer, examples, B_I_mapping):
    # tokenize into a BatchEncoding instance
    tokenized_inputs = tokenizer(examples["mentions"], truncation = True, is_split_into_words = True)

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
