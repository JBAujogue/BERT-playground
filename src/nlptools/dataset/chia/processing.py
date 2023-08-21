import copy
import re
from typing import List, Tuple, Iterable, Callable, Optional

# data
import pandas as pd

# **********************************************************
# *                          NER                           *
# **********************************************************


def flatten_entities(df_texts: pd.DataFrame, df_ents: pd.DataFrame) -> pd.DataFrame:
    '''
    Replace multi-span entities by single-span entities.

    :param pd.DataFrame df_text: Dataframe with 'Id' and 'Text' columns
    :param pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    :return pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    '''
    df_ents = copy.deepcopy(df_ents)
    id2text = {k: v for k, v in df_texts.values.tolist()}

    # compute flat mention spans
    df_ents["Mention"] = df_ents.apply(
        func=lambda row: id2text[row.Id][row.Start_char : row.End_char],
        axis=1,
    )
    df_ents = df_ents[
        ["Id", "Entity_id", "Mention", "Category", "Start_char", "End_char"]
    ]
    df_ents = df_ents.drop_duplicates(ignore_index=True)

    # group by unique mention
    df_ents = (
        df_ents.groupby(["Id", "Mention", "Start_char", "End_char"])
        .apply(lambda g: (g.Entity_id.tolist(), g.Category.tolist()[0]))
        .reset_index()
        .rename(columns={0: "Tmp"})
    )
    # merge entity ids
    df_ents["Entity_id"] = df_ents.Tmp.apply(lambda t: tuple(t[0]))

    # remove duplicate Categories
    df_ents["Category"] = df_ents.Tmp.apply(lambda t: t[1])

    # clean result
    df_ents = df_ents.drop(columns=["Tmp"])
    df_ents = df_ents.sort_values(by=["Id", "Start_char"], ignore_index=True)
    return df_ents


def get_overlapping_entities(row, df: pd.DataFrame) -> Tuple[List[int]]:
    '''
    Given an input DataFrame row, extracts ids of overlapping entities
    found in param df.
    
    :param row: DataFrame row with columns 'Id', 'Entity_id', 'Start_char', 'End_char'.
    :param df pd.DataFrame: DataFrame with columns 'Id', 'Entity_id', 'Start_char', 'End_char'.
    :return tuple: tuple (idx_short, idx_long), with idx_short (resp. idx_long) containing the 
        list of ids of stricty shorter (resp. longer) entities from "df" param.
    '''
    # get entities on same Id
    df = df[df.Id == row.Id]
    df = df[df.Entity_id != row.Entity_id]

    lengths = df.End_char - df.Start_char
    length = row.End_char - row.Start_char

    # get longer overlaping entities
    df_long = df[lengths > length]
    long = df_long.apply(
        func=lambda r: len(
            set(range(r.Start_char, r.End_char))
            & set(range(row.Start_char, row.End_char))
        )
        > 0,
        axis=1,
    )
    idx_long = df_long[long].index.tolist()

    # get shorter overlaping entities
    df_short = df[lengths < length]
    short = df_short.apply(
        func=lambda r: len(
            set(range(r.Start_char, r.End_char))
            & set(range(row.Start_char, row.End_char))
        )
        > 0,
        axis=1,
    )
    idx_short = df_short[short].index.tolist()
    return (idx_short, idx_long)


def map_entity_id_to_parent(df_ents: pd.DataFrame, short_overlaps: Iterable) -> pd.DataFrame:
    '''
    Complete the list of ids of an entity by ids associated to overlapped entities.
    
    :param pd.DataFrame df_ents: DataFrame of entities with column 'Entity_id'. 
    :param Iterable short_overlaps: list of tuples of same length than "df_ents" param,
        the i'th tuple consisting of of indices of entities overlapped by the i'th entity.
    :return pd.DataFrame df_ents: DataFrame of entities with column 'Entity_id'.
    '''
    df_ents = copy.deepcopy(df_ents)
    df_ents["Child"] = short_overlaps

    df_ents.Entity_id = df_ents.apply(
        func=lambda row: row.Entity_id
        + tuple(_id for ids in df_ents.loc[row.Child].Entity_id for _id in ids),
        axis=1,
    )
    df_ents = df_ents.drop(columns="Child")
    return df_ents


def solve_overlapping_entities(df_ents: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute overlap between entities, and:
        - retain only maximal entities, e.g. entities not overlapping any larger entity.
        - map ids of any overlapped entity towards its maximal overlapping entity.
    
    :param pd.DataFrame df_ents: DataFrame of entities with columns 
        'Id', 'Entity_id', 'Start_char', 'End_char'.
    :return pd.DataFrame df_ents: DataFrame of entities with columns 
        'Id', 'Entity_id', 'Start_char', 'End_char'.
    '''
    overlaps = df_ents.apply(
        func=lambda row: get_overlapping_entities(row, df_ents),
        axis=1,
    ).tolist()
    short_overlaps = [o[0] for o in overlaps]
    long_overlaps = [o[1] for o in overlaps]

    df_ents = map_entity_id_to_parent(df_ents, short_overlaps)
    df_ents = df_ents[[len(c) == 0 for c in long_overlaps]]
    return df_ents


def extract_flat_entities(df_texts: pd.DataFrame, df_ents: pd.DataFrame, categories: list) -> pd.DataFrame:
    '''
    Transform multi-span entities into single-span entities, and:
        - retain only maximal entities, e.g. entities not overlapping any larger entity.
        - map ids of any overlapped entity towards its maximal overlapping entity.
    
    :param pd.DataFrame df_text: Dataframe with 'Id' and 'Text' columns
    :param pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    :param list categories: List of categories to retain from "df_ents" param.
    :return pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    '''
    df_ents = df_ents[df_ents.Category.isin(categories)].reset_index(drop=True)
    df_ents = flatten_entities(df_texts, df_ents)
    df_ents = solve_overlapping_entities(df_ents)
    df_ents = df_ents.reset_index(drop=True)
    return df_ents


def convert_to_bio(df_texts: pd.DataFrame, df_ents: pd.DataFrame, tokenizer: Optional[Callable] = None) -> pd.DataFrame:
    '''
    Convert entities into a dataframe of spans, each span being either a full entity,
    or a full inter-entity span of text.
    If "tokenizer" param is not None, spans are subsequently tokenized, and in-entity tokens
    are labelled using a BIO tagging scheme.
    
    :param pd.DataFrame df_text: Dataframe with 'Id' and 'Text' columns
    :param pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    :param tokenizer: Callable that converts an input string into a list of token strings.
    :param pd.DataFrame df_bio: Dataframe of spans with columns
            "Id", "Sequence_id", "Mention", "Category".
    '''
    id2text = {k: v for k, v in df_texts.values.tolist()}
    all_ents = []
    for _id, text in id2text.items():
        sent_id = 0
        df_tmp = df_ents[df_ents.Id == _id]
        starts = [0] + df_tmp.End_char.tolist()
        ends = df_tmp.Start_char.tolist() + [len(text)]
        for i, (start, end) in enumerate(zip(starts, ends)):

            # add negative mentions, with spliting into sentences by linebreaks
            mention = text[start:end]
            for j, m in enumerate(mention.split("\n")):
                sent_id += j
                if m:
                    if tokenizer:
                        neg_ents = [
                            (_id, str(_id) + "_" + str(sent_id), t, "O")
                            for t in tokenizer(m)
                        ]
                        all_ents += neg_ents
                    else:
                        neg_ent = (_id, str(_id) + "_" + str(sent_id), m, "O")
                        all_ents.append(neg_ent)

            # add next positive entity
            if i < len(df_tmp):
                ent = df_tmp.iloc[i]
                ent_mention, ent_cat = ent.Mention, ent.Category
                if tokenizer:
                    tokens = tokenizer(ent_mention)
                    labels = ["B-" + ent_cat] + ["I-" + ent_cat] * (len(tokens) - 1)
                    ents = [
                        (_id, str(_id) + "_" + str(sent_id), t, l)
                        for t, l in zip(tokens, labels)
                    ]
                    all_ents += ents
                else:
                    ent = (_id, str(_id) + "_" + str(sent_id), ent_mention, ent_cat)
                    all_ents.append(ent)

    df_bio = pd.DataFrame(
        all_ents,
        columns=["Id", "Sequence_id", "Mention", "Category"],
    )
    # remove blank spans
    df_bio = df_bio[df_bio.Mention.apply(lambda s: re.sub("\n+", "", s) != "")]
    return df_bio


def complete_relations_with_entity_data(df_ents: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    '''
    Add to relationships some metadata describing source and target objects.
    
    :param pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    :param pd.DataFrame df_rels: Dataframe of relationships with columns
            "Id", "Source_id", "Target_id".
    :return pd.DataFrame: Dataframe of relationships with columns
            "Id", "Source_id", "Target_id",         
            "Source_mention", "Source_start_char", "Source_end_char", "Source_category",
            "Target_mention", "Target_start_char", "Target_end_char", "Target_category",
    '''
    def retrieve_relation_data(r, df_ents: pd.DataFrame) -> List:
        '''
        Given "Id", "Source_id", "Target_id" from "r" param,
        returns from "df_ents" param
            "Source_mention", "Source_start_char", "Source_end_char", "Source_category",
            "Target_mention", "Target_start_char", "Target_end_char", "Target_category",
        '''
        src_info = df_ents[df_ents.Entity_id.apply(lambda ids: r.Source_id in ids)][
            ["Mention", "Start_char", "End_char", "Category"]
        ].values.tolist()
        tar_info = df_ents[df_ents.Entity_id.apply(lambda ids: r.Target_id in ids)][
            ["Mention", "Start_char", "End_char", "Category"]
        ].values.tolist()

        # some source & target entity may no longer exist in df_ents due to previous filtering
        if src_info and tar_info:
            return [r.tolist() + s + t for s in src_info for t in tar_info]
        return []

    infos = df_rels.apply(
        lambda r: retrieve_relation_data(r, df_ents[df_ents.Id == r.Id]), axis=1
    ).tolist()
    infos = [rst for rsts in infos for rst in rsts]
    info_cols = df_rels.columns.tolist() + [
        "Source_mention",
        "Source_start_char",
        "Source_end_char",
        "Source_category",
        "Target_mention",
        "Target_start_char",
        "Target_end_char",
        "Target_category",
    ]
    return pd.DataFrame(infos, columns=info_cols)


def compute_separating_space(df_texts: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute the text separating the source and target entity of all relationships
    contained in "df_rels" param, and append it to a new "Separating_mention" column
    of "df_rels".
    
    :param pd.DataFrame df_texts: Dataframe of texts with columns "Id", "Text".
    :param pd.DataFrame df_rels: Dataframe of relationships with columns
            "Id", "Source_start_char", "Target_end_char".
    :return pd.DataFrame: Dataframe of relationships with columns
            "Id", "Source_start_char", "Target_end_char", "Separating_mention".
    '''
    def compute_separating_text(r, id2text):
        return id2text[r.Id][r.Target_end_char : r.Source_start_char]

    id2text = {i: t for i, t in df_texts.values.tolist()}
    seps = df_rels.apply(lambda r: compute_separating_text(r, id2text), axis=1)
    return pd.concat((df_rels, seps.rename("Separating_mention")), axis=1)


def select_qualifier_relations(df_texts: pd.DataFrame, df_ents: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    '''
    Extract from the "df_rels" param a subset of relationships with target being a Qualifier entity, 
    along with additional conditions to fulfill.
    
    :param pd.DataFrame df_texts: Dataframe of texts with columns "Id", "Text".
    :param pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    :param pd.DataFrame df_rels: Dataframe of relationships with columns
            "Id", "Source_start_char", "Target_end_char".
    :return pd.DataFrame df_rels_infos: Dataframe of relationships with columns
            "Id", "Source_id", "Target_id",         
            "Source_mention", "Source_start_char", "Source_end_char", "Source_category",
            "Target_mention", "Target_start_char", "Target_end_char", "Target_category",
            "Separating_mention".
    '''
    df_rels_infos = complete_relations_with_entity_data(df_ents, df_rels)

    # Select the relationships we want to merge, see df_rels_infos.Relation.unique()
    df_rels_infos = df_rels_infos[df_rels_infos.Relation.isin(["Has_qualifier"])]

    # only keep qualifiers placed before domain entity
    df_rels_infos = df_rels_infos[
        df_rels_infos.Target_end_char <= df_rels_infos.Source_start_char
    ]

    # add separating text between src and tar entities
    df_rels_infos = compute_separating_space(df_texts, df_rels_infos)

    # keep Qualifiers only !
    df_rels_infos = df_rels_infos[df_rels_infos.Target_category == "Qualifier"]

    # merge qualifiers with separation text obeying some strict rules
    # after carefully inspecting <= 3 characters long separators,
    # we chose spacings only as allowed separating texts (no letter, no dot, no coma, no linebreack, no parenthesis)
    df_rels_infos = df_rels_infos[df_rels_infos.Separating_mention.isin(["", " "])]

    # merge qualifiers to Conditions only
    df_rels_infos = df_rels_infos[df_rels_infos.Source_category == "Condition"]

    # forget about qualifiers containing a 'or', as it mislead ner afterward
    df_rels_infos = df_rels_infos[
        df_rels_infos.Target_mention.apply(lambda t: " or " not in t)
    ]

    df_rels_infos = df_rels_infos.drop_duplicates(
        subset=["Id", "Source_id"], ignore_index=True
    )
    return df_rels_infos


def merge_qualifiers(df_texts: pd.DataFrame, df_ents: pd.DataFrame, df_rels: pd.DataFrame) -> pd.DataFrame:
    '''
    Extend entities found in "df_ents" param whenever they are linked to a
    Qualifier entity using relationships found in "df_rels" param, provided 
    ceetains conditions on the relationship are fulfilled.
    
    :param pd.DataFrame df_texts: Dataframe of texts with columns "Id", "Text".
    :param pd.DataFrame df_ents: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    :param pd.DataFrame df_rels: Dataframe of relationships with columns
            "Id", "Source_start_char", "Target_end_char".
    :return pd.DataFrame: Dataframe of entities with columns
            "Id", "Entity_id", "Mention", "Category", "Start_char", "End_char".
    '''
    def get_least_start_char_extension(r, df_rels_infos):
        return min(
            [r.Start_char]
            + df_rels_infos[
                (df_rels_infos.Id == r.Id) & (df_rels_infos.Source_id.isin(r.Entity_id))
            ].Target_start_char.tolist()
        )

    df_rels_infos = select_qualifier_relations(df_texts, df_ents, df_rels)
    id2text = {k: v for k, v in df_texts.values.tolist()}
    df_ents = df_ents[df_ents.Category != "Qualifier"].copy()
    df_ents["Start_char"] = df_ents.apply(
        lambda r: get_least_start_char_extension(r, df_rels_infos), axis=1
    )
    df_ents["Mention"] = df_ents.apply(
        lambda r: id2text[r.Id][r.Start_char : r.End_char], axis=1
    )
    return df_ents.reset_index(drop=True)


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





#**********************************************************
#*                      Causal LM                         *
#**********************************************************

def convert_to_prompts(df_texts, df_ents, bos_term = '', sep_term = '\n\n###\n\n', end_term = '\n\nEND'):
    df_spans = convert_to_bio(df_texts, df_ents)
    return df_spans.groupby('Sequence_id').apply(
        lambda g: {
            'id': g.Sequence_id.iat[0],
            'prompt': bos_term + ''.join(g.Mention.tolist()) + sep_term,
            'completion': '\n\n'.join(['\n'.join((text, label)) for text, label in g[g.Category != 'O'][['Mention', 'Category']].values.tolist()]) + end_term,
        }
    ).tolist()



