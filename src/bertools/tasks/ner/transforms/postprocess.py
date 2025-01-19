import copy
import re
import string

import pandas as pd


def parse_trf_ner_output(ent_dict: dict):
    df_ents = pd.DataFrame(ent_dict)
    if not df_ents.empty:
        df_ents = df_ents.rename({
            'word': 'Mention',
            'entity_group' : 'Category',
            'start': 'Start_char',
            'end': 'End_char',
        }, axis = 'columns')
        df_ents = df_ents[['Mention', 'Category', 'Start_char', 'End_char']]
        df_ents = df_ents.sort_values(by = 'Start_char').reset_index(drop = True)
    else:
        df_ents = pd.DataFrame(columns = ['Mention', 'Category',  'Start_char', 'End_char'])
    return df_ents


def correct_entity_boundaries(text, df_ents):
    '''
    Modify the start and end character indices of each entity, by:
    - adding the beginning of a word when missing in the entity,
    - removing symbols (commas, dots) lying at the boundary of each entity
    
    Parameters
    ----------
    text : string.
    The text on which entities where found.
    
    df_ents: pd.DataFrame.
    A dataframe with rows being medical mentions, described by columns 
    containing (at least):
        - Mention : The raw mention
        - Category : The category of the mention
        - Start_char : The mention start index when captured from a medical text
        - End_char : The mention end index when captured from a medical text
    
    Returns
    -------
    df_ents: pd.DataFrame.
    A dataframe with rows being medical mentions, with columns
    "Text", "Start_char" and "End_char" corrected.
    '''
    def add_alphanumeric_prefix(text, start_char):
        # find, for each entity of the dataframe, the greatest prefix entirely
        # consisting of letters, and add it to the entity
        left_text = text[:start_char]
        prefix = re.sub('[^a-zA-Z0-9]', ' ', left_text).split(' ')[-1]
        return start_char - len(prefix)

    def add_alphanumeric_suffix(text, end_char):
        # find, for each entity of the dataframe, the greatest suffix entirely
        # consisting of letters, and add it to the entity
        right_text = text[end_char:]
        suffix = re.sub('[^a-zA-Z0-9]', ' ', right_text).split(' ')[0]
        return end_char + len(suffix)
    
    def drop_punct_prefix(text, chart_char, end_char):
        keep_punct = '<=>+-'
        if ']' in text[:end_char]:
            keep_punct += '['
        if ')' in text[:end_char]:
            keep_punct += '('
        if '}' in text[:end_char]:
            keep_punct += '{'
        return len(text) - len(text[chart_char:].lstrip(''.join(set(string.punctuation) - set(keep_punct)))) # string.punctuation

    def drop_punct_suffix(text, start_char, end_char):
        keep_punct = '%'
        if '[' in text[start_char:]:
            keep_punct += ']'
        if '(' in text[start_char:]:
            keep_punct += ')'
        if '{' in text[start_char:]:
            keep_punct += '}'
        return len(text[:end_char].rstrip(''.join(set(string.punctuation) - set(keep_punct)))) # string.punctuation
    
    if not df_ents.empty:
        df_ents = copy.deepcopy(df_ents)

        # fix prefix
        df_ents.Start_char = df_ents.Start_char.apply(lambda start_char: add_alphanumeric_prefix(text, start_char))
        df_ents.Start_char = df_ents.apply(
            func = lambda r: drop_punct_prefix(text, r.Start_char, r.End_char),
            axis = 1,
        )
        # fix suffix
        df_ents.End_char = df_ents.End_char.apply(lambda end_char: add_alphanumeric_suffix(text, end_char))
        df_ents.End_char = df_ents.apply(
            func = lambda r: drop_punct_suffix(text, r.Start_char, r.End_char),
            axis = 1,
        )        
        # update entities text
        df_ents.Mention = df_ents.apply(
            func = lambda row: text[row.Start_char: row.End_char],
            axis = 1,
        )
        df_ents = df_ents.sort_values(by = 'Start_char', ignore_index = True)
    return df_ents


def remove_entity_overlaps(df_ents):
    '''
    Resolves overlap issues between medical entities. 
    Overlap resolution is based on the following columns of the input dataframe :
        - Start_char : The mention start index when captured from a medical text
        - End_char : The index next to the mention's end when captured from a medical text
    The heuristic is to process entities, in case two entities overlap, to keep the longuest,
    and in case of equality keep the first occurence.
        
    Parameters
    ----------
    df_ents: pandas.DataFrame.
    A dataframe with rows being medical mentions, described by (at least)
    the two columns listed above.
    
    Returns
    -------
    df_ents: pandas.DataFrame.
    A dataframe with rows being medical mentions, without any overlap on the
    spans occupied by each medical mention.
    '''
    df_ents = copy.deepcopy(df_ents)
    df_ents = df_ents.drop_duplicates(ignore_index = True)

    # sort by descending length and ascending order of appearance
    df_ents['Length'] = df_ents.End_char - df_ents.Start_char
    df_ents = df_ents.sort_values(by = 'Start_char', ignore_index = True)
    df_ents = df_ents.sort_values(by = 'Length', ascending = False, ignore_index = True)

    df_ents['Keep'] = True
    for i in range(len(df_ents)):
        # get current entity
        row = df_ents.iloc[i]

        if row['Keep'] == True:
            # get all subsequent entities
            sub = df_ents.iloc[i+1:, :]
            sub = sub[sub['Keep'] == True]

            # get subsequent entities that overlap the current entity
            filter_start1 = (row.Start_char <= sub.Start_char)
            filter_start2 = (sub.Start_char < row.End_char)
            filter_end1 = (row.Start_char < sub.End_char)
            filter_end2 = (sub.End_char <= row.End_char)
            filter_overlap = ((filter_start1 & filter_start2) | (filter_end1 & filter_end2))
            sub = sub[filter_overlap]

            # if some overlap is detected
            if not sub.empty:
                sub_idx = sub.index.tolist()
                for j in sub_idx:
                    df_ents.at[j, 'Keep'] = False

    df_ents = df_ents[df_ents.Keep == True]
    df_ents = df_ents.drop(columns = ['Keep', 'Length']).reset_index(drop = True)
    df_ents = df_ents.sort_values(by = 'Start_char', ignore_index = True)
    return df_ents
