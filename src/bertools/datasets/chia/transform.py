import copy
import pandas as pd


def append_lines_to_spans(df_lines: pd.DataFrame, df_spans: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve the correct line_id of each span.
    """
    # compute start and end offset of each line wrt its underlying text
    line_lengths = df_lines['content'].str.len()
    
    df_lines_cp = copy.deepcopy(df_lines)
    df_lines_cp.loc[:, 'start'] = df_lines_cp.apply(
        func = lambda line: (line_lengths
            [
                  (df_lines_cp['text_id'] == line['text_id']) 
                & (df_lines_cp['line_id'] < line['line_id'])
            ]
            .sum()
        ),
        axis = 1,
    )
    df_lines_cp.loc[:, 'end'] = df_lines_cp['start'] + line_lengths
    
    # retrieve the list of spans included in each line
    df_lines_cp.loc[:, 'spans'] = df_lines_cp.apply(
        func = lambda line: (df_spans
            [
                  (df_spans['text_id'] == line['text_id'])
                & (df_spans['start'] >= line['start'])
                & (df_spans['end'] <= line['end'])
            ]
            .to_dict('records')            
        ),
        axis = 1,
    )
    # replace empty lists of span by a default blank span
    df_lines_cp.loc[:, 'spans'] = df_lines_cp.apply(
        func = lambda r: r['spans'] or [
            {
                'text_id': r['text_id'], 
                'span_id': '', 
                'start': 0, 
                'end': 0, 
                'label': '',
            },
        ],
        axis = 1,
    )
    # duplicate lines for each associated span
    # adjust span offsets to be wrt the line and not the text
    fusion = [
        line | span | {'start': span['start'] - line['start'], 'end': span['end'] - line['start']}
        for line in df_lines_cp.to_dict('records') 
        for span in line.pop('spans')
    ]
    return pd.DataFrame.from_records(fusion)


def flatten_spans(df_spans: pd.DataFrame) -> pd.DataFrame:
    """
    Transform multi-span entities into single-span entities.
    """
    df_spans['start'] = df_spans.groupby(['text_id', 'span_id'])['start'].transform('min')
    df_spans['end'] = df_spans.groupby(['text_id', 'span_id'])['end'].transform('max')
    df_spans = df_spans.drop_duplicates(ignore_index = True)

    # update text to account for change of offsets
    return update_span_text(df_spans)


def update_span_text(df_spans: pd.DataFrame) -> pd.DataFrame:
    """
    Replace text of a span by the actual text covered by offsets.
    """
    df_spans.loc[:, 'text'] = df_spans.apply(
        func = lambda r: r['content'][r['start']: r['end']],
        axis = 1,
    )
    return df_spans


def drop_overlapped_spans(df_spans: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overlap between entities, and:
    - retain only maximal entities, e.g. entities not overlapping any larger entity.
    - map ids of any overlapped entity towards its maximal overlapping entity.
    """
    # compute overlaping child and parent spans
    overlaps = df_spans.apply(
        func = lambda row: get_overlapping_spans(row, df_spans),
        axis = 1,
    ).tolist()
    
    # append ids of child spans
    df_spans.loc[:, 'child_ids'] = [o['child_ids'] for o in overlaps]

    # only retain maximal spans
    df_spans = df_spans.loc[[len(o['parent_ids']) == 0 for o in overlaps]]
    return df_spans.reset_index(drop = True)


def get_overlapping_spans(span, df: pd.DataFrame) -> dict[str, tuple[int,...]]:
    """
    Given an input DataFrame row, extracts row indices of overlapping spans
    found in param df.
    """
    # get dataframe of spans appearning in same text
    df = df.loc[(df['text_id'] == span['text_id']) & (df['line_id'] == span['line_id'])]

    lengths = df['end'] - df['start']
    length = span['end'] - span['start']
    overlaps = (df['end'] >= span['start']) & (df['start'] < span['end'])
    
    # get ids for longer overlaping spans
    greaters = (lengths > length) | ((lengths == length) & (df['start'] < span['start']))
    parent_ids = df.loc[overlaps & greaters, 'span_id'].unique().tolist()

    # get ids of shorter overlaping spans
    shorters = ~greaters
    child_ids = df.loc[overlaps & shorters, 'span_id'].unique().tolist()
    return {'child_ids': tuple(child_ids), 'parent_ids': tuple(parent_ids)}
