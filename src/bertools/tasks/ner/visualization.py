import pandas as pd

from spacy import displacy


ENT2COLOR = {
    # red
    'Condition': '#ff6e70', 
    
    # green
    'Drug': '#50ebb2', 
    'Device': '#50ebb2',
    
    # yellow
    'Measurement': '#fffad4',

    # blue
    'Procedure' : '#84bee8', 
    
    # dark blue
    'Person': '#377aab', 
    
    # pink
    'Visit': '#de1260', 
    
    # purple
    'Value': '#de97f7',
    'Temporal': '#de97f7',
    'Observation': '#de97f7',
    'Qualifier': '#de97f7',

    # shiny blue,
    'Negation': '#97f7f2',
}


def render_ner_as_html(
    text: str, 
    df_entities: pd.DataFrame, 
    ent2color: dict = ENT2COLOR,
    column_start_char: str = 'Start_char',
    column_end_char: str = 'End_char',
    column_cat: str = 'Category',
    ):
    '''
    A simple wrap of spacy.displacy.render function in order
    to highlight a collection of entities in an input text.
    The collection of entities should be a DataFrame with (at least) 
    3 columns (here names are default values of dedicated input parameters) :
        - Entity : the entity's category
        - Start_char : The index of the first character of the span
        - End_char : the index of the first character following the span
     
    Parameters
    ----------
    text: str.
    The text on which to highlight mentions.
    
    df_entities: pandas.DataFrame.
    A dataframe describing a set of entities to highlight in the input text.
    
    ent2color: dict, default = ent2color.
    A dictionary mapping a set of category names to colors as hex color code.
    
    column_start_char: str, default = 'Start_char'.
    The name of column corresponding to index of the first character of each 
    span in df_entities.
    
    column_end_char: str, default = 'End_char'.
    The name of column corresponding to index of the first character following 
    each span in df_entities.
    
    column_entity: str, default = 'Entity'.
    The name of column containing the category name of each span in 
    df_entities.
    
    Returns
    -------
    html: str.
    The inputy text as html syntax, with html tags surrounding each entity 
    to highlight.
    '''
    # get entities into list of dicts
    ent_list = df_entities.rename(columns = {
        column_start_char : 'start', 
        column_end_char : 'end', 
        column_cat : 'label',
    }).to_dict('records')
    
    # wrap entities into dict
    ent_dict = {"text": text, "ents": ent_list, "title": None}
    
    # compute html
    return displacy.render(
        ent_dict, 
        style = "ent", 
        manual = True, 
        jupyter = False, 
        options = {'colors': ent2color},
    )
