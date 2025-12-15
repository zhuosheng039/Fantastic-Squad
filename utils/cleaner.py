import numpy as np
import pandas as pd
from .common import DF, MF, FW, ATTRS, STATS


def simplify_position(pos):
    """Map many EA position strings into DF/MF/FW."""
    p = str(pos).upper()
    result = set()
    if 'GK' in p:
        result.add('GK')
    # defenders
    if any(x in p for x in DF):
        result.add('DF')
    # midfield
    if any(x in p for x in MF):
        result.add('MF')
    # forwards
    if any(x in p for x in FW):
        result.add('FW')
    return result

def combine_positions(row):
    """Combine Position string and Alternative positions list/NaN into a single set"""
    full = set()
    if isinstance(row['Position'], str):
        full.add(row['Position'])
    alt = row['Alternative positions']
    if isinstance(alt, str):
        alt_list = eval(alt)
        full.update(alt_list)
    return full

def clean_data(df_players, df_matches):
    """
    Ensurre key attributes of players, matches are numeric
    Use cm/kg value for height/weight
    Integrate alternative positions with main positions
    Simplfy positions category
    """
    df_players[ATTRS] = df_players[ATTRS].apply(pd.to_numeric, errors='coerce')
    df_players[STATS] = df_players[STATS].apply(pd.to_numeric, errors='coerce')
    df_players['Height'] = df_players['Height'].str.extract(r'(\d+)\s*cm').astype(float)
    df_players['Weight'] = df_players['Weight'].str.extract(r'(\d+)\s*kg').astype(float)
    df_players['Pos'] = df_players['Position'].apply(simplify_position)
    df_players['Alt'] = df_players['Alternative positions'].apply(simplify_position)
    df_players['full'] = df_players.apply(combine_positions, axis=1)
    df_players['all'] = df_players['full'].apply(simplify_position)
    df_matches[['home_goals', 'away_goals']] = df_matches[['home_goals', 'away_goals']].apply(pd.to_numeric, errors='coerce')
    df_matches['date'] = df_matches['date'].apply(pd.to_datetime, errors = 'coerce')
    return df_players, df_matches