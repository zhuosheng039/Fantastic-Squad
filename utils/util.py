import pandas as pd
from .common import POSITION_SLOTS

def df_with_pos(df_players, pos):
    """Return the dataframe if the player can play pos"""
    fields = df_players['all'].explode()
    return df_players.explode(column='all')[fields.isin([pos])]

def baseline_team(df_players, position_slots=POSITION_SLOTS):
    """Select baseline 4-3-3 squad by highest OVR with position priority FW > MF > DF > GK"""
    df = df_players.copy()

    selected_names = set()
    result = {}

    for slot, group in position_slots:
        candidates = df[
            df['full'].apply(lambda s: slot in s)
            & (~df['Name'].isin(selected_names))
        ]

        if candidates.empty:
            continue

        best = candidates.sort_values('OVR', ascending=False).iloc[0]
        name = best['Name']

        selected_names.add(name)

        entry = {'name': name, 'group': group}

        if slot not in result:
            result[slot] = [entry]
        else:
            result[slot].append(entry)

    for k, v in result.items():
        if len(v) == 1:
            result[k] = v[0]

    return result