import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .common import ATTRS, POS, ATTRSEX, FORMATION_POSITIONS, POS_COLORS
from .util import df_with_pos, baseline_team
from .cleaner import simplify_position

def plot_attribute_distribution(df_players, attrs=ATTRSEX):
    """Hist + KDE for each attribute in attrs."""
    df_players = df_players.copy()
    for a in attrs:
        data = df_players[a].dropna().sort_values()
        fig, ax = plt.subplots()
        ax.hist(data, bins=30, density=True, alpha=0.65)
        data.plot(kind='kde', ax=ax)
        ax.set_xlabel(a)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {a}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plot_box_by_position(df_players, attrs=ATTRSEX):
    """Boxplots of attrs by simplified positions."""
    df_players = df_players.copy()
    for a in attrs:
        groups = []
        labels = []
        for pos in POS:
            g = df_with_pos(df_players, pos)
            vals = g[a].dropna()
            if len(vals) > 0:
                groups.append(vals)
                labels.append(pos)
            if not groups:
                continue
        fig, ax = plt.subplots()
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_xlabel('Position')
        ax.set_ylabel(a)
        ax.set_title(f'{a} by Position (simplified)')

def plot_corr_matrix(df_players, cols=ATTRSEX):
    """Correlation matrix with numeric columns of interest annotated."""
    present = [c for c in cols if c in df_players.columns]
    corr = df_players[present].corr()
    fig, ax = plt.subplots(figsize=(9,7))
    cax = ax.imshow(corr.values,
                    interpolation='nearest',
                    aspect='auto',          
                    cmap='coolwarm',      
                    vmin=-1, vmax=1)        
    ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha='right'); ax.set_yticklabels(present)
    for (i,j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title('Correlation matrix of player attributes')

def plot_height_weight(df_players):
    """plot height weight distribution with regard to ovr"""
    df_players = df_players.copy()
    sub = df_players.dropna(subset=['Height','Weight','OVR'])

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    scatter = ax.scatter(
        sub['Height'], sub['Weight'],
        c=sub['OVR'],
        cmap='viridis',
        alpha=0.75,
        edgecolor='white',
        linewidth=0.8
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('OVR')

    ax.set_xlabel('Height (cm)', fontsize=12)
    ax.set_ylabel('Weight (kg)', fontsize=12)
    ax.set_title = 'Height vs Weight colored by Overall Rating (OVR)'

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()    

def plot_position_scatter_markers(df_players, k=8):
    """Position-specific 2D scatter: FW SHO vs PAC, MF PAS vs DRI, DF DEF vs PHY"""
    df_players = df_players.copy()
    combos = {
        'FW': ('SHO','PAC','Forwards: Shooting vs Pace'),
        'MF': ('PAS','DRI','Midfielders: Passing vs Dribbling'),
        'DF': ('DEF','PHY','Defenders: Defending vs Physical'),
    }
    for pos, (xcol,ycol,title) in combos.items():
        sub = df_with_pos(df_players, pos).dropna(subset=[xcol, ycol, 'OVR'])
        
        fig, ax = plt.subplots(figsize=(7,6))
        sc = ax.scatter(
            sub[xcol], sub[ycol],
            c=sub['OVR'],
            cmap='viridis',
            s=25,
            alpha=0.6,
            edgecolors='none'
        )

        plt.colorbar(sc, ax=ax, label='OVR')
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title)   
        # label top by OVR
        top = sub.sort_values('OVR', ascending=False).head(k)
        for _, r in top.iterrows():
            ax.annotate(r['Name'], (r[xcol], r[ycol]), fontsize=7, alpha=0.8)

def plot_goals_distribution(df_matches):
    """
    Distribution of goals
    """
    df_matches = df_matches.copy()
    df_matches['total_goals'] = df_matches['home_goals'] + df_matches['away_goals']
    for col in ['home_goals','away_goals', 'total_goals']:
        seri = df_matches[col]
        fig, ax = plt.subplots()
        maxg = int(max(seri.max(), 6))
        bins = range(0, maxg+2)
        ax.hist(seri, bins=bins, density=True, alpha=0.7)
        seri.plot(kind='kde', ax=ax)
        ax.set_xlabel('Goals per match'); ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {col}')

def plot_league_mean_goals(df_matches):
    """
    Distribution of goals across leagues
    """
    df_matches = df_matches.copy()
    df_matches['total_goals'] = df_matches['home_goals'] + df_matches['away_goals']
    stats = df_matches.groupby('League')['total_goals'].agg(['mean','median','std','count']).sort_values('mean', ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(stats.index, stats['mean'])
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    ax.set_ylabel('Mean total goals per match')
    ax.set_title('League mean total goals per match')

def plot_team_med_ovr_vs_goals(df_players, df_matches, attrs=ATTRS):
    """
    This shows whether team with high overall rating tends to have more goals
    """
    team_attr = df_players.groupby('Team')[ATTRS].median()
    teams = set(df_matches['home_team']).union(df_matches['away_team'])
    records = []
    for team in teams:
        home = df_matches[df_matches['home_team']==team]
        away = df_matches[df_matches['away_team']==team]
        gf = home['home_goals'].sum() + away['away_goals'].sum()
        gc = home['away_goals'].sum() + away['home_goals'].sum()
        matches = len(home)+len(away)
        if matches == 0:
            continue
        records.append({'Team':team, 'gf_per_match': gf/matches, 'ga_per_match': gc/matches})
    df_team_match = pd.DataFrame(records).set_index('Team')
    merged = team_attr.join(df_team_match, how='inner')
    fig, ax = plt.subplots()
    ax.scatter(merged['OVR'], merged['gf_per_match'], c=merged['OVR'], cmap='viridis', s=30, alpha=0.7)
    ax.set_xlabel('Team median OVR (from players)')
    ax.set_ylabel('Goals for per match')
    ax.set_title('Team med OVR vs Goals for per match')
    for team in merged.index:
        if merged.loc[team,'OVR'] >= merged['OVR'].quantile(0.90) or merged.loc[team,'gf_per_match'] >= merged['gf_per_match'].quantile(0.90):
            ax.annotate(team, (merged.loc[team,'OVR'], merged.loc[team,'gf_per_match']), fontsize=8, xytext=(3,3), textcoords='offset points')

def plot_team_med_ovr_vs_loss(df_players, df_matches, attrs=ATTRS):
    """
    This shows whether team with low overall ratings tends to loss more goals
    """
    team_attr = df_players.groupby('Team')[ATTRS].median()
    teams = set(df_matches['home_team']).union(df_matches['away_team'])
    records = []
    for team in teams:
        home = df_matches[df_matches['home_team']==team]
        away = df_matches[df_matches['away_team']==team]
        gf = home['home_goals'].sum() + away['away_goals'].sum()
        gc = home['away_goals'].sum() + away['home_goals'].sum()
        matches = len(home)+len(away)
        if matches == 0:
            continue
        records.append({'Team':team, 'gf_per_match': gf/matches, 'ga_per_match': gc/matches})
    df_team_match = pd.DataFrame(records).set_index('Team')
    merged = team_attr.join(df_team_match, how='inner')
    fig, ax = plt.subplots()
    ax.scatter(merged['OVR'], merged['ga_per_match'], c=merged['OVR'], cmap='viridis', s=30, alpha=0.7)
    ax.set_xlabel('Team median OVR (from players)')
    ax.set_ylabel('Loss for per match')
    ax.set_title('Team med OVR vs Loss for per match')
    for team in merged.index:
        if merged.loc[team,'OVR'] >= merged['OVR'].quantile(0.90) or merged.loc[team,'ga_per_match'] >= merged['ga_per_match'].quantile(0.90):
            ax.annotate(team, (merged.loc[team,'OVR'], merged.loc[team,'ga_per_match']), fontsize=8, xytext=(3,3), textcoords='offset points')

def plot_team_med_ovr_vs_win_rate(df_players, df_matches, attrs=ATTRS):
    """
    This shows whether team with low overall ratings tends to loss more goals
    """
    team_attr = df_players.groupby('Team')[ATTRS].median()
    teams = set(df_matches['home_team']).union(df_matches['away_team'])
    records = []
    for team in teams:
        home = df_matches[df_matches['home_team']==team]
        away = df_matches[df_matches['away_team']==team]
        w = (home['result'] == 'H').sum() + (away['result'] == 'A').sum()
        gf = home['home_goals'].sum() + away['away_goals'].sum()
        gc = home['away_goals'].sum() + away['home_goals'].sum()
        matches = len(home)+len(away)
        if matches == 0:
            continue
        records.append({'Team':team, 'gf_per_match': gf/matches, 'ga_per_match': gc/matches, 'win_rate': w/matches})
    df_team_match = pd.DataFrame(records).set_index('Team')
    merged = team_attr.join(df_team_match, how='inner')
    fig, ax = plt.subplots()
    ax.scatter(merged['OVR'], merged['win_rate'], c=merged['OVR'], cmap='viridis', s=30, alpha=0.7)
    ax.set_xlabel('Team median OVR (from players)')
    ax.set_ylabel('Win rate')
    ax.set_title('Team med OVR vs win rate')
    for team in merged.index:
        if merged.loc[team,'OVR'] >= merged['OVR'].quantile(0.90) or merged.loc[team,'win_rate'] >= merged['win_rate'].quantile(0.90):
            ax.annotate(team, (merged.loc[team,'OVR'], merged.loc[team,'win_rate']), fontsize=8, xytext=(3,3), textcoords='offset points')

def plot_team_med_ovr_vs_non_defeat_rate(df_players, df_matches, attrs=ATTRS):
    """
    This shows whether team with low overall ratings tends to loss more goals
    """
    team_attr = df_players.groupby('Team')[ATTRS].median()
    teams = set(df_matches['home_team']).union(df_matches['away_team'])
    records = []
    for team in teams:
        home = df_matches[df_matches['home_team']==team]
        away = df_matches[df_matches['away_team']==team]
        w = ((home['result'] == 'H') | (home['result'] == 'D')).sum() + ((away['result'] == 'A') | (away['result'] == 'D')).sum()
        gf = home['home_goals'].sum() + away['away_goals'].sum()
        gc = home['away_goals'].sum() + away['home_goals'].sum()
        matches = len(home)+len(away)
        if matches == 0:
            continue
        records.append({'Team':team, 'gf_per_match': gf/matches, 'ga_per_match': gc/matches, 'win_rate': w/matches})
    df_team_match = pd.DataFrame(records).set_index('Team')
    merged = team_attr.join(df_team_match, how='inner')
    fig, ax = plt.subplots()
    ax.scatter(merged['OVR'], merged['win_rate'], c=merged['OVR'], cmap='viridis', s=30, alpha=0.7)
    ax.set_xlabel('Team median OVR (from players)')
    ax.set_ylabel('Non-defeat rate')
    ax.set_title('Team med OVR vs non-defeat rate')
    for team in merged.index:
        if merged.loc[team,'OVR'] >= merged['OVR'].quantile(0.90) or merged.loc[team,'win_rate'] >= merged['win_rate'].quantile(0.90):
            ax.annotate(team, (merged.loc[team,'OVR'], merged.loc[team,'win_rate']), fontsize=8, xytext=(3,3), textcoords='offset points')

def parallel_topk_by_position(df_players, attrs=ATTRS, top_k=5):
    """Radar graph of topk by position"""
    GLOBAL_MIN = df_players[ATTRS].min()
    GLOBAL_MAX = df_players[ATTRS].max()
    range_val = GLOBAL_MAX - GLOBAL_MIN
    range_val[range_val == 0] = 1

    for pos in POS:
        sub = df_players[df_players['Pos'].apply(lambda x: pos in x)]
        if sub.empty:
            continue

        top = sub.sort_values('OVR', ascending=False).head(top_k)
        df_norm = (top[ATTRS] - GLOBAL_MIN) / range_val
            
        N = len(attrs)
            
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] 

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)   
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(attrs, fontsize=12)
            
        ax.set_rlabel_position(0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)], color="grey", size=8)
        ax.set_ylim(0, 1)

        for index, row in df_norm.iterrows():
            values = row.values.flatten().tolist()
            values += values[:1] 
                
            player_name = top.loc[index, 'Name']
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=player_name, alpha=0.7)
            ax.fill(angles, values, alpha=0.1)

        ax.set_title(f'Radar Chart for Top {top_k} {pos} Players', size=16, y=1.1)
        ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0.1), fontsize=8)
        plt.show()

def baseline_plot(df_players, pos_colors=POS_COLORS):
    """Baseline team with highest ovr for each pos""" 
    baseline_dict = baseline_team(df_players)

    rows = []

    for slot, entry in baseline_dict.items():
        if isinstance(entry, list):
            for e in entry:
                row = df_players[df_players['Name'] == e['name']].iloc[0].copy()
                row['Selected_Position'] = slot
                row['Position_Group'] = e['group']
                rows.append(row)
        else:
            row = df_players[df_players['Name'] == entry['name']].iloc[0].copy()
            row['Selected_Position'] = slot
            row['Position_Group'] = entry['group']
            rows.append(row)

    df_squad = pd.DataFrame(rows)

    
    colors = df_squad['Position_Group'].map(pos_colors)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        df_squad['Name'],
        df_squad['OVR'],
        color=colors,
        edgecolor='black',
        linewidth=0.7
    )

    for i, ovr in enumerate(df_squad['OVR']):
        ax.text(ovr + 0.5, i, f'{ovr:.1f}', va='center', fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in pos_colors.values()]
    ax.legend(handles, pos_colors.keys(), title='Position Group', loc='lower right')

    ax.set_xlabel('OVR')
    ax.set_title('Baseline 4-3-3 Squad', fontsize=14, weight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def draw_football_pitch(ax):
    pitch = patches.Rectangle((0, 0), 1, 1, linewidth=2,
                              edgecolor='white', facecolor='#2d5016', zorder=0)
    ax.add_patch(pitch)

    ax.plot([0, 1], [0.5, 0.5], color='white', linewidth=2, zorder=1)

    circle = patches.Circle((0.5, 0.5), 0.1, linewidth=2,
                             edgecolor='white', facecolor='none', zorder=1)
    ax.add_patch(circle)

    box1 = patches.Rectangle((0.25, 0), 0.5, 0.15,
                             linewidth=2, edgecolor='white',
                             facecolor='none', zorder=1)
    box2 = patches.Rectangle((0.25, 0.85), 0.5, 0.15,
                             linewidth=2, edgecolor='white',
                             facecolor='none', zorder=1)
    ax.add_patch(box1)
    ax.add_patch(box2)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_aspect('equal')


def build_formation_data(baseline_dict, df_players):
    data = []

    for slot, entry in baseline_dict.items():
        if isinstance(entry, list):
            entries = entry
        else:
            entries = [entry]

        for e in entries:
            r = df_players[df_players['Name'] == e['name']].iloc[0]

            data.append({
                'slot': slot,
                'group': e['group'],
                'name': r['Name'],
                'ovr': r['OVR']
            })

    return data


def plot_formation(df_players, formation_positions=FORMATION_POSITIONS, pos_colors=POS_COLORS):
    baseline_dict = baseline_team(df_players)
    data = build_formation_data(baseline_dict, df_players)

    fig, ax = plt.subplots(figsize=(10, 14))
    draw_football_pitch(ax)

    slot_counter = {}

    for p in data:
        slot = p['slot']
        group = p['group']

        if slot not in formation_positions:
            continue

        idx = slot_counter.get(slot, 0)
        coords = formation_positions[slot]

        if idx >= len(coords):
            continue

        x, y = coords[idx]
        slot_counter[slot] = idx + 1

        color = pos_colors[group]

        c = patches.Circle((x, y), 0.035, color=color,
                           ec='white', linewidth=2.5, zorder=3)
        ax.add_patch(c)

        ax.text(x, y - 0.06, p['name'],
                ha='center', va='top', fontsize=9,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='black', alpha=0.7, edgecolor='none'),
                zorder=4)

        ax.text(x, y, f"{p['ovr']}",
                ha='center', va='center', fontsize=8,
                color='white', weight='bold', zorder=5)

    ax.text(0.5, 1.08, '4-3-3 Formation',
            ha='center', va='center', fontsize=18,
            color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#1a1a1a', alpha=0.9,
                      edgecolor='white', linewidth=2))

    legend_elements = [
        patches.Patch(facecolor=pos_colors['GK'], edgecolor='white', label='Goalkeeper'),
        patches.Patch(facecolor=pos_colors['DF'], edgecolor='white', label='Defender'),
        patches.Patch(facecolor=pos_colors['MF'], edgecolor='white', label='Midfielder'),
        patches.Patch(facecolor=pos_colors['FW'], edgecolor='white', label='Forward')
    ]

    ax.legend(handles=legend_elements, loc='upper left',
              framealpha=0.9, edgecolor='white',
              facecolor='#1a1a1a', labelcolor='white', fontsize=10)

    plt.tight_layout()
    plt.show()
