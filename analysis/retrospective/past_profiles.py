import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/nominees_outfield.csv')

data = data.rename(columns={
    'Carries 1/3': 'Carries into Final Third', 
    'Take-Ons Att': 'Take-On Attempts',
    'Pass Types TB': 'Through Balls',
    'Playing Time Mn/MP': 'Minutes per Match Played',
    'Aerial Duels Won%': 'Aerial Duel Win Percentage', 
    'Performance Fld': 'Fouls Drawn',
    'Expected Np:G-xG': 'xG Overperformance',
    'Per 90 Minutes G+A-PK': 'Non-Penalty Goals and Assists Per 90'
})

stat_cols = [
    'Carries into Final Third', 
    'Take-On Attempts',
    'Through Balls',
    'Minutes per Match Played',
    'Aerial Duel Win Percentage', 
    'Fouls Drawn',
    'xG Overperformance',
    'Non-Penalty Goals and Assists Per 90'
]

data = data[stat_cols + ['Name','Season','Winner']]
big_four = data[
    data['Name'].isin(['Lamine Yamal','Ousmane Dembele','Mohamed Salah','Raphinha'])
    & (data['Season'] == '2024-2025')
]

winners_df = data[
    (data['Winner'] == 1)
    & (data['Season'] != '2024-2025')
]

profiles = winners_df[stat_cols].mean().to_frame().T
profiles['Name'] = 'Average Previous Winner'

profiles = pd.concat(
    [big_four.drop(columns=['Season','Winner']), profiles],
    ignore_index=True
)

for col in stat_cols:
    col_min = min(0, profiles[col].min())
    col_max = profiles[col].max()
 
    profiles[col] = (profiles[col] - col_min) / (col_max - col_min)

angles = np.linspace(0, 2 * np.pi, len(stat_cols), endpoint=False).tolist()
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

colors = {
    'Average Previous Winner': '#000000',
    'Lamine Yamal': '#E69F00',
    'Ousmane Dembele': '#56B4E9',
    'Mohamed Salah': '#009E73',
    'Raphinha': '#D55E00'
}

ax.set_ylim(0, 1)
ax.tick_params(pad=12)
ax.spines['polar'].set_linewidth(3)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(stat_cols, fontsize=8)
ax.set_yticks([0.33,0.67])
ax.set_yticklabels([])

for label, row in profiles.groupby('Name'):
    vals = row.iloc[0][stat_cols].tolist()
    vals += [vals[0]]
    lw = 2

    color = colors.get(label, 'gray')
    ax.plot(angles, vals, color=color, linewidth=lw, label=label)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(stat_cols, fontsize=8)
ax.set_yticks([0.33,0.67])
ax.set_yticklabels([])

for i, label in enumerate(ax.get_xticklabels()):
    if stat_cols[i] == 'Team xG Differential When On vs. Off Field':
        label.set_y(label.get_position()[1] - 0.18)
    elif stat_cols[i] in ['Shot Creation Actions via Take-Ons','Expected Assists Per 90']:
        label.set_y(label.get_position()[1] - 0.15)
    if stat_cols[i] in ['Take On Attempts','Carries in Final Third','Through Balls']:
        label.set_y(label.get_position()[1] - 0.1)

ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), frameon=True)

plt.savefig(f'analysis/graphs/winner_spider.png', dpi=300, bbox_inches='tight')