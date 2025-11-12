import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helper_functions.bar_graph import bar_graph

data = pd.read_csv('data/nominees_outfield.csv')

data = data.rename(columns={
    'Carries 1/3': 'Carries in the Final Third', 
    'Take-Ons Att': 'Take-On Attempts',
    'Pass Types TB': 'Through Balls',
    'Playing Time Mn/MP': 'Minutes Per Match Played',
    'Blocks Sh': 'Shots Blocked',
    'Performance Fld': 'Fouls Drawn',
    'Expected Np:G-xG': 'xG Overperformance',
    'Per 90 Minutes G+A-PK': 'Non-Penalty Goals and Assists Per 90'
})

stat_cols = [
    'Carries in the Final Third', 
    'xG Overperformance',
    'Through Balls',
    'Minutes Per Match Played',
    'Fouls Drawn',
    'Shots Blocked', 
    'Take-On Attempts',
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

    profiles[col] = profiles[col].clip(lower=0.05)

angles = np.linspace(0, 2 * np.pi, len(stat_cols), endpoint=False).tolist()
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.spines['polar'].set_color('yellow')
ax.spines['polar'].set_linewidth(4)
ax.tick_params(colors='white', labelsize=8)

ax.grid(color='white', linestyle='-', linewidth=0.6)

colors = {
    'Average Previous Winner': '#ffffff',
    'Lamine Yamal': '#CC79A7',
    'Ousmane Dembele': '#56B4E9',
    'Mohamed Salah': '#009E73',
    'Raphinha': '#BB6BD9'
}

ax.set_ylim(0, 1)
ax.tick_params(pad=12)
ax.spines['polar'].set_linewidth(4)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(stat_cols, fontsize=8, color='white')
ax.set_yticks([0.33, 0.67])
ax.set_yticklabels([], color='white')

for label, row in profiles.groupby('Name'):
    vals = row.iloc[0][stat_cols].tolist()
    vals += [vals[0]]
    lw = 2

    color = colors.get(label, 'gray')
    ax.plot(angles, vals, color=color, linewidth=lw, label=label)

for i, label in enumerate(ax.get_xticklabels()):
    if stat_cols[i] == 'Non-Penalty Goals and Assists Per 90':
        label.set_y(label.get_position()[1] - 0.2)
    elif stat_cols[i] == 'Carries in the Final Third':
        label.set_y(label.get_position()[1] - 0.18)
    elif stat_cols[i] in ['Minutes Per Match Played', 'xG Overperformance']:
        label.set_y(label.get_position()[1] - 0.12)
    elif stat_cols[i] in ['Fouls Drawn','Shots Blocked',]:
        label.set_y(label.get_position()[1] - 0.07)

ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), frameon=True, facecolor='black', edgecolor='yellow', labelcolor='white')

plt.savefig(f'analysis/graphs/winner_spider.png', dpi=300, bbox_inches='tight')

similarity_scores = pd.DataFrame(columns=['Name','Similarity Score'])

for name in ['Lamine Yamal','Ousmane Dembele','Mohamed Salah','Raphinha']:
    player_vector = profiles.loc[profiles['Name'] == name, stat_cols].to_numpy().reshape(1, -1)
    avg_winner_vector = profiles.loc[profiles['Name'] == 'Average Previous Winner', stat_cols].to_numpy().reshape(1, -1)

    similarity = cosine_similarity(player_vector, avg_winner_vector)[0][0]

    similarity_scores.loc[len(similarity_scores)] = [name, float(similarity)]

bar_graph(similarity_scores, 'Name','Similarity Score','Player','Similarity to Average Past Winner Profile','Similarity to Average Past Winner Profile Per Player','similarity_scores')