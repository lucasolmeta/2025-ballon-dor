import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helper_functions.bar_graph import bar_graph

data = pd.read_csv('data/nominees_outfield.csv')

data = data.rename(columns={
    'Performance G+A': 'Goal Contributions', 
    'Standard SoT': 'Shots on Target',
    'Outcomes Off': 'Offsides',
    'PPA': 'Passes into Opponent Penalty Area',
    'Pass Types TB': 'Through Balls',
    'Touches Att Pen': 'Touches in Opponent Penalty Area',
    'GCA Types PassLive': 'Open Play Passes',
    'Expected NpxG+xAG': 'Expected Non-Penalty Goal Contributions'
})

stat_cols = [
    'Goal Contributions', 
    'Shots on Target',
    'Offsides',
    'Passes into Opponent Penalty Area',
    'Through Balls',
    'Touches in Opponent Penalty Area', 
    'Open Play Passes',
    'Expected Non-Penalty Goal Contributions'
]

data = data[stat_cols + ['Name','Season','Winner']]

for col in stat_cols:
    data[col] = data.groupby('Season')[col].rank(pct=True, method='average')
    data[col] = data[col].clip(lower=0.05)

big_four = data[
    data['Name'].isin(['Lamine Yamal','Ousmane Dembele','Mohamed Salah','Raphinha'])
    & (data['Season'] == '2024-2025')
]

winners_df = data[
    (data['Winner'] == 1)
    & (data['Season'] != '2024-2025')
]

profiles = winners_df[stat_cols].mean().to_frame().T
profiles['Name'] = 'Past-Winner Profile'

profiles = pd.concat(
    [big_four.drop(columns=['Season','Winner']), profiles],
    ignore_index=True
)

angles = np.linspace(0, 2 * np.pi, len(stat_cols), endpoint=False).tolist()
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.spines['polar'].set_linewidth(4)
ax.tick_params(labelsize=8)

ax.grid(linestyle='-', linewidth=0.6)

colors = {
    'Past-Winner Profile': '#1B2A41',
    'Lamine Yamal': '#E09F3E',
    'Ousmane Dembele': '#3A86A8',
    'Mohamed Salah': '#6D597A',
    'Raphinha': '#8D99AE'
}

markers = {
    'Past-Winner Profile': 'o',
    'Lamine Yamal': 's',
    'Ousmane Dembele': '^',
    'Mohamed Salah': 'D',
    'Raphinha': 'X'
}

ax.set_ylim(0, 1)
ax.tick_params(pad=12)
ax.spines['polar'].set_linewidth(4)

ax.set_xticklabels(stat_cols, fontsize=9, fontweight='medium')

ax.set_yticklabels([])
ax.set_yticks([0.25, 0.5, 0.75, 1])
ax.tick_params(labelsize=8)

for label, row in profiles.groupby('Name'):
    vals = row.iloc[0][stat_cols].tolist()
    vals += [vals[0]]
    lw = 2

    color = colors.get(label, 'gray')

    ax.plot(
        angles,
        vals,
        color=color,
        linewidth=lw,
        alpha=0.75,
        marker=markers.get(label, 'o'),
        markersize=5,
        markeredgewidth=0.9,
        markerfacecolor='white',
        label=label
    )

for i, label in enumerate(ax.get_xticklabels()):
    if stat_cols[i] in ['Passes into Opponent Penalty Area','Touches in Opponent Penalty Area']:
        label.set_y(label.get_position()[1] - 0.17)
    elif stat_cols[i] in ['Through Balls','Shots on Target']:
        label.set_y(label.get_position()[1] - 0.07)
    elif stat_cols[i] in ['Goal Contributions']:
        label.set_y(label.get_position()[1] - 0.1)
    elif stat_cols[i] in ['Expected Non-Penalty Goal Contributions',]:
        label.set_y(label.get_position()[1] - 0.2)

ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), frameon=True)

plt.savefig(f'analysis/graphs/winner_spider.png', dpi=300, bbox_inches='tight')

similarity_scores = pd.DataFrame(columns=['Name','Similarity Score'])

for name in ['Lamine Yamal','Ousmane Dembele','Mohamed Salah','Raphinha']:
    player_vector = profiles.loc[profiles['Name'] == name, stat_cols].to_numpy().reshape(1, -1)
    avg_winner_vector = profiles.loc[profiles['Name'] == 'Past-Winner Profile', stat_cols].to_numpy().reshape(1, -1)

    similarity = cosine_similarity(player_vector, avg_winner_vector)[0][0]

    similarity_scores.loc[len(similarity_scores)] = [name, float(similarity)]

bar_graph(similarity_scores, 'Name','Similarity Score','Player','Cosine Similarity to Past-Winner Profile','Cosine Similarity to Past-Winner Profile By Player','similarity_scores', minimum=0.8)