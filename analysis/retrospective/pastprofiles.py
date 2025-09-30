import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/outfield_finalists.csv')

stat_cols = data.columns[4:]

for col in stat_cols:
    data[f'{col}_relative'] = data.groupby('season')[col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

data.to_csv('data/test.csv')

winners = [
    ('Benzema', 2122),
    ('Messi', 2223),
    ('Rodri', 2324)
]

winners_df = data[data.set_index(['player', 'season']).index.isin(winners)]

relative_cols = [col for col in data.columns if col.endswith('_relative')]

winners_avg = winners_df[relative_cols].mean()

values = winners_avg.values
N = len(values)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values = np.concatenate((values, [values[0]])) 
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.plot(angles, values, color='black', linewidth=2)
ax.fill(angles, values, color='black', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([col.replace('_relative', '') for col in relative_cols], fontsize=10)

ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

plt.savefig(f'analysis/graphs/winnerspider.png', dpi=300, bbox_inches='tight')