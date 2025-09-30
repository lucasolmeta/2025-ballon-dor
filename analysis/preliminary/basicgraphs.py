import pandas as pd
import sklearn
import matplotlib.pyplot as plt

data = pd.read_csv('data/2425_outfield_finalists.csv')

data = data.sort_values(by='Expected_npxG+xAG')

plt.rcParams['font.family'] = 'DejaVu Sans'

plt.bar(data['player'],data['Expected_npxG+xAG'])
plt.xlabel('Name')
plt.ylabel('Non-Penalty xG + xA')
plt.title("2025 Outfield Ballon d'Or Finalists Ranked by Expected Non-Penalty Goal Contributions")
plt.xticks(rotation=90)

fig = plt.gcf()
ax = plt.gca()

fig.patch.set_facecolor('black')
ax.set_facecolor('black')

for bar in ax.patches:
    bar.set_color('white')

for spine in ax.spines.values():
    spine.set_color('yellow')

ax.tick_params(axis='x', colors='white', which='both', direction='out')
ax.tick_params(axis='y', colors='white', which='both', direction='out')
ax.xaxis.set_tick_params(color='yellow')
ax.yaxis.set_tick_params(color='yellow')

ax.set_xlabel("Name", color='white')
ax.set_ylabel("Non-Penalty xG + xA", color='white')
ax.set_title("2025 Outfield Ballon d'Or Finalists Ranked by Expected Non-Penalty Goal Contributions", color='white')

for bar in ax.patches:
    height = bar.get_height() + 0.4
    ax.text(
        bar.get_x() + bar.get_width() / 2, height,
        f'{height:.1f}',
        ha='center', va='bottom', color='yellow', fontsize=8, rotation=90
    )

ax.set_ylim(0, max(data['Expected_npxG+xAG']) * 1.12)

bars = ax.bar(data['player'], data['Expected_npxG+xAG'], color='white', width=0.9)
ax.set_xlim(-0.5, len(data['player']) - 0.5)

plt.savefig('analysis/graphs/nonpenxgxa.png', dpi=300, bbox_inches='tight')
plt.close()