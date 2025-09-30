import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/2425_outfield_finalists.csv')
data['Per 90 Minutes_npxG+xAG'] = data['Expected_npxG+xAG'] / data['Playing Time_90s']

def bar_graph(df, x_col, y_col, xlabel, ylabel, title, graph_name):
    df = df.sort_values(by=y_col)

    plt.rcParams['font.family'] = 'DejaVu Sans'

    plt.bar(df[x_col], df[y_col])
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

    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color='white')

    max_bar_height = ax.patches[-1].get_height()

    if max_bar_height < 1:
        round_to = .2
    else:
        round_to = .1

    for bar in ax.patches:
        height = bar.get_height() + max_bar_height * 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:{round_to}f}',
            ha='center', va='bottom', color='yellow', fontsize=8, rotation=90
        )

    ax.set_ylim(0, max(df[y_col]) * 1.12)

    bars = ax.bar(df[x_col], df[y_col], color='white', width=0.9)
    ax.set_xlim(-0.5, len(df[x_col]) - 0.5)

    plt.savefig(f'analysis/graphs/{graph_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

bar_graph(data, 'player', 'Expected_npxG+xAG', 'Name','Non-Penalty Expected Goal Contributions', "2025 Outfield Ballon d'Or Finalists Ranked by Non-Penalty Expected Goal Contributions",'nonpenxgxa')
bar_graph(data, 'player', 'Per 90 Minutes_npxG+xAG', 'Name','Non-Penalty Expected Goal Contributions Per 90', "2025 Outfield Ballon d'Or Finalists Ranked by Non-Penalty Expected Goal Contributions Per 90",'nonpenxgxaper90')