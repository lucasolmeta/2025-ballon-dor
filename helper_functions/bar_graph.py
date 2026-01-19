import matplotlib.pyplot as plt

def bar_graph(df, x_col, y_col, xlabel, ylabel, title, graph_name, limit_to = 0):
    df = df.sort_values(by=y_col)

    if limit_to > 0:
        df = df.iloc[-limit_to:]

    fig, ax = plt.subplots(figsize=(8, 8))

    bars = ax.bar(df[x_col], df[y_col], width=0.9)

    plt.rcParams['font.family'] = 'DejaVu Sans'

    plt.xticks(rotation=90)

    ax.tick_params(axis='x', which='both', direction='out')
    ax.tick_params(axis='y', which='both', direction='out')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    max_bar_height = max(bar.get_height() for bar in bars)

    if max_bar_height < 1:
        round_to = .2
    else:
        round_to = .1

    for bar in bars:
        height = bar.get_height() + max_bar_height * 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:{round_to}f}',
            ha='center', va='bottom', fontsize=8, rotation=90
        )

    ax.set_ylim(min(df[y_col]), max(df[y_col]) * 1.12)
    ax.set_xlim(-0.5, len(df[x_col]) - 0.5)

    plt.savefig(f'analysis/graphs/{graph_name}.png', dpi=300, bbox_inches='tight')
    plt.close()