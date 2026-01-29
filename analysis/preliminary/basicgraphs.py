import pandas as pd
from helper_functions.bar_graph import bar_graph

data = pd.read_csv('data/nominees_outfield.csv')
data = data[data['Season'] == '2024-2025']
data['Per 90 Minutes NpxG+xAG'] = data['Expected NpxG+xAG'] / data['Playing Time 90s']

bar_graph(data, 'Name', 'Expected NpxG+xAG', 'Name','Non-Penalty Expected Goal Contributions', "2025 Outfield Ballon d'Or Finalists Ranked by Non-Penalty Expected Goal Contributions",'nonpen_xgxa', minimum=0)
bar_graph(data, 'Name', 'Per 90 Minutes NpxG+xAG', 'Name','Non-Penalty Expected Goal Contributions Per 90', "2025 Outfield Ballon d'Or Finalists Ranked by Non-Penalty Expected Goal Contributions Per 90",'nonpen_xgxa_per90', minimum=0)