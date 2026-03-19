import pandas as pd
from helper_functions.bar_graph import bar_graph

data = pd.read_csv('data/nominees_outfield.csv')
data = data[data['Season'] == '2024-2025']

bar_graph(data, 'Name', 'Expected NpxG+xAG', 'Name','Non-Penalty Expected Goal Contributions', "2025 Outfield Ballon d'Or Nominees Ranked by Non-Penalty Expected Goal Contributions",'nonpen_xgxa', minimum=0)
bar_graph(data, 'Name', 'Per 90 Minutes NpxG+xAG', 'Name','Non-Penalty Expected Goal Contributions Per 90', "2025 Outfield Ballon d'Or Nominees Ranked by Non-Penalty Expected Goal Contributions Per 90",'nonpen_xgxa_per90', minimum=0)
bar_graph(data, 'Name', 'Total PrgDist', 'Name','Total Yards that Completed Passes or Carries Have Traveled Directly Towards Opponent Goal', "2025 Outfield Ballon d'Or Nominees Ranked by Total Progressive Distance",'prg_dist', minimum=0)
bar_graph(data, 'Name', 'Touches Live', 'Name','Touches from Open Play', "2025 Outfield Ballon d'Or Nominees Ranked by Touches from Open Play",'touches_live', minimum=0)
bar_graph(data, 'Name', 'Take-Ons Succ%', 'Name','Successful Take-On %', "2025 Outfield Ballon d'Or Nominees Ranked by Successful Take-On %",'successful_take_on_percent', minimum=0)
bar_graph(data, 'Name', 'SCA SCA90', 'Name','Shot Creating Actions per 90', "2025 Outfield Ballon d'Or Nominees Ranked by Shot Creating Actions Per 90",'SCA_per90', minimum=0)