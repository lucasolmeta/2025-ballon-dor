import pandas as pd
from helper_functions.bar_graph import bar_graph

data = pd.read_csv("data/nominees_outfield.csv")

data = data[data['Season'] != '2024-2025']

stat_cols = data.columns[4:].to_list()

stat_cols.remove("Voting Points")
stat_cols.remove("Winner")
stat_cols.remove("League Winner")
stat_cols.remove("UCL Winner")
stat_cols.remove("Cup Winner")
stat_cols.remove("Major International Continental Trophy Winner")
stat_cols.remove("World Cup Winner")

voting = data["Voting Points"]

correl_df = pd.DataFrame(columns=["Column","Correlation Value"])

for col in stat_cols:
    correl_df.loc[len(correl_df)] = [col, abs(voting.corr(data[col]))]

correl_df = correl_df.sort_values(by="Correlation Value")
correl_df.to_csv("data/feature_correlations.csv", index=False)

bar_graph(correl_df, "Column","Correlation Value", "Statistic", "Top 20 Statistics with Strongest Correlation to Voting Points", "Statistics with Strongest Correlation to Ballon d'or Voting Points","statistic_correlations", limit_to=20)