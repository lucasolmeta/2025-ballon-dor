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
stat_cols.remove("Vote Share Outfield")
stat_cols.remove("Vote Share Overall")

voting = data["Vote Share Outfield"]

correl_df = pd.DataFrame(columns=["Column","Correlation Value"])
abs_correl_df = pd.DataFrame(columns=["Column","Correlation Value"])

for col in stat_cols:
    correl_df.loc[len(correl_df)] = [col, voting.corr(data[col])]
    abs_correl_df.loc[len(abs_correl_df)] = [col, abs(voting.corr(data[col]))]

correl_df = correl_df.sort_values(by="Correlation Value")
correl_df.to_csv("data/feature_correlations.csv", index=False)

abs_correl_df = abs_correl_df.sort_values(by="Correlation Value")
abs_correl_df.to_csv("data/abs_feature_correlations.csv", index=False)

bar_graph(correl_df, "Column","Correlation Value", "Outfield Voting Share", "Top 20 Features with Strongest Positive Correlation to Outfield Voting Share", "Features with Strongest Positive Correlation to Ballon d'or Outfield Voting Share","feature_correlations", limit_to=30)
bar_graph(abs_correl_df, "Column","Correlation Value", "Outfield Voting Share", "Top 20 Features with Strongest Correlation to Outfield Voting Share", "Features with Strongest Correlation to Ballon d'or Outfield Voting Share","abs_feature_correlations", limit_to=30)