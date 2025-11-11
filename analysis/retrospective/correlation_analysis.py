import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/outfield_finalists.csv')

stat_cols = data.columns[4:]
stat_cols.remove('Voting Points')

voting = data['Voting Points']

correlation_values = [] * len(stat_cols)

for col in stat_cols:
    correlation_values.append(voting.corr(data[col]))

