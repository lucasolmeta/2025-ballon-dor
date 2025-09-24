import pandas as pd

raw_data = pd.read_csv('data/raw_data.csv', na_values=['', ' '])

stat_cols = [c for c in raw_data.columns if c not in ['player', 'season']]

data_cleaned = raw_data
data_cleaned = raw_data.dropna(subset=stat_cols, how='all')

data_cleaned.to_csv('data/data.csv', index=False)