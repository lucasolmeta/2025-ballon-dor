import pandas as pd

data = pd.read_csv('data/nominees.csv')

data = data[(data['Winner'] == 1) & (data['Season'] != '2024-2025')]
data = data[['Name','Carries 1/3']]

print(data)