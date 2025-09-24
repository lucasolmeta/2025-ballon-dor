import soccerdata as sd
import pandas as pd
import warnings, logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

stat_types = [
    'standard','shooting','passing','passing_types','goal_shot_creation',
    'defense','possession','playing_time','misc','keeper','keeper_adv'
]

competitions = [
    'Big 5 European Leagues Combined',
    'UEFA-Champions League',
    'UEFA-Europa League',
    'UEFA-Europa Conference League',
    'INT-Club World Cup',
    'INT-World Cup',
    'INT-World Cup Qualifying',
    'INT-European Championship',
    'INT-European Championship Qualifying',
    'INT-Copa America',
    'INT-Africa Cup of Nations',
    'INT-UEFA Nations League',
    'INT-CONCACAF Gold Cup',
    'INT-CONCACAF Nations League',
    'ENG-FA Cup','ENG-EFL Cup','ENG-Community Shield',
    'ESP-Copa del Rey','ESP-Supercopa',
    'GER-DFB Pokal','GER-Supercup',
    'ITA-Coppa Italia','ITA-Supercoppa',
    'FRA-Coupe de France','FRA-Trophee des Champions'
]

seasons=['2122','2223','2324','2425']

data = pd.DataFrame()

i = 0
iterations = len(seasons) * len(stat_types) * len(competitions)

for league in competitions:
    for season in seasons:
        for stat_type in stat_types:
            try:
                stat_data = (
                    sd.FBref(leagues=[league], seasons=season)
                    .read_player_season_stats(stat_type=stat_type)
                    .reset_index()
                    .sort_values(['player','season'])
                    .groupby(['player','season']).sum(numeric_only=True)
                )

                stat_data.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in stat_data.columns]
                stat_data = stat_data.drop(columns=stat_data.columns.intersection(data.columns))
                data = stat_data if data.empty else pd.merge(data, stat_data, on=['player','season'], how='outer')

                i += 1

                print(f'Completed call #{i} of {iterations}  ({round((i/iterations*100), 2)}%)')

            except Exception as e:
                i += 1

                print(f'⚠️ Skipping call #{i} of {iterations} ({round((i/iterations*100), 2)}%)')

data.index.names = ['player','season']
data.to_csv('data/raw_data.csv')