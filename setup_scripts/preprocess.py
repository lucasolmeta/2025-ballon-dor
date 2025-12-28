import pandas as pd
from unidecode import unidecode
import re
from helper_functions.date_to_str import date_to_str

RAW_DATA_PATH = "data/full_raw_data.csv"
VOTES_PATH = "data/ballon_dor_votes.csv"

# Keepers are excluded from most modeling/analysis in this repo.
GK_NAMES = {
    "Mike Maignan",
    "Andre Onana",
    "Thibaut Courtois",
    "Emiliano Martinez",
    "Yassine Bounou",
    "Gianluigi Donnarumma",
}

data = pd.read_csv(RAW_DATA_PATH)

data.columns = data.columns.str.replace('_', ' ')
data.columns = [
    re.sub(r'(^[a-zA-Z]|(?<=\s)[a-zA-Z])', lambda m: m.group(0).upper(), col)
    for col in data.columns
]
data = data.rename(columns={'Player':'Name'})

data['Name'] = (
    data['Name']
    .apply(lambda x: unidecode(str(x)))
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'[^\w\s-]', '', regex=True)
    .str.strip()
)

data['Season'] = data['Season'].apply(date_to_str)

votes = pd.read_csv(VOTES_PATH)

# Normalize scraped/manual vote names in the same way as FBref names so merges are stable.
votes["Name"] = (
    votes["Name"]
    .apply(lambda x: unidecode(str(x)))
    .str.replace(r"\s+", " ", regex=True)
    .str.replace(r"[^\w\s-]", "", regex=True)
    .str.strip()
)

vote_int_cols = [
    "Finalist",
    "Voting Points",
    "Winner",
    "League Winner",
    "UCL Winner",
    "Cup Winner",
    "Major International Continental Trophy Winner",
    "World Cup Winner",
]
for col in vote_int_cols:
    if col in votes.columns:
        votes[col] = votes[col].fillna(0).astype(int)

data = data.merge(votes, on=['Name', 'Season'], how='left')

data['Finalist'] = data['Finalist'].fillna(0).astype(int)
data['Voting Points'] = data['Voting Points'].fillna(0).astype(int)
data['Winner'] = data['Winner'].fillna(0).astype(int)

if "Aerial Duels Won" in data.columns and "Aerial Duels Lost" in data.columns:
    denom = (data["Aerial Duels Won"] + data["Aerial Duels Lost"]).replace(0, pd.NA)
    data["Aerial Duels Won%"] = (data["Aerial Duels Won"] / denom) * 100

is_finalist = data["Finalist"] == 1
is_keeper = data["Name"].isin(GK_NAMES)
is_outfield_finalist = is_finalist & (~is_keeper)

# Vote share is computed within-season to handle point-scale changes across years.
# - Vote Share Overall: among all finalists (incl. keepers)
# - Vote Share Outfield: among outfield finalists only (modeling target for Part 4)
data["Vote Share Overall"] = 0.0
data["Vote Share Outfield"] = 0.0

overall_totals = data.loc[is_finalist].groupby("Season")["Voting Points"].transform("sum")
data.loc[is_finalist, "Vote Share Overall"] = (
    data.loc[is_finalist, "Voting Points"] / overall_totals.replace(0, pd.NA)
)
outfield_totals = data.loc[is_outfield_finalist].groupby("Season")["Voting Points"].transform("sum")
data.loc[is_outfield_finalist, "Vote Share Outfield"] = (
    data.loc[is_outfield_finalist, "Voting Points"] / outfield_totals.replace(0, pd.NA)
)

data["Vote Share Overall"] = data["Vote Share Overall"].fillna(0.0)
data["Vote Share Outfield"] = data["Vote Share Outfield"].fillna(0.0)

data.to_csv('data/full_cleaned_data.csv', index=False)

finalists = data[data['Finalist'] == 1]
finalists = finalists.drop(columns='Finalist')
finalists.to_csv('data/nominees.csv', index=False)

gk_cols = [
    "Performance GA",
    "Performance GA90",
    "Performance SoTA",
    "Performance Saves",
    "Performance Save%",
    "Performance W",
    "Performance D",
    "Performance L",
    "Performance CS",
    "Performance CS%",
    "Penalty Kicks PKatt",
    "Penalty Kicks PKA",
    "Penalty Kicks PKsv",
    "Penalty Kicks PKm",
    "Penalty Kicks Save%",
    "Goals GA",
    "Goals PKA",
    "Goals FK",
    "Goals CK",
    "Goals OG",
    "Expected PSxG",
    "Expected PSxG/SoT",
    "Expected PSxG+/-",
    "Expected /90",
    "Launched Cmp",
    "Launched Att",
    "Launched Cmp%",
    "Passes Att (GK)",
    "Passes Thr",
    "Passes Launch%",
    "Passes AvgLen",
    "Goal Kicks Att",
    "Goal Kicks Launch%",
    "Goal Kicks AvgLen",
    "Crosses Opp",
    "Crosses Stp",
    "Crosses Stp%",
    "Sweeper #OPA",
    "Sweeper #OPA/90",
    "Sweeper AvgDist",
    "A-xAG",
    "Rec"
]

outfield_finalists = finalists[~finalists["Name"].isin(GK_NAMES)]

outfield_finalists = outfield_finalists.drop(columns=gk_cols)
outfield_finalists.to_csv('data/nominees_outfield.csv', index=False)

gk_finalists = finalists[finalists["Name"].isin(GK_NAMES)]
gk_finalists = gk_finalists[['Name','Season'] + gk_cols]
gk_finalists.to_csv('data/nominees_gk.csv', index=False)