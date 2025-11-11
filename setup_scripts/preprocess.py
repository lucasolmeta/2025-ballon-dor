import pandas as pd
from unidecode import unidecode
import re
from helper_functions.date_to_str import date_to_str

data = pd.read_csv('data/raw_data.csv')

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

votes = [
    # 2022 Ballon d’Or
    {'Name': 'Thibaut Courtois', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 82, 'Winner': 0},
    {'Name': 'Rafael Leao', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 2, 'Winner': 0},
    {'Name': 'Christopher Nkunku', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Mohamed Salah', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 116, 'Winner': 0},
    {'Name': 'Joshua Kimmich', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Trent Alexander-Arnold', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Vinicius Junior', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 61, 'Winner': 0},
    {'Name': 'Bernardo Silva', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Riyad Mahrez', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 4, 'Winner': 0},
    {'Name': 'Casemiro', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 1, 'Winner': 0},
    {'Name': 'Son Heung-min', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 5, 'Winner': 0},
    {'Name': 'Fabinho', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 2, 'Winner': 0},
    {'Name': 'Karim Benzema', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 549, 'Winner': 1},
    {'Name': 'Robert Lewandowski', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 170, 'Winner': 0},
    {'Name': 'Mike Maignan', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Harry Kane', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Darwin Nunez', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Phil Foden', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Sadio Mane', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 193, 'Winner': 0},
    {'Name': 'Sebastien Haller', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 2, 'Winner': 0},
    {'Name': 'Kylian Mbappe', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 85, 'Winner': 0},
    {'Name': 'Luka Modric', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 20, 'Winner': 0},
    {'Name': 'Antonio Rudiger', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Cristiano Ronaldo', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Kevin De Bruyne', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 175, 'Winner': 0},
    {'Name': 'Luis Diaz', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 1, 'Winner': 0},
    {'Name': 'Dusan Vlahovic', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 1, 'Winner': 0},
    {'Name': 'Joao Cancelo', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Erling Haaland', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 18, 'Winner': 0},
    {'Name': 'Virgil van Dijk', 'Season': '2021-2022', 'Finalist': 1, 'Voting Points': 1, 'Winner': 0},

    # 2023 Ballon d’Or
    {'Name': 'Julian Alvarez', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 28, 'Winner': 0},
    {'Name': 'Nicolo Barella', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Jude Bellingham', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 5, 'Winner': 0},
    {'Name': 'Karim Benzema', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 6, 'Winner': 0},
    {'Name': 'Yassine Bounou', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 10, 'Winner': 0},
    {'Name': 'Kevin De Bruyne', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 100, 'Winner': 0},
    {'Name': 'Ruben Dias', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Antoine Griezmann', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 4, 'Winner': 0},
    {'Name': 'Ilkay Gundogan', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 8, 'Winner': 0},
    {'Name': 'Josko Gvardiol', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 1, 'Winner': 0},
    {'Name': 'Erling Haaland', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 357, 'Winner': 0},
    {'Name': 'Harry Kane', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 4, 'Winner': 0},
    {'Name': 'Kim Min-jae', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 3, 'Winner': 0},
    {'Name': 'Kylian Mbappe', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 270, 'Winner': 0},
    {'Name': 'Lautaro Martinez', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 4, 'Winner': 0},
    {'Name': 'Lionel Messi', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 462, 'Winner': 1},
    {'Name': 'Khvicha Kvaratskhelia', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 6, 'Winner': 0},
    {'Name': 'Victor Osimhen', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 24, 'Winner': 0},
    {'Name': 'Andre Onana', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 2, 'Winner': 0},
    {'Name': 'Bukayo Saka', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 1, 'Winner': 0},
    {'Name': 'Mohamed Salah', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 13, 'Winner': 0},
    {'Name': 'Bernardo Silva', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 20, 'Winner': 0},
    {'Name': 'Vinicius Junior', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 49, 'Winner': 0},
    {'Name': 'Jamal Musiala', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Randal Kolo Muani', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Rodri', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 57, 'Winner': 0},
    {'Name': 'Emiliano Martinez', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 7, 'Winner': 0},
    {'Name': 'Martin Odegaard', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Luka Modric', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 19, 'Winner': 0},
    {'Name': 'Robert Lewandowski', 'Season': '2022-2023', 'Finalist': 1, 'Voting Points': 12, 'Winner': 0},

    # 2024 Ballon d’Or
    {'Name': 'Jude Bellingham', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 917, 'Winner': 0},
    {'Name': 'Hakan Calhanoglu', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 15, 'Winner': 0},
    {'Name': 'Dani Carvajal', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 550, 'Winner': 0},
    {'Name': 'Ruben Dias', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 8, 'Winner': 0},
    {'Name': 'Artem Dovbyk', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Phil Foden', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 157, 'Winner': 0},
    {'Name': 'Alex Grimaldo', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 2, 'Winner': 0},
    {'Name': 'Erling Haaland', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 432, 'Winner': 0},
    {'Name': 'Mats Hummels', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 0, 'Winner': 0},
    {'Name': 'Harry Kane', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 201, 'Winner': 0},
    {'Name': 'Toni Kroos', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 291, 'Winner': 0},
    {'Name': 'Ademola Lookman', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 82, 'Winner': 0},
    {'Name': 'Emiliano Martinez', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 28, 'Winner': 0},
    {'Name': 'Lautaro Martinez', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 402, 'Winner': 0},
    {'Name': 'Kylian Mbappe', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 420, 'Winner': 0},
    {'Name': 'Martin Odegaard', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 16, 'Winner': 0},
    {'Name': 'Dani Olmo', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 86, 'Winner': 0},
    {'Name': 'Cole Palmer', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 7, 'Winner': 0},
    {'Name': 'Declan Rice', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 5, 'Winner': 0},
    {'Name': 'Rodri', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 1170, 'Winner': 1},
    {'Name': 'Antonio Rudiger', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 13, 'Winner': 0},
    {'Name': 'Bukayo Saka', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 14, 'Winner': 0},
    {'Name': 'William Saliba', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 8, 'Winner': 0},
    {'Name': 'Federico Valverde', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 58, 'Winner': 0},
    {'Name': 'Vinicius Junior', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 1129, 'Winner': 0},
    {'Name': 'Vitinha', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 5, 'Winner': 0},
    {'Name': 'Nico Williams', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 73, 'Winner': 0},
    {'Name': 'Florian Wirtz', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 101, 'Winner': 0},
    {'Name': 'Granit Xhaka', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 60, 'Winner': 0},
    {'Name': 'Lamine Yamal', 'Season': '2023-2024', 'Finalist': 1, 'Voting Points': 383, 'Winner': 0},

    # 2025 Ballon d’Or
    {'Name': 'Jude Bellingham', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 21, 'Winner': 0},
    {'Name': 'Raphinha', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 620, 'Winner': 0},
    {'Name': 'Achraf Hakimi', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 484, 'Winner': 0},
    {'Name': 'Cole Palmer', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 211, 'Winner': 0},
    {'Name': 'Gianluigi Donnarumma', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 172, 'Winner': 0},
    {'Name': 'Nuno Mendes', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 171, 'Winner': 0},
    {'Name': 'Pedri', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 124, 'Winner': 0},
    {'Name': 'Desire Doue', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 74, 'Winner': 0},
    {'Name': 'Erling Haaland', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 18, 'Winner': 0},
    {'Name': 'Viktor Gyokeres', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 56, 'Winner': 0},
    {'Name': 'Robert Lewandowski', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 49, 'Winner': 0},
    {'Name': 'Scott McTominay', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 45, 'Winner': 0},
    {'Name': 'Joao Neves', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 40, 'Winner': 0},
    {'Name': 'Serhou Guirassy', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 25, 'Winner': 0},
    {'Name': 'Alexis Mac Allister', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 21, 'Winner': 0},
    {'Name': 'Harry Kane', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 112, 'Winner': 0},
    {'Name': 'Fabian Ruiz Pena', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 20, 'Winner': 0},
    {'Name': 'Denzel Dumfries', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 20, 'Winner': 0},
    {'Name': 'Khvicha Kvaratskhelia', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 123, 'Winner': 0},
    {'Name': 'Kylian Mbappe', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 378, 'Winner': 0},
    {'Name': 'Lamine Yamal', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 1059, 'Winner': 0},
    {'Name': 'Lautaro Martinez', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 37, 'Winner': 0},
    {'Name': 'Declan Rice', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 13, 'Winner': 0},
    {'Name': 'Virgil van Dijk', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 7, 'Winner': 0},
    {'Name': 'Mohamed Salah', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 657, 'Winner': 0},
    {'Name': 'Ousmane Dembele', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 1380, 'Winner': 1},
    {'Name': 'Florian Wirtz', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 5, 'Winner': 0},
    {'Name': 'Michael Olise', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 4, 'Winner': 0},
    {'Name': 'Vinicius Junior', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 51, 'Winner': 0},
    {'Name': 'Vitinha', 'Season': '2024-2025', 'Finalist': 1, 'Voting Points': 703, 'Winner': 0}
]

votes = pd.DataFrame(votes)

data = data.merge(votes, on=['Name', 'Season'], how='left')

data['Finalist'] = data['Finalist'].fillna(0).astype(int)
data['Voting Points'] = data['Voting Points'].fillna(0).astype(int)
data['Winner'] = data['Winner'].fillna(0).astype(int)

data.to_csv('data/cleaned_data.csv', index=False)

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

outfield_finalists = finalists[~finalists['Name'].isin(['Mike Maignan','Andre Onana','Thibaut Courtois','Emiliano Martinez','Yassine Bounou','Gianluigi Donnarumma'])]

outfield_finalists = outfield_finalists.drop(columns=gk_cols)
outfield_finalists.to_csv('data/nominees_outfield.csv', index=False)

gk_finalists = finalists[finalists['Name'].isin(['Mike Maignan','Andre Onana','Thibaut Courtois','Emiliano Martinez','Yassine Bounou','Gianluigi Donnarumma'])]
gk_finalists = gk_finalists[['Name','Season'] + gk_cols]
gk_finalists.to_csv('data/nominees_gk.csv', index=False)