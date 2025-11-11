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

data.to_csv('data/cleaned_data.csv', index=False)

finalists_list = [
    # 2022 Ballon d’Or
    ['Thibaut Courtois','2021-2022'],
    ['Rafael Leao','2021-2022'],
    ['Christopher Nkunku','2021-2022'],
    ['Mohamed Salah','2021-2022'],
    ['Joshua Kimmich','2021-2022'],
    ['Trent Alexander-Arnold','2021-2022'],
    ['Vinicius Junior','2021-2022'],
    ['Bernardo Silva','2021-2022'],
    ['Riyad Mahrez','2021-2022'],
    ['Casemiro','2021-2022'],
    ['Son Heung-min','2021-2022'],
    ['Fabinho','2021-2022'],
    ['Karim Benzema','2021-2022'],
    ['Robert Lewandowski','2021-2022'],
    ['Mike Maignan','2021-2022'],
    ['Harry Kane','2021-2022'],
    ['Darwin Nunez','2021-2022'],
    ['Phil Foden','2021-2022'],
    ['Sadio Mane','2021-2022'],
    ['Sebastien Haller','2021-2022'],
    ['Kylian Mbappe','2021-2022'],
    ['Luka Modric','2021-2022'],
    ['Antonio Rudiger','2021-2022'],
    ['Cristiano Ronaldo','2021-2022'],
    ['Kevin De Bruyne','2021-2022'],
    ['Luis Diaz','2021-2022'],
    ['Dusan Vlahovic','2021-2022'],
    ['Joao Cancelo','2021-2022'],
    ['Erling Haaland','2021-2022'],
    ['Virgil van Dijk','2021-2022'],

    # 2023 Ballon d’Or
    ['Julian Alvarez','2022-2023'],
    ['Nicolo Barella','2022-2023'],
    ['Jude Bellingham','2022-2023'],
    ['Karim Benzema','2022-2023'],
    ['Yassine Bounou','2022-2023'],
    ['Kevin De Bruyne','2022-2023'],
    ['Ruben Dias','2022-2023'],
    ['Antoine Griezmann','2022-2023'],
    ['Ilkay Gundogan','2022-2023'],
    ['Josko Gvardiol','2022-2023'],
    ['Erling Haaland','2022-2023'],
    ['Harry Kane','2022-2023'],
    ['Kim Min-jae','2022-2023'],
    ['Kylian Mbappe','2022-2023'],
    ['Lautaro Martinez','2022-2023'],
    ['Lionel Messi','2022-2023'],
    ['Khvicha Kvaratskhelia','2022-2023'],
    ['Victor Osimhen','2022-2023'],
    ['Andre Onana','2022-2023'],
    ['Bukayo Saka','2022-2023'],
    ['Mohamed Salah','2022-2023'],
    ['Bernardo Silva','2022-2023'],
    ['Vinicius Junior','2022-2023'],
    ['Jamal Musiala','2022-2023'],
    ['Randal Kolo Muani','2022-2023'],
    ['Rodri','2022-2023'],
    ['Emiliano Martinez','2022-2023'],
    ['Martin Odegaard','2022-2023'],
    ['Luka Modric','2022-2023'],
    ['Robert Lewandowski','2022-2023'],

    # 2024 Ballon d’Or
    ['Jude Bellingham','2023-2024'],
    ['Hakan Calhanoglu','2023-2024'],
    ['Dani Carvajal','2023-2024'],
    ['Ruben Dias','2023-2024'],
    ['Artem Dovbyk','2023-2024'],
    ['Phil Foden','2023-2024'],
    ['Alex Grimaldo','2023-2024'],
    ['Erling Haaland','2023-2024'],
    ['Mats Hummels','2023-2024'],
    ['Harry Kane','2023-2024'],
    ['Toni Kroos','2023-2024'],
    ['Ademola Lookman','2023-2024'],
    ['Emiliano Martinez','2023-2024'],
    ['Lautaro Martinez','2023-2024'],
    ['Kylian Mbappe','2023-2024'],
    ['Martin Odegaard','2023-2024'],
    ['Dani Olmo','2023-2024'],
    ['Cole Palmer','2023-2024'],
    ['Declan Rice','2023-2024'],
    ['Rodri','2023-2024'],
    ['Antonio Rudiger','2023-2024'],
    ['Bukayo Saka','2023-2024'],
    ['William Saliba','2023-2024'],
    ['Federico Valverde','2023-2024'],
    ['Vinicius Junior','2023-2024'],
    ['Vitinha','2023-2024'],
    ['Nico Williams','2023-2024'],
    ['Florian Wirtz','2023-2024'],
    ['Granit Xhaka','2023-2024'],
    ['Lamine Yamal','2023-2024'],

    # 2025 Ballon d’Or
    ['Jude Bellingham','2024-2025'],
    ['Raphinha','2024-2025'],
    ['Achraf Hakimi','2024-2025'],
    ['Cole Palmer','2024-2025'],
    ['Gianluigi Donnarumma','2024-2025'],
    ['Nuno Mendes','2024-2025'],
    ['Pedri','2024-2025'],
    ['Desire Doue','2024-2025'],
    ['Erling Haaland','2024-2025'],
    ['Viktor Gyokeres','2024-2025'],
    ['Robert Lewandowski','2024-2025'],
    ['Scott McTominay','2024-2025'],
    ['Joao Neves','2024-2025'],
    ['Serhou Guirassy','2024-2025'],
    ['Alexis Mac Allister','2024-2025'],
    ['Fabian Ruiz Pena','2024-2025'],
    ['Denzel Dumfries','2024-2025'],
    ['Khvicha Kvaratskhelia','2024-2025'],
    ['Kylian Mbappe','2024-2025'],
    ['Lamine Yamal','2024-2025'],
    ['Lautaro Martinez','2024-2025'],
    ['Declan Rice','2024-2025'],
    ['Virgil van Dijk','2024-2025'],
    ['Mohamed Salah','2024-2025'],
    ['Nico Williams','2024-2025'],
    ['Ousmane Dembele','2024-2025'],
    ['Florian Wirtz','2024-2025'],
    ['Michael Olise','2024-2025'],
    ['Vinicius Junior','2024-2025'],
    ['Vitinha','2024-2025']
]

finalists = pd.DataFrame(finalists_list, columns=['Name','Season'])
finalists = data.merge(finalists, on=['Name','Season'], how='inner')
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