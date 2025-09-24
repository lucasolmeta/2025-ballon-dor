import pandas as pd
from unidecode import unidecode

data = pd.read_csv('data/raw_data.csv')

data['player'] = (
    data['player']
    .apply(lambda x: unidecode(str(x)))
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'[^\w\s-]', '', regex=True)
    .str.strip()
)
data['season'] = data['season'].astype(int)

data.to_csv('data/data.csv', index=False)

finalists_list = [
    # 2022 Ballon d’Or
    ['Thibaut Courtois',2122],
    ['Rafael Leao',2122],
    ['Christopher Nkunku',2122],
    ['Mohamed Salah',2122],
    ['Joshua Kimmich',2122],
    ['Trent Alexander-Arnold',2122],
    ['Vinicius Junior',2122],
    ['Bernardo Silva',2122],
    ['Riyad Mahrez',2122],
    ['Casemiro',2122],
    ['Son Heung-min',2122],
    ['Fabinho',2122],
    ['Karim Benzema',2122],
    ['Robert Lewandowski',2122],
    ['Mike Maignan',2122],
    ['Harry Kane',2122],
    ['Darwin Nunez',2122],
    ['Phil Foden',2122],
    ['Sadio Mane',2122],
    ['Sebastien Haller',2122],
    ['Kylian Mbappe',2122],
    ['Luka Modric',2122],
    ['Antonio Rudiger',2122],
    ['Cristiano Ronaldo',2122],
    ['Kevin De Bruyne',2122],
    ['Luis Diaz',2122],
    ['Dusan Vlahovic',2122],
    ['Joao Cancelo',2122],
    ['Erling Haaland',2122],
    ['Virgil van Dijk',2122],

    # 2023 Ballon d’Or
    ['Julian Alvarez',2223],
    ['Nicolo Barella',2223],
    ['Jude Bellingham',2223],
    ['Karim Benzema',2223],
    ['Yassine Bounou',2223],
    ['Kevin De Bruyne',2223],
    ['Ruben Dias',2223],
    ['Antoine Griezmann',2223],
    ['Ilkay Gundogan',2223],
    ['Josko Gvardiol',2223],
    ['Erling Haaland',2223],
    ['Harry Kane',2223],
    ['Kim Min-jae',2223],
    ['Kylian Mbappe',2223],
    ['Lautaro Martinez',2223],
    ['Lionel Messi',2223],
    ['Khvicha Kvaratskhelia',2223],
    ['Victor Osimhen',2223],
    ['Andre Onana',2223],
    ['Bukayo Saka',2223],
    ['Mohamed Salah',2223],
    ['Bernardo Silva',2223],
    ['Vinicius Junior',2223],
    ['Jamal Musiala',2223],
    ['Randal Kolo Muani',2223],
    ['Rodri',2223],
    ['Emiliano Martinez',2223],
    ['Martin Odegaard',2223],
    ['Luka Modric',2223],
    ['Robert Lewandowski',2223],

    # 2024 Ballon d’Or
    ['Jude Bellingham',2324],
    ['Hakan Calhanoglu',2324],
    ['Dani Carvajal',2324],
    ['Ruben Dias',2324],
    ['Artem Dovbyk',2324],
    ['Phil Foden',2324],
    ['Alex Grimaldo',2324],
    ['Erling Haaland',2324],
    ['Mats Hummels',2324],
    ['Harry Kane',2324],
    ['Toni Kroos',2324],
    ['Ademola Lookman',2324],
    ['Emiliano Martinez',2324],
    ['Lautaro Martinez',2324],
    ['Kylian Mbappe',2324],
    ['Martin Odegaard',2324],
    ['Dani Olmo',2324],
    ['Cole Palmer',2324],
    ['Declan Rice',2324],
    ['Rodri',2324],
    ['Antonio Rudiger',2324],
    ['Bukayo Saka',2324],
    ['William Saliba',2324],
    ['Federico Valverde',2324],
    ['Vinicius Junior',2324],
    ['Vitinha',2324],
    ['Nico Williams',2324],
    ['Florian Wirtz',2324],
    ['Granit Xhaka',2324],
    ['Lamine Yamal',2324],

    # 2025 Ballon d’Or
    ['Jude Bellingham',2425],
    ['Raphinha',2425],
    ['Achraf Hakimi',2425],
    ['Cole Palmer',2425],
    ['Gianluigi Donnarumma',2425],
    ['Nuno Mendes',2425],
    ['Pedri',2425],
    ['Desire Doue',2425],
    ['Erling Haaland',2425],
    ['Viktor Gyokeres',2425],
    ['Robert Lewandowski',2425],
    ['Scott McTominay',2425],
    ['Joao Neves',2425],
    ['Serhou Guirassy',2425],
    ['Alexis Mac Allister',2425],
    ['Fabian Ruiz Pena',2425],
    ['Denzel Dumfries',2425],
    ['Khvicha Kvaratskhelia',2425],
    ['Kylian Mbappe',2425],
    ['Lamine Yamal',2425],
    ['Lautaro Martinez',2425],
    ['Declan Rice',2425],
    ['Virgil van Dijk',2425],
    ['Mohamed Salah',2425],
    ['Nico Williams',2425],
    ['Ousmane Dembele',2425],
    ['Florian Wirtz',2425],
    ['Michael Olise',2425],
    ['Vinicius Junior',2425],
    ['Vitinha',2425]
]

finalists = pd.DataFrame(finalists_list, columns=['player','season'])

for name, season in finalists_list:
    if name not in data['player'].unique():
        print("Missing:", repr(name))

finalists = data.merge(finalists, on=['player','season'], how='inner')

finalists.to_csv('data/finalists.csv', index=False)

finalists['player'].to_csv('data/test')