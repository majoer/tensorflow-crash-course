#https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=pandas-colab&hl=no#scrollTo=-GQFz8NZuS06

import pandas as pd
import numpy as np;

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population})

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
# cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['Supersaiyan city'] = cities['City name'].apply(lambda name: name.startswith('San')) & cities['Area square miles'].apply(lambda m2: m2 > 50)

print(cities)

#cities = cities.reindex(np.random.permutation(cities.index))
cities = cities.reindex([0, 1, 3])

print(cities)
