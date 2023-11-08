

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('us_births_2016_2021.csv')

df = df.drop(['State Abbreviation', 'Year', 'Gender', 'Education Level Code'], axis=1)

df = df.replace('Unknown or Not Stated', pd.NA)
df['Education Level of Mother'] = df['Education Level of Mother'].fillna('Unknown')

print(df.head())
print(df.describe())
print(df['Education Level of Mother'].value_counts())

grouped = df.groupby('Education Level of Mother').agg({
    'Number of Births': 'sum',
    'Average Age of Mother (years)': 'mean',
    'Average Birth Weight (g)': 'mean'
})

grouped.plot(kind='bar', subplots=True, layout=(1, 3), figsize=(12, 4))
plt.show()



