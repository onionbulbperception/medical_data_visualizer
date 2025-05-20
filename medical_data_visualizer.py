import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv', sep=',')
#print(df.head())

# 2
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html#pandas.DataFrame.astype
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)
#print(df['overweight'])

# 3
# Normalizes data by making 0 always good and 1 always bad.
# If the value of cholesterol or gluc is 1, sets the value to 0. 
# If the value is more than 1, sets the value to 1.
#df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
#df['gluc'] = (df['gluc'] > 1).astype(int)

# "Apply 1 if x is 0 else set x to 1"
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
#print(df['cholesterol'])
#print(df['gluc'])

# 4
def draw_cat_plot():
    # 5
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html # !!!Converst data into long format
    # Creates df for cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.SeriesGroupBy.size.html#pandas.core.groupby.SeriesGroupBy.size
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.reset_index.html#pandas.Series.reset_index
    # Groups and reformats data if df cat to split by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    # https://seaborn.pydata.org/generated/seaborn.catplot.html
    chart = sns.catplot(data=df_cat, kind='bar', x='variable', y='total', hue='value', col='cardio')


    # 8
    fig = chart.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # Filters out segment with incorrect data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    # Calculates the correlation matrix for heat map
    corr = df_heat.corr()

    # 13
    # https://numpy.org/doc/stable/reference/generated/numpy.triu.html
    # https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
    # Generates a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    # Setting up heatmap figure
    fig, ax = plt.subplots(figsize=(16,9))

    # 15
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(corr, mask=mask, square=True, linewidths=0.5, annot=True, fmt="0.1f")


    # 16
    fig.savefig('heatmap.png')
    return fig
