#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Created on 31 05 2020 - 4:48 PM
  
 @author Sarah Naik
"""
import pandas as pd
import zipfile
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

pd.options.mode.chained_assignment = None

# Read the data
zf = zipfile.ZipFile('../../dataset/Dataset.zip')
df = pd.read_csv(zf.open('Dataset/Questionaire.csv'))

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

cols = df.columns[:40] # first 40 columns
colours = ['#330000', '#ff4d4d']
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

