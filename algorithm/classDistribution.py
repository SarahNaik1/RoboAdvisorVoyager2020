#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:07:35 2020

@author: monikamishra
"""

import pandas as pd
from collections import Counter

data = pd.read_csv('../dataset/Questionaire.csv', header = [1])

print(data.shape)

target = data.values[:,-1]

counter = Counter(target)

for k,v in counter.items():
    per = v/len(target)*100
    print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

