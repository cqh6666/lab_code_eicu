# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test_api
   Description:   ...
   Author:        cqh
   date:          2022/12/25 14:05
-------------------------------------------------
   Change Activity:
                  2022/12/25:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv').squeeze().to_list()
race_list = ['Demo2_1','Demo2_2']
train_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(1))
columns = train_df.columns.to_list()
# train_x = train_df[disease_list]
train_x = train_df.drop(['Label'], axis=1)
train_y = train_df['Label']

lr = LogisticRegression(n_jobs=-1)
lr.fit(train_x, train_y)
score_y = lr.predict_proba(train_x)[:, 1]
auc = roc_auc_score(train_y, score_y)
print("auc:", auc)

score_y_df = pd.DataFrame(data={"score_y": score_y}, index=train_y.index.to_list())
all_data = pd.concat([train_x[race_list], train_x[disease_list], train_y, score_y_df], axis=1)

data1 = all_data[all_data['Demo2_1'] == 1]
data2 = all_data[all_data['Demo2_2'] == 1]
all_data_df = pd.concat([data1, data2], axis=0)
all_data_df.reset_index(drop=True, inplace=True)
all_data.to_feather("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/test_data_1.feather")
print("done!")