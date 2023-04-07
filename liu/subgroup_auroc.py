#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:26:34 2022

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

subgroup_result = {}
subgroup_result_total = pd.DataFrame()

result_name = {}
result_name['global'] = 'predict_proba'
result_name['personal'] = 'update_1921_mat_proba'
model_list = ['global', 'personal']

# auroc_record = pd.DataFrame()
# for disease_num in range(disease_list.shape[0]):
#     disease_id_name = disease_list.iloc[disease_num, 0]
#     subgroup_df = pd.read_csv('/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_id_name))
#
#     y_label = subgroup_df['Label']
#
#     for model in model_list:
#         y_predict = subgroup_df[result_name[model]]
#         # AUC
#         model_auc = roc_auc_score(y_label, y_predict)
#         auroc_record.loc[disease_id_name, model] = model_auc
# auroc_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_auroc.csv")

# global
data_range = [1, 2, 3, 4, 5]
auc_append = []
for data_idx in data_range:
    test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_idx))
    train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_idx))
    lr_All = LogisticRegression(n_jobs=-1)

    X_train = train_df.drop(['Label'], axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop(['Label'], axis=1)
    y_test = test_df['Label']

    lr_All.fit(X_train, y_train)
    y_predict = lr_All.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_predict)
    print("auc", auc)
    auc_append.append(auc)

print(np.mean(auc_append))


