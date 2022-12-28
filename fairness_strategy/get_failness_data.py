# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_failness_data
   Description:   ...
   Author:        cqh
   date:          2022/12/28 16:33
-------------------------------------------------
   Change Activity:
                  2022/12/28:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import numpy as np

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv').squeeze().to_list()
race_list = ['Demo2_1', 'Demo2_2']
# train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(1))
test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(1))
columns = test_df.columns.to_list()


# read personal coef_ and intercept and result
# person_coef = pd.read_csv(
#     '/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(1))
person_result = pd.read_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(1))
y_score = person_result['update_1921_mat_proba']
score_df = pd.DataFrame(data={"score_y": y_score}, index=test_df.index.to_list())
# personal_score
# test_original = test_df.drop(['Label'], axis=1)
# test_original['intercept'] = 1
# personal_score = np.multiply(test_original, person_coef)

# lr = LogisticRegression(n_jobs=-1)
# lr.fit(train_x, train_y)
# score_y = lr.predict_proba(test_x)[:, 1]
# auc = roc_auc_score(test_y, score_y)
# print("auc:", auc)
# score_y_df = pd.DataFrame(data={"score_y": score_y}, index=test_x.index.to_list())
# all_data = pd.concat([test_x[race_list], test_x[disease_list], test_y, score_y_df], axis=1)

all_data = pd.concat([test_df[race_list], test_df[disease_list], test_df[['Label']], score_df], axis=1)
data1 = all_data[all_data['Demo2_1'] == 1]
data2 = all_data[all_data['Demo2_2'] == 1]
all_data_df = pd.concat([data1, data2], axis=0)
# 保留4位有效数字
all_data_df = all_data_df.round(4)
all_data_df.reset_index(drop=True, inplace=True)
all_data_df.to_feather("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/test_data_1.feather")
