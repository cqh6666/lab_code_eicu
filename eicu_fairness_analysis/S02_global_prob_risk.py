# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S02_global_prob_risk
   Description:   获得全局模型的预测概率集合，顺便将个性化模型补充进来保存为一个csv
   Author:        cqh
   date:          2023/4/25 15:01
-------------------------------------------------
   Change Activity:
                  2023/4/25:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from api_utils import get_cross_data

valid_id_list = [1,2,3,4,5]
# 个性化建模的版本
version = 31

result_df = pd.DataFrame()

for cross_id in valid_id_list:
    person_result = pd.read_csv(
        f"/home/chenqinhai/code_eicu/my_lab/result/S04/0/S04_LR_test{cross_id}_tra1_iter100_select10_v{version}.csv",
        index_col=0)

    train_data_x, test_data_x, train_data_y, test_data_y = get_cross_data(cross_id)
    cur_result_df = pd.DataFrame()

    lr_All = LogisticRegression(n_jobs=-1)
    lr_All.fit(train_data_x, train_data_y)
    y_predict = lr_All.predict_proba(test_data_x)[:, 1]

    global_score = roc_auc_score(test_data_y, y_predict)
    personal_score = roc_auc_score(test_data_y, person_result['prob'])
    print(cross_id, global_score, personal_score)

    cur_result_df['real'] = person_result['real']
    cur_result_df['global_predict_proba'] = y_predict
    cur_result_df['personal_predict_proba'] = person_result['prob']
    cur_result_df.index = test_data_y.index
    print("get GM, PMTL prob, and save success!")
    result_df = pd.concat([result_df, cur_result_df], axis=0)

result_df.to_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S02_test_result_GM_PMTL_predict_prob_with_drg.csv")