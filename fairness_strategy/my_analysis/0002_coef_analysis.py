# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0002_coef_analysis
   Description:   根据20个高风险入院原因Drg 亚组LR得到的coef特征重要性，分别分析各亚组的Top-50重要特征因子
   Author:        cqh
   date:          2023/3/21 10:04
-------------------------------------------------
   Change Activity:
                  2023/3/21:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

coef_records = pd.read_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_feature_coef.csv', index_col=0)
disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

top_K = [10, 50, 100, 200]


coef_top_K_records = pd.DataFrame()
for k in top_K:
    for disease_num in range(disease_list.shape[0]):
        disease_id_name = disease_list.iloc[disease_num, 0]

        coef_temp = coef_records.loc[:, disease_id_name]
        sort_coef_temp = coef_temp.sort_values(ascending=False)

        coef_top_K_records.loc[disease_id_name, k] = sort_coef_temp[:k].sum() / sort_coef_temp.sum()

coef_top_K_records.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_coef_top_K_compare.csv')