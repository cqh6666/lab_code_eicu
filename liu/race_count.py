#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:30:17 2022

@author: liukang
"""

import numpy as np
import pandas as pd

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

race = ['white','black', 'no_white']
race_id = ['Demo2_1','Demo2_2', 'Demo2_1']
race_standard = [1,1, -1]
race_count = pd.DataFrame()
# 得到全部样本
test_total = pd.DataFrame()
for data_num in range(1,5):
    
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    test_total = pd.concat([test_total, test_data])


# 得到20个高风险亚组的样本
subgroup_data_all = {}
for subgroup_num in range(disease_list.shape[0]):
    
    subgroup_feature_true = test_total.loc[:,disease_list.iloc[subgroup_num,0]]>0  #
    subgroup_data_select = test_total.loc[subgroup_feature_true]
    
    subgroup_data_all[subgroup_num] = subgroup_data_select



for race_num in range(len(race_id)):
    race_select = race[race_num]
    if race_standard[race_num] == -1:
        data_race_true = test_total.loc[:,race_id[race_num]] == 0
    else:
        data_race_true = test_total.loc[:,race_id[race_num]] == race_standard[race_num]  # 对应种族群体的条件
    
    data_race_select = test_total.loc[data_race_true].copy()  # 对应种族的所有样本集
    data_race_select.reset_index(drop=True,inplace=True)
    data_race_all_subgroup = data_race_select.loc[:,disease_list.iloc[:,0]]  # 对应种族的20个入院原因的特征
    data_race_all_subgroup_sum = data_race_all_subgroup.sum(axis=1)  #
    data_race_all_subgroup_true = data_race_all_subgroup_sum >= 1  # 排除非这20个入院原因的样本
    data_race_all_subgroup_select = data_race_select.loc[data_race_all_subgroup_true]  # 选出是20个入院原因的对应种族群体
    data_race_all_subgroup_AKI = np.sum(data_race_all_subgroup_select['Label'])  # 对应种群的AKI患者总人数
    
    for disease_num in range(disease_list.shape[0]):
        
        subgroup_data_used = subgroup_data_all[disease_num]  # 当前亚组的总人数
        subgroup_AKI = np.sum(subgroup_data_used['Label'])  # 当前亚组的总AKI人数
        
        disease_id = disease_list.iloc[disease_num,0]
        race_data_subgroup_true = data_race_select.loc[:,disease_id]>0
        race_data_subgroup_select = data_race_select.loc[race_data_subgroup_true]  # 当前亚组的对应种族群体
        race_data_subgroup_AKI = np.sum(race_data_subgroup_select['Label'])  # 当前亚组对应的种族群体的AKI患者
        # 当前亚组的对应种族人数
        race_count.loc[disease_id, '{}_num'.format(race_select)] = race_data_subgroup_select.shape[0]
        # 当前亚组的对应种族人数 / 所有亚组的对应种族群体
        race_count.loc[disease_id, '{}_race%'.format(race_select)] = race_data_subgroup_select.shape[0] / data_race_all_subgroup_select.shape[0]
        # 当前亚组的对应种族人数 / 当前亚组的总人数
        race_count.loc[disease_id, '{}_subgroup%'.format(race_select)] = race_data_subgroup_select.shape[0] / subgroup_data_used.shape[0]
        # 当前亚组的对应种族群体的AKI患者人数
        race_count.loc[disease_id, '{}_AKI'.format(race_select)] = race_data_subgroup_AKI
        # 当前亚组的对应种族群体的AKI患者人数 / 当前亚组的对应种族人数
        race_count.loc[disease_id, '{}_AKI%'.format(race_select)] = race_data_subgroup_AKI / race_data_subgroup_select.shape[0]
        # 当前亚组的对应种族群体的AKI患者人数 / 所有亚组的对应种族群体的AKI患者人数
        race_count.loc[disease_id, '{}_AKI_race%'.format(race_select)] = race_data_subgroup_AKI / data_race_all_subgroup_AKI
        # 当前亚组的对应种族群体的AKI患者人数 / 当前亚组的AKI患者人数
        race_count.loc[disease_id, '{}_AKI_subgroup%'.format(race_select)] = race_data_subgroup_AKI / subgroup_AKI

for disease_num in range(disease_list.shape[0]):
    disease_id = disease_list.iloc[disease_num, 0]
    subgroup_data_used = subgroup_data_all[disease_num]  # 当前亚组的总人数
    subgroup_AKI = np.sum(subgroup_data_used['Label'])  # 当前亚组的总AKI人数

    # 当前亚组的AKI患者人数 / 当前亚组的总人数
    race_count.loc[disease_id, '当前亚组的AKI占比'] = subgroup_AKI / subgroup_data_used.shape[0]


# 个性化模型和全局模型召回的AKI患者组成


race_count.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/race_count_2.csv', encoding='gbk')