#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:30:17 2022

@author: liukang
"""

import numpy as np
import pandas as pd

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

race = ['white','black','non_white']
race_id = ['Demo2_1','Demo2_2','Demo2_1']
race_standard = [1,1,-1]
race_count = pd.DataFrame()

test_total = pd.DataFrame()
for data_num in range(1,5):
    
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    test_total = pd.concat([test_total, test_data])



subgroup_data_all = {}
for subgroup_num in range(disease_list.shape[0]):
    
    subgroup_feature_true = test_total.loc[:,disease_list.iloc[subgroup_num,0]]>0
    subgroup_data_select = test_total.loc[subgroup_feature_true]
    
    subgroup_data_all[subgroup_num] = subgroup_data_select



for race_num in range(len(race_id)):
    
    race_select = race[race_num]
    
    if race_standard[race_num] == -1:
        
        data_race_true = test_total.loc[:,race_id[race_num]] == 0
    
    else:
        
        data_race_true = test_total.loc[:,race_id[race_num]] == race_standard[race_num]
    
    data_race_select = test_total.loc[data_race_true].copy()
    data_race_select.reset_index(drop=True,inplace=True)
    data_race_all_subgroup = data_race_select.loc[:,disease_list.iloc[:,0]]
    data_race_all_subgroup_sum = data_race_all_subgroup.sum(axis=1)
    data_race_all_subgroup_true = data_race_all_subgroup_sum >= 1
    data_race_all_subgroup_select = data_race_select.loc[data_race_all_subgroup_true]
    data_race_all_subgroup_AKI = np.sum(data_race_all_subgroup_select['Label'])
    
    for disease_num in range(disease_list.shape[0]):
        
        subgroup_data_used = subgroup_data_all[disease_num]
        subgroup_AKI = np.sum(subgroup_data_used['Label'])
        
        disease_id = disease_list.iloc[disease_num,0]
        race_data_subgroup_true = data_race_select.loc[:,disease_id]>0
        race_data_subgroup_select = data_race_select.loc[race_data_subgroup_true]
        race_data_subgroup_AKI = np.sum(race_data_subgroup_select['Label'])
        
        race_count.loc[disease_id, '{}_num'.format(race_select)] = race_data_subgroup_select.shape[0]
        race_count.loc[disease_id, '{}_race%'.format(race_select)] = race_data_subgroup_select.shape[0] / data_race_all_subgroup_select.shape[0]
        race_count.loc[disease_id, '{}_subgroup%'.format(race_select)] = race_data_subgroup_select.shape[0] / subgroup_data_used.shape[0]
        race_count.loc[disease_id, '{}_AKI'.format(race_select)] = race_data_subgroup_AKI
        race_count.loc[disease_id, '{}_AKI%'.format(race_select)] = race_data_subgroup_AKI / race_data_subgroup_select.shape[0]
        race_count.loc[disease_id, '{}_AKI_race%'.format(race_select)] = race_data_subgroup_AKI / data_race_all_subgroup_AKI
        race_count.loc[disease_id, '{}_AKI_subgroup%'.format(race_select)] = race_data_subgroup_AKI / subgroup_AKI

race_count.to_csv('/home/liukang/Doc/fairness_analysis/race_count.csv')
        
        
        
        
    
    