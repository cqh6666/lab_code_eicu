#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:41:58 2022

@author: liukang
"""

import numpy as np
import pandas as pd

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv').iloc[:,0].tolist()
gender_list = ['Demo2_1','Demo2_2']


test_total = pd.DataFrame()
person_coef_total = pd.DataFrame()
for data_num in range(1,5):
    
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    person_coef = pd.read_csv('/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
    
    test_total = pd.concat([test_total,test_data])
    person_coef_total = pd.concat([person_coef_total,person_coef])
    

average_value_record = pd.DataFrame()
average_coef_record = pd.DataFrame()
for gender in gender_list:
    
    list_used = disease_list.copy()
    list_used.append(gender)
    test_data_true = test_total.loc[:,list_used].sum(axis=1)>=2
    test_data_select = test_total.loc[test_data_true]
    coef_data_select = person_coef_total.loc[test_data_true]
    
    average_value_record.loc[:,'all_subgroup_{}'.format(gender)] = test_data_select.mean()
    average_coef_record.loc[:,'all_subgroup_{}'.format(gender)] = coef_data_select.mean()
    

for subgroup in disease_list:
    
    for gender in gender_list:
        
        list_used = [subgroup,gender]
        test_data_true = test_total.loc[:,list_used].sum(axis=1)>=2
        test_data_select = test_total.loc[test_data_true]
        coef_data_select = person_coef_total.loc[test_data_true]
        
        average_value_record.loc[:,'{}_{}'.format(subgroup,gender)] = test_data_select.mean()
        average_coef_record.loc[:,'{}_{}'.format(subgroup,gender)] = coef_data_select.mean()
        
average_value_record.to_csv('/home/liukang/Doc/fairness_analysis/race_feature_avg.csv')
average_coef_record.to_csv('/home/liukang/Doc/fairness_analysis/race_person_coef_avg.csv')