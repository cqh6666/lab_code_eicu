#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:49:20 2022

@author: liukang
"""

import numpy as np
import pandas as pd

test_data = {}
test_total = pd.DataFrame()
disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

feature_list = pd.read_csv('/home/liukang/Doc/valid_df/test_1.csv')
feature_list.drop(['Label'],axis=1,inplace=True)
feature_list = feature_list.columns.tolist()

subgroup_average_record = pd.DataFrame(index=feature_list)
subgroup_nonMiss_record = pd.DataFrame(index=feature_list)
subgroup_nonMiss_average_record = pd.DataFrame(index=feature_list)

subgroup_average_race_record = pd.DataFrame(index=feature_list)
subgroup_nonMiss_race_record = pd.DataFrame(index=feature_list)
subgroup_nonMiss_average_race_record = pd.DataFrame(index=feature_list)

for i in range(1,5):
    
    test_data[i] = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(i))
    test_total =pd.concat([test_total,test_data[i]])

for disease_num in range(disease_list.shape[0]):
    
    test_feature_true=test_total.loc[:,disease_list.iloc[disease_num,0]]>0
    test_meaningful_sample=test_total.loc[test_feature_true]
    
    test_meaningful_sample_notZero = test_meaningful_sample != 0
    
    subgroup_average = test_meaningful_sample.mean(axis=0)
    subgroup_nonMiss = test_meaningful_sample_notZero.sum(axis=0) / test_meaningful_sample.shape[0]
    subgroup_nonMiss_average = test_meaningful_sample.sum(axis=0) / test_meaningful_sample_notZero.sum(axis=0)
    
    subgroup_average_record.loc[:,disease_list.iloc[disease_num,0]] = subgroup_average
    subgroup_nonMiss_record.loc[:,disease_list.iloc[disease_num,0]] = subgroup_nonMiss
    subgroup_nonMiss_average_record.loc[:,disease_list.iloc[disease_num,0]] = subgroup_nonMiss_average
    
    #Statistics of race subgroups
    subgroup_white = test_meaningful_sample.loc[:,'Demo2_1'] == 1
    subgroup_black = test_meaningful_sample.loc[:,'Demo2_2'] == 1
    
    race_sample = {}
    race_sample['white'] = test_meaningful_sample.loc[subgroup_white]
    race_sample['non_white'] = test_meaningful_sample.loc[~subgroup_white]
    race_sample['black'] = test_meaningful_sample.loc[subgroup_black]
    
    for race in ['white','non_white','black']:
        
        data_race = race_sample[race]
        data_race_notZero = data_race != 0
        
        subgroup_average_race_record.loc[:,'{}_{}'.format(disease_list.iloc[disease_num,0],race)] = data_race.mean(axis=0)
        subgroup_nonMiss_race_record.loc[:,'{}_{}'.format(disease_list.iloc[disease_num,0],race)] = data_race_notZero.sum(axis=0) / data_race.shape[0]
        subgroup_nonMiss_average_race_record.loc[:,'{}_{}'.format(disease_list.iloc[disease_num,0],race)] = data_race.sum(axis=0) / data_race_notZero.sum(axis=0)
        
        