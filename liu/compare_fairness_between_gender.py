#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:31:57 2022

@author: liukang
"""

import numpy as np
import pandas as pd

threshold_used = ['500','1120','2241','self_10%','self_20%','self_%div2','self_%','self_race_10%','self_race_20%','self_race_%div2','self_race_%']

for threshold_id in threshold_used:
    
    for fairness_measure in ['TPR','FPR','odds','PPV']:
        
        measure_select = '{}_threshold_{}'.format(fairness_measure,threshold_id)
        female_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_fairness_{}.csv'.format(measure_select),index_col=0)
        male_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_male_fairness_{}.csv'.format(measure_select),index_col=0)
        
        female_vs_male = female_result - male_result
        
        female_vs_male.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_vs_male_fairness_{}.csv'.format(measure_select))
        
for fairness_measure in ['TPR','FPR','odds']:
    
    measure_select = '{}_no_threshold'.format(fairness_measure)
    female_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_fairness_{}.csv'.format(measure_select),index_col=0)
    male_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_male_fairness_{}.csv'.format(measure_select),index_col=0)
    
    female_vs_male = female_result - male_result
    
    female_vs_male.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_vs_male_fairness_{}.csv'.format(measure_select))
    

female_AKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_fairness_AKI_num.csv',index_col=0)
male_AKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_male_fairness_AKI_num.csv',index_col=0)
AKI_female_vs_male = female_AKI_select - male_AKI_select
AKI_female_vs_male.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_vs_male_fairness_AKI_num.csv')

female_nonAKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_fairness_nonAKI_num.csv',index_col=0)
male_nonAKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_male_fairness_nonAKI_num.csv',index_col=0)
nonAKI_female_vs_male = female_nonAKI_select - male_nonAKI_select
nonAKI_female_vs_male.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_female_vs_male_fairness_nonAKI_num.csv')

