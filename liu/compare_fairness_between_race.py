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
        white_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_fairness_{}.csv'.format(measure_select),index_col=0)
        black_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_black_fairness_{}.csv'.format(measure_select),index_col=0)
        non_white_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_non_white_fairness_{}.csv'.format(measure_select),index_col=0)
        
        white_vs_black = white_result - black_result
        white_vs_non_white = white_result - non_white_result
        
        white_vs_black.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_black_fairness_{}.csv'.format(measure_select))
        white_vs_non_white.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_non_white_fairness_{}.csv'.format(measure_select))
        
for fairness_measure in ['TPR','FPR','odds']:
    
    measure_select = '{}_no_threshold'.format(fairness_measure)
    white_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_fairness_{}.csv'.format(measure_select),index_col=0)
    black_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_black_fairness_{}.csv'.format(measure_select),index_col=0)
    non_white_result = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_non_white_fairness_{}.csv'.format(measure_select),index_col=0)
    
    white_vs_black = white_result - black_result
    white_vs_non_white = white_result - non_white_result
    
    white_vs_black.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_black_fairness_{}.csv'.format(measure_select))
    white_vs_non_white.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_non_white_fairness_{}.csv'.format(measure_select))
    

white_AKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_fairness_AKI_num.csv',index_col=0)
black_AKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_black_fairness_AKI_num.csv',index_col=0)
non_white_AKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_non_white_fairness_AKI_num.csv',index_col=0)
AKI_white_vs_black = white_AKI_select - black_AKI_select
AKI_white_vs_non_white = white_AKI_select - non_white_AKI_select
AKI_white_vs_black.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_black_fairness_AKI_num.csv')
AKI_white_vs_non_white.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_non_white_fairness_AKI_num.csv')

white_nonAKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_fairness_nonAKI_num.csv',index_col=0)
black_nonAKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_black_fairness_nonAKI_num.csv',index_col=0)
non_white_nonAKI_select = pd.read_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_non_white_fairness_nonAKI_num.csv',index_col=0)
nonAKI_white_vs_black = white_nonAKI_select - black_nonAKI_select
nonAKI_white_vs_non_white = white_nonAKI_select - non_white_nonAKI_select
nonAKI_white_vs_black.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_black_fairness_nonAKI_num.csv')
nonAKI_white_vs_non_white.to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_white_vs_non_white_fairness_nonAKI_num.csv')


