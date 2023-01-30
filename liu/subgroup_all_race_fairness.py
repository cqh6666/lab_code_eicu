#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:26:34 2022

@author: liukang
"""

import numpy as np
import pandas as pd

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

subgroup_result = {}
subgroup_result_total = pd.DataFrame()

result_name = {}
result_name['global'] = 'predict_proba'
result_name['subgroup'] = 'subgroup_proba'
result_name['personal'] = 'update_1921_mat_proba'

threshold_record = pd.DataFrame()
threshold_used = ['500','1120','2241','self_10%','self_20%','self_%div2','self_%']

AKI_select_record = pd.DataFrame(index=threshold_used, columns=['global','subgroup','personal'])
nonAKI_select_record = pd.DataFrame(index=threshold_used ,columns=['global','subgroup','personal'])
AKI_select_record.loc[:,:] = 0
nonAKI_select_record.loc[:,:] = 0

fairness_record = {}
for threshold_num in threshold_used:
    
    for fairness_measure in ['TPR','FPR','odds','PPV']:
        
        fairness_record['{}_threshold_{}'.format(fairness_measure,threshold_num)] = pd.DataFrame()
        
for fairness_measure in ['TPR','FPR','odds']:
    
    fairness_record['{}_no_threshold'.format(fairness_measure)] = pd.DataFrame()

for disease_num in range(disease_list.shape[0]):
    
    subgroup_result[disease_list.iloc[disease_num,0]] = pd.read_csv('/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_list.iloc[disease_num,0]))
    subgroup_result_total = pd.concat([subgroup_result_total,subgroup_result[disease_list.iloc[disease_num,0]]])

for model in ['global','subgroup','personal']:
    
    subgroup_result_total.sort_values(result_name[model],inplace=True,ascending=False)
    subgroup_result_total.reset_index(drop=True, inplace=True)
    threshold_record.loc['{}_{}'.format(model,500),'threshold'] = subgroup_result_total.loc[500-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,1120),'threshold'] = subgroup_result_total.loc[1120-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,2241),'threshold'] = subgroup_result_total.loc[2241-1,result_name[model]]
    
for disease_num in range(disease_list.shape[0]):
    
    data_subgroup = subgroup_result[disease_list.iloc[disease_num,0]]
    data_subgroup_AKI_true = data_subgroup.loc[:,'Label'] == 1
    data_subgroup_AKI = data_subgroup.loc[data_subgroup_AKI_true]
    data_subgroup_nonAKI = data_subgroup.loc[~data_subgroup_AKI_true]
    
    for model in ['global','subgroup','personal']:
        
        data_subgroup = data_subgroup.sort_values(result_name[model],ascending=False)
        data_subgroup.reset_index(drop=True, inplace=True)
        
        # fairness_measure_without_threshold
        equal_TPR_no_threshold = np.mean(data_subgroup_AKI.loc[:,result_name[model]])
        equal_FPR_no_threshold = np.mean(data_subgroup_nonAKI.loc[:,result_name[model]])
        equal_odds_no_threshold = equal_TPR_no_threshold + (1-equal_FPR_no_threshold)
        
        fairness_record['TPR_no_threshold'].loc[disease_list.iloc[disease_num,0],model] = equal_TPR_no_threshold
        fairness_record['FPR_no_threshold'].loc[disease_list.iloc[disease_num,0],model] = equal_FPR_no_threshold
        fairness_record['odds_no_threshold'].loc[disease_list.iloc[disease_num,0],model] = equal_odds_no_threshold
        
        #fairness_measure_all_subgroup_use_same_threshold
        for threshold_id in threshold_used:
            
            if threshold_id == 'self_10%':
                
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.1),result_name[model]]
                
            elif threshold_id == 'self_20%':
                
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.2),result_name[model]]
            
            elif threshold_id == 'self_%div2':
                
                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] / 2),result_name[model]]
            
            elif threshold_id == 'self_%':
                
                threshold_value = data_subgroup.loc[data_subgroup_AKI.shape[0]-1,result_name[model]]
                
            else:
                
                threshold_value = threshold_record.loc['{}_{}'.format(model,threshold_id),'threshold']
            
            data_subgroup_break_threshold = data_subgroup.loc[:,result_name[model]] >= threshold_value
            data_subgroup_select = data_subgroup.loc[data_subgroup_break_threshold]
            data_subgroup_not_select = data_subgroup.loc[~data_subgroup_break_threshold]
            
            select_AKI_num = np.sum(data_subgroup_select.loc[:,'Label'])
            select_nonAKI_num = data_subgroup_select.shape[0] - select_AKI_num
            
            AKI_select_record.loc[threshold_id,model] = AKI_select_record.loc[threshold_id,model] + select_AKI_num
            nonAKI_select_record.loc[threshold_id,model] = nonAKI_select_record.loc[threshold_id,model] + select_nonAKI_num
            
            equal_TPR_threshold = select_AKI_num / data_subgroup_AKI.shape[0]
            equal_FPR_threshold = select_nonAKI_num / data_subgroup_nonAKI.shape[0]
            equal_odds_threshold = equal_TPR_threshold + (1-equal_FPR_threshold)
            
            equal_PPV_threshold = select_AKI_num / data_subgroup_select.shape[0]
            
            fairness_record['TPR_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_TPR_threshold
            fairness_record['FPR_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_FPR_threshold
            fairness_record['odds_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_odds_threshold
            fairness_record['PPV_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_PPV_threshold
            

measure_avg_record = pd.DataFrame(columns=['global','subgroup','personal'])
measure_std_record = pd.DataFrame(columns=['global','subgroup','personal'])
measure_CV_record = pd.DataFrame(columns=['global','subgroup','personal'])
for threshold_id in threshold_used:
    
    for fairness_measure in ['TPR','FPR','odds','PPV']:
        
        measure_select = '{}_threshold_{}'.format(fairness_measure,threshold_id)
        measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
        measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0,ddof=0)
        measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
        fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_{}.csv'.format(measure_select))
        
for fairness_measure in ['TPR','FPR','odds']:
    
    measure_select = '{}_no_threshold'.format(fairness_measure)
    measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
    measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0,ddof=0)
    measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
    fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_{}.csv'.format(measure_select))
    
AKI_select_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_AKI_raw_num.csv")
nonAKI_select_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_fairness_all_race_nonAKI_raw_num.csv")
AKI_select_record = AKI_select_record / np.sum(subgroup_result_total['Label'])
nonAKI_select_record = nonAKI_select_record / (subgroup_result_total.shape[0] - np.sum(subgroup_result_total['Label']))


measure_avg_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_avg.csv")
measure_std_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_std.csv")
measure_CV_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_CV.csv")
AKI_select_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_all_race_fairness_AKI_num.csv")
nonAKI_select_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_C005_top20_fairness_all_race_nonAKI_num.csv")