#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:09:26 2022

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

threshold_used = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

calibration_group_num = 10

model_used = ['global','personal']
result_name = {}
result_name['global'] = 'predict_proba'
result_name['subgroup'] = 'subgroup_proba'
result_name['personal'] = 'update_1921_mat_proba'

TPR_subgroup_record = pd.DataFrame()
FPR_subgroup_record = pd.DataFrame()
PPV_subgroup_record = pd.DataFrame()
general_result = pd.DataFrame()


subgroup_result = {}
subgroup_result_total = pd.DataFrame()
for disease_num in range(disease_list.shape[0]):
    
    subgroup_result[disease_list.iloc[disease_num,0]] = pd.read_csv('/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_list.iloc[disease_num,0]))
    subgroup_result_total = pd.concat([subgroup_result_total,subgroup_result[disease_list.iloc[disease_num,0]]])

subgroup_result_total_AKI_true = subgroup_result_total.loc[:,'Label'] == 1
subgroup_result_total_AKI = subgroup_result_total.loc[subgroup_result_total_AKI_true]
subgroup_result_total_nonAKI = subgroup_result_total.loc[~subgroup_result_total_AKI_true]


for model in model_used:
    
    for threshold in threshold_used:
        
        threshold_select_data_total = pd.DataFrame()
        
        for disease_num in range(disease_list.shape[0]):
            
            data_subgroup = subgroup_result[disease_list.iloc[disease_num,0]]
            data_subgroup_AKI_true = data_subgroup.loc[:,'Label'] == 1
            data_subgroup_AKI = data_subgroup.loc[data_subgroup_AKI_true]
            data_subgroup_nonAKI = data_subgroup.loc[~data_subgroup_AKI_true]
            
            threshold_num = int(data_subgroup_AKI.shape[0] * threshold)
            data_subgroup_AKI = data_subgroup_AKI.sort_values(result_name[model],ascending=False)
            data_subgroup_AKI.reset_index(drop=True, inplace=True)
            threshold_value = data_subgroup_AKI.loc[threshold_num,result_name[model]]
            
            data_break_threshold = data_subgroup.loc[:,result_name[model]] >= threshold_value
            data_break_threshold_select = data_subgroup.loc[data_break_threshold]
            
            threshold_select_data_total = pd.concat([threshold_select_data_total,data_break_threshold_select])
            
            data_select_AKI_true = data_break_threshold_select.loc[:,'Label'] == 1
            data_select_AKI = data_break_threshold_select.loc[data_select_AKI_true]
            data_select_nonAKI = data_break_threshold_select.loc[~data_select_AKI_true]
            
            TPR_subgroup_record.loc[disease_list.iloc[disease_num,0],'{}_{}'.format(model,threshold)] = data_select_AKI.shape[0] / data_subgroup_AKI.shape[0]
            FPR_subgroup_record.loc[disease_list.iloc[disease_num,0],'{}_{}'.format(model,threshold)] = data_select_nonAKI.shape[0] / data_subgroup_nonAKI.shape[0]
            PPV_subgroup_record.loc[disease_list.iloc[disease_num,0],'{}_{}'.format(model,threshold)] = data_select_AKI.shape[0] / data_break_threshold_select.shape[0]
            
        
        data_select_total_AKI_true = threshold_select_data_total.loc[:,'Label'] == 1
        data_select_total_AKI = threshold_select_data_total.loc[data_select_total_AKI_true]
        data_select_total_nonAKI = threshold_select_data_total.loc[~data_select_total_AKI_true]
        
        general_result.loc['AUROC_{}'.format(threshold),'{}_fair'.format(model)] = roc_auc_score(threshold_select_data_total['Label'],threshold_select_data_total.loc[:,result_name[model]])
        general_result.loc['AUPRC_{}'.format(threshold),'{}_fair'.format(model)] = average_precision_score(threshold_select_data_total['Label'],threshold_select_data_total.loc[:,result_name[model]])
        #observation, prediction = calibration_curve(threshold_select_data_total['Label'],threshold_select_data_total[result_name[model]], n_bins=calibration_group_num, strategy='quantile')
        #general_result.loc['brier_{}'.format(threshold),'{}_fair'.format(model)] = np.mean(np.square(np.array(observation - prediction)))
        general_result.loc['TPR_{}'.format(threshold),'{}_fair'.format(model)] = data_select_total_AKI.shape[0] / subgroup_result_total_AKI.shape[0]
        general_result.loc['FPR_{}'.format(threshold),'{}_fair'.format(model)] = data_select_total_nonAKI.shape[0] / subgroup_result_total_nonAKI.shape[0]
        general_result.loc['PPV_{}'.format(threshold),'{}_fair'.format(model)] = data_select_total_AKI.shape[0] / threshold_select_data_total.shape[0]


for model in model_used:
    
    sort_subgroup_total_AKI = subgroup_result_total_AKI.sort_values(result_name[model],ascending=False)
    sort_subgroup_total_AKI.reset_index(drop=True, inplace=True)
    
    for threshold in threshold_used:
        
        threshold_num = int(sort_subgroup_total_AKI.shape[0] * threshold)
        threshold_value = sort_subgroup_total_AKI.loc[threshold_num,result_name[model]]
        total_data_break_threshold = subgroup_result_total.loc[:,result_name[model]] >= threshold_value
        total_data_break_threshold_select = subgroup_result_total.loc[total_data_break_threshold]
        
        total_data_select_AKI_true = total_data_break_threshold_select.loc[:,'Label'] == 1
        total_data_select_AKI = total_data_break_threshold_select.loc[total_data_select_AKI_true]
        total_data_select_nonAKI = total_data_break_threshold_select.loc[~total_data_select_AKI_true]
        
        general_result.loc['AUROC_{}'.format(threshold),'{}_unfair'.format(model)] = roc_auc_score(total_data_break_threshold_select['Label'],total_data_break_threshold_select.loc[:,result_name[model]])
        general_result.loc['AUPRC_{}'.format(threshold),'{}_unfair'.format(model)] = average_precision_score(total_data_break_threshold_select['Label'],total_data_break_threshold_select.loc[:,result_name[model]])
        #observation, prediction = calibration_curve(threshold_select_data_total['Label'],threshold_select_data_total[result_name[model]], n_bins=calibration_group_num, strategy='quantile')
        #general_result.loc['brier_{}'.format(threshold),'{}_unfair'.format(model)] = np.mean(np.square(np.array(observation - prediction)))
        general_result.loc['TPR_{}'.format(threshold),'{}_unfair'.format(model)] = total_data_select_AKI.shape[0] / subgroup_result_total_AKI.shape[0]
        general_result.loc['FPR_{}'.format(threshold),'{}_unfair'.format(model)] = total_data_select_nonAKI.shape[0] / subgroup_result_total_nonAKI.shape[0]
        general_result.loc['PPV_{}'.format(threshold),'{}_unfair'.format(model)] = total_data_select_AKI.shape[0] / total_data_break_threshold_select.shape[0]

        
        