#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:43:24 2022

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve


test_total = pd.DataFrame()
for data_num in range(1,5):
    
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    test_total = pd.concat([test_total, test_data])

test_total.reset_index(drop=True,inplace=True)
test_result_total = pd.read_csv("/home/liukang/Doc/AUPRC/test_result_10_No_Com.csv")

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
disease_list_noDrg = pd.read_csv("/home/liukang/Doc/disease_top_20_no_drg_full_name.csv")

disease_race_list = disease_list.iloc[:,0].tolist()
race_list = ['Demo2_1','Demo2_2','Demo2_3','Demo2_4']
disease_race_list.extend(race_list)

threshold_used = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

calibration_group_num = 10

model_used = ['global','personal']
result_name = {}
result_name['global'] = 'predict_proba'
result_name['subgroup'] = 'subgroup_proba'
result_name['personal'] = 'update_1921_mat_proba'


test_total_all_subgroups_data = test_total.loc[:,disease_race_list]
test_total_all_subgroups_true = test_total_all_subgroups_data.sum(axis=1) >= 2
test_total_all_subgroups = test_total.loc[test_total_all_subgroups_true]

test_result_all_subgroups = test_result_total.loc[test_total_all_subgroups_true]

test_total_all_subgroups.reset_index(drop=True,inplace=True)
test_result_all_subgroups.reset_index(drop=True,inplace=True)

test_result_all_subgroups_AKI_true = test_result_all_subgroups.loc[:,'Label'] == 1
test_result_all_subgroups_AKI = test_result_all_subgroups.loc[test_result_all_subgroups_AKI_true]
test_result_all_subgroups_nonAKI = test_result_all_subgroups.loc[~test_result_all_subgroups_AKI_true]



subgroup_result_record = pd.DataFrame()
general_result = pd.DataFrame()
TPR_disease_record = pd.DataFrame()
FPR_disease_record = pd.DataFrame()
PPV_disease_record = pd.DataFrame()
for model in ['global','personal']:
    
    for threshold in threshold_used:
        
        threshold_select_data_total = pd.DataFrame()
        
        unfair_select_AKI = test_result_all_subgroups_AKI.sort_values(result_name[model],ascending=False)
        unfair_select_AKI.reset_index(drop=True, inplace=True)
        threshold_num = int(unfair_select_AKI.shape[0] * threshold)
        threshold_value = unfair_select_AKI.loc[threshold_num,result_name[model]]
        unfair_break_threshold = test_result_all_subgroups.loc[:,result_name[model]] >= threshold_value
        unfair_break_threshold_select = test_result_all_subgroups.loc[unfair_break_threshold]
        
        for race in race_list:
            
            data_subgroup_true = test_total_all_subgroups.loc[:,race] == 1
            data_subgroup = test_result_all_subgroups.loc[data_subgroup_true]
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
            
            subgroup_result_record.loc['{}_TPR'.format(race),'{}_{}'.format(model,threshold)] = data_select_AKI.shape[0] / data_subgroup_AKI.shape[0]
            subgroup_result_record.loc['{}_FPR'.format(race),'{}_{}'.format(model,threshold)] = data_select_nonAKI.shape[0] / data_subgroup_nonAKI.shape[0]
            subgroup_result_record.loc['{}_PPV'.format(race),'{}_{}'.format(model,threshold)] = data_select_AKI.shape[0] / data_break_threshold_select.shape[0]
            
        data_select_total_AKI_true = threshold_select_data_total.loc[:,'Label'] == 1
        data_select_total_AKI = threshold_select_data_total.loc[data_select_total_AKI_true]
        data_select_total_nonAKI = threshold_select_data_total.loc[~data_select_total_AKI_true]
        
        #general_result.loc['AUROC_{}'.format(threshold),'{}_fair'.format(model)] = roc_auc_score(threshold_select_data_total['Label'],threshold_select_data_total.loc[:,result_name[model]])
        #general_result.loc['AUPRC_{}'.format(threshold),'{}_fair'.format(model)] = average_precision_score(threshold_select_data_total['Label'],threshold_select_data_total.loc[:,result_name[model]])
        #observation, prediction = calibration_curve(threshold_select_data_total['Label'],threshold_select_data_total[result_name[model]], n_bins=calibration_group_num, strategy='quantile')
        #general_result.loc['brier_{}'.format(threshold),'{}_fair'.format(model)] = np.mean(np.square(np.array(observation - prediction)))
        general_result.loc['TPR_{}'.format(threshold),'{}_fair'.format(model)] = data_select_total_AKI.shape[0] / test_result_all_subgroups_AKI.shape[0]
        general_result.loc['FPR_{}'.format(threshold),'{}_fair'.format(model)] = data_select_total_nonAKI.shape[0] / test_result_all_subgroups_nonAKI.shape[0]
        general_result.loc['PPV_{}'.format(threshold),'{}_fair'.format(model)] = data_select_total_AKI.shape[0] / threshold_select_data_total.shape[0]
        
        for disease_num in range(disease_list.shape[0]):
            
            data_disease_true = test_result_all_subgroups.loc[:,'Drg'] == disease_list_noDrg.iloc[disease_num,0]
            data_disease_select = test_result_all_subgroups.loc[data_disease_true]
            data_disease_select_AKI_true = data_disease_select.loc[:,'Label'] == 1
            data_disease_select_AKI = data_disease_select.loc[data_disease_select_AKI_true]
            data_disease_select_nonAKI = data_disease_select.loc[~data_disease_select_AKI_true]
            
            data_disease_race_true = threshold_select_data_total.loc[:,'Drg'] == disease_list_noDrg.iloc[disease_num,0]
            data_disease_race_select = threshold_select_data_total.loc[data_disease_race_true]
            data_disease_race_select_AKI_true = data_disease_race_select.loc[:,'Label'] == 1
            data_disease_race_select_AKI = data_disease_race_select.loc[data_disease_select_AKI_true]
            data_disease_race_select_nonAKI = data_disease_race_select.loc[~data_disease_select_AKI_true]
            
            TPR_disease_record.loc[disease_list.iloc[disease_num,0],'fair_{}'.format(threshold)] = data_disease_race_select_AKI.shape[0] / data_disease_select_AKI.shape[0]
            FPR_disease_record.loc[disease_list.iloc[disease_num,0],'fair_{}'.format(threshold)] = data_disease_race_select_nonAKI.shape[0] / data_disease_select_nonAKI.shape[0]
            
            if data_disease_race_select.shape[0] == 0:
                PPV_disease_record.loc[disease_list.iloc[disease_num,0],'fair_{}'.format(threshold)] = 'NaN'
            else:
                PPV_disease_record.loc[disease_list.iloc[disease_num,0],'fair_{}'.format(threshold)] = data_disease_race_select_AKI.shape[0] / data_disease_race_select.shape[0]
                
            
            unfair_disease_true = unfair_break_threshold_select.loc[:,'Drg'] == disease_list_noDrg.iloc[disease_num,0]
            data_disease_break_threshold_select = unfair_break_threshold_select.loc[unfair_disease_true]
            data_disease_break_threshold_select_AKI_true = data_disease_break_threshold_select.loc[:,'Label'] ==1
            data_disease_break_threshold_select_AKI = data_disease_break_threshold_select.loc[data_disease_break_threshold_select_AKI_true]
            data_disease_break_threshold_select_nonAKI = data_disease_break_threshold_select.loc[~data_disease_break_threshold_select_AKI_true]
            
            TPR_disease_record.loc[disease_list.iloc[disease_num,0],'unfair_{}'.format(threshold)] = data_disease_break_threshold_select_AKI.shape[0] / data_disease_select_AKI.shape[0]
            FPR_disease_record.loc[disease_list.iloc[disease_num,0],'unfair_{}'.format(threshold)] = data_disease_break_threshold_select_nonAKI.shape[0] / data_disease_select_nonAKI.shape[0]
            
            if data_disease_race_select.shape[0] == 0:
                PPV_disease_record.loc[disease_list.iloc[disease_num,0],'unfair_{}'.format(threshold)] = 'NaN'
            else:
                PPV_disease_record.loc[disease_list.iloc[disease_num,0],'unfair_{}'.format(threshold)] = data_disease_break_threshold_select_AKI.shape[0] / data_disease_race_select.shape[0]
                