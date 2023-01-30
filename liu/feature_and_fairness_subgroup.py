#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:54:24 2022

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from gc import collect
from time import sleep
import os
import warnings
warnings.filterwarnings('ignore')

def aurcc_calculator(target_score,baseline_score):
    
    target_recall_list = []
    
    for i in range(len(baseline_score)):
        
        threshold_used = baseline_score[i]
        target_score_break_threshold = [j >= threshold_used for j in target_score]
        target_recall = np.sum(target_score_break_threshold) / len(target_score)
        target_recall_list.append(target_recall)
        
    #area under recall compare curve
    aurcc = np.mean(target_recall_list)
    return aurcc

def fairness_with_view_only(score_used,subgroup_used,view_id_used):
    
    #select score in view only
    #score in subgroup
    subgroup_true = all_subgroup_data.loc[:,subgroup_used] >= 1
    AKI_true = all_subgroup_data.loc[:,'Label'] == 1
    subgroup_final_true = subgroup_true & AKI_true
    subgroup_score_used = score_used.loc[subgroup_final_true]
    subgroup_score_used_fs = subgroup_score_used.loc[:,[feature_name[view_id_used]]].copy()
    selected_column_names = feature_name[view_id_used]
    subgroup_score_used_fs['predict_score'] = subgroup_score_used_fs.iloc[:,0]
    subgroup_AKI_average_score = np.mean(subgroup_score_used_fs['predict_score'].tolist())
    
    #score in remaining patients
    remain_final_true = (~subgroup_true) & AKI_true
    remain_score_used = score_used.loc[remain_final_true]
    remain_score_used_fs = remain_score_used.loc[:,[feature_name[view_id_used]]].copy()
    remain_score_used_fs['predict_score'] = remain_score_used_fs.iloc[:,0]
    
    #calculate area under recall compare curve
    subgroup_score_used_fs.sort_values('predict_score',ascending=False,inplace=True)
    subgroup_score_used_fs.reset_index(drop=True,inplace=True)
    remain_score_used_fs.sort_values('predict_score',ascending=False,inplace=True)
    remain_score_used_fs.reset_index(drop=True,inplace=True)
    
    subgroup_predict_score = subgroup_score_used_fs['predict_score'].tolist()
    remain_predict_score = remain_score_used_fs['predict_score'].tolist()
    
    subgroup_aurcc = aurcc_calculator(target_score=subgroup_predict_score,baseline_score=remain_predict_score)
    
    
    #select score drop view only
    drop_columns = selected_column_names
    subgroup_score_used_noView = subgroup_score_used.drop(drop_columns, axis=1)
    subgroup_score_used_noView['predict_score'] = subgroup_score_used_noView.sum(axis=1)
    subgroup_AKI_average_score_noView = np.mean(subgroup_score_used_noView['predict_score'].tolist())
    
    remain_score_used_noView = remain_score_used.drop(drop_columns, axis=1)
    remain_score_used_noView['predict_score'] = remain_score_used_noView.sum(axis=1)
    
    subgroup_score_used_noView.sort_values('predict_score',ascending=False,inplace=True)
    subgroup_score_used_noView.reset_index(drop=True,inplace=True)
    remain_score_used_noView.sort_values('predict_score',ascending=False,inplace=True)
    remain_score_used_noView.reset_index(drop=True,inplace=True)
    
    subgroup_predict_score_noView = subgroup_score_used_noView['predict_score'].tolist()
    remain_predict_score_noView = remain_score_used_noView['predict_score'].tolist()
    
    subgroup_aurcc_noView = aurcc_calculator(target_score=subgroup_predict_score_noView,baseline_score=remain_predict_score_noView)
    
    return subgroup_AKI_average_score, subgroup_AKI_average_score_noView, subgroup_aurcc, subgroup_aurcc_noView




disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')


#fairness between Drg subgroups
fairness_target = 'subgroups'
subgroup_list = disease_list.iloc[:,0].tolist()
select_subgroup_data_feature = subgroup_list
select_subgroup_data_feature_standard = 1
view_result_record = pd.DataFrame(columns=['AKI_score_global_view_CV','AKI_score_global_noview_CV','AKI_score_person_view_CV','AKI_score_person_noview_CV','AURCC_global_view_CV','AURCC_global_noview_CV','AURCC_person_view_CV','AURCC_person_noview_CV'])
'''
#fairness between race
subgroup_list = ['Demo2_1','Demo2_2']
fairness_target = 'race'
select_subgroup_data_feature = subgroup_list.copy()
select_subgroup_data_feature.extend(disease_list.iloc[:,0].tolist())
select_subgroup_data_feature_standard = 2
view_result_record = pd.DataFrame(columns=['AKI_score_global_view_diff','AKI_score_global_noview_diff','AKI_score_person_view_diff','AKI_score_person_noview_diff','AURCC_global_view_diff','AURCC_global_noview_diff','AURCC_person_view_diff','AURCC_person_noview_diff'])
'''

feature_name = test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_1.csv')
feature_name.drop('Label',axis=1,inplace=True)
feature_name = feature_name.columns.tolist()
feature_name.append('intercept')
#view_name = ['demo','race','gender','vital','Lab','Drg','Med','CCS']
#view_start = ['Demo2_1','Demo2_1','Demo3_1','Vital1','Lab1','Drg0','Med0','CCS0']
#view_end = ['Demo1','Demo2_4','Demo3_2','Vital5','Lab14','Drg314','Med1270','CCS279']

feature_list = pd.read_csv('/home/liukang/Doc/valid_df/test_1.csv')
feature_list.drop(['Label'],axis=1,inplace=True)
feature_name = np.append(feature_list.columns.tolist(),'intercept')

global_score_record = pd.DataFrame()
personal_score_record = pd.DataFrame()
test_total = pd.DataFrame()

for data_num in range(1,5):
    #read original data
    test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    test_total = pd.concat([test_total,test_df])
    
    test_original = test_df.copy()
    test_original = test_original.drop(['Label'], axis=1)
    test_original['intercept'] = 1
    
    #global modeling
    train_original = train_df.copy()
    lr_All = LogisticRegression(n_jobs=-1)
    X_train = train_original.drop(['Label'], axis=1)
    y_train = train_original['Label']
    lr_All.fit(X_train, y_train)
    #test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]
    #test_original['update_1921_mat_proba'] = person_result['update_1921_mat_proba']
    global_coef = lr_All.coef_[0]
    global_intercept = lr_All.intercept_
    global_final_weight = np.append(global_coef,global_intercept)
    global_score = test_original * global_final_weight
    global_score_record = pd.concat([global_score_record,global_score])
    
    
    #read personal coef_ and intercept and result
    person_coef = pd.read_csv('/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num)) 
    person_result = pd.read_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
    person_coef['intercept'] = person_result['update_1921_mat_intercept']
    #personal_score
    personal_score = np.multiply(test_original, person_coef)
    personal_score_record = pd.concat([personal_score_record, personal_score])
    
    

test_total.reset_index(drop=True,inplace=True)
global_score_record.reset_index(drop=True,inplace=True)
personal_score_record.reset_index(drop=True,inplace=True)

all_subgroup_feature_data = test_total.loc[:,select_subgroup_data_feature]
all_subgroup_data_sum = all_subgroup_feature_data.sum(axis=1)
all_subgroup_data_true = all_subgroup_data_sum >= select_subgroup_data_feature_standard
all_subgroup_data = test_total.loc[all_subgroup_data_true]

all_subgroup_global_score_record = global_score_record.loc[all_subgroup_data_true]
all_subgroup_personal_score_record = personal_score_record.loc[all_subgroup_data_true]



for view_id in range(len(feature_name)):
    
    view_avg_AKI_score_result = pd.DataFrame()
    view_aurcc_result = pd.DataFrame()
    
    for subgroup in subgroup_list:
        
        global_view_AKI_score, global_noview_AKI_score, global_view_aurcc, global_noview_aurcc = fairness_with_view_only(score_used=all_subgroup_global_score_record,subgroup_used=subgroup,view_id_used=view_id)
        view_avg_AKI_score_result.loc[subgroup,'global_view_only'] = global_view_AKI_score
        view_avg_AKI_score_result.loc[subgroup,'global_noview_only'] = global_noview_AKI_score
        view_aurcc_result.loc[subgroup,'global_view_only'] = global_view_aurcc
        view_aurcc_result.loc[subgroup,'global_noview_only'] = global_noview_aurcc
        
        
        person_view_AKI_score, person_noview_AKI_score, person_view_aurcc, person_noview_aurcc = fairness_with_view_only(score_used=all_subgroup_personal_score_record,subgroup_used=subgroup,view_id_used=view_id)
        view_avg_AKI_score_result.loc[subgroup,'person_view_only'] = person_view_AKI_score
        view_avg_AKI_score_result.loc[subgroup,'person_noview_only'] = person_noview_AKI_score
        view_aurcc_result.loc[subgroup,'person_view_only'] = person_view_aurcc
        view_aurcc_result.loc[subgroup,'person_noview_only'] = person_noview_aurcc
        
    #view_avg_AKI_score_result.to_csv('/home/liukang/Doc/fairness_analysis/{}_AKI_avg_score_w_or_wo_{}.csv'.format(fairness_target,feature_name[view_id]))
    #view_aurcc_result.to_csv('/home/liukang/Doc/fairness_analysis/{}_aurcc_w_or_wo_{}.csv'.format(fairness_target,feature_name[view_id]))
    
    if fairness_target == 'subgroups':
        
        score_std = view_avg_AKI_score_result.std(ddof=0)
        score_mean = view_avg_AKI_score_result.mean().abs()
        view_result_record.loc[feature_name[view_id],'AKI_score_global_view_CV':'AKI_score_person_noview_CV'] = np.array(score_std.values) / np.array(score_mean.values)
        
        aurcc_std = view_aurcc_result.std(ddof=0)
        aurcc_mean = view_aurcc_result.mean()
        view_result_record.loc[feature_name[view_id],'AURCC_global_view_CV':'AURCC_person_noview_CV'] = np.array(aurcc_std) / np.array(aurcc_mean.values)
        
    elif fairness_target == 'race':
        
        view_result_record.loc[feature_name[view_id],'AKI_score_global_view_diff':'AKI_score_person_noview_diff'] = np.array(view_avg_AKI_score_result.iloc[0,:] - view_avg_AKI_score_result.iloc[1,:])
        view_result_record.loc[feature_name[view_id],'AURCC_global_view_diff':'AURCC_person_noview_diff'] = np.array(view_aurcc_result.iloc[0,:] - view_aurcc_result.iloc[1,:])
        
view_result_record.to_csv('/home/liukang/Doc/fairness_analysis/{}_aurcc_AKIscore_change_feature.csv'.format(fairness_target))