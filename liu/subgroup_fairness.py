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
threshold_used = ['560','1120','2241','self_5%','self_10%','self_20%', 'self_div_25%', 'self_div_50%','self_div_100%']

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

subgroup_aki_nums = []
subgroup_nums = []
for disease_num in range(disease_list.shape[0]):
    temp_df = pd.read_csv('/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_list.iloc[disease_num,0]))
    subgroup_aki_nums.append(temp_df['Label'].sum())
    subgroup_nums.append(temp_df.shape[0])
    subgroup_result[disease_list.iloc[disease_num,0]] = temp_df
    subgroup_result_total = pd.concat([subgroup_result_total,subgroup_result[disease_list.iloc[disease_num,0]]])

subgroup_aki_rate = [t1/t2 for (t1, t2) in zip(subgroup_aki_nums, subgroup_nums)]

for model in ['global','subgroup','personal']:
    
    subgroup_result_total.sort_values(result_name[model],inplace=True,ascending=False)
    subgroup_result_total.reset_index(drop=True, inplace=True)
    threshold_record.loc['{}_{}'.format(model,560),'threshold'] = subgroup_result_total.loc[560-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,1120),'threshold'] = subgroup_result_total.loc[1000-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,2241),'threshold'] = subgroup_result_total.loc[2000-1,result_name[model]]
    
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

            if threshold_id == 'self_5%':
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.05),result_name[model]]

            elif threshold_id == 'self_10%':

                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.1),result_name[model]]

            elif threshold_id == 'self_20%':
                
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.2),result_name[model]]

            elif threshold_id == 'self_div_25%':

                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] / 4), result_name[model]]

            elif threshold_id == 'self_div_50%':
                
                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] / 2),result_name[model]]
            
            elif threshold_id == 'self_div_100%':
                
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

# ======================================================================================
"""
['500','1120','2241','self_5%','self_10%','self_20%', 'self_%div4', 'self_%div2','self_%div']
获取绘画的dataframe格式  
x, y, hue
阈值选择大小, 指标数值, 建模策略类型, 阈值选择类型, 指标类型
"""
threshold_type_dict = {
    "同等阈值": ['500', '1120', '2241'],
    "同比例阈值": ['self_5%','self_10%','self_20%'],
    "根据各亚组AKI数量比例阈值": ['self_div_25%', 'self_div_50%','self_div_100%']
}

# plot_df = pd.DataFrame(columns=["threshold", "score", "建模策略", "threshold_type", "score_type", "drg"])
# for score_type in ['TPR', 'FPR', 'PPV']:
#     for threshold_type, thresholds in threshold_type_dict.items():
#         for threshold in thresholds:
#
#             cur_res_df = fairness_record[f"{score_type}_threshold_{threshold}"]
#
#             if threshold_type == "同比例阈值":
#                 threshold = threshold[5:]
#             elif threshold_type == "根据各亚组AKI数量比例阈值":
#                 threshold = threshold[9:]
#
#             cur_cols = cur_res_df.columns.tolist()
#             col_name_dict = {
#                 "global": "GM",
#                 "subgroup": "SM",
#                 "personal": "PMTL"
#             }
#             cur_indexes = cur_res_df.index.tolist()
#
#             for col in cur_cols:
#                 for index in cur_indexes:
#                     temp_df = pd.DataFrame(data={
#                         "threshold": [threshold],
#                         "threshold_type": [threshold_type],
#                         "score_type": [score_type],
#                         "drg": [index],
#                         "建模策略": [col_name_dict[col]],
#                         "score": [cur_res_df.loc[index, col]]
#                     })
#                     plot_df = pd.concat([plot_df, temp_df], axis=0, ignore_index=True)
# plot_df.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_to_plot.csv")

# =====================================================================================

for threshold_id in threshold_used:

    for fairness_measure in ['TPR']:

        measure_select = '{}_threshold_{}'.format(fairness_measure,threshold_id)
        measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
        measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0)
        measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
        fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_all_race_fairness_{}.csv'.format(measure_select))

for fairness_measure in ['TPR']:

    measure_select = '{}_no_threshold'.format(fairness_measure)
    measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
    measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0)
    measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
    fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_all_race_fairness_{}.csv'.format(measure_select))
    

# measure_avg_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_top20_fairness_avg.csv")
# measure_std_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_top20_fairness_std.csv")
# measure_CV_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_top20_fairness_CV.csv")
# AKI_select_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_top20_fairness_AKI_num.csv")
# nonAKI_select_record.to_csv("/home/liukang/Doc/fairness_analysis/subgroup_top20_fairness_nonAKI_num.csv")

measure_avg_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_avg.csv")
measure_std_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_std.csv")
measure_CV_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_CV.csv")
AKI_select_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_AKI_num.csv")
nonAKI_select_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_nonAKI_num.csv")


# ===================================================================================
# 根据每个亚组AKI数量选取决策阈值召回率
