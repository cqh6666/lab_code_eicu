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
# threshold_used = ['self_subgroup_5%', 'self_subgroup_10%', 'self_subgroup_20%', 'self_subgroup_aki_25%', 'self_subgroup_aki_50%','self_subgroup_aki_100%']

# 选择公平性最差进行分析
# threshold_used = ['560', '1120', '2241', 'self_subgroup_aki_50%', 'self_subgroup_aki_75%', 'self_subgroup_aki_100%']
threshold_used = ['2241']

AKI_select_record = pd.DataFrame(index=threshold_used, columns=['global','subgroup','personal'])
break_select_record = pd.DataFrame(index=threshold_used, columns=['global','subgroup','personal'])
nonAKI_select_record = pd.DataFrame(index=threshold_used ,columns=['global','subgroup','personal'])
AKI_select_record.loc[:,:] = 0
nonAKI_select_record.loc[:,:] = 0

fairness_record = {}
for threshold_num in threshold_used:
    
    for fairness_measure in ['TPR','FPR','odds','PPV']:
        
        fairness_record['{}_threshold_{}'.format(fairness_measure,threshold_num)] = pd.DataFrame()
        
for fairness_measure in ['TPR','FPR','odds']:
    
    fairness_record['{}_no_threshold'.format(fairness_measure)] = pd.DataFrame()

subgroup_nums_record = pd.DataFrame()
for disease_num in range(disease_list.shape[0]):
    temp_df = pd.read_csv('/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_list.iloc[disease_num,0]))
    subgroup_nums_record.loc[disease_list.iloc[disease_num,0], "aki_nums"] = temp_df['Label'].sum()
    subgroup_nums_record.loc[disease_list.iloc[disease_num,0], "people_nums"] = temp_df.shape[0]
    subgroup_result[disease_list.iloc[disease_num,0]] = temp_df
    subgroup_result_total = pd.concat([subgroup_result_total,subgroup_result[disease_list.iloc[disease_num,0]]])

# subgroup_nums_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_people_aki_nums.csv")

for model in ['global','subgroup','personal']:

    subgroup_result_total.sort_values(result_name[model],inplace=True,ascending=False)
    subgroup_result_total.reset_index(drop=True, inplace=True)
    threshold_record.loc['{}_{}'.format(model,560),'threshold'] = subgroup_result_total.loc[560-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,1120),'threshold'] = subgroup_result_total.loc[1120-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,2241),'threshold'] = subgroup_result_total.loc[2241-1,result_name[model]]


for disease_num in range(disease_list.shape[0]):
    
    disease_id_name = disease_list.iloc[disease_num, 0]
    data_subgroup = subgroup_result[disease_id_name]
    data_subgroup_AKI_true = data_subgroup.loc[:,'Label'] == 1
    data_subgroup_AKI = data_subgroup.loc[data_subgroup_AKI_true]
    data_subgroup_nonAKI = data_subgroup.loc[~data_subgroup_AKI_true]
    
    for model in ['global', 'personal']:
        
        data_subgroup = data_subgroup.sort_values(result_name[model],ascending=False)
        data_subgroup.reset_index(drop=True, inplace=True)
        
        # fairness_measure_without_threshold
        equal_TPR_no_threshold = np.mean(data_subgroup_AKI.loc[:,result_name[model]])
        equal_FPR_no_threshold = np.mean(data_subgroup_nonAKI.loc[:,result_name[model]])
        equal_odds_no_threshold = equal_TPR_no_threshold + (1-equal_FPR_no_threshold)
        
        fairness_record['TPR_no_threshold'].loc[disease_id_name, model] = equal_TPR_no_threshold
        fairness_record['FPR_no_threshold'].loc[disease_id_name, model] = equal_FPR_no_threshold
        fairness_record['odds_no_threshold'].loc[disease_id_name, model] = equal_odds_no_threshold
        
        #fairness_measure_all_subgroup_use_same_threshold
        for threshold_id in threshold_used:

            if threshold_id == 'self_subgroup_5%':
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.05),result_name[model]]

            elif threshold_id == 'self_subgroup_10%':

                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.1),result_name[model]]

            elif threshold_id == 'self_subgroup_20%':
                
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.2),result_name[model]]

            elif threshold_id == 'self_subgroup_aki_50%':
                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] * 0.5),result_name[model]]
            elif threshold_id == 'self_subgroup_aki_75%':
                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] * 0.75),result_name[model]]
            elif threshold_id == 'self_subgroup_aki_100%':
                threshold_value = data_subgroup.loc[data_subgroup_AKI.shape[0]-1,result_name[model]]
                
            else:
                
                threshold_value = threshold_record.loc['{}_{}'.format(model,threshold_id),'threshold']
            
            data_subgroup_break_threshold = data_subgroup.loc[:,result_name[model]] >= threshold_value
            data_subgroup_select = data_subgroup.loc[data_subgroup_break_threshold]
            data_subgroup_not_select = data_subgroup.loc[~data_subgroup_break_threshold]
            
            select_AKI_num = np.sum(data_subgroup_select.loc[:,'Label'])
            select_nonAKI_num = data_subgroup_select.shape[0] - select_AKI_num

            break_select_record.loc[disease_id_name, model] = data_subgroup_select.shape[0]
            AKI_select_record.loc[disease_id_name, model] = select_AKI_num
            # AKI_select_record.loc[threshold_id,model] = AKI_select_record.loc[threshold_id,model] + select_AKI_num
            nonAKI_select_record.loc[threshold_id,model] = nonAKI_select_record.loc[threshold_id,model] + select_nonAKI_num
            
            equal_TPR_threshold = select_AKI_num / data_subgroup_AKI.shape[0]
            equal_FPR_threshold = select_nonAKI_num / data_subgroup_nonAKI.shape[0]
            equal_odds_threshold = equal_TPR_threshold + (1-equal_FPR_threshold)
            
            equal_PPV_threshold = select_AKI_num / data_subgroup_select.shape[0]
            
            fairness_record['TPR_threshold_{}'.format(threshold_id)].loc[disease_id_name, model] = equal_TPR_threshold
            fairness_record['FPR_threshold_{}'.format(threshold_id)].loc[disease_id_name, model] = equal_FPR_threshold
            fairness_record['odds_threshold_{}'.format(threshold_id)].loc[disease_id_name, model] = equal_odds_threshold
            fairness_record['PPV_threshold_{}'.format(threshold_id)].loc[disease_id_name, model] = equal_PPV_threshold


measure_avg_record = pd.DataFrame(columns=['global','subgroup','personal'])
measure_std_record = pd.DataFrame(columns=['global','subgroup','personal'])
measure_CV_record = pd.DataFrame(columns=['global','subgroup','personal'])

# ======================================================================================
"""
['500','1120','2241','self_subgroup_aki_25%', 'self_subgroup_aki_50%','self_subgroup_aki_100%']
获取绘画的dataframe格式  
x, y, hue
阈值选择大小, 指标数值, 建模策略类型, 阈值选择类型, 指标类型
"""
# threshold_type_dict = {
#     "Top-K from all subgroups": [560, 1120, 2241],
#     "Top-K% incidence rate from each subgroup": ['self_subgroup_aki_50%', 'self_subgroup_aki_75%', 'self_subgroup_aki_100%']
# }
#
# plot_df = pd.DataFrame(columns=["threshold", "score", "build_type", "threshold_type", "score_type", "drg"])
# for score_type in ['TPR']:
#     for threshold_type, thresholds in threshold_type_dict.items():
#         for threshold in thresholds:
#
#             cur_res_df = fairness_record[f"{score_type}_threshold_{threshold}"]
#
#             if threshold_type == "Top-K% incidence rate from each subgroup":
#                 threshold = threshold[18:]
#             elif threshold_type == "Top-K% rate from each subgroups":
#                 threshold = threshold[14:]
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
#                         "build_type": [col_name_dict[col]],
#                         "score": [cur_res_df.loc[index, col]]
#                     })
#                     plot_df = pd.concat([plot_df, temp_df], axis=0, ignore_index=True)
# plot_df.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_TPR_records_3-25.csv")

# =====================================================================================

for threshold_id in threshold_used:

    for fairness_measure in ['TPR']:

        measure_select = '{}_threshold_{}'.format(fairness_measure,threshold_id)
        measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
        measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0)
        measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
        # fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_all_race_fairness_{}.csv'.format(measure_select))

for fairness_measure in ['TPR']:

    measure_select = '{}_no_threshold'.format(fairness_measure)
    measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
    measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0)
    measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
    # fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_all_race_fairness_{}.csv'.format(measure_select))
    

# measure_avg_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_avg.csv")
# measure_std_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_std.csv")
# measure_CV_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_CV_3-25.csv")
# AKI_select_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_top2241_AKI_num.csv")
break_select_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_top2241_break_num.csv")
# nonAKI_select_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_nonAKI_num.csv")


# ===================================================================================
# 根据每个亚组AKI数量选取决策阈值召回率
