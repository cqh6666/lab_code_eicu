#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:26:34 2022

@author: liukang
"""

import numpy as np
import pandas as pd

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

race = 'black'
race_id = 'Demo2_2'

result_name = {}
result_name['global'] = 'predict_proba'
result_name['subgroup'] = 'subgroup_proba'
result_name['personal'] = 'update_1921_mat_proba'

threshold_record = pd.DataFrame()
threshold_used = ['500', '1120', '2241', 'self_5%', 'self_10%', 'self_20%', 'self_race_5%', 'self_race_10%', 'self_race_20%', 'self_race_div_25%', 'self_race_div_50%','self_race_div_100%']


AKI_select_record = pd.DataFrame(index=threshold_used, columns=['global','subgroup','personal'])
nonAKI_select_record = pd.DataFrame(index=threshold_used ,columns=['global','subgroup','personal'])
AKI_select_record.loc[:,:] = 0
nonAKI_select_record.loc[:,:] = 0

#all test data
test_total = pd.DataFrame()
for data_num in range(1,5):
    
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    test_total = pd.concat([test_total, test_data])


#build pd to record result
fairness_record = {}
for threshold_num in threshold_used:
    
    for fairness_measure in ['TPR','FPR','odds','PPV']:
        
        fairness_record['{}_threshold_{}'.format(fairness_measure,threshold_num)] = pd.DataFrame()
        
for fairness_measure in ['TPR','FPR','odds']:
    
    fairness_record['{}_no_threshold'.format(fairness_measure)] = pd.DataFrame()

#subgroup and race identify
subgroup_result = {} #sample in a subgroup in a race
subgroup_all_race_result = {} #sample in subgroup in all race
subgroup_result_total = pd.DataFrame()
subgroup_all_race_result_total = pd.DataFrame()
for disease_num in range(disease_list.shape[0]):
    
    subgroup_feature_true = test_total.loc[:,disease_list.iloc[disease_num,0]]>0
    subgroup_data_select = test_total.loc[subgroup_feature_true]
    subgroup_data_select = subgroup_data_select.reset_index(drop=True)
    
    subgroup_ori_result = pd.read_csv('/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_list.iloc[disease_num,0]))
    subgroup_all_race_result_total = pd.concat([subgroup_all_race_result_total,subgroup_ori_result])
    subgroup_all_race_result[disease_list.iloc[disease_num,0]] = subgroup_ori_result
    
    print (subgroup_ori_result['Label'].tolist() == subgroup_data_select['Label'].tolist())
    
    subgroup_race_true = subgroup_data_select.loc[:,race_id] == 1
    subgroup_race_select_result = subgroup_ori_result.loc[subgroup_race_true]
    subgroup_race_select_result = subgroup_race_select_result.reset_index(drop=True)
    
    subgroup_result[disease_list.iloc[disease_num,0]] = subgroup_race_select_result
    subgroup_result_total = pd.concat([subgroup_result_total,subgroup_result[disease_list.iloc[disease_num,0]]])

#threshold when all subgroup share all selected patients, 2241 is the AKI num in all top20 subgroups
for model in ['global','subgroup','personal']:
    
    subgroup_all_race_result_total.sort_values(result_name[model],inplace=True,ascending=False)
    subgroup_all_race_result_total.reset_index(drop=True, inplace=True)
    threshold_record.loc['{}_{}'.format(model,500),'threshold'] = subgroup_all_race_result_total.loc[500-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,1120),'threshold'] = subgroup_all_race_result_total.loc[1120-1,result_name[model]]
    threshold_record.loc['{}_{}'.format(model,2241),'threshold'] = subgroup_all_race_result_total.loc[2241-1,result_name[model]]

#fairness compute  
for disease_num in range(disease_list.shape[0]):
    # 已经筛选了黑人群体
    data_subgroup = subgroup_result[disease_list.iloc[disease_num,0]]
    data_subgroup_AKI_true = data_subgroup.loc[:,'Label'] == 1
    data_subgroup_AKI = data_subgroup.loc[data_subgroup_AKI_true]
    data_subgroup_nonAKI = data_subgroup.loc[~data_subgroup_AKI_true]
    
    data_all_race_subgroup = subgroup_all_race_result[disease_list.iloc[disease_num,0]]
    data_all_race_subgroup_AKI_true = data_all_race_subgroup.loc[:,'Label'] == 1
    data_all_race_subgroup_AKI = data_all_race_subgroup.loc[data_all_race_subgroup_AKI_true]
    data_all_race_subgroup_nonAKI = data_all_race_subgroup.loc[~data_all_race_subgroup_AKI_true]
    
    for model in ['global','subgroup','personal']:
        # 各亚组筛选了黑人
        data_subgroup = data_subgroup.sort_values(result_name[model],ascending=False)
        data_subgroup.reset_index(drop=True, inplace=True)
        # 各亚组没有筛选黑人
        data_all_race_subgroup = data_all_race_subgroup.sort_values(result_name[model],ascending=False)
        data_all_race_subgroup.reset_index(drop=True, inplace=True)
        
        # fairness_measure_without_threshold
        equal_TPR_no_threshold = np.mean(data_subgroup_AKI.loc[:,result_name[model]])
        equal_FPR_no_threshold = np.mean(data_subgroup_nonAKI.loc[:,result_name[model]])
        equal_odds_no_threshold = equal_TPR_no_threshold + (1-equal_FPR_no_threshold)
        
        fairness_record['TPR_no_threshold'].loc[disease_list.iloc[disease_num,0],model] = equal_TPR_no_threshold
        fairness_record['FPR_no_threshold'].loc[disease_list.iloc[disease_num,0],model] = equal_FPR_no_threshold
        fairness_record['odds_no_threshold'].loc[disease_list.iloc[disease_num,0],model] = equal_odds_no_threshold
        
        #fairness measure use threshold, all race use same threshold
        for threshold_id in threshold_used:
            
            if threshold_id == 'self_10%':
                threshold_value = data_all_race_subgroup.loc[int(data_all_race_subgroup.shape[0] * 0.1),result_name[model]]
            elif threshold_id == 'self_5%':
                threshold_value = data_all_race_subgroup.loc[int(data_all_race_subgroup.shape[0] * 0.05), result_name[model]]
            elif threshold_id == 'self_20%':
                
                threshold_value = data_all_race_subgroup.loc[int(data_all_race_subgroup.shape[0] * 0.2),result_name[model]]
            
            elif threshold_id == 'self_%div2':
                
                threshold_value = data_all_race_subgroup.loc[int(data_all_race_subgroup_AKI.shape[0] / 2),result_name[model]]
            
            elif threshold_id == 'self_%':
                
                threshold_value = data_all_race_subgroup.loc[data_all_race_subgroup_AKI.shape[0]-1,result_name[model]]

            elif threshold_id == 'self_race_5%':
                # 各亚组的黑人群体的10%作为阈值点
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.05),result_name[model]]
            elif threshold_id == 'self_race_10%':
                # 各亚组的黑人群体的10%作为阈值点
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.1),result_name[model]]
                
            elif threshold_id == 'self_race_20%':
                
                threshold_value = data_subgroup.loc[int(data_subgroup.shape[0] * 0.2),result_name[model]]

            elif threshold_id == 'self_race_div_25%':
                # 各亚组的黑人群体的AKI患者前25%作为阈值点
                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] / 4),result_name[model]]

            elif threshold_id == 'self_race_div_50%':
                # 各亚组的黑人群体的AKI患者前50%作为阈值点
                threshold_value = data_subgroup.loc[int(data_subgroup_AKI.shape[0] / 2),result_name[model]]

            elif threshold_id == 'self_race_div_100%':
                # 各亚组的黑人群体的AKI患者的人数作为阈值点
                threshold_value = data_subgroup.loc[data_subgroup_AKI.shape[0]-1,result_name[model]]
                
            else:
                
                threshold_value = threshold_record.loc['{}_{}'.format(model,threshold_id),'threshold']
            # 从筛选了黑人的亚群群体种找大于阈值的样本
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
            
            # equal_PPV_threshold = select_AKI_num / data_subgroup_select.shape[0]
            
            fairness_record['TPR_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_TPR_threshold
            fairness_record['FPR_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_FPR_threshold
            fairness_record['odds_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_odds_threshold
            # fairness_record['PPV_threshold_{}'.format(threshold_id)].loc[disease_list.iloc[disease_num,0],model] = equal_PPV_threshold
            

threshold_type_dict = {
    "同等阈值": ['500', '1120', '2241'],
    "同比例阈值": ['self_5%', 'self_10%', 'self_20%',],
    "根据各亚组黑人比例阈值": ['self_race_5%', 'self_race_10%', 'self_race_20%'],
    "根据各亚组黑人AKI数量比例阈值": ['self_race_div_25%', 'self_race_div_50%', 'self_race_div_100%']
}
plot_df = pd.DataFrame(columns=["threshold", "score", "建模策略", "threshold_type", "score_type", "drg"])
for score_type in ['TPR']:
    for threshold_type, thresholds in threshold_type_dict.items():
        for threshold in thresholds:

            cur_res_df = fairness_record[f"{score_type}_threshold_{threshold}"]

            if threshold_type == "同比例阈值":
                threshold = threshold[5:]
            elif threshold_type == "根据各亚组黑人AKI数量比例阈值":
                threshold = threshold[14:]
            elif threshold_type == "根据各亚组黑人比例阈值":
                threshold = threshold[10:]

            cur_cols = cur_res_df.columns.tolist()
            col_name_dict = {
                "global": "GM",
                "subgroup": "SM",
                "personal": "PMTL"
            }
            cur_indexes = cur_res_df.index.tolist()

            for col in cur_cols:
                for index in cur_indexes:
                    temp_df = pd.DataFrame(data={
                        "threshold": [threshold],
                        "threshold_type": [threshold_type],
                        "score_type": [score_type],
                        "drg": [index],
                        "建模策略": [col_name_dict[col]],
                        "score": [cur_res_df.loc[index, col]]
                    })
                    plot_df = pd.concat([plot_df, temp_df], axis=0, ignore_index=True)
plot_df.to_csv(f"/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/lab_03_subgroup_race_{race}_top20_fairness_to_plot.csv")


measure_avg_record = pd.DataFrame(columns=['global','subgroup','personal'])
measure_std_record = pd.DataFrame(columns=['global','subgroup','personal'])
measure_CV_record = pd.DataFrame(columns=['global','subgroup','personal'])
for threshold_id in threshold_used:

    for fairness_measure in ['TPR','FPR','odds','PPV']:

        measure_select = '{}_threshold_{}'.format(fairness_measure,threshold_id)
        measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
        measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0)
        measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
        # fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_{}_fairness_{}.csv'.format(race,measure_select))

for fairness_measure in ['TPR','FPR','odds']:

    measure_select = '{}_no_threshold'.format(fairness_measure)
    measure_avg_record.loc[measure_select,:] = fairness_record[measure_select].mean(axis=0)
    measure_std_record.loc[measure_select,:] = fairness_record[measure_select].std(axis=0)
    measure_CV_record.loc[measure_select,:] = measure_std_record.loc[measure_select,:] / measure_avg_record.loc[measure_select,:]
    # fairness_record[measure_select].to_csv('/home/liukang/Doc/fairness_analysis/subgroup_top20_{}_fairness_{}.csv'.format(race,measure_select))


AKI_select_record = AKI_select_record / np.sum(subgroup_result_total['Label'])
nonAKI_select_record = nonAKI_select_record / (subgroup_result_total.shape[0] - np.sum(subgroup_result_total['Label']))


measure_avg_record.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_{}_fairness_avg.csv'.format(race))
measure_std_record.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_{}_fairness_std.csv'.format(race))
measure_CV_record.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_{}_fairness_CV.csv'.format(race))
AKI_select_record.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_{}_fairness_AKI_num.csv'.format(race))
nonAKI_select_record.to_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_{}_fairness_nonAKI_num.csv'.format(race))