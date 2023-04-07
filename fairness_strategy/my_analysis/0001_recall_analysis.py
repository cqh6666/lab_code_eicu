# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     recall_analysis
   Description:   ...
   Author:        cqh
   date:          2023/3/20 11:20
-------------------------------------------------
   Change Activity:
                  2023/3/20:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import numpy as np

disease_list = pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

subgroup_result = {}
subgroup_result_total = pd.DataFrame()

result_name = {}
result_name['global'] = 'predict_proba'
result_name['subgroup'] = 'subgroup_proba'
result_name['personal'] = 'update_1921_mat_proba'

threshold_record = pd.DataFrame()
threshold_used = ['self_aki_25%', 'self_aki_50%', 'self_aki_100%']

AKI_select_record = pd.DataFrame(index=threshold_used, columns=['global', 'subgroup', 'personal'])
nonAKI_select_record = pd.DataFrame(index=threshold_used, columns=['global', 'subgroup', 'personal'])
AKI_select_record.loc[:, :] = 0
nonAKI_select_record.loc[:, :] = 0


def get_high_patients():
    # 收集高风险患者
    patients_total = pd.DataFrame()
    for disease_num in range(disease_list.shape[0]):
        high_risk_patients = pd.read_csv(
            '/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_list.iloc[disease_num, 0]))[
            [result_name['global'], result_name['subgroup'], result_name['personal'], 'Label']
        ]
        patients_total = pd.concat([patients_total, high_risk_patients], axis=0)

    return patients_total


def get_general_patients():
    # 收集一般患者
    general_risk_patients_total = pd.DataFrame()
    for idx in [1, 2, 3, 4, 5]:
        prob_result = \
            pd.read_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(idx))[
                [result_name['global'], result_name['personal'], 'Label']
            ]
        general_risk_patients_total = pd.concat([general_risk_patients_total, prob_result], axis=0)

    return general_risk_patients_total


def cal_patients_tpr(is_high_risk=False):
    """
    计算高风险患者群体在不同策略下的TPR
    :return:
    """
    # result_records
    result_records = pd.DataFrame()
    record_index = 0

    # 0. 获取数据集
    if is_high_risk:
        patients_total = get_high_patients()
        model_list = ['global', 'subgroup', 'personal']
    else:
        patients_total = get_general_patients()
        model_list = ['global', 'personal']

    # 1. 根据阈值选取方式选取阈值点
    # 1.1 获取AKI患者数目，然后根据比例选取Top-K%
    patients_AKI_true = patients_total.loc[:, 'Label'] == 1
    patients_AKI = patients_total.loc[patients_AKI_true]

    # 2. 根据这个阈值点在不同建模策略下计算TPR
    for model in model_list:
        patients_total.sort_values(result_name[model], inplace=True, ascending=False)
        patients_total.reset_index(drop=True, inplace=True)

        for threshold_id in threshold_used:
            if threshold_id == 'self_aki_25%':
                threshold_index = int(patients_AKI.shape[0] / 4 - 1)
            elif threshold_id == 'self_aki_50%':
                threshold_index = int(patients_AKI.shape[0] / 2 - 1)
            else:
                threshold_index = patients_AKI.shape[0] - 1
            threshold_value = patients_total.loc[threshold_index, result_name[model]]

            patients_break = patients_total.loc[:, result_name[model]] >= threshold_value
            patients_break_select = patients_total.loc[patients_break]

            select_AKI_num = np.sum(patients_break_select.loc[:, 'Label'])

            equal_TPR_threshold = select_AKI_num / patients_AKI.shape[0]

            show_columns = ['high_risk', 'build_model', 'threshold_select', 'TPR_score']
            show_results = [is_high_risk, model, threshold_id, str(round(equal_TPR_threshold * 100, 2))]
            result_records.loc[record_index, show_columns] = show_results
            record_index += 1

    return result_records


# high_risk_records = cal_patients_tpr(True)
# high_risk_records.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/0001_high_risk_TPR_records.csv")
# general_risk_records = cal_patients_tpr(False)
# general_risk_records.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/0001_general_risk_TPR_records.csv")


def get_high_prob_mean():
    # 计算各亚组的平均预测概率
    subgroup_prob_mean_records = pd.DataFrame()
    for disease_num in range(disease_list.shape[0]):
        disease_id_name = disease_list.iloc[disease_num, 0]
        high_risk_patients = pd.read_csv(
            '/home/liukang/Doc/Error_analysis/predict_score_C005_{}.csv'.format(disease_id_name))[
            [result_name['global'], result_name['subgroup'], result_name['personal'], 'Label']
        ]

        show_cols = ['GM_mean', 'SM_mean', 'PMTL_mean']
        value_cols = [
            high_risk_patients[result_name['global']].mean(),
            high_risk_patients[result_name['subgroup']].mean(),
            high_risk_patients[result_name['personal']].mean()
        ]
        subgroup_prob_mean_records.loc[disease_id_name, show_cols] = value_cols

    return subgroup_prob_mean_records


get_high_prob_mean().to_csv(
    "/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/0001_subgroup_prob_mean_records.csv")

print("done!")
