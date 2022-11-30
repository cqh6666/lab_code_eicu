# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S01_norm_data
   Description:   ...
   Author:        cqh
   date:          2022/11/25 21:15
-------------------------------------------------
   Change Activity:
                  2022/11/25:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import numpy as np
import os

from api_utils import get_all_norm_data, get_all_data


def get_feature_list(feature_flag):
    """
    获取特征映射
    :param feature_flag:
    :return:
    """
    feature_dict = pd.read_csv(os.path.join(save_path, "{}_feature_dict.csv".format(feature_flag)))
    return feature_dict['train_column'].values


def get_cont_feature():
    """
    获得连续特征
    :return:
    """
    demo_vital_feature_list = [
        'age', 'sbp',
       'dbp', 'paop', 'bmi', 'temperature', 'sao2', 'heartrate', 'respiration',
       'sao2.1', 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
       'pasystolic', 'padiastolic', 'pamean', 'cvp', 'etco2', 'st1', 'st2', 'st3', 'icp'
    ]

    lab_feature_list = get_feature_list("lab")
    med_feature_list = get_feature_list("med")
    treat_feature_list = get_feature_list("px")

    all_feature_list = np.concatenate([demo_vital_feature_list, lab_feature_list, med_feature_list, treat_feature_list]).tolist()
    pd.DataFrame(data={"continue_feature": all_feature_list}).to_csv(os.path.join(save_path, "continue_feature.csv"), index=False)

    return all_feature_list


def normalize_data(all_Data):
    """
    对连续特征使用avg-if标准化
    :param all_Data:
    :return:
    """
    c_f = get_cont_feature()

    feature_happen_count = (all_Data.loc[:, c_f] != 0).sum(axis=0)
    feature_sum = all_Data.loc[:, c_f].sum(axis=0)
    feature_average_if = feature_sum / feature_happen_count
    all_Data.loc[:, c_f] = all_Data.loc[:, c_f] / feature_average_if

    # save
    all_Data.reset_index(inplace=True, drop=True)
    all_Data.to_feather(normalize_file)

    return all_Data


def save_topK_data(topK_hospital, all_Data):
    for idx in topK_hospital:
        temp_data = all_Data[all_Data[hospital_id] == idx]
        temp_data.reset_index(inplace=True, drop=True)
        temp_data = temp_data.drop([hospital_id], axis=1)
        temp_path = group_data_file.format(idx)
        temp_data.to_feather(temp_path)
        print("save_success! - ", idx)
    print("done!")


def get_group_data(all_data, topK=5):
    group_data = all_data.groupby(hospital_id)
    # topK医院的ID列表
    topK_hos = group_data.count().sort_values(by=y_label, ascending=False).index.tolist()[:topK]
    # 保存前topK个中心的数据
    save_topK_data(topK_hos, all_data)

    return topK_hos


def normal_process():
    """
    总入口。标准化并获得全局数据和各个中心的数据
    :return:
    """
    all_df = get_all_data()
    all_norm_df = normalize_data(all_df)
    get_group_data(all_norm_df, 20)


def main_run():
    """
    主入口
    :return:
    """
    # 标准化
    normal_process()


if __name__ == '__main__':
    y_label = "aki_label"
    hospital_id = "hospitalid"
    data_path = "data/processeed_csv_result"
    save_path = "data/processeed_csv_result"
    version = 5
    """
    version = 1 均值填充 lab没均值填充
    version = 2 中位数填充
    version = 3 均值填充 lab 均值填充
    version = 5 中值填充 lab 0填充 bmi异常值处理，压缩范围处理
    """
    concat_file = os.path.join(save_path, f"all_data_df_v{version}.feather")
    normalize_file = os.path.join(save_path, f"all_data_df_norm_v{version}.feather")
    group_data_file = os.path.join(save_path, "all_data_df_norm_{}" + f"_v{version}.feather")

    main_run()
