# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     split_group_data
   Description:   将原始数据按照hos_id分割
   Author:        cqh
   date:          2022/9/27 13:49
-------------------------------------------------
   Change Activity:
                  2022/9/27:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

from api_utils import get_all_norm_data


def get_group_data(all_data, topK=5):
    group_data = all_data.groupby(hospital_id)
    # topK医院的ID列表
    topK_hos = group_data.count().sort_values(by=y_label, ascending=False).index.tolist()[:topK]
    # 保存前topK个中心的数据
    save_topK_data(topK_hos, all_data)

    return topK_hos


def save_topK_data(topK_hospital, all_Data):
    for idx in topK_hospital:
        temp_data = all_Data[all_Data[hospital_id] == idx]
        temp_data.reset_index(inplace=True)
        temp_data = temp_data.drop([hospital_id], axis=1)
        temp_path = group_data_file.format(idx)
        temp_data.to_feather(temp_path)
        print("save_success! - ", idx)
    print("done!")


def normal_process():
    """
    总入口。标准化并获得全局数据和各个中心的数据
    :return:
    """
    all_df = get_all_norm_data()
    get_group_data(all_df, 5)


if __name__ == '__main__':
    y_label = "aki_label"
    hospital_id = "hospitalid"
    save_path = "/home/chenqinhai/code_eicu/my_lab/data/train_file/"
    """
    version = 1 均值填充
    version = 2 中位数填充
    """
    group_data_file = os.path.join(save_path, "all_data_df_norm_process_{}_v2.feather")

    normal_process()
