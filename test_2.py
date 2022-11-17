# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test_2
   Description:   ...
   Author:        cqh
   date:          2022/10/28 15:31
-------------------------------------------------
   Change Activity:
                  2022/10/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import time
import psutil
from numpy.random import laplace
from diffprivlib import mechanisms
from api_utils import get_fs_each_hos_data_X_y, get_qid_columns, get_sensitive_columns
import pandas as pd

from lr_utils_api import get_init_similar_weight


def add_laplace_noise(test_data_x_, μ=0, b=1.0):
    """
    为qid特征增加拉普拉斯噪声
    :param test_data_x_:
    :param μ:
    :param b:
    :return:
    """
    qid_cols = get_qid_columns()
    patient_ids = test_data_x_.index

    for patient_id in patient_ids:
        laplace_noise = laplace(μ, b, len(qid_cols))  # 为原始数据添加μ为0，b为1的噪声
        for index, col in enumerate(qid_cols):
            test_data_x_.loc[patient_id, col] += laplace_noise[index]
    return test_data_x_


def concat_most_sensitive_feature_weight(similar_weight, concat_nums=5):
    """
    将敏感度最高的几个特征进行合并
    :param similar_weight:
    :param concat_nums:
    :return:
    """
    # 选出前k个敏感度最高的特征
    sensitive_feature = get_sensitive_columns()
    sensitive_data_x = test_data_x[sensitive_feature]
    top_sens_feature = sensitive_data_x.sum(axis=0).sort_values(ascending=False).index.to_list()[:concat_nums]

    # 构建series
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=similar_weight)

    # 均值化
    mean_weight = psm_df[psm_df.index.isin(top_sens_feature)].mean()
    psm_df[psm_df.index.isin(top_sens_feature)] = mean_weight

    return psm_df.to_list()


hos_id = 73
init_similar_weight = get_init_similar_weight(hos_id)
train_data_x, test_data_x, train_data_y, test_data_y = get_fs_each_hos_data_X_y(hos_id)
# new_test_data_x = add_laplace_noise(test_data_x, b=0.5)
concat_most_sensitive_feature_weight(init_similar_weight)
print("")


# print("pid", os.getpid())
#
# mem = psutil.virtual_memory()
# # 系统总计内存
# zj = float(mem.total) / 1024 / 1024 / 1024
# # 系统已经使用内存
# ysy = float(mem.used) / 1024 / 1024 / 1024
#
# # 系统空闲内存
# kx = float(mem.free) / 1024 / 1024 / 1024
#
# memory_capacity = zj * 0.25
#
# print('系统总计内存:%d.3GB' % zj)
# print('系统已经使用内存:%d.3GB' % ysy)
# print('系统空闲内存:%d.3GB' % kx)
# print("至少需要留有 {:.3f} GB的空间".format(memory_capacity))