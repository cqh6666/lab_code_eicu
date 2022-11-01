# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     lr_utils_api
   Description:   获取LR相关文件信息
   Author:        cqh
   date:          2022/7/8 14:41
-------------------------------------------------
   Change Activity:
                  2022/7/8:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os

MODEL_SAVE_PATH = '/home/chenqinhai/code_eicu/my_lab/result/S03/{}'
global_lr_iter = 1000
# 不做类权重 5  做类权重 6
version = 6


def get_init_similar_weight(hos_id):
    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH.format(hos_id),
                                            f"S03_0_psm_global_lr_{global_lr_iter}_v{version}.csv")
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()
    return init_similar_weight


def get_lr_init_similar_weight(hos_id):
    return get_init_similar_weight(hos_id)


def get_transfer_weight(hos_id):
    # 全局迁移策略 需要用到初始的csv
    init_weight_file_name = os.path.join(MODEL_SAVE_PATH.format(hos_id),
                                         f"S03_global_weight_lr_{global_lr_iter}_v{version}.csv")
    global_feature_weight = pd.read_csv(init_weight_file_name).squeeze().tolist()
    return global_feature_weight
