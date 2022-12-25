# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     xgb_utils_api
   Description:   获取xgb相关信息，文件
   Author:        cqh
   date:          2022/7/8 14:30
-------------------------------------------------
   Change Activity:
                  2022/7/8:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os
import pickle
from my_logger import logger

MODEL_SAVE_PATH = '/home/chenqinhai/code_eicu/my_lab/result/S03/{}'
global_xgb_boost = 1000
"""
version = 1 旧版本 不用类权重
version = 5 新版本 不用类权重参数
version = 7 类权重参数
version = 10 xgb特征选择新数据
version = 11 lr特征选择新数据
"""
version = "5a"


def get_xgb_model_pkl(hos_id):
    xgb_model_file = os.path.join(MODEL_SAVE_PATH.format(hos_id), f"S03_global_xgb_{global_xgb_boost}_v{version}.pkl")
    xgb_model = pickle.load(open(xgb_model_file, "rb"))

    logger.warning(f"读取xgb迁移模型: {xgb_model_file}, version:{version}")

    return xgb_model


def get_init_similar_weight(hos_id):
    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH.format(hos_id), f'S03_0_psm_global_xgb_{global_xgb_boost}_v{version}.csv')
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()

    logger.warning(f"读取xgb相似性度量({len(init_similar_weight)}):{init_similar_weight[:5]}, version:{version}")

    return init_similar_weight


def get_xgb_init_similar_weight(hos_id):
    return get_init_similar_weight(hos_id)


def get_local_xgb_para(xgb_thread_num=1, num_boost_round=50):
    """
    xgb local 参数
    :param xgb_thread_num: 线程数
    :param num_boost_round: 数迭代次数
    :return:
    """
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': xgb_thread_num,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 2022,
        'tree_method': 'hist',
    }
    return params, num_boost_round

