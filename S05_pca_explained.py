# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S05_pca_explained
   Description:   PCA降维维度解释
   Author:        cqh
   date:          2022/10/13 15:43
-------------------------------------------------
   Change Activity:
                  2022/10/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys
import os

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from api_utils import get_hos_data_X_y, get_train_test_data_X_y, get_fs_train_test_data_X_y, get_fs_hos_data_X_y
from lr_utils_api import get_init_similar_weight


def get_pca_exp():
    pca = PCA(n_components=train_data_x.shape[1] - 1)
    pca.fit(train_data_x)
    ev_r = pca.explained_variance_ratio_
    ev_r_sum = np.cumsum(pca.explained_variance_ratio_)
    pd.DataFrame({"ev_r": ev_r, "ev_r_sum": ev_r_sum}).to_csv(f"./result/S05_pca_get_explained_vr_{hos_id}.csv")


def get_pca_diff(n_comp):
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_data_x * init_similar_weight)
    new_test_data_x = pca_model.transform(test_data_x * init_similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_data_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_data_x.index)
    return pca_train_x, pca_test_x, pca_model


def process_weight(before_pca_weight):
    weight_list = []
    for weight in before_pca_weight:
        if weight == 0:
            weight_list.append(0.00000001)
        else:
            weight_list.append(weight)

    return weight_list


def get_pca_diff_weight():
    """
    比较 pca前的特征权重，pca（不用权重）后复原回来的特征权重
    :return:
    """
    columns = train_data_x.columns
    res_diff_df = pd.DataFrame(index=columns)

    before_pca_weight = get_init_similar_weight(hos_id)
    weight_df = pd.DataFrame(data={"weight": before_pca_weight}, index=test_data_x.columns)

    res_diff_df['before_pca'] = before_pca_weight

    # 用了pca后复原的权重 不用权重
    # pca_model = PCA(n_components=0.99, random_state=2022)
    # fit_test_data_x = pca_model.fit_transform(test_data_x)
    # inverse_train_data_x = pca_model.inverse_transform(fit_test_data_x)
    # inverse_train_data_x_df = pd.DataFrame(data=inverse_train_data_x, index=test_data_x.index, columns=test_data_x.columns)

    # 用了pca后复原的权重 用权重
    pca_model = PCA(n_components=0.999, random_state=2022)
    tran_test_data_x = test_data_x * before_pca_weight
    fit_tran_test_data_x = pca_model.fit_transform(tran_test_data_x)
    inverse_train_data_x = pca_model.inverse_transform(fit_tran_test_data_x)

    # 比较方差
    test_data_var = test_data_x.var()
    tran_test_data_var = tran_test_data_x.var()
    var_df = pd.DataFrame(index=test_data_x.columns)
    var_df['test_data_var'] = test_data_var
    var_df['tran_test_data_var'] = tran_test_data_var
    var_df['weight'] = weight_df['weight']

    # new_pca_weight = process_weight(before_pca_weight)
    # inverse_train_data_x_ori = inverse_train_data_x / new_pca_weight
    inverse_train_data_x_ori_df = pd.DataFrame(data=inverse_train_data_x, index=test_data_x.index, columns=test_data_x.columns)

    return res_diff_df


def get_lr_psm():
    columns = train_data_x.columns
    before_pca_weight = get_init_similar_weight(hos_id)
    return pd.DataFrame(index=columns, data={"feature": before_pca_weight})


def get_good_feature():
    """
    获得权重不为0的特征
    :return:
    """
    weight_df = get_lr_psm()
    return weight_df[weight_df['feature'] != 0]

if __name__ == '__main__':

    version = 14
    hos_id = 73
    MODEL_SAVE_PATH = f'./result/S03/{hos_id}'

    # 获取数据
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_fs_train_test_data_X_y(strategy=2)
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_fs_hos_data_X_y(hos_id, strategy=2)

    # res_df = get_pca_diff_weight()
    res = get_pca_diff_weight()