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

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from api_utils import get_hos_data_X_y, get_train_test_data_X_y
from lr_utils_api import get_init_similar_weight


def get_pca_exp():
    pca = PCA(n_components=train_data_x.shape[1] - 1)
    pca.fit(train_data_x)
    ev_r = pca.explained_variance_ratio_
    ev_r_sum = np.cumsum(pca.explained_variance_ratio_)
    pd.DataFrame({"ev_r": ev_r, "ev_r_sum": ev_r_sum}).to_csv(f"./result/S05_pca_get_explained_vr_{hos_id}.csv")


def get_pca_deff(n_comp):
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_data_x * init_similar_weight)
    new_test_data_x = pca_model.transform(test_data_x * init_similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_data_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_data_x.index)
    return pca_train_x, pca_test_x, pca_model

def get_pca_auc():
    pass

if __name__ == '__main__':

    # hos_id = int(sys.argv[1])
    hos_id = 0
    # 获取数据
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)

    init_similar_weight = get_init_similar_weight(hos_id)

    res1 = get_pca_deff(0.7)
    res2 = get_pca_deff(29)

