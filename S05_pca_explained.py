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

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from api_utils import get_all_data_X_y, get_hos_data_X_y


def get_pca_exp(hos_id):
    # 获取数据
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_all_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)
    pca = PCA(n_components=train_data_x.shape[1] - 1)
    pca.fit(train_data_x)
    ev_r = pca.explained_variance_ratio_
    ev_r_sum = np.cumsum(pca.explained_variance_ratio_)
    pd.DataFrame({"ev_r": ev_r, "ev_r_sum": ev_r_sum}).to_csv(f"./result/S05_pca_get_explained_vr_{hos_id}.csv")


if __name__ == '__main__':
    all_hos = (0, 73, 167, 264, 338, 420)

    for hid in all_hos:
        get_pca_exp(hid)