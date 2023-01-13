# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     pca_sklearn_use
   Description:   ...
   Author:        cqh
   date:          2023/1/12 16:41
-------------------------------------------------
   Change Activity:
                  2023/1/12:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from api_utils import get_fs_each_hos_data_X_y

from_hos_id = 73
to_hos_id = 0
n_comp = 0.95

# 0. 获取数据, 获取度量
_, target_data_x, _, target_data_y = get_fs_each_hos_data_X_y(from_hos_id)
match_data_x, _, match_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
target_data_x = target_data_x.iloc[:50, :]
match_data_x = match_data_x.iloc[:100, :]

# len_split = int(match_data_x.shape[0] * 0.1)
# columns_list = match_data_x.columns.to_list()

# pca降维
pca_model = PCA(n_components=n_comp, random_state=2022)
new_test_data_x = pca_model.fit_transform(target_data_x)
new_train_data_x = pca_model.transform(match_data_x)