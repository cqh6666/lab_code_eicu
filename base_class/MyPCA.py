# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyPCA
   Description:   ...
   Author:        cqh
   date:          2022/12/2 17:41
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
from sklearn.decomposition import PCA



class MyPCAWithPsm(PCA):
    def __init__(self, psm_weight=None, n_components=0.95):
        self.psm_weight = psm_weight
        self.n_components = n_components
        super().__init__(n_components)

    def set_psm_weight(self, psm_weight):
        """
        设置更改相似性度量
        :return:
        """
        self.psm_weight = psm_weight

    def fit_transform(self, data_df, y=None):
        assert not isinstance(data_df, pd.DataFrame)
        my_logger.warning(f"正在进行PCA降维...", data_df.shape)
        data_df = data_df * self.psm_weight
        pca_data_df = super().fit_transform(data_df)
        return pd.DataFrame(data=pca_data_df, index=data_df)

    def transform(self, data_df):
        assert not isinstance(data_df, pd.DataFrame)
        my_logger.warning("开始通过降维矩阵转换数据...", data_df.shape)
        data_df = data_df * self.psm_weight
        pca_data_df = super().transform(data_df)
        return pd.DataFrame(data=pca_data_df, index=data_df)

    def inverse_transform(self, data_df):
        my_logger.warning("开始恢复数据...")
        assert not isinstance(data_df, pd.DataFrame)
        rec_data = super().inverse_transform(data_df)
        rec_data_df = pd.DataFrame(data=rec_data, index=data_df.index)
        rec_data_df = rec_data_df / self.process_psm_weight()

        my_logger.info("成功恢复目标患者数据:  target_data:{}".format(rec_data_df.shape))

        return rec_data_df

    def process_psm_weight(self):
        """
        处理相似性度量
        :return:
        """
        new_weight = []
        for weight in self.psm_weight:
            if weight == 0:
                new_weight.append(1.0)
            else:
                new_weight.append(weight)
        return new_weight



