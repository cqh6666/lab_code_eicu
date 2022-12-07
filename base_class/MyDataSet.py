# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyDataSet
   Description:   ...
   Author:        cqh
   date:          2022/12/2 17:28
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'

from api_utils import get_fs_each_hos_data_X_y


class MyDataSet:
    """
    读取数据
    """

    def __init__(self, from_hos_id, to_hos_id):
        self.from_hos_id = from_hos_id
        self.to_hos_id = to_hos_id
        self.from_hos_data = get_fs_each_hos_data_X_y(self.from_hos_id)
        self.to_hos_data = get_fs_each_hos_data_X_y(self.to_hos_id)

    def get_from_hos_data(self):
        """
        目标患者数据
        :return:
        """
        _, test_data_x, _, test_data_y = self.from_hos_data

        # 避免过久，不要超过1w个样本
        start_idx = 0
        final_idx = test_data_x.shape[0]
        end_idx = final_idx if final_idx < 10000 else 10000  # 不要超过10000个样本

        # 分批次进行个性化建模
        test_data_x = test_data_x.iloc[start_idx:end_idx]
        test_data_y = test_data_y.iloc[start_idx:end_idx]

        return test_data_x, test_data_y

    def get_to_hos_data(self):
        """
        匹配患者数据
        :return:
        """
        train_data_x, _, train_data_y, _ = self.to_hos_data
        return train_data_x, train_data_y
