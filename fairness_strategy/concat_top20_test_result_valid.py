# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     concat_top20_test_result_valid
   Description:   ...
   Author:        cqh
   date:          2023/4/10 17:12
-------------------------------------------------
   Change Activity:
                  2023/4/10:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

import numpy as np
import pandas as pd

def get_top20_result(rate, stype, cross):
    """
    获取top20测试集的所有结果
    :param rate:
    :param stype:
    :param cross:
    :return:
    """
    pass


def concat_result(stype):
    """
    根据代价指标类型合并数据，求平均，并保存csv
    :param stype:
    :return:
    """
    pass


def run():
    """
    主函数入口
    :return:
    """
    type_list = [1, 2, 3]
    rate_list = [0.75, 0.8, 0.85, 0.9]
    cross_list = [1, 2, 3, 4, 5]
    for _type in type_list:
        for _rate in rate_list:
            cross_data_list = []
            for cross in cross_list:
                cur_data = pd.read_csv(data_file_name.format(_rate, _type, cross), index_col=0)
                cross_data_list.append(cur_data)

            columns = cross_data_list[0].columns
            index = cross_data_list[0].index
            avg_data = pd.DataFrame(0, columns=columns, index=index)

            for cross_data in cross_data_list:
                avg_data += cross_data

            avg_data = avg_data / len(cross_data_list)

            avg_data.to_csv(os.path.join(data_path, "sbdt_top20_rate{}_type{}_v15.csv".format(_rate, _type)))
            print("save success!", _rate, _type)


if __name__ == '__main__':

    data_path = "/home/chenqinhai/code_eicu/my_lab/fairness_strategy/result/top_20/"
    file_name = "sbdt_top20_rate{}_type{}_cross{}_v15.csv"
    data_file_name = os.path.join(data_path, file_name)

    run()