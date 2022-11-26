# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     columns_dict_exp
   Description:   ...
   Author:        cqh
   date:          2022/11/24 19:34
-------------------------------------------------
   Change Activity:
                  2022/11/24:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

from api_utils import get_feature_select_columns
from lr_utils_api import get_lr_init_similar_weight


def get_px_dict_csv():
    """获取px dict"""
    return pd.read_csv("/home/chenqinhai/code_eicu/my_lab/data/raw_file/px_feature_dict.csv", index_col=0).squeeze()


def get_med_dict_csv():
    """获取px dict"""
    return pd.read_csv("/home/chenqinhai/code_eicu/my_lab/data/raw_file/med_feature_dict.csv", index_col=0).squeeze()


def save_select_px_dict():
    """
    保存px映射关系
    :return:
    """
    dict_df = get_px_dict_csv()

    px_list = []
    for cur_col in cur_columns:
        if cur_col.startswith("px"):
            px_list.append(cur_col)

    px_dict_df = pd.DataFrame(index=px_list)

    for px_cur in px_list:
        px_dict_df.loc[px_cur, 'weight'] = init_weight_df.loc[px_cur]
        px_dict_df.loc[px_cur, 'explain'] = dict_df.loc[px_cur]

    px_dict_df.to_csv("/home/chenqinhai/code_eicu/my_lab/result/S02/feature_importance/select_xgb_columns_px_dict.csv")


def save_select_med_dict():
    """
    保存med映射关系
    :return:
    """
    dict_df = get_med_dict_csv()
    med_list = []
    for cur_col in cur_columns:
        if cur_col.startswith("med"):
            med_list.append(cur_col)

    med_dict_df = pd.DataFrame(index=med_list)
    for px_cur in med_list:
        med_dict_df.loc[px_cur, 'weight'] = init_weight_df.loc[px_cur]
        med_dict_df.loc[px_cur, 'explain'] = dict_df.loc[px_cur]

    med_dict_df.to_csv("/home/chenqinhai/code_eicu/my_lab/result/S02/feature_importance/select_xgb_columns_med_dict.csv")


if __name__ == '__main__':
    feature_select_version = 5
    cur_columns = get_feature_select_columns(columns_version=feature_select_version)
    init_weight = get_lr_init_similar_weight(0)
    init_weight_df = pd.Series(index=cur_columns, data=init_weight)
    # save_select_px_dict()
    save_select_med_dict()