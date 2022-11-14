# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     load_dataset
   Description:   ...
   Author:        cqh
   date:          2022/11/8 15:25
-------------------------------------------------
   Change Activity:
                  2022/11/8:
-------------------------------------------------
"""
__author__ = 'cqh'

import os.path

import shap

from api_utils import get_fs_hos_data_X_y, get_hos_data_X_y
from warnings import simplefilter
import pandas as pd

from xgb_utils_api import get_xgb_model_pkl

simplefilter(action='ignore', category=FutureWarning)


def get_ccs_columns():
    columns = train_data_x.columns
    res_list = []
    for col in columns:
        if col.startswith("ccs"):
            res_list.append(col)

    return res_list


def get_cat_columns():
    return pd.read_csv(os.path.join(save_path, "select_cat_columns_v2.csv"), index_col=0).squeeze().to_list()


def columns_map():
    """
    属性解释映射
    :return:
    """
    # 获取属性
    columns = train_data_x.columns

    # 获取特征med, px, lab, ccs对应的字典解释
    med_dict = pd.read_csv(os.path.join(dict_path, "med_feature_dict.csv"), index_col=0)
    px_dict = pd.read_csv(os.path.join(dict_path, "px_feature_dict.csv"), index_col=0)
    ccs_dict = pd.read_csv(os.path.join(dict_path, "ccs_feature_dict.csv"), index_col=0)
    lab_dict = pd.read_csv(os.path.join(dict_path, "lab_feature_dict.csv"), index_col=0)

    # 遍历属性，分别对应解释到一个新df里
    dict_df = pd.DataFrame(index=columns, columns=["explain"])
    for cols in columns:
        if cols.startswith("lab"):
            dict_df.loc[cols, 'explain'] = lab_dict.loc[cols, 'origin_column']
        elif cols.startswith("med"):
            dict_df.loc[cols, 'explain'] = med_dict.loc[cols, 'origin_column']
        elif cols.startswith("px"):
            dict_df.loc[cols, 'explain'] = px_dict.loc[cols, 'origin_column']
        elif cols.startswith("ccs"):
            dict_df.loc[cols, 'explain'] = ccs_dict.loc[cols, 'origin_column']

    # 保存
    dict_df.to_csv(os.path.join(save_path, "select_xgb_columns_dict.csv"))
    print("save success!")


def cal_shap():
    def get_shap_value(train_x, model):
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(train_x)
        res = pd.DataFrame(data=shap_value, columns=train_x.columns)
        res = res.abs().mean(axis=0)
        res = res / res.sum()
        res.fillna(0, inplace=True)
        return res

    # get train data
    train_x, _, _, _ = get_fs_hos_data_X_y(hos_id)
    # get xgb model
    xgb_model = get_xgb_model_pkl(hos_id=hos_id)

    return get_shap_value(train_x, xgb_model)


def process_anonymity():
    anonymity_feature = [
        "ccs_104", "ccs_147", "ccs_178", "ccs_201", "ccs_255", "ccs_259", "ccs_3", "ccs_6", "ccs_93", "ccs_115",
        "ccs_1", "age", "cvp", "gender_Male", "bmi"]

if __name__ == '__main__':
    save_path = "/home/chenqinhai/code_eicu/my_lab/result/S02/feature_importance/"
    dict_path = "/home/chenqinhai/code_eicu/my_lab/data/train_file/"
    hos_id = 73
    # train_data_x, test_data_x, train_data_y, test_data_y = get_fs_hos_data_X_y(hos_id, strategy=2)
    # train_data_x2, test_data_2x, train_data_y2, test_data_y2 = get_hos_data_X_y(hos_id)
    weight = cal_shap()