# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_shap
   Description:   ...
   Author:        cqh
   date:          2022/9/26 16:26
-------------------------------------------------
   Change Activity:
                  2022/9/26:
-------------------------------------------------
"""
__author__ = 'cqh'
import shap
import pandas as pd
import os
from api_utils import get_all_data_X_y, get_hos_data_X_y, get_top5_hospital
from xgb_utils_api import get_xgb_model_pkl


def get_shap_value(train_x, model):
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(train_x)
    res = pd.DataFrame(data=shap_value, columns=train_x.columns)
    res = res.abs().mean(axis=0)
    res = res / res.sum()
    res.fillna(0, inplace=True)
    return res


def get_top10_shap(shap_weight):
    shap_weight.sort_values(ascending=False, inplace=True)
    return shap_weight.iloc[:10]


if __name__ == '__main__':
    TRAIN_PATH = "/home/chenqinhai/code_eicu/my_lab/result/shap"
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)

    glo_tl_boost_num = 1000
    # hos_id = 0
    hos_id_list = get_top5_hospital()

    DATA_FILE = "/home/chenqinhai/code_eicu/my_lab/data/train_file/"
    lab_dict = pd.read_csv(os.path.join(DATA_FILE, "lab_feature_dict.csv"), index_col=0)
    med_dict = pd.read_csv(os.path.join(DATA_FILE, "med_feature_dict.csv"), index_col=0)
    css_dict = pd.read_csv(os.path.join(DATA_FILE, "ccs_feature_dict.csv"), index_col=0)
    px_dict = pd.read_csv(os.path.join(DATA_FILE, "px_feature_dict.csv"), index_col=0)

    lab_list = lab_dict.index.tolist()
    med_list = med_dict.index.tolist()
    css_list = css_dict.index.tolist()
    px_list = px_dict.index.tolist()

    for hos_id in hos_id_list:
        # get train data
        train_x, _, _, _ = get_hos_data_X_y(hos_id)
        # get xgb model
        xgb_model = get_xgb_model_pkl(hos_id=hos_id)

        # get shap value
        shap_weight = get_shap_value(train_x, xgb_model)

        shap_file_name = os.path.join(TRAIN_PATH, f'shap_hosid{hos_id}_boost{glo_tl_boost_num}.csv')
        print(f"shap weight shape: {shap_weight.shape}")
        shap_weight.to_csv(shap_file_name, index=True)
        print(f"save success! - {shap_file_name}")

        # get top k
        res_df = pd.DataFrame(get_top10_shap(shap_weight), columns=['value'])
        res_index = res_df.index.tolist()
        for cur_index in res_index:
            if cur_index in lab_list or cur_index in med_list or cur_index in css_list or cur_index in px_list:
                res_df.loc[cur_index, "feature_name"] = lab_dict.loc[cur_index, "origin_column"]

        res_df.to_csv(os.path.join(TRAIN_PATH, f"shap_hosid{hos_id}_boost{glo_tl_boost_num}_top10.csv"))