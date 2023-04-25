# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S04_top20_subgroup_data
   Description:   获取top20数据进行公平性分析
   Author:        cqh
   date:          2023/4/25 16:12
-------------------------------------------------
   Change Activity:
                  2023/4/25:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import numpy as np

from api.get_eicu_dataset import get_fairness_data
aki_label = "aki_label"
global_score = "score_y_global"
person_score = "score_y_person"
race_black = "race_African American"
race_white = "race_Caucasian"


def save_top20_drg_cols():
    """
    选择高风险top20患者，根据AKI发生率来选择，并最终保存top20的drg列表
    :return:
    """
    all_data_records = get_fairness_data()
    drg_cols = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/drg_all_dict.csv", squeeze=True, index_col=0).to_list()

    # 保存各亚组患者的AKI发生率，并最终进行降序排序
    aki_risk_records = pd.DataFrame()
    for drg in drg_cols:
        break_condition = (all_data_records.loc[:, drg] == 1)
        subgroup_data = all_data_records[break_condition]
        if subgroup_data[aki_label].sum() < 50:
            print(drg, "AKI患者数量低于50，不符合要求...")
            continue

        aki_risk_records.loc[drg, "aki_risk"] = subgroup_data[aki_label].mean()
        aki_risk_records.loc[drg, "subgroup_nums"] = subgroup_data.shape[0]
    aki_risk_records.sort_values(by="aki_risk", ascending=False, inplace=True)

    top20_drg_df = pd.Series(aki_risk_records.index.to_list()[:20])
    top20_drg_df.to_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S04_top20_drg_cols.csv")
    print("save success!")
    return aki_risk_records


def get_top20_drg_cols():
    return pd.read_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S04_top20_drg_cols.csv", squeeze=True, index_col=0).tolist()


def save_top20_drg_data():
    all_data_records = get_fairness_data()
    data_top20_subgroup = pd.DataFrame()

    drg_list = get_top20_drg_cols()

    for drg in drg_list:
        subgroup_data = all_data_records[(all_data_records.loc[:, drg] == 1)]
        data_top20_subgroup = pd.concat([data_top20_subgroup, subgroup_data], axis=0)

    data_top20_subgroup.to_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S04_top20_subgroup_data.csv")
    print("save success!")


def get_top20_drg_data():
    return pd.read_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S04_top20_subgroup_data.csv", index_col=0)

if __name__ == '__main__':
    save_top20_drg_data()
    data = get_top20_drg_data()
