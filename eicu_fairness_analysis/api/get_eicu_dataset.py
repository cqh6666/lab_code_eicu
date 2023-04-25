# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_eicu_dataset
   Description:   ...
   Author:        cqh
   date:          2023/1/18 21:41
-------------------------------------------------
   Change Activity:
                  2023/1/18:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import pandas as pd
import feather

from api_utils import get_all_data, get_all_norm_data, get_cross_data
from sklearn.model_selection import KFold


def cross_five_part_data():
    data_df = get_all_norm_data()
    # 增加病人ID索引
    data_df.index = data_df["index"].tolist()
    data_df.drop(['index', 'hospitalid'],axis=1, inplace=True)

    data_y = data_df['aki_label']
    data_x = data_df.drop(['aki_label'], axis=1)

    # 原始数据
    my_data = get_all_data()
    # 增加病人ID索引
    my_data.index = my_data["index"].tolist()
    my_data.drop(['index', 'hospitalid'], axis=1, inplace=True)

    kfold = KFold(n_splits=5)
    number = 1
    for train_index, test_index in kfold.split(data_x, data_y):
        train_data = data_df.iloc[train_index, :]
        test_data = data_df.iloc[test_index, :]
        feather.write_dataframe(train_data, os.path.join(data_path, f'train_norm_data_{number}.feather'))
        feather.write_dataframe(test_data, os.path.join(data_path, f'test_norm_data_{number}.feather'))

        train_ori_data = my_data.iloc[train_index, :]
        test_ori_data = my_data.iloc[test_index, :]
        feather.write_dataframe(train_ori_data, os.path.join(data_path, f'train_data_{number}.feather'))
        feather.write_dataframe(test_ori_data, os.path.join(data_path, f'test_data_{number}.feather'))

        print("save success!", number)
        number += 1


def subgroup_feature():
    all_data = get_all_data()
    columns = all_data.columns

    my_select_feature = []

    for col in columns:
        if col.startswith("ccs"):
            my_select_feature.append(col)


def process_subgroup_data(test_data):
    """
    分亚组专用的数据集
    :param test_data:
    :return:
    """
    # 黑人白人
    race_list = ["race_African American", "race_Caucasian"]
    black_white_true = (test_data[race_list[0]] == 1) | (test_data[race_list[1]] == 1)
    test_data = test_data[black_white_true]

    cols = test_data.columns
    med_px_list = []
    ccs_list = []
    demo_list = []
    for col in cols:
        if col.startswith("med") or col.startswith("px"):
            med_px_list.append(col)
        if col.startswith("ccs"):
            ccs_list.append(col)
        if col.startswith("gender"):
            demo_list.append(col)

    for med_px in med_px_list:
        data_true = test_data[med_px] >= 7
        test_data.loc[data_true, med_px] = 1
        test_data.loc[~data_true, med_px] = 0
        test_data[med_px] = test_data[med_px].astype(int)

    last_cols = demo_list + med_px_list + ccs_list
    # 不包含黑人白人特征
    pd.Series(last_cols).to_csv(os.path.join(data_path, "subgroup_select_feature.csv"))

    test_data["race_black"] = test_data[race_list[0]]
    test_data["race_white"] = test_data[race_list[1]]
    test_data.drop(race_list, axis=1, inplace=True)
    last_cols = ["race_black", "race_white"] + last_cols
    return test_data[last_cols]


def get_test_cross_valid(score_file):
    result_df = pd.read_csv(score_file, index_col=0)
    test_ids = result_df.index.to_list()
    predict_score = result_df['prob']
    test_ids = list(set(test_ids) - {889425, 592976, 815219, 1655468, 3091400, 2392424, 1121703, 1441035, 1534652,
                                     1526183, 1533890})

    # data
    my_data = get_all_data()
    # 增加病人ID索引
    my_data.index = my_data["index"].tolist()
    my_data.drop(['index', 'hospitalid'], axis=1, inplace=True)

    test_data = my_data.loc[test_ids, :]
    y_label = test_data['aki_label']
    test_data = process_subgroup_data(test_data)

    test_data['score_y'] = predict_score
    test_data['Label'] = y_label
    kfold = KFold(n_splits=5)
    number = 1

    for _, test_index in kfold.split(test_data):
        cur_data = test_data.iloc[test_index, :]
        feather.write_dataframe(cur_data, os.path.join(data_path, f'test_valid_{number}.feather'))
        print("save success!", number)
        number += 1

    return test_data


def split_five_data():
    """
    5折交叉数据
    :return:
    """
    all_norm_data = get_all_norm_data()
    all_norm_data.index = all_norm_data["index"].tolist()
    all_norm_data.drop(['index', 'hospitalid'], axis=1, inplace=True)

    kfold = KFold(n_splits=5)

    feather.write_dataframe(all_norm_data, os.path.join(data_path, f'all_norm_data.feather'))
    number = 1
    for _, test_index in kfold.split(all_norm_data):
        cur_data = all_norm_data.iloc[test_index, :]
        feather.write_dataframe(cur_data, os.path.join(data_path, f'test_valid_{number}.feather'))
        print("save success!", number)
        number += 1

    return all_norm_data


def process_eicu_subgroup_data(test_df):
    """
    提取黑人和白人的记录
    同时将med和px的相关属性进行 0 1 化处理
    :param test_df:
    :return:
    """
    # 黑人白人
    race_list = ["race_African American", "race_Caucasian"]
    black_white_true = (test_df[race_list[0]] == 1) | (test_df[race_list[1]] == 1)
    test_data = test_df[black_white_true]
    cur_cols = test_data.columns

    for col in cur_cols:
        if col.startswith("med") or col.startswith("px"):
            data_true = test_data[col] >= 7
            test_data.loc[data_true, col] = 1
            test_data.loc[~data_true, col] = 0
            test_data[col] = test_data[col].astype(int)

    return test_data


def get_all_test_data(test_range_ids=None):
    """
    :param test_range_ids: 交叉验证集的序号集合
    :return:
    """
    if test_range_ids is None:
        test_range_ids = [1, 2, 3, 4, 5]

    data_path = f'/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/eicu_data/'
    version = 30

    all_data_df = get_all_data()
    # 增加病人ID索引
    all_data_df.index = all_data_df["index"].tolist()
    all_data_df.drop(['index', 'hospitalid'], axis=1, inplace=True)

    # 增加drg属性
    drg_result_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/data_drg_onehot_df.csv", index_col=0)
    drg_cols = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/drg_all_dict.csv", squeeze=True, index_col=0).to_list()

    cur_drg_data_df = pd.DataFrame(index=all_data_df.index)
    cur_drg_data_df = pd.concat([cur_drg_data_df, drg_result_df], join="inner", axis=1)

    all_data_df = pd.concat([cur_drg_data_df, all_data_df], axis=1)

    cur_data_df = pd.DataFrame()
    select_cols = pd.read_csv(os.path.join(data_path, "subgroup_select_feature.csv"), index_col=0).squeeze().to_list()
    race_cols = ["race_African American", "race_Caucasian"]
    aki_cols = ['aki_label']
    all_cols = drg_cols + race_cols + select_cols + aki_cols

    # 风险概率(GM, PMTL)
    score_result = pd.read_csv(
        f"/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S02_test_result_GM_PMTL_predict_prob.csv",
        index_col=0
    )

    for idx in test_range_ids:
        test_index = pd.read_feather(os.path.join(data_path, f'test_valid_{idx}.feather')).index.to_list()
        test_df = all_data_df.loc[test_index, all_cols]

        test_df["score_y_global"] = score_result.loc[test_index, "global_predict_proba"]
        test_df["score_y_person"] = score_result.loc[test_index, "personal_predict_proba"]

        # person_result = pd.read_csv(
        #     f"/home/chenqinhai/code_eicu/my_lab/result/S04/0/S04_LR_test{idx}_tra1_boost100_select10_v{version}.csv", index_col=0)
        # y_score = person_result['prob']
        # test_df["score_y"] = y_score

        subgroup_data = process_eicu_subgroup_data(test_df)
        cur_data_df = pd.concat([cur_data_df, subgroup_data], axis=0)

    cur_data_df.reset_index(drop=True, inplace=True)
    print("load success!", cur_data_df.shape, test_range_ids)
    return cur_data_df


def save_fairness_data():
    """
    获得全局、个性化的预测概率
    结合 黑人和白人
    再加入 DRG
    :return:
    """
    # 获得所有记录
    all_test_data = get_all_test_data([1,2,3,4,5])

    race_cols = ["race_African American", "race_Caucasian"]
    drg_cols = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/drg_all_dict.csv", squeeze=True, index_col=0).to_list()
    aki_cols = ["aki_label", "score_y_global", "score_y_person"]

    all_cols = race_cols + drg_cols + aki_cols
    fairness_data = all_test_data[all_cols]

    fairness_data.to_csv(
        "/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S03_fairness_analysis_all_test_data.csv"
    )

    print("save fairness data success!", fairness_data.shape)
    return fairness_data


def get_fairness_data():
    fairness_data = pd.read_csv(
        "/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/S03_fairness_analysis_all_test_data.csv",
        index_col=0
    )
    return fairness_data


def process_drg_columns():
    """
    处理drg属性,最终保存为 data_drg_onehot_df
    :return:
    """
    # 获取病人ID list
    all_data_df = get_all_data()
    patient_id_list = all_data_df["index"].tolist()

    drg_file = f"/home/chenqinhai/code_eicu/my_lab/data/demographics_raw_4-24.csv"
    drg_raw_df = pd.read_csv(drg_file, index_col=0)[["patientunitstayid", "apacheadmissiondx"]]

    # drg缺失值填充
    drg_raw_df.fillna("Not Known", inplace=True)

    # 保存drg映射属性
    drg_flag = "drg"
    drg_index = drg_raw_df["apacheadmissiondx"].unique().tolist()
    drg_index = ["apacheadmissiondx_" + drg for drg in drg_index]
    cur_drg_series = pd.Series(index=drg_index)
    for idx, drg in enumerate(drg_index):
        cur_drg_series.loc[drg] = drg_flag + str(idx)

    cur_drg_series.to_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/drg_all_dict.csv")
    print("save drg list success!")

    # one_hot 处理
    drg_raw_df["apacheadmissiondx"] = drg_raw_df["apacheadmissiondx"].astype('category')
    drg_result_df = pd.get_dummies(drg_raw_df, columns=["apacheadmissiondx"])

    # 修改列名
    drg_dict = cur_drg_series.to_dict()
    drg_result_df = drg_result_df.rename(columns=drg_dict)

    # 修改patientID为索引
    drg_result_df.index = drg_result_df["patientunitstayid"]
    drg_result_df.drop(["patientunitstayid"], axis=1, inplace=True)

    # 保存drg_result_df
    drg_result_df.to_csv("/home/chenqinhai/code_eicu/my_lab/eicu_fairness_analysis/data_drg_onehot_df.csv")
    print("save drg one hot data success!")
    return drg_result_df


def test_get_data():
    return get_cross_data(1)


if __name__ == '__main__':
    version = 30
    data_path = f'/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/eicu_data/'
    # data = process_drg_columns()  # 处理DRG属性
    # all_test_data = get_all_test_data([1])
    save_fairness_data()
    fair_data = get_fairness_data()