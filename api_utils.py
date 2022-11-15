# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     api_utils
   Description:   ...
   Author:        cqh
   date:          2022/8/29 17:20
-------------------------------------------------
   Change Activity:
                  2022/8/29:
-------------------------------------------------
"""
__author__ = 'cqh'

import random

import numpy as np
import os
import time
import feather

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/home/chenqinhai/code_eicu/my_lab/data/train_file"
all_data_file_name = "all_data_df_v1.feather"
# all_data_norm_file_name = "all_data_df_norm_v2.feather"
all_data_norm_file_name = "all_data_df_norm_v1.feather"
hos_data_norm_file_name = "all_data_df_norm_{}_v1.feather"
y_label = "aki_label"
hospital_id = "hospitalid"
patient_id = "index"
random_state = 2022

feature_select_version = 3


def get_continue_feature():
    """
    获得连续特征列表
    :return:
    """
    return pd.read_csv(os.path.join(TRAIN_PATH, "continue_feature.csv")).iloc[:, 0].tolist()


def get_top5_hospital():
    """
    获取前5个多的医院id
    :return:
    """
    return [73, 167, 264, 420, 338]


def get_all_data():
    data_file = os.path.join(TRAIN_PATH, all_data_file_name)
    all_data = pd.read_feather(data_file)
    print("load all_data", all_data.shape)
    return all_data


def get_all_norm_data():
    data_file = os.path.join(TRAIN_PATH, all_data_norm_file_name)
    all_data = pd.read_feather(data_file)
    print("load all_data", all_data.shape)
    return all_data


@DeprecationWarning
def get_all_data_X_y():
    """
    获取所有数据
    :return:
    """
    all_data = get_all_norm_data()
    # 增加病人ID索引
    all_data.index = all_data[patient_id].tolist()

    all_data_x = all_data.drop([y_label, hospital_id, patient_id], axis=1)
    all_data_y = all_data[y_label]

    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                                            random_state=random_state)
    return train_data_x, test_data_x, train_data_y, test_data_y


def get_train_test_data_X_y():
    """
    获取训练集测试集
    :return:
    """
    train_X_file = os.path.join(TRAIN_PATH, "all_data_df_norm_train_X_v1.feather")
    test_X_file = os.path.join(TRAIN_PATH, "all_data_df_norm_test_X_v1.feather")
    train_y_file = os.path.join(TRAIN_PATH, "all_data_df_norm_train_y_v1.feather")
    test_y_file = os.path.join(TRAIN_PATH, "all_data_df_norm_test_y_v1.feather")

    # 不存在文件就分割数据
    if not os.path.exists(test_y_file):
        print("not exist! begin split...")
        all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y = \
            split_train_test_data(test_X_file, test_y_file, train_X_file, train_y_file)
    else:
        print("exist! start loading...")
        all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y = \
            pd.read_feather(train_X_file), pd.read_feather(test_X_file), \
            pd.read_feather(train_y_file).squeeze(), pd.read_feather(test_y_file).squeeze()

    # 去除hospital_id
    all_train_data_x.drop([hospital_id], axis=1, inplace=True)
    all_test_data_x.drop([hospital_id], axis=1, inplace=True)
    return all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y


def split_train_test_data(test_X_file, test_y_file, train_X_file, train_y_file):
    """
    根据每个中心按7:3分割，最后合并成总的7:3，包含hos_id
    :param test_X_file:
    :param test_y_file:
    :param train_X_file:
    :param train_y_file:
    :return:
    """
    all_data = get_all_norm_data()
    # 增加病人ID索引
    all_data.index = all_data[patient_id].tolist()
    hospital_ids = all_data[hospital_id].value_counts().index.tolist()
    all_train_data_x = pd.DataFrame()
    all_test_data_x = pd.DataFrame()
    all_train_data_y = pd.Series(dtype=np.int64)
    all_test_data_y = pd.Series(dtype=np.int64)
    for hos_id in hospital_ids:
        cur_data = all_data[all_data[hospital_id] == hos_id]
        # 不去除hos_id
        cur_data_x = cur_data.drop([y_label, patient_id], axis=1)
        cur_data_y = cur_data[y_label]
        if cur_data.shape[0] < 10:
            all_train_data_x = pd.concat([all_train_data_x, cur_data_x], axis=0)
            all_train_data_y = pd.concat([all_train_data_y, cur_data_y], axis=0)
            continue

        train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(cur_data_x, cur_data_y, test_size=0.3,
                                                                                random_state=random_state)
        all_train_data_x = pd.concat([all_train_data_x, train_data_x], axis=0)
        all_train_data_y = pd.concat([all_train_data_y, train_data_y], axis=0)
        all_test_data_x = pd.concat([all_test_data_x, test_data_x], axis=0)
        all_test_data_y = pd.concat([all_test_data_y, test_data_y], axis=0)
        print(hos_id, "done...")
    print("concat success!")
    # save
    feather.write_dataframe(all_train_data_x, train_X_file)
    feather.write_dataframe(all_test_data_x, test_X_file)
    feather.write_dataframe(pd.DataFrame(all_train_data_y), train_y_file)
    feather.write_dataframe(pd.DataFrame(all_test_data_y), test_y_file)
    print("save success!")
    return all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y


def get_match_all_data():
    """
    更新版本
    :param hos_id:
    :return:
    """
    match_data_X, _, match_data_y, _ = get_train_test_data_X_y()
    return match_data_X, match_data_y


def get_fs_match_all_data(strategy=2):
    """
    特征处理后的匹配数据
    :param strategy:
    :return:
    """
    match_data_X, _, match_data_y, _ = get_fs_train_test_data_X_y(strategy)
    return match_data_X, match_data_y


@DeprecationWarning
def get_match_all_data_from_hos_data(hos_id):
    """
    根据hos_id匹配全局数据，剔除当前hos_id的测试集数据
    :param hos_id:
    :return:
    """
    # 1. 获取全局数据训练集（包含patient_id）
    all_data = get_all_norm_data()
    all_data_x = all_data.drop([y_label, hospital_id], axis=1)
    all_data_y = all_data[y_label]
    train_data_x, _, train_data_y, _ = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                        random_state=random_state)
    # 2. 获取当下hos_id数据的 测试集 patient_id
    data_file = os.path.join(TRAIN_PATH, hos_data_norm_file_name.format(hos_id))
    hos_data = pd.read_feather(data_file)
    hos_data_x = hos_data.drop(["level_0", y_label], axis=1)
    hos_data_y = hos_data[y_label]
    _, test_data_x, _, _ = train_test_split(hos_data_x, hos_data_y, test_size=0.3,
                                            random_state=random_state)
    hos_test_data_id_list = test_data_x['index'].tolist()

    # 3. 去除全局数据中的训练集数据中包含测试集ID
    condition_df = train_data_x['index'].isin(hos_test_data_id_list)
    match_data_x = train_data_x[~condition_df]
    match_data_y = train_data_y[~condition_df]
    match_data_x = match_data_x.drop(['index'], axis=1)

    return match_data_x, match_data_y


@DeprecationWarning
def get_hos_test_data_id(hos_id):
    data_file = os.path.join(TRAIN_PATH, hos_data_norm_file_name.format(hos_id))
    all_data = pd.read_feather(data_file)
    all_data_x = all_data.drop(["level_0", y_label], axis=1)
    all_data_y = all_data[y_label]
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                                            random_state=random_state)
    return train_data_x, test_data_x, train_data_y, test_data_y


def get_hos_data_X_y(hos_id):
    """
    获取某个中心的数据Xy
    :param hos_id:
    :return:
    """
    data_file = os.path.join(TRAIN_PATH, hos_data_norm_file_name.format(hos_id))
    all_data = pd.read_feather(data_file)
    all_data.index = all_data[patient_id].tolist()
    print("load hosp_data", hos_id, all_data.shape)
    all_data_x = all_data.drop(["level_0", y_label, patient_id], axis=1)
    all_data_y = all_data[y_label]
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                                            random_state=random_state)
    return train_data_x, test_data_x, train_data_y, test_data_y


def get_feature_select_columns(columns_version, strategy=2):
    columns_file = "/home/chenqinhai/code_eicu/my_lab/result/S02/feature_importance/select_{}_columns_v{}.csv"
    if strategy == 1:
        # LR特征选择
        return pd.read_csv(columns_file.format("lr", columns_version), index_col=0).squeeze().to_list()
    elif strategy == 2:
        return pd.read_csv(columns_file.format("xgb", columns_version), index_col=0).squeeze().to_list()
    else:
        raise ValueError("策略参数不存在!")


def get_fs_train_test_data_X_y(strategy=2):
    """
    获取特征选择后的数据
    两种策略
    :return:
    """
    all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y = get_train_test_data_X_y()
    new_columns = get_feature_select_columns(columns_version=feature_select_version, strategy=strategy)
    return all_train_data_x[new_columns], all_test_data_x[new_columns], all_train_data_y, all_test_data_y


def get_fs_hos_data_X_y(hos_id, strategy=2):
    train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)
    new_columns = get_feature_select_columns(strategy=strategy, columns_version=feature_select_version)
    return train_data_x[new_columns], test_data_x[new_columns], train_data_y, test_data_y


def get_fs_each_hos_data_X_y(hos_id, strategy=2):
    """
    抽象成一个函数接口，无论是全局还是单个中心
    :param hos_id:
    :param strategy:
    :return:
    """
    if hos_id == 0:
        return get_fs_train_test_data_X_y(strategy)
    else:
        return get_fs_hos_data_X_y(hos_id, strategy)


def get_target_test_id(hos_id):
    """
    得到50个正样本，50个负样本来进行分析
    :return:
    """
    if hos_id == 0:
        _, _, _, test_data_y = get_train_test_data_X_y()
    else:
        _, _, _, test_data_y = get_hos_data_X_y(hos_id)

    test_data_ids_1 = test_data_y[test_data_y == 1].index[:50].values
    test_data_ids_0 = test_data_y[test_data_y == 0].index[:50].values

    return test_data_ids_1, test_data_ids_0


def get_top5_data():
    """
    获取每个中心的数据
    :return:
    """
    hos_data_list = {}

    hos_ids = get_top5_hospital()

    for hos in hos_ids:
        data_file = os.path.join(TRAIN_PATH, hos_data_norm_file_name.format(hos))
        all_data = pd.read_feather(data_file)
        hos_data_list[hos] = all_data
        print("load hosp_data", hos, all_data.shape)

    return hos_data_list


def covert_time_format(seconds):
    """将秒数转成比较好显示的格式
    # >>> covert_time_format(3600) == '1.0 h'
    # True
    # >>> covert_time_format(360) == '6.0 m'
    # True
    # >>> covert_time_format(6) == '36 s'
    # True
    """
    assert isinstance(seconds, (int, float))
    hour = seconds // 3600
    if hour > 0:
        return f"{round(hour + seconds % 3600 / 3600, 2)} h"

    minute = seconds // 60
    if minute > 0:
        return f"{round(minute + seconds % 60 / 60, 2)} m"

    return f"{round(seconds, 2)} s"


def save_to_csv_by_row(csv_file, new_df):
    """
    以行的方式插入csv文件之中，若文件存在则在尾行插入，否则新建一个新的csv；
    :param csv_file: 默认保存的文件
    :param new_df: dataFrame格式 需要包含header
    :return:
    """
    # 保存存入的是dataFrame格式
    assert isinstance(new_df, pd.DataFrame)
    # 不能存在NaN
    if new_df.isna().sum().sum() > 0:
        print("exist NaN...")
        return False

    if os.path.exists(csv_file):
        new_df.to_csv(csv_file, mode='a', index=True, header=False)
        print("append to csv file success!")
    else:
        new_df.to_csv(csv_file, index=True, header=True)
        print("create to csv file success!")

    return True


def create_path_if_not_exists(new_path):
    if not os.path.exists(new_path):
        try:
            os.makedirs(new_path)
            print("create new dirs... {}".format(new_path))
        except Exception as err:
            pass


def get_sensitive_columns(strategy=2):
    """
    获取敏感特征
    :return:
    """
    cur_columns = get_feature_select_columns(columns_version=feature_select_version, strategy=strategy)
    sens_cols = []
    for col in cur_columns:
        if col.startswith("ccs"):
            sens_cols.append(col)

    return sens_cols


def get_qid_columns(strategy=2, select_rate=0.1):
    """
    获取准标识符特征
    :return:
    """
    cur_columns = get_feature_select_columns(columns_version=feature_select_version, strategy=strategy)
    qid_meds = []
    qid_px = []
    qid_vital = ["bmi"]
    qid_demo = ["age"]
    for col in cur_columns:
        # demo
        if col.startswith("gender") or col.startswith("race"):
            qid_demo.append(col)
        elif col.startswith("med"):
            qid_meds.append(col)
        elif col.startswith("px"):
            qid_px.append(col)

    # 随机选取10%的特征
    random.seed(random_state)
    select_meds = random.sample(qid_meds, int(len(qid_meds) * select_rate))
    select_px = random.sample(qid_px, int(len(qid_px) * select_rate))

    return qid_demo + qid_vital + select_px + select_meds



if __name__ == '__main__':
    # data = get_all_norm_data()
    # all_data_x, t_data_x, all_data_y, t_data_y = get_all_data_X_y()
    # train_data_x2, test_data_x2, train_data_y2, test_data_y2 = get_hos_data_X_y(73)
    # # test1, test0 = get_target_test_id(73)
    # all_data_x2, t_data_x2, all_data_y2, t_data_y2 = get_train_test_data_X_y()

    # res1 = get_fs_train_test_data_X_y(strategy=1)
    # res2 = get_fs_train_test_data_X_y(strategy=2)
    cols = get_sensitive_columns()
