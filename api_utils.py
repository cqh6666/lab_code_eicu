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

import numpy as np
import os
import feather

import pandas as pd
from sklearn.model_selection import train_test_split
from my_logger import logger

version = "5"

if version == 1:
    TRAIN_PATH = "/home/chenqinhai/code_eicu/my_lab/data/train_file/"
else:
    TRAIN_PATH = "/home/chenqinhai/code_eicu/my_lab/data/processeed_csv_result/"

all_data_file_name = f"all_data_df_v{version}.feather"
all_data_norm_file_name = f"all_data_df_norm_v{version}.feather"
hos_data_norm_file_name = "all_data_df_norm_{}_v" + f"{version}.feather"
y_label = "aki_label"
hospital_id = "hospitalid"
patient_id = "index"
random_state = 2022
feature_select_version = 5


def get_continue_feature():
    """
    获得连续特征列表
    :return:
    """
    return pd.read_csv(os.path.join(TRAIN_PATH, "continue_feature.csv")).iloc[:, 0].tolist()


def get_topK_hospital(k=5):
    """
    获取前5个多的医院id
    :return:
    """
    all_data = get_all_norm_data()
    hospital_ids = all_data[hospital_id].value_counts(ascending=False).index.to_list()
    logger.warning("The count of hospital is {}".format(len(hospital_ids)))
    return hospital_ids[:k]


def get_all_data():
    data_file = os.path.join(TRAIN_PATH, all_data_file_name)
    all_data = pd.read_feather(data_file)

    # 过滤掉18岁以下的病人
    all_data = all_data.query("age >= 18")

    # 将bmi值变为2位有效数字
    # all_data['bmi'] = round(all_data['bmi'], 0)

    return all_data


def get_all_norm_data():
    data_file = os.path.join(TRAIN_PATH, all_data_norm_file_name)
    all_data = pd.read_feather(data_file)
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
    获取 全局 训练集测试集
    :return:
    """
    all_test_data_x, all_test_data_y, all_train_data_x, all_train_data_y = load_global_dataset()

    # 去除hospital_id
    all_train_data_x.drop([hospital_id], axis=1, inplace=True)
    all_test_data_x.drop([hospital_id], axis=1, inplace=True)

    logger.warning(f"当前中心:0, 读取数据版本为: {version}, 维度:{all_train_data_x.shape}")

    return all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y


def load_global_dataset():
    """
    获取全局数据集
    :return:
    """
    train_X_file = os.path.join(TRAIN_PATH, f"all_data_df_norm_train_X_v{version}.feather")
    test_X_file = os.path.join(TRAIN_PATH, f"all_data_df_norm_test_X_v{version}.feather")
    train_y_file = os.path.join(TRAIN_PATH, f"all_data_df_norm_train_y_v{version}.feather")
    test_y_file = os.path.join(TRAIN_PATH, f"all_data_df_norm_test_y_v{version}.feather")
    # 不存在文件就分割数据
    if not os.path.exists(test_y_file):
        logger.warning("not exist! begin split...")
        all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y = \
            split_train_test_data(test_X_file, test_y_file, train_X_file, train_y_file)
    else:
        logger.warning("exist! start loading...")
        all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y = \
            pd.read_feather(train_X_file), pd.read_feather(test_X_file), \
            pd.read_feather(train_y_file).squeeze(), pd.read_feather(test_y_file).squeeze()

    # 去除杂鱼属性
    other_columns = "level_0"
    t_columns = all_train_data_x.columns.to_list()
    if other_columns in t_columns:
        all_train_data_x = all_train_data_x.drop([other_columns], axis=1)
        all_test_data_x = all_test_data_x.drop([other_columns], axis=1)
        logger.warning(f"remove {other_columns} columns...")

    return all_test_data_x, all_test_data_y, all_train_data_x, all_train_data_y


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
        logger.info(hos_id, "done...")
    logger.info("concat success!")
    # save
    feather.write_dataframe(all_train_data_x, train_X_file)
    feather.write_dataframe(all_test_data_x, test_X_file)
    feather.write_dataframe(pd.DataFrame(all_train_data_y), train_y_file)
    feather.write_dataframe(pd.DataFrame(all_test_data_y), test_y_file)
    logger.info("save success!")
    return all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y


def get_match_all_data():
    """
    更新版本
    :param hos_id:
    :return:
    """
    match_data_X, _, match_data_y, _ = get_train_test_data_X_y()
    logger.warning(f"可知全局匹配患者数量为 {match_data_X.shape[0]}.")
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
    hos_data_x = hos_data.drop([y_label], axis=1)
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
    all_data_x = all_data.drop([y_label], axis=1)
    all_data_y = all_data[y_label]
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                                            random_state=random_state)
    return train_data_x, test_data_x, train_data_y, test_data_y


def get_hos_data_X_y(hos_id):
    """
    读取某个中心的 train test X,y
    :param hos_id:
    :return:
    """
    # 读取全局中心的数据
    all_test_data_x, all_test_data_y, all_train_data_x, all_train_data_y = load_global_dataset()

    # 根据hos_id获取数据
    test_data_hos_index = (all_test_data_x[hospital_id] == hos_id)
    train_data_hos_index = (all_train_data_x[hospital_id] == hos_id)

    all_test_data_x = all_test_data_x[test_data_hos_index]
    all_test_data_y = all_test_data_y[test_data_hos_index]
    all_train_data_x = all_train_data_x[train_data_hos_index]
    all_train_data_y = all_train_data_y[train_data_hos_index]

    # 去除hospital_id
    all_train_data_x.drop([hospital_id], axis=1, inplace=True)
    all_test_data_x.drop([hospital_id], axis=1, inplace=True)

    logger.warning(f"当前中心:{hos_id}, 读取数据版本为: {version}, 维度:{all_train_data_x.shape}")

    return all_train_data_x, all_test_data_x, all_train_data_y, all_test_data_y


# @DeprecationWarning
def get_hos_data_X_y_old(hos_id):
    """
    获取某个中心的数据Xy
    :param hos_id:
    :return:
    """
    data_file = os.path.join(TRAIN_PATH, hos_data_norm_file_name.format(hos_id))
    all_data = pd.read_feather(data_file)
    all_data.index = all_data[patient_id].tolist()
    all_data_x = all_data.drop([y_label, patient_id], axis=1)
    all_data_y = all_data[y_label]
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                                            random_state=random_state)

    logger.warning(f"当前中心:{hos_id}, 读取数据版本为: {version}, 维度:{train_data_x.shape}, 来源:{data_file}")

    return train_data_x, test_data_x, train_data_y, test_data_y


def get_hos_data_X_y_from_all(hos_id):
    """
    从所有数据中获取数据X,y
    :param hos_id:
    :return:
    """
    all_data = get_all_norm_data()
    # 增加病人ID索引
    all_data.index = all_data[patient_id].tolist()

    cur_data = all_data[all_data[hospital_id] == hos_id]
    # 不去除hos_id
    cur_data_x = cur_data.drop([y_label, patient_id], axis=1)
    cur_data_y = cur_data[y_label]

    return train_test_split(cur_data_x, cur_data_y, test_size=0.3, random_state=random_state)


def get_fs_hos_data_X_y_from_all(hos_id, strategy=2):
    train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y_from_all(hos_id)
    new_columns = get_feature_select_columns(strategy=strategy, columns_version=feature_select_version)
    return train_data_x[new_columns], test_data_x[new_columns], train_data_y, test_data_y


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


def get_each_hos_data_X_y(hos_id):
    """
    无论是全局还是单个中心，都调用这个函数
    :param hos_id:
    :return:
    """
    if hos_id == 0:
        res_df = get_train_test_data_X_y()
    else:
        res_df = get_hos_data_X_y(hos_id)

    return res_df


def get_fs_each_hos_data_X_y(hos_id, strategy=2):
    """
    抽象成一个函数接口，无论是全局还是单个中心
    :param hos_id:
    :param strategy:
    :return:
    """
    if hos_id == 0:
        res_df = get_fs_train_test_data_X_y(strategy)
    else:
        res_df = get_fs_hos_data_X_y(hos_id, strategy)

    logger.warning(f"做了特征选择后:{res_df[0].shape}, strategy:{strategy}")
    return res_df


def get_target_test_id(hos_id):
    """
    得到50个正样本，50个负样本来进行分析
    :return:
    """
    # if hos_id == 0:
    #     _, _, _, test_data_y = get_train_test_data_X_y()
    # else:
    #     _, _, _, test_data_y = get_hos_data_X_y(hos_id)
    _, _, _, test_data_y = get_fs_each_hos_data_X_y(hos_id)
    test_data_ids_1 = test_data_y[test_data_y == 1].index[:50].values
    test_data_ids_0 = test_data_y[test_data_y == 0].index[:50].values

    return test_data_ids_1, test_data_ids_0


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
        logger.error("exist NaN...")
        return False

    if os.path.exists(csv_file):
        new_df.to_csv(csv_file, mode='a', index=True, header=False)
        logger.warning("append to csv file success!")
    else:
        new_df.to_csv(csv_file, index=True, header=True)
        logger.warning("create to csv file success!")

    return True


def create_path_if_not_exists(new_path):
    if not os.path.exists(new_path):
        try:
            os.makedirs(new_path)
            logger.warning("create new dirs... {}".format(new_path))
        except Exception as err:
            pass


def get_sensitive_columns(strategy=2):
    """
    获取敏感特征
    :return:
    """
    cur_columns = get_feature_select_columns(columns_version=feature_select_version, strategy=strategy)
    cur_columns_set = set(cur_columns)
    # 肺结核, 冠心病, 神经病, 急性呼吸衰竭, 肾失调， 精神创伤， 传染病，
    sens_ccs = ['ccs_122', 'ccs_172', 'ccs_148', 'ccs_242', "ccs_42", "ccs_139", "ccs_229"]
    # 他克莫司（Tacrolimus）免疫抑制剂, 异丙酚（Propofol） 镇定剂
    # sens_med = ['med_1388', "med_1172", "med_121"]
    sens_med = []
    # 神经病治疗, 肾衰竭传染性疾病
    # sens_px = ['px_242', "px_534", "px_10", "px_238", "px_20"]
    sens_px = []

    sens_cols = sens_ccs + sens_med + sens_px
    for col in sens_cols:
        if col not in cur_columns_set:
            sens_cols.remove(col)
            logger.warning("当前特征列表不存在此敏感特征-{}, 已删除...".format(col))

    return sens_cols


def get_diff_sens():
    """
    获取不同类型的敏感特征
    :return:
    """
    sens_cols = get_sensitive_columns()
    # 连续特征和离散特征
    con_cols, cat_cols = [], []

    for col in sens_cols:
        if col.startswith("px") or col.startswith("med"):
            con_cols.append(col)
        elif col.startswith("ccs"):
            cat_cols.append(col)

    return con_cols, cat_cols


def get_qid_columns(strategy=2):
    """
    获取准标识符特征
    :return:
    """
    cur_columns = get_feature_select_columns(columns_version=feature_select_version, strategy=strategy)
    cur_columns_set = set(cur_columns)

    # 糖原代谢病, 休克/低血压, 胸痛, 高血压
    qid_ccs = ['ccs_10', 'ccs_13', 'ccs_199', 'ccs_214']
    # 肠胃手术, 高血压手术，胰岛素注射
    qid_px = ["px_596", "px_92", "px_497", "px_471"]
    # bmi
    qid_vital = ["bmi"]
    # age, gender, race
    #
    qid_demo = ["age", "gender_Female", "gender_Male", "race_Asian", "race_Caucasian", "race_African American",
                "race_Other/Unknown", "race_Hispanic"]

    qid_cols = qid_demo + qid_vital + qid_px + qid_ccs

    for col in qid_cols:
        if col not in cur_columns_set:
            logger.warning("移除", col)
            qid_cols.remove(col)
            # raise ValueError("当前特征列表不存在准标识符-{}".format(col))

    return qid_cols


def get_match_all_data_except_test_old(hos_id):
    """
    去除匹配样本中包含测试样本
    :param hos_id:
    :return:
    """
    res_old = get_hos_data_X_y_old(hos_id)[1]
    index_old = res_old.index

    match_data_X, match_data_y = get_match_all_data()
    origin_len = match_data_X.shape[0]

    index_all = match_data_X.index
    index_set = set(index_all)

    except_list = []
    for index in index_old:
        if index in index_set:
            except_list.append(index)

    match_condition = match_data_X.index.isin(except_list)
    match_data_X = match_data_X[~match_condition]
    match_data_y = match_data_y[~match_condition]

    logger.warning(f"去除了{len(except_list)}个错误匹配患者,匹配患者数量 {origin_len}->{match_data_X.shape[0]}.")

    return match_data_X, match_data_y



def get_index_diff():
    res_new = get_hos_data_X_y(73)[1]
    res_old = get_hos_data_X_y_old(73)[1]

    index_new = res_new.index
    index_old = res_old.index

    match_all = get_match_all_data()[0]
    index_all = match_all.index

    index_set = set(index_all)
    count_new = 0
    count_old = 0
    for index in index_new:
        if index in index_set:
            count_new += 1

    for index in index_old:
        if index in index_set:
            count_old += 1

    print("count_old", count_old)
    print("count_new", count_new)


if __name__ == '__main__':
    # data = get_all_norm_data()
    # all_data_x, t_data_x, all_data_y, t_data_y = get_all_data_X_y()
    # train_data_x2, test_data_x2, train_data_y2, test_data_y2 = get_hos_data_X_y(73)
    # # test1, test0 = get_target_test_id(73)
    # all_data_x2, t_data_x2, all_data_y2, t_data_y2 = get_train_test_data_X_y()
    # get_match_all_data_except_test(73)

    # version = 1
    # res_old = get_hos_data_X_y(73)
    print("done!")

    # res2 = get_fs_train_test_data_X_y(strategy=2)
    # get_topK_hospital()
