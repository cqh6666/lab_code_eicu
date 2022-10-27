# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_pid
   Description:   ...
   Author:        cqh
   date:          2022/10/26 10:18
-------------------------------------------------
   Change Activity:
                  2022/10/26:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from api_utils import get_match_all_data_from_hos_data, get_target_test_id, get_hos_data_X_y


def get_balanced(pid_dict, version):
    # 匹配样本y标签
    _, match_data_y = get_match_data(version)

    labels_dict = {}

    for test_data_ids in test_data_ids_all:
        label_list = []
        cur_pids = pid_dict[test_data_ids]
        for pid in cur_pids:
            label_list.append(match_data_y.loc[pid])

        labels_dict[test_data_ids] = label_list

    # save
    dict_save_name = os.path.join(save_path, 'target_match_ids_labels_v{}.npy'.format(version))
    np.save(dict_save_name, labels_dict)
    print(version, "save success!")


def get_match_data(version):
    if version == 10 or version == 12:
        match_flag = False
    else:
        match_flag = True

    if match_flag:
        match_data_X, match_data_y = get_match_all_data_from_hos_data(hos_id)
        # my_logger.warning("匹配全局数据 - 局部训练集修改为全局训练数据...train_data_shape:{}".format(train_data_x.shape))
        return match_data_X, match_data_y
    else:
        match_data_X, _, match_data_y, _ = get_hos_data_X_y(hos_id)
        return match_data_X, match_data_y


def get_all_match_labels():
    """
    入口。获取所有match_id的标签
    :return:
    """
    version_list = [10, 11, 12, 13, 14, 15]

    for cur_version in version_list:
        dict_file_name = os.path.join(save_path, 'target_match_ids_v{}.npy'.format(cur_version))
        load_dict = np.load(dict_file_name, allow_pickle=True).item()
        get_balanced(load_dict, cur_version)


def get_match_len(version):
    if version == 11 or version == 14:
        _, match_y = get_match_all_data_from_hos_data(hos_id)
        return int(match_y.shape[0] * 0.1)
    else:
        _, _, match_data_y, _ = get_hos_data_X_y(hos_id)
        return int(match_data_y.shape[0] * 0.1)


def analysis_labels_balanced(version):
    """
    分析分布
    :return:
    """
    dict_save_name = os.path.join(save_path, 'target_match_ids_labels_v{}.npy'.format(version))
    load_dict = np.load(dict_save_name, allow_pickle=True).item()

    select_range = np.arange(0.1, 1.01, 0.1)
    res_df_1 = pd.DataFrame(index=test_data_ids_1, columns=select_range)
    res_df_0 = pd.DataFrame(index=test_data_ids_0, columns=select_range)

    match_len = get_match_len(version)

    # labels 为 1 的测试样本
    for test_id in test_data_ids_1:
        cur_pid_list = load_dict[test_id]

        for select in select_range:
            select_nums = int(select * match_len)
            cur_count = cur_pid_list[:select_nums].count(1)
            res_df_1.loc[test_id, select] = cur_count

    res_1 = res_df_1.mean()
    # labels 为 0 的测试样本
    for test_id in test_data_ids_0:
        cur_pid_list = load_dict[test_id]

        for select in select_range:
            select_nums = int(select * match_len)
            cur_count = cur_pid_list[:select_nums].count(0)
            res_df_0.loc[test_id, select] = cur_count

    res_0 = res_df_0.mean()
    all_df = pd.concat([res_1, res_0], axis=1)
    all_df.columns = [f'v{version}_1', f'v{version}_0']
    return all_df


def analysis_and_save():
    """
    入口，分析占比
    :return:
    """
    version_list = [10, 11, 12, 13, 14, 15]

    for cur_version in version_list:
        tt = analysis_labels_balanced(cur_version)
        save_file = os.path.join(save_path, f'S04_count_percent_v{cur_version}.csv')
        tt.to_csv(save_file)
        print("save success!", cur_version)


def load_csv_file(version_list):
    """
    读取不同count csv文件
    :param version_list:
    :return:
    """
    count_csv_1 = pd.DataFrame()
    count_csv_0 = pd.DataFrame()

    # 读取csv文件
    for cur_version in version_list:
        save_file = os.path.join(save_path, f'S04_count_percent_v{cur_version}.csv')
        cur_csv = pd.read_csv(save_file, index_col=0)
        count_csv_1 = pd.concat([count_csv_1, cur_csv[f'v{cur_version}_1']], axis=1)
        count_csv_0 = pd.concat([count_csv_0, cur_csv[f'v{cur_version}_0']], axis=1)

    # 返回 0 1
    return count_csv_1, count_csv_0

if __name__ == '__main__':
    hos_id = 73
    save_path = f"./result/S04/{hos_id}/"

    # 根据key id获取当前匹配样本，得到对应的标签
    test_data_ids_1, test_data_ids_0 = get_target_test_id(hos_id)
    test_data_ids_all = np.concatenate((test_data_ids_1, test_data_ids_0), axis=0)

    # dict_file_name = os.path.join(save_path, 'target_match_ids_v{}.npy'.format(10))
    # load_dict = np.load(dict_file_name, allow_pickle=True).item()

    # analysis_and_save()
    plot_point()