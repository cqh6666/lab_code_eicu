# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S07_k_anonymity
   Description:   k匿名实现
   Author:        cqh
   date:          2022/11/10 16:06
-------------------------------------------------
   Change Activity:
                  2022/11/10:
-------------------------------------------------
"""
__author__ = 'cqh'

import os.path

from api_utils import get_qid_columns, get_sensitive_columns, get_fs_each_hos_data_X_y, get_topK_hospital, \
    get_fs_hos_data_X_y_from_all, create_path_if_not_exists
import pandas as pd
import random
import numpy as np

# 标识符
id_columns = []
# 准标识符
qid_columns = get_qid_columns()
# 敏感标识符
sens_columns = get_sensitive_columns()


def identify_risk(df, qid_cols, k=2):
    """
    遍历每一行数据，判断是否符合k匿名
    :param df: 数据dataframe
    :param qid_cols: 准标识符属性
    :param k: 是否查出有k行
    :return: 符合k匿名要求的概率
    """
    risk_count = 0
    all_count = df.shape[0]

    for index, row in df.iterrows():
        query = ' & '.join([f'`{col}` == {row[col]}' for col in qid_cols])
        rows = df.query(query)
        if rows.shape[0] < k:
            risk_count += 1

    return risk_count / all_count


def is_l_diversity(df, qid_cols, sens_cols, l=2):
    """
    即在公开的数据中，对于那些准标识符相同的数据中，敏感数据必须具有多样性，这样才能保证用户的隐私不能通过背景知识等方法推测出来。
    :param df:
    :param qid_cols:
    :param sens_cols:
    :param l:
    :return:
    """
    pass_count = 0
    all_count = df.shape[0] * len(sens_cols)
    for index, row in df.iterrows():
        query = ' & '.join([f'{col} == {row[col]}' for col in qid_cols])
        rows = df.query(query)
        for sen_col in sens_cols:
            if rows[sen_col].value_count().shape[0] < l:
                print(sen_col, "敏感特征暴露...")
            else:
                pass_count += 1

    return pass_count / all_count


def select_columns(all_columns, qid_rate=0.04, sens_rate=0.01):
    """
    随机选择特征(准标识符和敏感特征）
    :return:
    """
    random.seed(2022)
    len_cols = len(all_columns)
    qid_nums = int(len_cols * qid_rate)
    sens_nums = int(len_cols * sens_rate)

    select_cols = random.sample(all_columns, qid_nums + sens_nums)
    qid_cols = select_cols[:qid_nums]
    sens_cols = select_cols[qid_nums:]

    return qid_cols, sens_cols


def all_identify_risk():
    hos_ids = get_topK_hospital(50)

    res_df = pd.DataFrame(columns=['all'])

    for hod_idx in range(0, len(hos_ids), 2):
        train_data_x, _, _, _ = get_fs_hos_data_X_y_from_all(hos_ids[hod_idx])
        train_data_x2, _, _, _ = get_fs_hos_data_X_y_from_all(hos_ids[hod_idx + 1])
        train_data = pd.concat([train_data_x, train_data_x2], axis=0)
        print(f"hos{hos_ids[hod_idx]}_{hos_ids[hod_idx + 1]}", train_data.shape)

        prob = identify_risk(train_data, qid_cols=qid_columns)
        res_df.loc[f"hos{hos_ids[hod_idx]}_{hos_ids[hod_idx + 1]}", "all"] = prob
        print(f"hos{hos_ids[hod_idx]}_{hos_ids[hod_idx + 1]}", "all", prob)
        print("=============================================================")

    res_df.to_csv(os.path.join(save_path, f"S07_all_result_train_df_v{version}.csv"))
    print("done!")


def all_random_identify_risk(hos_nums=50, len_sample=500):
    hos_ids = get_topK_hospital(hos_nums)

    res_df = pd.DataFrame()

    for hod_idx in range(0, len(hos_ids), 2):
        train_data_x, _, _, _ = get_fs_hos_data_X_y_from_all(hos_ids[hod_idx])
        train_data_x2, _, _, _ = get_fs_hos_data_X_y_from_all(hos_ids[hod_idx + 1])
        train_data = pd.concat([train_data_x, train_data_x2], axis=0)
        cur_name = f"hos{hos_ids[hod_idx]}_{hos_ids[hod_idx + 1]}"

        print(cur_name, train_data.shape)

        if train_data.shape[0] < len_sample:
            break

        for idx in range(0, 5):
            temp_data_x = train_data.sample(n=len_sample)
            prob_temp = identify_risk(temp_data_x, qid_cols=qid_columns)
            res_df.loc[cur_name, "random_" + str(idx)] = prob_temp

        res_df.loc[cur_name, "median"] = res_df.loc[cur_name, :].median()

        print(cur_name, res_df.loc[cur_name, "median"], "done!")
        print("====================================================")
    res_df.to_csv(os.path.join(save_path, f"S07_all_random_{len_sample}_result_train_df_v{version}.csv"))
    print("done!")


def all_each_identify_risk():
    hos_ids = get_topK_hospital(50)
    # age demo vital px ccs

    qid_split = {
        "age": [qid_columns[0]],
        "race_gender": qid_columns[1:8],
        "bmi": [qid_columns[8]],
        "px": qid_columns[9:13],
        "ccs": qid_columns[13:]
    }

    res_cols = list(qid_split.keys()) + ["all"]
    res_df = pd.DataFrame(index=hos_ids, columns=res_cols)

    for hos_index, hos_id in enumerate(hos_ids):
        train_data_x, test_data_x, train_data_y, test_data_y = get_fs_hos_data_X_y_from_all(hos_id)
        # 是否有效
        flag = True

        for col_name, col_list in qid_split.items():
            prob_temp = identify_risk(test_data_x, qid_cols=col_list)

            res_df.loc[hos_id, col_name] = prob_temp
            print(hos_id, col_name, prob_temp)

            if prob_temp == 1.0:
                print(hos_id, "识别概率为1了，不做下一步...")
                print("=============================================================")
                flag = False
                break

        if flag:
            prob = identify_risk(test_data_x, qid_cols=qid_columns)
            res_df.loc[hos_id, "all"] = prob
            print(hos_id, "all", prob)
            print("=============================================================")

    res_df.to_csv("./S07_result_df.csv")
    print("done!")


if __name__ == '__main__':
    save_path = "/home/chenqinhai/code_eicu/my_lab/result/S07"
    create_path_if_not_exists(save_path)
    version = 6

    # all_identify_risk()
    sample_list = [1000, 1500]
    for sample_cur in sample_list:
        all_random_identify_risk(len_sample=sample_cur)
