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

from api_utils import get_fs_hos_data_X_y
import pandas as pd
import random

# 标识符
id_columns = []
# 准标识符
qid_columns = []
# 敏感标识符
sens_columns = []


def is_k_anonymity(df, qid_cols, k=2):
    """
    遍历每一行数据，判断是否符合k匿名
    :param df: 数据dataframe
    :param qid_cols: 准标识符属性
    :param k: 是否查出有k行
    :return: 符合k匿名要求的概率
    """
    pass_count = 0
    all_count = df.shape[0]

    for index, row in df.iterrows():
        query = ' & '.join([f'{col} == {row[col]}' for col in qid_cols])
        rows = df.query(query)
        if rows.shape[0] >= k:
            pass_count += 1
        else:
            # print(index, "不符合k匿名要求...")
            pass
    return pass_count / all_count


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


def genera_feature_values(data_X, target_column, k=2):
    cur_data = data_X[target_column]
    unique_values = cur_data.unique().sort().to_list()










def anony_data(data_X, select_qid_nums=5, select_sens_nums=3):
    qid_cols, sens_cols = select_columns(data_X.columns.to_list())
    # select_qid_cols = qid_cols[:select_qid_nums]
    # select_sens_cols = sens_cols[:select_sens_nums]
    select_qid_cols = ['age', 'med_1', 'px_1', 'ccs_1', 'lab_1', 'med_656']
    cur_data_X = data_X[select_qid_cols]

    # 我的目标是，让k匿名概率达到0.5以上 k=4
    target_column = 'age'
    gener_feature_values(data_X, target_column)



if __name__ == '__main__':
    hos_id = 73
    train_data_x, test_data_x, train_data_y, test_data_y = get_fs_hos_data_X_y(hos_id)

    all_data_X = pd.concat([train_data_x, test_data_x], axis=0)

    anony_data(all_data_X)
