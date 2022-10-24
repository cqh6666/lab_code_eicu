# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0006_global_LR
   Description:   全局数据的LR得到weight
   Author:        cqh
   date:          2022/5/23 10:39
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import pickle
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
from api_utils import get_hos_data_X_y, get_all_data_X_y

import time
import os
import pandas as pd
import warnings

from email_api import send_success_mail

warnings.filterwarnings('ignore')


def save_weight_importance_to_csv(train_iter, weight_important):
    # 不标准化 初始特征重要性
    weight_importance_df = pd.DataFrame({"feature_weight": weight_important})
    tra_file = transfer_weight_file.format(train_iter)
    weight_importance_df.to_csv(tra_file, index=False)
    print(f"save to csv success! - {tra_file}")

    # 标准化 用作psm_0
    weight_importance = [abs(i) for i in weight_important]
    normalize_weight_importance = [i / sum(weight_importance) for i in weight_importance]
    normalize_weight_importance_df = pd.DataFrame({"normalize_weight": normalize_weight_importance})
    psm_file = init_psm_weight_file.format(train_iter)
    normalize_weight_importance_df.to_csv(psm_file, index=False)
    print(f"save to csv success! - {psm_file}")


def global_train(train_iter):
    start_time = time.time()

    train_x_ft = train_data_x
    test_x_ft = test_data_x

    lr_all = LogisticRegression(solver='liblinear', max_iter=train_iter, n_jobs=-1)
    lr_all.fit(train_x_ft, train_data_y)
    y_predict = lr_all.decision_function(test_x_ft)
    auc = roc_auc_score(test_data_y, y_predict)
    recall = recall_score(test_data_y, lr_all.predict(test_data_x))
    run_time = round(time.time() - start_time, 2)

    # save feature weight
    weight_importance = lr_all.coef_[0]
    save_weight_importance_to_csv(train_iter, weight_importance)

    # save model
    file = model_file_name_file.format(train_iter)
    pickle.dump(lr_all, open(file, "wb"))
    print(f"save lr model to pkl - [{file}]")

    print(
        f'[global] - max_iter:{train_iter}, train_iter:{lr_all.n_iter_}, cost time: {run_time} s, auc: {auc}, recall: {recall}')

    return auc, recall, run_time


def get_train_data_for_random_idx(train_x, train_y, select_rate):
    """
    选取10%的索引列表，利用这索引列表随机选取数据
    :param train_x:
    :param train_y:
    :param select_rate:
    :return:
    """
    len_split = int(train_x.shape[0] * select_rate)
    random_idx = random.sample(list(range(train_x.shape[0])), len_split)

    train_x_ = train_x.iloc[random_idx, :]
    train_x_.reset_index(drop=True, inplace=True)

    train_y_ = train_y.iloc[random_idx]
    train_y_.reset_index(drop=True, inplace=True)

    print(f"select sub train x shape {train_x_.shape}.")
    return train_x_, train_y_


def sub_global_train(select_rate=0.1, is_transfer=1, local_iter_idx=100):
    """
    选取10%的数据进行训练
    :param select_rate:
    :param is_transfer:
    :param local_iter_idx:
    :return:
    """
    start_time = time.time()

    train_x_ft, train_y_ft = get_train_data_for_random_idx(train_data_x, train_data_y, select_rate)

    if is_transfer == 1:
        fit_train_x = train_x_ft * global_feature_weight
        fit_test_x = test_data_x * global_feature_weight
    else:
        fit_train_x = train_x_ft
        fit_test_x = test_data_x

    lr_local = LogisticRegression(max_iter=local_iter_idx, solver="liblinear")
    lr_local.fit(fit_train_x, train_y_ft)
    y_predict = lr_local.decision_function(fit_test_x)
    auc = roc_auc_score(test_data_y, y_predict)

    run_time = round(time.time() - start_time, 2)

    print(
        f'[sub_global] - solver:liblinear, max_iter:{local_iter_idx}, is_tra:{is_transfer}, train_iter:{lr_local.n_iter_}, cost time: {run_time} s, auc: {auc}')
    return auc, run_time


if __name__ == '__main__':
    run_start_time = time.time()
    global_max_iter = 1000
    hos_id = int(sys.argv[1])
    MODEL_SAVE_PATH = f'./result/S03/{hos_id}'
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_all_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)

    # ============================= save file ==================================== #
    program_name = f"S03_global_LR"
    """
    version = 3 不做类平衡权重
    version = 4 做类平衡权重
    """
    # version = 3 不做类平衡权重的AUC
    model_file_name_file = os.path.join(MODEL_SAVE_PATH, "S03_global_lr_{}_v4.pkl")
    transfer_weight_file = os.path.join(MODEL_SAVE_PATH, "S03_global_weight_lr_{}_v4.csv")
    init_psm_weight_file = os.path.join(MODEL_SAVE_PATH, "S03_0_psm_global_lr_{}_v4.csv")
    save_result_file = os.path.join(MODEL_SAVE_PATH, "S03_auc_global_lr_v4.csv")
    save_result_file2 = os.path.join(MODEL_SAVE_PATH, "S03_auc_sub_global_lr_v4.csv")
    # ============================= save file ==================================== #

    global_auc = pd.DataFrame()
    # for max_idx in range(200, 1001, 200):
    #     global_auc.loc[max_idx, 'auc_score'], global_auc.loc[max_idx, 'recall_score'], global_auc.loc[max_idx, 'cost_time'] = global_train(max_idx)
    global_auc.loc[global_max_iter, 'auc_score'], global_auc.loc[global_max_iter, 'recall_score'], global_auc.loc[global_max_iter, 'cost_time'] = global_train(global_max_iter)

    global_auc.to_csv(save_result_file)
    print("global done!")

    global_feature_weight = pd.read_csv(transfer_weight_file.format(global_max_iter)).squeeze().tolist()
    # frac_list = np.arange(0.05, 1.01, 0.05)
    frac_list = [0.1]
    transfers = [0]
    local_iter = [100]
    sub_global_auc = pd.DataFrame()
    index = 0
    for i in frac_list:
        for j in transfers:
            for k in local_iter:
                temp_auc, cost_time = sub_global_train(select_rate=i, is_transfer=j, local_iter_idx=k)
                sub_global_auc.loc[index, 'select_rate'] = i
                sub_global_auc.loc[index, 'is_transfer'] = j
                sub_global_auc.loc[index, 'local_iter'] = k
                sub_global_auc.loc[index, 'auc_score'] = temp_auc
                sub_global_auc.loc[index, 'cost_time'] = cost_time
                index += 1

    sub_global_auc.to_csv(save_result_file2)

    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=time.time())
    print("done!")
