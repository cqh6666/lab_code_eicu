# encoding=gbk
"""
train a xgboost for data,
the parameters refer to BR

"""
import random
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, recall_score
import pickle
import os
import numpy as np

from api_utils import get_all_data_X_y, get_hos_data_X_y, get_train_test_data_X_y
import time

from email_api import send_success_mail


def get_xgb_params(num_boost):
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': -1,
        'verbosity': 0,
        'seed': 2022,
        'tree_method': 'hist'
    }
    num_boost_round = num_boost
    return params, num_boost_round


def xgb_train_global(train_data_x_, train_data_y_, num_boost, xgb_model_, save=True):
    d_train = xgb.DMatrix(train_data_x_, label=train_data_y_)
    d_test = xgb.DMatrix(test_data_x, label=test_data_y)

    start = time.time()

    params, num_boost_round = get_xgb_params(num_boost)

    model = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=num_boost,
        xgb_model=xgb_model_,
        verbose_eval=False,
    )

    run_time = round(time.time() - start, 2)

    test_y_predict = model.predict(d_test)
    auc = roc_auc_score(test_data_y, test_y_predict)

    print(f'train time: {run_time} | num_boost_round: {num_boost} : The auc of this model is {auc}')

    # save model
    if save:
        pickle.dump(model, open(model_file_name_file.format(num_boost), "wb"))
        print(f"save xgb model to pkl ")
        save_weight_importance(model, num_boost)

    return auc, run_time


def xgb_train_sub_global(num_boost, is_transfer=1, select_rate=0.1):
    train_x, train_y = get_sub_train_data(select_rate=select_rate)
    if is_transfer == 1:
        model = xgb_model
    else:
        model = None

    print(
        f"local xgb train params: select_rate:{select_rate}, is_transfer:{is_transfer}, num_boost:{num_boost}")
    return xgb_train_global(train_x, train_y, num_boost, model, False)


def save_weight_importance(model, num_boost):
    # get weights of feature
    weight_importance = model.get_score(importance_type='weight')
    # gain_importance = model.get_score(importance_type='gain')
    # cover_importance = model.get_score(importance_type='cover')

    # 保存特征重要性
    result = pd.Series(index=test_data_x.columns.tolist(), dtype='float64')
    weight = pd.Series(weight_importance, dtype='float')
    result.loc[:] = weight
    result = result / result.sum()
    result.fillna(0, inplace=True)
    result.to_csv(init_psm_weight_file.format(num_boost), index=False)
    print(f"save feature important weight to csv success!", init_psm_weight_file.format(num_boost))


def get_important_weight(file_name):
    weight_result = os.path.join(MODEL_SAVE_PATH, file_name)
    normalize_weight = pd.read_csv(weight_result)
    print(normalize_weight.shape)
    print(normalize_weight.head())


def get_sub_train_data(select_rate):
    len_split = int(train_data_x.shape[0] * select_rate)
    random_idx = random.sample(list(range(train_data_x.shape[0])), len_split)

    train_x_ = train_data_x.iloc[random_idx, :]
    train_x_.reset_index(drop=True, inplace=True)

    train_y_ = train_data_y.iloc[random_idx]
    train_y_.reset_index(drop=True, inplace=True)

    return train_x_, train_y_


if __name__ == '__main__':
    run_start_time = time.time()
    # 自定义日志
    global_boost_nums = 1000
    hos_id = int(sys.argv[1])
    # hos_id = 0
    MODEL_SAVE_PATH = f'./result/S03/{hos_id}'
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)

    # select_srate = int(sys.argv[2])
    # ============================= save file ==================================== #
    """
    version = 5 重新按7:3分割数据，不做类平衡权重（只需要做全局训练【跟之前不一样】，各中心医院数据分割和之前一样）
    version = 6 重新按7:3分割数据，做类平衡权重（只需要做全局训练【跟之前不一样】，各中心医院数据分割和之前一样）
    """
    program_name = f"S03_global_XGB"
    version = 5
    model_file_name_file = os.path.join(MODEL_SAVE_PATH, "S03_global_xgb_{}_v" + "{}.pkl".format(version))
    init_psm_weight_file = os.path.join(MODEL_SAVE_PATH, "S03_0_psm_global_xgb_{}_v" + "{}.csv".format(version))
    save_result_file = os.path.join(MODEL_SAVE_PATH, "S03_auc_global_xgb_v" + "{}.csv".format(version))
    save_result_file2 = os.path.join(MODEL_SAVE_PATH, "S03_auc_sub_global_xgb_v" + "{}.csv".format(version))
    # ============================= save file ==================================== #

    global_auc = pd.DataFrame()
    # for max_idx in range(600, 1001, 100):
    #     global_auc.loc[max_idx, 'auc_score'], global_auc.loc[max_idx, 'cost_time'] = \
    #         xgb_train_global(train_data_x, train_data_y, xgb_model_=None, num_boost=max_idx, save=True)

    global_auc.loc[global_boost_nums, 'auc_score'], global_auc.loc[global_boost_nums, 'cost_time'] = \
        xgb_train_global(train_data_x, train_data_y, xgb_model_=None, num_boost=global_boost_nums, save=True)

    global_auc.to_csv(save_result_file)
    print("done!")

    xgb_model = pickle.load(open(model_file_name_file.format(global_boost_nums), "rb"))

    # frac_list = np.arange(0.05, 1.01, 0.05)
    frac_list = [0.1]
    transfers = [0, 1]
    num_boosts = [50]
    sub_global_auc = pd.DataFrame()
    index = 0
    for i in frac_list:
        for j in transfers:
            for k in num_boosts:
                temp_auc, cost_time = xgb_train_sub_global(num_boost=k, is_transfer=j, select_rate=i)
                sub_global_auc.loc[index, 'select_rate'] = i
                sub_global_auc.loc[index, 'is_transfer'] = j
                sub_global_auc.loc[index, 'local_iter'] = k
                sub_global_auc.loc[index, 'auc_score'] = temp_auc
                sub_global_auc.loc[index, 'cost_time'] = cost_time
                index += 1

    sub_global_auc.to_csv(save_result_file2)
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=time.time())
    print("done!")
