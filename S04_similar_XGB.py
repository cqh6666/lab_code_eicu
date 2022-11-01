# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     pca_similar
   Description:   没做PCA处理，使用初始相似性度量匹配相似样本，进行计算AUC
   Author:        cqh
   date:          2022/7/5 10:07
-------------------------------------------------
   Change Activity:
                  2022/7/5:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import xgboost as xgb

import pandas as pd
from sklearn.metrics import roc_auc_score

from api_utils import covert_time_format, save_to_csv_by_row, get_all_data_X_y, get_hos_data_X_y, \
    get_match_all_data_from_hos_data, get_train_test_data_X_y, get_match_all_data
from email_api import send_success_mail, send_an_error_message, get_run_time
from my_logger import MyLog
from xgb_utils_api import get_xgb_model_pkl, get_local_xgb_para, get_xgb_init_similar_weight

warnings.filterwarnings('ignore')


def get_similar_rank(pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = abs((train_data_x - pre_data_select.values) * init_similar_weight).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        # distance column
        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]

    except Exception as err:
        raise err

    return patient_ids, sample_ki


def xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y, weight=sample_ki)

    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=xgb_model)
    d_test_local = xgb.DMatrix(pre_data_select)
    predict_prob = xgb_local.predict(d_test_local)[0]
    return predict_prob


def personalized_modeling(test_id, pre_data_select):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - dataframe
    :return: 最终的相似样本
    """
    patient_ids, sample_ki = get_similar_rank(pre_data_select)

    try:
        fit_train_x = train_data_x.loc[patient_ids]
        fit_train_y = train_data_y.loc[patient_ids]

        predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)

        global_lock.acquire()
        test_result.loc[test_id, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(traceback.format_exc())
        sys.exit(1)


def print_result_info():
    # 不能存在NaN
    if test_result.isna().sum().sum() > 0:
        print("exist NaN...")
        sys.exit(1)
    test_result.to_csv(test_result_file_name)
    # 计算auc性能
    score = roc_auc_score(test_result['real'], test_result['prob'])
    my_logger.info(f"auc score: {score}")
    # save到全局结果集合里
    save_df = pd.DataFrame(columns=['start_time', 'end_time', 'run_time', 'auc_score_result'])
    start_time_date, end_time_date, run_date_time = get_run_time(run_start_time, run_end_time)
    save_df.loc[program_name + "_" + str(int(run_start_time)), :] = [start_time_date, end_time_date, run_date_time,
                                                                     score]
    save_to_csv_by_row(save_result_file, save_df)

    # 发送邮箱
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=run_end_time)
    print("end!")


if __name__ == '__main__':

    run_start_time = time.time()

    my_logger = MyLog().logger

    pool_nums = 4

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])  # 0 1

    m_sample_weight = 0.01
    xgb_boost_num = 50
    select = 10

    xgb_thread_num = 1
    select_ratio = select * 0.01
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)

    other_hos_id = 167 if hos_id == 73 else 73

    init_psm_id = hos_id  # 初始相似性度量
    is_train_same = False  # 训练样本数是否等样本量
    is_match_all = True  # 是否匹配全局样本
    is_match_other = False  # 是否匹配其他中心样本

    if is_match_all:
        transfer_id = 0
    else:
        transfer_id = hos_id if not is_match_other else other_hos_id

    assert not is_match_all & is_match_other, "不能同时匹配全局或其他中心的数据"
    my_logger.warning("init_psm:{}, is_train_same:{}, is_match_all:{}, transfer_id:{}".format(init_psm_id, is_train_same, is_match_all, transfer_id))

    # 获取xgb_model和初始度量  是否全局匹配
    init_similar_weight = get_xgb_init_similar_weight(init_psm_id)
    xgb_model = get_xgb_model_pkl(transfer_id) if is_transfer == 1 else None
    """
    version=1
    version = 4 中位数填充
    version = 5 类权重
    version = 6 匹配全局（错误版本，没用全局模型）
    version = 7 平均数填充，用全局模型
    version = 8 全局相似性度量 和 全局模型
    version = 9 全局
    ===============================================================
    T  代表重新进行了分割数据
    -6 用类权重的初始相似度量和全局迁移 （自己不做类权重）
    -5 不用类权重的初始相似度量和全局迁移（自己不做类权重）
    -4 用类权重的初始相似度量和全局迁移 （自己也做类权重）
    -3 不用类权重的初始相似度量和全局迁移 （自己也做类权重）
    ===============================================================
    version = 10 基于该中心相似度量匹配中心10%比例  init_psm_id = hos_id, is_train_same = False, is_match_all = False
    version = 11 基于该中心相似度量匹配全局样本10%  init_psm_id = hos_id, is_train_same = False, is_match_all = True
    version = 12 基于全局相似度量匹配该中心10%比例  init_psm_id = 0, is_train_same = False, is_match_all = False
    version = 13 基于该中心相似度量匹配全局样本等样本量  init_psm_id = hos_id, is_train_same = True, is_match_all = True
    version = 14 基于全局相似度量匹配全局样本10% init_psm_id = 0, is_train_same = False, is_match_all = True
    version = 15 基于全局相似度量匹配全局等样本（该中心） init_psm_id = 0, is_train_same = True, is_match_all = True
    
    version = 16 基于该中心相似度量匹配其他中心10%  init_weight hos_id global_weight other_hos_id
    version = 17 基于该中心相似度量匹配其他中心等样本量  init_weight hos_id global_weight other_hos_id
    version = 18 基于其他中心相似度量匹配当前中心10%  init_weight other_hos_id global_weight hos_id
    version = 19 基于其他中心相似度量匹配其他中心10%  init_weight other_hos_id global_weight other_hos_id
    version = 20 基于其他中心相似度量匹配其他中心等样本量  init_weight other_hos_id global_weight other_hos_id
    """
    version = "11"
    # ================== save file name ====================
    program_name = f"S04_XGB_id{hos_id}_tra{is_transfer}_v{version}"
    save_result_file = f"./result/S04_id{hos_id}_XGB_result_save.csv"
    save_path = f"./result/S04/{hos_id}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        my_logger.warning("create new dirs... {}".format(save_path))
    test_result_file_name = os.path.join(
        save_path, f"S04_XGB_test_tra{is_transfer}_boost{xgb_boost_num}_select{select}_v{version}.csv")
    # =====================================================
    # 获取数据
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_data_X_y()
        match_data_len = -1
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)
        # 计算匹配的样本
        match_data_len = int(select_ratio * train_data_x.shape[0])

    # 改为匹配全局，修改为全部数据
    if is_match_all:
        train_data_x, train_data_y = get_match_all_data()
        my_logger.warning(
                "匹配全局数据 - 局部训练集修改为全局训练数据...train_data_shape:{}".format(train_data_x.shape))

    # 改为匹配其他中心
    if is_match_other:
        train_data_x, _, train_data_y, _ = get_hos_data_X_y(other_hos_id)
        my_logger.warning(
            "匹配数据 - 局部训练集修改为其他中心{}训练数据...train_data_shape:{}".format(other_hos_id, train_data_x.shape))

    # 是否等样本量匹配
    if is_train_same:
        assert not hos_id == 0 & match_data_len == -1, "训练全局数据不需要等样本匹配"
        len_split = match_data_len
    else:
        len_split = int(select_ratio * train_data_x.shape[0])

    start_idx = 0
    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # 不要超过10000个样本

    # 分批次进行个性化建模
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))
    my_logger.warning(
        f"[params] - version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, "
        f"index_range:[{start_idx}, {end_idx}]")

    test_id_list = test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = threading.Lock()
    my_logger.warning("starting ...")

    s_t = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    run_end_time = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(run_end_time - s_t)}...")

    print_result_info()
