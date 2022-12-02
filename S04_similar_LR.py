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
import queue
import sys
import time
import warnings
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_EXCEPTION

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from api_utils import covert_time_format, save_to_csv_by_row,  \
    get_fs_hos_data_X_y, get_fs_match_all_data, create_path_if_not_exists, get_fs_each_hos_data_X_y
from lr_utils_api import get_transfer_weight, get_init_similar_weight
from email_api import send_success_mail, get_run_time
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param target_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = abs((train_data_x - target_pre_data_select.values) * init_similar_weight).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values
        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        exec_queue.put("Termination")
        my_logger.exception(err)
        raise Exception(err)

    return patient_ids, sample_ki


def lr_train(fit_train_x, fit_train_y, fit_test_x, sample_ki):
    if len(fit_train_y.value_counts()) <= 1:
        return train_data_y[0]

    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    # lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter, class_weight='balanced')
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(fit_test_x)[0][1]
    return predict_prob


def personalized_modeling(patient_id, pre_data_select_x):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - dataframe
    :return: 最终的相似样本
    """
    # 如果消息队列中消息不为空，说明已经有任务异常了
    if not exec_queue.empty():
        return

    try:
        patient_ids, sample_ki = get_similar_rank(pre_data_select_x)

        fit_train_y = train_data_y.loc[patient_ids]
        fit_test_x, fit_train_x = fit_train_test_data(patient_ids, pre_data_select_x)
        predict_prob = lr_train(fit_train_x, fit_train_y, fit_test_x, sample_ki)
        global_lock.acquire()
        test_result.loc[patient_id, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        exec_queue.put("Termination")
        my_logger.exception(err)
        raise Exception(err)


def fit_train_test_data(patient_ids, pre_data_select_x):
    if is_transfer == 1:
        fit_train_x = transfer_train_data_X.loc[patient_ids]
        fit_test_x = pre_data_select_x * global_feature_weight
    else:
        fit_train_x = train_data_x.loc[patient_ids]
        fit_test_x = pre_data_select_x

    return fit_test_x, fit_train_x


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
    start_time_date, end_time_date, run_date_time = get_run_time(run_start_time, time.time())
    save_df.loc[program_name + "_" + str(os.getpid()), :] = [start_time_date, end_time_date, run_date_time, score]
    save_to_csv_by_row(save_result_file, save_df)

    # 发送邮箱
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=time.time())
    print("end!")


def multi_thread_personal_modeling():
    """
    多线程跑程序
    :return:
    """
    my_logger.warning("starting personalized modelling...")

    s_t = time.time()
    # 匹配相似样本（从训练集） 建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select)
            thread_list.append(thread)

        # 若出现第一个错误, 反向逐个取消任务并终止
        wait(thread_list, return_when=FIRST_EXCEPTION)
        for cur_thread in reversed(thread_list):
            cur_thread.cancel()

        wait(thread_list, return_when=ALL_COMPLETED)

    # 若出现异常直接返回
    if not exec_queue.empty():
        my_logger.error("something task error... we have to stop!!!")
        return

    run_end_time = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(run_end_time - s_t)}...")


if __name__ == '__main__':

    run_start_time = time.time()

    my_logger = MyLog().logger

    pool_nums = 15

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])

    local_lr_iter = 100
    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    other_hos_id = 167 if hos_id == 73 else 73

    init_psm_id = 0  # 初始相似性度量
    is_train_same = False  # 训练样本数是否等样本量
    is_match_all = True  # 是否匹配全局样本

    is_match_other = False  # 是否匹配其他中心样本

    if is_match_all:
        transfer_id = 0
    else:
        transfer_id = hos_id if not is_match_other else other_hos_id

    assert not is_match_all & is_match_other, "不能同时匹配全局或其他中心的数据"

    my_logger.warning("pid:{}， init_psm:{}, is_train_same:{}, is_match_all:{}, transfer_id:{}".format(os.getpid(), init_psm_id, is_train_same, is_match_all, transfer_id))
    init_similar_weight = get_init_similar_weight(init_psm_id)
    global_feature_weight = get_transfer_weight(transfer_id)
    """
    version = 1  local_lr_iter = 100
    version = 2  有错误重新调整
    version = 3  压缩数据
    version = 4 中位数填充
    version = 5 不做类平衡权重会如何
    version = 6 匹配全局数据 （出错了，没用全局度量）
    version = 7 正确版本 使用全局相似性度量和全局迁移参数 平均数填充
    version = 8 使用全局相似性度量和全局初始度量
    version = 9 不用类平衡权重,全局
    
    ===============================================================
    T  代表重新进行了分割数据
    -11 基于该中心（v5）匹配全局迁移(v7)
    -10 基于全局（v7）匹配该中心迁移(v5)
    -9 用0.01：0.99的类权重的初始相似度量和全局迁移 （自己不做类权重）
    -8 用0.05：0.95的类权重的初始相似度量和全局迁移 （自己不做类权重）
    -7 用1：9的类权重的初始相似度量和全局迁移 （自己不做类权重）
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
    version = 19 基于其他中心相似度量匹配其他中心10%  init_weight hos_id global_weight other_hos_id
    version = 20 基于其他中心相似度量匹配其他中心等样本量  init_weight other_hos_id global_weight other_hos_id

    version = 21 全局个性化建模 hos_id = 0
    version = 22 xgb特征选择后的数据 加类权重（LR有影响，XGB没影响）
    version = 23 lr特征选择后的数据  加类权重（LR有影响，XGB没影响）
    version = 24 xgb特征选择后的数据 不加类权重
    version = 25 lr特征选择后的数据  不加类权重
    version = 26 xgb特征选择后的数据  不加类权重  增加离散特征
    version = 27 lr特征选择后的数据  不加类权重  增加离散特征
    version = 28 直接xgb特征选择后的数据  不加类权重
    
    version = 30 
    """
    version = "14"
    # ================== save file name ====================
    save_path = f"./result/S04/{hos_id}/"
    create_path_if_not_exists(save_path)

    program_name = f"S04_LR_id{hos_id}_tra{is_transfer}_v{version}"
    save_result_file = f"./result/S04_id{hos_id}_LR_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S04_LR_test_tra{is_transfer}_boost{local_lr_iter}_select{select}_v{version}.csv")
    # =====================================================
    # 获取数据
    train_data_x, test_data_x, train_data_y, test_data_y = get_fs_each_hos_data_X_y(hos_id)
    match_data_len = int(select_ratio * train_data_x.shape[0])

    # 改为匹配全局，修改为全部数据
    if is_match_all:
        train_data_x, train_data_y = get_fs_match_all_data()
        my_logger.warning("匹配全局数据 - 局部训练集修改为全局训练数据...train_data_shape:{}".format(train_data_x.shape))

    # 改为匹配其他中心
    if is_match_other:
        train_data_x, _, train_data_y, _ = get_fs_each_hos_data_X_y(other_hos_id)
        my_logger.warning(
            "匹配数据 - 局部训练集修改为其他中心{}训练数据...train_data_shape:{}".format(other_hos_id, train_data_x.shape))

    # 是否等样本量匹配
    if is_train_same:
        assert not hos_id == 0, "训练全局数据不需要等样本匹配"
        len_split = match_data_len
    else:
        len_split = int(select_ratio * train_data_x.shape[0])

    start_idx = 0
    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # 不要超过10000个样本

    # 分批次进行个性化建模
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    my_logger.warning(f"[params] - model_select:LR, pool_nums:{pool_nums}, is_transfer:{is_transfer}, "
                      f"max_iter:{local_lr_iter}, select:{select}, index_range:[{start_idx, end_idx}, "
                      f"version:{version}]")
    my_logger.warning("load data - train_data:{}, test_data:{}, len_split:{}".
                      format(train_data_x.shape, test_data_x.shape, len_split))

    # 测试集患者病人ID
    test_id_list = test_data_x.index.values
    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    # 提前计算迁移后的训练集和测试集
    if is_transfer == 1:
        my_logger.warning("is_tra is 1, pre get transfer_train_data_X, transfer_test_data_X...")
        transfer_train_data_X = train_data_x * global_feature_weight
        transfer_test_data_X = test_data_x * global_feature_weight

    # ===================================== 任务开始 ==========================================
    # 多线程用到的全局锁, 异常消息队列
    global_lock, exec_queue = Lock(), queue.Queue()
    # 开始跑程序
    multi_thread_personal_modeling()

    print_result_info()
