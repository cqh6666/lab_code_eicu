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
from threading import Lock
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from api_utils import covert_time_format, save_to_csv_by_row, get_all_data_X_y, get_hos_data_X_y
from email_api import send_success_mail, get_run_time
from lr_utils_api import get_transfer_weight, get_init_similar_weight
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param target_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=pca_train_data_x.index)
        similar_rank['distance'] = abs(pca_train_data_x - target_pre_data_select.values).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        print(err)
        sys.exit(1)

    return patient_ids, sample_ki


def lr_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki):
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(pre_data_select_)[0][1]
    return predict_prob


def fit_train_test_data(patient_ids, pre_data_select_x, is_tra):
    select_train_x = train_data_x.loc[patient_ids]
    if is_tra == 1:
        transfer_weight = global_feature_weight
        fit_train_x = select_train_x * transfer_weight
        fit_test_x = pre_data_select_x * transfer_weight
    else:
        fit_train_x = select_train_x
        fit_test_x = pre_data_select_x
    return fit_test_x, fit_train_x


def personalized_modeling(patient_id, pre_data_select_x, pca_pre_data_select_x):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    :param pre_data_select_x: 原始测试样本
    :param pca_pre_data_select_x:  处理后的测试样本
    :param patient_id:
    :return:
    """
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_x)
    try:
        fit_train_y = train_data_y.loc[patient_ids]
        fit_test_x, fit_train_x = fit_train_test_data(patient_ids, pre_data_select_x, is_transfer)
        predict_prob = lr_train(fit_train_x, fit_train_y, fit_test_x, sample_ki)
        global_lock.acquire()
        test_result.loc[patient_id, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(err)
        sys.exit(1)


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    if n_comp >= train_x.shape[1]:
        n_comp = train_x.shape[1] - 1

    my_logger.warning(f"starting pca by train_data...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


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
    save_df.loc[program_name + "_" + str(int(run_start_time)), :] = [start_time_date, end_time_date, run_date_time, score]
    save_to_csv_by_row(save_result_file, save_df)

    # 发送邮箱
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=run_end_time)
    print("end!")


if __name__ == '__main__':

    run_start_time = time.time()
    my_logger = MyLog().logger

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])
    # start_idx = int(sys.argv[3])
    # end_idx = int(sys.argv[4])  # 50520
    n_components = int(sys.argv[3])

    if hos_id == 0:
        pool_nums = 20
    else:
        pool_nums = 5

    start_idx = 0
    local_lr_iter = 100
    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    global_feature_weight = get_transfer_weight(hos_id)
    init_similar_weight = get_init_similar_weight(hos_id)
    """
    version = 1
    version = 2 不分批测试
    version = 3 不分批正式
    """
    version = 3
    # ================== save file name ====================
    program_name = f"S05_LR_id{hos_id}_tra{is_transfer}_comp{n_components}"
    save_result_file = f"./result/all_result_save.csv"
    save_path = f"./result/S05/{hos_id}/"
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
            my_logger.warning("create new dirs... {}".format(save_path))
        except Exception:
            pass

    test_result_file_name = os.path.join(
        save_path, f"S05_lr_test_tra{is_transfer}_boost{local_lr_iter}_comp{n_components}_v{version}.csv")
    # =====================================================

    # 获取数据
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_all_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)

    # final_idx = test_data_x.shape[0]
    # end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # 不要超过10000个样本

    # 分批次进行个性化建模
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    # PCA降维
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))
    my_logger.warning(
        f"[params] - model_select:LR, pool_nums:{pool_nums}, is_transfer:{is_transfer}, max_iter:{local_lr_iter}, select:{select}, version:{version}, test_idx:[{start_idx}, {end_idx}]")

    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = Lock()
    my_logger.warning("starting personalized modelling...")
    s_t = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            pca_pre_data_select = pca_test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select, pca_pre_data_select)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    run_end_time = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(run_end_time - s_t)}...")

    print_result_info()



