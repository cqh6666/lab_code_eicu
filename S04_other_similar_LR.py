# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     pca_similar
   Description:   û��PCA����ʹ�ó�ʼ�����Զ���ƥ���������������м���AUC
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

from api_utils import covert_time_format, save_to_csv_by_row,  create_path_if_not_exists, get_fs_each_hos_data_X_y
from lr_utils_api import get_transfer_weight, get_init_similar_weight
from email_api import send_success_mail, get_run_time
from my_logger import logger

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    ѡ��ǰ10%�����������Ҹ������Ƶõ�����Ȩ��
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
        logger.exception(err)
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
    ���ݾ���õ� ĳ��Ŀ�����������ÿ��ѵ�������ľ���
    test_id - patient id
    pre_data_select - dataframe
    :return: ���յ���������
    """
    # �����Ϣ��������Ϣ��Ϊ�գ�˵���Ѿ��������쳣��
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
        logger.exception(err)
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
    # ���ܴ���NaN
    if test_result.isna().sum().sum() > 0:
        print("exist NaN...")
        sys.exit(1)
    test_result.to_csv(test_result_file_name)
    # ����auc����
    score = roc_auc_score(test_result['real'], test_result['prob'])
    logger.info(f"auc score: {score}")
    # save��ȫ�ֽ��������
    save_df = pd.DataFrame(columns=['start_time', 'end_time', 'run_time', 'auc_score_result'])
    start_time_date, end_time_date, run_date_time = get_run_time(run_start_time, time.time())
    save_df.loc[program_name + "_" + str(os.getpid()), :] = [start_time_date, end_time_date, run_date_time, score]
    save_to_csv_by_row(save_result_file, save_df)

    # ��������
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=time.time())
    print("end!")


def multi_thread_personal_modeling():
    """
    ���߳��ܳ���
    :return:
    """
    logger.warning("starting personalized modelling...")

    s_t = time.time()
    # ƥ��������������ѵ������ ��ģ ���߳�
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select)
            thread_list.append(thread)

        # �����ֵ�һ������, �������ȡ��������ֹ
        wait(thread_list, return_when=FIRST_EXCEPTION)
        for cur_thread in reversed(thread_list):
            cur_thread.cancel()

        wait(thread_list, return_when=ALL_COMPLETED)

    # �������쳣ֱ�ӷ���
    if not exec_queue.empty():
        logger.error("something task error... we have to stop!!!")
        return

    run_end_time = time.time()
    logger.warning(f"done - cost_time: {covert_time_format(run_end_time - s_t)}...")


if __name__ == '__main__':

    run_start_time = time.time()

    pool_nums = 5

    from_hos_id = int(sys.argv[1])
    to_hos_id = int(sys.argv[2])
    is_transfer = int(sys.argv[3])

    local_lr_iter = 100
    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    init_psm_id = from_hos_id  # ��ʼ�����Զ���
    transfer_id = to_hos_id

    is_train_same = False  # ѵ���������Ƿ��������

    logger.warning("pid:{}�� init_psm:{}, is_train_same:{}, transfer_id:{}".format(os.getpid(), init_psm_id, is_train_same, transfer_id))
    init_similar_weight = get_init_similar_weight(init_psm_id)
    global_feature_weight = get_transfer_weight(transfer_id)
    """
    version = 1  ƥ���������� �����Զ���(from) Ǩ��(to)  ʹ�ö���������Զ���
    version = 2  ƥ���������� �����Զ���(from) Ǩ��(to)  ʹ���Լ��������Զ���
    version = 3  ƥ���������� �����Զ���(from) Ǩ��(to)  �õ�ǰ���ĵ�10%����
    
    version = 5  73����ƥ������������������
    version = 6  0����ƥ������������������
    
    """
    version = "6"
    # ================== save file name ====================
    save_path = f"./result/S04/{from_hos_id}/"
    create_path_if_not_exists(save_path)

    program_name = f"S04_LR_from{from_hos_id}_to{to_hos_id}_tra{is_transfer}_v{version}"
    save_result_file = f"./result/S04_id{from_hos_id}_other_LR_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S04_LR_test_tra{is_transfer}_boost{local_lr_iter}_select{select}_other_v{version}.csv")
    # =====================================================
    # ��ȡ����
    t_x, test_data_x, _, test_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    train_data_x, _, train_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
    match_data_len = int(select_ratio * t_x.shape[0])

    # # �Ƿ��������ƥ��
    if is_train_same:
        assert not from_hos_id == 0, "ѵ��ȫ�����ݲ���Ҫ������ƥ��"
        len_split = match_data_len
    else:
        len_split = int(select_ratio * train_data_x.shape[0])

    start_idx = 0
    final_idx = test_data_x.shape[0]
    # end_idx = final_idx if final_idx < 10000 else 10000  # ��Ҫ����10000������
    end_idx = final_idx

    # �����ν��и��Ի���ģ
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    logger.warning(f"[params] - model_select:LR, pool_nums:{pool_nums}, is_transfer:{is_transfer}, "
                   f"max_iter:{local_lr_iter}, select:{select}, index_range:[{start_idx, end_idx}, "
                   f"version:{version}]")
    logger.warning("load data - train_data:{}, test_data:{}, len_split:{}".
                   format(train_data_x.shape, test_data_x.shape, len_split))

    # ���Լ����߲���ID
    test_id_list = test_data_x.index.values
    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    # ��ǰ����Ǩ�ƺ��ѵ�����Ͳ��Լ�
    if is_transfer == 1:
        logger.warning("is_tra is 1, pre get transfer_train_data_X, transfer_test_data_X...")
        transfer_train_data_X = train_data_x * global_feature_weight

    # ===================================== ����ʼ ==========================================
    # ���߳��õ���ȫ����, �쳣��Ϣ����
    global_lock, exec_queue = Lock(), queue.Queue()
    # ��ʼ�ܳ���
    multi_thread_personal_modeling()

    print_result_info()
