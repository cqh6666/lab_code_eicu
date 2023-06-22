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

from api_utils import covert_time_format, save_to_csv_by_row, create_path_if_not_exists, get_cross_data_with_drg

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
        # logger.info(f"[{patient_id}]: {patient_ids[:5]}")

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


def process_test_data_no_match(train_data_y, test_data_x, test_data_y):
    """pass"""
    data_index_set = set(train_data_y.index)
    test_data_index = test_data_x.index

    good_list = []

    for index in test_data_index:
        if index not in data_index_set:
            good_list.append(index)

    logger.warning(f"�� {len(good_list)} ��Ŀ�껼�߲���match���ݼ�֮��")
    return test_data_x.loc[good_list], test_data_y.loc[good_list]


def process_test_data_match(train_data_y, test_data_x, test_data_y):
    """pass"""
    data_index_set = set(train_data_y.index)
    test_data_index = test_data_x.index

    good_list = []

    for index in test_data_index:
        if index in data_index_set:
            good_list.append(index)

    logger.warning(f"�� {len(good_list)} ��Ŀ�껼����match���ݼ�֮��")
    return test_data_x.loc[good_list], test_data_y.loc[good_list]


def global_train(train_iter=1000):
    train_x_ft = train_data_x
    test_x_ft = test_data_x

    lr_all = LogisticRegression(max_iter=train_iter, solver="liblinear")
    logger.warning("start global training...")
    start_time = time.time()
    lr_all.fit(train_x_ft, train_data_y)
    y_predict = lr_all.decision_function(test_x_ft)
    auc = roc_auc_score(test_data_y, y_predict)
    run_time = round(time.time() - start_time, 2)

    # save feature weight
    weight_importance = lr_all.coef_[0]
    abs_weight_importance = [abs(i) for i in weight_importance]
    normalize_weight_importance = [i / sum(abs_weight_importance) for i in weight_importance]

    print(
        f'[global] - max_iter:{train_iter}, train_iter:{lr_all.n_iter_}, cost time: {run_time} s, auc: {auc}')

    return normalize_weight_importance, weight_importance


if __name__ == '__main__':

    run_start_time = time.time()

    pool_nums = 2

    hos_id = 0
    is_transfer = 1
    test_valid_id = int(sys.argv[1])
    # test_valid_id = 1

    # ���LR�����Զ�����Ǩ�ƶ���
    train_data_x, test_data_x, train_data_y, test_data_y = get_cross_data_with_drg(test_valid_id=test_valid_id)
    init_similar_weight, global_feature_weight = global_train()

    local_lr_iter = 100
    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    init_psm_id = hos_id  # ��ʼ�����Զ���
    transfer_id = init_psm_id

    # init_similar_weight = get_init_similar_weight(transfer_id)
    # global_feature_weight = get_transfer_weight(transfer_id)
    # logger.warning(f"init weight: {init_similar_weight[:5]}")

    """
    version = 1  local_lr_iter = 100
    version = 2  �д������µ���
    version = 3  ѹ������
    version = 4 ��λ�����
    version = 5 ������ƽ��Ȩ�ػ����
    version = 6 ƥ��ȫ������ �������ˣ�û��ȫ�ֶ�����
    version = 7 ��ȷ�汾 ʹ��ȫ�������Զ�����ȫ��Ǩ�Ʋ��� ƽ�������
    version = 8 ʹ��ȫ�������Զ�����ȫ�ֳ�ʼ����
    version = 9 ������ƽ��Ȩ��,ȫ��
    
    ===============================================================
    T  �������½����˷ָ�����
    -11 ���ڸ����ģ�v5��ƥ��ȫ��Ǩ��(v7)
    -10 ����ȫ�֣�v7��ƥ�������Ǩ��(v5)
    -9 ��0.01��0.99����Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�� ���Լ�������Ȩ�أ�
    -8 ��0.05��0.95����Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�� ���Լ�������Ȩ�أ�
    -7 ��1��9����Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�� ���Լ�������Ȩ�أ�
    -6 ����Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�� ���Լ�������Ȩ�أ�
    -5 ������Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�ƣ��Լ�������Ȩ�أ�
    -4 ����Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�� ���Լ�Ҳ����Ȩ�أ�
    -3 ������Ȩ�صĳ�ʼ���ƶ�����ȫ��Ǩ�� ���Լ�Ҳ����Ȩ�أ�
    ===============================================================
    version = 10 ���ڸ��������ƶ���ƥ������10%����  init_psm_id = hos_id, is_train_same = False, is_match_all = False
    version = 11 ���ڸ��������ƶ���ƥ��ȫ������10%  init_psm_id = hos_id, is_train_same = False, is_match_all = True
    version = 12 ����ȫ�����ƶ���ƥ�������10%����  init_psm_id = 0, is_train_same = False, is_match_all = False
    version = 13 ���ڸ��������ƶ���ƥ��ȫ��������������  init_psm_id = hos_id, is_train_same = True, is_match_all = True
    version = 14 ����ȫ�����ƶ���ƥ��ȫ������10% init_psm_id = 0, is_train_same = False, is_match_all = True
    version = 15 ����ȫ�����ƶ���ƥ��ȫ�ֵ������������ģ� init_psm_id = 0, is_train_same = True, is_match_all = True
    
    version = 16 ���ڸ��������ƶ���ƥ����������10%  init_weight hos_id global_weight other_hos_id
    version = 17 ���ڸ��������ƶ���ƥ���������ĵ�������  init_weight hos_id global_weight other_hos_id
    version = 18 ���������������ƶ���ƥ�䵱ǰ����10%  init_weight other_hos_id global_weight hos_id
    version = 19 ���������������ƶ���ƥ����������10%  init_weight hos_id global_weight other_hos_id
    version = 20 ���������������ƶ���ƥ���������ĵ�������  init_weight other_hos_id global_weight other_hos_id

    version = 21 ȫ�ָ��Ի���ģ hos_id = 0
    version = 22 xgb����ѡ�������� ����Ȩ�أ�LR��Ӱ�죬XGBûӰ�죩
    version = 23 lr����ѡ��������  ����Ȩ�أ�LR��Ӱ�죬XGBûӰ�죩
    version = 24 xgb����ѡ�������� ������Ȩ��
    version = 25 lr����ѡ��������  ������Ȩ��
    version = 26 xgb����ѡ��������  ������Ȩ��  ������ɢ����
    version = 27 lr����ѡ��������  ������Ȩ��  ������ɢ����
    version = 28 ֱ��xgb����ѡ��������  ������Ȩ��
    
    version = 13 B ʹ�������� v5 , Ȩ�� v5a ��Ӧ��
    version = 13 C ʹ��ԭʼ���� v1,  Ȩ�� v1a
    version = 13 D ʹ��ԭʼ���� v1, Ȩ�� v5
    version = 13 E ʹ�������� v5, Ȩ�� v5
    version = 13 F ʹ�������� v5, Ȩ�� v5a ƥ��ȫ�����ݲ����д���(��ȫ�������Զ�����
    version = 13 G ʹ�������� v5, Ȩ�� v5a ʹ���ظ���ѵ������Ϊ���Լ�
    version = 13 H ʹ�������� v5, Ȩ�� v5a ȥ���������Լ���������ƥ������
    version = 13 H-2 ʹ�������� v5, Ȩ�� v5a ȥ���������Լ���������ƥ������(��ȫ�ֵĳ�ʼ���ƶ�����
    version = 13 I ʹ�������� v5, Ȩ�� v5a ȥ��������ƥ�������Ĳ�������  ����������
    version = 13 J ʹ�������� v5, Ȩ�� v5a ʹ�ð�����ƥ�������Ĳ�������  ��������
    
    version = 14 LR���Ի���ģ ��������ѡ�� �ɰ汾����
    
    version = 30 LR���Ի���ģ ������ѡ�� ������ 0
    version = 31 LR���Ի���ģ ������ѡ�� ������DRG���� 0
    """
    version = "31"
    # ================== save file name ====================
    save_path = f"./result/S04/{hos_id}/"
    create_path_if_not_exists(save_path)

    program_name = f"S04_LR_id{hos_id}_tra{is_transfer}_v{version}"
    save_result_file = f"./result/S04_id{hos_id}_LR_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S04_LR_test{test_valid_id}_tra{is_transfer}_iter{local_lr_iter}_select{select}_v{version}.csv")
    # =====================================================
    # ��ȡ����
    # train_data_x, test_data_x, train_data_y, test_data_y = get_fs_each_hos_data_X_y(hos_id)

    match_data_len = int(select_ratio * train_data_x.shape[0])

    # ���� ��ƥ�伯���д��ڵ�Ŀ�껼��
    # test_data_x, test_data_y = process_test_data_match(train_data_y, test_data_x, test_data_y)
    # ������ ��ƥ�伯���д��ڵ�Ŀ�껼��
    # test_data_x, test_data_y = process_test_data_no_match(train_data_y, test_data_x, test_data_y)

    # # ��Ϊƥ����������
    # if is_match_other:
    #     train_data_x, _, train_data_y, _ = get_fs_each_hos_data_X_y(other_hos_id)
    #     logger.warning(
    #         "ƥ������ - �ֲ�ѵ�����޸�Ϊ��������{}ѵ������...train_data_shape:{}".format(other_hos_id, train_data_x.shape))

    # �Ƿ��������ƥ��
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
        transfer_test_data_X = test_data_x * global_feature_weight

    # ===================================== ����ʼ ==========================================
    # ���߳��õ���ȫ����, �쳣��Ϣ����
    global_lock, exec_queue = Lock(), queue.Queue()
    # ��ʼ�ܳ���
    multi_thread_personal_modeling()

    print_result_info()
