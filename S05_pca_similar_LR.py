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
from threading import Lock
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_EXCEPTION
import pandas as pd
from numpy.random import laplace

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from api_utils import covert_time_format, save_to_csv_by_row, get_fs_train_test_data_X_y, get_fs_hos_data_X_y, \
    create_path_if_not_exists, get_fs_each_hos_data_X_y, get_sensitive_columns, get_qid_columns
from email_api import send_success_mail, get_run_time
from lr_utils_api import get_transfer_weight, get_init_similar_weight
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    ѡ��ǰ10%�����������Ҹ������Ƶõ�����Ȩ��
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
        exec_queue.put("Termination")
        my_logger.exception(err)
        raise Exception(err)

    return patient_ids, sample_ki


def lr_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki):
    if len(fit_train_y.value_counts()) <= 1:
        return train_data_y[0]
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(pre_data_select_)[0][1]
    return predict_prob


def fit_train_test_data(patient_ids, pre_data_select_x):
    if is_transfer == 1:
        fit_train_x = transfer_train_data_X.loc[patient_ids]
        fit_test_x = pre_data_select_x * global_feature_weight
    else:
        fit_train_x = train_data_x.loc[patient_ids]
        fit_test_x = pre_data_select_x

    return fit_test_x, fit_train_x


def personalized_modeling(patient_id, pre_data_select_x, pca_pre_data_select_x):
    """
    ���ݾ���õ� ĳ��Ŀ�����������ÿ��ѵ�������ľ���
    :param pre_data_select_x: ԭʼ��������
    :param pca_pre_data_select_x:  �����Ĳ�������
    :param patient_id:
    :return:
    """
    # �����Ϣ��������Ϣ��Ϊ�գ�˵���Ѿ��������쳣��
    if not exec_queue.empty():
        return

    try:
        patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_x)
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


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    """
    pca��ά, n_comp����Ϊ�ٷֱ�
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= 1:
        my_logger.warning("n_comp ���� 1 �ˣ�����������������Ϊ 1...")
        sys.exit(1)

    my_logger.warning(f"starting pca by train_data...")
    # pca��ά
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # ת����Ҫ * �����Զ���
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # new_train_data_x = pca_model.fit_transform(train_x)
    # new_test_data_x = pca_model.transform(test_x)
    # ת��df��ʽ
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}], svd_solver:{pca_model.svd_solver}")

    return pca_train_x, pca_test_x


def print_result_info():
    """
    �����������Ϣ
    :return:
    """
    if test_result.isna().sum().sum() > 0:
        print("exist NaN...")
        sys.exit(1)
    test_result.to_csv(test_result_file_name)
    # ����auc����
    score = roc_auc_score(test_result['real'], test_result['prob'])
    my_logger.info(f"auc score: {score}")
    # save��ȫ�ֽ��������
    save_df = pd.DataFrame(columns=['start_time', 'end_time', 'run_time', 'auc_score_result'])
    start_time_date, end_time_date, run_date_time = get_run_time(program_start_time, time.time())
    save_df.loc[program_name + "_" + str(os.getpid()), :] = [start_time_date, end_time_date, run_date_time, score]
    save_to_csv_by_row(save_result_file, save_df)

    # ��������
    send_success_mail(program_name, run_start_time=program_start_time, run_end_time=time.time())
    print("end!")


def multi_thread_personal_modeling():
    """
    ���߳��ܳ���
    :return:
    """
    my_logger.warning("starting personalized modelling...")

    mt_begin_time = time.time()
    # ƥ��������������ѵ������ XGB��ģ ���߳�
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            pca_pre_data_select = pca_test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select, pca_pre_data_select)
            thread_list.append(thread)

        # �����ֵ�һ������, �������ȡ��������ֹ
        wait(thread_list, return_when=FIRST_EXCEPTION)
        for cur_thread in reversed(thread_list):
            cur_thread.cancel()

        wait(thread_list, return_when=ALL_COMPLETED)

    # �������쳣ֱ�ӷ���
    if not exec_queue.empty():
        my_logger.error("something task error... we have to stop!!!")
        return

    mt_end_time = time.time()

    run_time = covert_time_format(mt_end_time - mt_begin_time)
    my_logger.warning(f"done - cost_time: {run_time}...")


def get_my_data():
    """
    ��ȡ���ݺʹ�������
    :return:
    """
    # ��ȡ����
    train_data_x, test_data_x, train_data_y, test_data_y = get_fs_each_hos_data_X_y(hos_id)
    # final_idx = test_data_x.shape[0]
    # end_idx = final_idx if end_idx > final_idx else end_idx  # ���ô�����ֵ
    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # ��Ҫ����10000������
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))

    return train_data_x, test_data_x, train_data_y, test_data_y


def process_sensitive_feature_weight(init_similar_weight_):
    """
    ����������Ȩ����Ϊ0
    :param init_similar_weight_:
    :return:
    """
    sens_cols = get_sensitive_columns()
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=init_similar_weight_)

    psm_df[psm_df.index.isin(sens_cols)] = 0

    my_logger.warning("�ѽ�{}����������Ȩ������Ϊ0...".format(len(sens_cols)))

    return psm_df.to_list()


def process_qid_feature_weight(init_similar_weight_):
    """
    ��qid����Ȩ����Ϊ0
    :param init_similar_weight_:
    :return:
    """
    sens_cols = get_qid_columns()
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=init_similar_weight_)

    psm_df[psm_df.index.isin(sens_cols)] = 0

    my_logger.warning("�ѽ�{}��׼��ʶ������Ȩ������Ϊ0...".format(len(sens_cols)))

    return psm_df.to_list()


def add_laplace_noise(test_data_x_, ��=0, b=1.0):
    """
    Ϊqid��������������˹����
    :param test_data_x_:
    :param ��:
    :param b:
    :return:
    """
    # qid_cols = get_qid_columns()
    qid_cols = get_sensitive_columns()
    patient_ids = test_data_x_.index

    for patient_id in patient_ids:
        laplace_noise = laplace(��, b, len(qid_cols))  # Ϊԭʼ������Ӧ�Ϊ0��bΪ1������
        for index, col in enumerate(qid_cols):
            test_data_x_.loc[patient_id, col] += laplace_noise[index]

    my_logger.warning("��׼��ʶ������({})����������˹��������...".format(len(qid_cols)))

    return test_data_x_


def concat_most_sensitive_feature_weight(similar_weight, concat_nums=5):
    """
    �����ж���ߵļ����������кϲ�
    :param similar_weight:
    :param concat_nums:
    :return:
    """
    # ѡ��ǰk�����ж���ߵ�����
    sensitive_feature = get_sensitive_columns()
    sensitive_data_x = test_data_x[sensitive_feature]
    top_sens_feature = sensitive_data_x.sum(axis=0).sort_values(ascending=False).index.to_list()[:concat_nums]

    # ����series
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=similar_weight)

    # ��ֵ��
    mean_weight = psm_df[psm_df.index.isin(top_sens_feature)].mean()
    psm_df[psm_df.index.isin(top_sens_feature)] = mean_weight

    return psm_df.to_list()


if __name__ == '__main__':

    program_start_time = time.time()
    my_logger = MyLog().logger

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])
    n_components = float(sys.argv[3])

    n_components_str = str(int(n_components * 100))
    pool_nums = 8
    start_idx = 0
    local_lr_iter = 100
    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    concat_nums = 20

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    global_feature_weight = get_transfer_weight(hos_id)
    init_similar_weight = get_init_similar_weight(hos_id)
    """
    version = 1
    version = 2 ����������
    version = 3 ��������ʽ
    version = 4 ʹ��ȫ��ƥ��
    version = 5 500 1000 ��ά
    version = 6 �°� 100 500 1000
    version = 7 0.7 0.8 0.9 0.95 0.99
    version = 8 0.7 0.8 0.9 0.95 0.99 ���������Զ���
    version = 9 ����ѡ�� lr��Ҫ��
    version = 10 ����ѡ�� xgb��Ҫ��
    version = 11 ����ѡ�� lr��Ҫ��
    version = 12 ����ѡ�� xgb��Ҫ�� ( ���������Զ�����
    version = 13 ����ѡ�� lr��Ҫ�� ���������Զ�����
    version = 14 ����ѡ�� xgb��Ҫ�� ���������Զ�����
    version = 16 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� 
    version = 17 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ����������Ȩ����Ϊ0
    version = 18 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ��׼��ʶ������Ȩ����Ϊ0
    version = 19 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� qid����������˹
    version = 20 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� sens����������˹
    version = 21 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� qid����������˹ b=0.5
    version = 22 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� sens����������˹ b=0.5
    version = 23 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ��concat_numsȨ�ؾ�ֵ�� 5 10 15 20
    
    version = 24 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ʹ��ȫ�������Զ���
    version = 25 ������ ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� 
    version = 26 ������ ֱ��xgb����ѡ�� xgb��Ҫ�� �����������Զ����� 
    """
    version = 25
    # ================== save file name ====================
    # �����ھʹ���
    save_path = f"./result/S05/{hos_id}/"
    create_path_if_not_exists(save_path)

    # �ļ������
    program_name = f"S05_LR_id{hos_id}_tra{is_transfer}_comp{n_components_str}_concat{concat_nums}_v{version}"
    save_result_file = f"./result/S05_hosid{hos_id}_LR_all_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S05_LR_test_tra{is_transfer}_boost{local_lr_iter}_comp{n_components_str}_v{version}.csv")
    # =====================================================

    # �������ز���չʾ
    my_logger.warning(
        f"[params] - pid:{os.getpid()}, model_select:LR, pool_nums:{pool_nums}, is_transfer:{is_transfer}, "
        f"max_iter:{local_lr_iter}, select:{select}, version:{version}")

    # ��ȡ����
    train_data_x, test_data_x, train_data_y, test_data_y = get_my_data()
    len_split = int(select_ratio * train_data_x.shape[0])

    # ��ǰ����Ǩ�ƺ��ѵ�����Ͳ��Լ�
    if is_transfer == 1:
        transfer_train_data_X = train_data_x * global_feature_weight
        transfer_test_data_X = test_data_x * global_feature_weight

    # ������������Ȩ����Ϊ0��ʹ��ƥ����ȫû����
    # init_similar_weight = process_sensitive_feature_weight(init_similar_weight)

    # ��׼��ʶ����Ȩ����Ϊ0�� ʹ��ƥ����ȫû����
    # init_similar_weight = process_qid_feature_weight(init_similar_weight)

    # ������˹����
    # test_data_x = add_laplace_noise(test_data_x, ��=0, b=0.5)

    # ����������������кϲ�
    # init_similar_weight = concat_most_sensitive_feature_weight(init_similar_weight, concat_nums=concat_nums)

    # PCA��ά
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)

    # ��ʼ�����Ի���ģ��Ҫ��df
    test_id_list = pca_test_data_x.index.values
    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    # ===================================== ����ʼ ==========================================
    # ���߳��õ���ȫ����, �쳣��Ϣ����
    global_lock, exec_queue = Lock(), queue.Queue()
    # ��ʼ�ܳ���
    multi_thread_personal_modeling()

    print_result_info()



