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
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_EXCEPTION
import xgboost as xgb

import pandas as pd
from numpy.random import laplace
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from my_logger import MyLog
from api_utils import covert_time_format, save_to_csv_by_row, get_hos_data_X_y, get_train_test_data_X_y, \
    get_fs_train_test_data_X_y, get_fs_hos_data_X_y, create_path_if_not_exists, get_fs_each_hos_data_X_y, \
    get_sensitive_columns, get_qid_columns
from email_api import send_success_mail, get_run_time
from xgb_utils_api import get_xgb_model_pkl, get_local_xgb_para, get_init_similar_weight

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


def xgb_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki):
    """
    xgbѵ��ģ�ͣ���ԭʼ���ݽ�ģ
    :param fit_train_x:
    :param fit_train_y:
    :param pre_data_select_:
    :param sample_ki:
    :return:
    """
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y, weight=sample_ki)
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=xgb_model)
    d_test_local = xgb.DMatrix(pre_data_select_)
    predict_prob = xgb_local.predict(d_test_local)[0]
    return predict_prob


def personalized_modeling(test_id_, pre_data_select_, pca_pre_data_select_):
    """
    ���ݾ���õ� ĳ��Ŀ�����������ÿ��ѵ�������ľ���
    test_id - patient id
    pre_data_select - Ŀ��ԭʼ����
    pca_pre_data_select: pca��ά���Ŀ������
    :return: ���յ���������
    """
    # �����Ϣ��������Ϣ��Ϊ�գ�˵���Ѿ��������쳣��
    if not exec_queue.empty():
        return

    try:
        patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_)
        fit_train_x = train_data_x.loc[patient_ids]
        fit_train_y = train_data_y.loc[patient_ids]
        predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki)
        global_lock.acquire()
        test_result.loc[test_id_, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        exec_queue.put("Termination")
        my_logger.exception(err)
        raise Exception(err)


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    """
    ����ѵ�����Ͳ��Լ���PCA��άǰ�ȵ��ȳ��������Զ���, n_compΪ(0,1]
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
    # ת��df��ʽ
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: [{pca_model.n_components},{pca_test_x.shape[1]}], svd_solver:{pca_model.svd_solver}.")

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


def add_qid_laplace_noise(test_data_x_, is_qid, ��=0, b=1.0):
    """
    Ϊqid��������������˹����
    :param is_qid:
    :param test_data_x_:
    :param ��:
    :param b:
    :return:
    """
    if is_qid:
        qid_cols = get_qid_columns()
    else:
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

    pool_nums = 5
    xgb_boost_num = 50
    xgb_thread_num = 1

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])
    n_components = float(sys.argv[3])

    n_components_str = str(n_components * 100)

    m_sample_weight = 0.01
    start_idx = 0
    select = 10
    select_ratio = select * 0.01
    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    concat_nums = 20

    # ��ȡxgb_model�ͳ�ʼ����  �Ƿ�ȫ��ƥ��
    xgb_model = get_xgb_model_pkl(hos_id) if is_transfer == 1 else None
    init_similar_weight = get_init_similar_weight(hos_id)

    """
    version = 3 ֱ����
    version = 4 ʹ��ȫ��ƥ��
    version = 5 ʹ����ȷ��Ǩ��xgb_model��ȫ��ƥ�䣩 ������
    version = 6 ��ȫ��ƥ��
    version = 7 ʹ����ȷ��Ǩ��xgb_model��ȫ��ƥ�䣩
    version = 8 0.7 0.8 0.9 0.95 0.99
    version = 10 xgb����ѡ���������� 0.7 0.9 0.99
    version = 11 ����ѡ�� lr
    version = 12 ����ѡ�� xgb
    version = 14 ����ѡ�� xgb��Ҫ�� ���������Զ�����

    version = 16 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� 
    
    version = 17 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ����������Ȩ����Ϊ0
    version = 18 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ��׼��ʶ������Ȩ����Ϊ0
    version = 19 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� qid����������˹ b=0.5
    version = 20 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� sens����������˹ b=0.5
    version = 21 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� qid����������˹ b=1
    version = 22 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� sens����������˹ b=1
    version = 23 ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� ��concat_numsȨ�ؾ�ֵ�� 5 10 15 20
    
    version = 25 ������ ֱ��xgb����ѡ�� xgb��Ҫ�� ���������Զ����� 
    """
    version = 25
    # ================== save file name ====================
    save_path = f"./result/S05/{hos_id}/"
    create_path_if_not_exists(save_path)

    # �ļ������
    program_name = f"S05_XGB_id{hos_id}_tra{is_transfer}_comp{n_components_str}_concat{concat_nums}_v{version}"
    save_result_file = f"./result/S05_hosid{hos_id}_XGB_all_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S05_XGB_test_tra{is_transfer}_boost{num_boost_round}_comp{n_components_str}_v{version}.csv")
    # =====================================================

    # ��ȡ����
    train_data_x, test_data_x, train_data_y, test_data_y = get_my_data()

    # ��Ȩ����Ϊ0�� ʹ��ƥ����ȫû����
    # init_similar_weight = process_sensitive_feature_weight(init_similar_weight)

    # ��Ȩ����Ϊ0�� ʹ��ƥ����ȫû����
    # init_similar_weight = process_qid_feature_weight(init_similar_weight)

    # ������˹���� True qid  False sens
    # test_data_x = add_qid_laplace_noise(test_data_x, False, ��=0, b=0.5)

    # ����������������кϲ�
    # init_similar_weight = concat_most_sensitive_feature_weight(init_similar_weight, concat_nums=concat_nums)

    # PCA��ά
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)

    my_logger.warning(
        f"[params] - pid:{os.getpid()}, version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}")

    # 10% ƥ�仼��
    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values
    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    # ===================================== ����ʼ ==========================================
    # ���߳��õ���ȫ����, �쳣��Ϣ����
    global_lock, exec_queue = threading.Lock(), queue.Queue()
    # ��ʼ�ܳ���
    multi_thread_personal_modeling()

    print_result_info()
