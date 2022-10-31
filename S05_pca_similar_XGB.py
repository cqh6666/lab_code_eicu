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
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import xgboost as xgb

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from my_logger import MyLog
from api_utils import covert_time_format, save_to_csv_by_row, get_all_data_X_y, get_hos_data_X_y
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
        print(err)
        sys.exit(1)

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
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_)
    try:
        fit_train_x = train_data_x.loc[patient_ids]
        fit_train_y = train_data_y.loc[patient_ids]
        predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki)
        global_lock.acquire()
        test_result.loc[test_id_, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(err)
        sys.exit(1)


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    """
    ����ѵ�����Ͳ��Լ���PCA��άǰ�ȵ��ȳ��������Զ���
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= train_x.shape[1]:
        n_comp = train_x.shape[1] - 1

    my_logger.warning(f"starting pca by train_data...")
    # pca��ά
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # ת����Ҫ * �����Զ���
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # ת��df��ʽ
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


def print_result_info():
    # ���ܴ���NaN
    if test_result.isna().sum().sum() > 0:
        print("exist NaN...")
        sys.exit(1)
    test_result.to_csv(test_result_file_name)
    # ����auc����
    score = roc_auc_score(test_result['real'], test_result['prob'])
    my_logger.info(f"auc score: {score}")
    # save��ȫ�ֽ��������
    save_df = pd.DataFrame(columns=['start_time', 'end_time', 'run_time', 'auc_score_result'])
    start_time_date, end_time_date, run_date_time = get_run_time(run_start_time, run_end_time)
    save_df.loc[program_name + "_" + str(int(run_start_time)), :] = [start_time_date, end_time_date, run_date_time,
                                                                     score]
    save_to_csv_by_row(save_result_file, save_df)

    # ��������
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=run_end_time)
    print("end!")


if __name__ == '__main__':

    run_start_time = time.time()
    my_logger = MyLog().logger

    pool_nums = 4
    xgb_boost_num = 50
    xgb_thread_num = 1

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])
    n_components = int(sys.argv[3])
    # hos_id = 167
    # is_transfer = 1
    # n_components = 1000

    m_sample_weight = 0.01
    start_idx = 0
    select = 10
    select_ratio = select * 0.01
    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    # ��ȡxgb_model�ͳ�ʼ����  �Ƿ�ȫ��ƥ��
    xgb_model = get_xgb_model_pkl(hos_id) if is_transfer == 1 else None
    init_similar_weight = get_init_similar_weight(hos_id)

    """
    version = 3 ֱ����
    version = 4 ʹ��ȫ��ƥ��
    version = 5 ʹ����ȷ��Ǩ��xgb_model��ȫ��ƥ�䣩 ������
    version = 6 ��ȫ��ƥ��
    version = 7 ʹ����ȷ��Ǩ��xgb_model��ȫ��ƥ�䣩
    """
    version = 6
    # ================== save file name ====================
    program_name = f"S05_XGB_id{hos_id}_tra{is_transfer}_comp{n_components}_v{version}"
    save_result_file = f"./result/S05_hosid{hos_id}_XGB_all_result_save.csv"
    save_path = f"./result/S05/{hos_id}/"
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
            my_logger.warning("create new dirs... {}".format(save_path))
        except Exception:
            pass

    test_result_file_name = os.path.join(
        save_path, f"S05_XGB_test_tra{is_transfer}_boost{num_boost_round}_comp{n_components}_v{version}.csv")
    # =====================================================

    # ��ȡ����
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_all_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)

    # ��Ϊƥ��ȫ�֣��޸�Ϊȫ������
    # train_data_x, _, train_data_y, _ = get_all_data_X_y()
    # my_logger.warning("ƥ��ȫ������ - �ֲ�ѵ�����޸�Ϊȫ��ѵ������...")

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # ��Ҫ����10000������

    # �����ν��и��Ի���ģ
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    # PCA��ά
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)

    my_logger.warning(
        f"[params] - version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, "
        f"test_idx:[{start_idx}, {end_idx}]")

    # 10%ƥ�仼��
    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = threading.Lock()
    my_logger.warning("starting personalized modelling...")
    s_t = time.time()
    # ƥ��������������ѵ������ XGB��ģ ���߳�
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
