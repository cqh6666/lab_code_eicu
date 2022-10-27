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
import time
import traceback
import warnings
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import json

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from api_utils import covert_time_format, save_to_csv_by_row, get_all_data_X_y, get_hos_data_X_y, \
    get_match_all_data_from_hos_data, get_target_test_id
from lr_utils_api import get_transfer_weight, get_init_similar_weight
from email_api import send_success_mail, send_an_error_message, get_run_time
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(match_data_X, target_pre_data_select, init_psm_weight, match_len):
    """
    ѡ��ǰ10%�����������Ҹ������Ƶõ�����Ȩ��
    :param match_len:
    :param init_psm_weight:
    :param match_data_X:
    :param target_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=match_data_X.index)
        similar_rank['distance'] = abs((match_data_X - target_pre_data_select.values) * init_psm_weight).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:match_len].values
        sample_ki = similar_rank.iloc[:match_len, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        raise err

    return patient_ids, sample_ki


def get_match_data(match_all_flag):
    if match_all_flag:
        match_data_X, _ = get_match_all_data_from_hos_data(hos_id)
        # my_logger.warning("ƥ��ȫ������ - �ֲ�ѵ�����޸�Ϊȫ��ѵ������...train_data_shape:{}".format(train_data_x.shape))
        return match_data_X
    else:
        match_data_X, _, _, _ = get_hos_data_X_y(hos_id)
        return match_data_X


def get_similar_weight(psm_id):
    return get_init_similar_weight(psm_id)


def get_len_split(match_data_X, train_same_flag):
    # �Ƿ��������ƥ��
    if train_same_flag:
        len_split_ = match_data_len
    else:
        len_split_ = int(select_ratio * match_data_X.shape[0])

    return len_split_


def save_patient_ids(v_info):
    """
    ���ݲ�����Ϣ��Ŀ�껼��ƥ�仼��ID������������
    :param v_info:
    :return:
    """
    try:
        res_dict = {}
        # ��Ҫ���Ƚ�������
        # 1. ƥ������ train_data_x
        match_data_X = get_match_data(v_info['is_match_all'])
        # 2. �����Զ��� similar_weight
        similar_weight = get_similar_weight(v_info['init_psm_id'])
        # 3. ƥ������ len_split
        match_len = get_len_split(match_data_X, v_info['is_train_same'])
        # 4. �汾��Ϣ version
        cur_version = v_info['version']

        # �Ի��߱���ƥ��
        for test_id in test_data_ids_1:
            patient_ids, _ = get_similar_rank(match_data_X, test_data_1.loc[[test_id]], similar_weight, match_len)
            res_dict[test_id] = patient_ids
        my_logger.info(f"[{cur_version}]: test_data_ids_1 done...")

        # �Էǻ��߱���ƥ��
        for test_id in test_data_ids_0:
            patient_ids, _ = get_similar_rank(match_data_X, test_data_0.loc[[test_id]], similar_weight, match_len)
            res_dict[test_id] = patient_ids
        my_logger.info(f"[{cur_version}]: test_data_ids_0 done...")

        # save
        dict_file_name = os.path.join(save_path, 'target_match_ids_v{}.npy'.format(cur_version))
        np.save(dict_file_name, res_dict)
        # load  ==> load_dict = np.load('xxx').item()
        my_logger.warning("[{}] - save dict({}) success! - {}".format(cur_version, len(res_dict), dict_file_name))
    except Exception as err:
        print(err)
        sys.exit(1)


if __name__ == '__main__':

    run_start_time = time.time()

    my_logger = MyLog().logger

    pool_nums = 10

    # hos_id = 264
    hos_id = int(sys.argv[1])

    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    """
    version = 10 ���ڸ��������ƶ���ƥ������10%����  init_psm_id = hos_id, is_train_same = False, is_match_all = False
    version = 11 ���ڸ��������ƶ���ƥ��ȫ������10%  init_psm_id = hos_id, is_train_same = False, is_match_all = True
    version = 12 ����ȫ�����ƶ���ƥ�������10%����  init_psm_id = 0, is_train_same = False, is_match_all = False
    version = 13 ���ڸ��������ƶ���ƥ��ȫ��������������  init_psm_id = hos_id, is_train_same = True, is_match_all = True
    version = 14 ����ȫ�����ƶ���ƥ��ȫ������10% init_psm_id = 0, is_train_same = False, is_match_all = True
    version = 15 ����ȫ�����ƶ���ƥ��ȫ�ֵ������������ģ� init_psm_id = 0, is_train_same = True, is_match_all = True
    """
    version = "1"
    # ================== save file name ====================
    program_name = f"S04_id{hos_id}_find_pid_v{version}"
    save_path = f"./result/S04/{hos_id}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        my_logger.warning("create new dirs... {}".format(save_path))
    # =====================================================
    # ��ȡ���� ֻ����hos_id
    train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)
    match_data_len = int(select_ratio * train_data_x.shape[0])
    my_logger.info("hos_id:{}, match_len:{}".format(hos_id, match_data_len))
    # ��ȡĿ���������з���
    test_data_ids_1, test_data_ids_0 = get_target_test_id(hos_id)
    test_data_1 = test_data_x.loc[test_data_ids_1]
    test_data_0 = test_data_x.loc[test_data_ids_0]

    # ���߳�ִ�ж���汾��forѭ��ִ��100��Ŀ�껼��
    version_list = [
        {"version": 10, "init_psm_id": hos_id, "is_train_same": False, "is_match_all": False},
        {"version": 11, "init_psm_id": hos_id, "is_train_same": False, "is_match_all": True},
        {"version": 12, "init_psm_id": 0, "is_train_same": False, "is_match_all": False},
        {"version": 13, "init_psm_id": hos_id, "is_train_same": True, "is_match_all": True},
        {"version": 14, "init_psm_id": 0, "is_train_same": False, "is_match_all": True},
        {"version": 15, "init_psm_id": 0, "is_train_same": True, "is_match_all": True},
    ]

    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for version_info in version_list:
            # ִ��ƥ�仼��IDs
            thread = executor.submit(save_patient_ids, version_info)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    run_end_time = time.time()
    my_logger.info("done!")
    # ��������
    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=run_end_time)
    print("end!")
