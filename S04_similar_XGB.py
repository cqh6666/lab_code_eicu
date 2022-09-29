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

from api_utils import covert_time_format, save_to_csv_by_row, get_all_data_X_y, get_hos_data_X_y
from email_api import send_success_mail, send_an_error_message
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
        global is_send
        if not is_send:
            send_an_error_message(program_name=program_name, error_name=repr(err), error_detail=traceback.format_exc())
            is_send = True
        sys.exit(1)


if __name__ == '__main__':

    run_start_time = time.time()

    my_logger = MyLog().logger

    pool_nums = 2

    hos_id = int(sys.argv[1])
    is_transfer = int(sys.argv[2])  # 0 1
    start_idx = int(sys.argv[3])
    end_idx = int(sys.argv[4])

    m_sample_weight = 0.01
    xgb_boost_num = 50
    select = 10
    # 分成5批，每一批1w个测试样本
    init_similar_weight = get_xgb_init_similar_weight(hos_id)

    xgb_thread_num = 1
    select_ratio = select * 0.01
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)

    if is_transfer == 1:
        xgb_model = get_xgb_model_pkl(hos_id)
    else:
        xgb_model = None
    """
    version=1
    version = 4 中位数填充
    version = 5 类权重
    """
    version = 5
    # ================== save file name ====================
    program_name = f"S04_XGB_{hos_id}_{is_transfer}_{start_idx}_{end_idx}"
    is_send = False
    save_path = f"./result/S04/{hos_id}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        my_logger.warning("create new dirs... {}".format(save_path))
    test_result_file_name = os.path.join(
        save_path, f"S04_XGB_test_tra{is_transfer}_boost{xgb_boost_num}_select{select}_v{version}.csv")
    # =====================================================

    # 获取数据
    if hos_id == 0:
        train_data_x, test_data_x, train_data_y, test_data_y = get_all_data_X_y()
    else:
        train_data_x, test_data_x, train_data_y, test_data_y = get_hos_data_X_y(hos_id)

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值

    # 分批次进行个性化建模
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))
    my_logger.warning(
        f"[params] - version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, "
        f"index_range:[{start_idx}, {end_idx}]")

    # 10%匹配患者
    len_split = int(select_ratio * train_data_x.shape[0])
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

    e_t = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(e_t - s_t)}...")

    # save concat test_result csv
    if save_to_csv_by_row(test_result_file_name, test_result):
        my_logger.info("save test result prob success!")
    else:
        my_logger.info("save error...")

    send_success_mail(program_name, run_start_time=run_start_time, run_end_time=time.time())
    print("end!")