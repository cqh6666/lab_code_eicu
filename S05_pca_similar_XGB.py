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
        exec_queue.put("Termination")
        my_logger.exception(err)
        raise Exception(err)

    return patient_ids, sample_ki


def xgb_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki):
    """
    xgb训练模型，用原始数据建模
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
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - 目标原始样本
    pca_pre_data_select: pca降维后的目标样本
    :return: 最终的相似样本
    """
    # 如果消息队列中消息不为空，说明已经有任务异常了
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
    传入训练集和测试集，PCA降维前先得先乘以相似性度量, n_comp为(0,1]
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= 1:
        my_logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    my_logger.warning(f"starting pca by train_data...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: [{pca_model.n_components},{pca_test_x.shape[1]}], svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


def print_result_info():
    """
    输出结果相关信息
    :return:
    """
    if test_result.isna().sum().sum() > 0:
        print("exist NaN...")
        sys.exit(1)
    test_result.to_csv(test_result_file_name)
    # 计算auc性能
    score = roc_auc_score(test_result['real'], test_result['prob'])
    my_logger.info(f"auc score: {score}")
    # save到全局结果集合里
    save_df = pd.DataFrame(columns=['start_time', 'end_time', 'run_time', 'auc_score_result'])
    start_time_date, end_time_date, run_date_time = get_run_time(program_start_time, time.time())
    save_df.loc[program_name + "_" + str(os.getpid()), :] = [start_time_date, end_time_date, run_date_time, score]
    save_to_csv_by_row(save_result_file, save_df)

    # 发送邮箱
    send_success_mail(program_name, run_start_time=program_start_time, run_end_time=time.time())
    print("end!")


def multi_thread_personal_modeling():
    """
    多线程跑程序
    :return:
    """
    my_logger.warning("starting personalized modelling...")

    mt_begin_time = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            pca_pre_data_select = pca_test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select, pca_pre_data_select)
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

    mt_end_time = time.time()

    run_time = covert_time_format(mt_end_time - mt_begin_time)
    my_logger.warning(f"done - cost_time: {run_time}...")


def get_my_data():
    """
    读取数据和处理数据
    :return:
    """
    # 获取数据
    train_data_x, test_data_x, train_data_y, test_data_y = get_fs_each_hos_data_X_y(hos_id)
    # final_idx = test_data_x.shape[0]
    # end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值
    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # 不要超过10000个样本
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))

    return train_data_x, test_data_x, train_data_y, test_data_y


def process_sensitive_feature_weight(init_similar_weight_):
    """
    将敏感特征权重设为0
    :param init_similar_weight_:
    :return:
    """
    sens_cols = get_sensitive_columns()
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=init_similar_weight_)

    psm_df[psm_df.index.isin(sens_cols)] = 0

    my_logger.warning("已将{}个敏感特征权重设置为0...".format(len(sens_cols)))

    return psm_df.to_list()


def process_qid_feature_weight(init_similar_weight_):
    """
    将qid特征权重设为0
    :param init_similar_weight_:
    :return:
    """
    sens_cols = get_qid_columns()
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=init_similar_weight_)

    psm_df[psm_df.index.isin(sens_cols)] = 0

    my_logger.warning("已将{}个准标识符特征权重设置为0...".format(len(sens_cols)))

    return psm_df.to_list()


def add_qid_laplace_noise(test_data_x_, is_qid, μ=0, b=1.0):
    """
    为qid特征增加拉普拉斯噪声
    :param is_qid:
    :param test_data_x_:
    :param μ:
    :param b:
    :return:
    """
    if is_qid:
        qid_cols = get_qid_columns()
    else:
        qid_cols = get_sensitive_columns()

    patient_ids = test_data_x_.index

    for patient_id in patient_ids:
        laplace_noise = laplace(μ, b, len(qid_cols))  # 为原始数据添加μ为0，b为1的噪声
        for index, col in enumerate(qid_cols):
            test_data_x_.loc[patient_id, col] += laplace_noise[index]

    my_logger.warning("将准标识符特征({})进行拉普拉斯噪声处理...".format(len(qid_cols)))

    return test_data_x_


def concat_most_sensitive_feature_weight(similar_weight, concat_nums=5):
    """
    将敏感度最高的几个特征进行合并
    :param similar_weight:
    :param concat_nums:
    :return:
    """
    # 选出前k个敏感度最高的特征
    sensitive_feature = get_sensitive_columns()
    sensitive_data_x = test_data_x[sensitive_feature]
    top_sens_feature = sensitive_data_x.sum(axis=0).sort_values(ascending=False).index.to_list()[:concat_nums]

    # 构建series
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=similar_weight)

    # 均值化
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

    # 获取xgb_model和初始度量  是否全局匹配
    xgb_model = get_xgb_model_pkl(hos_id) if is_transfer == 1 else None
    init_similar_weight = get_init_similar_weight(hos_id)

    """
    version = 3 直接跑
    version = 4 使用全局匹配
    version = 5 使用正确的迁移xgb_model（全局匹配） 【错误】
    version = 6 非全局匹配
    version = 7 使用正确的迁移xgb_model（全局匹配）
    version = 8 0.7 0.8 0.9 0.95 0.99
    version = 10 xgb特征选择后的新数据 0.7 0.9 0.99
    version = 11 特征选择 lr
    version = 12 特征选择 xgb
    version = 14 特征选择 xgb重要性 （做相似性度量）

    version = 16 直接xgb特征选择 xgb重要性 （做相似性度量） 
    
    version = 17 直接xgb特征选择 xgb重要性 （做相似性度量） 将敏感特征权重设为0
    version = 18 直接xgb特征选择 xgb重要性 （做相似性度量） 将准标识符特征权重设为0
    version = 19 直接xgb特征选择 xgb重要性 （做相似性度量） qid增加拉普拉斯 b=0.5
    version = 20 直接xgb特征选择 xgb重要性 （做相似性度量） sens增加拉普拉斯 b=0.5
    version = 21 直接xgb特征选择 xgb重要性 （做相似性度量） qid增加拉普拉斯 b=1
    version = 22 直接xgb特征选择 xgb重要性 （做相似性度量） sens增加拉普拉斯 b=1
    version = 23 直接xgb特征选择 xgb重要性 （做相似性度量） 将concat_nums权重均值化 5 10 15 20
    
    version = 25 新数据 直接xgb特征选择 xgb重要性 （做相似性度量） 
    """
    version = 25
    # ================== save file name ====================
    save_path = f"./result/S05/{hos_id}/"
    create_path_if_not_exists(save_path)

    # 文件名相关
    program_name = f"S05_XGB_id{hos_id}_tra{is_transfer}_comp{n_components_str}_concat{concat_nums}_v{version}"
    save_result_file = f"./result/S05_hosid{hos_id}_XGB_all_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S05_XGB_test_tra{is_transfer}_boost{num_boost_round}_comp{n_components_str}_v{version}.csv")
    # =====================================================

    # 读取数据
    train_data_x, test_data_x, train_data_y, test_data_y = get_my_data()

    # 将权重设为0， 使得匹配完全没用上
    # init_similar_weight = process_sensitive_feature_weight(init_similar_weight)

    # 将权重设为0， 使得匹配完全没用上
    # init_similar_weight = process_qid_feature_weight(init_similar_weight)

    # 拉普拉斯噪声 True qid  False sens
    # test_data_x = add_qid_laplace_noise(test_data_x, False, μ=0, b=0.5)

    # 将多个敏感特征进行合并
    # init_similar_weight = concat_most_sensitive_feature_weight(init_similar_weight, concat_nums=concat_nums)

    # PCA降维
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)

    my_logger.warning(
        f"[params] - pid:{os.getpid()}, version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}")

    # 10% 匹配患者
    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values
    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    # ===================================== 任务开始 ==========================================
    # 多线程用到的全局锁, 异常消息队列
    global_lock, exec_queue = threading.Lock(), queue.Queue()
    # 开始跑程序
    multi_thread_personal_modeling()

    print_result_info()
