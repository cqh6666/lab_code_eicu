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
from my_logger import logger
from MyPCA import MyPCA
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
        logger.exception(err)
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
    根据距离得到 某个目标测试样本对每个训练样本的距离
    :param pre_data_select_x: 原始测试样本
    :param pca_pre_data_select_x:  处理后的测试样本
    :param patient_id:
    :return:
    """
    # 如果消息队列中消息不为空，说明已经有任务异常了
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
        logger.exception(err)
        raise Exception(err)


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= 1:
        logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    logger.warning(f"starting pca by train_data...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # new_train_data_x = pca_model.fit_transform(train_x)
    # new_test_data_x = pca_model.transform(test_x)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}], svd_solver:{pca_model.svd_solver}")

    return pca_train_x, pca_test_x


def pca_reduction_no_similar(train_x, test_x, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= 1:
        logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    logger.warning(f"starting pca by train_data without similar weight...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_test_data_x = pca_model.fit_transform(test_x)
    new_train_data_x = pca_model.transform(train_x)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}], svd_solver:{pca_model.svd_solver}")

    return pca_train_x, pca_test_x


def pca_reduction_no_similar_all_data(train_x, test_x, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= 1:
        logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    logger.warning(f"starting pca by train_data without similar weight...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    all_data_x = pd.concat([test_x, train_x], axis=0)
    pca_model.fit(all_data_x)
    # 转换需要 * 相似性度量
    new_test_data_x = pca_model.transform(test_x)
    new_train_data_x = pca_model.transform(train_x)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}]")

    return pca_train_x, pca_test_x


def pca_reduction_all_data(train_x, test_x, similar_weight, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    if n_comp >= 1:
        logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    logger.warning(f"starting pca by train_data without similar weight...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    all_data_x = pd.concat([test_x, train_x], axis=0)
    pca_model.fit(all_data_x)
    # 转换需要 * 相似性度量
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    new_train_data_x = pca_model.transform(train_x * similar_weight)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}]")

    return pca_train_x, pca_test_x


def my_pca_reduction_concat_cor_all_data(train_x, test_x, similar_weight, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    logger.warning(f"starting pca by train_data without similar weight...")

    # from hos
    tempPCA = MyPCA(n_components=n_comp)
    test_X_cor = tempPCA.get_X_cov(test_x * similar_weight)
    train_X_cor = tempPCA.get_X_cov(train_x * similar_weight)

    concat_X_cor = (test_X_cor + train_X_cor) / 2

    # pca降维
    pca_model = MyPCA(n_components=n_comp)
    pca_model.set_X_cov(concat_X_cor)
    pca_model.fit_cov()
    # 转换需要 * 相似性度量
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    new_train_data_x = pca_model.transform(train_x * similar_weight)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}]")

    return pca_train_x, pca_test_x


def my_pca_reduction_concat_cor_no_similar_all_data(train_x, test_x, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    logger.warning(f"starting pca by train_data without similar weight...")

    tempPCA = MyPCA(n_components=n_comp)
    test_X_cor = tempPCA.get_X_cov(test_x)
    train_X_cor = tempPCA.get_X_cov(train_x)
    concat_X_cor = test_X_cor + train_X_cor

    # pca降维
    pca_model = MyPCA(n_components=n_comp)
    pca_model.set_X_cov(concat_X_cor)
    pca_model.fit_cov()
    # 转换需要
    new_test_data_x = pca_model.transform(test_x)
    new_train_data_x = pca_model.transform(train_x)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}]")

    return pca_train_x, pca_test_x
def my_pca_reduction_no_similar_all_data(train_x, test_x, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    logger.warning(f"starting pca by train_data without similar weight...")
    # pca降维
    pca_model = MyPCA(n_components=n_comp)
    all_data_x = pd.concat([test_x, train_x], axis=0)
    pca_model.fit(all_data_x)
    # 转换需要 * 相似性度量
    new_test_data_x = pca_model.transform(test_x)
    new_train_data_x = pca_model.transform(train_x)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}]")

    return pca_train_x, pca_test_x

def my_pca_reduction_all_data(train_x, test_x, similar_weight, n_comp):
    """
    pca降维, n_comp更改为百分比
    :param train_x:
    :param test_x:
    :param similar_weight:
    :param n_comp:
    :return:
    """
    logger.warning(f"starting pca by train_data without similar weight...")
    # pca降维
    pca_model = MyPCA(n_components=n_comp)
    all_data_x = pd.concat([test_x, train_x], axis=0)
    pca_model.fit(all_data_x)
    # 转换需要 * 相似性度量
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    new_train_data_x = pca_model.transform(train_x * similar_weight)
    # 转成df格式
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)

    logger.info(f"n_components: [{pca_model.n_components}, {pca_test_x.shape[1]}]")

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
    logger.info(f"auc score: {score}")
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
    logger.warning("starting personalized modelling...")

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
        logger.error("something task error... we have to stop!!!")
        return

    mt_end_time = time.time()

    run_time = covert_time_format(mt_end_time - mt_begin_time)
    logger.warning(f"done - cost_time: {run_time}...")


def get_my_data(match_same=False):
    """
    读取数据和处理数据
    :return:
    """
    # 获取数据
    t_x, test_data_x, _, test_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    train_data_x, _, train_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)

    # t_x 为 原本匹配样本的训练集
    if match_same:
        match_len = int(select_ratio * t_x.shape[0])
    else:
        match_len = train_data_x.shape[0]

    # final_idx = test_data_x.shape[0]
    # end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值
    final_idx = test_data_x.shape[0]
    end_idx = final_idx if final_idx < 10000 else 10000  # 不要超过10000个样本
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))

    return train_data_x, test_data_x, train_data_y, test_data_y, match_len


def process_sensitive_feature_weight(init_similar_weight_, sens_coef=0.5):
    """
    将敏感特征权重设为0
    :param sens_coef:
    :param init_similar_weight_:
    :return:
    """
    sens_cols = get_sensitive_columns()
    columns_name = train_data_x.columns.to_list()
    psm_df = pd.Series(index=columns_name, data=init_similar_weight_)

    psm_df[psm_df.index.isin(sens_cols)] = psm_df[psm_df.index.isin(sens_cols)] * sens_coef

    logger.warning("已将{}个敏感特征权重设置为0...".format(len(sens_cols)))

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

    logger.warning("已将{}个准标识符特征权重设置为0...".format(len(sens_cols)))

    return psm_df.to_list()


def add_laplace_noise(test_data_x_, μ=0, b=1.0):
    """
    为qid特征增加拉普拉斯噪声
    :param test_data_x_:
    :param μ:
    :param b:
    :return:
    """
    # qid_cols = get_qid_columns()
    qid_cols = get_sensitive_columns()
    patient_ids = test_data_x_.index

    for patient_id in patient_ids:
        laplace_noise = laplace(μ, b, len(qid_cols))  # 为原始数据添加μ为0，b为1的噪声
        for index, col in enumerate(qid_cols):
            test_data_x_.loc[patient_id, col] += laplace_noise[index]

    logger.warning("将准标识符特征({})进行拉普拉斯噪声处理...".format(len(qid_cols)))

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

    from_hos_id = int(sys.argv[1])
    to_hos_id = int(sys.argv[2])
    is_transfer = int(sys.argv[3])
    n_components = float(sys.argv[4])

    n_components_str = str(n_components * 100)
    pool_nums = 4
    start_idx = 0
    local_lr_iter = 100
    select = 10
    select_ratio = select * 0.01
    m_sample_weight = 0.01

    concat_nums = 25

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    init_similar_weight = get_init_similar_weight(from_hos_id)
    global_feature_weight = get_transfer_weight(to_hos_id)
    """
    version = 2 不使用相似性度量
    version = 2-B 不使用相似性度量, 但匹配原中心百分之10%
    version = 2-C 不使用相似性度量, 但匹配原中心百分之10% 并且 初始相似性权重设为当前中心
    version = 1 使用相似性度量
    version = 3 使用相似性度量 敏感特征*0
    version = 4 使用相似性度量 敏感特征*0.5
    version = 5 使用相似性度量 敏感特征增加噪声
    
    version = 6 特征选择 使用单中心相似性度量 PCA降维 
    version = 7 特征选择 不使用单中心相似性度量 PCA降维
    version = 8 特征选择 使用匹配中心相似性度量 PCA降维
    
    ----- 使用测试集转化矩阵
    version = 9 特征选择 使用单中心相似性度量 PCA降维
    version = 10 特征选择 不使用单中心相似性度量 PCA降维
    version = 11 特征选择 使用相似性度量 使用所有数据进行PCA降维
    version = 14 特征选择 不使用相似性度量 使用所有数据进行PCA降维
    
    ------- 自己的MyPCA
    version = 12 特征选择 使用单中心相似性度量 使用所有数据进行PCA降维
    version = 13 特征选择 不使用单中心相似性度量 使用所有数据进行PCA降维
    
    version = 15 特征选择 相似性度量 合并协方差矩阵 使用所有数据进行PCA降维
    version = 16 特征选择 不使用相似性度量 合并协方差矩阵 使用所有数据进行PCA降维
    """
    version = "15"
    # ================== save file name ====================
    # 不存在就创建
    save_path = f"./result/S05/{from_hos_id}/"
    create_path_if_not_exists(save_path)

    # 文件名相关
    program_name = f"S05_LR_from{from_hos_id}_to{to_hos_id}_tra{is_transfer}_comp{n_components_str}_select{select}_v{version}"
    save_result_file = f"./result/S05_from{from_hos_id}_to{to_hos_id}_LR_all_result_save.csv"
    test_result_file_name = os.path.join(
        save_path, f"S05_LR_test_from{from_hos_id}_to{to_hos_id}_tra{is_transfer}_comp{n_components_str}_select{select}_v{version}.csv")
    # =====================================================

    # 输入的相关参数展示
    logger.warning(
        f"[params] - pid:{os.getpid()}, model_select:LR, pool_nums:{pool_nums}, is_transfer:{is_transfer}, "
        f"max_iter:{local_lr_iter}, select:{select}, version:{version}")

    # 读取数据
    train_data_x, test_data_x, train_data_y, test_data_y, len_split = get_my_data(match_same=False)

    # 提前计算迁移后的训练集和测试集
    if is_transfer == 1:
        transfer_train_data_X = train_data_x * global_feature_weight

    # 将敏感特征的权重设为0，使得匹配完全没用上
    # init_similar_weight = process_sensitive_feature_weight(init_similar_weight, sens_coef=0.5)

    # 将准标识符的权重设为0， 使得匹配完全没用上
    # init_similar_weight = process_qid_feature_weight(init_similar_weight)

    # 拉普拉斯噪声
    # test_data_x = add_laplace_noise(test_data_x, μ=0, b=0.5)

    # 将多个敏感特征进行合并
    # init_similar_weight = concat_most_sensitive_feature_weight(init_similar_weight, concat_nums=concat_nums)

    # PCA降维
    # pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)
    # pca_train_data_x, pca_test_data_x = pca_reduction_no_similar(train_data_x, test_data_x, n_components)
    # pca_train_data_x, pca_test_data_x = pca_reduction_no_similar_all_data(train_data_x, test_data_x, n_components)
    # pca_train_data_x, pca_test_data_x = pca_reduction_all_data(train_data_x, test_data_x, init_similar_weight, n_components)
    # pca_train_data_x, pca_test_data_x = my_pca_reduction_all_data(train_data_x, test_data_x, init_similar_weight, n_components)
    # pca_train_data_x, pca_test_data_x = my_pca_reduction_no_similar_all_data(train_data_x, test_data_x, n_components)
    pca_train_data_x, pca_test_data_x = my_pca_reduction_concat_cor_all_data(train_data_x, test_data_x, init_similar_weight, n_components)
    # pca_train_data_x, pca_test_data_x = my_pca_reduction_concat_cor_no_similar_all_data(train_data_x, test_data_x, n_components)

    # 初始化个性化建模需要的df
    test_id_list = pca_test_data_x.index.values
    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    # ===================================== 任务开始 ==========================================
    # 多线程用到的全局锁, 异常消息队列
    global_lock, exec_queue = Lock(), queue.Queue()
    # 开始跑程序
    multi_thread_personal_modeling()

    print_result_info()



