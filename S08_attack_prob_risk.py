# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S09_attack_prob_risk
   Description:   攻击者通过推导攻击，计算损失
   Author:        cqh
   date:          2022/11/30 15:53
-------------------------------------------------
   Change Activity:
                  2022/11/30:
-------------------------------------------------
"""
__author__ = 'cqh'

import os.path
import sys
import threading

import numpy as np
from numpy.random import laplace
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from api_utils import get_fs_each_hos_data_X_y, get_sensitive_columns, create_path_if_not_exists, get_target_test_id
from lr_utils_api import get_init_similar_weight
from my_logger import logger
import pandas as pd


def process_sensitive_feature_weight(init_similar_weight_, columns_list, sens_coef=0.5):
    """
    将敏感特征权重设为0
    :param columns_list:
    :param sens_coef:
    :param init_similar_weight_:
    :return:
    """
    sens_cols = get_sensitive_columns()
    columns_name = columns_list
    psm_df = pd.Series(index=columns_name, data=init_similar_weight_)

    psm_df[psm_df.index.isin(sens_cols)] = psm_df[psm_df.index.isin(sens_cols)] * sens_coef

    logger.warning("已将{}个敏感特征权重设置为{}...".format(len(sens_cols), sens_coef))

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

    logger.warning("将敏感特征({})进行拉普拉斯噪声处理...".format(len(qid_cols)))

    return test_data_x_


def pca_reduction(match_data_x, target_data_x, n_comp):
    """
    直接进行pca降维
    :param match_data_x: 匹配样本
    :param target_data_x: 目标样本
    :param similar_weight: 相似性度量
    :param n_comp: 百分比
    :return:
    """
    if n_comp >= 1:
        logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    logger.warning(f"开始对目标患者和匹配样本进行PCA降维...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_match_data_x = pca_model.fit_transform(match_data_x)
    new_target_data_x = pca_model.transform(target_data_x)

    # 转成df格式
    pca_match_data_x = pd.DataFrame(data=new_match_data_x, index=match_data_x.index)
    pca_target_data_x = pd.DataFrame(data=new_target_data_x, index=target_data_x.index)

    logger.info(
        f"方差占比阈值: {pca_model.n_components}, 降维维度: {pca_model.n_components_}")

    # =============================================================================

    logger.warning("开始恢复数据...")
    rec_tar_x = pca_model.inverse_transform(pca_target_data_x)
    rec_tar_x_df = pd.DataFrame(data=rec_tar_x, index=target_data_x.index, columns=target_data_x.columns)

    logger.info("成功恢复目标患者数据:  target_data:{}".format(rec_tar_x_df.shape))

    return pca_match_data_x, pca_target_data_x, rec_tar_x_df


def pca_reduction_with_similar_weight(match_data_x, target_data_x, similar_weight, n_comp):
    """
    利用相似性度量进行pca降维
    :param match_data_x: 匹配样本
    :param target_data_x: 目标样本
    :param similar_weight: 相似性度量
    :param n_comp: 百分比
    :return:
    """
    if n_comp >= 1:
        logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    logger.warning(f"开始对目标患者和匹配样本进行PCA降维...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_match_data_x = pca_model.fit_transform(match_data_x * similar_weight)
    new_target_data_x = pca_model.transform(target_data_x * similar_weight)

    # 转成df格式
    pca_match_data_x = pd.DataFrame(data=new_match_data_x, index=match_data_x.index)
    pca_target_data_x = pd.DataFrame(data=new_target_data_x, index=target_data_x.index)

    logger.info(
        f"方差占比阈值: {pca_model.n_components}, 降维维度: {pca_model.n_components_}")

    # =============================================================================

    logger.warning(f"开始恢复数据...")
    # rec_match_x = pca_model.inverse_transform(pca_match_data_x)
    rec_tar_x = pca_model.inverse_transform(pca_target_data_x)

    # rec_match_x_df = pd.DataFrame(data=rec_match_x, index=match_data_x.index, columns=match_data_x.columns)
    rec_tar_x_df = pd.DataFrame(data=rec_tar_x, index=target_data_x.index, columns=target_data_x.columns)

    new_weight = []
    for weight in similar_weight:
        if weight == 0:
            new_weight.append(1.0)
        else:
            new_weight.append(weight)
    # rec_match_x_df = rec_match_x_df / new_weight
    rec_tar_x_df = rec_tar_x_df / new_weight

    logger.info("成功对目标患者进行恢复数据:  target_data: {}".
                   format(rec_tar_x_df.shape))

    return pca_match_data_x, pca_target_data_x, rec_tar_x_df


def do_cal_loss_info(target_sens_true, target_sens_predict):
    """
    计算损失
    :param target_sens_true: 目标敏感真实值
    :param target_sens_predict: 目标敏感预测值
    :return:
    """
    cols = target_sens_true.columns
    loss = 0.0
    for col in cols:
        loss += mean_squared_error(target_sens_true[col], target_sens_predict[col], squared=False)

    return loss


def buildRegressionModel(cur_col, match_hos_train_data_x, match_hos_train_data_y, target_hos_test_data_x):
    """
    建立回归模型
    :param cur_col:
    :param match_hos_train_data_x:
    :param match_hos_train_data_y:
    :param target_hos_test_data_x:
    :return:
    """
    model = LinearRegression()
    model.fit(match_hos_train_data_x, match_hos_train_data_y)
    target_hos_test_data_y_predict = model.predict(target_hos_test_data_x)
    # logger.warning("target_index:{}, {} 属性完成建模预测完成...".format(
    #     target_hos_test_data_x.index.to_list()[0], cur_col))
    return target_hos_test_data_y_predict


def do_build_model(match_hos_train_data_x, match_hos_train_data_ys, target_hos_test_data_x, sens_cols):
    """
    对每个敏感特征建立模型预测
    :param match_hos_train_data_x: 用于训练的数据X
    :param match_hos_train_data_ys: 用于训练的数据y
    :param target_hos_test_data_x: 用于测试的数据
    :param sens_cols: 敏感数据 （做预测）
    :return:
    """
    # 保存结果矩阵
    result_df = pd.DataFrame(columns=sens_cols, index=target_hos_test_data_x.index)
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for cur_col in sens_cols:
            match_hos_train_data_y = match_hos_train_data_ys[cur_col]

            # 建立回归模型
            thread_list.append(executor.submit(
                buildRegressionModel,
                cur_col,
                match_hos_train_data_x,
                match_hos_train_data_y,
                target_hos_test_data_x
            ))

        wait(thread_list, return_when=ALL_COMPLETED)
        for cur_col, thread in zip(sens_cols, thread_list):
            result_df[cur_col] = thread.result()

    if result_df.isna().sum().sum() > 0:
        logger.error("出现并发问题...缺失某个敏感特征...")
        raise Exception("多线程跑线程并发错误...")

    logger.warning(f"target_index:{target_hos_test_data_x.index[0]} 预测完成...")

    return result_df


def array_diff(a, b):
    # 创建数组在，且数组元素在a不在b中
    return [x for x in a if x not in b]


def get_similar_rank(len_split, init_similar_weight, target_pre_data_select, match_data_x):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param init_similar_weight:
    :param len_split:
    :param match_data_x:
    :param target_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=match_data_x.index)
        similar_rank['distance'] = abs((match_data_x - target_pre_data_select.values) * init_similar_weight).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values
        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        raise Exception(err)

    return patient_ids, sample_ki

def main_run_with_psm_noise(from_hos_id, to_hos_id, n_comp=0.95):
    # 0. 获取数据, 获取度量
    _, target_data_x, _, target_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    match_data_x, _, match_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
    match_init_similar_weight = get_init_similar_weight(to_hos_id)
    columns_list = match_data_x.columns.to_list()

    # 0.5 可能会增加噪声
    target_data_x = add_laplace_noise(target_data_x, μ=0, b=0.5)

    # 1. PCA降维 得到 降维矩阵 （不同处理方式），然后还原数据
    pca_match_data_x, pca_target_data_x, recover_target_data_x = pca_reduction_with_similar_weight(
        match_data_x, target_data_x, match_init_similar_weight, n_comp
    )

    # 3. 将还原的数据的某些非敏感特征推导预测敏感特征 （X,y) LR或XGB或...
    # 3.1 获取敏感特征集
    sens_cols = get_sensitive_columns()
    train_cols = array_diff(columns_list, sens_cols)

    # 3.2 对每个敏感特征预测建模(训练数据暂时使用全部匹配数据）
    # 3.3. 预测对应的敏感特征信息并生成新的矩阵
    match_hos_train_data = match_data_x
    match_hos_train_data_x = match_hos_train_data[train_cols]
    match_hos_train_data_y = match_hos_train_data[sens_cols]
    target_hos_test_data_x = recover_target_data_x[train_cols]

    target_sens_predict = do_build_model(
        match_hos_train_data_x, match_hos_train_data_y, target_hos_test_data_x, sens_cols
    )

    # 4. 通过损失函数计算对应的损失信息
    target_sens_true = target_data_x[sens_cols]  # 敏感真实值
    loss_value = do_cal_loss_info(target_sens_true, target_sens_predict)

    logger.info("[使用相似性度量-增加噪声] - comp:{} - 计算得知当前损失值为: {}".format(comp, loss_value))

    return loss_value


def main_run_with_psm_sens_weight(from_hos_id, to_hos_id, n_comp=0.95, sens_weight=0.0):
    """
    主入口 - 用相似性度量
    :return:
    """
    # 0. 获取数据, 获取度量
    _, target_data_x, _, target_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    match_data_x, _, match_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
    match_init_similar_weight = get_init_similar_weight(to_hos_id)
    len_split = int(match_data_x.shape[0] * 0.1)
    columns_list = match_data_x.columns.to_list()

    match_init_similar_weight = process_sensitive_feature_weight(
        match_init_similar_weight,
        columns_list,
        sens_coef=sens_weight
    )

    # 选取目标患者
    test_data_ids_1, test_data_ids_0 = get_target_test_id(from_hos_id)
    test_data_ids = np.concatenate((test_data_ids_1, test_data_ids_0), axis=0)
    target_data_x = target_data_x.loc[test_data_ids]
    target_data_y = target_data_y.loc[test_data_ids]

    # 1. PCA降维 得到 降维矩阵 （不同处理方式），然后还原数据
    pca_match_data_x, pca_target_data_x, recover_target_data_x = pca_reduction_with_similar_weight(
        match_data_x, target_data_x, match_init_similar_weight, n_comp
    )

    # 3. 将还原的数据的某些非敏感特征推导预测敏感特征 （X,y) LR或XGB或...
    # 3.1 获取敏感特征集
    sens_cols = get_sensitive_columns()
    train_cols = array_diff(columns_list, sens_cols)

    # 3.2 对每个敏感特征预测建模(训练数据暂时使用全部匹配数据） 需要针对训练集（也就是当前中心取10%的相似样本）建模
    match_hos_train_data = match_data_x
    match_hos_train_data_x = match_hos_train_data[train_cols]
    match_hos_train_data_y = match_hos_train_data[sens_cols]
    target_hos_test_data_x = recover_target_data_x[train_cols]

    # 3.2.1 for循环每个患者，找到对应的相似样本集合，多线程对每个敏感特征进行回归建模，保存下来后得到最终的result_df
    target_index_list = recover_target_data_x.index.to_list()
    target_sens_predict = pd.DataFrame(columns=sens_cols)
    for target_index in target_index_list:
        cur_target_x = recover_target_data_x.loc[[target_index], :]

        # 获取相似样本呢
        patient_ids, _ = get_similar_rank(
            len_split=len_split,
            init_similar_weight=match_init_similar_weight,
            target_pre_data_select=cur_target_x,
            match_data_x=match_hos_train_data
        )

        fit_train_y = match_hos_train_data_y.loc[patient_ids]
        fit_train_x = match_hos_train_data_x.loc[patient_ids]
        fit_test_x = target_hos_test_data_x.loc[[target_index], :]
        # 3.3. 预测对应的敏感特征信息并生成新的矩阵
        fit_test_y_predict = do_build_model(
            fit_train_x, fit_train_y, fit_test_x, sens_cols
        )
        target_sens_predict = pd.concat([target_sens_predict, fit_test_y_predict], axis=0)

    # 4. 通过损失函数计算对应的损失信息
    target_sens_true = target_data_x[sens_cols]  # 敏感真实值
    loss_value = do_cal_loss_info(target_sens_true, target_sens_predict)

    logger.info("[使用相似性度量-更改权重{}] - comp:{} - 计算得知当前损失值为: {}".format(sens_weight, comp, loss_value))

    return loss_value


def main_run_with_psm(from_hos_id, to_hos_id, n_comp):
    """
    主入口 - 用相似性度量
    :return:
    """
    # 0. 获取数据, 获取度量
    _, target_data_x, _, target_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    match_data_x, _, match_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
    match_init_similar_weight = get_init_similar_weight(to_hos_id)
    len_split = int(match_data_x.shape[0] * 0.1)
    columns_list = match_data_x.columns.to_list()

    # 选取目标患者
    test_data_ids_1, test_data_ids_0 = get_target_test_id(from_hos_id)
    test_data_ids = np.concatenate((test_data_ids_1, test_data_ids_0), axis=0)
    target_data_x = target_data_x.loc[test_data_ids]
    target_data_y = target_data_y.loc[test_data_ids]

    # 1. PCA降维 得到 降维矩阵 （不同处理方式），然后还原数据
    pca_match_data_x, pca_target_data_x, recover_target_data_x = pca_reduction_with_similar_weight(
        match_data_x, target_data_x, match_init_similar_weight, n_comp
    )

    # 3. 将还原的数据的某些非敏感特征推导预测敏感特征 （X,y) LR或XGB或...
    # 3.1 获取敏感特征集
    sens_cols = get_sensitive_columns()
    train_cols = array_diff(columns_list, sens_cols)

    # 3.2 对每个敏感特征预测建模(训练数据暂时使用全部匹配数据） 需要针对训练集（也就是当前中心取10%的相似样本）建模
    match_hos_train_data = match_data_x
    match_hos_train_data_x = match_hos_train_data[train_cols]
    match_hos_train_data_y = match_hos_train_data[sens_cols]
    target_hos_test_data_x = recover_target_data_x[train_cols]

    # 3.2.1 for循环每个患者，找到对应的相似样本集合，多线程对每个敏感特征进行回归建模，保存下来后得到最终的result_df
    target_index_list = recover_target_data_x.index.to_list()
    target_sens_predict = pd.DataFrame(columns=sens_cols)
    for target_index in target_index_list:
        cur_target_x = recover_target_data_x.loc[[target_index], :]

        # 获取相似样本呢
        patient_ids, _ = get_similar_rank(
            len_split=len_split,
            init_similar_weight=match_init_similar_weight,
            target_pre_data_select=cur_target_x,
            match_data_x=match_hos_train_data
        )

        fit_train_y = match_hos_train_data_y.loc[patient_ids]
        fit_train_x = match_hos_train_data_x.loc[patient_ids]
        fit_test_x = target_hos_test_data_x.loc[[target_index], :]
        # 3.3. 预测对应的敏感特征信息并生成新的矩阵
        fit_test_y_predict = do_build_model(
            fit_train_x, fit_train_y, fit_test_x, sens_cols
        )
        target_sens_predict = pd.concat([target_sens_predict, fit_test_y_predict], axis=0)

    # 4. 通过损失函数计算对应的损失信息
    target_sens_true = target_data_x[sens_cols]  # 敏感真实值
    loss_value = do_cal_loss_info(target_sens_true, target_sens_predict)

    logger.info("[使用相似性度量] - comp:{} - 计算得知当前损失值为: {}".format(comp, loss_value))

    return loss_value


def main_run_with_no_psm(from_hos_id, to_hos_id, n_comp):
    """
    主入口 - 不用相似性度量
    :return:
    """
    # 0. 获取数据, 获取度量
    _, target_data_x, _, target_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    match_data_x, _, match_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
    columns_list = match_data_x.columns.to_list()

    # 1. PCA降维 得到 降维矩阵 （不同处理方式），然后还原数据
    pca_match_data_x, pca_target_data_x, recover_target_data_x = pca_reduction(
        match_data_x, target_data_x, n_comp
    )

    # 3. 将还原的数据的某些非敏感特征推导预测敏感特征 （X,y) LR或XGB或...
    # 3.1 获取敏感特征集
    sens_cols = get_sensitive_columns()
    train_cols = array_diff(columns_list, sens_cols)

    # 3.2 对每个敏感特征预测建模(训练数据暂时使用全部匹配数据）
    # 3.3. 预测对应的敏感特征信息并生成新的矩阵
    match_hos_train_data = match_data_x
    match_hos_train_data_x = match_hos_train_data[train_cols]
    match_hos_train_data_ys = match_hos_train_data[sens_cols]
    target_hos_test_data_x = recover_target_data_x[train_cols]
    target_sens_predict = do_build_model(
        match_hos_train_data_x, match_hos_train_data_ys, target_hos_test_data_x, sens_cols
    )

    # 4. 通过损失函数计算对应的损失信息
    target_sens_true = target_data_x[sens_cols]  # 敏感真实值
    loss_value = do_cal_loss_info(target_sens_true, target_sens_predict)

    logger.info("[不使用相似性度量] - comp:{} - 计算得知当前损失值为: {}".format(comp, loss_value))

    return loss_value


if __name__ == '__main__':
    from_hospital_id = 73
    to_hospital_id = 0
    pool_nums = 22

    m_sample_weight = 0.01

    comp_list = [0.95]

    save_path = f"./result/S08/"
    create_path_if_not_exists(save_path)

    all_res_df = pd.DataFrame()

    for comp in comp_list:
        all_res_df.loc["pca_with_psm_loss-lr", comp] = main_run_with_psm(from_hospital_id, to_hospital_id, n_comp=comp)
        # all_res_df.loc["pca_with_no_psm_loss-lr", comp] = main_run_with_no_psm(from_hospital_id, to_hospital_id, n_comp=comp)
        # all_res_df.loc["pca_with_no_psm_loss-lr-noise", comp] = main_run_with_psm_noise(from_hospital_id, to_hospital_id, n_comp=comp)
        all_res_df.loc["pca_with_no_psm_loss-lr-sw0.0", comp] = main_run_with_psm_sens_weight(from_hospital_id, to_hospital_id, n_comp=comp, sens_weight=0.0)
        all_res_df.loc["pca_with_no_psm_loss-lr-sw0.5", comp] = main_run_with_psm_sens_weight(from_hospital_id, to_hospital_id, n_comp=comp, sens_weight=0.5)

    all_res_df.to_csv(os.path.join(save_path, f"S08_{from_hospital_id}_{to_hospital_id}_attack_loss1_with_plr.csv"))
