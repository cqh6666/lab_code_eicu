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

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error

from api_utils import get_fs_each_hos_data_X_y, get_sensitive_columns, get_diff_sens, create_path_if_not_exists
from lr_utils_api import get_init_similar_weight
from my_logger import MyLog
import pandas as pd
import numpy as np


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
        my_logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    my_logger.warning(f"开始对目标患者和匹配样本进行PCA降维...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_match_data_x = pca_model.fit_transform(match_data_x)
    new_target_data_x = pca_model.transform(target_data_x)

    # 转成df格式
    pca_match_data_x = pd.DataFrame(data=new_match_data_x, index=match_data_x.index)
    pca_target_data_x = pd.DataFrame(data=new_target_data_x, index=target_data_x.index)

    my_logger.info(
        f"方差占比阈值: {pca_model.n_components}, 降维维度: {pca_model.n_components_}")

    # =============================================================================

    my_logger.warning("开始恢复数据...")
    rec_tar_x = pca_model.inverse_transform(pca_target_data_x)
    rec_tar_x_df = pd.DataFrame(data=rec_tar_x, index=target_data_x.index, columns=target_data_x.columns)

    my_logger.info("成功恢复目标患者数据:  target_data:{}".format(rec_tar_x_df.shape))

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
        my_logger.warning("n_comp 超过 1 了，参数错误，重新设置为 1...")
        sys.exit(1)

    my_logger.warning(f"开始对目标患者和匹配样本进行PCA降维...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_match_data_x = pca_model.fit_transform(match_data_x * similar_weight)
    new_target_data_x = pca_model.transform(target_data_x * similar_weight)

    # 转成df格式
    pca_match_data_x = pd.DataFrame(data=new_match_data_x, index=match_data_x.index)
    pca_target_data_x = pd.DataFrame(data=new_target_data_x, index=target_data_x.index)

    my_logger.info(
        f"方差占比阈值: {pca_model.n_components}, 降维维度: {pca_model.n_components_}")

    # =============================================================================

    my_logger.warning(f"开始恢复数据...")
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

    my_logger.info("成功对目标患者进行恢复数据:  target_data: {}".
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


def buildRegressionModel(match_hos_train_data_x, match_hos_train_data_y, target_hos_test_data_x):
    """
    建立回归模型
    :param match_hos_train_data_x:
    :param match_hos_train_data_y:
    :param target_hos_test_data_x:
    :return:
    """
    model = LinearRegression()
    model.fit(match_hos_train_data_x, match_hos_train_data_y)
    target_hos_test_data_y_predict = model.predict(target_hos_test_data_x)
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

    for cur_col in sens_cols:
        match_hos_train_data_y = match_hos_train_data_ys[cur_col]
        # 建立回归模型
        result_df[cur_col] = buildRegressionModel(match_hos_train_data_x, match_hos_train_data_y,
                                                  target_hos_test_data_x)
        my_logger.warning("{} 属性完成建模预测完成...".format(cur_col))

    return result_df


def array_diff(a, b):
    # 创建数组在，且数组元素在a不在b中
    return [x for x in a if x not in b]


def main_run_with_psm(from_hos_id, to_hos_id, n_comp):
    """
    主入口 - 用相似性度量
    :return:
    """
    # 0. 获取数据, 获取度量
    _, target_data_x, _, target_data_y = get_fs_each_hos_data_X_y(from_hos_id)
    match_data_x, _, match_data_y, _ = get_fs_each_hos_data_X_y(to_hos_id)
    match_init_similar_weight = get_init_similar_weight(to_hos_id)
    columns_list = match_data_x.columns.to_list()

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

    my_logger.info("[使用相似性度量] - comp:{} - 计算得知当前损失值为: {}".format(comp, loss_value))

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

    my_logger.info("[不使用相似性度量] - comp:{} - 计算得知当前损失值为: {}".format(comp, loss_value))

    return loss_value


if __name__ == '__main__':
    my_logger = MyLog().logger
    from_hos_id = 73
    to_hos_id = 167
    n_component = 0.95
    comp_list = [0.9, 0.95, 0.99, 0.995, 0.999]

    save_path = f"./result/S08/"
    create_path_if_not_exists(save_path)

    all_res_df = pd.DataFrame()

    for comp in comp_list:
        loss_1 = main_run_with_psm(from_hos_id, to_hos_id, n_comp=comp)
        loss_2 = main_run_with_no_psm(from_hos_id, to_hos_id, n_comp=comp)
        all_res_df.loc["pca_with_psm_loss-lr", comp] = loss_1
        all_res_df.loc["pca_with_no_psm_loss-lr", comp] = loss_2

    all_res_df.to_csv(os.path.join(save_path, "S08_attack_loss_with_lr.csv"))
