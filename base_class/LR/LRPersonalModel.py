# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     LRPersonalModel
   Description:   ...
   Author:        cqh
   date:          2022/12/2 19:14
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_EXCEPTION
import pandas as pd
import queue
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from api_utils import covert_time_format
from base_class.BaseClass import BasePersonalModel
from my_logger import logger


class LRPersonalModel(BasePersonalModel):
    # 多线程用到的全局锁, 异常消息队列
    exec_queue = queue.Queue()

    def __init__(self, init_similar_weight, transfer_weight, pool_nums=20, m_sample_weight=0.01,
                 select_rate=0.1, local_lr_iter=100):
        super().__init__(init_similar_weight)
        self.transfer_weight = transfer_weight
        self.pool_nums = pool_nums
        self.m_sample_weight = m_sample_weight
        self.select_rate = select_rate
        self.local_lr_iter = local_lr_iter
        self.len_split = None
        self.is_transfer = None

    def output_params_info(self):
        """
        输出相关信息
        :return:
        """
        logger.warning(f"[params] - model_select:LR, pool_nums:{self.pool_nums}, is_transfer:{self.is_transfer}, "
                       f"max_iter:{self.local_lr_iter}, select:{self.select_rate}, {self.len_split}]")

    def fit(self, target_data_x, target_data_y, match_data_x, match_data_y, is_transfer=0):

        self.is_transfer = is_transfer
        self.len_split = int(match_data_x.shape[0] * self.select_rate)
        self.output_params_info()

        # 目标患者建模保存结果df
        target_id_list = target_data_x.index.values
        target_result = pd.DataFrame(index=target_id_list, columns=['real', 'prob'])
        target_result['real'] = target_data_y

        logger.warning("starting personalized modelling...")
        mt_begin_time = time.time()
        # 匹配相似样本（从训练集） 建模 多线程
        with ThreadPoolExecutor(max_workers=self.pool_nums) as executor:
            thread_list = []
            for test_id in target_id_list:
                pre_data_select_x = target_data_x.loc[[test_id]]
                thread = executor.submit(
                    self.personalized_modeling,
                    pre_data_select_x, match_data_x, match_data_y, target_data_y, is_transfer
                )
                thread_list.append(thread)

            # 若出现第一个错误, 反向逐个取消任务并终止
            wait(thread_list, return_when=FIRST_EXCEPTION)
            for cur_thread in reversed(thread_list):
                cur_thread.cancel()

            wait(thread_list, return_when=ALL_COMPLETED)

            # 若出现异常直接返回
            if not self.exec_queue.empty():
                logger.error("something task error... we have to stop!!!")
                return

            # 保存结果
            for test_id, thread in zip(target_id_list, thread_list):
                target_result.loc[test_id, "prob"] = thread.result()

        mt_end_time = time.time()
        run_time = covert_time_format(mt_end_time - mt_begin_time)
        logger.warning(f"done - cost_time: {run_time}...")

        if target_result.isna().sum().sum() > 0:
            raise ValueError("target_result exist NaN...")
        score = roc_auc_score(target_result['real'], target_result['prob'])
        logger.info(f"auc score: {score}")

        return score

    def personalized_modeling(self, pre_target_select_x, match_data_x, match_data_y, target_data_y, is_transfer):
        """
        根据距离得到 某个目标测试样本对每个训练样本的距离
        test_id - patient id
        pre_data_select - dataframe
        :return: 最终的相似样本
        """
        # 如果消息队列中消息不为空，说明已经有任务异常了
        if not self.exec_queue.empty():
            return

        try:
            patient_ids, sample_ki = self.get_similar_rank(pre_target_select_x, match_data_x)

            fit_train_y = match_data_y.loc[patient_ids]
            fit_train_x = match_data_x.loc[patient_ids]
            if is_transfer == 1:
                fit_train_x = fit_train_x * self.transfer_weight
                fit_test_x = pre_target_select_x * self.transfer_weight
            else:
                fit_test_x = pre_target_select_x

            # 如果匹配的全是一样的标签
            if len(fit_train_y.value_counts()) <= 1:
                return target_data_y[0]

            lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=self.local_lr_iter)
            lr_local.fit(fit_train_x, fit_train_y, sample_ki)
            predict_prob = lr_local.predict_proba(fit_test_x)[0][1]
        except Exception as err:
            self.exec_queue.put("Termination")
            logger.exception(err)
            raise Exception(err)

        # logger.info(f"{threading.currentThread().getName()}, {predict_prob}")
        return predict_prob

    def get_similar_rank(self, target_pre_data_select, match_data_x):
        """
        选择前10%的样本，并且根据相似得到样本权重
        :param match_data_x:
        :param target_pre_data_select:
        :return:
        """
        try:
            similar_rank = pd.DataFrame(index=match_data_x.index)
            similar_rank['distance'] = abs(
                (match_data_x - target_pre_data_select.values) * self.init_similar_weight).sum(
                axis=1)
            similar_rank.sort_values('distance', inplace=True)
            patient_ids = similar_rank.index[:self.len_split].values
            sample_ki = similar_rank.iloc[:self.len_split, 0].values
            sample_ki = [(sample_ki[0] + self.m_sample_weight) / (val + self.m_sample_weight) for val in sample_ki]
        except Exception as err:
            raise Exception(err)

        return patient_ids, sample_ki
