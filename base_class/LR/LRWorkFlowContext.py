# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     WorkFlowContext
   Description:   S04 LR 个性化建模
   Author:        cqh
   date:          2022/12/2 16:52
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import time

import pandas as pd

from api_utils import create_path_if_not_exists, save_to_csv_by_row
from base_class.BaseClass import BaseWorkFlowContext, BasePersonalModel
from base_class.LR.LRPersonalModel import LRPersonalModel
from base_class.LR.LRWeight import LRWeight
from base_class.MyDataSet import MyDataSet
from email_api import get_run_time, send_success_mail
from my_logger import logger


class LRWorkFlowContext(BaseWorkFlowContext):
    """
    工作流上下文
    """

    def __init__(self, data_set: MyDataSet, personal_model: BasePersonalModel, version, save_path, save_result_file,
                 program_name):
        super().__init__(data_set, personal_model, version)
        self.save_path = save_path
        self.save_result_file = save_result_file
        self.program_name = program_name

    def fit(self, is_transfer):
        """
        跑LR个性化建模
        :return:
        """
        # 读取数据
        target_patient_data_x, target_patient_data_y = self.dataset.get_from_hos_data()
        match_patient_data_x, match_patient_data_y = self.dataset.get_to_hos_data()
        logger.warning("load data - train_data:{}, test_data:{}".
                       format(target_patient_data_x.shape, match_patient_data_x.shape))

        # 初始化相关命名文件
        self.init_any_file()

        start_time = time.time()
        # 进行个性化建模（LR）
        score = self.personal_model.fit(
            target_patient_data_x, target_patient_data_y, match_patient_data_x, match_patient_data_y,
            is_transfer=is_transfer
        )
        end_time = time.time()

        # 输出结果，保存结果
        self.output_result_info(start_time, end_time, score)

    def output_result_info(self, start_time, end_time, score):
        # save到全局结果集合里
        save_df = pd.DataFrame(columns=['start_time', 'end_time', 'run_time', 'auc_score_result'])
        start_time_date, end_time_date, run_date_time = get_run_time(start_time, end_time)
        save_df.loc[self.program_name + "_" + str(os.getpid()), :] = [start_time_date, end_time_date, run_date_time, score]
        save_to_csv_by_row(self.save_result_file, save_df)

        # 发送邮箱
        send_success_mail(self.program_name, run_start_time=start_time, run_end_time=end_time)
        print("end!")

    def init_any_file(self):
        create_path_if_not_exists(self.save_path)


if __name__ == '__main__':
    from_hos_id = 73
    to_hos_id = 167

    mw = LRWeight(from_hos_id, to_hos_id)
    init_similar = mw.get_init_similar()
    transfer_weight = mw.get_transfer_similar()

    lrpm = LRPersonalModel(init_similar_weight=init_similar, transfer_weight=transfer_weight)
    dataset = MyDataSet(from_hos_id, to_hos_id)
    context = BaseWorkFlowContext(dataset=dataset, personal_model=lrpm)
    context.fit(is_transfer=1)
