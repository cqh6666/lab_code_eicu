# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     BaseClass
   Description:   ...
   Author:        cqh
   date:          2022/12/2 16:34
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'

from abc import ABC, abstractmethod

from base_class.MyDataSet import MyDataSet


class BasePersonalModel(ABC):
    def __init__(self, init_similar_weight):
        self.init_similar_weight = init_similar_weight

    @abstractmethod
    def fit(self, target_data_x, target_data_y, match_data_x, match_data_y, is_transfer):
        """
        必须实现
        :return:
        """


class BaseWeight(ABC):
    def __init__(self, psm_hos_id):
        self.psm_hos_id = psm_hos_id

    @abstractmethod
    def get_init_similar(self):
        """
        获取相似性度量
        :return:
        """


class BaseWorkFlowContext(ABC):
    def __init__(self,
                 dataset: MyDataSet,
                 personal_model: BasePersonalModel,
                 version=1,
                 ):
        self.dataset = dataset
        self.personal_model = personal_model
        self.version = version

    @abstractmethod
    def fit(self, is_transfer):
        """
        必须实现这个方法
        :param is_transfer:
        :return:
        """