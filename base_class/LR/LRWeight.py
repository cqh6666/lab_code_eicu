# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     PsmWeight
   Description:   ...
   Author:        cqh
   date:          2022/12/2 16:42
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'

from base_class.BaseClass import BaseWeight
from lr_utils_api import get_init_similar_weight, get_transfer_weight


class LRWeight(BaseWeight):
    """
     获取相似性度量和迁移度量
    """
    def __init__(self, psm_hos_id, trans_hos_id):
        super().__init__(psm_hos_id)
        self.trans_hos_id = trans_hos_id

    def get_init_similar(self):
        return get_init_similar_weight(self.psm_hos_id)

    def get_transfer_similar(self):
        return get_transfer_weight(self.trans_hos_id)