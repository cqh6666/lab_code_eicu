# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     DefaultLR
   Description:   ...
   Author:        cqh
   date:          2022/12/2 21:39
-------------------------------------------------
   Change Activity:
                  2022/12/2:
-------------------------------------------------
"""
__author__ = 'cqh'


from_hos_id = 73
to_hos_id = 167

mw = MyLRWeight(from_hos_id, to_hos_id)
init_similar = mw.get_init_similar()
transfer_weight = mw.get_transfer_similar()

lrpm = LRPersonalModel(init_similar_weight=init_similar, transfer_weight=transfer_weight)
dataset = MyDataSet(from_hos_id, to_hos_id)
context = BaseWorkFlowContext(dataset=dataset, personal_model=lrpm)
context.fit(is_transfer=1)