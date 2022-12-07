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

from base_class.LR.LRPersonalModel import LRPersonalModel
from base_class.LR.LRWeight import LRWeight
from base_class.LR.LRWorkFlowContext import LRWorkFlowContext
from base_class.MyDataSet import MyDataSet

from_hos_id = 73
to_hos_id = 0
local_lr_iter = 100
select = 10
select_ratio = select * 0.01
m_sample_weight = 0.01
version = 1
is_transfer = 0
save_path = f"/home/chenqinhai/code_eicu/my_lab/result/S04/{from_hos_id}/"
save_result_file = f"/home/chenqinhai/code_eicu/my_lab/result/S04_id{from_hos_id}_other_LR_result_save.csv"
program_name = f"S04_LR_from{from_hos_id}_to{to_hos_id}_tra{is_transfer}_v{version}"

# 获取数据集
dataset = MyDataSet(from_hos_id, to_hos_id)

# 获取度量权重
mw = LRWeight(from_hos_id, to_hos_id)

# LR个性化建模
lrpm = LRPersonalModel(init_similar_weight=mw.get_init_similar(), transfer_weight=mw.get_transfer_similar())

# 创建工作流
context = LRWorkFlowContext(
    data_set=dataset,
    personal_model=lrpm,
    version=version,
    save_path=save_path,
    save_result_file=save_result_file,
    program_name=program_name
)
context.fit(is_transfer=is_transfer)
