# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     concat_feature_name
   Description:   ...
   Author:        cqh
   date:          2023/4/10 20:21
-------------------------------------------------
   Change Activity:
                  2023/4/10:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

import pandas as pd
import numpy as np

data_path = "/home/chenqinhai/code_eicu/my_lab/fairness_strategy"
name_df = pd.read_csv(os.path.join(data_path, "AKI_age_map1994_v2.csv"), index_col=0)
result_df = pd.read_csv(os.path.join(data_path, "sbdt_feature_loss_rate0.85_v15.csv"), index_col=0)

index_list = result_df.index.to_list()

for index in index_list:
    try:
        d_index = index.upper()
        result_df.loc[index, "feature_name"] = name_df.loc[d_index, "Name"]
    except:
        print(index, "error!")
        pass


result_df.to_csv(os.path.join(data_path, "sbdt_feature_loss_rate0.85_name_v15.csv"))