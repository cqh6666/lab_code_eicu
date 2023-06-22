# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S01_load_eicu_fairness_data
   Description:   ...
   Author:        cqh
   date:          2023/4/24 17:19
-------------------------------------------------
   Change Activity:
                  2023/4/24:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
from get_eicu_dataset import get_subgroup_data

# eicu
my_cols = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/eicu_data/subgroup_select_feature.csv", index_col=0).squeeze().to_list()
data_file_name = "/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/eicu_data/test_valid_{}.feather"

data_df = get_subgroup_data([1])
data_df2 = get_subgroup_data([2])
data_df3 = get_subgroup_data([3])
data_df4 = get_subgroup_data([4])
data_df5 = get_subgroup_data([5])

