# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_all_columns
   Description:   ...
   Author:        cqh
   date:          2023/1/13 21:13
-------------------------------------------------
   Change Activity:
                  2023/1/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(1))
columns = test_df.columns

# Demo
demo_cols = ["Demo1"]

# Drg
drg_cols = []
for col in columns:
    if col.startswith("Drg"):
        drg_cols.append(col)

ccs_cols = []
for col in columns:
    if col.startswith("CCS"):
        ccs_cols.append(col)

all_cols = demo_cols + drg_cols + ccs_cols
all_cols_df = pd.Series(all_cols)
all_cols_df.to_csv("ku_data_select_cols.csv")

my_cols = pd.read_csv("ku_data_select_cols.csv", index_col=0).squeeze().to_list()
