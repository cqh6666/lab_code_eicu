# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     email_test
   Description:   ...
   Author:        cqh
   date:          2022/9/23 16:53
-------------------------------------------------
   Change Activity:
                  2022/9/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import time
import numpy as np
import pandas as pd

from api_utils import get_all_norm_data, get_continue_feature

all_data = get_all_norm_data()
print(all_data.info())
spe_feature = ['hospitalid', 'index']
con_feature = get_continue_feature()
all_feature = all_data.columns.tolist()
cat_feature = list(set(all_feature).difference(set(con_feature)))

# 方便合并
all_data.set_index(['index'], drop=False, inplace=True)

# 去除spe feature
for spe in spe_feature:
    cat_feature.remove(spe)

spe_data = all_data[spe_feature].copy()
cat_data = all_data[cat_feature].copy()
con_data = all_data[con_feature].copy()

# 修改cat_data类型
cat_data2 = cat_data.astype(np.int8)
# 判断是否完全一致
# assert_frame_equal(cat_data, cat_data2, check_dtype=False)

# 修改con_data类型
con_data2 = con_data.astype(np.float32)
print("astype success!")

new_df = pd.concat([spe_data, cat_data2, con_data2], axis=1)

new_df.reset_index(drop=True, inplace=True)
new_df = new_df[all_data.columns]

TRAIN_PATH = "/home/chenqinhai/code_eicu/my_lab/data/train_file"
new_df.to_feather(os.path.join(TRAIN_PATH, "all_data_df_norm_process_v2.feather"))
print("save success!")
print(new_df.info())
