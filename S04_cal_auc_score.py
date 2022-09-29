# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S02_cal_auc_score
   Description:   ...
   Author:        cqh
   date:          2022/7/17 13:29
-------------------------------------------------
   Change Activity:
                  2022/7/17:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

from sklearn.metrics import roc_auc_score
import pandas as pd

step = 4
version = 7
model = 'lr'
# model = 'XGB'

local_boost = [50, 100]
select_rate = [10]
hos_id_list = [0, 73, 167, 264, 420, 338]
# hos_id_list = [0]
columns = ['host_id', 'local_boost', 'select_rate', 'transfer', 'no_transfer']

result_df = pd.DataFrame(columns=columns)
all_save_path = f"/home/chenqinhai/code_eicu/my_lab/result/S04/"
hos_path = "/home/chenqinhai/code_eicu/my_lab/result/S04/{}"
for hid in hos_id_list:

    for boost in local_boost:
        for select in select_rate:
            try:
                # S04_lr_test_tra0_boost50_select10_v3.csv
                res_tra = pd.read_csv(os.path.join(hos_path.format(hid), f"S04_{model}_test_tra1_boost{boost}_select{select}_v{version}.csv"))
                res_no_tra = pd.read_csv(os.path.join(hos_path.format(hid), f"S04_{model}_test_tra0_boost{boost}_select{select}_v{version}.csv"))

                # 迁移和非迁移
                score = roc_auc_score(res_tra['real'], res_tra['prob'])
                score2 = roc_auc_score(res_no_tra['real'], res_no_tra['prob'])

                cur_res_df = pd.DataFrame([[hid, boost, select, score, score2]], columns=columns)
                result_df = pd.concat([result_df, cur_res_df], ignore_index=True)

            except Exception as err:
                continue

test_result_file_name = os.path.join(all_save_path, f"S04_{model}_test_auc_result_v{version}.csv")
result_df.to_csv(test_result_file_name, index=False)
print(f"===================== step:{step}, version:{version} ======================================")
print(result_df)
