# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lab_02_each_subgroup_fairness
   Description:   本文件是绘制3种建模策略的根据不同阈值选取方式在 各亚组的TPR均值和标准差 对比图
                  采用表格来比较
   Author:        cqh
   date:          2023/2/17 22:52
-------------------------------------------------
   Change Activity:
                  2023/2/17:
-------------------------------------------------
"""
__author__ = 'cqh'

# Import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

# Load an example dataset
my_data_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_to_plot.csv", index_col=0)

select_cond = (my_data_df["threshold_type"] == "根据各亚组AKI数量比例阈值") & (my_data_df["score_type"] == "TPR") & (my_data_df["threshold"] == "100%")
plot_df = my_data_df[select_cond]

# drg属性字典 index:Drg - Name, Chinese
drg_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/disease_top_20_no_drg_full_name.csv", encoding="gbk", index_col=0)
drg_list = drg_df.index.to_list()
result_df = pd.DataFrame(index=drg_list, columns=["Subgroups", "GM", "SM", "PMTL", "PMTL/GM", "PMTL/SM"])

rows, cols = plot_df.shape[0], plot_df.shape[1]

for row_id in range(rows):
    cur_df = plot_df.iloc[row_id, :]
    cur_drg = int(cur_df["drg"][3:])
    cur_build = cur_df["建模策略"]
    # cur_score = '%6.2f' % round(cur_df["score"] * 100, 2)
    # cur_score = round(cur_df["score"] * 100, 2)
    cur_score = cur_df["score"]
    result_df.loc[cur_drg, "Subgroups"] = drg_df.loc[cur_drg, "Chinese"]
    result_df.loc[cur_drg, cur_build] = cur_score

for row_id in result_df.index.to_list():
    result_df.loc[row_id, "PMTL/GM"] = round(result_df.loc[row_id, "PMTL"] / result_df.loc[row_id, "GM"], 2)
    result_df.loc[row_id, "PMTL/SM"] = round(result_df.loc[row_id, "PMTL"] / result_df.loc[row_id, "SM"], 2)

score_cols = ["GM", "SM", "PMTL", "PMTL/GM", "PMTL/SM"]
for col in score_cols:
    result_df[col] = result_df[col].apply(lambda x: format(x, '.2%'))

result_df.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/lab_02_subgroup_top20_TPR.csv", encoding="gbk")



