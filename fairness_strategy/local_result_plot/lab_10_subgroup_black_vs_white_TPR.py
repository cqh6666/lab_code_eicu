# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lab_10_subgroup_black_vs_white_TPR
   Description:   ...
   Author:        cqh
   date:          2023/2/20 22:38
-------------------------------------------------
   Change Activity:
                  2023/2/20:
-------------------------------------------------
"""
__author__ = 'cqh'
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
import numpy as np

# Apply the default theme
sns.set_theme()
plt.figure(figsize=(13,7))
plt.subplots_adjust(left=0.3) #调整子图间距

sns.set_style("whitegrid", {"font.sans-serif":['SimHei','Droid Sans Fallback']})
sns.set_context(rc={'font.size': 17})
result_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_white_vs_black_fairness_TPR_threshold_2241.csv", index_col=0)
result_df['drg'] = result_df.index

# process data
drg_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/disease_top_20_no_drg_full_name.csv", encoding="gbk")
result_df['各入院原因的亚组'] = drg_df['Chinese'].to_list()
# result_df.loc["Drg50", 'personal'] = result_df.loc["Drg50", 'personal'] / 2
# result_df.loc["Drg259", 'personal'] = result_df.loc["Drg259", 'personal'] / 2
result_df['种族之间的公平性(白人TPR-黑人TPR)'] = result_df['personal']
axs = sns.barplot(x="种族之间的公平性(白人TPR-黑人TPR)", y="各入院原因的亚组", data=result_df, palette="deep")
axs.set_xlabel('')
axs.set_ylabel('')
axs.xaxis.tick_top()
axs.set_title("种族群体之间的公平性(多数-少数)", fontsize=18)
# axs.set_xticks(np.arange(-0.8, 0.6, 0.1))
# plt.xticks(rotation=90)
plt.savefig('lab_10_result_4-11.jpg', dpi=600, bbox_inches="tight")

plt.show()

