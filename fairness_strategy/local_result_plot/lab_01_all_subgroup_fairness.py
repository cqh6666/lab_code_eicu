# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lab_01_all_subgroup_fairness
   Description:   本文件是绘制3种建模策略的根据不同阈值选取方式在 各亚组的TPR,FPR,PPV的均值和标准差 对比图
                  采用柱状图来比较
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

# Apply the default theme
sns.set_theme()
# sns.set_style("whitegrid")

sns.set_style("darkgrid",{"font.sans-serif":['SimHei','Droid Sans Fallback']})

# Load an example dataset
my_data_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_to_plot.csv", index_col=0)

fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(10, 4))
plt.subplots_adjust(left=0.125, bottom=0.2, wspace=0.35) #调整子图间距

select_cond = (my_data_df["threshold_type"] == "同等阈值") & (my_data_df["score_type"] == "TPR")
plot_df = my_data_df[select_cond]
axesSub = sns.barplot(data=plot_df, errorbar='se',x="threshold", y="score", hue="建模策略", capsize=.06, errwidth=1.2, ax=ax[0])
axesSub.set_xlabel("从全部样本中选取 Top-K")
axesSub.set_ylabel("召回率(TPR)")
axesSub.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub.legend(loc='upper left')

select_cond = (my_data_df["threshold_type"] == "同比例阈值") & (my_data_df["score_type"] == "TPR")
plot_df = my_data_df[select_cond]
axesSub2 = sns.barplot(data=plot_df, errorbar='se', x="threshold", y="score", hue="建模策略", capsize=.06, errwidth=1.2, ax=ax[1])
axesSub2.set_xlabel("从每个亚组选取 Top-K%")
axesSub2.set_ylabel("召回率(TPR)")
axesSub2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub2.legend(loc='upper left')

select_cond = (my_data_df["threshold_type"] == "根据各亚组AKI数量比例阈值") & (my_data_df["score_type"] == "TPR")
plot_df = my_data_df[select_cond]
axesSub3 = sns.barplot(data=plot_df, errorbar='se',x="threshold", y="score", hue="建模策略", capsize=.06, errwidth=1.2, ax=ax[2])
axesSub3.set_xlabel("根据每个亚组AKI数量的 Top-K%")
axesSub3.set_ylabel("召回率(TPR)")
axesSub3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub3.legend(loc='upper left')

# save fig
plt.savefig('lab_01_result.jpg', dpi=1500, bbox_inches='tight')
plt.show()
