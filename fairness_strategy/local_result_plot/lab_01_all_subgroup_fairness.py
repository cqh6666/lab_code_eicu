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

sns.set_style("darkgrid",{"font.sans-serif":['SimHei','Droid Sans Fallback'], "axes.titleweight":"bold"})
my_data_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result_plot/lab_01_result_plot.csv")

def process_df(data_df):
    """
    处理可以直接绘制直方图
    """
    build_types = ['GM', 'SM', 'PMTL']
    cur_result_df = pd.DataFrame(columns=['threshold', '建模策略', 'threshold_type', 'score'])
    for idx in data_df.index.to_list():
        thres = data_df.loc[idx, "result_index"]
        build = data_df.loc[idx, "result_type"]

        for bt in build_types:
            cur_result_df.loc[f"{build}_{thres}_{bt}", "threshold"] = thres
            cur_result_df.loc[f"{build}_{thres}_{bt}", "建模策略"] = bt
            cur_result_df.loc[f"{build}_{thres}_{bt}", "threshold_type"] = build
            cur_result_df.loc[f"{build}_{thres}_{bt}", "score"] = data_df.loc[idx, bt]

    return cur_result_df


my_data_df = process_df(my_data_df)
# Load an example dataset
# my_data_df2 = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_to_plot.csv", index_col=0)

fig, ax =plt.subplots(2,2,constrained_layout=True, figsize=(8, 8))
plt.subplots_adjust(left=0.125, bottom=0.125, wspace=0.45, hspace=0.45) #调整子图间距

select_cond = (my_data_df["threshold_type"] == "全部患者")
plot_df = my_data_df[select_cond]
axesSub = sns.barplot(data=plot_df, x="threshold", y="score", hue="建模策略", ax=ax[0][0])
axesSub.set_title("在所有患者上的TPR比较")
axesSub.set_xlabel("从全部样本中选取 Top-K")
axesSub.set_ylabel("召回率(TPR)")
axesSub.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub.legend(loc='upper left')

select_cond = (my_data_df["threshold_type"] == "入院原因")
plot_df = my_data_df[select_cond]
axesSub = sns.barplot(data=plot_df, x="threshold", y="score", hue="建模策略", ax=ax[0][1])
axesSub.set_title("在不同入院原因群体上的平均TPR比较")
axesSub.set_xlabel("根据各群体的AKI数量选取 Top-K%")
axesSub.set_ylabel("召回率(TPR)")
axesSub.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub.legend(loc='upper left')

select_cond = (my_data_df["threshold_type"] == "种族")
plot_df = my_data_df[select_cond]
axesSub = sns.barplot(data=plot_df, x="threshold", y="score", hue="建模策略", ax=ax[1][0])
axesSub.set_title("在不同种族群体上的平均TPR比较", fontweight="bold")
axesSub.set_xlabel("根据各群体的AKI数量选取 Top-K%")
axesSub.set_ylabel("召回率(TPR)")
axesSub.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub.legend(loc='upper left')

select_cond = (my_data_df["threshold_type"] == "性别")
plot_df = my_data_df[select_cond]
axesSub = sns.barplot(data=plot_df, x="threshold", y="score", hue="建模策略", ax=ax[1][1])
axesSub.set_title("在不同性别群体上的平均TPR比较")
axesSub.set_xlabel("根据各群体的AKI数量选取 Top-K%")
axesSub.set_ylabel("召回率(TPR)")
axesSub.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub.legend(loc='upper left')

# select_cond = (my_data_df["threshold_type"] == "根据各亚组AKI数量比例阈值") & (my_data_df["score_type"] == "TPR")
# plot_df = my_data_df[select_cond]
# axesSub3 = sns.barplot(data=plot_df, errorbar='se',x="threshold", y="score", hue="建模策略", capsize=.06, errwidth=1.2, ax=ax[2])
# axesSub3.set_xlabel("根据每个亚组AKI数量的 Top-K%")
# axesSub3.set_ylabel("召回率(TPR)")
# axesSub3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# axesSub3.legend(loc='upper left')

# save fig
# plt.savefig('lab_01_result_v2.jpg', dpi=1500, bbox_inches='tight')
plt.show()
