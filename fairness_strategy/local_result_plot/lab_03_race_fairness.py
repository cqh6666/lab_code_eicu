# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lab_01_all_subgroup_fairness
   Description:

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
my_data_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result_plot/lab_03_result_plot.csv")

def process_df(data_df):
    """
    处理可以直接绘制直方图
    """
    build_types = ['GM', 'SM', 'PMTL']
    cur_result_df = pd.DataFrame(columns=['threshold', 'thres_type', 'build_type', 'race_type', 'score'])
    for idx in data_df.index.to_list():
        threshold = data_df.loc[idx, "thresholds"]
        race = data_df.loc[idx, "race"]
        thres_type = data_df.loc[idx, "thres_type"]

        for bt in build_types:
            cur_result_df.loc[f"{race}_{threshold}_{bt}", "threshold"] = threshold
            cur_result_df.loc[f"{race}_{threshold}_{bt}", "build_type"] = bt
            cur_result_df.loc[f"{race}_{threshold}_{bt}", "race_type"] = race
            cur_result_df.loc[f"{race}_{threshold}_{bt}", "score"] = data_df.loc[idx, bt]
            cur_result_df.loc[f"{race}_{threshold}_{bt}", "thres_type"] = thres_type

    return cur_result_df


my_data_df = process_df(my_data_df)
# Load an example dataset
# my_data_df2 = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/subgroup_top20_fairness_to_plot.csv", index_col=0)

fig, ax =plt.subplots(1,2,constrained_layout=True, figsize=(8, 4))
plt.subplots_adjust(left=0.125, bottom=0.225, wspace=0.45, hspace=0.45) #调整子图间距

# select_cond = (my_data_df["race_type"] == "black") & (my_data_df["thres_type"] == 1)
# plot_df = my_data_df[select_cond]
# axesSub = sns.barplot(data=plot_df, x="threshold", y="score", hue="build_type", ax=ax[0][0])
# axesSub.set_title("黑人群体患者的TPR比较")
# axesSub.set_xlabel("从全部样本中选取Top-K%")
# axesSub.set_ylabel("召回率(TPR)")
# axesSub.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# axesSub.legend(loc='upper left')

select_cond = (my_data_df["race_type"] == "black") & (my_data_df["thres_type"] == 2)
plot_df = my_data_df[select_cond]
axesSub3 = sns.barplot(data=plot_df, x="threshold", y="score", hue="build_type", ax=ax[0])
axesSub3.set_title("黑人群体患者的TPR比较")
axesSub3.set_xlabel("根据各入院原因亚组AKI数量选取Top-K%")
axesSub3.set_ylabel("召回率(TPR)")
axesSub3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub3.legend(loc='upper left')

# select_cond = (my_data_df["race_type"] == "white") & (my_data_df["thres_type"] == 1)
# plot_df = my_data_df[select_cond]
# axesSub2 = sns.barplot(data=plot_df, x="threshold", y="score", hue="build_type", ax=ax[1][0])
# axesSub2.set_title("白人群体患者的TPR比较")
# axesSub2.set_xlabel("从全部样本中选取Top-K%")
# axesSub2.set_ylabel("召回率(TPR)")
# axesSub2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# axesSub2.legend(loc='upper left')

select_cond = (my_data_df["race_type"] == "white") & (my_data_df["thres_type"] == 2)
plot_df = my_data_df[select_cond]
axesSub4 = sns.barplot(data=plot_df, x="threshold", y="score", hue="build_type", ax=ax[1])
axesSub4.set_title("白人群体患者的TPR比较")
axesSub4.set_xlabel("根据各入院原因亚组AKI数量选取Top-K%")
axesSub4.set_ylabel("召回率(TPR)")
axesSub4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axesSub4.legend(loc='upper left')

# save fig
plt.savefig('lab_03_result_v2.jpg', dpi=1500, bbox_inches='tight')
plt.show()
