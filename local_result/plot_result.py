# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     plot_result
   Description:   ...
   Author:        cqh
   date:          2022/10/27 10:33
-------------------------------------------------
   Change Activity:
                  2022/10/27:
-------------------------------------------------
"""
__author__ = 'cqh'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv_file(version_list):
    """
    读取不同count csv文件
    :param version_list:
    :return:
    """
    count_csv_1 = pd.DataFrame()
    count_csv_0 = pd.DataFrame()

    # 读取csv文件
    for cur_version in version_list:
        save_file = os.path.join(save_path, f'S04_count_percent_v{cur_version}.csv')
        cur_csv = pd.read_csv(save_file, index_col=0)
        count_csv_1 = pd.concat([count_csv_1, cur_csv[f'v{cur_version}_1']], axis=1)
        count_csv_0 = pd.concat([count_csv_0, cur_csv[f'v{cur_version}_0']], axis=1)

    # 返回 0 1
    return count_csv_1, count_csv_0


def plot_point():
    """
    绘制点
    :return:
    """
    # version  10-12
    # version  11-14
    # version  13-15
    version_dict = {
        "当前中心使用不同相似度量匹配当前中心":[10, 12],
        "当前中心使用不同相似度量匹配全局": [11, 14],
        "当前中心使用不同相似度量匹配全局（等样本量）": [13, 15],
        "使用不同相似度量匹配当前中心或全局对比(相同样本下）": [10, 12, 13, 15]
    }

    for title, cur_ver in version_dict.items():
        df_1, df_0 = load_csv_file(cur_ver)

        df_1.plot(y=df_1.columns, marker='o', title=title + '(正样本)')
        plt.xlabel("匹配样本前百分比")
        plt.ylabel("匹配正样本数目")
        png_file = os.path.join(save_path, f"S04_comp_{title}_正.png")
        plt.savefig(png_file)
        print("save result to png success!", png_file)

        df_0.plot(y=df_0.columns, marker='o', title=title + '(负样本)')
        plt.xlabel("匹配样本前百分比")
        plt.ylabel("匹配负样本数目")

        png_file = os.path.join(save_path, f"S04_comp_{title}_负.png")
        plt.savefig(png_file)
        print("save result to png success!", png_file)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    hos_id = 73
    save_path = f"./"
    plot_point()