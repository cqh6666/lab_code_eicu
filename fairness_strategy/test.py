# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test
   Description:   ...
   Author:        cqh
   date:          2023/4/7 16:47
-------------------------------------------------
   Change Activity:
                  2023/4/7:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd


class Constant:
    label_column = "label"
    score_column = "score"

def find_next_black_threshold(black_data_df, black_threshold):
    """
    找到下一个阈值 阈值下降
    未被选中的最高风险的AKI黑人对应的分数
    :param black_data_df: 黑人数据集（已经按预测概率降序排序）
    :param black_threshold: 黑人风险阈值 已经排好序
    :return:
        thresholds 新的阈值
        add_nums 黑人阈值下降后 新增的风险人数
    """
    black_aki_df = black_data_df[black_data_df[Constant.label_column] == 1]

    before_index = -1
    new_black_threshold = black_threshold  # 新的阈值
    add_nums = 0  # 受影响人数

    for cur_index in black_aki_df.index.to_list():
        cur_score = black_aki_df.loc[cur_index, Constant.score_column]
        if cur_score < black_threshold:
            new_black_threshold = cur_score
            add_nums = cur_index - before_index
            break
        before_index = cur_index

    return new_black_threshold, add_nums


if __name__ == '__main__':
    black_df = pd.DataFrame({
        "score": [0.9,0.7,0.5,0.3,0.1],
        "label": [1,0,0,0,0,0,0,1,1]
    })

    threshold = 0.7

    print(find_next_black_threshold(black_df, threshold))