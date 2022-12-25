# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     getSubGroup
   Description:   ...
   Author:        cqh
   date:          2022/12/23 20:39
-------------------------------------------------
   Change Activity:
                  2022/12/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
from my_logger import logger


class SubGroupNode:
    def __init__(self, parent_index, thresholds, split_loss, recalls, depth, data_index, pre_split_features_dict: dict):
        self.parent_index = parent_index
        self.pre_split_features_dict = pre_split_features_dict
        self.data_index = data_index
        self.depth = depth
        self.recalls = recalls
        self.split_loss = split_loss
        self.thresholds = thresholds

        self.split_feature_name = None
        self.split_feature_value = None
        self.parent_thresholds = None
        self.index = -1
        self.is_leaf = False
        self.sub_nodes = []

    def set_inner_node(self, index, feature_name, feature_value):
        self.index = index
        self.is_leaf = False
        self.split_feature_name = feature_name
        self.split_feature_value = feature_value

    def set_leaf_node(self, index):
        self.index = index
        self.is_leaf = True
        self.sub_nodes = None
        self.split_feature_name = None
        self.split_feature_name = None

    def get_data_df(self):
        """
        根据数据分割索引得到数据df
        :return:
        """


class SplitLoss:
    def __init__(self):
        pass

    def cal_loss(self):
        return 0


class SubGroupDecisionTree:
    def __init__(self, data_df: pd.DataFrame, max_depth, init_threshold, select_features, splitLoss: SplitLoss):
        self.data_df = data_df
        self.init_threshold = init_threshold
        self.select_features = select_features
        self.max_depth = max_depth
        self.splitLoss = splitLoss

        self.node_count = 0
        self.node_tree = {}

    def build_tree(self, node: SubGroupNode, parent_index):
        node.parent_index = parent_index

        # 1. 获取当前节点数据 和 备选特征集（用来做亚组分割）
        cur_data_df = self.get_cur_data_df(node.pre_split_features_dict)
        remain_features = self.get_remain_features(list(node.pre_split_features_dict.keys()))

        # 2.1 若 无备选特征集，则把当前node设置为叶子节点，并添加到对应的决策树之中, return
        if len(remain_features) == 0:
            node.set_leaf_node(self.__get_node_index())
            self.node_tree[node.index] = node
            return node

        # 2.2 若 当前节点本来就为叶子节点，则添加到对应决策树之中，return
        if node.is_leaf:
            self.node_tree[node.index] = node
            return node

        # 3. 计算每一个备选特征集的loss值并得到最小loss的特征，此特征用来做亚组分割，并得到新分割节点列表
        best_split_feature, node_list = self.select_best_split_feature(node, cur_data_df, remain_features)
        node.sub_nodes = node_list
        self.node_tree[node.index] = node

        # 4. 将当前节点设置为内节点，并递归调用当前函数
        for sub_node in node_list:
            self.build_tree(sub_node, node.index)
        # 5. 返回根节点
        return node

    def __get_node_index(self):
        self.node_count += 1
        return self.node_count

    def show_node_info(self):
        pass

    def show_tree_info(self):
        pass

    def get_cur_data_df(self, pre_split_features_dict: dict):
        """
        根据之前当前节点之前分割过的特征，获取子数据集
        :param pre_split_features_dict:
        :return:
        """
        cur_data_df = self.data_df

        for feature_name, feature_value in pre_split_features_dict.items():
            query_str = f'{feature_name} == "{feature_value}"'
            cur_data_df = cur_data_df.query(query_str)

        return cur_data_df

    def get_remain_features(self, used_features: list):
        """
        排除使用过的特征
        :param used_features:
        :return:
        """
        all_features = self.select_features
        return list(set(all_features).difference(set(used_features)))

    def select_best_split_feature(self, node: SubGroupNode, cur_data_df: pd.DataFrame, remain_features: list):
        """
        获取最佳分割点
        :param node:
        :param cur_data_df:
        :param remain_features:
        :return:
        """
        # 1. 获得当前节点的阈值作为全局阈值 ( 黑人, 白人 )
        parent_thresholds = node.thresholds

        # 2. 循环对备选特征进行亚组分割，每个特征都会会分出子节点
        all_loss = []
        for feature in remain_features:
            # 2.1 接下来对每个子节点进行计算
            select_feature_df = cur_data_df[feature]
            select_feature_df.value_counts()

        # 2.1.1 首先在全局阈值下计算 黑人召回率和白人召回率

        # 2.1.2 若 黑人召回率 >= 白人召回率，则认为是公平的，不再继续分亚组。也就是把子节点初始化并赋为叶子节点，并且loss=0

        # 2.1.3 若 黑人召回率 < 白人召回率， 则需要进一步以最小损失进行分亚组。也就是需要对子节点进行修改阈值使得符合2.2。
        # 2.1.3 得到最佳阈值后，把子节点进行初始化并赋值为内节点，设置相关参数并计算loss值

        # 2.2 将每个子节点的loss值进行相加

        # 3. 每个备选特征的总loss值得到后，选出最小的loss值作为最优分割点。
        best_split_feature = ""
        best_node_list = []
        # 4. 返回结果（分割特征
        return best_split_feature, best_node_list

