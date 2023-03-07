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

import pickle
import sys

import pandas as pd
import numpy as np

from api_utils import save_to_csv_by_row
from get_failness_data import get_range_data
from get_eicu_dataset import get_subgroup_data
from my_logger import logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Constant:
    """
    eicu:
        score_column = “score_y”
        label_column = "Label"
        black_race_column = "race_black"
        white_race_column = "race_white"

    """
    score_column = "score_y"
    # label_column = "Label"
    # black_race_column = "Demo2_2"
    # white_race_column = "Demo2_1"

    # eicu
    label_column = "aki_label"
    # black_race_column = "race_African American"
    # white_race_column = "race_Caucasian"
    black_race_column = "race_Caucasian"
    white_race_column = "race_African American"

    black_race = 1
    white_race = 1


class SplitLoss:
    """
    Loss计算的基类
    """
    def __init__(self, name):
        self.name = name

    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        return 0


class RiskProbDiffLoss(SplitLoss):
    def __init__(self, name):
        super().__init__(name)

    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        """
        计算概率变化 原本被选样本的平均预测概率 与 新被选样本的平均预测概率 之差
        :param cur_data_df: 当前数据集
        :param old_thresholds: 黑人和白人的旧阈值
        :param new_thresholds: 黑人和白人的新阈值
        :return:
        """
        # 筛选AKI患者
        # cur_data_df = cur_data_df[cur_data_df[Constant.label_column] == 1]

        # 将数据集先分成黑人和白人，再降序排序
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race]
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race]

        black_score = black_data_df[Constant.score_column]
        white_score = white_data_df[Constant.score_column]

        old_black_prob_sum = black_score[black_score >= old_thresholds[0]].sum()
        old_white_prob_sum = white_score[white_score >= old_thresholds[1]].sum()
        new_black_prob_sum = black_score[black_score >= new_thresholds[0]].sum()
        new_white_prob_sum = white_score[white_score >= new_thresholds[1]].sum()

        old_sum = old_black_prob_sum + old_white_prob_sum
        new_sum = new_black_prob_sum + new_white_prob_sum

        return round(old_sum - new_sum, 4)

class RiskAkiDiffLoss(SplitLoss):

    def __init__(self, name):
        super().__init__(name)

    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        """
        计算 额外牺牲  原本选出的AKI人数和新选出的AKI人数之差
        :param cur_data_df: 当前数据集
        :param old_thresholds: 黑人和白人的旧阈值
        :param new_thresholds: 黑人和白人的新阈值
        :return:
        """
        # 筛选AKI患者
        cur_data_df = cur_data_df[cur_data_df[Constant.label_column] == 1]

        # 将数据集先分成黑人和白人
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race][
            [Constant.score_column, Constant.label_column]]
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race][
            [Constant.score_column, Constant.label_column]]

        old_black_aki_nums = black_data_df[black_data_df[Constant.score_column] >= old_thresholds[0]].shape[0]
        old_white_aki_nums = white_data_df[white_data_df[Constant.score_column] >= old_thresholds[1]].shape[0]
        new_black_aki_nums = black_data_df[black_data_df[Constant.score_column] >= new_thresholds[0]].shape[0]
        new_white_aki_nums = white_data_df[white_data_df[Constant.score_column] >= new_thresholds[1]].shape[0]

        old_aki_nums = old_black_aki_nums + old_white_aki_nums
        new_aki_nums = new_black_aki_nums + new_white_aki_nums

        return old_aki_nums - new_aki_nums


class RiskManDiffLoss(SplitLoss):

    def __init__(self, name):
        super().__init__(name)

    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        """
        计算换手率， 原本被选上的有风险人，经过新的阈值调整后，有多少人没有被选上。 指代 白种人的选取
        :param cur_data_df: 当前数据集
        :param old_thresholds: 黑人和白人的旧阈值
        :param new_thresholds: 黑人和白人的新阈值
        :return:
        """
        # 筛选AKI患者
        # cur_data_df = cur_data_df[cur_data_df[Constant.label_column] == 1]
        # 白种人
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race]

        old_white_threshold = old_thresholds[1]
        new_white_threshold = new_thresholds[1]

        score_values = white_data_df[Constant.score_column].values
        miss_count = 0
        for score in score_values:
            if old_white_threshold <= score < new_white_threshold:
                miss_count += 1

        return miss_count


class SubGroupNode:
    def __init__(self, thresholds, depth, data_index, pre_split_features_dict={}, split_loss=0,
                 recall_nums=None, threshold_nums=None, label_nums=None, recall_rate=None, split_feature_name=None, split_feature_value=None):
        """
        亚组节点
        :param thresholds: 当前节点的阈值 [黑人, 白人]
        :param split_loss: 分割损失计算
        :param depth: 深度（根节点为1）
        :param data_index: 当前节点的数据索引
        :param pre_split_features_dict: 当前节点由哪些特征分割而来（字典） {A:a} 代表 全局数据的特征A=a的来
        :param split_feature_name: 上一层分割而来的特征名（根节点为None）
        :param split_feature_value: 上一层分割而来的特征值
        """
        self.pre_split_features_dict = pre_split_features_dict
        self.data_index = data_index
        self.depth = depth
        self.split_loss = split_loss
        self.thresholds = thresholds
        self.split_feature_name = split_feature_name
        self.split_feature_value = split_feature_value

        # 黑人和白人的阈值召回人数和AKI人数
        self.threshold_nums = threshold_nums
        self.recall_nums = recall_nums
        self.label_nums = label_nums
        self.recall_rate = recall_rate

        self.all_nums = -1
        # 当前节点的ID(索引)
        self.index = -1
        # 父节点的ID(索引)
        self.parent_index = -1
        # 是否为叶子节点
        self.is_leaf = False
        # 子节点列表
        self.sub_nodes = []

    def set_inner_node(self):
        self.is_leaf = False

    def set_leaf_node(self):
        self.is_leaf = True
        self.sub_nodes = None

    def set_node_index(self, index):
        self.index = index

    def set_recall_info(self, recall_result_dict):
        self.threshold_nums = recall_result_dict['score_nums'][0], recall_result_dict['score_nums'][1],
        self.recall_nums = recall_result_dict['recall_nums'][0], recall_result_dict['recall_nums'][1]
        self.label_nums = recall_result_dict['label_nums'][0], recall_result_dict['label_nums'][1]
        self.recall_rate = recall_result_dict['recall_rate'][0], recall_result_dict['recall_rate'][1]
        self.all_nums = recall_result_dict['all_nums'][0], recall_result_dict['all_nums'][1]


class SubGroupDecisionTree:
    def __init__(self, data_df: pd.DataFrame, select_features, splitLoss: SplitLoss, global_thresholds):
        """
        保存节点的树
        :param data_df: 数据集
        :param select_features: 备选特征进行分亚组
        :param splitLoss: 不同计算损失器
        :param global_thresholds: 全局初始阈值
        """
        self.data_df = data_df
        self.select_features = select_features
        self.splitLoss = splitLoss
        self.global_thresholds = global_thresholds
        self.max_depth = 15

        # 根节点
        self.root_node = None
        # 根节点的loss（未做亚组划分的loss）
        self.root_loss = None
        self.split_plus_loss = 0

        self.node_count = -1
        self.node_tree = {}

    def build_tree(self, node: SubGroupNode, parent_index):

        # 设置树节点的索引和父节点的索引
        node.set_node_index(self.__get_node_index())
        node.parent_index = parent_index

        cur_data_df = self.get_cur_node_data_df(node)
        # 如果不公平，则不进行调整直接返回;如果公平，则调整后返回新结果;
        recall_result_dict, cur_thresholds, split_loss = self.do_adjust_thresholds(cur_data_df)

        # 如果是根节点
        if node.parent_index == -1:
            # 注入召回率阈值等相关信息
            node.set_recall_info(recall_result_dict)
            node.thresholds = cur_thresholds
            node.split_loss = split_loss
            self.root_loss = split_loss
            self.root_node = node

        # 2.1 若没有备选特征，黑白群体的AKI样本量较小，loss=0
        if self.stop_to_split_condition(node):
            node.set_leaf_node()
            self.node_tree[node.index] = node
            return

        node.set_inner_node()
        self.node_tree[node.index] = node

        # 3. 计算每一个备选特征集的loss值并得到最小loss的特征，此特征用来做亚组分割，并得到新分割节点列表
        node_list = self.select_best_split_feature(node)
        node.sub_nodes = node_list

        # 4. 将当前节点设置为内节点，并递归调用当前函数
        for sub_node in node_list:
            self.build_tree(sub_node, node.index)

    def __get_node_index(self):
        self.node_count += 1
        return self.node_count

    @DeprecationWarning
    def get_cur_data_df(self, pre_split_features_dict: dict):
        """
        根据之前当前节点之前分割过的特征，获取子数据集
        :param pre_split_features_dict:
        :return:
        """
        cur_data_df = self.data_df

        for feature_name, feature_value in pre_split_features_dict.items():
            cur_data_df = cur_data_df[cur_data_df[feature_name] == feature_value]

        return cur_data_df

    def get_cur_node_data_df(self, node: SubGroupNode):
        """
        根据当前node节点获取当前数据
        :param node:
        :return:
        """
        return self.data_df.loc[node.data_index, :]

    def get_remain_features(self, used_features: list):
        """
        排除使用过的特征
        :param used_features:
        :return:
        """
        all_features = self.select_features
        return list(set(all_features).difference(set(used_features)))

    def get_cur_node_remain_features(self, node: SubGroupNode):
        """
        根据当前节点的信息获取备选特征
        :param node:
        :return:
        """
        used_features = list(node.pre_split_features_dict.keys())
        return list(set(self.select_features).difference(set(used_features)))

    def select_best_split_feature(self, node: SubGroupNode):
        """
        获取最佳分割点
        :param node:
        :return:
        """
        cur_data_df = self.get_cur_node_data_df(node)

        # ============================================ 找到最佳划分点 ===========================================
        # 1. 获取当前分割的信息
        remain_features = self.get_cur_node_remain_features(node)
        pre_split_feature_dict = node.pre_split_features_dict

        # 2. 循环对备选特征进行亚组分割，每个特征都会会分出子节点
        all_loss_dict = {}
        all_node_dict = {}

        # ======================== 多线程计算 ================================
        # thread_list = []
        # for feature in remain_features:
        #     thread_list.append(executor.submit(
        #         self.buildFeatureNode, cur_data_df, node, feature
        #     ))
        # wait(thread_list, return_when=ALL_COMPLETED)
        #
        # for feature, thread in zip(remain_features, thread_list):
        #     # 添加到字典之中
        #     all_loss_dict[feature], all_node_dict[feature] = thread.result()

        # ======================== 单线程计算 ================================
        for feature in remain_features:
            # 2.1 接下来对每个子节点进行计算
            feature_unique_list = cur_data_df[feature].value_counts().index.sort_values(ascending=True).to_list()
            # 记录子节点loss总和
            cur_loss = 0
            cur_node_list = []
            cur_depth = node.depth + 1

            for feature_value in feature_unique_list:  # 一般是 0, 1
                # 获取数据
                temp_data_df = cur_data_df[cur_data_df[feature] == feature_value]
                # 添加新的分割点到分割字典之中
                temp_pre_split_feature_dict = pre_split_feature_dict.copy()
                temp_pre_split_feature_dict[feature] = feature_value

                # 在全局阈值下计算 黑人召回率和白人召回率
                recall_result_dict = cal_black_white_recall(temp_data_df, self.global_thresholds)
                black_recall, white_recall = recall_result_dict['recall_rate']
                if black_recall >= white_recall:
                    # 2.1.2 若 黑人召回率 >= 白人召回率，则认为是公平的，不再继续分亚组。也就是把子节点初始化并赋为叶子节点，并且loss=0
                    # 初始化为叶子节点，并且设置loss=0
                    temp_node = SubGroupNode(
                        thresholds=self.global_thresholds,
                        split_loss=0,
                        data_index=temp_data_df.index.to_list(),
                        pre_split_features_dict=temp_pre_split_feature_dict,
                        depth=cur_depth,
                        split_feature_name=feature,
                        split_feature_value=feature_value,
                    )
                    temp_node.set_recall_info(recall_result_dict)
                    # logger.info(f"{node.depth}-[parent_index:{node.index}]-[{feature}:{feature_value}]: 黑人召回率({black_recall})高于(或等于)白人召回率({white_recall}), 不再继续分亚组...")
                else:
                    # 2.1.3 若 黑人召回率 < 白人召回率， 则需要进一步以最小损失进行分亚组。也就是需要对子节点进行修改阈值使得符合2.2。
                    # logger.warning(f"{node.depth}-[parent_index:{node.index}]-[{feature}:{feature_value}]: 黑人召回率({black_recall})低于白人召回率({white_recall}), 还得继续分亚组...")
                    best_recall_result_dict, best_thresholds, split_loss = self.do_adjust_thresholds(temp_data_df)

                    # 2.1.4 得到最佳阈值后，把子节点进行初始化，设置相关参数并计算loss值
                    temp_node = SubGroupNode(
                        thresholds=best_thresholds,
                        split_loss=split_loss,
                        data_index=temp_data_df.index.to_list(),
                        pre_split_features_dict=temp_pre_split_feature_dict,
                        depth=cur_depth,
                        split_feature_name=feature,
                        split_feature_value=feature_value,
                    )
                    temp_node.set_recall_info(best_recall_result_dict)

                # 将当前节点保存到节点列表之中
                cur_node_list.append(temp_node)

            # 2.2 将每个子节点的loss值进行相加
            for cur_node in cur_node_list:
                cur_loss += cur_node.split_loss

            # 添加到字典之中
            all_loss_dict[feature] = cur_loss
            all_node_dict[feature] = cur_node_list

        # 3. 每个备选特征的总loss值得到后，选出最小的loss值作为最优分割点。
        # 若 两个子节点的loss之和还比父节点的loss还要高，则不进行分割
        best_split_feature = get_best_split_feature(all_loss_dict, node)
        if best_split_feature is not None:
            return all_node_dict[best_split_feature]
        else:
            return []

    def buildFeatureNode(self, cur_data_df, node, feature):
        """
        建立特征节点
        :param node:
        :param cur_data_df:
        :param feature:
        :return:
        """
        # start_time = time.time()
        pre_split_feature_dict = node.pre_split_features_dict

        # 2.1 接下来对每个子节点进行计算
        feature_unique_list = cur_data_df[feature].value_counts().index.sort_values(ascending=True).to_list()
        # 记录子节点loss总和
        cur_loss = 0
        cur_node_list = []
        cur_depth = node.depth + 1

        for feature_value in feature_unique_list:  # 一般是 0, 1
            # 获取数据
            temp_data_df = cur_data_df[cur_data_df[feature] == feature_value]
            # 添加新的分割点到分割字典之中
            temp_pre_split_feature_dict = pre_split_feature_dict.copy()
            temp_pre_split_feature_dict[feature] = feature_value

            # 在全局阈值下计算 黑人召回率和白人召回率
            recall_result_dict = cal_black_white_recall(temp_data_df, self.global_thresholds)
            black_recall, white_recall = recall_result_dict['recall_rate']
            if black_recall >= white_recall:
                # 2.1.2 若 黑人召回率 >= 白人召回率，则认为是公平的，不再继续分亚组。也就是把子节点初始化并赋为叶子节点，并且loss=0
                # 初始化为叶子节点，并且设置loss=0
                temp_node = SubGroupNode(
                    thresholds=self.global_thresholds,
                    split_loss=0,
                    data_index=temp_data_df.index.to_list(),
                    pre_split_features_dict=temp_pre_split_feature_dict,
                    depth=cur_depth,
                    split_feature_name=feature,
                    split_feature_value=feature_value,
                )
                temp_node.set_recall_info(recall_result_dict)
                # logger.info(f"{node.depth}-[parent_index:{node.index}]-[{feature}:{feature_value}]: 黑人召回率({black_recall})高于(或等于)白人召回率({white_recall}), 不再继续分亚组...")
            else:
                # 2.1.3 若 黑人召回率 < 白人召回率， 则需要进一步以最小损失进行分亚组。也就是需要对子节点进行修改阈值使得符合2.2。
                # logger.warning(f"{node.depth}-[parent_index:{node.index}]-[{feature}:{feature_value}]: 黑人召回率({black_recall})低于白人召回率({white_recall}), 还得继续分亚组...")
                best_recall_result_dict, best_thresholds, split_loss = self.do_adjust_thresholds(temp_data_df)

                # 2.1.4 得到最佳阈值后，把子节点进行初始化，设置相关参数并计算loss值
                temp_node = SubGroupNode(
                    thresholds=best_thresholds,
                    split_loss=split_loss,
                    data_index=temp_data_df.index.to_list(),
                    pre_split_features_dict=temp_pre_split_feature_dict,
                    depth=cur_depth,
                    split_feature_name=feature,
                    split_feature_value=feature_value,
                )
                temp_node.set_recall_info(best_recall_result_dict)

            # 将当前节点保存到节点列表之中
            cur_node_list.append(temp_node)

        # 2.2 将每个子节点的loss值进行相加
        for cur_node in cur_node_list:
            cur_loss += cur_node.split_loss

        # logger.info(f"{threading.currentThread().getName()}- id:{node.index}- feature:{feature}- loss:{cur_loss} [{start_time}]")
        return cur_loss, cur_node_list

    def do_adjust_thresholds(self, cur_data_df):
        """
        调整阈值使得 黑人召回率 高于 白人召回率
        调整策略：
            若白人召回率高于黑人召回率，则以 黑人阈值bt1 之下的第一个最高预测分数的黑人为 新的阈值 bt2，并计算得知新的阈值下 黑人新增了 p 人；
            而此时白人也要计算出 高于 白人阈值 wt1 p人后的新的阈值 wt2。
            若这么调整后 还是 白人高于黑人， 则黑人需要 继续上一步操作。
            最后将bt2和wt2作为新的阈值
        :param node:
        :param cur_data_df:
        :return:
        """
        black_threshold, white_threshold = self.global_thresholds

        recall_result_dict = cal_black_white_recall(cur_data_df, self.global_thresholds)
        black_recall, white_recall = recall_result_dict['recall_rate']

        # 如果黑人召回率高于白人召回率，则直接返回
        if black_recall >= white_recall:
            return recall_result_dict, [black_threshold, white_threshold], 0

        # 将数据集先分成黑人和白人，再降序排序
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race].sort_values(by=Constant.score_column, ascending=False)
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race].sort_values(by=Constant.score_column, ascending=False)

        while black_recall < white_recall:
            black_threshold, black_add_nums = find_next_black_threshold(black_data_df, black_threshold)
            white_threshold = find_next_white_threshold(white_data_df, white_threshold, black_add_nums)
            recall_result_dict = cal_black_white_recall(cur_data_df, [black_threshold, white_threshold])
            black_recall, white_recall = recall_result_dict['recall_rate']

        new_thresholds = [black_threshold, white_threshold]
        # 计算损失值
        split_loss = self.splitLoss.cal_loss(cur_data_df, self.global_thresholds, new_thresholds)

        return recall_result_dict, new_thresholds, split_loss

    def get_tree_info(self):
        logger.warning("==========================================================")
        logger.info(f'risk_rate = {risk_rate}')
        logger.info(f'global_thresholds = {self.global_thresholds}')
        logger.info(f'root node thresholds = {self.root_node.thresholds}')
        logger.info(f'root node recall_rate = {self.root_node.recall_rate}')
        logger.info(f'node_count = {self.node_count + 1}')
        logger.info(f'root_loss = {self.root_loss}')
        logger.warning("==========================================================")

    def convert_format_to_dot(self):
        """
        将我当前的类的树模型转化为可以输入的模型，最终得到 dot 文件， 可以输出png格式的图片
        :return:
        """
        dot_begin_str = """
        digraph Tree {
        node [shape=box, style="filled", fontname="helvetica"] ;
        edge [fontname="helvetica"] ; 
        """
        dot_explain_str = """
        -1 [shape=tab, fillcolor="pink", label="{}"];
        """
        dot_edge_str = """
        {} [label="{}", {}] ;
        """
        dot_link_str = """
        {} -> {} [labeldistance=2.5, headlabel="{}"] ; 
        """
        dot_end_str = "}"

        # edge 字符串添加
        max_depth = 8
        all_dot_edge_str = ""
        all_dot_link_str = ""

        explain_str = f'risk_rate = {risk_rate} \\n' \
                      f'global_thresholds = {self.global_thresholds} \\n' \
                      f'node_count = {self.node_count + 1} \\n' \
                      f'root_loss = {self.root_loss} \\n' \
                      f'split_plus_loss = {self.get_split_plus_loss()}'

        all_dot_explain_str = dot_explain_str.format(explain_str)

        for node_index, node in self.node_tree.items():
            if node.depth > max_depth:
                continue

            dot_info_str = f'LOSS({self.splitLoss.name}) = {node.split_loss} \\n' \
                           f'all_nums = {node.all_nums} \\n' \
                           f'thresholds = {node.thresholds} \\n' \
                           f'threshold_nums = {node.threshold_nums} \\n' \
                           f'recall_nums = {node.recall_nums} \\n' \
                           f'label_nums = {node.label_nums} \\n' \
                           f'recall_rate = {node.recall_rate} \\n' \
                           f'{node.split_feature_name} = {node.split_feature_value} \\n'

            if node.split_feature_value == 1:
                all_dot_edge_str += dot_edge_str.format(node_index, dot_info_str, 'fillcolor="#e6843d"')
            else:
                all_dot_edge_str += dot_edge_str.format(node_index, dot_info_str, '')

            if not node.is_leaf:
                if node.sub_nodes is not None:
                    temp_nodes = node.sub_nodes
                    for tmp in temp_nodes:
                        all_dot_link_str += dot_link_str.format(node_index, tmp.index, tmp.split_feature_value)

        all_dot_str = dot_begin_str + all_dot_explain_str + all_dot_edge_str + all_dot_link_str + dot_end_str

        with open(f"/home/chenqinhai/code_eicu/my_lab/fairness_strategy/tree_{self.splitLoss.name}.dot", "w") as f:
            f.write(all_dot_str)
            logger.info("save as dot success!")

    def stop_to_split_condition(self, node, lower_threshold=5):
        """
        :param lower_threshold:
        :param node:
        :return:
        """
        # 如果树的深度过大，提前终止
        if node.depth > self.max_depth:
            return True

        # 损失为0，提前终止
        if node.split_loss == 0:
            return True

        # 没有备选特征 则停止分裂
        if len(self.get_cur_node_remain_features(node)) == 0:
            return True

        cur_data_df = self.get_cur_node_data_df(node)

        # 将数据集先分成黑人和白人，再降序排序
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race].sort_values(by=Constant.score_column, ascending=False)
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race].sort_values(by=Constant.score_column, ascending=False)

        # 获取黑人和白人AKI为1的部分
        black_label_true = (black_data_df[Constant.label_column] == 1)
        white_label_true = (white_data_df[Constant.label_column] == 1)
        # 获取黑人和白人AKI=1的人数
        black_label_nums, white_label_nums = np.sum(black_label_true), np.sum(white_label_true)

        # 如果黑人或白人AKI人数<= 一定阈值
        if black_label_nums <= lower_threshold or white_label_nums <= lower_threshold:
            return True

        return False

    def get_split_plus_loss(self):
        """
        计算得到最终的分割Loss大小
        """
        for node_index, node in self.node_tree.items():
            if node.is_leaf:
                self.split_plus_loss += node.split_loss

        return self.split_plus_loss

    def predict(self, test_data_df: pd.DataFrame):
        """
        测试集数据，放入叶子节点中，计算出最终的loss值
        :param test_data_df:
        :return:
        """
        test_data_index = test_data_df.index.to_list()
        # 保存结果映射
        leaf_index_df = pd.DataFrame(index=test_data_index, columns=['leaf_index'])

        for index in test_data_index:
            temp_data = test_data_df.loc[index, :]
            # 根据当前树结构找到对应的叶子节点的ID
            leaf_index_df.loc[index, 'leaf_index'] = self.get_leaf_index(temp_data)

        # 计算根节点的LOSS
        root_loss = self.splitLoss.cal_loss(test_data_df, self.global_thresholds, self.root_node.thresholds)
        # 计算每个叶子节点的LOSS
        leaf_node_list = self.get_leaf_node()

        split_all_loss = 0
        for leaf_node in leaf_node_list:
            cur_index = leaf_node.index
            cur_subgroup_data_df = test_data_df.loc[leaf_index_df['leaf_index'] == cur_index, :]
            split_all_loss += self.splitLoss.cal_loss(cur_subgroup_data_df, self.global_thresholds, leaf_node.thresholds)

        return root_loss, split_all_loss

    def get_leaf_node(self):
        """
        获取叶子节点列表
        :return:
        """
        leaf_list = []
        for node_index, node in self.node_tree.items():
            if node.is_leaf:
                leaf_list.append(node)

        return leaf_list

    def get_leaf_index(self, temp_data):
        """
        当前测试节点，找到对应的叶子节点ID
        :param temp_data: pd.Series
        :return:
        """
        cur_node = self.root_node

        while not cur_node.is_leaf:
            cur_child_nodes = cur_node.sub_nodes
            is_match = False
            for cur_child in cur_child_nodes:
                feature_name = cur_child.split_feature_name
                feature_value = cur_child.split_feature_value
                if temp_data[feature_name] == feature_value:
                    cur_node = cur_child
                    is_match = True
                    break

            if not is_match:
                raise ValueError(f"index: {cur_node.index} - 匹配特征失败，中止匹配!")

        return cur_node.index


def cal_black_white_recall(cur_data_df: pd.DataFrame, thresholds: list):
    """
    根据数据集和阈值计算 黑人和白人 的风险人数
    :param cur_data_df: 数据集
    :param thresholds: 黑人和白人的阈值
    :return:
        black_threshold_nums, white_threshold_nums 黑人和白人超过阈值的人数
        black_recall_nums, white_recall_nums 黑人和白人召回AKI患者人数
        black_label_nums, white_label_nums 黑人和白人总AKI患者人数
        black_recall_rate, white_recall_rate 黑人和白人召回率
    """
    result_dict = {}

    black_threshold, white_threshold = thresholds[0], thresholds[1]

    # 获取黑人和白人各自的数据集
    black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race]
    white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race]

    # 获取黑人和白人高于阈值的部分
    black_score_true = (black_data_df[Constant.score_column] >= black_threshold)
    white_score_true = (white_data_df[Constant.score_column] >= white_threshold)

    # 获取黑人和白人AKI为1的部分
    black_label_true = (black_data_df[Constant.label_column] == 1)
    white_label_true = (white_data_df[Constant.label_column] == 1)

    # 两部分进行与运算，获取交集
    black_final_true = black_score_true & black_label_true
    white_final_true = white_score_true & white_label_true

    # 获取黑人和白人AKI=1的人数
    black_label_nums, white_label_nums = np.sum(black_label_true), np.sum(white_label_true)
    # 统计AKI召回人数
    black_recall_nums, white_recall_nums = np.sum(black_final_true), np.sum(white_final_true)

    # 统计召回率
    if black_label_nums == 0 or white_label_nums == 0:
        black_recall_rate, white_recall_rate = 0, 0
    else:
        black_recall_rate = round(black_recall_nums / black_label_nums, 4)
        white_recall_rate = round(white_recall_nums / white_label_nums, 4)

    # 保存结果
    result_dict['all_nums'] = black_data_df.shape[0], white_data_df.shape[0]
    result_dict['score_nums'] = np.sum(black_score_true), np.sum(white_score_true)  # 获取黑人和白人超过阈值的人数
    result_dict['label_nums'] = black_label_nums, white_label_nums
    result_dict['recall_nums'] = black_recall_nums, white_recall_nums
    result_dict['recall_rate'] = black_recall_rate, white_recall_rate

    return result_dict


def find_next_black_threshold(black_data_df, black_threshold):
    """
    找到下一个黑人阈值 阈值下降
    未被选中的最高风险的AKI黑人对应的分数
    :param black_data_df: 黑人数据集（已经按预测概率降序排序）
    :param black_threshold: 黑人风险阈值 已经排好序
    :return:
        thresholds 新的阈值
        add_nums 黑人阈值下降后 新增的风险人数
    """
    black_score = black_data_df[Constant.score_column].values
    len_black_score = len(black_score)

    threshold = black_threshold
    add_nums = 0

    for index, score in enumerate(black_score):
        # 找到第一个比原来阈值小的新阈值
        if score < black_threshold:
            threshold = score
            # 如果有重复的分数，统计数目
            while index < len_black_score:
                if black_score[index] == threshold:
                    index += 1
                    add_nums += 1
                else:
                    break
            break

    if threshold == black_threshold:
        logger.error(f"黑人阈值无法继续下降了...")
        raise ValueError(f"当前黑人阈值已经是最低值了...")

    # logger.info(f"黑人阈值下降! [{black_threshold}->{threshold}], 新增了{add_nums}人...")
    return threshold, add_nums


def find_next_white_threshold(white_data_df, white_threshold, black_add_nums):
    """
    根据黑人增加数目找到下一个白人阈值 阈值上升
    :param white_data_df:
    :param white_threshold:
    :param black_add_nums:
    :return:
    """
    white_score = white_data_df[Constant.score_column].values

    # 原来阈值的索引位置
    cur_index = 0
    for index, score in enumerate(white_score):
        if score < white_threshold:
            cur_index = index - 1
            break

    # 新索引位置
    new_index = cur_index - black_add_nums

    # 如果新索引的预测概率和原来的预测概率阈值相等 则需要进一步上升阈值
    while new_index >= 0 and white_score[new_index] == white_threshold:
        new_index -= 1

    if new_index < 0:
        # logger.warning("找不到使得白人阈值上升对应匹配的人数, 只能取最大值...")
        return white_score[0] + 0.5
    else:
        # logger.info(f"白人阈值上升! [{white_threshold}->{threshold}]...")
        return white_score[new_index]


def get_best_split_feature(cur_loss_dict: dict, cur_node):
    """
    根据loss字典获得最小损失的特征
    :param cur_node:
    :param cur_loss_dict:
    :return:
    """
    parent_split_loss = cur_node.split_loss
    min_loss = np.iinfo(np.int32).max
    min_feature = None
    for feature, loss in cur_loss_dict.items():
        if loss < min_loss:
            min_loss = loss
            min_feature = feature

    if min_loss >= parent_split_loss:
        cur_node.set_leaf_node()
        logger.warning(f"{cur_node.depth}-[index:{cur_node.index}]-当前节点分亚组后得到的损失之和（{min_loss}）大于等于父节点，不再继续分亚组...")
        return None

    logger.warning(f"{cur_node.depth}-[index:{cur_node.index}]-选出 {min_feature} 作为下一个分割特征...")
    return min_feature


def load_my_data(data_file):
    """
    读取数据
    :return:
    """
    return pd.read_feather(data_file)


def get_global_thresholds(data_df: pd.DataFrame, rate):
    """
    根据风险比例划分出一个公共的全局阈值
    :param data_df:
    :param rate:
    :return: [t, t]
    """
    # 降序
    score_array = data_df[Constant.score_column].sort_values(ascending=True).values
    # 计算比例人数
    risk_index = int(data_df.shape[0] * rate) - 1
    assert not risk_index <= 0, "当前得到的索引不能为负..."
    score_threshold = score_array[risk_index]

    return [score_threshold, score_threshold]


def get_black_white_global_thresholds(data_df: pd.DataFrame, rate):
    """
    根据风险比例划分出一个公共的黑人和白人全局阈值
    :param data_df:
    :param rate:
    :return:
    """
    # 将数据集先分成黑人和白人，再降序排序
    black_data_df = data_df[data_df[Constant.black_race_column] == Constant.black_race].sort_values(
        by=Constant.score_column, ascending=False)
    white_data_df = data_df[data_df[Constant.white_race_column] == Constant.white_race].sort_values(
        by=Constant.score_column, ascending=False)

    black_threshold = get_global_thresholds(black_data_df, rate)
    white_threshold = get_global_thresholds(white_data_df, rate)

    return [black_threshold[0], white_threshold[0]]



def cross_predict():
    # train_data = load_my_data(data_file_name.format("train"))
    # test_data = load_my_data(data_file_name.format("test"))

    all_range_list = [1, 2, 3, 4, 5]
    all_init_loss = []
    all_split_loss = []

    for index in all_range_list:
        if index != cross_index:
            continue
        train_index = [temp for temp in all_range_list if temp != index]
        test_index = [index]

        # train_data = get_range_data(train_index)
        # test_data = get_range_data(test_index)
        train_data = get_subgroup_data(train_index)
        test_data = get_subgroup_data(test_index)
        logger.warning(f"get_data ===== train:{train_data.shape}, test:{test_data.shape}")

        init_global_thresholds = get_global_thresholds(train_data, risk_rate)
        root_node = SubGroupNode(
            thresholds=init_global_thresholds,
            data_index=train_data.index.to_list(),
            depth=1,
        )

        subgroup_select_features = my_cols

        sbdt = SubGroupDecisionTree(
            data_df=train_data,
            global_thresholds=init_global_thresholds,
            select_features=subgroup_select_features,
            splitLoss=myLoss,
        )

        logger.warning(f"start build tree!")
        # 关键入口
        sbdt.build_tree(root_node, -1)
        # 保存pickle
        try:
            tree_str = pickle.dumps(sbdt)
            with open(pickle_file_name, "wb") as file:
                file.write(tree_str)
            logger.info(f"save pickle success!, memory: {round(sys.getsizeof(tree_str) / 1024 / 1024, 2)} MB")
        except:
            logger.error("save pickle error!")

        sbdt.get_tree_info()
        # sbdt.convert_format_to_dot()
        logger.warning("start predict loss!")
        init_loss, split_loss = sbdt.predict(test_data)

        all_init_loss.append(init_loss)
        all_split_loss.append(split_loss)

        logger.info(f"{init_loss},{split_loss} done!")

    save_result_file = f"/home/chenqinhai/code_eicu/my_lab/fairness_strategy/result/{myLoss.name}_测试集5折交叉结果_v{version}.csv"
    save_df = pd.DataFrame(index=[risk_rate], data={
        "test_valid_index": cross_index,
        "init_loss": np.sum(all_init_loss),
        "split_loss": np.sum(all_split_loss)
    })
    save_to_csv_by_row(save_result_file, save_df)
    logger.warning(f"save success! - {save_result_file}")


if __name__ == '__main__':
    # risk_rate = float(sys.argv[1])
    # split_type = int(sys.argv[2])
    # # 五折交叉分批执行
    # cross_index = int(sys.argv[3])
    # pool_nums = 10
    risk_rate = 0.75
    split_type = 1
    cross_index = 2
    """
    version = 1  drg top 20
    version = 2  all drg, ccs, demo3
    version = 3  all drg, ccs, demo3   概率变化修复bug
    version = 4  all drg, ccs, demo3   三个损失计算的修正
    version = 5  all drg, ccs, demo3   多线程计算，损失计算的修正v2
    version = 6  eicu data
    version = 7  eicu data  分批执行
    version = 8   eciu cross3
    version = 9  server7 数据
    version = 10  eicu 种族反过来
    """
    version = 10
    # drg_cols = "/home/liukang/Doc/disease_top_20.csv"
    # drg_list = pd.read_csv(drg_cols).squeeze().to_list()
    # my_cols = pd.read_csv("ku_data_select_cols.csv", index_col=0).squeeze().to_list()
    # data_file_name = "/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/{}_data.feather"

    # eicu
    my_cols = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/eicu_data/subgroup_select_feature.csv", index_col=0).squeeze().to_list()
    data_file_name = "/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/eicu_data/test_valid_{}.feather"
    pickle_file_name = f"/home/chenqinhai/code_eicu/my_lab/fairness_strategy/result/sbdt_pickle_rate{risk_rate}_type{split_type}_cross{cross_index}_v{version}.pkl"
    loss_dict = {
        1: RiskAkiDiffLoss(name="额外牺牲"),
        2: RiskProbDiffLoss(name="概率变化"),
        3: RiskManDiffLoss(name=f"换手率")
    }

    # 线程池
    # executor = ThreadPoolExecutor(max_workers=pool_nums)

    myLoss = loss_dict[split_type]

    cross_predict()

    # 读取pickle文件
    # with open(pickle_file_name, "rb") as f:
    #     obj = pickle.loads(f.read())
