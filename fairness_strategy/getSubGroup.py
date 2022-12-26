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
import numpy as np

from my_logger import logger


class SubGroupNode:
    def __init__(self, thresholds, split_loss, depth, data_index, pre_split_features_dict: dict,
                 split_feature_name=None, split_feature_value=None):
        self.pre_split_features_dict = pre_split_features_dict
        self.data_index = data_index
        self.depth = depth
        self.split_loss = split_loss
        self.thresholds = thresholds
        self.split_feature_name = split_feature_name
        self.split_feature_value = split_feature_value

        self.risk_nums = None
        self.index = -1
        self.parent_index = -1
        self.is_leaf = False
        self.sub_nodes = []

    def set_inner_node(self):
        self.is_leaf = False

    def set_leaf_node(self):
        self.is_leaf = True
        self.sub_nodes = None

    def set_node_index(self, index):
        self.index = index

    def get_data_df(self):
        """
        根据数据分割索引得到数据df
        :return:
        """


class SplitLoss:
    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        return 0


class RiskProbDiffLoss(SplitLoss):
    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        """
        计算概率变化 原本被选样本的平均预测概率 与 新被选样本的平均预测概率 之差
        :param cur_data_df:
        :param old_thresholds:
        :param new_thresholds:
        :return:
        """
        # 将数据集先分成黑人和白人，再降序排序
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race].sort_values(
            by=Constant.score_column, ascending=False)
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race].sort_values(
            by=Constant.score_column, ascending=False)

        black_score = black_data_df[Constant.score_column].values
        white_score = white_data_df[Constant.score_column].values

        old_prob, new_prob = [], []

        for b_score in black_score:
            if b_score > old_thresholds[0]:
                old_prob.append(b_score)
            if b_score > new_thresholds[0]:
                new_prob.append(b_score)

        for w_score in white_score:
            if w_score > old_thresholds[1]:
                old_prob.append(w_score)
            if w_score > new_thresholds[1]:
                new_prob.append(w_score)

        old_mean = np.mean(old_prob)
        new_mean = np.mean(new_prob)

        return np.abs(old_mean - new_mean)


class RiskAkiDiffLoss(SplitLoss):
    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        """
        计算 额外牺牲  原本选出的AKI人数和新选出的AKI人数之差
        :param cur_data_df:
        :param old_thresholds:
        :param new_thresholds:
        :return:
        """
        # 将数据集先分成黑人和白人，再降序排序
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race][[Constant.score_column, Constant.label_column]]
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race][[Constant.score_column, Constant.label_column]]

        old_black_aki_nums = black_data_df[black_data_df[Constant.score_column] > old_thresholds[0]][Constant.label_column].sum()
        old_white_aki_nums = white_data_df[white_data_df[Constant.score_column] > old_thresholds[1]][Constant.label_column].sum()
        new_black_aki_nums = black_data_df[black_data_df[Constant.score_column] > new_thresholds[0]][Constant.label_column].sum()
        new_white_aki_nums = white_data_df[white_data_df[Constant.score_column] > new_thresholds[1]][Constant.label_column].sum()

        old_aki_nums = old_black_aki_nums + old_white_aki_nums
        new_aki_nums = new_black_aki_nums + new_white_aki_nums

        return abs(old_aki_nums - new_aki_nums)

class RiskManDiffLoss(SplitLoss):
    def __init__(self):
        super().__init__()

    def cal_loss(self, cur_data_df, old_thresholds, new_thresholds):
        """
        计算换手率， 原本被选上的有风险人，经过新的阈值调整后，有多少人没有被选上。 指代 白种人的选取
        :param cur_data_df:
        :param old_thresholds:
        :param new_thresholds:
        :return:
        """
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race].sort_values(
            by=Constant.score_column, ascending=False)

        old_white_threshold = old_thresholds[1]
        new_white_threshold = new_thresholds[1]

        score_values = white_data_df[Constant.score_column].values
        miss_count = 0
        risk_count = 0
        for score in score_values:
            if old_white_threshold <= score < new_white_threshold:
                miss_count += 1

            if score >= old_white_threshold:
                risk_count += 1

        return miss_count / risk_count


class Constant:
    score_column = "score_y"
    label_column = "Label"
    # race_columns = "race"
    black_race_column = "Demo2_1"
    white_race_column = "Demo2_2"
    black_race = 1
    white_race = 1


def cal_black_white_recall_nums(cur_data_df: pd.DataFrame, thresholds: list):
    """
    根据数据集和阈值计算 黑人和白人 的召回率
    召回率计算方式： 统计当前数据集超过阈值的所有数据集（黑人+白人） N
    黑人召回率： 黑人超过阈值的人数 / N
    白人召回率： 白人超过阈值的人数 / N

    :param cur_data_df: 数据集
    :param thresholds: 黑人和白人的阈值
    :return:
    """
    black_threshold, white_threshold = thresholds[0], thresholds[1]

    # black
    black_score = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race][Constant.score_column].values
    target_black_score_nums = np.sum([j >= black_threshold for j in black_score])

    # white
    white_score = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race][Constant.score_column].values
    target_white_score_nums = np.sum([j >= white_threshold for j in white_score])

    score_nums = target_black_score_nums + target_white_score_nums
    # logger.info(f"总人数:{score_nums}, 黑人/白人: {target_black_score_nums}/{target_white_score_nums}")
    return [target_black_score_nums, target_white_score_nums]


def find_next_black_threshold(black_data_df, black_threshold):
    """
    找到下一个黑人阈值 阈值下降
    :param black_data_df:
    :param black_threshold:
    :return:
    """
    black_score = black_data_df[Constant.score_column].values
    len_black_score = len(black_score)
    threshold = black_threshold
    add_nums = 0

    for index, score in enumerate(black_score):
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

    cur_index = 0
    for index, score in enumerate(white_score):
        if score < white_threshold:
            cur_index = index - 1
            break

    new_index = cur_index - black_add_nums

    while new_index >= 0 and white_score[new_index] == white_threshold:
        new_index -= 1

    if new_index < 0:
        logger.warning("找不到使得白人阈值上升对应匹配的人数, 只能取最大值...")
        return white_score[0]

    threshold = white_score[new_index]

    # logger.info(f"白人阈值上升! [{white_threshold}->{threshold}]...")
    return threshold


def get_best_split_feature(loss_dict: dict):
    """
    根据loss字典获得最小损失的特征
    :param loss_dict:
    :return:
    """
    min_loss = np.iinfo(np.int32).max
    min_feature = None
    for feature, loss in loss_dict.items():
        if loss < min_loss:
            min_loss = loss
            min_feature = feature

    return min_feature


def cal_black_white_recall_rate(temp_data_df: pd.DataFrame, global_thresholds: list, risk_nums: list):
    subgroup_risk_nums = cal_black_white_recall_nums(temp_data_df, global_thresholds)
    black_recall = subgroup_risk_nums[0] / risk_nums[0]
    white_recall = subgroup_risk_nums[1] / risk_nums[1]
    return black_recall, white_recall


class SubGroupDecisionTree:
    def __init__(self, data_df: pd.DataFrame, select_features, splitLoss: SplitLoss, global_thresholds):
        self.data_df = data_df
        self.select_features = select_features
        self.splitLoss = splitLoss
        self.global_thresholds = global_thresholds

        self.node_count = 0
        self.node_tree = {}

    def build_tree(self, node: SubGroupNode, parent_index):

        node.set_node_index(self.__get_node_index())
        node.parent_index = parent_index

        # 1. 获取当前节点数据 和 备选特征集（用来做亚组分割）
        # cur_data_df = self.get_cur_data_df(node.pre_split_features_dict)
        # pre_split_features = list(node.pre_split_features_dict.keys())
        # remain_features = self.get_remain_features(pre_split_features)
        used_features = list(node.pre_split_features_dict.keys())
        remain_nums = len(self.select_features) - len(used_features)

        # 2.1 若 无备选特征集，则把当前node设置为叶子节点，并添加到对应的决策树之中, return
        if remain_nums == 0 or len(node.data_index) <= 1:
            node.set_leaf_node()
            self.node_tree[node.index] = node
            return node

        # 2.2 若 当前节点本来就为叶子节点，则添加到对应决策树之中，return
        if node.is_leaf:
            self.node_tree[node.index] = node
            return node

        # 3. 计算每一个备选特征集的loss值并得到最小loss的特征，此特征用来做亚组分割，并得到新分割节点列表
        best_split_feature, node_list = self.select_best_split_feature(node)
        node.sub_nodes = node_list
        self.node_tree[node.index] = node

        # 4. 将当前节点设置为内节点，并递归调用当前函数
        for sub_node in node_list:
            self.build_tree(sub_node, node.index)
        # 5. 返回根节点
        return node

    def get_tree_info(self):
        max_count = self.node_count
        max_depth = 5
        cur_count = 1
        cur_depth = 1
        while cur_count <= max_count:
            if cur_depth > max_depth:
                break
            logger.info("")
            logger.info(f"======================== depth: {cur_depth} ======================")
            for _, node in self.node_tree.items():
                if node.depth == cur_depth:
                    self.print_node_info(node)
                    cur_count += 1
            logger.info(f"======================== depth: {cur_depth} ======================")
            cur_depth += 1
            logger.info("")






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
        remain_features = self.get_cur_node_remain_features(node)

        # 计算当前父节点可获取到的黑人和白人风险人数（计算子节点recall）
        node.risk_nums = cal_black_white_recall_nums(cur_data_df, self.global_thresholds)
        logger.info(f"D{node.depth}-[ID{node.index}]-[黑人/白人风险人数:{node.risk_nums}]")
        # 1. 获取当前分割的信息
        pre_split_feature_dict = node.pre_split_features_dict

        # 2. 循环对备选特征进行亚组分割，每个特征都会会分出子节点
        all_loss_dict = {}
        all_node_dict = {}
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
                black_recall, white_recall = cal_black_white_recall_rate(temp_data_df, self.global_thresholds, node.risk_nums)
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
                    temp_node.set_leaf_node()
                    logger.info(
                        f"{node.depth}-[parent_index:{node.index}]-[{feature}:{feature_value}]: 黑人召回率({black_recall})高于白人召回率({white_recall}), 不再继续分亚组...")
                else:
                    # 2.1.3 若 黑人召回率 < 白人召回率， 则需要进一步以最小损失进行分亚组。也就是需要对子节点进行修改阈值使得符合2.2。
                    logger.warning(
                        f"{node.depth}-[parent_index:{node.index}]-[{feature}:{feature_value}]: 黑人召回率({black_recall})低于白人召回率({white_recall}), 还得继续分亚组...")
                    best_thresholds, split_loss = self.do_adjust_thresholds(temp_data_df, node.risk_nums)
                    # 2.1.4 得到最佳阈值后，把子节点进行初始化并赋值为内节点，设置相关参数并计算loss值

                    temp_node = SubGroupNode(
                        thresholds=best_thresholds,
                        split_loss=split_loss,
                        data_index=temp_data_df.index.to_list(),
                        pre_split_features_dict=temp_pre_split_feature_dict,
                        depth=cur_depth,
                        split_feature_name=feature,
                        split_feature_value=feature_value
                    )
                    temp_node.set_inner_node()

                # 将当前节点保存到节点列表之中
                cur_node_list.append(temp_node)

            # 2.2 将每个子节点的loss值进行相加
            for cur_node in cur_node_list:
                cur_loss += cur_node.split_loss

            # 添加到字典之中
            all_loss_dict[feature] = cur_loss
            all_node_dict[feature] = cur_node_list

        # 3. 每个备选特征的总loss值得到后，选出最小的loss值作为最优分割点。
        best_split_feature = get_best_split_feature(all_loss_dict)
        best_node_list = all_node_dict[best_split_feature]

        # 4. 返回结果
        return best_split_feature, best_node_list

    def do_adjust_thresholds(self, cur_data_df, risk_nums):
        """
        调整阈值使得 黑人召回率 高于 白人召回率
        调整策略：
            若白人召回率高于黑人召回率，则以 黑人阈值bt1 之下的第一个最高预测分数的黑人为 新的阈值 bt2，并计算得知新的阈值下 黑人新增了 p 人；
            而此时白人也要计算出 高于 白人阈值 wt1 p人后的新的阈值 wt2。
            若这么调整后 还是 白人高于黑人， 则黑人需要 继续上一步操作。
            最后将bt2和wt2作为新的阈值
        :param risk_nums:
        :param cur_data_df:
        :return:
        """
        black_threshold, white_threshold = self.global_thresholds

        black_recall, white_recall = cal_black_white_recall_rate(cur_data_df, self.global_thresholds, risk_nums)
        assert not black_recall >= white_recall, "黑人召回率已经高于白人召回率了，不需要调整..."

        # 将数据集先分成黑人和白人，再降序排序
        black_data_df = cur_data_df[cur_data_df[Constant.black_race_column] == Constant.black_race].sort_values(
            by=Constant.score_column, ascending=False)
        white_data_df = cur_data_df[cur_data_df[Constant.white_race_column] == Constant.white_race].sort_values(
            by=Constant.score_column, ascending=False)

        while black_recall < white_recall:
            black_threshold, black_add_nums = find_next_black_threshold(black_data_df, black_threshold)
            white_threshold = find_next_white_threshold(white_data_df, white_threshold, black_add_nums)
            black_recall, white_recall = cal_black_white_recall_rate(cur_data_df, [black_threshold, white_threshold], risk_nums)

        # 新的阈值
        new_thresholds = [black_threshold, white_threshold]

        split_loss = self.splitLoss.cal_loss(cur_data_df, self.global_thresholds, new_thresholds)

        return new_thresholds, split_loss

    def print_node_info(self, node: SubGroupNode):
        """
        输出节点信息
        :param node:
        :return:
        """
        logger.info(f"======================== id: {node.index} <<< p_id: {node.parent_index}======================")
        logger.info(f"depth: {node.depth}, is_leaf: {node.is_leaf}")
        logger.info(f"subGroupFrom: {node.split_feature_name}:{node.split_feature_value}")
        logger.info(f"split_loss: {node.split_loss}")
        logger.info(f"thresholds: [{round(node.thresholds[0])}, {round(node.thresholds[1])}]")
        logger.info(f"risk_nums: {node.risk_nums}")
        logger.info(f"======================== id: {node.index} <<< p_id: {node.parent_index}======================")

def load_my_data(data_file):
    """
    读取数据
    :return:
    """
    return pd.read_feather(data_file)


def get_global_thresholds(data, rate):
    """
    根据风险比例划分出一个公共的全局阈值
    :param data:
    :param rate:
    :return: [t, t]
    """
    score_array = data[Constant.score_column].sort_values(ascending=True).values
    risk_index = int(data.shape[0] * rate) - 1
    assert not risk_index <= 0, "当前得到的索引不能为负..."
    score_threshold = score_array[risk_index]

    return [score_threshold, score_threshold]


if __name__ == '__main__':
    risk_rate = 0.9
    data_file_name = "/home/chenqinhai/code_eicu/my_lab/fairness_strategy/data/test_data_1.feather"
    drg_cols = "/home/liukang/Doc/disease_top_20.csv"
    drg_list = pd.read_csv(drg_cols).squeeze().to_list()

    data_df = load_my_data(data_file_name)
    global_thresholds = get_global_thresholds(data_df, risk_rate)
    root_node = SubGroupNode(
        thresholds=global_thresholds,
        split_loss=0,
        data_index=data_df.index.to_list(),
        pre_split_features_dict={},
        depth=1
    )

    subgroup_select_features = drg_list

    myLoss = RiskProbDiffLoss()

    sbdt = SubGroupDecisionTree(
        data_df=data_df,
        global_thresholds=global_thresholds,
        select_features=subgroup_select_features,
        splitLoss=myLoss,
    )

    sbdt.build_tree(root_node, -1)
    sbdt.get_tree_info()
    print("done!")
