# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyDecitionTree
   Description:   ...
   Author:        cqh
   date:          2022/12/26 21:28
-------------------------------------------------
   Change Activity:
                  2022/12/26:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.tree import BaseDecisionTree
from sklearn.tree._tree import Tree

from fairness_strategy.getSubGroup import SubGroupDecisionTree


class MyDecisionTree(BaseDecisionTree):

    def __init__(self,
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                class_weight=None,
                ccp_alpha=0.0,
        ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

        self.tree_ = None

    def set_params(self, group_dt: SubGroupDecisionTree):
        self.criterion = group_dt.splitLoss.name
        leftChild, rightChild = [], []
        node_tree = group_dt.node_tree
        impurity_loss = []
        # leftChild, rightChild
        for node_index, node in node_tree.items():
            if node.is_leaf:
                leftChild[node_index] = -1
                rightChild[node_index] = -1
            else:
                leftChild[node_index] = node.sub_nodes[0].index
                rightChild[node_index] = node.sub_nodes[1].index

            impurity_loss[node_index] = node.split_loss



        self.tree_ = MyTree(
            capacity=len(list(node_tree.keys())),
            children_left=leftChild,
            children_right=rightChild,
            value=[],
            feature=[],
            threshold=[],
            impurity=impurity_loss,
            n_classes=[],
        )


class MyTree(Tree):

    def __init__(self, capacity, n_classes, children_left, children_right, feature, impurity, threshold, value):
        self.impurity = impurity
        self.feature = feature
        self.children_right = children_right
        self.threshold = threshold
        self.value = value
        self.children_left = children_left
        self.capacity = capacity
        self.n_classes = n_classes
        self.n_outputs = 1


