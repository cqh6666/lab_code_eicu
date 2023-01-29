# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     dot_to_pic
   Description:   ...
   Author:        cqh
   date:          2022/12/26 20:32
-------------------------------------------------
   Change Activity:
                  2022/12/26:
-------------------------------------------------
"""
__author__ = 'cqh'

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import plot_tree, export_graphviz
# from sklearn.tree import DecisionTreeClassifier
# import graphviz
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from sklearn.datasets import load_iris, load_breast_cancer
# import matplotlib.pyplot as plt
# # 模型搭建代码汇总
# # 1.读取数据与简单预处理
# data = load_breast_cancer()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target
# # Arrange Data into Features Matrix and Target Vector
# X = df.loc[:, df.columns != 'target']
# y = df.loc[:, 'target'].values
# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
# # Random Forests in `scikit-learn` (with N = 100)
# clf = DecisionTreeClassifier(max_depth = 2,
#                              random_state = 0)
# # Step 3: Train the model on the data
# clf.fit(X_train, Y_train)
#
# fn=data.feature_names
# cn=data.target_names
# dot_data = export_graphviz(clf, out_file="tree.dot", feature_names=fn, class_names=cn, filled=True)


import pickle
import sys

from fairness_strategy.getSubGroup2 import SubGroupDecisionTree
from fairness_strategy.getSubGroup2 import SubGroupNode

root_node = SubGroupNode(
            thresholds=[0, 0],
            data_index=[1, 2, 3],
            depth=1,
        )

tree_str = pickle.dumps(root_node)
with open("my_tree_test.pkl", "wb") as file:
    file.write(tree_str)
    print(sys.getsizeof(tree_str) / 1024 / 1024)
with open("my_tree_test.pkl", "rb") as f:
    obj = pickle.loads(f.read())