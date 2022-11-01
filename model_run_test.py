# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test_2
   Description:   ...
   Author:        cqh
   date:          2022/10/28 15:31
-------------------------------------------------
   Change Activity:
                  2022/10/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import os.path

import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

# eicu_rf_data = r"D:\lab\my_lab_eicu\data\eICU_rf.csv"
data_path = "/home/chenqinhai/code_eicu/my_lab/data/"
eicu_rf_data = 'all_data_rf2_df_v5.csv'
# eicu_rf_data = 'all_data_rf2_df_v4.csv'
data = os.path.join(data_path, eicu_rf_data)

print(data)

df = pd.read_csv(data, index_col=0)


def eicu_data_screen(df):
    df_copy = df.copy()
    df_copy = df_copy[df_copy["age"] >= 18]
    # df_copy.drop(columns='patientunitstayid', inplace=True)
    label = 'aki_label'

    return df_copy, label


df, label = eicu_data_screen(df)

X = df
y = X[[label]]
X = X.drop(columns=label)
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(X, y, test_size=0.3, random_state=42)


def lr_global_train(train_iter, class_weigth_flag):
    train_x_ft = train_data_x
    test_x_ft = test_data_x

    if class_weigth_flag:
        lr_all = LogisticRegression(solver='liblinear', max_iter=train_iter, n_jobs=-1, class_weight='balanced')
    else:
        lr_all = LogisticRegression(solver='liblinear', max_iter=train_iter, n_jobs=-1)

    lr_all.fit(train_x_ft, train_data_y)
    y_predict = lr_all.decision_function(test_x_ft)
    auc = roc_auc_score(test_data_y, y_predict)
    recall = recall_score(test_data_y, lr_all.predict(test_data_x))

    print(
        f'[global] - max_iter:{train_iter}, train_iter:{lr_all.n_iter_}, auc: {auc}, recall: {recall}')

    weight_importance = lr_all.coef_[0]
    weight_importance_df = pd.DataFrame({"feature_weight": weight_importance})
    weight_importance_df.to_csv(f"weight_important_{train_iter}_{class_weigth_flag}.csv")
    return auc, recall


def get_xgb_params(num_boost):
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': -1,
        'verbosity': 0,
        'seed': 2022,
        'tree_method': 'hist'
    }
    num_boost_round = num_boost
    return params, num_boost_round


def xgb_train_global(num_boost):
    d_train = xgb.DMatrix(train_data_x, label=train_data_y)
    d_test = xgb.DMatrix(test_data_x, label=test_data_y)

    params, num_boost_round = get_xgb_params(num_boost)

    model = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=num_boost,
        verbose_eval=False,
    )

    test_y_predict = model.predict(d_test)
    auc = roc_auc_score(test_data_y, test_y_predict)

    print(f'num_boost_round: {num_boost} : The auc of this model is {auc}')

    return auc, model


def get_shap_value(train_x, model):
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(train_x)
    res = pd.DataFrame(data=shap_value, columns=train_x.columns)
    res = res.abs().mean(axis=0)
    res = res / res.sum()
    return res


# my_auc, my_model = xgb_train_global(100)
# shap_value = get_shap_value(train_data_x, my_model)

lr_global_train(100, True)
lr_global_train(100, False)

lr_global_train(500, True)
lr_global_train(500, False)
