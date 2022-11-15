# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S02_feature_select
   Description:   多种特征选择方式
   Author:        cqh
   date:          2022/11/6 17:00
-------------------------------------------------
   Change Activity:
                  2022/11/6:
-------------------------------------------------
"""
__author__ = 'cqh'

import feather
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import cross_val_score

from api_utils import get_all_norm_data, get_train_test_data_X_y
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from my_logger import MyLog
import numpy as np
import os
from sklearn.feature_selection import chi2, f_classif, VarianceThreshold
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from kydavra import MUSESelector, ChiSquaredSelector, PValueSelector, PearsonCorrelationSelector, PCAFilter, \
    M3USelector, SpearmanCorrelationSelector, KendallCorrelationSelector


def var_feature_select(data_X, data_y, threshold=0.05):
    """
    对全部特征处理，方差阈值特征选择
    :param threshold:
    :param data_X:
    :param data_y:
    :return:
    """
    # 将阈值设置为0.05,即会删除标准差小于0.05的特征（列）
    var_thres = VarianceThreshold(threshold=threshold)
    var_thres.fit(data_X)

    select_columns = var_thres.get_support(indices=True)
    select_columns = data_X.columns[select_columns].to_list()
    my_logger.warning("使用方差阈值筛选出来的特征结果有 {}/{} 个, 前5个是 {}".format(
        len(select_columns), len(data_X.columns), select_columns[:5]))

    return select_columns


def chi_feature_select(data_X, data_y, threshold=0.05):
    """
    卡方特征选择
    :param data_X:
    :param data_y:
    :param threshold:
    :return:
    """
    select_columns = data_X.iloc[:, chi2(np.abs(data_X), data_y)[1] < threshold].columns.tolist()

    my_logger.warning("使用卡方检验阈值筛选出来的特征结果有 {}/{} 个, 前5个是 {}".format(
        len(select_columns), len(data_X.columns), select_columns[:5]))

    return select_columns


def f_classif_feature_select(data_X, data_y, threshold=0.05):
    """
    F检验
    :param data_X: 
    :param data_y: 
    :param threshold: 
    :return: 
    """
    select_columns = data_X.iloc[:, f_classif(np.abs(data_X), data_y)[1] < threshold].columns.tolist()
    my_logger.warning("使用F检验阈值筛选出来的特征结果有 {}/{} 个, 前5个是 {}".format(
        len(select_columns), len(data_X.columns), select_columns[:5]))

    return select_columns


def t_test_feature_select(data_X, data_y, threshold=0.05):
    """
    T检验
    :param data_X:
    :param data_y:
    :param threshold:
    :return:
    """
    columns = data_X.columns.values
    labels = data_y.values
    datas = data_X.values
    feature_num = datas.shape[1]
    positive_sample_index = labels == 1
    positive_sample = datas[positive_sample_index, :]
    negative_sample_index = labels == 0
    negative_sample = datas[negative_sample_index, :]
    final_feature = []

    for i in range(feature_num):
        positive_pre_feature = positive_sample[:, i]
        negative_pre_feature = negative_sample[:, i]
        _, p = stats.ttest_ind(positive_pre_feature, negative_pre_feature, equal_var=True)
        if p < threshold:
            final_feature.append(i)
    return columns[final_feature].tolist()


def lr_feature_select(data_X, data_y):
    """
    LR进行特征选择
    :param data_X:
    :param data_y:
    :return:
    """
    columns = data_X.columns.values
    feature_num = data_X.shape[1]
    final_feature = []
    for i in range(feature_num):
        pre_data = data_X.iloc[:, i].values.reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(pre_data, data_y)
        Weight_importance = lr.coef_[0]
        predProbs = lr.predict_proba(pre_data)
        # Design matrix -- add column of 1's at the beginning of your X_train matrix
        X_design = np.hstack([np.ones((pre_data.shape[0], 1)), pre_data])
        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        V = np.product(predProbs, axis=1)
        # Note that the @-operater does matrix multiplication in Python 3.5+, so if you're running
        # Python 3.5+, you can replace the covLogit-line below with the more readable:
        # covLogit = np.linalg.inv(X_design.T @ V @ X_design)
        covLogit = np.linalg.pinv(X_design.T * V @ X_design)
        # Standard errors
        Var = np.diag(covLogit)
        Var_feature = np.zeros(1)
        Var_feature.setflags(write=True)
        Var_feature = Var[1:]
        se = np.sqrt(Var_feature)
        if Weight_importance < 0:
            final = (Weight_importance + 1.96 * se) < 0
        else:
            final = (Weight_importance - 1.96 * se) > 0
        if final:
            final_feature.append(i)
    return columns[final_feature].tolist()


def rf_feature_select(data_X, data_y, topK=500):
    """
    随机森林特征提取
    :param data_X:
    :param data_y:
    :param topK:
    :return:
    """
    rf_clf = RandomForestClassifier()
    rf_clf.fit(data_X, data_y)

    feature_importance = rf_clf.feature_importances_

    feature_importance = pd.DataFrame(feature_importance, columns=['importance'])
    feature_importance['name'] = data_X.columns
    rank_K = feature_importance.sort_values(by='importance', ascending=False)[:topK]
    rank_K.to_csv(os.path.join(save_path, "rf_fi.csv"))

    return rank_K['name'].to_list()


def xgb_feature_select(data_X, data_y, threshold=5):
    """
    xgb特征提取
    :param threshold:
    :param data_X:
    :param data_y:
    :return:
    """
    import xgboost as xgb

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

    d_train = xgb.DMatrix(data_X, label=data_y)
    params, num_boost_round = get_xgb_params(1000)

    model = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )

    weight_importance = model.get_score(importance_type='weight')
    # 保存特征重要性
    result = pd.Series(index=data_X.columns.tolist(), dtype='float64')
    weight = pd.Series(weight_importance, dtype='float')
    result.loc[:] = weight
    result.fillna(0, inplace=True)
    result.to_csv(os.path.join(save_path, "xgb_weight_fi.csv"))
    print("save success!")

    return result[result > threshold].index.to_list()
    # return result[result != 0].index.to_list()


def pearson_feature_select(all_data):
    """
    M3USelector
    :return:
    """
    from kydavra import PearsonCorrelationSelector

    method = PearsonCorrelationSelector()
    select_columns = method.select(all_data, y_label)
    return select_columns

def get_data_X_y():
    """X,y"""
    all_data = get_all_norm_data()
    all_data.index = all_data[patient_id].tolist()
    all_data_x = all_data.drop([y_label, hospital_id, patient_id], axis=1)
    all_data_y = all_data[y_label]
    return all_data_x, all_data_y


def get_cat_feature():
    """
    获得类别特征
    :return:
    """
    return pd.read_csv(os.path.join(data_path, "category_feature.csv")).squeeze().to_list()


def get_con_feature():
    """
    获得连续数据
    :return:
    """
    return pd.read_csv(os.path.join(data_path, "continue_feature.csv")).squeeze().to_list()


def feature_select_valid(select_columns, select_str):
    """
    特征选择后进行验证
    :param select_columns: 选择的特征 list
    :param select_str: 特征选择方法
    :return:
    """
    # 验证AUC性能
    my_logger.warning("选择 {} 方式特征选择 {} 个特征后的结果: {}".format(
          select_str, len(select_columns), lr_auc_train(select_columns)))


def run():
    """
    主入口
    :return:
    """
    # 获取数据X,y
    all_data_X, all_data_y = get_data_X_y()
    # 连续特征，类别特征
    cat_feature = get_cat_feature()
    con_feature = get_con_feature()

    cat_data_X = all_data_X[cat_feature]
    con_data_X = all_data_X[con_feature]

    cat_data = pd.concat([cat_data_X, all_data_y], axis=1)
    con_data = pd.concat([con_data_X, all_data_y], axis=1)

    print("chi+pear...")
    # chi特征选择 + Pearson特征选择
    method = ChiSquaredSelector()
    select_columns = method.select(cat_data, y_label)
    method2 = PearsonCorrelationSelector()
    select_columns2 = method2.select(con_data, y_label)
    select_columns_new1 = select_columns + select_columns2

    print("rf...")
    # rf随机森林特征选择
    select_columns_new2 = rf_feature_select(all_data_X, all_data_y, topK=500)

    print("xgb...")
    # xgb特征选择
    select_columns_new3 = xgb_feature_select(all_data_X, all_data_y, 5)
    select_columns_new4 = xgb_feature_select(all_data_X, all_data_y, 0)

    print("lr...")
    select_columns_new5 = lr_feature_select(all_data_X, all_data_y)

    # 特征选择前的结果
    my_logger.warning("特征选择前的AUC结果: {}".format(lr_auc_train(all_data_X.columns)))

    feature_select_valid(select_columns_new1, select_str="chi+pearson")
    feature_select_valid(select_columns_new2, select_str="chi+pearson")
    feature_select_valid(select_columns_new3, select_str="chi+pearson")
    feature_select_valid(select_columns_new4, select_str="chi+pearson")
    feature_select_valid(select_columns_new5, select_str="chi+pearson")


def model_select():
    """
    过滤法
    :return:
    """
    # 获取数据X,y
    all_data_X, all_data_y = get_data_X_y()
    # 连续特征，类别特征
    cat_feature = get_cat_feature()
    con_feature = get_con_feature()

    cat_data_X = all_data_X[cat_feature]
    con_data_X = all_data_X[con_feature]

    cat_data = pd.concat([cat_data_X, all_data_y], axis=1)
    con_data = pd.concat([con_data_X, all_data_y], axis=1)
    all_data = pd.concat([all_data_X, all_data_y], axis=1)

    print("chi+pear...")
    # chi特征选择 + Pearson特征选择
    method = ChiSquaredSelector()
    select_columns = method.select(cat_data, y_label)
    method2 = PearsonCorrelationSelector()
    select_columns2 = method2.select(con_data, y_label)
    select_columns_new1 = select_columns + select_columns2
    feature_select_valid(select_columns_new1, select_str="chi+pearson")

    print("chi+Spearman...")
    # chi特征选择 + Pearson特征选择
    method2 = SpearmanCorrelationSelector()
    select_columns2 = method2.select(con_data, y_label)
    select_columns_new1 = select_columns + select_columns2
    feature_select_valid(select_columns_new1, select_str="chi+Spearman")

    print("chi+Kendall...")
    # chi特征选择 + Pearson特征选择
    method2 = KendallCorrelationSelector()
    select_columns2 = method2.select(con_data, y_label)
    select_columns_new1 = select_columns + select_columns2
    feature_select_valid(select_columns_new1, select_str="chi+Kendall")

    print("M3USelector...")
    method = M3USelector(n_features=1000)
    select_columns_new2 = method.select(all_data, y_label)
    feature_select_valid(select_columns_new2, select_str="M3USelector(1000)")

    method = M3USelector(n_features=1500)
    select_columns_new2 = method.select(all_data, y_label)
    feature_select_valid(select_columns_new2, select_str="M3USelector(1500)")

    method = M3USelector(n_features=500)
    select_columns_new2 = method.select(all_data, y_label)
    feature_select_valid(select_columns_new2, select_str="M3USelector(500)")


def lr_auc_train(columns_list):
    # 特征选择前的结果
    lr_all = LogisticRegression(max_iter=1000, solver="liblinear", random_state=2022)
    train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_data_X_y()

    train_data_x = train_data_x[columns_list]
    test_data_x = test_data_x[columns_list]

    lr_all.fit(train_data_x, train_data_y)
    y_predict = lr_all.decision_function(test_data_x)
    auc = roc_auc_score(test_data_y, y_predict)
    # recall = recall_score(test_data_y, lr_all.predict(test_data_x))
    return auc


def rf_get_fi():
    all_data_X, all_data_y = get_data_X_y()
    rf_feature_select(all_data_X, all_data_y, topK=all_data_X.shape[1] - 1)

def xgb_get_fi():
    all_data_X, all_data_y = get_data_X_y()
    xgb_feature_select(all_data_X, all_data_y)


def last_feature_select():
    """
    最终的特征选择方案
    :return:
    """
    # 获取数据X,y
    all_data_X, all_data_y = get_data_X_y()
    # 连续特征，类别特征
    cat_feature = get_cat_feature()
    con_feature = get_con_feature()
    print("离散特征: {}, 连续特征: {}".format(len(cat_feature), len(con_feature)))

    cat_data_X = all_data_X[cat_feature]
    con_data_X = all_data_X[con_feature]

    cat_data = pd.concat([cat_data_X, all_data_y], axis=1)
    con_data = pd.concat([con_data_X, all_data_y], axis=1)

    # 针对离散数据 - 卡方过滤
    method = ChiSquaredSelector()
    select_cat_columns = method.select(cat_data, y_label)
    print("对离散特征经过卡方过滤后特征数: {}".format(len(select_cat_columns)))

    # 方差过滤
    select_var_columns = var_feature_select(all_data_X[con_feature], all_data_y)
    print("对连续特征进行方差过滤后特征数: {}".format(len(select_var_columns)))

    # 保存离散特征
    pd.DataFrame({"cat_feature": select_cat_columns}).to_csv(os.path.join(save_path, "select_cat_columns_v2.csv"))

    # 离散+连续特征
    select_columns = select_cat_columns + select_var_columns
    # todo 相关性过滤 - 互信息过滤

    # 嵌入法过滤
    # 基于XGB特征重要性
    select_xgb_columns = xgb_feature_select(all_data_X[select_columns], all_data_y, 0)
    print("根据XGB重要性过滤后特征数: {}".format(len(select_xgb_columns)))
    select_columns_list = list(set(select_cat_columns) | set(select_xgb_columns))
    pd.DataFrame({"feature": select_columns_list}).to_csv(os.path.join(save_path, "select_xgb_columns_v2.csv"))

    # 基于LR特征重要性
    select_lr_columns = lr_feature_select(all_data_X[select_columns], all_data_y)
    print("根据LR重要性过滤后特征数: {}".format(len(select_lr_columns)))
    select_columns_list = list(set(select_cat_columns) | set(select_lr_columns))
    pd.DataFrame({"feature": select_columns_list}).to_csv(os.path.join(save_path, "select_lr_columns_v2.csv"))

    # 进行性能评估
    feature_select_valid(select_xgb_columns, select_str="xgb特征重要性")
    feature_select_valid(select_lr_columns, select_str="lr特征重要性")


def last_feature_select2():
    """
    最终的特征选择方案 - 只根据XGB或LR特征重要性 进行筛选
    :return:
    """
    # 获取数据X,y
    all_data_X, all_data_y = get_data_X_y()

    # 嵌入法过滤
    # 基于XGB特征重要性
    select_xgb_columns = xgb_feature_select(all_data_X, all_data_y, 0)
    print("根据XGB重要性过滤后特征数: {}".format(len(select_xgb_columns)))
    pd.DataFrame({"feature": select_xgb_columns}).to_csv(os.path.join(save_path, f"select_xgb_columns_v{version}.csv"))

    # 基于LR特征重要性
    select_lr_columns = lr_feature_select(all_data_X, all_data_y)
    print("根据LR重要性过滤后特征数: {}".format(len(select_lr_columns)))
    pd.DataFrame({"feature": select_lr_columns}).to_csv(os.path.join(save_path, f"select_lr_columns_v{version}.csv"))

    # 进行性能评估
    feature_select_valid(select_xgb_columns, select_str="xgb特征重要性")
    feature_select_valid(select_lr_columns, select_str="lr特征重要性")


if __name__ == '__main__':
    my_logger = MyLog().logger

    """
    version = 3 只进行XGB或LR特征重要性选择特征
    """
    version = 3
    y_label = "aki_label"
    hospital_id = "hospitalid"
    patient_id = "index"
    data_path = "/home/chenqinhai/code_eicu/my_lab/data/raw_file"
    save_path = "/home/chenqinhai/code_eicu/my_lab/result/S02/feature_importance/"
    # run()
    last_feature_select2()
