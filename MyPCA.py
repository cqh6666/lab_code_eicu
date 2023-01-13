# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyPCA
   Description:   ...
   Author:        cqh
   date:          2023/1/12 21:07
-------------------------------------------------
   Change Activity:
                  2023/1/12:
-------------------------------------------------
"""
__author__ = 'cqh'
import numpy as np
from sklearn.preprocessing import StandardScaler



class MyPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
        if n_components > 1:
            raise ValueError("n_components 不能大于1！")
        self.X_cov = None

    def get_X_cov(self, X):
        self.standar_scaler = StandardScaler()
        X = self.standar_scaler.fit_transform(X)
        self.X_cov = np.matmul(X.T, X) / len(X)
        return self.X_cov

    def set_X_cov(self, X_cov):
        self.X_cov = X_cov

    def fit(self, X):
        self.standar_scaler = StandardScaler()
        X = self.standar_scaler.fit_transform(X)
        X_cov = np.matmul(X.T, X) / len(X)
        # 计算特征值和特征向量
        w, v = np.linalg.eig(X_cov)
        # v[:,i] 是特征值w[i]所对应的特征向量
        idx = np.argsort(w)[::-1]  # 获取特征值降序排序的索引
        self.w = w[idx]  # [k,]  进行降序排列

        # 按百分比
        total_var = self.w.sum()
        explained_variance_ratio_ = self.w / total_var
        cur_ratio = 0
        n_comp = 0
        for var in explained_variance_ratio_:
            cur_ratio += var
            if cur_ratio >= self.n_components:
                break
            n_comp += 1
        self.v = v[:, idx][:, :n_comp]  # [n,k]，  排序
        return self

    def fit_cov(self):
        if self.X_cov is None:
            raise NotImplementedError("请先通过 set_X_cov 设置协方差矩阵!")

        # 计算特征值和特征向量
        w, v = np.linalg.eig(self.X_cov)
        # v[:,i] 是特征值w[i]所对应的特征向量
        idx = np.argsort(w)[::-1]  # 获取特征值降序排序的索引
        self.w = w[idx]  # [k,]  进行降序排列

        # 按百分比
        total_var = self.w.sum()
        explained_variance_ratio_ = self.w / total_var
        cur_ratio = 0
        n_comp = 0
        for var in explained_variance_ratio_:
            cur_ratio += var
            if cur_ratio >= self.n_components:
                break
            n_comp += 1
        self.v = v[:, idx][:, :n_comp]  # [n,k]，  排序
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        # 降维
        self.standar_scaler = StandardScaler()
        X = self.standar_scaler.fit_transform(X)
        return np.matmul(X, self.v)  # [m,n] @ [n,k] = [m,k]

    def inverse_transform(self, X):
        return self.standar_scaler.inverse_transform(np.matmul(X, self.v.T))


if __name__ == '__main__':
    from api_utils import get_fs_each_hos_data_X_y

    pca_model = MyPCA(n_components=0.95)
    # 获取数据
    t_x, test_data_x, _, test_data_y = get_fs_each_hos_data_X_y(0)

    pca_model.fit_transform(test_data_x)