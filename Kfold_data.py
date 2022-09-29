# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     Kfold_data
   Description:   5折交叉
   Author:        cqh
   date:          2022/9/27 20:10
-------------------------------------------------
   Change Activity:
                  2022/9/27:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from api_utils import get_all_norm_data

y_label = "aki_label"
hospital_id = "hospitalid"
patient_id = "index"

folder = KFold(n_splits=5, random_state=2022, shuffle=True)

all_data = get_all_norm_data()
all_data_x = all_data.drop([y_label, hospital_id, patient_id], axis=1)
all_data_y = all_data[y_label]

train_list = []
test_list = []

for train, test in folder.split(all_data_x):
    train_list.append(train)
    test_list.append(test)


cur_train = all_data.iloc[train_list[0], :]
cur_test = all_data.iloc[test_list[0], :]
