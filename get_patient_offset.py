# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_patient_offset
   Description:   ...
   Author:        cqh
   date:          2022/9/29 10:48
-------------------------------------------------
   Change Activity:
                  2022/9/29:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import numpy as np
from api_utils import get_all_norm_data

patient_df = pd.read_csv("/home/chenqinhai/code_eicu/my_lab/data/train_file/offset_patients.csv")
patient_list = patient_df.iloc[:, 0].tolist()

