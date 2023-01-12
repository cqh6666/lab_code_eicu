# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     lr_utils_api
   Description:   ��ȡLR����ļ���Ϣ
   Author:        cqh
   date:          2022/7/8 14:41
-------------------------------------------------
   Change Activity:
                  2022/7/8:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os
from my_logger import logger

MODEL_SAVE_PATH = '/home/chenqinhai/code_eicu/my_lab/result/S03/{}'
global_lr_iter = 1000
"""
version = 1 ������Ȩ�أ��ϰ汾���ݣ�
version = 1a ������Ȩ�أ��ϰ汾���ݣ� 7��3�ָ�
version = 5 ������Ȩ�أ��°汾���ݣ�
version = 5a ������Ȩ�أ��°汾���ݣ� 7:3�ָ�
version = 6 ��ƽ����Ȩ��
version = 7 1��9 ��Ȩ��
version = 8 0.05��0.95 ��Ȩ��
version = 10 �����ݺ��Ȩ�� xgb ������Ȩ��
version = 11 �����ݺ��Ȩ�� lr ������Ȩ��
version = 12 �����ݺ��Ȩ�� xgb ������Ȩ��
version = 13 �����ݺ��Ȩ�� lr ������Ȩ��
version = 14 �����ݺ��Ȩ�� xgb ������Ȩ�� ��ɢ���� 1420
version = 15 �����ݺ��Ȩ�� lr ������Ȩ�� ��ɢ����
version = 16 ֱ��xgb����ѡ�񣬲��������ͷ�����  1600
version = 20 ������Ԥ���� 1596

version = 5b xgb����ѡ�� ������
"""
# global_version = 10
# hos_version = 10
version = "5b"


def get_init_similar_weight(hos_id):
    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH.format(hos_id),
                                            f"S03_0_psm_global_lr_{global_lr_iter}_v{version}.csv")
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()
    logger.warning(f"��ȡ�����Զ���({len(init_similar_weight)}):{init_similar_weight[:5]}, version:{version}")
    return init_similar_weight


def get_lr_init_similar_weight(hos_id):
    return get_init_similar_weight(hos_id)


def get_transfer_weight(hos_id):
    # ȫ��Ǩ�Ʋ��� ��Ҫ�õ���ʼ��csv
    init_weight_file_name = os.path.join(MODEL_SAVE_PATH.format(hos_id),
                                         f"S03_global_weight_lr_{global_lr_iter}_v{version}.csv")
    global_feature_weight = pd.read_csv(init_weight_file_name).squeeze().tolist()
    logger.warning(f"��ȡǨ�ƶ���({len(global_feature_weight)}):{global_feature_weight[:5]}, version:{version}")
    return global_feature_weight
