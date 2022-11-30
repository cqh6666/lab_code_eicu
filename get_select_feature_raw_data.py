# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_select_feature_data
   Description:   获取和友光所获取数据的特征
   Author:        cqh
   date:          2022/10/29 15:49
-------------------------------------------------
   Change Activity:
                  2022/10/29:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def get_demo_data():
    """
    获得demo特征数据
    age是连续特征,但只有一个缺失值,直接均值填充
    :return:
    """
    demo_data = pd.read_csv(os.path.join(data_path, "demographics_raw.csv"), index_col=0)

    # 将patient_id作为索引,并排序
    demo_data.index = demo_data['patientunitstayid'].tolist()
    demo_data.sort_index(inplace=True)

    # aki_label放到第一列
    columns = demo_data.columns.tolist()
    columns.insert(0, columns.pop(columns.index('aki_label')))
    demo_data = demo_data.loc[:, columns]

    # gender one_hot编码
    gender_onehot_df = pd.get_dummies(demo_data['gender'], prefix="gender")

    # ethnicity one_hot编码
    ethnicity_onehot_df = pd.get_dummies(demo_data['ethnicity'], prefix="race")

    # 去除多余列
    demo_data.drop(['patientunitstayid', 'gender', 'ethnicity', 'admissionheight', 'admissionweight'], axis=1,
                   inplace=True)
    # 合并
    demo_data = pd.concat([demo_data, gender_onehot_df, ethnicity_onehot_df], axis=1)

    # 处理年龄 age 特征
    age_df = demo_data['age']
    age_df = age_df.replace('> 89', 89)
    age_df = pd.to_numeric(age_df, errors='coerce')
    age_df.fillna(int(age_df.mean()), inplace=True)
    demo_data['age'] = age_df.astype(np.int8)

    print("get demo data...")

    return demo_data


def get_vital_data():
    """
    vital feature
    vital全是连续特征,有缺失值
    :return:
    """
    # sbp,dbp,bmi 都要
    vital_df_1 = pd.read_csv(os.path.join(data_path, "vitalaperiodic_raw.csv"), index_col=0)

    # fio2, polys,
    vital_feature_list = [
        "temperature", "heartrate", "respiration", "sao2", 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
        'pasystolic', 'padiastolic', 'pamean', 'cvp', 'etco2', 'st1', 'st2', 'st3', 'icp'
    ]

    # 重新调整index
    vital_df_2 = pd.read_csv(os.path.join(data_path, "vital_raw.csv"), index_col=0)
    vital_df_2.index = vital_df_2['patientunitstayid'].tolist()
    vital_df_2.drop(['row_id'], axis=1, inplace=True)
    vital_df_2.drop(['patientunitstayid'], axis=1, inplace=True)
    vital_df_2 = pd.concat([pd.DataFrame(index=vital_df_1.index.tolist()), vital_df_2], axis=1)
    vital_df_2 = vital_df_2[vital_feature_list]

    # 合并
    vital_df = pd.concat([vital_df_1, vital_df_2], axis=1)

    print("get vital data...")
    return vital_df


def get_lab_data():
    """
    lab全是连续特征,有缺失值
    :return:
    """
    lab_df = pd.read_csv(os.path.join(data_path, "lab_raw.csv"), index_col=0)

    fea_list = ["sodium", "potassium", "bicarbonate", "anion gap",
                "glucose", "bedside glucose", "glucose - CSF", "calcium", "BUN", "phosphate", "total protein",
                "albumin", "total bilirubin", "AST (SGOT)", "WBC x 1000", "WBC''s in urine",
                "WBC''s in cerebrospinal fluid", "WBC''s in body fluid",
                "platelets x 1000", "lactate", "troponin - I", "troponin - T", "pH", "serum ketones", "ALT (SGPT)",
                "chloride", "magnesium", "PTT", "-basos", "-eos", "Hct", "PT - INR", "-lymphs", "-monos",
                "RBC", "RDW", "paCO2", "paO2", "CRP", "fibrinogen", "FiO2"]

    lab_df = lab_df[fea_list]

    lab_df.replace(0, np.NaN, inplace=True)

    print("get lab data...")

    return lab_df


def get_med_data():
    """
    med不用填充,没有缺失值
    :return:
    """
    med_df = pd.read_csv(os.path.join(data_path, "medication_raw.csv"), index_col=0)
    # "insulin", "lactulose"
    med_feature_list = [
        "INSULIN-LISPRO (rDNA) *UNIT* INJ",
        "LACTULOSE 20 GRAM/30 ML UD LIQ"
    ]

    med_df = med_df[med_feature_list]
    columns = med_df.columns
    # 处理inf问题
    for column in columns:
        isinf_df = np.isinf(med_df[column])
        if isinf_df.sum() > 0:
            max_value = med_df.loc[~isinf_df, column].sort_values(ascending=False).max()
            med_df.loc[isinf_df, column] = max_value
            print("process inf value", column, isinf_df.sum())

    print("get med data...")

    return med_df


def get_diag_data():
    """
    diag没有缺失值
    :return:
    """
    dia_df = pd.read_csv(os.path.join(data_path, "diagnosis_one_hot.csv"), index_col=0)

    diag_feature_list = [
        "primary lung cancer", "respiratory failure", "acute respiratory failure",
        "diabetes mellitus", "congestive heart failure", "hypertension", "stroke",
        "atrial fibrillation", "asthma / bronchospasm", "coronary artery disease",
        "chronic kidney disease",
    ]
    dia_df = dia_df[diag_feature_list]

    print("get diag data...")

    return dia_df


def get_treat_data():
    """
    treat没有缺失值
    :return:
    """
    treatment_df = pd.read_csv(os.path.join(data_path, "treatment_raw.csv"), index_col=0)

    treatment_feature_list = [
        "pulmonary|radiologic procedures / bronchoscopy|chest x-ray",
        "pulmonary|radiologic procedures / bronchoscopy|CT scan|with contrast",
        "neurologic|procedures / diagnostics|head CT scan",
        "pulmonary|radiologic procedures / bronchoscopy|CT scan",
        "neurologic|procedures / diagnostics|head CT scan|without contrast",
        "gastrointestinal|radiology, diagnostic and procedures|CT scan|abdomen",
        "gastrointestinal|radiology, diagnostic and procedures|CT scan|pelvis"
    ]
    treatment_df = treatment_df[treatment_feature_list]
    columns = treatment_df.columns
    # 处理inf问题
    for column in columns:
        isinf_df = np.isinf(treatment_df[column])
        if isinf_df.sum() > 0:
            max_value = treatment_df.loc[~isinf_df, column].sort_values(ascending=False).tempMax()
            treatment_df.loc[isinf_df, column] = max_value
            print("process inf value", column, isinf_df.sum())

    print("get treat data...")

    return treatment_df


def concat_all_data():
    """
    合并所有不同种特征
    :return:
    """
    demo_df = get_demo_data()
    vital_df = get_vital_data()
    lab_df = get_lab_data()
    med_df = get_med_data()
    diag_df = get_diag_data()
    treat_df = get_treat_data()

    all_data = pd.concat([demo_df, vital_df, lab_df, med_df, diag_df, treat_df], axis=1)
    # v5 提取友光版本的特征
    all_data.to_csv(all_data_file)
    print("save success!")
    return all_data


def missing_value_processing(data_X, label):
    # 随机森林填补缺失值
    index = data_X.index
    data_X.index = range(len(data_X))
    X_missing_reg = data_X.copy()
    column_list = X_missing_reg.isnull().sum(axis=0).index
    sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
    for i in sortindex:
        df = X_missing_reg
        fillc = df.loc[:, column_list[i]]
        print(i, column_list[i], "...")
        if fillc.isnull().sum() == 0:
            continue
        df = pd.concat([df.loc[:, df.columns != column_list[i]], pd.DataFrame(label)], axis=1)
        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
        Ytrain = fillc[fillc.notnull()]
        Ytest = fillc[fillc.isnull()]
        Xtrain = df_0[Ytrain.index, :]
        Xtest = df_0[Ytest.index, :]
        if len(Xtrain) != 0:
            rfc = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            rfc = rfc.fit(Xtrain, Ytrain)
            Ypredict = rfc.predict(Xtest)
        else:
            Ypredict = [0] * len(Xtest)
        X_missing_reg.loc[X_missing_reg.loc[:, column_list[i]].isnull(), column_list[i]] = Ypredict
    X_missing_reg.index = index

    return X_missing_reg


def three_sigma(ser1):  # ser1表示传入DataFrame的某一列
    mean_value = ser1.mean()  # 求平均值
    std_value = ser1.std()  # 求标准差
    rule = (mean_value - 3 * std_value > ser1) | (ser1.mean() + 3 * ser1.std() < ser1)
    # 位于（u-3std,u+3std）区间的数据是正常的，不在这个区间的数据为异常的
    # 一旦发现有异常值，就标注为True，否则标注为False
    index = np.arange(ser1.shape[0])[rule]  # 返回异常值的位置索引
    outrange = ser1.iloc[index]  # 获取异常数据
    return outrange


def rf_fill(data_X_y, bmi_flag=False):
    vital_df = get_vital_data()
    lab_df = get_lab_data()
    # 获得连续特征，需要填充的特征列表
    continue_features = vital_df.columns.to_list() + lab_df.columns.to_list()
    print("共有{}个连续特征需要填充...".format(len(continue_features)))

    if bmi_flag:
        for i in range(3):  # bmi异常值处理
            bmi_index = three_sigma(data_X_y['bmi']).index
            # 修改bmi异常值为NaN
            for i in bmi_index:
                data_X_y.loc[i, 'bmi'] = np.nan

    data_cont_X = data_X_y[continue_features]
    data_y = data_X_y[y_label]
    data_X_y[continue_features] = missing_value_processing(data_cont_X, data_y)
    print("填充数据完毕...", bmi_flag)
    return data_X_y


if __name__ == '__main__':
    data_path = "/home/chenqinhai/code_eicu/my_lab/data/raw_file"
    y_label = 'aki_label'
    # youguang_df = pd.read_csv(r"D:\lab\my_lab_eicu\data\eicuData.csv", index_col=0)
    # youguang_raw_df = pd.read_csv(r"D:\lab\my_lab_eicu\raw_data.csv", index_col=0)

    all_data_file = os.path.join(data_path, "all_data_df_v5.csv")
    # all_data_df = concat_all_data()
    all_data_df = pd.read_csv(all_data_file, index_col=0)
    print("读取数据成功...", all_data_df.shape)

    version = 4
    all_data_df = pd.read_feather(os.path.join(data_path, f"all_data_df_v{version}.feather"))

    # 进行bmi异常处理 随机森林填充
    all_data_rf1_df = rf_fill(all_data_df, False)
    all_data_rf1_df.to_csv(os.path.join(data_path, "all_data_rf1_df_v5.csv"))

    # 进行bmi异常处理 随机森林填充
    all_data_rf2_df = rf_fill(all_data_df, True)
    all_data_rf2_df.to_csv(os.path.join(data_path, "all_data_rf2_df_v5.csv"))
