# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     process_diagnosis_table
   Description:   不进行任何处理
   Author:        cqh
   date:          2022/9/7 21:30
-------------------------------------------------
   Change Activity:
                  2022/9/7:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def get_patientUnitStayID_list():
    """
    获取有效患者ID list
    :return:
    """
    pat_df = pd.read_csv(os.path.join(data_path, "patientUnitStayID.csv"), index_col=0)
    print("all patients", pat_df.shape[0])
    return pat_df['patientunitstayid'].tolist()


def get_999_feature(feature_df, miss_rate=0.001):
    """
    输入一个特征名+count的df，筛选掉缺失率高达99.9的病人
    :return:
    """
    # 患者数量
    records_sum = 168399

    # 每个特征最低至少得有的记录数
    threshold_sum = int(miss_rate * records_sum)

    feature_df_new = feature_df[feature_df.iloc[:, 1] >= threshold_sum]
    print("at least records", threshold_sum)
    print("find {}/{} features".format(feature_df_new.shape[0], feature_df.shape[0]))

    return feature_df_new


def process_diagnosis_to_onehot():
    """
    处理并发症成为one_hot
    :return:
    """
    global data, index, data_df
    data = pd.read_csv(os.path.join(data_path, "diagnosis_raw.csv"), index_col=0)
    patient_ids = get_patientUnitStayID_list()
    # 处理符合patient_ids的数据 （减少行数）
    data = data[data['patientunitstayid'].isin(patient_ids)]
    print("筛选patient_ids的数据...")
    index = set(data.index.values)
    all_diagnosis_set = set()
    for pat_id in patient_ids:
        if pat_id not in index:
            continue

        temp_str = data.loc[pat_id, "diagnosisstring"]
        split_str = temp_str.split("|")
        for t in split_str:
            all_diagnosis_set.add(t)
    # 2430
    len_set = len(all_diagnosis_set)
    all_diagnosis_list = list(all_diagnosis_set)
    print("遍历完获得总共有多少并发症...")
    col_shape = (len(patient_ids), len_set)
    data_df = pd.DataFrame(data=np.zeros(col_shape), index=patient_ids, columns=all_diagnosis_list, dtype=np.int8)
    data_df.index = data_df.index.astype(np.int64)
    for pat_id in patient_ids:
        if pat_id not in index:
            continue

        temp_str = data.loc[pat_id, "diagnosisstring"]
        split_str = temp_str.split("|")
        for t in split_str:
            data_df.loc[pat_id, t] = 1
    data_df.to_csv(os.path.join(data_path, "diagnosis_one_hot.csv"))
    print("将各种并发症放入dataFrame表中...")


@DeprecationWarning
def process_lab_data_old():
    """
    原来提取数据的时候已经放入dataframe了,此方法过期了
    :return:
    """
    data = pd.read_csv(os.path.join(data_path, "lab_raw.csv"))
    patient_ids = pd.read_csv(os.path.join(data_path, "patientUnitStayID.csv"), index_col=0)[
        'patientunitstayid'].tolist()

    columns = data['labname'].unique().tolist()
    rows = len(patient_ids)
    cols = len(columns)
    res_df = pd.DataFrame(data=np.zeros((rows, cols)), index=patient_ids, columns=columns)

    for i in range(data.shape[0]):
        temp_data = data.iloc[i]
        pid = temp_data['patientunitstayid']
        labname = temp_data['labname']
        labvalue = temp_data['labresult']
        try:
            res_df.loc[pid, labname] = labvalue
        except Exception as e:
            print(pid, labname, labvalue, "error!")
            continue

    res_df.to_csv(os.path.join(data_path, "lab_data_df.csv"))

    return res_df


def process_age(age_df):
    """
    年龄特征设置为连续值
    :param age_df:
    :return:
    """
    # 更改特殊特征
    age_df = age_df.replace('> 89', 89)
    # 转化为数字
    age_df = pd.to_numeric(age_df, errors='coerce')

    # 处理缺失值
    mean = int(age_df.mean())
    age_df.fillna(int(mean), inplace=True)

    # 转化为int
    return age_df.astype(np.int8)


def process_demo_data():
    """
    处理demo数据
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

    # 处理相关细节
    # age特征
    demo_data['age'] = process_age(demo_data['age'])
    # 类型转换
    demo_data['aki_label'] = demo_data['aki_label'].astype(np.int64)
    demo_data['hospitalid'] = demo_data['hospitalid'].astype(np.int64)

    demo_data.to_csv(os.path.join(save_path, "demo_data_df.csv"))
    return demo_data


def process_diagnosis_data():
    """
    处理并发症数据
    :return:
    """
    dia_df = pd.read_csv(os.path.join(data_path, "diagnosis_one_hot.csv"), index_col=0)

    rows = dia_df.shape[0]
    print("处理缺失率高的特征...")

    # 处理缺失率过高的特征
    count_df = pd.DataFrame(columns=['feature', 'pat_count'])
    count_df['pat_count'] = rows - (dia_df == 0).sum()
    count_df['feature'] = dia_df.columns.tolist()
    new_feature_df = get_999_feature(count_df)
    new_feature_df.to_csv(os.path.join(save_path, "diag_feature_miss999.csv"), index=False)

    # 处理后的新数据
    print("保存为新数据...")
    new_columns = new_feature_df['feature']
    dia_df = dia_df[new_columns]
    dia_df.to_csv(os.path.join(save_path, "diag_data_df.csv"))
    return dia_df


def process_med_data():
    """
    med系列特征预处理
    :return:
    """
    med_data = pd.read_csv(os.path.join(data_path, "medication_raw.csv"), index_col=0)
    rows = med_data.shape[0]
    # 处理缺失率过高的特征
    count_df = pd.DataFrame(columns=['feature', 'pat_count'])
    count_df['pat_count'] = rows - (med_data == 0).sum()
    count_df['feature'] = med_data.columns.tolist()
    new_feature_df = get_999_feature(count_df)
    new_feature_df.to_csv(os.path.join(save_path, "med_feature_miss999.csv"), index=False)

    # 处理后的新数据
    new_columns = new_feature_df['feature']
    med_df = med_data[new_columns]

    # 处理inf问题
    for column in new_columns:
        isinf_df = np.isinf(med_df[column])
        if isinf_df.sum() > 0:
            max_value = med_df.loc[~isinf_df, column].sort_values(ascending=False).max()
            med_df.loc[isinf_df, column] = max_value
            print("process inf value", column, isinf_df.sum())

    med_df.to_csv(os.path.join(save_path, "med_data_df.csv"))
    print("save success!", med_df.shape)
    return med_df


def process_treatment_data():
    """
    手术预处理
    :return:
    """
    treatment_df = pd.read_csv(os.path.join(data_path, "treatment_raw.csv"), index_col=0)
    rows = treatment_df.shape[0]
    # 处理缺失率过高的特征
    count_df = pd.DataFrame(columns=['feature', 'pat_count'])
    count_df['pat_count'] = rows - (treatment_df == 0).sum()
    count_df['feature'] = treatment_df.columns.tolist()
    new_feature_df = get_999_feature(count_df)
    new_feature_df.to_csv(os.path.join(save_path, "treatment_feature_miss999.csv"), index=False)

    # 处理后的新数据
    new_columns = new_feature_df['feature']
    treatment_df = treatment_df[new_columns]

    # 处理inf问题
    for column in new_columns:
        isinf_df = np.isinf(treatment_df[column])
        if isinf_df.sum() > 0:
            max_value = treatment_df.loc[~isinf_df, column].sort_values(ascending=False).max()
            treatment_df.loc[isinf_df, column] = max_value
            print("process inf value", column, isinf_df.sum())

    treatment_df.to_csv(os.path.join(save_path, "treatment_data_df.csv"))
    return treatment_df


def process_lab_data():
    """
    处理lab数据
    :return:
    """
    lab_df = pd.read_csv(os.path.join(data_path, "lab_raw.csv"), index_col=0)
    rows = lab_df.shape[0]
    # 处理缺失率过高的特征
    count_df = pd.DataFrame(columns=['feature', 'pat_count'])
    count_df['pat_count'] = rows - (lab_df == 0).sum()
    count_df['feature'] = lab_df.columns.tolist()
    new_feature_df = get_999_feature(count_df)
    new_feature_df.to_csv(os.path.join(save_path, "lab_feature_miss999.csv"), index=False)

    # 处理后的新数据
    new_columns = new_feature_df['feature']
    lab_df = lab_df[new_columns]

    lab_df.replace(0, np.nan, inplace=True)

    # 处理缺失值， 将 nan 填充为均值
    # for column in list(lab_df.columns[lab_df.isna().sum() > 0]):
    #     mean_val = lab_df[column].mean()
    #     lab_df[column].fillna(mean_val, inplace=True)

    lab_df.to_csv(os.path.join(save_path, "lab_data_df.csv"))

    return lab_df


def process_sbpdbpbmi_vitals_data():
    """
    处理vital_signs_raw.csv 包含 sbp,dbp,bmi
    :return:
    """
    # ====================第二个 vital 表=================
    vital_df = pd.read_csv(os.path.join(data_path, "vitalaperiodic_raw.csv"), index_col=0)
    # 中值填充
    # for column in list(vital_df.columns[vital_df.isnull().sum() > 0]):
    #     mean_val = vital_df[column].mean()
    #     vital_df[column].fillna(mean_val, inplace=True)
    return vital_df


def process_other_vitals_data():
    """
    vital集成
    :return:
    """
    vital_df1 = pd.read_csv(os.path.join(data_path, "vital_raw.csv"), index_col=0)
    vital_df1.drop(['row_id'], axis=1, inplace=True)
    rows = vital_df1.shape[0]

    # 处理缺失率过高的特征
    count_df = pd.DataFrame(columns=['feature', 'pat_count'])
    count_df['pat_count'] = rows - (vital_df1.isna()).sum()
    count_df['feature'] = vital_df1.columns.tolist()
    new_feature_df = get_999_feature(count_df)
    new_feature_df.to_csv(os.path.join(save_path, "vital1_other_feature_miss999.csv"), index=False)

    # 处理后的新数据
    new_columns = new_feature_df['feature']
    vital_df1 = vital_df1[new_columns]
    vital_df1.index = vital_df1['patientunitstayid'].tolist()
    vital_df1.drop(['patientunitstayid'], axis=1, inplace=True)

    # 重新调整
    patientUnitStayId = get_patientUnitStayID_list()
    new_df = pd.DataFrame(index=patientUnitStayId)
    new_df = pd.concat([new_df, vital_df1], axis=1)

    # 中值填充
    # for column in new_df.columns[new_df.isna().sum() > 0]:
    #     mean_val = new_df[column].mean()
    #     new_df[column].fillna(mean_val, inplace=True)

    return new_df


def process_all_vitals_data():
    """
    两个vital表合并
    :return:
    """
    vital_df1 = process_sbpdbpbmi_vitals_data()
    vital_df2 = process_other_vitals_data()
    vital_df = pd.concat([vital_df1, vital_df2], axis=1)
    vital_df.to_csv(os.path.join(save_path, "vital_all_data_df.csv"))
    print("save success!")
    return vital_df


def process_columns_data(cur_data_df, prefix_name):
    """
    处理相关columns
    :param prefix_name: 特征前缀名
    :param cur_data_df:
    :return:
    """
    cur_columns = cur_data_df.columns.tolist()

    new_columns = []
    for idx, col in enumerate(cur_columns):
        new_columns.append(prefix_name + "_" + str(idx + 1))

    # 保存字典映射
    feature_dict_df = pd.DataFrame(columns=['train_column', 'origin_column'])
    feature_dict_df['train_column'] = new_columns
    feature_dict_df['origin_column'] = cur_columns
    feature_dict_df.to_csv(os.path.join(save_path, "{}_feature_dict.csv".format(prefix_name)), index=False)
    print("save dict", prefix_name, "success !")

    cur_data_df.columns = new_columns
    return cur_data_df


def process_all_feature_data_first():
    """
    第一次合并数据
    :return:
    """
    demo_df = process_demo_data()
    vital_df = process_all_vitals_data()

    lab_df = process_lab_data()
    lab_df = process_columns_data(lab_df, "lab")

    med_df = process_med_data()
    med_df = process_columns_data(med_df, "med")

    treat_df = process_treatment_data()
    treat_df = process_columns_data(treat_df, "px")

    diag_df = process_diagnosis_data()
    diag_df = process_columns_data(diag_df, "ccs")

    all_data_df = pd.concat([demo_df, vital_df, diag_df, lab_df, med_df, treat_df], axis=1)
    all_data_df.reset_index(inplace=True)
    all_data_df.to_feather(os.path.join(save_path, f"all_data_df_v{version}.feather"))
    print("save success!")
    return all_data_df


def process_all_feature_data():
    """
    合并所有数据，成为一个csv表格
    :return:
    """
    demo_df = pd.read_csv(os.path.join(save_path, "demo_data_df.csv"), index_col=0)
    vital_df = pd.read_csv(os.path.join(save_path, "vital_all_data_df.csv"), index_col=0)

    lab_df = pd.read_csv(os.path.join(save_path, "lab_data_df.csv"), index_col=0)
    lab_df = process_columns_data(lab_df, "lab")

    med_df = pd.read_csv(os.path.join(save_path, "med_data_df.csv"), index_col=0)
    med_df = process_columns_data(med_df, "med")

    treat_df = pd.read_csv(os.path.join(save_path, "treatment_data_df.csv"), index_col=0)
    treat_df = process_columns_data(treat_df, "px")

    diag_df = pd.read_csv(os.path.join(save_path, "diag_data_df.csv"), index_col=0)
    diag_df = process_columns_data(diag_df, "ccs")

    all_data_df = pd.concat([demo_df, vital_df, diag_df, lab_df, med_df, treat_df], axis=1)
    """
    v0 初始
    v1 columns替换,index 移出来
    """
    all_data_df.reset_index(inplace=True)
    # v4 不进行任何填充处理的数据
    all_data_df.to_feather(os.path.join(save_path, f"all_data_df_v{version}.feather"))
    print("save success!")
    return all_data_df


def get_feature_list(feature_flag):
    """
    获取特征映射
    :param feature_flag:
    :return:
    """
    feature_dict = pd.read_csv(os.path.join(save_path, "{}_feature_dict.csv".format(feature_flag)))
    return feature_dict['train_column'].values


def get_cont_feature():
    """
    获得连续特征
    :return:
    """
    demo_vital_feature_list = [
        'age', 'sbp',
        'dbp', 'paop', 'bmi', 'temperature', 'sao2', 'heartrate', 'respiration',
        'sao2.1', 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
        'pasystolic', 'padiastolic', 'pamean', 'cvp', 'etco2', 'st1', 'st2', 'st3', 'icp'
    ]

    lab_feature_list = get_feature_list("lab")
    med_feature_list = get_feature_list("med")
    treat_feature_list = get_feature_list("px")

    all_feature_list = np.concatenate(
        [demo_vital_feature_list, lab_feature_list, med_feature_list, treat_feature_list]).tolist()
    pd.DataFrame(data={"continue_feature": all_feature_list}).to_csv(os.path.join(save_path, "continue_feature.csv"),
                                                                     index=False)

    return all_feature_list


def normalize_data(all_Data):
    """
    对连续特征使用avg-if标准化
    :param all_Data:
    :return:
    """
    c_f = get_cont_feature()
    feature_happen_count = (all_Data.loc[:, c_f] != 0).sum(axis=0)
    feature_sum = all_Data.loc[:, c_f].sum(axis=0)
    feature_average_if = feature_sum / feature_happen_count
    all_Data.loc[:, c_f] = all_Data.loc[:, c_f] / feature_average_if

    # save
    all_Data.to_feather(normalize_file)

    return all_Data


def get_group_data(all_data, topK=5):
    group_data = all_data.groupby(hospital_id)
    # topK医院的ID列表
    topK_hos = group_data.count().sort_values(by=y_label, ascending=False).index.tolist()[:topK]
    # 保存前topK个中心的数据
    save_topK_data(topK_hos, all_data)

    return topK_hos


def save_topK_data(topK_hospital, all_Data):
    for idx in topK_hospital:
        temp_data = all_Data[all_Data[hospital_id] == idx]
        temp_data.reset_index(inplace=True)
        temp_data = temp_data.drop([hospital_id], axis=1)
        temp_path = group_data_file.format(idx)
        temp_data.to_feather(temp_path)
        print("save_success! - ", idx)
    print("done!")


def normal_process():
    """
    总入口。标准化并获得全局数据和各个中心的数据
    :return:
    """
    all_df = pd.read_feather(os.path.join(data_path, f"all_data_df_v{version}.feather"))
    all_norm_df = normalize_data(all_df)
    get_group_data(all_norm_df, 5)


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
    # 获得连续特征，需要填充的特征列表
    continue_features = get_cont_feature()
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
    y_label = "aki_label"
    hospital_id = "hospitalid"
    data_path = "/home/chenqinhai/code_eicu/my_lab/data/"

    save_path = "/home/chenqinhai/code_eicu/my_lab/data/raw_file"
    """
    version = 1 均值填充 lab没均值填充
    version = 2 中位数填充
    version = 3 均值填充 lab 均值填充
    version = 4 不进行任何填充
    """
    version = 4
    normalize_file = os.path.join(save_path, f"all_data_df_norm_v{version}.feather")
    # group_data_file = os.path.join(save_path, "all_data_df_norm_{}_v5.feather")
    # process_all_feature_data_first()

    all_data_df = pd.read_feather(os.path.join(save_path, f"all_data_df_v{version}.feather"))

    # 进行bmi异常处理 随机森林填充
    all_data_rf1_df = rf_fill(all_data_df, False)
    all_data_rf1_df.to_csv(os.path.join(save_path, f"all_data_rf1_df_v{version}.csv"))

    # 进行bmi异常处理 随机森林填充
    all_data_rf2_df = rf_fill(all_data_df, True)
    all_data_rf2_df.to_csv(os.path.join(save_path, f"all_data_rf2_df_v{version}.csv"))
