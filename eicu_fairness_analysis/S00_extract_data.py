# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S00_extract_data.py
   Description:   本地运行，不在server7上运行
   Author:        cqh
   date:          2023年4月24日
-------------------------------------------------
   Change Activity:
                  2023年4月24日
-------------------------------------------------
"""
import os

import pymysql
import psycopg2
import pandas as pd
from psycopg2 import extras as ex
import numpy as np
import joblib


def query(sql):
    cursor.execute(sql)
    columns_names = [dec[0] for dec in cursor.description]
    result = cursor.fetchall()
    # cursor.close()
    # 最终可以把查询出来的数据放在数据框中
    df = pd.DataFrame(result, columns=columns_names)
    return df


def insert(sql):
    cursor.execute(sql)
    print("Total updated rows:", cursor.rowcount)
    connection.commit()


def batch_insert():
    df = pd.read_csv('Demographics.csv')
    sql = 'insert into fl_eicu.demographics values %s'
    params_list = []
    for i in range(len(df)):
        patientunitstayid = str(df.loc[i, 'patientunitstayid'])
        if (not df.loc[i, 'gender'] is np.nan):
            sex = df.loc[i, 'gender']
        else:
            sex = ''
        if (not df.loc[i, 'age'] is np.nan):
            age = str(df.loc[i, 'age'])
        else:
            age = ''
        if (not df.loc[i, 'ethnicity'] is np.nan):
            race = df.loc[i, 'ethnicity']
        else:
            race = ''
        params_list.append((patientunitstayid, sex, age, race))
    ex.execute_values(cursor, sql, params_list, page_size=len(params_list))
    connection.commit()


def add_aki_column(table_name):
    sql = """
    ALTER TABLE {}
    ADD COLUMN IF NOT EXISTS aki_label integer DEFAULT -1,
    ADD COLUMN IF NOT EXISTS aki_offset integer,
    ADD COLUMN IF NOT EXISTS unitdischargeoffset integer;
    """.format(table_name)
    insert(sql)


def add_aki_column_mysql(table_name):
    add_sql = """
    ALTER TABLE {}
    ADD COLUMN aki_label integer DEFAULT -1,
    ADD COLUMN aki_offset integer,
    ADD COLUMN unitdischargeoffset integer
    """.format(table_name)
    insert(add_sql)


def insert_aki_info_to_table(table_name):
    """
    插入三列属性到各种表 aki_label, aki_offset, unitdischargeoffset
    :return:
    """
    # 添加属性列
    add_aki_column(table_name)

    # 更新1
    df = pd.read_csv(os.path.join(save_path, "scr_aki_label_offset.csv"))
    sql1 = " UPDATE {} SET aki_label = f.v1, aki_offset = f.v2 from (values %s) f(id, v1, v2 ) WHERE patientunitstayid = f.id ".format(
        table_name)
    df1 = df[df['label'] == 1]
    params_list = []
    for i in range(len(df1)):
        patientunitstayid = df1.iloc[i, 0]
        label = df1.iloc[i, 1]
        aki_offset = int(df1.iloc[i, 2])
        params_list.append((int(patientunitstayid), int(label), int(aki_offset)))

    ex.execute_values(cursor, sql1, params_list, page_size=len(params_list))
    updated_rows1 = cursor.rowcount
    connection.commit()
    print("1 update:", updated_rows1)

    # 更新0
    sql0 = " UPDATE {} SET aki_label = f.v1 from (values %s) f(id, v1 ) WHERE patientunitstayid = f.id ".format(
        table_name)
    df0 = df[df['label'] == 0]
    params_list = []
    for i in range(len(df0)):
        patientunitstayid = df0.iloc[i, 0]
        label = df0.iloc[i, 1]
        params_list.append((int(patientunitstayid), int(label)))
    ex.execute_values(cursor, sql0, params_list, page_size=len(params_list))
    update_rows0 = cursor.rowcount
    connection.commit()

    print("0 update:", update_rows0)

    # 更新离院
    update_row0_of = update_unitdischargeoffset(table_name)

    return updated_rows1, update_rows0, update_row0_of


def batch_update():
    df = pd.read_csv("data/processed_csv_result/scr_aki_label_offset.csv")
    sql1 = " UPDATE patient SET aki_label = f.v1, aki_offset = f.v2 from (values %s) f(id, v1, v2 ) WHERE patientunitstayid = f.id "
    df1 = df[df['label'] == 1]
    params_list = []
    for i in range(len(df1)):
        patientunitstayid = df1.iloc[i, 0]
        label = df1.iloc[i, 1]
        aki_offset = int(df1.iloc[i, 2])
        params_list.append((int(patientunitstayid), int(label), int(aki_offset)))
    ex.execute_values(cursor, sql1, params_list, page_size=len(params_list))
    connection.commit()

    sql0 = " UPDATE patient SET aki_label = f.v1 from (values %s) f(id, v1 ) WHERE patientunitstayid = f.id "
    df0 = df[df['label'] == 0]
    params_list = []
    for i in range(len(df0)):
        patientunitstayid = df0.iloc[i, 0]
        label = df0.iloc[i, 1]
        params_list.append((int(patientunitstayid), int(label)))
    ex.execute_values(cursor, sql0, params_list, page_size=len(params_list))
    connection.commit()

    unitdischargeoffset_df = pd.read_csv("processed_csv_result/aki_csv_result/unitDischargeOffset.csv")
    sql = " UPDATE patient SET unitdischargeoffset = f.v1  from (values %s) f(id, v1 ) WHERE patientunitstayid = f.id "
    params_list = []
    for i in range(len(unitdischargeoffset_df)):
        patientunitstayid = unitdischargeoffset_df.iloc[i, 0]
        unitdischargeoffset = unitdischargeoffset_df.iloc[i, 1]
        params_list.append((int(patientunitstayid), int(unitdischargeoffset)))
    ex.execute_values(cursor, sql, params_list, page_size=len(params_list))
    connection.commit()


def determine_AKI():
    SCR_TIME = 48 * 60
    # URINE_TIME = 60 * 6

    aki_dic = {}
    scr_df = pd.read_csv("data/processeed_csv_result/lab_offset.csv")
    scr_count, urine_count = 0, 0

    for patientunitstayid, group in scr_df.groupby('patientunitstayid'):
        aki_dic[patientunitstayid] = {}
        if (len(group) < 2):  # 若只有1条数据，则跳过
            aki_dic[patientunitstayid]['label'] = -1
            continue

        offset_queue = []
        result_queue = []
        offset_0 = group.iloc[0, 2]
        offset_queue.append(int(offset_0))
        result_queue.append(group.iloc[0, 3])
        flag = False
        for i in range(1, len(group)):
            offset_1 = int(group.iloc[i, 2])
            result = group.iloc[i, 3]
            while len(offset_queue) > 0 and offset_1 - offset_queue[0] > SCR_TIME:
                del offset_queue[0]
                del result_queue[0]
            offset_queue.append(offset_1)
            result_queue.append(result)
            result_min = min(result_queue)
            if result - result_min >= 0.299:
                aki_dic[patientunitstayid]['label'] = 1
                aki_dic[patientunitstayid]['offset'] = offset_1
                flag = True
                scr_count += 1
                break
        if not flag:
            aki_dic[patientunitstayid]['label'] = 0

    aki_dic_d = pd.DataFrame(aki_dic.values(), index=list(aki_dic.keys()), columns=['label', 'offset'])
    # aki_dic_d = pd.DataFrame(aki_dic.values(), index=list(aki_dic.keys()), columns=['label'])
    aki_dic_d.to_csv("./data/scr_aki_label_offset.csv")
    # batch_update()


def get_bmi():
    """
    获得身高体重
    :return:
    """
    sql = """
    select patientunitstayid, admissionheight, admissionweight
    from patient
    where aki_label = 1 or aki_label = 0
    order by patientunitstayid asc
    """
    res_df = query(sql)
    res_df['admissionheight'] = pd.to_numeric(res_df['admissionheight'], errors='coerce').fillna(0)
    res_df['admissionweight'] = pd.to_numeric(res_df['admissionweight'], errors='coerce').fillna(0)

    res_df['admissionheight'] = 0.01 * res_df['admissionheight']

    res_df['bmi'] = res_df['admissionweight'] / (res_df['admissionheight'] * res_df['admissionheight'])

    res_df.index = res_df['patientunitstayid'].tolist()
    res_df.drop(['patientunitstayid', "admissionheight", "admissionweight"], axis=1, inplace=True)
    res_df.replace(np.inf, np.NaN, inplace=True)
    res_df.replace(0, np.NaN, inplace=True)

    return res_df


def get_vitals_data():
    """
    vitalaperiodic表
    :return:
    """
    patient_ids = get_patientUnitStayID_list()

    sql = """
    SELECT
	vap.patientunitstayid,
	vap.noninvasivesystolic SBP,
	vap.noninvasivediastolic DBP,
	vap.paop 
    FROM
        vitalaperiodic vap,
        patient pt 
    WHERE
        vap.patientunitstayid = pt.patientunitstayid 
        AND ( ( aki_label = 0 AND unitdischargeoffset - observationOffset >= {} ) OR ( aki_label = 1 AND aki_offset - observationOffset >= {} ) ) 
    ORDER BY
        patientunitstayid ASC,
        observationoffset DESC
    """.format(offset_time, offset_time)

    id_pd = pd.DataFrame(index=patient_ids)
    spb_dbp = query(sql)
    ids = []
    sbp_list = []
    dbp_list = []
    paop_list = []
    for pid, group in spb_dbp.groupby('patientunitstayid'):
        ids.append(group.iloc[0, 0])
        sbp_list.append(group.iloc[0, 1])
        dbp_list.append(group.iloc[0, 2])
        paop_list.append(group.iloc[0, 3])
    id_pd.loc[ids, 'sbp'] = sbp_list
    id_pd.loc[ids, 'dbp'] = dbp_list
    id_pd.loc[ids, 'paop'] = paop_list

    # 获得BMI
    bmi_df = get_bmi()

    id_pd = pd.concat([id_pd, bmi_df], axis=1)

    # 均值填充
    # for column in list(id_pd.columns[id_pd.isnull().sum() > 0]):
    #     mean_val = id_pd[column].mean()
    #     id_pd[column].fillna(mean_val, inplace=True)

    id_pd.to_csv(os.path.join(save_path, "vitalaperiodic_raw.csv"))

    print("save success!", id_pd.shape)
    return id_pd


def get_valid_patient_id():
    """
    获得到有效的患者id（有aki_label = 0/1)
    :return:
    """
    sql = """
    select patientunitstayid
    FROM patient
    where aki_label = 1 or aki_label = 0
    order by patientunitstayid asc
    """
    file = os.path.join(save_path, "patientUnitStayID.csv")
    res_df = query(sql)
    res_df.to_csv(file)
    print("get", res_df.shape)
    return res_df.shape


def get_patient_unitdischargeoffset():
    sql = """
    select patientunitstayid, unitdischargeoffset
    FROM patient
    where aki_label = 1 or aki_label = 0
    """
    file = os.path.join(save_path, "patientUnitdischargeoffset.csv")
    res_df = query(sql)
    res_df.to_csv(file)
    print("get", res_df.shape)
    return res_df.shape


def update_unitdischargeoffset(table_name):
    # 更新aki=0时的离院时间
    unitdischargeoffset_df = pd.read_csv(os.path.join(save_path, "patientUnitdischargeoffset.csv"))
    sql = """
    UPDATE {} SET unitdischargeoffset = f.v1  from (values %s) f(id, v1 ) WHERE patientunitstayid = f.id
    """.format(table_name)

    params_list = []
    for i in range(len(unitdischargeoffset_df)):
        patientunitstayid = unitdischargeoffset_df.iloc[i, 1]
        unitdischargeoffset = unitdischargeoffset_df.iloc[i, 2]
        params_list.append((int(patientunitstayid), int(unitdischargeoffset)))
    ex.execute_values(cursor, sql, params_list, page_size=len(params_list))
    update_rows0 = cursor.rowcount
    connection.commit()

    print("update_unitdischargeoffset", update_rows0)
    return update_rows0


def get_column_by_table(table_name):
    sql = """
    select COLUMN_NAME 
    from information_schema.COLUMNS
    where table_name = '{}'
    """.format(table_name)

    res_df = query(sql)
    print("get_columns", res_df.shape)
    return res_df['column_name'].tolist()


def get_patientUnitStayID_list():
    """
    获取有效患者ID list
    :return:
    """
    pat_df = pd.read_csv(os.path.join(save_path, "patientUnitStayID.csv"), index_col=0)
    print("all patients", pat_df.shape[0])
    return pat_df['patientunitstayid'].tolist()


def get_vitalssigns_data2():
    """
    获取vitals多个特征
    by 海
    :return:
    """
    sql = """
    SELECT
        * 
    FROM
        (
        SELECT
            vp.patientunitstayid,
            ROW_NUMBER ( ) OVER ( PARTITION BY vp.patientunitstayid ORDER BY vp.observationoffset DESC ) row_id,
            temperature,
            sao2,
            heartrate,
            respiration,
            sao2,
            systemicsystolic,
            systemicdiastolic,
            systemicMean,
            paSystolic,
            paDiastolic,
            paMean,
            cvp,
            etCo2,
            st1,
            st2,
            st3,
            ICP 
        FROM
            vitalperiodic vp,
            patient pt 
        WHERE
            vp.patientunitstayid = pt.patientunitstayid 
            AND (
                ( pt.aki_label = 0 AND pt.unitdischargeoffset - vp.observationOffset >= 1440 ) 
                OR ( pt.aki_label = 1 AND pt.aki_offset - vp.observationOffset >= 1440 ) 
            ) 
        ) AS tab
    WHERE
        row_id = 1 
    ORDER BY patientunitstayid ASC
    """

    res_df = query(sql)
    res_df.to_csv(os.path.join(save_path, "vital_raw.csv"))
    print("save success!")



def get_vitalsigns_data():
    """
    by guo
    :return:
    """
    patient_ids = get_patientUnitStayID_list()

    # 特征数
    vitalsigns = ['temperature', 'heartrate', 'respiration', 'sao2', 'systemicsystolic', 'systemicdiastolic',
                  'systemicMean', 'paSystolic', 'paDiastolic', 'paMean'
        , 'cvp', 'etCo2', 'st1', 'st2', 'st3', 'ICP']

    mask = np.zeros((len(patient_ids), len(vitalsigns)), dtype='uint8')
    res_df = pd.DataFrame(data=mask, index=patient_ids)

    for idx, vital in enumerate(vitalsigns):
        sql = """
            select * from(
            select v.patientUnitStayID,{},observationOffset
            from vitalperiodic v,patient p
            where v.patientUnitStayID=p.patientUnitStayID and aki_label=1 and {} is not null and aki_offset-observationOffset>={}
            union
            select v.patientUnitStayID,{},observationOffset
            from vitalperiodic v,patient p
            where v.patientUnitStayID=p.patientUnitStayID and aki_label=0 and {} is not null and unitdischargeoffset-observationOffset>={}
            ) as t
            order by patientUnitStayID asc,observationOffset desc
            """.format(vital, vital, offset_time, vital, vital, offset_time)
        df = query(sql)

        ids = []
        vals = []
        for patientunitstayid, group in df.groupby('patientunitstayid'):
            ids.append(group.iloc[0, 0])
            vals.append(group.iloc[0, 1])
        res_df.loc[ids, vital] = vals
        print("[", idx, "] insert vital - ", vital, " - success!", vals[0])

    res_df.to_csv(os.path.join(save_path, "vitalsigns_raw.csv"))
    print("save success!", res_df.shape)
    return res_df


def get_diagnosis_feature():
    """
    选择并发症特征
    :return:
    """
    sql = """
    select ds.diagnosisstring, count(ds.patientunitstayid) pat_count
    from diagnosis ds, patient pt
    where ds.patientunitstayid = pt.patientunitstayid and (aki_label = 1 or aki_label = 0)
    GROUP BY ds.diagnosisstring
    ORDER BY pat_count desc
    """
    diag_feature = query(sql)
    diag_feature = get_999_feature(diag_feature)

    diag_feature.to_csv(os.path.join(save_path, "diagnosis_feature.csv"))
    print("save diagnosis feature success!")
    return diag_feature


def get_diagnosis_data():
    """
    diagnosis表的并发症信息
    :return:
    """
    sql = """
    select patientunitstayid, diagnosisstring
    from diagnosis d
    """
    diagnosis = query(sql)
    diagnosis.to_csv(os.path.join(save_path, "diagnosis_raw.csv"))


def get_demo_data():
    """
    获取病人基本信息
    :return:
    """
    sql = """
    select patientunitstayid, gender, age, ethnicity, hospitalid, admissionheight, admissionweight, aki_label, apacheadmissiondx
    from patient
    where aki_label = 0 or aki_label = 1
    order by patientunitstayid asc
    """
    patient_demo = query(sql)
    patient_demo.to_csv(os.path.join(save_path, "demographics_raw_4-24.csv"))
    return patient_demo

def get_lab_feature():
    """
    统计每个lab特征的病人数量
    :return:
    """
    threshold = 24 * 60
    sql = """
        select labname, count(patientunitstayid) as patient_count
        from lab
        where (aki_label = 1 and aki_offset - labresultoffset >= {}) or (aki_label = 0 and unitdischargeoffset - labresultoffset >= {})
        group by labname
        order by patient_count desc
        """.format(threshold, threshold)
    df = query(sql)

    # 筛选999特征
    df = get_999_feature(df)
    df.to_csv(os.path.join(save_path, "lab_feature.csv"))
    print("save success!", df.shape)
    # 筛选
    return df['labname'].tolist()


def get_lab_data():
    """
    获取lab data
    :return:
    """
    print("start get lab data...")
    patient_ids = get_patientUnitStayID_list()

    # 获取lab所有信息，可以截取一部分
    all_labs = get_lab_feature()

    print("labs nums", len(all_labs))
    mask = np.zeros((len(patient_ids), len(all_labs)), dtype='uint8')
    res_df = pd.DataFrame(data=mask, index=patient_ids, columns=all_labs)
    for idx, lab in enumerate(all_labs):
        if "'" in lab:
            lab = lab.replace("'", "''")
        # 把aki_label = 1 和 aki_label = 0 的数据全部拿出来，需要取最近的（最大）的offset记录
        select_group_sql = """
        select patientunitstayid, labresult, labresultoffset
        from lab
        where labname='{}' and ((aki_label = 1 and aki_offset - labresultoffset >= {}) or (aki_label = 0 and unitdischargeoffset - labresultoffset >= {}))
        order by patientunitstayid asc
        """.format(lab, offset_time, offset_time)

        try:
            select_df = query(select_group_sql)
        except Exception as exec:
            print(lab, "something error...", exec)
            connection.rollback()
            continue

        ids = []
        results = []
        for pid, group in select_df.groupby("patientunitstayid"):
            group.sort_values(by=['labresultoffset'], ascending=False, inplace=True)
            ids.append(group.iloc[0, 0])
            results.append(group.iloc[0, 1])
        res_df.loc[ids, lab] = results
        print("[", idx, "] insert lab - ", lab, " - success!")

    res_df.fillna(np.NaN, inplace=True)
    res_df.to_csv(os.path.join(save_path, "lab_raw.csv"))
    print("save success!", res_df.shape)


def select_px_data():
    """
    获取px数据
    :return:
    """
    patient_ids = pd.read_csv(os.path.join(save_path, "patientUnitStayID.csv"), index_col=0)[
        'patientunitstayid'].tolist()
    print("all patients", len(patient_ids))
    offset_time = 24 * 60
    print("start...")
    # 获取lab所有信息，可以截取一部分
    lab_df = get_lab_feature()
    all_labs = lab_df['labname'].tolist()
    print("labs nums", len(all_labs))
    res_df = pd.DataFrame(index=patient_ids)
    for lab in all_labs:
        if "'" in lab:
            lab.replace("'", "''")
        # 把aki_label = 1 和 aki_label = 0 的数据全部拿出来，需要取最近的（最大）的offset记录
        select_group_sql = """
        select patientunitstayid, labresult, labresultoffset
        from lab
        where labname='{}' and ((aki_label = 1 and aki_offset - labresultoffset >= {}) or (aki_label = 0 and unitdischargeoffset - labresultoffset >= {}))
        order by patientunitstayid asc, labresultoffset desc 
        """.format(lab, offset_time, offset_time)

        try:
            select_df = query(select_group_sql)
        except Exception as ex:
            print(lab, "something error...", ex)
            continue

        ids = []
        results = []
        for pid, group in select_df.groupby("patientunitstayid"):
            ids.append(group.iloc[0, 0])
            results.append(group.iloc[0, 1])
        res_df.loc[ids, lab] = results
        print("insert lab - ", lab, " - success!")

    res_df.to_csv(os.path.join(save_path, "lab_raw.csv"), index=False)
    print("save success!", res_df.shape)


def procedures():
    id_pd = pd.read_csv("processed_csv_result/patientUnitStayID.csv", index_col=0)
    procedures = ['transfusion', 'CT scan', 'x-ray', 'insertion', 'injection', 'albumin', 'infus', 'culture',
                  'echocardiography']
    for p in procedures:
        cursor = connection.cursor()
        sql = """
        select * from 
        (
        select t.patientUnitStayID,treatmentOffset,treatmentString,aki_label,aki_offset,unitdischargeoffset 
        from treatment t,fl_eicu.patient p 
        where t.patientUnitStayID=p.patientUnitStayID and treatmentString like " + "'%" + p + "%'" + " and aki_label=1 and aki_offset-treatmentOffset>=2880 
        union 
        select t.patientUnitStayID,treatmentOffset,treatmentString,aki_label,aki_offset,unitdischargeoffset 
        from treatment t,fl_eicu.patient p 
        where t.patientUnitStayID=p.patientUnitStayID and treatmentString like " + "'%" + p + "%'" + " and aki_label=0 and unitdischargeoffset-treatmentOffset>=2880 
        ) as t 
        order by patientUnitStayID asc,treatmentOffset desc
        """
        df = query(sql)

        ids = []
        label = []
        for patientunitstayid, group in df.groupby('patientunitstayid'):
            ids.append(group.iloc[0, 0])
            label.append(str(1))
        id_pd.loc[ids, p] = label
    id_pd = id_pd.fillna(0)
    id_pd.drop(columns=['placeholder'], inplace=True)
    id_pd.to_csv("./processed_csv_result/procedures_csv_result/scr_procedures.csv")
    print(id_pd)


def get_treatment_feature():
    """
    获取treatment有关特征
    :return:
    """
    sql = """
    select tt.treatmentstring, count(tt.patientunitstayid) p_count
    from treatment tt, patient pt
    WHERE
            tt.patientunitstayid = pt.patientunitstayid and (
            (pt.aki_label = 1 and pt.aki_offset - tt.treatmentoffset >= 0) 
            or 
            (pt.aki_label = 0 and pt.unitdischargeoffset - tt.treatmentoffset >= 0))
    GROUP BY tt.treatmentstring
    ORDER BY p_count desc
    """
    df = query(sql)
    df = get_999_feature(df)
    df.to_csv(os.path.join(save_path, "treatment_feature.csv"))
    print("save success!", df.shape[0])
    return df['treatmentstring'].tolist()


def get_treatment_data():
    """
    查询获得手术的符合数据
    :return:
    """
    print("start get treatment...")
    patient_ids = pd.read_csv(os.path.join(save_path, "patientUnitStayID.csv"), index_col=0)[
        'patientunitstayid'].tolist()

    # 获取手术特征
    treatment_list = get_treatment_feature()

    mask = np.zeros((len(patient_ids), len(treatment_list)), dtype='uint8')
    res_df = pd.DataFrame(data=mask, index=patient_ids, columns=treatment_list)
    for idx, treat in enumerate(treatment_list):
        # 获取手术最近的距离
        sql = """
        select patientunitstayid, max(tab.dist) max_offset  from (
            (
                select tt.patientunitstayid, (pt.aki_offset - tt.treatmentoffset) dist
                from treatment tt, patient pt
                where 
                        treatmentstring = '{}' 
                        and
                        tt.patientunitstayid = pt.patientunitstayid 
                        and (
                            pt.aki_label = 1 and pt.aki_offset - tt.treatmentoffset >= 0
                        )
            )
            union
            (
                select tt.patientunitstayid, (pt.unitdischargeoffset - tt.treatmentoffset) dist
                from treatment tt, patient pt
                where 
                        treatmentstring = '{}' 
                        and
                        tt.patientunitstayid = pt.patientunitstayid 
                        and (
                            pt.aki_label = 0 and pt.unitdischargeoffset - tt.treatmentoffset >= 0
                        )
            )
        ) as tab
        group by tab.patientunitstayid
        order by tab.patientunitstayid asc
        """.format(treat, treat)

        try:
            sql_df = query(sql)
        except Exception as err:
            print(treat, "something error...", err)
            connection.rollback()
            continue

        ids = sql_df['patientunitstayid'].tolist()
        res = (1.0 / (sql_df['max_offset'] / 60.0 / 24.0)).tolist()

        res_df.loc[ids, treat] = res
        print("[", idx, "] insert treatment - ", treat, " - success!", res[0])

    res_df.to_csv(os.path.join(save_path, "treatment_raw.csv"))
    print("save success!", res_df.shape)

    return res_df


def get_treatment_str_feature():
    """
    获得手术特征
    :return:
    """
    sql = """
    select tt.treatmentstring, count(tt.patientunitstayid) pat_count
    from treatment tt
    GROUP BY tt.treatmentstring
    order by pat_count desc
    """
    res_df = query(sql)

    print("shape", res_df.shape)
    # save
    res_df.to_csv(os.path.join(save_path, "treatment_feature.csv"))

    return res_df['treatmentstring'].tolist()[:1000]


def get_medication_feature():
    """
    获得药物特征
    :return:
    """
    sql = """
    SELECT drugname, count(med.patientunitstayid) pat_count
    FROM
        medication med,
        patient pt 
    WHERE
        med.patientunitstayid = pt.patientunitstayid 
        AND (
            ( pt.aki_label = 1 AND pt.aki_offset - med.drugstartoffset >= 0 ) 
            OR ( aki_label = 0 AND pt.unitdischargeoffset - med.drugstartoffset >= 0 ) 
        ) 
    GROUP BY
        med.drugname
    ORDER BY pat_count DESC
    """
    res_df = query(sql)
    # 去除null列
    res_df.dropna(inplace=True)
    # 筛选999列
    res_df = get_999_feature(res_df)
    # save
    res_df.to_csv(os.path.join(save_path, "medication_feature.csv"))
    return res_df['drugname'].tolist()


def get_medication_data():
    """
    查询获得药物的符合数据
    :return:
    """
    print("start get medication...")
    patient_ids = get_patientUnitStayID_list()

    # 获取药物特征
    medication_list = get_medication_feature()

    mask = np.zeros((len(patient_ids), len(medication_list)), dtype='uint8')
    res_df = pd.DataFrame(data=mask, index=patient_ids, columns=medication_list)

    for idx, treat in enumerate(medication_list):
        if "'" in treat:
            treat = treat.replace("'", "''")
        # 获取手术最近的距离
        sql = """
        select patientunitstayid, max(tab.dist) max_offset  from (
            (
                select med.patientunitstayid, (pt.aki_offset - med.drugstartoffset) dist
                from medication med, patient pt
                where 
                        med.drugname = '{}'
                        and
                        pt.patientunitstayid = med.patientunitstayid 
                        and (
                            pt.aki_label = 1 and pt.aki_offset - med.drugstartoffset >= 0
                        )
            )
            union
            (
                select med.patientunitstayid, (pt.unitdischargeoffset - med.drugstartoffset) dist
                from medication med, patient pt
                where 
                        med.drugname = '{}'
                        and
                        pt.patientunitstayid = med.patientunitstayid 
                        and (
                            pt.aki_label = 0 and pt.unitdischargeoffset - med.drugstartoffset >= 0
                        )
            )
        ) as tab
        group by tab.patientunitstayid
        order by tab.patientunitstayid asc
        """.format(treat, treat)

        try:
            sql_df = query(sql)
        except Exception as err:
            print(treat, "something error...", err)
            connection.rollback()
            continue

        ids = sql_df['patientunitstayid'].tolist()
        # 按天计算
        res = (1.0 / (sql_df['max_offset'] / 60.0 / 24.0)).tolist()
        res = [round(x, 2) for x in res]
        res_df.loc[ids, treat] = res
        print("[", idx, "] insert medication - ", treat, " - success!", res[0])

    res_df.to_csv(os.path.join(save_path, "medication_raw.csv"))
    print("save success!", res_df.shape)

    return res_df


def get_999_feature(feature_df):
    """
    输入一个特征名+count的df，筛选掉缺失率高达99.9的病人
    :return:
    """
    # 患者数量
    records_sum = 168399
    miss_rate = 0.001

    # 每个特征最低至少得有的记录数
    threshold_sum = int(miss_rate * records_sum)

    feature_df_new = feature_df[feature_df.iloc[:, 1] >= threshold_sum]
    print("at least records", threshold_sum)
    print("find {}/{} features".format(feature_df_new.shape[0], feature_df.shape[0]))

    return feature_df_new


def get_offset_after_patient():
    """
    获取在出ICU后患的病人ID
    :return:
    """
    sql = """
    select patientunitstayid
    from patient
    where aki_label = 1 and unitdischargeoffset < aki_offset
    """
    res_df = query(sql)
    res_df.to_csv(os.path.join(save_path, "offset_patients.csv"), index=False)
    print("save success!")


if __name__ == "__main__":
    save_path = f"data/processeed_csv_result"
    connection = psycopg2.connect(
        database="eicu",
        user="eicu",
        password="2022",
        host="172.18.48.162",
        port="5407"
    )

    # 24小时之前
    offset_time = 24 * 60

    # connection = psycopg2.connect(
    #     database="EICU",
    #     user="postgres",
    #     password="root",
    #     host="localhost",
    #     port="5432"
    # )
    # connection = pymysql.connect(
    #         database="eicu_data",
    #         user="root",
    #         password="haige123",
    #         host="localhost",
    #         port=3306
    #     )

    cursor = connection.cursor()

    # drg相关
    # demo相关
    demo_df = get_demo_data()

    # vitals相关
    # vital_df = get_vitals_data()
    # vitalsigns_df = get_vitalssigns_data2()
    # med相关
    # med_df = get_medication_data()

    # treatment相关
    # treat_df = get_treatment_data()

    # lab相关
    # lab_df = get_lab_data()

    # diagnosis Comorbidity相关
    # diag_df = get_diagnosis_data()

    # get_offset_after_patient()
    print("ok!")
