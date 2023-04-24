# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: liukang
"""
from scipy import stats
from decimal import Decimal
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


#auc_drg=pd.DataFrame()
#auc_total=pd.DataFrame()
brier_score_record = pd.DataFrame()
brier_score_se_record = pd.DataFrame()
brier_score_compare_p_record = pd.DataFrame()
disease_drg_df=pd.read_csv('/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/disease_top_20_no_drg_full_name.csv', encoding='gbk')
disease_list = disease_drg_df[['Drg','Chinese']]
#drg_list=pd.read_csv('/home/liukang/Doc/drg_list.csv')
result=pd.read_csv("/home/liukang/Doc/calibration/test_result_10_No_Com.csv")
#result=pd.read_csv("/home/liukang/Doc/calibration/test_result_10_No_Com_without_top_subgroup_AKI50.csv")
#result['drg']=drg_list.iloc[:,0]
group_num = 10
round_num = 2000

#compute calibration in global patients
# plt.figure(figsize=(12, 7.4))
# #ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

#ax2 = plt.subplot2grid((3, 1), (2, 0))

#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)

#ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
# plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

model_list = result.columns.tolist()[1:4]

"""
对于GM,SM,PMTL
通过校准度曲线函数计算obs,pre (等比例分箱5个）
计算brier_score， 放入到General
plt 精确显示小数点，然后绘图
"""
# for result_type in result_list:
#
#     observation, prediction = calibration_curve(result['Label'], result[result_type], n_bins=group_num, strategy='quantile')
#
#     brier_score = np.mean(np.square(np.array(observation - prediction)))
#     brier_score_record.loc['General',result_type] = brier_score
#     brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")
#
#     #ax1.plot(prediction, observation, "s-", label="%s = %s" % (result_type, brier_score_short))
#     plt.plot(prediction, observation, "s-", label="%s (%s)" % (result_type, brier_score_short))
#
#     #ax2.hist(result[result_type], range=(0, 1), bins=10, label=result_type, histtype="step", lw=2)

#ax1.set_ylabel("Fraction of positives",fontsize=20)
#ax1.set_xlabel("Mean predicted value",fontsize=20)
#ax1.set_ylim([-0.05, 1.05])
#ax1.legend(loc="lower right",fontsize=15)
#ax1.set_title('General patients',fontsize=20)

# plt.ylabel("Fraction of positives",fontsize=20)
# plt.xlabel("Mean predicted value",fontsize=20)
# plt.ylim([-0.05, 1.05])
# plt.legend(loc="lower right",fontsize=15)

#ax2.set_xlabel("Mean predicted value",fontsize=20)
#ax2.set_ylabel("Count",fontsize=20)
#ax2.legend(loc="upper center", ncol=2, fontsize=15)

# plt.show()

# plt.tight_layout()
# plt.savefig("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/calibration_quantile_global.png")
# print("save global success!")
# plt.savefig("/home/liukang/Doc/calibration/result/calibration_quantile_global.eps",format='eps')
#plt.show()

boostrap_brier = pd.DataFrame()
"""
通过重复round_num次，随机抽取可重复的100%样例，计算calibration的obs、pre，然后计算brier分数。
计算标准差，P值
"""
# for i in range(round_num):
#     # 跑2k次
#     sample_result = result.sample(frac=1,replace=True)
#
#     for result_type in result_list:
#          observation, prediction = calibration_curve(sample_result['Label'], sample_result[result_type], n_bins=group_num, strategy='quantile')
#          brier_score = np.mean(np.square(np.array(observation - prediction)))
#          boostrap_brier.loc[i,result_type] = brier_score
# for result_type in result_list:
#     brier_score_se_record.loc['General',result_type] = np.std(boostrap_brier[result_type])
# # 计算P值
# for i in range(len(result_list)):
#     #first_result = boostrap_brier.loc[:,result_list[i]].values
#     for j in range(i+1,len(result_list)):
#         #second_result = boostrap_brier.loc[:,result_list[j]].values
#         brier_boostrap_diff = pd.DataFrame()
#         brier_boostrap_diff['diff'] = boostrap_brier[result_list[i]] - boostrap_brier[result_list[j]]
#         brier_diff_below_0 = brier_boostrap_diff.loc[:,'diff'] < 0
#         sum_below_0 = np.sum(brier_diff_below_0)
#         if sum_below_0 < (round_num/2):
#             boostrap_p = 2*(sum_below_0 / round_num)
#         else:
#             boostrap_p = 2*((round_num - sum_below_0) / round_num)
#         brier_score_compare_p_record.loc['General','{}_V_{}'.format(result_list[i],result_list[j])] = boostrap_p


"""
绘制每个亚组的calibration图
"""
model_list = ['GM', 'PMTL']
total_drg_result = pd.DataFrame()
plt.rc('font',family='SimHei')

fig, axes = plt.subplots(5, 4, constrained_layout=True, figsize=(20, 30))
plt.subplots_adjust(left=0.5, bottom=0.5, wspace=0.6, hspace=0.6) #调整子图间距

axes_list = []
disease_index = 0

for row in range(axes.shape[0]):
    for col in range(axes.shape[1]):
        result_true = result.loc[:, 'Drg'] == disease_list.iloc[disease_index, 0]
        meaningful_result = result.loc[result_true]
        total_drg_result = pd.concat([total_drg_result, meaningful_result])

        # axes[row][col].set_xticks(fontsize=14)
        # axes[row][col].set_yticks(fontsize=14)
        axes[row][col].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for result_type in model_list:
            observation, prediction = calibration_curve(meaningful_result['Label'], meaningful_result[result_type],n_bins=5, strategy='quantile')
            brier_score = np.mean(np.square(np.array(observation - prediction)))
            brier_score_record.loc[disease_list.iloc[disease_index, 1], result_type] = brier_score
            brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
            axes[row][col].plot(prediction, observation, "s-", label="%s (%s)" % (result_type, brier_score_short))

        axes[row][col].set_ylim([-0.05, 1.05])
        axes[row][col].legend(loc="lower right")
        axes[row][col].set_title('{}'.format(disease_list.iloc[disease_index, 1]), fontsize=20)

        print(f"add subplot[{row}, {col}] success!, subgroup name: [{disease_list.iloc[disease_index, 1]}]")
        # 下一个亚组
        disease_index += 1

plt.tight_layout()
plt.savefig("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/calibration_quantile_Drg_top20_4-11.png", dpi=200)
plt.show()

print("done!")
# for disease_num in range(disease_list.shape[0]):
#
#     # plt.figure(figsize=(8, 6))
#
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#
#     plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#
#     result_true=result.loc[:,'Drg']==disease_list.iloc[disease_num,0]
#     meaningful_result=result.loc[result_true]
#     total_drg_result = pd.concat([total_drg_result,meaningful_result])
#
#     for result_type in result_list:
#
#         observation, prediction = calibration_curve(meaningful_result['Label'], meaningful_result[result_type], n_bins=5, strategy='quantile')
#
#         brier_score = np.mean(np.square(np.array(observation - prediction)))
#         brier_score_record.loc[disease_list.iloc[disease_num,1],result_type] = brier_score
#         brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")
#
#         plt.plot(prediction, observation, "s-", label="%s (%s)" % (result_type, brier_score_short))
#
#     #ax1.set_ylabel("Fraction of positives",fontsize=20)
#     #ax1.set_xlabel("Mean predicted value",fontsize=20)
#     #ax1.set_ylim([-0.05, 1.05])
#     #ax1.legend(loc="lower right",fontsize=15)
#     #ax1.set_title('{}'.format(disease_list.iloc[disease_num,1]),fontsize=20)
#
#
#     plt.ylim([-0.05, 1.05])
#     plt.legend(loc="lower right",fontsize=14)
#     plt.title('{}'.format(disease_list.iloc[disease_num,1]),fontsize=18)
#     # plt.ylabel("Fraction of positives", fontsize=20)
#     # plt.xlabel("Mean predicted value", fontsize=20)
#
#     #ax2.set_xlabel("Mean predicted value",fontsize=20)
#     #ax2.set_ylabel("Count",fontsize=20)
#     #ax2.legend(loc="upper center", ncol=2,fontsize=15)
#     plt.show()
#
#     plt.tight_layout()
    # plt.savefig("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/calibration_quantile_Drg{}.png".format(disease_list.iloc[disease_num,0]))
    # print("save Drg{} success!".format(disease_list.iloc[disease_num,0]))
    # plt.savefig("/home/liukang/Doc/calibration/result/calibration_quantile_Drg{}.eps".format(disease_list.iloc[disease_num,0]),format='eps')
    #plt.show()



    # boostrap_brier = pd.DataFrame()
    # for i in range(round_num):
    #
    #     sample_result = meaningful_result.sample(frac=1,replace=True)
    #
    #     for result_type in result_list:
    #
    #         observation, prediction = calibration_curve(sample_result['Label'], sample_result[result_type], n_bins=5, strategy='quantile')
    #         brier_score = np.mean(np.square(np.array(observation - prediction)))
    #         boostrap_brier.loc[i,result_type] = brier_score
    #
    # for result_type in result_list:
    #
    #     brier_score_se_record.loc[disease_list.iloc[disease_num,1],result_type] = np.std(boostrap_brier[result_type])
    #
    # for i in range(len(result_list)):
    #
    #     #first_result = boostrap_brier.loc[:,result_list[i]].values
    #
    #     for j in range(i+1,len(result_list)):
    #         #second_result = boostrap_brier.loc[:,result_list[j]].values
    #         brier_boostrap_diff = pd.DataFrame()
    #         brier_boostrap_diff['diff'] = boostrap_brier[result_list[i]] - boostrap_brier[result_list[j]]
    #         brier_diff_below_0 = brier_boostrap_diff.loc[:,'diff'] < 0
    #         sum_below_0 = np.sum(brier_diff_below_0)
    #
    #         if sum_below_0 < (round_num/2):
    #
    #             boostrap_p = 2 * (sum_below_0 / round_num)
    #
    #         else:
    #
    #             boostrap_p = 2 * ((round_num - sum_below_0) / round_num)
    #
    #         brier_score_compare_p_record.loc[disease_list.iloc[disease_num,1],'{}_V_{}'.format(result_list[i],result_list[j])] = boostrap_p



# #compute calibration in all high-risk patients
# plt.figure(figsize=(12, 7.4))
# #ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#
# #ax2 = plt.subplot2grid((3, 1), (2, 0))
#
# #plt.xticks(fontsize=15)
# #plt.yticks(fontsize=15)
#
# #ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
# plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#
# for result_type in result_list:
#
#     observation, prediction = calibration_curve(total_drg_result['Label'], total_drg_result[result_type], n_bins=group_num, strategy='quantile')
#
#     brier_score = np.mean(np.square(np.array(observation - prediction)))
#     brier_score_record.loc['All Top-20',result_type] = brier_score
#     brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")
#
#     #ax1.plot(prediction, observation, "s-", label="%s = %s" % (result_type, brier_score_short))
#     plt.plot(prediction, observation, "s-", label="%s = %s" % (result_type, brier_score_short))
#     #ax2.hist(total_drg_result[result_type], range=(0, 1), bins=10, label=result_type, histtype="step", lw=2)
#
# #ax1.set_ylabel("Fraction of positives",fontsize=20)
# #ax1.set_xlabel("Mean predicted value",fontsize=20)
# #ax1.set_ylim([-0.05, 1.05])
# #ax1.legend(loc="lower right",fontsize=15)
#
# plt.ylabel("Fraction of positives",fontsize=20)
# plt.xlabel("Mean predicted value",fontsize=20)
# plt.ylim([-0.05, 1.05])
# plt.legend(loc="lower right",fontsize=15)
#
# #ax1.set_title('Top-20 high risk admissions',fontsize=20)
#
# #ax2.set_xlabel("Mean predicted value",fontsize=20)
# #ax2.set_ylabel("Count",fontsize=20)
# #ax2.legend(loc="upper center", ncol=2,fontsize=15)
# plt.rc('font',family='SimHei')
#
# plt.tight_layout()
# plt.savefig("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/calibration_quantile_Top20.png")
# # plt.savefig("/home/liukang/Doc/calibration/result/calibration_quantile_Top20.eps",format='eps')
# #plt.show()
#
# boostrap_brier = pd.DataFrame()
# for i in range(round_num):
#
#     sample_result = total_drg_result.sample(frac=1,replace=True)
#
#     for result_type in result_list:
#
#          observation, prediction = calibration_curve(sample_result['Label'], sample_result[result_type], n_bins=group_num, strategy='quantile')
#          brier_score = np.mean(np.square(np.array(observation - prediction)))
#          boostrap_brier.loc[i,result_type] = brier_score
#
# for result_type in result_list:
#
#     brier_score_se_record.loc['All Top-20',result_type] = np.std(boostrap_brier[result_type])
#
# for i in range(len(result_list)):
#
#     #first_result = boostrap_brier.loc[:,result_list[i]].values
#
#     for j in range(i+1,len(result_list)):
#
#         #second_result = boostrap_brier.loc[:,result_list[j]].values
#         brier_boostrap_diff = pd.DataFrame()
#         brier_boostrap_diff['diff'] = boostrap_brier[result_list[i]] - boostrap_brier[result_list[j]]
#         brier_diff_below_0 = brier_boostrap_diff.loc[:,'diff'] < 0
#         sum_below_0 = np.sum(brier_diff_below_0)
#
#         if sum_below_0 < (round_num/2):
#
#             boostrap_p = 2*(sum_below_0 / round_num)
#
#         else:
#
#             boostrap_p = 2*((round_num - sum_below_0) / round_num)
#
#         brier_score_compare_p_record.loc['All_Top-20','{}_V_{}'.format(result_list[i],result_list[j])] = boostrap_p


#output
# brier_score_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/brier_score_all.csv")
# brier_score_se_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/brier_se_all.csv")
# brier_score_compare_p_record.to_csv("/home/chenqinhai/code_eicu/my_lab/fairness_strategy/local_result/calibration/brier_p_all.csv")

