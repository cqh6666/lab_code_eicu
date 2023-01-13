#!/bin/sh
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
curStamp=$((timeStamp*1000+10#`date "+%N"`/1000000)) #将current转换为时间戳，精确到毫秒

#all_hos=(0)
all_hos=(73)
comps_list=(0.8)
to_hos_id=0
#comps_list=(0.7 0.8 0.9 0.95 0.99)
for comp in ${comps_list[@]}
do
  for hos_id in ${all_hos[@]}
  do
      nohup python S05_pca_similar_other_concat_LR.py ${hos_id} ${to_hos_id} 0 ${comp} > log/S05_LR_${hos_id}_tra0_${curStamp}.log 2>&1 &
      echo "submit ["${hos_id} ${to_hos_id} tra0"] success!"
      nohup python S05_pca_similar_other_concat_LR.py ${hos_id} ${to_hos_id} 1 ${comp} > log/S05_LR_${hos_id}_tra1_${curStamp}.log 2>&1 &
      echo "submit ["${hos_id} ${to_hos_id} tra1"] success!"
  done
done

