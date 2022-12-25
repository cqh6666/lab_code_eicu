#!/bin/sh
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
curStamp=$((timeStamp*1000+10#`date "+%N"`/1000000)) #将current转换为时间戳，精确到毫秒

#source xxx.sh
all_hos=(73 0)
#all_hos=(264)
#all_hos=(73 167)
#all_hos=(73)
for hos_id in ${all_hos[@]}
do
#  nohup python S03_global_LR.py ${hos_id} > log/S03_LR_${hos_id}_${curStamp}.log 2>&1 &
  nohup python S03_global_XGB.py ${hos_id} > log/S03_XGB_${hos_id}_${curStamp}.log 2>&1 &
done
