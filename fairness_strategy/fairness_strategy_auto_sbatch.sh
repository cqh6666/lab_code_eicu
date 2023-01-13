#!/bin/sh
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
curStamp=$((timeStamp*1000+10#`date "+%N"`/1000000)) #将current转换为时间戳，精确到毫秒

all_risk_rate=(0.8 0.85 0.9 0.95)
all_loss_type=(1 2 3)
for risk_rate in ${all_risk_rate[@]}
do
  for loss_type in ${all_loss_type[@]}
  do
    nohup python getSubGroup2.py ${risk_rate} ${loss_type} > log/getSubGroup2_${risk_rate}_${loss_type}_${curStamp}.log 2>&1 &
    echo "submit ["${risk_rate} ${loss_type}"] success!"
  done
done
