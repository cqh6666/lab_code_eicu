#!/bin/sh
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s`
curStamp=$((timeStamp*1000+10#`date "+%N"`/1000000)) #将current转换为时间戳，精确到毫秒

#valid_array=(1 2 3 4 5)
valid_array=(3)
#all_hos=(0)
for valid in ${valid_array[@]}
do
#    nohup python S04_similar_LR.py ${hos_id} 0 > log/S04_LR_${hos_id}_tra0_${curStamp}.log 2>&1 &
#    echo "submit ["${hos_id} tra0"] success!"
    nohup python S04_similar_LR.py ${valid} > log/S04_LR_0_tra1_valid${valid}_${curStamp}.log 2>&1 &
    echo "submit ["valid: ${valid}"] success!"
done
