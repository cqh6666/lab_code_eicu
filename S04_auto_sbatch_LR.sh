#!/bin/sh
start=0
end=10000
all_hos=(73)
#all_hos=(0)
for hos_id in ${all_hos[@]}
do
    nohup python S04_similar_LR.py ${hos_id} 0 ${start} ${end} > log/S04_LR_${hos_id}_tra0.log 2>&1 &
    echo "submit ["${hos_id} tra0"] success!"
    nohup python S04_similar_LR.py ${hos_id} 1 ${start} ${end} > log/S04_LR_${hos_id}_tra1.log 2>&1 &
    echo "submit ["${hos_id} tra1"] success!"
done
