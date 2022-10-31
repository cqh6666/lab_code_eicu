#!/bin/sh
all_hos=(0)
for hos_id in ${all_hos[@]}
do
    nohup python S04_similar_XGB.py ${hos_id} 0 > log/S04_XGB_${hos_id}_tra0.log 2>&1 &
    echo "submit ["${hos_id} tra0"] success!"
    nohup python S04_similar_XGB.py ${hos_id} 1 > log/S04_XGB_${hos_id}_tra1.log 2>&1 &
    echo "submit ["${hos_id} tra1"] success!"
done
