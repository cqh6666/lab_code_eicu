#!/bin/sh
#source xxx.sh
#all_hos=(0)
all_hos=(0 73 167 264 338 420)
version=12
#all_hos=(73)
for hos_id in ${all_hos[@]}
do
  nohup python S03_global_LR.py ${hos_id} > log/S03_LR_${hos_id}_v${version}.log 2>&1 &
#  nohup python S03_global_XGB.py ${hos_id} > log/S03_XGB_${hos_id}_v${version}.log 2>&1 &
done
