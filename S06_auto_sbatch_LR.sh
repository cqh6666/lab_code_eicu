#!/bin/sh
#all_hos=(73 167 264 338 420)
all_hos=(167)
mean_list=(5 50 100)
for comp in ${mean_list[@]}
do
  for hos_id in ${all_hos[@]}
  do
      nohup python S06_kth_avg_LR.py ${hos_id} 0 ${comp} > log/S06_LR_${hos_id}_tra0.log 2>&1 &
      echo "submit ["id${hos_id} tra0 mean${comp}"] success!"
      nohup python S06_kth_avg_LR.py ${hos_id} 1 ${comp} > log/S06_LR_${hos_id}_tra1.log 2>&1 &
      echo "submit ["id${hos_id} tra1 mean${comp}"] success!"
  done
done

