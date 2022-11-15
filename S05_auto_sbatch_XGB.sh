#!/bin/sh
all_hos=(0)
#all_hos=(73)
#comps_list=(0.7 0.8 0.9 0.95 0.99)
comps_list=(0.9 0.95 0.99)
for comp in ${comps_list[@]}
do
  for hos_id in ${all_hos[@]}
  do
      nohup python S05_pca_similar_XGB.py ${hos_id} 0 ${comp} > log/S05_XGB_${hos_id}_tra0.log 2>&1 &
      echo "submit ["${hos_id} tra0"] success!"
#      nohup python S05_pca_similar_XGB.py ${hos_id} 1 ${comp} > log/S05_XGB_${hos_id}_tra1.log 2>&1 &
#      echo "submit ["${hos_id} tra1"] success!"
  done
done

