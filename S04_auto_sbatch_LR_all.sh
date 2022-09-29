#!/bin/sh
sleep 10h
step=5000
final=10000
let start=0
let end=${start}+${step}
while [ $start -lt $final ]
do
  nohup python S04_similar_LR.py 0 0 ${start} ${end} > log/S04_LR_0_tra0_${start}_${end}.log 2>&1 &
  echo "submit [ ${start},${end} ] tra0 success!"
  nohup python S04_similar_LR.py 0 1 ${start} ${end} > log/S04_LR_0_tra1_${start}_${end}.log 2>&1 &
  echo "submit [ ${start},${end} ] tra1 success!"
  let start=start+$step
  let end=end+$step
done