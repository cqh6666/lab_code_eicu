#!/bin/sh
step=2000
final=10000
let start=0
let end=$step
while [ $start -lt $final ]
do
  nohup python S04_similar_XGB.py 0 0 ${start} ${end} > log/S04_XGB_0_tra0_${start}_${end}.log 2>&1 &
  nohup python S04_similar_XGB.py 0 1 ${start} ${end} > log/S04_XGB_0_tra1_${start}_${end}.log 2>&1 &
  echo "submit [ ${start},${end} ]success!"
  let start=start+$step
  let end=end+$step
done

