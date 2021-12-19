#!/bin/bash

python3 optimized/run_test.py &> /dev/null

input="optimized/logs_trt.log"
while IFS= read -r line
do
  echo "$line"
done < "$input"