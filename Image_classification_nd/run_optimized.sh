#!/bin/bash

option=$1
if [[ $option == "-debug" ]]
then 
  python3 optimized/run_test.py
  input="optimized/logs_trt.log"
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
elif [[ $option == "-no-debug" ]]
then
  python3 optimized/run_test.py &> /dev/null
  input="optimized/logs_trt.log"
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
else
  echo "Please provide no-debug or debug option"
fi